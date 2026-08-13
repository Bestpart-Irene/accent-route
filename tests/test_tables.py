"""T15: 结果表生成 —— 三臂消融、按源分层、supported-class 域外、措辞校验。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accentroute.eval.tables import (
    BANNED_PHRASES,
    ablation_table,
    assert_budget_alignment,
    check_wording,
    format_delta_line,
    per_source_class_f1,
    supported_class_report,
)

CLASSES = ["en-US", "en-GB", "L1-Korean"]


def _run_dir(tmp_path: Path, arm: str, seed: int, steps: int, shared_hash="h1") -> Path:
    d = tmp_path / f"{arm}-seed{seed}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps({"arm": arm, "seed": seed, "total_steps": steps,
                    "shared_config_hash": shared_hash, "best_val_macro_f1": 0.5})
    )
    return d


class TestBudgetAlignment:
    def test_matching_steps_and_hash_pass(self, tmp_path):
        for arm, steps in [("b_gold_oversampled", 900), ("c_gold_weak", 900)]:
            for seed in (17, 42):
                _run_dir(tmp_path, arm, seed, steps)
        assert_budget_alignment(tmp_path)

    def test_step_mismatch_raises(self, tmp_path):
        _run_dir(tmp_path, "b_gold_oversampled", 17, 800)
        _run_dir(tmp_path, "c_gold_weak", 17, 900)
        with pytest.raises(AssertionError, match="step"):
            assert_budget_alignment(tmp_path)

    def test_shared_hash_mismatch_raises(self, tmp_path):
        _run_dir(tmp_path, "b_gold_oversampled", 17, 900, shared_hash="h1")
        _run_dir(tmp_path, "c_gold_weak", 17, 900, shared_hash="h2")
        with pytest.raises(AssertionError, match="shared_config_hash"):
            assert_budget_alignment(tmp_path)


def _preds(y, acc, rng, n_seeds=3):
    return np.stack(
        [np.where(rng.random(len(y)) < acc, y, rng.choice(CLASSES, len(y)))
         for _ in range(n_seeds)]
    )


@pytest.fixture()
def eval_set():
    rng = np.random.default_rng(0)
    y = np.array([CLASSES[i % 3] for i in range(90)])
    spk = np.array([f"{y[i]}:spk{i % 6}" for i in range(90)])
    src = np.array(["common_voice" if i % 2 else "l2_arctic" for i in range(90)])
    return y, spk, src, rng


class TestAblationTable:
    def test_headline_is_c_minus_b(self, eval_set):
        y, spk, _, rng = eval_set
        preds = {"a_gold": _preds(y, 0.60, rng), "b_gold_oversampled": _preds(y, 0.65, rng),
                 "c_gold_weak": _preds(y, 0.85, rng)}
        table = ablation_table(y, preds, spk, n_boot=200, seed=1)
        headline = table[table["comparison"] == "C-B"].iloc[0]
        assert headline["headline"] is np.True_ or headline["headline"] is True
        assert headline["delta_macro_f1"] > 0
        assert headline["ci_excludes_zero"] in (True, np.True_)
        budget = table[table["comparison"] == "A-B"].iloc[0]
        assert budget["headline"] in (False, np.False_)

    def test_no_significance_column(self, eval_set):
        y, spk, _, rng = eval_set
        preds = {a: _preds(y, 0.7, rng) for a in
                 ["a_gold", "b_gold_oversampled", "c_gold_weak"]}
        table = ablation_table(y, preds, spk, n_boot=100, seed=1)
        assert not any("signific" in c.lower() for c in table.columns)
        assert {"ci_low", "ci_high", "ci_excludes_zero", "seed_delta_std"} <= set(table.columns)


class TestPerSourceStratification:
    def test_per_source_class_f1(self, eval_set):
        y, _, src, rng = eval_set
        pred = _preds(y, 0.8, rng, n_seeds=1)[0]
        table = per_source_class_f1(y, pred, src)
        assert set(table["source"]) == {"common_voice", "l2_arctic"}
        assert set(table.columns) >= {"source", "accent_label", "f1", "n_clips"}
        # 每个 (source, class) 组合都在表里 → 混杂问题肉眼可查
        assert len(table) == 6


class TestSupportedClassReport:
    def test_excluded_classes_reported_and_in_domain_control(self, eval_set):
        y, _, _, rng = eval_set
        pred = _preds(y, 0.8, rng, n_seeds=1)[0]
        coverage = pd.DataFrame(
            {"accent_label": CLASSES, "n_speakers": [8, 6, 2],
             "include": [True, True, False]}
        )
        rep = supported_class_report(y, pred, coverage, in_domain_y=y, in_domain_pred=pred)
        assert rep["supported_classes"] == ["en-GB", "en-US"]
        assert rep["excluded_classes"] == ["L1-Korean"]
        assert "supported_class_macro_f1" in rep
        # 域内对照必须在同一支持类子集上算,才可比
        assert "in_domain_supported_class_macro_f1" in rep
        assert "full_8class_macro_f1" not in rep  # 不得混入不可比的数字


class TestWording:
    def test_banned_phrases_detected(self):
        with pytest.raises(ValueError, match="statistically significant"):
            check_wording("The gain is statistically significant (p<0.05).")

    def test_case_insensitive(self):
        with pytest.raises(ValueError):
            check_wording("Statistically Significant improvement")

    def test_sota_banned(self):
        with pytest.raises(ValueError, match="(?i)sota"):
            check_wording("Our SOTA result beats everything")

    def test_approved_wording_passes(self):
        check_wording(
            "Delta macro-F1 = 0.081; test-speaker stratified bootstrap 95% CI "
            "[0.021, 0.142] excludes zero; per-seed delta std = 0.014."
        )

    def test_format_delta_line_is_compliant(self):
        from accentroute.eval.bootstrap import AblationStats

        stats = AblationStats(0.081, 0.021, 0.142, 10000, True, (0.07, 0.08, 0.09), 0.014)
        line = format_delta_line("C-B", stats)
        check_wording(line)  # 生成的模板本身必须合规
        assert "excludes zero: yes" in line
        assert "CI" in line

    def test_all_banned_phrases_listed(self):
        assert "statistically significant" in BANNED_PHRASES
        assert "state-of-the-art" in BANNED_PHRASES
