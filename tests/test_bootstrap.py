"""T12: 按类分层的 test-speaker paired bootstrap + seed 变异分开报告。

措辞纪律在数据结构里固化:报告端只允许引用 ci_excludes_zero,
不存在任何叫 significant 的字段。
"""

import numpy as np
import pytest

from accentroute.eval.bootstrap import (
    AblationStats,
    stratified_cluster_bootstrap,
    stratified_cluster_resample,
)

CLASSES = ["en-US", "en-GB", "L1-Korean"]


def _make_eval_set(n_per_class=40, n_speakers_per_class=8, seed=0):
    rng = np.random.default_rng(seed)
    y, spk = [], []
    for c_i, cls in enumerate(CLASSES):
        for i in range(n_per_class):
            y.append(cls)
            spk.append(f"{cls}:spk{i % n_speakers_per_class}")
    return np.array(y), np.array(spk), rng


def _preds_with_accuracy(y, acc, rng, n_seeds=3):
    preds = []
    for _ in range(n_seeds):
        correct = rng.random(len(y)) < acc
        wrong = rng.choice(CLASSES, len(y))
        preds.append(np.where(correct, y, wrong))
    return np.stack(preds)


class TestResample:
    def test_every_class_present_in_every_resample(self):
        """分层重采样:每类的 cluster 数保持不变 → 稀有类不会消失。"""
        rng = np.random.default_rng(0)
        strata = {
            "en-US": np.array(["a", "b", "c"]),
            "rare": np.array(["only-one"]),
        }
        for _ in range(50):
            chosen = stratified_cluster_resample(rng, strata)
            assert len(chosen) == 4
            assert "only-one" in chosen  # 单 cluster 类每次必在


class TestBootstrap:
    def test_identical_preds_ci_straddles_zero(self):
        y, spk, rng = _make_eval_set()
        preds = _preds_with_accuracy(y, 0.7, rng)
        stats = stratified_cluster_bootstrap(y, preds, preds.copy(), spk, y, n_boot=300, seed=1)
        assert isinstance(stats, AblationStats)
        assert stats.ci_low <= 0 <= stats.ci_high
        assert stats.ci_excludes_zero is False
        assert stats.delta_mean == pytest.approx(0.0, abs=1e-9)

    def test_injected_shift_detected(self):
        y, spk, rng = _make_eval_set()
        preds_good = _preds_with_accuracy(y, 0.85, rng)
        preds_bad = _preds_with_accuracy(y, 0.55, rng)
        stats = stratified_cluster_bootstrap(
            y, preds_good, preds_bad, spk, y, n_boot=300, seed=1
        )
        assert stats.ci_excludes_zero is True
        assert stats.delta_mean > 0.05

    def test_seed_deterministic(self):
        y, spk, rng = _make_eval_set()
        a = _preds_with_accuracy(y, 0.8, rng)
        b = _preds_with_accuracy(y, 0.7, rng)
        s1 = stratified_cluster_bootstrap(y, a, b, spk, y, n_boot=200, seed=7)
        s2 = stratified_cluster_bootstrap(y, a, b, spk, y, n_boot=200, seed=7)
        assert s1 == s2

    def test_seed_deltas_reported_separately(self):
        y, spk, rng = _make_eval_set()
        a = _preds_with_accuracy(y, 0.8, rng, n_seeds=3)
        b = _preds_with_accuracy(y, 0.7, rng, n_seeds=3)
        stats = stratified_cluster_bootstrap(y, a, b, spk, y, n_boot=100, seed=1)
        assert len(stats.seed_deltas) == 3
        assert stats.seed_delta_std >= 0
        # seed_deltas 是全测试集上的逐 seed 配对差,不是 bootstrap 均值
        from accentroute.eval.metrics import macro_f1

        expected0 = macro_f1(y, a[0], labels=CLASSES) - macro_f1(y, b[0], labels=CLASSES)
        assert stats.seed_deltas[0] == pytest.approx(expected0)

    def test_no_significant_field(self):
        """措辞纪律:不存在 significant 字段。"""
        assert "significant" not in AblationStats.__dataclass_fields__
        assert "ci_excludes_zero" in AblationStats.__dataclass_fields__
