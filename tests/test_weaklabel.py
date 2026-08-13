"""T13: 弱标注共识规则与三池审计 —— 项目核心论点的机器契约。"""

import pandas as pd
import pytest

from accentroute.weaklabel.audit import audit_report, draw_audit_sample, wilson_interval
from accentroute.weaklabel.consensus import (
    EVIDENCE_WEIGHT,
    WeakLabelDecision,
    apply_consensus,
    consensus,
)


class TestConsensusRule:
    @pytest.mark.parametrize(
        "evidence,prior,votes,expected",
        [
            # 证据 E3 → 直接拒(频道地区元数据不可靠,决策 #3)
            ("E3", "en-AU", ["en-AU", "en-AU", "en-AU"], (False, None, 0.0, "evidence_E3")),
            # Qwen 与先验不一致 → review 池
            ("E1", "en-AU", ["en-US", "en-US", "en-AU"], (False, None, 0.0, "qwen_disagrees")),
            # 多数强度不足(3 票分散)
            ("E1", "en-AU", ["en-AU", "en-US", "en-GB"], (False, None, 0.0, "qwen_disagrees")),
            # E1 全票 → conf 1.0
            ("E1", "en-AU", ["en-AU", "en-AU", "en-AU"], (True, "en-AU", 1.0, "consensus")),
            # E2 2/3 票 → 2/3 * 0.85
            ("E2", "L1-Korean", ["L1-Korean", "L1-Korean", "unsure"],
             (True, "L1-Korean", pytest.approx(0.5667, abs=1e-3), "consensus")),
            # unsure 占多数 → 拒
            ("E1", "en-GB", ["unsure", "unsure", "en-GB"], (False, None, 0.0, "qwen_disagrees")),
        ],
    )
    def test_table_driven(self, evidence, prior, votes, expected):
        got = consensus(evidence, prior, votes)
        assert (got.accepted, got.label, got.consensus_score, got.reason) == expected

    def test_k1_degradation_still_works(self):
        """k_votes 降到 1(止损阶梯 ②)时规则不变:单票即多数。"""
        got = consensus("E1", "en-AU", ["en-AU"])
        assert got.accepted is True
        assert got.consensus_score == 1.0

    def test_returns_frozen_decision(self):
        got = consensus("E1", "en-AU", ["en-AU"])
        assert isinstance(got, WeakLabelDecision)
        with pytest.raises(AttributeError):
            got.accepted = False

    def test_evidence_weights_locked(self):
        assert EVIDENCE_WEIGHT == {"E1": 1.0, "E2": 0.85}


class TestApplyConsensus:
    def _df(self):
        return pd.DataFrame(
            [
                {"clip_id": "y1", "prior_label": "en-AU", "evidence_level": "E1",
                 "qwen_votes": ["en-AU", "en-AU", "en-AU"]},
                {"clip_id": "y2", "prior_label": "en-GB", "evidence_level": "E2",
                 "qwen_votes": ["en-US", "en-US", "en-GB"]},
                {"clip_id": "y3", "prior_label": "L1-Arabic", "evidence_level": "E3",
                 "qwen_votes": ["L1-Arabic", "L1-Arabic", "L1-Arabic"]},
            ]
        )

    def test_status_and_fields(self):
        out = apply_consensus(self._df())
        by_id = out.set_index("clip_id")
        assert by_id.loc["y1", "status"] == "accepted"
        assert by_id.loc["y1", "accent_label"] == "en-AU"
        assert by_id.loc["y1", "label_source"] == "weak"
        assert by_id.loc["y1", "split"] == "train"  # 弱标签只进 train
        assert by_id.loc["y2", "status"] == "review"
        assert by_id.loc["y3", "status"] == "rejected"
        assert by_id.loc["y3", "reject_reason"] == "evidence_E3"

    def test_acceptance_rate_reported(self):
        out = apply_consensus(self._df())
        assert (out["status"] == "accepted").mean() == pytest.approx(1 / 3)


class TestAuditSampling:
    def _pool(self):
        rows = []
        for cls in ["en-AU", "L1-Korean"]:
            for i in range(60):
                rows.append({"clip_id": f"acc-{cls}-{i}", "status": "accepted",
                             "accent_label": cls, "reject_reason": None})
        for reason in ["evidence_E3", "qwen_disagrees"]:
            for i in range(80):
                rows.append({"clip_id": f"rej-{reason}-{i}", "status": "rejected",
                             "accent_label": None, "reject_reason": reason})
        return pd.DataFrame(rows)

    def test_stratified_accepted_sample(self):
        sample = draw_audit_sample(self._pool(), accepted_per_class=25, reject_pool_n=50, seed=0)
        acc = sample[sample["pool"] == "accepted"]
        assert len(acc) == 50
        assert acc.groupby("accent_label").size().tolist() == [25, 25]

    def test_reject_pool_stratified_by_reason(self):
        """决策 #4:审计必须覆盖 reject 池,否则看不到筛选器的选择偏差。"""
        sample = draw_audit_sample(self._pool(), accepted_per_class=25, reject_pool_n=50, seed=0)
        rej = sample[sample["pool"] == "rejected"]
        assert len(rej) == 50
        assert rej.groupby("reject_reason").size().tolist() == [25, 25]

    def test_blind_csv_has_no_label_columns(self):
        sample = draw_audit_sample(self._pool(), seed=0)
        assert "accent_label" in sample.columns  # 内部表保留
        blind = sample.drop(columns=[c for c in ("accent_label", "reject_reason") if c in sample])
        assert "accent_label" not in blind.columns

    def test_deterministic(self):
        a = draw_audit_sample(self._pool(), seed=0)
        b = draw_audit_sample(self._pool(), seed=0)
        pd.testing.assert_frame_equal(a, b)


class TestAuditReport:
    def test_precision_and_kill_rule(self):
        rows = []
        # en-AU:25 条 24 对 → 0.96,通过
        for i in range(25):
            rows.append({"pool": "accepted", "accent_label": "en-AU",
                         "human_label": "en-AU" if i < 24 else "en-US", "reject_reason": None})
        # L1-Korean:25 条 15 对 → 0.60,低于 0.80 → kill
        for i in range(25):
            rows.append({"pool": "accepted", "accent_label": "L1-Korean",
                         "human_label": "L1-Korean" if i < 15 else "L1-Mandarin",
                         "reject_reason": None})
        # reject 池:10 条中 3 条其实是对的 → false-reject 0.3
        for i in range(10):
            rows.append({"pool": "rejected", "accent_label": None, "reject_reason": "qwen_disagrees",
                         "human_label": "en-AU" if i < 3 else "unsure",
                         "prior_label": "en-AU"})
        report = audit_report(pd.DataFrame(rows), kill_precision=0.80)
        prec = report.accepted_precision
        assert prec["en-AU"] == pytest.approx(0.96)
        assert prec["L1-Korean"] == pytest.approx(0.60)
        assert report.killed_classes == ["L1-Korean"]
        assert report.false_reject_rate["qwen_disagrees"] == pytest.approx(0.3)

    def test_wilson_interval_widens_with_small_n(self):
        lo25, hi25 = wilson_interval(24, 25)
        lo250, hi250 = wilson_interval(240, 250)
        assert (hi25 - lo25) > (hi250 - lo250)  # n=25 的区间必须明显更宽
        assert 0.0 <= lo25 <= hi25 <= 1.0

    def test_report_carries_interval_per_class(self):
        rows = [{"pool": "accepted", "accent_label": "en-AU", "human_label": "en-AU",
                 "reject_reason": None} for _ in range(25)]
        report = audit_report(pd.DataFrame(rows))
        lo, hi = report.accepted_precision_ci["en-AU"]
        assert lo < 1.0 <= hi  # 25/25 也不能报成确定的 1.0
