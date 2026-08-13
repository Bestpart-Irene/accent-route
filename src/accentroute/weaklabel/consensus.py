"""弱标签共识:证据等级 × Qwen 自洽投票(决策 #3/#5)。

接受条件(全满足):
  1. evidence_level ∈ {E1, E2} —— 频道地区元数据单独不足以定口音;
  2. Qwen 多数票 == 人工先验标签;
  3. 多数强度 ≥ 2/3。
E3 或不一致 → review 池(进人工抽查,不进训练)。
consensus_score = 多数票比例 × 证据权重,是工程排序分,不是校准置信度。
"""

from collections import Counter
from dataclasses import dataclass

import pandas as pd

EVIDENCE_WEIGHT = {"E1": 1.0, "E2": 0.85}
_MIN_MAJORITY = 2 / 3


@dataclass(frozen=True)
class WeakLabelDecision:
    accepted: bool
    label: str | None
    consensus_score: float
    reason: str


def consensus(
    evidence_level: str, prior_label: str, qwen_votes: list[str]
) -> WeakLabelDecision:
    if evidence_level not in EVIDENCE_WEIGHT:
        return WeakLabelDecision(False, None, 0.0, "evidence_E3")
    if not qwen_votes:
        return WeakLabelDecision(False, None, 0.0, "qwen_disagrees")
    top, n = Counter(qwen_votes).most_common(1)[0]
    if top != prior_label or n / len(qwen_votes) < _MIN_MAJORITY:
        return WeakLabelDecision(False, None, 0.0, "qwen_disagrees")
    score = (n / len(qwen_votes)) * EVIDENCE_WEIGHT[evidence_level]
    return WeakLabelDecision(True, prior_label, score, "consensus")


def apply_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """逐行判定 → status/accent_label/consensus_score/split。

    接受行固定 label_source="weak"、split="train"(schema 的 weak_never_in_eval
    再把这条不变量兜住一次)。
    """
    out = df.copy()
    decisions = [
        consensus(row["evidence_level"], row["prior_label"], list(row["qwen_votes"]))
        for row in out.to_dict("records")
    ]
    out["accent_label"] = [d.label for d in decisions]
    out["consensus_score"] = [d.consensus_score for d in decisions]
    out["label_source"] = "weak"
    out["status"] = [
        "accepted" if d.accepted else ("rejected" if d.reason == "evidence_E3" else "review")
        for d in decisions
    ]
    out["reject_reason"] = [
        d.reason if not d.accepted and d.reason == "evidence_E3" else None for d in decisions
    ]
    out["split"] = ["train" if d.accepted else "unassigned" for d in decisions]
    return out
