"""Weak-label consensus: evidence level × Qwen self-consistency voting (decisions #3/#5).

A label is accepted only when all three hold:
  1. evidence_level ∈ {E1, E2} — channel/region metadata alone cannot establish an accent;
  2. the Qwen majority vote == the human prior label;
  3. the majority is at least 2/3.
E3, or any disagreement, goes to the review pool: it is sampled for human audit and never
reaches training.
consensus_score = majority share × evidence weight. It is an engineering ranking score,
not a calibrated confidence.
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
    """Decide row by row → status/accent_label/consensus_score/split.

    Accepted rows are pinned to label_source="weak" and split="train"; the schema's
    weak_never_in_eval check backstops that invariant a second time.
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
