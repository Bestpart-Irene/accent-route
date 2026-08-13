"""Three-pool blind audit of weak labels (decision #4).

Auditing only the accepted pool would hide the filter's own selection bias, so the
reject/review pools are sampled alongside it, stratified by reject_reason — the
false-reject rate goes into the datasheet.
Precision on n=25 per class is noisy, so it is always reported with a Wilson interval;
bare point estimates are not allowed.
Kill rule: if a class's precision in the accepted pool falls below kill_precision, that
class's weak labels are dropped wholesale.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def wilson_interval(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a small-sample proportion; at n=25 it is noticeably wider
    than the normal approximation."""
    if n_total == 0:
        return (0.0, 1.0)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def draw_audit_sample(
    df: pd.DataFrame,
    accepted_per_class: int = 25,
    reject_pool_n: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Stratify the accepted pool by class and the reject/review pools by reject_reason.

    Returns the internal table, labels included; the caller drops the label columns before
    exporting the blind-listening CSV.
    """
    rng = np.random.default_rng(seed)

    def _sample(group: pd.DataFrame, n: int) -> pd.DataFrame:
        n = min(n, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        return group.loc[np.sort(idx)]

    accepted = df[df["status"] == "accepted"]
    acc_parts = [
        _sample(grp, accepted_per_class) for _, grp in accepted.groupby("accent_label")
    ]

    rejected = df[df["status"].isin(["rejected", "review"])]
    reasons = sorted(rejected["reject_reason"].dropna().unique())
    per_reason = reject_pool_n // len(reasons) if reasons else 0
    rej_parts = [
        _sample(rejected[rejected["reject_reason"] == r], per_reason) for r in reasons
    ]

    acc_df = pd.concat(acc_parts) if acc_parts else accepted.iloc[:0]
    rej_df = pd.concat(rej_parts) if rej_parts else rejected.iloc[:0]
    acc_df = acc_df.assign(pool="accepted")
    rej_df = rej_df.assign(pool="rejected")
    return pd.concat([acc_df, rej_df], ignore_index=True)


@dataclass(frozen=True)
class AuditReport:
    accepted_precision: dict[str, float]
    accepted_precision_ci: dict[str, tuple[float, float]]
    accepted_n: dict[str, int]
    killed_classes: list[str]
    false_reject_rate: dict[str, float] = field(default_factory=dict)


def audit_report(annotated: pd.DataFrame, kill_precision: float = 0.80) -> AuditReport:
    """annotated must carry a human_label column (the blind-listening result).

    Accepted pool: precision = the fraction where human_label == accent_label.
    Reject pool: false-reject = the fraction where human_label agrees with prior_label,
    i.e. clips that should have been kept.
    """
    acc = annotated[annotated["pool"] == "accepted"]
    precision, ci, ns = {}, {}, {}
    for label, grp in acc.groupby("accent_label"):
        n_correct = int((grp["human_label"] == label).sum())
        precision[label] = n_correct / len(grp)
        ci[label] = wilson_interval(n_correct, len(grp))
        ns[label] = len(grp)

    killed = sorted(lab for lab, p in precision.items() if p < kill_precision)

    rej = annotated[annotated["pool"] == "rejected"]
    false_reject = {}
    if len(rej) and "prior_label" in rej.columns:
        for reason, grp in rej.groupby("reject_reason"):
            false_reject[reason] = float((grp["human_label"] == grp["prior_label"]).mean())

    return AuditReport(
        accepted_precision=precision,
        accepted_precision_ci=ci,
        accepted_n=ns,
        killed_classes=killed,
        false_reject_rate=false_reject,
    )
