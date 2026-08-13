"""弱标签三池盲审(决策 #4)。

只审 accepted 池看不到筛选器的选择偏差,所以 reject/review 池按 reject_reason
分层同抽 —— false-reject 率进 datasheet。
每类 n=25 的 precision 噪声大,一律带 Wilson 区间报告,禁止裸报点估计。
kill 规则:accepted 池某类 precision < kill_precision → 该类弱标签整体剔除。
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def wilson_interval(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间:小样本比例的区间,n=25 时明显宽于正态近似。"""
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
    """accepted 池按类分层 + reject/review 池按 reject_reason 分层。

    返回内部表(含标签列);盲听 CSV 由调用方 drop 标签列后导出。
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
    """annotated 需含 human_label 列(盲听结果)。

    accepted 池:precision = human_label == accent_label 的比例。
    reject 池:false-reject = human_label 与 prior_label 一致(即本该收录)的比例。
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
