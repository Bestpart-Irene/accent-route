"""结果表生成 + 公平性/措辞的机器校验(决策 #1/#2/#3/#7)。

三张核心表:
  1. 三臂消融(头条 C−B,附 A−B 预算效应)—— CI 与 seed 变异分列;
  2. 按源分层的每类 F1 —— source shortcut 肉眼可查;
  3. supported-class 域外报告 —— 附域内模型在同一支持类子集的对照值。
外加两道机器闸门:预算对齐断言、报告措辞校验。
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from accentroute.eval.bootstrap import AblationStats, stratified_cluster_bootstrap
from accentroute.eval.metrics import macro_f1

BANNED_PHRASES = (
    "statistically significant",
    "statistical significance",
    "significant improvement",
    "sota",
    "state-of-the-art",
    "state of the art",
)


def check_wording(text: str) -> None:
    """报告文本闸门:CI 未覆盖训练随机性,不得泛称显著;不得称 SOTA。"""
    for phrase in BANNED_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
            raise ValueError(
                f"banned phrase {phrase!r} in report text; use "
                '"test-speaker bootstrap CI excludes zero" instead'
            )


def format_delta_line(comparison: str, stats: AblationStats) -> str:
    """固定措辞模板 —— 报告里的每个 Δ 都走这里生成。"""
    seeds = ", ".join(f"{d:.4f}" for d in stats.seed_deltas)
    return (
        f"{comparison}: delta macro-F1 = {stats.delta_mean:.4f}; "
        f"test-speaker stratified bootstrap 95% CI "
        f"[{stats.ci_low:.4f}, {stats.ci_high:.4f}] "
        f"(excludes zero: {'yes' if stats.ci_excludes_zero else 'no'}); "
        f"per-seed delta = [{seeds}], std = {stats.seed_delta_std:.4f}"
    )


def assert_budget_alignment(runs_dir: Path) -> None:
    """三臂公平性硬门:B/C 总步数必须相等,所有 run 的共享协议哈希必须一致。"""
    runs = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/metrics.json"))]
    if not runs:
        raise AssertionError(f"no runs found under {runs_dir}")

    hashes = {r["shared_config_hash"] for r in runs}
    if len(hashes) > 1:
        raise AssertionError(
            f"shared_config_hash differs across runs: {sorted(hashes)}; "
            "arms did not share the training protocol"
        )

    steps_by_arm: dict[str, set] = {}
    for r in runs:
        steps_by_arm.setdefault(r["arm"], set()).add(r["total_steps"])
    b_steps = steps_by_arm.get("b_gold_oversampled", set())
    c_steps = steps_by_arm.get("c_gold_weak", set())
    if b_steps and c_steps and b_steps != c_steps:
        raise AssertionError(
            f"B/C total step counts differ (B={sorted(b_steps)}, C={sorted(c_steps)}); "
            "the C-B comparison would confound data content with training budget"
        )


def ablation_table(
    y_true: np.ndarray,
    preds_by_arm: dict[str, np.ndarray],
    speaker_keys: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """preds_by_arm: {arm: [n_seeds, n]}。头条 C−B,附 A−B(预算效应)。"""
    comparisons = [
        ("C-B", "c_gold_weak", "b_gold_oversampled", True),
        ("A-B", "a_gold", "b_gold_oversampled", False),
    ]
    rows = []
    for name, arm_a, arm_b, headline in comparisons:
        if arm_a not in preds_by_arm or arm_b not in preds_by_arm:
            continue
        stats = stratified_cluster_bootstrap(
            y_true, preds_by_arm[arm_a], preds_by_arm[arm_b],
            speaker_keys, y_true, n_boot=n_boot, seed=seed,
        )
        rows.append(
            {
                "comparison": name,
                "headline": headline,
                "delta_macro_f1": stats.delta_mean,
                "ci_low": stats.ci_low,
                "ci_high": stats.ci_high,
                "ci_excludes_zero": stats.ci_excludes_zero,
                "seed_deltas": list(stats.seed_deltas),
                "seed_delta_std": stats.seed_delta_std,
                "n_boot": stats.n_boot,
                "report_line": format_delta_line(name, stats),
            }
        )
    return pd.DataFrame(rows)


def per_source_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray
) -> pd.DataFrame:
    """逐 (source, class) 的 F1:某类只在一个源上好用,这张表会直接暴露。"""
    rows = []
    for source in sorted(np.unique(sources)):
        mask = sources == source
        for label in sorted(np.unique(y_true)):
            rows.append(
                {
                    "source": source,
                    "accent_label": label,
                    "f1": macro_f1(y_true[mask], y_pred[mask], labels=[label]),
                    "n_clips": int(np.sum(mask & (y_true == label))),
                }
            )
    return pd.DataFrame(rows)


def supported_class_report(
    ood_y: np.ndarray,
    ood_pred: np.ndarray,
    coverage: pd.DataFrame,
    in_domain_y: np.ndarray,
    in_domain_pred: np.ndarray,
) -> dict:
    """supported-class macro-F1 + 同一子集上的域内对照(唯一可比的对照)。

    刻意不返回完整 8 类数字 —— 排除类后的指标与全 8 类不可横比。
    """
    supported = sorted(coverage[coverage["include"]]["accent_label"].tolist())
    excluded = sorted(coverage[~coverage["include"]]["accent_label"].tolist())
    ood_mask = np.isin(ood_y, supported)
    in_mask = np.isin(in_domain_y, supported)
    return {
        "supported_classes": supported,
        "excluded_classes": excluded,
        "supported_class_macro_f1": macro_f1(
            ood_y[ood_mask], ood_pred[ood_mask], labels=supported
        ),
        "in_domain_supported_class_macro_f1": macro_f1(
            in_domain_y[in_mask], in_domain_pred[in_mask], labels=supported
        ),
        "n_ood_clips": int(ood_mask.sum()),
    }
