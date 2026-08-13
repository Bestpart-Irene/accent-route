"""按真实类别分层的 test-speaker paired bootstrap(决策 #2 的统计口径)。

两个量分开呈现、绝不合并:
  1. bootstrap CI —— 只重采样测试集 speaker cluster(类内等量有放回),
     对 seed 平均后的 Δmacro-F1 出 percentile CI;
  2. seed 变异 —— 全测试集上逐 seed 配对 Δ 的明细与 std。
CI 未覆盖训练随机性,所以报告措辞只允许 "test-speaker bootstrap CI excludes
zero",不得泛称 statistically significant —— 数据结构里没有 significant 字段。
"""

from dataclasses import dataclass

import numpy as np

from accentroute.eval.metrics import macro_f1


@dataclass(frozen=True)
class AblationStats:
    delta_mean: float  # seed 平均后 Δmacro-F1 的 bootstrap 均值
    ci_low: float
    ci_high: float
    n_boot: int
    ci_excludes_zero: bool  # 报告措辞只允许引用这个字段
    seed_deltas: tuple[float, ...]  # 全测试集上逐 seed 配对 Δ
    seed_delta_std: float


def stratified_cluster_resample(
    rng: np.random.Generator, strata: dict[str, np.ndarray]
) -> np.ndarray:
    """每类内部有放回重采样该类的 cluster(等量)→ 稀有类不会在采样中消失。"""
    return np.concatenate([rng.choice(ks, size=len(ks)) for ks in strata.values()])


def stratified_cluster_bootstrap(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    speaker_keys: np.ndarray,
    classes: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> AblationStats:
    """preds_a/b: [n_seeds, n](如 C 臂 / B 臂各 3 seeds 的测试集预测)。

    classes 通常就是 y_true(每个 speaker 一个类);labels 取其去重集,
    supported-class 场景由调用方先行子集化。
    """
    y_true = np.asarray(y_true)
    speaker_keys = np.asarray(speaker_keys)
    classes = np.asarray(classes)
    labels = sorted(np.unique(classes).tolist())

    rng = np.random.default_rng(seed)
    idx_of = {k: np.flatnonzero(speaker_keys == k) for k in np.unique(speaker_keys)}
    strata = {c: np.unique(speaker_keys[classes == c]) for c in labels}

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        chosen = stratified_cluster_resample(rng, strata)
        idx = np.concatenate([idx_of[k] for k in chosen])
        fa = np.mean([macro_f1(y_true[idx], p[idx], labels=labels) for p in preds_a])
        fb = np.mean([macro_f1(y_true[idx], p[idx], labels=labels) for p in preds_b])
        deltas[b] = fa - fb

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    seed_deltas = tuple(
        float(macro_f1(y_true, a, labels=labels) - macro_f1(y_true, b, labels=labels))
        for a, b in zip(preds_a, preds_b)
    )
    return AblationStats(
        delta_mean=float(deltas.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
        ci_excludes_zero=bool(lo > 0 or hi < 0),
        seed_deltas=seed_deltas,
        seed_delta_std=float(np.std(seed_deltas)),
    )
