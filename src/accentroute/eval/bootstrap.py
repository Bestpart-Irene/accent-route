"""Paired test-speaker bootstrap, stratified by true class (the statistical convention
behind decision #2).

Two quantities are reported side by side and never merged:
  1. the bootstrap CI — resamples only test-set speaker clusters (with replacement, equal
     count within each class) and gives a percentile CI on the seed-averaged Δmacro-F1;
  2. seed variation — the per-seed paired Δ over the full test set, plus its std.
The CI does not cover training randomness, so the report may only say "test-speaker
bootstrap CI excludes zero" and never "statistically significant" — which is why this data
structure has no `significant` field to reach for.
"""

from dataclasses import dataclass

import numpy as np

from accentroute.eval.metrics import macro_f1


@dataclass(frozen=True)
class AblationStats:
    delta_mean: float  # bootstrap mean of the seed-averaged Δmacro-F1
    ci_low: float
    ci_high: float
    n_boot: int
    ci_excludes_zero: bool  # the only field report wording is allowed to cite
    seed_deltas: tuple[float, ...]  # per-seed paired Δ over the full test set
    seed_delta_std: float


def stratified_cluster_resample(
    rng: np.random.Generator, strata: dict[str, np.ndarray]
) -> np.ndarray:
    """Resample each class's clusters with replacement, keeping the count per class → rare
    classes can never vanish from a bootstrap draw."""
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
    """preds_a/b: [n_seeds, n] (e.g. test-set predictions from 3 seeds each of arm C/arm B).

    classes is normally just y_true (one class per speaker); labels is its unique set. For
    the supported-class case the caller subsets the arrays first.
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
