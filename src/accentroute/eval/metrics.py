"""Metrics: a standalone implementation (no sklearn dependency), cross-checked against
sklearn reference values in the unit tests.

labels defaults to the full 8 classes; supported-class macro-F1 (after EdAcc's excluded
classes are dropped) passes its subset explicitly. The two are not comparable — the table
generation layer (T15) is responsible for naming them distinctly.
"""

from collections.abc import Sequence

import numpy as np

from accentroute.schema import ACCENTS


def confusion(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str] = tuple(ACCENTS)
) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            m[index[t], index[p]] += 1
    return m


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str] = tuple(ACCENTS)
) -> float:
    """Unweighted mean of per-class F1; a class with no support and no predictions scores 0
    (matching sklearn's zero_division=0).

    Deliberately not routed through the confusion matrix: when labels is a subset, samples
    predicted outside that subset must still count as FN for their true class (otherwise
    supported-class macro-F1 comes out inflated), so TP/FP/FN are counted per class here.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for lab in labels:
        tp = float(np.sum((y_true == lab) & (y_pred == lab)))
        fp = float(np.sum((y_pred == lab) & (y_true != lab)))
        fn = float(np.sum((y_true == lab) & (y_pred != lab)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))
