"""指标:独立实现(不依赖 sklearn),单测对照 sklearn 参考值。

labels 默认锁定 8 类;supported-class macro-F1(EdAcc 排除类后)显式传子集,
两者不可横比 —— 表格生成端(T15)负责命名区分。
"""

from collections.abc import Sequence

import numpy as np

from accentroute.schema import ACCENTS


def confusion(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str] = tuple(ACCENTS)
) -> np.ndarray:
    """行=真实类,列=预测类。"""
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            m[index[t], index[p]] += 1
    return m


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str] = tuple(ACCENTS)
) -> float:
    """逐类 F1 的未加权均值;无支持且无预测的类记 0(与 sklearn zero_division=0 一致)。"""
    m = confusion(y_true, y_pred, labels)
    tp = np.diag(m).astype(np.float64)
    fp = m.sum(axis=0) - tp
    fn = m.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())
