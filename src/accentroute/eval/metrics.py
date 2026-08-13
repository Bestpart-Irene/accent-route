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
    """逐类 F1 的未加权均值;无支持且无预测的类记 0(与 sklearn zero_division=0 一致)。

    不走 confusion 矩阵:labels 取子集时,预测值落在子集外的样本必须计入
    真实类的 FN(否则 supported-class macro-F1 虚高),逐类直接数 TP/FP/FN。
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
