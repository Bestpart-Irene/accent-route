"""三个基线:多数类、frozen ECAPA embedding probe、Qwen2-Audio 零样本。

- ecapa_embedding_probe 的命名即措辞纪律:冻结 embedding + 逻辑回归探针,
  不是完整 ECAPA-TDNN 口音模型,报告里必须这么称呼。
- qwen_zero_shot 与弱标注(T13)共用 pin 的 revision + prompt;
  generate_fn 注入,单测与 CI 零下载。
"""

import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from accentroute.schema import ACCENTS

GenerateFn = Callable[[str, str], str]  # (clip_id, prompt) -> 模型输出文本


def majority_baseline(train_labels: np.ndarray, n_test: int) -> np.ndarray:
    values, counts = np.unique(train_labels, return_counts=True)
    return np.full(n_test, values[counts.argmax()])


def ecapa_embedding_probe(
    train_emb: np.ndarray, y_train: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    """冻结 ECAPA embedding + 逻辑回归探针(绝不 fine-tune)。"""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_emb, y_train)
    return clf.predict(test_emb)


_LABEL_PATTERNS = [(lab, re.compile(re.escape(lab), re.IGNORECASE)) for lab in ACCENTS]


def parse_qwen_label(text: str) -> str:
    """强制 8 选 1 + unsure 的输出解析:恰好命中一个标签才算数,否则 unsure。"""
    hits = [lab for lab, pat in _LABEL_PATTERNS if pat.search(text)]
    if len(hits) == 1:
        return hits[0]
    return "unsure"


def qwen_zero_shot(
    df: pd.DataFrame, *, generate_fn: GenerateFn, prompt: str
) -> np.ndarray:
    """逐 clip 生成 → 解析。unsure 保留原样(评测端把 unsure 当错判计分)。"""
    return np.array(
        [parse_qwen_label(generate_fn(clip_id, prompt)) for clip_id in df["clip_id"]]
    )
