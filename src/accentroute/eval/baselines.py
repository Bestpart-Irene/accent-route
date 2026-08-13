"""Three baselines: majority class, a frozen ECAPA embedding probe, and Qwen2-Audio
zero-shot.

- The name ecapa_embedding_probe *is* the wording discipline: it is a frozen embedding plus
  a logistic-regression probe, not a full ECAPA-TDNN accent model, and the report must call
  it exactly that.
- qwen_zero_shot shares the pinned revision + prompt with weak labeling (T13); generate_fn
  is injected, so unit tests and CI download nothing.
"""

import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from accentroute.schema import ACCENTS

GenerateFn = Callable[[str, str], str]  # (clip_id, prompt) -> raw model output text


def majority_baseline(train_labels: np.ndarray, n_test: int) -> np.ndarray:
    values, counts = np.unique(train_labels, return_counts=True)
    return np.full(n_test, values[counts.argmax()])


def ecapa_embedding_probe(
    train_emb: np.ndarray, y_train: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    """Frozen ECAPA embedding + logistic-regression probe (never fine-tuned)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_emb, y_train)
    return clf.predict(test_emb)


_LABEL_PATTERNS = [(lab, re.compile(re.escape(lab), re.IGNORECASE)) for lab in ACCENTS]


def parse_qwen_label(text: str) -> str:
    """Parse the forced 1-of-8 (plus unsure) output: exactly one label match counts,
    anything else is unsure."""
    hits = [lab for lab, pat in _LABEL_PATTERNS if pat.search(text)]
    if len(hits) == 1:
        return hits[0]
    return "unsure"


def qwen_zero_shot(
    df: pd.DataFrame, *, generate_fn: GenerateFn, prompt: str
) -> np.ndarray:
    """Generate per clip → parse. unsure is kept as-is (evaluation scores it as a miss)."""
    return np.array(
        [parse_qwen_label(generate_fn(clip_id, prompt)) for clip_id in df["clip_id"]]
    )
