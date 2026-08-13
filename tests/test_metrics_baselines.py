"""T11: metrics (checked against sklearn) and the three baselines.

Naming is wording: ecapa_embedding_probe is a frozen ECAPA embedding probe, not a full
ECAPA-TDNN accent baseline, and the function name pins that down.
"""

import numpy as np
import pytest

from accentroute.eval.baselines import (
    ecapa_embedding_probe,
    majority_baseline,
    parse_qwen_label,
    qwen_zero_shot,
)
from accentroute.eval.metrics import confusion, macro_f1
from accentroute.schema import ACCENTS

sklearn_metrics = pytest.importorskip("sklearn.metrics")


class TestMetricsVsSklearn:
    def _data(self):
        rng = np.random.default_rng(0)
        y_true = rng.choice(ACCENTS, 200)
        y_pred = np.where(rng.random(200) < 0.6, y_true, rng.choice(ACCENTS, 200))
        return y_true, y_pred

    def test_macro_f1_matches_sklearn(self):
        y_true, y_pred = self._data()
        expected = sklearn_metrics.f1_score(
            y_true, y_pred, labels=ACCENTS, average="macro", zero_division=0
        )
        assert macro_f1(y_true, y_pred) == pytest.approx(expected)

    def test_macro_f1_subset_labels(self):
        """supported-class macro-F1: pass the label subset explicitly."""
        y_true, y_pred = self._data()
        subset = ACCENTS[:4]
        mask = np.isin(y_true, subset)
        expected = sklearn_metrics.f1_score(
            y_true[mask], y_pred[mask], labels=subset, average="macro", zero_division=0
        )
        assert macro_f1(y_true[mask], y_pred[mask], labels=subset) == pytest.approx(expected)

    def test_confusion_matches_sklearn(self):
        y_true, y_pred = self._data()
        expected = sklearn_metrics.confusion_matrix(y_true, y_pred, labels=ACCENTS)
        np.testing.assert_array_equal(confusion(y_true, y_pred), expected)


class TestMajorityBaseline:
    def test_predicts_most_common_train_class(self):
        train = np.array(["en-US"] * 5 + ["en-GB"] * 3)
        preds = majority_baseline(train, n_test=4)
        assert list(preds) == ["en-US"] * 4


class TestEcapaProbe:
    def test_separable_embeddings_learned(self):
        rng = np.random.default_rng(0)
        centers = {"en-US": [1, 0, 0], "en-GB": [0, 1, 0], "L1-Korean": [0, 0, 1]}
        labels = list(centers)
        y_train = np.array([labels[i % 3] for i in range(90)])
        x_train = np.stack([centers[y] + rng.normal(0, 0.05, 3) for y in y_train])
        y_test = np.array([labels[i % 3] for i in range(30)])
        x_test = np.stack([centers[y] + rng.normal(0, 0.05, 3) for y in y_test])
        preds = ecapa_embedding_probe(x_train, y_train, x_test)
        assert (preds == y_test).mean() > 0.95


class TestQwenZeroShot:
    def test_parse_exact_label(self):
        assert parse_qwen_label("en-US") == "en-US"
        assert parse_qwen_label("The accent is L1-Mandarin.") == "L1-Mandarin"

    def test_parse_unsure_and_garbage(self):
        assert parse_qwen_label("unsure") == "unsure"
        assert parse_qwen_label("I cannot tell") == "unsure"

    def test_parse_multiple_labels_is_unsure(self):
        assert parse_qwen_label("either en-US or en-GB") == "unsure"

    def test_zero_shot_harness(self):
        import pandas as pd

        df = pd.DataFrame({"clip_id": ["a", "b", "c"]})
        outputs = {"a": "en-AU", "b": "gibberish", "c": "It sounds like L1-Spanish"}
        preds = qwen_zero_shot(df, generate_fn=lambda clip_id, prompt: outputs[clip_id],
                               prompt="<test prompt>")
        assert list(preds) == ["en-AU", "unsure", "L1-Spanish"]
