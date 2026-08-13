"""A deliberately leaky split, used only to measure how much leakage inflates a score.

Clip-level random splitting is the common mistake in accent-classification work: the same
speaker lands in train and test, so the model can score highly by recognizing voices.
Quantifying that gap requires being able to reproduce the mistake on purpose.
"""

import pandas as pd
from conftest import make_manifest

from accentroute.split import assign_splits, assign_splits_leaky


class TestLeakySplit:
    def test_speakers_do_leak_across_splits(self):
        """Its defining property. A leaky split that did not leak would measure nothing."""
        out = assign_splits_leaky(make_manifest(), seed=17)
        assigned = out[out["split"].isin(["train", "val", "test"])]
        spanning = (assigned.groupby("speaker_key")["split"].nunique() > 1).sum()
        assert spanning > 0

    def test_speaker_disjoint_split_does_not_leak(self):
        """The contrast the demo rests on."""
        out = assign_splits(make_manifest(), seed=17, fixed_split_by_source={})
        assigned = out[out["split"].isin(["train", "val", "test"])]
        assert (assigned.groupby("speaker_key")["split"].nunique() == 1).all()

    def test_same_rows_are_assigned_by_both_rules(self):
        """Only the assignment differs, so a score gap cannot be blamed on dataset size."""
        leaky = assign_splits_leaky(make_manifest(), seed=17)
        strict = assign_splits(make_manifest(), seed=17, fixed_split_by_source={})
        assert set(leaky[leaky.split != "unassigned"].clip_id) == set(
            strict[strict.split != "unassigned"].clip_id
        )

    def test_deterministic(self):
        a = assign_splits_leaky(make_manifest(), seed=17)
        b = assign_splits_leaky(make_manifest(), seed=17)
        pd.testing.assert_series_equal(a["split"], b["split"])

    def test_output_passes_split_schema(self):
        from accentroute.schema import validate_manifest

        validate_manifest(assign_splits_leaky(make_manifest(), seed=17), stage="split")

    def test_rejected_rows_stay_unassigned(self):
        out = assign_splits_leaky(make_manifest(), seed=17)
        rejected = out[out["status"] == "rejected"]
        assert set(rejected["split"]) == {"unassigned"}

    def test_requires_explicit_opt_in(self):
        """It must be impossible to reach this rule by passing a stray argument to the
        real splitter."""
        import inspect

        sig = inspect.signature(assign_splits)
        assert "leaky" not in sig.parameters
        assert "allow_leakage" not in sig.parameters

    def test_docstring_warns(self):
        doc = assign_splits_leaky.__doc__ or ""
        assert "never" in doc.lower()
