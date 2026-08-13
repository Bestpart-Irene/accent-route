"""External comparison axis: report on a published benchmark's own label set.

The project's 8-class taxonomy is custom, which makes its numbers impossible for a reader
to calibrate. Vox-Profile (arXiv 2505.14648) aggregates 11 corpora under a regional label
set with speaker-disjoint splits and publishes models, so reporting a second axis in its
vocabulary gives an external reference point.

The two label sets do NOT nest: ours is partly L1-based (L1-Mandarin), Vox-Profile's is
regional (East Asia). Only the unambiguous correspondences are mapped; everything else is
declared unmappable rather than forced.
"""

import pandas as pd
import pytest

from accentroute.eval.external import (
    VOX_PROFILE_NARROW,
    to_vox_profile,
    vox_profile_report,
)


class TestLabelMapping:
    def test_unambiguous_correspondences(self):
        assert to_vox_profile("en-US") == "North America"
        assert to_vox_profile("en-GB") == "English"
        assert to_vox_profile("en-AU") == "Oceania"
        assert to_vox_profile("en-IN") == "South Asia"

    def test_l1_classes_map_to_regional_groups(self):
        assert to_vox_profile("L1-Mandarin") == "East Asia"
        assert to_vox_profile("L1-Korean") == "East Asia"
        assert to_vox_profile("L1-Spanish") == "Romance"
        assert to_vox_profile("L1-Arabic") == "Semitic"

    def test_targets_are_valid_vox_profile_labels(self):
        for cls in ["en-US", "en-GB", "en-AU", "en-IN",
                    "L1-Mandarin", "L1-Spanish", "L1-Korean", "L1-Arabic"]:
            assert to_vox_profile(cls) in VOX_PROFILE_NARROW

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            to_vox_profile("en-ZZ")


class TestReport:
    def test_collapses_predictions_and_scores(self):
        y_true = pd.Series(["L1-Mandarin", "L1-Korean", "en-US", "en-GB"])
        y_pred = pd.Series(["L1-Korean", "L1-Mandarin", "en-US", "en-GB"])
        rep = vox_profile_report(y_true, y_pred)
        # Mandarin/Korean confusion vanishes under the regional label set — both are
        # East Asia — so the external axis scores it as correct.
        assert rep["vox_profile_macro_f1"] == pytest.approx(1.0)
        assert rep["n_clips"] == 4

    def test_reports_collapsed_pairs_explicitly(self):
        """Collapsing hides real errors, so the report has to name what it merged."""
        y_true = pd.Series(["L1-Mandarin", "L1-Korean"])
        y_pred = pd.Series(["L1-Korean", "L1-Mandarin"])
        rep = vox_profile_report(y_true, y_pred)
        assert rep["collapsed_groups"]["East Asia"] == ["L1-Korean", "L1-Mandarin"]
        assert rep["n_errors_hidden_by_collapse"] == 2

    def test_no_collapse_when_groups_are_singletons(self):
        y_true = pd.Series(["en-US", "en-GB"])
        y_pred = pd.Series(["en-GB", "en-US"])
        rep = vox_profile_report(y_true, y_pred)
        assert rep["n_errors_hidden_by_collapse"] == 0
        assert rep["vox_profile_macro_f1"] == pytest.approx(0.0)

    def test_report_labels_are_vox_profile_vocabulary(self):
        y = pd.Series(["en-US", "L1-Spanish"])
        rep = vox_profile_report(y, y)
        assert set(rep["classes"]) <= set(VOX_PROFILE_NARROW)
