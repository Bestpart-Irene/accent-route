"""T5: source × accent 混杂矩阵、confounded 标记、EdAcc supported-class 覆盖。"""

import pandas as pd
import pytest

from accentroute.reports.coverage_confounding import (
    edacc_class_coverage,
    flag_confounded,
    source_accent_matrix,
)


def _mini_manifest() -> pd.DataFrame:
    """en-US 双源均衡;L1-Korean 95% 时长来自单源(应标 confounded)。"""
    rows = [
        # en-US: cv 2 speakers 2h, edacc… 不掺,用第二源 l2_arctic 模拟均衡双源
        {"source": "common_voice", "accent_label": "en-US", "speaker_id_raw": "s1", "duration_s": 3600.0},
        {"source": "common_voice", "accent_label": "en-US", "speaker_id_raw": "s2", "duration_s": 3600.0},
        {"source": "l2_arctic", "accent_label": "en-US", "speaker_id_raw": "x1", "duration_s": 7200.0},
        # L1-Korean: cv 只占 5%
        {"source": "l2_arctic", "accent_label": "L1-Korean", "speaker_id_raw": "k1", "duration_s": 9500.0},
        {"source": "common_voice", "accent_label": "L1-Korean", "speaker_id_raw": "k2", "duration_s": 500.0},
    ]
    return pd.DataFrame(rows)


class TestSourceAccentMatrix:
    def test_matrix_values(self):
        m = source_accent_matrix(_mini_manifest())
        row = m[(m["source"] == "common_voice") & (m["accent_label"] == "en-US")].iloc[0]
        assert row["n_speakers"] == 2
        assert row["n_clips"] == 2
        assert row["hours"] == pytest.approx(2.0)

    def test_matrix_covers_all_pairs_present(self):
        m = source_accent_matrix(_mini_manifest())
        assert len(m) == 4  # (cv,en-US) (l2,en-US) (l2,Korean) (cv,Korean)


class TestFlagConfounded:
    def test_dominant_single_source_flagged(self):
        report = flag_confounded(source_accent_matrix(_mini_manifest()), dominance=0.9)
        by_label = report.set_index("accent_label")
        assert bool(by_label.loc["L1-Korean", "confounded"]) is True
        assert by_label.loc["L1-Korean", "dominant_source"] == "l2_arctic"
        assert by_label.loc["L1-Korean", "dominant_share"] == pytest.approx(0.95)

    def test_balanced_class_not_flagged(self):
        report = flag_confounded(source_accent_matrix(_mini_manifest()), dominance=0.9)
        by_label = report.set_index("accent_label")
        assert bool(by_label.loc["en-US", "confounded"]) is False


class TestEdAccCoverage:
    def _edacc_df(self) -> pd.DataFrame:
        rows = []
        # en-GB: 6 speakers → include;L1-Korean: 2 speakers → exclude
        for i in range(6):
            rows.append({"source": "edacc", "accent_label": "en-GB",
                         "speaker_id_raw": f"g{i}", "duration_s": 600.0})
        for i in range(2):
            rows.append({"source": "edacc", "accent_label": "L1-Korean",
                         "speaker_id_raw": f"k{i}", "duration_s": 600.0})
        return pd.DataFrame(rows)

    def test_include_flag(self):
        cov = edacc_class_coverage(self._edacc_df(), min_speakers=5).set_index("accent_label")
        assert bool(cov.loc["en-GB", "include"]) is True
        assert bool(cov.loc["L1-Korean", "include"]) is False

    def test_all_eight_classes_listed_with_zero_fill(self):
        cov = edacc_class_coverage(self._edacc_df(), min_speakers=5).set_index("accent_label")
        assert len(cov) == 8
        assert cov.loc["en-AU", "n_speakers"] == 0
        assert bool(cov.loc["en-AU", "include"]) is False

    def test_non_edacc_rows_rejected(self):
        df = self._edacc_df()
        df.loc[0, "source"] = "common_voice"
        with pytest.raises(ValueError, match="edacc"):
            edacc_class_coverage(df)
