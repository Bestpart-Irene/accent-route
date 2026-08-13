"""T2: accent_raw → 8-class mapping (versioned YAML; off-whitelist values dropped, counted)."""

from pathlib import Path

import pytest

from accentroute.schema import ACCENTS
from accentroute.taxonomy import Taxonomy, load_taxonomy

TAXONOMY_V1 = Path(__file__).parent.parent / "configs" / "taxonomy_v1.yaml"


@pytest.fixture()
def tax() -> Taxonomy:
    return load_taxonomy(TAXONOMY_V1)


class TestMapping:
    def test_version(self, tax):
        assert tax.version == "v1"

    def test_cv_us_english(self, tax):
        assert tax.map("united states english") == "en-US"

    def test_case_and_whitespace_robust(self, tax):
        assert tax.map("  United  States   English ") == "en-US"

    def test_l2_arctic_l1_values(self, tax):
        assert tax.map("Mandarin") == "L1-Mandarin"
        assert tax.map("Korean") == "L1-Korean"
        assert tax.map("Arabic") == "L1-Arabic"
        assert tax.map("Spanish") == "L1-Spanish"

    def test_unmapped_returns_none(self, tax):
        assert tax.map("scottish english") is None

    def test_unmapped_is_counted(self, tax):
        tax.map("scottish english")
        tax.map("Scottish English")  # same normalized key
        tax.map("filipino")
        assert tax.unmapped_counts["scottish english"] == 2
        assert tax.unmapped_counts["filipino"] == 1

    def test_empty_and_none_input(self, tax):
        assert tax.map("") is None
        assert tax.map(None) is None
        # empty input is not counted as unmapped (missing ≠ off-whitelist)
        assert "" not in tax.unmapped_counts


class TestIntegrity:
    def test_all_targets_are_valid_accents(self, tax):
        assert set(tax.mapping.values()) <= set(ACCENTS)

    def test_every_accent_class_reachable(self, tax):
        assert set(tax.mapping.values()) == set(ACCENTS)

    def test_keys_are_normalized(self, tax):
        for key in tax.mapping:
            assert key == " ".join(key.lower().split()), f"non-normalized key: {key!r}"
