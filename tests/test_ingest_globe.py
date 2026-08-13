"""GLOBE (`MushanW/GLOBE_V2`): Common-Voice-derived, CC0, with speaker_id and duration.

GLOBE supplies the four native varieties — including en-AU, which no other reachable
corpus covers — but structurally none of the L1-* classes: its accent field is Common
Voice's self-reported *variety*, not the speaker's first language.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from accentroute.ingest.globe import GlobeIngestor, quota_plan
from accentroute.schema import validate_manifest

SR = 44100  # GLOBE V2 ships 44.1 kHz


def _flac(seconds: float) -> dict:
    n = int(seconds * SR)
    wav = 0.2 * np.sin(2 * np.pi * 180 * np.arange(n) / SR)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="FLAC")
    return {"bytes": buf.getvalue(), "path": "x.flac"}


@pytest.fixture()
def globe_dir(tmp_path: Path) -> Path:
    root = tmp_path / "globe" / "data"
    root.mkdir(parents=True)
    rows = [
        {"speaker_id": "S_1", "accent": "United States English", "duration": 6.0,
         "transcript": "a", "age": "twenties", "gender": "female", "audio": _flac(6.0)},
        {"speaker_id": "S_2", "accent": "Australian English", "duration": 7.0,
         "transcript": "b", "age": "thirties", "gender": "male", "audio": _flac(7.0)},
        # multi-valued self-report: ambiguous, skipped like the Common Voice ingestor
        {"speaker_id": "S_3", "accent": "United States English,England English",
         "duration": 5.0, "transcript": "c", "age": "forties", "gender": "male",
         "audio": _flac(5.0)},
        # empty self-report
        {"speaker_id": "S_4", "accent": "", "duration": 5.0, "transcript": "d",
         "age": "fifties", "gender": "female", "audio": _flac(5.0)},
    ]
    pd.DataFrame(rows).to_parquet(root / "train-00000-of-00002.parquet")
    return tmp_path / "globe"


class TestIngestor:
    def test_only_single_valued_self_reports_yielded(self, globe_dir):
        records = list(GlobeIngestor(root=globe_dir).iter_records())
        assert len(records) == 2
        assert {r["accent_raw"] for r in records} == {
            "United States English", "Australian English",
        }

    def test_valid_raw_manifest(self, globe_dir):
        records = list(GlobeIngestor(root=globe_dir).iter_records())
        validate_manifest(pd.DataFrame.from_records(records), stage="raw")

    def test_metadata_fields(self, globe_dir):
        r = next(iter(GlobeIngestor(root=globe_dir).iter_records()))
        assert r["source"] == "globe"
        assert r["license"] == "CC0-1.0"
        assert r["speaker_id_raw"] == "S_1"
        assert r["sample_rate_orig"] == SR
        assert r["duration_s"] == pytest.approx(6.0, abs=0.05)
        assert r["clip_id"].startswith("globe:train:00000:")

    def test_stats_report_skips(self, globe_dir):
        ing = GlobeIngestor(root=globe_dir)
        list(ing.iter_records())
        assert ing.stats["n_rows"] == 4
        assert ing.stats["n_no_accent"] == 1
        assert ing.stats["n_multi_accent_skipped"] == 1
        assert ing.stats["n_yielded"] == 2

    def test_clip_ids_unique_across_shards(self, globe_dir):
        pd.DataFrame([{
            "speaker_id": "S_9", "accent": "England English", "duration": 6.0,
            "transcript": "e", "age": "twenties", "gender": "male", "audio": _flac(6.0),
        }]).to_parquet(globe_dir / "data" / "train-00001-of-00002.parquet")
        ids = [r["clip_id"] for r in GlobeIngestor(root=globe_dir).iter_records()]
        assert len(ids) == len(set(ids)) == 3


class TestQuotaPlan:
    """GLOBE is 122 GB and wildly imbalanced (en-US ~15k speakers, en-IN ~600), so
    materializing it whole is neither affordable nor useful."""

    def test_caps_per_class(self):
        counts = {"en-US": 24617, "en-GB": 4760, "en-AU": 1419, "en-IN": 1450}
        plan = quota_plan(counts, per_class=1200)
        assert plan == {"en-US": 1200, "en-GB": 1200, "en-AU": 1200, "en-IN": 1200}

    def test_never_exceeds_availability(self):
        plan = quota_plan({"en-AU": 800, "en-US": 24617}, per_class=1200)
        assert plan["en-AU"] == 800
        assert plan["en-US"] == 1200

    def test_empty_input(self):
        assert quota_plan({}, per_class=100) == {}
