"""Streaming a balanced GLOBE subset.

GLOBE is 122 GB and 69% en-US, so it is streamed under two caps rather than downloaded:
per class, and per speaker. The per-speaker cap is the one that matters for this task —
1,200 clips from 12 speakers would teach the model voices, not accents.
"""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from accentroute.ingest.globe import fetch_globe_subset
from accentroute.schema import validate_manifest
from accentroute.taxonomy import Taxonomy

SR = 44100


def _row(speaker: str, accent: str, seconds: float = 6.0) -> dict:
    n = int(seconds * SR)
    wav = 0.2 * np.sin(2 * np.pi * 180 * np.arange(n) / SR)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="FLAC")
    return {
        "speaker_id": speaker,
        "accent": accent,
        "duration": seconds,
        "audio": {"bytes": buf.getvalue(), "path": f"{speaker}.flac"},
    }


@pytest.fixture()
def tax() -> Taxonomy:
    return Taxonomy(version="v1", mapping={
        "united states english": "en-US",
        "australian english": "en-AU",
    })


def _stream(n_us_speakers=6, clips_each=10, n_au_speakers=2):
    for s in range(n_us_speakers):
        for _ in range(clips_each):
            yield _row(f"US_{s}", "United States English")
    for s in range(n_au_speakers):
        for _ in range(clips_each):
            yield _row(f"AU_{s}", "Australian English")


class TestQuotas:
    def test_per_class_cap(self, tmp_path, tax):
        df = fetch_globe_subset(tmp_path, tax, per_class=15, max_per_speaker=100,
                                row_iter=_stream())
        assert (df.accent_label == "en-US").sum() == 15
        assert (df.accent_label == "en-AU").sum() == 15

    def test_per_speaker_cap_forces_speaker_diversity(self, tmp_path, tax):
        """The point of the cap: 30 clips must come from 6 speakers, not 3."""
        df = fetch_globe_subset(tmp_path, tax, per_class=30, max_per_speaker=5,
                                row_iter=_stream())
        us = df[df.accent_label == "en-US"]
        assert len(us) == 30
        assert us.speaker_id_raw.nunique() == 6
        assert us.groupby("speaker_id_raw").size().max() == 5

    def test_takes_all_when_supply_is_short(self, tmp_path, tax):
        df = fetch_globe_subset(tmp_path, tax, per_class=500, max_per_speaker=5,
                                row_iter=_stream())
        # 6 US speakers x 5 = 30, 2 AU speakers x 5 = 10
        assert (df.accent_label == "en-US").sum() == 30
        assert (df.accent_label == "en-AU").sum() == 10

    def test_unmapped_accents_skipped(self, tmp_path, tax):
        stream = [_row("X_0", "Scottish English"), *list(_stream(1, 2, 1))]
        df = fetch_globe_subset(tmp_path, tax, per_class=10, max_per_speaker=10,
                                row_iter=iter(stream))
        assert "X_0" not in set(df.speaker_id_raw)

    def test_stops_early_once_every_expected_label_is_full(self, tmp_path, tax):
        """Streaming 122 GB to the end after the quotas are full would defeat the point."""
        consumed = 0

        def counting():
            nonlocal consumed
            for row in _stream(n_us_speakers=50, clips_each=10, n_au_speakers=50):
                consumed += 1
                yield row

        fetch_globe_subset(tmp_path, tax, per_class=10, max_per_speaker=2,
                           row_iter=counting(), expected_labels={"en-US", "en-AU"})
        assert consumed < 600, f"streamed {consumed} rows after every quota was full"

    def test_does_not_stop_before_a_later_class_appears(self, tmp_path, tax):
        """GLOBE's stream is not interleaved by accent. Stopping on 'every label seen so
        far is full' would return zero en-AU here, silently."""
        df = fetch_globe_subset(tmp_path, tax, per_class=10, max_per_speaker=10,
                                row_iter=_stream(), expected_labels={"en-US", "en-AU"})
        assert (df.accent_label == "en-AU").sum() == 10

    def test_max_rows_backstop(self, tmp_path, tax):
        df = fetch_globe_subset(tmp_path, tax, per_class=1000, max_per_speaker=1000,
                                row_iter=_stream(), max_rows=12)
        assert len(df) == 12


class TestOutput:
    def test_manifest_is_raw_valid(self, tmp_path, tax):
        df = fetch_globe_subset(tmp_path, tax, per_class=10, max_per_speaker=5,
                                row_iter=_stream())
        validate_manifest(df.drop(columns=["accent_label"]), stage="raw")

    def test_writes_16k_mono_wavs(self, tmp_path, tax):
        df = fetch_globe_subset(tmp_path, tax, per_class=5, max_per_speaker=5,
                                row_iter=_stream())
        for clip_id in df.clip_id:
            data, sr = sf.read(Path(tmp_path) / f"{clip_id}.wav")
            assert sr == 16000
            assert data.ndim == 1

    def test_clip_ids_unique(self, tmp_path, tax):
        df = fetch_globe_subset(tmp_path, tax, per_class=20, max_per_speaker=4,
                                row_iter=_stream())
        assert df.clip_id.is_unique
