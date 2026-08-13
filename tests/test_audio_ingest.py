"""T3: audio normalization (to 16 kHz mono PCM16) and the ingest base class."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandera.errors
import pytest
import soundfile as sf

from accentroute.audio import to_wav16k_mono
from accentroute.ingest.base import SourceIngestor, run_ingest
from accentroute.schema import validate_manifest


@pytest.fixture()
def stereo_44k(tmp_path: Path) -> Path:
    """Two seconds of 44.1 kHz stereo; the channels are sines at different frequencies."""
    sr = 44100
    t = np.arange(2 * sr) / sr
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    path = tmp_path / "stereo44k.wav"
    sf.write(path, np.stack([left, right], axis=1), sr)
    return path


class TestToWav16kMono:
    def test_converts_to_16k_mono(self, stereo_44k, tmp_path):
        dst = tmp_path / "out.wav"
        meta = to_wav16k_mono(stereo_44k, dst)
        data, sr = sf.read(dst)
        assert sr == 16000
        assert data.ndim == 1
        assert meta.sample_rate_orig == 44100
        assert meta.n_channels_orig == 2
        assert meta.duration_s == pytest.approx(2.0, abs=0.01)
        assert len(data) == pytest.approx(2 * 16000, abs=16)

    def test_output_is_pcm16(self, stereo_44k, tmp_path):
        dst = tmp_path / "out.wav"
        to_wav16k_mono(stereo_44k, dst)
        assert sf.info(dst).subtype == "PCM_16"

    def test_slice_offsets_exact(self, stereo_44k, tmp_path):
        dst = tmp_path / "slice.wav"
        meta = to_wav16k_mono(stereo_44k, dst, start_s=0.5, end_s=1.5)
        data, _ = sf.read(dst)
        assert meta.duration_s == pytest.approx(1.0, abs=0.005)
        assert len(data) == pytest.approx(16000, abs=16)

    def test_slice_beyond_eof_raises(self, stereo_44k, tmp_path):
        with pytest.raises(ValueError, match="beyond"):
            to_wav16k_mono(stereo_44k, tmp_path / "x.wav", start_s=1.0, end_s=5.0)


def _record(clip_id: str) -> dict:
    return {
        "clip_id": clip_id,
        "source": "common_voice",
        "source_uri": "hf://mozilla-foundation/common_voice_17_0",
        "orig_file": f"clips/{clip_id}.mp3",
        "offset_start_s": 0.0,
        "offset_end_s": 6.0,
        "sample_rate_orig": 48000,
        "duration_s": 6.0,
        "license": "CC0-1.0",
        "speaker_id_raw": "spk1",
        "accent_raw": "united states english",
    }


class DummyIngestor(SourceIngestor):
    source = "common_voice"
    license = "CC0-1.0"

    def __init__(self, records: list[dict]):
        self._records = records

    def iter_records(self) -> Iterator[dict]:
        yield from self._records


class TestRunIngest:
    def test_writes_valid_raw_manifest(self, tmp_path):
        out = run_ingest(DummyIngestor([_record("a"), _record("b")]), tmp_path / "raw.parquet")
        import pandas as pd

        df = pd.read_parquet(out)
        validate_manifest(df, stage="raw")
        assert len(df) == 2

    def test_invalid_records_rejected(self, tmp_path):
        with pytest.raises(pandera.errors.SchemaErrors):
            run_ingest(
                DummyIngestor([_record("a"), _record("a")]),  # duplicate clip_id
                tmp_path / "raw.parquet",
            )

    def test_empty_ingestor_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no records"):
            run_ingest(DummyIngestor([]), tmp_path / "raw.parquet")
