"""EdAcc as published on Hugging Face: pre-segmented clips with embedded audio.

The DataShare release ships long conversations plus segments.csv/speakers.csv; the HF
release (`edinburghcstr/edacc`) ships one row per clip with the audio inline and columns
speaker/text/accent/raw_accent/gender/l1. Two layouts, two ingestors.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from accentroute.ingest.edacc_hf import EdAccHFIngestor, extract_audio
from accentroute.schema import validate_manifest

SR = 16000


def _audio_struct(seconds: float, path: str) -> dict:
    """The real on-disk form: HF stores the Audio feature as encoded file bytes.

    Reading the parquet directly with pandas yields {"bytes", "path"} — NOT the decoded
    {"array", "sampling_rate"} that `datasets` produces after decoding. The first version
    of this fixture used the decoded form, so the tests passed while the real corpus
    crashed with KeyError: 'array'.
    """
    n = int(seconds * SR)
    wav = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return {"bytes": buf.getvalue(), "path": path}


def _decoded_audio_struct(seconds: float, path: str) -> dict:
    """The decoded form, as produced by `datasets` — also supported."""
    n = int(seconds * SR)
    return {"array": 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR),
            "sampling_rate": SR, "path": path}


@pytest.fixture()
def edacc_dir(tmp_path: Path) -> Path:
    root = tmp_path / "edacc_hf" / "data"
    root.mkdir(parents=True)
    rows = [
        {"speaker": "P001", "text": "hello there", "accent": "Southern British English",
         "raw_accent": "british", "gender": "F", "l1": "English",
         "audio": _audio_struct(6.0, "P001_0.wav")},
        {"speaker": "P002", "text": "good morning", "accent": "Latin American",
         "raw_accent": "spanish", "gender": "M", "l1": "Spanish",
         "audio": _audio_struct(7.5, "P002_0.wav")},
        {"speaker": "P003", "text": "a third one", "accent": "Irish English",
         "raw_accent": "irish", "gender": "M", "l1": "English",
         "audio": _audio_struct(5.0, "P003_0.wav")},
    ]
    pd.DataFrame(rows).to_parquet(root / "validation-00000-of-00001.parquet")
    return tmp_path / "edacc_hf"


class TestIngestor:
    def test_produces_valid_raw_manifest(self, edacc_dir):
        records = list(EdAccHFIngestor(root=edacc_dir).iter_records())
        assert len(records) == 3
        validate_manifest(pd.DataFrame.from_records(records), stage="raw")

    def test_l1_precedence_rule(self, edacc_dir):
        """Same rule as the DataShare ingestor: L2 speakers are labeled by l1, native
        speakers by the linguist-standardized accent field."""
        by_spk = {r["speaker_id_raw"]: r for r in EdAccHFIngestor(root=edacc_dir).iter_records()}
        assert by_spk["P001"]["accent_raw"] == "Southern British English"
        assert by_spk["P002"]["accent_raw"] == "Spanish"  # l1 wins for L2 speakers
        assert by_spk["P003"]["accent_raw"] == "Irish English"  # dropped later by taxonomy

    def test_metadata_fields(self, edacc_dir):
        r = next(iter(EdAccHFIngestor(root=edacc_dir).iter_records()))
        assert r["source"] == "edacc"
        assert r["license"] == "CC-BY-SA-4.0"
        assert r["sample_rate_orig"] == SR
        assert r["duration_s"] == pytest.approx(6.0, abs=0.01)
        assert r["offset_start_s"] == 0.0
        assert r["clip_id"].startswith("edacc:")

    def test_clip_ids_unique_across_shards(self, edacc_dir):
        rows = [{"speaker": "P009", "text": "x", "accent": "Southern British English",
                 "raw_accent": "b", "gender": "F", "l1": "English",
                 "audio": _audio_struct(6.0, "P009_0.wav")}]
        pd.DataFrame(rows).to_parquet(edacc_dir / "data" / "test-00000-of-00001.parquet")
        ids = [r["clip_id"] for r in EdAccHFIngestor(root=edacc_dir).iter_records()]
        assert len(ids) == len(set(ids)) == 4

    def test_split_recorded_in_source_uri(self, edacc_dir):
        r = next(iter(EdAccHFIngestor(root=edacc_dir).iter_records()))
        assert "validation" in r["source_uri"] or "edinburghcstr/edacc" in r["source_uri"]


class TestExtractAudio:
    def test_writes_16k_mono_wav_per_clip(self, edacc_dir, tmp_path):
        out = tmp_path / "wav"
        n = extract_audio(edacc_dir, out)
        assert n == 3
        wavs = sorted(out.glob("*.wav"))
        assert len(wavs) == 3
        data, sr = sf.read(wavs[0])
        assert sr == SR
        assert data.ndim == 1

    def test_filenames_match_clip_ids(self, edacc_dir, tmp_path):
        out = tmp_path / "wav"
        extract_audio(edacc_dir, out)
        clip_ids = {r["clip_id"] for r in EdAccHFIngestor(root=edacc_dir).iter_records()}
        assert {p.stem for p in out.glob("*.wav")} == clip_ids

    def test_skips_existing(self, edacc_dir, tmp_path):
        out = tmp_path / "wav"
        extract_audio(edacc_dir, out)
        assert extract_audio(edacc_dir, out, skip_existing=True) == 0

    def test_audio_content_round_trips(self, edacc_dir, tmp_path):
        """Extraction must preserve the signal, not just produce a file of the right shape."""
        out = tmp_path / "wav"
        extract_audio(edacc_dir, out)
        data, sr = sf.read(out / "edacc:validation:00000:000000.wav")
        expected = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(6.0 * SR)) / SR)
        assert sr == SR
        assert len(data) == len(expected)
        assert np.max(np.abs(data - expected)) < 1e-3


class TestDecodedAudioForm:
    """`datasets` hands back decoded arrays; both forms must work."""

    @pytest.fixture()
    def decoded_dir(self, tmp_path: Path) -> Path:
        root = tmp_path / "dec" / "data"
        root.mkdir(parents=True)
        pd.DataFrame([{
            "speaker": "P010", "text": "t", "accent": "Southern British English",
            "raw_accent": "b", "gender": "F", "l1": "English",
            "audio": _decoded_audio_struct(6.0, "P010_0.wav"),
        }]).to_parquet(root / "validation-00000-of-00001.parquet")
        return tmp_path / "dec"

    def test_ingest_and_extract(self, decoded_dir, tmp_path):
        records = list(EdAccHFIngestor(root=decoded_dir).iter_records())
        assert records[0]["duration_s"] == pytest.approx(6.0, abs=0.01)
        assert extract_audio(decoded_dir, tmp_path / "w") == 1
