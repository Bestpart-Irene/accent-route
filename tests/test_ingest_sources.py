"""T4: the three core source adapters (Common Voice / L2-ARCTIC / EdAcc).

The fixtures are miniature copies of each source's real directory layout. An adapter only
enumerates clips and carries metadata — label mapping belongs to the taxonomy — but CV
also reports the fill rate of its `accents` column.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from accentroute.ingest.common_voice import CommonVoiceIngestor
from accentroute.ingest.edacc import EdAccIngestor
from accentroute.ingest.l2_arctic import L2ArcticIngestor


def _write_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, 0.3 * np.sin(2 * np.pi * 220 * t), sr)


# ── Common Voice ──────────────────────────────────────────────────────────


@pytest.fixture()
def cv_root(tmp_path: Path) -> Path:
    root = tmp_path / "cv"
    for name in ["a", "b", "c", "d"]:
        _write_wav(root / "clips" / f"{name}.wav")
    tsv = pd.DataFrame(
        {
            "client_id": ["spk1", "spk1", "spk2", "spk3"],
            "path": ["a.wav", "b.wav", "c.wav", "d.wav"],
            "sentence": ["s1", "s2", "s3", "s4"],
            "accents": [
                "United States English",
                "",  # no self-report → skipped
                "United States English,England English",  # multi-value → skipped and counted
                "India and South Asia (India, Pakistan, Sri Lanka)",
            ],
        }
    )
    tsv.to_csv(root / "validated.tsv", sep="\t", index=False)
    return root


class TestCommonVoice:
    def test_records(self, cv_root):
        ing = CommonVoiceIngestor(root=cv_root)
        records = list(ing.iter_records())
        assert len(records) == 2  # only a and d carry a single self-reported accent
        by_id = {r["clip_id"]: r for r in records}
        a = by_id["cv:a"]
        assert a["accent_raw"] == "United States English"
        assert a["speaker_id_raw"] == "spk1"
        assert a["license"] == "CC0-1.0"
        assert a["source"] == "common_voice"
        assert a["sample_rate_orig"] == 16000
        assert a["duration_s"] == pytest.approx(1.0, abs=0.01)
        assert a["offset_start_s"] == 0.0
        assert a["offset_end_s"] == pytest.approx(1.0, abs=0.01)

    def test_accent_fill_rate_stats(self, cv_root):
        ing = CommonVoiceIngestor(root=cv_root)
        list(ing.iter_records())
        assert ing.stats == {
            "n_rows": 4,
            "n_no_accent": 1,
            "n_multi_accent_skipped": 1,
            "n_yielded": 2,
            "accent_fill_rate": 0.75,  # 3 of the 4 rows carry accents (multi-value included)
        }


# ── L2-ARCTIC ─────────────────────────────────────────────────────────────


@pytest.fixture()
def l2_root(tmp_path: Path) -> Path:
    root = tmp_path / "l2arctic"
    for spk in ["BWC", "YKWK", "ASI"]:
        for utt in ["arctic_a0001", "arctic_a0002"]:
            _write_wav(root / spk / "wav" / f"{utt}.wav")
    return root


class TestL2Arctic:
    SPEAKER_L1: ClassVar[dict[str, str]] = {
        "BWC": "Mandarin", "YKWK": "Korean", "ASI": "Hindi",
    }

    def test_records(self, l2_root):
        ing = L2ArcticIngestor(root=l2_root, speaker_l1=self.SPEAKER_L1)
        records = list(ing.iter_records())
        assert len(records) == 6
        by_id = {r["clip_id"]: r for r in records}
        r = by_id["l2arctic:BWC:arctic_a0001"]
        assert r["accent_raw"] == "Mandarin"
        assert r["speaker_id_raw"] == "BWC"
        assert r["license"] == "CC-BY-NC-4.0"
        # Hindi speakers are still emitted; the taxonomy stage drops and counts them,
        # ingest never quietly cuts rows
        assert by_id["l2arctic:ASI:arctic_a0001"]["accent_raw"] == "Hindi"

    def test_unknown_speaker_dir_raises(self, l2_root):
        (l2_root / "XXX" / "wav").mkdir(parents=True)
        _write_wav(l2_root / "XXX" / "wav" / "arctic_a0001.wav")
        ing = L2ArcticIngestor(root=l2_root, speaker_l1=self.SPEAKER_L1)
        with pytest.raises(KeyError, match="XXX"):
            list(ing.iter_records())


# ── EdAcc ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def edacc_root(tmp_path: Path) -> Path:
    root = tmp_path / "edacc"
    _write_wav(root / "data" / "conv1.wav", seconds=30.0)
    pd.DataFrame(
        {
            "segment_id": ["conv1-001", "conv1-002"],
            "audio_file": ["data/conv1.wav", "data/conv1.wav"],
            "speaker_id": ["P001", "P002"],
            "start_s": [0.5, 12.0],
            "end_s": [8.5, 20.0],
        }
    ).to_csv(root / "segments.csv", index=False)
    pd.DataFrame(
        {
            "speaker_id": ["P001", "P002"],
            "accent": ["Southern British English", "Latin American English"],
            "l1": ["English", "Spanish"],
        }
    ).to_csv(root / "speakers.csv", index=False)
    return root


class TestEdAcc:
    def test_l1_precedence_rule(self, edacc_root):
        """L2 speakers (non-English l1) take accent_raw from l1; natives use the accent field."""
        ing = EdAccIngestor(root=edacc_root)
        by_spk = {r["speaker_id_raw"]: r for r in ing.iter_records()}
        assert by_spk["P001"]["accent_raw"] == "Southern British English"
        assert by_spk["P002"]["accent_raw"] == "Spanish"

    def test_segment_offsets(self, edacc_root):
        ing = EdAccIngestor(root=edacc_root)
        r = next(iter(ing.iter_records()))
        assert r["clip_id"] == "edacc:conv1-001"
        assert r["offset_start_s"] == 0.5
        assert r["offset_end_s"] == 8.5
        assert r["duration_s"] == pytest.approx(8.0)
        assert r["orig_file"] == "data/conv1.wav"
        assert r["license"] == "CC-BY-SA-4.0"
