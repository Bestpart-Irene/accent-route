"""T6: quality filtering → the qc manifest. Every model (VAD/Whisper/LID) is injected,
so CI downloads nothing.
"""

import math

import numpy as np
import pandas as pd
import pytest

from accentroute.filter import (
    FilterConfig,
    apply_filters,
    compute_vad_ratio,
    estimate_snr_proxy_db,
)
from accentroute.schema import validate_manifest
from accentroute.taxonomy import Taxonomy

SR = 16000


# ── injected fake models ──────────────────────────────────────


def fake_vad(wav: np.ndarray, sr: int) -> list[dict]:
    """Energy-based VAD stand-in: contiguous runs where |x| > 0.01."""
    mask = np.abs(wav) > 0.01
    ts, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            ts.append({"start": start, "end": i})
            start = None
    if start is not None:
        ts.append({"start": start, "end": len(wav)})
    return ts


def fake_transcribe(wav: np.ndarray) -> str:
    return "hello world this is a test"


def fake_lid_en(text: str) -> tuple[str, float]:
    return ("en", 0.99)


def make_taxonomy() -> Taxonomy:
    return Taxonomy(version="v1", mapping={"united states english": "en-US"})


# ── pure functions ────────────────────────────────────────────


class TestVadRatio:
    def test_half_speech(self):
        wav = np.zeros(2 * SR)
        wav[:SR] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
        ratio = compute_vad_ratio(wav, SR, get_speech_ts=fake_vad)
        assert ratio == pytest.approx(0.5, abs=0.02)

    def test_silence_is_zero(self):
        assert compute_vad_ratio(np.zeros(SR), SR, get_speech_ts=fake_vad) == 0.0


class TestSnrProxy:
    def test_known_snr_within_1db(self):
        rng = np.random.default_rng(0)
        amp, sigma = 0.5, 0.05
        wav = np.zeros(2 * SR)
        wav[:SR] = amp * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
        wav[SR:] = rng.normal(0, sigma, SR)
        expected = 10 * math.log10((amp**2 / 2) / sigma**2)
        got = estimate_snr_proxy_db(wav, [{"start": 0, "end": SR}])
        assert got == pytest.approx(expected, abs=1.0)

    def test_no_noise_samples_is_inf(self):
        wav = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
        assert estimate_snr_proxy_db(wav, [{"start": 0, "end": SR}]) == math.inf


# ── apply_filters ─────────────────────────────────────────────


def _raw_row(clip_id: str, accent_raw="united states english", duration=6.0) -> dict:
    return {
        "clip_id": clip_id,
        "source": "common_voice",
        "source_uri": "hf://x",
        "orig_file": f"clips/{clip_id}.wav",
        "offset_start_s": 0.0,
        "offset_end_s": duration,
        "sample_rate_orig": SR,
        "duration_s": duration,
        "license": "CC0-1.0",
        "speaker_id_raw": "spk1",
        "accent_raw": accent_raw,
    }


def _speech_wav(seconds: float = 6.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    wav = np.zeros_like(t)
    # 90% speech + a 10% low-noise tail, so the SNR proxy comes out high
    n_speech = int(0.9 * len(t))
    wav[:n_speech] = 0.5 * np.sin(2 * np.pi * 220 * t[:n_speech])
    wav[n_speech:] = 0.001 * np.sin(2 * np.pi * 60 * t[n_speech:])
    return wav


def _run(df: pd.DataFrame, audio_map: dict[str, np.ndarray], **lid_override):
    return apply_filters(
        df,
        FilterConfig(),
        taxonomy=make_taxonomy(),
        audio_loader=lambda clip_id: (audio_map[clip_id], SR),
        vad_fn=fake_vad,
        transcribe_fn=fake_transcribe,
        lid_fn=lid_override.get("lid_fn", fake_lid_en),
    )


class TestApplyFilters:
    def test_good_clip_accepted_and_qc_valid(self):
        df = pd.DataFrame([_raw_row("good")])
        out = _run(df, {"good": _speech_wav()})
        validate_manifest(out, stage="qc")
        row = out.iloc[0]
        assert row["status"] == "accepted"
        assert row["accent_label"] == "en-US"
        assert row["taxonomy_version"] == "v1"
        assert row["transcript"] == "hello world this is a test"

    def test_silence_rejected_low_vad(self):
        df = pd.DataFrame([_raw_row("quiet")])
        out = _run(df, {"quiet": np.zeros(6 * SR)})
        row = out.iloc[0]
        assert row["status"] == "rejected"
        assert row["reject_reason"] == "low_vad"

    def test_unmapped_accent_rejected_without_loading_audio(self):
        df = pd.DataFrame([_raw_row("scot", accent_raw="scottish english")])
        out = _run(df, {})  # audio_map is empty: it must never be touched
        row = out.iloc[0]
        assert row["status"] == "rejected"
        assert row["reject_reason"] == "unmapped_accent"

    def test_too_short_rejected(self):
        df = pd.DataFrame([_raw_row("short", duration=3.0)])
        out = _run(df, {})
        assert out.iloc[0]["reject_reason"] == "too_short"

    def test_non_english_rejected(self):
        df = pd.DataFrame([_raw_row("de")])
        out = _run(df, {"de": _speech_wav()}, lid_fn=lambda t: ("de", 0.97))
        assert out.iloc[0]["reject_reason"] == "not_english"

    def test_mixed_batch_validates_qc(self):
        df = pd.DataFrame([_raw_row("good"), _raw_row("short", duration=1.0)])
        out = _run(df, {"good": _speech_wav()})
        validate_manifest(out, stage="qc")
        assert list(out["status"]) == ["accepted", "rejected"]
