"""Decoding for Hugging Face's Audio feature, shared by the HF-hosted source adapters.

The feature reaches us in one of two shapes:

- *encoded* ``{"bytes": <file bytes>, "path": str}`` — what reading the parquet directly
  gives, which is what these ingestors do. EdAcc stores WAV, GLOBE stores FLAC.
- *decoded* ``{"array": np.ndarray, "sampling_rate": int}`` — what ``datasets`` produces
  after decoding.

Assuming only the decoded shape is a real bug that unit tests with synthetic fixtures do
not catch: the fixture encodes the same wrong assumption, so the suite passes while the
real corpus raises KeyError. Both shapes are supported here on purpose.
"""

import io

import numpy as np
import soundfile as sf


def decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Return (mono float64 samples, sample rate)."""
    if "array" in audio:
        return np.asarray(audio["array"], dtype=np.float64), int(audio["sampling_rate"])
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float64", always_2d=True)
    return wav.mean(axis=1), int(sr)


def audio_duration(audio: dict) -> tuple[float, int]:
    """Return (duration_s, sample rate), reading the header only when possible."""
    if "array" in audio:
        sr = int(audio["sampling_rate"])
        return len(audio["array"]) / sr, sr
    info = sf.info(io.BytesIO(audio["bytes"]))
    return info.duration, int(info.samplerate)
