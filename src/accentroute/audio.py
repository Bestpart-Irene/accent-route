"""Audio normalization: any input → 16 kHz mono PCM16 WAV, the one format the whole
pipeline works in."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000


@dataclass(frozen=True)
class AudioMeta:
    sample_rate_orig: int
    n_channels_orig: int
    duration_s: float  # duration of the emitted segment
    n_samples_out: int


def to_wav16k_mono(
    src: Path,
    dst: Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> AudioMeta:
    """Read src (wav/flac/mp3/ogg), optionally slice it by seconds, downmix to mono,
    resample to 16k, and write PCM16.

    Slicing happens by sample index at the original sample rate, so offsets are exact to
    1/sr of a second.
    """
    data, sr = sf.read(src, always_2d=True, dtype="float64")
    n_channels = data.shape[1]
    n_total = data.shape[0]

    i0 = 0 if start_s is None else round(start_s * sr)
    i1 = n_total if end_s is None else round(end_s * sr)
    if i0 < 0 or i1 > n_total or i0 >= i1:
        raise ValueError(
            f"slice [{start_s}, {end_s}]s is beyond file bounds "
            f"({n_total / sr:.3f}s @ {sr}Hz): {src}"
        )
    mono = data[i0:i1].mean(axis=1)

    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR)

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.clip(mono, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
    return AudioMeta(
        sample_rate_orig=sr,
        n_channels_orig=n_channels,
        duration_s=len(mono) / TARGET_SR,
        n_samples_out=len(mono),
    )
