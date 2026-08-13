"""EdAcc as published on Hugging Face (`edinburghcstr/edacc`).

The DataShare release ships long conversation files plus segments.csv/speakers.csv;
the HF release ships one row per clip with the audio embedded and the columns
speaker / text / accent / raw_accent / gender / l1. Same corpus, different layout, so
this is a separate ingestor rather than a config remap of the DataShare one.

accent_raw follows the same rule as the DataShare ingestor: L2 speakers (l1 is not
English) are labeled by their l1, native speakers by the linguist-standardized accent
field.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from accentroute.ingest.base import SourceIngestor

TARGET_SR = 16000


def _decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """HF's Audio feature reaches us in one of two shapes.

    Reading the parquet directly (what this ingestor does) yields the *encoded* form
    {"bytes": <file bytes>, "path": str}; going through `datasets` yields the *decoded*
    form {"array": np.ndarray, "sampling_rate": int}. Support both — the encoded form is
    the one the real corpus uses.
    """
    if "array" in audio:
        return np.asarray(audio["array"], dtype=np.float64), int(audio["sampling_rate"])
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float64", always_2d=True)
    return wav.mean(axis=1), int(sr)


def _audio_duration(audio: dict) -> tuple[float, int]:
    """(duration_s, sample_rate) without decoding the samples when we can avoid it."""
    if "array" in audio:
        return len(audio["array"]) / int(audio["sampling_rate"]), int(audio["sampling_rate"])
    info = sf.info(io.BytesIO(audio["bytes"]))
    return info.duration, int(info.samplerate)


def _accent_raw(row: dict) -> str:
    l1 = str(row.get("l1") or "").strip()
    if l1 and l1.lower() not in ("english", "nan"):
        return l1
    return str(row.get("accent") or "").strip()


def _shard_files(root: Path) -> list[Path]:
    return sorted((Path(root) / "data").glob("*.parquet"))


def _split_of(shard: Path) -> str:
    return "test" if shard.name.startswith("test") else "validation"


def _clip_prefix(shard: Path) -> str:
    """`validation-00003-of-00006-<hash>` → `edacc:validation:00003`, unique per shard."""
    parts = shard.stem.split("-")
    return f"edacc:{parts[0]}:{parts[1]}"


class EdAccHFIngestor(SourceIngestor):
    source = "edacc"
    license = "CC-BY-SA-4.0"

    def __init__(
        self,
        root: Path,
        source_uri: str = "hf://edinburghcstr/edacc",
    ):
        self.root = Path(root)
        self.source_uri = source_uri

    def iter_records(self) -> Iterator[dict]:
        for shard in _shard_files(self.root):
            split = _split_of(shard)
            prefix = _clip_prefix(shard)
            df = pd.read_parquet(shard, columns=["speaker", "accent", "l1", "audio"])
            for i, row in enumerate(df.to_dict("records")):
                duration, sr = _audio_duration(row["audio"])
                yield {
                    "clip_id": f"{prefix}:{i:06d}",
                    "source": self.source,
                    "source_uri": f"{self.source_uri}#{split}",
                    "orig_file": f"data/{shard.name}",
                    "offset_start_s": 0.0,
                    "offset_end_s": duration,
                    "sample_rate_orig": sr,
                    "duration_s": duration,
                    "license": self.license,
                    "speaker_id_raw": str(row["speaker"]),
                    "accent_raw": _accent_raw(row),
                }


def extract_audio(root: Path, out_dir: Path, skip_existing: bool = False) -> int:
    """Write one 16 kHz mono PCM16 WAV per clip, named by clip_id. Returns files written.

    The pipeline works on WAV files on disk, so the embedded audio has to be materialized
    once before filter/train can read it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for shard in _shard_files(root):
        df = pd.read_parquet(shard, columns=["audio"])
        prefix = _clip_prefix(shard)
        for i, row in enumerate(df.to_dict("records")):
            dst = out_dir / f"{prefix}:{i:06d}.wav"
            if skip_existing and dst.exists():
                continue
            wav, sr = _decode_audio(row["audio"])
            if sr != TARGET_SR:
                import soxr

                wav = soxr.resample(wav, sr, TARGET_SR)
            sf.write(dst, np.clip(wav, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
            written += 1
    return written
