"""GLOBE (`MushanW/GLOBE_V2`) — Common-Voice-derived, CC0, 23,519 speakers.

Why this source exists in the project: it is the only reachable corpus that covers
**en-AU**, and it supplies the other three native varieties with speaker counts far above
the G1 floor (measured on a 14% sample: en-US 2196 speakers, en-GB 485, en-AU 137,
en-IN 86). It supplies **none** of the L1-* classes — its accent field is Common Voice's
self-reported English *variety*, not the speaker's first language — so L2-ARCTIC stays
the only gold source for those four.

Two caveats the datasheet has to carry:
- GLOBE is a TTS corpus, curated for clean audio, so it is not representative of
  in-the-wild recordings; that is a selection bias, not a neutral filter.
- V2 supersamples to 44.1 kHz from Common Voice's original rate. The pipeline resamples
  to 16 kHz anyway, but the upsampling means the extra bandwidth carries no information.

The accent field carries Common Voice's free-text self-report, so the same rule as the
Common Voice ingestor applies: only single-valued reports are used, and multi-valued or
empty ones are skipped and counted.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from accentroute.ingest.base import SourceIngestor
from accentroute.ingest.common_voice import _split_accents
from accentroute.ingest.hf_audio import audio_duration


def _shard_files(root: Path) -> list[Path]:
    return sorted((Path(root) / "data").glob("*.parquet"))


def _clip_prefix(shard: Path) -> str:
    """`train-00003-of-00160` → `globe:train:00003`, unique per shard."""
    parts = shard.stem.split("-")
    return f"globe:{parts[0]}:{parts[1]}"


class GlobeIngestor(SourceIngestor):
    source = "globe"
    license = "CC0-1.0"

    def __init__(self, root: Path, source_uri: str = "hf://MushanW/GLOBE_V2"):
        self.root = Path(root)
        self.source_uri = source_uri
        self.stats: dict = {}

    def iter_records(self) -> Iterator[dict]:
        n_rows = n_no_accent = n_multi = n_yielded = 0
        for shard in _shard_files(self.root):
            prefix = _clip_prefix(shard)
            df = pd.read_parquet(shard, columns=["speaker_id", "accent", "duration", "audio"])
            for i, row in enumerate(df.to_dict("records")):
                n_rows += 1
                accents = _split_accents(str(row["accent"] or ""))
                if not accents:
                    n_no_accent += 1
                    continue
                if len(accents) > 1:
                    n_multi += 1
                    continue
                duration, sr = audio_duration(row["audio"])
                n_yielded += 1
                yield {
                    "clip_id": f"{prefix}:{i:06d}",
                    "source": self.source,
                    "source_uri": self.source_uri,
                    "orig_file": f"data/{shard.name}",
                    "offset_start_s": 0.0,
                    "offset_end_s": duration,
                    "sample_rate_orig": sr,
                    "duration_s": duration,
                    "license": self.license,
                    "speaker_id_raw": str(row["speaker_id"]),
                    "accent_raw": accents[0],
                }
        self.stats = {
            "n_rows": n_rows,
            "n_no_accent": n_no_accent,
            "n_multi_accent_skipped": n_multi,
            "n_yielded": n_yielded,
        }


def quota_plan(available: dict[str, int], per_class: int) -> dict[str, int]:
    """How many clips to take per class: the cap, or everything available if that is less.

    GLOBE is 122 GB and heavily imbalanced (en-US outnumbers en-IN roughly 17:1), so it is
    materialized under a per-class quota rather than downloaded whole. Capping equally is
    also what keeps the class-balanced sampler from having to fight a 17:1 prior.
    """
    return {label: min(per_class, n) for label, n in available.items()}
