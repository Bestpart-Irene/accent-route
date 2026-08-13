"""Common Voice adapter: validated.tsv + a clips/ directory (the same layout you get from
an HF common_voice_17_0 download).

Only rows with a *single-valued* self-reported accent are emitted: rows with no accents are
skipped, and multi-valued (comma-separated) rows are skipped as ambiguous and counted.
Fill-rate statistics land in self.stats, which the G1 datasheet cites directly.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import soundfile as sf

from accentroute.ingest.base import SourceIngestor

# Common Voice separates multiple values with commas, but the official option strings can
# contain commas inside parentheses — "India and South Asia (India, Pakistan, Sri Lanka)"
# is a single value.
_ACCENT_SEP = re.compile(r",(?![^()]*\))")


def _split_accents(raw: str) -> list[str]:
    return [a.strip() for a in _ACCENT_SEP.split(raw) if a.strip()]


class CommonVoiceIngestor(SourceIngestor):
    source = "common_voice"
    license = "CC0-1.0"

    def __init__(
        self,
        root: Path,
        tsv: str = "validated.tsv",
        clips_dir: str = "clips",
        source_uri: str = "hf://mozilla-foundation/common_voice_17_0",
    ):
        self.root = Path(root)
        self.tsv = self.root / tsv
        self.clips = self.root / clips_dir
        self.source_uri = source_uri
        self.stats: dict = {}

    def iter_records(self) -> Iterator[dict]:
        df = pd.read_csv(self.tsv, sep="\t", dtype=str, keep_default_na=False)
        n_no_accent = 0
        n_multi = 0
        n_yielded = 0
        for row in df.itertuples(index=False):
            accents = _split_accents(row.accents)
            if not accents:
                n_no_accent += 1
                continue
            if len(accents) > 1:
                n_multi += 1
                continue
            clip = self.clips / row.path
            info = sf.info(clip)
            n_yielded += 1
            yield {
                "clip_id": f"cv:{Path(row.path).stem}",
                "source": self.source,
                "source_uri": self.source_uri,
                "orig_file": f"{self.clips.name}/{row.path}",
                "offset_start_s": 0.0,
                "offset_end_s": info.duration,
                "sample_rate_orig": info.samplerate,
                "duration_s": info.duration,
                "license": self.license,
                "speaker_id_raw": row.client_id,
                "accent_raw": accents[0],
            }
        n_rows = len(df)
        self.stats = {
            "n_rows": n_rows,
            "n_no_accent": n_no_accent,
            "n_multi_accent_skipped": n_multi,
            "n_yielded": n_yielded,
            "accent_fill_rate": (n_rows - n_no_accent) / n_rows if n_rows else 0.0,
        }
