"""EdAcc adapter: long conversational wavs + a segment table + a speaker language-background
table.

How accent_raw is chosen (documented in the datasheet):
  - L2 speakers (l1 is non-empty and not English) → accent_raw = l1, since the spec defines
    the L2 classes by native language
  - native speakers → accent_raw = accent, the linguist-normalized accent field
Column names can be remapped in configs/sources/edacc.yaml, so landing the real data is a
config change rather than a code change.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from accentroute.ingest.base import SourceIngestor


class EdAccIngestor(SourceIngestor):
    source = "edacc"
    license = "CC-BY-SA-4.0"

    def __init__(
        self,
        root: Path,
        segments_csv: str = "segments.csv",
        speakers_csv: str = "speakers.csv",
        source_uri: str = "https://datashare.ed.ac.uk/handle/10283/8983",
        columns: dict[str, str] | None = None,
    ):
        self.root = Path(root)
        self.segments_csv = self.root / segments_csv
        self.speakers_csv = self.root / speakers_csv
        self.source_uri = source_uri
        # Remap in the config when the real EdAcc metadata columns differ from the fixtures
        self.col = {
            "segment_id": "segment_id",
            "audio_file": "audio_file",
            "speaker_id": "speaker_id",
            "start_s": "start_s",
            "end_s": "end_s",
            "accent": "accent",
            "l1": "l1",
            **(columns or {}),
        }

    def _accent_raw(self, spk_row: pd.Series) -> str:
        l1 = str(spk_row[self.col["l1"]]).strip()
        if l1 and l1.lower() not in ("english", "nan", ""):
            return l1
        return str(spk_row[self.col["accent"]]).strip()

    def iter_records(self) -> Iterator[dict]:
        segments = pd.read_csv(self.segments_csv)
        speakers = pd.read_csv(self.speakers_csv).set_index(self.col["speaker_id"])
        for row in segments.itertuples(index=False):
            seg = row._asdict()
            spk_id = seg[self.col["speaker_id"]]
            spk = speakers.loc[spk_id]
            start = float(seg[self.col["start_s"]])
            end = float(seg[self.col["end_s"]])
            audio_file = seg[self.col["audio_file"]]
            # Reading the sample rate per segment is slow (many segments share one long
            # conversation file), so cache sf.info by file
            info = self._info(audio_file)
            yield {
                "clip_id": f"edacc:{seg[self.col['segment_id']]}",
                "source": self.source,
                "source_uri": self.source_uri,
                "orig_file": audio_file,
                "offset_start_s": start,
                "offset_end_s": end,
                "sample_rate_orig": info.samplerate,
                "duration_s": end - start,
                "license": self.license,
                "speaker_id_raw": str(spk_id),
                "accent_raw": self._accent_raw(spk),
            }

    def _info(self, rel_path: str):
        if not hasattr(self, "_info_cache"):
            self._info_cache: dict = {}
        if rel_path not in self._info_cache:
            import soundfile as sf

            self._info_cache[rel_path] = sf.info(self.root / rel_path)
        return self._info_cache[rel_path]
