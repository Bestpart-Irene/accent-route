"""EdAcc 适配器:长对话 wav + 分段表 + 说话人语言背景表。

accent_raw 选择规则(进 datasheet):
  - L2 说话人(l1 非空且非 English)→ accent_raw = l1(spec 定义 L2 类按母语)
  - 母语说话人 → accent_raw = accent(语言学家标准化的口音字段)
列名可在 configs/sources/edacc.yaml 重映射,真实数据落地时只改配置不改代码。
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
        # 真实 EdAcc 元数据列名与 fixture 不同时,在 config 里重映射
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
            # 采样率读一次会很慢(长对话文件被多段引用),用 sf.info 带缓存
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
