"""L2-ARCTIC 适配器:{SPEAKER}/wav/*.wav 布局,speaker→L1 表来自 configs/sources/l2_arctic.yaml。

全部 6 个 L1 都产出(含 Hindi/Vietnamese):丢弃是 taxonomy 的职责,这里不偷偷砍,
保证丢弃统计完整。未知说话人目录直接报错——说明下载不完整或表过期。
"""

from collections.abc import Iterator
from pathlib import Path

import soundfile as sf

from accentroute.ingest.base import SourceIngestor


class L2ArcticIngestor(SourceIngestor):
    source = "l2_arctic"
    license = "CC-BY-NC-4.0"

    def __init__(
        self,
        root: Path,
        speaker_l1: dict[str, str],
        source_uri: str = "https://psi.engr.tamu.edu/l2-arctic-corpus/",
    ):
        self.root = Path(root)
        self.speaker_l1 = speaker_l1
        self.source_uri = source_uri

    def iter_records(self) -> Iterator[dict]:
        for spk_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            spk = spk_dir.name
            if spk not in self.speaker_l1:
                raise KeyError(
                    f"unknown L2-ARCTIC speaker dir {spk!r}: not in speaker_l1 table"
                )
            for wav in sorted((spk_dir / "wav").glob("*.wav")):
                info = sf.info(wav)
                yield {
                    "clip_id": f"l2arctic:{spk}:{wav.stem}",
                    "source": self.source,
                    "source_uri": self.source_uri,
                    "orig_file": str(wav.relative_to(self.root)),
                    "offset_start_s": 0.0,
                    "offset_end_s": info.duration,
                    "sample_rate_orig": info.samplerate,
                    "duration_s": info.duration,
                    "license": self.license,
                    "speaker_id_raw": spk,
                    "accent_raw": self.speaker_l1[spk],
                }
