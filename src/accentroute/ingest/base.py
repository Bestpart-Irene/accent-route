"""ingest 基类:各源适配器产出 raw-schema 记录,run_ingest 收集、校验、落 Parquet。"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd

from accentroute.schema import validate_manifest


class SourceIngestor(ABC):
    """每个数据源一个子类。iter_records 产出的 dict 必须能通过 raw 阶段校验。"""

    source: ClassVar[str]
    license: ClassVar[str]

    @abstractmethod
    def iter_records(self) -> Iterator[dict]: ...


def run_ingest(ingestor: SourceIngestor, out: Path) -> Path:
    """收集记录 → raw 校验 → 写 Parquet。校验失败直接抛出,不落盘半成品。"""
    records = list(ingestor.iter_records())
    if not records:
        raise ValueError(f"ingestor {type(ingestor).__name__} produced no records")
    df = pd.DataFrame.from_records(records)
    validate_manifest(df, stage="raw")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
