"""Ingest base class: each source adapter yields raw-schema records, and run_ingest
collects, validates, and persists them to Parquet."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd

from accentroute.schema import validate_manifest


class SourceIngestor(ABC):
    """One subclass per data source. The dicts iter_records yields must pass raw-stage
    validation."""

    source: ClassVar[str]
    license: ClassVar[str]

    @abstractmethod
    def iter_records(self) -> Iterator[dict]: ...


def run_ingest(ingestor: SourceIngestor, out: Path) -> Path:
    """Collect records → validate against raw → write Parquet. Validation failures raise
    immediately so no half-finished manifest ever hits disk."""
    records = list(ingestor.iter_records())
    if not records:
        raise ValueError(f"ingestor {type(ingestor).__name__} produced no records")
    df = pd.DataFrame.from_records(records)
    validate_manifest(df, stage="raw")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
