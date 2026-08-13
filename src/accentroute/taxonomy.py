"""accent_raw → 8-class mapping: a versioned YAML allowlist; anything not in the table is
dropped and counted."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from accentroute.schema import ACCENTS


def _normalize(raw: str) -> str:
    return " ".join(raw.lower().split())


@dataclass
class Taxonomy:
    version: str
    mapping: dict[str, str]
    unmapped_counts: Counter = field(default_factory=Counter)

    def map(self, raw: str | None) -> str | None:
        """Normalize, then look up the allowlist; anything not in the table returns None and
        is counted. Empty input is not counted — missing is not the same as unmapped.
        """
        if not raw:
            return None
        key = _normalize(raw)
        if not key:
            return None
        label = self.mapping.get(key)
        if label is None:
            self.unmapped_counts[key] += 1
        return label


def load_taxonomy(path: Path) -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text())
    mapping = {_normalize(k): v for k, v in data["mapping"].items()}
    invalid = set(mapping.values()) - set(ACCENTS)
    if invalid:
        raise ValueError(f"taxonomy {data['version']}: invalid target labels {invalid}")
    return Taxonomy(version=str(data["version"]), mapping=mapping)
