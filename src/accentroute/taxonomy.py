"""accent_raw → 8 类映射:YAML 版本化白名单,表外丢弃并计数。"""

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
        """归一化后查白名单;表外返回 None 并计数。空输入不计数(缺失 ≠ 映射外)。"""
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
