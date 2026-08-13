"""Emit the three-arm datasets: A/B carry gold and self-reported labels only, C adds the
accepted weak labels, and LOSO drops L2-ARCTIC from the training side.

All three arms share one set of val/test/ood_test splits — the evaluation data must not
vary by arm, or the ablation is not comparable. The unit tests enforce this.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from accentroute.schema import validate_manifest

EVAL_SPLITS = ("val", "test", "ood_test")
VARIANTS = ("a_gold", "b_gold_oversampled", "c_gold_weak", "loso_l2")


@dataclass(frozen=True)
class DatasetStats:
    variant: str
    n_total: int
    n_weak_train: int
    per_split: dict[str, int] = field(default_factory=dict)
    per_class_train: dict[str, int] = field(default_factory=dict)
    hours_per_split: dict[str, float] = field(default_factory=dict)


def emit_dataset(df: pd.DataFrame, variant: str, out_dir: Path) -> DatasetStats:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    keep = df.copy()
    is_eval = keep["split"].isin(EVAL_SPLITS)
    is_weak = keep["label_source"] == "weak"

    if variant in ("a_gold", "b_gold_oversampled", "loso_l2"):
        keep = keep[is_eval | ~is_weak]
    if variant == "loso_l2":
        # L2-ARCTIC stays on the evaluation side only: training is left with Common Voice
        # self-reported L2 speech, which is what quantifies cross-source generalization
        keep = keep[keep["split"].isin(EVAL_SPLITS) | (keep["source"] != "l2_arctic")]

    keep = keep.reset_index(drop=True)
    validate_manifest(keep, stage="split")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(out_dir / "manifest.parquet", index=False)

    train = keep[keep["split"] == "train"]
    stats = DatasetStats(
        variant=variant,
        n_total=len(keep),
        n_weak_train=int((train["label_source"] == "weak").sum()),
        per_split=keep["split"].value_counts().to_dict(),
        per_class_train=train["accent_label"].value_counts().to_dict(),
        hours_per_split=(keep.groupby("split")["duration_s"].sum() / 3600).round(3).to_dict(),
    )
    (out_dir / "stats.json").write_text(json.dumps(asdict(stats), indent=2))
    return stats
