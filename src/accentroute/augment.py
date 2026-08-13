"""Training-set augmentation: speed perturbation + MUSAN noise + RIR reverb, applied to
train rows only.

An augmented row reuses the source row's speaker_key and label — it never invents a new
speaker, which is what keeps augmentation from breaking the speaker-disjoint split. The
clip_id gets an "#aug" suffix so the provenance stays visible.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import soxr


@dataclass(frozen=True)
class AugmentConfig:
    speed: tuple[float, ...] = (0.9, 1.1)
    musan: bool = True
    rir: bool = True


def speed_perturb(wav: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Resampling-based speed change: shifts duration and formants together (the standard
    speed perturbation)."""
    if factor == 1.0:
        return wav
    return soxr.resample(wav, int(sr * factor), sr)


def augment_train(df: pd.DataFrame, cfg: AugmentConfig) -> pd.DataFrame:
    """Add one augmented row per train row per speed factor, at the manifest level; the
    audio itself is generated on demand during training.

    MUSAN and RIR are random online augmentations applied at training time and never enter
    the manifest — only speed perturbation changes duration, which the manifest has to
    reflect.
    """
    train = df[df["split"] == "train"]
    aug_rows = []
    for factor in cfg.speed:
        if factor == 1.0:
            continue
        for row in train.to_dict("records"):
            aug = dict(row)
            aug["clip_id"] = f"{row['clip_id']}#aug-sp{factor}"
            aug["duration_s"] = row["duration_s"] / factor
            aug["offset_end_s"] = row["offset_start_s"] + aug["duration_s"]
            aug_rows.append(aug)
    if not aug_rows:
        return df.copy()
    return pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)
