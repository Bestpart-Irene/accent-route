"""训练集增强:变速 + MUSAN 加噪 + RIR 混响,只作用于 train 行。

增强行沿用原行的 speaker_key 与标签 —— 绝不制造新说话人,否则
speaker-disjoint 切分会被自己的增强破坏。clip_id 加 "#aug" 后缀标记来源。
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
    """重采样式变速:同时改变时长与共振峰(标准 speed perturbation)。"""
    if factor == 1.0:
        return wav
    return soxr.resample(wav, int(sr * factor), sr)


def augment_train(df: pd.DataFrame, cfg: AugmentConfig) -> pd.DataFrame:
    """为每个 train 行按 speed 因子生成增强行(manifest 层;音频在训练侧按需生成)。

    MUSAN/RIR 是训练时的随机在线增强,不进 manifest —— 只有变速改变时长,
    需要在 manifest 里体现。
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
