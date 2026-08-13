"""质量过滤:raw manifest → qc manifest(taxonomy 映射 + VAD + SNR 代理 + 转写 + LID)。

模型依赖全部经参数注入(vad_fn / transcribe_fn / lid_fn / audio_loader):
单测与 CI 用假实现,生产入口再绑真模型(Silero VAD、faster-whisper tiny、fastText)。
拒绝按最先命中的原因记录,后续昂贵步骤跳过(unmapped/too_short 不加载音频)。
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from accentroute.schema import validate_manifest
from accentroute.taxonomy import Taxonomy

# silero 约定:[{"start": 样本索引, "end": 样本索引}, ...]
SpeechTs = list[dict]
VadFn = Callable[[np.ndarray, int], SpeechTs]
TranscribeFn = Callable[[np.ndarray], str]
LidFn = Callable[[str], tuple[str, float]]
AudioLoader = Callable[[str], tuple[np.ndarray, int]]


@dataclass(frozen=True)
class FilterConfig:
    min_dur_s: float = 5.0
    max_dur_s: float = 30.0
    min_snr_proxy_db: float = 10.0
    min_vad_ratio: float = 0.5
    min_lang_prob: float = 0.8

    @classmethod
    def from_yaml(cls, path: Path) -> "FilterConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))


def _ratio_from_ts(speech_ts: SpeechTs, n_samples: int) -> float:
    speech = sum(seg["end"] - seg["start"] for seg in speech_ts)
    return speech / n_samples if n_samples else 0.0


def compute_vad_ratio(wav: np.ndarray, sr: int, get_speech_ts: VadFn) -> float:
    """语音样本占比。"""
    return _ratio_from_ts(get_speech_ts(wav, sr), len(wav))


def estimate_snr_proxy_db(wav: np.ndarray, speech_ts: SpeechTs) -> float:
    """语音段与非语音段的能量比(dB)。

    单通道代理量:假设噪声平稳、以非语音段能量近似——不是真实 SNR,
    字段名 snr_proxy_db 即为此。无非语音样本时返回 +inf。
    """
    mask = np.zeros(len(wav), dtype=bool)
    for seg in speech_ts:
        mask[seg["start"] : seg["end"]] = True
    speech, noise = wav[mask], wav[~mask]
    if len(speech) == 0:
        return -math.inf
    if len(noise) == 0:
        return math.inf
    p_speech = float(np.mean(speech**2))
    p_noise = float(np.mean(noise**2))
    if p_noise == 0.0:
        return math.inf
    if p_speech == 0.0:
        return -math.inf
    return 10.0 * math.log10(p_speech / p_noise)


def apply_filters(
    df: pd.DataFrame,
    cfg: FilterConfig,
    *,
    taxonomy: Taxonomy,
    audio_loader: AudioLoader,
    vad_fn: VadFn,
    transcribe_fn: TranscribeFn,
    lid_fn: LidFn,
) -> pd.DataFrame:
    """raw manifest → qc manifest(通过 qc 阶段校验后返回)。"""
    out_rows = []
    for row in df.to_dict("records"):
        qc = {
            **row,
            "accent_label": None,
            "taxonomy_version": taxonomy.version,
            "snr_proxy_db": None,
            "vad_speech_ratio": None,
            "lang_prob": None,
            "transcript": None,
            "status": "rejected",
            "reject_reason": None,
        }
        out_rows.append(qc)

        qc["accent_label"] = taxonomy.map(row["accent_raw"])
        if qc["accent_label"] is None:
            qc["reject_reason"] = "unmapped_accent"
            continue
        if row["duration_s"] < cfg.min_dur_s:
            qc["reject_reason"] = "too_short"
            continue
        if row["duration_s"] > cfg.max_dur_s:
            qc["reject_reason"] = "too_long"
            continue

        wav, sr = audio_loader(row["clip_id"])
        speech_ts = vad_fn(wav, sr)
        qc["vad_speech_ratio"] = _ratio_from_ts(speech_ts, len(wav))
        if qc["vad_speech_ratio"] < cfg.min_vad_ratio:
            qc["reject_reason"] = "low_vad"
            continue

        qc["snr_proxy_db"] = estimate_snr_proxy_db(wav, speech_ts)
        if qc["snr_proxy_db"] < cfg.min_snr_proxy_db:
            qc["reject_reason"] = "low_snr_proxy"
            continue

        qc["transcript"] = transcribe_fn(wav)
        lang, prob = lid_fn(qc["transcript"])
        qc["lang_prob"] = prob
        if lang != "en" or prob < cfg.min_lang_prob:
            qc["reject_reason"] = "not_english"
            continue

        qc["status"] = "accepted"

    out = pd.DataFrame(out_rows)
    validate_manifest(out, stage="qc")
    return out
