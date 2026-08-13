"""masked mean-pooling:whisper encoder 输出恒为 1500 帧(30s pad/截断),
有效帧数从 feature extractor 的 attention mask 推导,padding 帧绝不进池。

策略(进模型卡):>30s 取中心 30s 窗;<5s 已被 filter 丢弃;单窗不滑窗;
归一化沿用 WhisperFeatureExtractor 默认。
"""

import torch

N_ENC_MAX = 1500  # whisper 30s 窗:3000 mel 帧 → conv2(k=3,s=2,p=1) → 1500


def valid_encoder_frames(mel_attention_mask: torch.Tensor) -> torch.Tensor:
    """mel_attention_mask: [B, 3000],来自 WhisperFeatureExtractor(return_attention_mask=True)。

    conv1 保长;conv2 k=3,s=2,p=1 → L_out = (L_in - 1)//2 + 1。
    """
    n_mel = mel_attention_mask.sum(-1).to(torch.long)
    return torch.clamp((n_mel - 1) // 2 + 1, max=N_ENC_MAX)


def num_valid_encoder_frames_ref(n_samples: int, hop: int = 160) -> int:
    """仅作单测参考实现,与 mask 推导路径交叉验证;生产用 valid_encoder_frames。"""
    n_mel = n_samples // hop
    return min((n_mel - 1) // 2 + 1, N_ENC_MAX)


def masked_mean(hidden: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
    """hidden: [B, T, D]; n_valid: [B] → [B, D],只对前 n_valid 帧取均值。"""
    idx = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    mask = (idx < n_valid[:, None].to(hidden.device)).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
