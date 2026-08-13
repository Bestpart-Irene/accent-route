"""Masked mean-pooling. The whisper encoder always emits 1500 frames (audio is padded or
truncated to 30s), so the valid frame count is derived from the feature extractor's
attention mask and padding frames never enter the pool.

Policy (documented in the model card): clips >30s are cropped to the center 30s window;
clips <5s were already dropped by the filter; one window per clip, no sliding; and
normalization stays at the WhisperFeatureExtractor default.
"""

import torch

N_ENC_MAX = 1500  # whisper's 30s window: 3000 mel frames → conv2(k=3,s=2,p=1) → 1500


def valid_encoder_frames(mel_attention_mask: torch.Tensor) -> torch.Tensor:
    """mel_attention_mask: [B, 3000], from WhisperFeatureExtractor(return_attention_mask=True).

    conv1 preserves length; conv2 with k=3, s=2, p=1 gives L_out = (L_in - 1)//2 + 1.
    """
    n_mel = mel_attention_mask.sum(-1).to(torch.long)
    return torch.clamp((n_mel - 1) // 2 + 1, max=N_ENC_MAX)


def num_valid_encoder_frames_ref(n_samples: int, hop: int = 160) -> int:
    """Reference implementation for unit tests only, cross-checked against the mask-derived
    path; production uses valid_encoder_frames."""
    n_mel = n_samples // hop
    return min((n_mel - 1) // 2 + 1, N_ENC_MAX)


def masked_mean(hidden: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
    """hidden: [B, T, D]; n_valid: [B] → [B, D], averaging only the first n_valid frames."""
    idx = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    mask = (idx < n_valid[:, None].to(hidden.device)).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
