"""冻结 whisper encoder + LoRA(r=16, q/v)+ masked mean-pooling + 线性头。"""

import torch
from torch import nn

from accentroute.model.pooling import masked_mean


class WhisperEncoderClassifier(nn.Module):
    """encoder 全冻结,LoRA 只打 q_proj/v_proj,分类头可训。"""

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        n_classes: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        for p in encoder.parameters():
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.encoder = get_peft_model(encoder, lora_cfg)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_features: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        """input_features: [B, 80, 3000]; n_valid: [B] → logits [B, n_classes]。"""
        hidden = self.encoder(input_features).last_hidden_state  # [B, 1500, D]
        return self.head(masked_mean(hidden, n_valid))


def build_model(
    base_model: str = "openai/whisper-small",
    n_classes: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    revision: str | None = None,
) -> WhisperEncoderClassifier:
    """生产入口:下载底座权重(测试不走这里,用微型随机 encoder 直构)。"""
    from transformers import WhisperModel

    whisper = WhisperModel.from_pretrained(base_model, revision=revision)
    return WhisperEncoderClassifier(
        whisper.get_encoder(),
        d_model=whisper.config.d_model,
        n_classes=n_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
