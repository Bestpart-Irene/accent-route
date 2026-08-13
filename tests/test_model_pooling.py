"""T9: masked mean-pooling frame math + the LoRA classifier.

CI does not install the `ml` extra, so importorskip skips this file there; it runs in
full locally. WhisperFeatureExtractor is default-constructed (parameters match whisper,
zero network), and the classifier uses a tiny randomly initialized WhisperConfig, so no
weights are downloaded.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

from accentroute.model.pooling import (
    N_ENC_MAX,
    masked_mean,
    num_valid_encoder_frames_ref,
    valid_encoder_frames,
)
from accentroute.model.whisper_lora import WhisperEncoderClassifier

SR = 16000


class TestFrameMathRef:
    def test_reference_values(self):
        assert num_valid_encoder_frames_ref(30 * SR) == 1500
        assert num_valid_encoder_frames_ref(5 * SR) == 250
        assert num_valid_encoder_frames_ref(60 * SR) == N_ENC_MAX  # capped by truncation


class TestMaskDerivedPath:
    @pytest.mark.parametrize("seconds", [5.0, 7.3, 30.0])
    def test_agrees_with_reference_on_real_extractor(self, seconds):
        extractor = WhisperFeatureExtractor()
        n_samples = int(seconds * SR)
        audio = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n_samples) / SR)
        feats = extractor(
            audio, sampling_rate=SR, return_attention_mask=True, return_tensors="pt"
        )
        got = valid_encoder_frames(feats.attention_mask)
        assert got.shape == (1,)
        assert int(got[0]) == num_valid_encoder_frames_ref(n_samples)


class TestMaskedMean:
    def test_padding_excluded_exactly(self):
        hidden = torch.randn(2, 10, 4)
        hidden[0, 6:] = 999.0  # garbage padding
        hidden[1, 3:] = -999.0
        n_valid = torch.tensor([6, 3])
        got = masked_mean(hidden, n_valid)
        assert torch.allclose(got[0], hidden[0, :6].mean(0))
        assert torch.allclose(got[1], hidden[1, :3].mean(0))

    def test_full_length_equals_plain_mean(self):
        hidden = torch.randn(1, 8, 4)
        got = masked_mean(hidden, torch.tensor([8]))
        assert torch.allclose(got[0], hidden[0].mean(0))


def _tiny_encoder():
    cfg = WhisperConfig(
        d_model=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        decoder_layers=1,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
    )
    return WhisperModel(cfg).get_encoder()


class TestClassifier:
    def test_only_lora_and_head_trainable(self):
        model = WhisperEncoderClassifier(_tiny_encoder(), d_model=32, n_classes=8, lora_r=4)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable, "no trainable params"
        for name in trainable:
            assert "lora_" in name or "head" in name, f"unexpected trainable param: {name}"

    def test_lora_targets_q_and_v_only(self):
        model = WhisperEncoderClassifier(_tiny_encoder(), d_model=32, n_classes=8, lora_r=4)
        lora_params = [n for n, p in model.named_parameters() if "lora_" in n]
        assert lora_params
        for name in lora_params:
            assert ("q_proj" in name) or ("v_proj" in name), name

    def test_forward_shape(self):
        model = WhisperEncoderClassifier(_tiny_encoder(), d_model=32, n_classes=8, lora_r=4)
        feats = torch.randn(2, 80, 3000)
        n_valid = torch.tensor([250, 1500])
        logits = model(feats, n_valid)
        assert logits.shape == (2, 8)
