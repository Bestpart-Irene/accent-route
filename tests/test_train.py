"""T10: 训练循环 + 三臂预算协议。

公平性由机制保证:共享字段(预算/采样/增强/ckpt 规则)只能来自
train_common.yaml,臂配置试图覆盖 → 配置加载器直接拒绝;
B 臂步数 == C 臂步数由 resolve_total_steps 保证并在测试断言。
"""

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from transformers import WhisperConfig, WhisperModel

from accentroute.model.whisper_lora import WhisperEncoderClassifier
from accentroute.train import (
    SHARED_FIELDS,
    TrainConfig,
    load_train_config,
    resolve_total_steps,
    train,
)

COMMON = Path(__file__).parent.parent / "configs" / "train_common.yaml"
ARMS = Path(__file__).parent.parent / "configs" / "arms"


class TestConfigLoader:
    def test_loads_arm_over_common(self):
        cfg = load_train_config(COMMON, ARMS / "a_gold.yaml", seed=17)
        assert cfg.arm == "a_gold"
        assert cfg.budget == "epoch_matched"
        assert cfg.seed == 17
        assert cfg.batch_size > 0

    def test_arm_overriding_shared_field_rejected(self, tmp_path):
        bad = tmp_path / "bad_arm.yaml"
        bad.write_text("arm: evil\nbudget: epoch_matched\nbatch_size: 4\n")
        with pytest.raises(ValueError, match="batch_size"):
            load_train_config(COMMON, bad, seed=17)

    def test_shared_hash_identical_across_arms(self):
        a = load_train_config(COMMON, ARMS / "a_gold.yaml", seed=17)
        c = load_train_config(COMMON, ARMS / "c_gold_weak.yaml", seed=42)
        assert a.shared_config_hash == c.shared_config_hash

    def test_all_arm_configs_load(self):
        for arm_file in sorted(ARMS.glob("*.yaml")):
            cfg = load_train_config(COMMON, arm_file, seed=17)
            assert cfg.arm

    def test_seed_must_be_in_common_seeds(self):
        with pytest.raises(ValueError, match="seed"):
            load_train_config(COMMON, ARMS / "a_gold.yaml", seed=999)

    def test_every_shared_field_is_locked(self, tmp_path):
        """协议字段逐个验证不可覆盖 —— 公平性不能只靠一个抽样测试。"""
        for field in sorted(SHARED_FIELDS):
            bad = tmp_path / f"arm_{field}.yaml"
            bad.write_text(f"arm: evil\nbudget: epoch_matched\n{field}: 1\n")
            with pytest.raises(ValueError, match=field):
                load_train_config(COMMON, bad, seed=17)


class TestBudgetProtocol:
    def test_b_steps_equal_c_steps(self):
        n_c, n_a, bs, epochs_c = 5000, 3000, 32, 15
        steps_c = resolve_total_steps("epoch_matched", n_c, bs, epochs_c)
        steps_b = resolve_total_steps("step_matched_to_c", n_a, bs, epochs_c, steps_c=steps_c)
        assert steps_b == steps_c == epochs_c * math.ceil(n_c / bs)

    def test_epoch_matched_uses_own_size(self):
        assert resolve_total_steps("epoch_matched", 3000, 32, 15) == 15 * math.ceil(3000 / 32)

    def test_step_matched_requires_steps_c(self):
        with pytest.raises(ValueError, match="steps_c"):
            resolve_total_steps("step_matched_to_c", 3000, 32, 15)


def _tiny_model(n_classes=4):
    cfg = WhisperConfig(
        d_model=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        decoder_layers=1,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=150,  # mel T=300 → encoder 150 帧,测试提速
    )
    return WhisperEncoderClassifier(
        WhisperModel(cfg).get_encoder(), d_model=32, n_classes=n_classes, lora_r=4
    )


class _SyntheticDs(torch.utils.data.Dataset):
    """16 clips、4 类;每类的 mel 有独特强模式,可快速过拟合。"""

    def __init__(self, n=16, n_classes=4):
        g = torch.Generator().manual_seed(0)
        self.items = []
        for i in range(n):
            label = i % n_classes
            feats = 0.01 * torch.randn(80, 300, generator=g)
            feats[label * 20 : (label + 1) * 20, :] += 3.0  # 类模式
            self.items.append(
                {"input_features": feats, "n_valid": 150, "label": label}
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _train_cfg(tmp_path, total_steps=400, seed=17) -> TrainConfig:
    return TrainConfig(
        arm="a_gold",
        budget="epoch_matched",
        seed=seed,
        out_dir=tmp_path / "run",
        epochs_c=15,
        batch_size=8,
        # 测试用的随机初始化 encoder 特征近乎退化(pooled 两两余弦 >0.99),
        # 需要比生产(1e-4)大得多的 lr 才能在合成任务上过拟合。
        lr=2e-2,
        warmup_ratio=0.05,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        base_model="tiny-test",
        n_classes=4,
        total_steps=total_steps,
        shared_config_hash="testhash",
    )


class TestTrainLoop:
    def test_overfits_synthetic_batch(self, tmp_path):
        ds = _SyntheticDs()
        result = train(_train_cfg(tmp_path), model=_tiny_model(), train_ds=ds, val_ds=ds)
        assert result.val_macro_f1 > 0.9
        assert result.total_steps == 400
        assert Path(result.ckpt_path).exists()

    def test_metrics_json_written(self, tmp_path):
        ds = _SyntheticDs()
        result = train(_train_cfg(tmp_path), model=_tiny_model(), train_ds=ds, val_ds=ds)
        metrics = json.loads((Path(result.ckpt_path).parent / "metrics.json").read_text())
        assert metrics["arm"] == "a_gold"
        assert metrics["seed"] == 17
        assert metrics["total_steps"] == 400
        assert metrics["shared_config_hash"] == "testhash"
        assert metrics["best_val_macro_f1"] == pytest.approx(result.val_macro_f1)
        assert len(metrics["history"]) >= 2
        assert metrics["history"][0]["train_loss"] > metrics["history"][-1]["train_loss"]

    def test_ckpt_contains_only_trainable_params(self, tmp_path):
        ds = _SyntheticDs()
        result = train(_train_cfg(tmp_path), model=_tiny_model(), train_ds=ds, val_ds=ds)
        state = torch.load(result.ckpt_path, weights_only=True)
        assert state, "empty checkpoint"
        for key in state:
            assert "lora_" in key or "head" in key, key
