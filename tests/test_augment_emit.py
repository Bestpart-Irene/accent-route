"""T14: augmentation (train only) and the three-arm dataset emit.

Key invariants: the A and B variants contain zero weak rows, augmented rows only ever
land in train, and the emitted stats reconcile.
"""

import numpy as np
import pandas as pd
import pytest

from accentroute.augment import AugmentConfig, augment_train, speed_perturb
from accentroute.emit import emit_dataset

SR = 16000


def _row(clip_id, split, label_source="self_report", label="en-US", source="common_voice"):
    return {
        "clip_id": clip_id,
        "source": source,
        "source_uri": "x://",
        "orig_file": f"{clip_id}.wav",
        "offset_start_s": 0.0,
        "offset_end_s": 6.0,
        "sample_rate_orig": SR,
        "duration_s": 6.0,
        "license": "CC0-1.0",
        "speaker_id_raw": f"spk-{clip_id}",
        "speaker_key": f"{source}:spk-{clip_id}",
        "accent_raw": label,
        "accent_label": label,
        "taxonomy_version": "v1",
        "snr_proxy_db": 20.0,
        "vad_speech_ratio": 0.9,
        "lang_prob": 0.99,
        "transcript": "t",
        "status": "accepted",
        "reject_reason": None,
        "split": split,
        "label_source": label_source,
        "consensus_score": 1.0 if label_source == "weak" else None,
        "evidence_level": "E1" if label_source == "weak" else None,
    }


@pytest.fixture()
def manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("g1", "train"),
            _row("g2", "val"),
            _row("g3", "test"),
            _row("w1", "train", label_source="weak", label="en-AU", source="youtube"),
            _row("e1", "ood_test", label_source="gold", source="edacc"),
        ]
    )


class TestSpeedPerturb:
    def test_length_scales_inversely(self):
        wav = np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
        assert len(speed_perturb(wav, SR, 0.9)) == pytest.approx(SR / 0.9, rel=0.01)
        assert len(speed_perturb(wav, SR, 1.1)) == pytest.approx(SR / 1.1, rel=0.01)

    def test_factor_one_is_identity(self):
        wav = np.sin(2 * np.pi * 220 * np.arange(1000) / SR)
        np.testing.assert_allclose(speed_perturb(wav, SR, 1.0), wav)


class TestAugmentTrain:
    def test_only_train_rows_augmented(self, manifest):
        out = augment_train(manifest, AugmentConfig(speed=(0.9, 1.1), musan=False, rir=False))
        aug = out[out["clip_id"].str.contains("#aug")]
        assert len(aug) > 0
        assert set(aug["split"]) == {"train"}

    def test_augmented_rows_keep_speaker_key_and_label(self, manifest):
        out = augment_train(manifest, AugmentConfig(speed=(0.9,), musan=False, rir=False))
        aug = out[out["clip_id"].str.contains("#aug")]
        for _, r in aug.iterrows():
            base = out[out["clip_id"] == r["clip_id"].split("#")[0]].iloc[0]
            assert r["speaker_key"] == base["speaker_key"]  # must not invent a new speaker
            assert r["accent_label"] == base["accent_label"]
            assert r["label_source"] == base["label_source"]

    def test_augmented_clip_ids_unique(self, manifest):
        out = augment_train(manifest, AugmentConfig(speed=(0.9, 1.1), musan=False, rir=False))
        assert out["clip_id"].is_unique

    def test_duration_adjusted_by_speed(self, manifest):
        out = augment_train(manifest, AugmentConfig(speed=(0.9,), musan=False, rir=False))
        aug = out[out["clip_id"].str.contains("#aug")].iloc[0]
        assert aug["duration_s"] == pytest.approx(6.0 / 0.9, rel=0.01)


class TestEmitDataset:
    def test_gold_variants_have_zero_weak(self, manifest, tmp_path):
        for variant in ["a_gold", "b_gold_oversampled"]:
            stats = emit_dataset(manifest, variant, tmp_path / variant)
            df = pd.read_parquet(tmp_path / variant / "manifest.parquet")
            assert (df["label_source"] == "weak").sum() == 0
            assert stats.n_weak_train == 0

    def test_c_variant_includes_weak_in_train_only(self, manifest, tmp_path):
        stats = emit_dataset(manifest, "c_gold_weak", tmp_path / "c")
        df = pd.read_parquet(tmp_path / "c" / "manifest.parquet")
        weak = df[df["label_source"] == "weak"]
        assert len(weak) == 1
        assert set(weak["split"]) == {"train"}
        assert stats.n_weak_train == 1

    def test_eval_splits_identical_across_variants(self, manifest, tmp_path):
        """All three arms must share the same eval splits, or the ablation is not comparable."""
        frames = {}
        for variant in ["a_gold", "b_gold_oversampled", "c_gold_weak"]:
            emit_dataset(manifest, variant, tmp_path / variant)
            df = pd.read_parquet(tmp_path / variant / "manifest.parquet")
            frames[variant] = set(df[df["split"].isin(["val", "test", "ood_test"])]["clip_id"])
        assert frames["a_gold"] == frames["b_gold_oversampled"] == frames["c_gold_weak"]

    def test_loso_variant_drops_l2arctic_from_train(self, tmp_path):
        df = pd.DataFrame(
            [
                _row("l2t", "train", label_source="gold", label="L1-Korean", source="l2_arctic"),
                _row("l2e", "test", label_source="gold", label="L1-Korean", source="l2_arctic"),
                _row("cvk", "train", label="L1-Korean"),
            ]
        )
        emit_dataset(df, "loso_l2", tmp_path / "loso")
        out = pd.read_parquet(tmp_path / "loso" / "manifest.parquet")
        train = out[out["split"] == "train"]
        assert "l2_arctic" not in set(train["source"])  # dropped from train
        assert "l2t" not in set(out["clip_id"])
        assert "l2e" in set(out["clip_id"])  # kept on the test side

    def test_stats_reconcile(self, manifest, tmp_path):
        stats = emit_dataset(manifest, "c_gold_weak", tmp_path / "c")
        df = pd.read_parquet(tmp_path / "c" / "manifest.parquet")
        assert stats.n_total == len(df)
        assert sum(stats.per_split.values()) == stats.n_total
        assert sum(stats.per_class_train.values()) == stats.per_split["train"]

    def test_stats_json_written(self, manifest, tmp_path):
        import json

        emit_dataset(manifest, "a_gold", tmp_path / "a")
        stats = json.loads((tmp_path / "a" / "stats.json").read_text())
        assert stats["variant"] == "a_gold"
        assert "per_split" in stats and "per_class_train" in stats

    def test_unknown_variant_rejected(self, manifest, tmp_path):
        with pytest.raises(ValueError, match="unknown variant"):
            emit_dataset(manifest, "d_mystery", tmp_path / "d")
