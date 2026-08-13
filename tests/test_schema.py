"""T1: behavioral contract for the staged manifest schema.

Three invariants have to be machine-enforced:
  1. a rejected row must carry a reject_reason
  2. a row with label_source == "weak" may never appear in val/test/ood_test
  3. an accepted row from youtube must carry an E1 or E2 evidence level
"""

import pandas as pd
import pandera.errors
import pytest

from accentroute.schema import ACCENTS, SOURCES, SPLITS, validate_manifest


def make_raw_df(**overrides) -> pd.DataFrame:
    row = {
        "clip_id": "cv:abc123:0",
        "source": "common_voice",
        "source_uri": "hf://mozilla-foundation/common_voice_17_0",
        "orig_file": "clips/common_voice_en_123.mp3",
        "offset_start_s": 0.0,
        "offset_end_s": 6.2,
        "sample_rate_orig": 48000,
        "duration_s": 6.2,
        "license": "CC0-1.0",
        "speaker_id_raw": "cv_speaker_1",
        "accent_raw": "united states english",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def make_split_df(**overrides) -> pd.DataFrame:
    row = {
        **make_raw_df().iloc[0].to_dict(),
        # qc-stage fields
        "accent_label": "en-US",
        "taxonomy_version": "v1",
        "snr_proxy_db": 22.5,
        "vad_speech_ratio": 0.9,
        "lang_prob": 0.99,
        "transcript": "hello world",
        "status": "accepted",
        "reject_reason": None,
        # split-stage fields
        "speaker_key": "common_voice:cv_speaker_1",
        "split": "train",
        "label_source": "self_report",
        "consensus_score": None,
        "evidence_level": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestRawStage:
    def test_valid_raw_row_passes(self):
        validate_manifest(make_raw_df(), stage="raw")

    def test_raw_row_missing_split_columns_fails_split_stage(self):
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_manifest(make_raw_df(), stage="split")

    def test_duplicate_clip_id_rejected(self):
        df = pd.concat([make_raw_df(), make_raw_df()], ignore_index=True)
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_manifest(df, stage="raw")

    def test_unknown_source_rejected(self):
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_manifest(make_raw_df(source="librispeech"), stage="raw")


class TestSplitStage:
    def test_valid_split_row_passes(self):
        validate_manifest(make_split_df(), stage="split")

    def test_weak_label_in_test_rejected(self):
        df = make_split_df(label_source="weak", split="test")
        with pytest.raises(pandera.errors.SchemaErrors) as exc:
            validate_manifest(df, stage="split")
        assert "weak_never_in_eval" in str(exc.value)

    @pytest.mark.parametrize("split", ["val", "ood_test"])
    def test_weak_label_in_any_eval_split_rejected(self, split):
        df = make_split_df(label_source="weak", split=split)
        with pytest.raises(pandera.errors.SchemaErrors):
            validate_manifest(df, stage="split")

    def test_weak_label_in_train_passes(self):
        validate_manifest(make_split_df(label_source="weak", split="train"), stage="split")

    def test_rejected_without_reason_rejected(self):
        df = make_split_df(status="rejected", reject_reason=None)
        with pytest.raises(pandera.errors.SchemaErrors) as exc:
            validate_manifest(df, stage="split")
        assert "rejected_has_reason" in str(exc.value)

    def test_rejected_with_reason_passes(self):
        validate_manifest(
            make_split_df(status="rejected", reject_reason="low_vad"), stage="split"
        )

    def test_youtube_accepted_without_evidence_rejected(self):
        df = make_split_df(
            source="youtube", status="accepted", evidence_level="E3", label_source="weak"
        )
        with pytest.raises(pandera.errors.SchemaErrors) as exc:
            validate_manifest(df, stage="split")
        assert "youtube_requires_evidence" in str(exc.value)

    def test_youtube_accepted_with_e1_passes(self):
        validate_manifest(
            make_split_df(
                source="youtube", status="accepted", evidence_level="E1", label_source="weak"
            ),
            stage="split",
        )


class TestConstants:
    def test_eight_accent_classes_locked(self):
        assert ACCENTS == [
            "en-US", "en-GB", "en-AU", "en-IN",
            "L1-Mandarin", "L1-Spanish", "L1-Korean", "L1-Arabic",
        ]

    def test_core_sources(self):
        assert set(SOURCES) == {"common_voice", "l2_arctic", "edacc", "youtube"}

    def test_splits_include_ood(self):
        assert "ood_test" in SPLITS
