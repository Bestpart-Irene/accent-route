"""T8: speaker-disjoint splits. Core invariant: one speaker_key never spans two splits."""

from pathlib import Path

import pandas as pd
import pytest

from accentroute.split import assign_splits, write_speaker_report

SR = 16000


def _row(clip_id, source, speaker, label, status="accepted", duration=6.0):
    return {
        "clip_id": clip_id,
        "source": source,
        "source_uri": "x://",
        "orig_file": f"{clip_id}.wav",
        "offset_start_s": 0.0,
        "offset_end_s": duration,
        "sample_rate_orig": SR,
        "duration_s": duration,
        "license": "CC0-1.0",
        "speaker_id_raw": speaker,
        "accent_raw": label,
        "accent_label": label if status == "accepted" else None,
        "taxonomy_version": "v1",
        "snr_proxy_db": 20.0,
        "vad_speech_ratio": 0.9,
        "lang_prob": 0.99,
        "transcript": f"transcript {clip_id}",
        "status": status,
        "reject_reason": None if status == "accepted" else "low_vad",
        "speaker_key": f"{source}:{speaker}",
    }


def make_manifest() -> pd.DataFrame:
    rows = []
    # CV: 2 classes × 10 speakers × 4 clips
    for label in ["en-US", "L1-Korean"]:
        for s in range(10):
            spk = f"{label}-cv{s}"
            for c in range(4):
                rows.append(_row(f"cv-{label}-{s}-{c}", "common_voice", spk, label))
    # L2-ARCTIC: L1-Korean, 4 speakers × 5 clips (mirrors the real constraint)
    for s, spk in enumerate(["HJK", "HKK", "YDCK", "YKWK"]):
        for c in range(5):
            rows.append(_row(f"l2-{s}-{c}", "l2_arctic", spk, "L1-Korean"))
    # EdAcc: goes to ood_test only
    for s in range(3):
        rows.append(_row(f"ed-{s}", "edacc", f"P{s}", "en-US"))
    # YouTube weak labels: train only
    rows.append(
        {**_row("yt-0", "youtube", "chan1:v1", "en-AU"), "evidence_level": "E1"}
    )
    # one rejected row: its split must come out "unassigned"
    rows.append(_row("cv-rej", "common_voice", "rejspk", "en-US", status="rejected"))
    return pd.DataFrame(rows)


@pytest.fixture()
def split_df() -> pd.DataFrame:
    return assign_splits(make_manifest(), ratios=(0.8, 0.1, 0.1), seed=17)


class TestInvariants:
    def test_no_speaker_key_spans_splits(self, split_df):
        assigned = split_df[split_df["split"].isin(["train", "val", "test"])]
        per_spk = assigned.groupby("speaker_key")["split"].nunique()
        assert (per_spk == 1).all()

    def test_edacc_only_ood_test(self, split_df):
        assert set(split_df[split_df["source"] == "edacc"]["split"]) == {"ood_test"}

    def test_weak_only_train(self, split_df):
        yt = split_df[split_df["source"] == "youtube"]
        assert set(yt["split"]) == {"train"}
        assert set(yt["label_source"]) == {"weak"}

    def test_rejected_unassigned(self, split_df):
        assert split_df[split_df["clip_id"] == "cv-rej"]["split"].iloc[0] == "unassigned"

    def test_label_source_by_source(self, split_df):
        by_source = split_df.groupby("source")["label_source"].unique()
        assert list(by_source["l2_arctic"]) == ["gold"]
        assert list(by_source["common_voice"]) == ["self_report"]
        assert list(by_source["edacc"]) == ["gold"]


class TestStratification:
    def test_each_class_source_stratum_covers_eval(self, split_df):
        """Every (class, source) stratum with ≥3 speakers is non-empty in train/val/test."""
        pool = split_df[
            split_df["source"].isin(["common_voice", "l2_arctic"])
            & (split_df["status"] == "accepted")
        ]
        for (_label, _source), grp in pool.groupby(["accent_label", "source"]):
            if grp["speaker_key"].nunique() >= 3:
                assert {"train", "val", "test"} <= set(grp["split"])

    def test_l2_arctic_holdout_in_test(self, split_df):
        """A 4-speaker gold stratum splits 2 train / 1 val / 1 test."""
        l2 = split_df[split_df["source"] == "l2_arctic"]
        spk_split = l2.groupby("speaker_key")["split"].first()
        assert sorted(spk_split.values) == ["test", "train", "train", "val"]

    def test_train_share_within_tolerance(self, split_df):
        cv = split_df[(split_df["source"] == "common_voice") & (split_df["split"] != "unassigned")]
        share = (cv["split"] == "train").mean()
        assert 0.65 <= share <= 0.9

    def test_deterministic_under_seed(self):
        a = assign_splits(make_manifest(), seed=17)
        b = assign_splits(make_manifest(), seed=17)
        pd.testing.assert_series_equal(a["split"], b["split"])

    def test_output_passes_split_schema(self, split_df):
        from accentroute.schema import validate_manifest

        validate_manifest(split_df, stage="split")


class TestSpeakerReport:
    def test_report_flags_single_source_test(self, split_df, tmp_path: Path):
        summary = write_speaker_report(split_df, tmp_path / "speakers.csv")
        assert (tmp_path / "speakers.csv").exists()
        by_label = summary.set_index("accent_label")
        # the en-US test set has a single source, CV (EdAcc sits in ood_test) → flag
        assert bool(by_label.loc["en-US", "single_source_test"]) is True
        # the L1-Korean test set has two sources, CV + L2-ARCTIC
        assert by_label.loc["L1-Korean", "n_test_sources"] == 2
        assert bool(by_label.loc["L1-Korean", "single_source_test"]) is False

    def test_speaker_csv_is_reviewable(self, split_df, tmp_path: Path):
        write_speaker_report(split_df, tmp_path / "speakers.csv")
        table = pd.read_csv(tmp_path / "speakers.csv")
        assert {"speaker_key", "source", "accent_label", "split", "n_clips"} <= set(table.columns)
        # one row per speaker
        assert table["speaker_key"].is_unique
