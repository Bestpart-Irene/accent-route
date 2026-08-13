"""T7: scoped dedup — speaker keys, ANN + union-find speaker merging, CV near-duplicates,
and threshold calibration.
"""

import numpy as np
import pandas as pd

from accentroute.dedup import (
    assign_speaker_keys,
    calibration_table,
    candidate_edges,
    dedup_youtube_speakers,
    find_near_duplicates,
    suggest_threshold,
    union_find_clusters,
)


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class TestSpeakerKeys:
    def test_default_key(self):
        df = pd.DataFrame(
            [
                {"source": "common_voice", "speaker_id_raw": "abc"},
                {"source": "youtube", "speaker_id_raw": "chan1:v9"},
            ]
        )
        out = assign_speaker_keys(df)
        assert list(out["speaker_key"]) == ["common_voice:abc", "youtube:chan1:v9"]


class TestCandidateEdges:
    def test_near_identical_linked_orthogonal_not(self):
        embs = np.stack(
            [
                _unit([1.0, 0.01, 0.0]),
                _unit([1.0, 0.02, 0.0]),  # nearly collinear with row 0
                _unit([0.0, 0.0, 1.0]),  # orthogonal
            ]
        )
        edges = candidate_edges(embs, k=2, sim_threshold=0.45)
        pairs = {tuple(sorted(e)) for e in edges}
        assert (0, 1) in pairs
        assert all(2 not in p for p in pairs)


class TestUnionFind:
    def test_chain_merges(self):
        labels = union_find_clusters(4, [(0, 1), (1, 2)])
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]

    def test_no_edges_all_singletons(self):
        assert len(set(union_find_clusters(3, []))) == 3


class TestDedupYoutubeSpeakers:
    def test_same_voice_across_videos_merged(self):
        df = assign_speaker_keys(
            pd.DataFrame(
                [
                    {"source": "youtube", "speaker_id_raw": "vidA:spk0"},
                    {"source": "youtube", "speaker_id_raw": "vidB:spk0"},
                    {"source": "youtube", "speaker_id_raw": "vidC:spk1"},
                    {"source": "common_voice", "speaker_id_raw": "cv1"},
                ]
            )
        )
        centroids = {
            "youtube:vidA:spk0": _unit([1.0, 0.05, 0.0]),
            "youtube:vidB:spk0": _unit([1.0, 0.03, 0.0]),  # same interviewee, different video
            "youtube:vidC:spk1": _unit([0.0, 1.0, 0.0]),
        }
        out = dedup_youtube_speakers(df, centroids, sim_threshold=0.45)
        yt = out[out["source"] == "youtube"]
        merged = yt[yt["speaker_id_raw"].str.contains("spk0")]["speaker_key"]
        assert merged.nunique() == 1
        assert yt[yt["speaker_id_raw"] == "vidC:spk1"]["speaker_key"].iloc[0] != merged.iloc[0]
        # non-youtube rows are left alone
        assert out[out["source"] == "common_voice"]["speaker_key"].iloc[0] == "common_voice:cv1"


class TestNearDuplicates:
    def _df(self, rows):
        base = {
            "source": "common_voice",
            "status": "accepted",
            "reject_reason": None,
        }
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_same_transcript_close_duration_rejected(self):
        df = self._df(
            [
                {"clip_id": "a", "duration_s": 6.0, "transcript": "the cat sat on the mat"},
                {"clip_id": "b", "duration_s": 6.2, "transcript": "the cat sat on the mat"},
            ]
        )
        out = find_near_duplicates(df)
        assert list(out["status"]) == ["accepted", "rejected"]
        assert out.iloc[1]["reject_reason"] == "near_duplicate"

    def test_different_transcript_kept(self):
        df = self._df(
            [
                {"clip_id": "a", "duration_s": 6.0, "transcript": "the cat sat on the mat"},
                {"clip_id": "b", "duration_s": 6.1, "transcript": "completely different words here now"},
            ]
        )
        out = find_near_duplicates(df)
        assert list(out["status"]) == ["accepted", "accepted"]

    def test_duration_gap_kept(self):
        df = self._df(
            [
                {"clip_id": "a", "duration_s": 6.0, "transcript": "the cat sat on the mat"},
                {"clip_id": "b", "duration_s": 9.0, "transcript": "the cat sat on the mat"},
            ]
        )
        out = find_near_duplicates(df)
        assert list(out["status"]) == ["accepted", "accepted"]


class TestCalibration:
    def test_table_and_threshold(self):
        rng = np.random.default_rng(0)
        pos = rng.normal(0.85, 0.03, 200)  # same-speaker pairs
        neg = rng.normal(0.10, 0.05, 2000)  # cross-source hard negatives
        table = calibration_table(pos, neg)
        assert set(table["pair_type"]) == {"same_speaker", "cross_source_negative"}
        assert len(table) == 2200
        thr = suggest_threshold(pos, neg, max_false_merge=1e-3)
        # contract: the lowest threshold that meets the false-merge budget (maximizing
        # same-speaker recall); it has to land between the two distributions
        assert (neg >= thr).mean() <= 1e-3
        assert (pos >= thr).mean() > 0.99
        assert neg.mean() < thr < pos.mean()
