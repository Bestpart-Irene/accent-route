"""Scoped deduplication (v1.2): speaker_key assignment, within-YouTube speaker merging,
and near-duplicate detection on Common Voice.

The architecture was built to scale from the start: ANN candidate edges + union-find.
Global cross-source clustering (backlog) only needs candidate_edges to swap its exact
nearest-neighbor search for faiss; the signature stays the same.
speaker_key is the key the split is built on — its correctness is what makes every
headline number trustworthy.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def assign_speaker_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Default speaker_key = f"{source}:{speaker_id_raw}"."""
    out = df.copy()
    out["speaker_key"] = out["source"] + ":" + out["speaker_id_raw"]
    return out


def candidate_edges(
    embs: np.ndarray, k: int = 20, sim_threshold: float = 0.45
) -> list[tuple[int, int]]:
    """embs: [n, d], L2-normalized. Builds candidate edges from top-k cosine neighbors,
    avoiding the full O(n²) similarity matrix.

    In the scope that matters here n is only a few hundred (within YouTube), so sklearn's
    exact neighbor search is plenty; switching to faiss only touches this function.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(embs) < 2:
        return []
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(embs)), metric="cosine").fit(embs)
    dist, idx = nn.kneighbors(embs)
    return [
        (i, int(j))
        for i in range(len(embs))
        for j, d in zip(idx[i][1:], dist[i][1:])
        if 1.0 - d >= sim_threshold
    ]


def union_find_clusters(n: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def dedup_youtube_speakers(
    df: pd.DataFrame,
    centroids: dict[str, np.ndarray],
    sim_threshold: float = 0.45,
) -> pd.DataFrame:
    """Within YouTube: speaker_keys with similar ECAPA centroids collapse to the smallest
    key in their cluster.

    The speaker-disjoint split only truly holds once the same interviewee appearing across
    channels and videos has been merged. Non-YouTube rows pass through untouched (global
    cross-source clustering is on the backlog).
    """
    out = df.copy()
    keys = sorted(k for k in centroids if k.startswith("youtube:"))
    if len(keys) < 2:
        return out
    embs = np.stack([centroids[k] for k in keys])
    labels = union_find_clusters(len(keys), candidate_edges(embs, sim_threshold=sim_threshold))
    canonical: dict[int, str] = {}
    for key, lab in zip(keys, labels):
        canonical.setdefault(lab, key)  # keys are sorted → smallest key in the cluster
    remap = {key: canonical[lab] for key, lab in zip(keys, labels)}
    mask = out["source"] == "youtube"
    out.loc[mask, "speaker_key"] = out.loc[mask, "speaker_key"].map(lambda k: remap.get(k, k))
    return out


def find_near_duplicates(
    df: pd.DataFrame,
    max_dur_delta_s: float = 0.5,
    min_jaccard: float = 0.8,
) -> pd.DataFrame:
    """Transcript Jaccard ≥ min_jaccard and |Δduration| ≤ max_dur_delta_s → keep one,
    reject the rest.

    Sorts by duration and pairs within a sliding window to avoid O(n²). The clip with the
    smallest clip_id survives; the others get status=rejected and
    reject_reason="near_duplicate".
    """
    out = df.copy()
    cand = out[out["status"] == "accepted"].dropna(subset=["transcript"])
    order = cand.sort_values(["duration_s", "clip_id"]).index.to_list()
    tokens = {i: frozenset(str(out.at[i, "transcript"]).split()) for i in order}
    rejected: set = set()
    for a_pos, i in enumerate(order):
        if i in rejected:
            continue
        for j in order[a_pos + 1 :]:
            if out.at[j, "duration_s"] - out.at[i, "duration_s"] > max_dur_delta_s:
                break  # sorted by duration, so everything after this is further away
            if j in rejected:
                continue
            ti, tj = tokens[i], tokens[j]
            union = len(ti | tj)
            if union and len(ti & tj) / union >= min_jaccard:
                rejected.add(j)
    out.loc[list(rejected), "status"] = "rejected"
    out.loc[list(rejected), "reject_reason"] = "near_duplicate"
    return out


def calibration_table(pos_sims: np.ndarray, neg_sims: np.ndarray) -> pd.DataFrame:
    """Calibration histogram data: similarity distribution of same-speaker pairs vs.
    cross-source hard negatives."""
    return pd.DataFrame(
        {
            "similarity": np.concatenate([pos_sims, neg_sims]),
            "pair_type": ["same_speaker"] * len(pos_sims)
            + ["cross_source_negative"] * len(neg_sims),
        }
    )


def suggest_threshold(
    pos_sims: np.ndarray, neg_sims: np.ndarray, max_false_merge: float = 1e-3
) -> float:
    """Lowest threshold satisfying the false-merge rate constraint (maximizes same-speaker
    recall)."""
    for thr in np.sort(np.unique(np.concatenate([pos_sims, neg_sims]))):
        if (neg_sims >= thr).mean() <= max_false_merge:
            return float(thr)
    return 1.0
