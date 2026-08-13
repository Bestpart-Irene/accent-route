"""范围化去重(v1.2):speaker_key 分配、YouTube 集内说话人合并、CV 近重复检测。

架构从一开始按可扩展设计:ANN 候选边 + union-find,全局跨源聚类(backlog)
只需把 candidate_edges 的精确近邻换成 faiss,签名不变。
split 的键是 speaker_key —— 它的正确性决定所有头条数字的可信度。
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def assign_speaker_keys(df: pd.DataFrame) -> pd.DataFrame:
    """缺省 speaker_key = f"{source}:{speaker_id_raw}"。"""
    out = df.copy()
    out["speaker_key"] = out["source"] + ":" + out["speaker_id_raw"]
    return out


def candidate_edges(
    embs: np.ndarray, k: int = 20, sim_threshold: float = 0.45
) -> list[tuple[int, int]]:
    """embs: [n, d] L2 归一化。top-k 余弦近邻生成候选边,避免 O(n²) 全矩阵。

    核心范围 n 只有数百(YouTube 集内),sklearn 精确近邻足够;
    换 faiss 只改这里的实现。
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
    """YouTube 集内:ECAPA 质心相似的 speaker_key 合并为簇内最小键。

    跨频道/视频的同一访谈对象合并后,speaker-disjoint 切分才真正成立。
    非 youtube 行原样返回(全局跨源聚类在 backlog)。
    """
    out = df.copy()
    keys = sorted(k for k in centroids if k.startswith("youtube:"))
    if len(keys) < 2:
        return out
    embs = np.stack([centroids[k] for k in keys])
    labels = union_find_clusters(len(keys), candidate_edges(embs, sim_threshold=sim_threshold))
    canonical: dict[int, str] = {}
    for key, lab in zip(keys, labels):
        canonical.setdefault(lab, key)  # keys 已排序 → 簇内最小键
    remap = {key: canonical[lab] for key, lab in zip(keys, labels)}
    mask = out["source"] == "youtube"
    out.loc[mask, "speaker_key"] = out.loc[mask, "speaker_key"].map(lambda k: remap.get(k, k))
    return out


def find_near_duplicates(
    df: pd.DataFrame,
    max_dur_delta_s: float = 0.5,
    min_jaccard: float = 0.8,
) -> pd.DataFrame:
    """转写 Jaccard ≥ min_jaccard 且 |Δ时长| ≤ max_dur_delta_s → 留一拒余。

    按时长排序 + 滑窗配对,避免 O(n²);保留 clip_id 最小的一条,
    其余 status=rejected, reject_reason="near_duplicate"。
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
                break  # 已按时长排序,后面只会更远
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
    """校准直方图数据:同人对 vs 跨源 hard negatives 的相似度分布。"""
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
    """满足误合并率约束的最低阈值(最大化同人召回)。"""
    for thr in np.sort(np.unique(np.concatenate([pos_sims, neg_sims]))):
        if (neg_sims >= thr).mean() <= max_false_merge:
            return float(thr)
    return 1.0
