"""speaker-disjoint 切分:同一 speaker_key 绝不跨 split —— 头条数字可信度的基石。

规则:
  - edacc → ood_test(只作域外);youtube(weak)→ train(决策 #4:弱标签绝不进评测)
  - 其余源按 (accent_label, source) 分层,speaker 级分配:
    每层 ≥3 speakers 保证 train/val/test 各非空(4 人金标层 → 2/1/1),
    这就是「测试集每类尽量 ≥2 源」的机制
  - 被拒行 split=unassigned
label_source 由源决定:l2_arctic/edacc=gold,common_voice=self_report,youtube=weak。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from accentroute.schema import validate_manifest

LABEL_SOURCE_BY_SOURCE = {
    "l2_arctic": "gold",
    "edacc": "gold",
    "common_voice": "self_report",
    "youtube": "weak",
}

_FIXED_SPLIT_BY_SOURCE = {"edacc": "ood_test", "youtube": "train"}


def _quota(n_speakers: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    """(n_test, n_val)。≥3 人保证三个 split 各至少 1 人;1–2 人优先 train/test。"""
    if n_speakers <= 1:
        return (0, 0)
    if n_speakers == 2:
        return (1, 0)
    n_test = max(1, round(n_speakers * ratios[2]))
    n_val = max(1, round(n_speakers * ratios[1]))
    while n_test + n_val >= n_speakers:  # train 必须非空
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_test, n_val


def assign_splits(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 17,
) -> pd.DataFrame:
    """qc manifest(含 speaker_key)→ split manifest(过 split 阶段校验)。"""
    out = df.copy()
    out["label_source"] = out["source"].map(LABEL_SOURCE_BY_SOURCE)
    for col in ("consensus_score", "evidence_level"):
        if col not in out.columns:
            out[col] = None
    out["split"] = "unassigned"

    accepted = out["status"] == "accepted"
    for source, fixed in _FIXED_SPLIT_BY_SOURCE.items():
        out.loc[accepted & (out["source"] == source), "split"] = fixed

    # speaker 级表:每个 speaker 恰好一行(标签取众数),保证不会被分到两个 split
    pool = out[accepted & ~out["source"].isin(_FIXED_SPLIT_BY_SOURCE)]
    speakers = (
        pool.groupby("speaker_key")
        .agg(
            accent_label=("accent_label", lambda s: s.mode().iloc[0]),
            source=("source", "first"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    spk_split: dict[str, str] = {}
    for (_label, _source), grp in speakers.groupby(["accent_label", "source"]):
        keys = sorted(grp["speaker_key"])
        rng.shuffle(keys)
        n_test, n_val = _quota(len(keys), ratios)
        for i, key in enumerate(keys):
            if i < n_test:
                spk_split[key] = "test"
            elif i < n_test + n_val:
                spk_split[key] = "val"
            else:
                spk_split[key] = "train"

    mask = accepted & ~out["source"].isin(_FIXED_SPLIT_BY_SOURCE)
    out.loc[mask, "split"] = out.loc[mask, "speaker_key"].map(spk_split)

    validate_manifest(out, stage="split")
    return out


def write_speaker_report(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """可复核的 speaker 清单(CSV)+ 逐类测试集源数摘要(返回值)。

    single_source_test=True 的类,其域内测试结论必须限定措辞(confounding 防线之一)。
    """
    assigned = df[df["split"] != "unassigned"]
    table = (
        assigned.groupby("speaker_key")
        .agg(
            source=("source", "first"),
            accent_label=("accent_label", lambda s: s.mode().iloc[0]),
            split=("split", "first"),
            n_clips=("clip_id", "size"),
        )
        .reset_index()
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    test_rows = assigned[assigned["split"] == "test"]
    summary = (
        test_rows.groupby("accent_label")
        .agg(
            n_test_sources=("source", "nunique"),
            n_test_speakers=("speaker_key", "nunique"),
            n_test_clips=("clip_id", "size"),
        )
        .reset_index()
    )
    summary["single_source_test"] = summary["n_test_sources"] < 2
    return summary
