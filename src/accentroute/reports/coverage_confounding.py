"""source × accent 混杂矩阵与 EdAcc 覆盖报告(G1 门禁输入,决策 #3/#6)。

模型可能学到「数据源指纹」而非口音——类别与源高度绑定时,speaker-disjoint
也挡不住 source shortcut。这里量化绑定程度:单源时长占比 > dominance 的类
标记 confounded,datasheet 必须携带,结论措辞必须限定。
"""

import pandas as pd

from accentroute.schema import ACCENTS

_REQUIRED = ["source", "accent_label", "speaker_id_raw", "duration_s"]


def source_accent_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """逐 (source, accent_label) 的 n_speakers / n_clips / hours。"""
    labeled = df[df["accent_label"].notna()][_REQUIRED]
    out = (
        labeled.groupby(["source", "accent_label"], as_index=False)
        .agg(
            n_speakers=("speaker_id_raw", "nunique"),
            n_clips=("speaker_id_raw", "size"),
            hours=("duration_s", "sum"),
        )
    )
    out["hours"] = out["hours"] / 3600.0
    return out


def flag_confounded(matrix: pd.DataFrame, dominance: float = 0.9) -> pd.DataFrame:
    """逐类的主导源与时长占比;占比 > dominance → confounded=True。

    调用方负责先过滤矩阵范围(例如剔除只进 ood_test 的 edacc)。
    """
    rows = []
    for label, grp in matrix.groupby("accent_label"):
        total = grp["hours"].sum()
        top = grp.loc[grp["hours"].idxmax()]
        share = top["hours"] / total if total > 0 else 0.0
        rows.append(
            {
                "accent_label": label,
                "n_sources": len(grp),
                "dominant_source": top["source"],
                "dominant_share": share,
                "confounded": share > dominance,
            }
        )
    return pd.DataFrame(rows)


def edacc_class_coverage(df: pd.DataFrame, min_speakers: int = 5) -> pd.DataFrame:
    """EdAcc 每类 speaker 覆盖;n_speakers < min_speakers 的类 include=False。

    include=True 的类构成 supported-class 集合:域外 macro-F1 只在这些类上算
    (称 supported-class macro-F1,不与完整 8 类横比)。8 类全部列出,缺失补零。
    """
    if (df["source"] != "edacc").any():
        raise ValueError("edacc_class_coverage expects an edacc-only manifest slice")
    agg = (
        df[df["accent_label"].notna()]
        .groupby("accent_label")
        .agg(
            n_speakers=("speaker_id_raw", "nunique"),
            n_clips=("speaker_id_raw", "size"),
            hours=("duration_s", "sum"),
        )
    )
    agg["hours"] = agg["hours"] / 3600.0
    out = agg.reindex(ACCENTS).fillna(0).reset_index(names="accent_label")
    out[["n_speakers", "n_clips"]] = out[["n_speakers", "n_clips"]].astype(int)
    out["include"] = out["n_speakers"] >= min_speakers
    return out
