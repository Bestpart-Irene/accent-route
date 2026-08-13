"""source × accent confounding matrix and EdAcc coverage report (inputs to the G1 gate,
decisions #3/#6).

A model can learn the "fingerprint of a data source" instead of the accent: when a class is
tightly bound to one source, even a speaker-disjoint split will not stop the source
shortcut. This module quantifies how tight that binding is — any class whose dominant
source holds more than `dominance` of its audio hours is flagged confounded, the flag must
travel with the datasheet, and conclusions about that class must be worded accordingly.
"""

import pandas as pd

from accentroute.schema import ACCENTS

_REQUIRED = ["source", "accent_label", "speaker_id_raw", "duration_s"]


def source_accent_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """n_speakers / n_clips / hours per (source, accent_label)."""
    labeled = df[df["accent_label"].notna()][_REQUIRED]
    out = (
        labeled.groupby(["source", "accent_label"], as_index=False)
        .agg(
            n_speakers=("speaker_id_raw", "nunique"),
            n_clips=("speaker_id_raw", "size"),
            hours=("duration_s", "sum"),
            median_duration_s=("duration_s", "median"),
            p10_duration_s=("duration_s", lambda s: s.quantile(0.10)),
            p90_duration_s=("duration_s", lambda s: s.quantile(0.90)),
        )
    )
    out["hours"] = out["hours"] / 3600.0
    return out


def flag_duration_confound(matrix: pd.DataFrame) -> pd.DataFrame:
    """Flag classes whose clip-duration range does not overlap every other class's.

    Duration is a source fingerprint in disguise. GLOBE's TTS utterances run around 4-6 s
    while L2-ARCTIC's run longer, so if each class comes from one source a model can read
    the clip length instead of the accent — and a speaker-disjoint split does nothing about
    it. Disjoint [p10, p90] ranges mean the shortcut is available.
    """
    per_class = (
        matrix.groupby("accent_label")
        .agg(p10=("p10_duration_s", "min"), p90=("p90_duration_s", "max"))
        .reset_index()
    )
    rows = []
    for _, row in per_class.iterrows():
        others = per_class[per_class["accent_label"] != row["accent_label"]]
        overlaps = (
            (others["p10"] <= row["p90"]) & (others["p90"] >= row["p10"])
        )
        rows.append(
            {
                "accent_label": row["accent_label"],
                "p10_duration_s": row["p10"],
                "p90_duration_s": row["p90"],
                "n_overlapping_classes": int(overlaps.sum()),
                "duration_disjoint": bool(len(others) > 0 and not overlaps.any()),
            }
        )
    return pd.DataFrame(rows)


def flag_confounded(matrix: pd.DataFrame, dominance: float = 0.9) -> pd.DataFrame:
    """Per class, the dominant source and its share of hours; share > dominance →
    confounded=True.

    The caller is responsible for scoping the matrix first (e.g. dropping edacc, which only
    ever lands in ood_test).
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
    """Per-class speaker coverage in EdAcc; classes with n_speakers < min_speakers get
    include=False.

    The include=True classes form the supported-class set: out-of-domain macro-F1 is
    computed over those classes only and reported as supported-class macro-F1, never
    compared against the full 8-class number. All 8 classes are listed, missing ones zeroed.
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
