"""Speaker-disjoint split: a speaker_key never straddles two splits — the foundation for
trusting any headline number.

Rules:
  - edacc → ood_test (out-of-domain only); youtube (weak) → train (decision #4: weak
    labels never enter evaluation)
  - every other source is stratified by (accent_label, source) and assigned at the speaker
    level: a stratum with ≥3 speakers guarantees train/val/test are all non-empty (a
    4-speaker gold stratum → 2/1/1), which is the mechanism behind "aim for ≥2 sources per
    class in the test set"
  - rejected rows get split=unassigned
label_source follows the source: l2_arctic/edacc=gold, common_voice=self_report,
youtube=weak.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from accentroute.schema import validate_manifest

LABEL_SOURCE_BY_SOURCE = {
    "l2_arctic": "gold",
    "edacc": "gold",
    "common_voice": "self_report",
    "globe": "self_report",  # Common-Voice-derived; the accent field is the same self-report
    "youtube": "weak",
}

_FIXED_SPLIT_BY_SOURCE = {"edacc": "ood_test", "youtube": "train"}


def _quota(n_speakers: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    """(n_test, n_val). With ≥3 speakers every split gets at least one; with 1-2 speakers,
    train/test take priority."""
    if n_speakers <= 1:
        return (0, 0)
    if n_speakers == 2:
        return (1, 0)
    n_test = max(1, round(n_speakers * ratios[2]))
    n_val = max(1, round(n_speakers * ratios[1]))
    while n_test + n_val >= n_speakers:  # train must stay non-empty
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
    fixed_split_by_source: dict[str, str] | None = None,
) -> pd.DataFrame:
    """qc manifest (with speaker_key) → split manifest (validated at the split stage).

    fixed_split_by_source overrides which sources bypass the speaker-level allocation.
    The default is the production rule (edacc → ood_test, youtube → train); a diagnostic
    run that needs, say, EdAcc to be trainable has to pass an explicit override rather
    than edit the rule itself.
    """
    fixed_by_source = (
        _FIXED_SPLIT_BY_SOURCE if fixed_split_by_source is None else fixed_split_by_source
    )
    out = df.copy()
    out["label_source"] = out["source"].map(LABEL_SOURCE_BY_SOURCE)
    for col in ("consensus_score", "evidence_level"):
        if col not in out.columns:
            out[col] = None
    out["split"] = "unassigned"

    accepted = out["status"] == "accepted"
    for source, fixed in fixed_by_source.items():
        out.loc[accepted & (out["source"] == source), "split"] = fixed

    # Speaker-level table: exactly one row per speaker (label = mode), so no speaker can
    # ever be assigned to two splits
    pool = out[accepted & ~out["source"].isin(fixed_by_source)]
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

    mask = accepted & ~out["source"].isin(fixed_by_source)
    out.loc[mask, "split"] = out.loc[mask, "speaker_key"].map(spk_split)

    validate_manifest(out, stage="split")
    return out


def write_speaker_report(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Auditable speaker roster (CSV) + a per-class summary of test-set source counts
    (the return value).

    For classes with single_source_test=True, any in-domain test claim must be worded with
    that caveat — one of the lines of defense against confounding.
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
