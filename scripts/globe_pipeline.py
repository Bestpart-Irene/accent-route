#!/usr/bin/env python
"""Fetch a balanced GLOBE subset and run it through the pipeline.

GLOBE supplies the four native varieties (and is the only reachable source for en-AU).
The four L1-* classes are absent by construction, so this run covers half the taxonomy;
L2-ARCTIC fills the rest once its access form clears.

Usage:
    python scripts/globe_pipeline.py fetch [per_class] [max_per_speaker]
    python scripts/globe_pipeline.py pipeline
    python scripts/globe_pipeline.py report
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WAV = Path(os.environ.get("GLOBE_WAV", "data/work/globe_wav"))
MAN = Path(os.environ.get("GLOBE_MANIFESTS", "data/manifests"))
NATIVE_CLASSES = {"en-US", "en-GB", "en-AU", "en-IN"}


def fetch() -> None:
    from accentroute.ingest.globe import fetch_globe_subset
    from accentroute.taxonomy import load_taxonomy

    per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    max_per_speaker = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    tax = load_taxonomy("configs/taxonomy_v1.yaml")
    df = fetch_globe_subset(
        WAV, tax, per_class=per_class, max_per_speaker=max_per_speaker,
        expected_labels=NATIVE_CLASSES,
    )
    MAN.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["accent_label"]).to_parquet(MAN / "globe_raw.parquet", index=False)

    print(f"fetched {len(df)} clips, {df.speaker_id_raw.nunique()} speakers, "
          f"{df.duration_s.sum() / 3600:.1f} h")
    per = df.groupby("accent_label").agg(
        clips=("clip_id", "size"), speakers=("speaker_id_raw", "nunique"),
        hours=("duration_s", "sum"))
    per["hours"] = (per.hours / 3600).round(2)
    print(per.to_string())

    # `datasets` streaming leaves HTTP worker threads that abort the interpreter during
    # finalization ("PyGILState_Release: thread state must be current"), long after the
    # work is durably on disk. Skipping finalization is the workaround — but only once the
    # outputs are verified present, so a genuine failure still exits non-zero rather than
    # being papered over.
    manifest = MAN / "globe_raw.parquet"
    n_wavs = len(list(WAV.glob("*.wav")))
    if not manifest.exists() or n_wavs < len(df):
        raise SystemExit(
            f"fetch incomplete: manifest={manifest.exists()}, wavs={n_wavs}, rows={len(df)}"
        )
    print(f"verified {n_wavs} wavs and the manifest on disk; exiting before finalization")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def pipeline() -> None:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    from accentroute.dedup import assign_speaker_keys, find_near_duplicates
    from accentroute.filter import FilterConfig, apply_filters
    from accentroute.split import assign_splits, write_speaker_report
    from accentroute.taxonomy import load_taxonomy

    raw = pd.read_parquet(MAN / "globe_raw.parquet")
    tax = load_taxonomy("configs/taxonomy_v1.yaml")
    cfg = FilterConfig.from_yaml("configs/filter.yaml")
    vad_model = load_silero_vad()

    def vad_fn(wav, sr):
        return get_speech_timestamps(torch.as_tensor(wav, dtype=torch.float32), vad_model)

    def audio_loader(clip_id: str):
        return sf.read(WAV / f"{clip_id}.wav", dtype="float64")

    # GLOBE ships Common Voice transcripts, but this run keeps the transcript column null:
    # language ID is not informative on a corpus that is English by construction, and the
    # ASR pass would dominate runtime.
    qc = apply_filters(
        raw, cfg, taxonomy=tax, audio_loader=audio_loader, vad_fn=vad_fn,
        transcribe_fn=lambda wav: None, lid_fn=lambda text: ("en", 1.0),
    )
    print("filter outcome:", qc["status"].value_counts().to_dict())
    print("reject reasons:", qc["reject_reason"].value_counts().to_dict())

    keyed = assign_speaker_keys(qc)
    deduped = find_near_duplicates(keyed)
    n_dup = int((deduped["reject_reason"] == "near_duplicate").sum())
    print(f"near-duplicates rejected: {n_dup}")

    split = assign_splits(deduped, ratios=(0.7, 0.15, 0.15), seed=17)
    split.to_parquet(MAN / "globe_split.parquet", index=False)
    summary = write_speaker_report(split, MAN / "globe_speakers.csv")
    print(split[split.split != "unassigned"].groupby(["accent_label", "split"]).size())
    print("\ntest-set source coverage per class:")
    print(summary.to_string(index=False))


def report() -> None:
    from accentroute.reports.coverage_confounding import (
        flag_confounded,
        flag_duration_confound,
        source_accent_matrix,
    )

    split = pd.read_parquet(MAN / "globe_split.parquet")
    assigned = split[split.split.isin(["train", "val", "test"])]

    leak = int((assigned.groupby("speaker_key")["split"].nunique() > 1).sum())
    print(f"speaker_keys spanning more than one split: {leak}")

    print("\nspeakers per class per split:")
    print(assigned.groupby(["accent_label", "split"]).speaker_key.nunique()
          .unstack(fill_value=0).to_string())

    matrix = source_accent_matrix(assigned)
    print("\nsource x accent matrix:")
    print(matrix.to_string(index=False))
    print("\nsource confounding flags:")
    print(flag_confounded(matrix).to_string(index=False))
    print("\nduration confounding flags:")
    print(flag_duration_confound(matrix).to_string(index=False))


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    {"fetch": fetch, "pipeline": pipeline, "report": report}[action]()
