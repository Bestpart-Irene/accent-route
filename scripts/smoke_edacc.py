#!/usr/bin/env python
"""End-to-end pipeline smoke test on real EdAcc audio.

This is NOT the designed experiment. EdAcc is the out-of-domain test set, and it covers
only part of the taxonomy, so the number this produces is not the headline C−B. What it
does prove is that ingest → taxonomy → filter → dedup/split → train → evaluate runs on
real speech, and it surfaces integration bugs before the Common Voice and L2-ARCTIC data
land.

To keep the split honest even here, the split is speaker-disjoint and the reported metric
is macro-F1 over the classes EdAcc actually supports.

Usage:
    python scripts/smoke_edacc.py extract     # parquet → 16k mono wavs
    python scripts/smoke_edacc.py pipeline    # ingest → filter → split
    python scripts/smoke_edacc.py train       # LoRA training on MPS/CPU
    python scripts/smoke_edacc.py report      # metrics + baselines
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RAW = Path("data/raw/edacc_hf")
WAV = Path("data/work/edacc_wav")
MAN = Path("data/manifests")
RUN = Path("runs/smoke_edacc")
MIN_SPEAKERS_PER_CLASS = 4


def extract() -> None:
    from accentroute.ingest.edacc_hf import extract_audio

    n = extract_audio(RAW, WAV, skip_existing=True)
    print(f"wrote {n} wav files to {WAV} (total {len(list(WAV.glob('*.wav')))})")


def pipeline() -> None:
    from accentroute.dedup import assign_speaker_keys
    from accentroute.filter import FilterConfig, apply_filters
    from accentroute.ingest.base import run_ingest
    from accentroute.ingest.edacc_hf import EdAccHFIngestor
    from accentroute.split import assign_splits
    from accentroute.taxonomy import load_taxonomy

    MAN.mkdir(parents=True, exist_ok=True)
    raw_path = run_ingest(EdAccHFIngestor(root=RAW), MAN / "smoke_raw.parquet")
    raw = pd.read_parquet(raw_path)
    print(f"ingested {len(raw)} clips, {raw.speaker_id_raw.nunique()} speakers")

    tax = load_taxonomy("configs/taxonomy_v1.yaml")
    cfg = FilterConfig.from_yaml("configs/filter.yaml")

    # Silero VAD only; whisper-tiny transcription and LID are skipped here because EdAcc
    # is already known-English conversational speech and the ASR pass would dominate
    # runtime. The transcript column stays null, which the qc schema allows.
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad_model = load_silero_vad()

    import soundfile as sf
    import torch

    def vad_fn(wav, sr):
        return get_speech_timestamps(torch.as_tensor(wav, dtype=torch.float32), vad_model)

    def audio_loader(clip_id: str):
        wav, sr = sf.read(WAV / f"{clip_id}.wav", dtype="float64")
        return wav, sr

    qc = apply_filters(
        raw, cfg, taxonomy=tax, audio_loader=audio_loader, vad_fn=vad_fn,
        transcribe_fn=lambda wav: None, lid_fn=lambda text: ("en", 1.0),
    )
    qc.to_parquet(MAN / "smoke_qc.parquet", index=False)
    print("filter outcome:", qc["status"].value_counts().to_dict())
    print("reject reasons:", qc["reject_reason"].value_counts().to_dict())
    print("unmapped accents (top):", dict(tax.unmapped_counts.most_common(8)))

    accepted = qc[qc["status"] == "accepted"]
    per_class = accepted.groupby("accent_label")["speaker_id_raw"].nunique()
    keep = per_class[per_class >= MIN_SPEAKERS_PER_CLASS].index.tolist()
    print(f"classes with >={MIN_SPEAKERS_PER_CLASS} speakers: {keep}")
    qc.loc[~qc["accent_label"].isin(keep) & (qc["status"] == "accepted"), ["status", "reject_reason"]] = [
        "rejected", "too_few_speakers_for_smoke_test",
    ]

    # The production rule pins EdAcc to ood_test. This diagnostic needs it trainable, so
    # it opts out explicitly — the rule itself is never edited.
    split = assign_splits(
        assign_speaker_keys(qc), ratios=(0.7, 0.15, 0.15), seed=17,
        fixed_split_by_source={},
    )
    split.to_parquet(MAN / "smoke_split.parquet", index=False)
    print(split[split.split != "unassigned"].groupby(["accent_label", "split"]).size())


def _dataset(manifest: Path, split_name: str, classes: list[str]):
    import torch
    from transformers import WhisperFeatureExtractor

    from accentroute.model.pooling import valid_encoder_frames

    df = pd.read_parquet(manifest)
    df = df[(df["split"] == split_name) & df["accent_label"].isin(classes)].reset_index(drop=True)
    extractor = WhisperFeatureExtractor()
    index = {c: i for i, c in enumerate(classes)}

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            import soundfile as sf

            row = df.iloc[i]
            wav, _ = sf.read(WAV / f"{row['clip_id']}.wav", dtype="float32")
            wav = wav[: 30 * 16000]
            feats = extractor(wav, sampling_rate=16000, return_attention_mask=True,
                              return_tensors="pt")
            return {
                "input_features": feats.input_features[0],
                "n_valid": int(valid_encoder_frames(feats.attention_mask)[0]),
                "label": index[row["accent_label"]],
            }

    return _DS(), df


def train() -> None:
    import torch

    from accentroute.model.whisper_lora import build_model
    from accentroute.train import TrainConfig
    from accentroute.train import train as run_train

    split = pd.read_parquet(MAN / "smoke_split.parquet")
    classes = sorted(split[split["split"] == "train"]["accent_label"].dropna().unique())
    print("training classes:", classes)

    train_ds, _ = _dataset(MAN / "smoke_split.parquet", "train", classes)
    val_ds, _ = _dataset(MAN / "smoke_split.parquet", "val", classes)
    print(f"train {len(train_ds)} clips, val {len(val_ds)} clips")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(n_classes=len(classes)).to(device)

    cfg = TrainConfig(
        arm="smoke_edacc", budget="epoch_matched", seed=17, out_dir=RUN,
        epochs_c=3, batch_size=8, lr=1e-3, warmup_ratio=0.1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        base_model="openai/whisper-small", n_classes=len(classes),
        total_steps=int(sys.argv[2]) if len(sys.argv) > 2 else 300,
        shared_config_hash="smoke",
    )
    result = run_train(cfg, model=model, train_ds=train_ds, val_ds=val_ds)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "classes.json").write_text(json.dumps(classes))
    print(result)


def report() -> None:
    import torch

    from accentroute.eval.baselines import majority_baseline
    from accentroute.eval.metrics import confusion, macro_f1
    from accentroute.model.whisper_lora import build_model

    classes = json.loads((RUN / "classes.json").read_text())
    test_ds, test_df = _dataset(MAN / "smoke_split.parquet", "test", classes)
    train_df = pd.read_parquet(MAN / "smoke_split.parquet")
    train_df = train_df[train_df["split"] == "train"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(n_classes=len(classes)).to(device)
    state = torch.load(RUN / "ckpt_best.pt", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    preds, golds = [], []
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=8,
        collate_fn=lambda items: (
            torch.stack([it["input_features"] for it in items]),
            torch.as_tensor([it["n_valid"] for it in items]),
            torch.as_tensor([it["label"] for it in items]),
        ),
    )
    with torch.no_grad():
        for feats, n_valid, ys in loader:
            logits = model(feats.to(device), n_valid.to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            golds.extend(ys.tolist())

    y_true = np.array([classes[i] for i in golds])
    y_pred = np.array([classes[i] for i in preds])
    maj = majority_baseline(train_df["accent_label"].dropna().to_numpy(), len(y_true))

    out = {
        "classes": classes,
        "n_test_clips": len(y_true),
        "n_test_speakers": int(test_df["speaker_key"].nunique()),
        "model_macro_f1": macro_f1(y_true, y_pred, labels=classes),
        "majority_baseline_macro_f1": macro_f1(y_true, maj, labels=classes),
        "confusion": confusion(y_true, y_pred, labels=classes).tolist(),
    }
    (RUN / "smoke_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=2))
    print("confusion (rows=true, cols=pred):")
    print(pd.DataFrame(out["confusion"], index=classes, columns=classes).to_string())


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    {"extract": extract, "pipeline": pipeline, "train": train, "report": report}[action]()
