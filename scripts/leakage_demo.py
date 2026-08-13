#!/usr/bin/env python
"""Measure how much speaker leakage inflates an accent-classification score.

Same clips, same model, same optimizer budget, same seed. The only thing that differs is
how the data was split:

  strict : speaker-disjoint — a speaker appears in exactly one of train/val/test
  leaky  : clip-level random — the common mistake, which puts the same voice on both sides

The gap between the two test scores is the size of the illusion. It is the reason
published accent-classification accuracies should be read with suspicion, and it is
measured here rather than asserted.

Usage:
    python scripts/leakage_demo.py run [steps]
    python scripts/leakage_demo.py report
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

WAV = Path(os.environ.get("GLOBE_WAV", "data/work/globe_wav"))
MAN = Path(os.environ.get("GLOBE_MANIFESTS", "data/manifests"))
RUN = Path(os.environ.get("LEAKAGE_RUN", "runs/leakage_demo"))
SEED = 17


def _pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset(df: pd.DataFrame, split_name: str, classes: list[str]):
    import torch
    from transformers import WhisperFeatureExtractor

    from accentroute.model.pooling import valid_encoder_frames

    rows = df[(df["split"] == split_name) & df["accent_label"].isin(classes)]
    rows = rows.reset_index(drop=True)
    extractor = WhisperFeatureExtractor()
    index = {c: i for i, c in enumerate(classes)}

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            import soundfile as sf

            row = rows.iloc[i]
            wav, _ = sf.read(WAV / f"{row['clip_id']}.wav", dtype="float32")
            wav = wav[: 30 * 16000]
            feats = extractor(wav, sampling_rate=16000, return_attention_mask=True,
                              return_tensors="pt")
            return {
                "input_features": feats.input_features[0],
                "n_valid": int(valid_encoder_frames(feats.attention_mask)[0]),
                "label": index[row["accent_label"]],
            }

    return _DS(), rows


def _train_and_score(tag: str, split_df: pd.DataFrame, classes: list[str], steps: int) -> dict:
    import torch

    from accentroute.eval.baselines import majority_baseline
    from accentroute.eval.metrics import confusion, macro_f1
    from accentroute.model.whisper_lora import build_model
    from accentroute.train import TrainConfig
    from accentroute.train import train as run_train

    train_ds, train_rows = _dataset(split_df, "train", classes)
    val_ds, _ = _dataset(split_df, "val", classes)
    test_ds, test_rows = _dataset(split_df, "test", classes)

    device = _pick_device(torch)
    model = build_model(n_classes=len(classes)).to(device)
    cfg = TrainConfig(
        arm=f"leakage_{tag}", budget="epoch_matched", seed=SEED, out_dir=RUN / tag,
        epochs_c=3, batch_size=16, lr=1e-3, warmup_ratio=0.1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        base_model="openai/whisper-small", n_classes=len(classes),
        total_steps=steps, shared_config_hash="leakage-demo",
    )
    run_train(cfg, model=model, train_ds=train_ds, val_ds=val_ds)

    model.eval()
    preds, golds = [], []
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16,
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
    maj = majority_baseline(train_rows["accent_label"].to_numpy(), len(y_true))

    shared = len(set(train_rows.speaker_key) & set(test_rows.speaker_key))
    return {
        "split_rule": tag,
        "n_train_clips": len(train_rows),
        "n_test_clips": len(test_rows),
        "n_train_speakers": int(train_rows.speaker_key.nunique()),
        "n_test_speakers": int(test_rows.speaker_key.nunique()),
        "speakers_in_both_train_and_test": shared,
        "test_macro_f1": macro_f1(y_true, y_pred, labels=classes),
        "majority_baseline_macro_f1": macro_f1(y_true, maj, labels=classes),
        "confusion": confusion(y_true, y_pred, labels=classes).tolist(),
    }


def run() -> None:
    from accentroute.dedup import assign_speaker_keys
    from accentroute.split import assign_splits, assign_splits_leaky

    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    qc = pd.read_parquet(MAN / "globe_qc.parquet")
    qc = assign_speaker_keys(qc) if "speaker_key" not in qc.columns else qc
    classes = sorted(qc[qc.status == "accepted"]["accent_label"].dropna().unique())
    print(f"classes: {classes}, accepted clips: {(qc.status == 'accepted').sum()}")

    RUN.mkdir(parents=True, exist_ok=True)
    results = []
    for tag, fn in [
        ("strict", lambda d: assign_splits(d, ratios=(0.7, 0.15, 0.15), seed=SEED,
                                           fixed_split_by_source={})),
        ("leaky", lambda d: assign_splits_leaky(d, ratios=(0.7, 0.15, 0.15), seed=SEED)),
    ]:
        split_df = fn(qc)
        print(f"\n=== {tag} split ===")
        res = _train_and_score(tag, split_df, classes, steps)
        print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=2))
        results.append(res)

    (RUN / "leakage_results.json").write_text(
        json.dumps({"classes": classes, "steps": steps, "seed": SEED, "runs": results},
                   indent=2)
    )
    report()


def report() -> None:
    data = json.loads((RUN / "leakage_results.json").read_text())
    by_tag = {r["split_rule"]: r for r in data["runs"]}
    strict, leaky = by_tag["strict"], by_tag["leaky"]
    gap = leaky["test_macro_f1"] - strict["test_macro_f1"]

    print("\n" + "=" * 64)
    print("Same clips, same model, same budget, same seed. Only the split differs.")
    print("=" * 64)
    print(f"speaker-disjoint split : macro-F1 {strict['test_macro_f1']:.3f}  "
          f"({strict['speakers_in_both_train_and_test']} speakers on both sides)")
    print(f"clip-level random split: macro-F1 {leaky['test_macro_f1']:.3f}  "
          f"({leaky['speakers_in_both_train_and_test']} speakers on both sides)")
    print(f"\ninflation from speaker leakage: {gap:+.3f} macro-F1")
    print(f"majority-class baseline: {strict['majority_baseline_macro_f1']:.3f}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "report": report}[action]()
