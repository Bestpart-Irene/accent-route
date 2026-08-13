#!/usr/bin/env python
"""Train the accent classifier and evaluate it honestly.

Produces the numbers that actually mean something:

  in-domain   held-out GLOBE speakers — speakers the model never heard, same corpus
  OOD         EdAcc — a different corpus AND a different speaking style (spontaneous
              conversation vs GLOBE's read speech). This is the number that says whether
              the model learned accent or learned "read-speech accent".

Trained over several seeds, reported as mean±std, against a majority-class baseline, with
a per-class breakdown and a confusion matrix. EdAcc does not cover every class, so the OOD
figure is a supported-class macro-F1 paired with an in-domain control computed on that
same subset — the only comparable reference.

Usage:
    python scripts/train_eval.py train [steps]
    python scripts/train_eval.py evaluate
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GLOBE_WAV = Path(os.environ.get("GLOBE_WAV", "data/work/globe_wav"))
EDACC_WAV = Path(os.environ.get("ACCENTROUTE_WAV", "data/work/edacc_wav"))
MAN = Path(os.environ.get("GLOBE_MANIFESTS", "data/manifests"))
RUN = Path(os.environ.get("TRAINEVAL_RUN", "runs/accent_v1"))
SEEDS = [17, 42, 1337]


def _pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset(rows: pd.DataFrame, classes: list[str], wav_dir: Path):
    import torch
    from transformers import WhisperFeatureExtractor

    from accentroute.model.pooling import valid_encoder_frames

    rows = rows[rows["accent_label"].isin(classes)].reset_index(drop=True)
    extractor = WhisperFeatureExtractor()
    index = {c: i for i, c in enumerate(classes)}

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            import soundfile as sf

            row = rows.iloc[i]
            wav, _ = sf.read(wav_dir / f"{row['clip_id']}.wav", dtype="float32")
            wav = wav[: 30 * 16000]
            feats = extractor(wav, sampling_rate=16000, return_attention_mask=True,
                              return_tensors="pt")
            return {
                "input_features": feats.input_features[0],
                "n_valid": int(valid_encoder_frames(feats.attention_mask)[0]),
                "label": index[row["accent_label"]],
            }

    return _DS(), rows


def _predict(model, ds, device, batch_size=16):
    import torch

    model.eval()
    preds, golds = [], []
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
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
    return np.array(preds), np.array(golds)


def _load_splits():
    split = pd.read_parquet(MAN / "globe_split.parquet")
    classes = sorted(split[split.split == "train"]["accent_label"].dropna().unique())
    return split, classes


def train() -> None:
    import torch

    from accentroute.model.whisper_lora import build_model
    from accentroute.train import TrainConfig
    from accentroute.train import train as run_train

    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    split, classes = _load_splits()
    print(f"classes: {classes}")

    train_ds, train_rows = _dataset(split[split.split == "train"], classes, GLOBE_WAV)
    val_ds, _ = _dataset(split[split.split == "val"], classes, GLOBE_WAV)
    print(f"train {len(train_ds)} clips / {train_rows.speaker_key.nunique()} speakers, "
          f"val {len(val_ds)} clips")

    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "classes.json").write_text(json.dumps(classes))

    device = _pick_device(torch)
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        model = build_model(n_classes=len(classes)).to(device)
        cfg = TrainConfig(
            arm="accent_v1", budget="epoch_matched", seed=seed, out_dir=RUN / f"seed{seed}",
            epochs_c=3, batch_size=16, lr=1e-3, warmup_ratio=0.1,
            lora_r=16, lora_alpha=32, lora_dropout=0.05,
            base_model="openai/whisper-small", n_classes=len(classes),
            total_steps=steps, shared_config_hash="accent_v1",
        )
        print(run_train(cfg, model=model, train_ds=train_ds, val_ds=val_ds))


def evaluate() -> None:
    import torch

    from accentroute.eval.baselines import majority_baseline
    from accentroute.eval.metrics import confusion, macro_f1
    from accentroute.model.whisper_lora import build_model

    classes = json.loads((RUN / "classes.json").read_text())
    split, _ = _load_splits()
    device = _pick_device(torch)

    in_ds, in_rows = _dataset(split[split.split == "test"], classes, GLOBE_WAV)
    train_rows = split[split.split == "train"]

    # EdAcc: different corpus, spontaneous conversation instead of read speech
    ood_rows = pd.DataFrame()
    edacc_path = MAN / "smoke_qc.parquet"
    if edacc_path.exists():
        edacc = pd.read_parquet(edacc_path)
        ood_rows = edacc[(edacc.status == "accepted") & edacc.accent_label.isin(classes)]

    results = {"classes": classes, "seeds": SEEDS, "in_domain": [], "ood": []}
    for seed in SEEDS:
        ckpt = RUN / f"seed{seed}" / "ckpt_best.pt"
        if not ckpt.exists():
            print(f"skipping seed {seed}: no checkpoint")
            continue
        model = build_model(n_classes=len(classes)).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)

        preds, golds = _predict(model, in_ds, device)
        y_true = np.array([classes[i] for i in golds])
        y_pred = np.array([classes[i] for i in preds])
        results["in_domain"].append({
            "seed": seed,
            "macro_f1": macro_f1(y_true, y_pred, labels=classes),
            "per_class_f1": {c: macro_f1(y_true, y_pred, labels=[c]) for c in classes},
            "confusion": confusion(y_true, y_pred, labels=classes).tolist(),
        })

        if len(ood_rows):
            ood_ds, ood_used = _dataset(ood_rows, classes, EDACC_WAV)
            o_preds, o_golds = _predict(model, ood_ds, device)
            oy_true = np.array([classes[i] for i in o_golds])
            oy_pred = np.array([classes[i] for i in o_preds])
            supported = sorted(set(oy_true))
            results["ood"].append({
                "seed": seed,
                "supported_classes": supported,
                "supported_class_macro_f1": macro_f1(oy_true, oy_pred, labels=supported),
                "in_domain_control_same_subset": macro_f1(
                    y_true, y_pred, labels=supported),
                "n_clips": len(oy_true),
                "n_speakers": int(ood_used.speaker_id_raw.nunique()),
                "confusion": confusion(oy_true, oy_pred, labels=supported).tolist(),
            })

    maj = majority_baseline(train_rows["accent_label"].dropna().to_numpy(), len(in_rows))
    in_true = in_rows["accent_label"].to_numpy()
    results["majority_baseline_macro_f1"] = macro_f1(in_true, maj, labels=classes)
    results["n_test_clips"] = len(in_rows)
    results["n_test_speakers"] = int(in_rows.speaker_key.nunique())

    (RUN / "results.json").write_text(json.dumps(results, indent=2))
    _print(results)


def _print(r: dict) -> None:
    classes = r["classes"]
    ind = [x["macro_f1"] for x in r["in_domain"]]
    print("\n" + "=" * 66)
    print(f"classes: {', '.join(classes)}")
    print(f"test set: {r['n_test_clips']} clips from {r['n_test_speakers']} held-out speakers")
    print("=" * 66)
    if ind:
        print(f"in-domain macro-F1 : {np.mean(ind):.3f} +/- {np.std(ind):.3f}  "
              f"(seeds: {', '.join(f'{v:.3f}' for v in ind)})")
    print(f"majority baseline  : {r['majority_baseline_macro_f1']:.3f}")

    if r["in_domain"]:
        print("\nper-class F1 (mean over seeds):")
        for c in classes:
            vals = [x["per_class_f1"][c] for x in r["in_domain"]]
            print(f"  {c:<14} {np.mean(vals):.3f}")
        print("\nconfusion, seed 1 (rows=true, cols=pred):")
        print(pd.DataFrame(r["in_domain"][0]["confusion"],
                           index=classes, columns=classes).to_string())

    if r["ood"]:
        o = [x["supported_class_macro_f1"] for x in r["ood"]]
        ctrl = [x["in_domain_control_same_subset"] for x in r["ood"]]
        first = r["ood"][0]
        print(f"\nout-of-domain (EdAcc, spontaneous conversation, {first['n_clips']} clips "
              f"from {first['n_speakers']} speakers)")
        print(f"  supported classes  : {', '.join(first['supported_classes'])}")
        print(f"  supported-class F1 : {np.mean(o):.3f} +/- {np.std(o):.3f}")
        print(f"  in-domain control  : {np.mean(ctrl):.3f}   (same class subset)")
        print(f"  domain gap         : {np.mean(ctrl) - np.mean(o):+.3f}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "evaluate": evaluate}[action]()
