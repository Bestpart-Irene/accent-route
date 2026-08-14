#!/usr/bin/env python
"""Score a published accent model on this project's test set, for a calibrated comparison.

A macro-F1 in isolation tells a reader nothing: this taxonomy is custom and the field has
no canonical benchmark. Running a published model on the *same clips, same speaker-disjoint
split, same metric* gives the only comparison that is actually apples-to-apples.

The published model (Vox-Profile) predicts 16 regional labels while this one predicts four
accent classes, so both sides are mapped onto the Vox-Profile vocabulary before scoring.
Predictions from the published model that fall outside the four relevant regions count as
errors rather than being discarded — dropping them would flatter it.

STATUS: not yet run. The published checkpoint loads through the authors' `WhisperWrapper`,
whose custom encoder layer targets an older transformers API — its `forward` rejects
`output_hidden_states` and requires a positional `layer_head_mask` that current
transformers no longer passes. Patching one signature just surfaces the next, so running
this needs an environment with transformers pinned to the version the authors used
(their install instructions specify a separate Python 3.8 conda env), not a patch on top
of ours. The obstacle is version skew in their code, not anything about the model.

Usage:
    python scripts/compare_published.py     # requires the pinned environment above
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GLOBE_WAV = Path(os.environ.get("GLOBE_WAV", "data/work/globe_wav"))
MAN = Path(os.environ.get("GLOBE_MANIFESTS", "data/manifests"))
RUN = Path(os.environ.get("TRAINEVAL_RUN", "runs/accent_v1"))
PUBLISHED = "tiantiaf/whisper-large-v3-narrow-accent"


def main() -> None:
    import torch

    from accentroute.eval.external import to_vox_profile
    from accentroute.eval.metrics import confusion, macro_f1

    classes = json.loads((RUN / "classes.json").read_text())
    split = pd.read_parquet(MAN / "globe_split.parquet")
    test = split[(split.split == "test") & split.accent_label.isin(classes)]
    test = test.reset_index(drop=True)
    print(f"test set: {len(test)} clips, {test.speaker_key.nunique()} speakers, "
          f"classes {classes}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- published model -------------------------------------------------------------
    # Loaded through the authors' own wrapper class, not AutoModel: the repo ships no
    # preprocessor_config.json because the model takes a raw waveform, not features.
    vox_repo = Path(os.environ.get("VOX_PROFILE_REPO", "vox-profile-release"))
    sys.path.insert(0, str(vox_repo))
    from src.model.accent.whisper_accent import WhisperWrapper

    # Their label order, from the model card; the checkpoint carries no id2label.
    VP_LABELS = [
        "East Asia", "English", "Germanic", "Irish", "North America", "Northern Irish",
        "Oceania", "Other", "Romance", "Scottish", "Semitic", "Slavic", "South African",
        "Southeast Asia", "South Asia", "Welsh",
    ]
    # The checkpoint ships in fp16, but the wrapper's own forward runs a
    # WhisperFeatureExtractor internally and hands the conv stack float32 features, so the
    # weights have to be float32 too. Casting the input instead does nothing — the input
    # never reaches the conv, only the extractor's output does.
    model = WhisperWrapper.from_pretrained(PUBLISHED).to(device).float().eval()
    print(f"published model dtype: {next(model.parameters()).dtype}")

    import soundfile as sf

    # The model card states their training filtered audio longer than 15 s, so clips are
    # truncated to that here rather than fed at a length the model never saw.
    max_len = 15 * 16000
    pub_preds = []
    with torch.no_grad():
        for _, row in test.iterrows():
            wav, _ = sf.read(GLOBE_WAV / f"{row['clip_id']}.wav", dtype="float32")
            data = torch.from_numpy(wav[:max_len]).float().unsqueeze(0).to(device)
            logits, _ = model(data, return_feature=True)
            pub_preds.append(VP_LABELS[int(logits.argmax(-1).item())])

    # --- both sides in the published model's vocabulary -------------------------------
    y_true_vp = np.array([to_vox_profile(c) for c in test.accent_label])
    pub_vp = np.array(pub_preds)

    ours = json.loads((RUN / "results.json").read_text())
    labels_vp = sorted(set(y_true_vp))

    out = {
        "n_clips": len(test),
        "n_speakers": int(test.speaker_key.nunique()),
        "vox_profile_labels": labels_vp,
        "published_model": PUBLISHED,
        "published_macro_f1": macro_f1(y_true_vp, pub_vp, labels=labels_vp),
        "published_out_of_vocab_rate": float(np.mean(~np.isin(pub_vp, labels_vp))),
        "ours_macro_f1_native_taxonomy": float(
            np.mean([x["macro_f1"] for x in ours["in_domain"]])
        ),
        "published_confusion": confusion(y_true_vp, pub_vp, labels=labels_vp).tolist(),
    }
    (RUN / "comparison.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 66)
    print("Same clips, same speaker-disjoint split, same metric.")
    print("=" * 66)
    print(f"published {PUBLISHED}: macro-F1 {out['published_macro_f1']:.3f}")
    print(f"  (predicted outside the four relevant regions on "
          f"{out['published_out_of_vocab_rate']:.1%} of clips, counted as errors)")
    print(f"ours (4-class taxonomy)          : macro-F1 "
          f"{out['ours_macro_f1_native_taxonomy']:.3f}")
    print("\npublished-model confusion (rows=true, cols=pred):")
    print(pd.DataFrame(out["published_confusion"], index=labels_vp,
                       columns=labels_vp).to_string())


if __name__ == "__main__":
    main()
