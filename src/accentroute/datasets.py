"""Training dataset: manifest → {input_features, n_valid, label}.

The valid frame count comes from WhisperFeatureExtractor's attention mask (decision #8).
Clips longer than 30s are cropped to their center window, and augmented rows (clip_id
carrying #aug-sp) have their audio generated on the fly from the speed factor.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from accentroute.model.pooling import valid_encoder_frames
from accentroute.schema import ACCENTS

TARGET_SR = 16000
MAX_SECONDS = 30


class ManifestAudioDataset:
    def __init__(
        self,
        manifest: Path,
        split: str,
        wav_dir: Path = Path("data/work/wav16k"),
        extractor=None,
    ):
        df = pd.read_parquet(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.wav_dir = Path(wav_dir)
        if extractor is None:
            from transformers import WhisperFeatureExtractor

            extractor = WhisperFeatureExtractor()
        self.extractor = extractor
        self.label_index = {lab: i for i, lab in enumerate(ACCENTS)}

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, row: pd.Series) -> np.ndarray:
        from accentroute.augment import speed_perturb

        base_id, _, aug = str(row["clip_id"]).partition("#")
        wav, sr = sf.read(self.wav_dir / f"{base_id}.wav", dtype="float32")
        if aug.startswith("aug-sp"):
            wav = speed_perturb(wav, sr, float(aug.removeprefix("aug-sp")))
        max_len = MAX_SECONDS * TARGET_SR
        if len(wav) > max_len:  # >30s → crop to the center window
            start = (len(wav) - max_len) // 2
            wav = wav[start : start + max_len]
        return wav

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        wav = self._load_audio(row)
        feats = self.extractor(
            wav, sampling_rate=TARGET_SR, return_attention_mask=True, return_tensors="pt"
        )
        return {
            "input_features": feats.input_features[0],
            "n_valid": int(valid_encoder_frames(feats.attention_mask)[0]),
            "label": self.label_index[row["accent_label"]],
        }
