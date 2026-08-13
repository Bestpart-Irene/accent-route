"""Staged manifest schemas: raw → qc → split, tightening one stage at a time.

Every pipeline stage is the same shape: read Parquet → transform →
validate_manifest(df, stage) → write Parquet. Three project-level invariants are
machine-enforced at the split stage, so no downstream code has to re-prove them:
  1. rejected rows always carry a reject_reason
  2. weak labels never reach val/test/ood_test
  3. accepted youtube rows must carry an E1/E2 evidence level
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ACCENTS = [
    "en-US", "en-GB", "en-AU", "en-IN",
    "L1-Mandarin", "L1-Spanish", "L1-Korean", "L1-Arabic",
]
SOURCES = ["common_voice", "l2_arctic", "edacc", "youtube"]  # extend when vctk/saa land
SPLITS = ["train", "val", "test", "ood_test", "unassigned"]


class RawManifestSchema(pa.DataFrameModel):
    """Produced by the ingest stage: audio and provenance only, no labeling decisions."""

    clip_id: Series[str] = pa.Field(unique=True)
    source: Series[str] = pa.Field(isin=SOURCES)
    source_uri: Series[str]
    orig_file: Series[str]
    offset_start_s: Series[float] = pa.Field(ge=0)
    offset_end_s: Series[float] = pa.Field(gt=0)
    sample_rate_orig: Series[int] = pa.Field(gt=0)
    duration_s: Series[float] = pa.Field(gt=0)
    license: Series[str]
    speaker_id_raw: Series[str]
    accent_raw: Series[str] = pa.Field(nullable=True)


class QCManifestSchema(RawManifestSchema):
    """After taxonomy + filter: mapped labels plus quality metrics."""

    accent_label: Series[str] = pa.Field(isin=ACCENTS, nullable=True)
    taxonomy_version: Series[str]
    # Single-channel estimate: a proxy, not true SNR
    snr_proxy_db: Series[float] = pa.Field(nullable=True, coerce=True)
    vad_speech_ratio: Series[float] = pa.Field(ge=0, le=1, nullable=True, coerce=True)
    lang_prob: Series[float] = pa.Field(ge=0, le=1, nullable=True, coerce=True)
    transcript: Series[str] = pa.Field(nullable=True)
    status: Series[str] = pa.Field(isin=["pending", "accepted", "rejected", "review"])
    reject_reason: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_check
    def rejected_has_reason(cls, df: pd.DataFrame) -> Series[bool]:
        return ~((df["status"] == "rejected") & df["reject_reason"].isna())


class SplitManifestSchema(QCManifestSchema):
    """After dedup + split + weak labeling: ready for training and evaluation."""

    speaker_key: Series[str]  # defaults to f"{source}:{speaker_id_raw}"; dedup may remap it
    split: Series[str] = pa.Field(isin=SPLITS)
    label_source: Series[str] = pa.Field(isin=["gold", "self_report", "weak"])
    # Majority-vote share × evidence weight: an engineering ranking score,
    # not a calibrated confidence
    consensus_score: Series[float] = pa.Field(ge=0, le=1, nullable=True, coerce=True)
    evidence_level: Series[str] = pa.Field(isin=["E1", "E2", "E3"], nullable=True)

    @pa.dataframe_check
    def weak_never_in_eval(cls, df: pd.DataFrame) -> Series[bool]:
        return ~(
            (df["label_source"] == "weak") & df["split"].isin(["val", "test", "ood_test"])
        )

    @pa.dataframe_check
    def youtube_requires_evidence(cls, df: pd.DataFrame) -> Series[bool]:
        return ~(
            (df["source"] == "youtube")
            & (df["status"] == "accepted")
            & ~df["evidence_level"].isin(["E1", "E2"])
        )


STAGE_SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "raw": RawManifestSchema,
    "qc": QCManifestSchema,
    "split": SplitManifestSchema,
}


def validate_manifest(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Validate a manifest for one stage; lazy=True surfaces every violation at once."""
    return STAGE_SCHEMAS[stage].validate(df, lazy=True)
