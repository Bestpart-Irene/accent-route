"""阶段化 manifest schema：raw → qc → split，逐阶段收紧。

管线每段都是「读 Parquet → 变换 → validate_manifest(df, stage) → 写 Parquet」。
三条项目级不变量在 split 阶段被机器强制，任何下游代码都不需要再自证清白：
  1. rejected 行必有 reject_reason
  2. weak 标签绝不进 val/test/ood_test
  3. youtube 的 accepted 行必须有 E1/E2 证据等级
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ACCENTS = [
    "en-US", "en-GB", "en-AU", "en-IN",
    "L1-Mandarin", "L1-Spanish", "L1-Korean", "L1-Arabic",
]
SOURCES = ["common_voice", "l2_arctic", "edacc", "youtube"]  # vctk/saa 进 backlog 时再扩
SPLITS = ["train", "val", "test", "ood_test", "unassigned"]


class RawManifestSchema(pa.DataFrameModel):
    """ingest 阶段产出：只描述音频与来源，不含标签决策。"""

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
    """taxonomy + filter 之后：标签映射与质量指标。"""

    accent_label: Series[str] = pa.Field(isin=ACCENTS, nullable=True)
    taxonomy_version: Series[str]
    # 单通道估算的代理量，非真实 SNR
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
    """dedup + split + weak-label 之后：训练/评测就绪。"""

    speaker_key: Series[str]  # 缺省 f"{source}:{speaker_id_raw}"，去重合并后更新
    split: Series[str] = pa.Field(isin=SPLITS)
    label_source: Series[str] = pa.Field(isin=["gold", "self_report", "weak"])
    # 多数票比例 × 证据权重的工程排序分，非校准置信度
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
    """按阶段校验 manifest；lazy=True 让所有违规一次性报出。"""
    return STAGE_SCHEMAS[stage].validate(df, lazy=True)
