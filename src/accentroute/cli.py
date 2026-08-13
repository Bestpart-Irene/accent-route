"""AccentRoute 管线 CLI。

各阶段都是 manifest → manifest;真实模型依赖(VAD/whisper/LID/ECAPA/Qwen)
在这里才绑定,库代码本身保持可注入、可测试。
"""

from pathlib import Path

import pandas as pd
import typer
import yaml

app = typer.Typer(name="accentroute", no_args_is_help=True)
ingest_app = typer.Typer(no_args_is_help=True, help="各源 → raw manifest")
weaklabel_app = typer.Typer(no_args_is_help=True, help="YouTube 弱标注(GPU)")
train_app = typer.Typer(no_args_is_help=True, help="训练(GPU)")
app.add_typer(ingest_app, name="ingest")
app.add_typer(weaklabel_app, name="weaklabel")
app.add_typer(train_app, name="train")

CONFIGS = Path("configs")


@app.callback()
def main() -> None:
    """AccentRoute 数据管线。"""


@app.command()
def version() -> None:
    """打印版本。"""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("accentroute"))


@ingest_app.command("common-voice")
def ingest_common_voice(
    config: Path = CONFIGS / "sources" / "common_voice.yaml",
    out: Path = Path("data/manifests/raw_common_voice.parquet"),
) -> None:
    from accentroute.ingest.base import run_ingest
    from accentroute.ingest.common_voice import CommonVoiceIngestor

    cfg = yaml.safe_load(config.read_text())
    ing = CommonVoiceIngestor(
        root=Path(cfg["root"]), tsv=cfg["tsv"], clips_dir=cfg["clips_dir"],
        source_uri=cfg["source_uri"],
    )
    path = run_ingest(ing, out)
    typer.echo(f"{path}\naccent stats: {ing.stats}")


@ingest_app.command("l2-arctic")
def ingest_l2_arctic(
    config: Path = CONFIGS / "sources" / "l2_arctic.yaml",
    out: Path = Path("data/manifests/raw_l2_arctic.parquet"),
) -> None:
    from accentroute.ingest.base import run_ingest
    from accentroute.ingest.l2_arctic import L2ArcticIngestor

    cfg = yaml.safe_load(config.read_text())
    ing = L2ArcticIngestor(
        root=Path(cfg["root"]), speaker_l1=cfg["speaker_l1"], source_uri=cfg["source_uri"]
    )
    typer.echo(str(run_ingest(ing, out)))


@ingest_app.command("edacc")
def ingest_edacc(
    config: Path = CONFIGS / "sources" / "edacc.yaml",
    out: Path = Path("data/manifests/raw_edacc.parquet"),
) -> None:
    from accentroute.ingest.base import run_ingest
    from accentroute.ingest.edacc import EdAccIngestor

    cfg = yaml.safe_load(config.read_text())
    ing = EdAccIngestor(
        root=Path(cfg["root"]), segments_csv=cfg["segments_csv"],
        speakers_csv=cfg["speakers_csv"], source_uri=cfg["source_uri"],
        columns=cfg.get("columns") or {},
    )
    typer.echo(str(run_ingest(ing, out)))


@app.command("filter")
def filter_cmd(
    manifest: Path,
    out: Path,
    wav_dir: Path = Path("data/work/wav16k"),
    config: Path = CONFIGS / "filter.yaml",
    taxonomy: Path = CONFIGS / "taxonomy_v1.yaml",
) -> None:
    """质量过滤:raw → qc(绑定 Silero VAD 与 faster-whisper tiny 的转写 + 音频 LID)。"""
    import soundfile as sf
    from faster_whisper import WhisperModel
    from silero_vad import get_speech_timestamps, load_silero_vad

    from accentroute.filter import FilterConfig, apply_filters
    from accentroute.taxonomy import load_taxonomy

    vad_model = load_silero_vad()
    asr = WhisperModel("tiny", device="cpu", compute_type="int8")

    def vad_fn(wav, sr):
        import torch

        return get_speech_timestamps(torch.as_tensor(wav, dtype=torch.float32), vad_model)

    # 语言判定用 whisper 自己的音频 LID(比对转写文本再判语言更可靠):
    # 转写时记录该 clip 的语言,lid_fn 按转写文本取回。
    detected: dict[str, tuple[str, float]] = {}

    def transcribe_fn(wav):
        segments, info = asr.transcribe(wav, language=None)
        text = " ".join(s.text for s in segments).strip()
        detected[text] = (info.language, info.language_probability)
        return text

    def lid_fn(text):
        return detected.get(text, ("unknown", 0.0))

    def audio_loader(clip_id: str):
        wav, sr = sf.read(Path(wav_dir) / f"{clip_id}.wav")
        return wav, sr

    out_df = apply_filters(
        pd.read_parquet(manifest), FilterConfig.from_yaml(config),
        taxonomy=load_taxonomy(taxonomy), audio_loader=audio_loader,
        vad_fn=vad_fn, transcribe_fn=transcribe_fn, lid_fn=lid_fn,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    typer.echo(f"{out}\n{out_df['status'].value_counts().to_dict()}")


@app.command("split")
def split_cmd(
    manifest: Path,
    out: Path,
    config: Path = CONFIGS / "split.yaml",
    speaker_report: Path = Path("data/manifests/speakers.csv"),
) -> None:
    """speaker-disjoint 切分 + 可复核 speaker 清单。"""
    from accentroute.dedup import assign_speaker_keys
    from accentroute.split import assign_splits, write_speaker_report

    cfg = yaml.safe_load(config.read_text())
    df = assign_speaker_keys(pd.read_parquet(manifest))
    out_df = assign_splits(df, ratios=tuple(cfg["ratios"]), seed=cfg["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    summary = write_speaker_report(out_df, speaker_report)
    typer.echo(f"{out}\n{summary.to_string(index=False)}")


@app.command("report")
def report_cmd(
    manifest: Path,
    out_dir: Path = Path("data/reports"),
) -> None:
    """G1 门禁报告:source × accent 混杂矩阵 + confounded 标记 + EdAcc 覆盖。"""
    from accentroute.reports.coverage_confounding import (
        edacc_class_coverage,
        flag_confounded,
        source_accent_matrix,
    )

    df = pd.read_parquet(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = source_accent_matrix(df[df["source"] != "edacc"])
    matrix.to_csv(out_dir / "source_accent_matrix.csv", index=False)
    confounded = flag_confounded(matrix)
    confounded.to_csv(out_dir / "confounded_classes.csv", index=False)
    edacc = df[df["source"] == "edacc"]
    if len(edacc):
        edacc_class_coverage(edacc).to_csv(out_dir / "edacc_coverage.csv", index=False)
    typer.echo(f"{out_dir}\n{confounded.to_string(index=False)}")


@app.command("emit")
def emit_cmd(
    manifest: Path,
    variant: str,
    out_dir: Path,
    augment: bool = True,
) -> None:
    """产出训练集变体(a_gold / b_gold_oversampled / c_gold_weak / loso_l2)。"""
    from accentroute.augment import AugmentConfig, augment_train
    from accentroute.emit import emit_dataset

    df = pd.read_parquet(manifest)
    if augment:
        common = yaml.safe_load((CONFIGS / "train_common.yaml").read_text())
        df = augment_train(df, AugmentConfig(speed=tuple(common["augment"]["speed"])))
    stats = emit_dataset(df, variant, out_dir)
    typer.echo(f"{out_dir}\n{stats}")


@weaklabel_app.command("run")
def weaklabel_run(
    manifest: Path,
    out: Path,
    config: Path = CONFIGS / "weaklabel.yaml",
    wav_dir: Path = Path("data/work/youtube_wav"),
) -> None:
    """Qwen2-Audio 批量投票(需 GPU;revision 未 pin 时直接失败)。"""
    from accentroute.weaklabel.qwen import WeakLabelConfig, build_generate_fn, qwen_label_batch

    cfg = WeakLabelConfig.from_yaml(config)
    if cfg.revision == "PIN_ME":
        raise typer.BadParameter("configs/weaklabel.yaml still has revision: PIN_ME")
    path = qwen_label_batch(
        manifest, cfg, out, generate_fn=build_generate_fn(cfg),
        wav_path_fn=lambda cid: str(Path(wav_dir) / f"{cid}.wav"),
    )
    typer.echo(str(path))


@weaklabel_app.command("consensus")
def weaklabel_consensus(manifest: Path, out: Path) -> None:
    """共识判定 + 接受率。"""
    from accentroute.weaklabel.consensus import apply_consensus

    df = apply_consensus(pd.read_parquet(manifest))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    rate = (df["status"] == "accepted").mean()
    typer.echo(f"{out}\nacceptance rate: {rate:.3f}\n{df['status'].value_counts().to_dict()}")


@weaklabel_app.command("audit-sample")
def weaklabel_audit_sample(
    manifest: Path,
    out: Path = Path("data/audit/blind_sample.csv"),
    accepted_per_class: int = 25,
    reject_pool_n: int = 50,
) -> None:
    """抽三池盲听样本(导出的 CSV 不含标签列)。"""
    from accentroute.weaklabel.audit import draw_audit_sample

    sample = draw_audit_sample(
        pd.read_parquet(manifest), accepted_per_class=accepted_per_class,
        reject_pool_n=reject_pool_n,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out.with_suffix(".internal.parquet"), index=False)
    blind_cols = [c for c in ("clip_id", "pool") if c in sample.columns]
    sample[blind_cols].assign(human_label="").to_csv(out, index=False)
    typer.echo(f"blind CSV: {out}\ninternal key: {out.with_suffix('.internal.parquet')}")


@train_app.command("run")
def train_run(
    common: Path = CONFIGS / "train_common.yaml",
    arm: Path = CONFIGS / "arms" / "a_gold.yaml",
    seed: int = 17,
    data_dir: Path = Path("data/datasets"),
    steps_c: int | None = None,
) -> None:
    """单次训练(需 GPU)。b_gold_oversampled 必须传 --steps-c。"""
    from accentroute.datasets import ManifestAudioDataset
    from accentroute.model.whisper_lora import build_model
    from accentroute.train import load_train_config, resolve_total_steps, train

    cfg = load_train_config(common, arm, seed=seed)
    variant_dir = data_dir / cfg.dataset_variant
    train_ds = ManifestAudioDataset(variant_dir / "manifest.parquet", split="train")
    val_ds = ManifestAudioDataset(variant_dir / "manifest.parquet", split="val")
    total = resolve_total_steps(
        cfg.budget, len(train_ds), cfg.batch_size, cfg.epochs_c, steps_c=steps_c
    )
    cfg = type(cfg)(**{**cfg.__dict__, "total_steps": total})
    model = build_model(cfg.base_model, cfg.n_classes, cfg.lora_r, cfg.lora_alpha,
                        cfg.lora_dropout)
    result = train(cfg, model=model, train_ds=train_ds, val_ds=val_ds)
    typer.echo(f"{result}")


if __name__ == "__main__":
    app()
