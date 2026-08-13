"""Qwen2-Audio 弱标注推理(GPU 阶段,AICR rtx)。

防循环论证(决策 #2):模型 revision sha 与 prompt 文件 sha256 都在
configs/weaklabel.yaml 里 pin 死,并随每次运行写进输出元数据 —— 弱标签来源
与零样本基线用的是同一份冻结配置,报告里必须披露这一双重角色。
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from accentroute.eval.baselines import parse_qwen_label

GenerateFn = Callable[[str, str], str]  # (wav_path, prompt) -> 模型输出文本


@dataclass(frozen=True)
class WeakLabelConfig:
    model_id: str
    revision: str
    prompt_file: Path
    prompt_sha256: str
    k_votes: int
    temperature: float
    kill_precision: float

    @classmethod
    def from_yaml(cls, path: Path) -> "WeakLabelConfig":
        data = yaml.safe_load(Path(path).read_text())
        prompt_file = Path(path).parent.parent / data["prompt_file"]
        return cls(
            model_id=data["model_id"],
            revision=data["revision"],
            prompt_file=prompt_file,
            prompt_sha256=data["prompt_sha256"],
            k_votes=data["k_votes"],
            temperature=data["temperature"],
            kill_precision=data["kill_precision"],
        )

    def load_prompt(self) -> str:
        text = self.prompt_file.read_text()
        got = hashlib.sha256(text.encode()).hexdigest()
        if self.prompt_sha256 not in ("", got):
            raise ValueError(
                f"prompt file changed: {self.prompt_file} sha256={got} "
                f"but config pins {self.prompt_sha256}; bump the prompt version instead"
            )
        return text


def build_generate_fn(cfg: WeakLabelConfig, device: str = "cuda") -> GenerateFn:
    """真实 Qwen2-Audio 推理入口(需 GPU,~17GB BF16)。单测不走这里。"""
    import librosa
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.revision)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.model_id, revision=cfg.revision, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    sr = processor.feature_extractor.sampling_rate

    def generate(wav_path: str, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": wav_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio, _ = librosa.load(wav_path, sr=sr)
        inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature or None,
            )
        gen = out[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(gen, skip_special_tokens=True)[0]

    return generate


def qwen_label_batch(
    manifest: Path,
    cfg: WeakLabelConfig,
    out: Path,
    *,
    generate_fn: GenerateFn,
    wav_path_fn: Callable[[str], str],
) -> Path:
    """逐 clip k 次自洽投票 → 写 Parquet(qwen_votes 列)+ 冻结元数据 sidecar。"""
    df = pd.read_parquet(manifest)
    prompt = cfg.load_prompt()
    votes = [
        [parse_qwen_label(generate_fn(wav_path_fn(clip_id), prompt)) for _ in range(cfg.k_votes)]
        for clip_id in df["clip_id"]
    ]
    df["qwen_votes"] = votes
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    out.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model_id": cfg.model_id,
                "revision": cfg.revision,
                "prompt_file": str(cfg.prompt_file),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "k_votes": cfg.k_votes,
                "temperature": cfg.temperature,
            },
            indent=2,
        )
    )
    return out
