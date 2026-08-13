"""训练循环 + 三臂预算协议(决策 #1)。

公平性的机制保证:
  - 共享字段只能来自 train_common.yaml,臂配置试图覆盖 → 加载器 ValueError;
  - shared_config_hash 写进每个 run 的 metrics.json,T15 汇总时断言一致;
  - B 臂步数 == C 臂步数由 resolve_total_steps("step_matched_to_c") 给出。
固定步数预算(无 early stopping),checkpoint 按 val macro-F1 选择。
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from accentroute.eval.metrics import macro_f1

# 消融公平性协议字段:锁定在 train_common.yaml,臂配置不得覆盖
SHARED_FIELDS = frozenset(
    {
        "epochs_c", "batch_size", "lr", "lr_schedule", "warmup_ratio",
        "sampler", "augment", "ckpt_select", "seeds", "base_model",
        "lora_r", "lora_alpha", "lora_dropout", "n_classes",
    }
)


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    budget: str  # epoch_matched | step_matched_to_c
    seed: int
    out_dir: Path
    epochs_c: int
    batch_size: int
    lr: float
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    base_model: str
    n_classes: int
    shared_config_hash: str
    dataset_variant: str = ""
    total_steps: int | None = None  # B 臂由 C 臂步数注入;None → 由数据量解出


@dataclass(frozen=True)
class TrainResult:
    ckpt_path: str
    val_macro_f1: float
    seed: int
    total_steps: int


def load_train_config(
    common_path: Path, arm_path: Path, seed: int, out_dir: Path | None = None
) -> TrainConfig:
    common = yaml.safe_load(Path(common_path).read_text())
    arm = yaml.safe_load(Path(arm_path).read_text())

    clash = SHARED_FIELDS & set(arm)
    if clash:
        raise ValueError(
            f"arm config {arm_path} overrides shared protocol fields {sorted(clash)}; "
            "shared fields are locked in train_common.yaml"
        )
    if seed not in common["seeds"]:
        raise ValueError(f"seed {seed} not in protocol seeds {common['seeds']}")

    shared_hash = hashlib.sha256(
        json.dumps({k: common[k] for k in sorted(SHARED_FIELDS)}, sort_keys=True).encode()
    ).hexdigest()[:16]

    return TrainConfig(
        arm=arm["arm"],
        budget=arm["budget"],
        seed=seed,
        out_dir=Path(out_dir) if out_dir else Path("runs") / f"{arm['arm']}-seed{seed}",
        epochs_c=common["epochs_c"],
        batch_size=common["batch_size"],
        lr=common["lr"],
        warmup_ratio=common["warmup_ratio"],
        lora_r=common["lora_r"],
        lora_alpha=common["lora_alpha"],
        lora_dropout=common["lora_dropout"],
        base_model=common["base_model"],
        n_classes=common["n_classes"],
        shared_config_hash=shared_hash,
        dataset_variant=arm.get("dataset_variant", ""),
        total_steps=None,
    )


def resolve_total_steps(
    budget: str,
    n_train: int,
    batch_size: int,
    epochs_c: int,
    steps_c: int | None = None,
) -> int:
    """epoch_matched → 自身数据量;step_matched_to_c → 必须注入 C 臂步数。"""
    if budget == "epoch_matched":
        return epochs_c * math.ceil(n_train / batch_size)
    if budget == "step_matched_to_c":
        if steps_c is None:
            raise ValueError("step_matched_to_c requires steps_c from the C-arm run")
        return steps_c
    raise ValueError(f"unknown budget rule: {budget}")


def _cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: TrainConfig, *, model, train_ds, val_ds) -> TrainResult:
    """数据集条目:{"input_features": [80,T] float, "n_valid": int, "label": int}。

    model 由调用方构建(生产走 build_model 下载底座;测试用微型随机 encoder),
    训练循环本身与模型尺寸无关。
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(cfg.seed)
    np.random.default_rng(cfg.seed)

    total_steps = cfg.total_steps
    if total_steps is None:
        total_steps = resolve_total_steps(
            cfg.budget, len(train_ds), cfg.batch_size, cfg.epochs_c
        )

    labels = np.array([train_ds[i]["label"] for i in range(len(train_ds))])
    counts = np.bincount(labels, minlength=cfg.n_classes).astype(np.float64)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights),
        num_samples=total_steps * cfg.batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    def collate(items):
        return (
            torch.stack([torch.as_tensor(it["input_features"]) for it in items]),
            torch.as_tensor([it["n_valid"] for it in items]),
            torch.as_tensor([it["label"] for it in items]),
        )

    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)
    warmup = int(cfg.warmup_ratio * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _cosine_warmup(s, total_steps, warmup)
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt_best.pt"
    eval_every = max(1, total_steps // 5)
    history: list[dict] = []
    best_f1 = -1.0
    running_loss: list[float] = []

    def _trainable_state():
        return {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if "lora_" in k or "head" in k
        }

    def _evaluate() -> float:
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            eval_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate)
            for feats, n_valid, ys in eval_loader:
                logits = model(feats, n_valid)
                preds.extend(logits.argmax(-1).tolist())
                golds.extend(ys.tolist())
        model.train()
        class_labels = list(range(cfg.n_classes))
        return macro_f1(np.array(golds), np.array(preds), labels=class_labels)

    model.train()
    for step, (feats, n_valid, ys) in enumerate(loader, start=1):
        opt.zero_grad()
        loss = loss_fn(model(feats, n_valid), ys)
        loss.backward()
        opt.step()
        sched.step()
        running_loss.append(float(loss.detach()))

        if step % eval_every == 0 or step == total_steps:
            val_f1 = _evaluate()
            history.append(
                {
                    "step": step,
                    "train_loss": float(np.mean(running_loss)),
                    "val_macro_f1": val_f1,
                }
            )
            running_loss = []
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(_trainable_state(), ckpt_path)
        if step >= total_steps:
            break

    metrics = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "total_steps": total_steps,
        "best_val_macro_f1": best_f1,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return TrainResult(
        ckpt_path=str(ckpt_path),
        val_macro_f1=best_f1,
        seed=cfg.seed,
        total_steps=total_steps,
    )
