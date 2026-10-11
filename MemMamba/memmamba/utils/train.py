from __future__ import annotations

import json
import math
from pathlib import Path
import random

import numpy as np

from memmamba.model_spec import MODEL_SPEC


CHECKPOINT_VERSION = 5


def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_resolved_config(path, config) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def scheduled_aux_weight(
    epoch, total_epochs, schedule, start_weight, end_weight,
    decay_start_ratio, decay_end_ratio,
) -> float:
    if schedule not in ("constant", "cosine_decay"):
        raise ValueError("lambda_aux_schedule must be constant or cosine_decay")
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must lie in [1,total_epochs]")
    if schedule == "constant":
        return float(start_weight)
    if not 0 <= decay_start_ratio < decay_end_ratio <= 1:
        raise ValueError("invalid lambda_aux decay ratios")
    if not 0 <= end_weight <= start_weight:
        raise ValueError("lambda_aux_end must lie in [0,lambda_aux]")
    progress = epoch / total_epochs
    if progress <= decay_start_ratio:
        return float(start_weight)
    if progress >= decay_end_ratio:
        return float(end_weight)
    normalized = (progress - decay_start_ratio) / (decay_end_ratio - decay_start_ratio)
    return float(
        end_weight
        + 0.5 * (start_weight - end_weight) * (1.0 + math.cos(math.pi * normalized))
    )


def knowledge_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    if epoch <= 0 or warmup_epochs < 0:
        raise ValueError("invalid knowledge warmup epoch")
    return 1.0 if warmup_epochs == 0 else min(1.0, epoch / warmup_epochs)


def warmup_cosine_lr(
    step, total_steps, warmup_steps, peak_lr, warmup_start_lr, min_lr
) -> float:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("invalid warmup/total steps")
    if not 0 < min_lr <= warmup_start_lr <= peak_lr:
        raise ValueError("learning rates must satisfy min <= warmup <= peak")
    step = max(0, min(int(step), total_steps - 1))
    if warmup_steps and step < warmup_steps:
        progress = 1.0 if warmup_steps == 1 else step / (warmup_steps - 1)
        return warmup_start_lr + (peak_lr - warmup_start_lr) * progress
    decay_steps = total_steps - warmup_steps
    progress = 1.0 if decay_steps == 1 else (step - warmup_steps) / (decay_steps - 1)
    return min_lr + (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress)) / 2.0


def build_warmup_cosine_scheduler(
    optimizer, total_steps, warmup_steps, peak_lr, warmup_start_lr, min_lr
):
    import torch

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_lr(
            step, total_steps, warmup_steps, peak_lr, warmup_start_lr, min_lr
        ) / peak_lr,
    )


def save_checkpoint(
    path, model, optimizer, epoch, resolved_config,
    class_snapshot, artifact_metadata, scheduler,
) -> None:
    import torch

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            # Reserved version-5 field; training does not compute validation metrics.
            "best_metric": None,
            "resolved_config": dict(resolved_config),
            "class_snapshot": list(class_snapshot),
            "artifact_metadata": dict(artifact_metadata),
        },
        output,
    )


def _torch_load(path, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_checkpoint(path, map_location="cpu"):
    """Read and validate a checkpoint from this MemMamba release."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {artifact}")
    payload = _torch_load(artifact, map_location)
    required = {
        "format_version", "model", "optimizer", "scheduler", "epoch", "best_metric",
        "resolved_config", "class_snapshot", "artifact_metadata",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("malformed MemMamba checkpoint")
    if payload["format_version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version: MemMamba requires format version 5")
    stored_spec = payload["resolved_config"]
    if not isinstance(stored_spec, dict) or not isinstance(payload["artifact_metadata"], dict):
        raise ValueError("malformed MemMamba checkpoint metadata")
    for key, value in MODEL_SPEC.to_dict().items():
        if type(stored_spec.get(key)) is not type(value) or stored_spec.get(key) != value:
            raise ValueError(f"checkpoint model specification mismatch: {key}")
    return payload


def load_checkpoint(
    path, model, optimizer=None, map_location="cpu", expected_config=None,
    expected_class_snapshot=None, expected_artifact_metadata=None, scheduler=None,
):
    payload = read_checkpoint(path, map_location)
    if expected_config is not None:
        stored = payload["resolved_config"]
        strict_keys = tuple(MODEL_SPEC.to_dict()) + (
            "dataset_name", "modality", "image_size", "memory_visual_dim",
            "memory_semantic_dim",
        )
        if optimizer is not None or scheduler is not None:
            strict_keys += ("epochs", "batch_size", "learning_rate")
        for key in strict_keys:
            if stored.get(key) != expected_config.get(key):
                raise ValueError(f"checkpoint configuration mismatch: {key}")
    if expected_class_snapshot is not None and payload["class_snapshot"] != list(
        expected_class_snapshot
    ):
        raise ValueError("checkpoint class snapshot mismatch")
    if expected_artifact_metadata is not None and payload["artifact_metadata"] != dict(
        expected_artifact_metadata
    ):
        raise ValueError("checkpoint artifact metadata mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def should_save_epoch_checkpoint(epoch, interval=50) -> bool:
    return interval > 0 and epoch > 0 and epoch % interval == 0
