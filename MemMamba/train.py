from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from memmamba.config import load_class_config, load_defaults, resolve_config
from memmamba.utils.logging import configure_runtime_logging


def load_runtime_config(argv=None) -> dict:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/task05_prostate.json")
    known, _ = pre.parse_known_args(raw_argv)
    defaults = load_defaults(known.config)
    parser = argparse.ArgumentParser(
        description="Train MemMamba on Task05_Prostate T2/ADC MRI.",
        parents=[pre],
    )
    for name in (
        "train_npz_dir", "test_h5_dir", "memory_bank_path",
        "alignment_teacher_path", "checkpoint_dir", "log_dir", "device",
        "resume_checkpoint",
    ):
        parser.add_argument(f"--{name}", default=None)
    for name in ("epochs", "batch_size", "num_workers", "save_interval"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args(raw_argv)
    overrides = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    return resolve_config(defaults, overrides).to_dict()


def load_training_alignment_teachers(model, classes, path, lambda_align):
    if lambda_align <= 0:
        return None, 0
    from memmamba.memory.alignment_teachers import load_alignment_teacher_bank

    bank = load_alignment_teacher_bank(path)
    if bank.classes != classes:
        raise ValueError("Alignment Teacher classes differ from training classes")
    if bank.feature_dim != model.knowledge_retrieval.query_projector.out_features:
        raise ValueError("Alignment Teacher and query dimensions differ")
    return bank.teacher_lookup(), bank.feature_dim


def main() -> None:
    config = load_runtime_config()
    log_dir = Path(config["log_dir"])
    epoch_logger = configure_runtime_logging(log_dir)
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from memmamba.data.task05_prostate import (
        ProstateNPZDataset,
        list_h5_volumes,
        list_npz_slices,
        validate_case_split,
    )
    from memmamba.factory import build_model_from_config
    from memmamba.utils.artifacts import build_artifact_metadata
    from memmamba.utils.losses import DiceCELoss, alignment_teacher_batch, compute_training_loss
    from memmamba.utils.train import (
        build_warmup_cosine_scheduler, knowledge_warmup_scale, load_checkpoint,
        save_checkpoint, save_resolved_config, scheduled_aux_weight, set_seed,
        should_save_epoch_checkpoint,
    )

    classes = load_class_config(config["class_config"])
    if len(classes) != config["num_classes"]:
        raise ValueError("class config differs from num_classes")
    train_paths = list_npz_slices(config["train_npz_dir"])
    eval_paths = list_h5_volumes(config["test_h5_dir"])
    validate_case_split(
        train_paths, eval_paths, config["expected_train_cases"], config["expected_eval_cases"]
    )
    set_seed(config["seed"])
    device = torch.device(
        config["device"] if config["device"].startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config)
    model.memory_bank = model.memory_bank.to(device)
    model = model.to(device)
    config["memory_visual_dim"] = model.memory_bank.visual_dim
    config["memory_semantic_dim"] = model.memory_bank.semantic_dim
    class_snapshot = [asdict(item) for item in classes]
    teacher_lookup, teacher_dim = load_training_alignment_teachers(
        model, classes, config["alignment_teacher_path"], config["lambda_align"]
    )
    artifact_metadata = build_artifact_metadata(model, config, teacher_lookup is not None)
    loader = DataLoader(
        ProstateNPZDataset(train_paths, classes),
        batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * len(loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps, config["warmup_epochs"] * len(loader),
        config["learning_rate"], config["warmup_start_lr"], config["min_learning_rate"],
    )
    criterion = DiceCELoss(len(classes), config["dice_weight"], config["ce_weight"])
    writer = SummaryWriter(log_dir=log_dir)
    start_epoch = 1
    if config["resume_checkpoint"]:
        payload = load_checkpoint(
            config["resume_checkpoint"], model, optimizer, device, config,
            class_snapshot, artifact_metadata, scheduler,
        )
        start_epoch = int(payload["epoch"]) + 1
    save_resolved_config(checkpoint_dir / "resolved_config.json", config)

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        knowledge_scale = knowledge_warmup_scale(epoch, config["knowledge_warmup_epochs"])
        lambda_aux = scheduled_aux_weight(
            epoch, config["epochs"], config["lambda_aux_schedule"], config["lambda_aux"],
            config["lambda_aux_end"], config["lambda_aux_decay_start_ratio"],
            config["lambda_aux_decay_end_ratio"],
        )
        sums = {key: 0.0 for key in ("total", "final", "coarse", "presence", "alignment")}
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(image, knowledge_scale=knowledge_scale)
            teacher_batch = None
            if teacher_lookup is not None:
                teacher_batch = alignment_teacher_batch(
                    batch["case_id"], batch["slice_id"], label, teacher_lookup,
                    config["num_foreground_classes"], teacher_dim, device,
                )
            losses = compute_training_loss(
                output, label, criterion, lambda_aux, config["lambda_presence"], config["lambda_align"],
                None if teacher_batch is None else teacher_batch.teacher,
                None if teacher_batch is None else teacher_batch.valid,
                None if teacher_batch is None else teacher_batch.confidence,
            )
            losses.total.backward()
            if config["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for key in sums:
                sums[key] += float(getattr(losses, key).detach())

        count = max(len(loader), 1)
        epoch_logger.info("epoch=%03d", epoch)
        for key, value in sums.items():
            writer.add_scalar(f"loss/{key}", value / count, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch,
            config, class_snapshot, artifact_metadata, scheduler,
        )
        if should_save_epoch_checkpoint(epoch, config["save_interval"]):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch,
                config, class_snapshot, artifact_metadata, scheduler,
            )
    writer.close()


if __name__ == "__main__":
    main()
