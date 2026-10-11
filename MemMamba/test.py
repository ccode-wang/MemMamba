from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from memmamba.config import load_class_config, load_defaults, resolve_config
from memmamba.utils.logging import configure_runtime_logging


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/task05_prostate.json")
    known, _ = pre.parse_known_args(argv)
    defaults = load_defaults(known.config)
    parser = argparse.ArgumentParser(
        description="Evaluate MemMamba on six Task05_Prostate T2/ADC H5 volumes.",
        parents=[pre],
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_h5_dir", default=None)
    parser.add_argument("--memory_bank_path", default=None)
    parser.add_argument("--alignment_teacher_path", default=None)
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--prediction_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--z_spacing", type=float, default=1.0)
    args = parser.parse_args(argv)
    overrides = {
        key: value for key, value in vars(args).items()
        if key not in ("config", "checkpoint", "z_spacing")
        and value is not None
    }
    config = resolve_config(defaults, overrides).to_dict()
    return args, config


def next_prediction_run_dir(root: str | Path) -> Path:
    """Create a monotonically numbered prediction directory without overwriting."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = output / f"run_{index:03d}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1


def append_test_report(result_path, report, checkpoint, config, prediction_run_dir, timestamp=None) -> None:
    """Append one identifiable test run without replacing older results."""

    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_time = timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
    header = (
        f"=== Test run {run_time} ===\n"
        f"checkpoint: {Path(checkpoint).resolve()}\n"
        f"prediction_dir: {Path(prediction_run_dir).resolve()}\n"
        f"dataset_name: {config['dataset_name']}\n"
        f"seed: {config.get('seed', 'unknown')}\n"
        f"model_name: {config['model_name']}\n"
        f"kcm_scan_backend: {config.get('kcm_scan_backend', 'not_applicable')}"
    )
    block = f"{header}\n{report.rstrip()}\n"
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        if output.stat().st_size:
            handle.write("\n")
        handle.write(block)


def main() -> None:
    configure_runtime_logging()
    args, config = parse_args()
    import numpy as np
    import torch

    from memmamba.data.task05_prostate import ProstateH5Dataset, list_h5_volumes
    from memmamba.utils.artifacts import build_artifact_metadata
    from memmamba.utils.evaluation import (
        aggregate_metric_list, dice_hd95_metric_list, format_metric_report,
        infer_multimodal_volume_slices, write_nii_triplet,
    )
    from memmamba.utils.train import load_checkpoint, read_checkpoint
    from memmamba.factory import build_model_from_config

    classes = load_class_config(config["class_config"])
    device = torch.device(
        config["device"] if config["device"].startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    raw = read_checkpoint(args.checkpoint, "cpu")
    if raw.get("class_snapshot") != [asdict(item) for item in classes]:
        raise ValueError("checkpoint class snapshot differs from Task05_Prostate class config")
    model = build_model_from_config(config)
    model.memory_bank = model.memory_bank.to(device)
    model = model.to(device)
    config["memory_visual_dim"] = model.memory_bank.visual_dim
    config["memory_semantic_dim"] = model.memory_bank.semantic_dim
    include_alignment = raw["artifact_metadata"].get("alignment_teachers") is not None
    metadata = build_artifact_metadata(model, config, include_alignment)
    config["seed"] = raw["resolved_config"]["seed"]
    del raw
    load_checkpoint(
        args.checkpoint, model, None, device, config,
        [asdict(item) for item in classes], metadata, None,
    )
    model.eval()

    h5_paths = list_h5_volumes(config["test_h5_dir"])
    if len(h5_paths) != config["expected_eval_cases"]:
        raise ValueError(f"expected {config['expected_eval_cases']} evaluation cases, found {len(h5_paths)}")
    prediction_dir = next_prediction_run_dir(config["prediction_dir"])
    all_metrics, lines = [], []

    dataset = ProstateH5Dataset(h5_paths, classes)
    for sample in dataset:
        image_volume = sample["image"].numpy()
        label_volume = sample["label"].numpy()
        prediction = infer_multimodal_volume_slices(
            model, image_volume, (config["image_size"], config["image_size"]), device
        )
        # Preserve the existing three-file prediction contract; T2 is the reference image.
        write_nii_triplet(
            image_volume[:, 0], prediction, label_volume, prediction_dir,
            sample["case_id"], args.z_spacing,
        )
        current = dice_hd95_metric_list(prediction, label_volume, len(classes))
        all_metrics.append(current)
        mean = np.asarray(current, dtype=np.float64).mean(axis=0)
        lines.append(
            f"idx {len(all_metrics) - 1} case {sample['case_id']} "
            f"mean_dice {mean[0]:.5f} mean_hd95 {mean[1]:.5f}"
        )
        for class_info, (dice_value, hd95_value) in zip(classes[1:], current):
            lines.append(
                f"case {sample['case_id']} class {class_info.id} {class_info.name}: "
                f"dice={dice_value:.5f} hd95={hd95_value:.5f}"
            )

    per_class, mean_dice, mean_hd95 = aggregate_metric_list(all_metrics)
    lines.append(format_metric_report(per_class, classes, mean_dice, mean_hd95))
    lines.append(f"prediction_dir={prediction_dir.resolve()}")
    report = "\n".join(lines)
    print(report)
    append_test_report(config["result_path"], report, args.checkpoint, config, prediction_dir)


if __name__ == "__main__":
    main()
