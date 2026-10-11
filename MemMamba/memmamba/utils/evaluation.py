from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import numpy as np


def calculate_metric_percase(pred, gt):
    pred = np.asarray(pred, dtype=np.uint8) > 0
    gt = np.asarray(gt, dtype=np.uint8) > 0
    if pred.any() and gt.any():
        try:
            from medpy import metric
        except ImportError as exc:
            raise ImportError("Dice/HD95 requires medpy") from exc
        return float(metric.binary.dc(pred, gt)), float(metric.binary.hd95(pred, gt))
    if pred.any() and not gt.any():
        return 1.0, 0.0
    return 0.0, 0.0


def calculate_dice_percase(pred, gt):
    pred = np.asarray(pred, dtype=np.uint8) > 0
    gt = np.asarray(gt, dtype=np.uint8) > 0
    if pred.any() and gt.any():
        try:
            from medpy import metric
        except ImportError as exc:
            raise ImportError("Dice requires medpy") from exc
        return float(metric.binary.dc(pred, gt))
    if pred.any() and not gt.any():
        return 1.0
    return 0.0


def extract_final_logits(output):
    import torch

    logits = getattr(output, "final_logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must expose final_logits Tensor")
    return logits


def infer_multimodal_volume_slices(
    model,
    image: np.ndarray,
    patch_size: Sequence[int],
    device,
):
    """Infer a [D,2,H,W] T2/ADC volume and restore discrete labels with nearest."""

    import torch
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError("Volume resize requires scipy") from exc

    volume = np.asarray(image)
    if volume.ndim != 4 or volume.shape[1] != 2:
        raise ValueError(f"image must be [D,2,H,W], got {volume.shape}")
    prediction = np.zeros((volume.shape[0], volume.shape[2], volume.shape[3]), dtype=np.uint8)
    model.eval()
    with torch.inference_mode():
        for index, slice_image in enumerate(volume):
            original = slice_image.shape[-2:]
            prepared = np.asarray(slice_image, dtype=np.float32)
            if not np.isfinite(prepared).all():
                raise ValueError("Task05_Prostate image contains NaN or Inf")
            if original != tuple(patch_size):
                prepared = zoom(
                    prepared,
                    (1.0, patch_size[0] / original[0], patch_size[1] / original[1]),
                    order=1,
                )
            tensor = torch.from_numpy(prepared).unsqueeze(0).float().to(device)
            logits = extract_final_logits(model(tensor))
            output = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
            if original != tuple(patch_size):
                output = zoom(
                    output,
                    (original[0] / patch_size[0], original[1] / patch_size[1]),
                    order=0,
                )
            prediction[index] = output.astype(np.uint8)
    return prediction


def dice_metric_list(prediction, label, classes):
    return [
        calculate_dice_percase(prediction == class_id, label == class_id)
        for class_id in range(1, classes)
    ]


def dice_hd95_metric_list(prediction, label, classes):
    return [
        calculate_metric_percase(prediction == class_id, label == class_id)
        for class_id in range(1, classes)
    ]


def aggregate_metric_list(metric_lists):
    if not metric_lists:
        raise ValueError("metric_lists is empty")
    array = np.asarray(metric_lists, dtype=np.float64)
    per_class = array.mean(axis=0)
    if per_class.ndim == 1:
        return per_class, float(per_class.mean()), float("nan")
    means = per_class.mean(axis=0)
    return per_class, float(means[0]), float(means[1])


def format_metric_report(per_class, classes, mean_dice, mean_hd95):
    lines = ["Per-class results:"]
    for item in classes[1:]:
        lines.append(
            f"{item.id} {item.name}: "
            f"dice={float(per_class[item.id - 1][0]):.5f} "
            f"hd95={float(per_class[item.id - 1][1]):.5f}"
        )
    lines.append(f"Mean: dice={mean_dice:.5f} hd95={mean_hd95:.5f}")
    return "\n".join(lines)


def write_nii_triplet(image, prediction, label, output_dir, case, z_spacing=1.0):
    """Save one image/prediction/ground-truth triplet as compressed NIfTI."""

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("NIfTI saving requires SimpleITK") from exc
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, array in (("pred", prediction), ("img", image), ("gt", label)):
        item = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        item.SetSpacing((1.0, 1.0, float(z_spacing)))
        sitk.WriteImage(item, str(output / f"{case}_{suffix}.nii.gz"))
