from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np

from memmamba.config import ClassInfo


_TRAIN_NAME = re.compile(r"^(?P<case>prostate_\d+)_slice(?P<slice>\d+)\.npz$")
_TEST_NAME = re.compile(r"^(?P<case>prostate_\d+)\.npy\.h5$")


def _validate_integer_label_values(label: np.ndarray) -> None:
    array = np.asarray(label)
    if array.size == 0 or not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError("label must contain finite integer values")


def validate_label_range(label: np.ndarray, classes: Sequence[ClassInfo] | int) -> None:
    array = np.asarray(label)
    _validate_integer_label_values(array)
    maximum_id = int(classes) if isinstance(classes, int) else len(classes) - 1
    minimum, maximum = int(array.min()), int(array.max())
    if minimum < 0 or maximum > maximum_id:
        raise ValueError(f"label values must be in [0, {maximum_id}], got min={minimum} max={maximum}")


def parse_train_name(path_name: str | Path) -> tuple[str, str, str, int]:
    name = Path(path_name).name
    match = _TRAIN_NAME.fullmatch(name)
    if match is None:
        raise ValueError("training filename must match 'prostate_00_slice000.npz', got " + repr(name))
    case_id = match.group("case")
    slice_id = int(match.group("slice"))
    return case_id, case_id, f"{case_id}_slice{slice_id:03d}", slice_id


def parse_test_name(path_name: str | Path) -> tuple[str, str]:
    name = Path(path_name).name
    match = _TEST_NAME.fullmatch(name)
    if match is None:
        raise ValueError("evaluation filename must match 'prostate_04.npy.h5', got " + repr(name))
    case_id = match.group("case")
    return case_id, case_id


@dataclass
class ProstateTrainSlice:
    path: Path
    image: np.ndarray
    label: np.ndarray
    patient_id: str
    case_id: str
    sample_id: str
    slice_id: int
    filename: str

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"training slice does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            if "image" not in data or "label" not in data:
                raise KeyError(f"{self.path} must contain image and label arrays")
            image = np.asarray(data["image"])
            label = np.asarray(data["label"])
        if image.shape != (2, 224, 224) or label.shape != (224, 224):
            raise ValueError(
                "training image/label must be [2,224,224]/[224,224], got "
                f"image={image.shape} label={label.shape}"
            )
        if not np.all(np.isfinite(image)):
            raise ValueError("training image must contain only finite values")
        _validate_integer_label_values(label)
        self.image = image.astype(np.float32, copy=False)
        self.label = label.astype(np.int64, copy=False)
        self.patient_id, self.case_id, self.sample_id, self.slice_id = parse_train_name(self.path.name)
        self.filename = self.path.name


@dataclass
class ProstateTestVolume:
    path: Path
    image: np.ndarray
    label: np.ndarray
    patient_id: str
    case_id: str
    filename: str

    def __init__(self, path: str | Path, classes: Sequence[ClassInfo] | int):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"evaluation volume does not exist: {self.path}")
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("reading H5 volumes requires h5py") from exc
        with h5py.File(self.path, "r") as handle:
            if "image" not in handle or "label" not in handle:
                raise KeyError(f"{self.path} must contain image and label datasets")
            image = np.asarray(handle["image"][:])
            label = np.asarray(handle["label"][:])
        if image.ndim != 4 or image.shape[1] != 2 or any(size <= 0 for size in image.shape):
            raise ValueError(f"evaluation image must be non-empty [D,2,H,W], got {image.shape}")
        expected_label_shape = (image.shape[0], image.shape[2], image.shape[3])
        if label.ndim != 3 or label.shape != expected_label_shape:
            raise ValueError(
                "evaluation label must match image spatial shape [D,H,W], got "
                f"image={image.shape} label={label.shape}"
            )
        if not np.all(np.isfinite(image)):
            raise ValueError("evaluation image must contain only finite values")
        validate_label_range(label, classes)
        self.image = image.astype(np.float32, copy=False)
        self.label = label.astype(np.int64, copy=False)
        self.patient_id, self.case_id = parse_test_name(self.path.name)
        self.filename = self.path.name


class ProstateNPZDataset:
    def __init__(self, paths: Iterable[str | Path], classes: Sequence[ClassInfo]) -> None:
        self.paths = [Path(path) for path in paths]
        self.classes = classes
        if not self.paths:
            raise ValueError("ProstateNPZDataset received no NPZ files")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        sample = ProstateTrainSlice(self.paths[index])
        validate_label_range(sample.label, self.classes)
        return {
            "image": torch.from_numpy(sample.image).float(),
            "label": torch.from_numpy(sample.label).long(),
            "patient_id": sample.patient_id,
            "case_id": sample.case_id,
            "sample_id": sample.sample_id,
            "slice_id": sample.slice_id,
            "filename": sample.filename,
            "image_path": str(sample.path.resolve()),
        }


class ProstateH5Dataset:
    def __init__(self, paths: Iterable[str | Path], classes: Sequence[ClassInfo]) -> None:
        self.paths = [Path(path) for path in paths]
        self.classes = classes
        if not self.paths:
            raise ValueError("ProstateH5Dataset received no H5 files")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        volume = ProstateTestVolume(self.paths[index], self.classes)
        return {
            "image": torch.from_numpy(volume.image).float(),
            "label": torch.from_numpy(volume.label).long(),
            "patient_id": volume.patient_id,
            "case_id": volume.case_id,
            "filename": volume.filename,
        }


def list_npz_slices(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"training directory does not exist: {root}")
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No Task05_Prostate NPZ files found under {root}")
    for path in paths:
        parse_train_name(path.name)
    return paths


def list_h5_volumes(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation directory does not exist: {root}")
    paths = sorted(root.glob("*.npy.h5"))
    if not paths:
        raise FileNotFoundError(f"No Task05_Prostate H5 files found under {root}")
    for path in paths:
        parse_test_name(path.name)
    return paths


def training_case_ids(paths: Iterable[str | Path]) -> set[str]:
    return {parse_train_name(Path(path).name)[1] for path in paths}


def evaluation_case_ids(paths: Iterable[str | Path]) -> set[str]:
    return {parse_test_name(Path(path).name)[1] for path in paths}


def validate_case_split(
    train_paths: Iterable[str | Path],
    eval_paths: Iterable[str | Path],
    expected_train_cases: int = 26,
    expected_eval_cases: int = 6,
) -> None:
    train_cases = training_case_ids(train_paths)
    eval_cases = evaluation_case_ids(eval_paths)
    if len(train_cases) != expected_train_cases:
        raise ValueError(f"expected {expected_train_cases} training cases, found {len(train_cases)}")
    if len(eval_cases) != expected_eval_cases:
        raise ValueError(f"expected {expected_eval_cases} evaluation cases, found {len(eval_cases)}")
    overlap = sorted(train_cases & eval_cases)
    if overlap:
        raise ValueError(f"training/evaluation case leakage detected: {overlap}")


def slice_positions(paths: Iterable[str | Path]) -> dict[Path, float]:
    grouped: dict[str, list[tuple[Path, int]]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        _, case_id, _, slice_id = parse_train_name(path.name)
        grouped.setdefault(case_id, []).append((path, slice_id))
    positions: dict[Path, float] = {}
    for entries in grouped.values():
        indices = [index for _, index in entries]
        minimum, maximum = min(indices), max(indices)
        for path, index in entries:
            positions[path] = 0.5 if maximum == minimum else (index - minimum) / (maximum - minimum)
    return positions
