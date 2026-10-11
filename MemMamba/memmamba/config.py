from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from memmamba.model_spec import MODEL_SPEC


DESCRIPTION_FIELDS = (
    "organ_name",
    "anatomical_location",
    "morphology",
    "boundary",
    "neighboring_relations",
    "typical_appearance",
)
DATASET_NAME = "task05_prostate"
MODALITIES = ["T2", "ADC"]
ANATOMY = "prostate"


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str


@dataclass(frozen=True)
class ProjectConfig:
    values: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON configuration does not exist: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    return payload


def load_class_config(path: str | Path) -> tuple[ClassInfo, ...]:
    payload = _load_json(path)
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("class config must contain a non-empty classes list")
    classes = tuple(ClassInfo(int(item["id"]), str(item["name"]).strip()) for item in raw_classes)
    if classes[0] != ClassInfo(0, "background"):
        raise ValueError("class config must start with background class 0")
    if [item.id for item in classes] != list(range(len(classes))):
        raise ValueError("class ids must be contiguous from zero")
    if any(not item.name for item in classes) or len({item.name for item in classes}) != len(classes):
        raise ValueError("class names must be non-empty and unique")
    return classes


def foreground_classes(classes: tuple[ClassInfo, ...]) -> tuple[ClassInfo, ...]:
    if not classes or classes[0].id != 0:
        raise ValueError("classes must start with background class 0")
    return classes[1:]


def load_descriptions(path: str | Path, classes: tuple[ClassInfo, ...]) -> dict[int, dict[str, str]]:
    payload = _load_json(path)
    organs = payload.get("organs")
    if not isinstance(organs, dict):
        raise ValueError("description config must contain an organs object")
    expected = {item.id: item.name for item in foreground_classes(classes)}
    descriptions: dict[int, dict[str, str]] = {}
    for raw_id, raw_record in organs.items():
        class_id = int(raw_id)
        if class_id not in expected or not isinstance(raw_record, dict):
            raise ValueError(f"invalid description class_id={class_id}")
        record: dict[str, str] = {}
        for field in DESCRIPTION_FIELDS:
            value = raw_record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"class_id={class_id} requires non-empty {field}")
            record[field] = value.strip()
        if record["organ_name"] != expected[class_id]:
            raise ValueError(f"class_id={class_id} organ_name differs from class config")
        descriptions[class_id] = record
    if set(descriptions) != set(expected):
        raise ValueError("descriptions must cover every foreground class exactly once")
    return descriptions


def _number(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _validate_resolved(values: Mapping[str, Any]) -> None:
    if values.get("dataset_name") != DATASET_NAME:
        raise ValueError(f"dataset_name must be {DATASET_NAME}")
    if values.get("modality") != MODALITIES or values.get("anatomy") != ANATOMY:
        raise ValueError("modality/anatomy must be [T2, ADC]/prostate")
    for key, expected in MODEL_SPEC.to_dict().items():
        actual = values.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{key} is fixed by the MemMamba model specification")
    if values.get("image_size") != 224:
        raise ValueError("Task05_Prostate requires image_size=224")
    if values.get("expected_train_cases") != 26 or values.get("expected_eval_cases") != 6:
        raise ValueError("Task05_Prostate requires 26 training and 6 evaluation cases")
    morphology = values.get("class_morphology")
    expected_ids = {str(class_id) for class_id in range(1, int(values["num_classes"]))}
    if not isinstance(morphology, dict) or set(morphology) != expected_ids:
        raise ValueError("class_morphology must configure every foreground class exactly once")
    for class_id, options in morphology.items():
        if not isinstance(options, dict):
            raise ValueError(f"class_morphology[{class_id}] must be an object")
        for key in ("erosion_radius", "dilation_radius"):
            value = options.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"class_morphology[{class_id}].{key} must be non-negative int")
    for key in (
        "image_size",
        "min_pixels", "min_effective_patches", "prototypes_per_class",
        "kmeans_max_iter", "epochs", "batch_size", "learning_rate",
    ):
        if _number(values, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in (
        "lambda_aux", "lambda_aux_end", "lambda_presence", "lambda_align",
        "dice_weight", "ce_weight", "weight_decay", "max_grad_norm",
    ):
        if _number(values, key) < 0:
            raise ValueError(f"{key} must be non-negative")
    if _number(values, "warmup_epochs") >= _number(values, "epochs"):
        raise ValueError("warmup_epochs must be smaller than epochs")
    if not (
        0 < _number(values, "min_learning_rate")
        <= _number(values, "warmup_start_lr")
        <= _number(values, "learning_rate")
    ):
        raise ValueError("learning rates must satisfy min <= warmup <= peak")
    required_paths = (
        "train_npz_dir", "test_h5_dir", "class_config", "description_path",
        "semantic_embedding_path", "visual_feature_path",
        "alignment_teacher_path", "memory_bank_path", "checkpoint_dir", "log_dir",
        "result_path", "prediction_dir",
    )
    for key in required_paths:
        if not isinstance(values.get(key), str) or not values[key].strip():
            raise ValueError(f"{key} must be a non-empty path")


def load_defaults(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    defaults = payload.get("defaults")
    if not isinstance(defaults, dict) or not defaults:
        raise ValueError("configuration must contain a non-empty defaults object")
    fixed = MODEL_SPEC.to_dict()
    forbidden = set(defaults) & set(fixed)
    if forbidden:
        raise ValueError(f"Model settings are internal and must not appear in runtime JSON: {sorted(forbidden)}")
    result = {**defaults, **fixed}
    _validate_resolved(result)
    return result


def resolve_config(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> ProjectConfig:
    fixed_overrides = set(overrides) & set(MODEL_SPEC.to_dict())
    if fixed_overrides:
        raise ValueError(f"Cannot override fixed MemMamba settings: {sorted(fixed_overrides)}")
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise KeyError(f"unknown configuration keys: {unknown}")
    resolved = dict(defaults)
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    _validate_resolved(resolved)
    return ProjectConfig(resolved)
