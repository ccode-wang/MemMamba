from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memmamba.config import ClassInfo, DATASET_NAME, MODALITIES, foreground_classes


STRICT_SOURCE = "strict_regions"
ALIGNMENT_ARTIFACT_VERSION = 3
_ARTIFACT_TYPE = "alignment_teacher_bank"
_RECORD_FIELDS = {
    "case_id",
    "slice_id",
    "class_id",
    "class_name",
    "source",
    "effective_patch_count",
    "image_path",
}
_PAYLOAD_FIELDS = {
    "format_version",
    "artifact_type",
    "model_name",
    "feature_dim",
    "classes",
    "extraction_config",
    "class_ids",
    "features",
    "confidences",
    "records",
    "dataset_name",
    "modality",
    "split",
}


@dataclass(frozen=True)
class AlignmentTeacherRecord:
    case_id: str
    slice_id: int
    class_id: int
    class_name: str
    source: str
    effective_patch_count: int
    image_path: str


@dataclass(frozen=True)
class AlignmentTeacherValue:
    feature: Any
    confidence: float
    source: str


def _validate_classes(classes: tuple[ClassInfo, ...]) -> dict[int, str]:
    if not isinstance(classes, tuple) or len(classes) < 2:
        raise ValueError("alignment teacher classes must be a tuple with foreground classes")
    if any(not isinstance(item, ClassInfo) for item in classes):
        raise ValueError("alignment teacher classes must contain ClassInfo values")
    if any(type(item.id) is not int for item in classes):
        raise ValueError("alignment teacher class ids must use type int")
    if classes[0] != ClassInfo(0, "background"):
        raise ValueError("alignment teacher classes must start with id=0 name=background")
    ids = [item.id for item in classes]
    names = [item.name for item in classes]
    if ids != list(range(len(classes))):
        raise ValueError("alignment teacher class ids must be contiguous from zero")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("alignment teacher class names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("alignment teacher class names must be unique")
    return {item.id: item.name for item in foreground_classes(classes)}


@dataclass
class AlignmentTeacherBank:
    class_ids: Any
    features: Any
    confidences: Any
    records: list[AlignmentTeacherRecord]
    classes: tuple[ClassInfo, ...]
    model_name: str
    extraction_config: dict[str, Any]
    dataset_name: str = DATASET_NAME
    modality: list[str] | tuple[str, ...] = tuple(MODALITIES)
    split: str = "train"

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def validate(self, require_all_classes: bool = True) -> None:
        import torch

        configured = _validate_classes(self.classes)
        if self.dataset_name != DATASET_NAME or list(self.modality) != MODALITIES or self.split != "train":
            raise ValueError("alignment teachers must be Task05_Prostate T2/ADC train artifacts")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("alignment teacher model_name must be a non-empty string")
        if not isinstance(self.extraction_config, dict):
            raise ValueError("alignment teacher extraction_config must be a dict")
        if not isinstance(self.records, list) or any(
            not isinstance(record, AlignmentTeacherRecord) for record in self.records
        ):
            raise ValueError("alignment teacher records must be AlignmentTeacherRecord values")

        count = len(self.records)
        if not torch.is_tensor(self.class_ids) or self.class_ids.dtype != torch.int64:
            raise ValueError("alignment teacher class_ids must use dtype int64")
        if self.class_ids.shape != (count,):
            raise ValueError(
                f"alignment teacher class_ids must be [{count}], got {tuple(self.class_ids.shape)}"
            )
        if not torch.is_tensor(self.features) or self.features.dtype != torch.float32:
            raise ValueError("alignment teacher features must use dtype float32")
        if self.features.ndim != 2 or self.features.shape[0] != count or self.features.shape[1] <= 0:
            shape = tuple(self.features.shape) if torch.is_tensor(self.features) else None
            raise ValueError(f"alignment teacher features must be [{count},D] with D>0, got {shape}")
        if not torch.isfinite(self.features).all():
            raise ValueError("alignment teacher features contain non-finite values")
        expected_norms = torch.ones(count, dtype=self.features.dtype, device=self.features.device)
        if not torch.allclose(
            self.features.norm(dim=-1), expected_norms, atol=1e-4, rtol=1e-4
        ):
            raise ValueError("alignment teacher feature rows must be L2-normalized")

        if not torch.is_tensor(self.confidences) or self.confidences.dtype != torch.float32:
            raise ValueError("alignment teacher confidences must use dtype float32")
        if self.confidences.shape != (count,):
            raise ValueError(
                f"alignment teacher confidences must be [{count}], got {tuple(self.confidences.shape)}"
            )
        if not torch.isfinite(self.confidences).all():
            raise ValueError("alignment teacher confidences contain non-finite values")
        if torch.any((self.confidences <= 0.0) | (self.confidences > 1.0)):
            raise ValueError("alignment teacher confidences must lie in (0,1]")

        observed = set(map(int, self.class_ids.tolist()))
        unknown = observed - set(configured)
        if unknown:
            raise ValueError(f"alignment teacher bank contains unknown classes: {sorted(unknown)}")

        keys: list[tuple[str, int, int]] = []
        for index, record in enumerate(self.records):
            if type(record.class_id) is not int:
                raise ValueError(f"alignment teacher record class_id must be an integer at index={index}")
            tensor_id = int(self.class_ids[index].item())
            if tensor_id != record.class_id:
                raise ValueError(
                    f"class_ids[{index}]={tensor_id} disagrees with "
                    f"record.class_id={record.class_id}"
                )
            if configured.get(record.class_id) != record.class_name:
                raise ValueError(f"alignment teacher record class mismatch at index={index}")
            if record.source != STRICT_SOURCE:
                raise ValueError(f"alignment teacher source is invalid at index={index}")
            confidence = float(self.confidences[index].item())
            if confidence != 1.0:
                raise ValueError("strict alignment teacher confidence must equal 1.0")
            if type(record.effective_patch_count) is not int or record.effective_patch_count <= 0:
                raise ValueError(
                    f"alignment teacher effective_patch_count must be positive at index={index}"
                )
            if type(record.slice_id) is not int or record.slice_id < 0:
                raise ValueError(f"alignment teacher slice_id must be non-negative int at index={index}")
            if any(
                not isinstance(value, str) or not value.strip()
                for value in (record.case_id, record.class_name, record.image_path)
            ):
                raise ValueError(f"alignment teacher record strings must be non-empty at index={index}")
            keys.append((record.case_id, record.slice_id, record.class_id))

        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (case_id, slice_id, class_id) alignment teacher records")
        if require_all_classes and observed != set(configured):
            raise ValueError(
                f"alignment teacher bank missing classes: {sorted(set(configured) - observed)}"
            )

    def teacher_lookup(self) -> dict[tuple[str, int, int], AlignmentTeacherValue]:
        return {
            (record.case_id, record.slice_id, record.class_id): AlignmentTeacherValue(
                feature=self.features[index],
                confidence=float(self.confidences[index].item()),
                source=record.source,
            )
            for index, record in enumerate(self.records)
        }


def _torch_load(path: Path, map_location: str | Any) -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_alignment_teacher_bank(bank: AlignmentTeacherBank, path: str | Path) -> None:
    import torch

    bank.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": ALIGNMENT_ARTIFACT_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "model_name": bank.model_name,
            "feature_dim": bank.feature_dim,
            "classes": [asdict(item) for item in bank.classes],
            "extraction_config": dict(bank.extraction_config),
            "class_ids": bank.class_ids.cpu(),
            "features": bank.features.cpu(),
            "confidences": bank.confidences.cpu(),
            "records": [asdict(item) for item in bank.records],
            "dataset_name": bank.dataset_name,
            "modality": list(bank.modality),
            "split": bank.split,
        },
        output,
    )


def _load_classes(raw_classes: Any) -> tuple[ClassInfo, ...]:
    if not isinstance(raw_classes, list):
        raise ValueError("alignment teacher classes metadata must be a list")
    classes: list[ClassInfo] = []
    for index, item in enumerate(raw_classes):
        if not isinstance(item, Mapping) or set(item) != {"id", "name"}:
            raise ValueError(f"alignment teacher classes[{index}] has invalid fields")
        if type(item["id"]) is not int or not isinstance(item["name"], str):
            raise ValueError(f"alignment teacher classes[{index}] has invalid types")
        classes.append(ClassInfo(id=item["id"], name=item["name"]))
    result = tuple(classes)
    _validate_classes(result)
    return result


def _load_records(raw_records: Any) -> list[AlignmentTeacherRecord]:
    if not isinstance(raw_records, list):
        raise ValueError("alignment teacher records metadata must be a list")
    records: list[AlignmentTeacherRecord] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping) or set(item) != _RECORD_FIELDS:
            raise ValueError(f"alignment teacher records[{index}] has invalid fields")
        records.append(AlignmentTeacherRecord(**dict(item)))
    return records


def load_alignment_teacher_bank(
    path: str | Path, map_location: str | Any = "cpu"
) -> AlignmentTeacherBank:
    import torch

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Alignment Teacher Bank does not exist: {artifact}")
    payload = _torch_load(artifact, map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Alignment Teacher Bank payload must be a mapping")
    payload_fields = set(payload)
    if payload_fields != _PAYLOAD_FIELDS:
        raise KeyError(
            "Alignment Teacher Bank fields differ: "
            f"missing={sorted(_PAYLOAD_FIELDS - payload_fields)} "
            f"extra={sorted(payload_fields - _PAYLOAD_FIELDS)}"
        )
    if type(payload["format_version"]) is not int:
        raise ValueError("Alignment Teacher Bank format_version must be an integer")
    if payload["format_version"] != ALIGNMENT_ARTIFACT_VERSION:
        raise ValueError("Unsupported Alignment Teacher Bank format_version")
    if payload["artifact_type"] != _ARTIFACT_TYPE:
        raise ValueError("Wrong Alignment Teacher Bank artifact_type")
    if type(payload["feature_dim"]) is not int or payload["feature_dim"] <= 0:
        raise ValueError("Alignment Teacher Bank feature_dim must be a positive integer")
    if not isinstance(payload["model_name"], str) or not payload["model_name"].strip():
        raise ValueError("Alignment Teacher Bank model_name must be a non-empty string")
    if not isinstance(payload["extraction_config"], dict):
        raise ValueError("Alignment Teacher Bank extraction_config must be a dict")
    for field in ("class_ids", "features", "confidences"):
        if not torch.is_tensor(payload[field]):
            raise ValueError(f"Alignment Teacher Bank {field} must be a tensor")

    bank = AlignmentTeacherBank(
        class_ids=payload["class_ids"],
        features=payload["features"],
        confidences=payload["confidences"],
        records=_load_records(payload["records"]),
        classes=_load_classes(payload["classes"]),
        model_name=payload["model_name"],
        extraction_config=dict(payload["extraction_config"]),
        dataset_name=str(payload["dataset_name"]),
        modality=list(payload["modality"]),
        split=str(payload["split"]),
    )
    bank.validate()
    if payload["feature_dim"] != bank.feature_dim:
        raise ValueError("Alignment Teacher Bank feature_dim metadata is inconsistent")
    return bank
