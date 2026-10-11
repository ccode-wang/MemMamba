from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from memmamba.config import ClassInfo, DATASET_NAME, MODALITIES, foreground_classes
from memmamba.memory.alignment_teachers import (
    STRICT_SOURCE,
    AlignmentTeacherBank,
    AlignmentTeacherRecord,
)


ARTIFACT_VERSION = 4
LOGGER = logging.getLogger(__name__)
REGIONS = ("interior", "boundary", "context")


@dataclass(frozen=True)
class VisualExtractionOptions:
    class_morphology: dict[int, dict[str, int]]
    min_effective_patches: int = 1
    min_pixels: int = 8
    batch_size: int = 8

    def validate(self) -> None:
        if not self.class_morphology:
            raise ValueError("class_morphology must not be empty")
        for class_id, values in self.class_morphology.items():
            if type(class_id) is not int or class_id <= 0 or not isinstance(values, dict):
                raise ValueError("class_morphology keys must be positive class ids")
            for name in ("erosion_radius", "dilation_radius"):
                value = values.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"class {class_id} {name} must be non-negative int")
        if min(self.min_effective_patches, self.min_pixels, self.batch_size) <= 0:
            raise ValueError("patch, pixel, and batch thresholds must be positive")


@dataclass(frozen=True)
class VisualFeatureRecord:
    patient_id: str
    case_id: str
    sample_id: str
    slice_id: int
    class_id: int
    class_name: str
    normalized_slice_position: float
    organ_pixel_count: int
    effective_patch_count: int
    interior_patch_count: int
    boundary_patch_count: int
    context_patch_count: int
    interior_fallback: bool
    image_path: str


def _validate_unit_rows(tensor, name: str, count: int) -> None:
    import torch

    if tensor.ndim != 2 or tensor.shape[0] != count:
        raise ValueError(f"{name} must be [M,D] with M={count}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    if not torch.allclose(
        tensor.norm(dim=-1), torch.ones(count, device=tensor.device), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"{name} rows must be L2-normalized")


@dataclass
class VisualFeatureBank:
    class_ids: Any
    interior_t2: Any
    interior_adc: Any
    interior: Any
    boundary_t2: Any
    boundary_adc: Any
    boundary: Any
    context_t2: Any
    context_adc: Any
    context: Any
    visual_key: Any
    records: list[VisualFeatureRecord]
    classes: tuple[ClassInfo, ...]
    extraction_config: dict[str, Any]
    dataset_name: str = DATASET_NAME
    modality: list[str] | tuple[str, ...] = tuple(MODALITIES)
    split: str = "train"

    @property
    def feature_dim(self) -> int:
        return int(self.visual_key.shape[1])

    def validate(self, require_all_classes: bool = True) -> None:
        count = len(self.records)
        if self.class_ids.ndim != 1 or self.class_ids.shape[0] != count:
            raise ValueError("visual class_ids and records have inconsistent lengths")
        tensor_names = (
            "interior_t2", "interior_adc", "interior",
            "boundary_t2", "boundary_adc", "boundary",
            "context_t2", "context_adc", "context", "visual_key",
        )
        for name in tensor_names:
            _validate_unit_rows(getattr(self, name), name, count)
        dimensions = {int(getattr(self, name).shape[1]) for name in tensor_names}
        if len(dimensions) != 1:
            raise ValueError(f"visual feature dimensions differ: {sorted(dimensions)}")
        if self.dataset_name != DATASET_NAME or list(self.modality) != MODALITIES or self.split != "train":
            raise ValueError("visual features must be Task05_Prostate T2/ADC train features")
        configured = {item.id: item.name for item in foreground_classes(self.classes)}
        keys: list[tuple[str, int, int]] = []
        for index, record in enumerate(self.records):
            if int(self.class_ids[index].item()) != record.class_id:
                raise ValueError(f"visual class id mismatch at index={index}")
            if configured.get(record.class_id) != record.class_name:
                raise ValueError(f"visual class name mismatch at index={index}")
            if (
                record.patient_id != record.case_id
                or record.sample_id != f"{record.case_id}_slice{record.slice_id:03d}"
                or type(record.slice_id) is not int
                or record.slice_id < 0
                or not record.image_path
            ):
                raise ValueError(f"visual record identity is invalid at index={index}")
            if not 0.0 <= record.normalized_slice_position <= 1.0:
                raise ValueError(f"slice position outside [0,1] at index={index}")
            if min(
                record.organ_pixel_count,
                record.effective_patch_count,
                record.interior_patch_count,
                record.boundary_patch_count,
                record.context_patch_count,
            ) <= 0:
                raise ValueError(f"visual record counts must be positive at index={index}")
            keys.append((record.case_id, record.slice_id, record.class_id))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (case_id, slice_id, class_id) visual records")
        observed = set(map(int, self.class_ids.tolist()))
        if not observed <= set(configured):
            raise ValueError(f"visual bank contains unknown classes: {sorted(observed - set(configured))}")
        if require_all_classes and observed != set(configured):
            raise ValueError(f"visual bank missing classes: {sorted(set(configured) - observed)}")


@dataclass
class SemanticFeatureBank:
    class_ids: Any
    embeddings: Any
    class_names: list[str]
    rendered_texts: list[str]
    classes: tuple[ClassInfo, ...]
    model_name: str
    description_path: str
    dataset_name: str = DATASET_NAME
    modality: list[str] | tuple[str, ...] = tuple(MODALITIES)

    @property
    def feature_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def validate(self) -> None:
        if self.dataset_name != DATASET_NAME or list(self.modality) != MODALITIES:
            raise ValueError("semantic features must be Task05_Prostate T2/ADC features")
        expected = foreground_classes(self.classes)
        if self.class_ids.tolist() != [item.id for item in expected]:
            raise ValueError("semantic class ids do not match the class configuration")
        if self.class_names != [item.name for item in expected]:
            raise ValueError("semantic class names do not match the class configuration")
        _validate_unit_rows(self.embeddings, "semantic embeddings", len(expected))
        if len(self.rendered_texts) != len(expected) or any(not text.strip() for text in self.rendered_texts):
            raise ValueError("semantic rendered texts are incomplete")


def _torch_load(path: Path, map_location: str | Any):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _classes_payload(classes: tuple[ClassInfo, ...]) -> list[dict[str, Any]]:
    return [asdict(item) for item in classes]


def save_visual_feature_bank(bank: VisualFeatureBank, path: str | Path) -> None:
    import torch

    bank.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_names = (
        "interior_t2", "interior_adc", "interior",
        "boundary_t2", "boundary_adc", "boundary",
        "context_t2", "context_adc", "context", "visual_key",
    )
    payload = {
        "format_version": ARTIFACT_VERSION,
        "artifact_type": "visual_feature_bank",
        "classes": _classes_payload(bank.classes),
        "feature_dim": bank.feature_dim,
        "extraction_config": bank.extraction_config,
        "dataset_name": bank.dataset_name,
        "modality": list(bank.modality),
        "split": bank.split,
        "class_ids": bank.class_ids.cpu(),
        "records": [asdict(item) for item in bank.records],
    }
    payload.update({name: getattr(bank, name).cpu() for name in tensor_names})
    torch.save(payload, output)


def load_visual_feature_bank(path: str | Path, map_location: str | Any = "cpu") -> VisualFeatureBank:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Visual Feature Bank does not exist: {artifact}")
    payload = _torch_load(artifact, map_location)
    tensor_names = (
        "interior_t2", "interior_adc", "interior",
        "boundary_t2", "boundary_adc", "boundary",
        "context_t2", "context_adc", "context", "visual_key",
    )
    required = {
        "format_version", "artifact_type", "classes", "feature_dim", "extraction_config",
        "dataset_name", "modality", "split", "class_ids", "records", *tensor_names,
    }
    if set(payload) != required:
        raise KeyError(f"Visual Feature Bank fields differ: missing={sorted(required-set(payload))} extra={sorted(set(payload)-required)}")
    if payload["format_version"] != ARTIFACT_VERSION or payload["artifact_type"] != "visual_feature_bank":
        raise ValueError(f"Unsupported or wrong visual artifact: {artifact}")
    classes = tuple(ClassInfo(**item) for item in payload["classes"])
    values = [payload[name].float() for name in tensor_names]
    bank = VisualFeatureBank(
        payload["class_ids"].long(), *values,
        [VisualFeatureRecord(**item) for item in payload["records"]],
        classes, dict(payload["extraction_config"]), str(payload["dataset_name"]),
        list(payload["modality"]), str(payload["split"]),
    )
    bank.validate()
    if int(payload["feature_dim"]) != bank.feature_dim:
        raise ValueError("Visual Feature Bank feature_dim metadata is inconsistent")
    return bank


def save_semantic_feature_bank(bank: SemanticFeatureBank, path: str | Path) -> None:
    import torch

    bank.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": ARTIFACT_VERSION,
            "artifact_type": "semantic_feature_bank",
            "classes": _classes_payload(bank.classes),
            "feature_dim": bank.feature_dim,
            "class_ids": bank.class_ids.cpu(),
            "embeddings": bank.embeddings.cpu(),
            "class_names": bank.class_names,
            "rendered_texts": bank.rendered_texts,
            "model_name": bank.model_name,
            "description_path": bank.description_path,
            "dataset_name": bank.dataset_name,
            "modality": list(bank.modality),
        },
        output,
    )


def load_semantic_feature_bank(path: str | Path, map_location: str | Any = "cpu") -> SemanticFeatureBank:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Semantic Feature Bank does not exist: {artifact}")
    payload = _torch_load(artifact, map_location)
    required = {
        "format_version", "artifact_type", "classes", "feature_dim", "class_ids",
        "embeddings", "class_names", "rendered_texts", "model_name",
        "description_path", "dataset_name", "modality",
    }
    if set(payload) != required:
        raise KeyError(f"Semantic Feature Bank fields differ: missing={sorted(required-set(payload))} extra={sorted(set(payload)-required)}")
    if payload["format_version"] != ARTIFACT_VERSION or payload["artifact_type"] != "semantic_feature_bank":
        raise ValueError(f"Unsupported or wrong semantic artifact: {artifact}")
    bank = SemanticFeatureBank(
        payload["class_ids"].long(), payload["embeddings"].float(),
        list(payload["class_names"]), list(payload["rendered_texts"]),
        tuple(ClassInfo(**item) for item in payload["classes"]),
        str(payload["model_name"]), str(payload["description_path"]),
        str(payload["dataset_name"]), list(payload["modality"]),
    )
    bank.validate()
    if int(payload["feature_dim"]) != bank.feature_dim:
        raise ValueError("Semantic Feature Bank feature_dim metadata is inconsistent")
    return bank


def extract_visual_and_alignment_feature_banks(
    paths: Iterable[str | Path],
    encoder,
    classes: tuple[ClassInfo, ...],
    options: VisualExtractionOptions,
) -> tuple[VisualFeatureBank, AlignmentTeacherBank]:
    import torch
    import torch.nn.functional as functional
    from memmamba.data.task05_prostate import (
        ProstateTrainSlice,
        slice_positions,
        training_case_ids,
        validate_label_range,
    )
    from memmamba.memory.regions import build_region_masks, pool_patch_region, resize_mask_to_patch_grid

    options.validate()
    resolved_paths = [Path(path) for path in paths]
    if not resolved_paths:
        raise ValueError("visual extraction received no training slices")
    if len(training_case_ids(resolved_paths)) != 26:
        raise ValueError("visual features and Alignment Teacher must use all 26 training cases")
    positions = slice_positions(resolved_paths)
    organs = foreground_classes(classes)
    if set(options.class_morphology) != {item.id for item in organs}:
        raise ValueError("class_morphology must configure every foreground class exactly once")
    tensor_names = (
        "interior_t2", "interior_adc", "interior",
        "boundary_t2", "boundary_adc", "boundary",
        "context_t2", "context_adc", "context", "visual_key",
    )
    vectors: dict[str, list[Any]] = {name: [] for name in tensor_names}
    class_ids: list[int] = []
    records: list[VisualFeatureRecord] = []
    teacher_class_ids: list[int] = []
    teacher_features: list[Any] = []
    teacher_confidences: list[float] = []
    teacher_records: list[AlignmentTeacherRecord] = []
    stat_names = ("samples", "pixel_filtered", "patch_filtered", "region_filtered", "interior_fallback")
    stats = {name: {item.id: 0 for item in organs} for name in stat_names}
    patch_stats = {name: {item.id: [] for item in organs} for name in ("organ", *REGIONS)}

    def add_teacher(sample, info, feature, confidence, source, patch_count):
        teacher_class_ids.append(info.id)
        teacher_features.append(feature.detach().cpu())
        teacher_confidences.append(confidence)
        teacher_records.append(
            AlignmentTeacherRecord(sample.case_id, sample.slice_id, info.id, info.name, source, patch_count, str(sample.path.resolve()))
        )

    for start in range(0, len(resolved_paths), options.batch_size):
        samples = [ProstateTrainSlice(path) for path in resolved_paths[start:start + options.batch_size]]
        for sample in samples:
            validate_label_range(sample.label, classes)
        t2_tokens, t2_grid = encoder.encode_patch_tokens([sample.image[0] for sample in samples])
        adc_tokens, adc_grid = encoder.encode_patch_tokens([sample.image[1] for sample in samples])
        if t2_grid != adc_grid or t2_tokens.shape != adc_tokens.shape:
            raise ValueError("T2 and ADC BiomedCLIP patch token geometry differs")
        for batch_index, sample in enumerate(samples):
            modality_tokens = {
                "t2": t2_tokens[batch_index:batch_index + 1],
                "adc": adc_tokens[batch_index:batch_index + 1],
            }
            for info in organs:
                organ_mask = sample.label == info.id
                pixels = int(organ_mask.sum())
                if pixels == 0:
                    continue
                stats["samples"][info.id] += 1
                if pixels < options.min_pixels:
                    stats["pixel_filtered"][info.id] += 1
                    continue
                morphology = options.class_morphology[info.id]
                regions = build_region_masks(organ_mask, morphology["erosion_radius"], morphology["dilation_radius"])
                if regions.interior_fallback:
                    stats["interior_fallback"][info.id] += 1
                organ_patch = resize_mask_to_patch_grid(encoder.transform_mask(organ_mask.astype(np.uint8)), t2_grid)
                effective = int(organ_patch.sum().item())
                patch_stats["organ"][info.id].append(effective)
                if effective < options.min_effective_patches:
                    stats["patch_filtered"][info.id] += 1
                    continue
                patch_masks: dict[str, Any] = {}
                counts: dict[str, int] = {}
                for region_name in REGIONS:
                    transformed = encoder.transform_mask(getattr(regions, region_name).astype(np.uint8))
                    patch_masks[region_name] = resize_mask_to_patch_grid(transformed, t2_grid)
                    counts[region_name] = int(patch_masks[region_name].sum().item())
                    patch_stats[region_name][info.id].append(counts[region_name])
                invalid = [name for name in REGIONS if counts[name] == 0]
                if invalid:
                    stats["region_filtered"][info.id] += 1
                    continue
                fused_regions: dict[str, Any] = {}
                for region_name in REGIONS:
                    modal = {
                        name: pool_patch_region(tokens, patch_masks[region_name])[0][0]
                        for name, tokens in modality_tokens.items()
                    }
                    fused = functional.normalize((modal["t2"] + modal["adc"]) / 2.0, dim=0)
                    vectors[f"{region_name}_t2"].append(modal["t2"].detach().cpu())
                    vectors[f"{region_name}_adc"].append(modal["adc"].detach().cpu())
                    vectors[region_name].append(fused.detach().cpu())
                    fused_regions[region_name] = fused
                visual_key = functional.normalize(sum(fused_regions.values()) / len(REGIONS), dim=0)
                vectors["visual_key"].append(visual_key.detach().cpu())
                class_ids.append(info.id)
                records.append(
                    VisualFeatureRecord(
                        sample.patient_id, sample.case_id, sample.sample_id, sample.slice_id,
                        info.id, info.name, float(positions[sample.path]), pixels, effective,
                        counts["interior"], counts["boundary"], counts["context"],
                        regions.interior_fallback, str(sample.path.resolve()),
                    )
                )
                add_teacher(sample, info, visual_key, 1.0, STRICT_SOURCE, effective)
    if not records:
        raise RuntimeError("No valid foreground visual features were extracted")
    extraction_config = {
        **asdict(options), **stats, "dataset_name": DATASET_NAME,
        "modality": list(MODALITIES), "anatomy": "prostate", "split": "train",
        "fusion": "l2_normalize((t2 + adc) / 2)",
    }
    bank = VisualFeatureBank(
        torch.tensor(class_ids, dtype=torch.long),
        *(torch.stack(vectors[name]) for name in tensor_names),
        records, classes, extraction_config,
    )

    def patch_range(values: list[int]) -> tuple[str, str, str]:
        return ("n/a", "n/a", "n/a") if not values else (str(min(values)), f"{sum(values)/len(values):.2f}", str(max(values)))

    for info in organs:
        sample_count = stats["samples"][info.id]
        valid_count = int((bank.class_ids == info.id).sum())
        LOGGER.info(
            "class=%d %s samples=%d valid_features=%d retention_rate=%.2f%% pixel_filtered=%d patch_filtered=%d region_filtered=%d interior_fallback=%d",
            info.id, info.name, sample_count, valid_count, 100.0 * valid_count / max(sample_count, 1),
            stats["pixel_filtered"][info.id], stats["patch_filtered"][info.id],
            stats["region_filtered"][info.id], stats["interior_fallback"][info.id],
        )
        for region_name in ("organ", *REGIONS):
            minimum, mean, maximum = patch_range(patch_stats[region_name][info.id])
            LOGGER.info("class=%d %s effective_patch_count[%s] min=%s mean=%s max=%s", info.id, info.name, region_name, minimum, mean, maximum)
    LOGGER.info("Feature Bank final records=%d", len(records))
    bank.validate()
    model_name = str(getattr(encoder, "model_name", "")).strip() or encoder.__class__.__qualname__
    alignment_bank = AlignmentTeacherBank(
        torch.tensor(teacher_class_ids, dtype=torch.long), torch.stack(teacher_features).float(),
        torch.tensor(teacher_confidences, dtype=torch.float32), teacher_records,
        classes, model_name, dict(extraction_config),
    )
    alignment_bank.validate()
    LOGGER.info(
        "Alignment Teacher Bank records=%d strict=%d dim=%d",
        len(teacher_records), sum(record.source == STRICT_SOURCE for record in teacher_records),
        alignment_bank.feature_dim,
    )
    return bank, alignment_bank
