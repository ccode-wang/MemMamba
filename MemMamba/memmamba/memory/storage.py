from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memmamba.config import ANATOMY, ClassInfo, DATASET_NAME, MODALITIES, foreground_classes


MEMORY_VERSION = 3


@dataclass
class MultimodalMedicalKnowledgeMemory:
    class_ids: Any
    class_names: list[str]
    visual_key: Any
    interior: Any
    boundary: Any
    context: Any
    semantic: Any
    normalized_slice_position: Any
    cluster_size: Any
    classes: tuple[ClassInfo, ...]
    build_config: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def visual_dim(self) -> int:
        return int(self.visual_key.shape[1])

    @property
    def semantic_dim(self) -> int:
        return int(self.semantic.shape[1])

    @property
    def class_offsets(self) -> dict[int, tuple[int, int]]:
        offsets = {}
        for info in foreground_classes(self.classes):
            indices = self.indices_for_class(info.id)
            if indices.numel() == 0:
                continue
            start, end = int(indices[0]), int(indices[-1]) + 1
            if indices.tolist() != list(range(start, end)):
                raise ValueError(f"Memory prototypes for class {info.id} are not contiguous")
            offsets[info.id] = (start, end)
        return offsets

    def validate(self) -> None:
        import torch

        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if self.class_ids.dtype not in integer_dtypes or self.cluster_size.dtype not in integer_dtypes:
            raise ValueError("Memory integer fields have invalid dtype")
        count = int(self.class_ids.numel())
        if self.class_ids.ndim != 1 or count == 0:
            raise ValueError("Memory class_ids must be non-empty [P]")
        tensors = {
            "visual_key": self.visual_key, "interior": self.interior,
            "boundary": self.boundary, "context": self.context, "semantic": self.semantic,
        }
        for name, tensor in tensors.items():
            if not tensor.is_floating_point() or tensor.ndim != 2 or tensor.shape[0] != count or not torch.isfinite(tensor).all():
                raise ValueError(f"Memory {name} must be floating finite [P,D]")
            if not torch.allclose(tensor.norm(dim=-1), torch.ones(count, device=tensor.device), atol=1e-4, rtol=1e-4):
                raise ValueError(f"Memory {name} rows must be L2-normalized")
        if len({self.visual_dim, self.interior.shape[1], self.boundary.shape[1], self.context.shape[1]}) != 1:
            raise ValueError("Memory visual prototype dimensions differ")
        if self.normalized_slice_position.shape != (count,) or not torch.isfinite(self.normalized_slice_position).all():
            raise ValueError("Memory normalized_slice_position must be finite [P]")
        if torch.any((self.normalized_slice_position < 0) | (self.normalized_slice_position > 1)):
            raise ValueError("Memory slice positions must lie in [0,1]")
        if self.cluster_size.shape != (count,) or torch.any(self.cluster_size <= 0):
            raise ValueError("Memory cluster_size must be positive [P]")
        configured = {item.id: item.name for item in foreground_classes(self.classes)}
        if set(map(int, self.class_ids.tolist())) != set(configured):
            raise ValueError("Memory must contain every foreground class")
        if len(self.class_names) != count:
            raise ValueError("Memory class_names length mismatch")
        for index, class_id in enumerate(self.class_ids.tolist()):
            if self.class_names[index] != configured[int(class_id)]:
                raise ValueError(f"Memory class name mismatch at {index}")
        if set(self.class_offsets) != set(configured):
            raise ValueError("Memory class_offsets missing classes")
        expected_metadata = {
            "dataset_name": DATASET_NAME,
            "modality": list(MODALITIES),
            "anatomy": ANATOMY,
            "num_classes": len(self.classes),
            "foreground_class_ids": list(configured),
        }
        if self.metadata != expected_metadata:
            raise ValueError(f"Memory metadata differs from Task05_Prostate contract: expected={expected_metadata} got={self.metadata}")

    def indices_for_class(self, class_id: int):
        import torch

        return torch.nonzero(self.class_ids == int(class_id), as_tuple=False).flatten()

    def to(self, device) -> "MultimodalMedicalKnowledgeMemory":
        return MultimodalMedicalKnowledgeMemory(
            self.class_ids.to(device), self.class_names, self.visual_key.to(device),
            self.interior.to(device), self.boundary.to(device), self.context.to(device),
            self.semantic.to(device), self.normalized_slice_position.to(device),
            self.cluster_size.to(device), self.classes, self.build_config, self.metadata,
        )


def _payload(bank: MultimodalMedicalKnowledgeMemory) -> dict[str, Any]:
    return {
        "format_version": MEMORY_VERSION, "artifact_type": "multimodal_memory_bank",
        "classes": [asdict(item) for item in bank.classes], "visual_dim": bank.visual_dim,
        "semantic_dim": bank.semantic_dim, "build_config": bank.build_config,
        "class_offsets": bank.class_offsets, "class_ids": bank.class_ids.cpu(),
        "metadata": bank.metadata, "class_names": bank.class_names,
        "visual_key": bank.visual_key.cpu(), "interior_prototype": bank.interior.cpu(),
        "boundary_prototype": bank.boundary.cpu(), "context_prototype": bank.context.cpu(),
        "semantic_embedding": bank.semantic.cpu(),
        "normalized_slice_position": bank.normalized_slice_position.cpu(),
        "cluster_size": bank.cluster_size.cpu(),
    }


def save_memory_bank(bank: MultimodalMedicalKnowledgeMemory, path: str | Path) -> None:
    import torch

    bank.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_payload(bank), output)


def load_memory_bank(path: str | Path, map_location: str | Any = "cpu", expected_dataset_name: str | None = None, expected_num_classes: int | None = None) -> MultimodalMedicalKnowledgeMemory:
    import torch

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Memory Bank does not exist: {artifact}")
    try:
        payload = torch.load(artifact, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(artifact, map_location=map_location)
    required = {
        "format_version", "artifact_type", "classes", "visual_dim", "semantic_dim",
        "build_config", "metadata", "class_offsets", "class_ids", "class_names",
        "visual_key", "interior_prototype", "boundary_prototype", "context_prototype",
        "semantic_embedding", "normalized_slice_position", "cluster_size",
    }
    if set(payload) != required:
        raise KeyError(f"Memory Bank fields differ: missing={sorted(required-set(payload))} extra={sorted(set(payload)-required)}")
    if payload["format_version"] != MEMORY_VERSION or payload["artifact_type"] != "multimodal_memory_bank":
        raise ValueError("Wrong Memory artifact")
    bank = MultimodalMedicalKnowledgeMemory(
        payload["class_ids"], list(payload["class_names"]), payload["visual_key"],
        payload["interior_prototype"], payload["boundary_prototype"], payload["context_prototype"],
        payload["semantic_embedding"], payload["normalized_slice_position"], payload["cluster_size"],
        tuple(ClassInfo(**item) for item in payload["classes"]), dict(payload["build_config"]), dict(payload["metadata"]),
    )
    bank.validate()
    if {int(key): tuple(value) for key, value in payload["class_offsets"].items()} != bank.class_offsets:
        raise ValueError("Memory class_offsets metadata is inconsistent")
    if int(payload["visual_dim"]) != bank.visual_dim or int(payload["semantic_dim"]) != bank.semantic_dim:
        raise ValueError("Memory dimension metadata inconsistent")
    if expected_dataset_name is not None and bank.metadata["dataset_name"] != expected_dataset_name:
        raise ValueError("Memory dataset_name does not match the requested dataset")
    if expected_num_classes is not None and bank.metadata["num_classes"] != int(expected_num_classes):
        raise ValueError("Memory num_classes does not match the requested class count")
    return bank
