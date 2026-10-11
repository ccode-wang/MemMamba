from __future__ import annotations

import hashlib
from pathlib import Path


def file_identity(path: str | Path) -> dict[str, str | int]:
    """Return stable artifact identity for checkpoint/resume compatibility."""

    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(artifact), "size": artifact.stat().st_size, "sha256": digest.hexdigest()}


def build_artifact_metadata(model, config: dict, include_alignment: bool) -> dict:
    """Record the memory and optional alignment supervision used for a run."""
    return {
        "memory": file_identity(config["memory_bank_path"]),
        "alignment_teachers": file_identity(config["alignment_teacher_path"])
        if include_alignment else None,
        "model_name": config["model_name"],
        "dataset_name": config["dataset_name"],
        "num_classes": config["num_classes"],
        "memory_visual_dim": model.memory_bank.visual_dim,
        "memory_semantic_dim": model.memory_bank.semantic_dim,
    }
