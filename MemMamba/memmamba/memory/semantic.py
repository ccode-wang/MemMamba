from __future__ import annotations

from pathlib import Path
from typing import Mapping

from memmamba.config import ClassInfo, foreground_classes
from memmamba.memory.features import SemanticFeatureBank


def render_organ_description(record: Mapping[str, str]) -> str:
    """Render structured organ fields through one deterministic medical template."""

    return (
        f"Organ name: {record['organ_name']}. "
        f"Anatomical location: {record['anatomical_location']} "
        f"Morphology: {record['morphology']} "
        f"Boundary characteristics: {record['boundary']} "
        f"Neighboring relations: {record['neighboring_relations']} "
        f"Typical T2-weighted and ADC MRI appearance: {record['typical_appearance']}"
    )


def extract_semantic_feature_bank(
    descriptions: Mapping[int, Mapping[str, str]],
    encoder,
    classes: tuple[ClassInfo, ...],
    description_path: str | Path = "configs/task05_prostate_organ_descriptions.json",
) -> SemanticFeatureBank:
    import torch
    import torch.nn.functional as functional

    organs = foreground_classes(classes)
    expected = [item.id for item in organs]
    if sorted(descriptions) != expected:
        raise ValueError(f"description class ids must be {expected}, got {sorted(descriptions)}")
    texts = [render_organ_description(descriptions[item.id]) for item in organs]
    embeddings = encoder.encode_texts(texts)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(organs):
        raise ValueError(
            f"BiomedCLIP semantic embeddings must be [C-1,D], got {tuple(embeddings.shape)}"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError("BiomedCLIP semantic embeddings contain non-finite values")
    if torch.any(embeddings.norm(dim=-1) <= 0):
        raise ValueError("BiomedCLIP semantic embeddings contain zero-norm rows")
    bank = SemanticFeatureBank(
        class_ids=torch.tensor(expected, dtype=torch.long),
        embeddings=functional.normalize(embeddings.float(), dim=-1).detach().cpu(),
        class_names=[item.name for item in organs],
        rendered_texts=texts,
        classes=classes,
        model_name=str(getattr(encoder, "model_name", "unknown")),
        description_path=str(Path(description_path).resolve()),
    )
    bank.validate()
    return bank
