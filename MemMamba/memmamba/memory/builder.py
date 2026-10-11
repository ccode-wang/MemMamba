from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from memmamba.config import DATASET_NAME, MODALITIES, foreground_classes
from memmamba.memory.features import SemanticFeatureBank, VisualFeatureBank
from memmamba.memory.storage import MultimodalMedicalKnowledgeMemory

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryBuildOptions:
    prototypes_per_class: int = 8
    seed: int = 2026
    max_iter: int = 100
    tolerance: float = 1e-4

    def validate(self):
        if self.prototypes_per_class <= 0 or self.max_iter <= 0 or self.tolerance <= 0:
            raise ValueError("K-means options must be positive")


def _force_nonempty(assignment, similarity, k: int):
    import torch

    counts = torch.bincount(assignment, minlength=k)
    for empty in torch.nonzero(counts == 0, as_tuple=False).flatten().tolist():
        donors = torch.nonzero(counts[assignment] > 1, as_tuple=False).flatten()
        if donors.numel() == 0:
            raise RuntimeError("cannot repair empty spherical cluster")
        errors = 1.0 - similarity[donors, assignment[donors]]
        donor = int(donors[errors.argmax()].item())
        old = int(assignment[donor].item())
        assignment[donor] = int(empty)
        counts[old] -= 1
        counts[empty] += 1
    return assignment


def _centers_from_assignment(x, assignment, k: int):
    import torch
    import torch.nn.functional as functional

    centers = []
    for cluster in range(k):
        members = x[assignment == cluster]
        if members.shape[0] == 0:
            raise RuntimeError("empty cluster after repair")
        mean = members.mean(dim=0)
        centers.append(members[0] if float(mean.norm()) <= 1e-12 else functional.normalize(mean, dim=0))
    return torch.stack(centers)


def spherical_kmeans(x, k: int, seed: int, max_iter: int, tolerance: float):
    import torch
    import torch.nn.functional as functional

    if x.ndim != 2 or x.shape[0] == 0 or not torch.isfinite(x).all() or torch.any(x.norm(dim=-1) <= 0):
        raise ValueError("input must be finite nonzero [N,D]")
    if not 1 <= k <= x.shape[0] or max_iter <= 0 or tolerance <= 0:
        raise ValueError("invalid K-means parameters")
    x = functional.normalize(x.float(), dim=-1)
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    chosen = [int(torch.randint(x.shape[0], (1,), generator=generator, device=x.device))]
    while len(chosen) < k:
        distance = 1.0 - (x @ x[chosen].T).max(dim=1).values
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    centers = x[chosen].clone()
    assignment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(max_iter):
        similarity = x @ centers.T
        assignment = _force_nonempty(similarity.argmax(dim=1), similarity, k)
        new_centers = _centers_from_assignment(x, assignment, k)
        shift = (1.0 - (centers * new_centers).sum(dim=1)).abs().max()
        centers = new_centers
        if float(shift) <= tolerance:
            break
    similarity = x @ centers.T
    assignment = _force_nonempty(similarity.argmax(dim=1), similarity, k)
    return functional.normalize(_centers_from_assignment(x, assignment, k), dim=-1), assignment


def build_multimodal_memory(
    visual: VisualFeatureBank,
    semantic: SemanticFeatureBank,
    options: MemoryBuildOptions,
) -> MultimodalMedicalKnowledgeMemory:
    import torch
    import torch.nn.functional as functional

    options.validate()
    visual.validate()
    semantic.validate()
    if visual.classes != semantic.classes:
        raise ValueError("visual and semantic class configurations differ")
    if visual.dataset_name != DATASET_NAME or list(visual.modality) != MODALITIES or visual.split != "train":
        raise ValueError("Memory may only use Task05_Prostate T2/ADC training features")
    if semantic.dataset_name != DATASET_NAME or list(semantic.modality) != MODALITIES:
        raise ValueError("Memory requires Task05_Prostate T2/ADC semantic features")
    semantics = {int(cid): semantic.embeddings[i] for i, cid in enumerate(semantic.class_ids.tolist())}
    out = {name: [] for name in ("key", "interior", "boundary", "context", "semantic", "position", "size")}
    ids, names = [], []
    positions = torch.tensor([record.normalized_slice_position for record in visual.records])
    for info in foreground_classes(visual.classes):
        indices = torch.nonzero(visual.class_ids == info.id, as_tuple=False).flatten()
        actual = min(options.prototypes_per_class, len(indices))
        if actual == 0:
            raise ValueError(f"class {info.id} has no visual samples")
        if actual < options.prototypes_per_class:
            LOGGER.warning("class=%d %s requested=%d actual=%d", info.id, info.name, options.prototypes_per_class, actual)
        centers, assignment = spherical_kmeans(
            visual.visual_key[indices], actual, options.seed + info.id, options.max_iter, options.tolerance
        )
        for cluster in range(actual):
            members = indices[assignment == cluster]
            out["key"].append(centers[cluster])
            for name in ("interior", "boundary", "context"):
                values = getattr(visual, name)[members]
                mean = values.mean(0)
                out[name].append(values[0] if float(mean.norm()) <= 1e-12 else functional.normalize(mean, dim=0))
            out["semantic"].append(semantics[info.id])
            out["position"].append(positions[members].mean())
            out["size"].append(torch.tensor(len(members)))
            ids.append(info.id)
            names.append(info.name)
    configured = foreground_classes(visual.classes)
    metadata = {
        "dataset_name": DATASET_NAME,
        "modality": list(MODALITIES),
        "anatomy": "prostate",
        "num_classes": len(visual.classes),
        "foreground_class_ids": [item.id for item in configured],
    }
    bank = MultimodalMedicalKnowledgeMemory(
        torch.tensor(ids, dtype=torch.long), names, torch.stack(out["key"]),
        torch.stack(out["interior"]), torch.stack(out["boundary"]),
        torch.stack(out["context"]), torch.stack(out["semantic"]),
        torch.stack(out["position"]).float(), torch.stack(out["size"]).long(),
        visual.classes, asdict(options), metadata,
    )
    bank.validate()
    return bank
