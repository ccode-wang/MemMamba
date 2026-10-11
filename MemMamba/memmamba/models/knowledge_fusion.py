"""Class-Aware Dynamic Knowledge Retrieval and Knowledge to Spatial Mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as functional

from memmamba.memory.retrieval import ClasswiseRetrievalResult, classwise_topk
from memmamba.memory.storage import MultimodalMedicalKnowledgeMemory
from memmamba.model_spec import MODEL_SPEC
from memmamba.models.backbone import generate_class_queries


@dataclass
class RegionFusionOutput:
    final_knowledge: torch.Tensor
    retrieved_knowledge: torch.Tensor
    region_weights: torch.Tensor
    topk_weights: torch.Tensor
    reliability_gate: torch.Tensor


class RegionKnowledgeFusion(nn.Module):
    """Fuse regional prototypes and semantics into presence-weighted class knowledge."""

    def __init__(self, visual_dim: int, semantic_dim: int, knowledge_dim: int):
        super().__init__()
        self.knowledge_dim = knowledge_dim
        self.interior_proj = nn.Linear(visual_dim, knowledge_dim)
        self.boundary_proj = nn.Linear(visual_dim, knowledge_dim)
        self.context_proj = nn.Linear(visual_dim, knowledge_dim)
        self.region_query = nn.Linear(visual_dim, knowledge_dim)
        self.region_score = nn.Linear(knowledge_dim, 1, bias=False)
        self.semantic_proj = nn.Linear(semantic_dim, knowledge_dim)
        self.attention_query = nn.Linear(visual_dim, knowledge_dim)
        self.attention_key = nn.Linear(knowledge_dim, knowledge_dim)
        self.attention_value = nn.Linear(knowledge_dim, knowledge_dim)
        self.gate_query = nn.Linear(visual_dim, knowledge_dim)
        self.gate_knowledge = nn.Linear(knowledge_dim, knowledge_dim)
        self.gate_presence = nn.Linear(1, knowledge_dim)

    def forward(self, query, retrieval: ClasswiseRetrievalResult, presence_probability):
        if query.ndim != 3 or presence_probability.shape != query.shape[:2]:
            raise ValueError("query/presence shapes invalid")
        regions = torch.stack([
            self.interior_proj(retrieval.interior),
            self.boundary_proj(retrieval.boundary),
            self.context_proj(retrieval.context),
        ], -2)
        region_weights = functional.softmax(
            self.region_score(torch.tanh(
                regions + self.region_query(query)[:, :, None, None, :]
            )).squeeze(-1), -1,
        )
        visual = (region_weights[..., None] * regions).sum(-2)
        paired = visual + self.semantic_proj(retrieval.semantic)
        logits = (
            self.attention_query(query)[:, :, None, :] * self.attention_key(paired)
        ).sum(-1) / math.sqrt(self.knowledge_dim)
        valid = retrieval.valid_mask
        logits = torch.where(valid, logits, torch.zeros_like(logits))
        weights = functional.softmax(logits, -1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        retrieved = (weights[..., None] * self.attention_value(paired)).sum(-2)
        gate = torch.sigmoid(
            self.gate_query(query) + self.gate_knowledge(retrieved)
            + self.gate_presence(presence_probability[..., None])
        )
        return RegionFusionOutput(
            presence_probability[..., None] * gate * retrieved,
            retrieved, region_weights, weights, gate,
        )


class KnowledgeToSpatialMapping(nn.Module):
    """Map class knowledge to the deep-feature grid to obtain spatial knowledge Z."""

    def forward(self, foreground_probability, final_knowledge, output_size):
        if foreground_probability.ndim != 4 or final_knowledge.ndim != 3:
            raise ValueError("foreground_probability and final_knowledge must be [B,K,H,W] and [B,K,D]")
        if foreground_probability.shape[:2] != final_knowledge.shape[:2]:
            raise ValueError("foreground probabilities and class knowledge must share [B,K]")
        probability = functional.interpolate(
            foreground_probability, size=output_size, mode="bilinear", align_corners=False,
        )
        return torch.einsum("bkhw,bkd->bdhw", probability, final_knowledge)


@dataclass
class DynamicRetrievalOutput:
    class_query: torch.Tensor
    projected_query: torch.Tensor
    fused: RegionFusionOutput
    spatial_knowledge: torch.Tensor


class ClassAwareDynamicKnowledgeRetrieval(nn.Module):
    """Generate class queries, retrieve top-k prototypes per class, and construct Z."""

    def __init__(self, feature_channels: int, visual_dim: int, semantic_dim: int):
        super().__init__()
        self.query_projector = nn.Linear(feature_channels, visual_dim)
        self.region_fusion = RegionKnowledgeFusion(visual_dim, semantic_dim, MODEL_SPEC.knowledge_dim)
        self.knowledge_to_spatial_mapping = KnowledgeToSpatialMapping()

    def forward(
        self,
        deep_feature: torch.Tensor,
        coarse_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        memory_bank: MultimodalMedicalKnowledgeMemory,
    ) -> DynamicRetrievalOutput:
        query = generate_class_queries(deep_feature, coarse_logits)
        projected = self.query_projector(query)
        presence = torch.sigmoid(presence_logits)
        # Train with all foreground classes; threshold predicted presence at evaluation.
        active_mask = None if self.training else presence >= MODEL_SPEC.presence_threshold
        retrieval = classwise_topk(projected, memory_bank, MODEL_SPEC.retrieval_top_k, active_mask)
        fused = self.region_fusion(projected, retrieval, presence)
        foreground_probability = functional.softmax(coarse_logits, dim=1)[:, 1:]
        spatial = self.knowledge_to_spatial_mapping(
            foreground_probability, fused.final_knowledge, deep_feature.shape[-2:],
        )
        return DynamicRetrievalOutput(query, projected, fused, spatial)
