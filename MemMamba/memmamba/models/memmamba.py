"""MemMamba: memory retrieval followed by two Knowledge-Conditioned Mamba blocks."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from torch import nn

from memmamba.config import DATASET_NAME
from memmamba.memory.storage import MultimodalMedicalKnowledgeMemory
from memmamba.model_spec import MODEL_SPEC
from memmamba.models.backbone import EncoderDecoderBackbone
from memmamba.models.knowledge_fusion import ClassAwareDynamicKnowledgeRetrieval
from memmamba.models.kcm import KCMBottleneck


@dataclass
class MemMambaOutput:
    final_logits: torch.Tensor
    coarse_logits: torch.Tensor
    presence_logits: torch.Tensor
    class_query: torch.Tensor
    projected_query: torch.Tensor
    retrieved_knowledge: torch.Tensor
    final_knowledge: torch.Tensor
    spatial_knowledge: torch.Tensor
    region_weights: torch.Tensor
    reliability_gate: torch.Tensor
    bottleneck_before_kcm: torch.Tensor | None = None
    bottleneck_after_block1: torch.Tensor | None = None
    bottleneck_after_block2: torch.Tensor | None = None


class MemMamba(nn.Module):
    """Task05_Prostate model with one retrieval and two KCM blocks sharing spatial Z.

    Architecture settings live in ``memmamba.model_spec.MODEL_SPEC``. The memory
    is an external, frozen artifact and is not included in the model state dict.
    """

    def __init__(self, memory_bank: MultimodalMedicalKnowledgeMemory) -> None:
        super().__init__()
        memory_bank.validate()
        if len(memory_bank.classes) != MODEL_SPEC.num_classes:
            raise ValueError("Memory classes differ from the MemMamba model specification")
        if memory_bank.metadata["dataset_name"] != DATASET_NAME:
            raise ValueError("MemMamba requires the Task05_Prostate memory bank")
        self.memory_bank = memory_bank
        self.backbone = EncoderDecoderBackbone()
        self.knowledge_retrieval = ClassAwareDynamicKnowledgeRetrieval(
            self.backbone.bottleneck_channels, memory_bank.visual_dim, memory_bank.semantic_dim,
        )
        self.kcm_bottleneck = KCMBottleneck(
            self.backbone.bottleneck_channels, MODEL_SPEC.knowledge_dim,
        )
        logging.info(
            "MemMamba KCM_blocks=2 retrievals_per_forward=1 shared_spatial_knowledge=true "
            "modulate_delta=true modulate_B=true modulate_C=true modulate_A=false scan=%s",
            self.kcm_bottleneck.scan_description,
        )

    def forward(
        self,
        image: torch.Tensor,
        knowledge_scale: float | torch.Tensor = 1.0,
        return_intermediates: bool = False,
    ) -> MemMambaOutput:
        if image.ndim != 4 or image.shape[1] != MODEL_SPEC.in_channels:
            raise ValueError("image must be [B,2,H,W] with channels [T2,ADC]")
        features = self.backbone.encode(image)
        heads = self.backbone.predict_heads(features, image.shape[-2:])
        retrieved = self.knowledge_retrieval(
            features.deep_feature, heads.coarse_logits, heads.presence_logits, self.memory_bank,
        )
        # X0 -> KCM -> X1 -> KCM -> X2; both blocks receive the same spatial Z.
        kcm = self.kcm_bottleneck(features.deep_feature, retrieved.spatial_knowledge, knowledge_scale)
        final = self.backbone.decode(features.with_deep_feature(kcm.enhanced_feature))
        return MemMambaOutput(
            final_logits=final,
            coarse_logits=heads.coarse_logits,
            presence_logits=heads.presence_logits,
            class_query=retrieved.class_query,
            projected_query=retrieved.projected_query,
            retrieved_knowledge=retrieved.fused.retrieved_knowledge,
            final_knowledge=retrieved.fused.final_knowledge,
            spatial_knowledge=retrieved.spatial_knowledge,
            region_weights=retrieved.fused.region_weights,
            reliability_gate=retrieved.fused.reliability_gate,
            bottleneck_before_kcm=features.deep_feature if return_intermediates else None,
            bottleneck_after_block1=kcm.after_block1 if return_intermediates else None,
            bottleneck_after_block2=kcm.after_block2 if return_intermediates else None,
        )
