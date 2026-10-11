from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from memmamba.model_spec import MODEL_SPEC
from memmamba.models.selective_scan import (
    SCAN_DIRECTIONS,
    describe_scan_backend,
    reverse_scan_2d,
    scan_2d,
    selective_scan,
)


DELTA_KNOWLEDGE_INJECTION_FACTOR = 0.1
DELTA_LOGIT_MIN = -20.0
DELTA_LOGIT_MAX = 20.0
DELTA_MIN_VALUE = 1e-4
DELTA_MAX_VALUE = 1.0


def _tensor_diagnostics(name: str, value: torch.Tensor) -> str:
    """Return a compact tensor summary for an exceptional numerical path."""
    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    finite_values = detached[finite_mask]
    finite_range = (
        f"finite_min={float(finite_values.min().item()):.8g} "
        f"finite_max={float(finite_values.max().item()):.8g}"
        if finite_count else "finite_min=none finite_max=none"
    )
    return (
        f"{name}[shape={tuple(detached.shape)} dtype={detached.dtype} "
        f"finite={finite_count}/{detached.numel()} "
        f"nan={int(torch.isnan(detached).sum().item())} "
        f"posinf={int(torch.isposinf(detached).sum().item())} "
        f"neginf={int(torch.isneginf(detached).sum().item())} {finite_range}]"
    )


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class KCMBottleneckOutput:
    enhanced_feature: torch.Tensor
    after_block1: torch.Tensor
    after_block2: torch.Tensor


class KnowledgeConditionedMamba(nn.Module):
    """One 2D Mamba block whose input and selective Δ/B/C are conditioned by Z."""

    def __init__(self, image_channels: int, knowledge_channels: int, block_index: int) -> None:
        super().__init__()
        spec = MODEL_SPEC
        d_model, d_state = spec.kcm_d_model, spec.kcm_d_state
        expand, conv_kernel = spec.kcm_expand, spec.kcm_conv_kernel
        alpha_init, layer_scale_init = spec.kcm_alpha_init, spec.kcm_layer_scale_init
        resolved_rank = math.ceil(d_model / 16)
        self.image_channels = image_channels
        self.knowledge_channels = knowledge_channels
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = resolved_rank
        self.scan_backend = spec.kcm_scan_backend
        self.last_scan_source = None
        self.block_index = block_index
        self._delta_logit_bound_warning_logged = False
        self._delta_ceiling_warning_logged = False

        self.image_norm = nn.GroupNorm(_group_count(image_channels), image_channels)
        self.knowledge_norm = nn.GroupNorm(_group_count(knowledge_channels), knowledge_channels)
        self.image_projection = nn.Conv2d(image_channels, d_model, 1)
        self.knowledge_projection = nn.Conv2d(knowledge_channels, d_model, 1)
        self.input_gate_x = nn.Conv2d(d_model, d_model, 1)
        self.input_gate_z = nn.Conv2d(d_model, d_model, 1)
        self.input_mixer = nn.Sequential(
            nn.Conv2d(d_model, d_model * expand, 1),
            nn.SiLU(),
            nn.Conv2d(d_model * expand, d_model, 1),
            nn.Conv2d(
                d_model, d_model, conv_kernel, padding=conv_kernel // 2,
                groups=d_model,
            ),
        )

        self.delta_x_rank = nn.Linear(d_model, resolved_rank, bias=False)
        self.delta_x_projection = nn.Linear(resolved_rank, d_model, bias=False)
        self.delta_z_rank = nn.Linear(d_model, resolved_rank, bias=False)
        self.delta_z_projection = nn.Linear(resolved_rank, d_model, bias=False)
        self.delta_gate_x = nn.Linear(d_model, d_model)
        self.delta_gate_z = nn.Linear(d_model, d_model)
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        )
        self.delta_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.b_x_projection = nn.Linear(d_model, d_state, bias=False)
        self.b_z_projection = nn.Linear(d_model, d_state, bias=False)
        self.b_gate_x = nn.Linear(d_model, d_state)
        self.b_gate_z = nn.Linear(d_model, d_state)
        self.c_x_projection = nn.Linear(d_model, d_state, bias=False)
        self.c_z_projection = nn.Linear(d_model, d_state, bias=False)
        self.c_gate_x = nn.Linear(d_model, d_state)
        self.c_gate_z = nn.Linear(d_model, d_state)

        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1)
        )
        self.d_skip = nn.Parameter(torch.ones(d_model))
        self.alpha_in = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_delta = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_b = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_c = nn.Parameter(torch.tensor(float(alpha_init)))
        self.output_projection = nn.Conv2d(d_model, image_channels, 1)
        self.layer_scale = nn.Parameter(
            torch.full((image_channels, 1, 1), float(layer_scale_init))
        )

    @staticmethod
    def _sequence_scale(
        knowledge_scale: float | torch.Tensor,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.as_tensor(knowledge_scale, device=reference.device, dtype=reference.dtype)
        if scale.ndim == 0:
            return scale
        if scale.ndim == 1 and scale.shape[0] == batch_size:
            return scale[:, None, None]
        raise ValueError("knowledge_scale must be a scalar or a [B] tensor")

    def _directional_scan(
        self,
        image_sequence: torch.Tensor,
        knowledge_sequence: torch.Tensor,
        knowledge_scale: float | torch.Tensor,
        direction: str,
    ) -> tuple[torch.Tensor, str]:
        # image_sequence/knowledge_sequence: [B,L,d_model]
        scale = self._sequence_scale(knowledge_scale, image_sequence.shape[0], image_sequence)
        delta_x = self.delta_x_projection(self.delta_x_rank(image_sequence))
        delta_z = self.delta_z_projection(self.delta_z_rank(knowledge_sequence))
        delta_gate = torch.sigmoid(
            self.delta_gate_x(image_sequence) + self.delta_gate_z(knowledge_sequence)
        )
        delta_term = (
            DELTA_KNOWLEDGE_INJECTION_FACTOR
            * scale * self.alpha_delta * delta_gate * delta_z
        )
        delta_term_tensor = delta_term
        raw_delta_logits = (
            delta_x.float()
            + delta_term_tensor.float()
            + self.delta_bias.float()
        )
        finite_logits = torch.isfinite(raw_delta_logits)
        if not bool(finite_logits.all()):
            reasons: list[str] = []
            counts = (
                ("delta_logits_nan", torch.isnan(raw_delta_logits)),
                ("delta_logits_posinf", torch.isposinf(raw_delta_logits)),
                ("delta_logits_neginf", torch.isneginf(raw_delta_logits)),
            )
            for reason, mask in counts:
                count = int(mask.sum().item())
                if count:
                    reasons.append(f"{reason}:{count}")
            scale_tensor = torch.as_tensor(
                knowledge_scale,
                device=raw_delta_logits.device,
                dtype=raw_delta_logits.dtype,
            )
            raise FloatingPointError(
                "knowledge-conditioned Delta is invalid "
                f"block={self.block_index} direction={direction} "
                f"reasons={','.join(reasons) or 'unknown'} "
                f"knowledge_injection_factor={DELTA_KNOWLEDGE_INJECTION_FACTOR:.8g}; "
                + "; ".join((
                    _tensor_diagnostics("knowledge_scale", scale_tensor),
                    _tensor_diagnostics("alpha_delta", self.alpha_delta),
                    _tensor_diagnostics("delta_x", delta_x),
                    _tensor_diagnostics("delta_z", delta_z),
                    _tensor_diagnostics("delta_gate", delta_gate),
                    _tensor_diagnostics("delta_term", delta_term_tensor),
                    _tensor_diagnostics("delta_bias", self.delta_bias),
                    _tensor_diagnostics("raw_delta_logits", raw_delta_logits),
                ))
            )
        bounded_delta_logits = raw_delta_logits.clamp(
            min=DELTA_LOGIT_MIN,
            max=DELTA_LOGIT_MAX,
        )
        if not self._delta_logit_bound_warning_logged:
            lower_logit_mask = raw_delta_logits < DELTA_LOGIT_MIN
            upper_logit_mask = raw_delta_logits > DELTA_LOGIT_MAX
            if bool((lower_logit_mask | upper_logit_mask).any()):
                lower_logit_count = int(lower_logit_mask.sum().item())
                upper_logit_count = int(upper_logit_mask.sum().item())
                logging.warning(
                    "KCM Delta logits bounded block=%d direction=%s "
                    "logit_lower_clipped=%d logit_upper_clipped=%d "
                    "logit_bounds=[%.8g,%.8g] knowledge_injection_factor=%.8g; "
                    "%s; %s; %s; %s; %s; %s",
                    self.block_index,
                    direction,
                    lower_logit_count,
                    upper_logit_count,
                    DELTA_LOGIT_MIN,
                    DELTA_LOGIT_MAX,
                    DELTA_KNOWLEDGE_INJECTION_FACTOR,
                    _tensor_diagnostics("alpha_delta", self.alpha_delta),
                    _tensor_diagnostics("delta_x", delta_x),
                    _tensor_diagnostics("delta_term", delta_term_tensor),
                    _tensor_diagnostics("delta_bias", self.delta_bias),
                    _tensor_diagnostics("raw_delta_logits", raw_delta_logits),
                    _tensor_diagnostics("bounded_delta_logits", bounded_delta_logits),
                )
                self._delta_logit_bound_warning_logged = True
        unbounded_delta = functional.softplus(bounded_delta_logits) + DELTA_MIN_VALUE
        if not self._delta_ceiling_warning_logged:
            ceiling_mask = unbounded_delta > DELTA_MAX_VALUE
            if bool(ceiling_mask.any()):
                ceiling_count = int(ceiling_mask.sum().item())
                logging.warning(
                    "KCM Delta ceiling applied block=%d direction=%s "
                    "delta_ceiling_clipped=%d delta_range=[%.8g,%.8g]; %s; %s",
                    self.block_index,
                    direction,
                    ceiling_count,
                    DELTA_MIN_VALUE,
                    DELTA_MAX_VALUE,
                    _tensor_diagnostics("raw_delta_logits", raw_delta_logits),
                    _tensor_diagnostics("unbounded_delta", unbounded_delta),
                )
                self._delta_ceiling_warning_logged = True
        delta = unbounded_delta.clamp_max(DELTA_MAX_VALUE)
        valid_delta = (
            torch.isfinite(delta)
            & (delta >= DELTA_MIN_VALUE)
            & (delta <= DELTA_MAX_VALUE)
        )
        if not bool(valid_delta.all()):
            raise FloatingPointError(
                "knowledge-conditioned Delta remained invalid after range protection "
                f"block={self.block_index} direction={direction}; "
                + _tensor_diagnostics("delta", delta)
            )

        b_x = self.b_x_projection(image_sequence)
        b_z = self.b_z_projection(knowledge_sequence)
        b_gate = torch.sigmoid(self.b_gate_x(image_sequence) + self.b_gate_z(knowledge_sequence))
        b_value = b_x + (
            scale * self.alpha_b * b_gate * b_z
        )
        c_x = self.c_x_projection(image_sequence)
        c_z = self.c_z_projection(knowledge_sequence)
        c_gate = torch.sigmoid(self.c_gate_x(image_sequence) + self.c_gate_z(knowledge_sequence))
        c_value = c_x + (
            scale * self.alpha_c * c_gate * c_z
        )
        a = -torch.exp(self.a_log.float())
        output, source = selective_scan(
            image_sequence.transpose(1, 2).contiguous(),
            delta.transpose(1, 2).contiguous(),
            a,
            b_value.transpose(1, 2).contiguous(),
            c_value.transpose(1, 2).contiguous(),
            self.d_skip,
            backend=self.scan_backend,
        )
        return output.transpose(1, 2).contiguous(), source

    def forward(
        self,
        image_feature: torch.Tensor,
        spatial_knowledge: torch.Tensor,
        knowledge_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if image_feature.ndim != 4 or image_feature.shape[1] != self.image_channels:
            raise ValueError("image_feature must have shape [B,image_channels,H,W]")
        if spatial_knowledge.ndim != 4 or spatial_knowledge.shape[:2] != (
            image_feature.shape[0], self.knowledge_channels
        ):
            raise ValueError("spatial_knowledge must have shape [B,knowledge_channels,H,W]")
        if spatial_knowledge.shape[-2:] != image_feature.shape[-2:]:
            spatial_knowledge = functional.interpolate(
                spatial_knowledge, size=image_feature.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        if not torch.isfinite(image_feature).all() or not torch.isfinite(spatial_knowledge).all():
            raise FloatingPointError("KCM inputs contain NaN or Inf")
        image_m = self.image_projection(self.image_norm(image_feature))
        knowledge_m = self.knowledge_projection(self.knowledge_norm(spatial_knowledge))
        input_gate = torch.sigmoid(self.input_gate_x(image_m) + self.input_gate_z(knowledge_m))
        spatial_scale = self._sequence_scale(
            knowledge_scale, image_feature.shape[0], image_feature
        )
        if spatial_scale.ndim == 3:
            spatial_scale = spatial_scale.unsqueeze(-1)
        mixed = image_m + spatial_scale * self.alpha_in * input_gate * knowledge_m
        mixed = mixed + self.input_mixer(mixed)

        height, width = mixed.shape[-2:]
        restored: list[torch.Tensor] = []
        sources: set[str] = set()
        for direction in SCAN_DIRECTIONS:
            image_sequence = scan_2d(mixed, direction)
            knowledge_sequence = scan_2d(knowledge_m, direction)
            sequence_output, source = self._directional_scan(
                image_sequence, knowledge_sequence, knowledge_scale,
                direction,
            )
            restored.append(reverse_scan_2d(sequence_output, direction, height, width))
            sources.add(source)
        if len(sources) != 1:
            raise RuntimeError("one KCM block used inconsistent selective-scan backends")
        stacked = torch.stack(restored, dim=0)
        merged = stacked.mean(0)
        projected = self.output_projection(merged)
        enhanced = image_feature + (projected * self.layer_scale)
        if enhanced.shape != image_feature.shape or not torch.isfinite(enhanced).all():
            raise FloatingPointError("KCM output is non-finite or shape-incompatible")
        self.last_scan_source = next(iter(sources))
        return enhanced


class KCMBottleneck(nn.Module):
    """Two independent KCM blocks sharing one spatial knowledge tensor."""

    def __init__(self, image_channels: int, knowledge_channels: int) -> None:
        super().__init__()
        self.knowledge_dropout = MODEL_SPEC.knowledge_dropout
        self._scan_source_logged = False
        self.blocks = nn.ModuleList([
            KnowledgeConditionedMamba(image_channels, knowledge_channels, block_index=index + 1)
            for index in range(MODEL_SPEC.kcm_num_blocks)
        ])

    @property
    def scan_description(self) -> str:
        return describe_scan_backend(self.blocks[0].scan_backend)

    def _drop_knowledge(self, knowledge: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return knowledge
        keep = knowledge.new_empty((knowledge.shape[0], 1, 1, 1)).bernoulli_(
            1.0 - self.knowledge_dropout
        )
        return knowledge * keep

    def forward(
        self,
        image_feature: torch.Tensor,
        spatial_knowledge: torch.Tensor,
        knowledge_scale: float | torch.Tensor = 1.0,
    ) -> KCMBottleneckOutput:
        # One sample-wise dropout mask is applied once; both blocks receive this same Z.
        shared_knowledge = self._drop_knowledge(spatial_knowledge)
        current = image_feature
        intermediates: list[torch.Tensor] = []
        for block in self.blocks:
            current = block(
                current, shared_knowledge, knowledge_scale
            )
            intermediates.append(current)
        if not self._scan_source_logged:
            logging.info(
                "KCM selective_scan_source=%s blocks=%d shared_spatial_knowledge=true",
                self.blocks[0].last_scan_source, len(self.blocks),
            )
            self._scan_source_logged = True
        return KCMBottleneckOutput(
            current,
            intermediates[0],
            intermediates[1],
        )
