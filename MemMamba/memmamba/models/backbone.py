"""Convolutional encoder, skip connections, decoder, and retrieval heads."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
import torch.nn.functional as functional

from memmamba.model_spec import MODEL_SPEC


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        mid = out_channels if mid_channels is None else mid_channels
        super().__init__(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, low: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        low = self.up(low)
        dy, dx = skip.size(2) - low.size(2), skip.size(3) - low.size(3)
        low = functional.pad(low, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, low], dim=1))


@dataclass(frozen=True)
class EncoderFeatures:
    level1: torch.Tensor
    level2: torch.Tensor
    level3: torch.Tensor
    level4: torch.Tensor
    deep_feature: torch.Tensor

    def with_deep_feature(self, feature: torch.Tensor) -> "EncoderFeatures":
        return replace(self, deep_feature=feature)


@dataclass(frozen=True)
class RetrievalHeadsOutput:
    coarse_logits: torch.Tensor
    presence_logits: torch.Tensor


def generate_class_queries(
    feature: torch.Tensor, coarse_logits: torch.Tensor, epsilon: float = 1e-6,
) -> torch.Tensor:
    """Class Query Generation: pool deep features with coarse foreground probabilities."""
    if (
        feature.ndim != 4 or coarse_logits.ndim != 4
        or coarse_logits.shape[0] != feature.shape[0] or coarse_logits.shape[1] < 2
    ):
        raise ValueError("feature/coarse_logits must be [B,C,H,W] with foreground classes")
    probabilities = functional.softmax(coarse_logits, dim=1)[:, 1:]
    probabilities = functional.interpolate(
        probabilities, size=feature.shape[-2:], mode="bilinear", align_corners=False,
    )
    numerator = torch.einsum("bkhw,bdhw->bkd", probabilities, feature)
    denominator = probabilities.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    return numerator / (denominator + epsilon)


class EncoderDecoderBackbone(nn.Module):
    """MemMamba's fixed convolutional backbone; the deep feature is enhanced by KCM."""

    def __init__(self):
        super().__init__()
        spec = MODEL_SPEC
        base = spec.base_channels
        # Preserve layer construction order and initialization of the original model.
        self.in_conv = DoubleConv(spec.in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.bottleneck_channels = base * 8
        self.down4 = Down(base * 8, self.bottleneck_channels)
        self.up1 = Up(base * 16, base * 4)
        self.up2 = Up(base * 8, base * 2)
        self.up3 = Up(base * 4, base)
        self.up4 = Up(base * 2, base)
        self.segmentation_head = nn.Conv2d(base, spec.num_classes, 1)
        self.coarse_segmentation_head = nn.Conv2d(self.bottleneck_channels, spec.num_classes, 1)
        self.presence_pool = nn.AdaptiveAvgPool2d(1)
        self.class_presence_head = nn.Linear(self.bottleneck_channels, spec.num_foreground_classes)

    def encode(self, image: torch.Tensor) -> EncoderFeatures:
        level1 = self.in_conv(image)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        return EncoderFeatures(level1, level2, level3, level4, self.down4(level4))

    def predict_heads(self, features: EncoderFeatures, output_size) -> RetrievalHeadsOutput:
        coarse = functional.interpolate(
            self.coarse_segmentation_head(features.deep_feature), size=output_size,
            mode="bilinear", align_corners=False,
        )
        presence = self.class_presence_head(self.presence_pool(features.deep_feature).flatten(1))
        return RetrievalHeadsOutput(coarse, presence)

    def decode(self, features: EncoderFeatures) -> torch.Tensor:
        decoded = self.up1(features.deep_feature, features.level4)
        decoded = self.up2(decoded, features.level3)
        decoded = self.up3(decoded, features.level2)
        decoded = self.up4(decoded, features.level1)
        return self.segmentation_head(decoded)
