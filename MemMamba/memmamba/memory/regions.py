from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionMasks:
    interior: np.ndarray
    boundary: np.ndarray
    context: np.ndarray
    interior_fallback: bool


def disk_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius < 0:
        raise ValueError("morphology radius must be non-negative")
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return (xx * xx + yy * yy <= radius * radius).astype(bool)


def build_region_masks(mask: np.ndarray, erosion_radius: int, dilation_radius: int) -> RegionMasks:
    erosion_radius = int(erosion_radius)
    dilation_radius = int(dilation_radius)
    if erosion_radius < 0 or dilation_radius < 0:
        raise ValueError("erosion_radius and dilation_radius must be non-negative")
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
    except ImportError as exc:
        raise ImportError("Region morphology requires scipy") from exc
    organ = np.asarray(mask, dtype=bool)
    if organ.ndim != 2 or not organ.any():
        raise ValueError("organ mask must be a non-empty 2D array")
    eroded = binary_erosion(organ, structure=disk_structure(erosion_radius)) if erosion_radius else organ.copy()
    fallback = not bool(eroded.any())
    interior = organ.copy() if fallback else eroded
    # Use a one-pixel morphological gradient for a stable inner/outer band.
    # The class-specific erosion remains reserved for the interior region.
    boundary_eroded = binary_erosion(organ, structure=disk_structure(1))
    boundary_dilated = binary_dilation(organ, structure=disk_structure(1))
    boundary = boundary_dilated & ~boundary_eroded
    dilated = binary_dilation(organ, structure=disk_structure(dilation_radius)) if dilation_radius else organ.copy()
    context = dilated & ~organ
    return RegionMasks(interior, boundary, context, fallback)


def resize_mask_to_patch_grid(mask, grid: tuple[int, int]):
    """Map a binary image mask to a binary patch-occupancy mask.

    Adaptive max pooling preserves a patch whenever any source pixel belongs to
    the region. The threshold keeps the downstream pooling strictly binary.
    """

    import torch
    import torch.nn.functional as functional

    height, width = int(grid[0]), int(grid[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"patch grid must be positive, got {grid}")
    # PIL-backed arrays may be read-only.  Copy before torch conversion so
    # PyTorch never shares storage with a non-writable NumPy view.
    writable = np.array(mask, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(writable)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    elif tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"mask must be [H,W], [B,H,W], or [B,1,H,W], got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all() or torch.any(tensor < 0):
        raise ValueError("patch mask weights must be finite and non-negative")
    pooled = functional.adaptive_max_pool2d(
        tensor, output_size=(height, width)
    )
    return (pooled > 0).to(dtype=tensor.dtype)


def pool_patch_region(tokens, mask):
    import torch
    import torch.nn.functional as functional

    if tokens.ndim != 3:
        raise ValueError(f"patch_tokens must be [B,N,D], got {tuple(tokens.shape)}")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"patch mask must be [B,1,Hp,Wp], got {tuple(mask.shape)}")
    batch, count, _ = tokens.shape
    if mask.shape[0] not in (1, batch):
        raise ValueError("patch mask batch dimension does not match tokens")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1)
    if mask.shape[-2] * mask.shape[-1] != count:
        raise ValueError(f"patch mask grid {tuple(mask.shape[-2:])} does not match token count N={count}")
    weights = mask[:, 0].reshape(batch, count).to(device=tokens.device, dtype=tokens.dtype)
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("patch mask weights must be finite and non-negative")
    effective_counts = (weights > 0).sum(dim=1).long()
    weight_sums = weights.sum(dim=1, keepdim=True)
    if torch.any(weight_sums <= 0):
        raise ValueError("region has zero effective patches")
    pooled = torch.einsum("bn,bnd->bd", weights, tokens) / weight_sums
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled region feature contains non-finite values")
    if torch.any(pooled.norm(dim=-1) <= 0):
        raise ValueError("pooled region feature has zero norm")
    return functional.normalize(pooled, dim=-1), effective_counts
