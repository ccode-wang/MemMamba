from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch


SCAN_DIRECTIONS = ("right", "left", "down", "up")
SCAN_BACKENDS = ("auto", "mamba_ssm", "reference")


def scan_2d(feature: torch.Tensor, direction: str) -> torch.Tensor:
    """Serialize ``[B,C,H,W]`` into aligned ``[B,H*W,C]`` tokens."""
    if not torch.is_tensor(feature) or feature.ndim != 4:
        raise ValueError("feature must be a [B,C,H,W] tensor")
    if direction not in SCAN_DIRECTIONS:
        raise ValueError(f"direction must be one of {SCAN_DIRECTIONS}")
    if direction == "right":
        ordered = feature
    elif direction == "left":
        ordered = feature.flip(-1)
    elif direction == "down":
        ordered = feature.transpose(-2, -1).contiguous()
    else:
        ordered = feature.flip(-2).transpose(-2, -1).contiguous()
    return ordered.flatten(2).transpose(1, 2).contiguous()


def reverse_scan_2d(
    sequence: torch.Tensor,
    direction: str,
    height: int,
    width: int,
) -> torch.Tensor:
    """Restore a directional ``[B,H*W,C]`` sequence to ``[B,C,H,W]``."""
    if not torch.is_tensor(sequence) or sequence.ndim != 3:
        raise ValueError("sequence must be a [B,L,C] tensor")
    if direction not in SCAN_DIRECTIONS:
        raise ValueError(f"direction must be one of {SCAN_DIRECTIONS}")
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")
    if sequence.shape[1] != height * width:
        raise ValueError("sequence length must equal height * width")
    channels_first = sequence.transpose(1, 2).contiguous()
    if direction == "right":
        return channels_first.reshape(sequence.shape[0], sequence.shape[2], height, width)
    if direction == "left":
        return channels_first.reshape(sequence.shape[0], sequence.shape[2], height, width).flip(-1)
    restored = channels_first.reshape(sequence.shape[0], sequence.shape[2], width, height)
    restored = restored.transpose(-2, -1).contiguous()
    return restored if direction == "down" else restored.flip(-2)


@lru_cache(maxsize=1)
def _mamba_scan() -> Callable[..., torch.Tensor] | None:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except (ImportError, OSError):
        return None
    return selective_scan_fn


def describe_scan_backend(preference: str) -> str:
    if preference not in SCAN_BACKENDS:
        raise ValueError(f"scan backend must be one of {SCAN_BACKENDS}")
    if preference == "reference":
        return "pytorch_reference"
    if preference == "mamba_ssm":
        return "mamba_ssm_required"
    return "auto:mamba_ssm_cuda_else_pytorch_reference"


def _validate_scan_inputs(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> None:
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must have the same [B,D,L] shape")
    if a.ndim != 2 or a.shape[0] != u.shape[1]:
        raise ValueError("A must have shape [D,N]")
    expected_bc = (u.shape[0], a.shape[1], u.shape[2])
    if b.shape != expected_bc or c.shape != expected_bc:
        raise ValueError("B and C must have shape [B,N,L]")
    if d.shape != (u.shape[1],):
        raise ValueError("D must have shape [D]")
    if any(value.device != u.device for value in (delta, a, b, c, d)):
        raise ValueError("all selective-scan tensors must be on the same device")
    if not all(value.is_floating_point() for value in (u, delta, a, b, c, d)):
        raise ValueError("selective-scan tensors must be floating point")
    if not all(torch.isfinite(value).all() for value in (u, delta, a, b, c, d)):
        raise FloatingPointError("selective-scan inputs contain NaN or Inf")
    if bool((delta <= 0).any()):
        raise FloatingPointError("selective-scan Delta must be positive")


def reference_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Mamba recurrence used for CPU checks and as an explicit fallback.

    Inputs follow the public ``mamba_ssm.selective_scan_fn`` layout. Computation
    is performed in float32 for AMP stability, without an explicit matrix inverse.
    """
    _validate_scan_inputs(u, delta, a, b, c, d)
    output_dtype = u.dtype
    u32, delta32, a32, b32, c32, d32 = (
        value.float() for value in (u, delta, a, b, c, d)
    )
    state = torch.zeros(
        u.shape[0], u.shape[1], a.shape[1], device=u.device, dtype=torch.float32
    )
    outputs: list[torch.Tensor] = []
    for index in range(u.shape[2]):
        dt = delta32[:, :, index].unsqueeze(-1)
        transition = torch.exp(dt * a32.unsqueeze(0))
        state_input = dt * b32[:, :, index].unsqueeze(1) * u32[:, :, index].unsqueeze(-1)
        state = transition * state + state_input
        current = (state * c32[:, :, index].unsqueeze(1)).sum(-1)
        outputs.append(current + d32.unsqueeze(0) * u32[:, :, index])
    result = torch.stack(outputs, dim=-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("PyTorch reference selective scan produced NaN or Inf")
    return result.to(dtype=output_dtype)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    backend: str = "auto",
) -> tuple[torch.Tensor, str]:
    """Run a real selective scan with externally supplied knowledge-conditioned Δ/B/C."""
    _validate_scan_inputs(u, delta, a, b, c, d)
    if backend not in SCAN_BACKENDS:
        raise ValueError(f"scan backend must be one of {SCAN_BACKENDS}")
    function = _mamba_scan()
    use_kernel = backend == "mamba_ssm" or (backend == "auto" and u.is_cuda and function is not None)
    if use_kernel:
        if function is None:
            raise RuntimeError(
                "kcm_scan_backend='mamba_ssm' requires a compatible mamba_ssm "
                "installation exposing mamba_ssm.ops.selective_scan_interface.selective_scan_fn"
            )
        output_dtype = u.dtype
        try:
            result = function(
                u.float(), delta.float(), a.float(), b.float(), c.float(),
                D=d.float(), z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False,
            )
        except TypeError as exc:
            raise RuntimeError(
                "The installed mamba_ssm selective_scan_fn API cannot accept externally "
                "provided Δ/B/C; use a compatible mamba_ssm installation"
            ) from exc
        if isinstance(result, tuple):
            result = result[0]
        result = result.to(dtype=output_dtype)
        source = "mamba_ssm"
    else:
        result = reference_selective_scan(u, delta, a, b, c, d)
        source = "pytorch_reference"
    if result.shape != u.shape:
        raise RuntimeError("selective scan returned an incompatible output shape")
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"{source} selective scan produced NaN or Inf")
    return result, source
