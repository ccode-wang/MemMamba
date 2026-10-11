from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def configure_offline_hf_cache(cache_dir: str | Path | None) -> None:
    """Expose one flat Hugging Face cache to open_clip and its Transformers text tower."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if cache_dir is None:
        return
    cache_path = str(Path(cache_dir).expanduser().resolve())
    # open_clip gets cache_dir directly, but its nested HFTextEncoder does not.
    # Point Hugging Face Hub and Transformers at the same flat models--* root.
    os.environ["HF_HUB_CACHE"] = cache_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
    os.environ["TRANSFORMERS_CACHE"] = cache_path



def _collect_tensors(value: Any, path: str = "output") -> list[tuple[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("BiomedCLIP patch extraction requires PyTorch") from exc
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    tensors: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            tensors.extend(_collect_tensors(item, f"{path}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            tensors.extend(_collect_tensors(item, f"{path}[{index}]"))
    return tensors


def coerce_patch_tokens(output: Any, grid_size: tuple[int, int] | int, context: str):
    """Coerce one unambiguous spatial output to `[B,N,D]`; pooled fallback is forbidden."""

    import torch

    grid = (int(grid_size), int(grid_size)) if isinstance(grid_size, int) else tuple(map(int, grid_size))
    if len(grid) != 2 or min(grid) <= 0:
        raise RuntimeError(f"Invalid visual patch grid {grid!r} for {context}")
    patch_count = grid[0] * grid[1]
    candidates: list[tuple[str, torch.Tensor]] = []
    observed: list[str] = []
    for path, tensor in _collect_tensors(output):
        observed.append(f"{path}:{tuple(tensor.shape)}")
        if tensor.ndim == 3 and tensor.shape[1] in (patch_count, patch_count + 1):
            candidates.append((f"{path}:BND", tensor[:, -patch_count:, :]))
        if tensor.ndim == 4 and tuple(tensor.shape[-2:]) == grid:
            candidates.append((f"{path}:NCHW", tensor.flatten(2).transpose(1, 2)))
        if tensor.ndim == 4 and tuple(tensor.shape[1:3]) == grid:
            candidates.append((f"{path}:NHWC", tensor.reshape(tensor.shape[0], patch_count, tensor.shape[-1])))
    if len(candidates) != 1:
        detail = ", ".join(observed) if observed else f"type={type(output).__name__}"
        reason = "no" if not candidates else "ambiguous"
        raise RuntimeError(
            f"Could not obtain unique spatial patch tokens ({reason} candidate) for {context}; "
            f"grid={grid}, observed={detail}. Global pooled features are not allowed."
        )
    tokens = candidates[0][1]
    if not torch.isfinite(tokens).all():
        raise RuntimeError(f"Spatial patch tokens contain non-finite values for {context}")
    return tokens.float(), grid


class FrozenBiomedCLIPEncoder:
    """Local-only frozen BiomedCLIP adapter used exclusively by offline scripts."""

    _NON_GEOMETRIC_TRANSFORMS = {
        "ToTensor",
        "PILToTensor",
        "Normalize",
        "ConvertImageDtype",
        "ToImage",
        "ToDtype",
        "_ConvertImageToRGB",
        "MaybeConvertMode",
        "MaybeToTensor",
    }
    _GEOMETRIC_TRANSFORMS = {
        "Resize",
        "CenterCrop",
        "ResizeKeepRatio",
        "CenterCropOrPad",
    }

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        pretrained: str | None = None,
        cache_dir: str | None = None,
        local_model_dir: str | None = None,
        device: str = "cuda",
    ):
        cache_path = Path(cache_dir).expanduser() if cache_dir else None
        configure_offline_hf_cache(cache_path)
        local_path = Path(local_model_dir).expanduser() if local_model_dir else None
        if cache_path is None and local_path is None:
            raise ValueError("Provide an existing cache_dir or local_model_dir for offline BiomedCLIP")
        if cache_path is not None and not cache_path.is_dir():
            raise FileNotFoundError(f"BiomedCLIP cache_dir does not exist: {cache_path}")
        if local_path is not None and not local_path.is_dir():
            raise FileNotFoundError(f"BiomedCLIP local_model_dir is not a directory: {local_path}")
        pretrained_path: Path | None = None
        if pretrained is not None:
            pretrained_path = Path(pretrained).expanduser()
            if not pretrained_path.is_file():
                raise FileNotFoundError(
                    f"pretrained must be an existing local weight file; registry tags are forbidden: {pretrained}"
                )
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError("BiomedCLIP loading requires torch and open_clip_torch") from exc
        self.torch = torch
        self.open_clip = open_clip
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model_name = str(local_path or model_name)
        kwargs: dict[str, Any] = {}
        if cache_path is not None:
            kwargs["cache_dir"] = str(cache_path)
        if pretrained_path is not None:
            kwargs["pretrained"] = str(pretrained_path)
        errors: list[str] = []
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, **kwargs)
        except Exception as exc:
            errors.append(f"create_model_and_transforms: {exc}")
            try:
                created = open_clip.create_model_from_pretrained(self.model_name, **kwargs)
                self.model, self.preprocess = created[0], created[-1]
            except Exception as fallback_exc:
                errors.append(f"create_model_from_pretrained: {fallback_exc}")
                raise RuntimeError(
                    f"Failed to load BiomedCLIP strictly offline from model={self.model_name!r}, "
                    f"cache={str(cache_path)!r}: {' | '.join(errors)}"
                ) from fallback_exc
        try:
            self.tokenizer = open_clip.get_tokenizer(
                self.model_name,
                cache_dir=str(cache_path) if cache_path is not None else None,
            )
        except TypeError:
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if self.model.training:
            raise RuntimeError("BiomedCLIP must remain in eval mode")
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f"BiomedCLIP contains trainable parameters: {trainable[:5]}")

    def _grid_size(self) -> tuple[int, int]:
        visual = self.model.visual
        value = getattr(visual, "grid_size", None)
        if value is None and hasattr(visual, "patch_embed"):
            value = getattr(visual.patch_embed, "grid_size", None)
        trunk = getattr(visual, "trunk", None)
        if value is None and trunk is not None:
            value = getattr(trunk, "grid_size", None)
            if value is None and hasattr(trunk, "patch_embed"):
                value = getattr(trunk.patch_embed, "grid_size", None)
        if value is None:
            raise RuntimeError(f"BiomedCLIP visual encoder exposes no patch grid: {self.model_name}")
        return (int(value), int(value)) if isinstance(value, int) else tuple(map(int, value))

    def _forward_visual_features(self, image_batch):
        visual = self.model.visual
        if callable(getattr(visual, "forward_features", None)):
            return visual.forward_features(image_batch)
        trunk = getattr(visual, "trunk", None)
        if trunk is not None and callable(getattr(trunk, "forward_features", None)):
            return trunk.forward_features(image_batch)
        raise RuntimeError(
            f"BiomedCLIP visual encoder has no forward_features API for model={self.model_name}; "
            "pooled image features are forbidden"
        )

    @staticmethod
    def _to_rgb_pil(image: np.ndarray):
        from PIL import Image

        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D medical image, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("image contains non-finite values")
        gray = Image.fromarray(np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        return gray.convert("RGB")

    def prepare_images(self, images: Sequence[np.ndarray]):
        if not images:
            raise ValueError("images must not be empty")
        return self.torch.stack([self.preprocess(self._to_rgb_pil(image)) for image in images]).to(self.device)

    def transform_mask(self, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        from torchvision.transforms import CenterCrop, Resize
        from torchvision.transforms import functional as transform_functional
        from torchvision.transforms.functional import InterpolationMode

        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {array.shape}")
        transformed = Image.fromarray(array.astype(np.uint8), mode="L")
        operations = getattr(self.preprocess, "transforms", None)
        if operations is None:
            raise RuntimeError("BiomedCLIP preprocess does not expose deterministic transforms")
        for operation in operations:
            operation_name = operation.__class__.__name__
            if isinstance(operation, Resize):
                transformed = transform_functional.resize(
                    transformed,
                    operation.size,
                    interpolation=InterpolationMode.NEAREST,
                    max_size=operation.max_size,
                    antialias=False,
                )
            elif isinstance(operation, CenterCrop):
                transformed = transform_functional.center_crop(transformed, operation.size)
            elif operation_name == "ResizeKeepRatio":
                random_scale_prob = float(getattr(operation, "random_scale_prob", 0.0))
                random_aspect_prob = float(getattr(operation, "random_aspect_prob", 0.0))
                if random_scale_prob > 0.0 or random_aspect_prob > 0.0:
                    raise RuntimeError("Random ResizeKeepRatio cannot be synchronized with an offline mask")
                size = operation.get_params(
                    transformed,
                    operation.size,
                    operation.longest,
                    random_scale_prob,
                    getattr(operation, "random_scale_range", (1.0, 1.0)),
                    random_aspect_prob,
                    getattr(operation, "random_aspect_range", (1.0, 1.0)),
                )
                transformed = transform_functional.resize(
                    transformed,
                    size,
                    interpolation=InterpolationMode.NEAREST,
                    antialias=False,
                )
            elif operation_name == "CenterCropOrPad":
                transformed = transform_functional.center_crop(transformed, operation.size)
            elif operation_name in self._NON_GEOMETRIC_TRANSFORMS:
                continue
            else:
                pipeline = " -> ".join(item.__class__.__name__ for item in operations)
                raise RuntimeError(
                    "Unsupported BiomedCLIP preprocess operation for synchronized mask geometry: "
                    f"{operation}; full_pipeline={pipeline}"
                )
        return np.asarray(transformed, dtype=np.uint8)

    def encode_patch_tokens(self, images: Sequence[np.ndarray]):
        self.assert_frozen()
        with self.torch.inference_mode():
            output = self._forward_visual_features(self.prepare_images(images))
            version = getattr(self.open_clip, "__version__", "unknown")
            return coerce_patch_tokens(output, self._grid_size(), f"model={self.model_name} open_clip={version}")

    def encode_texts(self, texts: Sequence[str]):
        if not texts or any(not str(text).strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        self.assert_frozen()
        with self.torch.inference_mode():
            tokens = self.tokenizer(list(texts)).to(self.device)
            feature = self.model.encode_text(tokens)
            if feature.ndim != 2 or not self.torch.isfinite(feature).all():
                raise RuntimeError(f"Invalid BiomedCLIP text output shape={tuple(feature.shape)}")
            return self.torch.nn.functional.normalize(feature.float(), dim=-1)


