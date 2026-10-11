from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemMambaSpec:
    model_name: str = "MemMamba"
    in_channels: int = 2
    num_classes: int = 3
    num_foreground_classes: int = 2
    base_channels: int = 64
    knowledge_dim: int = 256
    retrieval_top_k: int = 4
    presence_threshold: float = 0.3
    kcm_num_blocks: int = 2
    kcm_d_model: int = 256
    kcm_d_state: int = 16
    kcm_dt_rank: str = "auto"
    kcm_expand: int = 2
    kcm_conv_kernel: int = 3
    kcm_direction_fusion: str = "mean"
    kcm_modulate_delta: bool = True
    kcm_modulate_B: bool = True
    kcm_modulate_C: bool = True
    kcm_modulate_A: bool = False
    kcm_input_knowledge_gate: bool = True
    kcm_alpha_init: float = 0.0
    kcm_layer_scale_init: float = 1e-4
    kcm_drop_path: float = 0.0
    knowledge_dropout: float = 0.1
    kcm_scan_backend: str = "auto"

    def to_dict(self) -> dict:
        """Return a snapshot; modifying it never changes the model specification."""
        return asdict(self)


MODEL_SPEC = MemMambaSpec()
