"""Public MemMamba model interface."""

__all__ = ["MemMamba", "MemMambaOutput"]


def __getattr__(name):
    # Feature extraction and CLI help need not import the segmentation model.
    if name in __all__:
        from memmamba.models.memmamba import MemMamba, MemMambaOutput

        return {"MemMamba": MemMamba, "MemMambaOutput": MemMambaOutput}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

