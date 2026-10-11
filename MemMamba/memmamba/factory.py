from memmamba.model_spec import MODEL_SPEC


def build_model_from_config(config: dict):
    from memmamba.memory.storage import load_memory_bank
    from memmamba.models import MemMamba

    memory_bank = load_memory_bank(
        config["memory_bank_path"],
        expected_dataset_name=config["dataset_name"],
        expected_num_classes=MODEL_SPEC.num_classes,
    )
    return MemMamba(memory_bank)
