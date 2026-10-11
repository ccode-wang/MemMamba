from __future__ import annotations

import argparse
import logging

from memmamba.config import load_class_config, load_defaults, load_descriptions
from memmamba.memory.features import save_semantic_feature_bank
from memmamba.memory.semantic import extract_semantic_feature_bank
from memmamba.models.biomedclip_adapter import FrozenBiomedCLIPEncoder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/task05_prostate.json")
    known, _ = config_parser.parse_known_args(argv)
    defaults = load_defaults(known.config)
    parser = argparse.ArgumentParser(description="Extract fixed structured organ semantics with local BiomedCLIP.", parents=[config_parser])
    parser.add_argument("--description_path", default=defaults["description_path"])
    parser.add_argument("--class_config", default=defaults["class_config"])
    parser.add_argument("--output_path", default=defaults["semantic_embedding_path"])
    parser.add_argument("--biomedclip_model_name", default=defaults["biomedclip_model_name"])
    parser.add_argument("--cache_dir", default=defaults["cache_dir"])
    parser.add_argument("--local_model_dir", default=defaults["local_model_dir"])
    parser.add_argument("--pretrained", default=None, help="Existing local weight file only.")
    parser.add_argument("--device", default=defaults["device"])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    classes = load_class_config(args.class_config)
    descriptions = load_descriptions(args.description_path, classes)
    encoder = FrozenBiomedCLIPEncoder(args.biomedclip_model_name, args.pretrained, args.cache_dir, args.local_model_dir, args.device)
    bank = extract_semantic_feature_bank(descriptions, encoder, classes, args.description_path)
    save_semantic_feature_bank(bank, args.output_path)
    logging.info("Saved semantic Feature Bank: classes=%d dim=%d path=%s", len(bank.class_names), bank.feature_dim, args.output_path)


if __name__ == "__main__":
    main()
