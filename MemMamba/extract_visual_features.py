from __future__ import annotations

import argparse
import logging

from memmamba.config import load_class_config, load_defaults
from memmamba.data.task05_prostate import list_npz_slices
from memmamba.memory.alignment_teachers import save_alignment_teacher_bank
from memmamba.memory.features import (
    VisualExtractionOptions,
    extract_visual_and_alignment_feature_banks,
    save_visual_feature_bank,
)
from memmamba.models.biomedclip_adapter import FrozenBiomedCLIPEncoder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/task05_prostate.json")
    known, _ = config_parser.parse_known_args(argv)
    defaults = load_defaults(known.config)
    parser = argparse.ArgumentParser(description="Extract strict T2/ADC regional BiomedCLIP features.", parents=[config_parser])
    parser.add_argument("--train_npz_dir", default=defaults["train_npz_dir"])
    parser.add_argument("--output_path", default=defaults["visual_feature_path"])
    parser.add_argument("--alignment_teacher_path", default=defaults["alignment_teacher_path"])
    parser.add_argument("--class_config", default=defaults["class_config"])
    parser.add_argument("--biomedclip_model_name", default=defaults["biomedclip_model_name"])
    parser.add_argument("--cache_dir", default=defaults["cache_dir"])
    parser.add_argument("--local_model_dir", default=defaults["local_model_dir"])
    parser.add_argument("--pretrained", default=None, help="Existing local weight file only.")
    parser.add_argument("--min_pixels", type=int, default=defaults["min_pixels"])
    parser.add_argument("--min_effective_patches", type=int, default=defaults["min_effective_patches"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log per-sample filtering details in addition to class summaries.",
    )
    parser.add_argument("--device", default=defaults["device"])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    encoder = FrozenBiomedCLIPEncoder(args.biomedclip_model_name, args.pretrained, args.cache_dir, args.local_model_dir, args.device)
    strict_bank, alignment_bank = extract_visual_and_alignment_feature_banks(
        list_npz_slices(args.train_npz_dir),
        encoder,
        load_class_config(args.class_config),
        VisualExtractionOptions(
            class_morphology={
                int(class_id): dict(values)
                for class_id, values in load_defaults(args.config)["class_morphology"].items()
            },
            min_effective_patches=args.min_effective_patches,
            min_pixels=args.min_pixels,
            batch_size=args.batch_size,
        ),
    )
    save_visual_feature_bank(strict_bank, args.output_path)
    save_alignment_teacher_bank(alignment_bank, args.alignment_teacher_path)
    logging.info("Saved visual Feature Bank to %s", args.output_path)
    logging.info("Saved Alignment Teacher Bank to %s", args.alignment_teacher_path)


if __name__ == "__main__":
    main()
