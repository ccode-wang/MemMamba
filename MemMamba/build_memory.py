from __future__ import annotations

import argparse
import logging

from memmamba.config import load_defaults
from memmamba.memory.builder import MemoryBuildOptions, build_multimodal_memory
from memmamba.memory.features import load_semantic_feature_bank, load_visual_feature_bank
from memmamba.memory.storage import save_memory_bank


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config", default="configs/task05_prostate.json")
    known, _ = pre.parse_known_args(argv); defaults = load_defaults(known.config)
    parser = argparse.ArgumentParser(description="Build class-wise multimodal Memory Bank.", parents=[pre])
    parser.add_argument("--visual_feature_path", default=defaults["visual_feature_path"])
    parser.add_argument("--semantic_embedding_path", default=defaults["semantic_embedding_path"])
    parser.add_argument("--output_path", default=defaults["memory_bank_path"])
    parser.add_argument("--prototypes_per_class", type=int, default=defaults["prototypes_per_class"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--max_iter", type=int, default=defaults["kmeans_max_iter"])
    parser.add_argument("--tolerance", type=float, default=defaults["kmeans_tolerance"])
    return parser.parse_args(argv)


def main():
    args = parse_args(); logging.basicConfig(level=logging.INFO)
    memory = build_multimodal_memory(load_visual_feature_bank(args.visual_feature_path),
        load_semantic_feature_bank(args.semantic_embedding_path),
        MemoryBuildOptions(args.prototypes_per_class, args.seed, args.max_iter, args.tolerance))
    save_memory_bank(memory, args.output_path)
    logging.info("Saved Memory Bank prototypes=%d visual_dim=%d semantic_dim=%d path=%s", len(memory.class_ids), memory.visual_dim, memory.semantic_dim, args.output_path)


if __name__ == "__main__": main()
