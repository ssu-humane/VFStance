from __future__ import annotations

import argparse
from pathlib import Path

from train import default_dataset_path, default_image_dir, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate T-MAD multimodal baseline.")
    parser.add_argument("--dataset_path", type=str, default=str(default_dataset_path(__file__)))
    parser.add_argument("--image_dir", type=str, default=str(default_image_dir(__file__)))
    parser.add_argument("--model_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "best_model"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "test_outputs"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_max_length", type=int, default=512)
    parser.add_argument("--reasoning_depth", type=int, default=5)
    parser.add_argument("--attention_heads", type=int, default=12)
    parser.add_argument("--projection_dim", type=int, default=768)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    run_evaluation(args.dataset_path, args.image_dir, args.model_dir, args.output_dir, args.batch_size, args.num_workers, args.text_max_length, args.reasoning_depth, args.attention_heads, args.projection_dim, args.gpu, args.split)


if __name__ == "__main__":
    main()
