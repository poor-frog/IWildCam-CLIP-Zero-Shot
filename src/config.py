import os
import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="Dataset name for zero-shot classifier text embeddings.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="The type of model (e.g. ViT-B-16, ViT-L-14, RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optionally load a pretrained classifier.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a classifier.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="PoorFrogs")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parsed_args = parser.parse_args()
    if torch.cuda.is_available():
        parsed_args.device = "cuda"
    elif torch.backends.mps.is_available():
        parsed_args.device = "mps"
    else:
        parsed_args.device = "cpu"
    return parsed_args
