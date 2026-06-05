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
    parser.add_argument("--n-ctx", type=int, default=16, help="Number of CoOp context tokens.")
    parser.add_argument("--ctx-init", type=str, default="", help="Optional text initialization for CoOp context tokens.")
    parser.add_argument("--class-token-position", type=str, default="end", choices=["end", "middle", "front"])
    parser.add_argument("--csc", action="store_true", help="Use class-specific CoOp context tokens.")
    parser.add_argument("--epochs", type=int, default=50, help="CoOp training epochs.")
    parser.add_argument("--lr", type=float, default=0.002, help="CoOp prompt learner learning rate.")
    parser.add_argument("--wd", type=float, default=1e-5, help="CoOp prompt learner weight decay.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional train batch cap for smoke tests.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional eval batch cap for smoke tests.")
    parser.add_argument("--no-data-parallel", action="store_true", help="Disable CoOp DataParallel on multi-GPU CUDA hosts.")
    parser.add_argument("--val-dataset", type=str, default="IWildCamIDVal", help="Validation dataset for CoOp best checkpoint selection.")
    parser.add_argument("--best-metric", type=str, default="F1-macro_all", help="Validation metric used to select the best CoOp checkpoint; falls back to top1 when missing.")
    parser.add_argument("--best-checkpoint", type=str, default=None, help="Optional path for the best CoOp checkpoint.")
    parser.add_argument("--no-load-best-for-eval", action="store_true", help="Skip loading the best CoOp checkpoint before final eval.")

    parsed_args = parser.parse_args()
    if torch.cuda.is_available():
        parsed_args.device = "cuda"
    elif torch.backends.mps.is_available():
        parsed_args.device = "mps"
    else:
        parsed_args.device = "cpu"
    return parsed_args
