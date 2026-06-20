import os
import argparse

from src.device import resolve_device_choice


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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu", "xla"],
        help="Training device. 'auto' preserves the default priority: CUDA, MPS, TPU/XLA, then CPU.",
    )
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
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "linear"], help="Training LR scheduler.")
    parser.add_argument("--warmup-length", type=int, default=0, help="Number of optimizer steps for linear LR warmup.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional train batch cap for smoke tests.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional eval batch cap for smoke tests.")
    parser.add_argument("--no-data-parallel", action="store_true", help="Disable CoOp DataParallel on multi-GPU CUDA hosts.")
    parser.add_argument("--val-dataset", type=str, default="IWildCamIDVal", help="Validation dataset for CoOp best checkpoint selection.")
    parser.add_argument("--best-metric", type=str, default="F1-macro_all", help="Validation metric used to select the best CoOp checkpoint; falls back to top1 when missing.")
    parser.add_argument("--best-checkpoint", type=str, default=None, help="Optional path for the best CoOp checkpoint.")
    parser.add_argument("--no-load-best-for-eval", action="store_true", help="Skip loading the best CoOp checkpoint before final eval.")
    parser.add_argument("--maple-precision", type=str, default="fp32", choices=["fp32", "amp"], help="Precision mode for MaPLe.")
    parser.add_argument("--maple-prompt-depth", type=int, default=9, help="MaPLe prompt depth; 1 means shallow coupled prompts, >1 adds deep compound prompts.")
    parser.add_argument("--maple-lora-rank", type=int, default=0, help="Optional MaPLe LoRA rank; 0 disables LoRA.")
    parser.add_argument("--maple-lora-alpha", type=int, default=None, help="Optional MaPLe LoRA alpha; defaults to 2 * rank when omitted.")
    parser.add_argument("--maple-lora-dropout", type=float, default=0.0, help="Reserved for optional MaPLe LoRA adapters; currently must remain 0.0 for out_proj weight parametrization.")
    parser.add_argument("--maple-lora-target", type=str, default="vision_out_proj", choices=["vision_out_proj"], help="Optional MaPLe LoRA target modules.")
    parser.add_argument("--maple-lora-layers", type=str, default="last6", help="Optional MaPLe LoRA layer selection: 'lastN' or 'all'.")
    parser.add_argument("--class-balanced-ce", action="store_true", help="Use train-only inverse-frequency class weights for MaPLe cross-entropy.")
    parser.add_argument("--kl-weight", type=float, default=0.0, help="Optional GLAD-lite KL distillation weight; 0 disables KL.")
    parser.add_argument("--kl-temperature", type=float, default=1.0, help="Temperature for optional GLAD-lite KL distillation.")
    parser.add_argument("--logit-adjustment-tau", type=float, default=None, help="Optional fixed post-hoc logit-adjustment tau; 0 disables adjustment.")
    parser.add_argument("--logit-adjustment-tau-grid", type=str, default=None, help="Optional comma-separated tau grid for IWildCamVal selection, e.g. '0,1'.")
    parser.add_argument("--selection-split", type=str, default="IWildCamVal", help="Validation split used for tau selection. IWildCamOOD is forbidden.")
    parser.add_argument("--num-ood-hp-examples", type=int, default=-1, help="Subsample IWildCamVal/IWildCamOODVal to N examples for HP selection; -1 uses the full split.")
    parser.add_argument("--class-balanced-ood", action="store_true", help="Use class-balanced sampling when --num-ood-hp-examples subsamples OOD validation.")
    parser.add_argument("--no-class-balanced-ood", dest="class_balanced_ood", action="store_false", help="Disable class-balanced OOD validation subsampling when a Kaggle mode enables it by default.")

    parsed_args = parser.parse_args()
    parsed_args.device = resolve_device_choice(parsed_args.device)
    return parsed_args
