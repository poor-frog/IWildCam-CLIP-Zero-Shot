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
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs with different seeds for mean+-std reporting.")
    parser.add_argument("--deterministic-training", action="store_true", help="Enable strict deterministic training controls for paper-grade runs.")
    parser.add_argument("--determinism-receipt", type=str, default=None, help="Immutable JSON receipt path required by --deterministic-training.")
    parser.add_argument("--wandb", dest="wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging when a wrapper enables it by default.")
    parser.set_defaults(wandb=False)
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
    parser.add_argument("--val-dataset", type=str, default="IWildCamVal", help="Validation dataset for CoOp and MaPLe best checkpoint selection.")
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
    parser.add_argument("--maple-lora-gamma", type=float, default=1.0, help="Evaluation/training strength multiplier for MaPLe LoRA deltas; 1 uses the learned adapter, 0 disables LoRA deltas.")
    parser.add_argument("--class-balanced-ce", action="store_true", help="Use train-only inverse-frequency class weights for MaPLe cross-entropy.")
    parser.add_argument("--kl-weight", type=float, default=0.0, help="Optional GLAD-lite KL distillation weight; 0 disables KL.")
    parser.add_argument("--kl-temperature", type=float, default=1.0, help="Temperature for optional GLAD-lite KL distillation.")
    parser.add_argument("--logit-adjustment-tau", type=float, default=None, help="Optional fixed post-hoc logit-adjustment tau; 0 disables adjustment.")
    parser.add_argument("--logit-adjustment-tau-grid", type=str, default=None, help="Optional comma-separated tau grid for IWildCamVal selection, e.g. '0,1'.")
    parser.add_argument("--class-bias-calibration", action="store_true", help="Select a post-hoc per-class bias vector on the validation split and apply it during final eval.")
    parser.add_argument("--class-bias-scale-grid", type=str, default="-2,-1,-0.5,0,0.5,1,2", help="Comma-separated scales for post-hoc class-bias candidates based on train class priors.")
    parser.add_argument("--selection-split", type=str, default="IWildCamVal", help="Validation split used for tau selection. IWildCamOOD is forbidden.")
    parser.add_argument("--num-ood-hp-examples", type=int, default=-1, help="Subsample IWildCamVal/IWildCamOODVal to N examples for HP selection; -1 uses the full split.")
    parser.add_argument("--class-balanced-ood", action="store_true", help="Use class-balanced sampling when --num-ood-hp-examples subsamples OOD validation.")
    parser.add_argument("--no-class-balanced-ood", dest="class_balanced_ood", action="store_false", help="Disable class-balanced OOD validation subsampling when a Kaggle mode enables it by default.")
    parser.add_argument("--drm-weight", type=float, default=0.0, help="Optional normalized anchor-L2 weight toward the initial CLIP weights during FLYP training.")
    parser.add_argument("--drm-warmup-epochs", type=int, default=0, help="Linearly warm up --drm-weight over this many epochs; 0 disables warmup.")
    parser.add_argument("--wise-alphas", type=str, default=None, help="Optional comma-separated WiSE-FT alpha grid; alpha=0 is fine-tuned, alpha=1 is zero-shot init. Omit to disable WiSE selection.")
    parser.add_argument("--wise-eval-alpha", type=float, default=None, help="Optional single WiSE-FT alpha to apply before final FLYP evaluation.")
    parser.add_argument("--adaptive-bins", type=str, default="0,0.4,0.6,0.8,0.9,1.0", help="Comma-separated confidence bin edges for adaptive WiSE evaluation.")
    parser.add_argument("--adaptive-min-bin-examples", type=int, default=100, help="Minimum validation examples per confidence bin before selecting a bin-specific WiSE alpha.")
    parser.add_argument("--cd-path", type=str, default=None, help="Optional class concept-description JSON path for DRM-style evaluation.")
    parser.add_argument("--cd-beta", type=float, default=0.5, help="DRM-style blend weight: beta * default_prompt_probs + (1 - beta) * concept_description_probs.")
    parser.add_argument("--tail-proto-weight", type=float, default=0.0, help="Tail-Aware FLYP auxiliary weight lambda_tail; 0 disables train-time tail prototypes.")
    parser.add_argument("--tail-proto-scale", type=float, default=50.0, help="Logit scale for Tail-Aware FLYP class-prototype logits.")
    parser.add_argument("--tail-proto-objective", type=str, default="ce", choices=["ce", "distill", "fixed_distill"], help="Train-time tail objective: hard prototype CE, self TPA-logit distillation, or fixed-teacher TPA distillation.")
    parser.add_argument("--tail-proto-temperature", type=float, default=1.0, help="Temperature for --tail-proto-objective=distill.")
    parser.add_argument("--tail-proto-teacher-load", type=str, default=None, help="Fixed teacher CLIPEncoder checkpoint for --tail-proto-objective=fixed_distill.")
    parser.add_argument("--tail-proto-max-batches", type=int, default=None, help="Optional cap for prototype-building batches, intended for smoke tests only.")
    parser.add_argument("--btel-weight", type=float, default=0.0, help="Burst-aware Tail Evidence Learning auxiliary weight; 0 disables BTEL exactly.")
    parser.add_argument("--btel-prototype-scale", type=float, default=50.0, help="Frozen prototype-logit scale used by BTEL evidence.")
    parser.add_argument("--btel-negative-quantile", type=float, default=0.95, help="Train-only negative-burst evidence quantile used for BTEL calibration.")
    parser.add_argument("--btel-max-frames-per-sequence", type=int, default=8, help="Maximum frames sampled from one train burst in a BTEL batch.")
    parser.add_argument("--btel-audit-only", action="store_true", help="Print train and validation burst audits, then exit before training.")

    parsed_args = parser.parse_args()
    parsed_args.device = resolve_device_choice(parsed_args.device)
    return parsed_args
