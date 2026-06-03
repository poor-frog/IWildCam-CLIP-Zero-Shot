"""
PoorFrogs: Zero-shot CLIP evaluation on IWildCam.

Usage:
    # Zero-shot evaluation on all IWildCam splits
    python src/main.py \
        --model=ViT-B-16 \
        --template=iwildcam_template \
        --train-dataset=IWildCam \
        --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
        --data-location=./datasets/data/
"""
import getpass
import socket
import torch
import random
import numpy as np

from src.config import parse_arguments
from src.models.clip_encoder import CLIPEncoder
from src.models.zeroshot import get_zeroshot_classifier
from src.models.eval import eval_single_dataset
import src.datasets as datasets


def init_wandb(args):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb or run ./install.sh before using --wandb.") from error

    init_kwargs = {
        "project": args.wandb_project,
        "config": {
            **vars(args),
            "system_hostname": socket.gethostname(),
            "system_user": getpass.getuser(),
        },
    }
    if args.wandb_entity is not None:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        init_kwargs["name"] = args.wandb_run_name

    wandb.init(**init_kwargs)
    return wandb


def log_wandb_split(wandb, dataset_name, top1, f1_macro):
    if wandb is None:
        return

    metrics = {}
    if top1 is not None:
        metrics[f"eval/{dataset_name}/top1"] = top1
    if f1_macro is not None:
        metrics[f"eval/{dataset_name}/f1_macro"] = f1_macro
    if metrics:
        wandb.log(metrics)


def log_wandb_summary(wandb, summary_rows):
    if wandb is None:
        return

    table = wandb.Table(columns=["split", "top1", "f1_macro", "top1_percent", "f1_macro_percent"])
    for dataset_name, top1, f1_macro in summary_rows:
        table.add_data(
            dataset_name,
            top1,
            f1_macro,
            top1 * 100 if top1 is not None else None,
            f1_macro * 100 if f1_macro is not None else None,
        )
    wandb.log({"eval/summary": table})


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    wandb = init_wandb(args)

    # Load CLIP encoder (open_clip for ViT-B-16, ViT-L-14; original clip for RN50 etc.)
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    clip_encoder.eval()
    clip_encoder.to(args.device)

    # Build zero-shot classification head from class names + text templates
    classification_head = get_zeroshot_classifier(args, clip_encoder.model)
    classification_head.eval()
    classification_head.to(args.device)

    # Free text transformer to save memory (only image encoder needed for eval)
    if hasattr(clip_encoder.model, 'transformer'):
        delattr(clip_encoder.model, 'transformer')

    # Evaluate on each dataset
    print(f"\n=== Zero-shot evaluation on: {args.eval_datasets} ===\n")
    summary_rows = []
    for dataset_name in args.eval_datasets:
        print(f"Evaluating on {dataset_name}...")
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            clip_encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )

        results = eval_single_dataset(
            clip_encoder, dataset, args, classification_head,
        )

        if 'top1' in results:
            print(f"  {dataset_name} Top-1 accuracy: {results['top1']:.4f}")

        f1_macro = None
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"  {dataset_name} {key}: {val:.4f}")
            if key.lower() == 'f1-macro_all':
                f1_macro = val

        summary_rows.append((dataset_name, results.get('top1'), f1_macro))
        log_wandb_split(wandb, dataset_name, results.get('top1'), f1_macro)

    if summary_rows:
        print("\n=== Summary ===")
        print("| Split         | Top-1  | F1-macro |")
        print("| ------------- | ------ | -------- |")
        for dataset_name, top1, f1_macro in summary_rows:
            top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
            f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
            print(f"| {dataset_name:<13} | {top1_text:<6} | {f1_text:<8} |")

    log_wandb_summary(wandb, summary_rows)
    if wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
