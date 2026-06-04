import getpass
import os
import random
import socket

import numpy as np
import torch

import src.datasets as datasets
from src.config import parse_arguments
from src.models.clip_encoder import CLIPEncoder
from src.models.coop import (
    CustomCLIP,
    ensure_openai_clip_for_coop,
    eval_coop_single_dataset,
    get_prompt_learner,
    load_prompt_learner,
    maybe_data_parallel,
    save_prompt_learner,
    train_one_epoch,
)


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
            "method": "coop",
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


def print_summary(summary_rows):
    if not summary_rows:
        return
    print("\n=== CoOp Summary ===")
    print("| Split         | Top-1  | F1-macro |")
    print("| ------------- | ------ | -------- |")
    for dataset_name, top1, f1_macro in summary_rows:
        top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
        f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
        print(f"| {dataset_name:<13} | {top1_text:<6} | {f1_text:<8} |")


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

    clip_encoder = CLIPEncoder(args, keep_lang=True)
    ensure_openai_clip_for_coop(clip_encoder.model, args.model)
    clip_encoder.model.to(args.device)
    clip_encoder.model.eval()

    dataset_class = getattr(datasets, args.train_dataset)
    train_data = dataset_class(
        clip_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    classnames = train_data.classnames
    model = CustomCLIP(args, classnames, clip_encoder.model).to(args.device)
    model = maybe_data_parallel(model, args)
    if hasattr(model, "module"):
        print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices")

    if args.load is not None:
        load_prompt_learner(model, args.load, args.device)

    optimizer = torch.optim.AdamW(get_prompt_learner(model).parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    wandb = init_wandb(args)

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_data.train_loader, optimizer, args, epoch, wandb=wandb)
        scheduler.step()
        print(f"Epoch {epoch}: loss={stats.loss:.4f}, acc={stats.accuracy:.4f}")
        if wandb is not None:
            wandb.log({
                "train/epoch_loss": stats.loss,
                "train/epoch_accuracy": stats.accuracy,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })

    if args.save is not None:
        save_path = args.save
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "coop_prompt_learner.pt")
        save_prompt_learner(model, save_path, args, classnames)
        print(f"Saved CoOp prompt learner to {save_path}")

    summary_rows = []
    if args.eval_datasets is not None:
        for dataset_name in args.eval_datasets:
            print(f"Evaluating CoOp on {dataset_name}...")
            eval_dataset_class = getattr(datasets, dataset_name)
            eval_dataset = eval_dataset_class(
                clip_encoder.val_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )
            results = eval_coop_single_dataset(model, eval_dataset, args)
            top1 = results.get("top1")
            f1_macro = results.get("F1-macro_all")
            print(f"  {dataset_name} Top-1 accuracy: {top1:.4f}")
            if f1_macro is not None:
                print(f"  {dataset_name} F1-macro_all: {f1_macro:.4f}")
            summary_rows.append((dataset_name, top1, f1_macro))
            if wandb is not None:
                metrics = {f"eval/{dataset_name}/top1": top1}
                if f1_macro is not None:
                    metrics[f"eval/{dataset_name}/f1_macro"] = f1_macro
                wandb.log(metrics)

    print_summary(summary_rows)
    log_wandb_summary(wandb, summary_rows)
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main(parse_arguments())
