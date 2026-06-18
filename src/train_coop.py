import getpass
import os
import random
import socket
from pathlib import Path

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
from src.models.logit_adjustment import (
    build_train_class_priors_for_dataset,
    describe_tau_selection,
    parse_tau_grid,
    select_best_tau,
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


def get_validation_score(results, metric_name):
    value = results.get(metric_name)
    if value is None:
        value = results.get("top1")
    if value is None:
        raise KeyError(f"Validation results include neither {metric_name!r} nor 'top1'.")
    return float(value)


def resolve_save_path(save_path):
    if save_path is None:
        return None
    if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
        return os.path.join(save_path, "coop_prompt_learner.pt")
    return save_path


def resolve_best_checkpoint_path(args):
    if args.best_checkpoint is not None:
        return args.best_checkpoint
    if args.save is not None:
        save_path = Path(args.save)
        if os.path.isdir(args.save) or save_path.suffix == "":
            return str(save_path / "coop_prompt_learner_best.pt")
        return str(save_path.with_name(f"{save_path.stem}_best{save_path.suffix}"))
    return os.path.join("checkpoints", "coop_prompt_learner_best.pt")


def build_eval_dataset(dataset_name, clip_encoder, args):
    eval_dataset_class = getattr(datasets, dataset_name)
    return eval_dataset_class(
        clip_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


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
    best_score = None
    best_epoch = None
    best_checkpoint_path = resolve_best_checkpoint_path(args)
    val_dataset = None
    if args.val_dataset is not None and args.epochs > 0:
        val_dataset = build_eval_dataset(args.val_dataset, clip_encoder, args)

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

        if val_dataset is not None:
            print(f"Validating CoOp on {args.val_dataset} at epoch {epoch}...")
            val_results = eval_coop_single_dataset(model, val_dataset, args)
            val_top1 = val_results.get("top1")
            val_f1_macro = val_results.get("F1-macro_all")
            val_score = get_validation_score(val_results, args.best_metric)
            print(f"  {args.val_dataset} Top-1 accuracy: {val_top1:.4f}")
            if val_f1_macro is not None:
                print(f"  {args.val_dataset} F1-macro_all: {val_f1_macro:.4f}")
            print(f"  {args.val_dataset} best metric {args.best_metric}: {val_score:.4f}")
            if wandb is not None:
                val_metrics = {
                    f"val/{args.val_dataset}/top1": val_top1,
                    f"val/{args.val_dataset}/{args.best_metric}": val_score,
                    "epoch": epoch,
                }
                if val_f1_macro is not None:
                    val_metrics[f"val/{args.val_dataset}/f1_macro"] = val_f1_macro
                wandb.log(val_metrics)
            if best_score is None or val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                save_prompt_learner(model, best_checkpoint_path, args, classnames)
                print(
                    f"Saved best CoOp prompt learner to {best_checkpoint_path} "
                    f"(epoch {best_epoch}, {args.best_metric}={best_score:.4f})"
                )
                if wandb is not None:
                    wandb.log({
                        "val/best_epoch": best_epoch,
                        f"val/best_{args.best_metric}": best_score,
                    })

    if args.save is not None:
        save_path = resolve_save_path(args.save)
        save_prompt_learner(model, save_path, args, classnames)
        print(f"Saved CoOp prompt learner to {save_path}")

    if args.eval_datasets is not None and best_epoch is not None and not args.no_load_best_for_eval:
        load_prompt_learner(model, best_checkpoint_path, args.device)
        print(
            f"Loaded best CoOp prompt learner from {best_checkpoint_path} "
            f"for final eval (epoch {best_epoch}, {args.best_metric}={best_score:.4f})"
        )

    _, class_priors = build_train_class_priors_for_dataset(train_data, args.device)
    selected_tau = args.logit_adjustment_tau
    tau_grid = parse_tau_grid(args.logit_adjustment_tau_grid)
    if tau_grid is not None:
        selection_dataset = build_eval_dataset(args.selection_split, clip_encoder, args)
        selection = select_best_tau(
            eval_coop_single_dataset,
            model,
            selection_dataset,
            args,
            tau_grid=tau_grid,
            class_priors=class_priors,
        )
        for line in describe_tau_selection(selection):
            print(line)
        selected_tau = selection.best_tau

    summary_rows = []
    if args.eval_datasets is not None:
        for dataset_name in args.eval_datasets:
            print(f"Evaluating CoOp on {dataset_name}...")
            eval_dataset = build_eval_dataset(dataset_name, clip_encoder, args)
            results = eval_coop_single_dataset(model, eval_dataset, args, tau=selected_tau, class_priors=class_priors)
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
