import getpass
import os
import random
import socket
from pathlib import Path

import numpy as np
import torch

import src.datasets as datasets
from src.config import parse_arguments
from src.datasets.iwildcam import compute_inverse_frequency_weights
from src.models.coop import maybe_data_parallel
from src.models import maple_clip
from src.models.maple_full import (
    CustomFullMaPLeCLIP,
    build_maple_design_details,
    eval_full_maple_single_dataset,
    load_full_maple_prompt_learner,
    save_full_maple_prompt_learner,
    train_full_maple_one_epoch,
)
from src.models.logit_adjustment import (
    build_train_class_priors_for_dataset,
    describe_tau_selection,
    parse_tau_grid,
    select_best_tau,
)
from src.models.zeroshot import (
    FrozenZeroShotAnchor,
    MaPLeZeroPromptImageEncoder,
    get_compatible_zeroshot_classifier,
)
from src.train_coop import build_eval_dataset, get_validation_score, log_wandb_summary


def print_summary(summary_rows):
    if not summary_rows:
        return
    print("\n=== Full MaPLe Summary ===")
    print("| Split         | Top-1  | F1-macro |")
    print("| ------------- | ------ | -------- |")
    for dataset_name, top1, f1_macro in summary_rows:
        top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
        f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
        print(f"| {dataset_name:<13} | {top1_text:<6} | {f1_text:<8} |")


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
            "method": getattr(args, "training_method", "maple_full"),
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


def resolve_save_path(save_path):
    if save_path is None:
        return None
    if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
        return os.path.join(save_path, "maple_full_prompt_learner.pt")
    return save_path


def resolve_best_checkpoint_path(args):
    if args.best_checkpoint is not None:
        return args.best_checkpoint
    if args.save is not None:
        save_path = Path(args.save)
        if os.path.isdir(args.save) or save_path.suffix == "":
            return str(save_path / "maple_full_prompt_learner_best.pt")
        return str(save_path.with_name(f"{save_path.stem}_best{save_path.suffix}"))
    return os.path.join("checkpoints", "maple_full_prompt_learner_best.pt")


def build_class_balanced_ce_weights(train_data, device):
    source_dataset = train_data if hasattr(train_data, "get_subset") else train_data.dataset
    train_subset = source_dataset.get_subset("train", transform=None)
    labels = getattr(train_subset, "y_array", None)
    if labels is None:
        raise ValueError("Train subset does not expose y_array labels.")
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    num_classes = len(train_data.classnames)
    _, _, weights = compute_inverse_frequency_weights(labels, num_classes)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    design_details = build_maple_design_details(prompt_depth=args.maple_prompt_depth, n_ctx=args.n_ctx)
    clip_model, preprocess = maple_clip.load(args.model, device="cpu", jit=False, design_details=design_details)
    if getattr(args, "maple_precision", "fp32") == "fp32":
        clip_model.float()
    clip_model.to(args.device)
    clip_model.eval()

    dataset_class = getattr(datasets, args.train_dataset)
    train_data = dataset_class(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    classnames = train_data.classnames
    model = CustomFullMaPLeCLIP(args, classnames, clip_model).to(args.device)
    model = maybe_data_parallel(model, args)
    if hasattr(model, "module"):
        print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices")

    if args.load is not None:
        load_full_maple_prompt_learner(model, args.load, args.device)

    class_weights = None
    if getattr(args, "class_balanced_ce", False):
        class_weights = build_class_balanced_ce_weights(train_data, args.device)

    anchor_model = None
    kl_weight = getattr(args, "kl_weight", 0.0)
    kl_temperature = getattr(args, "kl_temperature", 1.0)
    if kl_weight != 0.0:
        classification_head = get_compatible_zeroshot_classifier(args, device=args.device).to(args.device)
        anchor_image_encoder = MaPLeZeroPromptImageEncoder(
            clip_model.visual,
            n_ctx=args.n_ctx,
            prompt_depth=args.maple_prompt_depth,
        ).to(args.device)
        anchor_model = FrozenZeroShotAnchor(anchor_image_encoder, classification_head).to(args.device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    wandb = init_wandb(args)
    best_score = None
    best_epoch = None
    best_checkpoint_path = resolve_best_checkpoint_path(args)
    val_dataset = None
    if args.val_dataset is not None and args.epochs > 0:
        val_dataset = build_eval_dataset(args.val_dataset, SimpleNamespaceEncoder(clip_model, preprocess), args)

    for epoch in range(1, args.epochs + 1):
        stats = train_full_maple_one_epoch(
            model,
            train_data.train_loader,
            optimizer,
            args,
            epoch,
            wandb=wandb,
            class_weights=class_weights,
            anchor_model=anchor_model,
            kl_weight=kl_weight,
            kl_temperature=kl_temperature,
        )
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
            print(f"Validating full MaPLe on {args.val_dataset} at epoch {epoch}...")
            val_results = eval_full_maple_single_dataset(model, val_dataset, args)
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
                save_full_maple_prompt_learner(model, best_checkpoint_path, args, classnames)
                print(
                    f"Saved best full MaPLe prompt learner to {best_checkpoint_path} "
                    f"(epoch {best_epoch}, {args.best_metric}={best_score:.4f})"
                )

    if args.save is not None:
        save_path = resolve_save_path(args.save)
        save_full_maple_prompt_learner(model, save_path, args, classnames)
        print(f"Saved full MaPLe prompt learner to {save_path}")

    if args.eval_datasets is not None and best_epoch is not None and not args.no_load_best_for_eval:
        load_full_maple_prompt_learner(model, best_checkpoint_path, args.device)
        print(
            f"Loaded best full MaPLe prompt learner from {best_checkpoint_path} "
            f"for final eval (epoch {best_epoch}, {args.best_metric}={best_score:.4f})"
        )

    _, class_priors = build_train_class_priors_for_dataset(train_data, args.device)
    selected_tau = args.logit_adjustment_tau
    tau_grid = parse_tau_grid(args.logit_adjustment_tau_grid)
    if tau_grid is not None:
        eval_encoder = SimpleNamespaceEncoder(clip_model, preprocess)
        selection_dataset = build_eval_dataset(args.selection_split, eval_encoder, args)
        selection = select_best_tau(
            eval_full_maple_single_dataset,
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
        eval_encoder = SimpleNamespaceEncoder(clip_model, preprocess)
        for dataset_name in args.eval_datasets:
            print(f"Evaluating full MaPLe on {dataset_name}...")
            eval_dataset = build_eval_dataset(dataset_name, eval_encoder, args)
            results = eval_full_maple_single_dataset(model, eval_dataset, args, tau=selected_tau, class_priors=class_priors)
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


class SimpleNamespaceEncoder:
    def __init__(self, model, preprocess):
        self.model = model
        self.train_preprocess = preprocess
        self.val_preprocess = preprocess


if __name__ == "__main__":
    main(parse_arguments())
