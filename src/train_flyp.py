import os
import random
from pathlib import Path

import numpy as np
import torch

import src.datasets as datasets
import src.templates as templates
from src.config import parse_arguments
from src.models.clip_encoder import CLIPEncoder
from src.models.coop import maybe_data_parallel, unwrap_model
from src.models.flyp import eval_flyp_single_dataset, train_flyp_one_epoch, wise_interpolate_state_dict
from src.train_coop import build_eval_dataset, get_validation_score, log_wandb_summary
from src.train_maple_full import build_step_lr_scheduler, init_wandb


OPEN_CLIP_FLYP_MODELS = {"ViT-B-16", "ViT-L-14"}


def resolve_flyp_save_path(save_path):
    if save_path is None:
        return None
    if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
        return os.path.join(save_path, "flyp_clip_encoder.pt")
    return save_path


def resolve_flyp_best_checkpoint_path(args):
    if args.best_checkpoint is not None:
        return args.best_checkpoint
    if args.save is not None:
        save_path = Path(args.save)
        if os.path.isdir(args.save) or save_path.suffix == "":
            return str(save_path / "flyp_clip_encoder_best.pt")
        return str(save_path.with_name(f"{save_path.stem}_best{save_path.suffix}"))
    return os.path.join("checkpoints", "flyp_clip_encoder_best.pt")


def ensure_open_clip_for_flyp(model_name):
    if model_name not in OPEN_CLIP_FLYP_MODELS:
        raise ValueError("FLYP baseline currently supports open_clip models: ViT-B-16 or ViT-L-14.")


def print_flyp_summary(summary_rows):
    if not summary_rows:
        return
    print("\n=== FLYP Summary ===")
    print("| Split         | Top-1  | F1-macro |")
    print("| ------------- | ------ | -------- |")
    for dataset_name, top1, f1_macro in summary_rows:
        top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
        f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
        print(f"| {dataset_name:<13} | {top1_text:<6} | {f1_text:<8} |")


def clone_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parse_wise_alphas(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    return [float(value.strip()) for value in str(raw_value).split(",") if value.strip()]


def load_wise_interpolated_state(model, finetuned_state_dict, zeroshot_state_dict, alpha):
    interpolated = wise_interpolate_state_dict(finetuned_state_dict, zeroshot_state_dict, alpha)
    model.load_state_dict(interpolated)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.template is None:
        args.template = "iwildcam_template"
    ensure_open_clip_for_flyp(args.model)

    clip_encoder = CLIPEncoder(args, keep_lang=True).to(args.device)
    train_dataset_class = getattr(datasets, args.train_dataset)
    train_data = train_dataset_class(
        clip_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    model = maybe_data_parallel(clip_encoder, args)
    if hasattr(model, "module"):
        print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices")
    zeroshot_state_dict = clone_state_dict(unwrap_model(model))

    if args.load is not None:
        loaded = CLIPEncoder.load(args.load)
        model = maybe_data_parallel(loaded.to(args.device), args)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )
    effective_batches_per_epoch = len(train_data.train_loader)
    if args.max_train_batches is not None and args.max_train_batches > 0:
        effective_batches_per_epoch = min(effective_batches_per_epoch, args.max_train_batches)
    total_steps = max(args.epochs * effective_batches_per_epoch, 1)
    scheduler = build_step_lr_scheduler(optimizer, args, total_steps)
    requested_precision = getattr(args, "maple_precision", "amp")
    use_amp = str(args.device).startswith("cuda") and requested_precision in ("amp", "fp32")
    args.use_amp = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    args.training_method = "flyp"
    wandb = init_wandb(args)
    template_fns = getattr(templates, args.template)
    best_score = None
    best_epoch = None
    best_checkpoint_path = resolve_flyp_best_checkpoint_path(args)
    wise_alphas = parse_wise_alphas(getattr(args, "wise_alphas", None))
    selected_wise_alpha = getattr(args, "wise_eval_alpha", None)
    if selected_wise_alpha is not None and not (0.0 <= selected_wise_alpha <= 1.0):
        raise ValueError(f"--wise-eval-alpha must be in [0, 1], got {selected_wise_alpha}")
    val_dataset = None
    needs_val_dataset = args.epochs > 0 or bool(wise_alphas)
    if args.val_dataset is not None and needs_val_dataset:
        val_dataset = build_eval_dataset(args.val_dataset, unwrap_model(model), args, allow_ood_hp_subsample=True)

    for epoch in range(1, args.epochs + 1):
        stats = train_flyp_one_epoch(
            model,
            train_data.train_loader,
            optimizer,
            args,
            train_data.classnames,
            template_fns,
            epoch,
            init_state_dict=zeroshot_state_dict,
            drm_weight=getattr(args, "drm_weight", 0.0),
            scheduler=scheduler,
            scaler=scaler,
        )
        print(f"Epoch {epoch}: loss={stats.loss:.4f}")
        if wandb is not None:
            wandb.log({
                "train/epoch_loss": stats.loss,
                "train/lr": stats.lr,
                "epoch": epoch,
            })

        if val_dataset is not None:
            print(f"Validating FLYP on {args.val_dataset} at epoch {epoch}...")
            val_results = eval_flyp_single_dataset(model, val_dataset, args)
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
                unwrap_model(model).save(best_checkpoint_path)
                print(f"Saved best FLYP encoder to {best_checkpoint_path} (epoch {best_epoch}, {args.best_metric}={best_score:.4f})")
                if wandb is not None:
                    wandb.log({
                        "val/best_epoch": best_epoch,
                        f"val/best_{args.best_metric}": best_score,
                    })

    # Reload best checkpoint before WiSE interpolation and final eval.
    # Without this, WiSE base and final eval use the last epoch, not the
    # best-validation checkpoint.
    if best_epoch is not None and not getattr(args, "no_load_best_for_eval", False):
        print(f"Reloading best FLYP checkpoint from epoch {best_epoch} for final eval...")
        loaded = CLIPEncoder.load(best_checkpoint_path)
        model = maybe_data_parallel(loaded.to(args.device), args)

    finetuned_state_dict = clone_state_dict(unwrap_model(model))
    if wise_alphas:
        if val_dataset is None:
            raise ValueError("--wise-alphas requires a validation dataset for alpha selection.")
        wise_best_score = None
        selected_wise_alpha = None
        for alpha in wise_alphas:
            load_wise_interpolated_state(unwrap_model(model), finetuned_state_dict, zeroshot_state_dict, alpha)
            print(f"Validating WiSE-FT alpha={alpha:g} on {args.val_dataset}...")
            wise_results = eval_flyp_single_dataset(model, val_dataset, args)
            wise_score = get_validation_score(wise_results, args.best_metric)
            print(f"  WiSE alpha={alpha:g} {args.best_metric}: {wise_score:.4f}")
            if wandb is not None:
                wandb.log({
                    "wise/alpha": alpha,
                    f"wise/{args.val_dataset}/{args.best_metric}": wise_score,
                    f"wise/{args.val_dataset}/top1": wise_results.get("top1"),
                })
            if wise_best_score is None or wise_score > wise_best_score:
                wise_best_score = wise_score
                selected_wise_alpha = alpha
        print(f"Selected WiSE-FT alpha={selected_wise_alpha:g} ({args.best_metric}={wise_best_score:.4f})")
        if wandb is not None:
            wandb.log({"wise/best_alpha": selected_wise_alpha, f"wise/best_{args.best_metric}": wise_best_score})

    if selected_wise_alpha is not None:
        load_wise_interpolated_state(unwrap_model(model), finetuned_state_dict, zeroshot_state_dict, selected_wise_alpha)

    if args.save is not None:
        save_path = resolve_flyp_save_path(args.save)
        unwrap_model(model).save(save_path)
        print(f"Saved FLYP encoder to {save_path}")

    summary_rows = []
    if args.eval_datasets is not None:
        for dataset_name in args.eval_datasets:
            print(f"Evaluating FLYP on {dataset_name}...")
            eval_dataset = build_eval_dataset(dataset_name, unwrap_model(model), args)
            results = eval_flyp_single_dataset(model, eval_dataset, args)
            top1 = results.get("top1")
            f1_macro = results.get("F1-macro_all")
            print(f"  {dataset_name} Top-1 accuracy: {top1:.4f}")
            if f1_macro is not None:
                print(f"  {dataset_name} F1-macro_all: {f1_macro:.4f}")
            summary_rows.append((dataset_name, top1, f1_macro))

    print_flyp_summary(summary_rows)
    log_wandb_summary(wandb, summary_rows)


if __name__ == "__main__":
    main(parse_arguments())
