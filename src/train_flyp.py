import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

import src.datasets as datasets
import src.templates as templates
from src.config import parse_arguments
from src.models.clip_encoder import CLIPEncoder
from src.models.btel import BurstBatchSampler
from src.models.btel_artifacts import audit_sequences, build_btel_artifacts, print_sequence_audit, validate_btel_validation_split
from src.models.coop import maybe_data_parallel, unwrap_model
from src.models.flyp import eval_flyp_single_dataset, train_flyp_one_epoch, wise_interpolate_state_dict
from src.models.tail_prototype import build_class_prototypes_from_loader
from src.models.zeroshot import get_zeroshot_classifier
from src.train_coop import build_eval_dataset, get_validation_score, log_wandb_summary
from src.train_maple_full import build_step_lr_scheduler, init_wandb
from src.training_determinism import (
    build_determinism_receipt,
    configure_training_determinism,
    make_torch_generator,
    seed_data_loader_worker,
    write_json_receipt_refusing_overwrite,
)


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


def initialize_flyp_model(args):
    model = maybe_data_parallel(CLIPEncoder(args, keep_lang=True).to(args.device), args)
    if hasattr(model, "module"):
        print(f"Using DataParallel on {torch.cuda.device_count()} CUDA devices")
    if args.load is not None:
        loaded = CLIPEncoder.load(args.load)
        model = maybe_data_parallel(loaded.to(args.device), args)
    anchor_state_dict = clone_state_dict(unwrap_model(model))
    return model, anchor_state_dict


def parse_wise_alphas(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    return [float(value.strip()) for value in str(raw_value).split(",") if value.strip()]


def final_eval_datasets(args):
    eval_datasets = list(args.eval_datasets or [])
    validation_split = getattr(args, "val_dataset", None)
    used_for_tuning = validation_split is not None and (
        getattr(args, "epochs", 0) > 0 or bool(parse_wise_alphas(getattr(args, "wise_alphas", None)))
    )
    if not used_for_tuning:
        return eval_datasets
    return [dataset_name for dataset_name in eval_datasets if dataset_name != validation_split]


def load_wise_interpolated_state(model, finetuned_state_dict, zeroshot_state_dict, alpha):
    interpolated = wise_interpolate_state_dict(finetuned_state_dict, zeroshot_state_dict, alpha)
    model.load_state_dict(interpolated)


def should_use_flyp_amp(args):
    return str(args.device).startswith("cuda") and getattr(args, "maple_precision", "fp32") == "amp"


def maybe_build_tail_teacher_model(args):
    tail_weight = float(getattr(args, "tail_proto_weight", 0.0) or 0.0)
    tail_objective = getattr(args, "tail_proto_objective", "ce")
    if tail_weight == 0.0 or tail_objective != "fixed_distill":
        return None
    teacher_path = getattr(args, "tail_proto_teacher_load", None)
    if teacher_path is None:
        raise ValueError("--tail-proto-teacher-load is required when --tail-proto-objective=fixed_distill.")

    print(f"Loading fixed Tail-Aware FLYP teacher from {teacher_path}...")
    teacher_model = maybe_data_parallel(CLIPEncoder.load(teacher_path).to(args.device), args)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    return teacher_model


def maybe_prepare_btel(model, train_data, args):
    btel_weight = float(getattr(args, "btel_weight", 0.0) or 0.0)
    if btel_weight == 0.0:
        return train_data.train_loader, None
    max_batches = getattr(args, "max_train_batches", None)
    if max_batches is not None and max_batches <= 0:
        max_batches = None
    print(
        "Building frozen BTEL artifacts "
        f"(scale={getattr(args, 'btel_prototype_scale', 50.0):g}, "
        f"negative_quantile={getattr(args, 'btel_negative_quantile', 0.95):g})..."
    )
    artifacts = build_btel_artifacts(
        model,
        train_data.train_loader,
        train_data.train_dataset,
        classnames=train_data.classnames,
        device=args.device,
        prototype_scale=getattr(args, "btel_prototype_scale", 50.0),
        negative_quantile=getattr(args, "btel_negative_quantile", 0.95),
        max_batches=max_batches,
    )
    metadata = getattr(train_data.train_dataset, "metadata_array", None)
    if not isinstance(metadata, torch.Tensor):
        raise ValueError("BTEL requires train_dataset.metadata_array from the WILDS iWildCam split.")
    sampler = BurstBatchSampler(
        metadata,
        sequence_field_index=artifacts.sequence_field_index,
        frame_budget=args.batch_size,
        max_frames_per_sequence=getattr(args, "btel_max_frames_per_sequence", 8),
        seed=args.seed,
    )
    loader = DataLoader(
        train_data.train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        generator=make_torch_generator(args.seed),
        worker_init_fn=seed_data_loader_worker,
    )
    print(
        "Built BTEL artifacts: "
        f"{int((artifacts.class_counts > 0).sum().item())}/{len(artifacts.class_counts)} class prototypes, "
        f"{len(loader)} burst-preserving batches per epoch"
    )
    return loader, artifacts


def planned_training_steps(train_loader, args) -> int:
    """Count scheduled optimizer steps for the exact deterministic epoch batch plans."""
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    max_batches = getattr(args, "max_train_batches", None)
    if max_batches is not None and max_batches <= 0:
        max_batches = None
    if hasattr(batch_sampler, "batch_count_for_epoch"):
        epoch_batch_counts = [batch_sampler.batch_count_for_epoch(epoch) for epoch in range(1, args.epochs + 1)]
    else:
        epoch_batch_counts = [len(train_loader)] * args.epochs
    if max_batches is not None:
        epoch_batch_counts = [min(count, max_batches) for count in epoch_batch_counts]
    return max(sum(epoch_batch_counts), 1)


def print_btel_audits(train_data, val_dataset, args):
    num_classes = len(train_data.classnames)
    print_sequence_audit(
        audit_sequences(
            train_data.train_dataset,
            split_name="train",
            num_classes=num_classes,
            classnames=train_data.classnames,
        )
    )
    if val_dataset is not None:
        print_sequence_audit(
            audit_sequences(
                val_dataset.test_dataset,
                split_name=args.val_dataset,
                num_classes=num_classes,
                classnames=train_data.classnames,
            )
        )


def tail_prototype_source_model(model, tail_teacher_model, args):
    if getattr(args, "tail_proto_objective", "ce") == "fixed_distill":
        if tail_teacher_model is None:
            raise ValueError("tail_teacher_model is required for fixed_distill prototypes.")
        return tail_teacher_model
    return model


def maybe_build_tail_prototypes(model, train_data, args, tail_teacher_model=None):
    tail_weight = float(getattr(args, "tail_proto_weight", 0.0) or 0.0)
    if tail_weight == 0.0:
        return None, None

    source_model = tail_prototype_source_model(model, tail_teacher_model, args)
    print(
        "Building frozen Tail-Aware FLYP class prototypes "
        f"(scale={getattr(args, 'tail_proto_scale', 50.0):g}, weight={tail_weight:g})..."
    )
    prototypes, class_counts = build_class_prototypes_from_loader(
        source_model,
        train_data.train_loader,
        args.device,
        num_classes=len(train_data.classnames),
        max_batches=getattr(args, "tail_proto_max_batches", None),
    )
    missing_classes = int((class_counts == 0).sum().item())
    print(
        "Built Tail-Aware FLYP prototypes: "
        f"{len(class_counts) - missing_classes}/{len(class_counts)} classes covered"
    )
    if missing_classes:
        print(
            "Warning: some classes have no prototype. This is expected only for smoke tests "
            "or deliberately capped --tail-proto-max-batches runs."
        )
    return prototypes, class_counts


def maybe_build_tail_distillation_head(model, args):
    tail_weight = float(getattr(args, "tail_proto_weight", 0.0) or 0.0)
    tail_objective = getattr(args, "tail_proto_objective", "ce")
    if tail_weight == 0.0 or tail_objective not in {"distill", "fixed_distill"}:
        return None

    print("Building frozen student zero-shot head for Tail-Aware FLYP distillation...")
    classifier = get_zeroshot_classifier(args, unwrap_model(model).model).to(args.device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    return classifier


def maybe_build_tail_teacher_head(tail_teacher_model, args):
    tail_weight = float(getattr(args, "tail_proto_weight", 0.0) or 0.0)
    tail_objective = getattr(args, "tail_proto_objective", "ce")
    if tail_weight == 0.0 or tail_objective != "fixed_distill":
        return None
    if tail_teacher_model is None:
        raise ValueError("tail_teacher_model is required when --tail-proto-objective=fixed_distill.")

    print("Building frozen teacher zero-shot head for Tail-Aware FLYP distillation...")
    classifier = get_zeroshot_classifier(args, unwrap_model(tail_teacher_model).model).to(args.device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    return classifier


def main(args):
    deterministic_training = bool(getattr(args, "deterministic_training", False))
    determinism_receipt_path = getattr(args, "determinism_receipt", None)
    if deterministic_training != (determinism_receipt_path is not None):
        raise ValueError("--deterministic-training and --determinism-receipt must be used together.")
    if determinism_receipt_path is not None and Path(determinism_receipt_path).exists():
        raise FileExistsError(f"Refusing to overwrite determinism receipt: {determinism_receipt_path}")
    runtime_determinism = configure_training_determinism(args.seed, deterministic=deterministic_training)
    if args.template is None:
        args.template = "iwildcam_template"
    ensure_open_clip_for_flyp(args.model)
    if float(getattr(args, "btel_weight", 0.0) or 0.0) != 0.0:
        validate_btel_validation_split(args.val_dataset)

    model, zeroshot_state_dict = initialize_flyp_model(args)
    train_dataset_class = getattr(datasets, args.train_dataset)
    train_data = train_dataset_class(
        unwrap_model(model).train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    val_dataset = None
    wise_alphas = parse_wise_alphas(getattr(args, "wise_alphas", None))
    needs_val_dataset = args.epochs > 0 or bool(wise_alphas) or getattr(args, "btel_audit_only", False)
    if args.val_dataset is not None and needs_val_dataset:
        val_dataset = build_eval_dataset(args.val_dataset, unwrap_model(model), args, allow_ood_hp_subsample=True)
    if float(getattr(args, "btel_weight", 0.0) or 0.0) != 0.0 or getattr(args, "btel_audit_only", False):
        print_btel_audits(train_data, val_dataset, args)
    if getattr(args, "btel_audit_only", False):
        return
    train_loader, btel_artifacts = maybe_prepare_btel(model, train_data, args)
    tail_teacher_model = maybe_build_tail_teacher_model(args)
    tail_prototypes, tail_class_counts = maybe_build_tail_prototypes(model, train_data, args, tail_teacher_model=tail_teacher_model)
    tail_zeroshot_classifier = maybe_build_tail_distillation_head(model, args)
    tail_teacher_classifier = maybe_build_tail_teacher_head(tail_teacher_model, args)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )
    total_steps = planned_training_steps(train_loader, args)
    scheduler = build_step_lr_scheduler(optimizer, args, total_steps)
    use_amp = should_use_flyp_amp(args)
    args.use_amp = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    args.training_method = "flyp"
    wandb = init_wandb(args)
    template_fns = getattr(templates, args.template)
    best_score = None
    best_epoch = None
    best_checkpoint_path = resolve_flyp_best_checkpoint_path(args)
    selected_wise_alpha = getattr(args, "wise_eval_alpha", None)
    amp_skipped_step_count = 0
    if selected_wise_alpha is not None and not (0.0 <= selected_wise_alpha <= 1.0):
        raise ValueError(f"--wise-eval-alpha must be in [0, 1], got {selected_wise_alpha}")
    for epoch in range(1, args.epochs + 1):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        stats = train_flyp_one_epoch(
            model,
            train_loader,
            optimizer,
            args,
            train_data.classnames,
            template_fns,
            epoch,
            init_state_dict=zeroshot_state_dict,
            drm_weight=getattr(args, "drm_weight", 0.0),
            tail_prototypes=tail_prototypes,
            tail_class_counts=tail_class_counts,
            tail_zeroshot_classifier=tail_zeroshot_classifier,
            tail_teacher_model=tail_teacher_model,
            tail_teacher_classifier=tail_teacher_classifier,
            btel_artifacts=btel_artifacts,
            scheduler=scheduler,
            scaler=scaler,
        )
        print(f"Epoch {epoch}: loss={stats.loss:.4f}")
        amp_skipped_step_count += stats.optimizer_skipped_step_count
        if wandb is not None:
            wandb.log({
                "train/epoch_loss": stats.loss,
                "train/clip_loss": stats.clip_loss,
                "train/drm_loss": stats.drm_loss,
                "train/tail_loss": stats.tail_loss,
                "train/btel_loss": stats.btel_loss,
                "train/drm_to_clip_ratio": stats.drm_to_clip_ratio,
                "train/tail_to_clip_ratio": stats.tail_to_clip_ratio,
                "train/btel_to_clip_ratio": stats.btel_to_clip_ratio,
                "train/drm_effective_weight": stats.drm_effective_weight,
                "train/tail_proto_weight": stats.tail_proto_weight,
                "train/tail_proto_scale": stats.tail_proto_scale,
                "train/tail_proto_objective": stats.tail_proto_objective,
                "train/btel_weight": stats.btel_weight,
                "train/btel_prototype_scale": stats.btel_prototype_scale,
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
    eval_dataset_names = final_eval_datasets(args)
    if eval_dataset_names:
        for dataset_name in eval_dataset_names:
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
    if determinism_receipt_path is not None:
        receipt = build_determinism_receipt(
            args,
            runtime_determinism,
            best_epoch=best_epoch,
            selected_wise_alpha=selected_wise_alpha,
            amp_skipped_step_count=amp_skipped_step_count,
        )
        output_path = write_json_receipt_refusing_overwrite(determinism_receipt_path, receipt)
        print(f"Wrote immutable training determinism receipt to {output_path}")


if __name__ == "__main__":
    main(parse_arguments())
