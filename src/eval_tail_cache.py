import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import src.datasets as datasets
from src.device import resolve_device_choice
from src.datasets.dataloader import maybe_dictionarize
from src.models.clip_encoder import CLIPEncoder
from src.models.coop import maybe_data_parallel, unwrap_model
from src.models.flyp import get_cached_flyp_zeroshot_classifier, wise_interpolate_state_dict
from src.models.logit_adjustment import apply_logit_adjustment, get_metric_for_tau_selection
from src.models.stmp_adapter import (
    apply_sequence_consensus,
    build_multi_prototypes,
    metadata_fields_from_dataset,
    multi_prototype_logits,
    print_metadata_audit,
    resolve_metadata_field_index,
)
from src.models.tail_prototype import apply_tail_class_weights, tail_class_weights
from src.train_coop import build_eval_dataset, log_wandb_summary
from src.train_flyp import clone_state_dict, ensure_open_clip_for_flyp
from src.train_maple_full import init_wandb


def parse_float_grid(raw_value):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("grid must include at least one numeric value.")
    return [float(value) for value in values]


def parse_int_grid(raw_value):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("grid must include at least one integer value.")
    return [int(value) for value in values]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Eval-only tail prototype/cache adapter for FLYP checkpoints.")
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--train-dataset", type=str, default="IWildCam")
    parser.add_argument("--val-dataset", type=str, default="IWildCamVal")
    parser.add_argument("--eval-datasets", type=lambda value: value.split(","), required=True)
    parser.add_argument("--template", type=str, default="iwildcam_template")
    parser.add_argument("--data-location", type=str, default="~/data")
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu", "xla"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--wise-eval-alpha", type=float, default=None)
    parser.add_argument("--prototype-scale-grid", type=str, default="0,1,2,5,10,20,50,100")
    parser.add_argument("--cache-tau-grid", type=str, default="0,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--tail-gamma-grid", type=str, default="0")
    parser.add_argument("--tail-weight-max", type=float, default=5.0)
    parser.add_argument("--gate-mode-grid", type=str, default="none")
    parser.add_argument("--gate-strength-grid", type=str, default="0")
    parser.add_argument("--sequence-consensus-grid", type=str, default="0")
    parser.add_argument("--sequence-id-field", type=str, default="auto")
    parser.add_argument("--multi-prototype-k-grid", type=str, default="1")
    parser.add_argument("--multi-prototype-reduction", type=str, default="max", choices=["max", "logsumexp"])
    parser.add_argument("--audit-metadata", action="store_true")
    parser.add_argument("--report-key-ablation-candidates", action="store_true")
    parser.add_argument("--cd-path", type=str, default=None, help="Optional DRM concept-description JSON for concept ablations.")
    parser.add_argument("--concept-beta-grid", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--max-cache-examples-per-class", type=int, default=0)
    parser.add_argument("--best-metric", type=str, default="F1-macro_all")
    parser.add_argument("--num-ood-hp-examples", type=int, default=-1)
    parser.add_argument("--class-balanced-ood", action="store_true")
    parser.add_argument("--wandb", dest="wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=False)
    parser.add_argument("--wandb-project", type=str, default="PoorFrogs")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    args.data_location = os.path.expanduser(args.data_location)
    args.device = resolve_device_choice(args.device)
    return args


def clone_args(args):
    return copy.copy(args)


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def maybe_apply_wise(model, args):
    if args.wise_eval_alpha is None:
        return model
    if not 0.0 <= args.wise_eval_alpha <= 1.0:
        raise ValueError(f"--wise-eval-alpha must be in [0, 1], got {args.wise_eval_alpha}.")
    print(f"Applying WiSE alpha={args.wise_eval_alpha:g}")
    finetuned_state_dict = clone_state_dict(unwrap_model(model))
    anchor_args = clone_args(args)
    anchor_args.load = None
    anchor = CLIPEncoder(anchor_args, keep_lang=True).to(args.device)
    zeroshot_state_dict = clone_state_dict(anchor)
    del anchor
    interpolated = wise_interpolate_state_dict(finetuned_state_dict, zeroshot_state_dict, args.wise_eval_alpha)
    unwrap_model(model).load_state_dict(interpolated)
    return model


def build_train_dataset(args, clip_encoder):
    dataset_class = getattr(datasets, args.train_dataset)
    return dataset_class(
        clip_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


def extract_features(model, loader, args, desc, is_train=False):
    encoder = unwrap_model(model)
    encoder.eval()
    features = []
    labels = []
    metadata = []
    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(loader, desc=desc)):
            max_batches = args.max_train_batches if is_train else args.max_eval_batches
            if max_batches is not None and batch_index >= max_batches:
                break
            data = maybe_dictionarize(data)
            images = data["images"].to(args.device)
            batch_labels = data["labels"].to(args.device)
            batch_features = normalize_features(encoder(images))
            features.append(batch_features.cpu())
            labels.append(batch_labels.cpu())
            if "metadata" in data:
                metadata.extend(data["metadata"])
            elif "image_paths" in data:
                metadata.extend(data["image_paths"])
    if not features:
        raise RuntimeError(f"No features were extracted for {desc}.")
    return {
        "features": torch.cat(features).float(),
        "labels": torch.cat(labels).long(),
        "metadata": metadata,
    }


def select_cache_examples(features, labels, num_classes, max_per_class, seed):
    if max_per_class is None or max_per_class <= 0:
        return features, labels
    rng = np.random.default_rng(seed)
    selected = []
    labels_np = labels.numpy()
    for class_index in range(num_classes):
        class_indices = np.where(labels_np == class_index)[0]
        if len(class_indices) == 0:
            continue
        if len(class_indices) > max_per_class:
            class_indices = rng.choice(class_indices, size=max_per_class, replace=False)
        selected.extend(class_indices.tolist())
    if not selected:
        raise RuntimeError("No cache examples were selected.")
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return features[selected], labels[selected]


def build_class_priors(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    if total <= 0:
        raise RuntimeError("Cannot build class priors from empty labels.")
    priors = counts / total
    return counts, priors


def build_prototypes(features, labels, num_classes):
    feature_dim = features.shape[1]
    sums = torch.zeros(num_classes, feature_dim, dtype=features.dtype)
    counts = torch.zeros(num_classes, dtype=features.dtype)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
    prototypes = normalize_features(prototypes)
    present = counts > 0
    return prototypes, present


def prototype_logits(features, prototypes, present_mask, beta):
    logits = float(beta) * normalize_features(features) @ prototypes.t()
    if not present_mask.all():
        logits[:, ~present_mask] = torch.finfo(logits.dtype).min
    return logits


def concept_logits(features, classification_head):
    if classification_head is None:
        return None
    classification_head = classification_head.cpu().eval()
    with torch.no_grad():
        return classification_head(features).float()


def default_logits(features, classification_head):
    classification_head = classification_head.cpu().eval()
    with torch.no_grad():
        return classification_head(features).float()


def metrics_from_logits(dataset, logits, labels, metadata, args):
    preds = logits.argmax(dim=1)
    metrics = {"top1": (preds == labels).float().mean().item()}
    if hasattr(dataset, "post_loop_metrics"):
        wilds_metrics = dataset.post_loop_metrics(labels, logits, metadata, args)
        metrics.update(wilds_metrics)
        if "acc" in metrics:
            metrics["top1"] = metrics["acc"]
    return metrics


def confidence_gate(base_logits, mode, strength):
    strength = float(strength)
    if mode == "none" or strength == 0.0:
        return torch.ones(base_logits.shape[0], 1, dtype=base_logits.dtype)
    probs = F.softmax(base_logits, dim=1)
    if mode == "entropy":
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(base_logits.shape[1], device=base_logits.device, dtype=base_logits.dtype))
        uncertainty = entropy / max_entropy.clamp_min(1e-12)
    elif mode == "margin":
        top2 = probs.topk(k=2, dim=1).values
        uncertainty = 1.0 - (top2[:, 0] - top2[:, 1])
    else:
        raise ValueError(f"Unsupported gate mode: {mode}")
    return (1.0 + strength * (uncertainty.clamp(0.0, 1.0) - 0.5)).clamp_min(0.0).unsqueeze(1)


def build_candidate_predictions(base_logits, prototype_raw_logits_by_k, concept_raw_logits, class_priors, tail_weights_by_gamma, row, metadata=None, sequence_field_index=None):
    head = row["head"]
    if head == "default":
        return base_logits

    prototype_scale = float(row.get("prototype_scale", 0.0))
    tau = float(row.get("tau", 0.0))
    tail_gamma = float(row.get("tail_gamma", 0.0))
    prototype_k = int(row.get("prototype_k", 1))
    sequence_eta = float(row.get("sequence_eta", 0.0))
    gate_mode = row.get("gate_mode", "none")
    gate_strength = float(row.get("gate_strength", 0.0))
    prototype_raw_logits = prototype_raw_logits_by_k[prototype_k]
    weighted_prototype_logits = apply_tail_class_weights(prototype_raw_logits, tail_weights_by_gamma[tail_gamma])
    gate = confidence_gate(base_logits, gate_mode, gate_strength)
    prototype_combo_logits = base_logits + prototype_scale * gate * weighted_prototype_logits
    prototype_combo_logits = apply_logit_adjustment(prototype_combo_logits, class_priors, tau)
    prototype_combo_logits = apply_sequence_consensus(prototype_combo_logits, metadata, sequence_field_index, sequence_eta)

    if head in {"prototype", "prototype_tau"}:
        return prototype_combo_logits

    if concept_raw_logits is None:
        raise ValueError(f"{head} requires --cd-path.")

    concept_beta = float(row.get("concept_beta", 0.5))
    concept_probs = F.softmax(concept_raw_logits, dim=1)
    if head == "concept":
        base_probs = F.softmax(base_logits, dim=1)
        return concept_beta * base_probs + (1.0 - concept_beta) * concept_probs
    if head == "concept_prototype":
        prototype_probs = F.softmax(prototype_combo_logits, dim=1)
        return concept_beta * prototype_probs + (1.0 - concept_beta) * concept_probs

    raise ValueError(f"Unsupported candidate head: {head}")


def evaluate_candidate(dataset, base_logits, prototype_raw_logits_by_k, concept_raw_logits, labels, metadata, args, class_priors, tail_weights_by_gamma, row, sequence_field_index=None):
    predictions = build_candidate_predictions(
        base_logits,
        prototype_raw_logits_by_k,
        concept_raw_logits,
        class_priors,
        tail_weights_by_gamma,
        row,
        metadata=metadata,
        sequence_field_index=sequence_field_index,
    )
    return metrics_from_logits(dataset, predictions, labels, metadata, args)


def make_candidate_rows(prototype_scale_grid, tau_grid, tail_gamma_grid, gate_mode_grid, gate_strength_grid, sequence_eta_grid, prototype_k_grid, concept_beta_grid, include_concept):
    rows = [{
        "head": "default",
        "prototype_scale": 0.0,
        "tau": 0.0,
        "tail_gamma": 0.0,
        "prototype_k": 1,
        "sequence_eta": 0.0,
        "gate_mode": "none",
        "gate_strength": 0.0,
        "concept_beta": None,
    }]
    for scale in prototype_scale_grid:
        for prototype_k in prototype_k_grid:
            for sequence_eta in sequence_eta_grid:
                for tail_gamma in tail_gamma_grid:
                    for gate_mode in gate_mode_grid:
                        for gate_strength in gate_strength_grid:
                            rows.append({
                                "head": "prototype",
                                "prototype_scale": float(scale),
                                "tau": 0.0,
                                "tail_gamma": float(tail_gamma),
                                "prototype_k": int(prototype_k),
                                "sequence_eta": float(sequence_eta),
                                "gate_mode": gate_mode,
                                "gate_strength": float(gate_strength),
                                "concept_beta": None,
                            })
    for scale in prototype_scale_grid:
        for prototype_k in prototype_k_grid:
            for sequence_eta in sequence_eta_grid:
                for tau in tau_grid:
                    for tail_gamma in tail_gamma_grid:
                        for gate_mode in gate_mode_grid:
                            for gate_strength in gate_strength_grid:
                                rows.append({
                                    "head": "prototype_tau",
                                    "prototype_scale": float(scale),
                                    "tau": float(tau),
                                    "tail_gamma": float(tail_gamma),
                                    "prototype_k": int(prototype_k),
                                    "sequence_eta": float(sequence_eta),
                                    "gate_mode": gate_mode,
                                    "gate_strength": float(gate_strength),
                                    "concept_beta": None,
                                })
    if include_concept:
        for concept_beta in concept_beta_grid:
            rows.append({
                "head": "concept",
                "prototype_scale": 0.0,
                "tau": 0.0,
                "tail_gamma": 0.0,
                "prototype_k": 1,
                "sequence_eta": 0.0,
                "gate_mode": "none",
                "gate_strength": 0.0,
                "concept_beta": float(concept_beta),
            })
        for concept_beta in concept_beta_grid:
            for scale in prototype_scale_grid:
                for tau in tau_grid:
                    for prototype_k in prototype_k_grid:
                        for sequence_eta in sequence_eta_grid:
                            for tail_gamma in tail_gamma_grid:
                                for gate_mode in gate_mode_grid:
                                    for gate_strength in gate_strength_grid:
                                        rows.append({
                                            "head": "concept_prototype",
                                            "prototype_scale": float(scale),
                                            "tau": float(tau),
                                            "tail_gamma": float(tail_gamma),
                                            "prototype_k": int(prototype_k),
                                            "sequence_eta": float(sequence_eta),
                                            "gate_mode": gate_mode,
                                            "gate_strength": float(gate_strength),
                                            "concept_beta": float(concept_beta),
                                        })
    return rows


def select_adapter_params(dataset, base_logits, prototype_raw_logits_by_k, concept_raw_logits, labels, metadata, args, prototype_scale_grid, tau_grid, tail_gamma_grid, gate_mode_grid, gate_strength_grid, sequence_eta_grid, prototype_k_grid, concept_beta_grid, class_priors, tail_weights_by_gamma, sequence_field_index=None):
    rows = []
    best_by_head = {}
    candidate_rows = make_candidate_rows(
        prototype_scale_grid,
        tau_grid,
        tail_gamma_grid,
        gate_mode_grid,
        gate_strength_grid,
        sequence_eta_grid,
        prototype_k_grid,
        concept_beta_grid,
        concept_raw_logits is not None,
    )
    for candidate in candidate_rows:
        results = evaluate_candidate(
            dataset,
            base_logits,
            prototype_raw_logits_by_k,
            concept_raw_logits,
            labels,
            metadata,
            args,
            class_priors,
            tail_weights_by_gamma,
            candidate,
            sequence_field_index=sequence_field_index,
        )
        row = {
            **candidate,
            "score": get_metric_for_tau_selection(results, args.best_metric),
            "top1": float(results.get("top1", 0.0)),
            "F1-macro_all": float(results.get("F1-macro_all", 0.0)) if "F1-macro_all" in results else None,
        }
        rows.append(row)
        current_best = best_by_head.get(row["head"])
        if current_best is None or row["score"] > current_best["score"]:
            best_by_head[row["head"]] = row
    if not best_by_head:
        raise RuntimeError("No cache adapter candidate was evaluated.")
    return best_by_head, rows


def print_selection(rows, best_by_head, limit=16):
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
    print("\n=== Tail Cache Ablation Selection on Validation ===")
    print("| Rank | Head              | Scale | K | Seq eta | Tau  | Tail gamma | Gate     | Strength | Concept beta | Score  | Top-1  | F1-macro |")
    print("| ---- | ----------------- | ----- | - | ------- | ---- | ---------- | -------- | -------- | ------------ | ------ | ------ | -------- |")
    for rank, row in enumerate(ranked, start=1):
        top1 = f"{row['top1'] * 100:.2f}%"
        f1 = "N/A" if row["F1-macro_all"] is None else f"{row['F1-macro_all'] * 100:.2f}%"
        concept_beta = "N/A" if row["concept_beta"] is None else f"{row['concept_beta']:g}"
        print(
            f"| {rank:<4} | {row['head']:<17} | {row['prototype_scale']:<5g} | {row['prototype_k']:<1d} | "
            f"{row['sequence_eta']:<7g} | {row['tau']:<4g} | {row['tail_gamma']:<10g} | {row['gate_mode']:<8} | {row['gate_strength']:<8g} | "
            f"{concept_beta:<12} | {row['score']:<6.4f} | {top1:<6} | {f1:<8} |"
        )

    print("\n=== Best by Ablation Head ===")
    print("| Head              | Scale | K | Seq eta | Tau  | Tail gamma | Gate     | Strength | Concept beta | Score  | Top-1  | F1-macro |")
    print("| ----------------- | ----- | - | ------- | ---- | ---------- | -------- | -------- | ------------ | ------ | ------ | -------- |")
    for head in ("default", "prototype", "prototype_tau", "concept", "concept_prototype"):
        row = best_by_head.get(head)
        if row is None:
            continue
        top1 = f"{row['top1'] * 100:.2f}%"
        f1 = "N/A" if row["F1-macro_all"] is None else f"{row['F1-macro_all'] * 100:.2f}%"
        concept_beta = "N/A" if row["concept_beta"] is None else f"{row['concept_beta']:g}"
        print(
            f"| {head:<17} | {row['prototype_scale']:<5g} | {row['prototype_k']:<1d} | "
            f"{row['sequence_eta']:<7g} | {row['tau']:<4g} | {row['tail_gamma']:<10g} | {row['gate_mode']:<8} | {row['gate_strength']:<8g} | "
            f"{concept_beta:<12} | {row['score']:<6.4f} | {top1:<6} | {f1:<8} |"
        )


def print_tail_cache_summary(summary_rows):
    print("\n=== Tail Cache Summary ===")
    print("| Split         | Head          | Top-1  | F1-macro |")
    print("| ------------- | ------------- | ------ | -------- |")
    for dataset_name, head_name, top1, f1_macro in summary_rows:
        top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
        f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
        print(f"| {dataset_name:<13} | {head_name:<13} | {top1_text:<6} | {f1_text:<8} |")


def candidate_matches(row, *, sequence_eta, prototype_k, gate_mode, gate_strength):
    return (
        row["head"] == "prototype"
        and float(row["prototype_scale"]) == 50.0
        and float(row["tau"]) == 0.0
        and float(row["tail_gamma"]) == 0.0
        and int(row["prototype_k"]) == int(prototype_k)
        and float(row["sequence_eta"]) == float(sequence_eta)
        and row["gate_mode"] == gate_mode
        and float(row["gate_strength"]) == float(gate_strength)
    )


def find_key_ablation_rows(selection_rows):
    targets = [
        ("tpa_baseline", {"sequence_eta": 0.0, "prototype_k": 1, "gate_mode": "none", "gate_strength": 0.0}),
        ("sequence_only", {"sequence_eta": 0.5, "prototype_k": 1, "gate_mode": "none", "gate_strength": 0.0}),
        ("selected_gate", {"sequence_eta": 0.5, "prototype_k": 1, "gate_mode": "margin", "gate_strength": 0.25}),
        ("multiproto_sanity", {"sequence_eta": 0.5, "prototype_k": 8, "gate_mode": "entropy", "gate_strength": 1.0}),
    ]
    rows = []
    for label, target in targets:
        match = next((row for row in selection_rows if candidate_matches(row, **target)), None)
        if match is not None:
            rows.append((label, match))
    return rows


def print_key_ablation_summary(summary_rows):
    if not summary_rows:
        return
    print("\n=== Key STMP Final Ablation Candidates ===")
    print("| Split         | Candidate        | K | Seq eta | Gate   | Strength | Val F1 | Top-1  | F1-macro |")
    print("| ------------- | ---------------- | - | ------- | ------ | -------- | ------ | ------ | -------- |")
    for dataset_name, label, row, top1, f1_macro in summary_rows:
        val_f1 = "N/A" if row["F1-macro_all"] is None else f"{row['F1-macro_all'] * 100:.2f}%"
        top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
        f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
        print(
            f"| {dataset_name:<13} | {label:<16} | {row['prototype_k']:<1d} | "
            f"{row['sequence_eta']:<7g} | {row['gate_mode']:<6} | {row['gate_strength']:<8g} | "
            f"{val_f1:<6} | {top1_text:<6} | {f1_text:<8} |"
        )


def main(args):
    if args.load is None:
        raise ValueError("--load must point to a FLYP CLIPEncoder checkpoint.")
    if args.val_dataset == "IWildCamOOD":
        raise ValueError("IWildCamOOD is final-test-only and cannot be used for cache hyperparameter selection.")
    ensure_open_clip_for_flyp(args.model)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.training_method = "tail_cache_eval"
    wandb = init_wandb(args)

    print(f"Loading fine-tuned FLYP checkpoint from {args.load}")
    model = maybe_data_parallel(CLIPEncoder.load(args.load).to(args.device), args)
    model = maybe_apply_wise(model, args)
    encoder = unwrap_model(model)

    train_data = build_train_dataset(args, encoder)
    num_classes = len(train_data.classnames)
    if args.audit_metadata:
        print_metadata_audit("train", train_data, args.sequence_id_field)
    train_features = extract_features(model, train_data.train_loader, args, "Tail cache train features", is_train=True)
    cache_features, cache_labels = select_cache_examples(
        train_features["features"],
        train_features["labels"],
        num_classes,
        args.max_cache_examples_per_class,
        args.seed,
    )
    counts, class_priors = build_class_priors(train_features["labels"], num_classes)
    prototypes, present_mask = build_prototypes(cache_features, cache_labels, num_classes)
    prototype_k_grid = parse_int_grid(args.multi_prototype_k_grid)
    if any(k <= 0 for k in prototype_k_grid):
        raise ValueError("--multi-prototype-k-grid values must be positive.")
    max_prototype_k = max(prototype_k_grid)
    multi_prototypes, multi_prototype_mask = build_multi_prototypes(
        cache_features,
        cache_labels,
        num_classes,
        max_prototype_k,
    )
    print(f"Built tail cache with {cache_features.shape[0]} examples across {int((counts > 0).sum().item())}/{num_classes} classes.")

    classification_head = get_cached_flyp_zeroshot_classifier(args, encoder)
    concept_head = None
    if args.cd_path is not None:
        from src.eval_drm_blend import build_drm_classifiers

        concept_args = clone_args(args)
        concept_args.cd_beta = 0.5
        _, concept_head = build_drm_classifiers(concept_args, encoder.model)
    prototype_scale_grid = parse_float_grid(args.prototype_scale_grid)
    tau_grid = parse_float_grid(args.cache_tau_grid)
    tail_gamma_grid = parse_float_grid(args.tail_gamma_grid)
    gate_mode_grid = [item.strip() for item in args.gate_mode_grid.split(",") if item.strip()]
    gate_strength_grid = parse_float_grid(args.gate_strength_grid)
    sequence_eta_grid = parse_float_grid(args.sequence_consensus_grid)
    concept_beta_grid = parse_float_grid(args.concept_beta_grid)
    tail_weights_by_gamma = {
        float(gamma): tail_class_weights(counts, gamma, max_weight=args.tail_weight_max)
        for gamma in tail_gamma_grid
    }

    val_dataset = build_eval_dataset(args.val_dataset, encoder, args, allow_ood_hp_subsample=True)
    sequence_field_index = print_metadata_audit(args.val_dataset, val_dataset, args.sequence_id_field) if args.audit_metadata else resolve_metadata_field_index(
        metadata_fields_from_dataset(val_dataset),
        args.sequence_id_field,
        ["seq_id", "sequence_id", "sequence"],
    )
    val_features = extract_features(model, val_dataset.test_loader, args, f"{args.val_dataset} features")
    val_base_logits = default_logits(val_features["features"], classification_head)
    val_prototype_logits_by_k = {
        1: prototype_logits(val_features["features"], prototypes, present_mask, beta=1.0)
    }
    for prototype_k in prototype_k_grid:
        if prototype_k == 1:
            continue
        val_prototype_logits_by_k[prototype_k] = multi_prototype_logits(
            val_features["features"],
            multi_prototypes[:, :prototype_k],
            multi_prototype_mask[:, :prototype_k],
            beta=1.0,
            reduction=args.multi_prototype_reduction,
        )
    val_concept_logits = concept_logits(val_features["features"], concept_head)
    best_by_head, selection_rows = select_adapter_params(
        val_dataset,
        val_base_logits,
        val_prototype_logits_by_k,
        val_concept_logits,
        val_features["labels"],
        val_features["metadata"],
        args,
        prototype_scale_grid,
        tau_grid,
        tail_gamma_grid,
        gate_mode_grid,
        gate_strength_grid,
        sequence_eta_grid,
        prototype_k_grid,
        concept_beta_grid,
        class_priors,
        tail_weights_by_gamma,
        sequence_field_index=sequence_field_index,
    )
    print_selection(selection_rows, best_by_head)
    key_ablation_rows = find_key_ablation_rows(selection_rows) if args.report_key_ablation_candidates else []

    summary_rows = []
    key_ablation_summary_rows = []
    for dataset_name in args.eval_datasets:
        print(f"Evaluating tail cache on {dataset_name}...")
        dataset = build_eval_dataset(dataset_name, encoder, args)
        if args.audit_metadata and dataset_name != args.val_dataset:
            print_metadata_audit(dataset_name, dataset, args.sequence_id_field)
        features = val_features if dataset_name == args.val_dataset else extract_features(model, dataset.test_loader, args, f"{dataset_name} features")
        base_logits = val_base_logits if dataset_name == args.val_dataset else default_logits(features["features"], classification_head)
        if dataset_name == args.val_dataset:
            proto_logits_by_k = val_prototype_logits_by_k
        else:
            proto_logits_by_k = {1: prototype_logits(features["features"], prototypes, present_mask, beta=1.0)}
            for prototype_k in prototype_k_grid:
                if prototype_k == 1:
                    continue
                proto_logits_by_k[prototype_k] = multi_prototype_logits(
                    features["features"],
                    multi_prototypes[:, :prototype_k],
                    multi_prototype_mask[:, :prototype_k],
                    beta=1.0,
                    reduction=args.multi_prototype_reduction,
                )
        cd_logits = val_concept_logits if dataset_name == args.val_dataset else concept_logits(features["features"], concept_head)
        for head_name in ("default", "prototype", "prototype_tau", "concept", "concept_prototype"):
            best = best_by_head.get(head_name)
            if best is None:
                continue
            results = evaluate_candidate(
                dataset,
                base_logits,
                proto_logits_by_k,
                cd_logits,
                features["labels"],
                features["metadata"],
                args,
                class_priors,
                tail_weights_by_gamma,
                best,
                sequence_field_index=sequence_field_index,
            )
            top1 = results.get("top1")
            f1_macro = results.get("F1-macro_all")
            print(f"  {dataset_name} {head_name} Top-1 accuracy: {top1:.4f}")
            if f1_macro is not None:
                print(f"  {dataset_name} {head_name} F1-macro_all: {f1_macro:.4f}")
            summary_rows.append((dataset_name, head_name, top1, f1_macro))
            if wandb is not None:
                wandb.log({
                    f"tail_cache/{dataset_name}/{head_name}/top1": top1,
                    f"tail_cache/{dataset_name}/{head_name}/f1_macro": f1_macro,
                    f"tail_cache/{head_name}/prototype_scale": best["prototype_scale"],
                    f"tail_cache/{head_name}/prototype_k": best["prototype_k"],
                    f"tail_cache/{head_name}/sequence_eta": best["sequence_eta"],
                    "tail_cache/best_tau": best["tau"],
                    f"tail_cache/{head_name}/tail_gamma": best["tail_gamma"],
                    f"tail_cache/{head_name}/gate_strength": best["gate_strength"],
                    f"tail_cache/{head_name}/concept_beta": best["concept_beta"],
                })
        for label, candidate in key_ablation_rows:
            results = evaluate_candidate(
                dataset,
                base_logits,
                proto_logits_by_k,
                cd_logits,
                features["labels"],
                features["metadata"],
                args,
                class_priors,
                tail_weights_by_gamma,
                candidate,
                sequence_field_index=sequence_field_index,
            )
            key_ablation_summary_rows.append((
                dataset_name,
                label,
                candidate,
                results.get("top1"),
                results.get("F1-macro_all"),
            ))

    print_tail_cache_summary(summary_rows)
    print_key_ablation_summary(key_ablation_summary_rows)
    preferred_head = "concept_prototype" if "concept_prototype" in best_by_head else "prototype_tau"
    cache_summary_rows = [(dataset, top1, f1) for dataset, head, top1, f1 in summary_rows if head == preferred_head]
    log_wandb_summary(wandb, cache_summary_rows)
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main(parse_arguments())
