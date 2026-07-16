import argparse
import hashlib
import copy
import json
import os
import random
from pathlib import Path

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
from src.models.loo_bcpd import (
    build_loo_bcpd_logits,
    build_loo_linear_logits,
    deterministic_derangement,
    shuffled_sequence_groups,
)
from src.models.sctr import apply_tail_protective_sctr
from src.models.stp_diagnostics import build_stp_diagnostics, paired_sequence_bootstrap
from src.models.tail_prototype import apply_tail_class_weights, tail_class_weights
from src.models.tip_adapter import tip_adapter_cache_logits
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


def parse_tip_adapter_grids(args):
    if args.tip_adapter_beta_grid is None and args.tip_adapter_alpha_grid is None:
        return [], []
    if args.tip_adapter_beta_grid is None or args.tip_adapter_alpha_grid is None:
        raise ValueError("Tip-Adapter requires both --tip-adapter-beta-grid and --tip-adapter-alpha-grid.")
    beta_grid = parse_float_grid(args.tip_adapter_beta_grid)
    alpha_grid = parse_float_grid(args.tip_adapter_alpha_grid)
    if any(beta <= 0.0 for beta in beta_grid):
        raise ValueError("--tip-adapter-beta-grid values must be positive.")
    if any(alpha < 0.0 for alpha in alpha_grid):
        raise ValueError("--tip-adapter-alpha-grid values must be non-negative.")
    return beta_grid, alpha_grid


def parse_tip_adapter_support_shots_grid(args, beta_grid):
    if not beta_grid:
        return []
    raw_value = getattr(args, "tip_adapter_support_shots_grid", None)
    if raw_value is None:
        return [0]
    support_shots_grid = parse_int_grid(raw_value)
    if any(shots < 0 for shots in support_shots_grid):
        raise ValueError("--tip-adapter-support-shots-grid values must be non-negative.")
    return support_shots_grid


def validate_tip_adapter_protocol(args, beta_grid, support_shots_grid=None):
    if not beta_grid:
        return
    support_shots_grid = [0] if support_shots_grid is None else support_shots_grid
    if args.val_dataset != "IWildCamVal":
        raise ValueError("The Tip-Adapter control requires --val-dataset=IWildCamVal.")
    if args.max_train_batches is not None:
        raise ValueError("The Tip-Adapter control does not allow --max-train-batches.")
    if args.max_eval_batches is not None:
        raise ValueError("The Tip-Adapter control does not allow --max-eval-batches.")
    if 0 in support_shots_grid and args.max_cache_examples_per_class > 0:
        raise ValueError("The full-data Tip-Adapter control requires --max-cache-examples-per-class=0.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Eval-only tail prototype/cache adapter for FLYP checkpoints.")
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--train-dataset", type=str, default="IWildCam")
    parser.add_argument("--val-dataset", type=str, default="IWildCamVal")
    parser.add_argument("--eval-datasets", type=lambda value: value.split(","), required=True)
    parser.add_argument("--template", type=str, default="iwildcam_template")
    parser.add_argument("--data-location", type=str, default="~/data")
    parser.add_argument("--cache-dir", type=str, default=None)
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
    parser.add_argument("--sctr-strength-grid", type=str, default="0")
    parser.add_argument("--sctr-tail-protection-grid", type=str, default="1")
    parser.add_argument("--loo-bcpd-strength-grid", type=str, default="0")
    parser.add_argument("--loo-bcpd-diagnostics-report", type=str, default=None)
    parser.add_argument("--loo-bcpd-diagnostics-split", type=str, default="IWildCamVal")
    parser.add_argument("--sequence-id-field", type=str, default="auto")
    parser.add_argument("--multi-prototype-k-grid", type=str, default="1")
    parser.add_argument("--multi-prototype-reduction", type=str, default="max", choices=["max", "logsumexp"])
    parser.add_argument("--tip-adapter-beta-grid", type=str, default=None)
    parser.add_argument("--tip-adapter-alpha-grid", type=str, default=None)
    parser.add_argument("--tip-adapter-support-shots-grid", type=str, default=None)
    parser.add_argument("--tip-adapter-query-chunk-size", type=int, default=256)
    parser.add_argument("--tip-adapter-cache-chunk-size", type=int, default=16384)
    parser.add_argument("--audit-metadata", action="store_true")
    parser.add_argument("--report-key-ablation-candidates", action="store_true")
    parser.add_argument("--stp-diagnostics-report", type=str, default=None)
    parser.add_argument("--stp-diagnostics-split", type=str, default="IWildCamOOD")
    parser.add_argument("--stp-diagnostics-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--cd-path", type=str, default=None, help="Optional DRM concept-description JSON for concept ablations.")
    parser.add_argument("--concept-beta-grid", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--max-cache-examples-per-class", type=int, default=0)
    parser.add_argument("--best-metric", type=str, default="F1-macro_all")
    parser.add_argument("--selection-output", type=str, default=None)
    parser.add_argument("--summary-head", type=str, default=None)
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


def resolve_tip_adapter_support_shots_grid(labels, num_classes, support_shots_grid):
    class_counts = torch.bincount(labels, minlength=num_classes)
    minimum_class_count = int(class_counts.min().item())
    valid_support_shots = [shots for shots in support_shots_grid if shots == 0 or shots <= minimum_class_count]
    skipped_support_shots = [shots for shots in support_shots_grid if shots not in valid_support_shots]
    if skipped_support_shots:
        print(
            "Skipping balanced Tip-Adapter support shots "
            f"{skipped_support_shots}: the smallest train class has {minimum_class_count} examples."
        )
    if not valid_support_shots:
        raise RuntimeError("No Tip-Adapter support-shot setting is available for every train class.")
    return valid_support_shots


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


def class_mapping_checksum(classnames, classifier):
    if classifier.weight.shape[0] != len(classnames):
        raise ValueError(
            f"Classifier rows ({classifier.weight.shape[0]}) do not match dataset classes ({len(classnames)})."
        )
    encoded = "\n".join(f"{index}:{name}" for index, name in enumerate(classnames)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prototype_logits(features, prototypes, present_mask, beta):
    logits = float(beta) * normalize_features(features) @ prototypes.t()
    if not present_mask.all():
        logits[:, ~present_mask] = torch.finfo(logits.dtype).min
    return logits


def build_tip_adapter_logits_by_config(features, train_features, train_labels, num_classes, beta_grid, support_shots_grid, args):
    logits_by_config = {}
    for support_shots in support_shots_grid:
        support_features, support_labels = select_cache_examples(
            train_features,
            train_labels,
            num_classes,
            support_shots,
            args.seed,
        )
        for beta in beta_grid:
            logits_by_config[(int(support_shots), float(beta))] = tip_adapter_cache_logits(
                features,
                support_features,
                support_labels,
                num_classes=num_classes,
                beta=beta,
                query_chunk_size=args.tip_adapter_query_chunk_size,
                cache_chunk_size=args.tip_adapter_cache_chunk_size,
                device=args.device,
            )
    return logits_by_config


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


def build_candidate_predictions(base_logits, prototype_raw_logits_by_k, concept_raw_logits, class_priors, tail_weights_by_gamma, row, metadata=None, sequence_field_index=None, tip_cache_logits_by_config=None, loo_bcpd_logits_by_strength=None):
    head = row["head"]
    if head == "default":
        return base_logits
    if head == "tip_adapter":
        if tip_cache_logits_by_config is None:
            raise ValueError("Tip-Adapter requires cache logits.")
        tip_support_shots = int(row["tip_support_shots"])
        tip_beta = float(row["tip_beta"])
        cache_key = (tip_support_shots, tip_beta)
        if cache_key not in tip_cache_logits_by_config:
            raise ValueError(f"No Tip-Adapter cache logits were built for support_shots={tip_support_shots}, beta={tip_beta:g}.")
        return base_logits + float(row["tip_alpha"]) * tip_cache_logits_by_config[cache_key]
    if head == "loo_bcpd":
        if loo_bcpd_logits_by_strength is None:
            raise ValueError("LOO-BCPD requires precomputed logits.")
        strength = float(row["loo_bcpd_strength"])
        if strength not in loo_bcpd_logits_by_strength:
            raise ValueError(f"No LOO-BCPD logits were built for strength={strength:g}.")
        return loo_bcpd_logits_by_strength[strength]

    prototype_scale = float(row.get("prototype_scale", 0.0))
    tau = float(row.get("tau", 0.0))
    tail_gamma = float(row.get("tail_gamma", 0.0))
    prototype_k = int(row.get("prototype_k", 1))
    sequence_eta = float(row.get("sequence_eta", 0.0))
    sctr_strength = float(row.get("sctr_strength", 0.0))
    sctr_tail_protection = float(row.get("sctr_tail_protection", 0.0))
    gate_mode = row.get("gate_mode", "none")
    gate_strength = float(row.get("gate_strength", 0.0))
    prototype_raw_logits = prototype_raw_logits_by_k[prototype_k]
    weighted_prototype_logits = apply_tail_class_weights(prototype_raw_logits, tail_weights_by_gamma[tail_gamma])
    gate = confidence_gate(base_logits, gate_mode, gate_strength)
    prototype_combo_logits = base_logits + prototype_scale * gate * weighted_prototype_logits
    prototype_combo_logits = apply_logit_adjustment(prototype_combo_logits, class_priors, tau)
    if head == "sctr":
        return apply_tail_protective_sctr(
            prototype_combo_logits,
            metadata,
            sequence_field_index,
            class_priors,
            sctr_strength,
            sctr_tail_protection,
        )
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


def evaluate_candidate(dataset, base_logits, prototype_raw_logits_by_k, concept_raw_logits, labels, metadata, args, class_priors, tail_weights_by_gamma, row, sequence_field_index=None, tip_cache_logits_by_config=None, loo_bcpd_logits_by_strength=None):
    predictions = build_candidate_predictions(
        base_logits,
        prototype_raw_logits_by_k,
        concept_raw_logits,
        class_priors,
        tail_weights_by_gamma,
        row,
        metadata=metadata,
        sequence_field_index=sequence_field_index,
        tip_cache_logits_by_config=tip_cache_logits_by_config,
        loo_bcpd_logits_by_strength=loo_bcpd_logits_by_strength,
    )
    return metrics_from_logits(dataset, predictions, labels, metadata, args)


def write_stp_diagnostics(path, split_name, labels, frame_logits, stp_logits, train_class_counts, metadata, sequence_field_index, bootstrap_samples, seed):
    diagnostics = build_stp_diagnostics(
        labels=labels,
        frame_logits=frame_logits,
        stp_logits=stp_logits,
        train_class_counts=train_class_counts,
        metadata=metadata,
        sequence_field_index=sequence_field_index,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(diagnostics.to_markdown(split_name), encoding="utf-8")
    print(f"Saved STP diagnostics to {report_path}")
    return diagnostics


def _macro_f1(labels, predictions, num_classes):
    if labels.numel() == 0:
        return 0.0
    encoded = labels * num_classes + predictions
    confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()
    true_positives = confusion.diag()
    denominators = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = 2.0 * true_positives / denominators.clamp_min(1.0)
    present = confusion.sum(dim=1) > 0
    return f1[present].mean().item() if present.any() else 0.0


def _tail_bin_rows(labels, logits_by_name, train_class_counts):
    num_classes = logits_by_name["tpa"].shape[1]
    label_counts = train_class_counts.to(device=labels.device)[labels]
    bins = (
        ("tail_1_20", label_counts <= 20),
        ("medium_21_100", (label_counts >= 21) & (label_counts <= 100)),
        ("head_101_plus", label_counts >= 101),
    )
    rows = []
    for name, mask in bins:
        active_classes = int(labels[mask].unique().numel()) if mask.any() else 0
        guard_eligible = active_classes >= 5
        metrics = {
            method: _macro_f1(labels[mask], logits.argmax(dim=1)[mask], num_classes)
            for method, logits in logits_by_name.items()
        }
        rows.append((name, int(mask.sum().item()), active_classes, guard_eligible, metrics))
    return rows


def write_loo_bcpd_diagnostics(path, split_name, dataset, args, labels, metadata, sequence_field_index, train_class_counts, class_checksum, logits_by_name, selected_result, shuffled, selected_strength, bootstrap_samples, seed):
    bcpd_predictions = logits_by_name["loo_bcpd"].argmax(dim=1)
    comparison_rows = []
    for method, logits in logits_by_name.items():
        if method == "loo_bcpd":
            continue
        bootstrap = paired_sequence_bootstrap(
            labels=labels,
            reference_predictions=logits.argmax(dim=1),
            candidate_predictions=bcpd_predictions,
            metadata=metadata,
            sequence_field_index=sequence_field_index,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        comparison_rows.append((method, bootstrap))
    metric_rows = {
        method: metrics_from_logits(dataset, logits, labels, metadata, args)
        for method, logits in logits_by_name.items()
    }
    changed_predictions = int((logits_by_name["tpa"].argmax(dim=1) != bcpd_predictions).sum().item())
    tpa_predictions = logits_by_name["tpa"].argmax(dim=1)
    corrected = int(((tpa_predictions != labels) & (bcpd_predictions == labels)).sum().item())
    regressed = int(((tpa_predictions == labels) & (bcpd_predictions != labels)).sum().item())
    lines = [
        f"# LOO-BCPD Diagnostics: {split_name}",
        "",
        "## Configuration",
        "",
        f"- Class mapping SHA-256: `{class_checksum}`.",
        f"- Selected BCPD strength: {selected_strength:g}.",
        f"- Valid bursts: {selected_result.valid_burst_count}; corrected frames: {selected_result.corrected_frame_count}.",
        f"- Burst-shuffle changed-frame fraction: {shuffled.changed_frame_fraction:.4f}; unavailable groups: {shuffled.unavailable_group_count}.",
        "",
        "## Overall",
        "",
        "| Method | Top-1 | Macro-F1 |",
        "| --- | ---: | ---: |",
    ]
    for method, metrics in metric_rows.items():
        lines.append(f"| {method} | {metrics.get('top1', 0.0) * 100:.2f} | {metrics.get('F1-macro_all', 0.0) * 100:.2f} |")
    lines.extend([
        "",
        "## Paired Sequence Bootstrap",
        "",
        "| Reference | Delta BCPD-reference | 95% CI | Positive fraction | Median class coverage | Stability samples |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for method, bootstrap in comparison_rows:
        lines.append(
            f"| {method} | {bootstrap.delta * 100:+.2f} | [{bootstrap.low * 100:+.2f}, {bootstrap.high * 100:+.2f}] | "
            f"{bootstrap.positive_delta_fraction:.3f} | {bootstrap.median_class_coverage:.3f} | {bootstrap.stability_sample_count} |"
        )
    lines.extend([
        "",
        "## Tail Bins",
        "",
        "The promotion guard applies only to bins with at least five supported validation classes; other bins are not reliable for promotion decisions.",
        "",
        "| Bin | Frames | Supported classes | Guard | TPA F1 | BCPD F1 | Delta |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    tail_guard_failures = []
    for name, frame_count, class_count, guard_eligible, metrics in _tail_bin_rows(labels, logits_by_name, train_class_counts):
        delta = metrics["loo_bcpd"] - metrics["tpa"]
        guard = "pass" if not guard_eligible or delta >= -0.02 else "fail"
        if guard == "fail":
            tail_guard_failures.append(name)
        lines.append(
            f"| {name} | {frame_count} | {class_count} | {guard} | {metrics['tpa'] * 100:.2f} | "
            f"{metrics['loo_bcpd'] * 100:.2f} | {delta * 100:+.2f} |"
        )
    lines.extend([
        "",
        "## Promotion Guard",
        "",
        "- Tail-bin guard: " + ("FAIL for " + ", ".join(tail_guard_failures) if tail_guard_failures else "PASS for all reliable bins") + ".",
        "",
        "## Aggregate Mechanism Statistics",
        "",
        f"- Mean responsibility entropy: {selected_result.mean_responsibility_entropy:.4f}.",
        f"- Mean maximum responsibility: {selected_result.mean_max_responsibility:.4f}.",
        f"- Mean effective class count: {selected_result.mean_effective_class_count:.4f}.",
        f"- Mean LOO support mass: {selected_result.mean_support_mass:.4f}.",
        f"- Mean tangent rotation: {selected_result.mean_rotation_degrees:.4f} degrees.",
        f"- Mean normalization penalty: {selected_result.mean_normalization_penalty:.4f}.",
        f"- TPA-to-BCPD changed predictions: {changed_predictions}; wrong-to-correct: {corrected}; correct-to-wrong: {regressed}.",
        "",
    ])
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LOO-BCPD diagnostics to {report_path}")


def make_candidate_rows(prototype_scale_grid, tau_grid, tail_gamma_grid, gate_mode_grid, gate_strength_grid, sequence_eta_grid, prototype_k_grid, concept_beta_grid, include_concept, sctr_strength_grid=None, sctr_tail_protection_grid=None, tip_beta_grid=None, tip_alpha_grid=None, tip_support_shots_grid=None, loo_bcpd_strength_grid=None):
    sctr_strength_grid = [0.0] if sctr_strength_grid is None else sctr_strength_grid
    sctr_tail_protection_grid = [1.0] if sctr_tail_protection_grid is None else sctr_tail_protection_grid
    rows = [{
        "head": "default",
        "prototype_scale": 0.0,
        "tau": 0.0,
        "tail_gamma": 0.0,
        "prototype_k": 1,
        "sequence_eta": 0.0,
        "gate_mode": "none",
        "gate_strength": 0.0,
        "sctr_strength": 0.0,
        "sctr_tail_protection": 0.0,
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
                                "sctr_strength": 0.0,
                                "sctr_tail_protection": 0.0,
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
                                    "sctr_strength": 0.0,
                                    "sctr_tail_protection": 0.0,
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
                "sctr_strength": 0.0,
                "sctr_tail_protection": 0.0,
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
                                            "sctr_strength": 0.0,
                                            "sctr_tail_protection": 0.0,
                                            "concept_beta": float(concept_beta),
                                        })
    for scale in prototype_scale_grid:
        for prototype_k in prototype_k_grid:
            for tail_gamma in tail_gamma_grid:
                for routing_strength in sctr_strength_grid:
                    for tail_protection in sctr_tail_protection_grid:
                        rows.append({
                            "head": "sctr",
                            "prototype_scale": float(scale),
                            "tau": 0.0,
                            "tail_gamma": float(tail_gamma),
                            "prototype_k": int(prototype_k),
                            "sequence_eta": 0.0,
                            "gate_mode": "none",
                            "gate_strength": 0.0,
                            "sctr_strength": float(routing_strength),
                            "sctr_tail_protection": float(tail_protection),
                            "concept_beta": None,
                        })
    effective_tip_support_shots_grid = [] if tip_beta_grid is None else ([0] if tip_support_shots_grid is None else tip_support_shots_grid)
    for tip_support_shots in effective_tip_support_shots_grid:
        for tip_beta in ([] if tip_beta_grid is None else tip_beta_grid):
            for tip_alpha in ([] if tip_alpha_grid is None else tip_alpha_grid):
                rows.append({
                    "head": "tip_adapter",
                    "prototype_scale": 0.0,
                    "tau": 0.0,
                    "tail_gamma": 0.0,
                    "prototype_k": 1,
                    "sequence_eta": 0.0,
                    "gate_mode": "none",
                    "gate_strength": 0.0,
                    "sctr_strength": 0.0,
                    "sctr_tail_protection": 0.0,
                    "concept_beta": None,
                    "tip_support_shots": int(tip_support_shots),
                    "tip_beta": float(tip_beta),
                    "tip_alpha": float(tip_alpha),
                })
    for strength in ([] if loo_bcpd_strength_grid is None else loo_bcpd_strength_grid):
        rows.append({
                "head": "loo_bcpd",
                "prototype_scale": 50.0,
                "tau": 0.0,
                "tail_gamma": 0.0,
                "prototype_k": 1,
                "sequence_eta": 0.0,
                "gate_mode": "none",
                "gate_strength": 0.0,
                "sctr_strength": 0.0,
                "sctr_tail_protection": 0.0,
                "concept_beta": None,
                "loo_bcpd_strength": float(strength),
        })
    for row in rows:
        row.setdefault("loo_bcpd_strength", 0.0)
    return rows


def select_adapter_params(dataset, base_logits, prototype_raw_logits_by_k, concept_raw_logits, labels, metadata, args, prototype_scale_grid, tau_grid, tail_gamma_grid, gate_mode_grid, gate_strength_grid, sequence_eta_grid, prototype_k_grid, concept_beta_grid, class_priors, tail_weights_by_gamma, sctr_strength_grid=None, sctr_tail_protection_grid=None, sequence_field_index=None, tip_beta_grid=None, tip_alpha_grid=None, tip_support_shots_grid=None, tip_cache_logits_by_config=None, loo_bcpd_strength_grid=None, loo_bcpd_logits_by_strength=None, train_class_counts=None):
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
        sctr_strength_grid=sctr_strength_grid,
        sctr_tail_protection_grid=sctr_tail_protection_grid,
        tip_beta_grid=tip_beta_grid,
        tip_alpha_grid=tip_alpha_grid,
        tip_support_shots_grid=tip_support_shots_grid,
        loo_bcpd_strength_grid=loo_bcpd_strength_grid,
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
            tip_cache_logits_by_config=tip_cache_logits_by_config,
            loo_bcpd_logits_by_strength=loo_bcpd_logits_by_strength,
        )
        row = {
            **candidate,
            "score": get_metric_for_tau_selection(results, args.best_metric),
            "top1": float(results.get("top1", 0.0)),
            "F1-macro_all": float(results.get("F1-macro_all", 0.0)) if "F1-macro_all" in results else None,
        }
        if row["head"] == "loo_bcpd" and train_class_counts is not None:
            candidate_logits = build_candidate_predictions(
                base_logits,
                prototype_raw_logits_by_k,
                concept_raw_logits,
                class_priors,
                tail_weights_by_gamma,
                candidate,
                metadata=metadata,
                sequence_field_index=sequence_field_index,
                tip_cache_logits_by_config=tip_cache_logits_by_config,
                loo_bcpd_logits_by_strength=loo_bcpd_logits_by_strength,
            )
            tail_mask = train_class_counts.to(device=labels.device)[labels] <= 20
            row["tail_macro_f1"] = _macro_f1(labels[tail_mask], candidate_logits.argmax(dim=1)[tail_mask], candidate_logits.shape[1])
        else:
            row["tail_macro_f1"] = None
        rows.append(row)
        current_best = best_by_head.get(row["head"])
        if current_best is None or row["score"] > current_best["score"] + 1e-4 or (
            abs(row["score"] - current_best["score"]) <= 1e-4
            and row["head"] == "loo_bcpd"
            and (
                row["loo_bcpd_strength"] < current_best["loo_bcpd_strength"]
                or (
                    row["loo_bcpd_strength"] == current_best["loo_bcpd_strength"]
                    and (row["tail_macro_f1"] or float("-inf")) > (current_best["tail_macro_f1"] or float("-inf"))
                )
            )
        ):
            best_by_head[row["head"]] = row
    if not best_by_head:
        raise RuntimeError("No cache adapter candidate was evaluated.")
    return best_by_head, rows


def print_selection(rows, best_by_head, limit=16):
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
    include_tip_columns = any(row["head"] == "tip_adapter" for row in rows)
    print("\n=== Tail Cache Ablation Selection on Validation ===")
    if include_tip_columns:
        print("| Rank | Head              | Scale | K | Seq eta | SCTR route | Tail protect | Tau  | Tail gamma | Gate     | Adapter strength | Concept beta | Tip shots | Tip beta | Tip alpha | Score  | Top-1  | F1-macro |")
        print("| ---- | ----------------- | ----- | - | ------- | ---------- | ------------ | ---- | ---------- | -------- | ---------------- | ------------ | --------- | -------- | --------- | ------ | ------ | -------- |")
    else:
        print("| Rank | Head              | Scale | K | Seq eta | SCTR route | Tail protect | Tau  | Tail gamma | Gate     | Adapter strength | Concept beta | Score  | Top-1  | F1-macro |")
        print("| ---- | ----------------- | ----- | - | ------- | ---------- | ------------ | ---- | ---------- | -------- | ---------------- | ------------ | ------ | ------ | -------- |")
    for rank, row in enumerate(ranked, start=1):
        top1 = f"{row['top1'] * 100:.2f}%"
        f1 = "N/A" if row["F1-macro_all"] is None else f"{row['F1-macro_all'] * 100:.2f}%"
        concept_beta = "N/A" if row["concept_beta"] is None else f"{row['concept_beta']:g}"
        adapter_strength = row.get("loo_bcpd_strength", 0.0) if row["head"] == "loo_bcpd" else row["gate_strength"]
        common = (
            f"| {rank:<4} | {row['head']:<17} | {row['prototype_scale']:<5g} | {row['prototype_k']:<1d} | "
            f"{row['sequence_eta']:<7g} | {row['sctr_strength']:<10g} | {row['sctr_tail_protection']:<12g} | {row['tau']:<4g} | {row['tail_gamma']:<10g} | {row['gate_mode']:<8} | {adapter_strength:<16g} | "
            f"{concept_beta:<12}"
        )
        if include_tip_columns:
            tip_support_shots = "full" if row.get("tip_support_shots") == 0 else str(row.get("tip_support_shots", "N/A"))
            tip_beta = "N/A" if row.get("tip_beta") is None else f"{row['tip_beta']:g}"
            tip_alpha = "N/A" if row.get("tip_alpha") is None else f"{row['tip_alpha']:g}"
            print(f"{common} | {tip_support_shots:<9} | {tip_beta:<8} | {tip_alpha:<9} | {row['score']:<6.4f} | {top1:<6} | {f1:<8} |")
        else:
            print(f"{common} | {row['score']:<6.4f} | {top1:<6} | {f1:<8} |")

    print("\n=== Best by Ablation Head ===")
    if include_tip_columns:
        print("| Head              | Scale | K | Seq eta | SCTR route | Tail protect | Tau  | Tail gamma | Gate     | Adapter strength | Concept beta | Tip shots | Tip beta | Tip alpha | Score  | Top-1  | F1-macro |")
        print("| ----------------- | ----- | - | ------- | ---------- | ------------ | ---- | ---------- | -------- | ---------------- | ------------ | --------- | -------- | --------- | ------ | ------ | -------- |")
    else:
        print("| Head              | Scale | K | Seq eta | SCTR route | Tail protect | Tau  | Tail gamma | Gate     | Adapter strength | Concept beta | Score  | Top-1  | F1-macro |")
        print("| ----------------- | ----- | - | ------- | ---------- | ------------ | ---- | ---------- | -------- | ---------------- | ------------ | ------ | ------ | -------- |")
    for head in ("default", "tip_adapter", "prototype", "prototype_tau", "loo_bcpd", "sctr", "concept", "concept_prototype"):
        row = best_by_head.get(head)
        if row is None:
            continue
        top1 = f"{row['top1'] * 100:.2f}%"
        f1 = "N/A" if row["F1-macro_all"] is None else f"{row['F1-macro_all'] * 100:.2f}%"
        concept_beta = "N/A" if row["concept_beta"] is None else f"{row['concept_beta']:g}"
        adapter_strength = row.get("loo_bcpd_strength", 0.0) if row["head"] == "loo_bcpd" else row["gate_strength"]
        common = (
            f"| {head:<17} | {row['prototype_scale']:<5g} | {row['prototype_k']:<1d} | "
            f"{row['sequence_eta']:<7g} | {row['sctr_strength']:<10g} | {row['sctr_tail_protection']:<12g} | {row['tau']:<4g} | {row['tail_gamma']:<10g} | {row['gate_mode']:<8} | {adapter_strength:<16g} | "
            f"{concept_beta:<12}"
        )
        if include_tip_columns:
            tip_support_shots = "full" if row.get("tip_support_shots") == 0 else str(row.get("tip_support_shots", "N/A"))
            tip_beta = "N/A" if row.get("tip_beta") is None else f"{row['tip_beta']:g}"
            tip_alpha = "N/A" if row.get("tip_alpha") is None else f"{row['tip_alpha']:g}"
            print(f"{common} | {tip_support_shots:<9} | {tip_beta:<8} | {tip_alpha:<9} | {row['score']:<6.4f} | {top1:<6} | {f1:<8} |")
        else:
            print(f"{common} | {row['score']:<6.4f} | {top1:<6} | {f1:<8} |")


def write_selection_output(path, val_dataset, best_metric, best_by_head):
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_dataset": val_dataset,
        "best_metric": best_metric,
        "best_by_head": best_by_head,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved validation selection to {output_path}")


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
        ("multiproto_k2", {"sequence_eta": 0.5, "prototype_k": 2, "gate_mode": "none", "gate_strength": 0.0}),
        ("multiproto_k4", {"sequence_eta": 0.5, "prototype_k": 4, "gate_mode": "none", "gate_strength": 0.0}),
        ("multiproto_k8", {"sequence_eta": 0.5, "prototype_k": 8, "gate_mode": "none", "gate_strength": 0.0}),
    ]
    rows = []
    for label, target in targets:
        match = next((row for row in selection_rows if candidate_matches(row, **target)), None)
        if match is not None:
            rows.append((label, match))
    return rows


def select_preferred_head(best_by_head):
    return max(best_by_head.items(), key=lambda item: item[1]["score"])[0]


def select_summary_head(best_by_head, requested_head):
    if requested_head is None:
        return select_preferred_head(best_by_head)
    if requested_head not in best_by_head:
        raise ValueError(f"--summary-head={requested_head!r} has no selected candidate.")
    return requested_head


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
    tip_beta_grid, tip_alpha_grid = parse_tip_adapter_grids(args)
    tip_support_shots_grid = parse_tip_adapter_support_shots_grid(args, tip_beta_grid)
    validate_tip_adapter_protocol(args, tip_beta_grid, tip_support_shots_grid)
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
    if tip_support_shots_grid:
        tip_support_shots_grid = resolve_tip_adapter_support_shots_grid(
            train_features["labels"],
            num_classes,
            tip_support_shots_grid,
        )
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
    class_checksum = class_mapping_checksum(train_data.classnames, classification_head)
    print(f"LOO-BCPD class mapping checksum={class_checksum}")
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
    sctr_strength_grid = parse_float_grid(args.sctr_strength_grid)
    sctr_tail_protection_grid = parse_float_grid(args.sctr_tail_protection_grid)
    concept_beta_grid = parse_float_grid(args.concept_beta_grid)
    loo_bcpd_strength_grid = parse_float_grid(args.loo_bcpd_strength_grid)
    if any(strength < 0.0 or strength > 1.0 for strength in loo_bcpd_strength_grid):
        raise ValueError("--loo-bcpd-strength-grid values must be in [0, 1].")
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
    val_tip_cache_logits_by_config = build_tip_adapter_logits_by_config(
        val_features["features"],
        train_features["features"],
        train_features["labels"],
        num_classes,
        tip_beta_grid,
        tip_support_shots_grid,
        args,
    )
    if tip_beta_grid:
        print(f"Built Tip-Adapter cache logits for {len(tip_support_shots_grid)} support-shot settings and {len(tip_beta_grid)} beta values.")
    val_tpa_logits = val_base_logits + 50.0 * val_prototype_logits_by_k[1]
    val_loo_bcpd_results = {
        float(strength): build_loo_bcpd_logits(
            val_features["features"],
            val_base_logits,
            prototypes,
            val_features["metadata"],
            sequence_field_index=sequence_field_index,
            prototype_scale=50.0,
            strength=float(strength),
            cached_tpa_logits=val_tpa_logits,
        )
        for strength in loo_bcpd_strength_grid
    }
    val_loo_bcpd_logits_by_strength = {
        strength: result.logits for strength, result in val_loo_bcpd_results.items()
    }
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
        sctr_strength_grid=sctr_strength_grid,
        sctr_tail_protection_grid=sctr_tail_protection_grid,
        sequence_field_index=sequence_field_index,
        tip_beta_grid=tip_beta_grid,
        tip_alpha_grid=tip_alpha_grid,
        tip_support_shots_grid=tip_support_shots_grid,
        tip_cache_logits_by_config=val_tip_cache_logits_by_config,
        loo_bcpd_strength_grid=loo_bcpd_strength_grid,
        loo_bcpd_logits_by_strength=val_loo_bcpd_logits_by_strength,
        train_class_counts=counts,
    )
    print_selection(selection_rows, best_by_head)
    if args.selection_output is not None:
        write_selection_output(args.selection_output, args.val_dataset, args.best_metric, best_by_head)
    key_ablation_rows = find_key_ablation_rows(selection_rows) if args.report_key_ablation_candidates else []
    selected_tip = best_by_head.get("tip_adapter")
    selected_tip_beta_grid = [] if selected_tip is None else [selected_tip["tip_beta"]]
    selected_tip_support_shots_grid = [] if selected_tip is None else [selected_tip["tip_support_shots"]]

    summary_rows = []
    key_ablation_summary_rows = []
    stp_diagnostics_written = False
    loo_bcpd_diagnostics_written = False
    for dataset_name in args.eval_datasets:
        print(f"Evaluating tail cache on {dataset_name}...")
        dataset = build_eval_dataset(dataset_name, encoder, args)
        if args.audit_metadata and dataset_name != args.val_dataset:
            print_metadata_audit(dataset_name, dataset, args.sequence_id_field)
        features = val_features if dataset_name == args.val_dataset else extract_features(model, dataset.test_loader, args, f"{dataset_name} features")
        base_logits = val_base_logits if dataset_name == args.val_dataset else default_logits(features["features"], classification_head)
        if dataset_name == args.val_dataset:
            proto_logits_by_k = val_prototype_logits_by_k
            tip_cache_logits_by_config = val_tip_cache_logits_by_config
            loo_bcpd_results = val_loo_bcpd_results
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
            tip_cache_logits_by_config = build_tip_adapter_logits_by_config(
                features["features"],
                train_features["features"],
                train_features["labels"],
                num_classes,
                selected_tip_beta_grid,
                selected_tip_support_shots_grid,
                args,
            )
            tpa_logits = base_logits + 50.0 * proto_logits_by_k[1]
            loo_bcpd_results = {
                float(strength): build_loo_bcpd_logits(
                    features["features"],
                    base_logits,
                    prototypes,
                    features["metadata"],
                    sequence_field_index=sequence_field_index,
                    prototype_scale=50.0,
                    strength=float(strength),
                    cached_tpa_logits=tpa_logits,
                )
                for strength in loo_bcpd_strength_grid
            }
        cd_logits = val_concept_logits if dataset_name == args.val_dataset else concept_logits(features["features"], concept_head)
        loo_bcpd_logits_by_strength = {strength: result.logits for strength, result in loo_bcpd_results.items()}
        if args.loo_bcpd_diagnostics_report is not None and dataset_name == args.loo_bcpd_diagnostics_split:
            selected_bcpd = best_by_head.get("loo_bcpd")
            if selected_bcpd is None:
                raise ValueError("LOO-BCPD diagnostics require a selected LOO-BCPD candidate.")
            selected_strength = float(selected_bcpd["loo_bcpd_strength"])
            selected_result = loo_bcpd_results[selected_strength]
            metadata_fields = metadata_fields_from_dataset(dataset)
            location_field_index = resolve_metadata_field_index(
                metadata_fields,
                "auto",
                ["location", "camera", "camera_id", "location_id"],
            )
            hour_field_index = resolve_metadata_field_index(metadata_fields, "auto", ["hour"])
            shuffled = shuffled_sequence_groups(
                features["metadata"],
                sequence_field_index,
                location_field_index,
                hour_field_index,
                args.seed,
            )
            tpa_logits = base_logits + 50.0 * proto_logits_by_k[1]
            derangement = deterministic_derangement(num_classes, args.seed).to(features["features"].device)
            linear_result = build_loo_linear_logits(
                features["features"], base_logits, prototypes, features["metadata"],
                sequence_field_index=sequence_field_index, prototype_scale=50.0, strength=selected_strength,
                cached_tpa_logits=tpa_logits,
            )
            permuted_result = build_loo_bcpd_logits(
                features["features"], base_logits, prototypes, features["metadata"],
                sequence_field_index=sequence_field_index, prototype_scale=50.0, strength=selected_strength,
                cached_tpa_logits=tpa_logits, responsibility_permutation=derangement,
            )
            shuffled_result = build_loo_bcpd_logits(
                features["features"], base_logits, prototypes, features["metadata"],
                sequence_field_index=sequence_field_index, prototype_scale=50.0, strength=selected_strength,
                cached_tpa_logits=tpa_logits, groups=shuffled.groups,
            )
            self_including_result = build_loo_bcpd_logits(
                features["features"], base_logits, prototypes, features["metadata"],
                sequence_field_index=sequence_field_index, prototype_scale=50.0, strength=selected_strength,
                cached_tpa_logits=tpa_logits, include_target=True,
            )
            unconstrained_result = build_loo_bcpd_logits(
                features["features"], base_logits, prototypes, features["metadata"],
                sequence_field_index=sequence_field_index, prototype_scale=50.0, strength=selected_strength,
                cached_tpa_logits=tpa_logits, variant="unconstrained",
            )
            diagnostics_logits = {
                "frame": base_logits,
                "tpa": tpa_logits,
                "stp_mean": apply_sequence_consensus(tpa_logits, features["metadata"], sequence_field_index, eta=0.5),
                "loo_linear": linear_result.logits,
                "loo_bcpd": selected_result.logits,
                "class_derangement": permuted_result.logits,
                "burst_shuffle": shuffled_result.logits,
                "self_including": self_including_result.logits,
                "unconstrained_mixing": unconstrained_result.logits,
            }
            write_loo_bcpd_diagnostics(
                args.loo_bcpd_diagnostics_report,
                dataset_name,
                dataset,
                args,
                features["labels"],
                features["metadata"],
                sequence_field_index,
                counts,
                class_checksum,
                diagnostics_logits,
                selected_result,
                shuffled,
                selected_strength,
                2000,
                args.seed,
            )
            loo_bcpd_diagnostics_written = True
        if args.stp_diagnostics_report is not None and dataset_name == args.stp_diagnostics_split:
            stp_row = best_by_head.get("prototype")
            if stp_row is None:
                raise ValueError("STP diagnostics require a selected prototype candidate.")
            frame_row = dict(stp_row)
            frame_row["sequence_eta"] = 0.0
            frame_logits = build_candidate_predictions(
                base_logits,
                proto_logits_by_k,
                cd_logits,
                class_priors,
                tail_weights_by_gamma,
                frame_row,
                metadata=features["metadata"],
                sequence_field_index=sequence_field_index,
                tip_cache_logits_by_config=tip_cache_logits_by_config,
            )
            stp_logits = build_candidate_predictions(
                base_logits,
                proto_logits_by_k,
                cd_logits,
                class_priors,
                tail_weights_by_gamma,
                stp_row,
                metadata=features["metadata"],
                sequence_field_index=sequence_field_index,
                tip_cache_logits_by_config=tip_cache_logits_by_config,
            )
            diagnostics = write_stp_diagnostics(
                args.stp_diagnostics_report,
                dataset_name,
                features["labels"],
                frame_logits,
                stp_logits,
                counts,
                features["metadata"],
                sequence_field_index,
                args.stp_diagnostics_bootstrap_samples,
                args.seed,
            )
            stp_diagnostics_written = True
            if wandb is not None:
                wandb.log({
                    f"stp_diagnostics/{dataset_name}/frame_macro_f1": diagnostics.frame_macro_f1,
                    f"stp_diagnostics/{dataset_name}/stp_macro_f1": diagnostics.stp_macro_f1,
                    f"stp_diagnostics/{dataset_name}/delta_macro_f1": diagnostics.bootstrap.delta,
                    f"stp_diagnostics/{dataset_name}/bootstrap_low": diagnostics.bootstrap.low,
                    f"stp_diagnostics/{dataset_name}/bootstrap_high": diagnostics.bootstrap.high,
                    f"stp_diagnostics/{dataset_name}/frame_wrong_stp_correct": diagnostics.frame_wrong_stp_correct,
                    f"stp_diagnostics/{dataset_name}/frame_correct_stp_wrong": diagnostics.frame_correct_stp_wrong,
                })
        for head_name in ("default", "tip_adapter", "prototype", "prototype_tau", "loo_bcpd", "sctr", "concept", "concept_prototype"):
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
                tip_cache_logits_by_config=tip_cache_logits_by_config,
                loo_bcpd_logits_by_strength=loo_bcpd_logits_by_strength,
            )
            top1 = results.get("top1")
            f1_macro = results.get("F1-macro_all")
            print(f"  {dataset_name} {head_name} Top-1 accuracy: {top1:.4f}")
            if f1_macro is not None:
                print(f"  {dataset_name} {head_name} F1-macro_all: {f1_macro:.4f}")
            summary_rows.append((dataset_name, head_name, top1, f1_macro))
            if wandb is not None:
                log_payload = {
                    f"tail_cache/{dataset_name}/{head_name}/top1": top1,
                    f"tail_cache/{dataset_name}/{head_name}/f1_macro": f1_macro,
                    f"tail_cache/{head_name}/prototype_scale": best["prototype_scale"],
                    f"tail_cache/{head_name}/prototype_k": best["prototype_k"],
                    f"tail_cache/{head_name}/sequence_eta": best["sequence_eta"],
                    "tail_cache/best_tau": best["tau"],
                    f"tail_cache/{head_name}/tail_gamma": best["tail_gamma"],
                    f"tail_cache/{head_name}/gate_strength": best["gate_strength"],
                    f"tail_cache/{head_name}/loo_bcpd_strength": best.get("loo_bcpd_strength", 0.0),
                    f"tail_cache/{head_name}/sctr_strength": best["sctr_strength"],
                    f"tail_cache/{head_name}/sctr_tail_protection": best["sctr_tail_protection"],
                    f"tail_cache/{head_name}/concept_beta": best["concept_beta"],
                }
                if head_name == "tip_adapter":
                    log_payload.update({
                        f"tail_cache/{head_name}/tip_support_shots": best["tip_support_shots"],
                        f"tail_cache/{head_name}/tip_beta": best["tip_beta"],
                        f"tail_cache/{head_name}/tip_alpha": best["tip_alpha"],
                    })
                wandb.log(log_payload)
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
    if args.stp_diagnostics_report is not None and not stp_diagnostics_written:
        raise ValueError(f"--stp-diagnostics-split={args.stp_diagnostics_split} is not in --eval-datasets.")
    if args.loo_bcpd_diagnostics_report is not None and not loo_bcpd_diagnostics_written:
        raise ValueError(f"--loo-bcpd-diagnostics-split={args.loo_bcpd_diagnostics_split} is not in --eval-datasets.")
    preferred_head = select_summary_head(best_by_head, args.summary_head)
    cache_summary_rows = [(dataset, top1, f1) for dataset, head, top1, f1 in summary_rows if head == preferred_head]
    log_wandb_summary(wandb, cache_summary_rows)
    if wandb is not None:
        wandb.summary["tail_cache/selected_head"] = preferred_head
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main(parse_arguments())
