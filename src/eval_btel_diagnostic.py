import argparse
import os
import random

import numpy as np
import torch

from src.device import resolve_device_choice
from src.eval_tail_cache import (
    build_prototypes,
    build_train_dataset,
    default_logits,
    extract_features,
    metrics_from_logits,
    normalize_features,
)
from src.models.btel import calibrated_burst_residual, leave_one_out_topk_evidence, negative_thresholds_from_train_evidence
from src.models.btel_artifacts import find_empty_class_index, sequence_field_index, validate_btel_validation_split
from src.models.clip_encoder import CLIPEncoder
from src.models.coop import maybe_data_parallel, unwrap_model
from src.models.flyp import get_cached_flyp_zeroshot_classifier
from src.train_coop import build_eval_dataset


class BTELDiagnosticError(ValueError):
    pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-backbone BTEL diagnostic without OOD hyperparameter selection.")
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--train-dataset", type=str, default="IWildCam")
    parser.add_argument("--val-dataset", type=str, default="IWildCamVal")
    parser.add_argument("--eval-datasets", type=lambda value: value.split(","), required=True)
    parser.add_argument("--template", type=str, default="iwildcam_drm_template")
    parser.add_argument("--data-location", type=str, required=True)
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu", "xla"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--btel-prototype-scale", type=float, default=50.0)
    parser.add_argument("--btel-negative-quantile", type=float, default=0.95)
    parser.add_argument("--best-metric", type=str, default="F1-macro_all")
    args = parser.parse_args()
    try:
        validate_btel_validation_split(args.val_dataset)
    except ValueError as error:
        raise BTELDiagnosticError(str(error)) from error
    args.data_location = os.path.expanduser(args.data_location)
    args.device = resolve_device_choice(args.device)
    return args


def metadata_tensor(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise BTELDiagnosticError("BTEL diagnostic did not receive metadata rows.")
    return torch.stack([row.detach().cpu().long() for row in rows])


def build_thresholds(
    features: torch.Tensor,
    labels: torch.Tensor,
    metadata: torch.Tensor,
    prototypes: torch.Tensor,
    class_counts: torch.Tensor,
    *,
    sequence_index: int,
    empty_index: int | None,
    scale: float,
    quantile: float,
) -> torch.Tensor:
    train_logits = float(scale) * normalize_features(features) @ prototypes.t()
    return negative_thresholds_from_train_evidence(
        train_logits,
        labels,
        metadata,
        sequence_field_index=sequence_index,
        class_counts=class_counts,
        empty_class_index=empty_index,
        quantile=quantile,
    )


def btel_diagnostic_logits(
    base_logits: torch.Tensor,
    features: torch.Tensor,
    metadata: torch.Tensor,
    prototypes: torch.Tensor,
    class_counts: torch.Tensor,
    negative_thresholds: torch.Tensor,
    *,
    sequence_index: int,
    scale: float,
) -> torch.Tensor:
    prototype_logits = float(scale) * normalize_features(features) @ prototypes.t()
    evidence = leave_one_out_topk_evidence(
        prototype_logits,
        metadata,
        sequence_field_index=sequence_index,
        class_counts=class_counts,
    )
    return base_logits + calibrated_burst_residual(evidence, negative_thresholds)


def evaluate_split(dataset_name: str, dataset, model, classifier, prototypes, class_counts, thresholds, sequence_index: int, args) -> tuple[float, float | None]:
    extracted = extract_features(model, dataset.test_loader, args, f"BTEL {dataset_name} features")
    metadata = metadata_tensor(extracted["metadata"])
    base = default_logits(extracted["features"], classifier)
    logits = btel_diagnostic_logits(
        base,
        extracted["features"],
        metadata,
        prototypes,
        class_counts,
        thresholds,
        sequence_index=sequence_index,
        scale=args.btel_prototype_scale,
    )
    metrics = metrics_from_logits(dataset, logits, extracted["labels"], extracted["metadata"], args)
    return float(metrics["top1"]), metrics.get("F1-macro_all")


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Loading frozen backbone from {args.load}")
    model = maybe_data_parallel(CLIPEncoder.load(args.load).to(args.device), args)
    encoder = unwrap_model(model)
    train_data = build_train_dataset(args, encoder)
    train = extract_features(model, train_data.train_loader, args, "BTEL diagnostic train features", is_train=True)
    train_metadata = metadata_tensor(train["metadata"])
    class_counts = torch.bincount(train["labels"], minlength=len(train_data.classnames))
    prototypes, _ = build_prototypes(train["features"], train["labels"], len(train_data.classnames))
    sequence_index = sequence_field_index(train_data.train_dataset)
    thresholds = build_thresholds(
        train["features"],
        train["labels"],
        train_metadata,
        prototypes,
        class_counts,
        sequence_index=sequence_index,
        empty_index=find_empty_class_index(train_data.classnames),
        scale=args.btel_prototype_scale,
        quantile=args.btel_negative_quantile,
    )
    classifier = get_cached_flyp_zeroshot_classifier(args, encoder)
    names = [args.val_dataset, *[name for name in args.eval_datasets if name != args.val_dataset]]
    print("\n=== Frozen-Backbone BTEL Diagnostic ===")
    print("| Split | Top-1 | F1-macro |")
    print("| ----- | ----- | -------- |")
    for name in names:
        dataset = build_eval_dataset(name, encoder, args, allow_ood_hp_subsample=name == args.val_dataset)
        top1, f1 = evaluate_split(name, dataset, model, classifier, prototypes, class_counts, thresholds, sequence_index, args)
        f1_text = "N/A" if f1 is None else f"{f1 * 100:.2f}%"
        print(f"| {name} | {top1 * 100:.2f}% | {f1_text} |")


if __name__ == "__main__":
    main(parse_arguments())
