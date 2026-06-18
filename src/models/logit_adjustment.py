from __future__ import annotations

from dataclasses import dataclass

import torch

from src.datasets.iwildcam import get_train_class_priors


FINAL_TEST_ONLY_SPLIT = "IWildCamOOD"
DEFAULT_SELECTION_SPLIT = "IWildCamVal"


def parse_tau_grid(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("tau grid must include at least one numeric value.")
    return [float(value) for value in values]


def validate_selection_split(selection_split):
    if selection_split == FINAL_TEST_ONLY_SPLIT:
        raise ValueError("IWildCamOOD is final-test-only and cannot be used for tau selection.")


def get_metric_for_tau_selection(results, metric_name="F1-macro_all"):
    value = results.get(metric_name)
    if value is None:
        value = results.get("top1")
    if value is None:
        raise KeyError(f"Selection results include neither {metric_name!r} nor 'top1'.")
    return float(value)


def apply_logit_adjustment(logits, class_priors, tau, eps=1e-12):
    if tau is None or float(tau) == 0.0:
        return logits
    if class_priors is None:
        raise ValueError("class_priors are required when tau is non-zero.")

    if not torch.is_tensor(class_priors):
        class_priors = torch.as_tensor(class_priors, dtype=logits.dtype, device=logits.device)
    else:
        class_priors = class_priors.to(device=logits.device, dtype=logits.dtype)

    if class_priors.ndim != 1:
        raise ValueError("class_priors must be a 1D tensor.")
    if logits.shape[-1] != class_priors.shape[0]:
        raise ValueError("logits last dimension must match class_priors length.")

    return logits - float(tau) * torch.log(class_priors.clamp_min(eps))


def build_train_class_priors_for_dataset(dataset, device):
    counts, priors = get_train_class_priors(dataset.dataset, len(dataset.classnames))
    counts_tensor = torch.tensor(counts, dtype=torch.float32, device=device)
    priors_tensor = torch.tensor(priors, dtype=torch.float32, device=device)
    return counts_tensor, priors_tensor


@dataclass
class TauSelectionResult:
    best_tau: float
    best_score: float
    metric_name: str
    selection_split: str
    rows: list[dict]


def select_best_tau(eval_fn, model, dataset, args, tau_grid, metric_name="F1-macro_all", class_priors=None, classification_head=None):
    validate_selection_split(args.selection_split)
    rows = []
    best_tau = None
    best_score = None

    for tau in tau_grid:
        if classification_head is None:
            results = eval_fn(model, dataset, args, tau=tau, class_priors=class_priors)
        else:
            results = eval_fn(model, dataset, args, classification_head, tau=tau, class_priors=class_priors)
        score = get_metric_for_tau_selection(results, metric_name)
        row = {
            "tau": float(tau),
            "selection_split": args.selection_split,
            "metric_name": metric_name,
            "score": score,
            "top1": float(results.get("top1", 0.0)),
        }
        if "F1-macro_all" in results:
            row["F1-macro_all"] = float(results["F1-macro_all"])
        rows.append(row)

        if best_score is None or score > best_score:
            best_tau = float(tau)
            best_score = score

    return TauSelectionResult(
        best_tau=best_tau,
        best_score=best_score,
        metric_name=metric_name,
        selection_split=args.selection_split,
        rows=rows,
    )


def describe_tau_selection(selection_result):
    lines = []
    for row in selection_result.rows:
        f1_text = f", F1-macro_all={row['F1-macro_all']:.4f}" if "F1-macro_all" in row else ""
        lines.append(
            f"tau={row['tau']:.4f} selection_split={row['selection_split']} score={row['score']:.4f} top1={row['top1']:.4f}{f1_text}"
        )
    lines.append(
        f"selected_tau={selection_result.best_tau:.4f} selection_split={selection_result.selection_split} metric={selection_result.metric_name} best_score={selection_result.best_score:.4f}"
    )
    return lines
