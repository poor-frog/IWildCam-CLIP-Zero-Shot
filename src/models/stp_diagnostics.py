from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from src.models.stmp_adapter import metadata_value


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    low: float
    delta: float
    high: float


@dataclass(frozen=True, slots=True)
class PairedSequenceBootstrap:
    low: float
    delta: float
    high: float
    positive_delta_fraction: float
    median_class_coverage: float
    minimum_class_coverage: float
    stability_low: float | None
    stability_high: float | None
    stability_sample_count: int


@dataclass(frozen=True, slots=True)
class StratifiedMacroF1:
    label: str
    examples: int
    frame_macro_f1: float
    stp_macro_f1: float


@dataclass(frozen=True, slots=True)
class StpDiagnostics:
    frame_macro_f1: float
    stp_macro_f1: float
    frame_wrong_stp_correct: int
    frame_correct_stp_wrong: int
    sequence_count: int
    bootstrap: BootstrapInterval
    tail_breakdown: tuple[StratifiedMacroF1, ...]
    sequence_breakdown: tuple[StratifiedMacroF1, ...]
    sequence_length_counts: Mapping[str, int]

    def to_markdown(self, split_name: str) -> str:
        lines = [
            f"# STP Diagnostics: {split_name}",
            "",
            "## Overall",
            "",
            "| Comparison | Macro-F1 |",
            "| --- | ---: |",
            f"| Frame-only TPA | {self.frame_macro_f1 * 100:.2f} |",
            f"| STP sequence-aware | {self.stp_macro_f1 * 100:.2f} |",
            f"| Delta | {(self.stp_macro_f1 - self.frame_macro_f1) * 100:+.2f} |",
            "",
            f"Sequence-cluster bootstrap 95% CI for delta: [{self.bootstrap.low * 100:+.2f}, {self.bootstrap.high * 100:+.2f}] ",
            f"(point estimate {self.bootstrap.delta * 100:+.2f}; {self.sequence_count} sequences).",
            "",
            "## Corrections",
            "",
            f"- Frame-only TPA wrong, STP correct: **{self.frame_wrong_stp_correct}** examples.",
            f"- Frame-only TPA correct, STP wrong: **{self.frame_correct_stp_wrong}** examples.",
            "",
            "## Tail Classes",
            "",
            "Training-frequency bins: tail <=20, medium 21-100, head >100 images.",
            "",
            *_format_breakdown(self.tail_breakdown),
            "",
            "## Sequence Length",
            "",
            *_format_breakdown(self.sequence_breakdown),
            "",
            "Sequence membership counts: "
            + ", ".join(f"{name}={count}" for name, count in self.sequence_length_counts.items())
            + ".",
        ]
        return "\n".join(lines) + "\n"


def _format_breakdown(rows: Sequence[StratifiedMacroF1]) -> tuple[str, ...]:
    if not rows:
        return ()
    formatted_rows = [
        "| Bin | Examples | Frame-only TPA F1 | STP F1 | Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    formatted_rows.extend(
        f"| {row.label} | {row.examples} | {row.frame_macro_f1 * 100:.2f} | {row.stp_macro_f1 * 100:.2f} | "
        f"{(row.stp_macro_f1 - row.frame_macro_f1) * 100:+.2f} |"
        for row in rows
    )
    return tuple(formatted_rows)


def _macro_f1(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> float:
    if labels.numel() == 0:
        return 0.0
    encoded = labels * num_classes + predictions
    confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()
    true_positives = confusion.diag()
    denominators = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = (2.0 * true_positives) / denominators.clamp_min(1.0)
    present = confusion.sum(dim=1) > 0
    return f1[present].mean().item() if present.any() else 0.0


def _sequence_groups(metadata: Sequence[torch.Tensor], sequence_field_index: int | None, num_examples: int) -> tuple[tuple[int, ...], ...]:
    if sequence_field_index is None or len(metadata) != num_examples:
        return tuple((index,) for index in range(num_examples))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        value = metadata_value(row, sequence_field_index)
        key = f"row-{index}" if value is None else str(value)
        groups[key].append(index)
    return tuple(tuple(indices) for indices in groups.values())


def _bootstrap_delta(
    labels: torch.Tensor,
    frame_predictions: torch.Tensor,
    stp_predictions: torch.Tensor,
    groups: Sequence[Sequence[int]],
    num_classes: int,
    bootstrap_samples: int,
    seed: int,
) -> BootstrapInterval:
    point_delta = _macro_f1(labels, stp_predictions, num_classes) - _macro_f1(labels, frame_predictions, num_classes)
    generator = torch.Generator().manual_seed(seed)
    deltas = torch.empty(bootstrap_samples)
    for sample_index in range(bootstrap_samples):
        group_indices = torch.randint(len(groups), (len(groups),), generator=generator)
        sampled_rows = torch.tensor([row for group_index in group_indices.tolist() for row in groups[group_index]])
        deltas[sample_index] = _macro_f1(labels[sampled_rows], stp_predictions[sampled_rows], num_classes) - _macro_f1(
            labels[sampled_rows], frame_predictions[sampled_rows], num_classes
        )
    quantiles = torch.quantile(deltas, torch.tensor([0.025, 0.975]))
    return BootstrapInterval(low=quantiles[0].item(), delta=point_delta, high=quantiles[1].item())


def paired_sequence_bootstrap(
    *,
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    sequence_field_index: int | None,
    bootstrap_samples: int,
    seed: int,
) -> PairedSequenceBootstrap:
    if labels.ndim != 1 or reference_predictions.shape != labels.shape or candidate_predictions.shape != labels.shape:
        raise ValueError("Paired bootstrap tensors must describe the same examples.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    num_classes = int(torch.maximum(labels.max(), torch.maximum(reference_predictions.max(), candidate_predictions.max())).item()) + 1
    groups = _sequence_groups(metadata, sequence_field_index, labels.shape[0])
    original_supported_classes = max(int(labels.unique().numel()), 1)
    point_delta = _macro_f1(labels, candidate_predictions, num_classes) - _macro_f1(labels, reference_predictions, num_classes)
    generator = torch.Generator().manual_seed(seed)
    deltas = torch.empty(bootstrap_samples)
    coverages = torch.empty(bootstrap_samples)
    for sample_index in range(bootstrap_samples):
        group_indices = torch.randint(len(groups), (len(groups),), generator=generator)
        sampled_rows = torch.tensor([row for group_index in group_indices.tolist() for row in groups[group_index]])
        sampled_labels = labels[sampled_rows]
        deltas[sample_index] = _macro_f1(sampled_labels, candidate_predictions[sampled_rows], num_classes) - _macro_f1(
            sampled_labels,
            reference_predictions[sampled_rows],
            num_classes,
        )
        coverages[sample_index] = sampled_labels.unique().numel() / original_supported_classes
    quantiles = torch.quantile(deltas, torch.tensor([0.025, 0.975]))
    stability_deltas = deltas[coverages >= 0.9]
    stability_quantiles = None if stability_deltas.numel() == 0 else torch.quantile(stability_deltas, torch.tensor([0.025, 0.975]))
    return PairedSequenceBootstrap(
        low=quantiles[0].item(),
        delta=point_delta,
        high=quantiles[1].item(),
        positive_delta_fraction=(deltas > 0.0).float().mean().item(),
        median_class_coverage=coverages.median().item(),
        minimum_class_coverage=coverages.min().item(),
        stability_low=None if stability_quantiles is None else stability_quantiles[0].item(),
        stability_high=None if stability_quantiles is None else stability_quantiles[1].item(),
        stability_sample_count=int(stability_deltas.numel()),
    )


def _breakdown(
    labels: torch.Tensor,
    frame_predictions: torch.Tensor,
    stp_predictions: torch.Tensor,
    masks: Sequence[tuple[str, torch.Tensor]],
    num_classes: int,
) -> tuple[StratifiedMacroF1, ...]:
    rows = []
    for label, mask in masks:
        count = int(mask.sum().item())
        if count == 0:
            rows.append(StratifiedMacroF1(label, 0, 0.0, 0.0))
            continue
        rows.append(
            StratifiedMacroF1(
                label,
                count,
                _macro_f1(labels[mask], frame_predictions[mask], num_classes),
                _macro_f1(labels[mask], stp_predictions[mask], num_classes),
            )
        )
    return tuple(rows)


def build_stp_diagnostics(
    *,
    labels: torch.Tensor,
    frame_logits: torch.Tensor,
    stp_logits: torch.Tensor,
    train_class_counts: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    sequence_field_index: int | None,
    bootstrap_samples: int,
    seed: int,
) -> StpDiagnostics:
    if labels.ndim != 1 or frame_logits.shape != stp_logits.shape or frame_logits.shape[0] != labels.shape[0]:
        raise ValueError("Labels and logit tensors must describe the same examples.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    num_classes = frame_logits.shape[1]
    frame_predictions = frame_logits.argmax(dim=1)
    stp_predictions = stp_logits.argmax(dim=1)
    groups = _sequence_groups(metadata, sequence_field_index, labels.shape[0])
    sequence_lengths = torch.ones(labels.shape[0], dtype=torch.long)
    for group in groups:
        sequence_lengths[list(group)] = len(group)
    class_counts = train_class_counts.to(device=labels.device)
    label_counts = class_counts[labels]
    tail_masks = (
        ("tail_1_20", label_counts <= 20),
        ("medium_21_100", (label_counts >= 21) & (label_counts <= 100)),
        ("head_101_plus", label_counts >= 101),
    )
    length_masks = (
        ("singleton", sequence_lengths == 1),
        ("short_2_4", (sequence_lengths >= 2) & (sequence_lengths <= 4)),
        ("long_5_plus", sequence_lengths >= 5),
    )
    frame_wrong_stp_correct = int(((frame_predictions != labels) & (stp_predictions == labels)).sum().item())
    frame_correct_stp_wrong = int(((frame_predictions == labels) & (stp_predictions != labels)).sum().item())
    return StpDiagnostics(
        frame_macro_f1=_macro_f1(labels, frame_predictions, num_classes),
        stp_macro_f1=_macro_f1(labels, stp_predictions, num_classes),
        frame_wrong_stp_correct=frame_wrong_stp_correct,
        frame_correct_stp_wrong=frame_correct_stp_wrong,
        sequence_count=len(groups),
        bootstrap=_bootstrap_delta(
            labels,
            frame_predictions,
            stp_predictions,
            groups,
            num_classes,
            bootstrap_samples,
            seed,
        ),
        tail_breakdown=_breakdown(labels, frame_predictions, stp_predictions, tail_masks, num_classes),
        sequence_breakdown=_breakdown(labels, frame_predictions, stp_predictions, length_masks, num_classes),
        sequence_length_counts={label: int(mask.sum().item()) for label, mask in length_masks},
    )
