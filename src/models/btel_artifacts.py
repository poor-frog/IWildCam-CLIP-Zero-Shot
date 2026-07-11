"""Frozen BTEL artifact construction from the training split only."""

from dataclasses import dataclass
from typing import Final, Iterable, Protocol

import torch
from tqdm import tqdm

from src.datasets.dataloader import maybe_dictionarize
from src.models.btel import BTELArtifacts, BTELConfigurationError, negative_thresholds_from_train_evidence, sequence_groups
from src.models.tail_prototype import normalize_features
from src.models.stmp_adapter import metadata_fields_from_dataset, resolve_metadata_field_index


BTEL_VALIDATION_SPLIT: Final = "IWildCamVal"


@dataclass(frozen=True, slots=True)
class SequenceAudit:
    """Summary of one labeled iWildCam split grouped by burst ID."""

    split_name: str
    sequence_count: int
    frame_count: int
    singleton_sequences: int
    pure_label_sequences: int
    empty_only_sequences: int
    mixed_empty_sequences: int
    tail_frames: int
    medium_frames: int
    head_frames: int


class MetadataDataset(Protocol):
    metadata_fields: list[str]


class WILDSSubset(Protocol):
    dataset: MetadataDataset
    metadata_array: torch.Tensor
    y_array: torch.Tensor


def find_empty_class_index(classnames: list[str]) -> int | None:
    """Resolve the canonical empty camera-trap class when present."""
    for index, class_name in enumerate(classnames):
        if class_name.strip().lower().replace("_", " ") == "empty":
            return index
    return None


def validate_btel_validation_split(validation_split: str | None) -> None:
    if validation_split != BTEL_VALIDATION_SPLIT:
        raise BTELConfigurationError(
            f"BTEL requires {BTEL_VALIDATION_SPLIT} for validation selection, got {validation_split!r}."
        )


def sequence_field_index(dataset: WILDSSubset) -> int:
    """Resolve the required iWildCam sequence metadata column or fail fast."""
    fields = metadata_fields_from_dataset(dataset)
    index = resolve_metadata_field_index(fields, "auto", ["seq_id", "sequence_id", "sequence"])
    if index is None:
        raise BTELConfigurationError(f"BTEL requires iWildCam sequence metadata; available fields={fields}.")
    return index


def audit_sequences(
    subset: WILDSSubset,
    *,
    split_name: str,
    num_classes: int,
    classnames: list[str],
) -> SequenceAudit:
    """Audit burst labels and frequency buckets without loading image pixels."""
    metadata = getattr(subset, "metadata_array")
    labels = getattr(subset, "y_array")
    if not isinstance(metadata, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise BTELConfigurationError("BTEL audit requires WILDS metadata_array and y_array tensors.")
    index = sequence_field_index(subset)
    empty_index = find_empty_class_index(classnames)
    labels = labels.long()
    counts = torch.bincount(labels, minlength=num_classes)
    groups = sequence_groups(metadata, index)
    singleton = 0
    pure = 0
    empty_only = 0
    mixed_empty = 0
    for group in groups:
        group_labels = labels[torch.tensor(group)]
        unique = group_labels.unique()
        singleton += int(len(group) == 1)
        pure += int(unique.numel() == 1)
        if empty_index is not None:
            has_empty = bool((unique == empty_index).any())
            empty_only += int(has_empty and unique.numel() == 1)
            mixed_empty += int(has_empty and unique.numel() > 1)
    return SequenceAudit(
        split_name=split_name,
        sequence_count=len(groups),
        frame_count=int(labels.numel()),
        singleton_sequences=singleton,
        pure_label_sequences=pure,
        empty_only_sequences=empty_only,
        mixed_empty_sequences=mixed_empty,
        tail_frames=int((counts[labels] <= 20).sum().item()),
        medium_frames=int(((counts[labels] > 20) & (counts[labels] <= 100)).sum().item()),
        head_frames=int((counts[labels] > 100).sum().item()),
    )


def print_sequence_audit(audit: SequenceAudit) -> None:
    """Print a machine-readable-enough audit table for terminal and Kaggle logs."""
    print(f"\n=== BTEL Sequence Audit: {audit.split_name} ===")
    print("| sequences | frames | singleton | pure labels | empty-only | mixed-empty | tail frames | medium frames | head frames |")
    print("| --------- | ------ | --------- | ----------- | ---------- | ----------- | ----------- | ------------- | ----------- |")
    print(
        f"| {audit.sequence_count} | {audit.frame_count} | {audit.singleton_sequences} | "
        f"{audit.pure_label_sequences} | {audit.empty_only_sequences} | {audit.mixed_empty_sequences} | "
        f"{audit.tail_frames} | {audit.medium_frames} | {audit.head_frames} |"
    )


def _collect_train_features(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    metadata_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="BTEL frozen train features")):
            if max_batches is not None and batch_index >= max_batches:
                break
            data = maybe_dictionarize(batch)
            if "metadata" not in data:
                raise BTELConfigurationError("BTEL training loader must return metadata.")
            feature_chunks.append(normalize_features(model(data["images"].to(device))).cpu())
            label_chunks.append(data["labels"].detach().cpu().long())
            metadata_chunks.append(data["metadata"].detach().cpu().long())
    if not feature_chunks:
        raise BTELConfigurationError("BTEL could not build artifacts from an empty training loader.")
    return torch.cat(feature_chunks), torch.cat(label_chunks), torch.cat(metadata_chunks)


def build_btel_artifacts(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dataset: WILDSSubset,
    *,
    classnames: list[str],
    device: str,
    prototype_scale: float,
    negative_quantile: float,
    max_batches: int | None,
) -> BTELArtifacts:
    """Build frozen prototypes and train-negative evidence thresholds once before training."""
    features, labels, metadata = _collect_train_features(model, loader, device, max_batches)
    num_classes = len(classnames)
    class_counts = torch.bincount(labels, minlength=num_classes)
    sums = torch.zeros((num_classes, features.shape[1]), dtype=features.dtype)
    sums.index_add_(0, labels, features)
    prototypes = normalize_features(sums / class_counts.clamp_min(1).unsqueeze(1))
    index = sequence_field_index(dataset)
    empty_index = find_empty_class_index(classnames)
    logits = float(prototype_scale) * features @ prototypes.t()
    thresholds = negative_thresholds_from_train_evidence(
        logits,
        labels,
        metadata,
        sequence_field_index=index,
        class_counts=class_counts,
        empty_class_index=empty_index,
        quantile=negative_quantile,
    )
    return BTELArtifacts(prototypes, class_counts, thresholds, index, empty_index)
