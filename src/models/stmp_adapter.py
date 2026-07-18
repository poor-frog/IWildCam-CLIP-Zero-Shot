from collections import defaultdict
import math

import torch


INVALID_METADATA_GROUP_KEYS = frozenset({"-1", "nan", "none", "null", "missing"})


def normalize_features(features, eps=1e-12):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)


def farthest_first_indices(class_features, k):
    if class_features.shape[0] <= k:
        return torch.arange(class_features.shape[0])
    class_features = normalize_features(class_features)
    class_mean = normalize_features(class_features.mean(dim=0, keepdim=True))[0]
    first = torch.argmax(class_features @ class_mean).view(1)
    selected = [int(first.item())]
    best_sim = class_features @ class_features[selected[0]]
    for _ in range(1, k):
        next_index = int(torch.argmin(best_sim).item())
        selected.append(next_index)
        best_sim = torch.maximum(best_sim, class_features @ class_features[next_index])
    return torch.tensor(selected, dtype=torch.long)


def build_mean_prototypes(features, labels, num_classes):
    feature_dim = features.shape[1]
    sums = torch.zeros(num_classes, feature_dim, dtype=features.dtype)
    counts = torch.zeros(num_classes, dtype=features.dtype)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
    prototypes = normalize_features(prototypes)
    return prototypes, counts > 0


def build_multi_prototypes(features, labels, num_classes, max_k):
    if max_k <= 1:
        prototypes, present = build_mean_prototypes(features, labels, num_classes)
        return prototypes.unsqueeze(1), present.unsqueeze(1)

    feature_dim = features.shape[1]
    prototypes = torch.zeros(num_classes, max_k, feature_dim, dtype=features.dtype)
    prototype_mask = torch.zeros(num_classes, max_k, dtype=torch.bool)
    for class_index in range(num_classes):
        class_features = features[labels == class_index]
        if class_features.numel() == 0:
            continue
        selected = farthest_first_indices(class_features, min(max_k, class_features.shape[0]))
        selected_features = normalize_features(class_features[selected])
        prototypes[class_index, : selected_features.shape[0]] = selected_features
        prototype_mask[class_index, : selected_features.shape[0]] = True
    return prototypes, prototype_mask


def prototype_logits(features, prototypes, present_mask, beta):
    logits = float(beta) * normalize_features(features) @ prototypes.t()
    if not present_mask.all():
        logits[:, ~present_mask] = torch.finfo(logits.dtype).min
    return logits


def multi_prototype_logits(features, prototypes, prototype_mask, beta, reduction="max"):
    if prototypes.shape[1] == 1:
        return prototype_logits(features, prototypes[:, 0], prototype_mask[:, 0], beta)

    normalized = normalize_features(features)
    prototype_features = normalize_features(prototypes)
    logits = torch.einsum("bd,ckd->bck", normalized, prototype_features)
    invalid_value = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~prototype_mask.unsqueeze(0), invalid_value)
    if reduction == "max":
        class_logits = logits.max(dim=2).values
    elif reduction == "logsumexp":
        class_logits = torch.logsumexp(logits, dim=2)
    else:
        raise ValueError(f"Unsupported multi-prototype reduction: {reduction}")
    missing_classes = ~prototype_mask.any(dim=1)
    if missing_classes.any():
        class_logits[:, missing_classes] = invalid_value
    return float(beta) * class_logits


def metadata_fields_from_dataset(dataset):
    wilds_dataset = getattr(dataset, "dataset", dataset)
    nested_dataset = getattr(wilds_dataset, "dataset", None)
    if nested_dataset is not None and hasattr(nested_dataset, "metadata_fields"):
        return list(nested_dataset.metadata_fields)
    if hasattr(wilds_dataset, "metadata_fields"):
        return list(wilds_dataset.metadata_fields)
    return []


def normalize_field_name(value):
    return str(value).lower().replace("-", "_").replace(" ", "_")


def resolve_metadata_field_index(fields, requested, candidates):
    if requested is None or requested == "none":
        return None
    if str(requested).isdigit():
        index = int(requested)
        if not fields:
            return index
        return index if 0 <= index < len(fields) else None
    normalized = [normalize_field_name(field) for field in fields]
    if requested != "auto":
        requested_name = normalize_field_name(requested)
        return normalized.index(requested_name) if requested_name in normalized else None
    for candidate in candidates:
        candidate_name = normalize_field_name(candidate)
        if candidate_name in normalized:
            return normalized.index(candidate_name)
    for index, field_name in enumerate(normalized):
        if any(candidate in field_name for candidate in candidates):
            return index
    return None


def metadata_value(metadata_row, field_index):
    if field_index is None:
        return None
    if hasattr(metadata_row, "detach"):
        metadata_row = metadata_row.detach().cpu().tolist()
    elif hasattr(metadata_row, "tolist"):
        metadata_row = metadata_row.tolist()
    if isinstance(metadata_row, (list, tuple)):
        if 0 <= field_index < len(metadata_row):
            value = metadata_row[field_index]
            if hasattr(value, "item"):
                value = value.item()
            return value
        return None
    return metadata_row if field_index == 0 else None


def metadata_group_key(metadata_row, field_index):
    value = metadata_value(metadata_row, field_index)
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = str(int(value))
    elif isinstance(value, str):
        value = value.strip()
    else:
        value = str(value)
    return None if not value or value.lower() in INVALID_METADATA_GROUP_KEYS else value


def sequence_groups(metadata, sequence_field_index):
    groups = defaultdict(list)
    for row_index, metadata_row in enumerate(metadata):
        sequence_key = metadata_group_key(metadata_row, sequence_field_index)
        if sequence_key is not None:
            groups[sequence_key].append(row_index)
    return groups


def apply_sequence_consensus(logits, metadata, sequence_field_index, eta):
    eta = float(eta)
    if eta == 0.0 or sequence_field_index is None or not metadata:
        return logits

    groups = sequence_groups(metadata, sequence_field_index)
    if not groups:
        return logits

    sequence_logits = logits.clone()
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=logits.device)
        sequence_mean = logits.index_select(0, index_tensor).mean(dim=0, keepdim=True)
        sequence_logits[index_tensor] = sequence_mean
    return (1.0 - eta) * logits + eta * sequence_logits


def apply_target_selective_sequence_consensus(
    logits,
    metadata,
    sequence_field_index,
    eta,
    margin_threshold,
    min_sequence_length=3,
):
    eta = float(eta)
    if eta == 0.0 or sequence_field_index is None or not metadata:
        return logits
    if min_sequence_length < 2:
        raise ValueError("min_sequence_length must be at least 2.")
    if logits.shape[1] < 2:
        return logits

    groups = sequence_groups(metadata, sequence_field_index)
    if not groups:
        return logits

    selective_logits = logits.clone()
    for indices in groups.values():
        if len(indices) < min_sequence_length:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=logits.device)
        group_logits = logits.index_select(0, index_tensor)
        probabilities = torch.softmax(group_logits.float(), dim=1)
        top_two = probabilities.topk(k=2, dim=1).values
        target_mask = top_two[:, 0] - top_two[:, 1] <= float(margin_threshold)
        if not target_mask.any():
            continue
        target_indices = index_tensor[target_mask]
        sequence_mean = group_logits.mean(dim=0, keepdim=True)
        selective_logits[target_indices] = (1.0 - eta) * logits.index_select(0, target_indices) + eta * sequence_mean
    return selective_logits


def sample_metadata_rows(subset, limit=3):
    metadata_array = getattr(subset, "metadata_array", None)
    if metadata_array is None:
        return []
    rows = metadata_array[:limit]
    if hasattr(rows, "detach"):
        return rows.detach().cpu().tolist()
    if hasattr(rows, "tolist"):
        return rows.tolist()
    return rows


def print_metadata_audit(name, dataset, sequence_id_field):
    fields = metadata_fields_from_dataset(dataset)
    sequence_index = resolve_metadata_field_index(fields, sequence_id_field, ["seq_id", "sequence_id", "sequence"])
    camera_index = resolve_metadata_field_index(fields, "auto", ["location", "camera", "camera_id", "location_id"])
    datetime_index = resolve_metadata_field_index(fields, "auto", ["datetime", "date_time", "timestamp", "date"])
    subset = getattr(dataset, "train_dataset", None) or getattr(dataset, "test_dataset", None)
    rows = sample_metadata_rows(subset)
    print(f"\n=== Metadata Audit: {name} ===")
    print(f"metadata_fields={fields}")
    print(f"resolved_sequence_index={sequence_index}")
    print(f"resolved_camera_index={camera_index}")
    print(f"resolved_datetime_index={datetime_index}")
    print(f"sample_metadata_rows={rows}")
    if sequence_index is None:
        print("Sequence consensus disabled unless --sequence-id-field is set to a valid field/index.")
    return sequence_index
