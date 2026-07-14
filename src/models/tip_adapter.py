import torch


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def tip_adapter_cache_logits(
    query_features,
    cache_features,
    cache_labels,
    num_classes,
    beta,
    query_chunk_size,
    cache_chunk_size,
    device,
):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if query_chunk_size <= 0 or cache_chunk_size <= 0:
        raise ValueError("Tip-Adapter chunk sizes must be positive.")
    if cache_features.shape[0] != cache_labels.shape[0]:
        raise ValueError("Tip-Adapter cache features and labels must have the same length.")
    if cache_features.shape[0] == 0:
        raise ValueError("Tip-Adapter cache must contain at least one example.")

    compute_device = torch.device(device)
    cache_keys = normalize_features(cache_features.float()).to(compute_device)
    cache_values = cache_labels.long().to(compute_device)
    outputs = []
    for query_start in range(0, query_features.shape[0], query_chunk_size):
        query_end = min(query_start + query_chunk_size, query_features.shape[0])
        queries = normalize_features(query_features[query_start:query_end].float()).to(compute_device)
        cache_logits = torch.zeros((queries.shape[0], num_classes), dtype=queries.dtype, device=compute_device)
        for cache_start in range(0, cache_keys.shape[0], cache_chunk_size):
            cache_end = min(cache_start + cache_chunk_size, cache_keys.shape[0])
            affinity = torch.exp(-float(beta) * (1.0 - queries @ cache_keys[cache_start:cache_end].t()))
            labels = cache_values[cache_start:cache_end].expand(queries.shape[0], -1)
            cache_logits.scatter_add_(1, labels, affinity)
        outputs.append(cache_logits.cpu())
    return torch.cat(outputs, dim=0)
