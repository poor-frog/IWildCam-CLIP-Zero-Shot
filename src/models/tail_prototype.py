import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.dataloader import maybe_dictionarize
from src.models.coop import unwrap_model


def normalize_features(features, eps=1e-12):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)


def prototypes_from_sums(feature_sums, class_counts):
    counts = class_counts.to(device=feature_sums.device, dtype=feature_sums.dtype)
    prototypes = feature_sums / counts.clamp_min(1.0).unsqueeze(1)
    prototypes = normalize_features(prototypes)
    prototypes = torch.where(counts[:, None] > 0, prototypes, torch.zeros_like(prototypes))
    return prototypes


def build_class_prototypes_from_loader(
    model,
    dataloader,
    device,
    num_classes,
    max_batches=None,
    desc="Tail prototype features",
):
    encoder = unwrap_model(model)
    was_training = encoder.training
    encoder.eval()

    feature_sums = None
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches is not None and batch_index >= max_batches:
                break
            data = maybe_dictionarize(data)
            images = data["images"].to(device)
            labels = data["labels"].detach().cpu().long()
            if labels.numel() == 0:
                continue
            if labels.min().item() < 0 or labels.max().item() >= num_classes:
                raise IndexError(f"Prototype label is outside [0, {num_classes}).")

            features = normalize_features(encoder(images)).detach().cpu()
            if feature_sums is None:
                feature_sums = torch.zeros(num_classes, features.shape[-1], dtype=features.dtype)
            feature_sums.index_add_(0, labels, features)
            class_counts.index_add_(0, labels, torch.ones_like(labels, dtype=class_counts.dtype))

    if was_training:
        encoder.train()

    if feature_sums is None:
        raise RuntimeError("Tail prototype builder saw zero examples.")
    return prototypes_from_sums(feature_sums, class_counts), class_counts


def tail_prototype_logits(image_features, class_prototypes, prototype_scale):
    image_features = normalize_features(image_features)
    class_prototypes = normalize_features(class_prototypes.to(device=image_features.device, dtype=image_features.dtype))
    return float(prototype_scale) * image_features @ class_prototypes.t()


def tail_class_weights(class_counts, gamma, max_weight=5.0):
    counts = class_counts.float()
    present = counts > 0
    if not present.any():
        raise ValueError("Cannot build tail weights without present classes.")

    weights = torch.zeros_like(counts)
    if float(gamma) == 0.0:
        weights[present] = 1.0
        return weights

    max_count = counts[present].max().clamp_min(1.0)
    raw_weights = (counts[present].clamp_min(1.0) / max_count).pow(-float(gamma))
    raw_weights = raw_weights.clamp(max=float(max_weight))
    weights[present] = raw_weights / raw_weights.mean().clamp_min(1e-12)
    return weights


def apply_tail_class_weights(logits, class_weights):
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    weights = torch.where(weights > 0, weights, torch.ones_like(weights))
    return logits * weights.unsqueeze(0)


def tail_prototype_loss(image_features, labels, class_prototypes, prototype_scale, class_counts=None):
    labels = labels.to(device=image_features.device, dtype=torch.long)
    if class_counts is not None:
        counts = class_counts.to(device=image_features.device)
        missing = counts[labels] <= 0
        if missing.any():
            missing_labels = sorted(set(labels[missing].detach().cpu().tolist()))
            raise ValueError(f"Tail prototypes are missing labels present in batch: {missing_labels}")
    logits = tail_prototype_logits(image_features, class_prototypes, prototype_scale)
    return F.cross_entropy(logits, labels)


def tail_prototype_distillation_loss(
    image_features,
    class_prototypes,
    prototype_scale,
    classification_head,
    temperature=1.0,
    class_counts=None,
):
    if float(temperature) <= 0.0:
        raise ValueError("Tail prototype distillation temperature must be positive.")

    student_logits = classification_head(image_features)
    with torch.no_grad():
        prototype_residual = tail_prototype_logits(image_features.detach(), class_prototypes, 1.0)
        if class_counts is not None:
            counts = class_counts.to(device=prototype_residual.device)
            missing = counts <= 0
            prototype_residual[:, missing] = 0.0
        teacher_logits = student_logits.detach() + float(prototype_scale) * prototype_residual

    temperature = float(temperature)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
