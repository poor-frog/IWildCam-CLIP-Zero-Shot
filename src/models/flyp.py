from dataclasses import dataclass
from contextlib import nullcontext

import open_clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.dataloader import maybe_dictionarize
from src.device import optimizer_step
from src.models.clip_encoder import ImageClassifier
from src.models.coop import eval_coop_single_dataset
from src.models.coop import unwrap_model
from src.models.zeroshot import get_zeroshot_classifier


@dataclass
class FlypTrainStats:
    epoch: int
    loss: float
    clip_loss: float = 0.0
    drm_loss: float = 0.0
    lr: float | None = None


DRM_CHUNK_SIZE = 1_000_000


def build_flyp_captions(labels, classnames, templates, offset=0):
    captions = []
    template_count = len(templates)
    if template_count == 0:
        raise ValueError("FLYP requires at least one text template.")
    for sample_index, label in enumerate(labels.detach().cpu().tolist()):
        class_index = int(label)
        if class_index < 0 or class_index >= len(classnames):
            raise IndexError(f"Label {class_index} is outside classnames length {len(classnames)}.")
        template = templates[(offset + sample_index) % template_count]
        captions.append(template(classnames[class_index]))
    return captions


def multi_positive_cross_entropy(logits, positive_mask):
    log_probs = F.log_softmax(logits, dim=1)
    masked_log_probs = log_probs.masked_fill(~positive_mask, float("-inf"))
    return -torch.logsumexp(masked_log_probs, dim=1).mean()


def compute_flyp_clip_loss(image_features, text_features, logit_scale, class_labels=None):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = logit_scale.flatten()[0] if hasattr(logit_scale, "flatten") else logit_scale
    logits_per_image = scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    if class_labels is None:
        return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

    class_labels = class_labels.to(logits_per_image.device)
    positive_mask = class_labels[:, None].eq(class_labels[None, :])
    image_loss = multi_positive_cross_entropy(logits_per_image, positive_mask)
    text_loss = multi_positive_cross_entropy(logits_per_text, positive_mask.t())
    return (image_loss + text_loss) / 2


def unpack_clip_forward(outputs):
    if isinstance(outputs, dict):
        return outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
    if len(outputs) < 3:
        raise ValueError("FLYP expects CLIP forward to return image_features, text_features, and logit_scale.")
    return outputs[0], outputs[1], outputs[2]


def compute_drm_loss(model, init_state_dict, drm_weight):
    model = unwrap_model(model)
    if drm_weight == 0.0:
        return next(model.parameters()).new_zeros(())

    total = None
    current_state = dict(model.named_parameters())
    for name, param in current_state.items():
        if not param.requires_grad:
            continue
        if name not in init_state_dict:
            raise KeyError(f"Initial state dict is missing parameter {name!r}.")
        param_flat = param.flatten()
        initial_flat = init_state_dict[name].flatten()
        for start in range(0, param_flat.numel(), DRM_CHUNK_SIZE):
            end = start + DRM_CHUNK_SIZE
            param_chunk = param_flat[start:end]
            initial_chunk = initial_flat[start:end].to(device=param.device, dtype=param.dtype)
            component = (param_chunk - initial_chunk).pow(2).sum()
            total = component if total is None else total + component

    if total is None:
        return next(model.parameters()).new_zeros(())
    return float(drm_weight) * total


def wise_interpolate_state_dict(finetuned_state_dict, zeroshot_state_dict, alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("WiSE alpha must be in [0, 1].")
    finetuned_keys = set(finetuned_state_dict.keys())
    zeroshot_keys = set(zeroshot_state_dict.keys())
    if finetuned_keys != zeroshot_keys:
        missing_from_zeroshot = sorted(finetuned_keys - zeroshot_keys)
        missing_from_finetuned = sorted(zeroshot_keys - finetuned_keys)
        raise ValueError(
            "WiSE interpolation requires matching state dict keys. "
            f"Missing from zeroshot={missing_from_zeroshot}, missing from finetuned={missing_from_finetuned}"
        )

    interpolated = {}
    for name, finetuned_tensor in finetuned_state_dict.items():
        zeroshot_tensor = zeroshot_state_dict[name].to(device=finetuned_tensor.device, dtype=finetuned_tensor.dtype)
        if not torch.is_floating_point(finetuned_tensor):
            if not torch.equal(finetuned_tensor, zeroshot_tensor):
                raise ValueError(f"WiSE cannot interpolate changed non-floating tensor {name!r}.")
            interpolated[name] = finetuned_tensor.clone()
            continue
        interpolated[name] = (1.0 - alpha) * finetuned_tensor + alpha * zeroshot_tensor
    return interpolated


def flyp_zeroshot_cache_key(args, clip_encoder):
    model = clip_encoder.model
    state_versions = tuple((name, param._version) for name, param in model.named_parameters())
    return (
        getattr(args, "device", None),
        getattr(args, "train_dataset", None),
        getattr(args, "template", None),
        getattr(args, "data_location", None),
        getattr(args, "batch_size", None),
        state_versions,
    )


def get_cached_flyp_zeroshot_classifier(args, clip_encoder):
    cache_key = flyp_zeroshot_cache_key(args, clip_encoder)
    cached = getattr(clip_encoder, "_flyp_zeroshot_classifier_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["classifier"]

    classifier = get_zeroshot_classifier(args, clip_encoder.model).to(args.device)
    clip_encoder._flyp_zeroshot_classifier_cache = {"key": cache_key, "classifier": classifier}
    return classifier


def train_flyp_one_epoch(
    model,
    dataloader,
    optimizer,
    args,
    classnames,
    templates,
    epoch,
    init_state_dict=None,
    drm_weight=0.0,
    scheduler=None,
    scaler=None,
):
    model.train()
    tokenizer = open_clip.get_tokenizer(args.model)
    total_loss = 0.0
    total_clip_loss = 0.0
    total_drm_loss = 0.0
    total_seen = 0
    max_batches = args.max_train_batches

    for batch_index, data in enumerate(tqdm(dataloader, desc=f"FLYP train epoch {epoch}")):
        if max_batches is not None and batch_index >= max_batches:
            break

        data = maybe_dictionarize(data)
        images = data["images"].to(args.device)
        labels = data["labels"]
        captions = build_flyp_captions(labels, classnames, templates, offset=batch_index)
        text_tokens = tokenizer(captions).to(args.device)

        use_autocast = getattr(args, "use_amp", False) and str(args.device).startswith("cuda")
        autocast_context = torch.amp.autocast("cuda", enabled=True) if use_autocast else nullcontext()
        with autocast_context:
            image_features, text_features, logit_scale = unpack_clip_forward(model(images, text_tokens))
            clip_loss = compute_flyp_clip_loss(image_features, text_features, logit_scale, class_labels=labels)
            if drm_weight != 0.0:
                if init_state_dict is None:
                    raise ValueError("init_state_dict is required when drm_weight is non-zero.")
                drm_loss = compute_drm_loss(model, init_state_dict, drm_weight)
            else:
                drm_loss = clip_loss.new_zeros(())
            loss = clip_loss + drm_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"FLYP produced non-finite loss at epoch {epoch}, batch {batch_index}")

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if scaler is None:
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"FLYP produced non-finite gradient for {name} at epoch {epoch}, batch {batch_index}")
        previous_scale = scaler.get_scale() if scaler is not None and hasattr(scaler, "get_scale") else None
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            current_scale = scaler.get_scale() if hasattr(scaler, "get_scale") else previous_scale
            optimizer_was_skipped = previous_scale is not None and current_scale < previous_scale
        else:
            optimizer_step(optimizer, args.device)
            optimizer_was_skipped = False
        if scheduler is not None and not optimizer_was_skipped:
            scheduler.step()

        batch_size = images.shape[0]
        total_loss += (clip_loss.item() + drm_loss.item()) * batch_size
        total_clip_loss += clip_loss.item() * batch_size
        total_drm_loss += drm_loss.item() * batch_size
        total_seen += batch_size

    if total_seen == 0:
        raise RuntimeError("FLYP saw zero training examples.")
    current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
    return FlypTrainStats(
        epoch=epoch,
        loss=total_loss / total_seen,
        clip_loss=total_clip_loss / total_seen,
        drm_loss=total_drm_loss / total_seen,
        lr=current_lr,
    )


def eval_flyp_single_dataset(model, dataset, args):
    clip_encoder = unwrap_model(model)
    classification_head = get_cached_flyp_zeroshot_classifier(args, clip_encoder)
    image_classifier = ImageClassifier(clip_encoder, classification_head).to(args.device)
    return eval_coop_single_dataset(image_classifier, dataset, args, desc="FLYP eval")
