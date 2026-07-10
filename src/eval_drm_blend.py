import copy
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip.clip as clip
import src.datasets as datasets
import src.templates as templates
from src.config import parse_arguments
from src.datasets.dataloader import maybe_dictionarize
from src.models.clip_encoder import CLIPEncoder, ClassificationHead
from src.models.coop import maybe_data_parallel, unwrap_model
from src.models.flyp import wise_interpolate_state_dict
from src.train_coop import build_eval_dataset, get_validation_score, log_wandb_summary
from src.train_flyp import clone_state_dict, ensure_open_clip_for_flyp, parse_wise_alphas
from src.train_maple_full import init_wandb


class DrmBlendConfigurationError(ValueError):
    pass


def clone_args(args):
    return copy.copy(args)


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def get_classnames(args):
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=getattr(args, "workers", 0),
    )
    return dataset.classnames


def encode_text_classifier(args, clip_model, prompt_groups):
    clip_model.eval()
    clip_model.to(args.device)
    tokenizer_device = args.device
    weights = []
    with torch.no_grad():
        for prompts in tqdm(prompt_groups, desc="Building text classifier"):
            tokens = clip.tokenize(prompts).to(tokenizer_device)
            embeddings = clip_model.encode_text(tokens)
            embeddings = normalize_features(embeddings)
            embedding = embeddings.mean(dim=0, keepdim=True)
            embedding = normalize_features(embedding)
            weights.append(embedding)

        weights = torch.stack(weights, dim=0).to(args.device)
        weights = torch.transpose(weights, 0, 2)
        weights *= clip_model.logit_scale.exp()
        weights = weights.squeeze().float()
        weights = torch.transpose(weights, 0, 1)
    return ClassificationHead(normalize=True, weights=weights).to(args.device)


def load_concept_descriptions(path, class_count):
    with open(path) as file:
        raw = json.load(file)
    if len(raw) != class_count:
        raise DrmBlendConfigurationError(f"{path} has {len(raw)} concept descriptions, expected {class_count}.")
    descriptions = []
    for class_index in range(class_count):
        key = str(class_index)
        if key in raw:
            descriptions.append(raw[key])
        else:
            descriptions.append(list(raw.values())[class_index])
    return descriptions


def build_drm_classifiers(args, clip_model):
    if args.cd_path is None:
        raise DrmBlendConfigurationError("--cd-path is required for DRM blend evaluation.")
    if not 0.0 <= args.cd_beta <= 1.0:
        raise DrmBlendConfigurationError(f"--cd-beta must be in [0, 1], got {args.cd_beta}.")
    if args.template is None:
        args.template = "iwildcam_template"
    template_fns = getattr(templates, args.template)
    classnames = get_classnames(args)

    default_prompt_groups = [[template(classname) for template in template_fns] for classname in classnames]
    concept_descriptions = load_concept_descriptions(args.cd_path, len(classnames))
    concept_prompt_groups = [[description] for description in concept_descriptions]

    default_head = encode_text_classifier(args, clip_model, default_prompt_groups)
    concept_head = encode_text_classifier(args, clip_model, concept_prompt_groups)
    return default_head, concept_head


def metric_from_predictions(dataset, labels, predictions, metadata, correct, seen, args):
    metrics = {"top1": correct / max(seen, 1)}
    if hasattr(dataset, "post_loop_metrics"):
        labels = torch.cat(labels)
        predictions = torch.cat(predictions)
        metrics.update(dataset.post_loop_metrics(labels, predictions, metadata, args))
        if "acc" in metrics:
            metrics["top1"] = metrics["acc"]
    return metrics


def evaluate_drm_blend(model, dataset, args, default_head, concept_head):
    encoder = unwrap_model(model)
    encoder.eval()
    default_head.eval()
    concept_head.eval()

    state = {
        "default": {"labels": [], "predictions": [], "metadata": [], "correct": 0, "seen": 0},
        "concept": {"labels": [], "predictions": [], "metadata": [], "correct": 0, "seen": 0},
        "blend": {"labels": [], "predictions": [], "metadata": [], "correct": 0, "seen": 0},
    }

    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(dataset.test_loader, desc="DRM blend eval")):
            if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                break
            data = maybe_dictionarize(data)
            images = data["images"].to(args.device)
            labels = data["labels"].to(args.device)
            metadata = data["metadata"] if "metadata" in data else data.get("image_paths", [])

            features = encoder(images)
            default_logits = default_head(features)
            concept_logits = concept_head(features)
            default_probs = F.softmax(default_logits, dim=1)
            concept_probs = F.softmax(concept_logits, dim=1)
            blend_probs = args.cd_beta * default_probs + (1.0 - args.cd_beta) * concept_probs

            for name, predictions in (
                ("default", default_probs),
                ("concept", concept_probs),
                ("blend", blend_probs),
            ):
                predicted_labels = predictions.argmax(dim=1)
                state[name]["correct"] += (predicted_labels == labels).sum().item()
                state[name]["seen"] += labels.numel()
                if hasattr(dataset, "post_loop_metrics"):
                    state[name]["labels"].append(labels.cpu().clone())
                    state[name]["predictions"].append(predictions.cpu().clone())
                    state[name]["metadata"].extend(metadata)

    return {
        name: metric_from_predictions(
            dataset,
            values["labels"],
            values["predictions"],
            values["metadata"],
            values["correct"],
            values["seen"],
            args,
        )
        for name, values in state.items()
    }


def print_drm_summary(summary_rows):
    print("\n=== DRM Concept-Description Blend Summary ===")
    print("| Split         | Head    | Top-1  | F1-macro |")
    print("| ------------- | ------- | ------ | -------- |")
    for dataset_name, head_name, top1, f1_macro in summary_rows:
        top1_text = f"{top1 * 100:.2f}%" if top1 is not None else "N/A"
        f1_text = f"{f1_macro * 100:.2f}%" if f1_macro is not None else "N/A"
        print(f"| {dataset_name:<13} | {head_name:<7} | {top1_text:<6} | {f1_text:<8} |")


def load_eval_model(args):
    print(f"Loading fine-tuned FLYP checkpoint from {args.load}")
    model = maybe_data_parallel(CLIPEncoder.load(args.load).to(args.device), args)
    if args.wise_eval_alpha is None:
        return model

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


def main(args):
    if args.load is None:
        raise DrmBlendConfigurationError("--load must point to a FLYP CLIPEncoder checkpoint.")
    if args.train_dataset is None:
        raise DrmBlendConfigurationError("--train-dataset is required to align concept descriptions with class names.")
    if args.eval_datasets is None:
        raise DrmBlendConfigurationError("--eval-datasets is required.")
    if args.template is None:
        args.template = "iwildcam_template"
    ensure_open_clip_for_flyp(args.model)
    if args.wise_alphas:
        wise_alphas = parse_wise_alphas(args.wise_alphas)
        raise DrmBlendConfigurationError(f"Use --wise-eval-alpha for this eval-only script, not --wise-alphas={wise_alphas}.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.training_method = "drm_blend_eval"
    wandb = init_wandb(args)

    model = load_eval_model(args)
    default_head, concept_head = build_drm_classifiers(args, unwrap_model(model).model)

    summary_rows = []
    for dataset_name in args.eval_datasets:
        print(f"Evaluating DRM blend heads on {dataset_name}...")
        dataset = build_eval_dataset(dataset_name, unwrap_model(model), args)
        results_by_head = evaluate_drm_blend(model, dataset, args, default_head, concept_head)
        for head_name in ("default", "concept", "blend"):
            results = results_by_head[head_name]
            top1 = results.get("top1")
            f1_macro = results.get("F1-macro_all")
            print(f"  {dataset_name} {head_name} Top-1 accuracy: {top1:.4f}")
            if f1_macro is not None:
                print(f"  {dataset_name} {head_name} F1-macro_all: {f1_macro:.4f}")
            summary_rows.append((dataset_name, head_name, top1, f1_macro))
            if wandb is not None:
                wandb.log({
                    f"drm_blend/{dataset_name}/{head_name}/top1": top1,
                    f"drm_blend/{dataset_name}/{head_name}/f1_macro": f1_macro,
                    "drm_blend/cd_beta": args.cd_beta,
                })

    print_drm_summary(summary_rows)
    blend_rows = [(dataset, top1, f1) for dataset, head, top1, f1 in summary_rows if head == "blend"]
    log_wandb_summary(wandb, blend_rows)


if __name__ == "__main__":
    main(parse_arguments())
