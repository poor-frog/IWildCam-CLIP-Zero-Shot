from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from src.models.lesvi import LesviPrior, load_prior_artifact
from src.models.lesvi_cct20 import (
    CCT_ROTATION_SEEDS,
    CctRecord,
    build_cct_evaluation_support,
    cct_label_diagnostics,
    evaluate_cct_lesvi,
    file_sha256,
    load_cct_records,
    verify_lesvi_cct_stage,
    write_lesvi_cct_receipt,
)
from src.models.lesvi_evaluation import write_metric_support
from src.models.stp_confirmation import build_source_bundle_checksum


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen LESVI-v0 on one CCT-20 protocol stage.")
    parser.add_argument("--stage", choices=("cis_validation", "trans_validation", "trans_test"), required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--stage-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frozen-spec", type=Path, required=True)
    parser.add_argument("--prior-artifact", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--prior-viability", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--next-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def verify_runtime_bundle(args: argparse.Namespace) -> tuple[dict[str, object], tuple[str, ...], LesviPrior]:
    spec = verify_lesvi_cct_stage(args.stage, args.frozen_spec, args.ledger)
    source_files = spec.get("source_files")
    if not isinstance(source_files, list) or not all(isinstance(value, str) for value in source_files):
        raise ValueError("CCT LESVI frozen specification has no valid source bundle list.")
    source_sha256 = build_source_bundle_checksum(args.workspace_root.resolve(), source_files)
    if source_sha256 != spec["source_bundle_sha256"]:
        raise ValueError("CCT LESVI runtime source bundle differs from the frozen source bundle.")
    expected_artifacts = (
        (args.prior_artifact, "prior_artifact_sha256"),
        (args.class_mapping, "class_mapping_artifact_sha256"),
        (args.split_manifest, "split_manifest_sha256"),
        (args.prior_viability, "prior_viability_sha256"),
    )
    for path, field in expected_artifacts:
        if file_sha256(path) != spec.get(field):
            raise ValueError(f"CCT LESVI {field} does not match the frozen specification.")
    annotation_hashes = spec.get("annotation_sha256")
    if not isinstance(annotation_hashes, dict):
        raise ValueError("CCT LESVI frozen annotation checksums are missing.")
    if file_sha256(args.train_annotations) != annotation_hashes.get("train"):
        raise ValueError("CCT train annotation checksum differs from the frozen specification.")
    if file_sha256(args.stage_annotations) != annotation_hashes.get(args.stage):
        raise ValueError(f"CCT {args.stage} annotation checksum differs from the frozen specification.")
    mapping = _load_json(args.class_mapping, "CCT class mapping")
    viability = _load_json(args.prior_viability, "CCT prior viability")
    if viability.get("viability_pass") is not True:
        raise ValueError("CCT train-only LESVI prior failed frozen viability.")
    class_names = mapping.get("class_names")
    if not isinstance(class_names, list) or not class_names or not all(isinstance(value, str) for value in class_names):
        raise ValueError("CCT class mapping must contain ordered class_names.")
    if class_names[0] != "empty" or mapping.get("empty_class_index") != 0:
        raise ValueError("CCT LESVI requires canonical class 0 to be empty.")
    if mapping.get("class_mapping_sha256") != spec["class_mapping_sha256"]:
        raise ValueError("CCT ordered class mapping differs from the frozen specification.")
    prior = load_prior_artifact(args.prior_artifact, expected_class_mapping_sha256=str(spec["class_mapping_sha256"]))
    return spec, tuple(class_names), prior


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def _encode_records(model, preprocess, records: Sequence[CctRecord], device, batch_size: int, workers: int):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class ImageDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            with Image.open(records[index].image_path) as image:
                return preprocess(image.convert("RGB"))

    missing = [str(record.image_path) for record in records if not record.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"CCT image files are missing; first missing path: {missing[0]}")
    loader = DataLoader(
        ImageDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    encoded = []
    with torch.inference_mode():
        for images in loader:
            features = model.encode_image(images.to(device))
            encoded.append(F.normalize(features.float(), dim=1).cpu())
    if not encoded:
        raise ValueError("CCT split contains no images.")
    return torch.cat(encoded, dim=0)


def _encode_text_directions(model, tokenizer, class_names: Sequence[str], device):
    import torch
    import torch.nn.functional as F

    directions = []
    with torch.inference_mode():
        for class_name in class_names:
            prompts = [f"a photo of {class_name}.", f"{class_name} in the wild."]
            embeddings = F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=1)
            directions.append(F.normalize(embeddings.mean(dim=0), dim=0).cpu())
    return torch.stack(directions)


def _class_prototypes(features, records: Sequence[CctRecord], class_count: int):
    import torch
    import torch.nn.functional as F

    labels = torch.tensor([record.label for record in records], dtype=torch.long)
    sums = torch.zeros(class_count, features.shape[1], dtype=torch.float32)
    counts = torch.zeros(class_count, dtype=torch.long)
    sums.index_add_(0, labels, features.float())
    counts.index_add_(0, labels, torch.ones_like(labels))
    prototypes = torch.zeros_like(sums)
    present = counts > 0
    prototypes[present] = F.normalize(sums[present] / counts[present, None].float(), dim=1)
    return prototypes, present, counts


def run_evaluation(args: argparse.Namespace, class_names: Sequence[str], prior: LesviPrior) -> tuple[dict[str, object], object, object]:
    import open_clip
    import torch

    from src.models.vfep_cct20 import build_cct_tpa_logits

    train_records, train_names, train_mapping = load_cct_records(args.train_annotations, args.image_root)
    stage_records, stage_names, stage_mapping = load_cct_records(args.stage_annotations, args.image_root)
    if tuple(train_names) != tuple(class_names) or tuple(stage_names) != tuple(class_names):
        raise ValueError("CCT runtime class names differ from the frozen class mapping.")
    if train_mapping != prior.class_mapping_sha256 or stage_mapping != prior.class_mapping_sha256:
        raise ValueError("CCT runtime class checksum differs from the frozen prior.")
    labels = torch.tensor([record.label for record in stage_records], dtype=torch.long)
    frozen_support = build_cct_evaluation_support(stage_records, labels, CCT_ROTATION_SEEDS)
    device = _resolve_device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    train_features = _encode_records(model, preprocess, train_records, device, args.batch_size, args.workers)
    stage_features = _encode_records(model, preprocess, stage_records, device, args.batch_size, args.workers)
    text_directions = _encode_text_directions(model, tokenizer, class_names, device)
    prototypes, present, counts = _class_prototypes(train_features, train_records, len(class_names))
    tpa_logits, base_diagnostics = build_cct_tpa_logits(
        stage_features,
        text_directions,
        prototypes,
        present,
        model.logit_scale.detach().cpu(),
    )
    report, _ = evaluate_cct_lesvi(
        stage=args.stage,
        labels=labels,
        tpa_logits=tpa_logits,
        records=stage_records,
        prior=prior,
        frozen_support=frozen_support,
    )
    report["cct20"]["base_diagnostics"] = base_diagnostics
    report["cct20"]["train_class_image_counts"] = counts.tolist()
    report["cct20"]["class_names"] = list(class_names)
    report["cct20"]["stage_frame_count"] = len(stage_records)
    report["cct20"]["stage_location_count"] = len({record.location for record in stage_records})
    report["cct20"]["label_diagnostics"] = cct_label_diagnostics(args.stage_annotations)
    return report, labels, frozen_support.metric_support


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main(args: argparse.Namespace) -> None:
    if args.next_ledger.resolve() == args.ledger.resolve():
        raise ValueError("CCT LESVI next ledger must not overwrite the immutable input ledger.")
    _, class_names, prior = verify_runtime_bundle(args)
    final_output = args.output_dir.resolve()
    if final_output.exists():
        raise FileExistsError(f"CCT LESVI output already exists: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{final_output.name}.", dir=final_output.parent))
    try:
        report, labels, metric_support = run_evaluation(args, class_names, prior)
        support_path = temporary_output / "confirm_metric_support.json"
        support_sha256 = write_metric_support(support_path, metric_support, labels)
        report["cct20"]["metric_support_sha256"] = support_sha256
        report_path = temporary_output / "lesvi_cct20_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_ledger = temporary_output / "lesvi_cct20_next_ledger.json"
        shutil.copy2(args.ledger, temporary_ledger)
        write_lesvi_cct_receipt(
            args.stage,
            report_path,
            args.frozen_spec,
            temporary_ledger,
            temporary_output / "lesvi_cct20_receipt.json",
        )
        os.replace(temporary_output, final_output)
        _atomic_copy(final_output / "lesvi_cct20_next_ledger.json", args.next_ledger.resolve())
    except BaseException:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise
    print(f"CCT-20 LESVI {args.stage} report written to {final_output}")


if __name__ == "__main__":
    main(parse_arguments())
