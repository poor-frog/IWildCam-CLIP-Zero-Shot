import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "paper_grade_training_foundation_v0"


def _load(name):
    return json.loads((EXPERIMENT_DIR / name).read_text(encoding="utf-8"))


def test_preregistration_identity_and_predecessor_are_frozen():
    preregistration = _load("preregistration.json")
    closure_path = ROOT / preregistration["predecessor"]["closure_receipt_path"]

    assert preregistration["experiment"] == "Paper-Grade Training Foundation v0"
    assert preregistration["status"] == "locked_before_implementation_and_execution"
    assert preregistration["predecessor"]["frozen_outcome"] == "close_all_sequence_inference_development"
    assert hashlib.sha256(closure_path.read_bytes()).hexdigest() == preregistration["predecessor"]["closure_receipt_sha256"]


def test_seed_protocol_requires_five_unique_independent_runs():
    protocol = _load("preregistration.json")["independent_seed_protocol"]

    assert protocol["pilot_seeds"] == [20260721, 20260722, 20260723]
    assert protocol["completion_seeds"] == [20260724, 20260725]
    assert protocol["final_seed_set"] == protocol["pilot_seeds"] + protocol["completion_seeds"]
    assert len(set(protocol["final_seed_set"])) == 5
    assert protocol["checkpoint_reuse_across_seeds"] is False


def test_training_configuration_cannot_be_misreported_as_official_drm_or_flyp():
    preregistration = _load("preregistration.json")
    configuration = preregistration["frozen_training_configuration"]

    assert configuration["training_text"] == "deterministic class-label templates from iwildcam_template"
    assert configuration["loss"] == "multi-positive symmetric CLIP contrastive loss"
    assert configuration["drm_weight"] == 0.0
    assert configuration["tail_proto_weight"] == 0.0
    assert configuration["btel_weight"] == 0.0
    assert "official FLYP reproduction" in preregistration["claim_boundary"]["forbidden"]
    assert "official DRM reproduction" in preregistration["claim_boundary"]["forbidden"]


def test_selection_and_pilot_firewall_exclude_all_final_splits():
    preregistration = _load("preregistration.json")
    selection = preregistration["selection_protocol"]
    firewall = preregistration["data_firewall"]
    final_splits = set(firewall["final_evaluation_splits"])

    assert selection["selection_split"] == "IWildCamVal"
    assert selection["selection_metric"] == "F1-macro_all"
    assert final_splits <= set(selection["forbidden_selection_inputs"])
    assert final_splits <= set(firewall["pilot_forbidden_splits"])
    assert set(firewall["pilot_allowed_splits"]).isdisjoint(final_splits)
    assert firewall["cct20_opened"] is False


def test_stp_is_a_fixed_head_and_does_not_reopen_sequence_development():
    preregistration = _load("preregistration.json")
    heads = {head["name"]: head["configuration"] for head in preregistration["frozen_evaluation_heads"]}

    assert heads["label_template_flyp_wise_stp"] == (
        "prototype_count_per_class=1, prototype_scale=50, sequence_eta=0.5, self-inclusive mean"
    )
    assert "new sequence-inference method" in preregistration["claim_boundary"]["forbidden"]


def test_schema_locks_core_protocol_constants():
    schema = _load("preregistration.schema.json")
    properties = schema["properties"]

    assert properties["experiment"]["const"] == "Paper-Grade Training Foundation v0"
    assert properties["frozen_training_configuration"]["properties"]["drm_weight"]["const"] == 0.0
    assert properties["independent_seed_protocol"]["properties"]["final_seed_set"]["const"] == [
        20260721,
        20260722,
        20260723,
        20260724,
        20260725,
    ]
    assert properties["selection_protocol"]["properties"]["selection_split"]["const"] == "IWildCamVal"
    assert properties["data_firewall"]["properties"]["cct20_opened"]["const"] is False


def test_seeded_generators_reproduce_order_and_separate_seeds():
    from src.training_determinism import make_torch_generator

    first = torch.randperm(64, generator=make_torch_generator(20260721))
    repeated = torch.randperm(64, generator=make_torch_generator(20260721))
    different = torch.randperm(64, generator=make_torch_generator(20260722))

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_worker_seed_controls_python_and_numpy(monkeypatch):
    from src.training_determinism import seed_data_loader_worker

    monkeypatch.setattr(torch, "initial_seed", lambda: 123456)
    seed_data_loader_worker(7)
    first = (random.random(), np.random.random())
    seed_data_loader_worker(7)
    repeated = (random.random(), np.random.random())

    assert first == repeated


def test_determinism_receipt_is_hash_bound_and_immutable(tmp_path):
    from types import SimpleNamespace

    from src.training_determinism import build_determinism_receipt, write_json_receipt_refusing_overwrite

    args = SimpleNamespace(seed=20260721, model="ViT-B-16", device="cpu", deterministic_training=True)
    receipt = build_determinism_receipt(
        args,
        {"seed": 20260721, "source_commit": "abc"},
        best_epoch=9,
        selected_wise_alpha=0.1,
        amp_skipped_step_count=0,
        best_validation_score=0.42,
        selected_wise_score=0.43,
        validation_trace=[{"epoch": 9, "metric_value": 0.42}],
        wise_selection_trace=[{"alpha": 0.1, "metric_value": 0.43}],
    )
    output = tmp_path / "runtime_determinism_receipt.json"
    write_json_receipt_refusing_overwrite(output, receipt)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["configuration_sha256"] == hashlib.sha256(
        json.dumps(loaded["configuration"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert loaded["training"]["non_finite_loss_or_gradient_observed"] is False
    assert loaded["training"]["best_validation_score"] == 0.42
    assert loaded["training"]["wise_selection_trace"][0]["alpha"] == 0.1
    with pytest.raises(FileExistsError):
        write_json_receipt_refusing_overwrite(output, receipt)


def test_iwildcam_loaders_receive_seeded_generators_and_worker_initializer():
    import pandas as pd

    from src.datasets.iwildcam import IWildCam
    from src.training_determinism import seed_data_loader_worker

    class FakeDataset:
        n_classes = 2

        def get_subset(self, split, transform=None):
            return object()

    with patch("src.datasets.iwildcam.wilds.get_dataset", return_value=FakeDataset()), \
            patch("src.datasets.iwildcam.get_train_loader", return_value=object()) as train_loader, \
            patch("src.datasets.iwildcam.get_eval_loader", return_value=object()) as eval_loader, \
            patch(
                "src.datasets.iwildcam.pd.read_csv",
                return_value=pd.DataFrame({"y": [0, 1], "english": ["Empty", "Frog"]}),
            ):
        IWildCam(object(), location="unused", batch_size=4, num_workers=2, seed=20260721)

    train_kwargs = train_loader.call_args.kwargs
    eval_kwargs = eval_loader.call_args.kwargs
    assert train_kwargs["generator"].initial_seed() == 20260721
    assert eval_kwargs["generator"].initial_seed() == 20260721
    assert train_kwargs["worker_init_fn"] is seed_data_loader_worker
    assert eval_kwargs["worker_init_fn"] is seed_data_loader_worker


def _paper_grade_args(run_directory, seed=20260721, **overrides):
    values = {
        "seed": seed,
        "deterministic_training": True,
        "load": None,
        "train_dataset": "IWildCam",
        "val_dataset": "IWildCamVal",
        "eval_datasets": None,
        "determinism_receipt": str(run_directory / "run_receipt.json"),
        "save": str(run_directory / "final_checkpoint.pt"),
        "best_checkpoint": str(run_directory / "best_checkpoint.pt"),
        "wandb_run_name": f"pgf-v0-seed-{seed}",
        "model": "ViT-B-16",
        "template": "iwildcam_template",
        "epochs": 20,
        "batch_size": 256,
        "workers": 4,
        "lr": 0.00001,
        "wd": 0.2,
        "lr_scheduler": "cosine",
        "warmup_length": 0,
        "maple_precision": "amp",
        "best_metric": "F1-macro_all",
        "drm_weight": 0.0,
        "drm_warmup_epochs": 0,
        "tail_proto_weight": 0.0,
        "btel_weight": 0.0,
        "wise_alphas": "0,0.05,0.1,0.15,0.2,0.3",
        "wise_eval_alpha": None,
        "max_train_batches": None,
        "max_eval_batches": None,
        "num_ood_hp_examples": -1,
        "class_balanced_ood": False,
        "cd_path": None,
        "no_load_best_for_eval": False,
        "btel_audit_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validation_split_firewall_blocks_before_recording_forbidden_split():
    from src.models.paper_grade_training_foundation import ValidationSplitFirewall

    firewall = ValidationSplitFirewall()
    firewall.record("IWildCam")
    firewall.record("IWildCamVal")

    with pytest.raises(ValueError, match="firewall blocked"):
        firewall.record("IWildCamOOD")

    assert firewall.receipt_payload()["accessed_datasets"] == ["IWildCam", "IWildCamVal"]
    assert firewall.receipt_payload()["passed"] is True


def test_validation_run_requires_unique_seed_bound_artifact_paths(tmp_path):
    from src.models.paper_grade_training_foundation import validate_paper_grade_validation_run

    run_directory = tmp_path / "seed-20260721"
    args = _paper_grade_args(run_directory)
    paths = validate_paper_grade_validation_run(args, args.best_checkpoint)

    assert paths.run_directory == run_directory.resolve()
    assert paths.receipt.name == "run_receipt.json"

    args.eval_datasets = ["IWildCamOOD"]
    with pytest.raises(ValueError, match="blocked eval datasets"):
        validate_paper_grade_validation_run(args, args.best_checkpoint)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 1),
        ("lr", 9.9),
        ("maple_precision", "fp32"),
        ("drm_weight", 1.0),
        ("wise_alphas", "1.0"),
        ("max_train_batches", 1),
    ],
)
def test_validation_run_rejects_frozen_configuration_drift(tmp_path, field, value):
    from src.models.paper_grade_training_foundation import validate_paper_grade_validation_run

    args = _paper_grade_args(tmp_path / "seed-20260721", **{field: value})

    with pytest.raises(ValueError, match=field):
        validate_paper_grade_validation_run(args, args.best_checkpoint)


def test_validation_run_refuses_existing_checkpoint(tmp_path):
    from src.models.paper_grade_training_foundation import validate_paper_grade_validation_run

    run_directory = tmp_path / "seed-20260721"
    run_directory.mkdir()
    args = _paper_grade_args(run_directory)
    Path(args.save).write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validate_paper_grade_validation_run(args, args.best_checkpoint)


def test_validation_run_refuses_nonempty_seed_directory(tmp_path):
    from src.models.paper_grade_training_foundation import validate_paper_grade_validation_run

    run_directory = tmp_path / "seed-20260721"
    run_directory.mkdir()
    (run_directory / "stale.log").write_text("old run", encoding="utf-8")
    args = _paper_grade_args(run_directory)

    with pytest.raises(FileExistsError, match="run directory is not empty"):
        validate_paper_grade_validation_run(args, args.best_checkpoint)


def test_protocol_configuration_hash_ignores_only_preregistered_run_fields(tmp_path):
    from src.models.paper_grade_training_foundation import protocol_configuration
    from src.training_determinism import sha256_json

    first = _paper_grade_args(tmp_path / "seed-20260721", seed=20260721)
    second = _paper_grade_args(tmp_path / "seed-20260722", seed=20260722)

    assert sha256_json(protocol_configuration(first)) == sha256_json(protocol_configuration(second))
    second.model = "ViT-L-14"
    assert sha256_json(protocol_configuration(first)) != sha256_json(protocol_configuration(second))


def test_source_provenance_hash_changes_with_source_content(tmp_path):
    from src.models.paper_grade_training_foundation import build_source_provenance

    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = build_source_provenance(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = build_source_provenance(tmp_path)

    assert first["hashed_file_count"] == 1
    assert first["source_tree_sha256"] != second["source_tree_sha256"]


def test_dataset_provenance_binds_metadata_labels_and_splits(tmp_path):
    from src.models.paper_grade_training_foundation import build_iwildcam_dataset_provenance

    data_dir = tmp_path / "iwildcam_v2.0"
    data_dir.mkdir()
    (data_dir / "metadata.csv").write_text("image_id,y\n1,0\n2,1\n", encoding="utf-8")
    dataset = SimpleNamespace(
        _data_dir=data_dir,
        version="2.0",
        n_classes=2,
        split_dict={"train": 0, "val": 1},
        split_array=np.array([0, 1], dtype=np.int64),
        y_array=torch.tensor([0, 1]),
    )

    provenance = build_iwildcam_dataset_provenance(dataset, tmp_path)

    assert provenance["metadata_csv"]["sha256"] == hashlib.sha256(
        (data_dir / "metadata.csv").read_bytes()
    ).hexdigest()
    assert provenance["split_array"]["sha256"]
    assert provenance["label_array"]["sha256"]
    assert provenance["dataset_snapshot_sha256"]


def test_paper_grade_receipt_contains_all_hash_bound_provenance(tmp_path):
    from src.models.paper_grade_training_foundation import (
        build_checkpoint_provenance,
        enrich_paper_grade_run_receipt,
        validate_paper_grade_validation_run,
    )

    run_directory = tmp_path / "seed-20260721"
    run_directory.mkdir()
    args = _paper_grade_args(run_directory)
    paths = validate_paper_grade_validation_run(args, args.best_checkpoint)
    paths.best_checkpoint.write_bytes(b"best")
    paths.final_checkpoint.write_bytes(b"final")
    checkpoint_provenance = build_checkpoint_provenance(paths)
    validation_trace = [
        {"epoch": epoch, "metric_value": epoch / 20, "metric_name": "F1-macro_all"}
        for epoch in range(1, 21)
    ]
    wise_alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    wise_trace = [
        {"alpha": alpha, "metric_value": index / 10, "metric_name": "F1-macro_all"}
        for index, alpha in enumerate(wise_alphas)
    ]
    receipt = enrich_paper_grade_run_receipt(
        {
            "status": "complete",
            "seed": args.seed,
            "training": {
                "best_epoch": 20,
                "best_validation_score": 1.0,
                "selected_wise_alpha": 0.3,
                "selected_wise_score": 0.5,
                "validation_trace": validation_trace,
                "wise_selection_trace": wise_trace,
                "amp_skipped_step_count": 0,
                "non_finite_loss_or_gradient_observed": False,
            },
        },
        args,
        source_provenance={"git_commit": "abc", "source_tree_sha256": "source-hash"},
        dataset_provenance={"dataset_snapshot_sha256": "dataset-hash"},
        checkpoint_provenance=checkpoint_provenance,
        split_firewall={"passed": True, "accessed_datasets": ["IWildCam", "IWildCamVal"]},
    )

    assert receipt["receipt"] == "paper_grade_training_foundation_v0_validation_run"
    assert receipt["provenance"]["checkpoints"]["best_validation"]["sha256"] == hashlib.sha256(b"best").hexdigest()
    assert receipt["provenance"]["checkpoints"]["final_wise"]["sha256"] == hashlib.sha256(b"final").hexdigest()
    assert receipt["status"] == "complete"
    assert all(receipt["validity"].values())


@pytest.mark.parametrize(
    ("training_override", "failed_gate"),
    [
        ({"best_epoch": None}, "selection_complete"),
        ({"selected_wise_alpha": None}, "selection_complete"),
        ({"amp_skipped_step_count": 1}, "zero_amp_skipped_steps"),
        ({"non_finite_loss_or_gradient_observed": True}, "no_non_finite_event"),
    ],
)
def test_paper_grade_receipt_is_invalid_when_training_gate_fails(tmp_path, training_override, failed_gate):
    from src.models.paper_grade_training_foundation import enrich_paper_grade_run_receipt

    args = _paper_grade_args(tmp_path / "seed-20260721")
    validation_trace = [{"epoch": epoch, "metric_value": epoch / 20} for epoch in range(1, 21)]
    wise_alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    training = {
        "best_epoch": 20,
        "best_validation_score": 1.0,
        "selected_wise_alpha": 0.3,
        "selected_wise_score": 0.5,
        "validation_trace": validation_trace,
        "wise_selection_trace": [
            {"alpha": alpha, "metric_value": index / 10} for index, alpha in enumerate(wise_alphas)
        ],
        "amp_skipped_step_count": 0,
        "non_finite_loss_or_gradient_observed": False,
    }
    training.update(training_override)

    receipt = enrich_paper_grade_run_receipt(
        {"status": "complete", "seed": args.seed, "training": training},
        args,
        source_provenance={"source_tree_sha256": "source"},
        dataset_provenance={"dataset_snapshot_sha256": "dataset"},
        checkpoint_provenance={"best_validation": {"sha256": "best"}, "final_wise": {"sha256": "final"}},
        split_firewall={"passed": True},
    )

    assert receipt["status"] == "invalid"
    assert receipt["validity"][failed_gate] is False
