import hashlib
import json
import random
from pathlib import Path

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
    )
    output = tmp_path / "runtime_determinism_receipt.json"
    write_json_receipt_refusing_overwrite(output, receipt)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["configuration_sha256"] == hashlib.sha256(
        json.dumps(loaded["configuration"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert loaded["training"]["non_finite_loss_or_gradient_observed"] is False
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
