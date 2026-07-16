import torch
import torch.nn.functional as F


def test_loo_bcpd_candidate_rows_keep_other_adapter_heads_unchanged():
    from src.eval_tail_cache import make_candidate_rows

    rows = make_candidate_rows(
        prototype_scale_grid=[50.0],
        tau_grid=[0.0],
        tail_gamma_grid=[0.0],
        gate_mode_grid=["none"],
        gate_strength_grid=[0.0],
        sequence_eta_grid=[0.0],
        prototype_k_grid=[1],
        concept_beta_grid=[0.5],
        include_concept=False,
        loo_bcpd_strength_grid=[0.25, 0.5],
    )

    loo_rows = [row for row in rows if row["head"] == "loo_bcpd"]
    prototype_row = next(row for row in rows if row["head"] == "prototype")

    assert [row["loo_bcpd_strength"] for row in loo_rows] == [0.25, 0.5]
    assert prototype_row["loo_bcpd_strength"] == 0.0
    assert all(row["sequence_eta"] == 0.0 for row in loo_rows)


def test_loo_bcpd_head_uses_precomputed_logits_without_sequence_postprocessing():
    from src.eval_tail_cache import build_candidate_predictions

    base_logits = torch.tensor([[2.0, 1.0]])
    expected = torch.tensor([[0.5, 4.0]])
    actual = build_candidate_predictions(
        base_logits,
        prototype_raw_logits_by_k={1: torch.tensor([[0.0, 0.0]])},
        concept_raw_logits=None,
        class_priors=torch.tensor([0.5, 0.5]),
        tail_weights_by_gamma={0.0: torch.ones(2)},
        row={
            "head": "loo_bcpd",
            "prototype_scale": 50.0,
            "prototype_k": 1,
            "tau": 0.0,
            "tail_gamma": 0.0,
            "sequence_eta": 0.0,
            "gate_mode": "none",
            "gate_strength": 0.0,
            "sctr_strength": 0.0,
            "sctr_tail_protection": 0.0,
            "concept_beta": None,
            "loo_bcpd_strength": 0.5,
        },
        loo_bcpd_logits_by_strength={0.5: expected},
    )

    assert torch.equal(actual, expected)


def test_selection_output_displays_loo_bcpd_strength(capsys):
    from src.eval_tail_cache import print_selection

    row = {
        "head": "loo_bcpd",
        "prototype_scale": 50.0,
        "tau": 0.0,
        "tail_gamma": 0.0,
        "prototype_k": 1,
        "sequence_eta": 0.0,
        "gate_mode": "none",
        "gate_strength": 0.0,
        "sctr_strength": 0.0,
        "sctr_tail_protection": 0.0,
        "concept_beta": None,
        "loo_bcpd_strength": 0.5,
        "score": 0.4,
        "top1": 0.5,
        "F1-macro_all": 0.4,
    }

    print_selection([row], {"loo_bcpd": row})

    output = capsys.readouterr().out
    assert "Adapter strength" in output
    assert "| 0.5" in output


def test_class_mapping_checksum_rejects_classifier_with_wrong_class_count():
    from src.eval_tail_cache import class_mapping_checksum

    classifier = torch.nn.Linear(4, 2, bias=False)

    try:
        class_mapping_checksum(["empty", "species_a", "species_b"], classifier)
    except ValueError as error:
        assert "Classifier rows" in str(error)
    else:
        raise AssertionError("Expected a class-count mismatch to fail.")


def test_loo_bcpd_diagnostics_writes_controls_and_bootstrap_report(tmp_path):
    from src.eval_tail_cache import write_loo_bcpd_diagnostics
    from src.models.loo_bcpd import LooBcpdResult, ShuffledGroups

    labels = torch.tensor([0, 0, 1, 1])
    tpa = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    bcpd = torch.tensor([[3.0, 0.0], [2.0, 1.0], [0.0, 3.0], [1.0, 2.0]])
    result = LooBcpdResult(
        logits=bcpd,
        prototype_scores=F.normalize(bcpd, dim=1),
        valid_burst_count=2,
        corrected_frame_count=4,
        mean_responsibility_entropy=0.1,
        mean_max_responsibility=0.9,
        mean_effective_class_count=1.1,
        mean_support_mass=0.5,
        mean_rotation_degrees=2.0,
        mean_normalization_penalty=0.01,
    )
    output = tmp_path / "loo_bcpd.md"
    metadata = [torch.tensor([1]), torch.tensor([1]), torch.tensor([2]), torch.tensor([2])]

    write_loo_bcpd_diagnostics(
        output,
        "IWildCamVal",
        dataset=object(),
        args=object(),
        labels=labels,
        metadata=metadata,
        sequence_field_index=0,
        train_class_counts=torch.tensor([10, 100]),
        class_checksum="checksum",
        logits_by_name={"frame": tpa, "tpa": tpa, "loo_bcpd": bcpd, "class_derangement": tpa},
        selected_result=result,
        shuffled=ShuffledGroups(groups=((0, 1), (2, 3)), changed_frame_fraction=0.5, unavailable_group_count=0),
        selected_strength=0.5,
        bootstrap_samples=20,
        seed=1,
    )

    report = output.read_text(encoding="utf-8")
    assert "Paired Sequence Bootstrap" in report
    assert "class_derangement" in report
    assert "checksum" in report
    assert "Selected BCPD strength: 0.5" in report
    assert "not reliable" in report
