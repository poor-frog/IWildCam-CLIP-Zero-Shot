import torch


def test_audit_report_uses_location_primary_and_sequence_sensitivity():
    from src.models.stp_audit_report import build_mechanism_audit_report

    labels = torch.tensor([0, 1, 0, 1])
    tpa = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
    mean = torch.tensor([[2.0, 1.0], [1.0, 2.0], [1.5, 0.5], [0.5, 1.5]])
    loo = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    metadata = [torch.tensor([index // 2]) for index in range(4)]
    report = build_mechanism_audit_report(
        labels=labels,
        tpa_logits=tpa,
        stp_mean_logits=mean,
        stp_loo_logits=loo,
        metadata=metadata,
        location_keys=("a", "a", "b", "b"),
        sequence_field_index=0,
        train_class_counts=torch.tensor([30, 10]),
        bootstrap_samples=16,
        seed=7,
    )

    comparison = report["tpa_to_stp_mean"]
    assert "location_bootstrap" in comparison
    assert "sequence_bootstrap_sensitivity" in comparison
    assert set(report["audit_bin_edges"]) == {"confidence", "margin"}


def test_audit_bin_edges_are_derived_only_from_tpa_logits():
    from src.models.stp_audit_report import audit_bin_edges

    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [1.0, 1.0], [2.0, 0.0]])

    first = audit_bin_edges(logits)
    second = audit_bin_edges(logits)

    assert first == second


def test_phase_a_artifacts_do_not_materialize_confirm_performance(tmp_path):
    from src.models.stp_audit_report import write_audit_artifacts

    write_audit_artifacts(
        tmp_path,
        {"phase": "val_audit_only"},
        {"confirm_location_count": 12, "viability_pass": True},
        {"tpa_to_stp_mean": {"reference_macro_f1": 0.1, "candidate_macro_f1": 0.2, "location_bootstrap": {"delta": 0.1}},
         "tpa_to_stp_loo": {"reference_macro_f1": 0.1, "candidate_macro_f1": 0.2, "location_bootstrap": {"delta": 0.1}},
         "stp_mean_to_stp_loo": {"reference_macro_f1": 0.1, "candidate_macro_f1": 0.2, "location_bootstrap": {"delta": 0.1}}},
        {"empty_class_index": 0},
    )

    names = {path.name for path in tmp_path.iterdir()}
    assert "stp_mechanism_audit.json" in names
    assert not any("confirm" in name and "viability" not in name for name in names)
