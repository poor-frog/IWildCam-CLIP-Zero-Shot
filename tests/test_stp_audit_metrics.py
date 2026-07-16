import torch


def test_location_bootstrap_preserves_cluster_multiplicity():
    from src.models.stp_audit_metrics import paired_location_bootstrap

    labels = torch.tensor([0, 1, 0, 1])
    reference = torch.tensor([0, 1, 1, 0])
    candidate = torch.tensor([0, 0, 1, 1])
    locations = ("a", "a", "b", "b")

    result = paired_location_bootstrap(labels, reference, candidate, locations, bootstrap_samples=40, seed=3)

    assert result.requested_samples == 40
    assert result.valid_samples == 40
    assert result.location_count == 2
    assert result.minimum_class_coverage > 0.0


def test_public_viability_report_does_not_expose_confirm_class_composition():
    from src.models.stp_audit_split import build_location_audit_split

    metadata = [torch.tensor([index, index, 8]) for index in range(12)]
    labels = torch.tensor([index % 3 for index in range(12)])
    train_counts = torch.tensor([30, 10, 5])
    result = build_location_audit_split(metadata, labels, train_counts, sequence_field_index=1, location_field_index=0)

    payload = result.viability.to_public_dict()

    assert "confirm_supported_class_ids" not in payload
    assert "confirm_class_counts" not in payload
    assert {"confirm_location_count", "confirm_frame_count", "viability_pass"}.issubset(payload)


def test_audit_payload_contains_no_confirm_performance_keys():
    from src.models.stp_audit_metrics import build_audit_payload

    payload = build_audit_payload(
        split_name="val_audit",
        comparison_payloads={"stp_mean": {"delta_macro_f1": 0.1}, "stp_loo": {"delta_macro_f1": 0.05}},
        viability_payload={"confirm_location_count": 12, "viability_pass": True},
        audit_bin_edges={"margin": [0.1, 0.2, 0.3], "confidence": [0.2, 0.3, 0.4]},
    )

    assert all("confirm" not in key or key == "viability" for key in payload)
    assert "comparisons" in payload
