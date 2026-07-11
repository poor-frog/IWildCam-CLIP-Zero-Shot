import sys
from pathlib import Path


def test_selection_command_uses_only_validation_split(tmp_path: Path):
    from src.eval_drm_wise_stp import build_selection_command

    command = build_selection_command(
        alpha=0.3,
        selection_output=tmp_path / "selection.json",
        evaluator_args=["--model=ViT-B/16", "--val-dataset=IWildCamVal"],
    )

    assert "--eval-datasets=IWildCamVal" in command
    assert all("IWildCamOOD" not in argument for argument in command)
    assert "--wise-eval-alpha=0.3" in command
    assert "--sctr-strength-grid=0" in command
    assert "--sctr-tail-protection-grid=0" in command


def test_final_command_locks_validation_selected_wise_and_stp_values():
    from src.eval_drm_wise_stp import build_final_command

    candidate = {
        "prototype_scale": 50.0,
        "prototype_k": 1,
        "sequence_eta": 0.5,
        "tau": 0.0,
        "tail_gamma": 0.0,
        "gate_mode": "none",
        "gate_strength": 0.0,
        "sctr_tail_protection": 0.0,
    }
    command = build_final_command(alpha=0.3, candidate=candidate, evaluator_args=["--model=ViT-B/16"])

    assert "--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD" in command
    assert "--wise-eval-alpha=0.3" in command
    assert "--prototype-scale-grid=50" in command
    assert "--sequence-consensus-grid=0.5" in command
    assert "--multi-prototype-k-grid=1" in command
    assert "--summary-head=prototype" in command


def test_selection_output_writes_the_prototype_candidate(tmp_path: Path):
    from src.eval_tail_cache import write_selection_output

    output = tmp_path / "selection.json"
    candidate = {"head": "prototype", "score": 0.42, "sequence_eta": 0.5}

    write_selection_output(output, "IWildCamVal", "F1-macro_all", {"prototype": candidate})

    assert '"prototype"' in output.read_text(encoding="utf-8")
    assert '"score": 0.42' in output.read_text(encoding="utf-8")


def test_requested_summary_head_overrides_global_best_head():
    from src.eval_tail_cache import select_summary_head

    best_by_head = {
        "default": {"score": 0.5},
        "prototype": {"score": 0.4},
    }

    assert select_summary_head(best_by_head, "prototype") == "prototype"


def test_tail_cache_parser_provides_cache_dir_for_wise_anchor(monkeypatch):
    from src.eval_tail_cache import parse_arguments

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_tail_cache.py", "--eval-datasets=IWildCamVal", "--load=checkpoint.pt", "--device=cpu"],
    )

    assert parse_arguments().cache_dir is None
