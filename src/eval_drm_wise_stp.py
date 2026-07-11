import argparse
import json
import subprocess
import sys
from pathlib import Path


FINAL_EVAL_DATASETS = "IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD"


class WiseStpSelectionError(ValueError):
    pass


def parse_wise_alpha_grid(raw_value):
    values = [float(value.strip()) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise WiseStpSelectionError("--wise-alpha-grid must include at least one value.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise WiseStpSelectionError("WiSE alpha values must be in [0, 1].")
    return values


def format_float(value):
    return f"{float(value):g}"


def build_selection_command(alpha, selection_output, evaluator_args):
    return [
        sys.executable,
        "src/eval_tail_cache.py",
        *evaluator_args,
        "--eval-datasets=IWildCamVal",
        f"--wise-eval-alpha={format_float(alpha)}",
        "--sctr-strength-grid=0",
        "--sctr-tail-protection-grid=0",
        f"--selection-output={selection_output}",
    ]


def build_final_command(alpha, candidate, evaluator_args):
    return [
        sys.executable,
        "src/eval_tail_cache.py",
        *evaluator_args,
        f"--eval-datasets={FINAL_EVAL_DATASETS}",
        f"--wise-eval-alpha={format_float(alpha)}",
        f"--prototype-scale-grid={format_float(candidate['prototype_scale'])}",
        f"--cache-tau-grid={format_float(candidate['tau'])}",
        f"--tail-gamma-grid={format_float(candidate['tail_gamma'])}",
        f"--gate-mode-grid={candidate['gate_mode']}",
        f"--gate-strength-grid={format_float(candidate['gate_strength'])}",
        f"--sequence-consensus-grid={format_float(candidate['sequence_eta'])}",
        "--sctr-strength-grid=0",
        f"--sctr-tail-protection-grid={format_float(candidate['sctr_tail_protection'])}",
        f"--multi-prototype-k-grid={int(candidate['prototype_k'])}",
        "--summary-head=prototype",
    ]


def read_prototype_selection(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        candidate = payload["best_by_head"]["prototype"]
    except KeyError as error:
        raise WiseStpSelectionError(f"Selection output {path} does not contain a prototype candidate.") from error
    if candidate["head"] != "prototype":
        raise WiseStpSelectionError(f"Selection output {path} returned an unexpected head {candidate['head']!r}.")
    return candidate


def choose_best_wise_selection(records):
    if not records:
        raise WiseStpSelectionError("No WiSE selection records were produced.")
    return max(records, key=lambda record: record[1]["score"])


def alpha_label(alpha):
    return format_float(alpha).replace(".", "p")


def evaluator_args(args):
    command = [
        f"--model={args.model}",
        f"--train-dataset={args.train_dataset}",
        f"--val-dataset={args.val_dataset}",
        f"--template={args.template}",
        f"--data-location={args.data_location}",
        f"--load={args.load}",
        f"--prototype-scale-grid={args.prototype_scale_grid}",
        f"--cache-tau-grid={args.cache_tau_grid}",
        f"--tail-gamma-grid={args.tail_gamma_grid}",
        f"--gate-mode-grid={args.gate_mode_grid}",
        f"--gate-strength-grid={args.gate_strength_grid}",
        f"--sequence-consensus-grid={args.sequence_consensus_grid}",
        f"--sequence-id-field={args.sequence_id_field}",
        f"--multi-prototype-k-grid={args.multi_prototype_k_grid}",
        f"--multi-prototype-reduction={args.multi_prototype_reduction}",
        f"--batch-size={args.batch_size}",
        f"--workers={args.workers}",
        f"--device={args.device}",
        f"--best-metric={args.best_metric}",
    ]
    if args.audit_metadata:
        command.append("--audit-metadata")
    if args.wandb:
        command.extend(["--wandb", f"--wandb-project={args.wandb_project}"])
    else:
        command.append("--no-wandb")
    return command


def run_command(command, cwd):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validation-selected DRM + WiSE + STP evaluation.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--val-dataset", type=str, default="IWildCamVal")
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--data-location", type=str, required=True)
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--wise-alpha-grid", type=str, required=True)
    parser.add_argument("--prototype-scale-grid", type=str, default="50")
    parser.add_argument("--cache-tau-grid", type=str, default="0")
    parser.add_argument("--tail-gamma-grid", type=str, default="0")
    parser.add_argument("--gate-mode-grid", type=str, default="none")
    parser.add_argument("--gate-strength-grid", type=str, default="0")
    parser.add_argument("--sequence-consensus-grid", type=str, default="0,0.25,0.5")
    parser.add_argument("--sequence-id-field", type=str, default="auto")
    parser.add_argument("--multi-prototype-k-grid", type=str, default="1")
    parser.add_argument("--multi-prototype-reduction", type=str, default="max")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--best-metric", type=str, default="F1-macro_all")
    parser.add_argument("--selection-dir", type=Path, default=Path("/kaggle/working/drm_wise_stp_selection"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=False)
    parser.add_argument("--wandb-project", type=str, default="PoorFrogs")
    parser.add_argument("--wandb-run-prefix", type=str, default="drm-wise-stp-vitb16-iwildcamval")
    parser.add_argument("--audit-metadata", action="store_true")
    return parser.parse_args()


def main(args):
    if args.val_dataset != "IWildCamVal":
        raise WiseStpSelectionError("DRM + WiSE + STP selection requires --val-dataset=IWildCamVal.")
    alpha_grid = parse_wise_alpha_grid(args.wise_alpha_grid)
    args.selection_dir.mkdir(parents=True, exist_ok=True)
    shared_args = evaluator_args(args)
    records = []
    for alpha in alpha_grid:
        alpha_name = alpha_label(alpha)
        selection_output = args.selection_dir / f"selection_alpha{alpha_name}.json"
        command = build_selection_command(alpha, selection_output, shared_args)
        if args.wandb:
            command.append(f"--wandb-run-name={args.wandb_run_prefix}-selection-alpha{alpha_name}")
        run_command(command, cwd=Path.cwd())
        records.append((alpha, read_prototype_selection(selection_output)))

    selected_alpha, selected_candidate = choose_best_wise_selection(records)
    selected_path = args.selection_dir / "selected_config.json"
    selected_path.write_text(
        json.dumps({"wise_alpha": selected_alpha, "prototype": selected_candidate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Selected WiSE alpha={selected_alpha:g} with validation score={selected_candidate['score']:.4f}")
    final_command = build_final_command(selected_alpha, selected_candidate, shared_args)
    if args.wandb:
        final_command.append(f"--wandb-run-name={args.wandb_run_prefix}-final-alpha{alpha_label(selected_alpha)}")
    run_command(final_command, cwd=Path.cwd())


if __name__ == "__main__":
    main(parse_arguments())
