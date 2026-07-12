import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


DRM_REPOSITORY = "https://github.com/vaynexie/DRM.git"
DRM_ROOT = Path("/kaggle/working/DRM")
CHECKPOINT_NAME = "iwildcam_vit_b16.pt"
MODEL = "ViT-B/16"
BATCH_SIZE = 512
BETA = 0.5
WISE_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HARDCODED_WANDB_API_KEY = ""
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def run(command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd)


def ensure_dependencies():
    run([sys.executable, "-m", "pip", "install", "-q", "ftfy", "numpy", "open-clip-torch", "pandas", "regex", "scikit-learn", "tqdm", "wandb", "wilds"])


def configure_wandb():
    if os.environ.get("WANDB_API_KEY"):
        return True
    if HARDCODED_WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY
        return True
    try:
        from kaggle_secrets import UserSecretsClient
    except ModuleNotFoundError:
        return False
    secrets = UserSecretsClient()
    for name in WANDB_SECRET_NAMES:
        try:
            value = secrets.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ["WANDB_API_KEY"] = value
            return True
    return False


def clone_or_update_repo():
    if DRM_ROOT.exists():
        run(["git", "-C", DRM_ROOT, "pull", "--ff-only"])
    else:
        run(["git", "clone", DRM_REPOSITORY, DRM_ROOT])
    return DRM_ROOT


def find_iwildcam_data_location():
    roots = (Path("/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"), Path("/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset"))
    for root in roots:
        for candidate in (root, root / "iwildcam_v2.0", root / "archive" / "iwildcam_v2.0"):
            if (candidate / "metadata.csv").exists():
                return candidate
    for metadata_path in Path("/kaggle/input").rglob("metadata.csv"):
        if metadata_path.parent.name == "iwildcam_v2.0":
            return metadata_path.parent
    raise FileNotFoundError("Could not find iwildcam_v2.0/metadata.csv under /kaggle/input.")


def find_checkpoint():
    matches = sorted(Path("/kaggle/input").rglob(CHECKPOINT_NAME))
    if not matches:
        raise FileNotFoundError(f"Could not find {CHECKPOINT_NAME} under /kaggle/input.")
    return matches[0]


def stage_checkpoint(checkpoint_path):
    target = DRM_ROOT / "ckpts" / CHECKPOINT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(checkpoint_path)
    return target


def parse_drm_args(data_location, checkpoint_path, alpha, eval_datasets, exp_name):
    from src.args import parse_arguments

    original_argv = sys.argv
    sys.argv = [
        "official_wise_eval.py",
        "--train-dataset=IWildCamIDVal",
        f"--batch-size={BATCH_SIZE}",
        f"--model={MODEL}",
        f"--eval-datasets={eval_datasets}",
        "--template=iwildcam_template",
        f"--data-location={data_location}",
        f"--exp_name={exp_name}",
        "--cd_path=prompts/iwildcam_cd.json",
        f"--beta={BETA}",
        f"--checkpoint_path={checkpoint_path}",
    ]
    try:
        return parse_arguments()
    finally:
        sys.argv = original_argv


def interpolate_wise(zeroshot_state, finetuned, alpha):
    import torch

    finetuned_state = finetuned.state_dict()
    if set(zeroshot_state) != set(finetuned_state):
        raise RuntimeError("Zero-shot and DRM checkpoint state dictionaries have different keys.")
    merged = {}
    for key, fine_value in finetuned_state.items():
        zero_value = zeroshot_state[key]
        if torch.is_floating_point(fine_value):
            merged[key] = alpha * zero_value.to(dtype=fine_value.dtype) + (1.0 - alpha) * fine_value
        else:
            merged[key] = fine_value
    finetuned.load_state_dict(merged, strict=True)
    return finetuned


def stats_path(exp_name):
    return DRM_ROOT / "expt_logs" / exp_name / f"_BS{BATCH_SIZE}_WD0.1_LR0.001_run1" / "stats.tsv"


def read_metrics(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Official DRM evaluation produced no rows in {path}.")
    return {key: float(value) for key, value in rows[-1].items() if key and value}


def find_macro_f1(metrics, dataset_name):
    matches = [value for key, value in metrics.items() if key.startswith(dataset_name) and "f1" in key.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one macro-F1 metric for {dataset_name}, found {matches}.")
    return matches[0]


def evaluate_alpha(data_location, checkpoint_path, zeroshot_state, alpha, eval_datasets, label, logger):
    import torch
    from src.models.DRM_eval import drm_eval
    from src.models.modeling import CLIPEncoder

    args = parse_drm_args(data_location, checkpoint_path, alpha, eval_datasets, label)
    finetuned = CLIPEncoder.load(checkpoint_path).cpu()
    merged = interpolate_wise(zeroshot_state, finetuned, alpha)
    drm_eval(args, merged, logger)
    metrics = read_metrics(stats_path(label))
    del merged
    torch.cuda.empty_cache()
    return metrics


def make_logger():
    logger = logging.getLogger("official_drm_wise")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    return logger


def log_metrics(wandb_run, prefix, metrics):
    if wandb_run is None:
        return
    wandb_run.log({f"{prefix}/{key.replace(' ', '_')}": value for key, value in metrics.items()})


def main():
    ensure_dependencies()
    wandb_enabled = configure_wandb()
    clone_or_update_repo()
    sys.path.insert(0, str(DRM_ROOT))
    import torch
    from src.models.modeling import CLIPEncoder

    if not torch.cuda.is_available():
        raise RuntimeError("Official DRM WiSE baseline requires a Kaggle GPU runtime.")
    data_location = find_iwildcam_data_location()
    checkpoint_path = stage_checkpoint(find_checkpoint())
    zeroshot_args = parse_drm_args(data_location, checkpoint_path, 0.0, "IWildCamIDVal", "iwildcam/zero_shot_anchor")
    zeroshot_state = {key: value.detach().cpu().clone() for key, value in CLIPEncoder(zeroshot_args, keep_lang=True).cpu().state_dict().items()}
    wandb_run = None
    if wandb_enabled:
        import wandb
        wandb_run = wandb.init(project="PoorFrogs", name="drm-official-wise-vitb16-idval", config={"method": "official_drm_wise", "model": MODEL, "beta": BETA, "wise_alphas": WISE_ALPHAS, "selection_split": "IWildCamIDVal", "checkpoint": str(checkpoint_path)})
    logger = make_logger()
    selection = []
    for alpha in WISE_ALPHAS:
        metrics = evaluate_alpha(data_location, checkpoint_path, zeroshot_state, alpha, "IWildCamIDVal", f"iwildcam/wise_selection_alpha{str(alpha).replace('.', 'p')}", logger)
        score = find_macro_f1(metrics, "IWildCamIDVal")
        selection.append({"alpha": alpha, "idval_f1": score})
        log_metrics(wandb_run, f"selection/alpha_{str(alpha).replace('.', 'p')}", metrics)
    selected = max(selection, key=lambda record: record["idval_f1"])
    output_dir = Path("/kaggle/working/official_drm_wise")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(json.dumps({"selection": selection, "selected": selected}, indent=2), encoding="utf-8")
    final_metrics = evaluate_alpha(data_location, checkpoint_path, zeroshot_state, selected["alpha"], "IWildCamIDVal,IWildCamID,IWildCamOOD", "iwildcam/wise_final", logger)
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Selected official WiSE alpha={selected['alpha']:.1f} on IWildCamIDVal F1={selected['idval_f1']:.4f}", flush=True)
    print(json.dumps(final_metrics, indent=2, sort_keys=True), flush=True)
    log_metrics(wandb_run, "final", final_metrics)
    if wandb_run is not None:
        wandb_run.log({"selection/selected_alpha": selected["alpha"], "selection/selected_idval_f1": selected["idval_f1"]})
        wandb_run.save(str(output_dir / "selection.json"), base_path="/kaggle/working")
        wandb_run.save(str(output_dir / "final_metrics.json"), base_path="/kaggle/working")
        wandb_run.finish()


if __name__ == "__main__":
    main()
