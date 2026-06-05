import os
import subprocess
import sys
from pathlib import Path


DEFAULT_KAGGLE_DATASET = "/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"


def strip_mode_args(argv):
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--mode":
            skip_next = True
            continue
        if arg.startswith("--mode="):
            continue
        stripped.append(arg)
    return stripped


def parse_mode(argv):
    for index, arg in enumerate(argv):
        if arg == "--mode" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1]
    return "coop"


def resolve_kaggle_data_location(repo_root, fallback_input_path=DEFAULT_KAGGLE_DATASET):
    local_data = Path(repo_root) / "data" / "iwildcam_v2.0"
    if local_data.exists():
        return str(local_data.parent)
    return fallback_input_path


def _ensure_deps():
    packages = [
        "braceexpand",
        "ftfy",
        "open-clip-torch",
        "pandas",
        "regex",
        "tqdm",
        "wandb",
        "webdataset",
        "wilds",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def _ensure_local_clip_installed(repo_root):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)])


def _inject_default_args(argv, repo_root):
    args = strip_mode_args(argv)
    if not any(arg.startswith("--data-location") for arg in args):
        args.extend(["--data-location", resolve_kaggle_data_location(repo_root)])
    return args


def main():
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    os.environ.setdefault("PYTHONPATH", str(repo_root))
    _ensure_deps()
    _ensure_local_clip_installed(repo_root)

    mode = parse_mode(sys.argv)
    sys.argv = _inject_default_args(sys.argv, repo_root)
    if mode == "zeroshot":
        from src.main import main as run_zeroshot
    elif mode == "coop":
        from src.train_coop import main as run_coop
    else:
        raise ValueError("Unsupported --mode. Use 'coop' or 'zeroshot'.")

    from src.config import parse_arguments

    run_coop(parse_arguments()) if mode == "coop" else run_zeroshot(parse_arguments())


if __name__ == "__main__":
    main()
