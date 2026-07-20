from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = frozenset({".git", ".venv", "node_modules"})
WANDB_SECRET_PATTERN = re.compile(rb"wandb_v1_[A-Za-z0-9_]{20,}")


def repository_python_files():
    for path in ROOT.rglob("*.py"):
        if not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            yield path


def test_python_sources_do_not_embed_wandb_api_keys():
    offenders = [
        str(path.relative_to(ROOT))
        for path in repository_python_files()
        if WANDB_SECRET_PATTERN.search(path.read_bytes())
    ]

    assert offenders == []


def test_drm_wise_stp_launcher_uses_injected_credentials_only():
    launcher = (ROOT / "kaggle-drm-wise-stp/kaggle_main.py").read_text(encoding="utf-8")

    assert 'HARDCODED_WANDB_API_KEY = ""' in launcher
    assert 'env.get("WANDB_API_KEY")' in launcher
