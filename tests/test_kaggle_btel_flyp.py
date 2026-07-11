import ast
import re
from pathlib import Path

import pytest


def test_btel_launcher_rejects_a_stale_clone(tmp_path: Path):
    import kaggle_btel_flyp as launcher

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("--btel-weight\n", encoding="utf-8")
    (tmp_path / "src" / "audit_btel_sequences.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "train_flyp.py").write_text("btel_artifacts", encoding="utf-8")
    (tmp_path / "src" / "models").mkdir()
    (tmp_path / "src" / "models" / "btel.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "models" / "btel_artifacts.py").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        launcher.assert_btel_support(tmp_path)


def test_btel_launcher_rejects_clone_missing_the_btel_model_modules(tmp_path: Path):
    import kaggle_btel_flyp as launcher

    (tmp_path / "src").mkdir()
    flags = "\n".join(launcher.REQUIRED_BTEL_FLAGS)
    (tmp_path / "src" / "config.py").write_text(flags, encoding="utf-8")
    (tmp_path / "src" / "audit_btel_sequences.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "train_flyp.py").write_text("btel_artifacts", encoding="utf-8")

    with pytest.raises(RuntimeError, match="lacks BTEL runtime files"):
        launcher.assert_btel_support(tmp_path)


def test_btel_launcher_accepts_the_current_checkout():
    import kaggle_btel_flyp as launcher

    launcher.assert_btel_support(Path(__file__).parents[1])


def test_btel_launcher_rejects_placeholder_btel_sources(tmp_path: Path):
    import kaggle_btel_flyp as launcher

    (tmp_path / "src" / "models").mkdir(parents=True)
    flags = "\n".join(f'parser.add_argument("{flag}")' for flag in launcher.REQUIRED_BTEL_FLAGS)
    (tmp_path / "src" / "config.py").write_text(flags, encoding="utf-8")
    (tmp_path / "src" / "audit_btel_sequences.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "train_flyp.py").write_text(
        "def maybe_prepare_btel():\n    pass\n\ndef main():\n    train(btel_artifacts=None)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "models" / "btel.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "models" / "btel_artifacts.py").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete BTEL runtime code"):
        launcher.assert_btel_support(tmp_path)


def test_btel_launcher_rejects_symbol_only_btel_runtime(tmp_path: Path):
    import kaggle_btel_flyp as launcher

    (tmp_path / "src" / "models").mkdir(parents=True)
    flags = "\n".join(f'parser.add_argument("{flag}")' for flag in launcher.REQUIRED_BTEL_FLAGS)
    (tmp_path / "src" / "config.py").write_text(flags, encoding="utf-8")
    (tmp_path / "src" / "audit_btel_sequences.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "train_flyp.py").write_text(
        "def maybe_prepare_btel():\n    pass\n\ndef main():\n    train(btel_artifacts=None)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "models" / "btel.py").write_text(
        "class BTELArtifacts:\n    pass\n\nclass BurstBatchSampler:\n    pass\n\ndef btel_sequence_loss():\n    pass\n\ndef class_topk(_):\n    return []\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "models" / "btel_artifacts.py").write_text(
        "def audit_sequences():\n    pass\n\ndef build_btel_artifacts():\n    pass\n\ndef validate_btel_validation_split():\n    pass\n\ndef find_empty_class_index(_):\n    return 1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="incomplete BTEL runtime behavior"):
        launcher.assert_btel_support(tmp_path)


def _embedded_secret_literals(source: str) -> list[str]:
    tree = ast.parse(source)
    sensitive_name = re.compile(r"(?:key|token|secret|credential|password|api)", re.IGNORECASE)
    suspicious = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if any(sensitive_name.search(name) for name in names) and len(node.value.value) >= 20:
            suspicious.append(node.value.value)
    suspicious.extend(re.findall(r"wandb_v1_[A-Za-z0-9_-]+", source))
    return suspicious


def test_btel_launcher_does_not_embed_a_wandb_key():
    launcher_path = Path(__file__).parents[1] / "kaggle_btel_flyp.py"

    source = launcher_path.read_text(encoding="utf-8")
    assert _embedded_secret_literals(source) == []
    assert "WANDB_API_KEY" in source


def test_secret_scan_rejects_legacy_credential_literal():
    source = 'credential = "0123456789abcdef0123456789abcdef01234567"\n'
    assert _embedded_secret_literals(source) == ["0123456789abcdef0123456789abcdef01234567"]


def test_secret_scan_rejects_annotated_credential_literal():
    source = 'credential: str = "0123456789abcdef0123456789abcdef01234567"\n'
    assert _embedded_secret_literals(source) == ["0123456789abcdef0123456789abcdef01234567"]
