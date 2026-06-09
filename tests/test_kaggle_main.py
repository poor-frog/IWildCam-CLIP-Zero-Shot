import tempfile
import os
import sys
import unittest
from pathlib import Path


class KaggleMainTest(unittest.TestCase):
    def test_strip_mode_args_removes_mode_pair_before_main_parser(self):
        from kaggle_main import strip_mode_args

        argv = [
            "kaggle_main.py",
            "--mode",
            "coop",
            "--model=ViT-B/32",
            "--epochs=10",
        ]

        self.assertEqual(
            strip_mode_args(argv),
            ["kaggle_main.py", "--model=ViT-B/32", "--epochs=10"],
        )

    def test_resolve_kaggle_data_location_prefers_local_working_tree_data(self):
        from kaggle_main import resolve_kaggle_data_location

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            local_data = repo_root / "data" / "iwildcam_v2.0"
            local_data.mkdir(parents=True)
            resolved = resolve_kaggle_data_location(str(repo_root), "/kaggle/input/some-dataset")
            self.assertEqual(resolved, str(repo_root / "data"))

    def test_resolve_kaggle_data_location_falls_back_to_input_path(self):
        from kaggle_main import resolve_kaggle_data_location

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            resolved = resolve_kaggle_data_location(str(repo_root), "/kaggle/input/some-dataset")
            self.assertEqual(resolved, "/kaggle/input/some-dataset")

    def test_prepare_iwildcam_layout_uses_nested_iwildcam_mount(self):
        from kaggle_main import prepare_iwildcam_layout

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            kaggle_root = Path(tmpdir) / "kaggle" / "input" / "datasets" / "thanhquang71" / "iwildcam-v2-0-2020-wilds-dataset"
            source_dataset = kaggle_root / "iwildcam_v2.0"
            source_dataset.mkdir(parents=True)
            (source_dataset / "metadata.csv").write_text("metadata", encoding="utf-8")
            (source_dataset / "train").mkdir()

            data_location = prepare_iwildcam_layout(repo_root, kaggle_root)
            target_dataset = Path(data_location) / "iwildcam_v2.0"

            self.assertEqual(data_location, str(repo_root / "data"))
            self.assertTrue((target_dataset / "metadata.csv").exists())
            self.assertTrue((target_dataset / "train").exists())

    def test_prepare_iwildcam_layout_falls_back_when_default_mount_is_missing(self):
        from kaggle_main import prepare_iwildcam_layout

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            missing_default = Path(tmpdir) / "kaggle" / "input" / "iwildcam-v2-0-2020-wilds-dataset"
            nested_root = Path(tmpdir) / "kaggle" / "input" / "datasets" / "thanhquang71" / "iwildcam-v2-0-2020-wilds-dataset"
            source_dataset = nested_root / "iwildcam_v2.0"
            source_dataset.mkdir(parents=True)
            (source_dataset / "metadata.csv").write_text("metadata", encoding="utf-8")
            (source_dataset / "train").mkdir()

            data_location = prepare_iwildcam_layout(
                repo_root,
                missing_default,
                kaggle_dataset_candidates=[nested_root],
            )
            target_dataset = Path(data_location) / "iwildcam_v2.0"

            self.assertEqual(data_location, str(repo_root / "data"))
            self.assertTrue((target_dataset / "metadata.csv").exists())
            self.assertTrue((target_dataset / "train").exists())

    def test_resolve_iwildcam_source_root_uses_default_kaggle_candidates(self):
        import kaggle_main

        with tempfile.TemporaryDirectory() as tmpdir:
            default_root = Path(tmpdir) / "missing-default"
            nested_root = Path(tmpdir) / "datasets" / "thanhquang71" / "iwildcam-v2-0-2020-wilds-dataset"
            source_dataset = nested_root / "iwildcam_v2.0"
            source_dataset.mkdir(parents=True)
            (source_dataset / "metadata.csv").write_text("metadata", encoding="utf-8")

            original_candidates = kaggle_main.DEFAULT_KAGGLE_DATASET_CANDIDATES
            try:
                kaggle_main.DEFAULT_KAGGLE_DATASET_CANDIDATES = [str(default_root), str(nested_root)]

                self.assertEqual(kaggle_main.resolve_iwildcam_source_root(), source_dataset)
            finally:
                kaggle_main.DEFAULT_KAGGLE_DATASET_CANDIDATES = original_candidates

    def test_build_coop_training_argv_uses_phase11_defaults(self):
        from kaggle_main import build_coop_training_argv

        argv = build_coop_training_argv("./data")

        self.assertEqual(argv[0], "kaggle_main.py")
        self.assertIn("--model=ViT-B/32", argv)
        self.assertIn("--train-dataset=IWildCam", argv)
        self.assertIn("--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--epochs=15", argv)
        self.assertIn("--best-metric=F1-macro_all", argv)
        self.assertIn("--wandb-project=PoorFrogs", argv)
        self.assertIn("--wandb-run-name=coop-vit-b32-phase11-best-f1", argv)
        self.assertIn("--wandb", argv)
        self.assertIn("--save=./checkpoints/coop_prompt_learner.pt", argv)

    def test_find_repo_root_accepts_directory_with_project_markers(self):
        from kaggle_main import find_repo_root

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / "src").mkdir(parents=True)
            (repo_root / "src" / "train_coop.py").write_text("", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            self.assertEqual(find_repo_root([repo_root]), repo_root)

    def test_ensure_repo_root_clones_when_candidates_are_flat_kaggle_src(self):
        from kaggle_main import ensure_repo_root

        calls = []

        with tempfile.TemporaryDirectory() as tmpdir:
            flat_src = Path(tmpdir) / "src"
            clone_target = Path(tmpdir) / "working" / "IWildCam-CLIP-Zero-Shot"
            flat_src.mkdir(parents=True)

            def fake_check_call(command):
                calls.append(command)
                (clone_target / "src").mkdir(parents=True)
                (clone_target / "src" / "train_coop.py").write_text("", encoding="utf-8")
                (clone_target / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            resolved = ensure_repo_root([flat_src], clone_target, check_call=fake_check_call)

            self.assertEqual(resolved, clone_target)
            self.assertEqual(calls[0][0:2], ["git", "clone"])

    def test_local_package_install_uses_no_deps_to_preserve_kaggle_torch_runtime(self):
        from kaggle_main import _ensure_local_package_installed

        calls = []

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / "src").mkdir(parents=True)
            (repo_root / "src" / "train_coop.py").write_text("", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            _ensure_local_package_installed(repo_root, check_call=calls.append)

        self.assertEqual(calls[0][-3:], ["-e", str(repo_root), "--no-deps"])

    def test_configure_import_path_adds_repo_root_to_python_imports(self):
        from kaggle_main import configure_import_path

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = str(Path(tmpdir) / "repo")
            old_pythonpath = os.environ.get("PYTHONPATH")
            original_sys_path = list(sys.path)
            try:
                os.environ.pop("PYTHONPATH", None)
                sys.path = [path for path in sys.path if path != repo_root]

                configure_import_path(repo_root)

                self.assertEqual(os.environ["PYTHONPATH"], repo_root)
                self.assertEqual(sys.path[0], repo_root)
            finally:
                if old_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old_pythonpath
                sys.path = original_sys_path

    def test_build_coop_training_argv_preserves_user_overrides(self):
        from kaggle_main import build_coop_training_argv

        argv = build_coop_training_argv("./data", ["--epochs=1", "--wandb-run-name=debug"])

        self.assertIn("--epochs=1", argv)
        self.assertIn("--wandb-run-name=debug", argv)
        self.assertNotIn("--epochs=15", argv)
        self.assertNotIn("--wandb-run-name=coop-vit-b32-phase11-best-f1", argv)

    def test_build_maple_lora_training_argv_uses_separate_defaults(self):
        from kaggle_main import build_maple_lora_training_argv

        argv = build_maple_lora_training_argv("./data")

        self.assertIn("--model=ViT-B/32", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--maple-lora-rank=8", argv)
        self.assertIn("--maple-lora-alpha=16", argv)
        self.assertIn("--maple-lora-layers=last6", argv)
        self.assertIn("--wandb-run-name=maple-lora-vit-b32-r8-last6", argv)
        self.assertIn("--save=./checkpoints/maple_lora_r8_last6.pt", argv)

    def test_train_maple_full_imports_without_removed_shallow_module(self):
        import src.train_maple_full as train_maple_full

        self.assertTrue(callable(train_maple_full.print_summary))


if __name__ == "__main__":
    unittest.main()
