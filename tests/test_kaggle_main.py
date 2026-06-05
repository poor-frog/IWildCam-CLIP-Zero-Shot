import tempfile
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

    def test_build_coop_training_argv_preserves_user_overrides(self):
        from kaggle_main import build_coop_training_argv

        argv = build_coop_training_argv("./data", ["--epochs=1", "--wandb-run-name=debug"])

        self.assertIn("--epochs=1", argv)
        self.assertIn("--wandb-run-name=debug", argv)
        self.assertNotIn("--epochs=15", argv)
        self.assertNotIn("--wandb-run-name=coop-vit-b32-phase11-best-f1", argv)


if __name__ == "__main__":
    unittest.main()
