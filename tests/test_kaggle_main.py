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
        self.assertIn("--epochs=50", argv)
        self.assertIn("--best-metric=F1-macro_all", argv)
        self.assertIn("--wandb-project=PoorFrogs", argv)
        self.assertIn("--wandb-run-name=coop-vit-b32-phase11-best-f1", argv)
        self.assertIn("--save=./checkpoints/coop_prompt_learner.pt", argv)

    def test_build_coop_training_argv_preserves_user_overrides(self):
        from kaggle_main import build_coop_training_argv

        argv = build_coop_training_argv("./data", ["--epochs=1", "--wandb-run-name=debug"])

        self.assertIn("--epochs=1", argv)
        self.assertIn("--wandb-run-name=debug", argv)
        self.assertNotIn("--epochs=50", argv)
        self.assertNotIn("--wandb-run-name=coop-vit-b32-phase11-best-f1", argv)


if __name__ == "__main__":
    unittest.main()
