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


if __name__ == "__main__":
    unittest.main()
