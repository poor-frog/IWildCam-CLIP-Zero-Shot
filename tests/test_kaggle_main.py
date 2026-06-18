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



    def test_patch_iwildcam_val_patches_dataset_and_iwildcam_modules_without_all(self):
        import types
        from unittest import mock
        import kaggle_main

        fake_src = types.ModuleType("src")
        fake_src.__path__ = []
        fake_package = types.ModuleType("src.datasets")
        fake_package.__path__ = []
        fake_iwildcam = types.ModuleType("src.datasets.iwildcam")

        class FakeIWildCam:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

        fake_iwildcam.IWildCam = FakeIWildCam

        with mock.patch.dict(sys.modules, {
            "src": fake_src,
            "src.datasets": fake_package,
            "src.datasets.iwildcam": fake_iwildcam,
        }):
            kaggle_main._patch_iwildcam_val()

            self.assertIs(fake_package.IWildCamVal, fake_iwildcam.IWildCamVal)
            patched = fake_package.IWildCamVal(None, location="data")
            self.assertEqual(patched.kwargs["subset"], "val")

    def test_main_patches_iwildcam_val_after_dependencies_are_installed(self):
        import types
        from unittest import mock
        import kaggle_main

        calls = []
        config_module = types.ModuleType("src.config")
        config_module.parse_arguments = lambda: object()
        train_module = types.ModuleType("src.train_coop")
        train_module.main = lambda args: calls.append("run_coop")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            original_argv = sys.argv
            try:
                sys.argv = ["kaggle_main.py", "--mode=coop"]
                with mock.patch.object(kaggle_main, "ensure_repo_root", return_value=repo_root), \
                     mock.patch.object(kaggle_main.os, "chdir"), \
                     mock.patch.object(kaggle_main, "configure_import_path", side_effect=lambda root: calls.append("path")), \
                     mock.patch.object(kaggle_main, "_ensure_deps", side_effect=lambda: calls.append("deps")), \
                     mock.patch.object(kaggle_main, "_ensure_local_package_installed", side_effect=lambda root: calls.append("install")), \
                     mock.patch.object(kaggle_main, "_patch_iwildcam_val", side_effect=lambda: calls.append("patch")), \
                     mock.patch.object(kaggle_main, "_configure_wandb_from_kaggle_secret", side_effect=lambda: calls.append("wandb")), \
                     mock.patch.object(kaggle_main, "prepare_iwildcam_layout", return_value="./data"), \
                     mock.patch.dict(sys.modules, {"src.config": config_module, "src.train_coop": train_module}):
                    kaggle_main.main()
            finally:
                sys.argv = original_argv

        self.assertLess(calls.index("deps"), calls.index("patch"))
        self.assertLess(calls.index("install"), calls.index("patch"))
        self.assertLess(calls.index("patch"), calls.index("run_coop"))

    def test_build_coop_training_argv_preserves_user_overrides(self):
        from kaggle_main import build_coop_training_argv

        argv = build_coop_training_argv("./data", ["--epochs=1", "--wandb-run-name=debug"])

        self.assertIn("--epochs=1", argv)
        self.assertIn("--wandb-run-name=debug", argv)
        self.assertNotIn("--epochs=15", argv)
        self.assertNotIn("--wandb-run-name=coop-vit-b32-phase11-best-f1", argv)


    def test_build_maple_cbce_training_argv_uses_a1_protocol_defaults(self):
        from kaggle_main import build_maple_cbce_training_argv

        argv = build_maple_cbce_training_argv("./data")

        self.assertIn("--model=ViT-B/32", argv)
        self.assertIn("--train-dataset=IWildCam", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--eval-datasets=IWildCamVal", argv)
        self.assertIn("--val-dataset=IWildCamVal", argv)
        self.assertNotIn("--class-balanced-ce", argv)
        self.assertIn("--wandb", argv)
        self.assertIn("--wandb-run-name=a1-maple-cbce-iwildcamval", argv)
        self.assertIn("--save=/kaggle/working/checkpoints/a1_maple_cbce_iwildcamval.pt", argv)
        self.assertNotIn("--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD", argv)

    def test_build_maple_cbce_training_argv_preserves_protocol_overrides(self):
        from kaggle_main import build_maple_cbce_training_argv

        argv = build_maple_cbce_training_argv("./data", ["--epochs=30", "--no-wandb", "--save=/tmp/a1.pt"])

        self.assertIn("--epochs=30", argv)
        self.assertIn("--no-wandb", argv)
        self.assertIn("--save=/tmp/a1.pt", argv)
        self.assertNotIn("--wandb", argv)
        self.assertNotIn("--save=/kaggle/working/checkpoints/a1_maple_cbce_iwildcamval.pt", argv)

    def test_build_maple_tau_sweep_eval_argv_uses_vanilla_maple_tau_defaults(self):
        from kaggle_main import build_maple_tau_sweep_eval_argv

        argv = build_maple_tau_sweep_eval_argv("./data")

        self.assertIn("--model=ViT-B/32", argv)
        self.assertIn("--train-dataset=IWildCam", argv)
        self.assertIn("--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--epochs=0", argv)
        self.assertIn("--selection-split=IWildCamVal", argv)
        self.assertIn("--logit-adjustment-tau-grid=0,0.25,0.5,0.75,1,1.5,2", argv)
        self.assertIn("--wandb-run-name=maple-vanilla-tau-sweep-iwildcamval", argv)
        self.assertNotIn("--class-balanced-ce", argv)

    def test_build_maple_tau_sweep_eval_argv_preserves_checkpoint_and_tau_overrides(self):
        from kaggle_main import build_maple_tau_sweep_eval_argv

        argv = build_maple_tau_sweep_eval_argv("./data", [
            "--load=/kaggle/input/model/maple.pt",
            "--logit-adjustment-tau-grid=0,1",
            "--no-wandb",
        ])

        self.assertIn("--load=/kaggle/input/model/maple.pt", argv)
        self.assertIn("--logit-adjustment-tau-grid=0,1", argv)
        self.assertIn("--no-wandb", argv)
        self.assertNotIn("--load=/kaggle/input/maple-vanilla-checkpoint/maple_full_prompt_learner_best.pt", argv)
        self.assertNotIn("--logit-adjustment-tau-grid=0,0.25,0.5,0.75,1,1.5,2", argv)
        self.assertNotIn("--wandb", argv)

    def test_build_maple_lora_training_argv_uses_separate_defaults(self):
        from kaggle_main import build_maple_lora_training_argv

        argv = build_maple_lora_training_argv("./data")

        self.assertIn("--model=ViT-B/32", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--maple-lora-rank=4", argv)
        self.assertIn("--maple-lora-alpha=8", argv)
        self.assertIn("--maple-lora-layers=last6", argv)
        self.assertIn("--wandb-run-name=maple-lora-vit-b32-r4-last6-e3-lr1e-3", argv)
        self.assertIn("--save=./checkpoints/maple_lora_r4_last6_e3_lr1e-3.pt", argv)


    def test_build_c1_training_argv_uses_c1_protocol_defaults(self):
        from kaggle_main import build_c1_training_argv

        argv = build_c1_training_argv("./data")

        self.assertIn("--model=ViT-B/32", argv)
        self.assertIn("--train-dataset=IWildCam", argv)
        self.assertIn("--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--batch-size=32", argv)
        self.assertIn("--maple-lora-rank=4", argv)
        self.assertIn("--maple-lora-alpha=8", argv)
        self.assertIn("--maple-lora-layers=last6", argv)
        self.assertNotIn("--class-balanced-ce", argv)
        self.assertIn("--epochs=9", argv)
        self.assertNotIn("--kl-weight=0.1", argv)
        self.assertNotIn("--kl-temperature=1.0", argv)
        self.assertIn("--val-dataset=IWildCamVal", argv)
        self.assertIn("--best-metric=F1-macro_all", argv)
        self.assertIn("--wandb-run-name=c1-maple-lora-cbce-kl-vit-b32-bs32", argv)
        self.assertIn("--save=/kaggle/working/checkpoints/c1_maple_lora_cbce_kl_vitb32_bs32.pt", argv)
        self.assertIn("--wandb", argv)

    def test_build_c1_training_argv_preserves_overrides(self):
        from kaggle_main import build_c1_training_argv

        argv = build_c1_training_argv("./data", [
            "--batch-size=64",
            "--epochs=1",
            "--no-wandb",
            "--save=/tmp/c1.pt",
        ])

        self.assertIn("--batch-size=64", argv)
        self.assertIn("--epochs=1", argv)
        self.assertIn("--no-wandb", argv)
        self.assertIn("--save=/tmp/c1.pt", argv)
        self.assertNotIn("--batch-size=32", argv)
        self.assertNotIn("--epochs=9", argv)
        self.assertNotIn("--kl-weight=0.1", argv)
        self.assertNotIn("--wandb", argv)
        self.assertNotIn("--save=/kaggle/working/checkpoints/c1_maple_lora_cbce_kl_vitb32_bs32.pt", argv)

    def test_train_maple_full_imports_without_removed_shallow_module(self):
        import src.train_maple_full as train_maple_full

        self.assertTrue(callable(train_maple_full.print_summary))


if __name__ == "__main__":
    unittest.main()
