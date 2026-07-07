import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class KaggleMainTest(unittest.TestCase):
    def test_strip_mode_args_removes_legacy_mode_before_main_parser(self):
        from kaggle_main import strip_mode_args

        argv = [
            "kaggle_main.py",
            "--mode",
            "flyp",
            "--model=ViT-B-16",
            "--epochs=10",
        ]

        self.assertEqual(
            strip_mode_args(argv),
            ["kaggle_main.py", "--model=ViT-B-16", "--epochs=10"],
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

                resolved = kaggle_main.resolve_iwildcam_source_root()
            finally:
                kaggle_main.DEFAULT_KAGGLE_DATASET_CANDIDATES = original_candidates

        self.assertEqual(resolved, source_dataset)

    def test_find_repo_root_accepts_directory_with_project_markers(self):
        from kaggle_main import find_repo_root

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / "src").mkdir(parents=True)
            (repo_root / "src" / "train_coop.py").write_text("", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            resolved = find_repo_root([repo_root])

        self.assertEqual(resolved, repo_root)

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

    def test_assert_cloned_repo_supports_runtime_flags_rejects_stale_config(self):
        from kaggle_main import assert_cloned_repo_supports_runtime_flags

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            config_path = repo_root / "src" / "config.py"
            config_path.parent.mkdir(parents=True)
            config_path.write_text('parser.add_argument("--maple-precision", choices=["fp32"])\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Tail-Aware FLYP flags"):
                assert_cloned_repo_supports_runtime_flags(repo_root)

    def test_assert_cloned_repo_supports_runtime_flags_accepts_current_config(self):
        from kaggle_main import assert_cloned_repo_supports_runtime_flags

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            config_path = repo_root / "src" / "config.py"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                'parser.add_argument("--lr-scheduler")\n'
                'parser.add_argument("--warmup-length")\n'
                'parser.add_argument("--maple-precision", choices=["fp32", "amp"])\n'
                'parser.add_argument("--drm-weight")\n'
                'parser.add_argument("--drm-warmup-epochs")\n'
                'parser.add_argument("--wise-alphas")\n'
                'parser.add_argument("--wise-eval-alpha")\n'
                'parser.add_argument("--tail-proto-weight")\n'
                'parser.add_argument("--tail-proto-scale")\n'
                'parser.add_argument("--tail-proto-objective")\n'
                'parser.add_argument("--tail-proto-temperature")\n'
                'parser.add_argument("--tail-proto-teacher-load")\n'
                'parser.add_argument("--tail-proto-max-batches")\n',
                encoding="utf-8",
            )

            assert_cloned_repo_supports_runtime_flags(repo_root)

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

    def test_main_patches_iwildcam_val_before_tail_aware_flyp_runs(self):
        import kaggle_main

        calls = []
        captured = {}
        args = types.SimpleNamespace(
            drm_weight=0.0,
            wise_alphas="0.0,0.1",
            tail_proto_weight=0.001,
            tail_proto_scale=20.0,
            tail_proto_objective="ce",
            tail_proto_temperature=1.0,
            tail_proto_teacher_load=None,
            tail_proto_max_batches=None,
        )
        config_module = types.ModuleType("src.config")
        config_module.parse_arguments = lambda: args
        train_module = types.ModuleType("src.train_flyp")
        train_module.main = lambda parsed_args: captured.setdefault("args", parsed_args)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            original_argv = sys.argv
            try:
                sys.argv = ["kaggle_main.py"]
                with mock.patch.object(kaggle_main, "ensure_repo_root", return_value=repo_root), \
                     mock.patch.object(kaggle_main.os, "chdir"), \
                     mock.patch.object(kaggle_main, "configure_import_path", side_effect=lambda root: calls.append("path")), \
                     mock.patch.object(kaggle_main, "_ensure_deps", side_effect=lambda: calls.append("deps")), \
                     mock.patch.object(kaggle_main, "_ensure_local_package_installed", side_effect=lambda root: calls.append("install")), \
                     mock.patch.object(kaggle_main, "assert_cloned_repo_supports_runtime_flags", side_effect=lambda root: calls.append("guard")), \
                     mock.patch.object(kaggle_main, "_patch_iwildcam_val", side_effect=lambda: calls.append("patch")), \
                     mock.patch.object(kaggle_main, "_configure_wandb_from_kaggle_secret", side_effect=lambda: calls.append("wandb")), \
                     mock.patch.object(kaggle_main, "prepare_iwildcam_layout", return_value="./data"), \
                     mock.patch.dict(sys.modules, {"src.config": config_module, "src.train_flyp": train_module}):
                    kaggle_main.main()
            finally:
                sys.argv = original_argv

        self.assertLess(calls.index("deps"), calls.index("patch"))
        self.assertLess(calls.index("install"), calls.index("patch"))
        self.assertLess(calls.index("guard"), calls.index("patch"))
        self.assertLess(calls.index("patch"), calls.index("wandb"))
        self.assertIs(captured["args"], args)

    def test_main_applies_tail_aware_flyp_env_overrides_after_parsing(self):
        import kaggle_main

        captured = {}
        args = types.SimpleNamespace(
            drm_weight=1.0,
            wise_alphas="0.0,0.1",
            tail_proto_weight=0.001,
            tail_proto_scale=20.0,
            tail_proto_objective="ce",
            tail_proto_temperature=1.0,
            tail_proto_teacher_load=None,
            tail_proto_max_batches=None,
        )
        config_module = types.ModuleType("src.config")
        config_module.parse_arguments = lambda: args
        train_module = types.ModuleType("src.train_flyp")
        train_module.main = lambda parsed_args: captured.setdefault("args", parsed_args)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            original_argv = sys.argv
            originals = {
                "FLYP_DRM_WEIGHT": os.environ.get("FLYP_DRM_WEIGHT"),
                "FLYP_WISE_ALPHAS": os.environ.get("FLYP_WISE_ALPHAS"),
                "FLYP_TAIL_PROTO_WEIGHT": os.environ.get("FLYP_TAIL_PROTO_WEIGHT"),
                "FLYP_TAIL_PROTO_SCALE": os.environ.get("FLYP_TAIL_PROTO_SCALE"),
                "FLYP_TAIL_PROTO_OBJECTIVE": os.environ.get("FLYP_TAIL_PROTO_OBJECTIVE"),
                "FLYP_TAIL_PROTO_TEMPERATURE": os.environ.get("FLYP_TAIL_PROTO_TEMPERATURE"),
                "FLYP_TAIL_PROTO_TEACHER_LOAD": os.environ.get("FLYP_TAIL_PROTO_TEACHER_LOAD"),
                "FLYP_TAIL_PROTO_MAX_BATCHES": os.environ.get("FLYP_TAIL_PROTO_MAX_BATCHES"),
            }
            try:
                sys.argv = ["kaggle_main.py", "--mode=flyp"]
                os.environ["FLYP_DRM_WEIGHT"] = "0.5"
                os.environ["FLYP_WISE_ALPHAS"] = "0.0,0.05,0.1"
                os.environ["FLYP_TAIL_PROTO_WEIGHT"] = "0.03"
                os.environ["FLYP_TAIL_PROTO_SCALE"] = "100"
                os.environ["FLYP_TAIL_PROTO_OBJECTIVE"] = "distill"
                os.environ["FLYP_TAIL_PROTO_TEMPERATURE"] = "2"
                os.environ["FLYP_TAIL_PROTO_TEACHER_LOAD"] = "/kaggle/input/teacher/flyp.pt"
                os.environ["FLYP_TAIL_PROTO_MAX_BATCHES"] = "7"
                with mock.patch.object(kaggle_main, "ensure_repo_root", return_value=repo_root), \
                     mock.patch.object(kaggle_main.os, "chdir"), \
                     mock.patch.object(kaggle_main, "configure_import_path"), \
                     mock.patch.object(kaggle_main, "_ensure_deps"), \
                     mock.patch.object(kaggle_main, "_ensure_local_package_installed"), \
                     mock.patch.object(kaggle_main, "assert_cloned_repo_supports_runtime_flags"), \
                     mock.patch.object(kaggle_main, "_patch_iwildcam_val"), \
                     mock.patch.object(kaggle_main, "_configure_wandb_from_kaggle_secret"), \
                     mock.patch.object(kaggle_main, "prepare_iwildcam_layout", return_value="./data"), \
                     mock.patch.dict(sys.modules, {"src.config": config_module, "src.train_flyp": train_module}):
                    kaggle_main.main()
            finally:
                sys.argv = original_argv
                for name, value in originals.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

        self.assertEqual(captured["args"].drm_weight, 0.5)
        self.assertEqual(captured["args"].wise_alphas, "0.0,0.05,0.1")
        self.assertEqual(captured["args"].tail_proto_weight, 0.03)
        self.assertEqual(captured["args"].tail_proto_scale, 100.0)
        self.assertEqual(captured["args"].tail_proto_objective, "distill")
        self.assertEqual(captured["args"].tail_proto_temperature, 2.0)
        self.assertEqual(captured["args"].tail_proto_teacher_load, "/kaggle/input/teacher/flyp.pt")
        self.assertEqual(captured["args"].tail_proto_max_batches, 7)

    def test_build_flyp_training_argv_uses_tail_aware_wise_defaults(self):
        from kaggle_main import build_flyp_training_argv

        original_key = os.environ.get("WANDB_API_KEY")
        try:
            os.environ["WANDB_API_KEY"] = "test-key"

            argv = build_flyp_training_argv("./data")
        finally:
            if original_key is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = original_key

        self.assertIn("--model=ViT-B-16", argv)
        self.assertIn("--train-dataset=IWildCam", argv)
        self.assertIn("--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD", argv)
        self.assertIn("--data-location=./data", argv)
        self.assertIn("--batch-size=128", argv)
        self.assertIn("--workers=2", argv)
        self.assertIn("--epochs=20", argv)
        self.assertIn("--lr=1e-5", argv)
        self.assertIn("--wd=0.2", argv)
        self.assertIn("--maple-precision=amp", argv)
        self.assertIn("--template=iwildcam_template", argv)
        self.assertIn("--val-dataset=IWildCamVal", argv)
        self.assertIn("--best-metric=F1-macro_all", argv)
        self.assertIn("--drm-weight=0", argv)
        self.assertIn("--drm-warmup-epochs=0", argv)
        self.assertIn("--tail-proto-weight=0.003", argv)
        self.assertIn("--tail-proto-scale=50", argv)
        self.assertIn("--tail-proto-objective=fixed_distill", argv)
        self.assertIn("--tail-proto-temperature=1.0", argv)
        self.assertIn(
            "--tail-proto-teacher-load=/kaggle/input/datasets/thanhquang71/flyp-nodrm-wise-vitb16-iwildcamval-checkpoint/flyp_nodrm_wise_vitb16_iwildcamval_best.pt",
            argv,
        )
        self.assertIn("--wise-alphas=0.0,0.05,0.1,0.15,0.2,0.3", argv)
        self.assertIn("--wandb-run-name=tail-aware-flyp-fixedtpa-distill-lam0p003-scale50-bs128-wise-vitb16-iwildcamval", argv)
        self.assertIn("--save=/kaggle/working/checkpoints/tail_aware_flyp_fixedtpa_distill_lam0p003_scale50_bs128_wise_vitb16_iwildcamval.pt", argv)
        self.assertIn("--wandb", argv)

    def test_build_flyp_training_argv_preserves_overrides(self):
        from kaggle_main import build_flyp_training_argv

        argv = build_flyp_training_argv("./data", [
            "--model=ViT-L-14",
            "--batch-size=16",
            "--drm-weight=0.01",
            "--tail-proto-weight=0.03",
            "--tail-proto-scale=100",
            "--tail-proto-objective=ce",
            "--tail-proto-temperature=2",
            "--tail-proto-teacher-load=/tmp/teacher.pt",
            "--wise-alphas=0,0.5,1",
            "--no-wandb",
            "--save=/tmp/flyp.pt",
        ])

        self.assertIn("--model=ViT-L-14", argv)
        self.assertIn("--batch-size=16", argv)
        self.assertIn("--drm-weight=0.01", argv)
        self.assertIn("--tail-proto-weight=0.03", argv)
        self.assertIn("--tail-proto-scale=100", argv)
        self.assertIn("--tail-proto-objective=ce", argv)
        self.assertIn("--tail-proto-temperature=2", argv)
        self.assertIn("--tail-proto-teacher-load=/tmp/teacher.pt", argv)
        self.assertIn("--wise-alphas=0,0.5,1", argv)
        self.assertIn("--no-wandb", argv)
        self.assertIn("--save=/tmp/flyp.pt", argv)
        self.assertNotIn("--model=ViT-B-16", argv)
        self.assertNotIn("--batch-size=128", argv)
        self.assertNotIn("--tail-proto-weight=0.003", argv)
        self.assertNotIn("--tail-proto-scale=50", argv)
        self.assertNotIn("--tail-proto-objective=fixed_distill", argv)
        self.assertNotIn("--tail-proto-temperature=1.0", argv)
        self.assertNotIn(
            "--tail-proto-teacher-load=/kaggle/input/datasets/thanhquang71/flyp-nodrm-wise-vitb16-iwildcamval-checkpoint/flyp_nodrm_wise_vitb16_iwildcamval_best.pt",
            argv,
        )
        self.assertNotIn("--wandb", argv)
        self.assertNotIn("--save=/kaggle/working/checkpoints/tail_aware_flyp_fixedtpa_distill_lam0p003_scale50_bs128_wise_vitb16_iwildcamval.pt", argv)

    def test_parse_args_accepts_no_wandb_and_disables_wandb(self):
        from src.config import parse_arguments

        original_argv = sys.argv
        try:
            sys.argv = ["train_flyp.py", "--wandb", "--no-wandb"]

            args = parse_arguments()
        finally:
            sys.argv = original_argv

        self.assertFalse(args.wandb)


if __name__ == "__main__":
    unittest.main()
