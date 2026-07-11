import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock


class KaggleEvalDrmStmpAdapterTest(unittest.TestCase):
    def test_hardcoded_wandb_fallback_is_empty_by_default(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        self.assertEqual("", launcher.HARDCODED_WANDB_API_KEY)

    def test_wandb_environment_key_takes_priority_over_hardcoded_fallback(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "environment-key"}, clear=True), \
             mock.patch.object(launcher, "HARDCODED_WANDB_API_KEY", "fallback-key"):
            self.assertTrue(launcher.configure_wandb())
            self.assertEqual("environment-key", os.environ["WANDB_API_KEY"])

    def test_wandb_hardcoded_fallback_is_used_without_secret(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(launcher, "HARDCODED_WANDB_API_KEY", "fallback-key"):
            self.assertTrue(launcher.configure_wandb())
            self.assertEqual("fallback-key", os.environ["WANDB_API_KEY"])

    def test_preflight_rejects_repo_without_drm_blend_helper(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "src/eval_drm_blend.py"):
                launcher.assert_repo_supports_drm_concept_eval(Path(tmpdir))

    def test_main_builds_drm_stmp_command(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        captured = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "poorfrogs"
            drm_repo = root / "drm"
            helper_path = repo_root / "src" / "eval_drm_blend.py"
            helper_path.parent.mkdir(parents=True)
            helper_path.write_text("")
            with mock.patch.object(launcher, "ensure_deps"), \
                 mock.patch.object(launcher, "clone_or_update", side_effect=[repo_root, drm_repo]), \
                 mock.patch.object(launcher, "configure_import_path"), \
                 mock.patch.object(launcher, "ensure_local_package_installed"), \
                 mock.patch.object(launcher, "patch_tail_cache_eval_guard"), \
                 mock.patch.object(launcher, "patch_iwildcam_val"), \
                 mock.patch.object(launcher, "prepare_iwildcam_layout", return_value="/kaggle/working/data"), \
                 mock.patch.object(launcher, "find_drm_checkpoint", return_value=Path("/kaggle/input/drm.pt")), \
                 mock.patch.object(launcher, "export_drm_state_dict"), \
                 mock.patch.object(launcher, "build_poorfrogs_checkpoint"), \
                 mock.patch.object(launcher, "configure_wandb", return_value=True), \
                 mock.patch.object(launcher, "run", side_effect=lambda command, cwd=None, env=None: captured.append(command)):
                launcher.main()

        command = captured[-1]
        self.assertIn("--template=iwildcam_drm_template", command)
        self.assertIn("--prototype-scale-grid=50", command)
        self.assertIn("--sequence-consensus-grid=0,0.25,0.5", command)
        self.assertIn("--multi-prototype-k-grid=1", command)
        self.assertIn("--sctr-strength-grid=0.25,0.5,1", command)
        self.assertIn("--sctr-tail-protection-grid=0,0.5,1,2", command)
        self.assertIn("--wandb-run-name=drm-sctr-v1-route0p25-0p5-1-tail0-0p5-1-2-vitb16-iwildcamval", command)

    def test_main_uses_two_phase_driver_when_wise_grid_is_configured(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        captured = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "poorfrogs"
            drm_repo = root / "drm"
            with mock.patch.dict(os.environ, {"DRM_WISE_ALPHA_GRID": "0,0.1,0.2"}, clear=False), \
                 mock.patch.object(launcher, "WISE_ALPHA_GRID", "0,0.1,0.2"), \
                 mock.patch.object(launcher, "ensure_deps"), \
                 mock.patch.object(launcher, "clone_or_update", side_effect=[repo_root, drm_repo]), \
                 mock.patch.object(launcher, "configure_import_path"), \
                 mock.patch.object(launcher, "ensure_local_package_installed"), \
                 mock.patch.object(launcher, "patch_tail_cache_eval_guard"), \
                 mock.patch.object(launcher, "patch_iwildcam_val"), \
                 mock.patch.object(launcher, "assert_repo_supports_drm_wise_stp_eval"), \
                 mock.patch.object(launcher, "prepare_iwildcam_layout", return_value="/kaggle/working/data"), \
                 mock.patch.object(launcher, "find_drm_checkpoint", return_value=Path("/kaggle/input/drm.pt")), \
                 mock.patch.object(launcher, "export_drm_state_dict"), \
                 mock.patch.object(launcher, "build_poorfrogs_checkpoint"), \
                 mock.patch.object(launcher, "configure_wandb", return_value=True), \
                 mock.patch.object(launcher, "run", side_effect=lambda command, cwd=None, env=None: captured.append(command)):
                launcher.main()

        command = captured[-1]
        self.assertIn("src/eval_drm_wise_stp.py", command)
        self.assertIn("--wise-alpha-grid=0,0.1,0.2", command)
        self.assertIn("--selection-dir=/kaggle/working/drm_wise_stp_selection", command)
        self.assertIn("--wandb-run-prefix=drm-wise-stp-vitb16-iwildcamval", command)
        self.assertIn("--wandb", command)
        self.assertIn("--wandb-project=PoorFrogs", command)
        self.assertNotIn("--wandb-run-name=drm-sctr-v1-route0p25-0p5-1-tail0-0p5-1-2-vitb16-iwildcamval", command)


if __name__ == "__main__":
    unittest.main()
