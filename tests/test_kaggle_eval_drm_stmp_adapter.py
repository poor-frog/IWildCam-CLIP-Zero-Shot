import tempfile
import unittest
from pathlib import Path
from unittest import mock


class KaggleEvalDrmStmpAdapterTest(unittest.TestCase):
    def test_hardcoded_wandb_key_is_empty_by_default(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        self.assertEqual("", launcher.HARDCODED_WANDB_API_KEY)

    def test_preflight_rejects_repo_without_drm_blend_helper(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "src/eval_drm_blend.py"):
                launcher.assert_repo_supports_drm_concept_eval(Path(tmpdir))

    def test_main_builds_official_drm_concept_parity_command(self):
        import kaggle_eval_drm_stmp_adapter as launcher

        captured = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "poorfrogs"
            drm_repo = root / "drm"
            helper_path = repo_root / "src" / "eval_drm_blend.py"
            helper_path.parent.mkdir(parents=True)
            helper_path.write_text("")
            concept_path = drm_repo / "prompts" / "iwildcam_cd.json"
            concept_path.parent.mkdir(parents=True)
            concept_path.write_text("{}")

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
        self.assertIn(f"--cd-path={concept_path}", command)
        self.assertIn("--concept-beta-grid=0.5", command)
        self.assertIn("--prototype-scale-grid=0", command)
        self.assertIn("--sequence-consensus-grid=0", command)
        self.assertIn("--multi-prototype-k-grid=1", command)
        self.assertIn("--wandb-run-name=drm-official-vitb16-concept-parity-iwildcamval", command)


if __name__ == "__main__":
    unittest.main()
