import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class KaggleEvalTailPrototypeAdapterTest(unittest.TestCase):
    def test_main_builds_stmp_eval_command(self):
        import kaggle_eval_tail_prototype_adapter as launcher

        captured = []
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            try:
                with mock.patch.object(launcher, "ensure_repo_root", return_value=repo_root), \
                     mock.patch.object(launcher, "configure_import_path"), \
                     mock.patch.object(launcher, "ensure_deps"), \
                     mock.patch.object(launcher, "ensure_local_package_installed"), \
                     mock.patch.object(launcher, "assert_repo_supports_tail_adapter"), \
                     mock.patch.object(launcher, "patch_iwildcam_val"), \
                     mock.patch.object(launcher, "prepare_iwildcam_layout", return_value="/kaggle/working/data"), \
                     mock.patch.object(launcher, "find_checkpoint", return_value="/kaggle/input/checkpoint/flyp.pt"), \
                     mock.patch.object(launcher, "configure_wandb", return_value=False), \
                     mock.patch.object(launcher.subprocess, "check_call", side_effect=lambda command: captured.append(command)):
                    launcher.main()
            finally:
                os.chdir(original_cwd)

        command = captured[-1]
        self.assertIn("--sequence-consensus-grid=0,0.5", command)
        self.assertIn("--sequence-id-field=auto", command)
        self.assertIn("--multi-prototype-k-grid=1,8", command)
        self.assertIn("--multi-prototype-reduction=max", command)
        self.assertIn("--gate-mode-grid=none,margin,entropy", command)
        self.assertIn("--gate-strength-grid=0,0.25,1.0", command)
        self.assertIn("--audit-metadata", command)
        self.assertIn("--no-wandb", command)


if __name__ == "__main__":
    unittest.main()
