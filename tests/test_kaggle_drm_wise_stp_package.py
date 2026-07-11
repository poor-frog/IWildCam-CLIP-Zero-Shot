import json
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-drm-wise-stp"


class KaggleDrmWiseStpPackageTest(unittest.TestCase):
    def test_metadata_targets_the_dedicated_wise_stp_kernel(self):
        metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

        self.assertEqual("huyphmhakq/poorfrogs-drm-wise-stp-iwildcam", metadata["id"])
        self.assertEqual("kaggle_main.py", metadata["code_file"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertIn("klinh1912/drm-iwildcam-vitb16-checkpoint", metadata["dataset_sources"])

    def test_wrapper_uses_the_validation_only_wise_alpha_sweep(self):
        source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

        self.assertIn('DEFAULT_WISE_ALPHA_GRID = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"', source)
        self.assertIn('env.setdefault("DRM_WISE_ALPHA_GRID", DEFAULT_WISE_ALPHA_GRID)', source)
        self.assertIn('"kaggle_eval_drm_stmp_adapter.py"', source)


if __name__ == "__main__":
    unittest.main()
