import json
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-drm-wise-tpa-control"


class KaggleDrmWiseTpaControlPackageTest(unittest.TestCase):
    def test_metadata_targets_the_dedicated_tpa_control_kernel(self):
        metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

        self.assertEqual("huyphmhakq/poorfrogs-drm-wise-tpa-control-iwildcam", metadata["id"])
        self.assertEqual("kaggle_main.py", metadata["code_file"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertIn("klinh1912/drm-iwildcam-vitb16-checkpoint", metadata["dataset_sources"])

    def test_wrapper_forces_tpa_without_sequence_consensus(self):
        source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

        self.assertIn('env["DRM_STMP_SEQUENCE_CONSENSUS_GRID"] = "0"', source)
        self.assertIn('env.setdefault("DRM_WISE_ALPHA_GRID", DEFAULT_WISE_ALPHA_GRID)', source)
        self.assertIn('"kaggle_eval_drm_stmp_adapter.py"', source)


if __name__ == "__main__":
    unittest.main()
