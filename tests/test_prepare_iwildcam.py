import unittest

from scripts.prepare_iwildcam import build_parser


class PrepareIWildCamTest(unittest.TestCase):
    def test_defaults_match_current_data_layout(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.save_file, "./data/train.csv")
        self.assertEqual(args.metadata, "./data/iwildcam_v2.0/metadata.csv")
        self.assertEqual(args.data_dir, "./data/iwildcam_v2.0/train")
        self.assertEqual(args.english_label_path, "./src/datasets/iwildcam_metadata/labels.csv")


if __name__ == "__main__":
    unittest.main()
