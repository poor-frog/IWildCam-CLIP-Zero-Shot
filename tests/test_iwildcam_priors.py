import unittest

import numpy as np


class IWildCamClassPriorTest(unittest.TestCase):
    def test_synthetic_class_priors_are_normalized(self):
        from src.datasets.iwildcam import compute_class_priors

        counts, priors = compute_class_priors([0, 0, 1, 2, 2, 2], num_classes=3)

        np.testing.assert_array_equal(counts, np.array([2.0, 1.0, 3.0]))
        np.testing.assert_allclose(priors, np.array([2 / 6, 1 / 6, 3 / 6]))
        self.assertAlmostEqual(float(priors.sum()), 1.0)

    def test_zero_count_class_has_finite_weight(self):
        from src.datasets.iwildcam import compute_inverse_frequency_weights

        counts, priors, weights = compute_inverse_frequency_weights([0, 0, 2], num_classes=4)

        np.testing.assert_array_equal(counts, np.array([2.0, 0.0, 1.0, 0.0]))
        self.assertTrue(np.isfinite(priors).all())
        self.assertTrue(np.isfinite(weights).all())
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[3], 0.0)
        self.assertGreater(weights[0], 0.0)
        self.assertGreater(weights[2], 0.0)

    def test_empty_labels_error_clearly(self):
        from src.datasets.iwildcam import compute_class_priors

        with self.assertRaisesRegex(ValueError, "empty labels"):
            compute_class_priors([], num_classes=3)


if __name__ == "__main__":
    unittest.main()
