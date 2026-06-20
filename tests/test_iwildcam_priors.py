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

    def test_random_ood_hp_sampling_is_deterministic(self):
        from src.datasets.iwildcam import sample_indices

        labels = np.arange(20) % 4

        first = sample_indices(labels, n_examples=6, seed=7)
        second = sample_indices(labels, n_examples=6, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first.tolist())), 6)

    def test_class_balanced_ood_hp_sampling_uses_each_present_class(self):
        from src.datasets.iwildcam import sample_indices

        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        sampled = sample_indices(labels, n_examples=6, seed=3, class_balanced=True, num_classes=3)
        sampled_labels = labels[sampled]

        np.testing.assert_array_equal(np.bincount(sampled_labels, minlength=3), np.array([2, 2, 2]))

    def test_class_balanced_ood_hp_sampling_returns_requested_count_with_remainder(self):
        from src.datasets.iwildcam import sample_indices

        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        sampled = sample_indices(labels, n_examples=8, seed=4, class_balanced=True, num_classes=3)

        self.assertEqual(len(sampled), 8)


if __name__ == "__main__":
    unittest.main()
