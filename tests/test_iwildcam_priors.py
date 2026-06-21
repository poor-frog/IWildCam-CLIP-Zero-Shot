import unittest

import numpy as np
import torch


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

    def test_ood_hp_subsampling_preserves_wilds_collate(self):
        from src.datasets.iwildcam import maybe_subsample_ood_val

        class FakeWildsSubset:
            def __init__(self):
                self.y_array = torch.tensor([0, 1, 0, 1])
                self.indices = np.arange(4)
                self.dataset = type("FakeDataset", (), {"_collate": staticmethod(self.collate_batch)})()
                self.transform = "transform"
                self.collate = self.collate_batch

            @staticmethod
            def collate_batch(batch):
                return batch

            def __len__(self):
                return len(self.y_array)

        subset = FakeWildsSubset()

        sampled = maybe_subsample_ood_val(subset, n_examples=2, seed=0)

        self.assertNotIsInstance(sampled, torch.utils.data.Subset)
        self.assertTrue(hasattr(sampled, "collate"))
        self.assertIsNotNone(sampled.collate)

    def test_apply_class_bias_adds_per_class_offsets(self):
        from src.models.logit_adjustment import apply_class_bias

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        class_bias = torch.tensor([0.5, -1.0, 2.0])

        adjusted = apply_class_bias(logits, class_bias)

        self.assertTrue(torch.equal(adjusted, torch.tensor([[1.5, 1.0, 5.0]])))

    def test_select_best_class_bias_prefers_macro_f1_candidate(self):
        from src.models.logit_adjustment import select_best_class_bias

        class TinyDataset:
            test_loader = [{"images": torch.zeros(3, 1), "labels": torch.tensor([0, 1, 1])}]

        class TinyModel(torch.nn.Module):
            def forward(self, images):
                return torch.tensor([
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [2.0, 1.0],
                ])

        args = type("Args", (), {"device": "cpu", "max_eval_batches": None, "selection_split": "IWildCamVal"})()
        candidates = [torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 1.0])]

        selection = select_best_class_bias(TinyModel(), TinyDataset(), args, candidates)

        self.assertEqual(selection.best_index, 1)
        self.assertTrue(torch.equal(selection.best_bias, candidates[1].float()))
        self.assertEqual(selection.rows[1]["F1-macro_all"], 1.0)


if __name__ == "__main__":
    unittest.main()
