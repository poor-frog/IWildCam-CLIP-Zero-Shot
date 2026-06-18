import unittest

import torch
import torch.nn.functional as F

from src.models.maple_full import compute_maple_cross_entropy
from src.train_maple_full import build_class_balanced_ce_weights


class FakeSubset:
    def __init__(self, labels):
        self.y_array = torch.tensor(labels, dtype=torch.long)


class TrainOnlyDataset:
    classnames = ["common", "rare"]

    def __init__(self, labels):
        self.labels = labels
        self.requested_splits = []

    def get_subset(self, split, transform=None):
        self.requested_splits.append(split)
        if split != "train":
            raise AssertionError(f"unexpected non-train split requested: {split}")
        return FakeSubset(self.labels)


class ClassBalancedCETest(unittest.TestCase):
    def test_default_ce_matches_torch_cross_entropy(self):
        logits = torch.tensor(
            [[2.0, -1.0, 0.5], [-0.5, 1.0, 0.25], [0.1, 0.2, 2.2]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 2], dtype=torch.long)

        actual = compute_maple_cross_entropy(logits, labels, class_weights=None)
        expected = F.cross_entropy(logits, labels)

        self.assertTrue(torch.allclose(actual, expected))

    def test_weighted_ce_uses_finite_inverse_frequency_weights(self):
        logits = torch.tensor(
            [[2.0, -1.0], [1.5, -0.5], [0.25, 1.75], [0.1, 2.1]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 0, 1], dtype=torch.long)
        train_data = TrainOnlyDataset(labels.tolist())

        class_weights = build_class_balanced_ce_weights(train_data, device=torch.device("cpu"))
        unweighted_loss = compute_maple_cross_entropy(logits, labels, class_weights=None)
        weighted_loss = compute_maple_cross_entropy(logits, labels, class_weights=class_weights)

        self.assertEqual(train_data.requested_splits, ["train"])
        self.assertTrue(torch.isfinite(class_weights).all())
        self.assertEqual(class_weights.shape, torch.Size([2]))
        self.assertFalse(torch.allclose(weighted_loss, unweighted_loss))

    def test_weight_helper_uses_train_split_only(self):
        train_data = TrainOnlyDataset([0, 0, 1])

        build_class_balanced_ce_weights(train_data, device=torch.device("cpu"))

        self.assertEqual(train_data.requested_splits, ["train"])


if __name__ == "__main__":
    unittest.main()
