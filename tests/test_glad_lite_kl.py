import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from src.models.maple_full import compute_maple_cross_entropy, train_full_maple_one_epoch


class GLADLiteKLLossTest(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [[2.0, -1.0, 0.5], [-0.25, 1.5, 0.0], [0.1, -0.4, 2.2]],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 1, 2], dtype=torch.long)
        self.anchor_logits = torch.tensor(
            [[0.1, 1.2, -0.6], [1.25, -0.5, 0.2], [-0.2, 1.1, 0.7]],
            dtype=torch.float32,
        )

    def test_kl_disabled_default_path_matches_ce_only_loss(self):
        result = compute_maple_cross_entropy(self.logits, self.labels, return_components=True)

        expected_ce = F.cross_entropy(self.logits, self.labels)

        self.assertTrue(torch.allclose(result.ce, expected_ce))
        self.assertEqual(result.kl.item(), 0.0)
        self.assertTrue(torch.allclose(result.total, expected_ce))
        self.assertTrue(torch.allclose(result.loss, result.total))

    def test_kl_enabled_total_uses_weight_temperature_squared_scaling(self):
        kl_weight = 0.35
        temperature = 2.5

        result = compute_maple_cross_entropy(
            self.logits,
            self.labels,
            anchor_logits=self.anchor_logits,
            kl_weight=kl_weight,
            kl_temperature=temperature,
            return_components=True,
        )

        expected_ce = F.cross_entropy(self.logits, self.labels)
        expected_kl = F.kl_div(
            F.log_softmax(self.logits / temperature, dim=1),
            F.softmax(self.anchor_logits / temperature, dim=1),
            reduction="batchmean",
        )
        expected_total = expected_ce + kl_weight * temperature * temperature * expected_kl

        self.assertTrue(torch.isfinite(result.kl))
        self.assertGreater(result.kl.item(), 0.0)
        self.assertTrue(torch.allclose(result.ce, expected_ce))
        self.assertTrue(torch.allclose(result.kl, expected_kl))
        self.assertTrue(torch.allclose(result.total, expected_total))
        self.assertTrue(torch.allclose(result.loss, expected_total))

    def test_loss_result_exposes_logging_fields_for_smoke_logging(self):
        result = compute_maple_cross_entropy(
            self.logits,
            self.labels,
            anchor_logits=self.anchor_logits,
            kl_weight=0.25,
            kl_temperature=1.5,
            return_components=True,
        )

        log_fields = result.to_log_dict(prefix="train/batch")

        self.assertEqual(set(log_fields), {"train/batch_ce", "train/batch_kl", "train/batch_total"})
        self.assertAlmostEqual(log_fields["train/batch_ce"], result.ce.item())
        self.assertAlmostEqual(log_fields["train/batch_kl"], result.kl.item())
        self.assertAlmostEqual(log_fields["train/batch_total"], result.total.item())


class GLADLiteKLTrainLoopTest(unittest.TestCase):
    def test_train_loop_logs_ce_kl_total_when_anchor_kl_enabled(self):
        student_logits = torch.tensor(
            [[1.5, -0.25, 0.0], [-0.5, 1.25, 0.75]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 2], dtype=torch.long)
        anchor_logits = torch.tensor(
            [[-0.25, 1.0, 0.1], [0.8, -0.2, 1.4]],
            dtype=torch.float32,
        )

        class TinyLogitModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = torch.nn.Parameter(logits.clone())

            def forward(self, images):
                return self.logits[: images.shape[0]]

        class FixedAnchor(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.register_buffer("logits", logits.clone())
                self.grad_enabled = None

            def forward(self, images):
                self.grad_enabled = torch.is_grad_enabled()
                return self.logits[: images.shape[0]]

        class FakeWandb:
            def __init__(self):
                self.calls = []

            def log(self, fields):
                self.calls.append(fields)

        model = TinyLogitModel(student_logits)
        anchor = FixedAnchor(anchor_logits)
        wandb = FakeWandb()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dataloader = [{"images": torch.zeros(2, 1), "labels": labels}]
        args = SimpleNamespace(device=torch.device("cpu"), max_train_batches=None)
        kl_weight = 0.4
        temperature = 2.0

        expected = compute_maple_cross_entropy(
            student_logits,
            labels,
            anchor_logits=anchor_logits,
            kl_weight=kl_weight,
            kl_temperature=temperature,
            return_components=True,
        )

        stats = train_full_maple_one_epoch(
            model,
            dataloader,
            optimizer,
            args,
            epoch=3,
            wandb=wandb,
            anchor_model=anchor,
            kl_weight=kl_weight,
            kl_temperature=temperature,
        )

        self.assertEqual(len(wandb.calls), 1)
        logged = wandb.calls[0]
        self.assertFalse(anchor.grad_enabled)
        self.assertIn("train/batch_loss", logged)
        self.assertIn("train/batch_ce", logged)
        self.assertIn("train/batch_kl", logged)
        self.assertIn("train/batch_total", logged)
        self.assertGreater(logged["train/batch_kl"], 0.0)
        self.assertAlmostEqual(logged["train/batch_ce"], expected.ce.item())
        self.assertAlmostEqual(logged["train/batch_kl"], expected.kl.item())
        self.assertAlmostEqual(logged["train/batch_total"], expected.total.item())
        self.assertAlmostEqual(logged["train/batch_loss"], expected.total.item())
        self.assertAlmostEqual(stats.loss, expected.total.item())



class GLADLiteKLParserTest(unittest.TestCase):
    def test_parser_defaults_keep_kl_disabled(self):
        from src.config import parse_arguments

        with mock.patch.object(sys, "argv", ["train_maple_full.py"]):
            args = parse_arguments()

        self.assertEqual(args.kl_weight, 0.0)
        self.assertEqual(args.kl_temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
