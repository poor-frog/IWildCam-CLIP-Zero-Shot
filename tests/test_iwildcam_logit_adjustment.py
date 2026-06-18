import unittest
from types import SimpleNamespace

import torch


class LogitAdjustmentTest(unittest.TestCase):
    def test_tau_zero_returns_original_logits(self):
        from src.models.logit_adjustment import apply_logit_adjustment

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        priors = torch.tensor([0.5, 0.3, 0.2])
        adjusted = apply_logit_adjustment(logits, priors, tau=0.0)
        self.assertTrue(torch.equal(adjusted, logits))

    def test_positive_tau_subtracts_log_priors(self):
        from src.models.logit_adjustment import apply_logit_adjustment

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        priors = torch.tensor([0.5, 0.25, 0.25])
        adjusted = apply_logit_adjustment(logits, priors, tau=1.0)
        expected = logits - torch.log(priors)
        self.assertTrue(torch.allclose(adjusted, expected))

    def test_finite_with_zero_like_priors(self):
        from src.models.logit_adjustment import apply_logit_adjustment

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        priors = torch.tensor([0.5, 0.0, 0.5])
        adjusted = apply_logit_adjustment(logits, priors, tau=1.0)
        self.assertTrue(torch.isfinite(adjusted).all())

    def test_validate_selection_split_rejects_ood(self):
        from src.models.logit_adjustment import validate_selection_split

        with self.assertRaisesRegex(ValueError, "final-test-only"):
            validate_selection_split("IWildCamOOD")

    def test_parse_tau_grid_preserves_order(self):
        from src.models.logit_adjustment import parse_tau_grid

        self.assertEqual(parse_tau_grid("0,1,2.5"), [0.0, 1.0, 2.5])

    def test_select_best_tau_records_selection_split(self):
        from src.models.logit_adjustment import select_best_tau

        args = SimpleNamespace(selection_split="IWildCamVal")

        def fake_eval(_model, _dataset, _args, tau=None, class_priors=None):
            del _model, _dataset, _args, class_priors
            return {"top1": 0.5 + 0.1 * float(tau), "F1-macro_all": 0.1 + 0.2 * float(tau)}

        result = select_best_tau(fake_eval, object(), object(), args, tau_grid=[0.0, 1.0], class_priors=torch.ones(3))

        self.assertEqual(result.selection_split, "IWildCamVal")
        self.assertEqual(result.best_tau, 1.0)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0]["selection_split"], "IWildCamVal")


if __name__ == "__main__":
    unittest.main()
