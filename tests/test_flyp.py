import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class FlypModuleTest(unittest.TestCase):
    def test_build_flyp_captions_rotates_templates_from_labels(self):
        from src.models.flyp import build_flyp_captions

        labels = torch.tensor([1, 0])
        templates = [lambda c: f"a photo of {c}.", lambda c: f"{c} in the wild."]

        captions = build_flyp_captions(labels, ["frog", "deer"], templates, offset=1)

        self.assertEqual(captions, ["deer in the wild.", "a photo of frog."])

    def test_clip_loss_is_finite_and_backpropagates(self):
        from src.models.flyp import compute_flyp_clip_loss

        image_features = torch.eye(3, requires_grad=True)
        text_features = torch.eye(3, requires_grad=True)
        logit_scale = torch.tensor(10.0)

        loss = compute_flyp_clip_loss(image_features, text_features, logit_scale)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(image_features.grad)
        self.assertIsNotNone(text_features.grad)

    def test_train_flyp_one_epoch_honors_max_train_batches(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))
                self.calls = 0

            def forward(self, images, text):
                self.calls += 1
                features = images @ self.weight
                return features, features, torch.ones(())

        model = TinyModel()
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch, batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
            )

        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))

    def test_train_flyp_one_epoch_uses_grad_scaler_when_provided(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))

            def forward(self, images, text):
                features = images @ self.weight
                return features, features, torch.ones(())

        class FakeScaledLoss:
            def __init__(self, loss):
                self.loss = loss
                self.backward_called = False

            def backward(self):
                self.backward_called = True
                self.loss.backward()

        class FakeScaler:
            def __init__(self):
                self.scaled_loss = None
                self.unscale_called = False
                self.step_called = False
                self.update_called = False

            def scale(self, loss):
                self.scaled_loss = FakeScaledLoss(loss)
                return self.scaled_loss

            def unscale_(self, optimizer):
                self.unscale_called = True

            def step(self, optimizer):
                self.step_called = True
                optimizer.step()

            def update(self):
                self.update_called = True

        model = TinyModel()
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1, use_amp=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = FakeScaler()

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
                scaler=scaler,
            )

        self.assertTrue(scaler.scaled_loss.backward_called)
        self.assertTrue(scaler.unscale_called)
        self.assertTrue(scaler.step_called)
        self.assertTrue(scaler.update_called)
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))


class FlypDrmTest(unittest.TestCase):
    def test_drm_loss_is_zero_when_model_unchanged(self):
        from src.models.flyp import compute_drm_loss

        model = torch.nn.Linear(2, 2, bias=False)
        init_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        loss = compute_drm_loss(model, init_state, drm_weight=1.0)

        self.assertAlmostEqual(loss.item(), 0.0)

    def test_drm_loss_increases_with_deviation(self):
        from src.models.flyp import compute_drm_loss

        model = torch.nn.Linear(2, 2, bias=False)
        init_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(1.0)

        loss = compute_drm_loss(model, init_state, drm_weight=1.0)

        self.assertGreater(loss.item(), 0.0)

    def test_drm_loss_backpropagates(self):
        from src.models.flyp import compute_drm_loss

        model = torch.nn.Linear(2, 2, bias=False)
        init_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(1.0)

        loss = compute_drm_loss(model, init_state, drm_weight=1.0)
        loss.backward()

        self.assertIsNotNone(model.weight.grad)


class FlypWiseTest(unittest.TestCase):
    def test_wise_interpolate_alpha_zero_returns_finetuned_state(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"weight": torch.tensor([1.0, 2.0])}
        zeroshot = {"weight": torch.tensor([3.0, 4.0])}

        result = wise_interpolate_state_dict(finetuned, zeroshot, alpha=0.0)

        self.assertTrue(torch.equal(result["weight"], finetuned["weight"]))

    def test_wise_interpolate_alpha_one_returns_zeroshot_state(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"weight": torch.tensor([1.0, 2.0])}
        zeroshot = {"weight": torch.tensor([3.0, 4.0])}

        result = wise_interpolate_state_dict(finetuned, zeroshot, alpha=1.0)

        self.assertTrue(torch.equal(result["weight"], zeroshot["weight"]))

    def test_wise_interpolate_alpha_half_returns_midpoint(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"weight": torch.tensor([1.0, 2.0])}
        zeroshot = {"weight": torch.tensor([3.0, 4.0])}

        result = wise_interpolate_state_dict(finetuned, zeroshot, alpha=0.5)

        self.assertTrue(torch.allclose(result["weight"], torch.tensor([2.0, 3.0])))

    def test_wise_interpolate_requires_matching_keys(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"weight": torch.tensor([1.0])}
        zeroshot = {"bias": torch.tensor([1.0])}

        with self.assertRaises(ValueError):
            wise_interpolate_state_dict(finetuned, zeroshot, alpha=0.5)

    def test_wise_interpolate_preserves_equal_integer_buffers(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"buffer": torch.tensor([1, 2], dtype=torch.long)}
        zeroshot = {"buffer": torch.tensor([1, 2], dtype=torch.long)}

        result = wise_interpolate_state_dict(finetuned, zeroshot, alpha=0.5)

        self.assertTrue(torch.equal(result["buffer"], finetuned["buffer"]))

    def test_wise_interpolate_rejects_different_integer_buffers(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"buffer": torch.tensor([1, 2], dtype=torch.long)}
        zeroshot = {"buffer": torch.tensor([1, 3], dtype=torch.long)}

        with self.assertRaises(ValueError):
            wise_interpolate_state_dict(finetuned, zeroshot, alpha=0.5)


class FlypWiseParseTest(unittest.TestCase):
    def test_parse_wise_alphas_none_returns_empty(self):
        from src.train_flyp import parse_wise_alphas
        self.assertEqual(parse_wise_alphas(None), [])

    def test_parse_wise_alphas_string_csv(self):
        from src.train_flyp import parse_wise_alphas
        result = parse_wise_alphas("0.0,0.25,0.5,0.75,1.0")
        self.assertEqual(result, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_parse_wise_alphas_list(self):
        from src.train_flyp import parse_wise_alphas
        result = parse_wise_alphas([0.1, 0.5])
        self.assertEqual(result, [0.1, 0.5])

    def test_parse_wise_alphas_single_value(self):
        from src.train_flyp import parse_wise_alphas
        result = parse_wise_alphas("0.3")
        self.assertEqual(result, [0.3])

    def test_parse_wise_alphas_whitespace(self):
        from src.train_flyp import parse_wise_alphas
        result = parse_wise_alphas(" 0.1 , 0.9 ")
        self.assertEqual(result, [0.1, 0.9])


class FlypUnpackClipForwardTest(unittest.TestCase):
    def test_unpack_dict(self):
        from src.models.flyp import unpack_clip_forward
        img, txt, scale = unpack_clip_forward({
            "image_features": "i", "text_features": "t", "logit_scale": "s"
        })
        self.assertEqual((img, txt, scale), ("i", "t", "s"))

    def test_unpack_tuple(self):
        from src.models.flyp import unpack_clip_forward
        result = unpack_clip_forward(("i", "t", "s"))
        self.assertEqual(result, ("i", "t", "s"))

    def test_unpack_too_short_raises(self):
        from src.models.flyp import unpack_clip_forward
        with self.assertRaises(ValueError):
            unpack_clip_forward(("i",))


class FlypCloneStateDictTest(unittest.TestCase):
    def test_clone_creates_independent_copy(self):
        from src.train_flyp import clone_state_dict
        model = torch.nn.Linear(2, 2, bias=False)
        sd = clone_state_dict(model)
        with torch.no_grad():
            model.weight.add_(100.0)
        self.assertFalse(torch.equal(sd["weight"], model.weight))


class FlypSavePathTest(unittest.TestCase):
    def test_resolve_none(self):
        from src.train_flyp import resolve_flyp_save_path
        self.assertIsNone(resolve_flyp_save_path(None))

    def test_resolve_directory(self):
        import os
        import tempfile

        from src.train_flyp import resolve_flyp_save_path

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_flyp_save_path(d), os.path.join(d, "flyp_clip_encoder.pt"))

    def test_resolve_no_extension(self):
        from src.train_flyp import resolve_flyp_save_path
        self.assertEqual(resolve_flyp_save_path("checkpoints/myrun"), "checkpoints/myrun/flyp_clip_encoder.pt")

    def test_resolve_file(self):
        from src.train_flyp import resolve_flyp_save_path
        self.assertEqual(resolve_flyp_save_path("mydir/model.pt"), "mydir/model.pt")


class FlypBestCheckpointPathTest(unittest.TestCase):
    def test_best_checkpoint_explicit(self):
        from src.train_flyp import resolve_flyp_best_checkpoint_path
        args = SimpleNamespace(best_checkpoint="/tmp/explicit.pt", save=None)
        self.assertEqual(resolve_flyp_best_checkpoint_path(args), "/tmp/explicit.pt")

    def test_best_checkpoint_from_save_dir(self):
        import os
        import tempfile

        from src.train_flyp import resolve_flyp_best_checkpoint_path

        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(best_checkpoint=None, save=d)
            self.assertEqual(resolve_flyp_best_checkpoint_path(args), os.path.join(d, "flyp_clip_encoder_best.pt"))

    def test_best_checkpoint_from_save_file(self):
        from src.train_flyp import resolve_flyp_best_checkpoint_path
        args = SimpleNamespace(best_checkpoint=None, save="mydir/model.pt")
        self.assertEqual(resolve_flyp_best_checkpoint_path(args), "mydir/model_best.pt")

    def test_best_checkpoint_fallback(self):
        from src.train_flyp import resolve_flyp_best_checkpoint_path
        args = SimpleNamespace(best_checkpoint=None, save=None)
        self.assertEqual(resolve_flyp_best_checkpoint_path(args), "checkpoints/flyp_clip_encoder_best.pt")


class FlypDrmIntegrationTest(unittest.TestCase):
    def test_drm_contributes_to_total_loss(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))

            def forward(self, images, text):
                features = images @ self.weight
                return features, features, torch.ones(())

        model = TinyModel()
        init_state = {n: t.detach().cpu().clone() for n, t in model.named_parameters()}
        with torch.no_grad():
            model.weight.add_(0.5)

        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model, [batch], optimizer, args, ["frog", "deer"],
                [lambda c: f"a photo of {c}."], epoch=1,
                init_state_dict=init_state, drm_weight=1.0,
            )

        self.assertGreater(stats.drm_loss, 0.0)
        self.assertAlmostEqual(stats.loss, stats.clip_loss + stats.drm_loss, places=5)

    def test_drm_raises_without_init_state(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))

            def forward(self, images, text):
                features = images @ self.weight
                return features, features, torch.ones(())

        model = TinyModel()
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            with self.assertRaises(ValueError):
                train_flyp_one_epoch(
                    model, [batch], optimizer, args, ["frog", "deer"],
                    [lambda c: f"a photo of {c}."], epoch=1,
                    init_state_dict=None, drm_weight=1.0,
                )


class FlypWiseSelectionIntegrationTest(unittest.TestCase):
    def test_wise_grid_picks_best_alpha(self):
        from src.models.flyp import wise_interpolate_state_dict

        finetuned = {"weight": torch.tensor([1.0, 2.0])}
        zeroshot = {"weight": torch.tensor([3.0, 4.0])}

        best_alpha = None
        best_score = -1
        for alpha in [0.0, 0.5, 1.0]:
            interpolated = wise_interpolate_state_dict(finetuned, zeroshot, alpha)
            score = 1.0 - abs(interpolated["weight"].mean().item() - 2.5)
            if best_score is None or score > best_score:
                best_score = score
                best_alpha = alpha

        self.assertEqual(best_alpha, 0.5)

    def test_wise_alpha_bounds_validation(self):
        from src.models.flyp import wise_interpolate_state_dict
        finetuned = {"weight": torch.tensor([1.0])}
        zeroshot = {"weight": torch.tensor([3.0])}
        with self.assertRaises(ValueError):
            wise_interpolate_state_dict(finetuned, zeroshot, alpha=-0.1)
        with self.assertRaises(ValueError):
            wise_interpolate_state_dict(finetuned, zeroshot, alpha=1.1)


if __name__ == "__main__":
    unittest.main()
