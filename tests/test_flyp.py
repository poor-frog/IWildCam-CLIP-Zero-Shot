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

    def test_clip_loss_treats_duplicate_labels_as_positives(self):
        from src.models.flyp import compute_flyp_clip_loss

        image_features = torch.tensor([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ])
        text_features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        class_labels = torch.tensor([0, 0, 1])

        instance_loss = compute_flyp_clip_loss(image_features, text_features, torch.tensor(10.0))
        class_aware_loss = compute_flyp_clip_loss(
            image_features,
            text_features,
            torch.tensor(10.0),
            class_labels=class_labels,
        )

        self.assertLess(class_aware_loss.item(), instance_loss.item())

    def test_tail_prototypes_average_and_normalize_train_features(self):
        from src.models.tail_prototype import build_class_prototypes_from_loader

        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return images

        dataloader = [
            {"images": torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.0, 1.0]]), "labels": torch.tensor([0, 1, 1])}
        ]

        prototypes, counts = build_class_prototypes_from_loader(
            TinyEncoder(),
            dataloader,
            device="cpu",
            num_classes=3,
        )

        self.assertEqual(counts.tolist(), [1, 2, 0])
        self.assertTrue(torch.allclose(prototypes[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(prototypes[1], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(prototypes[2], torch.zeros(2)))

    def test_tail_prototype_loss_prefers_matching_class_prototypes(self):
        from src.models.tail_prototype import tail_prototype_loss

        image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prototypes = torch.eye(2)
        correct = tail_prototype_loss(image_features, torch.tensor([0, 1]), prototypes, prototype_scale=10.0)
        swapped = tail_prototype_loss(image_features, torch.tensor([1, 0]), prototypes, prototype_scale=10.0)

        self.assertLess(correct.item(), swapped.item())

    def test_tail_prototype_distillation_loss_prefers_tpa_consistent_prototypes(self):
        from src.models.tail_prototype import tail_prototype_distillation_loss

        classification_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classification_head.weight.copy_(torch.eye(2))
        image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        consistent = tail_prototype_distillation_loss(
            image_features,
            torch.eye(2),
            prototype_scale=10.0,
            classification_head=classification_head,
        )
        contradictory = tail_prototype_distillation_loss(
            image_features,
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            prototype_scale=10.0,
            classification_head=classification_head,
        )

        self.assertLess(consistent.item(), contradictory.item())

    def test_fixed_tail_prototype_distillation_uses_frozen_teacher_features(self):
        from src.models.tail_prototype import fixed_tail_prototype_distillation_loss

        student_head = torch.nn.Linear(2, 2, bias=False)
        teacher_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            student_head.weight.copy_(torch.eye(2))
            teacher_head.weight.copy_(torch.eye(2))
        student_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        teacher_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        loss = fixed_tail_prototype_distillation_loss(
            student_features,
            teacher_features,
            torch.eye(2),
            prototype_scale=10.0,
            student_classification_head=student_head,
            teacher_classification_head=teacher_head,
        )

        self.assertGreater(loss.item(), 0.0)

    def test_tail_class_weights_emphasize_rare_classes(self):
        from src.models.tail_prototype import tail_class_weights

        weights = tail_class_weights(torch.tensor([100, 25, 1, 0]), gamma=0.5, max_weight=5.0)

        self.assertGreater(weights[2].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[0].item())
        self.assertEqual(weights[3].item(), 0.0)
        self.assertAlmostEqual(weights[:3].mean().item(), 1.0, places=6)

    def test_tail_class_weights_gamma_zero_keeps_present_classes_unweighted(self):
        from src.models.tail_prototype import tail_class_weights

        weights = tail_class_weights(torch.tensor([100, 25, 1, 0]), gamma=0.0)

        self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0, 0.0])

    def test_apply_tail_class_weights_preserves_missing_class_mask(self):
        from src.models.tail_prototype import apply_tail_class_weights

        logits = torch.tensor([[2.0, torch.finfo(torch.float32).min]])
        weights = torch.tensor([1.5, 0.0])

        weighted = apply_tail_class_weights(logits, weights)

        self.assertEqual(weighted[0, 0].item(), 3.0)
        self.assertEqual(weighted[0, 1].item(), torch.finfo(torch.float32).min)

    def test_entropy_confidence_gate_boosts_uncertain_samples(self):
        from src.eval_tail_cache import confidence_gate

        confident = torch.tensor([[10.0, 0.0, -1.0]])
        uncertain = torch.tensor([[0.1, 0.0, -0.1]])
        gate = confidence_gate(torch.cat([confident, uncertain]), mode="entropy", strength=0.5)

        self.assertEqual(tuple(gate.shape), (2, 1))
        self.assertGreater(gate[1, 0].item(), gate[0, 0].item())
        self.assertLess(gate[0, 0].item(), 1.0)
        self.assertGreater(gate[1, 0].item(), 1.0)
        self.assertGreaterEqual(gate.min().item(), 0.75)
        self.assertLessEqual(gate.max().item(), 1.25)

    def test_margin_confidence_gate_boosts_small_margin_samples(self):
        from src.eval_tail_cache import confidence_gate

        confident = torch.tensor([[10.0, 0.0, -1.0]])
        uncertain = torch.tensor([[1.0, 0.9, -1.0]])
        gate = confidence_gate(torch.cat([confident, uncertain]), mode="margin", strength=1.0)

        self.assertGreater(gate[1, 0].item(), gate[0, 0].item())
        self.assertLess(gate[0, 0].item(), 1.0)
        self.assertGreater(gate[1, 0].item(), 1.0)
        self.assertGreaterEqual(gate.min().item(), 0.5)
        self.assertLessEqual(gate.max().item(), 1.5)

    def test_confidence_gate_strength_zero_matches_baseline(self):
        from src.eval_tail_cache import confidence_gate

        logits = torch.tensor([[10.0, 0.0, -1.0], [0.1, 0.0, -0.1]])

        self.assertTrue(torch.equal(confidence_gate(logits, mode="entropy", strength=0.0), torch.ones(2, 1)))
        self.assertTrue(torch.equal(confidence_gate(logits, mode="margin", strength=0.0), torch.ones(2, 1)))

    def test_train_flyp_one_epoch_adds_tail_prototype_auxiliary_loss(self):
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
        args = SimpleNamespace(
            device="cpu",
            model="ViT-B-16",
            max_train_batches=1,
            tail_proto_weight=0.5,
            tail_proto_scale=10.0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
                tail_prototypes=torch.eye(2),
                tail_class_counts=torch.tensor([1, 1]),
            )

        self.assertGreater(stats.tail_loss, 0.0)
        self.assertAlmostEqual(stats.loss, stats.clip_loss + stats.tail_loss, places=6)
        self.assertAlmostEqual(stats.tail_to_clip_ratio, stats.tail_loss / stats.clip_loss, places=6)

    def test_train_flyp_one_epoch_adds_tail_prototype_distillation_loss(self):
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
        classification_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classification_head.weight.copy_(torch.eye(2))
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(
            device="cpu",
            model="ViT-B-16",
            max_train_batches=1,
            tail_proto_weight=0.5,
            tail_proto_scale=10.0,
            tail_proto_objective="distill",
            tail_proto_temperature=1.0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
                tail_prototypes=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                tail_class_counts=torch.tensor([1, 1]),
                tail_zeroshot_classifier=classification_head,
            )

        self.assertEqual(stats.tail_proto_objective, "distill")
        self.assertGreater(stats.tail_loss, 0.0)
        self.assertAlmostEqual(stats.loss, stats.clip_loss + stats.tail_loss, places=6)

    def test_train_flyp_one_epoch_adds_fixed_tail_teacher_distillation_loss(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))

            def forward(self, images, text=None):
                del text
                features = images @ self.weight
                return features, features, torch.ones(())

        class FrozenTeacher(torch.nn.Module):
            def forward(self, images):
                return torch.flip(images, dims=[1])

        student_head = torch.nn.Linear(2, 2, bias=False)
        teacher_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            student_head.weight.copy_(torch.eye(2))
            teacher_head.weight.copy_(torch.eye(2))
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(
            device="cpu",
            model="ViT-B-16",
            max_train_batches=1,
            tail_proto_weight=0.5,
            tail_proto_scale=10.0,
            tail_proto_objective="fixed_distill",
            tail_proto_temperature=1.0,
        )
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
                tail_prototypes=torch.eye(2),
                tail_class_counts=torch.tensor([1, 1]),
                tail_zeroshot_classifier=student_head,
                tail_teacher_model=FrozenTeacher(),
                tail_teacher_classifier=teacher_head,
            )

        self.assertEqual(stats.tail_proto_objective, "fixed_distill")
        self.assertGreater(stats.tail_loss, 0.0)

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


class FlypEvalCacheTest(unittest.TestCase):
    def test_eval_reuses_zeroshot_classifier_when_state_unchanged(self):
        from src.models.flyp import eval_flyp_single_dataset

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2, bias=False)
                self.train_preprocess = object()
                self.val_preprocess = object()

        dataset = object()
        args = SimpleNamespace(device='cpu')
        encoder = TinyEncoder()
        classifier = torch.nn.Linear(2, 2, bias=False)

        with patch('src.models.flyp.get_zeroshot_classifier', return_value=classifier) as mocked_build, \
                patch('src.models.flyp.eval_coop_single_dataset', return_value={'top1': 0.1}) as mocked_eval:
            eval_flyp_single_dataset(encoder, dataset, args)
            eval_flyp_single_dataset(encoder, dataset, args)

        self.assertEqual(mocked_build.call_count, 1)
        self.assertEqual(mocked_eval.call_count, 2)

    def test_eval_rebuilds_zeroshot_classifier_when_state_changes(self):
        from src.models.flyp import eval_flyp_single_dataset

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2, bias=False)
                self.train_preprocess = object()
                self.val_preprocess = object()

        dataset = object()
        args = SimpleNamespace(device='cpu')
        encoder = TinyEncoder()
        classifier = torch.nn.Linear(2, 2, bias=False)

        with patch('src.models.flyp.get_zeroshot_classifier', return_value=classifier) as mocked_build, \
                patch('src.models.flyp.eval_coop_single_dataset', return_value={'top1': 0.1}):
            eval_flyp_single_dataset(encoder, dataset, args)
            with torch.no_grad():
                encoder.model.weight.add_(1.0)
            eval_flyp_single_dataset(encoder, dataset, args)

        self.assertEqual(mocked_build.call_count, 2)


class FlypDrmTest(unittest.TestCase):
    def test_drm_loss_uses_mean_squared_delta_not_parameter_sum(self):
        from src.models.flyp import compute_drm_loss

        model = torch.nn.Linear(2, 2, bias=False)
        init_state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
        with torch.no_grad():
            model.weight.fill_(2.0)

        loss = compute_drm_loss(model, init_state, drm_weight=0.5)

        self.assertAlmostEqual(loss.item(), 2.0)

    def test_drm_loss_excludes_logit_scale(self):
        from src.models.flyp import compute_drm_loss

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.logit_scale = torch.nn.Parameter(torch.tensor(10.0))

        model = TinyModel()
        init_state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}

        loss = compute_drm_loss(model, init_state, drm_weight=1.0)

        self.assertAlmostEqual(loss.item(), 1.0)

    def test_drm_warmup_scales_effective_weight_and_ratio(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))

            def forward(self, images, text):
                del text
                features = images @ self.weight
                return features, features, torch.ones(())

        model = TinyModel()
        init_state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1, drm_warmup_epochs=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=2,
                init_state_dict=init_state,
                drm_weight=2.0,
            )

        self.assertAlmostEqual(stats.drm_effective_weight, 1.0)
        self.assertGreater(stats.drm_to_clip_ratio, 0.0)

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

    def test_train_flyp_one_epoch_scales_clip_plus_drm_loss(self):
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

        class CapturingScaledLoss:
            def __init__(self, loss):
                self.loss = loss

            def backward(self):
                self.loss.backward()

        class CapturingScaler:
            def __init__(self):
                self.scaled_loss = None

            def scale(self, loss):
                self.scaled_loss = CapturingScaledLoss(loss)
                return self.scaled_loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        model = TinyModel()
        init_state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1, use_amp=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        scaler = CapturingScaler()

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            stats = train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
                init_state_dict=init_state,
                drm_weight=0.25,
                scaler=scaler,
            )

        self.assertGreater(stats.drm_loss, 0.0)
        self.assertAlmostEqual(scaler.scaled_loss.loss.item(), stats.loss, places=6)

    def test_train_flyp_one_epoch_lets_grad_scaler_skip_nonfinite_amp_step(self):
        from src.models.flyp import train_flyp_one_epoch

        class TinyTokenizer:
            def __call__(self, captions):
                return torch.zeros(len(captions), 4, dtype=torch.long)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))
                self.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))

            def forward(self, images, text):
                features = images @ self.weight
                return features, features, torch.ones(())

        class FakeScaledLoss:
            def __init__(self, loss):
                self.loss = loss

            def backward(self):
                self.loss.backward()

        class FakeSkippingScaler:
            def __init__(self):
                self.calls = []
                self.scale_value = 8.0

            def scale(self, loss):
                self.calls.append("scale")
                return FakeScaledLoss(loss)

            def unscale_(self, optimizer):
                self.calls.append("unscale")

            def get_scale(self):
                return self.scale_value

            def step(self, optimizer):
                self.calls.append("step")

            def update(self):
                self.calls.append("update")
                self.scale_value = 4.0

        class FakeScheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

        model = TinyModel()
        batch = {"images": torch.eye(2), "labels": torch.tensor([0, 1])}
        args = SimpleNamespace(device="cpu", model="ViT-B-16", max_train_batches=1, use_amp=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = FakeSkippingScaler()
        scheduler = FakeScheduler()

        with patch("src.models.flyp.open_clip.get_tokenizer", return_value=TinyTokenizer()):
            train_flyp_one_epoch(
                model,
                [batch],
                optimizer,
                args,
                ["frog", "deer"],
                [lambda c: f"a photo of {c}."],
                epoch=1,
                scaler=scaler,
                scheduler=scheduler,
            )

        self.assertEqual(scaler.calls, ["scale", "unscale", "step", "update"])
        self.assertEqual(scheduler.steps, 0)

    def test_drm_loss_gradient_matches_regularization_derivative(self):
        from src.models.flyp import compute_drm_loss

        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
        init_state = {"weight": torch.ones_like(model.weight)}

        compute_drm_loss(model, init_state, drm_weight=0.5).backward()

        expected = 2 * 0.5 * (model.weight.detach() - init_state["weight"]) / model.weight.numel()
        self.assertTrue(torch.allclose(model.weight.grad, expected))


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


class FlypFinalEvalDatasetTest(unittest.TestCase):
    def test_tuned_validation_split_is_excluded_from_final_eval(self):
        from src.train_flyp import final_eval_datasets

        args = SimpleNamespace(
            eval_datasets=["IWildCamIDVal", "IWildCamVal", "IWildCamID", "IWildCamOOD"],
            val_dataset="IWildCamVal",
            epochs=20,
            wise_alphas="0.0,0.1",
        )

        self.assertEqual(
            final_eval_datasets(args),
            ["IWildCamIDVal", "IWildCamID", "IWildCamOOD"],
        )

    def test_validation_split_remains_when_not_used_for_tuning(self):
        from src.train_flyp import final_eval_datasets

        args = SimpleNamespace(
            eval_datasets=["IWildCamVal", "IWildCamOOD"],
            val_dataset="IWildCamVal",
            epochs=0,
            wise_alphas=None,
        )

        self.assertEqual(final_eval_datasets(args), ["IWildCamVal", "IWildCamOOD"])


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


class FlypAmpConfigTest(unittest.TestCase):
    def test_flyp_amp_requires_cuda_and_amp_precision(self):
        from src.train_flyp import should_use_flyp_amp

        self.assertTrue(should_use_flyp_amp(SimpleNamespace(device="cuda", maple_precision="amp")))
        self.assertFalse(should_use_flyp_amp(SimpleNamespace(device="cuda", maple_precision="fp32")))
        self.assertFalse(should_use_flyp_amp(SimpleNamespace(device="cpu", maple_precision="amp")))


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


class FlypAnchorStateTest(unittest.TestCase):
    def test_initialize_flyp_model_anchors_to_loaded_state(self):
        import src.train_flyp as train_flyp

        class LoadedEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[9.0, 0.0], [0.0, 9.0]]))
                self.train_preprocess = object()

            def to(self, *args, **kwargs):
                return self

        class FreshEncoder(torch.nn.Module):
            def __init__(self, args, keep_lang=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
                self.train_preprocess = object()

            def to(self, *args, **kwargs):
                return self

            @classmethod
            def load(cls, filename):
                return LoadedEncoder()

        args = SimpleNamespace(model='ViT-B-16', device='cpu', load='checkpoint.pt')

        with patch.object(train_flyp, 'CLIPEncoder', FreshEncoder), \
                patch.object(train_flyp, 'maybe_data_parallel', side_effect=lambda model, args: model):
            model, anchor_state = train_flyp.initialize_flyp_model(args)

        self.assertEqual(model.weight.detach().tolist(), [[9.0, 0.0], [0.0, 9.0]])
        self.assertEqual(anchor_state['weight'].tolist(), [[9.0, 0.0], [0.0, 9.0]])

    def test_initialize_flyp_model_without_load_anchors_to_fresh_state(self):
        import src.train_flyp as train_flyp

        class FreshEncoder(torch.nn.Module):
            def __init__(self, args, keep_lang=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
                self.train_preprocess = object()

            def to(self, *args, **kwargs):
                return self

        args = SimpleNamespace(model='ViT-B-16', device='cpu', load=None)

        with patch.object(train_flyp, 'CLIPEncoder', FreshEncoder), \
                patch.object(train_flyp, 'maybe_data_parallel', side_effect=lambda model, args: model):
            model, anchor_state = train_flyp.initialize_flyp_model(args)

        self.assertEqual(model.weight.detach().tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(anchor_state['weight'].tolist(), [[1.0, 0.0], [0.0, 1.0]])


class FlypCloneStateDictTest(unittest.TestCase):
    def test_clone_creates_independent_copy(self):
        from src.train_flyp import clone_state_dict
        model = torch.nn.Linear(2, 2, bias=False)
        sd = clone_state_dict(model)
        with torch.no_grad():
            model.weight.add_(100.0)
        self.assertFalse(torch.equal(sd["weight"], model.weight))


class FlypMainAmpTest(unittest.TestCase):
    def _run_main_with_precision(self, maple_precision):
        import src.train_flyp as train_flyp

        class TinyClipEncoder(torch.nn.Module):
            def __init__(self, args, keep_lang=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2))
                self.train_preprocess = object()

            def to(self, *args, **kwargs):
                return self

            def save(self, path):
                pass

        class TinyTrainDataset:
            classnames = ["frog", "deer"]
            train_loader = [{"images": torch.eye(2), "labels": torch.tensor([0, 1])}]

            def __init__(self, preprocess, location, batch_size, num_workers):
                pass

        captured = {}

        def fake_train_flyp_one_epoch(*args, **kwargs):
            captured["scaler"] = kwargs.get("scaler")
            return SimpleNamespace(epoch=1, loss=0.0, lr=0.0)

        args = SimpleNamespace(
            seed=0,
            template="iwildcam_template",
            model="ViT-B-16",
            device="cuda",
            train_dataset="TinyTrainDataset",
            data_location="./data",
            batch_size=2,
            workers=0,
            load=None,
            lr=1e-5,
            wd=0.2,
            max_train_batches=None,
            epochs=1,
            save=None,
            best_checkpoint=None,
            lr_scheduler="cosine",
            warmup_length=0,
            maple_precision=maple_precision,
            wandb=False,
            val_dataset=None,
            wise_alphas=None,
            wise_eval_alpha=None,
            no_load_best_for_eval=False,
            eval_datasets=None,
            drm_weight=1.0,
        )

        with patch.object(train_flyp, "CLIPEncoder", TinyClipEncoder), \
                patch.object(train_flyp.datasets, "TinyTrainDataset", TinyTrainDataset, create=True), \
                patch.object(train_flyp, "maybe_data_parallel", side_effect=lambda model, args: model), \
                patch.object(train_flyp, "build_step_lr_scheduler", return_value=None), \
                patch.object(train_flyp, "init_wandb", return_value=None), \
                patch.object(train_flyp, "train_flyp_one_epoch", side_effect=fake_train_flyp_one_epoch), \
                patch.object(train_flyp, "parse_wise_alphas", return_value=[]), \
                patch("src.train_flyp.torch.cuda.is_available", return_value=True):
            train_flyp.main(args)

        return args, captured

    def test_main_passes_grad_scaler_when_amp_precision_requested(self):
        args, captured = self._run_main_with_precision("amp")

        self.assertTrue(args.use_amp)
        self.assertIsNotNone(captured["scaler"])

    def test_main_disables_grad_scaler_when_fp32_precision_requested(self):
        args, captured = self._run_main_with_precision("fp32")

        self.assertFalse(args.use_amp)
        self.assertIsNone(captured["scaler"])


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
