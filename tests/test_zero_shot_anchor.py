import unittest

import torch


class FakeImageEncoder(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.proj = torch.nn.Linear(3, feature_dim)

    def forward(self, images):
        return self.proj(images)


class FakeClassificationHead(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.linear(features)


class FrozenZeroShotAnchorTest(unittest.TestCase):
    def test_anchor_returns_logits_with_batch_and_class_shape(self):
        from src.models.zeroshot import FrozenZeroShotAnchor

        batch_size = 5
        feature_dim = 4
        num_classes = 3
        anchor = FrozenZeroShotAnchor(
            FakeImageEncoder(feature_dim),
            FakeClassificationHead(feature_dim, num_classes),
        )
        images = torch.randn(batch_size, 3)

        logits = anchor(images)

        self.assertEqual(logits.shape, torch.Size([batch_size, num_classes]))

    def test_anchor_freezes_all_parameters_and_stays_in_eval_mode(self):
        from src.models.zeroshot import FrozenZeroShotAnchor

        encoder = FakeImageEncoder(feature_dim=4)
        head = FakeClassificationHead(feature_dim=4, num_classes=3)
        encoder.train()
        head.train()

        anchor = FrozenZeroShotAnchor(encoder, head)

        self.assertFalse(anchor.training)
        self.assertFalse(anchor.image_encoder.training)
        self.assertFalse(anchor.classification_head.training)
        self.assertTrue(list(anchor.parameters()))
        self.assertTrue(all(not param.requires_grad for param in anchor.parameters()))

    def test_anchor_construction_uses_only_supplied_synthetic_modules(self):
        from src.models.zeroshot import FrozenZeroShotAnchor

        encoder = FakeImageEncoder(feature_dim=2)
        head = FakeClassificationHead(feature_dim=2, num_classes=2)

        anchor = FrozenZeroShotAnchor(encoder, head)
        logits = anchor(torch.ones(1, 3))

        self.assertEqual(logits.shape, torch.Size([1, 2]))


class FakeMaPLeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.seen_shared_ctx = None
        self.seen_deep_prompts = None

    def forward(self, images, shared_ctx, compound_deeper_prompts):
        self.seen_shared_ctx = shared_ctx
        self.seen_deep_prompts = compound_deeper_prompts
        return torch.ones(images.shape[0], 2, device=images.device)


class MaPLeZeroPromptImageEncoderTest(unittest.TestCase):
    def test_adapter_calls_maple_visual_with_frozen_zero_prompts(self):
        from src.models.zeroshot import MaPLeZeroPromptImageEncoder

        visual = FakeMaPLeVisual()
        adapter = MaPLeZeroPromptImageEncoder(visual, n_ctx=3, prompt_depth=4)
        images = torch.zeros(5, 3, 2, 2)

        features = adapter(images)

        self.assertEqual(features.shape, torch.Size([5, 2]))
        self.assertEqual(visual.seen_shared_ctx.shape, torch.Size([3, 4]))
        self.assertEqual(len(visual.seen_deep_prompts), 3)
        self.assertTrue(torch.equal(visual.seen_shared_ctx, torch.zeros_like(visual.seen_shared_ctx)))
        self.assertTrue(all(not param.requires_grad for param in adapter.parameters()))


if __name__ == "__main__":
    unittest.main()
