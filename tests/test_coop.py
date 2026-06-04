import unittest
from types import SimpleNamespace

import torch


class DummyVisual(torch.nn.Module):
    input_resolution = 224

    def forward(self, images):
        return torch.ones(images.shape[0], 4, dtype=images.dtype, device=images.device)


class DummyOpenAIClip(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(49408, 4)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(77, 4))
        self.transformer = torch.nn.Identity()
        self.ln_final = torch.nn.LayerNorm(4)
        self.text_projection = torch.nn.Parameter(torch.eye(4))
        self.visual = DummyVisual()
        self.logit_scale = torch.nn.Parameter(torch.ones([]))


class DummyHalfOpenAIClip(DummyOpenAIClip):
    dtype = torch.float16


class DummyOpenCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = DummyVisual()


class CoOpModuleTest(unittest.TestCase):
    def test_openai_clip_guard_accepts_openai_clip_api(self):
        from src.models.coop import ensure_openai_clip_for_coop

        ensure_openai_clip_for_coop(DummyOpenAIClip(), "ViT-B/32")

    def test_openai_clip_guard_rejects_open_clip_model_names(self):
        from src.models.coop import ensure_openai_clip_for_coop

        with self.assertRaisesRegex(ValueError, "OpenAI CLIP-only"):
            ensure_openai_clip_for_coop(DummyOpenCLIP(), "ViT-B-16")

    def test_prompt_learner_builds_one_prompt_per_class_and_only_ctx_trainable(self):
        from src.models.coop import CustomCLIP

        args = SimpleNamespace(
            n_ctx=2,
            ctx_init="",
            class_token_position="end",
            csc=False,
            device="cpu",
        )
        model = CustomCLIP(args, ["frog", "deer"], DummyOpenAIClip())

        prompts = model.prompt_learner()

        self.assertEqual(prompts.shape, (2, 77, 4))
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertEqual(trainable, ["prompt_learner.ctx"])

    def test_prompt_learner_context_stays_fp32_when_clip_dtype_is_fp16(self):
        from src.models.coop import CustomCLIP

        args = SimpleNamespace(
            n_ctx=2,
            ctx_init="",
            class_token_position="end",
            csc=False,
            device="cpu",
        )
        model = CustomCLIP(args, ["frog", "deer"], DummyHalfOpenAIClip())

        self.assertEqual(model.prompt_learner.ctx.dtype, torch.float32)

    def test_train_one_epoch_fails_fast_on_non_finite_logits_or_loss(self):
        from src.models.coop import train_one_epoch

        class BadModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))

            def forward(self, images):
                return torch.full((images.shape[0], 2), float("nan")) + self.param

        args = SimpleNamespace(device="cpu", max_train_batches=None)
        batch = {"images": torch.zeros(2, 3, 2, 2), "labels": torch.tensor([0, 1])}
        optimizer = torch.optim.SGD(BadModel().parameters(), lr=0.1)

        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            train_one_epoch(BadModel(), [batch], optimizer, args, epoch=1)

    def test_data_parallel_policy_uses_multiple_cuda_devices_only_when_enabled(self):
        from src.models.coop import should_use_data_parallel

        self.assertTrue(should_use_data_parallel("cuda", 2, disabled=False))
        self.assertFalse(should_use_data_parallel("cuda", 1, disabled=False))
        self.assertFalse(should_use_data_parallel("cpu", 2, disabled=False))
        self.assertFalse(should_use_data_parallel("cuda", 2, disabled=True))

    def test_get_prompt_learner_unwraps_data_parallel_like_module(self):
        from src.models.coop import CustomCLIP, get_prompt_learner

        args = SimpleNamespace(
            n_ctx=2,
            ctx_init="",
            class_token_position="end",
            csc=False,
            device="cpu",
        )
        model = CustomCLIP(args, ["frog", "deer"], DummyOpenAIClip())
        wrapped = SimpleNamespace(module=model)

        self.assertIs(get_prompt_learner(wrapped), model.prompt_learner)


if __name__ == "__main__":
    unittest.main()
