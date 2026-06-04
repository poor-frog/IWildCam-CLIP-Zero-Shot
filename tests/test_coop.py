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


if __name__ == "__main__":
    unittest.main()
