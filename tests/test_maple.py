import tempfile
import unittest
from types import SimpleNamespace

import torch


class DummyVisual(torch.nn.Module):
    input_resolution = 224

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 768, kernel_size=32, stride=32, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.zeros(768))
        self.positional_embedding = torch.nn.Parameter(torch.zeros(50, 768))
        self.ln_pre = torch.nn.LayerNorm(768)
        self.transformer = torch.nn.Identity()
        self.ln_post = torch.nn.LayerNorm(768)
        self.proj = torch.nn.Parameter(torch.zeros(768, 512))


class DummyOpenAIClip(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(49408, 512)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(77, 512))
        self.transformer = torch.nn.Identity()
        self.ln_final = torch.nn.LayerNorm(512)
        self.text_projection = torch.nn.Parameter(torch.eye(512))
        self.visual = DummyVisual()
        self.logit_scale = torch.nn.Parameter(torch.ones([]))


class MaPLeModuleTest(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            model="ViT-B/32",
            n_ctx=2,
            ctx_init="",
            maple_vision_n_ctx=3,
            device="cpu",
        )

    def test_prompt_learner_builds_text_and_visual_prompts(self):
        from src.models.maple import MultiModalPromptLearner

        learner = MultiModalPromptLearner(self.make_args(), ["frog", "deer"], DummyOpenAIClip())

        prompts = learner()

        self.assertEqual(prompts.shape, (2, 77, 512))
        self.assertEqual(learner.visual_ctx.shape, (3, 768))
        self.assertEqual(learner.ctx.dtype, torch.float32)
        self.assertEqual(learner.visual_ctx.dtype, torch.float32)

    def test_visual_prompted_vit_appends_prompt_tokens_and_returns_features(self):
        from src.models.maple import ShallowVisualPromptedViT

        args = self.make_args()
        clip_model = DummyOpenAIClip()
        wrapper = ShallowVisualPromptedViT(clip_model.visual)
        visual_ctx = torch.zeros(args.maple_vision_n_ctx, 768)

        features = wrapper(torch.zeros(2, 3, 224, 224), visual_ctx)

        self.assertEqual(features.shape, (2, 512))

    def test_custom_maple_clip_freezes_base_model_and_trains_prompts_only(self):
        from src.models.maple import CustomMaPLeCLIP

        model = CustomMaPLeCLIP(self.make_args(), ["frog", "deer"], DummyOpenAIClip())

        trainable = [name for name, param in model.named_parameters() if param.requires_grad]

        self.assertEqual(trainable, ["prompt_learner.ctx", "prompt_learner.visual_ctx"])

        logits = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(logits.shape, (2, 2))

    def test_maple_checkpoint_round_trips_prompt_learner(self):
        from src.models.maple import CustomMaPLeCLIP, load_maple_prompt_learner, save_maple_prompt_learner

        args = self.make_args()
        model = CustomMaPLeCLIP(args, ["frog", "deer"], DummyOpenAIClip())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/maple.pt"
            save_maple_prompt_learner(model, path, args, ["frog", "deer"])
            with torch.no_grad():
                model.prompt_learner.ctx.add_(1.0)
            load_maple_prompt_learner(model, path, "cpu")
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["method"], "maple_shallow")

    def test_maple_guard_rejects_non_vit_b32(self):
        from src.models.maple import ensure_openai_vit_b32_for_maple

        with self.assertRaisesRegex(ValueError, "ViT-B/32 only"):
            ensure_openai_vit_b32_for_maple(DummyOpenAIClip(), "RN50")


if __name__ == "__main__":
    unittest.main()
