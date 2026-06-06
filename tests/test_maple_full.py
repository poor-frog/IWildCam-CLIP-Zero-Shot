import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class DummyMaPLeVisual(torch.nn.Module):
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


class DummyMaPLeClip(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(49408, 512)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(77, 512))
        self.transformer = torch.nn.Identity()
        self.ln_final = torch.nn.LayerNorm(512)
        self.text_projection = torch.nn.Parameter(torch.eye(512))
        self.visual = DummyMaPLeVisual()
        self.logit_scale = torch.nn.Parameter(torch.ones([]))


class FullMaPLeModuleTest(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            model="ViT-B/32",
            n_ctx=2,
            ctx_init="",
            maple_prompt_depth=4,
            device="cpu",
        )

    def test_isolated_maple_clip_backend_import_does_not_replace_existing_clip_package(self):
        import clip as existing_clip
        from src.models import maple_clip

        self.assertIs(sys.modules["clip"], existing_clip)
        self.assertTrue(hasattr(maple_clip, "build_model"))
        self.assertTrue(hasattr(maple_clip, "tokenize"))
        self.assertNotEqual(maple_clip.__name__, existing_clip.__name__)

    def test_maple_clip_load_passes_design_details_to_build_model(self):
        from src.models import maple_clip
        import src.models.maple_clip.clip as maple_clip_loader

        design_details = {
            "trainer": "MaPLe",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": 2,
        }
        calls = []

        class DummyLoadedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = SimpleNamespace(input_resolution=224)

            def to(self, device):
                return self

        def fake_build_model(state_dict, received_design_details):
            calls.append(received_design_details)
            return DummyLoadedModel()

        with patch.object(maple_clip_loader, "_download", return_value="/tmp/fake.pt"), \
             patch.object(torch.jit, "load", side_effect=RuntimeError("not jit")), \
             patch.object(torch, "load", return_value={"fake": torch.zeros(1)}), \
             patch.object(maple_clip_loader, "build_model", side_effect=fake_build_model), \
             patch.object(maple_clip_loader, "_transform", return_value="preprocess"):
            model, preprocess = maple_clip.load(
                "ViT-B/32",
                device="cpu",
                jit=False,
                design_details=design_details,
            )

        self.assertIsInstance(model, DummyLoadedModel)
        self.assertEqual(preprocess, "preprocess")
        self.assertEqual(calls, [design_details])

    def test_prompt_learner_returns_coupled_shallow_and_deep_prompts(self):
        from src.models.maple_full import FullMaPLePromptLearner

        learner = FullMaPLePromptLearner(self.make_args(), ["frog", "deer"], DummyMaPLeClip())

        prompts, shared_ctx, deep_text_prompts, deep_vision_prompts = learner()

        self.assertEqual(prompts.shape, (2, 77, 512))
        self.assertEqual(shared_ctx.shape, (2, 768))
        self.assertEqual(len(deep_text_prompts), 3)
        self.assertEqual(len(deep_vision_prompts), 3)
        self.assertEqual(deep_text_prompts[0].shape, (2, 512))
        self.assertEqual(deep_vision_prompts[0].shape, (2, 768))
        self.assertIsInstance(learner.proj, torch.nn.Linear)
        self.assertIsInstance(learner.compound_prompts_text, torch.nn.ParameterList)
        self.assertIsInstance(learner.compound_prompt_projections, torch.nn.ModuleList)

    def test_custom_full_maple_clip_trains_prompt_learner_only(self):
        from src.models.maple_full import CustomFullMaPLeCLIP

        model = CustomFullMaPLeCLIP(self.make_args(), ["frog", "deer"], DummyMaPLeClip())

        trainable = [name for name, param in model.named_parameters() if param.requires_grad]

        self.assertEqual(
            trainable,
            [
                "prompt_learner.ctx",
                "prompt_learner.proj.weight",
                "prompt_learner.proj.bias",
                "prompt_learner.compound_prompts_text.0",
                "prompt_learner.compound_prompts_text.1",
                "prompt_learner.compound_prompts_text.2",
                "prompt_learner.compound_prompt_projections.0.weight",
                "prompt_learner.compound_prompt_projections.0.bias",
                "prompt_learner.compound_prompt_projections.1.weight",
                "prompt_learner.compound_prompt_projections.1.bias",
                "prompt_learner.compound_prompt_projections.2.weight",
                "prompt_learner.compound_prompt_projections.2.bias",
            ],
        )


if __name__ == "__main__":
    unittest.main()
