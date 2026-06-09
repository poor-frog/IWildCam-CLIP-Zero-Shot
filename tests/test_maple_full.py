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


class DummyAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(768, 12)


class DummyTransformerWithBlocks(torch.nn.Module):
    def __init__(self, block_count):
        super().__init__()
        self.resblocks = torch.nn.ModuleList([DummyAttentionBlock() for _ in range(block_count)])


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


class DummyLoRAMaPLeVisual(DummyMaPLeVisual):
    def __init__(self, block_count=12):
        super().__init__()
        self.transformer = DummyTransformerWithBlocks(block_count)


class DummyLoRAMaPLeClip(DummyMaPLeClip):
    def __init__(self, block_count=12):
        super().__init__()
        self.visual = DummyLoRAMaPLeVisual(block_count=block_count)


class FullMaPLeModuleTest(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            model="ViT-B/32",
            n_ctx=2,
            ctx_init="",
            maple_prompt_depth=4,
            device="cpu",
            maple_lora_rank=0,
            maple_lora_alpha=None,
            maple_lora_dropout=0.0,
            maple_lora_target="vision_out_proj",
            maple_lora_layers="last6",
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

    def test_maple_clip_load_converts_xla_model_to_float_before_device_transfer(self):
        from src.models import maple_clip
        import src.models.maple_clip.clip as maple_clip_loader

        calls = []

        class DummyLoadedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = SimpleNamespace(input_resolution=224)

            def float(self):
                calls.append("float")
                return self

            def to(self, device):
                calls.append(f"to:{device}")
                return self

        with patch.object(maple_clip_loader, "_download", return_value="/tmp/fake.pt"), \
             patch.object(torch.jit, "load", side_effect=RuntimeError("not jit")), \
             patch.object(torch, "load", return_value={"fake": torch.zeros(1)}), \
             patch.object(maple_clip_loader, "build_model", return_value=DummyLoadedModel()), \
             patch.object(maple_clip_loader, "_transform", return_value="preprocess"):
            maple_clip.load("ViT-B/32", device="xla:0", jit=False)

        self.assertEqual(calls, ["float", "to:xla:0"])

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

    def test_custom_full_maple_clip_injects_vision_out_proj_lora_last6_when_enabled(self):
        from src.models.maple_full import CustomFullMaPLeCLIP
        from src.models.maple_lora import collect_lora_state_dict, has_lora_weight_parametrization

        args = self.make_args()
        args.maple_lora_rank = 4
        args.maple_lora_alpha = 8
        clip_model = DummyLoRAMaPLeClip(block_count=12)

        model = CustomFullMaPLeCLIP(args, ["frog", "deer"], clip_model)

        resblocks = model.image_encoder.transformer.resblocks
        self.assertFalse(has_lora_weight_parametrization(resblocks[5].attn.out_proj))
        for block_index in range(6, 12):
            self.assertTrue(has_lora_weight_parametrization(resblocks[block_index].attn.out_proj))

        lora_state = collect_lora_state_dict(model)
        self.assertEqual(len(lora_state), 12)
        self.assertTrue(all(name.startswith("image_encoder.transformer.resblocks.") for name in lora_state))
        self.assertTrue(all("text_encoder" not in name for name in lora_state))

        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertIn("image_encoder.transformer.resblocks.6.attn.out_proj.parametrizations.weight.0.lora_down.weight", trainable)
        self.assertIn("image_encoder.transformer.resblocks.11.attn.out_proj.parametrizations.weight.0.lora_up.weight", trainable)
        self.assertNotIn("image_encoder.transformer.resblocks.5.attn.out_proj.weight", trainable)

    def test_lora_rank_zero_leaves_vision_out_proj_modules_unchanged(self):
        from src.models.maple_full import CustomFullMaPLeCLIP
        from src.models.maple_lora import collect_lora_state_dict, has_lora_weight_parametrization

        model = CustomFullMaPLeCLIP(self.make_args(), ["frog", "deer"], DummyLoRAMaPLeClip(block_count=12))

        for block in model.image_encoder.transformer.resblocks:
            self.assertFalse(has_lora_weight_parametrization(block.attn.out_proj))
        self.assertEqual(collect_lora_state_dict(model), {})

    def test_full_maple_checkpoint_round_trips_optional_lora_state(self):
        import tempfile
        from pathlib import Path

        from src.models.maple_full import CustomFullMaPLeCLIP, load_full_maple_prompt_learner, save_full_maple_prompt_learner
        from src.models.maple_lora import collect_lora_state_dict

        args = self.make_args()
        args.maple_lora_rank = 4
        args.maple_lora_alpha = 8
        source = CustomFullMaPLeCLIP(args, ["frog", "deer"], DummyLoRAMaPLeClip(block_count=12))
        for param in source.image_encoder.transformer.resblocks[6].attn.out_proj.parameters():
            if param.requires_grad:
                torch.nn.init.constant_(param, 0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "maple_lora.pt"
            save_full_maple_prompt_learner(source, str(path), args, ["frog", "deer"])

            target = CustomFullMaPLeCLIP(args, ["frog", "deer"], DummyLoRAMaPLeClip(block_count=12))
            load_full_maple_prompt_learner(target, str(path), "cpu")

        self.assertEqual(collect_lora_state_dict(source).keys(), collect_lora_state_dict(target).keys())
        for name, tensor in collect_lora_state_dict(source).items():
            self.assertTrue(torch.equal(tensor, collect_lora_state_dict(target)[name]))

    def test_collect_lora_state_dict_unwraps_data_parallel_prefix(self):
        from src.models.maple_full import CustomFullMaPLeCLIP
        from src.models.maple_lora import collect_lora_state_dict

        args = self.make_args()
        args.maple_lora_rank = 4
        model = torch.nn.DataParallel(CustomFullMaPLeCLIP(args, ["frog", "deer"], DummyLoRAMaPLeClip(block_count=12)))

        lora_state = collect_lora_state_dict(model)

        self.assertTrue(lora_state)
        self.assertTrue(all(not name.startswith("module.") for name in lora_state))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS device required for cross-device LoRA regression")
    def test_lora_weight_parametrization_matches_original_weight_device(self):
        from src.models.maple_lora import LoRAWeightParametrization

        parametrization = LoRAWeightParametrization(out_features=768, in_features=768, rank=8, alpha=16)
        original_weight = torch.zeros(768, 768, device="mps")

        parametrized_weight = parametrization(original_weight)

        self.assertEqual(parametrized_weight.device.type, "mps")

    def test_train_maple_lora_configures_separate_entrypoint_defaults(self):
        from src.train_maple_lora import configure_maple_lora_args

        args = self.make_args()
        args.save = None
        args.wandb_run_name = None

        configured = configure_maple_lora_args(args)

        self.assertEqual(configured.maple_lora_rank, 8)
        self.assertEqual(configured.maple_lora_alpha, 16)
        self.assertEqual(configured.maple_lora_layers, "last6")
        self.assertEqual(configured.training_method, "maple_lora")
        self.assertEqual(configured.wandb_run_name, "maple-lora-vit-b32-r8-last6")
        self.assertEqual(configured.save, "./checkpoints/maple_lora_r8_last6.pt")

    def test_train_maple_lora_preserves_user_rank_and_sets_matching_alpha(self):
        from src.train_maple_lora import configure_maple_lora_args

        args = self.make_args()
        args.maple_lora_rank = 4
        args.maple_lora_alpha = None
        args.save = "./custom.pt"
        args.wandb_run_name = "custom-run"

        configured = configure_maple_lora_args(args)

        self.assertEqual(configured.maple_lora_rank, 4)
        self.assertEqual(configured.maple_lora_alpha, 8)
        self.assertEqual(configured.save, "./custom.pt")
        self.assertEqual(configured.wandb_run_name, "custom-run")

    def test_prompt_dtype_uses_bfloat16_for_xla_reference_tensor(self):
        from src.device import prompt_tensor_dtype

        reference = torch.zeros(1, dtype=torch.float32)

        with patch("src.device.is_xla_device", return_value=True):
            self.assertEqual(prompt_tensor_dtype(reference), torch.bfloat16)

        with patch("src.device.is_xla_device", return_value=False):
            self.assertEqual(prompt_tensor_dtype(reference), torch.float32)


if __name__ == "__main__":
    unittest.main()
