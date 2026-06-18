import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

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
    def test_device_selection_keeps_cuda_mps_priority_and_uses_xla_before_cpu(self):
        from src.device import select_default_device

        with patch("src.device.get_xla_device", return_value="xla:0"), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.zeros", return_value=torch.tensor([0.0])), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_default_device(), "cuda")

        with patch("src.device.get_xla_device", return_value="xla:0"), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_default_device(), "mps")

        with patch("src.device.get_xla_device", return_value="xla:0"), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_default_device(), "xla:0")

    def test_device_selection_falls_back_to_cuda_mps_cpu_without_xla(self):
        from src.device import select_default_device

        with patch("src.device.get_xla_device", return_value=None), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.zeros", return_value=torch.tensor([0.0])):
            self.assertEqual(select_default_device(), "cuda")

    def test_device_selection_falls_back_when_cuda_is_advertised_but_unusable(self):
        from src.device import select_default_device

        with patch("src.device.get_xla_device", return_value=None), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.zeros", side_effect=AssertionError("Torch not compiled with CUDA enabled")), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_default_device(), "cpu")

        with patch("src.device.get_xla_device", return_value=None), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_default_device(), "mps")

        with patch("src.device.get_xla_device", return_value=None), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_default_device(), "cpu")

    def test_resolve_device_choice_supports_explicit_xla_and_existing_devices(self):
        from src.device import resolve_device_choice

        with patch("src.device.get_xla_device", return_value="xla:0"):
            self.assertEqual(resolve_device_choice("xla"), "xla:0")

        self.assertEqual(resolve_device_choice("cuda"), "cuda")
        self.assertEqual(resolve_device_choice("mps"), "mps")
        self.assertEqual(resolve_device_choice("cpu"), "cpu")

        with patch("src.device.get_xla_device", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "torch_xla"):
                resolve_device_choice("xla")

    def test_optimizer_step_uses_xla_helper_only_for_xla_devices(self):
        from src.device import optimizer_step

        optimizer = SimpleNamespace(step=unittest.mock.Mock())
        xla_model = SimpleNamespace(
            optimizer_step=unittest.mock.Mock(),
            mark_step=unittest.mock.Mock(),
        )

        with patch("src.device.get_xla_model", return_value=xla_model):
            optimizer_step(optimizer, "xla:0")

        xla_model.optimizer_step.assert_called_once_with(optimizer)
        xla_model.mark_step.assert_called_once_with()
        optimizer.step.assert_not_called()

        optimizer = SimpleNamespace(step=unittest.mock.Mock())
        with patch("src.device.get_xla_model", return_value=xla_model):
            optimizer_step(optimizer, "cuda")
        optimizer.step.assert_called_once_with()

    def test_xla_device_skips_torch_data_parallel_policy(self):
        from src.models.coop import should_use_data_parallel

        self.assertFalse(should_use_data_parallel("xla:0", 8, disabled=False))

    def test_validation_score_prefers_requested_metric_and_falls_back_to_top1(self):
        from src.train_coop import get_validation_score

        self.assertEqual(
            get_validation_score({"top1": 0.7, "F1-macro_all": 0.25}, "F1-macro_all"),
            0.25,
        )
        self.assertEqual(
            get_validation_score({"top1": 0.7}, "F1-macro_all"),
            0.7,
        )

    def test_best_checkpoint_path_uses_save_directory_when_no_explicit_path(self):
        from src.train_coop import resolve_best_checkpoint_path

        args = SimpleNamespace(save="checkpoints", best_checkpoint=None)

        self.assertEqual(
            resolve_best_checkpoint_path(args),
            "checkpoints/coop_prompt_learner_best.pt",
        )

    def test_save_prompt_learner_accepts_filename_without_parent_directory(self):
        from src.models.coop import CustomCLIP, save_prompt_learner

        args = SimpleNamespace(
            model="ViT-B/32",
            n_ctx=2,
            ctx_init="",
            class_token_position="end",
            csc=False,
            device="cpu",
        )
        model = CustomCLIP(args, ["frog", "deer"], DummyOpenAIClip())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = "prompt.pt"
            cwd = __import__("os").getcwd()
            try:
                __import__("os").chdir(tmpdir)
                save_prompt_learner(model, path, args, ["frog", "deer"])
                self.assertTrue(__import__("os").path.exists(path))
            finally:
                __import__("os").chdir(cwd)

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
