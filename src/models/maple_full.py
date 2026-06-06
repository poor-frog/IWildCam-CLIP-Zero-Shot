import copy
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.dataloader import maybe_dictionarize
from src.models import maple_clip


SUPPORTED_FULL_MAPLE_MODELS = {"ViT-B/32"}


def _get_clones(module, count):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def ensure_openai_vit_b32_for_full_maple(clip_model, model_name):
    required_attrs = (
        "token_embedding",
        "transformer",
        "positional_embedding",
        "ln_final",
        "text_projection",
        "visual",
        "logit_scale",
    )
    missing = [name for name in required_attrs if not hasattr(clip_model, name)]
    visual = getattr(clip_model, "visual", None)
    visual_required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post", "proj")
    visual_missing = [name for name in visual_required if visual is None or not hasattr(visual, name)]
    if model_name not in SUPPORTED_FULL_MAPLE_MODELS or missing or visual_missing:
        raise ValueError(
            "Full MaPLe supports OpenAI CLIP ViT-B/32 only. "
            f"Model {model_name!r} is missing CLIP attrs={missing}, visual attrs={visual_missing}"
        )


def load_maple_clip_to_cpu(model_name="ViT-B/32", prompt_depth=9, n_ctx=2):
    if model_name not in SUPPORTED_FULL_MAPLE_MODELS:
        raise ValueError("Full MaPLe supports OpenAI CLIP ViT-B/32 only.")
    design_details = build_maple_design_details(prompt_depth=prompt_depth, n_ctx=n_ctx)
    model, _ = maple_clip.load(model_name, device="cpu", jit=False, design_details=design_details)
    return model


def build_maple_design_details(prompt_depth, n_ctx):
    if prompt_depth < 1:
        raise ValueError("--maple-prompt-depth must be >= 1 for full MaPLe.")
    return {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": n_ctx,
    }


class FullMaPLePromptLearner(torch.nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init.replace("_", " ").strip()
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_ctx_dim = clip_model.visual.conv1.weight.shape[0]
        token_device = clip_model.token_embedding.weight.device
        prompt_depth = getattr(args, "maple_prompt_depth", 9)
        if prompt_depth < 1:
            raise ValueError("Full MaPLe prompt depth must be >= 1.")

        if ctx_init and n_ctx <= 4:
            tokenized_ctx = maple_clip.tokenize(ctx_init).to(token_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_ctx).float()
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, text_ctx_dim, dtype=torch.float32)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = torch.nn.Parameter(ctx_vectors)
        self.proj = torch.nn.Linear(text_ctx_dim, vision_ctx_dim)
        self.compound_prompts_depth = prompt_depth
        self.compound_prompts_text = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(n_ctx, text_ctx_dim, dtype=torch.float32))
            for _ in range(prompt_depth - 1)
        ])
        for prompt in self.compound_prompts_text:
            torch.nn.init.normal_(prompt, std=0.02)
        self.compound_prompt_projections = _get_clones(
            torch.nn.Linear(text_ctx_dim, vision_ctx_dim),
            prompt_depth - 1,
        )

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([maple_clip.tokenize(prompt) for prompt in prompts]).to(token_device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).float()
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix.to(dtype=ctx.dtype, device=ctx.device)
        suffix = self.token_suffix.to(dtype=ctx.dtype, device=ctx.device)
        prompts = self.construct_prompts(ctx, prefix, suffix)
        shared_ctx = self.proj(self.ctx)
        visual_deep_prompts = [
            layer(self.compound_prompts_text[index])
            for index, layer in enumerate(self.compound_prompt_projections)
        ]
        return prompts, shared_ctx, self.compound_prompts_text, visual_deep_prompts


class FullMaPLeTextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        outputs = self.transformer([x, compound_prompts_deeper_text, 0])
        x = outputs[0] if isinstance(outputs, list) else outputs
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class CustomFullMaPLeCLIP(torch.nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        ensure_openai_vit_b32_for_full_maple(clip_model, args.model)
        for param in clip_model.parameters():
            param.requires_grad = False
        self.prompt_learner = FullMaPLePromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = FullMaPLeTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        prompts, shared_ctx, deep_text_prompts, deep_vision_prompts = self.prompt_learner()
        prompts = prompts.type(self.dtype)
        shared_ctx = shared_ctx.type(self.dtype)
        deep_text_prompts = [prompt.type(self.dtype) for prompt in deep_text_prompts]
        deep_vision_prompts = [prompt.type(self.dtype) for prompt in deep_vision_prompts]
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_text_prompts)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()


def get_full_maple_prompt_learner(model):
    return model.module.prompt_learner if hasattr(model, "module") else model.prompt_learner


def save_full_maple_prompt_learner(model, path, args, classnames):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    prompt_learner = get_full_maple_prompt_learner(model)
    torch.save({
        "prompt_learner": prompt_learner.state_dict(),
        "args": vars(args),
        "classnames": classnames,
        "method": "maple_full",
    }, path)


def load_full_maple_prompt_learner(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    get_full_maple_prompt_learner(model).load_state_dict(checkpoint["prompt_learner"])


@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float


def train_full_maple_one_epoch(model, dataloader, optimizer, args, epoch, wandb=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    max_batches = args.max_train_batches

    for batch_index, data in enumerate(tqdm(dataloader, desc=f"Full MaPLe train epoch {epoch}")):
        if max_batches is not None and batch_index >= max_batches:
            break
        data = maybe_dictionarize(data)
        images = data["images"].to(args.device)
        labels = data["labels"].to(args.device)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Full MaPLe produced non-finite logits at epoch {epoch}, batch {batch_index}")
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Full MaPLe produced non-finite loss at epoch {epoch}, batch {batch_index}")
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"Full MaPLe produced non-finite gradient for {name} at epoch {epoch}, batch {batch_index}")
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size
        if wandb is not None:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/epoch": epoch,
                "train/batch": batch_index,
            })

    return TrainStats(epoch=epoch, loss=total_loss / total_seen, accuracy=total_correct / total_seen)


def eval_full_maple_single_dataset(model, dataset, args):
    from src.models.coop import eval_coop_single_dataset
    return eval_coop_single_dataset(model, dataset, args)
