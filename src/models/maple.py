import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip.clip as clip
from src.datasets.dataloader import maybe_dictionarize


SUPPORTED_MAPLE_MODELS = {"ViT-B/32"}


def ensure_openai_vit_b32_for_maple(clip_model, model_name):
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
    if model_name not in SUPPORTED_MAPLE_MODELS or missing or visual_missing:
        raise ValueError(
            "MaPLe Phase 2.1a supports OpenAI CLIP ViT-B/32 only. "
            f"Model {model_name!r} is missing CLIP attrs={missing}, visual attrs={visual_missing}"
        )


class MultiModalPromptLearner(torch.nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init.replace("_", " ").strip()
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_ctx_dim = clip_model.visual.conv1.weight.shape[0]
        token_device = clip_model.token_embedding.weight.device

        if ctx_init:
            tokenized_ctx = clip.tokenize(ctx_init).to(token_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_ctx).float()
            n_ctx = tokenized_ctx.argmax(dim=-1).item() - 1
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, text_ctx_dim, dtype=torch.float32)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        visual_ctx = torch.empty(args.maple_vision_n_ctx, vision_ctx_dim, dtype=torch.float32)
        torch.nn.init.normal_(visual_ctx, std=0.02)

        self.ctx = torch.nn.Parameter(ctx_vectors)
        self.visual_ctx = torch.nn.Parameter(visual_ctx)
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(token_device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).float()

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix.to(dtype=ctx.dtype, device=ctx.device)
        suffix = self.token_suffix.to(dtype=ctx.dtype, device=ctx.device)
        return torch.cat([prefix, ctx, suffix], dim=1)


class ShallowVisualPromptedViT(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, image, visual_ctx):
        visual = self.visual
        x = visual.conv1(image.type(visual.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype)
        class_token = class_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        prompt_tokens = visual_ctx.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt_tokens], dim=1)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        if isinstance(x, list):
            x = x[0]
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        return x


class MaPLeTextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class CustomMaPLeCLIP(torch.nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        ensure_openai_vit_b32_for_maple(clip_model, args.model)
        for param in clip_model.parameters():
            param.requires_grad = False
        self.prompt_learner = MultiModalPromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ShallowVisualPromptedViT(clip_model.visual)
        self.text_encoder = MaPLeTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        prompts = self.prompt_learner().type(self.dtype)
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image, self.prompt_learner.visual_ctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()


def get_maple_prompt_learner(model):
    return model.module.prompt_learner if hasattr(model, "module") else model.prompt_learner


def save_maple_prompt_learner(model, path, args, classnames):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    prompt_learner = get_maple_prompt_learner(model)
    torch.save({
        "prompt_learner": prompt_learner.state_dict(),
        "args": vars(args),
        "classnames": classnames,
        "method": "maple_shallow",
    }, path)


def load_maple_prompt_learner(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    get_maple_prompt_learner(model).load_state_dict(checkpoint["prompt_learner"])


@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float


def train_maple_one_epoch(model, dataloader, optimizer, args, epoch, wandb=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    max_batches = args.max_train_batches

    for batch_index, data in enumerate(tqdm(dataloader, desc=f"MaPLe train epoch {epoch}")):
        if max_batches is not None and batch_index >= max_batches:
            break

        data = maybe_dictionarize(data)
        images = data["images"].to(args.device)
        labels = data["labels"].to(args.device)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"MaPLe produced non-finite logits at epoch {epoch}, batch {batch_index}")
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"MaPLe produced non-finite loss at epoch {epoch}, batch {batch_index}")

        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"MaPLe produced non-finite gradient for {name} at epoch {epoch}, batch {batch_index}")
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


def eval_maple_single_dataset(model, dataset, args):
    from src.models.coop import eval_coop_single_dataset
    return eval_coop_single_dataset(model, dataset, args)
