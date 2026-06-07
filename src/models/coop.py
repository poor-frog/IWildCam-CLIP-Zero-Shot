import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip.clip as clip
from src.datasets.dataloader import maybe_dictionarize


OPENAI_CLIP_MODELS = {"RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"}


def should_use_data_parallel(device, cuda_device_count, disabled=False):
    return device == "cuda" and cuda_device_count > 1 and not disabled


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_prompt_learner(model):
    return unwrap_model(model).prompt_learner


def maybe_data_parallel(model, args):
    disabled = getattr(args, "no_data_parallel", False)
    if should_use_data_parallel(args.device, torch.cuda.device_count(), disabled):
        return torch.nn.DataParallel(model)
    return model


def ensure_openai_clip_for_coop(clip_model, model_name):
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
    if model_name not in OPENAI_CLIP_MODELS or missing:
        raise ValueError(
            "CoOp Phase 1 is OpenAI CLIP-only. Use --model=RN50 or --model=ViT-B/32. "
            f"Model {model_name!r} is missing: {missing}"
        )


class TextEncoder(torch.nn.Module):
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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(torch.nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init.replace("_", " ").strip()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        token_device = clip_model.token_embedding.weight.device

        if ctx_init:
            tokenized_ctx = clip.tokenize(ctx_init).to(token_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_ctx).float()
            n_ctx = tokenized_ctx.argmax(dim=-1).item() - 1
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        elif args.csc:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=torch.float32)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = torch.nn.Parameter(ctx_vectors)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = args.class_token_position

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(clip.tokenize(name)[0].nonzero()) - 2 for name in classnames]
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

        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)

        if self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prompts.append(torch.cat([
                    prefix[i:i + 1],
                    suffix[i:i + 1, :name_len],
                    ctx[i:i + 1],
                    suffix[i:i + 1, name_len:],
                ], dim=1))
            return torch.cat(prompts, dim=0)

        if self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prompts.append(torch.cat([
                    prefix[i:i + 1],
                    ctx[i:i + 1, :half_n_ctx],
                    suffix[i:i + 1, :name_len],
                    ctx[i:i + 1, half_n_ctx:],
                    suffix[i:i + 1, name_len:],
                ], dim=1))
            return torch.cat(prompts, dim=0)

        raise ValueError(f"Unsupported class token position: {self.class_token_position}")


class CustomCLIP(torch.nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        ensure_openai_clip_for_coop(clip_model, args.model if hasattr(args, "model") else "ViT-B/32")
        for param in clip_model.parameters():
            param.requires_grad = False
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner().type(self.dtype)
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()


@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float


def save_prompt_learner(model, path, args, classnames):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    prompt_learner = get_prompt_learner(model)
    torch.save({
        "prompt_learner": prompt_learner.state_dict(),
        "args": vars(args),
        "classnames": classnames,
    }, path)


def load_prompt_learner(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    get_prompt_learner(model).load_state_dict(checkpoint["prompt_learner"])


def train_one_epoch(model, dataloader, optimizer, args, epoch, wandb=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    max_batches = args.max_train_batches

    for batch_index, data in enumerate(tqdm(dataloader, desc=f"CoOp train epoch {epoch}")):
        if max_batches is not None and batch_index >= max_batches:
            break

        data = maybe_dictionarize(data)
        images = data["images"].to(args.device)
        labels = data["labels"].to(args.device)

        logits = model(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"CoOp produced non-finite logits at epoch {epoch}, batch {batch_index}")
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"CoOp produced non-finite loss at epoch {epoch}, batch {batch_index}")

        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"CoOp produced non-finite gradient for {name} at epoch {epoch}, batch {batch_index}")
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


def eval_coop_single_dataset(model, dataset, args, desc="CoOp eval"):
    model.eval()
    loader = dataset.test_loader
    correct = 0
    seen = 0
    all_labels = []
    all_preds = []
    all_metadata = []

    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(loader, desc=desc)):
            if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                break
            data = maybe_dictionarize(data)
            images = data["images"].to(args.device)
            labels = data["labels"].to(args.device)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.shape[0]
            if hasattr(dataset, "post_loop_metrics"):
                all_labels.append(labels.cpu())
                all_preds.append(logits.cpu())
                all_metadata.extend(data["metadata"])

    metrics = {"top1": correct / seen}
    if hasattr(dataset, "post_loop_metrics") and all_labels:
        wilds_metrics = dataset.post_loop_metrics(torch.cat(all_labels), torch.cat(all_preds), all_metadata, args)
        metrics.update(wilds_metrics)
        if "acc" in metrics:
            metrics["top1"] = metrics["acc"]
    return metrics
