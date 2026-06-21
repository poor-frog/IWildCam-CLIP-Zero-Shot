import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.dataloader import maybe_dictionarize
from src.device import optimizer_step
from src.models import maple_clip
from src.models.maple_lora import collect_lora_state_dict, inject_vision_out_proj_lora, load_lora_state_dict, set_lora_gamma


SUPPORTED_FULL_MAPLE_MODELS = {"ViT-B/32", "ViT-B/16"}


def _get_clones(module, count):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def ensure_openai_vit_for_full_maple(clip_model, model_name):
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
            "Full MaPLe supports OpenAI CLIP ViT-B/32 or ViT-B/16. "
            f"Model {model_name!r} is missing CLIP attrs={missing}, visual attrs={visual_missing}"
        )


def load_maple_clip_to_cpu(model_name="ViT-B/32", prompt_depth=9, n_ctx=2):
    if model_name not in SUPPORTED_FULL_MAPLE_MODELS:
        raise ValueError("Full MaPLe supports OpenAI CLIP ViT-B/32 or ViT-B/16.")
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
        ensure_openai_vit_for_full_maple(clip_model, args.model)
        for param in clip_model.parameters():
            param.requires_grad = False
        self.prompt_learner = FullMaPLePromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = copy.deepcopy(clip_model.visual)
        self.text_encoder = FullMaPLeTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        if getattr(args, "maple_lora_target", "vision_out_proj") != "vision_out_proj":
            raise ValueError("Only --maple-lora-target=vision_out_proj is currently supported.")
        inject_vision_out_proj_lora(
            self,
            rank=getattr(args, "maple_lora_rank", 0),
            alpha=getattr(args, "maple_lora_alpha", None),
            dropout=getattr(args, "maple_lora_dropout", 0.0),
            layers=getattr(args, "maple_lora_layers", "last6"),
        )
        set_lora_gamma(self, getattr(args, "maple_lora_gamma", 1.0))

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
    checkpoint = {
        "prompt_learner": prompt_learner.state_dict(),
        "args": vars(args),
        "classnames": classnames,
        "method": "maple_full",
    }
    lora_state = collect_lora_state_dict(model)
    if lora_state:
        checkpoint["lora"] = lora_state
    torch.save(checkpoint, path)


def load_full_maple_prompt_learner(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    get_full_maple_prompt_learner(model).load_state_dict(checkpoint["prompt_learner"])
    if "lora" in checkpoint:
        load_lora_state_dict(model.module if hasattr(model, "module") else model, checkpoint["lora"])


@dataclass
class TrainStats:
    epoch: int
    loss: float
    accuracy: float


@dataclass
class MapleLossResult:
    ce: torch.Tensor
    kl: torch.Tensor
    total: torch.Tensor

    @property
    def loss(self):
        return self.total

    def to_log_dict(self, prefix):
        return {
            f"{prefix}_ce": self.ce.item(),
            f"{prefix}_kl": self.kl.item(),
            f"{prefix}_total": self.total.item(),
        }


def compute_maple_cross_entropy(
    logits,
    labels,
    class_weights=None,
    anchor_logits=None,
    kl_weight=0.0,
    kl_temperature=1.0,
    return_components=False,
):
    if class_weights is None:
        ce_loss = F.cross_entropy(logits, labels)
    else:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.cross_entropy(logits, labels, weight=weights)

    use_kl = anchor_logits is not None and kl_weight != 0.0
    if use_kl:
        if kl_temperature <= 0:
            raise ValueError("kl_temperature must be > 0 when KL is used.")
        kl_loss = F.kl_div(
            F.log_softmax(logits / kl_temperature, dim=1),
            F.softmax(anchor_logits / kl_temperature, dim=1),
            reduction="batchmean",
        )
    else:
        kl_loss = ce_loss.new_zeros(())

    total_loss = ce_loss + kl_weight * kl_temperature * kl_temperature * kl_loss
    if return_components:
        return MapleLossResult(ce=ce_loss, kl=kl_loss, total=total_loss)
    return total_loss


def train_full_maple_one_epoch(
    model,
    dataloader,
    optimizer,
    args,
    epoch,
    wandb=None,
    class_weights=None,
    anchor_model=None,
    kl_weight=0.0,
    kl_temperature=1.0,
    desc="Full MaPLe",
    scheduler=None,
    scaler=None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    max_batches = args.max_train_batches

    for batch_index, data in enumerate(tqdm(dataloader, desc=f"{desc} train epoch {epoch}")):
        if max_batches is not None and batch_index >= max_batches:
            break
        data = maybe_dictionarize(data)
        images = data["images"].to(args.device)
        labels = data["labels"].to(args.device)
        use_autocast = getattr(args, "use_amp", False) and str(args.device).startswith("cuda")
        autocast_context = torch.amp.autocast("cuda", enabled=True) if use_autocast else nullcontext()
        with autocast_context:
            logits = model(images)
            if anchor_model is not None and kl_weight != 0.0:
                with torch.no_grad():
                    anchor_logits = anchor_model(images)
                loss_result = compute_maple_cross_entropy(
                    logits,
                    labels,
                    class_weights=class_weights,
                    anchor_logits=anchor_logits,
                    kl_weight=kl_weight,
                    kl_temperature=kl_temperature,
                    return_components=True,
                )
                loss = loss_result.loss
            else:
                loss_result = None
                loss = compute_maple_cross_entropy(logits, labels, class_weights=class_weights)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Full MaPLe produced non-finite logits at epoch {epoch}, batch {batch_index}")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Full MaPLe produced non-finite loss at epoch {epoch}, batch {batch_index}")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if scaler is None:
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Full MaPLe produced non-finite gradient for {name} at epoch {epoch}, batch {batch_index}")
        previous_scale = scaler.get_scale() if scaler is not None and hasattr(scaler, "get_scale") else None
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            current_scale = scaler.get_scale() if hasattr(scaler, "get_scale") else previous_scale
            optimizer_was_skipped = previous_scale is not None and current_scale < previous_scale
        else:
            optimizer_step(optimizer, args.device)
            optimizer_was_skipped = False
        if scheduler is not None and not optimizer_was_skipped:
            scheduler.step()
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size
        if wandb is not None:
            log_fields = {
                "train/batch_loss": loss.item(),
                "train/epoch": epoch,
                "train/batch": batch_index,
            }
            if loss_result is not None:
                log_fields.update(loss_result.to_log_dict(prefix="train/batch"))
            wandb.log(log_fields)

    return TrainStats(epoch=epoch, loss=total_loss / total_seen, accuracy=total_correct / total_seen)


def eval_full_maple_single_dataset(model, dataset, args, tau=None, class_priors=None, class_bias=None):
    from src.models.coop import eval_coop_single_dataset
    return eval_coop_single_dataset(model, dataset, args, desc="MaPLe eval", tau=tau, class_priors=class_priors, class_bias=class_bias)
