import math

import torch
from torch.nn.utils import parametrize


class LoRAWeightParametrization(torch.nn.Module):
    def __init__(self, out_features, in_features, rank, alpha=None, dropout=0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1.")
        if dropout != 0:
            raise ValueError("LoRA dropout is not supported for MultiheadAttention out_proj weight parametrization yet.")
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2
        self.scaling = self.alpha / self.rank
        self.register_buffer("gamma", torch.tensor(1.0, dtype=torch.float32))
        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def set_gamma(self, gamma):
        self.gamma.fill_(float(gamma))

    def forward(self, original_weight):
        delta = self.lora_up.weight @ self.lora_down.weight
        delta = delta.to(device=original_weight.device, dtype=original_weight.dtype)
        gamma = self.gamma.to(device=original_weight.device, dtype=original_weight.dtype)
        return original_weight + gamma * delta * self.scaling


def has_lora_weight_parametrization(linear_layer):
    return parametrize.is_parametrized(linear_layer, "weight")


def add_lora_to_linear_weight(linear_layer, rank, alpha=None, dropout=0.0):
    if not isinstance(linear_layer, torch.nn.Linear):
        raise TypeError("LoRA weight parametrization can only wrap torch.nn.Linear modules.")
    if has_lora_weight_parametrization(linear_layer):
        return linear_layer
    linear_layer.weight.requires_grad = False
    if linear_layer.bias is not None:
        linear_layer.bias.requires_grad = False
    parametrize.register_parametrization(
        linear_layer,
        "weight",
        LoRAWeightParametrization(
            out_features=linear_layer.out_features,
            in_features=linear_layer.in_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ),
    )
    return linear_layer


def _selected_layer_indices(total_layers, layer_spec):
    if layer_spec == "all":
        return range(total_layers)
    if layer_spec.startswith("last"):
        count = int(layer_spec.removeprefix("last"))
        if count < 1:
            raise ValueError("LoRA layer count must be >= 1.")
        start = max(total_layers - count, 0)
        return range(start, total_layers)
    raise ValueError("Unsupported LoRA layer spec. Use 'lastN' or 'all'.")


def inject_vision_out_proj_lora(model, rank, alpha=None, dropout=0.0, layers="last6"):
    if rank < 1:
        return []
    resblocks = model.image_encoder.transformer.resblocks
    selected_indices = list(_selected_layer_indices(len(resblocks), layers))
    injected_names = []
    for block_index in selected_indices:
        out_proj = resblocks[block_index].attn.out_proj
        add_lora_to_linear_weight(out_proj, rank=rank, alpha=alpha, dropout=dropout)
        injected_names.extend([
            f"image_encoder.transformer.resblocks.{block_index}.attn.out_proj.parametrizations.weight.0.lora_down.weight",
            f"image_encoder.transformer.resblocks.{block_index}.attn.out_proj.parametrizations.weight.0.lora_up.weight",
        ])
    return injected_names


def collect_lora_state_dict(model):
    model = model.module if hasattr(model, "module") else model
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if ".lora_down." in name or ".lora_up." in name
    }


def load_lora_state_dict(model, lora_state_dict):
    current_params = dict(model.named_parameters())
    missing = [name for name in lora_state_dict if name not in current_params]
    if missing:
        raise RuntimeError(f"LoRA checkpoint has missing model parameters: {missing}")
    with torch.no_grad():
        for name, tensor in lora_state_dict.items():
            current_params[name].copy_(tensor.to(device=current_params[name].device, dtype=current_params[name].dtype))


def set_lora_gamma(model, gamma):
    model = model.module if hasattr(model, "module") else model
    for module in model.modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None or not hasattr(parametrizations, "weight"):
            continue
        for parametrization in parametrizations.weight:
            if hasattr(parametrization, "set_gamma"):
                parametrization.set_gamma(gamma)
