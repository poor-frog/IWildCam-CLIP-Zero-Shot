import warnings

import torch


def get_xla_model():
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return None
    return xm


def get_xla_device():
    xm = get_xla_model()
    if xm is None:
        return None
    try:
        return xm.xla_device()
    except (RuntimeError, ValueError):
        return None


def is_xla_device(device):
    return str(device).startswith("xla")


def _cuda_is_usable():
    """Check whether CUDA is both available AND the GPU is compatible with the installed PyTorch build.

    On Kaggle the pre-installed PyTorch only supports sm_70+ but the allocated GPU
    may be a Tesla P100 (sm_60).  torch.cuda.is_available() still returns True in
    that case, so we do a tiny real kernel launch to confirm compatibility.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except (AssertionError, RuntimeError, torch.AcceleratorError) as exc:
        warnings.warn(f"CUDA device found but is not usable by this PyTorch build: {exc}")
        return False


def select_default_device():
    if _cuda_is_usable():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    xla_device = get_xla_device()
    if xla_device is not None:
        return xla_device
    return "cpu"


def resolve_device_choice(device_choice):
    if device_choice == "auto":
        return select_default_device()
    if device_choice == "xla":
        xla_device = get_xla_device()
        if xla_device is None:
            raise RuntimeError("--device=xla requires torch_xla and an available TPU/XLA runtime.")
        return xla_device
    return device_choice


def optimizer_step(optimizer, device):
    if is_xla_device(device):
        xm = get_xla_model()
        if xm is None:
            raise RuntimeError("TPU/XLA device selected but torch_xla is not available.")
        xm.optimizer_step(optimizer)
        xm.mark_step()
        return
    optimizer.step()


def prompt_tensor_dtype(tensor):
    if is_xla_device(tensor.device):
        return torch.bfloat16
    if tensor.is_floating_point():
        return tensor.dtype
    return torch.float32


def cast_prompt_like(prompt, reference):
    return prompt.to(dtype=prompt_tensor_dtype(reference), device=reference.device)
