import torch
import torch.nn.functional as F

from src.models.stmp_adapter import apply_sequence_consensus


def apply_tail_protective_sctr(
    frame_logits,
    metadata,
    sequence_field_index,
    class_priors,
    routing_strength,
    tail_protection_power,
):
    routing_strength = float(routing_strength)
    tail_protection_power = float(tail_protection_power)
    if not 0.0 <= routing_strength <= 1.0:
        raise ValueError(f"SCTR routing strength must be in [0, 1], got {routing_strength}.")
    if tail_protection_power < 0.0:
        raise ValueError(f"SCTR tail protection power must be non-negative, got {tail_protection_power}.")
    if routing_strength == 0.0 or sequence_field_index is None or not metadata:
        return frame_logits

    sequence_logits = apply_sequence_consensus(frame_logits, metadata, sequence_field_index, eta=1.0)
    sequence_delta = (sequence_logits - frame_logits).clamp_min(0.0)
    frame_probs = F.softmax(frame_logits, dim=1)
    sequence_probs = F.softmax(sequence_logits, dim=1)
    uncertainty = 1.0 - frame_probs.max(dim=1, keepdim=True).values
    agreement = torch.sqrt(frame_probs * sequence_probs)
    class_evidence = (1.0 - uncertainty) * agreement + uncertainty * sequence_probs
    normalized_priors = class_priors.to(device=frame_logits.device, dtype=frame_logits.dtype)
    normalized_priors = normalized_priors / normalized_priors.max().clamp_min(1e-12)
    frame_top_classes = frame_logits.argmax(dim=1)
    top_class_protection = normalized_priors[frame_top_classes].pow(tail_protection_power).unsqueeze(1)
    routing_gate = (routing_strength * top_class_protection * class_evidence).clamp(0.0, 1.0)
    return frame_logits + routing_gate * sequence_delta
