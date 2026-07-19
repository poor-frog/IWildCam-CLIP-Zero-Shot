from __future__ import annotations

import argparse
import json
from pathlib import Path

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report locked Val-Audit TPA calibration diagnostics for LESVI limitation wording only.")
    parser.add_argument("--val-audit-tpa-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import torch
    from src.models.lesvi_evaluation import stable_calibration_report

    payload = torch.load(args.val_audit_tpa_cache, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("split") != "val_audit":
        raise ValueError("Calibration cache must be explicitly tagged split='val_audit'.")
    logits = payload.get("logits")
    labels = payload.get("labels")
    frame_ids = payload.get("frame_ids")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor) or not isinstance(frame_ids, (list, tuple)):
        raise ValueError("Calibration cache must contain logits, labels, and stable frame_ids.")
    report = stable_calibration_report(logits, labels.long(), [str(value) for value in frame_ids], bins=15)
    artifact = {
        "method": "LESVI-v0",
        "split": "val_audit",
        "role": "limitation_wording_only",
        "may_change_inference_or_promotion": False,
        "calibration": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(parse_arguments())
