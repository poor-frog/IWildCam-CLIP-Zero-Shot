# `src/models/` — Model Components

CLIP encoder wrapper, classification head, zero-shot weight builder, and evaluation loop.

## Files

| File | Role |
|------|------|
| `clip_encoder.py` | `CLIPEncoder`, `ClassificationHead`, `ImageClassifier` — model definitions + save/load |
| `zeroshot.py` | `get_zeroshot_classifier()` — builds classification head weights from text prompts via CLIP text encoder |
| `eval.py` | `eval_single_dataset()` — runs evaluation loop over a dataloader |

## Pipeline

```
Text templates ──→ CLIP text encoder ──→ zeroshot weights ──→ ClassificationHead
                                                                        ↑
Images ──→ CLIP image encoder ──→ features ────────────────────────────┘
                                                                        ↓
                                                                     logits → metrics
```
