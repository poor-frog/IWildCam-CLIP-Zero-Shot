# `src/datasets/` — Dataset Classes

Wraps the WILDS IWildCam dataset into PyTorch dataloaders for zero-shot evaluation.

## Files

| File | Role |
|------|------|
| `__init__.py` | Exports IWildCam split wrappers, including optional `*NonEmpty` variants |
| `iwildcam.py` | WILDS `IWildCamDataset` wrapper — loads images, applies transforms, returns labels |
| `dataloader.py` | Shared utilities: `get_dataloader()`, `maybe_dictionarize()` |
| `iwildcam_metadata/labels.csv` | Class ID → class name mapping (182 classes) |

## Dataset Splits

| Export Name | WILDS Subset | Description |
|-------------|-------------|-------------|
| `IWildCamIDVal` | `id_val` | In-distribution validation |
| `IWildCamID` | `id_test` | In-distribution test |
| `IWildCamOOD` | `test` | Out-of-distribution test |
| `IWildCamNonEmpty` | `train` | Train split filtered by MegaDetector confidence |
| `IWildCamIDNonEmpty` | `id_test` | ID test filtered by MegaDetector confidence |
| `IWildCamOODNonEmpty` | `test` | OOD test filtered by MegaDetector confidence |
