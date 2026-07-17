from src.models import clip_encoder, modeling
from clip import model as clip_model
from clip import clip as clip_module
from pathlib import Path


def test_legacy_modeling_module_reexports_current_checkpoint_classes():
    assert modeling.CLIPEncoder is clip_encoder.CLIPEncoder
    assert modeling.ClassificationHead is clip_encoder.ClassificationHead
    assert modeling.ImageClassifier is clip_encoder.ImageClassifier
    assert modeling.ImageClassifier_Norm is clip_encoder.ImageClassifier_Norm
    assert modeling.ImageEncoder is clip_encoder.ImageEncoder


def test_legacy_visual_transformer_name_resolves_to_current_implementation():
    assert clip_model.VisualTransformer is clip_model.VisionTransformer


def test_legacy_preprocess_symbol_resolves_to_current_implementation():
    assert clip_module._convert_to_rgb is clip_module._convert_image_to_rgb


def test_official_flyp_pickle_checkpoint_loads_when_available():
    checkpoint = Path("checkpoints/flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt")
    if not checkpoint.is_file():
        return

    model = clip_encoder.CLIPEncoder.load(checkpoint)

    assert isinstance(model, clip_encoder.CLIPEncoder)
    assert hasattr(model, "model")
