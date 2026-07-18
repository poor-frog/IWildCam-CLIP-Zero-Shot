__all__ = ["CLIPEncoder", "get_zeroshot_classifier", "eval_single_dataset", "CustomCLIP"]


def __getattr__(name):
    if name == "CLIPEncoder":
        from .clip_encoder import CLIPEncoder

        return CLIPEncoder
    if name == "get_zeroshot_classifier":
        from .zeroshot import get_zeroshot_classifier

        return get_zeroshot_classifier
    if name == "eval_single_dataset":
        from .eval import eval_single_dataset

        return eval_single_dataset
    if name == "CustomCLIP":
        from .coop import CustomCLIP

        return CustomCLIP
    raise AttributeError(name)
