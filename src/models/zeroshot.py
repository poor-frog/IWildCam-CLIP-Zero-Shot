import torch
from tqdm import tqdm

import clip.clip as clip

import src.templates as templates
import src.datasets as datasets

from src.config import parse_arguments
from src.models.clip_encoder import ClassificationHead, CLIPEncoder
from src.models.eval import evaluate



class FrozenZeroShotAnchor(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        with torch.no_grad():
            features = self.image_encoder(images)
            return self.classification_head(features)


class MaPLeZeroPromptImageEncoder(torch.nn.Module):
    def __init__(self, visual, n_ctx, prompt_depth):
        super().__init__()
        self.visual = visual
        vision_width = visual.conv1.weight.shape[0]
        self.register_buffer("shared_ctx", torch.zeros(n_ctx, vision_width))
        self.compound_deeper_prompts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(n_ctx, vision_width), requires_grad=False)
            for _ in range(prompt_depth - 1)
        ])
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        prompts = [prompt.to(device=images.device, dtype=self.shared_ctx.dtype) for prompt in self.compound_deeper_prompts]
        return self.visual(images, self.shared_ctx.to(images.device), prompts)


def _build_zeroshot_classifier(args, clip_model, device):
    assert args.template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(None,
                            location=args.data_location,
                            batch_size=args.batch_size)
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(
                texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True,
                                             weights=zeroshot_weights)

    return classification_head


def get_zeroshot_classifier(args, clip_model):
    return _build_zeroshot_classifier(args, clip_model, args.device)


def get_compatible_zeroshot_classifier(args, device=None):
    device = device or args.device
    text_clip_model, _ = clip.load(args.model, device=device)
    return _build_zeroshot_classifier(args, text_clip_model, device)


def eval(args):
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    clip_encoder.eval()
    clip_encoder.to(args.device)

    classification_head = get_zeroshot_classifier(args, clip_encoder.model)
    classification_head.eval()
    classification_head.to(args.device)

    if hasattr(clip_encoder.model, 'transformer'):
        delattr(clip_encoder.model, 'transformer')

    evaluate(clip_encoder, args, classification_head)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
