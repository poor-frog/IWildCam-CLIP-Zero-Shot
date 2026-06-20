import os
import pandas as pd
import json
import numpy as np
import pathlib

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset


def get_mask_non_empty(dataset):
    metadf = pd.read_csv(dataset._data_dir / 'metadata.csv')
    filename = os.path.expanduser(dataset._data_dir / 'iwildcam2020_megadetector_results.json')
    with open(filename, 'r') as f:
        md_data = json.load(f)
    id_to_maxdet = {x['id']: x['max_detection_conf'] for x in md_data['images']}
    threshold = 0.95
    mask_non_empty = [id_to_maxdet[x] >= threshold for x in metadf['image_id']]
    return mask_non_empty


def get_nonempty_subset(dataset, split, frac=1.0, transform=None):
    if split not in dataset.split_dict:
        raise ValueError(f"Split {split} not found in dataset's split_dict.")
    split_mask = dataset.split_array == dataset.split_dict[split]

    # intersect split mask with non_empty. here is the only place this fn differs
    # from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/wilds_dataset.py#L56
    mask_non_empty = get_mask_non_empty(dataset)
    split_mask = split_mask & mask_non_empty

    split_idx = np.where(split_mask)[0]
    if frac < 1.0:
        num_to_retain = int(np.round(float(len(split_idx)) * frac))
        split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
    subset = WILDSSubset(dataset, split_idx, transform)
    return subset


def _to_numpy_labels(labels):
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels, dtype=np.int64)


def sample_indices(labels, n_examples, seed=0, class_balanced=False, num_classes=None):
    labels = _to_numpy_labels(labels)
    if n_examples <= 0 or n_examples >= len(labels) and not class_balanced:
        return np.arange(len(labels), dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot sample from an empty label array.")

    rng = np.random.default_rng(seed)
    if not class_balanced:
        return np.sort(rng.choice(len(labels), size=n_examples, replace=False)).astype(np.int64)

    if num_classes is None:
        num_classes = int(labels.max()) + 1
    present_classes = [class_id for class_id in range(num_classes) if np.any(labels == class_id)]
    if not present_classes:
        raise ValueError("Cannot class-balance sample without present classes.")

    base = n_examples // len(present_classes)
    remainder = n_examples % len(present_classes)
    selected = []
    for offset, class_id in enumerate(present_classes):
        target = base + (1 if offset < remainder else 0)
        class_indices = np.where(labels == class_id)[0]
        replace = len(class_indices) < target
        selected.extend(rng.choice(class_indices, size=target, replace=replace).tolist())

    rng.shuffle(selected)
    return np.asarray(selected[:n_examples], dtype=np.int64)


def maybe_subsample_ood_val(subset, n_examples=-1, seed=0, class_balanced=False, num_classes=None):
    if n_examples is None or n_examples <= 0:
        return subset
    labels = getattr(subset, 'y_array', None)
    if labels is None:
        raise ValueError("Cannot subsample OOD validation subset without y_array labels.")
    indices = sample_indices(labels, n_examples, seed=seed, class_balanced=class_balanced, num_classes=num_classes)
    source_indices = getattr(subset, 'indices', None)
    if source_indices is None:
        raise ValueError("Cannot subsample OOD validation subset without WILDS source indices.")
    sampled_indices = np.asarray(source_indices)[indices]
    return WILDSSubset(subset.dataset, sampled_indices, getattr(subset, 'transform', None))


def compute_class_counts(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot compute class counts from empty labels.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("labels must be in [0, num_classes).")
    return np.bincount(labels, minlength=num_classes).astype(np.float64)


def compute_class_priors(labels, num_classes):
    counts = compute_class_counts(labels, num_classes)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot compute class priors from zero total count.")
    return counts, counts / total


def compute_inverse_frequency_weights(labels, num_classes, eps=1e-12):
    counts, priors = compute_class_priors(labels, num_classes)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = 1.0 / np.maximum(priors[present], eps)
    if present.any():
        weights[present] *= present.sum() / weights[present].sum()
    return counts, priors, weights


def get_train_class_priors(dataset, num_classes):
    train_subset = dataset.get_subset('train', transform=None)
    labels = getattr(train_subset, 'y_array', None)
    if labels is None:
        raise ValueError("Train subset does not expose y_array labels.")
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    return compute_class_priors(labels, num_classes)


class IWildCam:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 remove_non_empty=False,
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 subset='train',
                 n_examples=-1,
                 use_class_balanced=False,
                 seed=0):
        self.dataset = wilds.get_dataset(dataset='iwildcam', root_dir=location)
        if remove_non_empty:
            self.train_dataset = get_nonempty_subset(self.dataset, 'train', transform=preprocess)
        else:
            self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        self.train_loader = get_train_loader("standard", self.train_dataset, num_workers=num_workers, batch_size=batch_size)

        if remove_non_empty:
            self.test_dataset = get_nonempty_subset(self.dataset, subset, transform=preprocess)
        else:
            self.test_dataset = self.dataset.get_subset(subset, transform=preprocess)
        if subset == 'val':
            self.test_dataset = maybe_subsample_ood_val(
                self.test_dataset,
                n_examples=n_examples,
                seed=seed,
                class_balanced=use_class_balanced,
                num_classes=getattr(self.dataset, 'n_classes', None),
            )

        self.test_loader = get_eval_loader(
            "standard", self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size)

        labels_csv = pathlib.Path(__file__).parent / 'iwildcam_metadata' / 'labels.csv'
        df = pd.read_csv(labels_csv)
        df = df[df['y'] < 99999]
        
        self.classnames = [s.lower() for s in list(df['english'])]

    def post_loop_metrics(self, labels, preds, metadata, args):
        preds = preds.argmax(dim=1, keepdim=True).view_as(labels)
        results = self.dataset.eval(preds, labels, metadata)
        return results[0]

class IWildCamIDVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_val'
        super().__init__(*args, **kwargs)

class IWildCamVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val'
        super().__init__(*args, **kwargs)

class IWildCamOODVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val'
        super().__init__(*args, **kwargs)

class IWildCamID(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)

class IWildCamOOD(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)


class IWildCamNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['remove_non_empty'] = True
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)


class IWildCamIDNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['remove_non_empty'] = True
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)


class IWildCamOODNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['remove_non_empty'] = True
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)
