from collections import Counter
from pathlib import Path
import random
from io import BytesIO

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image, ImageChops

try:
    from dataset import DocumentDataset, LABEL_MAP
except ImportError:
    from CNN.dataset import DocumentDataset, LABEL_MAP


# Default target size tuned for the L4 GPU (24 GB VRAM): full card at high resolution
# preserves guilloche patterns and typography artifacts engineered during data generation.
DEFAULT_TARGET_SIZE = (800, 600)

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
ELA_MEAN = 0.5
ELA_STD = 0.25


def _compute_ela_channel(image, ela_quality=90, ela_scale=12.0):
    """Compute a single-channel ELA map from a PIL RGB image."""
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=ela_quality)
    buffer.seek(0)
    recompressed_pil = Image.open(buffer).convert('RGB')
    ela = ImageChops.difference(image, recompressed_pil).convert('L')
    ela = ela.point(lambda p: min(255, int(p * ela_scale)))
    return transforms.functional.to_tensor(ela)


def _to_model_tensor(image, use_ela=False, ela_quality=90, ela_scale=12.0):
    """Convert PIL image to normalized RGB tensor, optionally appending ELA channel."""
    rgb = transforms.functional.to_tensor(image)
    if not use_ela:
        return transforms.functional.normalize(rgb, mean=RGB_MEAN, std=RGB_STD)

    ela = _compute_ela_channel(image, ela_quality=ela_quality, ela_scale=ela_scale)
    stacked = torch.cat([rgb, ela], dim=0)
    return transforms.functional.normalize(
        stacked,
        mean=RGB_MEAN + [ELA_MEAN],
        std=RGB_STD + [ELA_STD],
    )


def make_train_transform(target_size=DEFAULT_TARGET_SIZE, use_ela=False, ela_quality=90, ela_scale=12.0):
    return make_train_transform_with_profile(
        target_size=target_size,
        train_augmentation='full',
        use_ela=use_ela,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
    )


def make_train_transform_with_profile(
    target_size=DEFAULT_TARGET_SIZE,
    train_augmentation='full',
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
):
    if train_augmentation not in {'full', 'light', 'none'}:
        raise ValueError("train_augmentation must be one of: full, light, none")

    tensorize = transforms.Lambda(
        lambda img: _to_model_tensor(
            img,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
        )
    )

    if train_augmentation == 'none':
        return transforms.Compose([
            transforms.Resize(target_size),
            tensorize,
        ])

    if train_augmentation == 'light':
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.2),
            tensorize,
        ])

    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomRotation(degrees=3),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        tensorize,
    ])


def make_eval_transform(target_size=DEFAULT_TARGET_SIZE, use_ela=False, ela_quality=90, ela_scale=12.0):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.Lambda(
            lambda img: _to_model_tensor(
                img,
                use_ela=use_ela,
                ela_quality=ela_quality,
                ela_scale=ela_scale,
            )
        ),
    ])


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def _extract_group_id(file_path, label_id):
    """Return a stable group id so derived forgeries and source authentic stay in one split."""
    name = Path(file_path).name
    stem = Path(file_path).stem

    # Authentic filenames already encode unique citizen identity
    if label_id == LABEL_MAP['authentic']:
        return f"src:{stem}"

    # Forged samples must include source identity in filename, e.g. class1_0001__src__NCC-....jpg
    if '__src__' not in name:
        raise ValueError(
            "Leakage-safe split requires forged filenames with source tag '__src__'. "
            "Regenerate forged data with the updated generator."
        )

    source_id = name.split('__src__', 1)[1].rsplit('.', 1)[0]
    return f"src:{source_id}"


def _group_split_indices(samples, train_ratio, val_ratio, seed):
    groups = {}
    for idx, (file_path, label_id) in enumerate(samples):
        group_id = _extract_group_id(file_path, label_id)
        groups.setdefault(group_id, []).append(idx)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_samples = len(samples)
    train_target = int(train_ratio * total_samples)
    val_target = int(val_ratio * total_samples)

    train_indices, val_indices, test_indices = [], [], []

    for group_id in group_ids:
        group_indices = groups[group_id]
        if len(train_indices) < train_target:
            train_indices.extend(group_indices)
        elif len(val_indices) < val_target:
            val_indices.extend(group_indices)
        else:
            test_indices.extend(group_indices)

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Group split produced an empty split. Adjust ratios or dataset size.")

    return train_indices, val_indices, test_indices


def build_splits(
    authentic_dir="synthetic/generated/authentic",
    forged_dir="synthetic/generated/forged",
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42,
    target_size=DEFAULT_TARGET_SIZE,
    train_augmentation='full',
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
):
    full_dataset = DocumentDataset(
        authentic_dir=authentic_dir,
        forged_dir=forged_dir,
        transform=None,
    )

    total = len(full_dataset)
    if total == 0:
        raise ValueError("No images found in provided dataset directories.")

    train_indices, val_indices, test_indices = _group_split_indices(
        full_dataset.samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    train_set = TransformSubset(
        train_subset,
        make_train_transform_with_profile(
            target_size=target_size,
            train_augmentation=train_augmentation,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
        ),
    )
    val_set = TransformSubset(
        val_subset,
        make_eval_transform(
            target_size=target_size,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
        ),
    )
    test_set = TransformSubset(
        test_subset,
        make_eval_transform(
            target_size=target_size,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
        ),
    )

    return full_dataset, train_set, val_set, test_set


def build_dataloaders(
    train_set,
    val_set,
    test_set,
    batch_size=32,
    num_workers=0,
    pin_memory=None,
    persistent_workers=True,
    prefetch_factor=4,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def print_class_distribution(full_dataset):
    labels = [label for _, label in full_dataset.samples]
    dist = Counter(labels)
    label_names = {v: k for k, v in LABEL_MAP.items()}

    print("Class distribution:")
    for label_id, count in sorted(dist.items()):
        print(f"  {label_names[label_id]}: {count} samples")


def compute_class_weights(full_dataset):
    if isinstance(full_dataset, TransformSubset) and isinstance(full_dataset.subset, Subset):
        base_dataset = full_dataset.subset.dataset
        indices = full_dataset.subset.indices
        base_samples = getattr(base_dataset, 'samples', None)
        if base_samples is None:
            raise TypeError("Expected base dataset to expose a 'samples' attribute.")
        labels = [base_samples[i][1] for i in indices]
    elif isinstance(full_dataset, Subset):
        base_dataset = full_dataset.dataset
        indices = full_dataset.indices
        base_samples = getattr(base_dataset, 'samples', None)
        if base_samples is None:
            raise TypeError("Expected base dataset to expose a 'samples' attribute.")
        labels = [base_samples[i][1] for i in indices]
    else:
        samples = getattr(full_dataset, 'samples', None)
        if samples is None:
            raise TypeError("Expected dataset to expose a 'samples' attribute.")
        labels = [label for _, label in samples]

    dist = Counter(labels)
    total = len(labels)
    num_classes = len(LABEL_MAP)
    missing_classes = [i for i in range(num_classes) if dist.get(i, 0) == 0]
    if missing_classes:
        raise ValueError(
            f"Training split is missing class id(s): {missing_classes}. "
            "Adjust split/data generation before training."
        )
    # Inverse frequency weighting
    weights = [total / (num_classes * dist[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def run_sanity_check(train_loader):
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Pixel range: {images.min():.2f} to {images.max():.2f}")
    print(f"Labels in batch: {sorted(labels.unique().tolist())}")


def build_pipeline(
    authentic_dir="synthetic/generated/authentic",
    forged_dir="synthetic/generated/forged",
    batch_size=32,
    num_workers=0,
    pin_memory=None,
    seed=42,
    target_size=DEFAULT_TARGET_SIZE,
    train_augmentation='full',
    persistent_workers=True,
    prefetch_factor=4,
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
):
    full_dataset, train_set, val_set, test_set = build_splits(
        authentic_dir=authentic_dir,
        forged_dir=forged_dir,
        seed=seed,
        target_size=target_size,
        train_augmentation=train_augmentation,
        use_ela=use_ela,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return {
        "full_dataset": full_dataset,
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": compute_class_weights(train_set),
        "in_channels": 4 if use_ela else 3,
    }


if __name__ == "__main__":
    pipeline = build_pipeline()
    print_class_distribution(pipeline["full_dataset"])
    run_sanity_check(pipeline["train_loader"])
