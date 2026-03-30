from collections import Counter
from pathlib import Path
import random
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

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
COORD_MEAN = 0.5
COORD_STD = 0.5
DEFAULT_ADVERSARIAL_PROB = 0.3
DEFAULT_ADVERSARIAL_JPEG_MIN_QUALITY = 40
DEFAULT_ADVERSARIAL_JPEG_MAX_QUALITY = 60

# Adversarial attack recipes: sequence of attacks to compose
ADV_RECIPES = {
    'glare_jpeg': ('glare', 'jpeg', 'contrast'),
    'shadow_noise': ('shadow', 'noise', 'sharpness'),
    'compression_stack': ('jpeg', 'jpeg', 'noise'),
    'lighting_shift': ('gamma', 'contrast', 'shadow'),
    'hard_phone_capture': ('perspective', 'glare', 'jpeg', 'noise'),
    'washed_scan': ('contrast', 'blur', 'jpeg'),
}


# ---------------------------------------------------------------------------
# Adversarial Attack Functions (ported from adversarial_generator.py)
# ---------------------------------------------------------------------------

def _apply_glare(image, rng):
    """Simulate lens glare on document via diagonal white stripe."""
    width, height = image.size
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    glare_width = rng.randint(max(40, width // 10), max(80, width // 4))
    center = rng.randint(width // 4, (3 * width) // 4)
    for step in range(glare_width):
        alpha = int(150 * (1.0 - abs((2.0 * step / max(glare_width - 1, 1)) - 1.0)))
        draw.line(
            ((center + step - glare_width // 2, 0), (center + step, height)),
            fill=(255, 255, 255, max(alpha, 0)),
            width=3,
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(6, width // 80)))
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')


def _apply_shadow(image, rng):
    """Simulate scanner/lighting shadow via horizontal dark band."""
    width, height = image.size
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    start_y = rng.randint(0, height // 3)
    end_y = rng.randint((2 * height) // 3, height)
    for y in range(start_y, end_y):
        alpha = int(110 * ((y - start_y) / max(end_y - start_y, 1)))
        draw.line(((0, y), (width, y)), fill=(0, 0, 0, alpha), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(8, height // 90)))
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')


def _apply_noise(image, rng):
    """Add Gaussian noise to pixels (lossy capture simulation)."""
    array = np.asarray(image).astype(np.float32)
    noise_std = rng.uniform(6.0, 20.0)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0.0, noise_std, size=array.shape)
    noisy = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy)


def _apply_jpeg(image, rng, min_quality=DEFAULT_ADVERSARIAL_JPEG_MIN_QUALITY, max_quality=DEFAULT_ADVERSARIAL_JPEG_MAX_QUALITY):
    """Apply moderate JPEG compression without fully destroying local compression signals."""
    quality = rng.randint(min_quality, max_quality)
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def _apply_contrast(image, rng):
    """Randomly adjust contrast (bad lighting conditions)."""
    factor = rng.uniform(0.65, 1.45)
    return ImageEnhance.Contrast(image).enhance(factor)


def _apply_sharpness(image, rng):
    """Randomly adjust sharpness (OCR robustness)."""
    factor = rng.uniform(0.4, 1.8)
    return ImageEnhance.Sharpness(image).enhance(factor)


def _apply_blur(image, rng):
    """Apply Gaussian blur (out-of-focus capture)."""
    radius = rng.uniform(0.6, 2.2)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_gamma(image, rng):
    """Adjust gamma curve (overexposed/underexposed scan)."""
    gamma = rng.uniform(0.75, 1.35)
    table = [min(255, max(0, int(((value / 255.0) ** gamma) * 255.0))) for value in range(256)]
    return image.point(table * 3)


def _apply_perspective(image, rng):
    """Apply mild rotation + crop (skewed scan angle)."""
    angle = rng.uniform(-3.5, 3.5)
    translated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(242, 242, 242))
    crop_x = rng.randint(0, max(1, image.size[0] // 40))
    crop_y = rng.randint(0, max(1, image.size[1] // 40))
    cropped = translated.crop((crop_x, crop_y, image.size[0], image.size[1]))
    return cropped.resize(image.size, Image.Resampling.BICUBIC)


class AdversarialAugmentation:
    """
    On-the-fly adversarial augmentation: only a fraction of training images are attacked.

    Most samples stay clean so the model keeps a baseline for typography, geometry,
    and localized compression artifacts. A minority are degraded to teach robustness.
    """

    def __init__(
        self,
        p=DEFAULT_ADVERSARIAL_PROB,
        seed=None,
        jpeg_min_quality=DEFAULT_ADVERSARIAL_JPEG_MIN_QUALITY,
        jpeg_max_quality=DEFAULT_ADVERSARIAL_JPEG_MAX_QUALITY,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'adversarial probability must be in [0, 1], got {p}')
        if jpeg_min_quality < 1 or jpeg_max_quality > 100 or jpeg_min_quality > jpeg_max_quality:
            raise ValueError(
                'jpeg quality bounds must satisfy 1 <= min <= max <= 100, '
                f'got {jpeg_min_quality}, {jpeg_max_quality}'
            )

        self.p = p
        self.rng = random.Random(seed)
        self.recipe_names = list(ADV_RECIPES.keys())
        self.jpeg_min_quality = jpeg_min_quality
        self.jpeg_max_quality = jpeg_max_quality
        self.attack_map = {
            'glare': _apply_glare,
            'shadow': _apply_shadow,
            'noise': _apply_noise,
            'jpeg': self._apply_jpeg,
            'contrast': _apply_contrast,
            'sharpness': _apply_sharpness,
            'blur': _apply_blur,
            'gamma': _apply_gamma,
            'perspective': _apply_perspective,
        }

    def _apply_jpeg(self, image, rng):
        return _apply_jpeg(
            image,
            rng,
            min_quality=self.jpeg_min_quality,
            max_quality=self.jpeg_max_quality,
        )

    def apply_recipe(self, image, recipe_name):
        variant = image.copy()
        for attack_name in ADV_RECIPES[recipe_name]:
            variant = self.attack_map[attack_name](variant, self.rng)
        return variant

    def __call__(self, image):
        if self.rng.random() > self.p:
            return image

        recipe_name = self.rng.choice(self.recipe_names)
        return self.apply_recipe(image, recipe_name)


def _compute_ela_channel(image, ela_quality=90, ela_scale=12.0):
    """Compute a single-channel ELA map from a PIL RGB image."""
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=ela_quality)
    buffer.seek(0)
    recompressed_pil = Image.open(buffer).convert('RGB')
    ela = ImageChops.difference(image, recompressed_pil).convert('L')
    ela = ela.point(lambda p: min(255, int(p * ela_scale)))
    return transforms.functional.to_tensor(ela)


def _compute_coordinate_channels(height, width, dtype):
    """Generate normalized X/Y coordinate channels for CoordConv-style location cues."""
    y_coords = torch.linspace(0.0, 1.0, steps=height, dtype=dtype).view(1, height, 1).expand(1, height, width)
    x_coords = torch.linspace(0.0, 1.0, steps=width, dtype=dtype).view(1, 1, width).expand(1, height, width)
    return torch.cat([x_coords, y_coords], dim=0)


def _to_model_tensor(
    image,
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
    use_coord_channels=False,
):
    """Convert PIL image to model tensor with optional ELA and coordinate channels."""
    rgb = transforms.functional.to_tensor(image)
    channels = [rgb]
    mean = list(RGB_MEAN)
    std = list(RGB_STD)

    if use_ela:
        ela = _compute_ela_channel(image, ela_quality=ela_quality, ela_scale=ela_scale)
        channels.append(ela)
        mean.append(ELA_MEAN)
        std.append(ELA_STD)

    if use_coord_channels:
        coord_channels = _compute_coordinate_channels(
            height=rgb.shape[1],
            width=rgb.shape[2],
            dtype=rgb.dtype,
        )
        channels.append(coord_channels)
        mean.extend([COORD_MEAN, COORD_MEAN])
        std.extend([COORD_STD, COORD_STD])

    stacked = torch.cat(channels, dim=0)
    return transforms.functional.normalize(
        stacked,
        mean=mean,
        std=std,
    )


def make_train_transform(
    target_size=DEFAULT_TARGET_SIZE,
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
    use_coord_channels=False,
):
    return make_train_transform_with_profile(
        target_size=target_size,
        train_augmentation='full',
        use_ela=use_ela,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
        use_coord_channels=use_coord_channels,
    )


def make_train_transform_with_profile(
    target_size=DEFAULT_TARGET_SIZE,
    train_augmentation='full',
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
    use_coord_channels=False,
    adversarial_seed=None,
    adversarial_prob=DEFAULT_ADVERSARIAL_PROB,
    adversarial_jpeg_min_quality=DEFAULT_ADVERSARIAL_JPEG_MIN_QUALITY,
    adversarial_jpeg_max_quality=DEFAULT_ADVERSARIAL_JPEG_MAX_QUALITY,
):
    if train_augmentation not in {'full', 'light', 'none', 'adversarial'}:
        raise ValueError("train_augmentation must be one of: full, light, none, adversarial")

    # ROI heads assume a stable card layout, so horizontal flips would create invalid supervision.
    light_horizontal_flip = transforms.RandomHorizontalFlip(p=0.0)
    full_horizontal_flip = transforms.RandomHorizontalFlip(p=0.0)

    tensorize = transforms.Lambda(
        lambda img: _to_model_tensor(
            img,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
            use_coord_channels=use_coord_channels,
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
            light_horizontal_flip,
            tensorize,
        ])

    if train_augmentation == 'adversarial':
        return transforms.Compose([
            transforms.Resize(target_size),
            AdversarialAugmentation(
                p=adversarial_prob,
                seed=adversarial_seed,
                jpeg_min_quality=adversarial_jpeg_min_quality,
                jpeg_max_quality=adversarial_jpeg_max_quality,
            ),
            transforms.RandomRotation(degrees=3),
            full_horizontal_flip,
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            tensorize,
        ])

    # Default 'full' mode: standard augmentation without adversarial attacks
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomRotation(degrees=3),
        full_horizontal_flip,
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        tensorize,
    ])


def make_eval_transform(
    target_size=DEFAULT_TARGET_SIZE,
    use_ela=False,
    ela_quality=90,
    ela_scale=12.0,
    use_coord_channels=False,
):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.Lambda(
            lambda img: _to_model_tensor(
                img,
                use_ela=use_ela,
                ela_quality=ela_quality,
                ela_scale=ela_scale,
                use_coord_channels=use_coord_channels,
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
    use_coord_channels=False,
    adversarial_seed=None,
    adversarial_prob=DEFAULT_ADVERSARIAL_PROB,
    adversarial_jpeg_min_quality=DEFAULT_ADVERSARIAL_JPEG_MIN_QUALITY,
    adversarial_jpeg_max_quality=DEFAULT_ADVERSARIAL_JPEG_MAX_QUALITY,
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
            use_coord_channels=use_coord_channels,
            adversarial_seed=adversarial_seed,
            adversarial_prob=adversarial_prob,
            adversarial_jpeg_min_quality=adversarial_jpeg_min_quality,
            adversarial_jpeg_max_quality=adversarial_jpeg_max_quality,
        ),
    )
    val_set = TransformSubset(
        val_subset,
        make_eval_transform(
            target_size=target_size,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
            use_coord_channels=use_coord_channels,
        ),
    )
    test_set = TransformSubset(
        test_subset,
        make_eval_transform(
            target_size=target_size,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
            use_coord_channels=use_coord_channels,
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
    use_coord_channels=False,
    adversarial_seed=None,
    adversarial_prob=DEFAULT_ADVERSARIAL_PROB,
    adversarial_jpeg_min_quality=DEFAULT_ADVERSARIAL_JPEG_MIN_QUALITY,
    adversarial_jpeg_max_quality=DEFAULT_ADVERSARIAL_JPEG_MAX_QUALITY,
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
        use_coord_channels=use_coord_channels,
        adversarial_seed=adversarial_seed,
        adversarial_prob=adversarial_prob,
        adversarial_jpeg_min_quality=adversarial_jpeg_min_quality,
        adversarial_jpeg_max_quality=adversarial_jpeg_max_quality,
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
        "in_channels": 3 + int(use_ela) + (2 if use_coord_channels else 0),
    }


if __name__ == "__main__":
    pipeline = build_pipeline()
    print_class_distribution(pipeline["full_dataset"])
    run_sanity_check(pipeline["train_loader"])
