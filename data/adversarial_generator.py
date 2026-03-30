import argparse
from io import BytesIO
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'generated' / 'authentic'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'synthetic' / 'generated' / 'adversarial'

RECIPES = {
    'glare_jpeg': ('glare', 'jpeg', 'contrast'),
    'shadow_noise': ('shadow', 'noise', 'sharpness'),
    'compression_stack': ('jpeg', 'jpeg', 'noise'),
    'lighting_shift': ('gamma', 'contrast', 'shadow'),
    'hard_phone_capture': ('perspective', 'glare', 'jpeg', 'noise'),
    'washed_scan': ('contrast', 'blur', 'jpeg'),
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Generate adversarial document variants with photometric and compression attacks.'
    )
    parser.add_argument('--input', default=str(DEFAULT_INPUT_DIR), help='Input image or directory.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Output directory.')
    parser.add_argument('--count', type=int, default=8, help='Number of adversarial images to generate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser


def sample_inputs(input_path, count, rng):
    if input_path.is_file():
        return [input_path] * count

    candidates = sorted(
        path for path in input_path.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    if not candidates:
        raise ValueError(f'No image files found in {input_path}')
    return [rng.choice(candidates) for _ in range(count)]


def apply_glare(image, rng):
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


def apply_shadow(image, rng):
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


def apply_noise(image, rng):
    array = np.asarray(image).astype(np.float32)
    noise_std = rng.uniform(6.0, 20.0)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0.0, noise_std, size=array.shape)
    noisy = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg(image, rng):
    quality = rng.randint(18, 45)
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def apply_contrast(image, rng):
    factor = rng.uniform(0.65, 1.45)
    return ImageEnhance.Contrast(image).enhance(factor)


def apply_sharpness(image, rng):
    factor = rng.uniform(0.4, 1.8)
    return ImageEnhance.Sharpness(image).enhance(factor)


def apply_blur(image, rng):
    radius = rng.uniform(0.6, 2.2)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gamma(image, rng):
    gamma = rng.uniform(0.75, 1.35)
    table = [min(255, max(0, int(((value / 255.0) ** gamma) * 255.0))) for value in range(256)]
    return image.point(table * 3)


def apply_perspective(image, rng):
    angle = rng.uniform(-3.5, 3.5)
    translated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(242, 242, 242))
    crop_x = rng.randint(0, max(1, image.size[0] // 40))
    crop_y = rng.randint(0, max(1, image.size[1] // 40))
    cropped = translated.crop((crop_x, crop_y, image.size[0], image.size[1]))
    return cropped.resize(image.size, Image.Resampling.BICUBIC)


ATTACKS = {
    'glare': apply_glare,
    'shadow': apply_shadow,
    'noise': apply_noise,
    'jpeg': apply_jpeg,
    'contrast': apply_contrast,
    'sharpness': apply_sharpness,
    'blur': apply_blur,
    'gamma': apply_gamma,
    'perspective': apply_perspective,
}


def generate_variant(image, recipe_name, rng):
    variant = image.copy()
    for attack_name in RECIPES[recipe_name]:
        variant = ATTACKS[attack_name](variant, rng)
    return variant


def main():
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f'Input path not found: {input_path}')

    selected_inputs = sample_inputs(input_path, args.count, rng)
    recipe_names = list(RECIPES.keys())

    for index, image_path in enumerate(selected_inputs, start=1):
        recipe_name = recipe_names[(index - 1) % len(recipe_names)]
        image = Image.open(image_path).convert('RGB')
        variant = generate_variant(image, recipe_name, rng)
        output_name = f'{image_path.stem}__adv_{index:02d}__{recipe_name}.png'
        variant.save(output_dir / output_name)
        print(f'Saved {output_name}')


if __name__ == '__main__':
    main()