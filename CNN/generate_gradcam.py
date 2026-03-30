"""
Grad-CAM heatmap generator for the ForensicsNet document forgery detector.

Reuses the same preprocessing and Grad-CAM logic as inference.py so the
visualization is guaranteed to match exactly what the model sees during eval.

How it works
------------
1.  The full N-channel input tensor (RGB + optional ELA + optional XY coords)
    is built exactly as it is during training/inference.
2.  `compute_gradcam` taps the output of `backbone.features` (the last conv
    feature map, shape [1, 1280, H', W']), sends a gradient signal backwards
    from the winning class logit (or --target-class), and produces a [H, W]
    heatmap by GAP-weighting those feature channels.
3.  Only channels 0-2 (RGB, pre-normalisation) are used for display so the
    report image is always a natural-looking ID card.

Outputs (written to --output-dir)
----------------------------------
  {prefix}_gradcam_rgb.png           — resized RGB only
  {prefix}_gradcam_overlay.png       — overlay of heatmap on RGB
  {prefix}_gradcam_report_grid.png   — 4-panel report figure
                                       (RGB | ELA | raw heatmap | overlay)

Usage examples
--------------
  # Auto-detect target class from model confidence
  python CNN/generate_gradcam.py \\
      --image data/synthetic/generated/forged/class2_0001__src__XYZ.jpg \\
      --checkpoint results/best_model_phase2.pth

  # Force target class by name
  python CNN/generate_gradcam.py \\
      --image data/synthetic/generated/forged/class2_0001__src__XYZ.jpg \\
      --checkpoint results/best_model_phase2.pth \\
      --target-class class2_name

  # Force target class by integer index
  python CNN/generate_gradcam.py \\
      --image data/synthetic/generated/forged/class1_0001__src__XYZ.jpg \\
      --checkpoint results/best_model_phase2.pth \\
      --target-class 1
"""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    from inference import (
        compute_gradcam,
        heatmap_to_rgb,
        blend_overlay,
        infer_channel_flags,
        load_checkpoint,
        preprocess_image,
        resolve_ela_params,
        resolve_label_names,
        resolve_target_size,
        DEFAULT_CHECKPOINT,
    )
    from data_pipeline import _compute_ela_channel, RGB_MEAN, RGB_STD
except ImportError:
    from CNN.inference import (
        compute_gradcam,
        heatmap_to_rgb,
        blend_overlay,
        infer_channel_flags,
        load_checkpoint,
        preprocess_image,
        resolve_ela_params,
        resolve_label_names,
        resolve_target_size,
        DEFAULT_CHECKPOINT,
    )
    from CNN.data_pipeline import _compute_ela_channel, RGB_MEAN, RGB_STD

try:
    from model import build_model
except ImportError:
    from CNN.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'gradcam'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM heatmaps for the ForensicsNet model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--image', required=True, help='Path to source image (RGB).')
    parser.add_argument(
        '--checkpoint',
        default=str(DEFAULT_CHECKPOINT),
        help='Path to trained checkpoint (.pth). Defaults to results/best_model_phase2.pth.',
    )
    parser.add_argument(
        '--target-class',
        default=None,
        metavar='CLASS',
        help=(
            'Class to generate the heatmap for. '
            'Accepts an integer index OR a label name (e.g. class2_name). '
            'Defaults to the model\'s top prediction.'
        ),
    )
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Resize target (H W). Defaults to checkpoint config or 800 600.',
    )
    parser.add_argument('--ela-quality', type=int, default=None)
    parser.add_argument('--ela-scale', type=float, default=None)

    ela_group = parser.add_mutually_exclusive_group()
    ela_group.add_argument('--use-ela', dest='use_ela', action='store_true')
    ela_group.add_argument('--no-ela', dest='use_ela', action='store_false')
    parser.set_defaults(use_ela=None)

    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument('--use-coord-channels', dest='use_coord_channels', action='store_true')
    coord_group.add_argument('--no-coord-channels', dest='use_coord_channels', action='store_false')
    parser.set_defaults(use_coord_channels=None)

    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory for output images.',
    )
    parser.add_argument(
        '--prefix',
        default=None,
        help='Output filename prefix. Defaults to image stem.',
    )
    parser.add_argument('--alpha', type=float, default=0.45, help='Heatmap blend alpha (0-1).')
    parser.add_argument('--dpi', type=int, default=200, help='Figure resolution (DPI).')
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_target_class(raw_arg, label_names):
    """Accept int index or label name; resolve to int index."""
    if raw_arg is None:
        return None
    if raw_arg.isdigit():
        index = int(raw_arg)
        if index >= len(label_names):
            raise ValueError(f'--target-class {index} is out of range (0–{len(label_names)-1}).')
        return index
    lower_map = {name.lower(): i for i, name in enumerate(label_names)}
    if raw_arg.lower() not in lower_map:
        raise ValueError(
            f'Unknown class name "{raw_arg}". '
            f'Available: {label_names}'
        )
    return lower_map[raw_arg.lower()]


def denormalize_rgb(tensor_3chw):
    """Undo ImageNet normalisation and return a uint8 HxWx3 numpy array."""
    mean = torch.tensor(RGB_MEAN, dtype=tensor_3chw.dtype).view(3, 1, 1)
    std = torch.tensor(RGB_STD, dtype=tensor_3chw.dtype).view(3, 1, 1)
    rgb = tensor_3chw * std + mean
    rgb = rgb.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (rgb * 255.0).astype(np.uint8)


def compute_ela_display(pil_rgb, ela_quality, ela_scale):
    """Return a uint8 grayscale numpy array of the ELA map for display."""
    ela_tensor = _compute_ela_channel(pil_rgb, ela_quality=ela_quality, ela_scale=ela_scale)
    ela_np = ela_tensor.squeeze(0).cpu().numpy()
    return (np.clip(ela_np, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Report figure
# ---------------------------------------------------------------------------

def save_report_grid(
    output_dir,
    prefix,
    rgb_np,
    ela_np,
    heatmap_np,
    overlay_np,
    predicted_label,
    target_label,
    confidence,
    aux_scores,
    region_names,
    dpi,
):
    """4-panel academic report figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=dpi)

    axes[0].imshow(rgb_np)
    axes[0].set_title('Stage 1: Original RGB (C1–C3)', fontsize=11)

    if ela_np is not None:
        axes[1].imshow(ela_np, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Stage 2: ELA Map (C4)', fontsize=11)
    else:
        axes[1].imshow(np.zeros_like(rgb_np[..., 0]), cmap='gray')
        axes[1].set_title('Stage 2: ELA (not used)', fontsize=11)

    axes[2].imshow(heatmap_np, cmap='jet', vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f'Grad-CAM Heatmap\nTarget: {target_label}',
        fontsize=11,
    )

    axes[3].imshow(overlay_np)
    axes[3].set_title(
        f'Grad-CAM Overlay\nPredicted: {predicted_label} ({confidence*100:.1f}%)',
        fontsize=11,
    )

    # Aux region scores as a footnote
    if aux_scores:
        aux_text = '  '.join(
            f'{name}: {score*100:.0f}%'
            for name, score in zip(region_names, aux_scores)
        )
        fig.text(0.5, 0.01, f'Regional head scores — {aux_text}',
                 ha='center', fontsize=9, color='#444444')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.04, 1.0, 1.0])
    grid_path = output_dir / f'{prefix}_gradcam_report_grid.png'
    fig.savefig(grid_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return grid_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f'Error: image not found: {image_path}', file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f'Error: checkpoint not found: {checkpoint_path}', file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix if args.prefix else image_path.stem

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint & config ─────────────────────────────────────────────
    checkpoint, state_dict, in_channels = load_checkpoint(checkpoint_path, device)
    checkpoint_config = checkpoint.get('config')
    label_names = resolve_label_names(checkpoint_config)
    target_size = resolve_target_size(checkpoint_config, args.target_size)
    ela_quality, ela_scale = resolve_ela_params(checkpoint_config, args.ela_quality, args.ela_scale)
    use_ela, use_coord_channels = infer_channel_flags(
        in_channels,
        checkpoint_config,
        args.use_ela,
        args.use_coord_channels,
    )

    # ── Build model ──────────────────────────────────────────────────────────
    model, device = build_model(
        num_classes=len(label_names),
        dropout_rate=0.3,
        in_channels=in_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # ── Preprocess ───────────────────────────────────────────────────────────
    pil_rgb = Image.open(image_path).convert('RGB').resize(
        (target_size[1], target_size[0]), Image.Resampling.BILINEAR
    )

    input_tensor = preprocess_image(
        Image.open(image_path).convert('RGB'),
        target_size=target_size,
        use_ela=use_ela,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
        use_coord_channels=use_coord_channels,
    ).to(device)

    # ── Resolve target class ─────────────────────────────────────────────────
    target_index = resolve_target_class(args.target_class, label_names)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs['fusion_logits'], dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        predicted_label = label_names[predicted_index]
        predicted_confidence = float(probabilities[predicted_index].item())
        aux_scores = torch.sigmoid(outputs['aux_logits'])[0].detach().cpu().tolist()

    if target_index is None:
        target_index = predicted_index

    target_label = label_names[target_index]
    target_confidence = float(probabilities[target_index].item())

    print(f'Predicted class:  {predicted_label}  ({predicted_confidence*100:.2f}%)')
    print(f'Grad-CAM target:  {target_label}  ({target_confidence*100:.2f}%)')

    # ── Compute Grad-CAM ─────────────────────────────────────────────────────
    # compute_gradcam enables_grad internally; no need to wrap here.
    heatmap_np, _ = compute_gradcam(model, input_tensor, target_index)

    # ── Build display images ─────────────────────────────────────────────────
    # RGB display: de-normalize the first 3 channels from the model tensor
    rgb_display = denormalize_rgb(input_tensor[0, :3].cpu())

    # Heatmap overlay on RGB
    heatmap_rgb = heatmap_to_rgb(heatmap_np)
    overlay_np = np.array(blend_overlay(Image.fromarray(rgb_display), heatmap_rgb, alpha=args.alpha))

    # ELA display: recompute raw (un-normalized) for visualization
    ela_display = compute_ela_display(pil_rgb, ela_quality, ela_scale) if use_ela else None

    # ── Save individual images ────────────────────────────────────────────────
    Image.fromarray(rgb_display).save(output_dir / f'{prefix}_gradcam_rgb.png')
    Image.fromarray(overlay_np).save(output_dir / f'{prefix}_gradcam_overlay.png')

    # ── Save report grid ──────────────────────────────────────────────────────
    from inference import REGION_NAMES  # noqa: PLC0415
    grid_path = save_report_grid(
        output_dir=output_dir,
        prefix=prefix,
        rgb_np=rgb_display,
        ela_np=ela_display,
        heatmap_np=heatmap_np,
        overlay_np=overlay_np,
        predicted_label=predicted_label,
        target_label=target_label,
        confidence=predicted_confidence,
        aux_scores=aux_scores,
        region_names=REGION_NAMES,
        dpi=args.dpi,
    )

    print('Saved:')
    print(f'  {output_dir / f"{prefix}_gradcam_rgb.png"}')
    print(f'  {output_dir / f"{prefix}_gradcam_overlay.png"}')
    print(f'  {grid_path}')

    # Print prediction summary
    print('\nClass probabilities:')
    for i, name in enumerate(label_names):
        print(f'  {name:<20} {float(probabilities[i])*100:.2f}%')
    print('Regional head scores:')
    for name, score in zip(REGION_NAMES, aux_scores):
        print(f'  {name:<20} {score*100:.2f}%')


if __name__ == '__main__':
    main()
