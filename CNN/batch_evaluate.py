"""
Batch adversarial evaluation suite.

Processes entire adversarial directories in a single pass, computes the same
classification_report and confusion_matrix used during training, saves Grad-CAM
overlays for mispredictions only, and writes a per-attack breakdown + CSV.

True-label inference order (per image):
  1. --true-label flag bound to the image's --input-dir  (most explicit)
  2. Parent directory name contains a known class label string
     (e.g.  adversarial_authentic  →  authentic)
  3. Original filename stem before __adv_ prefix-matches a known class label
     (e.g.  class1_photo_007__adv_03__glare_jpeg.png  →  class1_photo)

Filename convention produced by adversarial_generator.py:
    {original_stem}__adv_{index:02d}__{recipe_name}.png

Usage examples
--------------
  # False-positive stress test (authentic images — should stay 'authentic')
  python CNN/batch_evaluate.py \\
      --input-dir data/synthetic/generated/adversarial_authentic \\
      --true-label authentic

  # Both authentic and forged batches in one report
  python CNN/batch_evaluate.py \\
      --input-dir data/synthetic/generated/adversarial_authentic \\
      --true-label authentic \\
      --input-dir data/synthetic/generated/adversarial_forged \\
      --true-label class1_photo

  # Auto-detect true labels from directory / filename (no --true-label needed)
  python CNN/batch_evaluate.py \\
      --input-dir data/synthetic/generated/adversarial_authentic \\
      --input-dir data/synthetic/generated/adversarial_forged
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

try:
    from inference import (
        DEFAULT_CHECKPOINT,
        REGION_NAMES,
        blend_overlay,
        compute_gradcam,
        heatmap_to_rgb,
        infer_channel_flags,
        load_checkpoint,
        preprocess_image,
        resolve_ela_params,
        resolve_label_names,
        resolve_target_size,
    )
    from model import build_model
except ImportError:
    from CNN.inference import (
        DEFAULT_CHECKPOINT,
        REGION_NAMES,
        blend_overlay,
        compute_gradcam,
        heatmap_to_rgb,
        infer_channel_flags,
        load_checkpoint,
        preprocess_image,
        resolve_ela_params,
        resolve_label_names,
        resolve_target_size,
    )
    from CNN.model import build_model


# Matches filenames produced by adversarial_generator.py:
#   {original_stem}__adv_{nn}__{recipe_name}
ADV_PATTERN = re.compile(r'^(.+)__adv_\d+__([a-zA-Z_]+)$')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'adversarial_eval'
LABEL_ALIASES = {
    'class1': 'class1_photo',
    'class2': 'class2_name',
    'class4': 'class4_overlay',
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Batch adversarial evaluation: loops through adversarial directories, '
            'outputs classification_report + confusion_matrix, saves Grad-CAM overlays '
            'for mispredictions only, and writes a per-attack breakdown CSV.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input-dir',
        dest='input_dirs',
        action='append',
        default=[],
        metavar='DIR',
        required=True,
        help='Adversarial image directory to evaluate. Repeat for multiple directories.',
    )
    parser.add_argument(
        '--true-label',
        dest='true_labels',
        action='append',
        default=[],
        metavar='LABEL',
        help=(
            'Ground-truth class for the corresponding --input-dir. '
            'Repeat to match each --input-dir, or omit entirely to auto-detect.'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        default=str(DEFAULT_CHECKPOINT),
        help='Path to trained checkpoint (.pth).',
    )
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Root output directory (CSV, report, confusion matrix, overlays).',
    )
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Resize target H W. Defaults to checkpoint config or 800 600.',
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

    return parser


# ---------------------------------------------------------------------------
# Label / attack-type inference
# ---------------------------------------------------------------------------

def normalize_label_name(raw_label, label_names):
    if raw_label is None:
        return None

    lowered = raw_label.lower()
    canonical_map = {label.lower(): label for label in label_names}
    if lowered in canonical_map:
        return canonical_map[lowered]

    alias_target = LABEL_ALIASES.get(lowered)
    if alias_target is None:
        return None
    return canonical_map.get(alias_target.lower())


def match_label_from_text(text, label_names, require_prefix=False):
    lowered = text.lower()

    for label in sorted(label_names, key=len, reverse=True):
        candidate = label.lower()
        if require_prefix:
            if lowered.startswith(candidate):
                return label
        elif candidate in lowered:
            return label

    for alias, canonical in sorted(LABEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if require_prefix:
            if lowered.startswith(alias):
                return normalize_label_name(canonical, label_names)
        elif alias in lowered:
            return normalize_label_name(canonical, label_names)

    return None

def infer_true_label_for_image(image_path, dir_true_label, label_names):
    """
    Three-tier cascade returning the true class string, or None on failure.

    Tier 1 — explicit CLI label bound to this directory.
    Tier 2 — parent directory name contains a known class label
              (adversarial_authentic  →  authentic).
    Tier 3 — original filename stem before __adv_ prefix-matches a label
              (class1_photo_007__adv_03__glare_jpeg.png  →  class1_photo,
               class1_0026__src__...png  →  class1_photo).
    """
    if dir_true_label is not None:
        normalized = normalize_label_name(dir_true_label, label_names)
        return normalized if normalized is not None else dir_true_label

    dir_name = image_path.parent.name.lower()
    matched_label = match_label_from_text(dir_name, label_names, require_prefix=False)
    if matched_label is not None:
        return matched_label

    match = ADV_PATTERN.match(image_path.stem)
    if match:
        original_stem = match.group(1).lower()
        return match_label_from_text(original_stem, label_names, require_prefix=True)

    return None


def parse_attack_type(image_stem):
    """Extract recipe name from __adv_NN__recipe_name suffix, or 'clean'."""
    match = ADV_PATTERN.match(image_stem)
    return match.group(2) if match else 'clean'


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images(input_dirs, true_labels, label_names):
    """
    Returns list of (image_path, true_label_or_None) across all input directories.
    true_labels must be either empty (auto-detect all) or the same length as input_dirs.
    """
    items = []
    for idx, dir_path in enumerate(input_dirs):
        dir_label = true_labels[idx] if idx < len(true_labels) else None
        for image_path in sorted(dir_path.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = infer_true_label_for_image(image_path, dir_label, label_names)
            items.append((image_path, label))
    return items


# ---------------------------------------------------------------------------
# Overlay saving (mispredictions only)
# ---------------------------------------------------------------------------

def save_misprediction_overlay(model, input_tensor, predicted_index, resized_rgb, output_path):
    heatmap, _ = compute_gradcam(model, input_tensor, predicted_index)
    heatmap_rgb = heatmap_to_rgb(heatmap)
    overlay = blend_overlay(resized_rgb, heatmap_rgb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(cm, label_names, output_path):
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), max(5, n * 1.6)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', xticks_rotation='vertical')
    ax.set_title('Adversarial Evaluation — Confusion Matrix', fontsize=13, pad=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_per_attack_table(results):
    attack_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for row in results:
        key = row['attack_type']
        attack_stats[key]['total'] += 1
        if row['correct']:
            attack_stats[key]['correct'] += 1

    sorted_attacks = sorted(attack_stats.items(), key=lambda kv: kv[1]['total'], reverse=True)
    col_w = max((len(k) for k in attack_stats), default=10) + 2
    sep = '-' * (col_w + 32)

    print('\nPer-Attack Accuracy')
    print(sep)
    print(f"{'Attack':<{col_w}}  {'Total':>6}  {'Correct':>7}  {'Accuracy':>9}")
    print(sep)

    grand_total = grand_correct = 0
    for attack, stats in sorted_attacks:
        total = stats['total']
        correct = stats['correct']
        acc = 100.0 * correct / total if total > 0 else 0.0
        grand_total += total
        grand_correct += correct
        print(f'{attack:<{col_w}}  {total:>6}  {correct:>7}  {acc:>8.1f}%')

    print(sep)
    grand_acc = 100.0 * grand_correct / grand_total if grand_total > 0 else 0.0
    print(f"{'TOTAL':<{col_w}}  {grand_total:>6}  {grand_correct:>7}  {grand_acc:>8.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()

    if args.true_labels and len(args.true_labels) != len(args.input_dirs):
        print(
            f'ERROR: --true-label count ({len(args.true_labels)}) must match '
            f'--input-dir count ({len(args.input_dirs)}) when specified.',
            file=sys.stderr,
        )
        sys.exit(1)

    input_dirs = []
    for raw in args.input_dirs:
        p = Path(raw)
        if not p.is_dir():
            print(f'ERROR: input directory not found: {p}', file=sys.stderr)
            sys.exit(1)
        input_dirs.append(p)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mispredictions_dir = output_dir / 'mispredictions'

    # --- Load model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f'ERROR: checkpoint not found: {checkpoint_path}', file=sys.stderr)
        sys.exit(1)

    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint, state_dict, in_channels = load_checkpoint(checkpoint_path, device)
    checkpoint_config = checkpoint.get('config')
    label_names = resolve_label_names(checkpoint_config)
    target_size = resolve_target_size(checkpoint_config, args.target_size)
    ela_quality, ela_scale = resolve_ela_params(checkpoint_config, args.ela_quality, args.ela_scale)
    use_ela, use_coord_channels = infer_channel_flags(
        in_channels, checkpoint_config, args.use_ela, args.use_coord_channels
    )
    label_to_index = {name: i for i, name in enumerate(label_names)}

    model, device = build_model(
        num_classes=len(label_names),
        dropout_rate=0.3,
        in_channels=in_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # --- Collect images ---
    all_items = collect_images(input_dirs, args.true_labels, label_names)
    valid_items = [
        (p, l) for p, l in all_items
        if l is not None and l in label_to_index
    ]
    n_skipped = len(all_items) - len(valid_items)
    if n_skipped:
        print(f'WARNING: {n_skipped} image(s) skipped — uninferable or unknown true label.')
    if not valid_items:
        print('ERROR: no images with valid true labels found.', file=sys.stderr)
        sys.exit(1)

    print(f'Evaluating {len(valid_items)} images on {device} ...\n')

    # --- Inference loop ---
    results = []
    for image_path, true_label in valid_items:
        image = Image.open(image_path).convert('RGB')
        resized_rgb = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        input_tensor = preprocess_image(
            image,
            target_size=target_size,
            use_ela=use_ela,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
            use_coord_channels=use_coord_channels,
        ).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs['fusion_logits'], dim=1)[0]
            class_index = int(probs.argmax().item())
            predicted_class = label_names[class_index]
            confidence = float(probs[class_index].item())

        correct = (class_index == label_to_index[true_label])
        attack_type = parse_attack_type(image_path.stem)

        overlay_path_str = ''
        if not correct:
            # Grad-CAM only for failures — shows what confused the model
            overlay_path = mispredictions_dir / f'{image_path.stem}_overlay.png'
            save_misprediction_overlay(model, input_tensor, class_index, resized_rgb, overlay_path)
            overlay_path_str = str(overlay_path)

        results.append({
            'filename': image_path.name,
            'input_dir': str(image_path.parent),
            'true_label': true_label,
            'predicted_class': predicted_class,
            'confidence': f'{confidence:.4f}',
            'attack_type': attack_type,
            'correct': correct,
            'overlay_path': overlay_path_str,
        })

    # --- Aggregate metrics ---
    y_true = [r['true_label'] for r in results]
    y_pred = [r['predicted_class'] for r in results]
    present_labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda x: label_names.index(x) if x in label_names else 999,
    )

    report_str = classification_report(y_true, y_pred, labels=present_labels, zero_division=0)
    print('Classification Report')
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_path = output_dir / 'confusion_matrix.png'
    save_confusion_matrix_plot(cm, present_labels, cm_path)
    print(f'Confusion matrix saved: {cm_path}')

    print_per_attack_table(results)

    n_wrong = sum(1 for r in results if not r['correct'])
    print(
        f'\nMispredictions: {n_wrong} / {len(results)}'
        f'  ({100.0 * n_wrong / len(results):.1f}% error rate)'
    )
    if n_wrong:
        print(f'Grad-CAM overlays for failures → {mispredictions_dir}')

    # --- Write outputs ---
    csv_path = output_dir / 'results.csv'
    fieldnames = [
        'filename', 'input_dir', 'true_label', 'predicted_class',
        'confidence', 'attack_type', 'correct', 'overlay_path',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f'Results CSV:           {csv_path}')

    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as fp:
        fp.write('Classification Report\n')
        fp.write(report_str)
        fp.write('\n\nConfusion Matrix (rows=true, cols=predicted)\n')
        fp.write('Labels: ' + ', '.join(present_labels) + '\n')
        fp.write(np.array2string(cm))
    print(f'Classification report: {report_path}')


if __name__ == '__main__':
    main()
