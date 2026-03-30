import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from data_pipeline import make_eval_transform
    from model import build_model
except ImportError:
    from CNN.data_pipeline import make_eval_transform
    from CNN.model import build_model


DEFAULT_LABEL_NAMES = ['authentic', 'class1_photo', 'class2_name', 'class4_overlay']
DEFAULT_TARGET_SIZE = (800, 600)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / 'results' / 'best_model_phase2.pth'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'inference'
REGION_NAMES = ('photo', 'name', 'expiry')


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Run single-image document forgery inference with Grad-CAM.'
    )
    parser.add_argument('--image', required=True, help='Path to a single image.')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT), help='Path to a trained checkpoint.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory for prediction artifacts.')
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Resize target (H W). Defaults to checkpoint config or 800 600.',
    )
    parser.add_argument('--ela-quality', type=int, default=None, help='JPEG quality for ELA computation.')
    parser.add_argument('--ela-scale', type=float, default=None, help='Scale factor for ELA intensity.')

    ela_group = parser.add_mutually_exclusive_group()
    ela_group.add_argument('--use-ela', dest='use_ela', action='store_true', help='Append ELA channel.')
    ela_group.add_argument('--no-ela', dest='use_ela', action='store_false', help='Disable ELA channel.')
    parser.set_defaults(use_ela=None)

    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument(
        '--use-coord-channels',
        dest='use_coord_channels',
        action='store_true',
        help='Append X/Y coordinate channels.',
    )
    coord_group.add_argument(
        '--no-coord-channels',
        dest='use_coord_channels',
        action='store_false',
        help='Disable X/Y coordinate channels.',
    )
    parser.set_defaults(use_coord_channels=None)
    return parser


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    stem_shape = tuple(state_dict['backbone.features.0.0.weight'].shape)
    return checkpoint, state_dict, stem_shape[1]


def infer_channel_flags(in_channels, checkpoint_config, cli_use_ela, cli_use_coord_channels):
    inferred_by_channels = {
        3: (False, False),
        4: (True, False),
        5: (False, True),
        6: (True, True),
    }
    if in_channels not in inferred_by_channels:
        raise ValueError(f'Unsupported input channel count in checkpoint: {in_channels}')

    default_use_ela, default_use_coord_channels = inferred_by_channels[in_channels]
    if checkpoint_config is not None:
        default_use_ela = checkpoint_config.get('use_ela', default_use_ela)
        default_use_coord_channels = checkpoint_config.get(
            'use_coord_channels', default_use_coord_channels
        )

    use_ela = default_use_ela if cli_use_ela is None else cli_use_ela
    use_coord_channels = (
        default_use_coord_channels if cli_use_coord_channels is None else cli_use_coord_channels
    )

    resolved_in_channels = 3 + int(use_ela) + (2 if use_coord_channels else 0)
    if resolved_in_channels != in_channels:
        raise ValueError(
            'Resolved preprocessing channels do not match checkpoint stem. '
            f'Checkpoint expects {in_channels}, but flags resolve to {resolved_in_channels}.'
        )

    return use_ela, use_coord_channels


def resolve_target_size(checkpoint_config, cli_target_size):
    if cli_target_size is not None:
        return tuple(cli_target_size)
    if checkpoint_config is not None and 'target_size' in checkpoint_config:
        return tuple(checkpoint_config['target_size'])
    return DEFAULT_TARGET_SIZE


def resolve_label_names(checkpoint_config):
    if checkpoint_config is not None and 'label_names' in checkpoint_config:
        return list(checkpoint_config['label_names'])
    return list(DEFAULT_LABEL_NAMES)


def resolve_ela_params(checkpoint_config, cli_quality, cli_scale):
    default_quality = 90
    default_scale = 12.0
    if checkpoint_config is not None:
        default_quality = checkpoint_config.get('ela_quality', default_quality)
        default_scale = checkpoint_config.get('ela_scale', default_scale)
    quality = default_quality if cli_quality is None else cli_quality
    scale = default_scale if cli_scale is None else cli_scale
    return quality, scale


def preprocess_image(image, target_size, use_ela, ela_quality, ela_scale, use_coord_channels):
    transform = make_eval_transform(
        target_size=target_size,
        use_ela=use_ela,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
        use_coord_channels=use_coord_channels,
    )
    return transform(image).unsqueeze(0)


def compute_gradcam(model, input_tensor, class_index):
    model.eval()
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        feature_map = model._extract_feature_map(input_tensor, freeze_backbone=False)
        feature_map.retain_grad()
        outputs = model._forward_from_feature_map(feature_map)
        logits = outputs['fusion_logits']
        selected_logit = logits[:, class_index].sum()
        selected_logit.backward()

    gradients = feature_map.grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    cam /= max(cam.max(), 1e-8)
    return cam, outputs


def heatmap_to_rgb(heatmap):
    rgba = plt.get_cmap('jet')(heatmap)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def blend_overlay(base_image, heatmap_rgb, alpha=0.42):
    base = np.asarray(base_image).astype(np.float32)
    overlay = heatmap_rgb.astype(np.float32)
    blended = np.clip((1.0 - alpha) * base + alpha * overlay, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def save_outputs(output_dir, image_stem, resized_rgb, heatmap, class_name, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    resized_path = output_dir / f'{image_stem}_resized.png'
    overlay_path = output_dir / f'{image_stem}_gradcam_overlay.png'
    json_path = output_dir / f'{image_stem}_prediction.json'

    heatmap_rgb = heatmap_to_rgb(heatmap)
    overlay_image = blend_overlay(resized_rgb, heatmap_rgb)

    resized_rgb.save(resized_path)
    overlay_image.save(overlay_path)
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)

    return resized_path, overlay_path, json_path, class_name


def main():
    args = build_arg_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if not image_path.is_file():
        raise FileNotFoundError(f'Image not found: {image_path}')

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

    model, device = build_model(
        num_classes=len(label_names),
        dropout_rate=0.3,
        in_channels=in_channels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    original_rgb = Image.open(image_path).convert('RGB')
    resized_rgb = original_rgb.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    input_tensor = preprocess_image(
        original_rgb,
        target_size=target_size,
        use_ela=use_ela,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
        use_coord_channels=use_coord_channels,
    ).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs['fusion_logits']
        probabilities = torch.softmax(logits, dim=1)[0]
        class_index = int(probabilities.argmax().item())
        class_name = label_names[class_index]
        confidence = float(probabilities[class_index].item())
        aux_scores = torch.sigmoid(outputs['aux_logits'])[0].detach().cpu().tolist()

    heatmap, _ = compute_gradcam(model, input_tensor, class_index)

    payload = {
        'image_path': str(image_path),
        'checkpoint_path': str(checkpoint_path),
        'predicted_class': class_name,
        'predicted_index': class_index,
        'confidence': confidence,
        'class_probabilities': {
            label_names[index]: float(probabilities[index].item())
            for index in range(len(label_names))
        },
        'aux_region_scores': {
            region_name: float(aux_scores[index])
            for index, region_name in enumerate(REGION_NAMES)
        },
        'preprocessing': {
            'target_size': list(target_size),
            'use_ela': use_ela,
            'ela_quality': ela_quality,
            'ela_scale': ela_scale,
            'use_coord_channels': use_coord_channels,
            'in_channels': in_channels,
        },
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_val_loss': checkpoint.get('val_loss'),
    }

    resized_path, overlay_path, json_path, _ = save_outputs(
        output_dir,
        image_path.stem,
        resized_rgb,
        heatmap,
        class_name,
        payload,
    )

    print(f'Predicted class: {class_name}')
    print(f'Confidence: {confidence * 100.0:.2f}%')
    print('Class probabilities:')
    for label_name, probability in payload['class_probabilities'].items():
        print(f'  {label_name}: {probability * 100.0:.2f}%')
    print('Auxiliary region scores:')
    for region_name, score in payload['aux_region_scores'].items():
        print(f'  {region_name}: {score * 100.0:.2f}%')
    print(f'Resized image:   {resized_path}')
    print(f'Grad-CAM overlay: {overlay_path}')
    print(f'Prediction JSON: {json_path}')


if __name__ == '__main__':
    main()