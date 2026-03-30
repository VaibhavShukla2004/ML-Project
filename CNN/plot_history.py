import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'results'
DEFAULT_HISTORY_FILE = DEFAULT_RESULTS_DIR / 'training_history.json'


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Plot two-phase curriculum metrics from training history.'
    )
    parser.add_argument(
        '--history-file',
        type=str,
        default=str(DEFAULT_HISTORY_FILE),
        help='Path to JSON containing phase1 and phase2 history.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help='Directory where plots are saved.',
    )
    parser.add_argument(
        '--phase2-lr',
        type=float,
        default=1e-5,
        help='Learning rate used in Phase 2 for annotation text.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure resolution (DPI).',
    )
    return parser


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def load_history(history_file):
    history_path = Path(history_file)
    if history_path.exists():
        history = _load_json(history_path)
        phase1 = history.get('phase1', {})
        phase2 = history.get('phase2', {})
        return phase1, phase2, history_path

    # Fallback to per-phase files emitted by train.py.
    results_dir = history_path.parent if history_path.parent.exists() else DEFAULT_RESULTS_DIR
    p1_path = results_dir / 'history_phase1.json'
    p2_path = results_dir / 'history_phase2.json'

    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(
            f'History not found at {history_path}, and fallback files are missing: '
            f'{p1_path} / {p2_path}'
        )

    return _load_json(p1_path), _load_json(p2_path), history_path


def _validate_phase(phase_name, phase_dict):
    keys = ('train_loss', 'val_loss', 'train_acc', 'val_acc')
    missing = [k for k in keys if k not in phase_dict]
    if missing:
        raise ValueError(f'{phase_name} is missing required keys: {missing}')

    lengths = {k: len(phase_dict[k]) for k in keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f'{phase_name} metric lengths do not match: {lengths}')



def save_combined_metrics_plot(output_dir, phase1, phase2, phase2_lr, dpi):
    p1_epochs = len(phase1['train_loss'])
    p2_epochs = len(phase2['train_loss'])
    total_epochs = p1_epochs + p2_epochs

    train_loss = phase1['train_loss'] + phase2['train_loss']
    val_loss = phase1['val_loss'] + phase2['val_loss']
    train_acc = phase1['train_acc'] + phase2['train_acc']
    val_acc = phase1['val_acc'] + phase2['val_acc']

    epochs = list(range(1, total_epochs + 1))

    sns.set_theme(style='whitegrid', context='talk')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)

    # Loss panel
    ax1.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2.2)
    ax1.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2.2)
    if p1_epochs > 0 and p2_epochs > 0:
        ax1.axvline(
            x=p1_epochs,
            color='#b22222',
            linestyle='--',
            linewidth=1.8,
            alpha=0.9,
            label='Backbone Unfrozen',
        )

    ymax_loss = max(train_loss + val_loss)
    if p1_epochs > 0:
        ax1.text(
            max(1, p1_epochs / 2.0),
            ymax_loss * 0.95,
            'Phase 1\nFrozen backbone',
            ha='center',
            va='top',
            fontsize=10,
            bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'},
        )
    if p2_epochs > 0:
        ax1.text(
            p1_epochs + max(1, p2_epochs / 2.0),
            ymax_loss * 0.95,
            f'Phase 2\nFine-tuning @ LR={phase2_lr:.0e}',
            ha='center',
            va='top',
            fontsize=10,
            bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'},
        )

    ax1.set_title('Two-Phase Loss Optimization', fontsize=14, pad=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', frameon=True)

    # Accuracy panel
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='#2ca02c', linewidth=2.2)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='#d62728', linewidth=2.2)
    if p1_epochs > 0 and p2_epochs > 0:
        ax2.axvline(
            x=p1_epochs,
            color='#b22222',
            linestyle='--',
            linewidth=1.8,
            alpha=0.9,
            label='Backbone Unfrozen',
        )
    ax2.set_title('Two-Phase Accuracy Tracking', fontsize=14, pad=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    save_path = output_dir / 'two_phase_curriculum_metrics.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return save_path


def save_phase_loss_plot(output_dir, phase_name, phase_data, subtitle, dpi):
    epochs = list(range(1, len(phase_data['train_loss']) + 1))

    sns.set_theme(style='whitegrid', context='talk')
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=dpi)
    ax.plot(epochs, phase_data['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2.2)
    ax.plot(epochs, phase_data['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.2)

    ax.set_title(f'{phase_name} Loss Curve', fontsize=14, pad=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.text(
        0.98,
        0.96,
        subtitle,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
    )
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    filename = f'{phase_name.lower().replace(" ", "_")}_loss_curve.png'
    save_path = output_dir / filename
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return save_path


def main():
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1, phase2, source_path = load_history(args.history_file)
    _validate_phase('phase1', phase1)
    _validate_phase('phase2', phase2)

    total_epochs = len(phase1['train_loss']) + len(phase2['train_loss'])
    if total_epochs == 0:
        print('No epoch metrics found. Nothing to plot.')
        return

    combined_path = save_combined_metrics_plot(
        output_dir=output_dir,
        phase1=phase1,
        phase2=phase2,
        phase2_lr=args.phase2_lr,
        dpi=args.dpi,
    )

    phase1_path = save_phase_loss_plot(
        output_dir=output_dir,
        phase_name='Phase 1',
        phase_data=phase1,
        subtitle='Frozen backbone\nHead adaptation stage',
        dpi=args.dpi,
    )

    phase2_path = save_phase_loss_plot(
        output_dir=output_dir,
        phase_name='Phase 2',
        phase_data=phase2,
        subtitle=f'Full fine-tuning\nMicro LR: {args.phase2_lr:.0e}',
        dpi=args.dpi,
    )

    print(f'Loaded history from: {source_path}')
    print('Saved plots:')
    print(f'  {combined_path}')
    print(f'  {phase1_path}')
    print(f'  {phase2_path}')


if __name__ == '__main__':
    main()
