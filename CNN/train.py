import os
import json
import random
import argparse
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.profiler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

try:
    from model import build_model
    from data_pipeline import build_pipeline
except ImportError:
    from CNN.model import build_model
    from CNN.data_pipeline import build_pipeline

# ─── Label names for reporting ───────────────────────────────────────────────
LABEL_NAMES = ['authentic', 'class1_photo', 'class2_name', 'class4_overlay']
CLASS_LABEL_IDS = list(range(len(LABEL_NAMES)))


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting."""

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def configure_gpu_runtime(device, deterministic=False, allow_tf32=True, matmul_precision='high'):
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(matmul_precision)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_amp_dtype(amp_dtype):
    if amp_dtype == 'bfloat16':
        return torch.bfloat16
    return torch.float16


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Train document forgery CNN with optional L4 GPU optimizations.')
    parser.add_argument('--authentic-dir', default='synthetic/generated/authentic')
    parser.add_argument('--forged-dir', default='synthetic/generated/forged')
    parser.add_argument(
        '--target-size', type=int, nargs=2, default=[800, 600],
        metavar=('HEIGHT', 'WIDTH'),
        help='Resize target (H W) for full-card input. Default: 800 600 (L4-optimised).',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=min(8, max(2, (os.cpu_count() or 2) // 2)))
    parser.add_argument('--train-augmentation', choices=['full', 'light', 'none'], default='full', help='CPU-side train augmentation intensity.')
    parser.add_argument('--use-ela', action='store_true', help='Append ELA as a 4th input channel (RGB+ELA).')
    parser.add_argument('--ela-quality', type=int, default=90, help='JPEG quality used when computing ELA channel.')
    parser.add_argument('--ela-scale', type=float, default=12.0, help='Intensity scaling factor for ELA channel.')
    parser.add_argument('--prefetch-factor', type=int, default=4, help='DataLoader prefetch factor (effective when num_workers > 0).')
    parser.add_argument('--no-persistent-workers', action='store_true', help='Disable DataLoader persistent workers.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Resume phase checkpoint if available.')
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--phase1-epochs', type=int, default=10)
    parser.add_argument('--phase2-epochs', type=int, default=20)
    parser.add_argument('--phase1-lr', type=float, default=1e-3)
    parser.add_argument('--phase2-lr', type=float, default=1e-5)
    parser.add_argument('--loss-type', choices=['focal', 'ce'], default='focal', help='Loss function to optimize.')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma for focal loss hard-example focusing.')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).')
    parser.add_argument('--phase1-stem-trainable', action='store_true', help='Keep stem trainable in Phase 1 (higher VRAM, useful for ELA adaptation).')
    parser.add_argument('--disable-phase1-backbone-no-grad', action='store_true', help='Disable Phase-1 memory-saving no-grad backbone path.')
    parser.add_argument('--use-amp', action='store_true', help='Enable mixed precision training on CUDA.')
    parser.add_argument('--amp-dtype', choices=['float16', 'bfloat16'], default='float16')
    parser.add_argument('--channels-last', action='store_true', help='Use channels-last memory format on CUDA.')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile to reduce compile/runtime overhead and instability.')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode (slower, reproducible).')
    parser.add_argument('--disable-tf32', action='store_true', help='Disable TF32 acceleration on supported NVIDIA GPUs.')
    parser.add_argument('--profile', action='store_true', help='Profile first epoch (5 batches) with torch.profiler; outputs to ./profiler_logs.')
    return parser


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)


# ─── Core training function for one epoch ────────────────────────────────────
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp=False,
    amp_dtype=torch.float16,
    scaler=None,
    use_channels_last=False,
    enable_profiler=False,
    grad_accum_steps=1,
    forward_fn=None,
):
    if len(loader) == 0:
        raise ValueError("train_loader is empty. Check split ratios and dataset generation.")

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batches_processed = 0
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=amp_dtype)
        if use_amp and device.type == 'cuda'
        else nullcontext()
    )

    profiler_ctx = None
    if enable_profiler and device.type == 'cuda':
        os.makedirs('profiler_logs', exist_ok=True)
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            acc_events=True,
            record_shapes=True,
            with_stack=True,
        )
        profiler_ctx.__enter__()

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(loader):
        batches_processed += 1
        images = images.to(device, non_blocking=(device.type == 'cuda'))
        if use_channels_last and device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=(device.type == 'cuda'))

        with autocast_ctx:
            outputs = forward_fn(images) if forward_fn is not None else model(images)
            raw_loss = criterion(outputs, labels)
            loss = raw_loss / grad_accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()   # Backpropagation
        else:
            loss.backward()                 # Backpropagation

        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))
        if should_step:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)          # Update weights
                scaler.update()
            else:
                optimizer.step()                # Update weights
            optimizer.zero_grad(set_to_none=True)

        if enable_profiler and profiler_ctx is not None:
            profiler_ctx.step()

        running_loss += raw_loss.item()
        _, predicted = outputs.max(1)   # Take the class with highest logit
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress indicator every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} "
                  f"| Loss: {running_loss / (batch_idx + 1):.4f} "
                  f"| Acc: {100. * correct / total:.1f}%")

        # Exit early during profiling after sampling ~5 batches
        if enable_profiler and batch_idx >= 5:
            print("Profiling complete. Exiting training snapshot.")
            break

    epoch_loss = running_loss / batches_processed
    epoch_acc = 100. * correct / total
    
    if enable_profiler and profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)

    return epoch_loss, epoch_acc


# ─── Validation / evaluation function ────────────────────────────────────────
def evaluate(model, loader, criterion, device, use_amp=False, amp_dtype=torch.float16, use_channels_last=False):
    if len(loader) == 0:
        raise ValueError("evaluation loader is empty. Check split ratios and dataset generation.")

    model.eval()    # Disables dropout, batch norm uses running stats
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=amp_dtype)
        if use_amp and device.type == 'cuda'
        else nullcontext()
    )

    with torch.no_grad():   # No gradient computation needed for evaluation
        for images, labels in loader:
            images = images.to(device, non_blocking=(device.type == 'cuda'))
            if use_channels_last and device.type == 'cuda':
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=(device.type == 'cuda'))

            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


# ─── Confusion matrix plot ────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, phase_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds, labels=CLASS_LABEL_IDS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.title(f'Confusion Matrix — {phase_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{phase_name}.png'))
    plt.close()
    print(f"Confusion matrix saved to results/confusion_matrix_{phase_name}.png")


# ─── Training phase runner ────────────────────────────────────────────────────
def run_phase(
    phase_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    use_amp=False,
    amp_dtype=torch.float16,
    scaler=None,
    use_channels_last=False,
    enable_profiler=False,
    grad_accum_steps=1,
    forward_fn=None,
    early_stopping_patience=5,
    resume=False,
    save_dir='results'
):
    os.makedirs(save_dir, exist_ok=True)
    last_checkpoint_path = os.path.join(save_dir, f'last_checkpoint_{phase_name}.pth')
    history_path = os.path.join(save_dir, f'history_{phase_name}.json')

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    val_preds, val_labels = None, None
    epochs_without_improvement = 0
    start_epoch = 1

    if resume and os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', history)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming {phase_name} from epoch {start_epoch}")

    if start_epoch > num_epochs:
        print(f"[{phase_name}] Already completed up to epoch {num_epochs}. Skipping training loop.")
        save_json(history_path, history)
        return history

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n[{phase_name}] Epoch {epoch}/{num_epochs}")
        print("-" * 50)

        profile_this_epoch = enable_profiler and epoch == start_epoch

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            use_channels_last=use_channels_last,
            enable_profiler=profile_this_epoch,
            grad_accum_steps=grad_accum_steps,
            forward_fn=forward_fn,
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
        )

        # Step the scheduler based on validation loss
        # ReduceLROnPlateau halves LR when val_loss stops improving
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(save_dir, f'best_model_{phase_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'history': history,
            }, checkpoint_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
            'epochs_without_improvement': epochs_without_improvement,
        }, last_checkpoint_path)

        save_json(history_path, history)

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"  Early stopping triggered after {epochs_without_improvement} "
                f"non-improving epoch(s)."
            )
            break

    # Print full classification report at end of phase
    if val_preds is not None and val_labels is not None:
        print(f"\n[{phase_name}] Final Classification Report:")
        print(
            classification_report(
                val_labels,
                val_preds,
                labels=CLASS_LABEL_IDS,
                target_names=LABEL_NAMES,
                zero_division=0,
            )
        )
        plot_confusion_matrix(val_labels, val_preds, phase_name, save_dir)
    else:
        print(f"\n[{phase_name}] Skipping report: no epochs were run.")

    save_json(history_path, history)

    return history


# ─── Main entry point ─────────────────────────────────────────────────────────
def main():
    args = build_arg_parser().parse_args()
    seed = args.seed
    set_global_seed(seed, deterministic=args.deterministic)

    # ── Build data pipeline ──────────────────────────────────────────────────
    print("Building data pipeline...")
    pipeline = build_pipeline(
        authentic_dir=args.authentic_dir,
        forged_dir=args.forged_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        seed=seed,
        target_size=tuple(args.target_size),
        train_augmentation=args.train_augmentation,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        use_ela=args.use_ela,
        ela_quality=args.ela_quality,
        ela_scale=args.ela_scale,
    )
    train_loader = pipeline['train_loader']
    val_loader   = pipeline['val_loader']
    test_loader  = pipeline['test_loader']
    class_weights = pipeline['class_weights']

    # ── Build model ──────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model, device = build_model(
        num_classes=4,
        dropout_rate=0.3,
        in_channels=pipeline['in_channels'],
    )
    configure_gpu_runtime(
        device=device,
        deterministic=args.deterministic,
        allow_tf32=not args.disable_tf32,
        matmul_precision='high',
    )

    amp_enabled = args.use_amp and device.type == 'cuda'
    amp_dtype = get_amp_dtype(args.amp_dtype)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if device.type == 'cuda' else None

    if args.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # JIT-compile model for ~15-30% speedup on CUDA (compilation overhead on first epoch)
    if device.type == 'cuda' and not args.no_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            compile_enabled = True
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing without compilation.")
            compile_enabled = False
    else:
        compile_enabled = False

    print(
        f"Runtime config | device={device} | amp={amp_enabled} ({args.amp_dtype}) "
        f"| channels_last={args.channels_last and device.type == 'cuda'} "
        f"| tf32={not args.disable_tf32 and device.type == 'cuda'} "
        f"| torch.compile={compile_enabled} "
        f"| train_aug={args.train_augmentation} "
        f"| use_ela={args.use_ela} "
        f"| ela_quality={args.ela_quality} "
        f"| ela_scale={args.ela_scale} "
        f"| grad_accum_steps={args.grad_accum_steps} "
        f"| persistent_workers={not args.no_persistent_workers} "
        f"| prefetch_factor={args.prefetch_factor} "
        f"| workers={args.num_workers} | batch_size={args.batch_size}"
    )

    class_weights = class_weights.to(device)
    if args.loss_type == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Loss config | type={args.loss_type} | focal_gamma={args.focal_gamma if args.loss_type == 'focal' else 'n/a'}")

    phase1_keep_stem_trainable = args.phase1_stem_trainable
    phase1_backbone_no_grad = (not phase1_keep_stem_trainable) and (not args.disable_phase1_backbone_no_grad)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Train classifier head with frozen backbone.
    # When RGB+ELA is enabled, also keep the stem trainable so the new channel
    # is integrated before full-network fine-tuning.
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    if phase1_keep_stem_trainable:
        print("PHASE 1: Training classifier head + stem")
    elif phase1_backbone_no_grad:
        print("PHASE 1: Training classifier head (frozen backbone in no-grad mode)")
    else:
        print("PHASE 1: Training classifier head (backbone frozen)")
    print("="*60)

    model.freeze_backbone(keep_stem_trainable=phase1_keep_stem_trainable)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Phase 1 trainable parameters: {trainable:,} / {total:,} "
        f"| stem_trainable={phase1_keep_stem_trainable} "
        f"| backbone_no_grad={phase1_backbone_no_grad}"
    )

    phase1_forward_fn = model.forward_head_with_frozen_features if phase1_backbone_no_grad else None

    # Higher LR is safe here — only the small head is being updated
    optimizer_p1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr
    )
    scheduler_p1 = ReduceLROnPlateau(
        optimizer_p1, mode='min', factor=0.5, patience=3
    )

    history_p1 = run_phase(
        phase_name='phase1',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        device=device,
        num_epochs=args.phase1_epochs,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        use_channels_last=args.channels_last,
        enable_profiler=args.profile,
        grad_accum_steps=args.grad_accum_steps,
        forward_fn=phase1_forward_fn,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume,
    )

    # Load best Phase 1 weights before starting Phase 2
    checkpoint = torch.load('results/best_model_phase1.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best Phase 1 model (val_loss: {checkpoint['val_loss']:.4f})")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune entire network — backbone unfrozen
    # Goal: Allow backbone to adapt its features to document forgery detection
    # Use a much lower LR to avoid catastrophic forgetting
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning full network (backbone unfrozen)")
    print("="*60)

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable == total, "Backbone is not fully unfrozen before Phase 2"

    # Much lower LR — backbone weights are precious, nudge don't shove
    optimizer_p2 = torch.optim.Adam(
        model.parameters(),
        lr=args.phase2_lr
    )
    scheduler_p2 = ReduceLROnPlateau(
        optimizer_p2, mode='min', factor=0.5, patience=3
    )

    history_p2 = run_phase(
        phase_name='phase2',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        device=device,
        num_epochs=args.phase2_epochs,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        use_channels_last=args.channels_last,
        enable_profiler=False,
        grad_accum_steps=args.grad_accum_steps,
        forward_fn=None,
        early_stopping_patience=args.early_stopping_patience,
        resume=args.resume,
    )

    # ════════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION on held-out test set
    # This is the only time test_loader is ever used — not during training
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    best_model_checkpoint = torch.load(
        'results/best_model_phase2.pth', map_location=device
    )
    model.load_state_dict(best_model_checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
        use_channels_last=args.channels_last,
    )

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.1f}%")
    print("\nFinal Classification Report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            labels=CLASS_LABEL_IDS,
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    plot_confusion_matrix(test_labels, test_preds, 'final_test')

    save_json(
        os.path.join('results', 'training_history.json'),
        {'phase1': history_p1, 'phase2': history_p2}
    )


if __name__ == "__main__":
    main()