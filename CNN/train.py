import os
import json
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    from model import build_model
    from data_pipeline import build_pipeline
except ImportError:
    from CNN.model import build_model
    from CNN.data_pipeline import build_pipeline

# ─── Label names for reporting ───────────────────────────────────────────────
LABEL_NAMES = ['authentic', 'class1_photo', 'class2_name', 'class4_overlay']
CLASS_LABEL_IDS = list(range(len(LABEL_NAMES)))


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)


# ─── Core training function for one epoch ────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    if len(loader) == 0:
        raise ValueError("train_loader is empty. Check split ratios and dataset generation.")

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()           # Clear gradients from previous batch
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)   # Take the class with highest logit
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress indicator every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} "
                  f"| Loss: {running_loss / (batch_idx + 1):.4f} "
                  f"| Acc: {100. * correct / total:.1f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ─── Validation / evaluation function ────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    if len(loader) == 0:
        raise ValueError("evaluation loader is empty. Check split ratios and dataset generation.")

    model.eval()    # Disables dropout, batch norm uses running stats
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():   # No gradient computation needed for evaluation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

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

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
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
    seed = 42
    set_global_seed(seed)
    # Set to False when changing model architecture or dataset.
    # Keep True only for strict same-run continuation compatibility.
    resume_training = False
    early_stopping_patience = 5

    # ── Build data pipeline ──────────────────────────────────────────────────
    print("Building data pipeline...")
    pipeline = build_pipeline(
        authentic_dir='synthetic/generated/authentic',
        forged_dir='synthetic/generated/forged',
        batch_size=32,
        num_workers=0,
        seed=seed,
    )
    train_loader = pipeline['train_loader']
    val_loader   = pipeline['val_loader']
    test_loader  = pipeline['test_loader']
    class_weights = pipeline['class_weights']

    # ── Build model ──────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model, device = build_model(num_classes=4, dropout_rate=0.3)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Train classifier head only — backbone frozen
    # Goal: Get the head to a reasonable state before touching backbone weights
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("="*60)

    model.freeze_backbone()

    # Higher LR is safe here — only the small head is being updated
    optimizer_p1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
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
        num_epochs=10,
        early_stopping_patience=early_stopping_patience,
        resume=resume_training,
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
        lr=1e-5
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
        num_epochs=20,
        early_stopping_patience=early_stopping_patience,
        resume=resume_training,
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
        model, test_loader, criterion, device
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