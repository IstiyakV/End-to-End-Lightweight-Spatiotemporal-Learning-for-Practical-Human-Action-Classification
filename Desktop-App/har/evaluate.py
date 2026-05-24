"""
Evaluation — confusion matrix, per-class metrics, figure generation.

Generates all data needed for paper:
    - Table IV: Per-class precision, recall, specificity, F1
    - Fig. 4: Training/validation curves
    - Fig. 5: Confusion matrix
    - Fig. 6: Per-class metrics bar chart
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score,
)

from . import config
from .model import Compact3DCNN


# ──────────────────────────────────────────────
# Publication-quality plot defaults
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


@torch.no_grad()
def get_predictions(
    model: Compact3DCNN,
    dataloader: DataLoader,
    device: str = config.DEVICE,
) -> tuple:
    """Run inference on dataloader. Returns (all_preds, all_labels, all_probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for videos, labels in dataloader:
        videos = videos.to(device, non_blocking=True)
        with autocast(enabled=config.USE_AMP and device == "cuda"):
            outputs = model(videos)
        probs = torch.softmax(outputs.float(), dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> dict:
    """Compute all metrics for paper Table IV."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Specificity per class
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    specificity = []
    for i in range(len(class_names)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)

    # Macro averages
    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "specificity": float(specificity[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "specificity": float(np.mean(specificity)),
            "f1": float(macro_f1),
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def plot_training_curves(history: dict, save_path: Path):
    """Generate Fig. 4: Training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Training Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    # Accuracy (Handle both 'train_acc' and 'train_accuracy' keys safely)
    train_acc = history.get("train_accuracy", history.get("train_acc", []))
    val_acc = history.get("val_accuracy", history.get("val_acc", []))
    ax2.plot(epochs, train_acc, label="Training Accuracy", linewidth=1.5)
    ax2.plot(epochs, val_acc, label="Validation Accuracy", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Training curves -> {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    normalize: bool = False,
):
    """Generate Fig. 5: Confusion matrix heatmap."""
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    # For many classes, reduce font size
    n = len(class_names)
    figsize = max(8, n * 0.15)
    annot = n <= 30  # Only annotate cells if manageable number of classes
    fontsize = max(4, 10 - n // 10)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sns.heatmap(
        cm, annot=annot, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, annot_kws={"size": fontsize},
        square=True, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix -> {save_path}")


def plot_per_class_metrics(
    metrics: dict,
    class_names: List[str],
    save_path: Path,
):
    """Generate Fig. 6: Per-class precision, recall, specificity, F1 bar chart."""
    n = len(class_names)
    prec = [metrics["per_class"][c]["precision"] for c in class_names]
    rec = [metrics["per_class"][c]["recall"] for c in class_names]
    spec = [metrics["per_class"][c]["specificity"] for c in class_names]
    f1 = [metrics["per_class"][c]["f1"] for c in class_names]

    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, n * 0.3), 6))
    ax.bar(x - 1.5 * width, prec, width, label="Precision", alpha=0.85)
    ax.bar(x - 0.5 * width, rec, width, label="Recall", alpha=0.85)
    ax.bar(x + 0.5 * width, spec, width, label="Specificity", alpha=0.85)
    ax.bar(x + 1.5 * width, f1, width, label="F1-Score", alpha=0.85)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=max(4, 10 - n // 10))
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Per-class metrics -> {save_path}")


def full_evaluation(
    model: Compact3DCNN,
    val_loader: DataLoader,
    class_names: List[str],
    history: Optional[dict] = None,
    experiment_name: str = "har_3dcnn",
    device: str = config.DEVICE,
) -> dict:
    """Run complete evaluation pipeline. Generates all figures + metrics JSON."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")

    # Get predictions
    preds, labels, probs = get_predictions(model, val_loader, device)

    # Compute metrics
    metrics = compute_metrics(preds, labels, class_names)
    print(f"\nOverall accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1: {metrics['macro']['f1']:.4f}")
    print(f"Macro precision: {metrics['macro']['precision']:.4f}")
    print(f"Macro recall: {metrics['macro']['recall']:.4f}")

    # Save metrics JSON
    metrics_path = config.METRICS_DIR / f"{experiment_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics -> {metrics_path}")

    # Generate figures (wrapped in a try-except to prevent crashes on other devices or headless environments)
    prefix = config.FIGURES_DIR / experiment_name
    try:
        if history:
            plot_training_curves(history, Path(f"{prefix}_training_curves.png"))

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, class_names, Path(f"{prefix}_confusion_matrix.png"))
        plot_confusion_matrix(cm, class_names, Path(f"{prefix}_confusion_matrix_norm.png"), normalize=True)
        plot_per_class_metrics(metrics, class_names, Path(f"{prefix}_per_class_metrics.png"))
    except Exception as e:
        print(f"\n⚠️ Warning: Failed to generate visualization plots/confusion matrix ({e}). Skipping figure generation.")

    # Print classification report
    print(f"\n{classification_report(labels, preds, target_names=class_names, zero_division=0)}")

    return metrics
