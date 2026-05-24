"""
Training loop — mixed precision, TensorBoard, checkpointing.

Implements paper Eq (9)-(10):
    Eq (9): Categorical cross-entropy loss
    Eq (10): Adam optimiser with bias-corrected moments
"""

import json
import time
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import config
from .model import Compact3DCNN

# ── GPU Performance Optimisations ──────────────────────────
# Auto-tune conv algorithms for this exact input shape (one-time ~30s warmup)
torch.backends.cudnn.benchmark = True
# Use TF32 precision on Ampere+ GPUs (Colab A100/T4) — harmless on older GPUs
torch.set_float32_matmul_precision('medium')


class Trainer:
    """
    Training engine with mixed precision, checkpointing, and TensorBoard.

    Args:
        model: Compact3DCNN instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_names: List of class name strings.
        experiment_name: Name for checkpoints and logs.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: Compact3DCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        experiment_name: str = "har_3dcnn",
        device: str = config.DEVICE,
    ):
        # torch.compile: fuses Conv3D+BN+ReLU into single GPU kernels
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[perf] torch.compile enabled with mode='reduce-overhead'")
        except Exception:
            try:
                model = torch.compile(model)
                print("[perf] torch.compile enabled with default mode")
            except Exception:
                print("[perf] torch.compile not available, skipping")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
        self.experiment_name = experiment_name

        # Loss — Eq (9): categorical cross-entropy + label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

        # Optimiser — Eq (10): Adam with bias correction
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.ADAM_BETAS,
            eps=config.ADAM_EPS,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Mixed precision
        self.scaler = GradScaler('cuda', enabled=config.USE_AMP)
        self.use_amp = config.USE_AMP and device == "cuda"

        # TensorBoard
        tb_dir = config.TENSORBOARD_DIR / experiment_name
        self.writer = SummaryWriter(str(tb_dir))

        # History tracking
        self.history = {
            "train_loss": [], "train_accuracy": [],
            "val_loss": [], "val_accuracy": [],
            "epoch_time": [], "lr": [],
        }
        self.best_val_acc = 0.0

    def train(self, epochs: int = config.EPOCHS) -> dict:
        """Full training loop. Returns history dict."""

        # Select scheduler based on config
        use_onecycle = getattr(config, "USE_ONECYCLE", False)
        accum_steps = config.GRAD_ACCUMULATION
        
        if use_onecycle:
            # Step per batch; total steps must account for gradient accumulation
            effective_batches = len(self.train_loader) // accum_steps
            if len(self.train_loader) % accum_steps != 0:
                effective_batches += 1
            total_steps = epochs * effective_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=getattr(config, "MAX_LR", 5e-4),
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=False,
            )
            scheduler_type = f"OneCycleLR (max_lr={getattr(config, 'MAX_LR', 5e-4)})"
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
            scheduler_type = "CosineAnnealingLR"

        patience = config.EARLY_STOPPING_PATIENCE
        epochs_no_improve = 0

        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"Epochs: {epochs} | Batch: {config.BATCH_SIZE} (eff {config.BATCH_SIZE * accum_steps}) | LR: {config.LEARNING_RATE}")
        print(f"Device: {self.device} | AMP: {self.use_amp} | Label smooth: {config.LABEL_SMOOTHING}")
        print(f"Scheduler: {scheduler_type} | Early stop: {patience} epochs")
        print(f"Classes: {len(self.class_names)} | Train batches: {len(self.train_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch, accum_steps, scheduler if use_onecycle else None)
            val_loss, val_acc = self._validate()

            # Step scheduler after each epoch only if using epoch-based scheduler
            if not use_onecycle:
                scheduler.step()

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["epoch_time"].append(elapsed)
            self.history["lr"].append(current_lr)

            # TensorBoard
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("Time/epoch_seconds", elapsed, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # Checkpoint best model + early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
                marker = " * best"
                epochs_no_improve = 0
            else:
                marker = ""
                epochs_no_improve += 1

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"acc {train_acc:.2%}/{val_acc:.2%} | "
                f"lr {current_lr:.2e} | "
                f"{elapsed:.1f}s{marker}"
            )

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
                break

        # Save final history
        self._save_history()
        self.writer.close()

        print(f"\nDone. Best val_accuracy: {self.best_val_acc:.2%}")
        return self.history

    def _train_epoch(self, epoch: int, accum_steps: int = 1, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[float, float]:
        """One training epoch with gradient accumulation. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)
        self.optimizer.zero_grad()

        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels) / accum_steps

            self.scaler.scale(loss).backward()

            # Step optimizer every accum_steps batches (or at end of epoch)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * accum_steps * videos.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.2%}")

        pbar.close()
        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Validation pass. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in self.val_loader:
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * videos.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_acc: float):
        """Save best model checkpoint."""
        path = config.CHECKPOINT_DIR / f"{self.experiment_name}_best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_acc,
            "class_names": self.class_names,
            "config": {
                "n_frames": config.N_FRAMES,
                "frame_step": config.FRAME_STEP,
                "img_size": config.IMG_SIZE,
                "conv_filters": config.CONV_FILTERS,
                "dropout": config.DROPOUT,
                "num_classes": len(self.class_names),
            },
        }, str(path))

    def _save_history(self):
        """Save training history as JSON."""
        path = config.METRICS_DIR / f"{self.experiment_name}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History -> {path}")
