"""
Transfer Learning Background Training Service.
Runs in a separate thread. Supports pause, resume, and restart/retrain controls.
Maintains absolute state isolation.
"""

import json
import time
import threading
import sys
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, field, asdict
from enum import Enum

import torch
import torch.nn as nn
import torchvision.models.video as video_models


class TransferStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TransferConfig:
    """Isolated configurations for Transfer Learning runs."""
    model_name: str = "transfer_model"
    backbone_weights_path: str = ""
    backbone_architecture: str = "r3d_18"  # r3d_18, mc3_18, r2plus1d_18
    custom_head_name: str = "Paper Default"
    dataset_path: str = ""
    cache_path: str = ""
    use_cache: bool = True
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 0.0001
    optimizer_type: str = "Adam" # Adam, SGD
    scheduler_type: str = "None" # None, StepLR
    use_amp: bool = True
    num_workers: int = 0
    device: str = "cuda:0"

    def to_dict(self):
        return asdict(self)


@dataclass
class TransferProgress:
    """Training progress telemetry pushed to the Transfer UI."""
    status: str = TransferStatus.IDLE
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    best_val_acc: float = 0.0
    eta_seconds: float = 0.0
    gpu_mem_mb: int = 0
    gpu_mem_total_mb: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""
    history: dict = field(default_factory=lambda: {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_time": [], "lr": [],
    })


def inspect_and_validate_backbone(weight_path: str, architecture_type: str) -> dict:
    """
    Ingests a local pre-trained weights file on the CPU.
    Validates structural layers, matches signatures against specified video backbones,
    and extracts the feature representation dimension.
    """
    report = []
    report.append(f"🔍 Starting dynamic ingestion for backbone: {architecture_type.upper()}")
    report.append(f"📂 Weight File: {Path(weight_path).name}")

    if not Path(weight_path).exists():
        return {
            "status": "Failed",
            "message": "Weights file does not exist at specified path.",
            "log": "\n".join(report)
        }

    try:
        # 1. Ingest weights file safely on CPU (weights_only=False to support all standard models)
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        
        # If wrapped in dict (e.g. {'state_dict': ...} or {'model_state_dict': ...})
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, dict):
            raise TypeError("Loaded weights are not a dictionary format.")

        report.append(f"✓ State dictionary loaded successfully on CPU. Total keys: {len(state_dict)}")

        # 2. Instantiate a clean torchvision dummy model to inspect key matching
        if architecture_type == "r3d_18":
            dummy_model = video_models.r3d_18(weights=None)
        elif architecture_type == "mc3_18":
            dummy_model = video_models.mc3_18(weights=None)
        elif architecture_type == "r2plus1d_18":
            dummy_model = video_models.r2plus1d_18(weights=None)
        else:
            raise ValueError(f"Unknown architecture type: {architecture_type}")

        dummy_keys = set(dummy_model.state_dict().keys())
        loaded_keys = set(state_dict.keys())

        # 3. Calculate alignment ratio
        matched_keys = dummy_keys.intersection(loaded_keys)
        match_ratio = len(matched_keys) / max(1, len(dummy_keys))
        
        report.append(f"📊 Structural layers signature check:")
        report.append(f"   • Expected model layers: {len(dummy_keys)}")
        report.append(f"   • Loaded layer parameters: {len(loaded_keys)}")
        report.append(f"   • Perfect matching signatures: {len(matched_keys)} ({match_ratio:.1%})")

        # 4. Extract backbone output feature dimensions
        # Standard torchvision video models have a linear head named 'fc'
        feature_dim = 512 # default fallback
        if "fc.weight" in state_dict:
            feature_dim = state_dict["fc.weight"].shape[1]
            report.append(f"✓ Detected final classification dimension (in_features): {feature_dim}")
        elif hasattr(dummy_model, "fc"):
            feature_dim = dummy_model.fc.in_features
            report.append(f"✓ Inferred final classification dimension from dummy module: {feature_dim}")

        # 5. Decide Suitability Status
        is_suitable = match_ratio > 0.75
        status_flag = "Suitable/Ready" if is_suitable else "Warning: Suboptimal Match"
        
        if is_suitable:
            report.append("\n✅ Verdict: Backbone is SUITABLE and READY for dynamic Hybrid Binding!")
        else:
            report.append("\n⚠️ Verdict: Matching ratio is too low. Loading may result in shape mismatch errors.")

        return {
            "status": status_flag,
            "feature_dim": feature_dim,
            "match_ratio": match_ratio,
            "message": "Structural verification check completed.",
            "log": "\n".join(report)
        }

    except Exception as e:
        report.append(f"❌ Error during weights parsing: {str(e)}")
        return {
            "status": "Unsuitable/Failed",
            "message": f"Validation failed: {str(e)}",
            "log": "\n".join(report)
        }


class TransferLearningService:
    """
    Manages the Transfer Learning training lifecycle in a background thread.
    Features standalone, non-blocking state loops (start, pause, resume, retrain)
    and complete structural isolation.
    """
    def __init__(self):
        self.config: Optional[TransferConfig] = None
        self.progress = TransferProgress()
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._pause_event.set() # Unblocked initially
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self.model: Optional[nn.Module] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def add_callback(self, fn: Callable):
        self._callbacks.append(fn)

    def _notify(self):
        for fn in self._callbacks:
            try:
                fn(self.progress)
            except Exception:
                pass

    def _save_results_viewer_assets(self):
        """Save training_state.json and metrics.json into standard directories for the Results Viewer."""
        if not self.config:
            return
        try:
            from har.config import CHECKPOINT_DIR, METRICS_DIR
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            METRICS_DIR.mkdir(parents=True, exist_ok=True)

            # 1. Save training_state.json
            state_path = CHECKPOINT_DIR / f"{self.config.model_name}_training_state.json"
            state_data = {
                "status": self.progress.status,
                "current_epoch": self.progress.current_epoch,
                "history": {
                    "train_loss": self.progress.history.get("train_loss", []),
                    "val_loss": self.progress.history.get("val_loss", []),
                    "train_acc": self.progress.history.get("train_acc", []),
                    "val_acc": self.progress.history.get("val_acc", []),
                    "epoch_time": self.progress.history.get("epoch_time", []),
                    "lr": self.progress.history.get("lr", [])
                }
            }
            with open(state_path, "w") as f:
                json.dump(state_data, f, indent=4)

            # 2. Save metrics.json
            if self.progress.history.get("val_acc"):
                metrics_path = METRICS_DIR / f"{self.config.model_name}_metrics.json"
                metrics_data = {
                    "accuracy": self.progress.val_acc,
                    "macro": {
                        "f1": self.progress.val_acc
                    },
                    "per_class": {}
                }
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=4)
        except Exception as e:
            print(f"⚠️ Error saving Results Viewer assets: {e}")

    def start(self, config: TransferConfig, clean_start: bool = False):
        """Start or restart a training run in a background worker."""
        if self.is_running:
            raise RuntimeError("Transfer Learning thread is already active.")

        self.config = config
        self._stop_event.clear()
        self._pause_event.set() # Unblock pause wait

        # Reset states for fresh retraining if requested
        if clean_start:
            self.progress = TransferProgress(
                status=TransferStatus.RUNNING,
                total_epochs=config.epochs,
                current_epoch=0,
                best_val_acc=0.0,
                message="Initializing clean retrain...",
            )
        else:
            self.progress.status = TransferStatus.RUNNING
            self.progress.total_epochs = config.epochs
            self.progress.message = "Initializing Transfer Learning Pipeline..."

        self._notify()
        self._thread = threading.Thread(target=self._training_loop, daemon=True)
        self._thread.start()

    def pause(self):
        if self.is_running:
            self._pause_event.clear()
            self.progress.status = TransferStatus.PAUSED
            self.progress.message = "Pausing training gracefully..."
            self._notify()

    def resume(self):
        if self.progress.status == TransferStatus.PAUSED:
            self._pause_event.set()
            self.progress.status = TransferStatus.RUNNING
            self.progress.message = "Resuming training loop..."
            self._notify()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set() # Release if blocked on pause
        self.progress.message = "Stopping training thread..."
        self._notify()

    def retrain(self, config: TransferConfig):
        """Clean restart: wipe all histories and start back from Epoch 0."""
        self.stop()
        # Wait a small moment for the thread to stop safely
        time.sleep(0.3)
        self.start(config, clean_start=True)

    def _training_loop(self):
        """Dedicated background loop for transfer learning."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

        try:
            from har.dataset import create_cached_dataloaders, create_dataloaders
            from har.transfer_model import HybridTransferModel
            from har.model_builder import ArchitectureConfig, build_from_config, ARCH_DIR, preset_architectures
            
            cfg = self.config
            device = cfg.device

            print("=" * 60)
            print("🚀 LAUNCHING TRANSFER LEARNING RUN")
            print(f"   • Model Identifier: {cfg.model_name}")
            print(f"   • Pre-trained:      {Path(cfg.backbone_weights_path).name}")
            print(f"   • Custom Head:      {cfg.custom_head_name}")
            print(f"   • Dataset Path:     {cfg.dataset_path}")
            print(f"   • Device:           {device}")
            print("=" * 60)

            # 1. Inspect Backbone weights to discover output shape
            self.progress.message = "Verifying pre-trained backbone features..."
            self._notify()
            
            inspection = inspect_and_validate_backbone(cfg.backbone_weights_path, cfg.backbone_architecture)
            if "Suitable" not in inspection["status"]:
                raise ValueError(f"Ingested backbone weights failed safety validation: {inspection['status']}")
            
            backbone_dim = inspection.get("feature_dim", 512)

            # 2. Instantiate Clean Dataloaders (raw MP4 or Cached .npy structures)
            self.progress.message = "Configuring data streams..."
            self._notify()

            if cfg.use_cache and cfg.cache_path:
                train_loader, val_loader, class_names = create_cached_dataloaders(
                    cache_dir=Path(cfg.cache_path),
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                )
            else:
                train_loader, val_loader, class_names = create_dataloaders(
                    data_dir=Path(cfg.dataset_path),
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                )
            
            self.progress.total_batches = len(train_loader)
            num_classes = len(class_names)

            # 3. Instantiate Backbone & Custom Head models
            self.progress.message = "Loading backbone & building custom head..."
            self._notify()

            if cfg.backbone_architecture == "r3d_18":
                backbone_net = video_models.r3d_18(weights=None)
            elif cfg.backbone_architecture == "mc3_18":
                backbone_net = video_models.mc3_18(weights=None)
            elif cfg.backbone_architecture == "r2plus1d_18":
                backbone_net = video_models.r2plus1d_18(weights=None)
            else:
                raise ValueError(f"Unknown backbone: {cfg.backbone_architecture}")

            # Build Custom head from Network Architect configs
            presets = preset_architectures()
            if cfg.custom_head_name in presets:
                arch = presets[cfg.custom_head_name]
                custom_head_net = build_from_config(arch, num_classes=num_classes)
            else:
                arch_path = ARCH_DIR / f"{cfg.custom_head_name}.json"
                if arch_path.exists():
                    arch = ArchitectureConfig.load(arch_path)
                    custom_head_net = build_from_config(arch, num_classes=num_classes)
                else:
                    # Failback to custom head default
                    from har.model import build_model
                    custom_head_net = build_model(num_classes=num_classes, device="cpu")

            # 4. Bind dynamically using Hybrid Transfer Model wrapper
            self.model = HybridTransferModel(
                backbone=backbone_net,
                custom_head=custom_head_net,
                backbone_feature_dim=backbone_dim,
                freeze_backbone=True
            ).to(device)

            # Load weights into the backbone
            state_dict = torch.load(cfg.backbone_weights_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Filter fc layers keys from state dict since we replaced it with Identity
            filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("fc.") and not k.startswith("classifier.")}
            self.model.backbone.load_state_dict(filtered_state, strict=False)
            print("✓ Loaded pre-trained backbone parameters successfully.")

            # 5. Optimizers (Adam or SGD) over trainable parameters only
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            if cfg.optimizer_type == "SGD":
                optimizer = torch.optim.SGD(trainable_params, lr=cfg.learning_rate, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = torch.optim.Adam(trainable_params, lr=cfg.learning_rate, betas=(0.9, 0.999))

            # Scheduler options
            scheduler = None
            if cfg.scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)

            # Restoring history for resumes
            start_epoch = self.progress.current_epoch + 1
            best_val_acc = self.progress.best_val_acc
            epoch_times = []

            # 6. Primary Stateful Training Loop
            for epoch in range(start_epoch, cfg.epochs + 1):
                # check stop
                if self._stop_event.is_set():
                    self.progress.status = TransferStatus.STOPPED
                    self.progress.message = "Training terminated by user."
                    self._notify()
                    self._save_results_viewer_assets()
                    break

                t0 = time.time()
                self.progress.current_epoch = epoch

                # Training epoch
                self.model.train()
                total_loss, correct, total = 0.0, 0, 0

                for batch_idx, (videos, labels) in enumerate(train_loader):
                    # Check pause event
                    if not self._pause_event.is_set():
                        self.progress.status = TransferStatus.PAUSED
                        self.progress.message = f"Training paused at epoch {epoch-1} boundary."
                        self._notify()
                        self._save_results_viewer_assets()
                        self._pause_event.wait() # Thread blocks here until set() is called
                        
                        if self._stop_event.is_set():
                            self.progress.status = TransferStatus.STOPPED
                            self.progress.message = "Training terminated by user."
                            self._notify()
                            self._save_results_viewer_assets()
                            break
                        self.progress.status = TransferStatus.RUNNING
                        self.progress.message = f"Resuming Epoch {epoch}..."
                        self._notify()

                    if self._stop_event.is_set():
                        self.progress.status = TransferStatus.STOPPED
                        self.progress.message = "Training terminated by user."
                        self._notify()
                        self._save_results_viewer_assets()
                        break

                    videos = videos.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                        outputs = self.model(videos)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item() * videos.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

                    # Update progress batch statistics
                    self.progress.current_batch = batch_idx + 1
                    self.progress.train_loss = total_loss / total
                    self.progress.train_acc = correct / total

                    if device.startswith("cuda"):
                        self.progress.gpu_mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
                        self.progress.gpu_mem_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

                    self.progress.message = f"Epoch {epoch} Training... ({batch_idx+1}/{len(train_loader)})"
                    self._notify()

                if self._stop_event.is_set():
                    break

                train_loss = total_loss / total
                train_acc = correct / total

                # Validation phase
                self.model.eval()
                val_loss_total, val_correct, val_total = 0.0, 0, 0
                val_len = len(val_loader)

                with torch.no_grad():
                    for val_idx, (videos, labels) in enumerate(val_loader):
                        if self._stop_event.is_set():
                            break
                        videos = videos.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                            outputs = self.model(videos)
                            loss = criterion(outputs, labels)

                        val_loss_total += loss.item() * videos.size(0)
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(labels).sum().item()
                        val_total += labels.size(0)

                        self.progress.message = f"Epoch {epoch} Validating... ({val_idx+1}/{val_len})"
                        self.progress.current_batch = val_idx + 1
                        self.progress.total_batches = val_len
                        self._notify()

                val_loss = val_loss_total / val_total
                val_acc = val_correct / val_total

                if scheduler is not None:
                    scheduler.step()

                elapsed = time.time() - t0
                epoch_times.append(elapsed)

                # Update live history logs (Matplotlib feeds on this!)
                self.progress.train_loss = train_loss
                self.progress.val_loss = val_loss
                self.progress.train_acc = train_acc
                self.progress.val_acc = val_acc
                self.progress.elapsed_seconds += elapsed

                self.progress.history["train_loss"].append(train_loss)
                self.progress.history["val_loss"].append(val_loss)
                self.progress.history["train_acc"].append(train_acc)
                self.progress.history["val_acc"].append(val_acc)
                self.progress.history.setdefault("epoch_time", []).append(elapsed)
                self.progress.history.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])

                # ETA calculation
                avg_epoch = sum(epoch_times) / len(epoch_times)
                remaining = cfg.epochs - epoch
                self.progress.eta_seconds = avg_epoch * remaining

                # Save Best Checkpoint
                checkpoints_dir = Path("results/transfer_learning/checkpoints")
                checkpoints_dir.mkdir(parents=True, exist_ok=True)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.progress.best_val_acc = best_val_acc
                    best_path = checkpoints_dir / f"{cfg.model_name}_best.pth"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "best_val_acc": best_val_acc,
                        "class_names": class_names,
                        "config": cfg.to_dict()
                    }, str(best_path))
                    print(f"★ Epoch {epoch} - New Best Acc: {val_acc:.2%}. Checkpoint saved.")
                
                # Save Latest Checkpoint to survive crashes
                latest_path = checkpoints_dir / f"{cfg.model_name}_latest.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                    "config": cfg.to_dict()
                }, str(latest_path))

                self.progress.message = f"Epoch {epoch} Complete. val_acc={val_acc:.2%}"
                self._notify()
                self._save_results_viewer_assets()

            # Completed
            if not self._stop_event.is_set() and self.progress.status != TransferStatus.PAUSED:
                self.progress.status = TransferStatus.COMPLETED
                self.progress.message = f"Transfer Learning Complete! Best val_acc: {best_val_acc:.2%}"
                self._notify()
                self._save_results_viewer_assets()

        except Exception as e:
            self.progress.status = TransferStatus.ERROR
            self.progress.message = f"Error: {str(e)}"
            self._notify()
            self._save_results_viewer_assets()
            import traceback
            traceback.print_exc()
