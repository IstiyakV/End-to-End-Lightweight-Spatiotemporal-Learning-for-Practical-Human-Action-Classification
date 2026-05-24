"""
Central configuration — all hyperparameters from the paper.

References:
    Table II in the submitted manuscript.
    Architecture: 3-layer 3D CNN, 279,683 params (3-class).
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "dataset" / "videos" / "train"  # Kinetics-121
UCF101_DIR = PROJECT_ROOT / "UCF-101"                            # UCF-101 (downloaded)
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"

# Create dirs
for d in [RESULTS_DIR, CHECKPOINT_DIR, FIGURES_DIR, METRICS_DIR, TENSORBOARD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Video preprocessing (Paper §III-B)
# ──────────────────────────────────────────────
N_FRAMES = 10          # n_frames — number of frames sampled per video
FRAME_STEP = 15        # Δt — temporal stride between sampled frames
IMG_SIZE = (224, 224)  # Spatial resolution after resize
# N_need = 1 + (N_FRAMES - 1) * FRAME_STEP = 136 frames of temporal coverage

# ──────────────────────────────────────────────
# Model architecture (Paper §III-C, Table II)
# ──────────────────────────────────────────────
CONV_FILTERS = [32, 64, 128]   # Channels per conv block
KERNEL_SIZE = 3                 # 3×3×3 spatiotemporal kernels
POOL_SIZE = (2, 2, 2)          # MaxPool3D pool size (all layers)
DROPOUT = 0.3                   # Dropout rate before classifier

# ──────────────────────────────────────────────
# Training (Paper §III-D, Eq. 9-10)
# ──────────────────────────────────────────────
BATCH_SIZE = 8         # Paper value — fits GTX 1660 6GB with mixed precision
EPOCHS = 100           # Paper: 100 epochs, converges by ~60
LEARNING_RATE = 1e-4   # Adam α (Paper Eq. 10)
ADAM_BETAS = (0.9, 0.999)  # β1, β2
ADAM_EPS = 1e-8        # ε
WEIGHT_DECAY = 0.0     # No L2 reg in paper
LABEL_SMOOTHING = 0.1  # Prevents overconfident predictions on 101 classes
EARLY_STOPPING_PATIENCE = 15  # Stop if val_acc doesn't improve for N epochs
VAL_SPLIT = 0.2        # 80/20 stratified split (Paper §III-B.4)
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Mixed precision + performance
# ──────────────────────────────────────────────
import sys
USE_AMP = True         # Automatic Mixed Precision -> halves VRAM
NUM_WORKERS = 0 if sys.platform == "win32" else 4        # DataLoader workers (0 is safest on Windows/local)
PIN_MEMORY = True      # Faster GPU transfer
GRAD_ACCUMULATION = 4  # Effective batch = 8 * 4 = 32 (better gradients, no extra VRAM)
PRELOAD_TO_RAM = False # Preload entire cached dataset to system RAM for 0 I/O overhead
USE_ONECYCLE = True    # Use OneCycleLR learning rate scheduler for faster convergence
MAX_LR = 5e-4          # Peak learning rate for OneCycleLR scheduler


# ──────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────
SAVE_BEST_ONLY = True  # Save only when val_accuracy improves
CHECKPOINT_METRIC = "val_accuracy"

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
