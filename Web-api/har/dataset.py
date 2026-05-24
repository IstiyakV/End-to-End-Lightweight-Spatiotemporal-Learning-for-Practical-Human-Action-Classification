"""
Video dataset — lazy-loading pipeline matching paper §III-B.

Key difference from old code: does NOT load all videos into RAM.
Decodes frames on-the-fly per batch -> handles 101/121 classes on 16GB RAM.

Implements paper equations:
    Eq (1): N_need = 1 + (n_frames - 1) * Δt
    Eq (2): Frame index sampling with stride
    Eq (3): Random temporal offset (training augmentation)
    Eq (4): Pixel normalisation to [0, 1]
    Eq (5): Bilinear resize with aspect-preserving padding
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from . import config


class VideoDataset(Dataset):
    """
    Lazy-loading video dataset. Each __getitem__ decodes one video on-the-fly.

    Args:
        video_paths: List of video file paths.
        labels: List of integer class labels.
        n_frames: Number of frames to sample per video.
        frame_step: Temporal stride between frames.
        img_size: (H, W) spatial resolution.
        is_train: If True, use random temporal offset (Eq. 3).
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        n_frames: int = config.N_FRAMES,
        frame_step: int = config.FRAME_STEP,
        img_size: Tuple[int, int] = config.IMG_SIZE,
        is_train: bool = True,
        uniform_sample: bool = False,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.n_frames = n_frames
        self.frame_step = frame_step
        self.img_size = img_size
        self.is_train = is_train
        self.uniform_sample = uniform_sample

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)
        # (T, H, W, C) -> (C, T, H, W) for PyTorch Conv3D
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        return frames, label

    def _extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract and preprocess frames from video file.

        Implements paper Eq (1)-(5) with a highly optimized single-seek sequential reading pipeline.
        This is up to 15x faster than multi-seeking in OpenCV, eliminating local I/O bottlenecks.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Return zero tensor if video can't be opened
            return np.zeros((self.n_frames, *self.img_size, 3), dtype=np.float32)

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if getattr(self, "uniform_sample", False) and video_length > self.n_frames:
            # Uniformly sample across the entire video
            target_idxs = set(np.linspace(0, video_length - 1, self.n_frames, dtype=int))
            start = 0
        else:
            # Standard paper stride-based sampling
            # Eq (1): Required span of frames
            need_length = 1 + (self.n_frames - 1) * self.frame_step

            # Eq (3): Starting frame — random offset during training
            if need_length > video_length:
                start = 0
            elif self.is_train:
                start = random.randint(0, video_length - need_length)
            else:
                start = 0  # Deterministic for validation
            
            target_idxs = {start + k * self.frame_step for k in range(self.n_frames)}

        # Seek EXACTLY ONCE at the starting frame (prevents expensive multi-seeking!)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        max_target = max(target_idxs) if target_idxs else start

        frames = []
        current_idx = start

        # Read sequentially to leverage OS/decoder filesystem read caches (blazing fast!)
        while len(frames) < self.n_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_idx in target_idxs:
                frame = self._preprocess_frame(frame)
                frames.append(frame)
            current_idx += 1
            if current_idx > max_target:
                break

        # Zero-pad remaining frames if video stream ended prematurely
        while len(frames) < self.n_frames:
            frames.append(np.zeros((*self.img_size, 3), dtype=np.float32))

        cap.release()
        return np.array(frames, dtype=np.float32)  # (T, H, W, C)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Eq (4): Normalise to [0, 1].
        Eq (5): Resize with aspect-preserving padding to IMG_SIZE.
        BGR -> RGB channel reorder.
        """
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Eq (4): Normalise pixel values
        frame = frame.astype(np.float32) / 255.0

        # Eq (5): Resize with padding to preserve aspect ratio
        frame = self._resize_with_pad(frame, self.img_size)

        return frame

    @staticmethod
    def _resize_with_pad(
        frame: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Bilinear resize with symmetric zero-padding (Paper Eq. 5)."""
        h, w = frame.shape[:2]
        target_h, target_w = target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Symmetric zero-padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded


def discover_classes(data_dir: Path) -> Tuple[List[str], dict]:
    """
    Scan directory for class folders. Returns sorted class names + name->idx map.

    Handles both UCF-101 (ApplyEyeMakeup) and Kinetics (applying_cream) naming.
    """
    classes = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    return classes, class_to_idx


def collect_videos(
    data_dir: Path,
    class_to_idx: dict,
    extensions: set = config.VIDEO_EXTENSIONS,
) -> Tuple[List[str], List[int]]:
    """Collect all video paths + labels from class folders."""
    paths, labels = [], []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in extensions:
                paths.append(str(f))
                labels.append(cls_idx)
    return paths, labels


def create_dataloaders(
    data_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
    num_workers: int = config.NUM_WORKERS,
    subset_classes: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train + val DataLoaders from a directory of class folders.

    Args:
        data_dir: Root directory with one subfolder per class.
        batch_size: Batch size for DataLoader.
        val_split: Fraction for validation (Paper §III-B.4: 0.2).
        seed: Random seed for reproducible splits.
        num_workers: DataLoader workers.
        subset_classes: Optional list of class names to use (for 3-class experiments).

    Returns:
        train_loader, val_loader, class_names
    """
    classes, class_to_idx = discover_classes(data_dir)

    # Optional: filter to subset
    if subset_classes:
        class_to_idx = {c: i for i, c in enumerate(subset_classes) if c in classes}
        classes = list(class_to_idx.keys())
        print(f"Subset mode: {len(classes)} classes -> {classes}")

    paths, labels = collect_videos(data_dir, class_to_idx)
    print(f"Dataset: {len(paths)} videos across {len(classes)} classes")

    if len(paths) == 0:
        raise ValueError(f"No videos found in {data_dir}")

    # Filter out classes with fewer than 2 samples (stratify requires >= 2)
    from collections import Counter
    label_counts = Counter(labels)
    min_samples = max(2, int(1 / val_split) + 1)  # Need enough for both splits
    valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples}
    if len(valid_labels) < len(set(labels)):
        dropped = len(set(labels)) - len(valid_labels)
        print(f"Dropped {dropped} classes with < {min_samples} samples")
        filtered = [(p, l) for p, l in zip(paths, labels) if l in valid_labels]
        paths = [p for p, _ in filtered]
        labels = [l for _, l in filtered]
        # Re-index labels to be contiguous
        old_to_new = {old: new for new, old in enumerate(sorted(valid_labels))}
        labels = [old_to_new[l] for l in labels]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = [idx_to_class[old] for old in sorted(valid_labels)]
        class_to_idx = {c: i for i, c in enumerate(classes)}

    # Stratified 80/20 split (Paper III-B.4)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )
    print(f"Split: {len(train_paths)} train / {len(val_paths)} val")

    train_ds = VideoDataset(train_paths, train_labels, is_train=True)
    val_ds = VideoDataset(val_paths, val_labels, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, classes


# ──────────────────────────────────────────────
# Cached dataset (loads preprocessed .npy files)
# ──────────────────────────────────────────────

class CachedVideoDataset(Dataset):
    """
    Loads preprocessed .npy frame tensors from disk or RAM.
    10-50x faster than decoding video files each epoch.
    """

    def __init__(self, npy_paths: List[str], labels: List[int], augment: bool = False, preload: bool = False):
        self.npy_paths = npy_paths
        self.labels = labels
        self.augment = augment
        self.preload = preload
        self.preloaded_frames = []

        if self.preload:
            from tqdm import tqdm
            print(f"[perf] Preloading {len(npy_paths)} samples to RAM (as uint8 to save memory)...")
            for path in tqdm(npy_paths, desc="RAM preloading", leave=False):
                loaded = np.load(path)
                if loaded.dtype != np.uint8:
                    if loaded.max() <= 1.0:
                        loaded = (loaded * 255.0).astype(np.uint8)
                    else:
                        loaded = loaded.astype(np.uint8)
                self.preloaded_frames.append(loaded)

    def __len__(self) -> int:
        return len(self.npy_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.preload:
            loaded = self.preloaded_frames[idx]
        else:
            loaded = np.load(self.npy_paths[idx])

        # Auto-detect format: uint8 is normalized on-the-fly, while legacy float16/float32 are loaded as-is
        if loaded.dtype == np.uint8:
            frames = loaded.astype(np.float32) / 255.0
        else:
            frames = loaded.astype(np.float32)

        if self.augment:
            # Random horizontal flip (50% chance) — mirrors spatial content
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            # Random brightness jitter ±10% — colour robustness
            brightness = 1.0 + random.uniform(-0.1, 0.1)
            frames = np.clip(frames * brightness, 0.0, 1.0).astype(np.float32)

        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)    # (C, T, H, W)
        return frames, self.labels[idx]


def create_cached_dataloaders(
    cache_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build DataLoaders from cached .npy files. Much faster than video decoding."""
    import json
    from collections import Counter

    meta_path = cache_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    classes = meta["classes"]
    class_to_idx = meta["class_to_idx"]

    # Collect .npy files
    paths, labels = [], []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = cache_dir / cls_name
        if not cls_dir.exists():
            continue
        for f in cls_dir.iterdir():
            if f.suffix == ".npy":
                paths.append(str(f))
                labels.append(cls_idx)

    print(f"Cached dataset: {len(paths)} samples across {len(classes)} classes")

    # Filter sparse classes
    label_counts = Counter(labels)
    min_samples = max(2, int(1 / val_split) + 1)
    valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples}
    if len(valid_labels) < len(set(labels)):
        dropped = len(set(labels)) - len(valid_labels)
        print(f"Dropped {dropped} classes with < {min_samples} samples")
        filtered = [(p, l) for p, l in zip(paths, labels) if l in valid_labels]
        paths = [p for p, _ in filtered]
        labels = [l for _, l in filtered]
        old_to_new = {old: new for new, old in enumerate(sorted(valid_labels))}
        labels = [old_to_new[l] for l in labels]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = [idx_to_class[old] for old in sorted(valid_labels)]

    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_split, random_state=seed, stratify=labels,
    )
    print(f"Split: {len(train_paths)} train / {len(val_paths)} val")

    preload = getattr(config, "PRELOAD_TO_RAM", False)
    train_ds = CachedVideoDataset(train_paths, train_labels, augment=True, preload=preload)
    val_ds = CachedVideoDataset(val_paths, val_labels, augment=False, preload=preload)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
        drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, classes
