"""
Single-video inference + GIF preview generation.
"""

import numpy as np
import torch
import imageio
from pathlib import Path
from typing import List, Tuple

from . import config
from .model import Compact3DCNN
from .dataset import VideoDataset


def load_model(checkpoint_path: str, device: str = config.DEVICE) -> Tuple[torch.nn.Module, List[str]]:
    """
    Load trained model from checkpoint.
    
    Automatically detects the state dictionary keys to support backward compatibility
    between legacy plain 3D CNN, Google Colab 3D CNN, and advanced R(2+1)D Residual model checkpoints.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    num_classes = ckpt["config"]["num_classes"]
    state_dict = ckpt["model_state_dict"]

    # Detect checkpoint architecture type
    is_colab_checkpoint = any("conv1.1.running_mean" in key or "fc.weight" in key for key in state_dict.keys())
    is_legacy_plain = any("conv1.0.weight" in key for key in state_dict.keys()) and not is_colab_checkpoint

    if is_colab_checkpoint:
        print("[COMPATIBILITY] Detected Google Colab Compact 3D CNN checkpoint (with BatchNorm3d). Instantiating Colab architecture...")
        from .model import ColabCompact3DCNN
        model = ColabCompact3DCNN(num_classes=num_classes).to(device)
        model.is_colab = True
    elif is_legacy_plain:
        print("[COMPATIBILITY] Detected legacy Plain 3D CNN checkpoint (no BatchNorm3d). Instantiating original architecture...")
        from .model import Plain3DCNN
        model = Plain3DCNN(num_classes=num_classes).to(device)
        model.is_colab = False
    else:
        print("[ADVANCED] Detected new Spatio-Temporal Residual (2+1)D CNN checkpoint. Instantiating R(2+1)D...")
        model = Compact3DCNN(num_classes=num_classes).to(device)
        model.is_colab = False

    model.load_state_dict(state_dict)
    model.eval()

    # Store dynamic image size resolution config on the model instance
    model.img_size = ckpt.get("config", {}).get("img_size", config.IMG_SIZE)

    val_acc = ckpt.get("val_accuracy", ckpt.get("best_val_acc", 0.0))
    print(f"Loaded: {checkpoint_path} ({num_classes} classes, val_acc={val_acc:.2%})")
    return model, class_names



@torch.no_grad()
def predict_video(
    model: Compact3DCNN,
    video_path: str,
    class_names: List[str],
    device: str = config.DEVICE,
) -> dict:
    """
    Predict action class for a single video.

    Returns:
        {"class": str, "confidence": float, "class_idx": int, "all_probs": dict}
    """
    img_size = getattr(model, "img_size", config.IMG_SIZE)
    ds = VideoDataset([video_path], [0], is_train=False, uniform_sample=True, img_size=img_size)
    frames, _ = ds[0]
    frames = frames.unsqueeze(0).to(device)  # (1, C, T, H, W)

    if getattr(model, "is_colab", False):
        frames = frames * 255.0

    outputs = model(frames)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = probs.argmax().item()

    return {
        "class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "class_idx": pred_idx,
        "all_probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def video_to_gif(video_path: str, gif_path: str, n_frames: int = config.N_FRAMES):
    """Extract frames from video -> save as GIF preview."""
    ds = VideoDataset([video_path], [0], is_train=False)
    frames, _ = ds[0]  # (C, T, H, W)
    frames = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(gif_path, frames, duration=100, loop=0)
    return gif_path


def measure_inference_latency(
    model: Compact3DCNN,
    device: str = config.DEVICE,
    n_runs: int = 50,
    warmup: int = 10,
) -> dict:
    """Measure inference latency (ms/video). For paper comparison table."""
    import time

    dummy = torch.randn(1, 3, config.N_FRAMES, *config.IMG_SIZE).to(device)
    model.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    latency = {
        "device": device,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }
    print(f"Inference latency ({device}): {latency['mean_ms']:.1f} ± {latency['std_ms']:.1f} ms")
    return latency



import cv2
import threading
import time
from collections import deque

@torch.no_grad()
def predict_realtime(
    model: Compact3DCNN,
    class_names: List[str],
    device: str = config.DEVICE,
    video_source = None,  # Can be None (webcam) or a YouTube URL string
):
    """
    Real-time spatiotemporal inference from webcam or YouTube live video stream 
    using a dedicated background inference thread and probability smoothing.
    """
    cap = None
    if isinstance(video_source, str) and any(k in video_source.lower() for k in ("youtube.com", "youtu.be")):
        print(f"Extracting YouTube stream URL for: {video_source}...")
        import yt_dlp
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_source, download=False)
            source = info['url']
        print("YouTube stream extracted successfully!")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("Failed to open YouTube video stream. Verify link or internet connection.")
    else:
        # Try indices 0, 1, 2 automatically to find any available webcam
        opened_idx = -1
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                opened_idx = idx
                print(f"Webcam successfully opened on index {idx}")
                break
            cap.release()
        
        if opened_idx == -1:
            raise RuntimeError("Could not open any webcam (tried indices 0, 1, 2). Verify connection.")

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Shared state between threads (history buffer for rolling aspect-padded frames)
    history_len = 1 + (config.N_FRAMES - 1) * config.FRAME_STEP
    frame_buffer = deque(maxlen=history_len)
    latest_prediction = {"class": "Waiting for frames...", "confidence": 0.0}
    stop_event = threading.Event()
    
    # Smooth probabilities using EMA to prevent flickering
    smoothed_probs = np.zeros(len(class_names), dtype=np.float32)
    alpha = 0.3  # Smoothing factor (lower = smoother but slower to react)

    def inference_thread():
        nonlocal latest_prediction, smoothed_probs
        
        while not stop_event.is_set():
            if len(frame_buffer) < config.N_FRAMES:
                time.sleep(0.01)
                continue
                
            # Snapshot the buffer to avoid mutation during inference
            history = list(frame_buffer)
            n = len(history)
            
            # Dynamically compute stride for instant startup responsiveness, scaling up to FRAME_STEP
            stride = min(config.FRAME_STEP, max(1, (n - 1) // (config.N_FRAMES - 1)))
            
            # Sample exactly N_FRAMES at the calculated stride
            frames = [history[i * stride] for i in range(config.N_FRAMES)]
            
            # shape: (T, H, W, C)
            tensor = np.array(frames)
            # convert to (1, C, T, H, W)
            tensor = torch.from_numpy(tensor).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
            
            if getattr(model, "is_colab", False):
                tensor = tensor * 255.0
            
            # GPU Forward Pass (explicitly disable gradients in background thread)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
            
            # Apply Exponential Moving Average (EMA) for multi-tasking smoothing
            if np.all(smoothed_probs == 0.0):
                smoothed_probs = probs.copy()
            else:
                smoothed_probs = alpha * probs + (1 - alpha) * smoothed_probs
            
            pred_idx = smoothed_probs.argmax()
            conf = smoothed_probs[pred_idx]
            
            # Update prediction dynamically
            latest_prediction = {
                "class": class_names[pred_idx],
                "confidence": float(conf)
            }
            
            # Cap inference rate at ~30 FPS to prevent thread starvation
            time.sleep(0.03)

    # Start the GPU inference in background
    infer_thread = threading.Thread(target=inference_thread, daemon=True)
    infer_thread.start()

    print("Real-Time Test Started. Press 'q' to quit.")

    # Main thread handles UI and camera reading (ensures max FPS)
    try:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get dynamic camera resolution (handles 1080p, 720p, etc.)
            h, w, _ = frame.shape

            # Preprocess for model using aspect-preserving symmetric padding
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            norm = rgb.astype(np.float32) / 255.0
            padded = VideoDataset._resize_with_pad(norm, config.IMG_SIZE)
            
            frame_buffer.append(padded)

            # Draw Dynamic UI Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, int(h * 0.1)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            text = f"{latest_prediction['class']} ({latest_prediction['confidence']:.1%})"
            color = (0, 255, 0) if latest_prediction['confidence'] > 0.6 else (0, 165, 255)
            
            # Calculate dynamic text scaling and positions based on frame width
            font_scale = max(0.5, w / 800)
            thickness = max(1, int(w / 400))
            
            cv2.putText(frame, text, (20, int(h * 0.06)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, "Press 'q' to quit", (w - int(w * 0.25), int(h * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (200, 200, 200), max(1, thickness - 1))

            cv2.imshow("Real Time Action Recognition (GPU Accelerated)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Crucial: Always set the stop event to terminate the background inference thread!
        stop_event.set()
        
        cap.release()
        cv2.destroyAllWindows()
        infer_thread.join(timeout=1.0)
        
        # Clear PyTorch CUDA cache to free up VRAM immediately on exit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
