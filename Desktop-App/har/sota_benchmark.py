"""
SOTA Video Action Recognition Benchmark Engine.

Runs pretrained state-of-the-art video classification models sequentially
on a given video and compares predictions against our compact R(2+1)D model.

Supports two model sources:
    1. HuggingFace (transformers) — VideoMAE, TimeSformer, ViViT
    2. Torchvision — R(2+1)D-18, S3D, Swin3D-T, MViT-v2-S

Plus reference-only models (paper accuracy only, no live inference):
    Two-Stream CNN, C3D, TSN, I3D, SlowFast, UniFormer-B

Design:
    - Sequential inference: loads one SOTA model at a time to fit 6 GB VRAM.
    - Each model is loaded -> inference -> offloaded before the next.
    - FP16 mode halves memory for large transformers.
    - Progress callback for GUI integration.
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Callable, Optional

# Suppress HuggingFace Hub warnings (symlinks, auth, progress bars)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ──────────────────────────────────────────────
# SOTA Model Registry — Benchmarkable Models
# ──────────────────────────────────────────────
SOTA_MODELS = {
    # ── HuggingFace models (require `transformers` package) ──
    "VideoMAE": {
        "source": "huggingface",
        "checkpoint": "MCG-NJU/videomae-base-finetuned-kinetics",
        "params": "87M",
        "flops": "180G",
        "year": 2022,
        "type": "Self-Supervised MAE + ViT",
        "num_frames": 16,
        "paper_acc": "96.1%",
    },
    "TimeSformer": {
        "source": "huggingface",
        "checkpoint": "facebook/timesformer-base-finetuned-k400",
        "params": "121M",
        "flops": "196G",
        "year": 2021,
        "type": "Divided Space-Time Attention",
        "num_frames": 8,
        "paper_acc": "96.0%",
    },
    "ViViT": {
        "source": "huggingface",
        "checkpoint": "google/vivit-b-16x2-kinetics400",
        "processor_class": "VivitImageProcessor",
        "model_class": "VivitForVideoClassification",
        "params": "88M",
        "flops": "260G",
        "year": 2021,
        "type": "Video Vision Transformer",
        "num_frames": 32,
        "paper_acc": "94.0%",
    },

    # ── Torchvision models (no extra package needed) ──
    "R(2+1)D-18": {
        "source": "torchvision",
        "model_fn": "r2plus1d_18",
        "weights_class": "R2Plus1D_18_Weights",
        "params": "31.5M",
        "flops": "41.6G",
        "year": 2018,
        "type": "(2+1)D Factorised 3D",
        "num_frames": 16,
        "paper_acc": "96.8%",
    },
    "S3D": {
        "source": "torchvision",
        "model_fn": "s3d",
        "weights_class": "S3D_Weights",
        "params": "8.3M",
        "flops": "6.1G",
        "year": 2018,
        "type": "Separable 3D CNN",
        "num_frames": 16,
        "paper_acc": "N/A",
    },
    "Swin3D-T": {
        "source": "torchvision",
        "model_fn": "swin3d_t",
        "weights_class": "Swin3D_T_Weights",
        "params": "28.2M",
        "flops": "79.4G",
        "year": 2022,
        "type": "Swin Transformer 3D",
        "num_frames": 16,
        "paper_acc": "84.9%",
    },
    "MViT-v2-S": {
        "source": "torchvision",
        "model_fn": "mvit_v2_s",
        "weights_class": "MViT_V2_S_Weights",
        "params": "34.5M",
        "flops": "70.5G",
        "year": 2022,
        "type": "Multi-scale ViT",
        "num_frames": 16,
        "paper_acc": "81.2%",
    },
}

# ──────────────────────────────────────────────
# Reference-Only Models (paper numbers, no live inference)
# ──────────────────────────────────────────────
REFERENCE_MODELS = {
    "Two-Stream CNN": {"year": 2014, "type": "2D + Optical Flow", "paper_acc": "88.0%", "params": "12M", "flops": "20G"},
    "C3D":            {"year": 2015, "type": "3D CNN",           "paper_acc": "82.3%", "params": "33M", "flops": "38.5G"},
    "TSN":            {"year": 2016, "type": "2D + Segments",    "paper_acc": "94.2%", "params": "23M", "flops": "16G"},
    "I3D (Two-Str.)": {"year": 2017, "type": "Inflated 3D",     "paper_acc": "97.9%", "params": "25M", "flops": "108G"},
    "R(2+1)D-34":     {"year": 2018, "type": "(2+1)D Factorised", "paper_acc": "96.8%", "params": "63.6M", "flops": "152.9G"},
    "SlowFast 8×8":   {"year": 2019, "type": "Dual-Path 3D",    "paper_acc": "94.2%", "params": "33.6M", "flops": "65.7G"},
    "Video Swin-B":   {"year": 2022, "type": "Swin Transformer", "paper_acc": "84.9%*", "params": "88M", "flops": "282G"},
    "UniFormer-B":    {"year": 2022, "type": "Conv + Attention", "paper_acc": "96.5%", "params": "50M", "flops": "259G"},
}


# ══════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════

def check_transformers_available() -> bool:
    """Check if HuggingFace transformers is installed."""
    try:
        import transformers
        return True
    except ImportError:
        return False


def is_model_cached(info: Dict) -> bool:
    """
    Check if a SOTA model is already in the local cache.

    Args:
        info: Model info dict from SOTA_MODELS.
    """
    source = info.get("source", "huggingface")

    if source == "huggingface":
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(info["checkpoint"], local_files_only=True)
            return True
        except Exception:
            return False

    elif source == "torchvision":
        try:
            import torchvision.models.video as vm
            weights_cls = getattr(vm, info["weights_class"])
            weights = weights_cls.DEFAULT
            url = weights.url
            hub_dir = torch.hub.get_dir()
            cached_file = os.path.join(hub_dir, 'checkpoints', os.path.basename(url))
            return os.path.exists(cached_file)
        except Exception:
            return False

    return False


def download_sota_model(info: Dict):
    """
    Pre-download a SOTA model to the persistent cache.

    HuggingFace models -> ~/.cache/huggingface/hub/
    Torchvision models -> ~/.cache/torch/hub/checkpoints/
    """
    source = info.get("source", "huggingface")

    if source == "huggingface":
        # Patch isatty for CustomTkinter compatibility
        for stream in [sys.stdout, sys.stderr]:
            if not hasattr(stream, 'isatty'):
                stream.isatty = lambda: False

        # Suppress "unauthenticated requests" warning from huggingface_hub
        import warnings
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*symlinks.*")

        import transformers

        # Use explicit class names if specified (e.g. ViViT), else Auto classes
        proc_cls_name = info.get("processor_class", "AutoImageProcessor")
        model_cls_name = info.get("model_class", "AutoModelForVideoClassification")
        ProcessorCls = getattr(transformers, proc_cls_name)
        ModelCls = getattr(transformers, model_cls_name)

        processor = ProcessorCls.from_pretrained(info["checkpoint"])
        del processor
        model = ModelCls.from_pretrained(info["checkpoint"])
        del model

    elif source == "torchvision":
        import torchvision.models.video as vm
        model_fn = getattr(vm, info["model_fn"])
        weights_cls = getattr(vm, info["weights_class"])
        weights = weights_cls.DEFAULT
        model = model_fn(weights=weights)
        del model

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════
# Frame Extraction
# ══════════════════════════════════════════════

def extract_video_frames(video_path: str, n_frames: int = 16) -> List[np.ndarray]:
    """
    Extract uniformly-sampled frames from a video file.

    Returns:
        List of numpy arrays, each (H, W, C) in uint8 [0, 255].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        total = len(all_frames)
        if total == 0:
            raise RuntimeError(f"No frames found in: {video_path}")
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        frames = [cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB) for i in indices]
    else:
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return frames[:n_frames]


def download_youtube_video(url: str) -> str:
    """
    Download a YouTube video to a temp file.

    Returns:
        Path to the downloaded temp MP4 file. Caller must delete after use.
    """
    import tempfile
    import yt_dlp

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    ydl_opts = {
        'format': 'best[ext=mp4][height<=480]/best[ext=mp4]/best',
        'outtmpl': tmp_path,
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return tmp_path


def extract_youtube_frames(url: str, n_frames: int = 16) -> List[np.ndarray]:
    """Download YouTube video, extract frames, cleanup."""
    tmp_path = download_youtube_video(url)
    frames = extract_video_frames(tmp_path, n_frames)
    try:
        Path(tmp_path).unlink()
    except OSError:
        pass
    return frames


# ══════════════════════════════════════════════
# Our Model Inference
# ══════════════════════════════════════════════

def run_our_model(
    video_path: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> Dict:
    """Run our compact R(2+1)D model on a video."""
    from . import config
    from .predict import load_model, predict_video

    t0 = time.perf_counter()
    model, class_names = load_model(checkpoint_path, device)
    result = predict_video(model, video_path, class_names, device)
    latency = (time.perf_counter() - t0) * 1000

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "label": result["class"],
        "confidence": result["confidence"],
        "latency_ms": latency,
        "params": f"{total_params / 1000:.0f}K",
        "params_raw": total_params,
        "flops": "0.47G",
        "all_probs": result.get("all_probs", {}),
        "year": 2026,
        "type": "(2+1)D Residual + GAP",
        "paper_acc": "93.83%",
    }


# ══════════════════════════════════════════════
# SOTA Inference — Dual-Source Engine
# ══════════════════════════════════════════════

def _infer_huggingface(info: Dict, frames: List[np.ndarray], device: str):
    """Run inference with a HuggingFace video classification model."""
    # Patch isatty for CustomTkinter
    for stream in [sys.stdout, sys.stderr]:
        if not hasattr(stream, 'isatty'):
            stream.isatty = lambda: False

    import warnings
    warnings.filterwarnings("ignore", message=".*unauthenticated.*")
    warnings.filterwarnings("ignore", message=".*symlinks.*")

    import transformers

    # Use explicit class names if specified (e.g. ViViT), else Auto classes
    proc_cls_name = info.get("processor_class", "AutoImageProcessor")
    model_cls_name = info.get("model_class", "AutoModelForVideoClassification")
    ProcessorCls = getattr(transformers, proc_cls_name)
    ModelCls = getattr(transformers, model_cls_name)

    processor = ProcessorCls.from_pretrained(info["checkpoint"])
    model = ModelCls.from_pretrained(info["checkpoint"])

    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device).half().eval()
    else:
        model = model.eval()
        device = "cpu"

    inputs = processor(frames, return_tensors="pt")
    processed = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.dtype == torch.float32 and device == "cuda":
                v = v.half()
        processed[k] = v

    with torch.no_grad():
        outputs = model(**processed)
        probs = torch.softmax(outputs.logits.float(), dim=-1)

    pred_idx = probs.argmax(-1).item()
    confidence = probs[0][pred_idx].item()

    # Get label — fall back to Kinetics-400 names if model has generic LABEL_X
    id2label = model.config.id2label
    sample_label = id2label.get(0, "")
    if sample_label.startswith("LABEL_"):
        # Model has generic labels — use Kinetics-400 class names from torchvision
        try:
            from torchvision.models.video import R2Plus1D_18_Weights
            k400_labels = R2Plus1D_18_Weights.DEFAULT.meta["categories"]
            id2label = {i: name for i, name in enumerate(k400_labels)}
        except Exception:
            pass  # Keep generic labels as fallback

    label = id2label.get(pred_idx, f"class_{pred_idx}")

    top5_vals, top5_idxs = probs[0].topk(5)
    top5 = [
        {"label": id2label.get(i.item(), f"class_{i.item()}"), "confidence": v.item()}
        for v, i in zip(top5_vals, top5_idxs)
    ]

    model.cpu()
    del model, processor, inputs, processed, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label, confidence, top5


def _infer_torchvision(info: Dict, frames: List[np.ndarray], device: str):
    """Run inference with a torchvision video classification model."""
    import torchvision.models.video as vm

    model_fn = getattr(vm, info["model_fn"])
    weights_cls = getattr(vm, info["weights_class"])
    weights = weights_cls.DEFAULT

    model = model_fn(weights=weights)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device).half().eval()
    else:
        model = model.eval()
        device = "cpu"

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    # VideoClassification transform expects: (T, C, H, W) uint8
    # It handles: float conversion -> resize -> crop -> normalize
    # Output: (T, C, H, W) float
    frames_tensor = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1)  # (H,W,C) uint8 -> (C,H,W) uint8
        for f in frames
    ])  # (T, C, H, W) uint8

    processed = preprocess(frames_tensor)  # (C, T, H, W) float

    # Model expects (B, C, T, H, W) — just add batch dim
    input_tensor = processed.unsqueeze(0)  # (1, C, T, H, W)
    if device == "cuda":
        input_tensor = input_tensor.to(device).half()
    else:
        input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output.float(), dim=-1)

    pred_idx = probs.argmax(-1).item()
    confidence = probs[0][pred_idx].item()
    label = categories[pred_idx]

    top5_vals, top5_idxs = probs[0].topk(5)
    top5 = [
        {"label": categories[i.item()], "confidence": v.item()}
        for v, i in zip(top5_vals, top5_idxs)
    ]

    model.cpu()
    del model, input_tensor, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label, confidence, top5


def run_sota_benchmark(
    frames: List[np.ndarray],
    device: str = "cuda",
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict[str, Dict]:
    """
    Run all enabled SOTA models sequentially on extracted video frames.

    Dispatches to HuggingFace or torchvision inference based on model source.
    """
    results = {}
    total = len(SOTA_MODELS)

    for step, (name, info) in enumerate(SOTA_MODELS.items()):
        if cancel_check and cancel_check():
            break

        source = info.get("source", "huggingface")

        # Check prerequisites
        if source == "huggingface" and not check_transformers_available():
            results[name] = {
                "label": "Error", "confidence": 0, "latency_ms": 0,
                "params": info["params"], "year": info["year"],
                "type": info["type"], "paper_acc": info.get("paper_acc", "N/A"),
                "top5": [], "status": "error",
                "error_msg": "Install: pip install transformers",
            }
            if progress_callback:
                progress_callback(name, step + 1, total, f"{name}: needs transformers package")
            continue

        if progress_callback:
            progress_callback(name, step, total, f"Loading {name}...")

        try:
            t0 = time.perf_counter()

            # Sample the required number of frames
            num_frames = info.get("num_frames", 16)
            if len(frames) >= num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled = [frames[int(i)] for i in indices]
            else:
                sampled = list(frames) + [frames[-1]] * (num_frames - len(frames))
                sampled = sampled[:num_frames]

            if progress_callback:
                progress_callback(name, step, total, f"Running {name} inference...")

            # Dispatch to the right inference engine
            if source == "huggingface":
                label, confidence, top5 = _infer_huggingface(info, sampled, device)
            elif source == "torchvision":
                label, confidence, top5 = _infer_torchvision(info, sampled, device)
            else:
                raise ValueError(f"Unknown source: {source}")

            latency = (time.perf_counter() - t0) * 1000

            results[name] = {
                "label": label,
                "confidence": confidence,
                "latency_ms": latency,
                "params": info["params"],
                "flops": info.get("flops", "N/A"),
                "year": info["year"],
                "type": info["type"],
                "paper_acc": info.get("paper_acc", "N/A"),
                "top5": top5,
                "status": "success",
            }

            if progress_callback:
                progress_callback(name, step + 1, total, f"{name}: {label} ({confidence:.1%})")

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results[name] = {
                "label": "Error",
                "confidence": 0.0,
                "latency_ms": 0,
                "params": info.get("params", "?"),
                "flops": info.get("flops", "?"),
                "year": info.get("year", 0),
                "type": info.get("type", "?"),
                "paper_acc": info.get("paper_acc", "N/A"),
                "top5": [],
                "status": "error",
                "error_msg": str(e),
            }

            if progress_callback:
                progress_callback(name, step + 1, total, f"{name}: FAILED - {str(e)[:60]}")

    return results


# ══════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════

def generate_benchmark_report(
    our_result: Dict,
    sota_results: Dict[str, Dict],
    video_source: str,
    output_path: Optional[str] = None,
) -> str:
    """Generate a comprehensive Markdown benchmark report."""
    lines = []
    lines.append("# SOTA Benchmark Report")
    lines.append(f"\n**Video Source:** `{video_source}`")
    lines.append(f"**Benchmark Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Device:** {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Live Benchmark Table ──
    lines.append("## Live Benchmark Results")
    lines.append("")
    lines.append("| Model | Year | Type | Prediction | Confidence | Latency (ms) | Params | GFLOPs | Paper Acc. |")
    lines.append("|-------|------|------|------------|:----------:|:------------:|:------:|:------:|:---------:|")

    # Our model row
    lines.append(
        f"| **Ours (Compact R(2+1)D)** | **{our_result.get('year', 2026)}** | "
        f"**{our_result.get('type', '(2+1)D')}** | "
        f"**{our_result['label']}** | **{our_result['confidence']:.1%}** | "
        f"**{our_result['latency_ms']:.0f}** | **{our_result['params']}** | "
        f"**{our_result.get('flops', '0.47G')}** | "
        f"**{our_result.get('paper_acc', 'N/A')}** |"
    )

    for name, data in sota_results.items():
        if data.get("status") == "success":
            lines.append(
                f"| {name} | {data['year']} | {data['type']} | "
                f"{data['label']} | {data['confidence']:.1%} | "
                f"{data['latency_ms']:.0f} | {data['params']} | "
                f"{data.get('flops', 'N/A')} | "
                f"{data.get('paper_acc', 'N/A')} |"
            )
        else:
            lines.append(
                f"| {name} | {data['year']} | {data['type']} | "
                f"Error | - | - | {data['params']} | {data.get('flops', '?')} | {data.get('paper_acc', 'N/A')} |"
            )

    lines.append("")

    # ── Reference Models Table ──
    lines.append("## Historical Reference (Published Accuracy)")
    lines.append("")
    lines.append("| Model | Year | Type | UCF-101 Acc. | Params | GFLOPs |")
    lines.append("|-------|------|------|:------------:|:------:|:------:|")

    our_params = our_result.get("params_raw", 279683)
    lines.append(
        f"| **Ours (Compact R(2+1)D)** | **2026** | **(2+1)D Residual + GAP** | "
        f"**{our_result.get('paper_acc', '93.83%')}** | **{our_result['params']}** | **{our_result.get('flops', '0.47G')}** |"
    )

    for name, ref in REFERENCE_MODELS.items():
        lines.append(
            f"| {name} | {ref['year']} | {ref['type']} | "
            f"{ref['paper_acc']} | {ref['params']} | {ref.get('flops', 'N/A')} |"
        )

    lines.append("")

    # ── Efficiency Analysis ──
    lines.append("## Efficiency Analysis")
    lines.append("")
    lines.append("| Model | Params | Ratio vs Ours | Latency (ms) | GFLOPs |")
    lines.append("|-------|:------:|:-------------:|:------------:|:------:|")
    lines.append(f"| **Ours** | **{our_params:,}** | **1×** | **{our_result['latency_ms']:.0f}** | **{our_result.get('flops', '0.47G')}** |")

    for name, data in sota_results.items():
        param_str = data["params"].replace("M", "")
        try:
            param_val = float(param_str) * 1e6
            ratio = param_val / max(our_params, 1)
            lines.append(f"| {name} | {data['params']} | {ratio:.0f}× | {data.get('latency_ms', 0):.0f} | {data.get('flops', 'N/A')} |")
        except ValueError:
            lines.append(f"| {name} | {data['params']} | - | {data.get('latency_ms', 0):.0f} | {data.get('flops', 'N/A')} |")

    lines.append("")

    # ── Top-5 Predictions ──
    lines.append("## Top-5 Predictions per Model")
    lines.append("")

    lines.append("### Ours (Compact R(2+1)D)")
    if our_result.get("all_probs"):
        sorted_probs = sorted(our_result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:5]
        for cls, prob in sorted_probs:
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            lines.append(f"- `{bar}` {prob:.1%}  {cls}")
    lines.append("")

    for name, data in sota_results.items():
        lines.append(f"### {name}")
        if data.get("top5"):
            for pred in data["top5"]:
                bar = "█" * int(pred["confidence"] * 20) + "░" * (20 - int(pred["confidence"] * 20))
                lines.append(f"- `{bar}` {pred['confidence']:.1%}  {pred['label']}")
        elif data.get("status") == "error":
            lines.append(f"- Error: {data.get('error_msg', 'Unknown')}")
        lines.append("")

    # ── Key Insights ──
    lines.append("## Key Insights")
    lines.append("")
    lines.append(f"- Our model uses only **{our_params:,} parameters** (~{our_params/1e6:.2f}M)")
    lines.append(f"- Inference latency: **{our_result['latency_ms']:.0f} ms** on {('GPU' if torch.cuda.is_available() else 'CPU')}")

    successful = {k: v for k, v in sota_results.items() if v.get("status") == "success"}
    if successful:
        avg_sota_latency = np.mean([v["latency_ms"] for v in successful.values()])
        speedup = avg_sota_latency / max(our_result['latency_ms'], 1)
        lines.append(f"- Average SOTA model latency: **{avg_sota_latency:.0f} ms** ({speedup:.1f}× slower)")

        largest = max(successful, key=lambda k: float(successful[k]["params"].replace("M", "")) * 1e6)
        ratio = float(successful[largest]["params"].replace("M", "")) * 1e6 / max(our_params, 1)
        lines.append(f"- Our model is **{ratio:.0f}× smaller** than the largest tested ({largest}, {successful[largest]['params']})")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by HAR Control Center SOTA Benchmark Engine*")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
