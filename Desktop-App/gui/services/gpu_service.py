"""
HAR Control Center — GPU detection + selection service.
Auto-detects available CUDA GPUs, provides info for UI.
"""

import subprocess
import re


def detect_gpus() -> list:
    """
    Detect all available GPUs. Returns list of dicts:
    [{"id": 0, "name": "GTX 1660", "vram_mb": 6144, "driver": "596.49"}, ...]
    Falls back to CPU if no GPU found.
    """
    gpus = []

    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "id": i,
                    "name": props.name,
                    "vram_mb": props.total_memory // (1024 * 1024),
                    "driver": "",
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass

    # Try nvidia-smi for driver info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx = int(parts[0])
                    # Update driver info if we have torch data
                    for g in gpus:
                        if g["id"] == idx:
                            g["driver"] = parts[3]
                            break
                    else:
                        # No torch data, use nvidia-smi
                        gpus.append({
                            "id": idx,
                            "name": parts[1],
                            "vram_mb": int(float(parts[2])),
                            "driver": parts[3],
                            "compute_capability": "",
                        })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpus


def get_gpu_memory_usage(device_id: int = 0) -> dict:
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_id) // (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device_id) // (1024 * 1024)
            total = torch.cuda.get_device_properties(device_id).total_memory // (1024 * 1024)
            return {"allocated_mb": allocated, "reserved_mb": reserved, "total_mb": total}
    except:
        pass
    return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}


def get_device_string(gpu_id: int = -1) -> str:
    """Get PyTorch device string. -1 = CPU."""
    if gpu_id < 0:
        return "cpu"
    return f"cuda:{gpu_id}"
