"""GPU / CUDA detection and utilities."""

import subprocess

import numpy as np

_gpu_info: dict = {}


def detect_gpu() -> dict:
    """Detect NVIDIA GPU and CUDA capabilities (idempotent)."""
    global _gpu_info
    if _gpu_info:
        return _gpu_info

    info = {
        "gpu_detected": False,
        "gpu_name": "N/A",
        "memory_mb": 0,
        "compute_cap": "N/A",
        "cupy_available": False,
        "cupy_version": "N/A",
        "xp": np,
    }

    # --- nvidia-smi detection ---
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            info["gpu_name"] = parts[0] if len(parts) > 0 else "N/A"
            mem_str = parts[1] if len(parts) > 1 else "0 MiB"
            info["memory_mb"] = int(mem_str.replace("MiB", "").strip()) if "MiB" in mem_str else 0
            info["compute_cap"] = parts[2] if len(parts) > 2 else "N/A"
            info["gpu_detected"] = True
    except Exception:
        pass

    # --- CuPy detection ---
    try:
        import cupy as cp
        _test = cp.arange(10, dtype=cp.float64)
        _ = cp.cumsum(_test)
        info["cupy_available"] = True
        info["cupy_version"] = cp.__version__
        info["xp"] = cp
    except Exception:
        pass

    _gpu_info = info
    return info


def gpu() -> dict:
    """Return cached GPU info (call detect_gpu first)."""
    return _gpu_info or detect_gpu()


def xp():
    """Return cupy if CUDA is available, else numpy."""
    return gpu()["xp"]


def print_gpu_info(info: dict) -> None:
    print(f"GPU: {info['gpu_name']}")
    print(f"  Memory:      {info['memory_mb']} MiB")
    print(f"  Compute Cap: {info['compute_cap']}")
    if info["cupy_available"]:
        print(f"  CuPy:        {info['cupy_version']}  ✓ (GPU acceleration enabled)")
    else:
        print(f"  CuPy:        not installed (CPU-only mode)")
