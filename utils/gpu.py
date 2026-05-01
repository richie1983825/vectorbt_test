"""GPU / CUDA 检测与工具函数。

提供统一的 GPU 检测、CuPy 导入和状态查询接口。
其他模块通过 xp() 获取 cupy（GPU）或 numpy（CPU）的统一数组接口。
"""

import subprocess

import numpy as np

# 缓存 GPU 检测结果，避免重复查询 nvidia-smi
_gpu_info: dict = {}


def detect_gpu() -> dict:
    """检测 NVIDIA GPU 和 CUDA 环境（幂等操作，结果缓存）。

    通过 nvidia-smi 获取 GPU 硬件信息，通过尝试导入 cupy 检测 CUDA 运行时。

    Returns:
        dict 包含:
          - gpu_detected:    是否检测到 NVIDIA GPU
          - gpu_name:        GPU 型号名称
          - memory_mb:       显存大小（MiB）
          - compute_cap:     计算能力版本
          - cupy_available:  CuPy 是否可用
          - cupy_version:    CuPy 版本号
          - xp:              cupy 模块（可用时）或 numpy（不可用时）
    """
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
        "xp": np,  # 默认使用 numpy
    }

    # ── 通过 nvidia-smi 获取 GPU 硬件信息 ──
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

    # ── 尝试导入 CuPy（GPU 加速库） ──
    try:
        import cupy as cp
        # 执行一次简单运算验证 CuPy 真正可用
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
    """返回缓存的 GPU 信息（若未检测则先执行检测）。"""
    return _gpu_info or detect_gpu()


def xp():
    """返回数组计算模块：CuPy 可用时返回 cupy，否则返回 numpy。

    用法：
        from utils.gpu import xp
        arr = xp().array([1, 2, 3])  # 自动选择 CPU 或 GPU
    """
    return gpu()["xp"]


def print_gpu_info(info: dict) -> None:
    """格式化打印 GPU 检测信息。"""
    print(f"GPU: {info['gpu_name']}")
    print(f"  Memory:      {info['memory_mb']} MiB")
    print(f"  Compute Cap: {info['compute_cap']}")
    if info["cupy_available"]:
        print(f"  CuPy:        {info['cupy_version']}  ✓ (GPU acceleration enabled)")
    else:
        print(f"  CuPy:        not installed (CPU-only mode)")
