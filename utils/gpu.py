"""GPU 检测与工具函数。

提供 MLX（Apple Silicon GPU）检测和状态查询接口。
不可用时回退到 numpy（CPU）。
"""

import numpy as np

_mlx_module = None
_mlx_checked = False


def _check_mlx():
    global _mlx_module, _mlx_checked
    if _mlx_checked:
        return _mlx_module
    _mlx_checked = True
    try:
        import mlx.core as mx
        _test = mx.arange(10, dtype=mx.float32)
        mx.eval(_test)
        _mlx_module = mx
    except Exception:
        pass
    return _mlx_module


def gpu() -> dict:
    """返回 GPU 检测信息。"""
    mx = _check_mlx()
    return {
        "mlx_available": mx is not None,
    }


def get_mlx():
    """返回 MLX 模块（Apple Silicon GPU），不可用时返回 None。"""
    _check_mlx()
    return _mlx_module


def has_gpu() -> bool:
    """是否有 GPU 加速可用（MLX）。"""
    return _check_mlx() is not None


def print_gpu_info() -> None:
    """格式化打印 GPU 检测信息。"""
    mx = _check_mlx()
    if mx is not None:
        print(f"  MLX:  available  ✓ (Apple Silicon GPU enabled)")
    else:
        print(f"  MLX:  not available (CPU-only mode)")


# 兼容旧代码的别名
def detect_gpu() -> dict:
    """兼容旧接口，返回 GPU 检测信息。"""
    return gpu()


def xp():
    """兼容旧代码：返回 numpy（MLX 不使用 cupy 的数组接口）。

    新代码应直接使用 mlx.core 或 numpy，不应依赖此函数。
    """
    return np
