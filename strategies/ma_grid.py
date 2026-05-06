"""Moving Average Dynamic Grid Strategy — 均线动态网格策略。

核心思想：以 SMA 均线为基准线，当价格偏离均线超出动态阈值时入场（均值回复），
当价格回到止盈/止损区域时离场。

信号逻辑：
  - 入场：价格低于均线，且偏离程度（MADevPct）达到动态网格步长的整数倍
  - 出场：持仓达到最大持有天数，或偏离回归到止盈网格线以上，或跌破止损网格线
  - 网格步长随波动率和趋势强度动态调整（高波动 → 宽网格，强趋势 → 宽网格）

GPU 加速：MLX（Apple Silicon）向量化批量回测。
"""

from typing import Tuple

import numpy as np
import pandas as pd

from utils.indicators import compute_ma_indicators
from utils.scan import indicator_and_scan


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成 — 逐 bar 循环的简单实现
# ══════════════════════════════════════════════════════════════════

def generate_grid_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,        # (close - MA) / MA，价格偏离均线的百分比
    dev_trend: np.ndarray,      # dev_pct 的短期斜率，反映趋势方向
    rolling_vol_pct: np.ndarray, # 滚动波动率（收益率的标准差）
    base_grid_pct: float = 0.012,     # 基础网格步长（如 1.2%）
    volatility_scale: float = 1.0,    # 波动率放大系数
    trend_sensitivity: float = 8.0,   # 趋势敏感度（趋势越强，网格越宽）
    max_grid_levels: int = 3,         # 最大网格层级（对应最多 3 倍步长的偏离）
    take_profit_grid: float = 0.85,   # 止盈：偏离回补到 entry_level * grid_step * 0.85
    stop_loss_grid: float = 1.6,      # 止损：偏离扩大到 entry_level * grid_step * 1.6
    max_holding_days: int = 30,       # 最大持仓天数
    cooldown_days: int = 1,           # 出场后的冷却期
    min_signal_strength: float = 0.45,# 最小信号强度（避免弱信号入场）
    position_size: float = np.inf,    # 单次仓位比例（inf = 全仓）
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成均线网格策略的入场/离场信号。

    返回值：
      entries: bool 数组 [n]，入场信号
      exits:   bool 数组 [n]，离场信号
      sizes:   float 数组 [n]，每次入场的仓位比例
    """
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)

    in_position = False
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    cooldown = 0

    for i in range(n):
        if (np.isnan(dev_pct[i]) or np.isnan(dev_trend[i])
                or np.isnan(rolling_vol_pct[i]) or close[i] <= 0):
            continue

        vol_mult = 1.0 + volatility_scale * max(rolling_vol_pct[i], 0.0)
        dynamic_grid_step = (
            base_grid_pct
            * (1.0 + trend_sensitivity * abs(dev_trend[i]))
            * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        if not in_position:
            if cooldown > 0:
                cooldown -= 1
                continue
            signal_strength = abs(dev_pct[i]) / max(dynamic_grid_step, 1e-9)
            entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
            entry_threshold = -entry_lvl * dynamic_grid_step
            if dev_pct[i] <= entry_threshold and signal_strength >= min_signal_strength:
                entries[i] = True
                sizes[i] = position_size
                in_position = True
                entry_bar = i
                entry_level = entry_lvl
                entry_grid_step = dynamic_grid_step
        else:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            if hold_limit or dev_pct[i] >= tp_threshold or dev_pct[i] <= -sl_threshold:
                exits[i] = True
                in_position = False
                cooldown = cooldown_days

    if in_position:
        exits[-1] = True

    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# 参数扫描入口
# ══════════════════════════════════════════════════════════════════

def scan_ma_strategy(close: pd.Series,
                     open_: pd.Series | None = None) -> pd.DataFrame:
    """MA 网格策略参数网格扫描。

    对 ma_window（MA 周期）、base_grid_pct（基础网格步长）、
    volatility_scale（波动率系数）、trend_sensitivity（趋势敏感度）、
    take_profit_grid（止盈倍数）、stop_loss_grid（止损倍数）
    进行全组合扫描，返回按 total_return 排序的结果。

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    return indicator_and_scan(
        close, "MA", compute_ma_indicators, "ma_window",
        windows=[20, 50, 75, 100, 150, 200],
        grid_pcts=[0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02],
        vol_scales=[0.5, 1.0, 1.5, 2.0, 3.0],
        trend_sens=[3.0, 6.0, 9.0, 12.0, 15.0],
        tpg_values=[0.5, 0.7, 0.85, 1.0],
        slg_values=[1.0, 1.3, 1.6, 2.0, 2.5],
        signal_fn=generate_grid_signals,
        signal_batch_fn=None,  # CPU-only, no MLX signal batch for MA grid
        open_=open_,
    )
