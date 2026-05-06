"""Polyfit Switch Strategy — 多项式拟合基线 + 均线交叉双模式策略。

与 MA-Switch 策略的核心区别：
  1. 基线：使用滑动窗口线性回归（Polyfit，通常 252 天）预测价格中枢，
     替代简单均线（SMA）。Polyfit 基线在趋势行情中滞后更小，
     且能更好地过滤震荡噪音。
  2. Switch 离场：使用最高价回撤止损（trailing stop），
     从持仓期间的最高收盘价回撤 switch_trailing_stop 比例后离场。
  3. 参数：最大持仓天数 45（vs 30），仓位系数更大（0.92-0.99 vs 0.5）。

双模式说明：
  - Grid 模式（默认）：均值回复网格交易，在 Polyfit 基线下方挂网格多单。
  - Switch 模式（趋势追踪）：当价格连续横盘后在基线上方且偏离超过阈值时激活，
    使用快慢均线金叉入场 + 最高价回撤追踪止损离场。

GPU 加速：MLX（Apple Silicon）向量化批量回测。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import gpu
from utils.indicators import compute_polyfit_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成
# ══════════════════════════════════════════════════════════════════

def generate_polyfit_switch_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 45,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,
    position_sizing_coef: float = 30.0,
    switch_deviation_m1: float = 0.03,
    switch_deviation_m2: float = 0.02,
    switch_trailing_stop: float = 0.05,
    # ── v4 趋势确认入场 ──
    trend_entry_dp: float = 0.0,              # dp 回归此阈值以上才检查趋势
    trend_confirm_dp_slope: float = 0.0003,
    trend_confirm_vol: float = 1.2,
    # ── v4 ATR / 量能 / MA 离场 ──
    trend_atr_mult: float = 2.5,
    trend_atr_window: int = 14,
    trend_vol_climax: float = 3.0,
    # ── v4 dp 连续下跌离场 ──
    trend_decline_days: int = 0,              # dp 连续下跌 N 天则离场，0=禁用
    # ── v4 额外数据 ──
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    ma5: np.ndarray | None = None,
    ma10: np.ndarray | None = None,
    ma20: np.ndarray | None = None,
    return_handovers: bool = False,
) -> Tuple:
    """Polyfit-Switch v4：Grid TP → 趋势确认 → ATR 追踪离场。

    Grid 入场/止损不变。止盈时不直接卖出，而是检测趋势条件：
      - dp 斜率加速（dev_trend > trend_confirm_dp_slope）
      - 放量确认（vol / vol_ema > trend_confirm_vol）
      - MA 多头排列（ma5 > ma10 > ma20）
      3 取 2 → 无缝切换到 Switch 持仓。

    Switch 离场（四层递进）：
      1. ATR Chandelier 追踪止损（close < peak - atr_mult × ATR）
      2. 量能衰竭（放量收阳 > trend_vol_climax × vol_ema）
      3. MA 死叉（ma5 < ma10 且前 bar ma5 >= ma10）
      4. dp 跌回基线下方 → 检查 Grid 条件，满足则切换，否则清仓
    """
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)
    entry_modes = np.zeros(n, dtype=np.int8)
    handover_bars = np.zeros(n, dtype=bool) if return_handovers else None

    # 预计算 ATR 和 vol_ema
    _use_atr = high is not None and low is not None
    atr_arr = np.zeros(n, dtype=np.float64)
    if _use_atr:
        alpha_a = 2.0 / (trend_atr_window + 1)
        atr_ema = 0.0
        for i in range(1, n):
            if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])
                or np.isnan(close[i-1])):
                continue
            tr = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            if i == 1:
                atr_ema = tr
            else:
                atr_ema = alpha_a * tr + (1.0 - alpha_a) * atr_ema
            atr_arr[i] = atr_ema

    _use_vol = volume is not None
    vol_ema_arr = np.zeros(n, dtype=np.float64)
    if _use_vol:
        alpha_v = 2.0 / 21.0  # ~20-day EMA
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    # 状态
    position_mode = 0       # 0=idle, 1=grid, 2=switch
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    entry_close_price = np.nan
    cooldown = 0
    switch_peak = 0.0
    prev_ma5 = 0.0
    prev_ma10 = 0.0
    prev_dp = np.nan
    decline_count = 0

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        pb = poly_base[i]
        cl = close[i]
        fma = ma_fast[i]
        sma = ma_slow[i]
        m5 = ma5[i] if ma5 is not None else 0.0
        m10 = ma10[i] if ma10 is not None else 0.0
        m20 = ma20[i] if ma20 is not None else 0.0

        if (np.isnan(dp) or np.isnan(dt) or np.isnan(vp) or np.isnan(pb)
            or np.isnan(fma) or np.isnan(sma) or pb <= 0 or cl <= 0):
            prev_ma5, prev_ma10 = m5, m10
            continue

        # 动态网格步长
        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        if cooldown > 0:
            cooldown -= 1

        # ── IDLE：Grid 入场 ──
        if position_mode == 0:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan
            entry_close_price = np.nan

            if cooldown <= 0:
                signal_strength = abs(dp) / max(dynamic_grid_step, 1e-9)
                entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
                entry_threshold = -entry_lvl * dynamic_grid_step
                if dp <= entry_threshold and signal_strength >= min_signal_strength:
                    size = float(np.clip(
                        abs(dp) * (1.0 + max(vp, 0.0)) * position_sizing_coef,
                        0.0, position_size,
                    ))
                    if size > 0:
                        entries[i] = True
                        sizes[i] = size
                        entry_modes[i] = 1
                        position_mode = 1
                        entry_bar = i
                        entry_level = entry_lvl
                        entry_grid_step = dynamic_grid_step
                        entry_close_price = cl

        # ── Grid 持仓 ──
        elif position_mode == 1:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            if hold_limit or dp <= -sl_threshold:
                exits[i] = True
                position_mode = 0
                cooldown = cooldown_days
            elif dp >= trend_entry_dp:
                # ── 趋势确认（dp 高于阈值，2 取 1 即触发）──
                cond_slope = dt > trend_confirm_dp_slope
                cond_ma = False
                if ma5 is not None and ma10 is not None:
                    cond_ma = (not np.isnan(m5) and not np.isnan(m10)
                               and m5 > m10)
                trend_score = 1 if (cond_slope or cond_ma) else 0

                if trend_score >= 1:
                    # 趋势确认 → 切换到 Switch
                    position_mode = 2
                    entry_bar = i
                    entry_grid_step = max(base_grid_pct, 1e-9)
                    switch_peak = cl
                    if return_handovers:
                        handover_bars[i] = True
                elif dp >= tp_threshold:
                    # dp 已达止盈位但趋势未确认 → 正常 Grid 止盈离场
                    exits[i] = True
                    position_mode = 0
                    cooldown = cooldown_days

        # ── Switch 持仓（v4：dp连跌 → ATR追踪 → 量能 → 死叉 → Grid回退）──
        elif position_mode == 2:
            # Layer 0: dp 连续下跌 N 天 → 趋势逆转，立即离场
            if trend_decline_days > 0 and decline_count >= trend_decline_days:
                exits[i] = True
                position_mode = 0
                switch_peak = 0.0
                cooldown = cooldown_days
                decline_count = 0
                continue

            # Layer 1: ATR Chandelier 追踪止损
            trail_exit = False
            if _use_atr and atr_arr[i] > 0:
                trail_exit = cl <= switch_peak - trend_atr_mult * atr_arr[i]
            else:
                # fallback: fixed trailing stop
                trail_exit = cl <= switch_peak * (1.0 - switch_trailing_stop)

            # Layer 2: 量能衰竭（放量收阳 = 趋势末端）
            vol_climax = False
            if _use_vol:
                rv = volume[i] / max(vol_ema_arr[i], 1e-9)
                vol_climax = rv > trend_vol_climax and cl > close[i-1] if i > 0 else False

            # Layer 3: MA 死叉（ma5 下穿 ma10）
            death_cross = False
            if ma5 is not None and ma10 is not None:
                death_cross = (m5 < m10 and prev_ma5 >= prev_ma10
                               and not np.isnan(m5) and not np.isnan(m10))

            if trail_exit or vol_climax or death_cross:
                exits[i] = True
                position_mode = 0
                switch_peak = 0.0
                cooldown = cooldown_days
            elif dp < 0:
                # Layer 4: dp 跌回基线下方 → 尝试切换回 Grid
                signal_strength = abs(dp) / max(dynamic_grid_step, 1e-9)
                entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
                grid_entry_threshold = -entry_lvl * dynamic_grid_step
                if dp <= grid_entry_threshold and signal_strength >= min_signal_strength:
                    position_mode = 1
                    entry_bar = i
                    entry_level = entry_lvl
                    entry_grid_step = dynamic_grid_step
                    entry_close_price = cl
                    switch_peak = 0.0
                    if return_handovers:
                        handover_bars[i] = True
                else:
                    exits[i] = True
                    position_mode = 0
                    switch_peak = 0.0
                    cooldown = cooldown_days
            else:
                switch_peak = max(switch_peak, cl)

        # dp 连续下跌追踪
        if not np.isnan(prev_dp) and not np.isnan(dp):
            if dp < prev_dp:
                decline_count += 1
            else:
                decline_count = 0
        prev_dp = dp
        prev_ma5, prev_ma10 = m5, m10

    # 末尾强制平仓
    if position_mode != 0:
        exits[-1] = True

    if return_handovers:
        return entries, exits, sizes, entry_modes, handover_bars
    return entries, exits, sizes, entry_modes


def generate_grid_priority_switch_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    grid_entries: np.ndarray,         # 预计算的 Grid 信号
    grid_exits: np.ndarray,
    ma20: np.ndarray | None = None,
    ma60: np.ndarray | None = None,
    # ── Switch 入场 ──
    trend_entry_dp: float = 0.01,
    trend_confirm_dp_slope: float = 0.0003,
    # ── Switch 离场 ──
    trend_atr_mult: float = 2.5,
    trend_atr_window: int = 14,
    trend_vol_climax: float = 3.0,
    trend_decline_days: int = 3,
    # ── OHLCV ──
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid 优先的 Switch 策略：Grid 持续生效，Switch 仅在 Grid 空仓时交易。

    Grid 信号优先级最高：
      - Grid 入场 → Switch 强制离场
      - Switch 仅在 Grid IDLE 时才能入场
      - 两个策略独立，不共享仓位
    """
    n = len(close)
    sw_entries = np.zeros(n, dtype=bool)
    sw_exits = np.zeros(n, dtype=bool)
    sw_sizes = np.ones(n) * 0.99  # Switch 满仓

    # Pre-compute ATR & vol EMA
    _use_atr = high is not None and low is not None
    atr_arr = np.zeros(n, dtype=np.float64)
    if _use_atr:
        alpha_a = 2.0 / (trend_atr_window + 1)
        atr_ema = 0.0
        for i in range(1, n):
            if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])
                or np.isnan(close[i-1])):
                continue
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            atr_ema = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema
            atr_arr[i] = atr_ema

    _use_vol = volume is not None
    vol_ema_arr = np.zeros(n, dtype=np.float64)
    if _use_vol:
        alpha_v = 2.0 / 21.0
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    grid_in = False
    sw_in = False
    sw_peak = 0.0
    prev_dp = np.nan
    decline_count = 0
    prev_ma20 = 0.0; prev_ma60 = 0.0

    for i in range(n):
        dp = dev_pct[i]; dt = dev_trend[i]; cl = close[i]
        m20 = ma20[i] if ma20 is not None else 0.0
        m60 = ma60[i] if ma60 is not None else 0.0
        if np.isnan(dp) or np.isnan(dt) or np.isnan(cl) or cl <= 0:
            prev_ma20 = m20; prev_ma60 = m60; continue

        # ── 1. Grid exits first ──
        grid_just_exited = False
        if grid_in and grid_exits[i]:
            grid_in = False
            grid_just_exited = True

        # ── 2. Grid entry = force Switch exit + Grid enters ──
        if grid_entries[i]:
            if sw_in:
                sw_exits[i] = True
                sw_in = False
                sw_peak = 0.0
            grid_in = True
            prev_ma20 = m20; prev_ma60 = m60
            continue

        # ── 3. Only if Grid is idle ──
        if not grid_in and not grid_just_exited:
            if not sw_in:
                # Check Switch entry: dp above threshold + momentum or MA
                if dp >= trend_entry_dp:
                    cond_s = dt > trend_confirm_dp_slope
                    cond_m = (not np.isnan(m20) and not np.isnan(m60) and m20 > m60)
                    if cond_s or cond_m:
                        sw_entries[i] = True
                        sw_in = True
                        sw_peak = cl
                        decline_count = 0
            else:
                # Switch exit checks
                exit_now = False

                # dp decline
                if trend_decline_days > 0 and decline_count >= trend_decline_days:
                    exit_now = True
                # ATR trail
                elif _use_atr and atr_arr[i] > 0:
                    if cl <= sw_peak - trend_atr_mult * atr_arr[i]:
                        exit_now = True
                # Vol climax
                elif _use_vol:
                    rv = volume[i] / max(vol_ema_arr[i], 1e-9)
                    if rv > trend_vol_climax and i > 0 and cl > close[i-1]:
                        exit_now = True
                # MA death cross (MA20 < MA60)
                elif ma20 is not None and ma60 is not None:
                    if (not np.isnan(m20) and not np.isnan(m60)
                        and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
                        and m20 < m60 and prev_ma20 >= prev_ma60):
                        exit_now = True

                if exit_now:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                else:
                    sw_peak = max(sw_peak, cl)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp):
            if dp < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp
        prev_ma20 = m20; prev_ma60 = m60

    # Force exit at end
    if sw_in:
        sw_exits[-1] = True

    return sw_entries, sw_exits, sw_sizes


# ══════════════════════════════════════════════════════════════════
# v6 — OHLCV 增强版 Grid-priority Switch（基于统计分析优化）
#
# 入场过滤（基于 analyze_switch_surge.py 的 109 笔 Switch 统计分析）：
#   1. 微涨 0 ~ +0.5% → 绝对不买（胜率 12.5%, n=32）
#   2. 长上影 ≥30%   → 不买（胜率 18.4%, n=38）
#   3. 正常量能 0.8-1.5x → 不买（胜率 13.7%, n=51）
#   4. 阳线 + 连涨<4 天 → 不买（阳线胜率 23.8%, n=63）
#   5. 高位 ≥70% + 连涨<4 天 → 不买（高位胜率 32.1%, n=53）
#
# 离场改进：
#   a. dp 连续下跌天数从 3→2（更早离场, dp_decline 本身胜率 71.7%）
#   b. 新增 Grid-zone 预警：dp 接近 Grid 入场区时提前离场（避免 grid_force 0%胜率）
#   c. ATR 乘数降到 2.0（收紧追踪止损）
# ══════════════════════════════════════════════════════════════════

def _compute_entry_ohlcv_features(close, open_, high, low, volume, vol_ema_arr, i):
    """在 bar i 处提取 OHLCV 特征，用于入场过滤。"""
    cl = close[i]
    op = open_[i] if open_ is not None else cl
    hi = high[i] if high is not None else cl
    lo = low[i] if low is not None else cl
    vol = volume[i] if volume is not None else 0

    # 涨跌
    entry_day_ret = (cl / close[i-1] - 1.0) if i > 0 else 0.0

    # 连涨天数
    consecutive_up = 0
    for j in range(i - 1, -1, -1):
        if close[j] > close[j-1]:
            consecutive_up += 1
        else:
            break

    # 上下影线
    total_range = hi - lo
    if total_range > 0:
        body_high = max(cl, op)
        body_low = min(cl, op)
        upper_wick = hi - body_high
        lower_wick = body_low - lo
        upper_wick_pct = upper_wick / total_range
    else:
        upper_wick_pct = 0.0
        lower_wick = 0.0

    # 相对成交量 (20日均)
    if i >= 21 and vol_ema_arr[i] > 0:
        rel_vol = vol / vol_ema_arr[i]
    else:
        rel_vol = 1.0

    # 价格位置 (相对10日高低点)
    lookback = min(10, i)
    if lookback > 0:
        n_high = np.max(high[max(0, i-lookback):i+1]) if high is not None else cl
        n_low = np.min(low[max(0, i-lookback):i+1]) if low is not None else cl
        price_pos = (cl - n_low) / (n_high - n_low) if n_high > n_low else 0.5
    else:
        price_pos = 0.5

    is_green = cl >= op

    return {
        "entry_day_ret": entry_day_ret,
        "consecutive_up": consecutive_up,
        "upper_wick_pct": upper_wick_pct,
        "rel_vol": rel_vol,
        "price_position": price_pos,
        "is_green": is_green,
    }


def _check_entry_filters(feat, trend_entry_dp, trend_confirm_dp_slope):
    """检查入场过滤条件。返回 (pass, reject_reason)。"""
    # Filter 1: 微涨 0 ~ +0.5% → 绝对不买
    if 0.0 < feat["entry_day_ret"] < 0.005:
        return False, "micro_up"

    # Filter 2: 长上影 ≥30% → 不买
    if feat["upper_wick_pct"] >= 0.30:
        return False, "long_upper_wick"

    # Filter 3: 正常量能 0.8-1.5x → 不买（缩量或显著放量才可以）
    if 0.8 <= feat["rel_vol"] <= 1.5:
        return False, "normal_volume"

    # Filter 4: 阳线 + 连涨<4 天 → 不买
    if feat["is_green"] and feat["consecutive_up"] < 4:
        return False, "green_without_momentum"

    # Filter 5: 高位 ≥70% + 连涨<4 天 → 不买
    if feat["price_position"] >= 0.70 and feat["consecutive_up"] < 4:
        return False, "high_position"

    return True, "pass"


def generate_grid_priority_switch_signals_v6(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    grid_entries: np.ndarray,
    grid_exits: np.ndarray,
    ma20: np.ndarray | None = None,
    ma60: np.ndarray | None = None,
    # ── Switch 入场 ──
    trend_entry_dp: float = 0.01,
    trend_confirm_dp_slope: float = 0.0003,
    # ── Switch 离场（v6 优化默认值）──
    trend_atr_mult: float = 2.0,
    trend_atr_window: int = 14,
    trend_vol_climax: float = 3.0,
    trend_decline_days: int = 2,
    # ── OHLCV ──
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    # ── v6 增强开关 ──
    enable_ohlcv_filter: bool = True,      # 是否启用 OHLCV 入场过滤
    enable_early_exit: bool = True,        # 是否启用 dp< -0.5% 预警离场
    # ── 调试 ──
    return_filter_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | dict:
    """OHLCV 增强版 Grid-priority Switch 策略。

    相比原版的核心改进：
    1. 入场前检查 OHLCV 特征，过滤掉统计上极低胜率的入场
    2. dp 连续下跌更快离场（2天 vs 3天）
    3. dp 接近 Grid 入场区时提前离场，避免被 Grid 强制踢出（grid_force 胜率 0%）
    4. ATR 追踪止损更紧（2.0x vs 2.5x）
    """
    n = len(close)
    sw_entries = np.zeros(n, dtype=bool)
    sw_exits = np.zeros(n, dtype=bool)
    sw_sizes = np.ones(n) * 0.99

    filter_stats = {"micro_up": 0, "long_upper_wick": 0, "normal_volume": 0,
                    "green_without_momentum": 0, "high_position": 0, "passed": 0}

    # Pre-compute ATR
    _use_atr = high is not None and low is not None
    atr_arr = np.zeros(n, dtype=np.float64)
    if _use_atr:
        alpha_a = 2.0 / (trend_atr_window + 1)
        atr_ema = 0.0
        for i in range(1, n):
            if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])
                or np.isnan(close[i-1])):
                continue
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            atr_ema = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema
            atr_arr[i] = atr_ema

    # Pre-compute vol EMA
    _use_vol = volume is not None
    vol_ema_arr = np.zeros(n, dtype=np.float64)
    if _use_vol:
        alpha_v = 2.0 / 21.0
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    grid_in = False
    sw_in = False
    sw_peak = 0.0
    sw_entry_bar = -1
    prev_dp = np.nan
    decline_count = 0
    prev_ma20 = 0.0; prev_ma60 = 0.0

    for i in range(n):
        dp = dev_pct[i]; dt = dev_trend[i]; cl = close[i]
        m20 = ma20[i] if ma20 is not None else 0.0
        m60 = ma60[i] if ma60 is not None else 0.0
        if np.isnan(dp) or np.isnan(dt) or np.isnan(cl) or cl <= 0:
            prev_ma20 = m20; prev_ma60 = m60; continue

        # ── 1. Grid exits first ──
        grid_just_exited = False
        if grid_in and grid_exits[i]:
            grid_in = False
            grid_just_exited = True

        # ── 2. Grid entry = force Switch exit + Grid enters ──
        if grid_entries[i]:
            if sw_in:
                sw_exits[i] = True
                sw_in = False
                sw_peak = 0.0
            grid_in = True
            prev_ma20 = m20; prev_ma60 = m60
            continue

        # ── 3. Only if Grid is idle ──
        if not grid_in and not grid_just_exited:
            if not sw_in:
                # Check Switch entry: dp above threshold + momentum or MA
                if dp >= trend_entry_dp:
                    cond_s = dt > trend_confirm_dp_slope
                    cond_m = (not np.isnan(m20) and not np.isnan(m60) and m20 > m60)
                    if cond_s or cond_m:
                        # ── v6 OHLCV 入场过滤 ──
                        if enable_ohlcv_filter and i >= 10:
                            feat = _compute_entry_ohlcv_features(
                                close, open_, high, low, volume, vol_ema_arr, i)
                            passed, reason = _check_entry_filters(
                                feat, trend_entry_dp, trend_confirm_dp_slope)
                            if not passed:
                                filter_stats[reason] += 1
                                prev_ma20 = m20; prev_ma60 = m60
                                continue
                            filter_stats["passed"] += 1

                        sw_entries[i] = True
                        sw_in = True
                        sw_peak = cl
                        sw_entry_bar = i
                        decline_count = 0
            else:
                exit_now = False

                # ── v6: dp 接近 Grid 入场区预警（避免 grid_force 0%胜率）──
                if enable_early_exit and dp < -0.005 and decline_count >= 1:
                    exit_now = True
                # dp decline (v6: 2天 vs 原来3天)
                elif trend_decline_days > 0 and decline_count >= trend_decline_days:
                    exit_now = True
                # ATR trail (v6: tighter multiplier)
                elif _use_atr and atr_arr[i] > 0:
                    if cl <= sw_peak - trend_atr_mult * atr_arr[i]:
                        exit_now = True
                # Vol climax
                elif _use_vol:
                    rv = volume[i] / max(vol_ema_arr[i], 1e-9)
                    if rv > trend_vol_climax and i > 0 and cl > close[i-1]:
                        exit_now = True
                # MA death cross (MA20 < MA60)
                elif ma20 is not None and ma60 is not None:
                    if (not np.isnan(m20) and not np.isnan(m60)
                        and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
                        and m20 < m60 and prev_ma20 >= prev_ma60):
                        exit_now = True

                if exit_now:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                else:
                    sw_peak = max(sw_peak, cl)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp):
            if dp < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp
        prev_ma20 = m20; prev_ma60 = m60

    # Force exit at end
    if sw_in:
        sw_exits[-1] = True

    if return_filter_stats:
        return {
            "sw_entries": sw_entries,
            "sw_exits": sw_exits,
            "sw_sizes": sw_sizes,
            "filter_stats": filter_stats,
        }
    return sw_entries, sw_exits, sw_sizes


# ══════════════════════════════════════════════════════════════════
# v7 — v6 + 顶部规避模块
#
# 基于 7 个历史顶部日的共同特征：
#   短期急涨(5日>5%) + 高位(>80%) + 异常K线(大振幅/长上影/巨量阴线)
#   → 后续3-10天跌3-5%
#
# 规避动作：
#   1. 检测到顶部信号 → 强制离场 Switch 持仓
#   2. 顶部信号后 N 天内禁止 Grid 新入场
#   3. 顶部信号后 N 天内禁止 Switch 新入场
# ══════════════════════════════════════════════════════════════════

def generate_grid_priority_switch_signals_v7(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    grid_entries: np.ndarray,
    grid_exits: np.ndarray,
    ma20: np.ndarray | None = None,
    ma60: np.ndarray | None = None,
    # ── Switch 入场 ──
    trend_entry_dp: float = 0.01,
    trend_confirm_dp_slope: float = 0.0003,
    # ── Switch 离场（v6 优化默认值）──
    trend_atr_mult: float = 2.0,
    trend_atr_window: int = 14,
    trend_vol_climax: float = 3.0,
    trend_decline_days: int = 2,
    # ── OHLCV ──
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    # ── v6 增强开关 ──
    enable_ohlcv_filter: bool = True,
    enable_early_exit: bool = True,
    # ── v7 顶部规避 ──
    enable_top_avoidance: bool = True,
    top_ret_5d: float = 0.05,          # 5日涨幅阈值
    top_price_pos: float = 0.80,        # 20日价位阈值
    top_amplitude: float = 0.025,       # 异常振幅阈值
    top_block_days: int = 3,            # 顶部后禁止入场天数
    # ── 调试 ──
    return_filter_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | dict:
    """v7: OHLCV增强 + 顶部规避 Grid-priority Switch 策略。

    在 v6 基础上增加顶部检测：
      条件：5日涨幅>top_ret_5d AND 价位>top_price_pos AND 收阴
            AND (振幅>top_amplitude OR 量异常)
      动作：强制离场Switch + 禁止入场 top_block_days 天
    """
    n = len(close)
    sw_entries = np.zeros(n, dtype=bool)
    sw_exits = np.zeros(n, dtype=bool)
    sw_sizes = np.ones(n) * 0.99

    filter_stats = {"micro_up": 0, "long_upper_wick": 0, "normal_volume": 0,
                    "green_without_momentum": 0, "high_position": 0, "passed": 0,
                    "top_avoided": 0}

    # Pre-compute ATR
    _use_atr = high is not None and low is not None
    atr_arr = np.zeros(n, dtype=np.float64)
    if _use_atr:
        alpha_a = 2.0 / (trend_atr_window + 1)
        atr_ema = 0.0
        for i in range(1, n):
            if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])
                or np.isnan(close[i-1])):
                continue
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            atr_ema = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema
            atr_arr[i] = atr_ema

    # Pre-compute vol EMA
    _use_vol = volume is not None
    vol_ema_arr = np.zeros(n, dtype=np.float64)
    if _use_vol:
        alpha_v = 2.0 / 21.0
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    grid_in = False
    sw_in = False
    sw_peak = 0.0
    sw_entry_bar = -1
    prev_dp = np.nan
    decline_count = 0
    prev_ma20 = 0.0; prev_ma60 = 0.0

    # 顶部规避状态
    top_block_until = -1
    top_signal_today = False

    # 预计算20日高低
    h20_arr = np.zeros(n, dtype=np.float64)
    l20_arr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i >= 19:
            h20_arr[i] = np.max(high[max(0,i-19):i+1]) if high is not None else close[i]
            l20_arr[i] = np.min(low[max(0,i-19):i+1]) if low is not None else close[i]

    for i in range(n):
        dp = dev_pct[i]; dt = dev_trend[i]; cl = close[i]
        m20 = ma20[i] if ma20 is not None else 0.0
        m60 = ma60[i] if ma60 is not None else 0.0
        if np.isnan(dp) or np.isnan(dt) or np.isnan(cl) or cl <= 0:
            prev_ma20 = m20; prev_ma60 = m60; continue

        # ── v7 顶部检测 ──
        top_signal_today = False
        if enable_top_avoidance and i >= 20 and high is not None and low is not None:
            ret_5d = close[i] / close[max(0, i-5)] - 1.0 if i >= 5 else 0.0
            price_pos = (close[i] - l20_arr[i]) / (h20_arr[i] - l20_arr[i]) if h20_arr[i] > l20_arr[i] else 0.5
            is_bearish = close[i] < open_[i] if open_ is not None else close[i] < close[i-1]
            amplitude = (high[i] - low[i]) / close[i]
            vol_ratio = volume[i] / vol_ema_arr[i] if _use_vol and vol_ema_arr[i] > 0 else 1.0

            if (ret_5d > top_ret_5d and price_pos > top_price_pos and is_bearish
                and amplitude > top_amplitude):
                top_signal_today = True
                top_block_until = i + top_block_days
                # 强制离场 Switch
                if sw_in:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                filter_stats["top_avoided"] += 1

        # ── 1. Grid exits first ──
        grid_just_exited = False
        if grid_in and grid_exits[i]:
            grid_in = False
            grid_just_exited = True

        # ── 2. Grid entry = force Switch exit + Grid enters ──
        #     但顶部规避期间禁止 Grid 入场
        if grid_entries[i] and i > top_block_until:
            if sw_in:
                sw_exits[i] = True
                sw_in = False
                sw_peak = 0.0
            grid_in = True
            prev_ma20 = m20; prev_ma60 = m60
            continue

        # ── 3. Only if Grid is idle ──
        if not grid_in and not grid_just_exited:
            if not sw_in:
                # 顶部规避期间禁止 Switch 入场
                if i <= top_block_until:
                    prev_ma20 = m20; prev_ma60 = m60
                    continue

                if dp >= trend_entry_dp:
                    cond_s = dt > trend_confirm_dp_slope
                    cond_m = (not np.isnan(m20) and not np.isnan(m60) and m20 > m60)
                    if cond_s or cond_m:
                        if enable_ohlcv_filter and i >= 10:
                            feat = _compute_entry_ohlcv_features(
                                close, open_, high, low, volume, vol_ema_arr, i)
                            passed, reason = _check_entry_filters(
                                feat, trend_entry_dp, trend_confirm_dp_slope)
                            if not passed:
                                filter_stats[reason] += 1
                                prev_ma20 = m20; prev_ma60 = m60
                                continue
                            filter_stats["passed"] += 1

                        sw_entries[i] = True
                        sw_in = True
                        sw_peak = cl
                        sw_entry_bar = i
                        decline_count = 0
            else:
                exit_now = False

                if enable_early_exit and dp < -0.005 and decline_count >= 1:
                    exit_now = True
                elif trend_decline_days > 0 and decline_count >= trend_decline_days:
                    exit_now = True
                elif _use_atr and atr_arr[i] > 0:
                    if cl <= sw_peak - trend_atr_mult * atr_arr[i]:
                        exit_now = True
                elif _use_vol:
                    rv = volume[i] / max(vol_ema_arr[i], 1e-9)
                    if rv > trend_vol_climax and i > 0 and cl > close[i-1]:
                        exit_now = True
                elif ma20 is not None and ma60 is not None:
                    if (not np.isnan(m20) and not np.isnan(m60)
                        and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
                        and m20 < m60 and prev_ma20 >= prev_ma60):
                        exit_now = True

                if exit_now:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                else:
                    sw_peak = max(sw_peak, cl)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp):
            if dp < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp
        prev_ma20 = m20; prev_ma60 = m60

    if sw_in:
        sw_exits[-1] = True

    if return_filter_stats:
        return {
            "sw_entries": sw_entries,
            "sw_exits": sw_exits,
            "sw_sizes": sw_sizes,
            "filter_stats": filter_stats,
        }
    return sw_entries, sw_exits, sw_sizes


# ══════════════════════════════════════════════════════════════════
# v8 — v7 + 恐慌低吸模块
#
# 恐慌低吸 vs 顶部规避的区别：
#   顶部反转：前期急涨>5% + 高位 + 阴线 + 大振幅 → BLOCK（V7 逻辑）
#   恐慌低吸：前期温和(<5%) + 大阴线 + 高量 + 大振幅 → BUY（V8 新增）
#
# 基于 2025-04-07 和 2026-03-23 等恐慌日的分析：
#   - 恐慌日前期涨幅温和（<5%），非顶部反转
#   - 大振幅阴线 + 放量（>1.5x 均量）
#   - 后续 1-3 天 V 型反弹概率高（12 个历史案例中 67% 正收益）
#
# 入场逻辑：
#   1. 检测到恐慌低吸 → Grid/Switch 均空闲时直接 Switch 入场
#   2. 跳过 OHLCV 入场过滤（恐慌模式与正常入场条件相反）
#   3. 顶部规避期间不发生恐慌低吸（顶部后禁止入场覆盖恐慌）
# ══════════════════════════════════════════════════════════════════

def generate_grid_priority_switch_signals_v8(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    grid_entries: np.ndarray,
    grid_exits: np.ndarray,
    ma20: np.ndarray | None = None,
    ma60: np.ndarray | None = None,
    # ── Switch 入场 ──
    trend_entry_dp: float = 0.01,
    trend_confirm_dp_slope: float = 0.0003,
    # ── Switch 离场（v6 优化默认值）──
    trend_atr_mult: float = 2.0,
    trend_atr_window: int = 14,
    trend_vol_climax: float = 3.0,
    trend_decline_days: int = 2,
    # ── OHLCV ──
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    # ── v6 增强开关 ──
    enable_ohlcv_filter: bool = True,
    enable_early_exit: bool = True,
    # ── v7 顶部规避 ──
    enable_top_avoidance: bool = True,
    top_ret_5d: float = 0.05,
    top_price_pos: float = 0.80,
    top_amplitude: float = 0.025,
    top_block_days: int = 3,
    # ── v8 恐慌低吸 ──
    enable_panic_dip_buy: bool = True,
    panic_ret_5d_max: float = 0.05,        # 5日涨幅上限（低于此值=恐慌，高于=顶部）
    panic_amplitude: float = 0.03,          # 最小振幅
    panic_vol_ratio: float = 1.5,           # 最小量比
    # ── 调试 ──
    return_filter_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | dict:
    """v8: v7顶部规避 + 恐慌低吸 Grid-priority Switch 策略。

    在 v7 基础上增加恐慌低吸检测：
      条件：收阴 AND 振幅>panic_amplitude AND 量比>panic_vol_ratio
            AND 5日涨幅<panic_ret_5d_max（区别于顶部反转）
      动作：Grid/Switch均空闲时直接Switch入场（跳过OHLCV过滤）
    """
    n = len(close)
    sw_entries = np.zeros(n, dtype=bool)
    sw_exits = np.zeros(n, dtype=bool)
    sw_sizes = np.ones(n) * 0.99

    filter_stats = {"micro_up": 0, "long_upper_wick": 0, "normal_volume": 0,
                    "green_without_momentum": 0, "high_position": 0, "passed": 0,
                    "top_avoided": 0, "panic_dip_buy": 0}

    # Pre-compute ATR
    _use_atr = high is not None and low is not None
    atr_arr = np.zeros(n, dtype=np.float64)
    if _use_atr:
        alpha_a = 2.0 / (trend_atr_window + 1)
        atr_ema = 0.0
        for i in range(1, n):
            if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])
                or np.isnan(close[i-1])):
                continue
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            atr_ema = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema
            atr_arr[i] = atr_ema

    # Pre-compute vol EMA
    _use_vol = volume is not None
    vol_ema_arr = np.zeros(n, dtype=np.float64)
    if _use_vol:
        alpha_v = 2.0 / 21.0
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    grid_in = False
    sw_in = False
    sw_peak = 0.0
    sw_entry_bar = -1
    prev_dp = np.nan
    decline_count = 0
    prev_ma20 = 0.0; prev_ma60 = 0.0

    # 顶部规避状态
    top_block_until = -1
    top_signal_today = False

    # 预计算20日高低
    h20_arr = np.zeros(n, dtype=np.float64)
    l20_arr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i >= 19:
            h20_arr[i] = np.max(high[max(0,i-19):i+1]) if high is not None else close[i]
            l20_arr[i] = np.min(low[max(0,i-19):i+1]) if low is not None else close[i]

    for i in range(n):
        dp = dev_pct[i]; dt = dev_trend[i]; cl = close[i]
        m20 = ma20[i] if ma20 is not None else 0.0
        m60 = ma60[i] if ma60 is not None else 0.0
        if np.isnan(dp) or np.isnan(dt) or np.isnan(cl) or cl <= 0:
            prev_ma20 = m20; prev_ma60 = m60; continue

        # ── v7 顶部检测 ──
        top_signal_today = False
        if enable_top_avoidance and i >= 20 and high is not None and low is not None:
            ret_5d = close[i] / close[max(0, i-5)] - 1.0 if i >= 5 else 0.0
            price_pos = (close[i] - l20_arr[i]) / (h20_arr[i] - l20_arr[i]) if h20_arr[i] > l20_arr[i] else 0.5
            is_bearish = close[i] < open_[i] if open_ is not None else close[i] < close[i-1]
            amplitude = (high[i] - low[i]) / close[i]
            vol_ratio = volume[i] / vol_ema_arr[i] if _use_vol and vol_ema_arr[i] > 0 else 1.0

            if (ret_5d > top_ret_5d and price_pos > top_price_pos and is_bearish
                and amplitude > top_amplitude):
                top_signal_today = True
                top_block_until = i + top_block_days
                if sw_in:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                filter_stats["top_avoided"] += 1

        # ── v8 恐慌低吸检测（顶部规避之后、Grid处理之前）──
        panic_signal_today = False
        if (enable_panic_dip_buy and i >= 20 and not top_signal_today
            and i > top_block_until
            and high is not None and low is not None and open_ is not None):
            ret_5d_p = close[i] / close[max(0, i-5)] - 1.0 if i >= 5 else 0.0
            is_bearish_p = close[i] < open_[i]
            amplitude_p = (high[i] - low[i]) / close[i]
            vol_ratio_p = volume[i] / vol_ema_arr[i] if _use_vol and vol_ema_arr[i] > 0 else 1.0

            # 恐慌条件：大阴线 + 高量 + 大振幅 + 前期涨幅温和（非顶部）
            if (is_bearish_p and amplitude_p > panic_amplitude
                and vol_ratio_p > panic_vol_ratio
                and ret_5d_p < panic_ret_5d_max):
                panic_signal_today = True
                filter_stats["panic_dip_buy"] += 1
                # 空闲时直接入场（跳过OHLCV过滤）
                if not grid_in and not sw_in:
                    sw_entries[i] = True
                    sw_in = True
                    sw_peak = cl
                    sw_entry_bar = i
                    decline_count = 0

        # ── 1. Grid exits first ──
        grid_just_exited = False
        if grid_in and grid_exits[i]:
            grid_in = False
            grid_just_exited = True

        # ── 2. Grid entry = force Switch exit + Grid enters ──
        #     但顶部规避期间禁止 Grid 入场
        if grid_entries[i] and i > top_block_until:
            if sw_in:
                sw_exits[i] = True
                sw_in = False
                sw_peak = 0.0
            grid_in = True
            prev_ma20 = m20; prev_ma60 = m60
            continue

        # ── 3. Only if Grid is idle ──
        if not grid_in and not grid_just_exited:
            if not sw_in:
                # 顶部规避期间禁止 Switch 入场（恐慌入场已在上面处理）
                if i <= top_block_until:
                    prev_ma20 = m20; prev_ma60 = m60
                    continue

                # 恐慌日已入场则跳过正常入场逻辑
                if panic_signal_today and sw_in:
                    prev_ma20 = m20; prev_ma60 = m60
                    continue

                if dp >= trend_entry_dp:
                    cond_s = dt > trend_confirm_dp_slope
                    cond_m = (not np.isnan(m20) and not np.isnan(m60) and m20 > m60)
                    if cond_s or cond_m:
                        if enable_ohlcv_filter and i >= 10:
                            feat = _compute_entry_ohlcv_features(
                                close, open_, high, low, volume, vol_ema_arr, i)
                            passed, reason = _check_entry_filters(
                                feat, trend_entry_dp, trend_confirm_dp_slope)
                            if not passed:
                                filter_stats[reason] += 1
                                prev_ma20 = m20; prev_ma60 = m60
                                continue
                            filter_stats["passed"] += 1

                        sw_entries[i] = True
                        sw_in = True
                        sw_peak = cl
                        sw_entry_bar = i
                        decline_count = 0
            else:
                exit_now = False

                if enable_early_exit and dp < -0.005 and decline_count >= 1:
                    exit_now = True
                elif trend_decline_days > 0 and decline_count >= trend_decline_days:
                    exit_now = True
                elif _use_atr and atr_arr[i] > 0:
                    if cl <= sw_peak - trend_atr_mult * atr_arr[i]:
                        exit_now = True
                elif _use_vol:
                    rv = volume[i] / max(vol_ema_arr[i], 1e-9)
                    if rv > trend_vol_climax and i > 0 and cl > close[i-1]:
                        exit_now = True
                elif ma20 is not None and ma60 is not None:
                    if (not np.isnan(m20) and not np.isnan(m60)
                        and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
                        and m20 < m60 and prev_ma20 >= prev_ma60):
                        exit_now = True

                if exit_now:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                else:
                    sw_peak = max(sw_peak, cl)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp):
            if dp < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp
        prev_ma20 = m20; prev_ma60 = m60

    if sw_in:
        sw_exits[-1] = True

    if return_filter_stats:
        return {
            "sw_entries": sw_entries,
            "sw_exits": sw_exits,
            "sw_sizes": sw_sizes,
            "filter_stats": filter_stats,
        }
    return sw_entries, sw_exits, sw_sizes




# ══════════════════════════════════════════════════════════════════
# 两阶段参数扫描（MLX）
# ══════════════════════════════════════════════════════════════════

def scan_polyfit_switch_two_stage(close: pd.Series,
                                   open_: pd.Series | None = None) -> pd.DataFrame:
    """两阶段扫描：Stage 1 复用 Polyfit-Grid 扫描，Stage 2 扫描 Switch 参数。

    Stage 1: 调用 scan_polyfit_grid() 获取纯 Grid 最优参数（与 Polyfit-Grid 策略一致）
    Stage 2: 固定最优 Grid 参数，CPU 逐组合生成信号 + MLX 批量回测

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    use_mlx = gpu()["mlx_available"]
    open_arr = open_.values if open_ is not None else None

    # ── Stage 1: 复用 Polyfit-Grid 扫描（纯Grid，确保与Polyfit-Grid策略一致） ──
    print("  [PolyfitSwitch Stage 1] Reusing Polyfit-Grid scan...")
    from strategies.polyfit_grid import scan_polyfit_grid as _scan_grid
    stage1_df = _scan_grid(close, open_=open_)
    if stage1_df.empty:
        return pd.DataFrame()

    best_grid = stage1_df.nlargest(1, "total_return").iloc[0]
    best_tw = int(best_grid["trend_window_days"])
    best_vw = int(best_grid["vol_window_days"])
    best_bgp = best_grid["base_grid_pct"]
    best_vs = best_grid["volatility_scale"]
    best_ts = best_grid["trend_sensitivity"]
    best_max_gl = int(best_grid["max_grid_levels"])
    best_tpg = best_grid["take_profit_grid"]
    best_slg = best_grid["stop_loss_grid"]
    best_pos_sz = best_grid["position_size"]
    best_pos_coef = best_grid["position_sizing_coef"]
    best_min_ss = best_grid["min_signal_strength"]

    print(f"  [PolyfitSwitch Stage 1] Best: tw={best_tw} vw={best_vw} "
          f"bgp={best_bgp:.4f} vs={best_vs:.1f} ts={best_ts:.0f} "
          f"max_gl={best_max_gl} tpg={best_tpg:.2f} slg={best_slg:.1f} "
          f"pos_sz={best_pos_sz:.2f} pos_coef={best_pos_coef:.0f} "
          f"min_ss={best_min_ss:.2f} ret={best_grid['total_return']:.1%}")

    # ── Stage 2: Switch 参数扫描（Grid 参数固定） ──
    print("  [PolyfitSwitch Stage 2] Scanning switch params (grid fixed)...")

    switch_m1_vals = [0.02, 0.03, 0.04, 0.05]
    switch_m2_vals = [0.005, 0.01, 0.015, 0.02]
    switch_trailing_stop_vals = [0.02, 0.03, 0.05, 0.07, 0.10]
    switch_fast_vals = [5, 10, 20]
    switch_slow_vals = [10, 20, 60]

    all_ma_windows2 = sorted(set(switch_fast_vals + switch_slow_vals))

    switch_combos = []
    for sw_m1, sw_m2, sw_tr, sw_fast, sw_slow in product(
        switch_m1_vals, switch_m2_vals,
        switch_trailing_stop_vals,
        switch_fast_vals, switch_slow_vals,
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_combos.append((sw_m1, sw_m2, sw_tr, sw_fast, sw_slow))

    indicators = compute_polyfit_switch_indicators(
        close, fit_window_days=252,
        ma_windows=all_ma_windows2,
        trend_window_days=best_tw, vol_window_days=best_vw,
    )
    common_idx = indicators.index
    if len(common_idx) == 0:
        return pd.DataFrame()
    cl_aligned = close.loc[common_idx]
    cl_arr = cl_aligned.values
    poly_base_arr = indicators["PolyBasePred"].values
    dev_pct_arr = indicators["PolyDevPct"].values
    dev_trend_arr = indicators["PolyDevTrend"].values
    vol_arr = indicators["RollingVolPct"].values

    n_switch = len(switch_combos)
    stage2_results = []

    if use_mlx:
        # MLX 路径: CPU 逐组合生成信号 + MLX 批量回测
        op_aligned = open_arr[close.index.get_indexer(common_idx)] if open_arr is not None else None

        # 先批量生成所有信号（CPU 循环，~2000 combos 可接受）
        all_entries = []
        all_exits = []
        all_sizes = []
        valid_indices = []
        for idx, (sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in enumerate(switch_combos):
            ma_fast_arr = indicators[f"MA{sw_fast}"].values
            ma_slow_arr = indicators[f"MA{sw_slow}"].values
            entries, exits, sizes, _modes = generate_polyfit_switch_signals(
                cl_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                poly_base_arr, ma_fast_arr, ma_slow_arr,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, max_grid_levels=best_max_gl,
                take_profit_grid=best_tpg, stop_loss_grid=best_slg,
                max_holding_days=45, cooldown_days=1,
                min_signal_strength=best_min_ss,
                position_size=best_pos_sz,
                position_sizing_coef=best_pos_coef,
                switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
            )
            if entries.sum() == 0:
                continue
            all_entries.append(entries)
            all_exits.append(exits)
            all_sizes.append(sizes)
            valid_indices.append(idx)

        if all_entries:
            entries_b = np.array(all_entries)
            exits_b = np.array(all_exits)
            sizes_b = np.array(all_sizes)
            bt = run_backtest_batch(cl_arr, entries_b, exits_b, sizes_b,
                                    n_combos=len(valid_indices), open_=op_aligned)
            for bi, orig_idx in enumerate(valid_indices):
                if int(bt[bi][4]) == 0:
                    continue
                sw_m1, sw_m2, sw_tr, sw_fast, sw_slow = switch_combos[orig_idx]
                stage2_results.append({
                    "total_return": bt[bi][0], "sharpe_ratio": bt[bi][1],
                    "max_drawdown": bt[bi][2], "calmar_ratio": bt[bi][3],
                    "num_trades": int(bt[bi][4]), "win_rate": bt[bi][5],
                    "fit_window_days": 252,
                    "trend_window_days": best_tw,
                    "vol_window_days": best_vw,
                    "base_grid_pct": best_bgp, "volatility_scale": best_vs,
                    "trend_sensitivity": best_ts, "max_grid_levels": best_max_gl,
                    "take_profit_grid": best_tpg, "stop_loss_grid": best_slg,
                    "max_holding_days": 45, "cooldown_days": 1,
                    "min_signal_strength": best_min_ss,
                    "position_size": best_pos_sz,
                    "position_sizing_coef": best_pos_coef,
                    "switch_deviation_m1": sw_m1,
                    "switch_deviation_m2": sw_m2,
                    "switch_trailing_stop": sw_tr,
                    "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
                })
    else:
        # CPU 路径: 逐组合生成信号 + VectorBT 回测
        cl_s = close.loc[common_idx]
        op_s = open_.reindex(common_idx) if open_ is not None else None
        for (sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in switch_combos:
            ma_fast_arr = indicators[f"MA{sw_fast}"].values
            ma_slow_arr = indicators[f"MA{sw_slow}"].values
            entries, exits, sizes, _modes = generate_polyfit_switch_signals(
                cl_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                poly_base_arr, ma_fast_arr, ma_slow_arr,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, max_grid_levels=best_max_gl,
                take_profit_grid=best_tpg, stop_loss_grid=best_slg,
                max_holding_days=45, cooldown_days=1,
                min_signal_strength=best_min_ss,
                position_size=best_pos_sz,
                position_sizing_coef=best_pos_coef,
                switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
            )
            if entries.sum() == 0:
                continue
            m = run_backtest(cl_s, entries, exits, sizes, open_=op_s)
            m["fit_window_days"] = 252
            m["trend_window_days"] = best_tw
            m["vol_window_days"] = best_vw
            m["base_grid_pct"] = best_bgp
            m["volatility_scale"] = best_vs
            m["trend_sensitivity"] = best_ts
            m["max_grid_levels"] = best_max_gl
            m["take_profit_grid"] = best_tpg
            m["stop_loss_grid"] = best_slg
            m["max_holding_days"] = 45
            m["cooldown_days"] = 1
            m["min_signal_strength"] = best_min_ss
            m["position_size"] = best_pos_sz
            m["position_sizing_coef"] = best_pos_coef
            m["switch_deviation_m1"] = sw_m1
            m["switch_deviation_m2"] = sw_m2
            m["switch_trailing_stop"] = sw_tr
            m["switch_fast_ma"] = sw_fast
            m["switch_slow_ma"] = sw_slow
            stage2_results.append(m)

    return pd.DataFrame(stage2_results)
