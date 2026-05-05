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

GPU 加速：通过 CuPy RawKernel 将多组参数组合的信号生成一次性在 GPU 上完成。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import xp, gpu
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
# GPU 批量信号生成 — CuPy RawKernel (v4)
#
# v4 改动：
#   - 移除独立 Switch IDLE 入场，仅在 Grid TP 时通过趋势确认切换
#   - Switch 离场：ATR Chandelier + 量能衰竭 + MA 死叉 + Grid 回退
# ══════════════════════════════════════════════════════════════════

_polyfit_switch_kernel = None

_POLYFIT_SWITCH_KERNEL_CODE = r"""
extern "C" __global__ void polyfit_switch_signals_kernel_v4(
    const double* __restrict__ close,
    const double* __restrict__ dev_pct,
    const double* __restrict__ dev_trend_all,
    const double* __restrict__ vol_pct_all,
    const int* __restrict__ indicator_idx,
    const double* __restrict__ poly_base,
    const double* __restrict__ ma_all,
    const int* __restrict__ fast_ma_idx,
    const int* __restrict__ slow_ma_idx,
    const int* __restrict__ ma5_idx,
    const int* __restrict__ ma10_idx,
    const int* __restrict__ ma20_idx,
    const double* __restrict__ atr_arr,
    const double* __restrict__ vol_ema_arr,
    const double* __restrict__ volume,
    const double* __restrict__ base_grid_pct,
    const double* __restrict__ volatility_scale,
    const double* __restrict__ trend_sensitivity,
    const double* __restrict__ take_profit_grid,
    const double* __restrict__ stop_loss_grid,
    const double* __restrict__ trend_entry_dp,
    const double* __restrict__ trend_confirm_dp_slope,
    const double* __restrict__ trend_confirm_vol,
    const double* __restrict__ trend_atr_mult,
    const double* __restrict__ trend_vol_climax,
    const int* __restrict__ trend_decline_days,
    const double* __restrict__ switch_trailing_stop,
    const int* __restrict__ max_grid_levels,
    const int* __restrict__ cooldown_days,
    const double* __restrict__ min_signal_strength,
    const double* __restrict__ position_size,
    const double* __restrict__ position_sizing_coef,
    bool* __restrict__ entries,
    bool* __restrict__ exits,
    double* __restrict__ sizes,
    int n_bars,
    int n_combos,
    int n_padded,
    int max_holding_days
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    int iset = indicator_idx[combo_idx];

    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];
    double tedp = trend_entry_dp[combo_idx];
    double tcds = trend_confirm_dp_slope[combo_idx];
    double tcv = trend_confirm_vol[combo_idx];
    double tam = trend_atr_mult[combo_idx];
    double tvc = trend_vol_climax[combo_idx];
    int tddecl = trend_decline_days[combo_idx];
    double sw_ts = switch_trailing_stop[combo_idx];
    int fi = fast_ma_idx[combo_idx];
    int si = slow_ma_idx[combo_idx];
    int m5i = ma5_idx[combo_idx];
    int m10i = ma10_idx[combo_idx];
    int m20i = ma20_idx[combo_idx];
    int mgl = max_grid_levels[combo_idx];
    int cd = cooldown_days[combo_idx];
    double mss = min_signal_strength[combo_idx];
    double ps = position_size[combo_idx];
    double psc = position_sizing_coef[combo_idx];

    // v4 状态机
    int position_mode = 0;       // 0=idle, 1=grid, 2=switch
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    double entry_close_price = 0.0 / 0.0;
    int cooldown = 0;
    double switch_peak = 0.0;
    double prev_ma5 = 0.0, prev_ma10 = 0.0;
    double prev_dp_k = 0.0 / 0.0;  // NaN
    int decline_count_k = 0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        double dp = dev_pct[i];
        double dt = dev_trend_all[iset * n_bars + i];
        double vp = vol_pct_all[iset * n_bars + i];
        double pb = poly_base[i];
        double fma = ma_all[fi * n_bars + i];
        double sma = ma_all[si * n_bars + i];
        double m5 = ma_all[m5i * n_bars + i];
        double m10 = ma_all[m10i * n_bars + i];
        double m20 = ma_all[m20i * n_bars + i];

        if (isnan(cl) || isnan(dp) || isnan(dt) || isnan(vp)
            || isnan(pb) || isnan(fma) || isnan(sma)
            || isnan(m5) || isnan(m10) || isnan(m20)
            || pb <= 0.0 || cl <= 0.0) {
            prev_ma5 = m5; prev_ma10 = m10;
            continue;
        }

        if (cooldown > 0) cooldown--;

        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        // ── IDLE: Grid 入场 ──
        if (position_mode == 0) {
            entry_bar = -1; entry_level = 1; entry_grid_step = -1.0;
            entry_close_price = 0.0 / 0.0;

            if (cooldown <= 0) {
                double sig = fabs(dp) / fmax(dgs, 1e-9);
                int el = (int)floor(sig);
                el = el < 1 ? 1 : (el > mgl ? mgl : el);
                double eth = -(double)el * dgs;
                if (dp <= eth && sig >= mss) {
                    double sz = fabs(dp) * (1.0 + fmax(vp, 0.0)) * psc;
                    sz = sz > ps ? ps : (sz < 0.0 ? 0.0 : sz);
                    if (sz > 0.0) {
                        entries[i * n_padded + combo_idx] = true;
                        sizes[i * n_padded + combo_idx] = sz;
                        position_mode = 1;
                        entry_bar = i; entry_level = el; entry_grid_step = dgs;
                        entry_close_price = cl;
                    }
                }
            }
        }

        // ── Grid 持仓 ──
        else if (position_mode == 1) {
            int hd = i - entry_bar;
            bool hl = hd >= max_holding_days;
            double rs = fmax(dgs, entry_grid_step);
            if (entry_grid_step < 0.0 || isnan(entry_grid_step)) rs = dgs;
            double tp_threshold = entry_level * rs * tpg;
            double sl_threshold = entry_level * rs * slg;
            if (hl || dp <= -sl_threshold) {
                exits[i * n_padded + combo_idx] = true;
                position_mode = 0;
                cooldown = cd;
            } else if (dp >= tedp) {
                // ── 趋势确认 (dp 高于阈值，2 取 1 即触发) ──
                int tscore = 0;
                if (dt > tcds) tscore++;
                if (!isnan(m5) && !isnan(m10) && m5 > m10) tscore++;

                if (tscore >= 1) {
                    position_mode = 2;
                    entry_bar = i; entry_grid_step = fmax(bgp, 1e-9);
                    switch_peak = cl;
                } else if (dp >= tp_threshold) {
                    // dp 已达止盈位但趋势未确认 → 正常 Grid 止盈离场
                    exits[i * n_padded + combo_idx] = true;
                    position_mode = 0;
                    cooldown = cd;
                }
            }
        }

        // ── Switch 持仓 (v4: dp连跌→ATR→量能→死叉→Grid回退) ──
        else if (position_mode == 2) {
            // Layer 0: dp 连续下跌 N 天 → 趋势逆转
            if (tddecl > 0 && decline_count_k >= tddecl) {
                exits[i * n_padded + combo_idx] = true;
                position_mode = 0;
                switch_peak = 0.0;
                cooldown = cd;
                decline_count_k = 0;
            } else {
            // Layer 1: ATR Chandelier trail
            bool trail_exit = false;
            if (atr_arr != NULL && atr_arr[i] > 0.0) {
                trail_exit = cl <= switch_peak - tam * atr_arr[i];
            } else {
                trail_exit = cl <= switch_peak * (1.0 - sw_ts);
            }

            // Layer 2: volume climax
            bool vol_climax = false;
            if (volume != NULL && vol_ema_arr != NULL && i > 0) {
                double rv = volume[i] / fmax(vol_ema_arr[i], 1e-9);
                vol_climax = rv > tvc && cl > close[i-1];
            }

            // Layer 3: MA death cross
            bool death_cross = false;
            if (!isnan(m5) && !isnan(m10) && !isnan(prev_ma5) && !isnan(prev_ma10)) {
                death_cross = m5 < m10 && prev_ma5 >= prev_ma10;
            }

            if (trail_exit || vol_climax || death_cross) {
                exits[i * n_padded + combo_idx] = true;
                position_mode = 0;
                switch_peak = 0.0;
                cooldown = cd;
            } else if (dp < 0.0) {
                // Layer 4: Grid fallback
                double sig = fabs(dp) / fmax(dgs, 1e-9);
                int el = (int)floor(sig);
                el = el < 1 ? 1 : (el > mgl ? mgl : el);
                double eth = -(double)el * dgs;
                if (dp <= eth && sig >= mss) {
                    position_mode = 1;
                    entry_bar = i; entry_level = el; entry_grid_step = dgs;
                    entry_close_price = cl;
                    switch_peak = 0.0;
                } else {
                    exits[i * n_padded + combo_idx] = true;
                    position_mode = 0;
                    switch_peak = 0.0;
                    cooldown = cd;
                }
            } else {
                switch_peak = fmax(switch_peak, cl);
            }
        }

            }  // end else (decline_count < tddecl)

        // dp 连续下跌追踪
        if (!isnan(prev_dp_k) && !isnan(dp)) {
            if (dp < prev_dp_k) decline_count_k++;
            else decline_count_k = 0;
        }
        prev_dp_k = dp;
        prev_ma5 = m5; prev_ma10 = m10;
    }

    if (position_mode != 0) {
        exits[(n_bars - 1) * n_padded + combo_idx] = true;
    }
}
"""


# ══════════════════════════════════════════════════════════════════
# GPU kernel v5 — Grid-priority Switch
# ══════════════════════════════════════════════════════════════════

_polyfit_switch_kernel_v5 = None

_POLYFIT_SWITCH_V5_KERNEL = r"""
extern "C" __global__ void polyfit_switch_grid_priority_kernel(
    const double* __restrict__ close,
    const double* __restrict__ dev_pct,
    const double* __restrict__ dev_trend,
    const bool* __restrict__ grid_entries,
    const bool* __restrict__ grid_exits,
    const double* __restrict__ ma20_arr,
    const double* __restrict__ ma60_arr,
    const double* __restrict__ atr_arr,
    const double* __restrict__ vol_ema_arr,
    const double* __restrict__ volume,
    const double* __restrict__ trend_entry_dp,
    const double* __restrict__ trend_confirm_slope,
    const double* __restrict__ trend_atr_mult,
    const double* __restrict__ trend_vol_climax,
    const int* __restrict__ trend_decline_days,
    bool* __restrict__ sw_entries,
    bool* __restrict__ sw_exits,
    int n_bars,
    int n_combos,
    int n_padded
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_combos) return;

    double tedp = trend_entry_dp[c];
    double tcds = trend_confirm_slope[c];
    double tam = trend_atr_mult[c];
    double tvc = trend_vol_climax[c];
    int tddecl = trend_decline_days[c];

    bool grid_in = false;
    bool sw_in = false;
    double sw_peak = 0.0;
    double prev_dp = 0.0 / 0.0;
    int decline = 0;
    double prev_m20 = 0.0, prev_m60 = 0.0;

    for (int i = 0; i < n_bars; i++) {
        double dp = dev_pct[i]; double dt = dev_trend[i]; double cl = close[i];
        double m20 = ma20_arr[i]; double m60 = ma60_arr[i];

        if (isnan(dp) || isnan(dt) || isnan(cl) || cl <= 0.0) {
            prev_m20 = m20; prev_m60 = m60; continue;
        }

        // Grid exits
        if (grid_in && grid_exits[i]) grid_in = false;

        // Grid entry = force Switch exit
        if (grid_entries[i]) {
            if (sw_in) { sw_exits[i * n_padded + c] = true; sw_in = false; sw_peak = 0.0; }
            grid_in = true;
            prev_m20 = m20; prev_m60 = m60; continue;
        }

        // Only if Grid idle
        if (!grid_in) {
            if (!sw_in) {
                if (dp >= tedp) {
                    bool cs = dt > tcds;
                    bool cm = !isnan(m20) && !isnan(m60) && m20 > m60;
                    if (cs || cm) {
                        sw_entries[i * n_padded + c] = true;
                        sw_in = true; sw_peak = cl; decline = 0;
                    }
                }
            } else {
                bool exit_now = false;
                if (tddecl > 0 && decline >= tddecl) exit_now = true;
                else if (atr_arr != NULL && atr_arr[i] > 0.0) {
                    if (cl <= sw_peak - tam * atr_arr[i]) exit_now = true;
                }
                if (!exit_now && vol_ema_arr != NULL && volume != NULL && i > 0) {
                    double rv = volume[i] / fmax(vol_ema_arr[i], 1e-9);
                    if (rv > tvc && cl > close[i-1]) exit_now = true;
                }
                if (!exit_now && !isnan(m20) && !isnan(m60)
                    && !isnan(prev_m20) && !isnan(prev_m60)
                    && m20 < m60 && prev_m20 >= prev_m60) exit_now = true;

                if (exit_now) {
                    sw_exits[i * n_padded + c] = true;
                    sw_in = false; sw_peak = 0.0;
                } else {
                    sw_peak = fmax(sw_peak, cl);
                }
            }
        }

        // dp decline tracking
        if (!isnan(prev_dp) && !isnan(dp)) {
            if (dp < prev_dp) decline++; else decline = 0;
        }
        prev_dp = dp; prev_m20 = m20; prev_m60 = m60;
    }

    if (sw_in) sw_exits[(n_bars - 1) * n_padded + c] = true;
}
"""

def _get_polyfit_switch_kernel_v5():
    global _polyfit_switch_kernel_v5
    if _polyfit_switch_kernel_v5 is None:
        cp = xp()
        _polyfit_switch_kernel_v5 = cp.RawKernel(
            _POLYFIT_SWITCH_V5_KERNEL, "polyfit_switch_grid_priority_kernel"
        )
    return _polyfit_switch_kernel_v5


def generate_grid_priority_switch_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    grid_entries: np.ndarray,
    grid_exits: np.ndarray,
    ma20_arr: np.ndarray,
    ma60_arr: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    trend_entry_dp_arr: np.ndarray | None = None,
    trend_confirm_slope_arr: np.ndarray | None = None,
    trend_atr_mult_arr: np.ndarray | None = None,
    trend_vol_climax_arr: np.ndarray | None = None,
    trend_decline_days_arr: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU 批量生成 Grid-priority Switch 信号。"""
    cp = xp()
    n_bars = len(close)
    n_combos = len(trend_entry_dp_arr) if trend_entry_dp_arr is not None else 1

    block_size = 256
    grid_size = (n_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    def _pad_f64(a):
        result = cp.zeros(padded, dtype=cp.float64)
        result[:n_combos] = cp.asarray(a, dtype=cp.float64)
        return result

    def _pad_i32(a):
        result = cp.zeros(padded, dtype=cp.int32)
        result[:n_combos] = cp.asarray(a, dtype=cp.int32)
        return result

    # Defaults
    _tedp = trend_entry_dp_arr if trend_entry_dp_arr is not None else np.full(n_combos, 0.01)
    _tcds = trend_confirm_slope_arr if trend_confirm_slope_arr is not None else np.full(n_combos, 0.0003)
    _tam = trend_atr_mult_arr if trend_atr_mult_arr is not None else np.full(n_combos, 2.5)
    _tvc = trend_vol_climax_arr if trend_vol_climax_arr is not None else np.full(n_combos, 3.0)
    _tddecl = trend_decline_days_arr if trend_decline_days_arr is not None else np.full(n_combos, 3, dtype=np.int32)

    # Pre-compute ATR & vol_ema
    atr_arr = np.zeros(n_bars, dtype=np.float64)
    if high is not None and low is not None:
        alpha_a = 2.0 / 15.0; atr_ema = 0.0
        for i in range(1, n_bars):
            if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(close[i-1]): continue
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            atr_ema = tr if i == 1 else alpha_a*tr + (1-alpha_a)*atr_ema
            atr_arr[i] = atr_ema

    vol_ema_arr = np.zeros(n_bars, dtype=np.float64)
    if volume is not None:
        alpha_v = 2.0/21.0; v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n_bars):
            if not np.isnan(volume[i]): v_ema = alpha_v*float(volume[i]) + (1-alpha_v)*v_ema
            vol_ema_arr[i] = v_ema

    # GPU arrays
    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    dt_d = cp.asarray(dev_trend, dtype=cp.float64)
    ge_d = cp.asarray(grid_entries, dtype=cp.bool_)
    gx_d = cp.asarray(grid_exits, dtype=cp.bool_)
    m20_d = cp.asarray(ma20_arr, dtype=cp.float64)
    m60_d = cp.asarray(ma60_arr, dtype=cp.float64)
    atr_d = cp.asarray(atr_arr, dtype=cp.float64)
    vema_d = cp.asarray(vol_ema_arr, dtype=cp.float64)
    vol_d = cp.asarray(volume if volume is not None else np.zeros(n_bars), dtype=cp.float64)

    tedp_d = _pad_f64(_tedp); tcds_d = _pad_f64(_tcds)
    tam_d = _pad_f64(_tam); tvc_d = _pad_f64(_tvc)
    tddecl_d = _pad_i32(_tddecl)

    sw_e_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sw_x_d = cp.zeros(padded * n_bars, dtype=cp.bool_)

    kernel = _get_polyfit_switch_kernel_v5()
    kernel((grid_size,), (block_size,),
           (close_d, dev_d, dt_d, ge_d, gx_d, m20_d, m60_d,
            atr_d, vema_d, vol_d,
            tedp_d, tcds_d, tam_d, tvc_d, tddecl_d,
            sw_e_d, sw_x_d, n_bars, n_combos, padded))

    sw_entries = cp.asnumpy(sw_e_d).reshape(n_bars, padded)[:, :n_combos].copy()
    sw_exits = cp.asnumpy(sw_x_d).reshape(n_bars, padded)[:, :n_combos].copy()
    return sw_entries, sw_exits


def _get_polyfit_switch_kernel():
    """原始 v4 kernel（已弃用，保留兼容）"""
    global _polyfit_switch_kernel
    if _polyfit_switch_kernel is None:
        cp = xp()
        _polyfit_switch_kernel = cp.RawKernel(
            _POLYFIT_SWITCH_KERNEL_CODE, "polyfit_switch_signals_kernel_v4"
        )
    return _polyfit_switch_kernel


def generate_polyfit_switch_signals_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend_all: np.ndarray,
    rolling_vol_pct_all: np.ndarray,
    poly_base: np.ndarray,
    ma_all: np.ndarray,             # [n_ma_windows, n_bars]
    fast_ma_idx: np.ndarray,        # [n_combos] int32
    slow_ma_idx: np.ndarray,        # [n_combos] int32
    ma5_idx: np.ndarray,            # [n_combos] int32
    ma10_idx: np.ndarray,           # [n_combos] int32
    ma20_idx: np.ndarray,           # [n_combos] int32
    base_grid_pcts: np.ndarray,
    volatility_scales: np.ndarray,
    trend_sensitivities: np.ndarray,
    take_profit_grids: np.ndarray,
    stop_loss_grids: np.ndarray,
    switch_deviation_m1_arr: np.ndarray,       # unused in v4, kept for compat
    switch_deviation_m2_arr: np.ndarray,       # unused in v4
    switch_trailing_stop_arr: np.ndarray | None = None,  # fallback trailing stop
    # ── v4 新参数 ──
    trend_entry_dp_arr: np.ndarray | None = None,
    trend_confirm_dp_slope_arr: np.ndarray | None = None,
    trend_confirm_vol_arr: np.ndarray | None = None,
    trend_atr_mult_arr: np.ndarray | None = None,
    trend_vol_climax_arr: np.ndarray | None = None,
    trend_decline_days_arr: np.ndarray | None = None,
    # ── v4 预计算数据 ──
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    # ── 其他 ──
    flat_wait_days_arr: np.ndarray | None = None,  # ignored
    indicator_idx: np.ndarray | None = None,
    max_grid_levels_arr: np.ndarray | None = None,
    max_holding_days: int = 45,
    cooldown_days_arr: np.ndarray | None = None,
    min_signal_strength_arr: np.ndarray | None = None,
    position_size_arr: np.ndarray | None = None,
    position_sizing_coef_arr: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU 批量生成 Polyfit-Switch v4 信号。"""
    cp = xp()
    n_bars = len(close)
    n_combos = len(base_grid_pcts)

    if dev_trend_all.ndim == 1:
        dev_trend_all = dev_trend_all.reshape(1, -1)
    if rolling_vol_pct_all.ndim == 1:
        rolling_vol_pct_all = rolling_vol_pct_all.reshape(1, -1)
    n_indicator_sets = dev_trend_all.shape[0]

    if indicator_idx is None:
        indicator_idx = np.zeros(n_combos, dtype=np.int32)

    block_size = 256
    grid_size = (n_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    def _pad_f64(a):
        result = cp.zeros(padded, dtype=cp.float64)
        result[:n_combos] = cp.asarray(a, dtype=cp.float64)
        if padded > n_combos:
            result[n_combos:] = cp.nan
        return result

    def _pad_i32(a):
        result = cp.zeros(padded, dtype=cp.int32)
        result[:n_combos] = cp.asarray(a, dtype=cp.int32)
        return result

    # 默认值
    _mgl = max_grid_levels_arr if max_grid_levels_arr is not None else np.full(n_combos, 3, dtype=np.int32)
    _cd  = cooldown_days_arr if cooldown_days_arr is not None else np.full(n_combos, 1, dtype=np.int32)
    _mss = min_signal_strength_arr if min_signal_strength_arr is not None else np.full(n_combos, 0.45, dtype=np.float64)
    _str = switch_trailing_stop_arr if switch_trailing_stop_arr is not None else np.full(n_combos, 0.05, dtype=np.float64)
    _ps  = position_size_arr if position_size_arr is not None else np.full(n_combos, 0.5, dtype=np.float64)
    _psc = position_sizing_coef_arr if position_sizing_coef_arr is not None else np.full(n_combos, 30.0, dtype=np.float64)
    _tedp = trend_entry_dp_arr if trend_entry_dp_arr is not None else np.full(n_combos, 0.0, dtype=np.float64)
    _tcds = trend_confirm_dp_slope_arr if trend_confirm_dp_slope_arr is not None else np.full(n_combos, 0.0003, dtype=np.float64)
    _tcv = trend_confirm_vol_arr if trend_confirm_vol_arr is not None else np.full(n_combos, 1.2, dtype=np.float64)
    _tam = trend_atr_mult_arr if trend_atr_mult_arr is not None else np.full(n_combos, 2.5, dtype=np.float64)
    _tvc = trend_vol_climax_arr if trend_vol_climax_arr is not None else np.full(n_combos, 3.0, dtype=np.float64)
    _tddecl = trend_decline_days_arr if trend_decline_days_arr is not None else np.full(n_combos, 0, dtype=np.int32)

    # Pre-compute ATR (percentage)
    atr_arr = np.zeros(n_bars, dtype=np.float64)
    if high is not None and low is not None:
        alpha_a = 2.0 / 15.0  # ~14-day EMA
        atr_ema = 0.0
        for i in range(1, n_bars):
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

    # Pre-compute vol EMA
    vol_ema_arr = np.zeros(n_bars, dtype=np.float64)
    if volume is not None:
        alpha_v = 2.0 / 21.0
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n_bars):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    # Pad per-combo arrays
    bgp_d = _pad_f64(base_grid_pcts); vs_d = _pad_f64(volatility_scales)
    ts_d = _pad_f64(trend_sensitivities); tpg_d = _pad_f64(take_profit_grids)
    slg_d = _pad_f64(stop_loss_grids)
    m1_d = _pad_f64(switch_deviation_m1_arr); m2_d = _pad_f64(switch_deviation_m2_arr)
    tedp_d = _pad_f64(_tedp)
    tcds_d = _pad_f64(_tcds); tcv_d = _pad_f64(_tcv)
    tam_d = _pad_f64(_tam); tvc_d = _pad_f64(_tvc)
    tddecl_d = _pad_i32(_tddecl)
    str_d = _pad_f64(_str)
    fi_d = _pad_i32(fast_ma_idx); si_d = _pad_i32(slow_ma_idx)
    m5i_d = _pad_i32(ma5_idx); m10i_d = _pad_i32(ma10_idx); m20i_d = _pad_i32(ma20_idx)
    mgl_d = _pad_i32(_mgl); cd_d = _pad_i32(_cd)
    mss_d = _pad_f64(_mss); ps_d = _pad_f64(_ps); psc_d = _pad_f64(_psc)
    iset_d = _pad_i32(indicator_idx)

    # GPU arrays
    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    trend_all_d = cp.asarray(dev_trend_all.ravel(), dtype=cp.float64)
    vol_all_d = cp.asarray(rolling_vol_pct_all.ravel(), dtype=cp.float64)
    pb_d = cp.asarray(poly_base, dtype=cp.float64)
    ma_d = cp.asarray(ma_all.ravel(), dtype=cp.float64)
    atr_d = cp.asarray(atr_arr, dtype=cp.float64)
    vema_d = cp.asarray(vol_ema_arr, dtype=cp.float64)
    vol_d = cp.asarray(volume if volume is not None else np.zeros(n_bars), dtype=cp.float64)

    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    kernel = _get_polyfit_switch_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, dev_d, trend_all_d, vol_all_d, iset_d,
            pb_d, ma_d, fi_d, si_d, m5i_d, m10i_d, m20i_d,
            atr_d, vema_d, vol_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d,
            tedp_d, tcds_d, tcv_d, tam_d, tvc_d, tddecl_d, str_d,
            mgl_d, cd_d, mss_d, ps_d, psc_d,
            entries_d, exits_d, sizes_d,
            n_bars, n_combos, padded, max_holding_days,
        ),
    )

    entries = cp.asnumpy(entries_d).reshape(n_bars, padded)[:, :n_combos].copy()
    exits = cp.asnumpy(exits_d).reshape(n_bars, padded)[:, :n_combos].copy()
    sizes = cp.asnumpy(sizes_d).reshape(n_bars, padded)[:, :n_combos].copy()
    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# 两阶段参数扫描（GPU）
# ══════════════════════════════════════════════════════════════════

def scan_polyfit_switch_two_stage(close: pd.Series,
                                   open_: pd.Series | None = None) -> pd.DataFrame:
    """两阶段扫描：Stage 1 复用 Polyfit-Grid 扫描，Stage 2 扫描 Switch 参数。

    Stage 1: 调用 scan_polyfit_grid() 获取纯 Grid 最优参数（与 Polyfit-Grid 策略一致）
    Stage 2: 固定最优 Grid 参数，GPU 批量扫描 Switch 参数（m1/m2/trailing_stop/MA）

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    use_gpu = gpu()["cupy_available"]
    open_arr = open_.values if open_ is not None else None

    # ── Stage 1: 复用 Polyfit-Grid 扫描（纯Grid，确保与Polyfit-Grid策略一致） ──
    print("  [PolyfitSwitch Stage 1] Reusing Polyfit-Grid scan…")
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
    print("  [PolyfitSwitch Stage 2] Scanning switch params (grid fixed)…")

    switch_m1_vals = [0.02, 0.03, 0.04, 0.05]
    switch_m2_vals = [0.005, 0.01, 0.015, 0.02]
    switch_trailing_stop_vals = [0.02, 0.03, 0.05, 0.07, 0.10]
    switch_fast_vals = [5, 10, 20]
    switch_slow_vals = [10, 20, 60]

    all_ma_windows2 = sorted(set(switch_fast_vals + switch_slow_vals))
    ma_to_idx2 = {w: i for i, w in enumerate(all_ma_windows2)}

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
    cl_aligned = close.loc[common_idx]
    cl_arr = cl_aligned.values
    poly_base_arr = indicators["PolyBasePred"].values
    dev_pct_arr = indicators["PolyDevPct"].values
    dev_trend_arr = indicators["PolyDevTrend"].values
    vol_arr = indicators["RollingVolPct"].values
    ma_all2 = np.array([indicators[f"MA{mw}"].values for mw in all_ma_windows2])

    n_switch = len(switch_combos)
    bgp_a = np.full(n_switch, best_bgp)
    vs_a = np.full(n_switch, best_vs)
    ts_a = np.full(n_switch, best_ts)
    tpg_a = np.full(n_switch, best_tpg)
    slg_a = np.full(n_switch, best_slg)
    m1_a = np.array([c[0] for c in switch_combos])
    m2_a = np.array([c[1] for c in switch_combos])
    str_a = np.array([c[2] for c in switch_combos])
    fi_a = np.array([ma_to_idx2[c[3]] for c in switch_combos], dtype=np.int32)
    si_a = np.array([ma_to_idx2[c[4]] for c in switch_combos], dtype=np.int32)

    stage2_results = []

    # Stage 2 GPU 批量：Grid 参数固定，Switch 参数变化
    mgl_a = np.full(n_switch, best_max_gl, dtype=np.int32)
    cd_a = np.full(n_switch, 1, dtype=np.int32)
    mss_a = np.full(n_switch, best_min_ss)
    ps_a = np.full(n_switch, best_pos_sz)
    psc_a = np.full(n_switch, best_pos_coef)

    if use_gpu:
        entries_b, exits_b, sizes_b = generate_polyfit_switch_signals_batch(
            cl_arr, dev_pct_arr, dev_trend_arr, vol_arr, poly_base_arr,
            ma_all2, fi_a, si_a,
            bgp_a, vs_a, ts_a, tpg_a, slg_a,
            m1_a, m2_a, str_a,
            max_grid_levels_arr=mgl_a,
            max_holding_days=45,
            cooldown_days_arr=cd_a,
            min_signal_strength_arr=mss_a,
            position_size_arr=ps_a,
            position_sizing_coef_arr=psc_a,
        )
        op_aligned = open_arr[close.index.get_indexer(common_idx)] if open_arr is not None else None
        bt = run_backtest_batch(cl_arr, entries_b, exits_b, sizes_b, n_combos=n_switch, transposed=True,
                                open_=op_aligned)
        for idx, (sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in enumerate(switch_combos):
            if int(bt[idx][4]) == 0:
                continue
            stage2_results.append({
                "total_return": bt[idx][0], "sharpe_ratio": bt[idx][1],
                "max_drawdown": bt[idx][2], "calmar_ratio": bt[idx][3],
                "num_trades": int(bt[idx][4]), "win_rate": bt[idx][5],
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
