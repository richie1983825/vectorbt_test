#!/usr/bin/env python3
"""
交易行为预测工具 — 基于 V7 策略，预测下一交易日操作并解释原因。

用法:
    uv run python workflows/predict.py SYMBOL STRATEGY TARGET

参数:
    SYMBOL    : 股票代码，如 sh512890、sh510880
    STRATEGY  : polyfit_switch (当前仅支持此策略)
    TARGET    : return 或 balanced

示例:
    uv run python workflows/predict.py sh512890 polyfit_switch return
    uv run python workflows/predict.py sh512890 polyfit_switch balanced

输出:
    - 控制台：当前持仓状态、下一日操作、详细解释
    - HTML 报告：reports/predict_<SYMBOL>_<TARGET>.html
"""

import os, sys, json, argparse
from datetime import date
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import (
    generate_grid_priority_switch_signals_v7,
    _compute_entry_ohlcv_features,
    _check_entry_filters,
)

REPORTS_DIR = "reports"

# V6/V7 最优固定参数
V6_FIXED = dict(
    trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_decline_days=1, trend_vol_climax=2.5,
    enable_ohlcv_filter=True, enable_early_exit=True,
)
V7_BEST_TOP = dict(
    enable_top_avoidance=True, top_ret_5d=0.05,
    top_price_pos=0.80, top_amplitude=0.02, top_block_days=3,
)

# Grid 默认参数（fallback）
DEFAULT_GRID = dict(
    trend_window_days=10, vol_window_days=10, base_grid_pct=0.01,
    volatility_scale=0.0, trend_sensitivity=4, max_grid_levels=3,
    take_profit_grid=0.8, stop_loss_grid=1.6, max_holding_days=45,
    cooldown_days=1, min_signal_strength=0.3,
    position_size=0.99, position_sizing_coef=60,
)


def _parse_symbol(sym: str) -> tuple:
    """解析股票代码，返回 (parquet_path, display_name)。

    sh512890 → data/1d/512890.SH_hfq.parquet, 512890.SH
    """
    sym = sym.lower().strip()
    if sym.startswith("sh"):
        code = sym[2:]
        market = "SH"
    elif sym.startswith("sz"):
        code = sym[2:]
        market = "SZ"
    else:
        code, market = sym.split(".")
    path = f"data/1d/{code}.{market}_hfq.parquet"
    return path, f"{code}.{market}"


def _load_best_grid_params(selector: str, data_end_date) -> dict:
    """加载最优 Grid 参数。优先从 WF 缓存读取最新窗口的参数。"""
    cache_path = os.path.join(REPORTS_DIR, "grid_wf_cache.csv")
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        cache = cache[cache["selector"] == selector]
        if not cache.empty:
            # 选择 test_start_date 最接近数据末端的窗口
            cache["test_start_dt"] = pd.to_datetime(cache["test_start_date"])
            latest = cache.nlargest(1, "test_start_dt").iloc[0]
            params = {}
            for k in ["trend_window_days", "vol_window_days", "base_grid_pct",
                       "volatility_scale", "trend_sensitivity", "max_grid_levels",
                       "take_profit_grid", "stop_loss_grid", "max_holding_days",
                       "cooldown_days", "min_signal_strength", "position_size",
                       "position_sizing_coef"]:
                if k in latest.index:
                    params[k] = latest[k]
            return params
    return DEFAULT_GRID.copy()


def _simulate_position(close, open_, high, low, volume, grid_params):
    """运行 V7 策略，返回最后一天的状态快照。"""
    n = len(close)
    grid_params = grid_params.copy()
    tw = int(grid_params.pop("trend_window_days", 10))
    vw = int(grid_params.pop("vol_window_days", 10))

    ind = compute_polyfit_switch_indicators(
        close, fit_window_days=252, ma_windows=[20, 60],
        trend_window_days=tw, vol_window_days=vw,
    )
    com_idx = ind.index
    cl_arr = close.loc[com_idx].values
    op_arr = open_.reindex(com_idx).values
    hi_arr = high.reindex(com_idx).values
    lo_arr = low.reindex(com_idx).values
    vol_arr = volume.reindex(com_idx).values

    e_grid, x_grid, s_grid = generate_grid_signals(
        cl_arr, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
        ind["RollingVolPct"].values, ind["PolyBasePred"].values,
        base_grid_pct=grid_params.get("base_grid_pct", 0.01),
        volatility_scale=grid_params.get("volatility_scale", 0.0),
        trend_sensitivity=grid_params.get("trend_sensitivity", 4),
        max_grid_levels=int(grid_params.get("max_grid_levels", 3)),
        take_profit_grid=grid_params.get("take_profit_grid", 0.8),
        stop_loss_grid=grid_params.get("stop_loss_grid", 1.6),
        max_holding_days=int(grid_params.get("max_holding_days", 45)),
        cooldown_days=int(grid_params.get("cooldown_days", 1)),
        min_signal_strength=grid_params.get("min_signal_strength", 0.3),
        position_size=grid_params.get("position_size", 0.99),
        position_sizing_coef=grid_params.get("position_sizing_coef", 60),
    )

    result = generate_grid_priority_switch_signals_v7(
        cl_arr, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
        ind["RollingVolPct"].values, ind["PolyBasePred"].values,
        e_grid, x_grid, ind["MA20"].values, ind["MA60"].values,
        trend_entry_dp=V6_FIXED["trend_entry_dp"],
        trend_confirm_dp_slope=V6_FIXED["trend_confirm_dp_slope"],
        trend_atr_mult=V6_FIXED["trend_atr_mult"], trend_atr_window=14,
        trend_vol_climax=V6_FIXED["trend_vol_climax"],
        trend_decline_days=V6_FIXED["trend_decline_days"],
        enable_ohlcv_filter=V6_FIXED["enable_ohlcv_filter"],
        enable_early_exit=V6_FIXED["enable_early_exit"],
        enable_top_avoidance=V7_BEST_TOP["enable_top_avoidance"],
        top_ret_5d=V7_BEST_TOP["top_ret_5d"],
        top_price_pos=V7_BEST_TOP["top_price_pos"],
        top_amplitude=V7_BEST_TOP["top_amplitude"],
        top_block_days=V7_BEST_TOP["top_block_days"],
        high=hi_arr, low=lo_arr, open_=op_arr, volume=vol_arr,
        return_filter_stats=True,
    )

    sw_entries = result["sw_entries"]
    sw_exits = result["sw_exits"]
    sw_sizes = result["sw_sizes"]
    filter_stats = result["filter_stats"]

    # 模拟仓位（bar i 信号 → bar i+1 执行）
    # positions[i] = 仓位状态 bar i 期间
    # signals at bar i → 影响 bar i+1 的状态
    n_bars = len(cl_arr)
    g_in = False; s_in = False
    g_entry_bar = -1; s_entry_bar = -1
    g_entry_price = np.nan; s_peak = 0.0
    last_g_entry = -1; last_g_exit = -1
    last_s_entry = -1; last_s_exit = -1
    decline_count = 0
    prev_dp = np.nan
    top_block_until = -1
    top_signal_today = False

    positions = np.zeros(n_bars, dtype=np.int8)

    for i in range(n_bars):
        dp_i = ind["PolyDevPct"].values[i]
        cl_i = cl_arr[i]

        # ── 记录 bar i 期间的仓位（信号未执行）──
        positions[i] = 1 if g_in else (2 if s_in else 0)

        # ── V7 顶部检测 ──
        top_signal_today = False
        if V7_BEST_TOP["enable_top_avoidance"] and i >= 20:
            h20 = np.max(hi_arr[max(0,i-19):i+1])
            l20 = np.min(lo_arr[max(0,i-19):i+1])
            ret_5d = cl_arr[i] / cl_arr[max(0,i-5)] - 1.0 if i >= 5 else 0.0
            price_pos = (cl_arr[i] - l20) / (h20 - l20) if h20 > l20 else 0.5
            is_bearish = cl_arr[i] < op_arr[i]
            amplitude = (hi_arr[i] - lo_arr[i]) / cl_arr[i]
            if (ret_5d > V7_BEST_TOP["top_ret_5d"]
                and price_pos > V7_BEST_TOP["top_price_pos"]
                and is_bearish and amplitude > V7_BEST_TOP["top_amplitude"]):
                top_signal_today = True
                top_block_until = i + V7_BEST_TOP["top_block_days"]
                if s_in:
                    # 顶部信号 → bar i+1 执行 Switch 离场
                    pass  # 信号已在 sw_exits 中

        # ── 执行 bar i 的信号（这些信号影响 bar i+1 的仓位）──
        # Grid exit at bar i → Grid 在 bar i+1 空闲
        if g_in and x_grid[i]:
            g_in = False
            last_g_exit = i
        # Switch exit at bar i → Switch 在 bar i+1 空闲
        if s_in and sw_exits[i]:
            s_in = False
            last_s_exit = i

        # Grid entry at bar i → Grid 在 bar i+1 持仓（强制 Switch 离场）
        if e_grid[i] and i > top_block_until:
            if s_in:
                s_in = False; last_s_exit = i
            g_in = True
            g_entry_bar = i
            g_entry_price = cl_i
            last_g_entry = i
            decline_count = 0
            prev_dp = dp_i
            continue

        # Switch entry at bar i → Switch 在 bar i+1 持仓
        if not g_in:
            if not s_in and sw_entries[i]:
                s_in = True
                s_entry_bar = i
                s_peak = cl_i
                last_s_entry = i
                decline_count = 0
            elif s_in:
                s_peak = max(s_peak, cl_i)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp_i):
            if dp_i < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp_i

    last_i = n_bars - 1
    return {
        "index": com_idx,
        "close": cl_arr, "open": op_arr, "high": hi_arr, "low": lo_arr, "volume": vol_arr,
        "indicators": ind,
        "e_grid": e_grid, "x_grid": x_grid, "s_grid": s_grid,
        "sw_entries": sw_entries, "sw_exits": sw_exits, "sw_sizes": sw_sizes,
        "filter_stats": filter_stats,
        "last_i": last_i,
        "grid_in": g_in, "sw_in": s_in,
        "g_entry_bar": g_entry_bar, "s_entry_bar": s_entry_bar,
        "g_entry_price": g_entry_price,
        "last_g_entry": last_g_entry, "last_g_exit": last_g_exit,
        "last_s_entry": last_s_entry, "last_s_exit": last_s_exit,
        "decline_count": decline_count,
        "top_block_until": top_block_until,
        "positions": positions,
    }


def _explain_signal(snap, grid_params):
    """解释最后一根 bar 的信号。"""
    i = snap["last_i"]
    idx = snap["index"]
    ind = snap["indicators"]

    dp = ind["PolyDevPct"].values[i]
    dt = ind["PolyDevTrend"].values[i]
    vp = ind["RollingVolPct"].values[i]
    pb = ind["PolyBasePred"].values[i]
    m20 = ind["MA20"].values[i]
    m60 = ind["MA60"].values[i]
    cl = snap["close"][i]
    op = snap["open"][i]
    hi = snap["high"][i]
    lo = snap["low"][i]
    vol = snap["volume"][i]

    lines = []
    lines.append(f"最新 bar: {idx[i].date()}  (共 {i+1} 根有效 bar)")

    # ── 价格概览 ──
    ret_1d = (cl / snap["close"][i-1] - 1) if i > 0 else 0
    ret_5d_val = cl / snap["close"][max(0,i-5)] - 1.0 if i >= 5 else 0
    h20 = np.max(snap["high"][max(0,i-19):i+1])
    l20 = np.min(snap["low"][max(0,i-19):i+1])
    price_pos = (cl - l20) / (h20 - l20) if h20 > l20 else 0.5
    amplitude = (hi - lo) / cl
    is_bearish = cl < op
    candle = "阴线" if is_bearish else "阳线"

    lines.append(f"\n═══ 当日行情 ═══")
    lines.append(f"  O={op:.4f}  H={hi:.4f}  L={lo:.4f}  C={cl:.4f}  V={vol:.0f}")
    lines.append(f"  涨跌: {ret_1d:+.2%}  |  振幅: {amplitude:.2%}  |  {candle}")
    lines.append(f"  5日涨幅: {ret_5d_val:+.2%}  |  20日价位: {price_pos:.1%}")

    # ── Polyfit 指标 ──
    lines.append(f"\n═══ Polyfit 指标 ═══")
    lines.append(f"  基线预测价: {pb:.4f}  |  偏离(dp): {dp:+.2%}  |  dp趋势(dt): {dt:+.4f}")
    lines.append(f"  波动率(vp): {vp:.2%}  |  MA20: {m20:.4f}  |  MA60: {m60:.4f}")
    lines.append(f"  MA20 > MA60: {m20 > m60}  |  dp连续变化: {snap['decline_count']}天")

    # ── 仓位状态 ──
    # positions[last_i] = bar last_i 期间的仓位（信号在 bar last_i 生成，bar last_i+1 执行）
    current_pos = snap["positions"][i]  # 0=idle, 1=Grid, 2=Switch
    pos_mode = "空仓" if current_pos == 0 else ("Grid持仓" if current_pos == 1 else "Switch持仓")
    holding_days = 0
    if current_pos == 1:
        holding_days = i - snap["g_entry_bar"]
    elif current_pos == 2:
        holding_days = i - snap["s_entry_bar"]

    lines.append(f"\n═══ 当前仓位 ═══")
    lines.append(f"  状态: {pos_mode}  (持仓 {holding_days} 天)")
    next_date = "下一交易日"
    lines.append(f"  ({idx[i].date()} 收盘信号 → {next_date} 开盘执行)")
    if current_pos == 1:
        entry_px = snap["g_entry_price"]
        px_chg = (cl - entry_px) / entry_px
        lines.append(f"  入场价: {entry_px:.4f}  |  浮动盈亏: {px_chg:+.2%}")

    # ── Grid 信号检查 ──
    bgp = grid_params.get("base_grid_pct", 0.01)
    vs = grid_params.get("volatility_scale", 0.0)
    ts = grid_params.get("trend_sensitivity", 4)
    mgl = int(grid_params.get("max_grid_levels", 3))
    tpg = grid_params.get("take_profit_grid", 0.8)
    slg = grid_params.get("stop_loss_grid", 1.6)

    vol_mult = 1.0 + vs * max(vp, 0.0)
    dgs = bgp * (1.0 + ts * abs(dt)) * vol_mult
    dgs = max(dgs, bgp * 0.3)
    signal_strength = abs(dp) / max(dgs, 1e-9)
    entry_lvl = int(np.clip(np.floor(signal_strength), 1, mgl))
    entry_threshold = -entry_lvl * dgs
    tp_threshold = entry_lvl * dgs * tpg
    sl_threshold = entry_lvl * dgs * slg

    lines.append(f"\n═══ Grid 入场条件 ═══")
    lines.append(f"  动态步长: {dgs:.4%}  |  信号强度: {signal_strength:.1f}")
    lines.append(f"  入场阈值: dp <= {entry_threshold:+.2%}  |  当前 dp = {dp:+.2%}  →  {'触发' if dp <= entry_threshold and signal_strength >= grid_params.get('min_signal_strength',0.45) else '不触发'}")
    lines.append(f"  止盈: dp >= {tp_threshold:+.2%}  |  止损: dp <= {-sl_threshold:+.2%}")

    # ── Switch 入场条件 ──
    tedp = V6_FIXED["trend_entry_dp"]
    tcds = V6_FIXED["trend_confirm_dp_slope"]
    cond_dp = dp >= tedp
    cond_slope = dt > tcds
    cond_ma = bool(not np.isnan(m20) and not np.isnan(m60) and m20 > m60)
    switch_entry_ok = cond_dp and (cond_slope or cond_ma)

    lines.append(f"\n═══ Switch 入场条件 ═══")
    lines.append(f"  dp >= {tedp:+.2%} (入场阈值): {cond_dp}  (dp={dp:+.4f})")
    lines.append(f"  dt > {tcds:+.4f} (斜率确认): {cond_slope}  (dt={dt:+.4f})")
    lines.append(f"  MA20 > MA60 (均线确认): {cond_ma}")
    lines.append(f"  → 综合: {'触发' if switch_entry_ok else '不触发'}")

    # ── OHLCV 入场过滤 ──
    if V6_FIXED["enable_ohlcv_filter"] and i >= 10:
        vol_ema = snap["volume"][:i+1]
        vol_ema = pd.Series(vol_ema).ewm(span=20, min_periods=1).mean().values
        feat = _compute_entry_ohlcv_features(
            snap["close"], snap["open"], snap["high"], snap["low"],
            snap["volume"], vol_ema, i)
        passed, reason = _check_entry_filters(feat, tedp, tcds)
        lines.append(f"\n═══ OHLCV 入场过滤 ═══")
        lines.append(f"  当日涨幅: {feat['entry_day_ret']:+.2%}  |  连涨: {feat['consecutive_up']}天")
        lines.append(f"  上影占比: {feat['upper_wick_pct']:.0%}  |  量比: {feat['rel_vol']:.1f}x")
        lines.append(f"  价位: {feat['price_position']:.0%}  |  {'阳线' if feat['is_green'] else '阴线'}")
        lines.append(f"  → 过滤结果: {'通过' if passed else reason}")
    else:
        lines.append(f"\n═══ OHLCV 入场过滤 ═══")
        lines.append(f"  (bar 数不足或已禁用)")

    # ── 顶部规避 ──
    lines.append(f"\n═══ 顶部规避(V7) ═══")
    top_triggered = False
    if i >= 20:
        top_cond_ret = ret_5d_val > V7_BEST_TOP["top_ret_5d"]
        top_cond_pos = price_pos > V7_BEST_TOP["top_price_pos"]
        top_cond_bear = is_bearish
        top_cond_amp = amplitude > V7_BEST_TOP["top_amplitude"]
        top_triggered = top_cond_ret and top_cond_pos and top_cond_bear and top_cond_amp
        lines.append(f"  5日涨幅 > {V7_BEST_TOP['top_ret_5d']:.0%}: {top_cond_ret}  ({ret_5d_val:+.1%})")
        lines.append(f"  价位 > {V7_BEST_TOP['top_price_pos']:.0%}: {top_cond_pos}  ({price_pos:.1%})")
        lines.append(f"  收阴: {top_cond_bear}")
        lines.append(f"  振幅 > {V7_BEST_TOP['top_amplitude']:.0%}: {top_cond_amp}  ({amplitude:.1%})")
        lines.append(f"  → 顶部信号: {'触发！禁止入场' if top_triggered else '未触发'}")
    else:
        lines.append(f"  (bar 数不足，跳过)")

    lines.append(f"  顶部封锁至 bar: {snap['top_block_until']}  (当前 bar: {i})  → {'封锁中' if i <= snap['top_block_until'] else '已过期'}")

    # ── 离场条件 ──
    if current_pos != 0:
        lines.append(f"\n═══ 离场条件 ═══")
        tdd = V6_FIXED["trend_decline_days"]
        dc_triggered = snap["decline_count"] >= tdd
        lines.append(f"  dp连跌 {snap['decline_count']}天 >= {tdd}天: {'触发离场' if dc_triggered else '未触发'}")

        ee_triggered = V6_FIXED["enable_early_exit"] and dp < -0.005 and snap["decline_count"] >= 1
        lines.append(f"  预警离场(dp<{-0.005}且连跌≥1天): {'触发' if ee_triggered else '未触发'}")

        mhd = grid_params.get("max_holding_days", 45)
        lines.append(f"  持仓天数({holding_days}d) >= 最大({int(mhd)}d): {holding_days >= int(mhd)}")

    # ── 最终决策 ──
    last_signal_grid_entry = snap["e_grid"][i]
    last_signal_grid_exit = snap["x_grid"][i]
    last_signal_sw_entry = snap["sw_entries"][i]
    last_signal_sw_exit = snap["sw_exits"][i]

    # ── 检测 Grid 强制平仓 ──
    # generate_grid_signals 会在末尾执行 `if in_position: exits[-1] = True`
    # 导致数据最后一根 bar 上出现虚假的离场信号。
    # 新数据加入后，该信号会漂移到新末尾，产生"未来数据"的错觉。
    # 这里通过重新检查离场条件来判断离场是否为强制平仓。
    grid_exit_forced = False
    if last_signal_grid_exit and current_pos == 1:
        entry_bar_idx = snap["g_entry_bar"]
        if entry_bar_idx >= 0:
            bgp = grid_params.get("base_grid_pct", 0.01)
            vs = grid_params.get("volatility_scale", 0.0)
            ts = grid_params.get("trend_sensitivity", 4)
            tpg = grid_params.get("take_profit_grid", 0.8)
            slg = grid_params.get("stop_loss_grid", 1.6)
            mhd = int(grid_params.get("max_holding_days", 45))
            # 重建入场时的动态步长和层级
            dt_entry = ind["PolyDevTrend"].values[entry_bar_idx]
            vp_entry = ind["RollingVolPct"].values[entry_bar_idx]
            vol_mult_e = 1.0 + vs * max(vp_entry, 0.0)
            dgs_entry = max(bgp * (1.0 + ts * abs(dt_entry)) * vol_mult_e, bgp * 0.3)
            sig_entry = abs(ind["PolyDevPct"].values[entry_bar_idx]) / max(dgs_entry, 1e-9)
            entry_lvl = int(np.clip(np.floor(sig_entry), 1, int(grid_params.get("max_grid_levels", 3))))
            # 当前 bar 的离场条件
            vol_mult_now = 1.0 + vs * max(vp, 0.0)
            dgs_now = max(bgp * (1.0 + ts * abs(dt)) * vol_mult_now, bgp * 0.3)
            ref_step = max(dgs_now, dgs_entry)
            tp_th = entry_lvl * ref_step * tpg
            sl_th = entry_lvl * ref_step * slg
            holding_d = i - entry_bar_idx
            natural_exit = (holding_d >= mhd or dp >= tp_th or dp <= -sl_th)
            if not natural_exit:
                grid_exit_forced = True
                last_signal_grid_exit = False

    lines.append(f"\n═══ 信号总结 ═══")
    lines.append(f"{'═' * 50}")

    if switch_entry_ok and not last_signal_sw_entry and V6_FIXED["enable_ohlcv_filter"] and i >= 10:
        lines.append(f"  ⚠ Switch 入场条件满足，但 OHLCV 过滤拒绝 (原因: {reason})")

    g_exit_valid = last_signal_grid_exit and current_pos == 1
    s_exit_valid = last_signal_sw_exit and current_pos == 2

    lines.append(f"  Grid  入场信号: {'★ BUY' if last_signal_grid_entry else '—'}")
    grid_exit_str = '★ SELL' if last_signal_grid_exit else '—'
    if grid_exit_forced:
        grid_exit_str += ' (强制平仓-已忽略)'
    elif snap["x_grid"][i] and not g_exit_valid:
        grid_exit_str += ' (无效-当前无Grid持仓)'
    lines.append(f"  Grid  离场信号: {grid_exit_str}")
    lines.append(f"  Switch 入场信号: {'★ BUY' if last_signal_sw_entry else '—'}")
    lines.append(f"  Switch 离场信号: {'★ SELL' if last_signal_sw_exit else '—'}{' (无效-当前无Switch持仓)' if last_signal_sw_exit and not s_exit_valid else ''}")

    # 下一日操作
    next_action = "HOLD (无操作)"
    action_detail = ""
    top_blocked = i <= snap["top_block_until"]
    if last_signal_grid_entry:
        if top_blocked:
            next_action = "HOLD (Grid入场被顶部规避封锁)"
        else:
            next_action = "BUY (Grid 入场)"
            action_detail = f"仓位: {grid_params.get('position_size', 0.99):.0%}, 信号强度: {signal_strength:.1f}"
    elif g_exit_valid:
        next_action = "SELL (Grid 离场)"
    elif last_signal_sw_entry:
        if top_blocked:
            next_action = "HOLD (Switch入场被顶部规避封锁)"
        else:
            next_action = "BUY (Switch 入场)"
            action_detail = "满仓 99%"
    elif s_exit_valid:
        next_action = "SELL (Switch 离场)"

    lines.append(f"\n  → 下一交易日操作: {next_action}")
    if action_detail:
        lines.append(f"    {action_detail}")

    return "\n".join(lines), {
        "next_action": next_action,
        "position_mode": pos_mode,
        "holding_days": holding_days,
        "dp": dp, "dt": dt, "price_pos": price_pos,
        "ret_5d": ret_5d_val, "ret_1d": ret_1d,
        "top_signal": top_triggered,
        "grid_entry": bool(last_signal_grid_entry),
        "grid_exit": bool(last_signal_grid_exit),
        "sw_entry": bool(last_signal_sw_entry),
        "sw_exit": bool(last_signal_sw_exit),
    }


def _generate_html_report(snap, explanation, meta, sym_name, target, grid_params, lookback=60):
    """生成 HTML 预测报告。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(REPORTS_DIR, exist_ok=True)

    idx = snap["index"]
    cl = snap["close"]; op = snap["open"]; hi = snap["high"]; lo = snap["low"]
    ind = snap["indicators"]
    pb = ind["PolyBasePred"].values
    dp = ind["PolyDevPct"].values
    e_g = snap["e_grid"]; x_g = snap["x_grid"]
    e_s = snap["sw_entries"]; x_s = snap["sw_exits"]
    positions = snap["positions"]

    n_show = min(lookback, len(cl))
    sl = slice(-n_show, None)

    # 每日涨跌 & PnL
    daily_ret = np.zeros(len(cl))
    for i in range(1, len(cl)):
        daily_ret[i] = cl[i] / cl[i-1] - 1
    pnl = np.where(positions > 0, daily_ret, 0)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.18, 0.22, 0.25],
        subplot_titles=("K线 + 信号", "成交量", "每日盈亏", "仓位模式"),
    )

    # ═══ Row 1: K 线 + 基线 + 买卖点 ═══
    fig.add_trace(go.Candlestick(
        x=idx[sl], open=op[sl], high=hi[sl], low=lo[sl], close=cl[sl],
        name="K线", increasing_line_color="red", decreasing_line_color="green",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx[sl], y=pb[sl], mode="lines", line=dict(color="gray", dash="dash", width=1.5),
        name="Polyfit基线",
    ), row=1, col=1)

    buy_x, buy_y = [], []; sell_x, sell_y = [], []
    last_idx = len(cl) - 1
    for j in range(max(0, len(cl)-n_show), len(cl)):
        is_last = (j == last_idx)
        # 最后一根 bar 使用 meta 中的修正信号（已排除强制平仓等伪信号）
        g_entry_j = meta["grid_entry"] if is_last else e_g[j]
        s_entry_j = meta["sw_entry"] if is_last else e_s[j]
        g_exit_j = meta["grid_exit"] if is_last else (x_g[j] and positions[j] == 1)
        s_exit_j = meta["sw_exit"] if is_last else (x_s[j] and positions[j] == 2)
        if g_entry_j or s_entry_j:
            buy_x.append(idx[j]); buy_y.append(lo[j] * 0.995)
        if g_exit_j or s_exit_j:
            sell_x.append(idx[j]); sell_y.append(hi[j] * 1.005)
    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers", marker=dict(symbol="triangle-up", size=10, color="red"),
            name=f"买入信号",
        ), row=1, col=1)
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers", marker=dict(symbol="triangle-down", size=10, color="green"),
            name=f"卖出信号",
        ), row=1, col=1)

    # ═══ Row 2: 成交量 ═══
    vol_arr = snap["volume"]
    vol_colors = ["red" if cl[i] >= op[i] else "green" for i in range(len(cl))]
    fig.add_trace(go.Bar(
        x=idx[sl], y=vol_arr[sl], marker_color=[vol_colors[i] for i in range(len(cl))][sl],
        name="成交量", showlegend=False,
    ), row=2, col=1)

    # ═══ Row 3: 每日盈亏 ═══
    pnl_slice = pnl[sl]
    pnl_colors = ["red" if v >= 0 else "green" for v in pnl_slice]
    fig.add_trace(go.Bar(
        x=idx[sl], y=pnl_slice, marker_color=pnl_colors,
        name="日盈亏", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)

    # ═══ Row 4: 仓位模式 (含图例) ═══
    pos_slice = positions[sl]
    # 用三个 trace 实现图例
    idle_mask = pos_slice == 0
    grid_mask = pos_slice == 1
    switch_mask = pos_slice == 2

    # 计算持仓 Y 值：用标记高度区分
    pos_y = np.where(pos_slice > 0, 1, 0)
    pos_color = np.array(["gray"] * len(pos_slice), dtype=object)
    pos_color[grid_mask] = "blue"
    pos_color[switch_mask] = "orange"

    fig.add_trace(go.Bar(
        x=idx[sl], y=pos_y, marker_color=pos_color,
        name="仓位", showlegend=False,
    ), row=4, col=1)

    # 添加不可见的 legend 项
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="gray"), name="空仓",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="blue"), name="Grid持仓",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="orange"), name="Switch持仓",
    ), row=4, col=1)

    fig.update_layout(
        title=f"V7 策略预测 — {sym_name} ({target})  |  操作: {meta['next_action']}  |  仓位: {meta['position_mode']}",
        height=950, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_xaxes(row=4, col=1, title="日期")
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="日盈亏 %", row=3, col=1, tickformat=".1%")
    fig.update_yaxes(title_text="仓位", row=4, col=1, tickvals=[0, 1], ticktext=["空仓", "满仓"], range=[-0.1, 1.3])

    html_path = os.path.join(REPORTS_DIR, f"predict_{sym_name.replace('.','_')}_{target}.html")
    fig.write_html(html_path)

    return html_path


def main():
    parser = argparse.ArgumentParser(description="V7 策略交易行为预测")
    parser.add_argument("symbol", help="股票代码，如 sh512890")
    parser.add_argument("strategy", help="策略名称，当前仅支持 polyfit_switch")
    parser.add_argument("target", choices=["return", "balanced"], help="评分目标")
    parser.add_argument("--lookback", type=int, default=60, help="图表显示天数 (默认 60)")
    args = parser.parse_args()

    if args.strategy != "polyfit_switch":
        print(f"错误: 不支持的策略 '{args.strategy}'，当前仅支持 polyfit_switch")
        sys.exit(1)

    data_path, sym_name = _parse_symbol(args.symbol)
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)

    print(f"加载数据: {data_path}")
    data = load_data(data_path)
    close = data["Close"]; open_ = data["Open"]
    high = data["High"]; low = data["Low"]; volume = data["Volume"]
    print(f"  {len(data)} bars  {data.index[0].date()} → {data.index[-1].date()}")

    # 加载最优 Grid 参数
    grid_params = _load_best_grid_params(args.target, data.index[-1])
    print(f"Grid 参数: tw={grid_params.get('trend_window_days','?')}, "
          f"bgp={grid_params.get('base_grid_pct','?'):.3f}, "
          f"ts={grid_params.get('trend_sensitivity','?')}")

    # 运行 V7 策略
    print("运行 V7 策略…")
    snap = _simulate_position(close, open_, high, low, volume, grid_params)

    # 解释信号
    print("分析信号…")
    explanation_text, meta = _explain_signal(snap, grid_params)

    print(f"\n{explanation_text}")

    # 过滤统计
    fs = snap["filter_stats"]
    total_filtered = sum(v for k, v in fs.items() if k != "passed" and k != "top_avoided")
    print(f"\n═══ V6/V7 过滤统计(全量) ═══")
    print(f"  OHLCV过滤: {total_filtered} 次  |  通过: {fs.get('passed',0)} 次")
    print(f"  微涨屏蔽: {fs.get('micro_up',0)}  |  长上影: {fs.get('long_upper_wick',0)}")
    print(f"  正常量能: {fs.get('normal_volume',0)}  |  阳线无势: {fs.get('green_without_momentum',0)}")
    print(f"  高位屏蔽: {fs.get('high_position',0)}  |  顶部规避: {fs.get('top_avoided',0)}")

    # 生成 HTML
    html_path = _generate_html_report(snap, explanation_text, meta, sym_name, args.target, grid_params, args.lookback)
    print(f"\nHTML 报告 → {html_path}")

    print(f"\n{'═' * 50}")
    print(f"  预测: {meta['next_action']}")
    print(f"  仓位: {meta['position_mode']} (持仓{meta['holding_days']}天)")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
