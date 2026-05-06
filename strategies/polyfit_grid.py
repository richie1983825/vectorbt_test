"""Polyfit Grid Strategy — 纯均值回复网格策略。

与 Polyfit-Switch 的区别：移除了 Switch（趋势追踪）模式，
仅保留 Grid（均值回复网格）模式，避免 Switch 的负贡献。

核心逻辑：
  - 基线：滑动窗口线性回归（252 天）预测价格中枢
  - 入场：价格偏离基线下方超过动态网格步长时，分批买入
  - 离场：止盈 / 止损 / 最大持仓天数
  - 执行：次日开盘价成交

GPU 加速：MLX（Apple Silicon）向量化批量信号生成 + 批量回测。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import gpu, get_mlx
from utils.indicators import compute_polyfit_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成（纯 Grid）
# ══════════════════════════════════════════════════════════════════

def generate_grid_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成纯 Grid 策略的入场/离场信号。

    相比 polyfit_switch：移除了 Switch 模式的全部逻辑
    （flat_wait_days / switch_deviation / trailing_stop / MA cross）。

    Returns:
        entries, exits, sizes
    """
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)

    in_position = False
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    entry_close_price = np.nan
    cooldown = 0

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        pb = poly_base[i]

        if np.isnan(dp) or np.isnan(dt) or np.isnan(vp) or np.isnan(pb) or pb <= 0 or close[i] <= 0:
            continue

        if not in_position:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan

        if cooldown > 0:
            cooldown -= 1

        # 动态网格步长
        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        if not in_position:
            if cooldown > 0:
                continue
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
                    in_position = True
                    entry_bar = i
                    entry_level = entry_lvl
                    entry_grid_step = dynamic_grid_step
                    entry_close_price = close[i]
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
# MLX 批量信号生成 + 回测（合并，单次 bar loop）
#
# 将信号生成和回测合并到一个 bar-by-bar 循环中，
# 避免存储巨大的信号数组 [n_combos, n_bars]。
# 每 bar 内所有 combo 在 MLX GPU 上并行处理。
# ══════════════════════════════════════════════════════════════════

def _grid_scan_mlx(
    close_arr: np.ndarray,
    dev_pct_arr: np.ndarray,
    dev_trend_arr: np.ndarray,
    vol_arr: np.ndarray,
    poly_base_arr: np.ndarray,
    bgp_arr: np.ndarray,
    vs_arr: np.ndarray,
    ts_arr: np.ndarray,
    mgl_arr: np.ndarray,
    tpg_arr: np.ndarray,
    slg_arr: np.ndarray,
    psz_arr: np.ndarray,
    psc_arr: np.ndarray,
    mss_arr: np.ndarray,
    cd_arr: np.ndarray,
    max_holding_days: int,
    open_arr: np.ndarray | None,
    init_cash: float = 100_000.0,
) -> np.ndarray:
    """MLX 合并信号生成 + 回测：单次 bar loop 处理所有 combo。

    Returns:
        numpy [n_combos, 6]: total_return, sharpe, max_dd, calmar, num_trades, win_rate
    """
    import mlx.core as mx

    n_bars = len(close_arr)
    n_combos = len(bgp_arr)
    if n_bars == 0 or n_combos == 0:
        return np.zeros((n_combos, 6), dtype=np.float64)

    # 成交价
    if open_arr is not None:
        fill_price = np.roll(open_arr, -1).astype(np.float32)
        fill_price[-1] = float(close_arr[-1])
    else:
        fill_price = close_arr.astype(np.float32)

    close_f32 = close_arr.astype(np.float32)
    dp_f32 = dev_pct_arr.astype(np.float32)
    dt_f32 = dev_trend_arr.astype(np.float32)
    vol_f32 = vol_arr.astype(np.float32)
    pb_f32 = poly_base_arr.astype(np.float32)

    # ── 参数数组 ──
    bgp_mx = mx.array(bgp_arr.astype(np.float32))
    vs_mx = mx.array(vs_arr.astype(np.float32))
    ts_mx = mx.array(ts_arr.astype(np.float32))
    mgl_mx = mx.array(mgl_arr.astype(np.float32))
    tpg_mx = mx.array(tpg_arr.astype(np.float32))
    slg_mx = mx.array(slg_arr.astype(np.float32))
    psz_mx = mx.array(psz_arr.astype(np.float32))
    psc_mx = mx.array(psc_arr.astype(np.float32))
    mss_mx = mx.array(mss_arr.astype(np.float32))
    cd_mx = mx.array(cd_arr.astype(np.float32))
    mhd_f = mx.array(float(max_holding_days), dtype=mx.float32)

    # ── 信号状态 ──
    in_position = mx.zeros(n_combos, dtype=mx.float32)
    entry_bar = mx.full(n_combos, mx.array(-1.0, dtype=mx.float32))
    entry_level = mx.ones(n_combos, dtype=mx.float32)
    entry_grid_step = mx.full(n_combos, mx.array(float("nan"), dtype=mx.float32))
    cooldown = mx.zeros(n_combos, dtype=mx.float32)

    # ── 回测状态 ──
    cash = mx.full(n_combos, mx.array(float(init_cash), dtype=mx.float32))
    position = mx.zeros(n_combos, dtype=mx.float32)
    entry_cost = mx.zeros(n_combos, dtype=mx.float32)
    peak_nav = mx.full(n_combos, mx.array(float(init_cash), dtype=mx.float32))
    min_drawdown = mx.zeros(n_combos, dtype=mx.float32)
    prev_nav = mx.full(n_combos, mx.array(float(init_cash), dtype=mx.float32))
    sum_ret = mx.zeros(n_combos, dtype=mx.float32)
    sum_ret2 = mx.zeros(n_combos, dtype=mx.float32)
    trade_count = mx.zeros(n_combos, dtype=mx.int32)
    win_count = mx.zeros(n_combos, dtype=mx.int32)

    zero_f = mx.array(0.0, dtype=mx.float32)
    one_f = mx.array(1.0, dtype=mx.float32)
    one_i = mx.array(1, dtype=mx.int32)
    nan_f = mx.array(float("nan"), dtype=mx.float32)

    for i in range(n_bars):
        cl = mx.array(float(close_f32[i]), dtype=mx.float32)
        fp = mx.array(float(fill_price[i]), dtype=mx.float32)
        dp = mx.array(float(dp_f32[i]), dtype=mx.float32)
        dt = mx.array(float(dt_f32[i]), dtype=mx.float32)
        vp = mx.array(float(vol_f32[i]), dtype=mx.float32)
        pb = mx.array(float(pb_f32[i]), dtype=mx.float32)

        # 有效性
        valid = (~mx.isnan(cl) & (cl > zero_f)
                 & ~mx.isnan(dp) & ~mx.isnan(dt) & ~mx.isnan(vp)
                 & ~mx.isnan(pb) & (pb > zero_f))

        # Cooldown 递减（不在持仓中的 combo）
        cooldown = mx.where((in_position <= zero_f) & (cooldown > zero_f) & valid,
                            cooldown - one_f, cooldown)

        # 动态网格步长
        vol_mult = one_f + vs_mx * mx.maximum(vp, zero_f)
        dgs = bgp_mx * (one_f + ts_mx * mx.abs(dt)) * vol_mult
        dgs = mx.maximum(dgs, bgp_mx * mx.array(0.3, dtype=mx.float32))

        # ── 入场逻辑 ──
        sig = mx.abs(dp) / mx.maximum(dgs, mx.array(1e-9, dtype=mx.float32))
        el = mx.floor(sig)
        el = mx.where(el < one_f, one_f, mx.where(el > mgl_mx, mgl_mx, el))
        eth = -el * dgs

        can_enter = ((in_position <= zero_f) & (cooldown <= zero_f)
                     & (dp <= eth) & (sig >= mss_mx) & valid)
        sz = mx.abs(dp) * (one_f + mx.maximum(vp, zero_f)) * psc_mx
        sz = mx.where(sz > psz_mx, psz_mx, mx.where(sz < zero_f, zero_f, sz))
        can_enter = can_enter & (sz > zero_f)

        # 执行入场（先保存入场前状态，避免同 bar 入场+离场）
        was_in = in_position
        buy_amount = cash * sz
        shares = mx.where(can_enter, buy_amount / fp, zero_f)
        cash = mx.where(can_enter, cash - shares * fp, cash)
        position = mx.where(can_enter, shares, position)
        entry_cost = mx.where(can_enter, shares * fp, entry_cost)
        in_position = mx.where(can_enter, one_f, in_position)
        entry_bar = mx.where(can_enter, mx.array(float(i), dtype=mx.float32), entry_bar)
        entry_level = mx.where(can_enter, el, entry_level)
        entry_grid_step = mx.where(can_enter, dgs, entry_grid_step)

        # ── 离场逻辑（仅对入场前已持仓的 combo，匹配 CPU 的 if-else 语义）──
        hd = mx.array(float(i), dtype=mx.float32) - entry_bar
        hl = hd >= mhd_f
        rs = mx.maximum(dgs, entry_grid_step)
        rs = mx.where(mx.isnan(entry_grid_step), dgs, rs)
        tp_threshold = entry_level * rs * tpg_mx
        sl_threshold = entry_level * rs * slg_mx

        can_exit = ((was_in > zero_f)
                    & (hl | (dp >= tp_threshold) | (dp <= -sl_threshold))
                    & valid)

        # 执行离场
        sell_amount = position * fp
        cash = mx.where(can_exit, cash + sell_amount, cash)
        win_trade = (sell_amount > entry_cost) & can_exit
        win_count = mx.where(win_trade, win_count + one_i, win_count)
        trade_count = mx.where(can_exit, trade_count + one_i, trade_count)
        position = mx.where(can_exit, zero_f, position)
        entry_cost = mx.where(can_exit, zero_f, entry_cost)
        in_position = mx.where(can_exit, zero_f, in_position)
        cooldown = mx.where(can_exit, cd_mx, cooldown)

        # ── NAV 统计 ──
        nav = cash + position * cl
        valid_prev = prev_nav > zero_f
        ret = mx.where(valid_prev, (nav - prev_nav) / prev_nav, zero_f)
        sum_ret = sum_ret + ret
        sum_ret2 = sum_ret2 + ret * ret
        peak_nav = mx.maximum(peak_nav, nav)
        dd = mx.where(peak_nav > zero_f, (nav - peak_nav) / peak_nav, zero_f)
        min_drawdown = mx.minimum(min_drawdown, dd)
        prev_nav = nav

        mx.eval(cash, position, in_position, entry_cost, entry_bar,
                entry_level, entry_grid_step, cooldown,
                peak_nav, min_drawdown, prev_nav,
                sum_ret, sum_ret2, trade_count, win_count)

    # ── 末尾强制平仓 ──
    final_cl = mx.array(float(close_f32[-1]), dtype=mx.float32)
    still_in = in_position > zero_f
    sell_amount = position * final_cl
    cash = mx.where(still_in, cash + sell_amount, cash)
    win_trade = sell_amount > entry_cost
    win_count = mx.where(still_in & win_trade, win_count + one_i, win_count)
    trade_count = mx.where(still_in, trade_count + one_i, trade_count)
    position = mx.where(still_in, zero_f, position)
    mx.eval(cash, position, win_count, trade_count)

    # ── 计算指标 ──
    final_nav = cash + position * final_cl
    init_cash_f = mx.array(float(init_cash), dtype=mx.float32)
    total_return = (final_nav - init_cash_f) / init_cash_f
    max_dd = min_drawdown

    n_bars_f = mx.array(float(n_bars), dtype=mx.float32)
    mean_ret = sum_ret / n_bars_f
    var = sum_ret2 / n_bars_f - mean_ret * mean_ret
    sharpe = mx.zeros(n_combos, dtype=mx.float32)
    valid_var = var > mx.array(1e-12, dtype=mx.float32)
    sharpe = mx.where(valid_var,
                      (mean_ret / mx.sqrt(var)) * mx.sqrt(mx.array(365.0, dtype=mx.float32)),
                      zero_f)

    calmar = mx.zeros(n_combos, dtype=mx.float32)
    valid_dd = max_dd < mx.array(-1e-9, dtype=mx.float32)
    calmar = mx.where(valid_dd, total_return / mx.abs(max_dd), zero_f)

    win_rate = mx.zeros(n_combos, dtype=mx.float32)
    has_trades = trade_count > 0
    win_rate = mx.where(has_trades,
                        win_count.astype(mx.float32) / trade_count.astype(mx.float32),
                        zero_f)

    mx.eval(total_return, sharpe, max_dd, calmar, trade_count, win_rate)

    return np.column_stack([
        np.array(total_return).astype(np.float64),
        np.array(sharpe).astype(np.float64),
        np.array(max_dd).astype(np.float64),
        np.array(calmar).astype(np.float64),
        np.array(trade_count).astype(np.float64),
        np.array(win_rate).astype(np.float64),
    ])


# ══════════════════════════════════════════════════════════════════
# 参数扫描
# ══════════════════════════════════════════════════════════════════

_grid_scan_cache: dict[tuple, pd.DataFrame] = {}


def _grid_cache_key(close: pd.Series, open_: pd.Series | None = None) -> tuple:
    return (close.index[0], close.index[-1], len(close), open_ is not None)


def clear_grid_cache() -> None:
    """清除 Grid 扫描缓存（用于跨品种对比等场景）。"""
    _grid_scan_cache.clear()


def scan_polyfit_grid(
    close: pd.Series, open_: pd.Series | None = None,
) -> pd.DataFrame:
    """Polyfit-Grid 策略参数扫描（MLX GPU 批量）。

    扫描参数：
      - indicator: trend_window_days × vol_window_days
      - grid: base_grid_pct × vol_scale × trend_sens × max_grid_levels
              × tpg × slg × position_size × position_sizing_coef
              × min_signal_strength
    """
    cache_key = _grid_cache_key(close, open_)
    if cache_key in _grid_scan_cache:
        print("  [PolyfitGrid] Cache hit — reusing scan results")
        return _grid_scan_cache[cache_key].copy()

    use_mlx = gpu()["mlx_available"]
    open_arr = open_.values if open_ is not None else None

    trend_windows = [10, 15, 20]
    vol_windows = [10, 20, 30]
    grid_pcts = [0.008, 0.010, 0.012, 0.015]
    vol_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    trend_sens = [4.0, 6.0, 8.0, 10.0]
    max_grid_levels_vals = [2, 3, 4]
    tpg_values = [0.6, 0.8, 1.0]
    slg_values = [1.2, 1.6, 2.0]
    position_sizes = [0.92, 0.94, 0.97, 0.99]
    position_sizing_coefs = [20.0, 30.0, 60.0]
    min_signal_strengths = [0.30, 0.45, 0.60]

    print("  [PolyfitGrid] MLX-scanning grid params…")

    indicator_combos = list(product(trend_windows, vol_windows))
    grid_combos = list(product(
        grid_pcts, vol_scales, trend_sens, max_grid_levels_vals,
        tpg_values, slg_values, position_sizes, position_sizing_coefs,
        min_signal_strengths,
    ))
    n_ind = len(indicator_combos)
    n_grid = len(grid_combos)
    n_total = n_ind * n_grid
    print(f"    indicator combos={n_ind}  grid combos={n_grid}  total={n_total}")

    # 预计算基线（复用）
    from utils.indicators import compute_polyfit_base_only, add_trend_vol_indicators
    base_indicators = compute_polyfit_base_only(close, fit_window_days=252, ma_windows=[])

    indicator_cache = {}
    for tw, vw in indicator_combos:
        key = (tw, vw)
        if key not in indicator_cache:
            indicator_cache[key] = add_trend_vol_indicators(
                base_indicators, close, trend_window_days=tw, vol_window_days=vw,
            )

    common_idx = base_indicators.index
    if len(common_idx) == 0:
        return pd.DataFrame()
    cl_aligned = close.loc[common_idx]
    cl_arr = cl_aligned.values
    op_aligned = open_arr[close.index.get_indexer(common_idx)] if open_arr is not None else None

    poly_base_arr = base_indicators["PolyBasePred"].values
    dev_pct_arr = base_indicators["PolyDevPct"].values

    # 预构建 per-combo 参数数组
    s1_bgp = np.array([p[0] for p in grid_combos], dtype=np.float64)
    s1_vs = np.array([p[1] for p in grid_combos], dtype=np.float64)
    s1_ts = np.array([p[2] for p in grid_combos], dtype=np.float64)
    s1_mgl = np.array([p[3] for p in grid_combos], dtype=np.float64)
    s1_tpg = np.array([p[4] for p in grid_combos], dtype=np.float64)
    s1_slg = np.array([p[5] for p in grid_combos], dtype=np.float64)
    s1_psz = np.array([p[6] for p in grid_combos], dtype=np.float64)
    s1_psc = np.array([p[7] for p in grid_combos], dtype=np.float64)
    s1_mss = np.array([p[8] for p in grid_combos], dtype=np.float64)
    s1_cd = np.full(n_grid, 1.0, dtype=np.float64)

    results = []

    if use_mlx:
        # MLX 路径：每个 indicator set 单独批量处理
        for ii, (tw, vw) in enumerate(indicator_combos):
            indicators = indicator_cache[(tw, vw)]
            dt_arr = indicators["PolyDevTrend"].reindex(common_idx).values
            vol_arr = indicators["RollingVolPct"].reindex(common_idx).values

            bt = _grid_scan_mlx(
                cl_arr, dev_pct_arr, dt_arr, vol_arr, poly_base_arr,
                s1_bgp, s1_vs, s1_ts, s1_mgl, s1_tpg, s1_slg,
                s1_psz, s1_psc, s1_mss, s1_cd,
                max_holding_days=45, open_arr=op_aligned,
            )

            for gi, (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss) in enumerate(grid_combos):
                if int(bt[gi][4]) == 0:
                    continue
                results.append({
                    "total_return": bt[gi][0], "sharpe_ratio": bt[gi][1],
                    "max_drawdown": bt[gi][2], "calmar_ratio": bt[gi][3],
                    "num_trades": int(bt[gi][4]), "win_rate": bt[gi][5],
                    "trend_window_days": tw, "vol_window_days": vw,
                    "base_grid_pct": bgp, "volatility_scale": vs,
                    "trend_sensitivity": ts, "max_grid_levels": max_gl,
                    "take_profit_grid": tpg, "stop_loss_grid": slg,
                    "max_holding_days": 45, "cooldown_days": 1,
                    "min_signal_strength": min_ss,
                    "position_size": pos_sz, "position_sizing_coef": pos_coef,
                })
    else:
        # CPU 路径
        for tw, vw in indicator_combos:
            indicators = indicator_cache[(tw, vw)]
            for (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss) in grid_combos:
                entries, exits, sizes = generate_grid_signals(
                    cl_arr, dev_pct_arr,
                    indicators["PolyDevTrend"].reindex(common_idx).values,
                    indicators["RollingVolPct"].reindex(common_idx).values,
                    poly_base_arr,
                    base_grid_pct=bgp, volatility_scale=vs,
                    trend_sensitivity=ts, max_grid_levels=max_gl,
                    take_profit_grid=tpg, stop_loss_grid=slg,
                    max_holding_days=45, cooldown_days=1,
                    min_signal_strength=min_ss,
                    position_size=pos_sz, position_sizing_coef=pos_coef,
                )
                if entries.sum() == 0:
                    continue
                m = run_backtest(cl_aligned, entries, exits, sizes, open_=op_aligned)
                m["trend_window_days"] = tw
                m["vol_window_days"] = vw
                m["base_grid_pct"] = bgp
                m["volatility_scale"] = vs
                m["trend_sensitivity"] = ts
                m["max_grid_levels"] = max_gl
                m["take_profit_grid"] = tpg
                m["stop_loss_grid"] = slg
                m["max_holding_days"] = 45
                m["cooldown_days"] = 1
                m["min_signal_strength"] = min_ss
                m["position_size"] = pos_sz
                m["position_sizing_coef"] = pos_coef
                results.append(m)

    result = pd.DataFrame(results)
    _grid_scan_cache[cache_key] = result
    return result
