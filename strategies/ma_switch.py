"""MA Switch Strategy — 均线网格 + 均线交叉双模式策略。

双模式说明：
  1. Grid 模式（默认）：均值回复网格交易，在 MA 基准线下方挂网格多单。
  2. Switch 模式（趋势追踪）：当价格连续 flat_wait_days 根 bar 在 MA 上方
     且偏离超过阈值时激活，使用快慢均线金叉入场 + 最高价回撤止损离场。

Switch 模式激活条件：flat_wait_days 后，close > MA 且 dev_pct > switch_deviation_m1
Switch 模式关闭条件：dev_pct < switch_deviation_m2（偏离回归到合理范围）

GPU 加速：MLX（Apple Silicon）向量化批量回测。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import gpu
from utils.indicators import compute_ma_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成
# ══════════════════════════════════════════════════════════════════

def generate_switch_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    ma_base: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,
    position_sizing_coef: float = 30.0,
    flat_wait_days: int = 8,
    switch_deviation_m1: float = 0.03,
    switch_deviation_m2: float = 0.01,
    switch_trailing_stop: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 MA-Switch 策略的入场/离场信号。

    两种模式互斥：同一时间只能处于一种模式。
    Switch 模式仅在 Grid 模式闲置（无持仓）且满足激活条件时触发。

    返回值：
      entries/exits/sizes 均为 n_bars 长度数组。
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
    switch_mode_active = False
    flat_days = 0
    peak_close = 0.0

    for i in range(n):
        dp = dev_pct[i]; dt = dev_trend[i]; vp = rolling_vol_pct[i]
        mb = ma_base[i]; cl = close[i]; fma = ma_fast[i]; sma = ma_slow[i]

        nan_vars = (np.isnan(dp) or np.isnan(dt) or np.isnan(vp)
                    or np.isnan(mb) or np.isnan(fma) or np.isnan(sma))
        if nan_vars or mb <= 0 or cl <= 0:
            continue

        if in_position and switch_mode_active:
            peak_close = max(peak_close, cl)

        if in_position:
            flat_days = 0
        else:
            entry_bar = -1; entry_level = 1; entry_grid_step = np.nan
            flat_days += 1

        if cooldown > 0:
            cooldown -= 1

        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        # Switch mode activation
        if (not switch_mode_active and not in_position
                and flat_days >= flat_wait_days
                and cl > mb and dp > switch_deviation_m1):
            switch_mode_active = True

        if switch_mode_active:
            if dp < switch_deviation_m2:
                switch_mode_active = False
                continue
            if cl <= mb or dp <= switch_deviation_m1:
                continue
            if fma > sma and not in_position:
                full_size = float(np.nextafter(1.0, 0.0))
                entries[i] = True; sizes[i] = full_size
                in_position = True; entry_bar = i; entry_level = 1
                entry_grid_step = max(base_grid_pct, 1e-9)
                flat_days = 0; peak_close = cl
                continue
            if in_position:
                if cl <= peak_close * (1.0 - switch_trailing_stop):
                    exits[i] = True
                    in_position = False
                    cooldown = cooldown_days
                    peak_close = 0.0
            continue

        # Grid mode logic
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
                    entries[i] = True; sizes[i] = size
                    in_position = True; entry_bar = i
                    entry_level = entry_lvl
                    entry_grid_step = dynamic_grid_step
        else:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            if hold_limit or dp >= tp_threshold or dp <= -sl_threshold:
                exits[i] = True
                in_position = False
                cooldown = cooldown_days

    if in_position:
        exits[-1] = True

    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# 单阶段参数扫描 — Grid + Switch 参数全排列（CPU）
# ══════════════════════════════════════════════════════════════════

def scan_switch_strategy(close: pd.Series) -> pd.DataFrame:
    """平坦参数扫描（grid 参数 + switch 参数全排列）。保留作为参考。

    与 scan_switch_two_stage 的区别：
      - 此函数对 grid 和 switch 参数做全排列，组合数巨大
      - scan_switch_two_stage 分两阶段：先优化 grid，再在最优 grid 上优化 switch
    """
    print("  Switch parameter scan...")

    windows = [50, 75]
    grid_pcts = [0.01, 0.02]
    vol_scales = [0.5, 2.0]
    trend_sens = [4.0, 12.0]
    tpg_values = [0.7, 1.0]
    slg_values = [1.2, 2.0]

    flat_wait_days_vals = [5, 8, 15]
    switch_m1_vals = [0.02, 0.03, 0.05]
    switch_m2_vals = [0.005, 0.01, 0.02]
    switch_fast_vals = [5, 20]
    switch_slow_vals = [10, 60]

    all_ma_windows = sorted(set(switch_fast_vals + switch_slow_vals))

    grid_params = list(product(grid_pcts, vol_scales, trend_sens, tpg_values, slg_values))
    switch_params = []
    for fw, sw_m1, sw_m2, sw_fast, sw_slow in product(
        flat_wait_days_vals, switch_m1_vals, switch_m2_vals,
        switch_fast_vals, switch_slow_vals,
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_params.append((fw, sw_m1, sw_m2, sw_fast, sw_slow))

    all_combos = list(product(windows, grid_params, switch_params))
    results = []
    count = 0

    for w, (bgp, vs, ts, tpg, slg), (fw, sw_m1, sw_m2, sw_fast, sw_slow) in all_combos:
        indicators = compute_ma_switch_indicators(close, w, ma_windows=all_ma_windows)
        close_arr = close.values
        ma_fast_arr = indicators[f"MA{sw_fast}"].values
        ma_slow_arr = indicators[f"MA{sw_slow}"].values

        entries, exits, sizes = generate_switch_signals(
            close_arr,
            indicators["MADevPct"].values,
            indicators["MADevTrend"].values,
            indicators["RollingVolPct"].values,
            indicators["MABase"].values,
            ma_fast_arr, ma_slow_arr,
            base_grid_pct=bgp, volatility_scale=vs,
            trend_sensitivity=ts, take_profit_grid=tpg,
            stop_loss_grid=slg,
            flat_wait_days=fw,
            switch_deviation_m1=sw_m1,
            switch_deviation_m2=sw_m2,
        )
        if entries.sum() == 0:
            continue
        m = run_backtest(close, entries, exits, sizes)
        m["ma_window"] = w
        m["base_grid_pct"] = bgp
        m["volatility_scale"] = vs
        m["trend_sensitivity"] = ts
        m["take_profit_grid"] = tpg
        m["stop_loss_grid"] = slg
        m["flat_wait_days"] = fw
        m["switch_deviation_m1"] = sw_m1
        m["switch_deviation_m2"] = sw_m2
        m["switch_fast_ma"] = sw_fast
        m["switch_slow_ma"] = sw_slow
        results.append(m)
        count += 1
        if count % 200 == 0:
            print(f"  [Switch] {count}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════
# 两阶段参数扫描 — 先优化 Grid 参数，再优化 Switch 参数
# ══════════════════════════════════════════════════════════════════

_SWITCH_PARAMS = dict(
    flat_wait_days_vals=[5, 8, 15],
    switch_m1_vals=[0.02, 0.03, 0.05],
    switch_m2_vals=[0.005, 0.01, 0.02],
    switch_trailing_stop_vals=[0.02, 0.03, 0.05, 0.07, 0.10],
    switch_fast_vals=[5, 20],
    switch_slow_vals=[10, 60],
)


def scan_switch_two_stage(close: pd.Series,
                          open_: pd.Series | None = None) -> pd.DataFrame:
    """两阶段扫描：先复用 MA Grid 最优参数，再优化 Switch 逻辑。

    Stage 1: 运行 MA Grid 扫描（scan_ma_strategy），找到最优网格参数
    Stage 2: 固定最优网格参数，扫描 switch 相关的入场/出场条件参数

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    from .ma_grid import scan_ma_strategy

    use_mlx = gpu()["mlx_available"]
    open_arr = open_.values if open_ is not None else None

    # ── Stage 1: 复用 MA Grid 扫描结果 ──
    print("  [Switch Stage 1] Reusing MA Grid optimal params...")
    ma_results = scan_ma_strategy(close, open_=open_)

    if ma_results.empty:
        return pd.DataFrame()

    best_grid = ma_results.nlargest(1, "total_return").iloc[0]
    best_w = int(best_grid["ma_window"])
    best_bgp = best_grid["base_grid_pct"]
    best_vs = best_grid["volatility_scale"]
    best_ts = best_grid["trend_sensitivity"]
    best_tpg = best_grid["take_profit_grid"]
    best_slg = best_grid["stop_loss_grid"]

    print(f"  [Switch Stage 1] MA Grid best: w={best_w} bgp={best_bgp:.4f} "
          f"vs={best_vs:.1f} ts={best_ts:.0f} tpg={best_tpg:.2f} slg={best_slg:.1f} "
          f"ret={best_grid['total_return']:.1%}")

    # ── Stage 2: 固定 Grid 参数，扫描 Switch 参数 ──
    print("  [Switch Stage 2] Optimizing switch params (grid fixed)...")

    all_ma_windows = sorted(set(_SWITCH_PARAMS["switch_fast_vals"]
                                + _SWITCH_PARAMS["switch_slow_vals"]))

    switch_combos = []
    for fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow in product(
        _SWITCH_PARAMS["flat_wait_days_vals"],
        _SWITCH_PARAMS["switch_m1_vals"],
        _SWITCH_PARAMS["switch_m2_vals"],
        _SWITCH_PARAMS["switch_trailing_stop_vals"],
        _SWITCH_PARAMS["switch_fast_vals"],
        _SWITCH_PARAMS["switch_slow_vals"],
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_combos.append((fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow))

    n_switch = len(switch_combos)
    indicators = compute_ma_switch_indicators(close, best_w, ma_windows=all_ma_windows)
    close_arr = close.values

    results = []

    if use_mlx:
        # MLX 路径：CPU 逐组合生成信号 + MLX 批量回测
        all_entries = []; all_exits = []; all_sizes = []; valid_indices = []
        for idx, (fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in enumerate(switch_combos):
            entries, exits, sizes = generate_switch_signals(
                close_arr,
                indicators["MADevPct"].values,
                indicators["MADevTrend"].values,
                indicators["RollingVolPct"].values,
                indicators["MABase"].values,
                indicators[f"MA{sw_fast}"].values,
                indicators[f"MA{sw_slow}"].values,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, take_profit_grid=best_tpg,
                stop_loss_grid=best_slg,
                flat_wait_days=fw, switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
                position_size=0.5, position_sizing_coef=30.0,
            )
            if entries.sum() == 0:
                continue
            all_entries.append(entries); all_exits.append(exits)
            all_sizes.append(sizes); valid_indices.append(idx)

        if all_entries:
            entries_b = np.array(all_entries); exits_b = np.array(all_exits)
            sizes_b = np.array(all_sizes)
            bt = run_backtest_batch(close_arr, entries_b, exits_b, sizes_b,
                                    n_combos=len(valid_indices), open_=open_arr)
            for bi, orig_idx in enumerate(valid_indices):
                if int(bt[bi][4]) == 0:
                    continue
                fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow = switch_combos[orig_idx]
                results.append({
                    "total_return": bt[bi][0], "sharpe_ratio": bt[bi][1],
                    "max_drawdown": bt[bi][2], "calmar_ratio": bt[bi][3],
                    "num_trades": int(bt[bi][4]), "win_rate": bt[bi][5],
                    "ma_window": best_w,
                    "base_grid_pct": best_bgp, "volatility_scale": best_vs,
                    "trend_sensitivity": best_ts, "take_profit_grid": best_tpg,
                    "stop_loss_grid": best_slg,
                    "flat_wait_days": fw, "switch_deviation_m1": sw_m1,
                    "switch_deviation_m2": sw_m2,
                    "switch_trailing_stop": sw_tr,
                    "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
                })
    else:
        # CPU 路径：逐组合生成信号 + 回测
        for (fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in switch_combos:
            entries, exits, sizes = generate_switch_signals(
                close_arr,
                indicators["MADevPct"].values,
                indicators["MADevTrend"].values,
                indicators["RollingVolPct"].values,
                indicators["MABase"].values,
                indicators[f"MA{sw_fast}"].values,
                indicators[f"MA{sw_slow}"].values,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, take_profit_grid=best_tpg,
                stop_loss_grid=best_slg,
                flat_wait_days=fw, switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
                position_size=0.5, position_sizing_coef=30.0,
            )
            if entries.sum() == 0:
                continue
            m = run_backtest(close, entries, exits, sizes, open_=open_)
            m["ma_window"] = best_w
            m["base_grid_pct"] = best_bgp
            m["volatility_scale"] = best_vs
            m["trend_sensitivity"] = best_ts
            m["take_profit_grid"] = best_tpg
            m["stop_loss_grid"] = best_slg
            m["flat_wait_days"] = fw
            m["switch_deviation_m1"] = sw_m1
            m["switch_deviation_m2"] = sw_m2
            m["switch_fast_ma"] = sw_fast
            m["switch_slow_ma"] = sw_slow
            results.append(m)

    return pd.DataFrame(results)
