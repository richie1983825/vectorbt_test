"""MA Switch Strategy — MA grid + MA-crossover switch mode.

GPU-batched signal generation via CuPy RawKernel when CuPy is available.
Two-stage parameter scan: grid params first (switch disabled), then switch params.
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import xp, gpu
from utils.indicators import compute_ma_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU signal generation
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate signals for MA Switch Strategy.

    Two modes:
    - Grid mode (default): mean-reversion grid entries below MA baseline.
    - Switch mode (activated after flat_wait_days flat bars when
      close > ma_base AND dev_pct > switch_deviation_m1): MA crossover
      trend-following. Switch mode deactivates when dev_pct < switch_deviation_m2.
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
    ending_position = 0.0
    switch_mode_active = False
    flat_days = 0

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        mb = ma_base[i]
        cl = close[i]
        fma = ma_fast[i]
        sma = ma_slow[i]

        nan_vars = (np.isnan(dp) or np.isnan(dt) or np.isnan(vp)
                    or np.isnan(mb) or np.isnan(fma) or np.isnan(sma))
        if nan_vars or mb <= 0 or cl <= 0:
            continue

        if in_position:
            flat_days = 0
        else:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan
            flat_days += 1

        if cooldown > 0:
            cooldown -= 1

        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        # switch mode activation
        if (not switch_mode_active and not in_position
                and flat_days >= flat_wait_days
                and cl > mb and dp > switch_deviation_m1):
            switch_mode_active = True

        # handle switch mode
        if switch_mode_active:
            if dp < switch_deviation_m2:
                switch_mode_active = False
                continue
            if cl <= mb or dp <= switch_deviation_m1:
                continue
            if fma > sma:
                if not in_position:
                    full_size = float(np.nextafter(1.0, 0.0))
                    entries[i] = True
                    sizes[i] = full_size
                    in_position = True
                    entry_bar = i
                    entry_level = 1
                    entry_grid_step = max(base_grid_pct, 1e-9)
                    flat_days = 0
                continue
            if fma < sma:
                if in_position:
                    exits[i] = True
                    in_position = False
                    cooldown = cooldown_days
                continue
            continue

        # grid mode
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
# GPU-batched switch signal generation (CuPy RawKernel)
# ══════════════════════════════════════════════════════════════════

_switch_kernel = None

_SWITCH_KERNEL_CODE = r"""
extern "C" __global__ void switch_signals_kernel(
    const double* close,
    const double* dev_pct,
    const double* dev_trend,
    const double* vol_pct,
    const double* ma_base,
    const double* ma_all,
    const int* fast_ma_idx,
    const int* slow_ma_idx,
    const double* base_grid_pct,
    const double* volatility_scale,
    const double* trend_sensitivity,
    const double* take_profit_grid,
    const double* stop_loss_grid,
    const int* flat_wait_days,
    const double* switch_deviation_m1,
    const double* switch_deviation_m2,
    bool* entries,
    bool* exits,
    double* sizes,
    int n_bars,
    int n_combos,
    int max_grid_levels,
    int max_holding_days,
    int cooldown_days,
    double min_signal_strength,
    double position_size,
    double position_sizing_coef
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];
    int fw = flat_wait_days[combo_idx];
    double sw_m1 = switch_deviation_m1[combo_idx];
    double sw_m2 = switch_deviation_m2[combo_idx];
    int fi = fast_ma_idx[combo_idx];
    int si = slow_ma_idx[combo_idx];

    bool in_position = false;
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    int cooldown = 0;
    bool switch_mode_active = false;
    int flat_days = 0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        double dp = dev_pct[i];
        double dt = dev_trend[i];
        double vp = vol_pct[i];
        double mb = ma_base[i];
        double fma = ma_all[fi * n_bars + i];
        double sma = ma_all[si * n_bars + i];

        if (isnan(cl) || isnan(dp) || isnan(dt) || isnan(vp)
            || isnan(mb) || isnan(fma) || isnan(sma)
            || mb <= 0.0 || cl <= 0.0) {
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        if (in_position) {
            flat_days = 0;
        } else {
            entry_bar = -1;
            entry_level = 1;
            entry_grid_step = -1.0;
            flat_days++;
        }
        if (cooldown > 0) cooldown--;

        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        if (!switch_mode_active && !in_position
                && flat_days >= fw && cl > mb && dp > sw_m1) {
            switch_mode_active = true;
        }

        if (switch_mode_active) {
            if (dp < sw_m2) {
                switch_mode_active = false;
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            if (cl <= mb || dp <= sw_m1) {
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            if (fma > sma) {
                if (!in_position) {
                    entries[combo_idx * n_bars + i] = true;
                    sizes[combo_idx * n_bars + i] = 0.9999999999999999;
                    in_position = true;
                    entry_bar = i;
                    entry_level = 1;
                    entry_grid_step = fmax(bgp, 1e-9);
                    flat_days = 0;
                }
                exits[combo_idx * n_bars + i] = false;
                continue;
            }
            if (fma < sma) {
                if (in_position) {
                    exits[combo_idx * n_bars + i] = true;
                    in_position = false;
                    cooldown = cooldown_days;
                }
                entries[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        // grid mode
        entries[combo_idx * n_bars + i] = false;
        exits[combo_idx * n_bars + i] = false;
        sizes[combo_idx * n_bars + i] = 0.0;

        if (!in_position) {
            if (cooldown > 0) continue;
            double signal_strength = fabs(dp) / fmax(dgs, 1e-9);
            int entry_lvl = (int)floor(signal_strength);
            entry_lvl = entry_lvl < 1 ? 1 : (entry_lvl > max_grid_levels ? max_grid_levels : entry_lvl);
            double entry_threshold = -(double)entry_lvl * dgs;
            if (dp <= entry_threshold && signal_strength >= min_signal_strength) {
                double size = fabs(dp) * (1.0 + fmax(vp, 0.0)) * position_sizing_coef;
                size = size > position_size ? position_size : (size < 0.0 ? 0.0 : size);
                if (size > 0.0) {
                    entries[combo_idx * n_bars + i] = true;
                    sizes[combo_idx * n_bars + i] = size;
                    in_position = true;
                    entry_bar = i;
                    entry_level = entry_lvl;
                    entry_grid_step = dgs;
                }
            }
        } else {
            int holding_days = i - entry_bar;
            bool hold_limit = holding_days >= max_holding_days;
            double ref_step = fmax(dgs, entry_grid_step);
            if (entry_grid_step < 0.0) ref_step = dgs;
            double tp_threshold = entry_level * ref_step * tpg;
            double sl_threshold = entry_level * ref_step * slg;
            if (hold_limit || dp >= tp_threshold || dp <= -sl_threshold) {
                exits[combo_idx * n_bars + i] = true;
                in_position = false;
                cooldown = cooldown_days;
            }
        }
    }

    if (in_position) {
        exits[combo_idx * n_bars + n_bars - 1] = true;
    }
}
"""


def _get_switch_kernel():
    global _switch_kernel
    if _switch_kernel is None:
        cp = xp()
        _switch_kernel = cp.RawKernel(_SWITCH_KERNEL_CODE, "switch_signals_kernel")
    return _switch_kernel


def generate_switch_signals_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    ma_base: np.ndarray,
    ma_all: np.ndarray,
    fast_ma_idx: np.ndarray,
    slow_ma_idx: np.ndarray,
    base_grid_pcts: np.ndarray,
    volatility_scales: np.ndarray,
    trend_sensitivities: np.ndarray,
    take_profit_grids: np.ndarray,
    stop_loss_grids: np.ndarray,
    flat_wait_days_arr: np.ndarray,
    switch_deviation_m1_arr: np.ndarray,
    switch_deviation_m2_arr: np.ndarray,
    max_grid_levels: int = 3,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,
    position_sizing_coef: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-batched switch signal generation."""
    cp = xp()
    n_bars = len(close)
    n_combos = len(base_grid_pcts)

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

    bgp_d = _pad_f64(base_grid_pcts)
    vs_d = _pad_f64(volatility_scales)
    ts_d = _pad_f64(trend_sensitivities)
    tpg_d = _pad_f64(take_profit_grids)
    slg_d = _pad_f64(stop_loss_grids)
    fw_d = _pad_i32(flat_wait_days_arr)
    m1_d = _pad_f64(switch_deviation_m1_arr)
    m2_d = _pad_f64(switch_deviation_m2_arr)
    fi_d = _pad_i32(fast_ma_idx)
    si_d = _pad_i32(slow_ma_idx)

    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    trend_d = cp.asarray(dev_trend, dtype=cp.float64)
    vol_d = cp.asarray(rolling_vol_pct, dtype=cp.float64)
    mb_d = cp.asarray(ma_base, dtype=cp.float64)
    ma_d = cp.asarray(ma_all.ravel(), dtype=cp.float64)

    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    kernel = _get_switch_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, dev_d, trend_d, vol_d, mb_d, ma_d, fi_d, si_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d, fw_d, m1_d, m2_d,
            entries_d, exits_d, sizes_d,
            n_bars, n_combos, max_grid_levels, max_holding_days, cooldown_days,
            min_signal_strength, position_size, position_sizing_coef,
        ),
    )

    entries = cp.asnumpy(entries_d).reshape(padded, n_bars)[:n_combos]
    exits = cp.asnumpy(exits_d).reshape(padded, n_bars)[:n_combos]
    sizes = cp.asnumpy(sizes_d).reshape(padded, n_bars)[:n_combos]
    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# Single-stage scan (flat grid of grid + switch params)
# ══════════════════════════════════════════════════════════════════

def scan_switch_strategy(close: pd.Series) -> pd.DataFrame:
    """Flat parameter scan (grid + switch together).  Kept for reference."""
    print("  Switch parameter scan…")

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
    ma_to_idx = {w: i for i, w in enumerate(all_ma_windows)}
    use_gpu = gpu()["cupy_available"]

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
# Two-stage scan: grid params first, then switch params
# ══════════════════════════════════════════════════════════════════

_GRID_PARAMS = dict(
    windows=[20, 50, 75, 100, 150, 200],
    grid_pcts=[0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02],
    vol_scales=[0.5, 1.0, 1.5, 2.0, 3.0],
    trend_sens=[3.0, 6.0, 9.0, 12.0, 15.0],
    tpg_values=[0.5, 0.7, 0.85, 1.0],
    slg_values=[1.0, 1.3, 1.6, 2.0, 2.5],
)

_SWITCH_PARAMS = dict(
    flat_wait_days_vals=[5, 8, 15],
    switch_m1_vals=[0.02, 0.03, 0.05],
    switch_m2_vals=[0.005, 0.01, 0.02],
    switch_fast_vals=[5, 20],
    switch_slow_vals=[10, 60],
)


def scan_switch_two_stage(close: pd.Series) -> pd.DataFrame:
    """Two-stage scan: optimize grid params (switch disabled), then switch params.

    Stage 1: Grid-only with switch mode disabled (flat_wait_days=∞).
    Stage 2: Fix best grid params, optimize switch parameters.
    """
    use_gpu = gpu()["cupy_available"]
    all_ma_windows = sorted(set(_SWITCH_PARAMS["switch_fast_vals"]
                                + _SWITCH_PARAMS["switch_slow_vals"]))
    ma_to_idx = {w: i for i, w in enumerate(all_ma_windows)}

    # ── Stage 1: grid params, switch disabled ──
    print("  [Switch Stage 1] Optimizing grid params (switch disabled)…")

    grid_combos = list(product(
        _GRID_PARAMS["grid_pcts"], _GRID_PARAMS["vol_scales"],
        _GRID_PARAMS["trend_sens"], _GRID_PARAMS["tpg_values"],
        _GRID_PARAMS["slg_values"],
    ))
    n_grid = len(grid_combos)
    fw_disabled = np.full(n_grid, 999999, dtype=np.int32)
    m1_disabled = np.full(n_grid, 999.0)
    m2_disabled = np.full(n_grid, 0.0)
    fi_disabled = np.full(n_grid, 0, dtype=np.int32)
    si_disabled = np.full(n_grid, 1, dtype=np.int32)

    stage1_results = []

    for w in _GRID_PARAMS["windows"]:
        n_bars = len(close)
        if w > n_bars - 5:
            continue

        indicators = compute_ma_switch_indicators(close, w, ma_windows=all_ma_windows)
        close_arr = close.values
        ma_all = np.array([indicators[f"MA{mw}"].values for mw in all_ma_windows])

        bgp_a = np.array([c[0] for c in grid_combos])
        vs_a = np.array([c[1] for c in grid_combos])
        ts_a = np.array([c[2] for c in grid_combos])
        tpg_a = np.array([c[3] for c in grid_combos])
        slg_a = np.array([c[4] for c in grid_combos])

        if use_gpu:
            entries_b, exits_b, sizes_b = generate_switch_signals_batch(
                close_arr,
                indicators["MADevPct"].values,
                indicators["MADevTrend"].values,
                indicators["RollingVolPct"].values,
                indicators["MABase"].values,
                ma_all, fi_disabled, si_disabled,
                bgp_a, vs_a, ts_a, tpg_a, slg_a,
                fw_disabled, m1_disabled, m2_disabled,
                position_size=0.5, position_sizing_coef=30.0,
            )
            bt = run_backtest_batch(close_arr, entries_b, exits_b, sizes_b,
                                    n_combos=n_grid)
            for idx, (bgp, vs, ts, tpg, slg) in enumerate(grid_combos):
                if int(bt[idx][4]) == 0:
                    continue
                stage1_results.append({
                    "ma_window": w,
                    "base_grid_pct": bgp, "volatility_scale": vs,
                    "trend_sensitivity": ts, "take_profit_grid": tpg,
                    "stop_loss_grid": slg,
                    "total_return": bt[idx][0],
                    "sharpe_ratio": bt[idx][1],
                })
        else:
            for idx, (bgp, vs, ts, tpg, slg) in enumerate(grid_combos):
                entries, exits, sizes = generate_switch_signals(
                    close_arr,
                    indicators["MADevPct"].values,
                    indicators["MADevTrend"].values,
                    indicators["RollingVolPct"].values,
                    indicators["MABase"].values,
                    indicators[f"MA{all_ma_windows[0]}"].values,
                    indicators[f"MA{all_ma_windows[1]}"].values,
                    base_grid_pct=bgp, volatility_scale=vs,
                    trend_sensitivity=ts, take_profit_grid=tpg,
                    stop_loss_grid=slg,
                    flat_wait_days=999999,
                    switch_deviation_m1=999.0, switch_deviation_m2=0.0,
                    position_size=0.5, position_sizing_coef=30.0,
                )
                if entries.sum() == 0:
                    continue
                m = run_backtest(close, entries, exits, sizes)
                stage1_results.append({
                    "ma_window": w,
                    "base_grid_pct": bgp, "volatility_scale": vs,
                    "trend_sensitivity": ts, "take_profit_grid": tpg,
                    "stop_loss_grid": slg,
                    "total_return": m["total_return"],
                    "sharpe_ratio": m["sharpe_ratio"],
                })

    if not stage1_results:
        return pd.DataFrame()

    df1 = pd.DataFrame(stage1_results)
    best_grid = df1.nlargest(1, "total_return").iloc[0]
    best_w = int(best_grid["ma_window"])
    best_bgp = best_grid["base_grid_pct"]
    best_vs = best_grid["volatility_scale"]
    best_ts = best_grid["trend_sensitivity"]
    best_tpg = best_grid["take_profit_grid"]
    best_slg = best_grid["stop_loss_grid"]

    print(f"  [Switch Stage 1] Best grid: w={best_w} bgp={best_bgp:.4f} "
          f"vs={best_vs:.1f} ts={best_ts:.0f} tpg={best_tpg:.2f} slg={best_slg:.1f} "
          f"ret={best_grid['total_return']:.1%}")

    # ── Stage 2: fix grid, optimize switch ──
    print("  [Switch Stage 2] Optimizing switch params (grid fixed)…")

    switch_combos = []
    for fw, sw_m1, sw_m2, sw_fast, sw_slow in product(
        _SWITCH_PARAMS["flat_wait_days_vals"],
        _SWITCH_PARAMS["switch_m1_vals"],
        _SWITCH_PARAMS["switch_m2_vals"],
        _SWITCH_PARAMS["switch_fast_vals"],
        _SWITCH_PARAMS["switch_slow_vals"],
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_combos.append((fw, sw_m1, sw_m2, sw_fast, sw_slow))

    n_switch = len(switch_combos)
    indicators = compute_ma_switch_indicators(close, best_w, ma_windows=all_ma_windows)
    close_arr = close.values
    ma_all = np.array([indicators[f"MA{mw}"].values for mw in all_ma_windows])

    bgp_a = np.full(n_switch, best_bgp)
    vs_a = np.full(n_switch, best_vs)
    ts_a = np.full(n_switch, best_ts)
    tpg_a = np.full(n_switch, best_tpg)
    slg_a = np.full(n_switch, best_slg)
    fw_a = np.array([c[0] for c in switch_combos], dtype=np.int32)
    m1_a = np.array([c[1] for c in switch_combos])
    m2_a = np.array([c[2] for c in switch_combos])
    fi_a = np.array([ma_to_idx[c[3]] for c in switch_combos], dtype=np.int32)
    si_a = np.array([ma_to_idx[c[4]] for c in switch_combos], dtype=np.int32)

    results = []

    if use_gpu:
        entries_b, exits_b, sizes_b = generate_switch_signals_batch(
            close_arr,
            indicators["MADevPct"].values,
            indicators["MADevTrend"].values,
            indicators["RollingVolPct"].values,
            indicators["MABase"].values,
            ma_all, fi_a, si_a,
            bgp_a, vs_a, ts_a, tpg_a, slg_a,
            fw_a, m1_a, m2_a,
            position_size=0.5, position_sizing_coef=30.0,
        )
        bt = run_backtest_batch(close_arr, entries_b, exits_b, sizes_b,
                                n_combos=n_switch)
        for idx, (fw, sw_m1, sw_m2, sw_fast, sw_slow) in enumerate(switch_combos):
            if int(bt[idx][4]) == 0:
                continue
            results.append({
                "total_return": bt[idx][0], "sharpe_ratio": bt[idx][1],
                "max_drawdown": bt[idx][2], "calmar_ratio": bt[idx][3],
                "num_trades": int(bt[idx][4]), "win_rate": bt[idx][5],
                "ma_window": best_w,
                "base_grid_pct": best_bgp, "volatility_scale": best_vs,
                "trend_sensitivity": best_ts, "take_profit_grid": best_tpg,
                "stop_loss_grid": best_slg,
                "flat_wait_days": fw, "switch_deviation_m1": sw_m1,
                "switch_deviation_m2": sw_m2,
                "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
            })
    else:
        for (fw, sw_m1, sw_m2, sw_fast, sw_slow) in switch_combos:
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
                position_size=0.5, position_sizing_coef=30.0,
            )
            if entries.sum() == 0:
                continue
            m = run_backtest(close, entries, exits, sizes)
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
