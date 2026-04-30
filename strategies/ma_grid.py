"""Moving Average Dynamic Grid Strategy.

Grid mean-reversion using SMA as baseline.
GPU-batched signal generation via CuPy RawKernel.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from utils.gpu import xp
from utils.indicators import compute_ma_indicators
from utils.scan import indicator_and_scan


# ══════════════════════════════════════════════════════════════════
# CPU signal generation
# ══════════════════════════════════════════════════════════════════

def generate_grid_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate entry/exit signals bar by bar for grid mean-reversion strategy."""
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
# GPU-batched signal generation (CuPy RawKernel)
# ══════════════════════════════════════════════════════════════════

_generate_grid_signals_kernel = None

_GRID_KERNEL_CODE = r"""
extern "C" __global__ void generate_grid_signals_kernel(
    const double* dev_pct,
    const double* dev_trend,
    const double* vol_pct,
    const double* base_grid_pct,
    const double* volatility_scale,
    const double* trend_sensitivity,
    const double* take_profit_grid,
    const double* stop_loss_grid,
    bool* entries,
    bool* exits,
    double* sizes,
    int n_bars,
    int n_combos,
    int max_grid_levels,
    int max_holding_days,
    int cooldown_days,
    double min_signal_strength,
    double pos_size
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];

    bool in_position = false;
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    int cooldown = 0;

    for (int i = 0; i < n_bars; i++) {
        double dp = dev_pct[i];
        double dt = dev_trend[i];
        double vp = vol_pct[i];

        if (isnan(dp) || isnan(dt) || isnan(vp)) {
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        if (!in_position) {
            if (cooldown > 0) {
                cooldown--;
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            double signal_strength = fabs(dp) / fmax(dgs, 1e-9);
            int entry_lvl = (int)floor(signal_strength);
            entry_lvl = entry_lvl < 1 ? 1 : (entry_lvl > max_grid_levels ? max_grid_levels : entry_lvl);
            double entry_threshold = -(double)entry_lvl * dgs;
            if (dp <= entry_threshold && signal_strength >= min_signal_strength) {
                entries[combo_idx * n_bars + i] = true;
                sizes[combo_idx * n_bars + i] = pos_size;
                in_position = true;
                entry_bar = i;
                entry_level = entry_lvl;
                entry_grid_step = dgs;
            } else {
                entries[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
            }
            exits[combo_idx * n_bars + i] = false;
        } else {
            entries[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
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
            } else {
                exits[combo_idx * n_bars + i] = false;
            }
        }
    }

    if (in_position) {
        exits[combo_idx * n_bars + n_bars - 1] = true;
    }
}
"""


def _get_grid_kernel():
    global _generate_grid_signals_kernel
    if _generate_grid_signals_kernel is None:
        cp = xp()
        _generate_grid_signals_kernel = cp.RawKernel(
            _GRID_KERNEL_CODE, "generate_grid_signals_kernel")
    return _generate_grid_signals_kernel


def generate_grid_signals_batch(
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    base_grid_pcts: np.ndarray,
    volatility_scales: np.ndarray,
    trend_sensitivities: np.ndarray,
    take_profit_grids: np.ndarray,
    stop_loss_grids: np.ndarray,
    max_grid_levels: int = 3,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-batched signal generation: one CUDA thread per parameter combo.

    Returns (entries, exits, sizes) as numpy arrays of shape [n_combos, n_bars].
    """
    cp = xp()
    n_bars = len(dev_pct)
    n_combos = len(base_grid_pcts)

    block_size = 256
    grid_size = (n_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    def _pad(a):
        result = cp.zeros(padded, dtype=cp.float64)
        result[:n_combos] = cp.asarray(a, dtype=cp.float64)
        if padded > n_combos:
            result[n_combos:] = cp.nan
        return result

    bgp_d = _pad(base_grid_pcts)
    vs_d = _pad(volatility_scales)
    ts_d = _pad(trend_sensitivities)
    tpg_d = _pad(take_profit_grids)
    slg_d = _pad(stop_loss_grids)

    dev_pct_d = cp.asarray(dev_pct, dtype=cp.float64)
    dev_trend_d = cp.asarray(dev_trend, dtype=cp.float64)
    vol_d = cp.asarray(rolling_vol_pct, dtype=cp.float64)

    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    pos_size = float(position_size)

    kernel = _get_grid_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            dev_pct_d, dev_trend_d, vol_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d,
            entries_d, exits_d, sizes_d,
            n_bars, n_combos, max_grid_levels, max_holding_days, cooldown_days,
            min_signal_strength, pos_size,
        ),
    )

    entries = cp.asnumpy(entries_d).reshape(padded, n_bars)[:n_combos]
    exits = cp.asnumpy(exits_d).reshape(padded, n_bars)[:n_combos]
    sizes = cp.asnumpy(sizes_d).reshape(padded, n_bars)[:n_combos]
    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# Parameter scan
# ══════════════════════════════════════════════════════════════════

def scan_ma_strategy(close: pd.Series) -> pd.DataFrame:
    print("  MA parameter scan…")
    return indicator_and_scan(
        close, "MA", compute_ma_indicators, "ma_window",
        windows=[20, 50, 75, 100, 150, 200],
        grid_pcts=[0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02],
        vol_scales=[0.5, 1.0, 1.5, 2.0, 3.0],
        trend_sens=[3.0, 6.0, 9.0, 12.0, 15.0],
        tpg_values=[0.5, 0.7, 0.85, 1.0],
        slg_values=[1.0, 1.3, 1.6, 2.0, 2.5],
        signal_fn=generate_grid_signals,
        signal_batch_fn=generate_grid_signals_batch,
    )
