"""VectorBT backtest runner — CPU (single) and GPU (batched)."""

import numpy as np
import pandas as pd
import vectorbt as vbt

from .gpu import xp, gpu

# ══════════════════════════════════════════════════════════════════
# CPU backtest (VectorBT)
# ══════════════════════════════════════════════════════════════════


def run_backtest(close: pd.Series, entries: np.ndarray, exits: np.ndarray,
                 sizes: np.ndarray, init_cash: float = 100_000.0) -> dict:
    """Run a single VectorBT backtest and return key metrics."""
    pf = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(entries, index=close.index),
        exits=pd.Series(exits, index=close.index),
        size=pd.Series(sizes, index=close.index),
        size_type="percent",
        init_cash=init_cash,
        freq="D",
    )
    dd = pf.max_drawdown()
    return {
        "total_return": pf.total_return(),
        "sharpe_ratio": pf.sharpe_ratio(),
        "max_drawdown": dd,
        "calmar_ratio": pf.total_return() / abs(dd) if dd != 0 else float("nan"),
        "num_trades": pf.trades.count(),
        "win_rate": pf.trades.win_rate() if pf.trades.count() > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════
# GPU-batched backtest (CuPy RawKernel)
# ══════════════════════════════════════════════════════════════════

_backtest_kernel = None

_BACKTEST_KERNEL_CODE = r"""
extern "C" __global__ void backtest_kernel(
    const double* close,
    const bool* entries,
    const bool* exits,
    const double* sizes,
    double* metrics,
    int n_bars,
    int n_combos,
    double init_cash,
    int annual_factor
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    int offset = combo_idx * n_bars;

    double cash = init_cash;
    double position = 0.0;       // number of shares
    double entry_cost = 0.0;     // cost basis for current trade
    bool in_position = false;

    double peak_nav = init_cash;
    double min_drawdown = 0.0;   // most negative drawdown (e.g., -0.1 for -10%)
    double prev_nav = init_cash;
    double sum_ret = 0.0;
    double sum_ret2 = 0.0;

    int trade_count = 0;
    int win_count = 0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        if (cl <= 0.0 || isnan(cl)) continue;

        bool entry_sig = entries[offset + i];
        bool exit_sig = exits[offset + i];
        double sz = sizes[offset + i];

        // --- entry ---
        if (entry_sig && !in_position) {
            double use_size = sz;
            if (isinf(use_size) || use_size > 1.0) use_size = 1.0;
            if (use_size <= 0.0) use_size = 1.0;

            double buy_amount = cash * use_size;
            double shares = buy_amount / cl;
            cash -= shares * cl;
            position = shares;
            entry_cost = shares * cl;
            in_position = true;
        }

        // --- exit ---
        if (exit_sig && in_position) {
            double sell_amount = position * cl;
            cash += sell_amount;

            // P&L for this trade
            if (sell_amount > entry_cost) win_count++;
            trade_count++;

            position = 0.0;
            entry_cost = 0.0;
            in_position = false;
        }

        // --- daily NAV ---
        double nav = cash + position * cl;

        // daily return
        if (prev_nav > 0.0) {
            double ret = (nav - prev_nav) / prev_nav;
            sum_ret += ret;
            sum_ret2 += ret * ret;
        }

        // drawdown
        peak_nav = fmax(peak_nav, nav);
        double dd = (nav - peak_nav) / peak_nav;
        min_drawdown = fmin(min_drawdown, dd);

        prev_nav = nav;
    }

    // Force close at end (matching VectorBT behavior)
    if (in_position) {
        double final_cl = close[n_bars - 1];
        if (final_cl > 0.0) {
            double sell_amount = position * final_cl;
            cash += sell_amount;
            if (sell_amount > entry_cost) win_count++;
            trade_count++;
        }
    }

    double final_nav = cash + position * close[n_bars - 1];
    double total_return = (final_nav - init_cash) / init_cash;
    double max_drawdown = min_drawdown;  // negative value, e.g., -0.0875

    // Sharpe ratio
    double mean_ret = sum_ret / (double)n_bars;
    double var = sum_ret2 / (double)n_bars - mean_ret * mean_ret;
    double sharpe = 0.0;
    if (var > 1e-12) {
        sharpe = mean_ret / sqrt(var) * sqrt((double)annual_factor);
    }

    // Calmar ratio
    double calmar = 0.0;
    if (max_drawdown < -1e-9) {
        calmar = total_return / fabs(max_drawdown);
    }

    // Win rate
    double win_rate = 0.0;
    if (trade_count > 0) {
        win_rate = (double)win_count / (double)trade_count;
    }

    // Write metrics: [total_return, sharpe, max_drawdown, calmar, num_trades, win_rate]
    int m_offset = combo_idx * 6;
    metrics[m_offset + 0] = total_return;
    metrics[m_offset + 1] = sharpe;
    metrics[m_offset + 2] = max_drawdown;
    metrics[m_offset + 3] = calmar;
    metrics[m_offset + 4] = (double)trade_count;
    metrics[m_offset + 5] = win_rate;
}
"""


def _get_backtest_kernel():
    global _backtest_kernel
    if _backtest_kernel is None:
        cp = xp()
        _backtest_kernel = cp.RawKernel(
            _BACKTEST_KERNEL_CODE,
            "backtest_kernel",
        )
    return _backtest_kernel


def run_backtest_batch(
    close: np.ndarray,
    entries: np.ndarray,    # [n_combos, n_bars] bool
    exits: np.ndarray,      # [n_combos, n_bars] bool
    sizes: np.ndarray,      # [n_combos, n_bars] float64
    init_cash: float = 100_000.0,
    annual_factor: int = 365,  # Calendar days, matching VectorBT freq="D"
    n_combos: int = 0,
) -> np.ndarray:
    """GPU-batched backtest: one CUDA thread per parameter combo.

    Returns numpy array of shape [n_combos, 6] with columns:
      total_return, sharpe_ratio, max_drawdown, calmar_ratio, num_trades, win_rate
    """
    cp = xp()
    n_bars = entries.shape[1]
    actual_combos = n_combos if n_combos > 0 else entries.shape[0]

    # Pad to multiple of 256
    block_size = 256
    grid_size = (actual_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    # Pad entries/exits/sizes to padded x n_bars
    def _pad_2d(arr, fill=0.0):
        if arr.shape[0] == padded:
            return cp.asarray(arr.ravel(), dtype=arr.dtype)
        p = cp.zeros(padded * n_bars, dtype=arr.dtype if arr.dtype != np.dtype('bool') else cp.float64)
        flat = arr.ravel()
        p[:len(flat)] = cp.asarray(flat)
        if padded > arr.shape[0]:
            fill_arr = cp.full((padded - arr.shape[0]) * n_bars, fill)
            # fill extra rows
        return p

    # Simpler: just flatten after padding
    entries_flat = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_flat = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_flat = cp.zeros(padded * n_bars, dtype=cp.float64)

    entries_cp = cp.asarray(entries.ravel())
    exits_cp = cp.asarray(exits.ravel())
    sizes_cp = cp.asarray(sizes.ravel())

    n_flat = actual_combos * n_bars
    entries_flat[:n_flat] = entries_cp
    exits_flat[:n_flat] = exits_cp
    sizes_flat[:n_flat] = sizes_cp

    close_d = cp.asarray(close, dtype=cp.float64)
    metrics_d = cp.zeros(padded * 6, dtype=cp.float64)

    kernel = _get_backtest_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, entries_flat, exits_flat, sizes_flat, metrics_d,
            n_bars, actual_combos, init_cash, annual_factor,
        ),
    )

    metrics = cp.asnumpy(metrics_d).reshape(padded, 6)[:actual_combos]
    return metrics


def metrics_array_to_dicts(metrics: np.ndarray) -> list[dict]:
    """Convert [n_combos, 6] metrics array to list of dicts."""
    keys = ["total_return", "sharpe_ratio", "max_drawdown",
            "calmar_ratio", "num_trades", "win_rate"]
    return [dict(zip(keys, row)) for row in metrics]
