"""回测执行模块 — CPU（VectorBT 单次）和 GPU（CuPy 批量）。

所有回测默认使用「次日开盘价成交」模型：
  - 信号在 bar i 的收盘后产生（基于 bar i 的收盘数据计算指标）
  - 实际成交发生在 bar i+1 的开盘价（消除了前视偏差）
  - 与 backtesting.py 库的执行模型一致

提供两种回测路径：
  - run_backtest:      使用 VectorBT 的 Portfolio.from_signals 做单次回测
  - run_backtest_batch: 使用 CuPy RawKernel 做批量回测，每个 CUDA 线程处理一组信号
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from .gpu import xp, gpu


# ══════════════════════════════════════════════════════════════════
# CPU 回测 — 基于 VectorBT
# ══════════════════════════════════════════════════════════════════


def run_backtest(close: pd.Series, entries: np.ndarray, exits: np.ndarray,
                 sizes: np.ndarray, init_cash: float = 100_000.0,
                 open_: pd.Series | None = None) -> dict:
    """使用 VectorBT 执行单次回测并返回关键指标。

    执行模型：
      - 未传 open_: 信号 bar Close 成交（简单测试用）
      - 传入 open_: 信号 bar i 的指令在 bar i+1 的 Open 成交（消除前视偏差）

    Args:
        close:     收盘价序列（带 DatetimeIndex）
        entries:   入场 bool 数组
        exits:     离场 bool 数组
        sizes:     仓位比例数组（percent 模式）
        init_cash: 初始资金
        open_:     开盘价序列。传入则使用 next-bar Open 成交

    Returns:
        dict: total_return, sharpe_ratio, max_drawdown, calmar_ratio,
              num_trades, win_rate
    """
    idx = close.index
    if open_ is not None:
        # shift(-1) 将次日开盘价对齐到信号 bar
        fill_price = open_.shift(-1).reindex(idx)
    else:
        fill_price = close

    pf = vbt.Portfolio.from_signals(
        fill_price,
        entries=pd.Series(entries, index=idx),
        exits=pd.Series(exits, index=idx),
        size=pd.Series(sizes, index=idx),
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
# GPU 批量回测 — CuPy RawKernel
#
# 每条 CUDA 线程独立模拟一个参数组合的完整交易过程。
# 成交价（fill_price）和估值价（close）分离：
#   - 入场/离场以 fill_price 成交（通常为次日开盘价）
#   - NAV 以 close 估值（每日收盘价）
#   - 末尾强制平仓
#
# 指标计算与 VectorBT 保持一致。
# ══════════════════════════════════════════════════════════════════

_backtest_kernel = None

_BACKTEST_KERNEL_CODE = r"""
extern "C" __global__ void backtest_kernel(
    const double* __restrict__ close,
    const double* __restrict__ fill_price,
    const bool* __restrict__ entries,
    const bool* __restrict__ exits,
    const double* __restrict__ sizes,
    double* __restrict__ metrics,
    int n_bars,
    int n_combos,
    int n_padded,
    double init_cash,
    int annual_factor
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    double cash = init_cash;
    double position = 0.0;
    double entry_cost = 0.0;
    bool in_position = false;

    double peak_nav = init_cash;
    double min_drawdown = 0.0;
    double prev_nav = init_cash;
    double sum_ret = 0.0;
    double sum_ret2 = 0.0;

    int trade_count = 0;
    int win_count = 0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        double fp = fill_price[i];
        if (cl <= 0.0 || isnan(cl)) continue;

        bool entry_sig = entries[i * n_padded + combo_idx];
        bool exit_sig = exits[i * n_padded + combo_idx];
        double sz = sizes[i * n_padded + combo_idx];

        // ── 入场：以 fill_price（次日开盘价）成交 ──
        if (entry_sig && !in_position && fp > 0.0 && !isnan(fp)) {
            double use_size = sz;
            if (isinf(use_size) || use_size > 1.0) use_size = 1.0;
            if (use_size <= 0.0) use_size = 1.0;

            double buy_amount = cash * use_size;
            double shares = buy_amount / fp;
            cash -= shares * fp;
            position = shares;
            entry_cost = shares * fp;
            in_position = true;
        }

        // ── 离场：以 fill_price 成交 ──
        if (exit_sig && in_position && fp > 0.0 && !isnan(fp)) {
            double sell_amount = position * fp;
            cash += sell_amount;

            if (sell_amount > entry_cost) win_count++;
            trade_count++;

            position = 0.0;
            entry_cost = 0.0;
            in_position = false;
        }

        // ── 每日净值 (NAV) — 以收盘价估值 ──
        double nav = cash + position * cl;

        if (prev_nav > 0.0) {
            double ret = (nav - prev_nav) / prev_nav;
            sum_ret += ret;
            sum_ret2 += ret * ret;
        }

        peak_nav = fmax(peak_nav, nav);
        double dd = (nav - peak_nav) / peak_nav;
        min_drawdown = fmin(min_drawdown, dd);

        prev_nav = nav;
    }

    // ── 末尾强制平仓（以最后收盘价成交）──
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
    double max_drawdown = min_drawdown;

    double mean_ret = sum_ret / (double)n_bars;
    double var = sum_ret2 / (double)n_bars - mean_ret * mean_ret;
    double sharpe = 0.0;
    if (var > 1e-12) {
        sharpe = mean_ret / sqrt(var) * sqrt((double)annual_factor);
    }

    double calmar = 0.0;
    if (max_drawdown < -1e-9) {
        calmar = total_return / fabs(max_drawdown);
    }

    double win_rate = 0.0;
    if (trade_count > 0) {
        win_rate = (double)win_count / (double)trade_count;
    }

    // 写入指标：[total_return, sharpe, max_drawdown, calmar, num_trades, win_rate]
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
    """延迟编译 CUDA kernel。"""
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
    entries: np.ndarray,    # [n_bars, n_combos] if transposed, else [n_combos, n_bars]
    exits: np.ndarray,
    sizes: np.ndarray,
    init_cash: float = 100_000.0,
    annual_factor: int = 365,
    n_combos: int = 0,
    open_: np.ndarray | None = None,
    transposed: bool = False,  # True: [n_bars, n_combos]; False: legacy [n_combos, n_bars]
) -> np.ndarray:
    """GPU 批量回测：每 CUDA 线程处理一组信号的完整回测。

    Args:
        close:  收盘价 [n_bars]，用于 NAV 估值
        entries/exits/sizes: 信号数组，布局由 transposed 参数决定
        transposed: True=[n_bars, n_combos]（合并写入）, False=[n_combos, n_bars]（兼容旧代码）
    Returns:
        numpy 数组 [n_combos, 6]
    """
    cp = xp()
    if transposed:
        n_bars = entries.shape[0]
        actual_combos = n_combos if n_combos > 0 else entries.shape[1]
    else:
        n_bars = entries.shape[1]
        actual_combos = n_combos if n_combos > 0 else entries.shape[0]
    if n_bars == 0 or actual_combos == 0:
        return np.zeros((actual_combos, 6))

    # 成交价
    if open_ is not None:
        fill_price = np.roll(open_, -1)
        fill_price[-1] = close[-1]
    else:
        fill_price = close.copy()

    block_size = 256
    grid_size = (actual_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    if transposed:
        entries_flat = cp.zeros((n_bars, padded), dtype=cp.bool_)
        exits_flat = cp.zeros((n_bars, padded), dtype=cp.bool_)
        sizes_flat = cp.zeros((n_bars, padded), dtype=cp.float64)
        entries_flat[:, :actual_combos] = cp.asarray(entries)
        exits_flat[:, :actual_combos] = cp.asarray(exits)
        sizes_flat[:, :actual_combos] = cp.asarray(sizes)
    else:
        entries_flat = cp.zeros(padded * n_bars, dtype=cp.bool_)
        exits_flat = cp.zeros(padded * n_bars, dtype=cp.bool_)
        sizes_flat = cp.zeros(padded * n_bars, dtype=cp.float64)
        n_flat = actual_combos * n_bars
        entries_flat[:n_flat] = cp.asarray(entries.ravel())
        exits_flat[:n_flat] = cp.asarray(exits.ravel())
        sizes_flat[:n_flat] = cp.asarray(sizes.ravel())

    close_d = cp.asarray(close, dtype=cp.float64)
    fill_d = cp.asarray(fill_price, dtype=cp.float64)
    metrics_d = cp.zeros(padded * 6, dtype=cp.float64)

    kernel = _get_backtest_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, fill_d,
            entries_flat.ravel() if transposed else entries_flat,
            exits_flat.ravel() if transposed else exits_flat,
            sizes_flat.ravel() if transposed else sizes_flat,
            metrics_d,
            n_bars, actual_combos, padded, init_cash, annual_factor,
        ),
    )

    metrics = cp.asnumpy(metrics_d).reshape(padded, 6)[:actual_combos]
    return metrics


def metrics_array_to_dicts(metrics: np.ndarray) -> list[dict]:
    """将 [n_combos, 6] 指标数组转换为 dict 列表。"""
    keys = ["total_return", "sharpe_ratio", "max_drawdown",
            "calmar_ratio", "num_trades", "win_rate"]
    return [dict(zip(keys, row)) for row in metrics]
