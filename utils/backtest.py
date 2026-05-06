"""回测执行模块 — CPU（VectorBT 单次）和批量（MLX GPU 向量化）。

所有回测默认使用「次日开盘价成交」模型：
  - 信号在 bar i 的收盘后产生（基于 bar i 的收盘数据计算指标）
  - 实际成交发生在 bar i+1 的开盘价（消除了前视偏差）

提供两种回测路径：
  - run_backtest:       使用 VectorBT 的 Portfolio.from_signals 做单次回测
  - run_backtest_batch: MLX GPU 向量化批量回测，所有 combo 同时处理
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


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
# 批量回测 — MLX GPU 向量化
#
# 将所有 combo 的 per-bar 操作向量化，在 Apple Silicon GPU 上运行。
# 成交价（fill_price）和估值价（close）分离：
#   - 入场/离场以 fill_price 成交（通常为次日开盘价）
#   - NAV 以 close 估值（每日收盘价）
#   - 末尾强制平仓
#
# 架构：外层 Python 逐 bar 循环（时序依赖不可消除），
# 内层每 bar 内所有 combo 在 MLX GPU 上并行处理。
# 每 bar 调用 mx.eval() 防止计算图膨胀。
# ══════════════════════════════════════════════════════════════════


def run_backtest_batch(
    close: np.ndarray,
    entries: np.ndarray,    # [n_combos, n_bars] bool (或 transposed=True 时 [n_bars, n_combos])
    exits: np.ndarray,      # [n_combos, n_bars] bool
    sizes: np.ndarray,      # [n_combos, n_bars] float64
    init_cash: float = 100_000.0,
    annual_factor: int = 365,
    n_combos: int = 0,
    open_: np.ndarray | None = None,
    transposed: bool = False,  # True: entries=[n_bars, n_combos]; False: [n_combos, n_bars]
) -> np.ndarray:
    """批量回测：MLX GPU 向量化处理所有 combo。

    成交模型（与 VectorBT 逻辑一致）：
      - 未传 open_: close 本身作为成交价
      - 传入 open_: fill_price = shift(open_, -1)，信号 bar i 在 bar i+1 Open 成交
      - NAV 以 close 估值，入场/离场以 fill_price 成交

    Args:
        close:  收盘价 [n_bars]，用于 NAV 估值
        entries/exits/sizes: 信号数组，默认 [n_combos, n_bars]
        init_cash / annual_factor: 初始资金 / 年化因子
        n_combos: 有效组合数
        open_:  可选，开盘价 [n_bars]，传入则启用 next-bar Open 成交
        transposed: True 表示 entries 布局为 [n_bars, n_combos]

    Returns:
        numpy 数组 [n_combos, 6]: total_return, sharpe, max_drawdown, calmar, num_trades, win_rate
    """
    from utils.gpu import get_mlx

    mx = get_mlx()
    if mx is None:
        # MLX 不可用时回退到 CPU（逐组合循环）
        return _run_backtest_batch_cpu(
            close, entries, exits, sizes, init_cash, annual_factor,
            n_combos, open_, transposed,
        )

    if transposed:
        n_bars = entries.shape[0]
        actual_combos = n_combos if n_combos > 0 else entries.shape[1]
        entries = entries.T.copy()  # → [n_combos, n_bars]
        exits = exits.T.copy()
        sizes = sizes.T.copy()
    else:
        n_bars = entries.shape[1]
        actual_combos = n_combos if n_combos > 0 else entries.shape[0]

    if n_bars == 0 or actual_combos == 0:
        return np.zeros((actual_combos, 6), dtype=np.float64)

    if open_ is not None:
        fill_price = np.roll(open_, -1).astype(np.float32)
        fill_price[-1] = float(close[-1])
    else:
        fill_price = close.astype(np.float32)

    close_f32 = close.astype(np.float32)

    # ── 转换信号数组为 MLX（bool → float32，MLX 不原生支持 bool） ──
    entries_mx = mx.array(entries[:actual_combos].astype(np.float32))
    exits_mx = mx.array(exits[:actual_combos].astype(np.float32))
    sizes_mx = mx.array(sizes[:actual_combos].astype(np.float32))

    # ── 每个 combo 的交易状态（MLX 数组，保留在 GPU） ──
    cash = mx.full(actual_combos, mx.array(float(init_cash), dtype=mx.float32))
    position = mx.zeros(actual_combos, dtype=mx.float32)
    in_position = mx.zeros(actual_combos, dtype=mx.float32)
    entry_cost = mx.zeros(actual_combos, dtype=mx.float32)

    # ── 统计累计量 ──
    peak_nav = mx.full(actual_combos, mx.array(float(init_cash), dtype=mx.float32))
    min_drawdown = mx.zeros(actual_combos, dtype=mx.float32)
    prev_nav = mx.full(actual_combos, mx.array(float(init_cash), dtype=mx.float32))
    sum_ret = mx.zeros(actual_combos, dtype=mx.float32)
    sum_ret2 = mx.zeros(actual_combos, dtype=mx.float32)

    trade_count = mx.zeros(actual_combos, dtype=mx.int32)
    win_count = mx.zeros(actual_combos, dtype=mx.int32)

    zero_f = mx.array(0.0, dtype=mx.float32)
    one_f = mx.array(1.0, dtype=mx.float32)
    one_i = mx.array(1, dtype=mx.int32)

    for i in range(n_bars):
        cl = mx.array(float(close_f32[i]), dtype=mx.float32)
        fp = mx.array(float(fill_price[i]), dtype=mx.float32)

        entry_sig = entries_mx[:, i]
        exit_sig = exits_mx[:, i]
        sz = sizes_mx[:, i]

        # 价格有效性
        cl_invalid = mx.isnan(cl) | (cl <= zero_f)
        fp_valid = ~mx.isnan(fp) & (fp > zero_f)

        # ── 入场逻辑 ──
        use_size = mx.where(mx.isinf(sz) | (sz > one_f), one_f, sz)
        use_size = mx.where(use_size <= zero_f, one_f, use_size)

        can_enter = (entry_sig > zero_f) & (in_position <= zero_f) & fp_valid
        can_enter = can_enter & ~cl_invalid

        buy_amount = cash * use_size
        shares = mx.where(can_enter, buy_amount / fp, zero_f)

        cash = mx.where(can_enter, cash - shares * fp, cash)
        position = mx.where(can_enter, shares, position)
        entry_cost = mx.where(can_enter, shares * fp, entry_cost)
        in_position = mx.where(can_enter, one_f, in_position)

        # ── 离场逻辑 ──
        was_in = in_position

        can_exit = (exit_sig > zero_f) & (was_in > zero_f) & fp_valid
        can_exit = can_exit & ~cl_invalid

        sell_amount = position * fp
        cash = mx.where(can_exit, cash + sell_amount, cash)

        win_trade = (sell_amount > entry_cost) & can_exit
        win_count = mx.where(win_trade, win_count + one_i, win_count)
        trade_count = mx.where(can_exit, trade_count + one_i, trade_count)

        position = mx.where(can_exit, zero_f, position)
        entry_cost = mx.where(can_exit, zero_f, entry_cost)
        in_position = mx.where(can_exit, zero_f, in_position)

        # ── NAV 与统计 ──
        nav = cash + position * cl

        valid_prev = prev_nav > zero_f
        ret = mx.where(valid_prev, (nav - prev_nav) / prev_nav, zero_f)
        sum_ret = sum_ret + ret
        sum_ret2 = sum_ret2 + ret * ret

        peak_nav = mx.maximum(peak_nav, nav)
        dd = mx.where(peak_nav > zero_f, (nav - peak_nav) / peak_nav, zero_f)
        min_drawdown = mx.minimum(min_drawdown, dd)

        prev_nav = nav

        # 每 bar 物化状态，防止计算图膨胀
        mx.eval(cash, position, in_position, entry_cost,
                peak_nav, min_drawdown, prev_nav,
                sum_ret, sum_ret2, trade_count, win_count)

    # ── 末尾强制平仓（以最后收盘价成交）──
    final_cl = mx.array(float(close_f32[-1]), dtype=mx.float32)
    still_in = in_position > zero_f

    sell_amount = position * final_cl
    cash = mx.where(still_in, cash + sell_amount, cash)
    win_trade = sell_amount > entry_cost
    win_count = mx.where(still_in & win_trade, win_count + one_i, win_count)
    trade_count = mx.where(still_in, trade_count + one_i, trade_count)
    position = mx.where(still_in, zero_f, position)

    mx.eval(cash, position, win_count, trade_count)

    # ── 计算最终指标 ──
    final_nav = cash + position * final_cl
    init_cash_f = mx.array(float(init_cash), dtype=mx.float32)
    total_return = (final_nav - init_cash_f) / init_cash_f
    max_dd = min_drawdown

    n_bars_f = mx.array(float(n_bars), dtype=mx.float32)
    mean_ret = sum_ret / n_bars_f
    var = sum_ret2 / n_bars_f - mean_ret * mean_ret
    sharpe = mx.zeros(actual_combos, dtype=mx.float32)
    valid_var = var > mx.array(1e-12, dtype=mx.float32)
    annual_f = mx.array(float(annual_factor), dtype=mx.float32)
    sharpe = mx.where(valid_var,
                      (mean_ret / mx.sqrt(var)) * mx.sqrt(annual_f),
                      zero_f)

    calmar = mx.zeros(actual_combos, dtype=mx.float32)
    valid_dd = max_dd < mx.array(-1e-9, dtype=mx.float32)
    calmar = mx.where(valid_dd, total_return / mx.abs(max_dd), zero_f)

    win_rate = mx.zeros(actual_combos, dtype=mx.float32)
    has_trades = trade_count > 0
    win_rate = mx.where(has_trades,
                        win_count.astype(mx.float32) / trade_count.astype(mx.float32),
                        zero_f)

    mx.eval(total_return, sharpe, max_dd, calmar, trade_count, win_rate)

    # ── 转回 numpy float64 ──
    metrics = np.column_stack([
        np.array(total_return).astype(np.float64),
        np.array(sharpe).astype(np.float64),
        np.array(max_dd).astype(np.float64),
        np.array(calmar).astype(np.float64),
        np.array(trade_count).astype(np.float64),
        np.array(win_rate).astype(np.float64),
    ])
    return metrics


def _run_backtest_batch_cpu(
    close: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    init_cash: float = 100_000.0,
    annual_factor: int = 365,
    n_combos: int = 0,
    open_: np.ndarray | None = None,
    transposed: bool = False,
) -> np.ndarray:
    """CPU 回退：逐组合循环（MLX 不可用时使用）。"""
    if transposed:
        n_bars = entries.shape[0]
        actual_combos = n_combos if n_combos > 0 else entries.shape[1]
        entries = entries.T.copy()
        exits = exits.T.copy()
        sizes = sizes.T.copy()
    else:
        n_bars = entries.shape[1]
        actual_combos = n_combos if n_combos > 0 else entries.shape[0]

    if n_bars == 0 or actual_combos == 0:
        return np.zeros((actual_combos, 6), dtype=np.float64)

    if open_ is not None:
        fill_price = np.roll(open_, -1).astype(np.float64)
        fill_price[-1] = float(close[-1])
    else:
        fill_price = close.astype(np.float64)

    close_f64 = close.astype(np.float64)
    metrics = np.zeros((actual_combos, 6), dtype=np.float64)

    for c in range(actual_combos):
        cash = float(init_cash)
        position = 0.0
        entry_cost = 0.0
        in_position = False

        peak_nav = float(init_cash)
        min_drawdown = 0.0
        prev_nav = float(init_cash)
        sum_ret = 0.0
        sum_ret2 = 0.0
        trade_count = 0
        win_count = 0

        for i in range(n_bars):
            cl = close_f64[i]
            fp = fill_price[i]
            if cl <= 0.0 or np.isnan(cl):
                continue

            entry_sig = entries[c, i]
            exit_sig = exits[c, i]
            sz = sizes[c, i]

            if entry_sig and not in_position and fp > 0.0 and not np.isnan(fp):
                use_size = sz
                if np.isinf(use_size) or use_size > 1.0:
                    use_size = 1.0
                if use_size <= 0.0:
                    use_size = 1.0
                buy_amount = cash * use_size
                shares = buy_amount / fp
                cash -= shares * fp
                position = shares
                entry_cost = shares * fp
                in_position = True

            if exit_sig and in_position and fp > 0.0 and not np.isnan(fp):
                sell_amount = position * fp
                cash += sell_amount
                if sell_amount > entry_cost:
                    win_count += 1
                trade_count += 1
                position = 0.0
                entry_cost = 0.0
                in_position = False

            nav = cash + position * cl
            if prev_nav > 0.0:
                ret = (nav - prev_nav) / prev_nav
                sum_ret += ret
                sum_ret2 += ret * ret
            peak_nav = max(peak_nav, nav)
            dd = (nav - peak_nav) / peak_nav
            min_drawdown = min(min_drawdown, dd)
            prev_nav = nav

        if in_position:
            final_cl = close_f64[-1]
            if final_cl > 0.0:
                sell_amount = position * final_cl
                cash += sell_amount
                if sell_amount > entry_cost:
                    win_count += 1
                trade_count += 1

        final_nav = cash
        total_return = (final_nav - init_cash) / init_cash
        max_dd = min_drawdown

        mean_ret = sum_ret / n_bars
        var = sum_ret2 / n_bars - mean_ret * mean_ret
        sharpe = 0.0
        if var > 1e-12:
            sharpe = mean_ret / np.sqrt(var) * np.sqrt(annual_factor)

        calmar = 0.0
        if max_dd < -1e-9:
            calmar = total_return / abs(max_dd)

        win_rate = 0.0
        if trade_count > 0:
            win_rate = win_count / trade_count

        metrics[c] = [total_return, sharpe, max_dd, calmar,
                       float(trade_count), win_rate]

    return metrics


def metrics_array_to_dicts(metrics: np.ndarray) -> list[dict]:
    """将 [n_combos, 6] 指标数组转换为 dict 列表。"""
    keys = ["total_return", "sharpe_ratio", "max_drawdown",
            "calmar_ratio", "num_trades", "win_rate"]
    return [dict(zip(keys, row)) for row in metrics]
