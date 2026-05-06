"""
Polyfit-Switch-v6 + 日内限价单增强策略。

策略 1 (Grid T+0): 已废弃——趋势日限价卖单截断利润，净收益为负。
策略 2 (Grid 入场限价买单): 连跌≥3天时，入场日挂 open×0.99 限价买单，
                          未成交则收盘买入。独立计算节省的成本。

日内收益独立于 Switch 基准收益计算。
总收益 = (1 + Switch基准收益) × (1 + 日内累计收益) - 1
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_intraday_pnl(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    sell_offset: float = 0.002,
    buy_offset: float = 0.002,
    enable_t0: bool = True,
) -> dict:
    """计算每日日内限价单的独立 P&L（不改变 Switch 基准收益）。

    信号在 bar i 产生，执行在 bar i+1 开盘。

    Returns:
        daily_intraday_pnl: 每天日内贡献（比例，相对当日开盘价）
        daily_type: 每天的操作类型 ('entry', 'exit', 't0', 'none')
        daily_fill_price: 每天的实际成交价（nan=无交易）
        stats: 统计汇总
    """
    n = len(close_arr)
    daily_pnl = np.zeros(n)
    daily_type = np.full(n, "none", dtype=object)
    daily_fill = np.full(n, np.nan)

    in_position = False
    entry_bar = -1

    n_buy_total = 0; n_buy_filled = 0
    n_sell_total = 0; n_sell_filled = 0
    n_t0_total = 0; n_t0_filled = 0
    total_buy_savings = 0.0
    total_sell_extra = 0.0
    total_t0_profit = 0.0

    for i in range(n):
        # 信号在 bar i，执行在 bar i+1
        exec_i = min(i + 1, n - 1)
        op = open_arr[exec_i]
        hi = high_arr[exec_i]
        lo = low_arr[exec_i]
        cl = close_arr[exec_i]

        if np.isnan(op) or op <= 0 or np.isnan(cl):
            continue

        # ── 入场日：节约的成本 = (open - fill_price) / open ──
        if entries[i]:
            n_buy_total += 1
            sz = float(sizes[i])
            limit_buy = op * (1.0 - buy_offset)
            if lo <= limit_buy:
                fp = limit_buy
                n_buy_filled += 1
            else:
                fp = cl  # 尾盘补齐：收盘价买入

            savings = (op - fp) / op * sz  # 正数=省了钱
            daily_pnl[i] = savings
            daily_type[i] = "entry"
            daily_fill[i] = fp
            total_buy_savings += savings
            in_position = True
            entry_bar = i

        # ── 离场日：额外收益 = (fill_price - open) / open ──
        if exits[i]:
            n_sell_total += 1
            limit_sell = op * (1.0 + sell_offset)
            if hi >= limit_sell:
                fp = limit_sell
                n_sell_filled += 1
            else:
                fp = cl

            extra = (fp - op) / op  # 正数=多赚了
            daily_pnl[i] = extra
            daily_type[i] = "exit"
            daily_fill[i] = fp
            total_sell_extra += extra
            in_position = False
            entry_bar = -1

        # ── 持仓日 T+0：日内波段 = (limit_sell - close) / open ──
        if enable_t0 and in_position and not exits[i] and i > entry_bar:
            n_t0_total += 1
            limit_sell = op * (1.0 + sell_offset)
            if hi >= limit_sell:
                t0_profit = (limit_sell - cl) / op
                daily_pnl[i] = t0_profit
                daily_type[i] = "t0"
                daily_fill[i] = limit_sell
                n_t0_filled += 1
                total_t0_profit += t0_profit

    stats = {
        "buy_fill_rate": n_buy_filled / max(n_buy_total, 1),
        "sell_fill_rate": n_sell_filled / max(n_sell_total, 1),
        "t0_trigger_rate": n_t0_filled / max(n_t0_total, 1),
        "n_buy_total": n_buy_total, "n_buy_filled": n_buy_filled,
        "n_sell_total": n_sell_total, "n_sell_filled": n_sell_filled,
        "n_t0_total": n_t0_total, "n_t0_filled": n_t0_filled,
        "total_buy_savings": total_buy_savings,
        "total_sell_extra": total_sell_extra,
        "total_t0_profit": total_t0_profit,
    }

    return {
        "daily_pnl": daily_pnl,
        "daily_type": daily_type,
        "daily_fill": daily_fill,
        "stats": stats,
    }


def run_intraday_backtest(
    close: pd.Series,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    sell_offset: float = 0.002,
    buy_offset: float = 0.002,
    enable_t0: bool = True,
) -> dict:
    """执行日内限价单增强回测。

    日内收益独立累加，不改变 Switch 基准 NAV。
    总收益 = (1 + 基准) × (1 + 日内累计) - 1

    Returns:
        baseline: 基准回测指标（开盘价成交）
        intraday_cumulative: 日内累计收益序列
        total_return: 总收益 = 基准 + 日内增量
        daily_detail: 逐日明细 DataFrame
        stats: 日内统计
    """
    from utils.backtest import run_backtest

    idx = close.index
    n = len(close)

    # ── 基准回测（开盘价成交，Switch 基准）──
    m_baseline = run_backtest(close, entries, exits, sizes, open_=open_)

    # ── 日内独立 P&L ──
    result = compute_intraday_pnl(
        open_.values, high.values, low.values, close.values,
        entries, exits, sizes,
        sell_offset=sell_offset, buy_offset=buy_offset,
        enable_t0=enable_t0,
    )

    # 日内累计收益序列（从 1.0 开始复利）
    intraday_cum = np.ones(n)
    cum = 1.0
    for i in range(n):
        cum *= (1.0 + result["daily_pnl"][i])
        intraday_cum[i] = cum

    intraday_total_return = cum - 1.0
    baseline_total_return = m_baseline["total_return"]

    # 总收益 = (1+基准) × (1+日内) - 1
    total_return = (1.0 + baseline_total_return) * (1.0 + intraday_total_return) - 1.0

    # 逐日明细
    detail_rows = []
    for i in range(n):
        if result["daily_type"][i] != "none":
            detail_rows.append({
                "date": idx[min(i + 1, n - 1)],  # 执行日期
                "signal_bar": idx[i],              # 信号日期
                "type": result["daily_type"][i],
                "intraday_pnl": result["daily_pnl"][i],
                "fill_price": result["daily_fill"][i],
                "open_price": open_.values[min(i + 1, n - 1)],
            })

    daily_detail = pd.DataFrame(detail_rows)

    intraday_series = pd.Series(intraday_cum, index=idx)

    return {
        "baseline": m_baseline,
        "intraday_total_return": intraday_total_return,
        "total_return": total_return,
        "excess_return": total_return - baseline_total_return,
        "intraday_cumulative": intraday_series,
        "intraday_stats": result["stats"],
        "daily_detail": daily_detail,
    }


# ══════════════════════════════════════════════════════════════
# Grid 入场条件限价买单：连跌≥3天 → 限价 open×0.99
# ══════════════════════════════════════════════════════════════

def compute_grid_entry_limit_buy(
    close_arr: np.ndarray,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    grid_entries: np.ndarray,
    grid_sizes: np.ndarray,
    cons_down_threshold: int = 3,
    cons_down_skip: int = 5,        # 连跌>=此天数 → 跳过限价，开盘直接买
    buy_offset: float = 0.01,
) -> dict:
    """Grid 入场日的条件限价买单。

    条件：入场信号前连跌 ∈ [cons_down_threshold, cons_down_skip)。
    连跌 >= cons_down_skip → V反概率大 → 跳过限价，开盘直接买（无日内损益）。

    执行：信号在 bar i，执行在 bar i+1 开盘。限价 = open × (1 - buy_offset)。
    未成交 → 收盘价买入。

    日内节省 = (open - fill_price) / open × size（相对开盘的节省比例）
    """
    n = len(close_arr)
    daily_pnl = np.zeros(n)
    daily_type = np.full(n, "none", dtype=object)
    daily_fill = np.full(n, np.nan)

    n_triggered = 0
    n_skipped = 0       # 因连跌过多被跳过
    n_filled = 0
    total_savings = 0.0

    for i in range(n):
        if not grid_entries[i]:
            continue

        cons_down = 0
        for j in range(i, max(0, i - 20), -1):
            if j > 0 and close_arr[j] < close_arr[j - 1]:
                cons_down += 1
            else:
                break

        if cons_down < cons_down_threshold:
            continue  # 不满足最低条件，正常开盘买入

        # 连跌过多 → V反概率大 → 跳过限价
        if cons_down >= cons_down_skip:
            n_skipped += 1
            # 开盘买，无日内损益（daily_pnl=0, daily_type='none'）
            continue

        n_triggered += 1

        exec_i = min(i + 1, n - 1)
        op = open_arr[exec_i]
        lo = low_arr[exec_i]
        cl = close_arr[exec_i]
        sz = float(grid_sizes[i])

        if np.isnan(op) or op <= 0:
            continue

        limit_price = op * (1.0 - buy_offset)

        if lo <= limit_price:
            fp = limit_price
            n_filled += 1
        else:
            fp = cl

        savings = (op - fp) / op * sz
        daily_pnl[i] = savings
        daily_type[i] = "grid_limit_buy"
        daily_fill[i] = fp
        total_savings += savings

    stats = {
        "n_triggered": n_triggered,
        "n_skipped": n_skipped,
        "n_filled": n_filled,
        "fill_rate": n_filled / max(n_triggered, 1),
        "total_savings": total_savings,
        "cons_down_threshold": cons_down_threshold,
        "cons_down_skip": cons_down_skip,
        "buy_offset": buy_offset,
    }

    return {
        "daily_pnl": daily_pnl,
        "daily_type": daily_type,
        "daily_fill": daily_fill,
        "stats": stats,
    }


def run_grid_entry_limit_backtest(
    close: pd.Series,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    grid_entries: np.ndarray,
    grid_sizes: np.ndarray,
    cons_down_threshold: int = 3,
    cons_down_skip: int = 5,
    buy_offset: float = 0.01,
) -> dict:
    """Grid 入场条件限价买单的完整回测。

    日内收益独立于基准，总收益 = (1+基准) × (1+日内) - 1
    """
    from utils.backtest import run_backtest

    idx = close.index
    n = len(close)

    # ── 基准回测 ──
    m_baseline = run_backtest(close, entries, exits, sizes, open_=open_)

    # ── 条件限价买单 ──
    result = compute_grid_entry_limit_buy(
        close.values, open_.values, high.values, low.values,
        grid_entries, grid_sizes,
        cons_down_threshold=cons_down_threshold,
        cons_down_skip=cons_down_skip,
        buy_offset=buy_offset,
    )

    # 日内累计收益
    intraday_cum = np.ones(n)
    cum = 1.0
    for i in range(n):
        if result["daily_pnl"][i] != 0:
            cum *= (1.0 + result["daily_pnl"][i])
        intraday_cum[i] = cum

    intraday_total_return = cum - 1.0
    baseline_total_return = m_baseline["total_return"]
    total_return = (1.0 + baseline_total_return) * (1.0 + intraday_total_return) - 1.0

    # 逐日明细
    detail_rows = []
    for i in range(n):
        if result["daily_type"][i] != "none":
            exec_i = min(i + 1, n - 1)
            detail_rows.append({
                "date": idx[exec_i],
                "signal_bar": idx[i],
                "type": result["daily_type"][i],
                "intraday_pnl": result["daily_pnl"][i],
                "fill_price": result["daily_fill"][i],
                "open_price": open_.values[exec_i],
            })

    daily_detail = pd.DataFrame(detail_rows)
    intraday_series = pd.Series(intraday_cum, index=idx)

    return {
        "baseline": m_baseline,
        "intraday_total_return": intraday_total_return,
        "total_return": total_return,
        "excess_return": total_return - baseline_total_return,
        "intraday_cumulative": intraday_series,
        "intraday_stats": result["stats"],
        "daily_detail": daily_detail,
    }
# ══════════════════════════════════════════════════════════════
# 参数扫描
# ══════════════════════════════════════════════════════════════

def scan_intraday_params(
    close: pd.Series,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    sell_offsets: list[float] | None = None,
    buy_offsets: list[float] | None = None,
    enable_t0_options: list[bool] | None = None,
) -> pd.DataFrame:
    """扫描日内限价单参数。"""
    if sell_offsets is None:
        sell_offsets = [0.001, 0.002, 0.003, 0.005]
    if buy_offsets is None:
        buy_offsets = [0.001, 0.002, 0.003, 0.005]
    if enable_t0_options is None:
        enable_t0_options = [True, False]

    results = []
    for so in sell_offsets:
        for bo in buy_offsets:
            for t0 in enable_t0_options:
                r = run_intraday_backtest(
                    close, open_, high, low, entries, exits, sizes,
                    sell_offset=so, buy_offset=bo, enable_t0=t0,
                )
                results.append({
                    "sell_offset": so, "buy_offset": bo, "enable_t0": t0,
                    "total_return": r["total_return"],
                    "baseline_return": r["baseline"]["total_return"],
                    "intraday_return": r["intraday_total_return"],
                    "excess_return": r["excess_return"],
                    "buy_fill_rate": r["intraday_stats"]["buy_fill_rate"],
                    "sell_fill_rate": r["intraday_stats"]["sell_fill_rate"],
                    "t0_trigger_rate": r["intraday_stats"]["t0_trigger_rate"],
                    "n_buy_total": r["intraday_stats"]["n_buy_total"],
                    "n_t0_total": r["intraday_stats"]["n_t0_total"],
                    "total_buy_savings": r["intraday_stats"]["total_buy_savings"],
                    "total_sell_extra": r["intraday_stats"]["total_sell_extra"],
                    "total_t0_profit": r["intraday_stats"]["total_t0_profit"],
                })

    return pd.DataFrame(results)
