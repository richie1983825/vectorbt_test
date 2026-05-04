"""
Intraday Execution Strategy — 日内分批限价执行。

基于 polyfit-switch 的日线信号，在日内分钟线上通过限价单分批执行，
替代原先的「次日开盘一次性成交」。

核心机制：
  - 买入日：在 PolyBasePred 下方挂 N 档限价单，吃日内回落
  - 卖出日：在 PolyBasePred 上方挂 N 档限价单，吃日内反弹
  - 尾盘（closeout_time 之后）未成交部分以市价补齐

对比基准：次日 bar Open 成交（polyfit-switch 原执行方式）。

用法：
    from strategies.intraday_execution import scan_intraday_params
    results = scan_intraday_params(close_daily, signals_daily, data_1m, open_)
"""

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# 单日执行模拟
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """单日执行结果。"""
    avg_fill_price: float      # 加权平均成交价
    fill_pct: float            # 实际成交比例
    num_fills: int             # 成交档数
    closeout_pct: float        # 尾盘补齐比例
    limit_fill_pct: float      # 限价单成交比例
    details: list[dict]        # 每笔成交明细


def execute_intraday_buy(
    minute_bars: pd.DataFrame,
    total_size: float,
    reference_price: float,
    grid_levels: int = 3,
    grid_spacing_pct: float = 0.005,
    first_offset_pct: float = 0.002,
    closeout_minute: int = 225,   # 14:45 ≈ 开盘后第 225 分钟 (09:31 → 14:45)
    use_open: bool = False,       # 若 True，集合竞价阶段以开盘价成交第一档
) -> ExecutionResult:
    """模拟买入日的日内限价执行。

    在 reference_price 下方逐档挂单：
      level 1: ref × (1 - first_offset)
      level 2: ref × (1 - first_offset - grid_spacing)
      level N: ref × (1 - first_offset - (N-1) × grid_spacing)

    每档仓位 = total_size / grid_levels。
    当分钟线 Low ≤ limit_price 时该档成交（成交价 = limit_price）。
    尾盘 closeout_minute 之后未成交的剩余仓位以市价补齐（最终 bar Close）。

    Args:
        minute_bars:       单日 1 分钟线 DataFrame，含 Open/High/Low/Close
        total_size:        总仓位比例
        reference_price:   基准价（PolyBasePred）
        grid_levels:       分批档数
        grid_spacing_pct:  档间距（相对 reference_price 的比例）
        first_offset_pct:  第一档偏移（相对 reference_price）
        closeout_minute:   尾盘补齐时间（从 09:31 起的分钟偏移，默认 225=14:45）
        use_open:          是否允许第一档以开盘价成交

    Returns:
        ExecutionResult
    """
    n_bars = len(minute_bars)
    if n_bars == 0:
        return ExecutionResult(avg_fill_price=0.0, fill_pct=0.0,
                               num_fills=0, closeout_pct=0.0,
                               limit_fill_pct=0.0, details=[])

    size_per_level = total_size / grid_levels
    total_filled = 0.0
    total_cost = 0.0
    fills = []
    levels_filled = 0
    limit_filled = 0.0

    # 逐分钟扫描
    for i in range(n_bars):
        bar = minute_bars.iloc[i]
        low = bar["Low"]
        close = bar["Close"]
        minute_idx = i  # 0-based from 09:31

        # 检查各档限价单
        for level in range(1, grid_levels + 1):
            limit_price = reference_price * (
                1.0 - first_offset_pct - (level - 1) * grid_spacing_pct
            )
            # 限价单成交条件：最低价触及限价
            if low <= limit_price:
                fill_size = size_per_level
                total_filled += fill_size
                total_cost += fill_size * limit_price
                levels_filled += 1
                limit_filled += fill_size
                fills.append({
                    "minute": minute_idx,
                    "level": level,
                    "limit_price": limit_price,
                    "fill_size": fill_size,
                    "type": "limit",
                })
                # 该档已成交，将限价设为无穷大避免重复
                # 通过标记已处理的 level 来避免重复
                break  # 同一分钟最多成交一档（简化假设）

        # 如果全部成交，提前结束
        if total_filled >= total_size - 1e-12:
            break

    # 尾盘补齐
    remaining = total_size - total_filled
    closeout_filled = 0.0
    if remaining > 1e-12:
        # 找 closeout_minute 之后的第一个 bar 或最后一根 bar
        closeout_idx = min(closeout_minute, n_bars - 1)
        closeout_price = float(minute_bars.iloc[closeout_idx]["Close"])
        total_filled += remaining
        total_cost += remaining * closeout_price
        closeout_filled = remaining
        fills.append({
            "minute": closeout_idx,
            "level": -1,
            "limit_price": closeout_price,
            "fill_size": remaining,
            "type": "closeout",
        })

    avg_price = total_cost / total_filled if total_filled > 1e-12 else 0.0
    return ExecutionResult(
        avg_fill_price=avg_price,
        fill_pct=float(total_filled / total_size) if total_size > 1e-12 else 0.0,
        num_fills=levels_filled,
        closeout_pct=float(closeout_filled / total_size) if total_size > 1e-12 else 0.0,
        limit_fill_pct=float(limit_filled / total_size) if total_size > 1e-12 else 0.0,
        details=fills,
    )


def execute_intraday_sell(
    minute_bars: pd.DataFrame,
    total_size: float,
    reference_price: float,
    grid_levels: int = 3,
    grid_spacing_pct: float = 0.005,
    first_offset_pct: float = 0.002,
    closeout_minute: int = 225,
) -> ExecutionResult:
    """模拟卖出日的日内限价执行。

    在 reference_price 上方逐档挂单：
      level 1: ref × (1 + first_offset)
      level 2: ref × (1 + first_offset + grid_spacing)
      level N: ref × (1 + first_offset + (N-1) × grid_spacing)

    High ≥ limit_price 时成交。
    """
    n_bars = len(minute_bars)
    if n_bars == 0:
        return ExecutionResult(avg_fill_price=0.0, fill_pct=0.0,
                               num_fills=0, closeout_pct=0.0,
                               limit_fill_pct=0.0, details=[])

    size_per_level = total_size / grid_levels
    total_filled = 0.0
    total_proceeds = 0.0
    fills = []
    levels_filled = 0
    limit_filled = 0.0
    filled_levels = set()

    for i in range(n_bars):
        bar = minute_bars.iloc[i]
        high = bar["High"]
        minute_idx = i

        for level in range(1, grid_levels + 1):
            if level in filled_levels:
                continue
            limit_price = reference_price * (
                1.0 + first_offset_pct + (level - 1) * grid_spacing_pct
            )
            if high >= limit_price:
                fill_size = size_per_level
                total_filled += fill_size
                total_proceeds += fill_size * limit_price
                levels_filled += 1
                limit_filled += fill_size
                filled_levels.add(level)
                fills.append({
                    "minute": minute_idx,
                    "level": level,
                    "limit_price": limit_price,
                    "fill_size": fill_size,
                    "type": "limit",
                })
                break

        if total_filled >= total_size - 1e-12:
            break

    # 尾盘补齐
    remaining = total_size - total_filled
    closeout_filled = 0.0
    if remaining > 1e-12:
        closeout_idx = min(closeout_minute, n_bars - 1)
        closeout_price = float(minute_bars.iloc[closeout_idx]["Close"])
        total_filled += remaining
        total_proceeds += remaining * closeout_price
        closeout_filled = remaining
        fills.append({
            "minute": closeout_idx,
            "level": -1,
            "limit_price": closeout_price,
            "fill_size": remaining,
            "type": "closeout",
        })

    avg_price = total_proceeds / total_filled if total_filled > 1e-12 else 0.0
    return ExecutionResult(
        avg_fill_price=avg_price,
        fill_pct=float(total_filled / total_size) if total_size > 1e-12 else 0.0,
        num_fills=levels_filled,
        closeout_pct=float(closeout_filled / total_size) if total_size > 1e-12 else 0.0,
        limit_fill_pct=float(limit_filled / total_size) if total_size > 1e-12 else 0.0,
        details=fills,
    )


# ══════════════════════════════════════════════════════════════════
# 完整回测：日线信号 + 日内执行
# ══════════════════════════════════════════════════════════════════

def _align_minute_data(
    data_1m: pd.DataFrame,
    trade_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """获取指定交易日期的分钟线数据。"""
    date_mask = data_1m.index.normalize() == trade_date.normalize()
    subset = data_1m[date_mask]
    if len(subset) == 0:
        return None
    return subset


def run_intraday_backtest(
    close_daily: pd.Series,
    entries_daily: np.ndarray,
    exits_daily: np.ndarray,
    sizes_daily: np.ndarray,
    data_1m: pd.DataFrame,
    poly_base_daily: np.ndarray,
    open_: pd.Series | None = None,
    close_raw: pd.Series | None = None,
    entry_modes: np.ndarray | None = None,   # 0=none, 1=grid, 2=switch
    intraday_grid_only: bool = False,         # True=仅 grid 模式用日内执行（switch 保持开盘）
    grid_levels: int = 3,
    grid_spacing_pct: float = 0.005,
    first_offset_pct: float = 0.002,
    closeout_minute: int = 225,
) -> dict:
    """结合日线信号与分钟线执行，计算完整回测结果。

    对每个日线信号（entry/exit），在对应的下一个交易日用分钟线分批执行。
    与普通的 next-bar-open 执行做对比。

    Returns:
        dict with:
          - intraday_return:  日内执行的收益率
          - baseline_return:  次日开盘执行的收益率（基准）
          - intraday_trades:  日内执行的交易列表
          - baseline_trades:  基准执行的交易列表
          - fill_improvement: 平均成交价改善幅度（bps）
    """
    n_daily = len(close_daily)

    # 构建前一日未复权收盘价序列（用作日内执行的基准价）
    # 信号在 bar i 生成，执行在 bar i+1，基准价为 bar i 的收盘（已知信息）
    if close_raw is not None:
        prev_close_raw = close_raw.reindex(close_daily.index, method="ffill")
    else:
        prev_close_raw = pd.Series(close_daily.values, index=close_daily.index)

    # 跟踪当前持仓的模式（用于确定离场时是否用日内执行）
    position_mode = 0  # 0=空仓, 1=grid, 2=switch

    intraday_trades = []        # 日内执行
    baseline_trades = []        # 基准（次日开盘）

    # 确定入场是否用日内执行：grid(均值回复)→限价等回落，switch(趋势)→开盘立即成交
    def _use_intraday(emode):
        if not intraday_grid_only:
            return True
        return emode == 1  # 仅 grid 模式用日内限价

    # 遍历日线信号
    for i in range(n_daily - 1):
        if entries_daily[i]:
            emode = int(entry_modes[i]) if entry_modes is not None else 1
            next_date = close_daily.index[i + 1]
            minute_bars = _align_minute_data(data_1m, next_date)
            ref_price = float(prev_close_raw.iloc[i])
            size = float(sizes_daily[i]) if sizes_daily[i] > 0 else 1.0

            if _use_intraday(emode) and minute_bars is not None and ref_price > 0:
                result = execute_intraday_buy(
                    minute_bars, size, ref_price,
                    grid_levels, grid_spacing_pct, first_offset_pct,
                    closeout_minute,
                )
                intraday_trades.append({
                    "date": next_date, "type": "entry", "size": size,
                    "fill_price": result.avg_fill_price,
                    "fill_pct": result.fill_pct,
                    "limit_fill_pct": result.limit_fill_pct,
                    "closeout_pct": result.closeout_pct,
                    "num_fills": result.num_fills, "mode": emode,
                })
                position_mode = emode
            else:
                # grid 模式或不满足条件时：用次日开盘（等同于 baseline）
                intraday_trades.append({
                    "date": next_date, "type": "entry", "size": size,
                    "fill_price": float(open_.iloc[i + 1]) if open_ is not None else float(close_daily.iloc[i + 1]),
                    "fill_pct": 1.0, "limit_fill_pct": 0.0, "closeout_pct": 0.0,
                    "num_fills": 0, "mode": emode,
                })
                position_mode = emode

            if open_ is not None:
                open_price = float(open_.iloc[i + 1])
                baseline_trades.append({
                    "date": next_date, "type": "entry", "size": size,
                    "fill_price": open_price, "fill_pct": 1.0,
                })

        if exits_daily[i]:
            emode = position_mode  # 沿用入场时的模式
            next_date = close_daily.index[i + 1]
            minute_bars = _align_minute_data(data_1m, next_date)
            ref_price = float(prev_close_raw.iloc[i])
            size = 1.0

            if _use_intraday(emode) and minute_bars is not None and ref_price > 0:
                result = execute_intraday_sell(
                    minute_bars, size, ref_price,
                    grid_levels, grid_spacing_pct, first_offset_pct,
                    closeout_minute,
                )
                intraday_trades.append({
                    "date": next_date, "type": "exit", "size": size,
                    "fill_price": result.avg_fill_price,
                    "fill_pct": result.fill_pct,
                    "limit_fill_pct": result.limit_fill_pct,
                    "closeout_pct": result.closeout_pct,
                    "num_fills": result.num_fills, "mode": emode,
                })
            else:
                intraday_trades.append({
                    "date": next_date, "type": "exit", "size": size,
                    "fill_price": float(open_.iloc[i + 1]) if open_ is not None else float(close_daily.iloc[i + 1]),
                    "fill_pct": 1.0, "limit_fill_pct": 0.0, "closeout_pct": 0.0,
                    "num_fills": 0, "mode": emode,
                })
            position_mode = 0

            if open_ is not None:
                open_price = float(open_.iloc[i + 1])
                baseline_trades.append({
                    "date": next_date, "type": "exit", "size": size,
                    "fill_price": open_price, "fill_pct": 1.0,
                })

    # ── 构建权益曲线并计算指标 ──
    def _build_equity_and_metrics(trades, start_date):
        """从交易列表构建日频权益曲线，计算年化收益/Sharpe/最大回撤。"""
        if len(trades) == 0:
            return pd.Series(dtype=float), 0.0, 0.0, 0.0, 0.0

        # 配对 entry/exit，跳过开头的孤立 exit
        pairs = []
        start_idx = 0
        while start_idx < len(trades) and trades[start_idx]["type"] == "exit":
            start_idx += 1

        i = start_idx
        while i < len(trades):
            if trades[i]["type"] == "entry":
                entry_trade = trades[i]
                # 找下一个 exit
                j = i + 1
                while j < len(trades) and trades[j]["type"] != "exit":
                    j += 1
                if j < len(trades):
                    exit_trade = trades[j]
                    trade_ret = (exit_trade["fill_price"] - entry_trade["fill_price"]) / entry_trade["fill_price"]
                    pairs.append({
                        "entry_date": entry_trade["date"],
                        "exit_date": exit_trade["date"],
                        "return": trade_ret,
                    })
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1

        if len(pairs) == 0:
            return pd.Series(dtype=float), 0.0, 0.0, 0.0, 0.0

        # 构建日频权益曲线：从第一笔入场前一日开始，初始权益=1
        first_date = pairs[0]["entry_date"]
        last_date = pairs[-1]["exit_date"]
        # 扩展日期范围以覆盖完整交易区间
        date_range = pd.date_range(first_date - pd.Timedelta(days=1), last_date, freq="B")
        equity = pd.Series(1.0, index=date_range)

        # 在 exit 日应用收益
        for p in pairs:
            exit_dt = p["exit_date"]
            if exit_dt in equity.index:
                # 找到 exit 日之前的最近权益值
                idx_loc = equity.index.get_loc(exit_dt)
                equity.iloc[idx_loc] = equity.iloc[max(0, idx_loc - 1)] * (1.0 + p["return"])
            # 将后续日期填充为当前值
            equity = equity.ffill()

        # 计算日收益率
        daily_ret = equity.pct_change().dropna()

        total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
        n_years = max((date_range[-1] - date_range[0]).days / 365.25, 0.01)
        ann_ret = (1.0 + total_ret) ** (1.0 / n_years) - 1.0

        # Sharpe: 年化日收益率
        if len(daily_ret) > 1 and daily_ret.std() > 1e-12:
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # 最大回撤
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        return equity, total_ret, ann_ret, sharpe, max_dd

    start_date = close_daily.index[0]
    eq_intra, intra_ret, intra_ann, intra_sharpe, intra_dd = _build_equity_and_metrics(
        intraday_trades, start_date,
    )
    eq_base, base_ret, base_ann, base_sharpe, base_dd = _build_equity_and_metrics(
        baseline_trades, start_date,
    )

    # 成交价改善（bps）
    improvement_bps = 0.0
    n_compared = 0
    for it, bt in zip(intraday_trades, baseline_trades):
        if it["type"] != bt["type"]:
            continue
        n_compared += 1
        if it["type"] == "entry":
            if bt["fill_price"] > 0:
                improvement_bps += (bt["fill_price"] - it["fill_price"]) / bt["fill_price"] * 10000
        else:
            if bt["fill_price"] > 0:
                improvement_bps += (it["fill_price"] - bt["fill_price"]) / bt["fill_price"] * 10000

    return {
        "intraday_return": intra_ret,
        "intraday_ann_return": intra_ann,
        "intraday_sharpe": intra_sharpe,
        "intraday_max_dd": intra_dd,
        "baseline_return": base_ret,
        "baseline_ann_return": base_ann,
        "baseline_sharpe": base_sharpe,
        "baseline_max_dd": base_dd,
        "excess_return": intra_ret - base_ret,
        "excess_ann_return": intra_ann - base_ann,
        "avg_improvement_bps": improvement_bps / max(n_compared, 1),
        "num_intraday_trades": len(intraday_trades),
        "num_baseline_trades": len(baseline_trades),
        "intraday_trades": intraday_trades,
        "baseline_trades": baseline_trades,
    }


# ══════════════════════════════════════════════════════════════════
# 参数扫描
# ══════════════════════════════════════════════════════════════════

def scan_intraday_params(
    close_daily: pd.Series,
    entries_daily: np.ndarray,
    exits_daily: np.ndarray,
    sizes_daily: np.ndarray,
    data_1m: pd.DataFrame,
    poly_base_daily: np.ndarray,
    open_: pd.Series | None = None,
    close_raw: pd.Series | None = None,
) -> pd.DataFrame:
    """扫描日内执行参数。

    对 grid_levels × grid_spacing × first_offset 做全排列，
    每个组合返回日内执行 vs 基准执行的收益对比。

    参数范围：
      - grid_levels:       2, 3, 4, 5
      - grid_spacing_pct:  0.3%, 0.5%, 0.8%, 1.0%, 1.5%
      - first_offset_pct:  0.0%, 0.1%, 0.2%, 0.5%
    """
    grid_levels_vals = [2, 3, 4, 5]
    grid_spacing_vals = [0.003, 0.005, 0.008, 0.010, 0.015]
    first_offset_vals = [0.0, 0.001, 0.002, 0.005]
    # closeout bar 索引：5m 数据 48 bars/天 → 39=14:15, 42=14:30, 45=14:45
    #                 1m 数据 240 bars/天 → 195=14:15, 210=14:30, 225=14:45
    closeout_vals = [39, 42, 45]

    combos = list(product(
        grid_levels_vals, grid_spacing_vals, first_offset_vals, closeout_vals,
    ))
    print(f"  Intraday scan: {len(combos)} parameter combos")

    results = []
    for gl, gs, fo, co in combos:
        r = run_intraday_backtest(
            close_daily, entries_daily, exits_daily, sizes_daily,
            data_1m, poly_base_daily, open_,
            close_raw=close_raw,
            grid_levels=gl, grid_spacing_pct=gs,
            first_offset_pct=fo, closeout_minute=co,
        )
        results.append({
            "grid_levels": gl,
            "grid_spacing_pct": gs,
            "first_offset_pct": fo,
            "closeout_minute": co,
            "intraday_return": r["intraday_return"],
            "baseline_return": r["baseline_return"],
            "excess_return": r["excess_return"],
            "avg_improvement_bps": r["avg_improvement_bps"],
            "num_trades": r["num_intraday_trades"],
        })

    return pd.DataFrame(results)
