"""
对比 v5 (原版) vs v6 (OHLCV增强版) Grid-priority Switch 策略。

v6 改进：
  入场过滤：
    1. 微涨 0~+0.5% → 不买（胜率 12.5%）
    2. 长上影 ≥30% → 不买（胜率 18.4%）
    3. 正常量能 0.8-1.5x → 不买（胜率 13.7%）
    4. 阳线 + 连涨<4天 → 不买（阳线胜率 23.8%）
    5. 高位 ≥70% + 连涨<4天 → 不买（高位胜率 32.1%）
  离场改进：
    a. dp连续下跌 2天即离场（原3天）
    b. dp接近Grid入场区时提前离场（避免grid_force 0%胜率）
    c. ATR乘数 2.5→2.0
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import (
    compute_polyfit_switch_indicators,
    compute_polyfit_base_only,
    add_trend_vol_indicators,
)
from utils.backtest import run_backtest
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import (
    generate_grid_priority_switch_signals,       # v5 原版
    generate_grid_priority_switch_signals_v6,    # v6 OHLCV增强
)

# ══════════════════════════════════════════════════════════════════
GRID_PARAMS = dict(
    base_grid_pct=0.008, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=1.0, stop_loss_grid=1.2,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=0.45, position_size=0.92, position_sizing_coef=30,
)

# v5 (原版) Switch 参数
SW_PARAMS_V5 = dict(
    trend_entry_dp=0.0,
    trend_confirm_dp_slope=0.0003,
    trend_atr_mult=2.5,
    trend_atr_window=14,
    trend_vol_climax=3.0,
    trend_decline_days=3,
)

# v6 (OHLCV增强) Switch 参数 — 离场参数已内置于函数默认值
SW_PARAMS_V6 = dict(
    trend_entry_dp=0.0,
    trend_confirm_dp_slope=0.0003,
    # atr_mult=2.0, decline_days=2 已作为函数默认值
)


def evaluate_window(close_warmup_all, open_warmup_all, high, low, volume,
                    test_offset, tw=10, vw=20):
    """在单个 WF 测试窗口上对比 v5 vs v6。"""
    test_start_label = close_warmup_all.index[test_offset]
    ind_s = compute_polyfit_switch_indicators(
        close_warmup_all, fit_window_days=252, ma_windows=[20, 60],
        trend_window_days=tw, vol_window_days=vw,
    )
    ind_test = ind_s.loc[ind_s.index >= test_start_label]
    if len(ind_test) < 10:
        return None

    cl_test = close_warmup_all.loc[ind_test.index]
    op_test = open_warmup_all.loc[ind_test.index]

    cl_arr = cl_test.values
    op_arr = op_test.values
    hi_arr = high.reindex(ind_test.index).values
    lo_arr = low.reindex(ind_test.index).values
    vol_arr = volume.reindex(ind_test.index).values
    dev_pct_arr = ind_test["PolyDevPct"].values
    dev_trend_arr = ind_test["PolyDevTrend"].values
    vol_pct_arr = ind_test["RollingVolPct"].values
    poly_base_arr = ind_test["PolyBasePred"].values
    ma20_arr = ind_test["MA20"].values
    ma60_arr = ind_test["MA60"].values

    # Grid 信号（v5 和 v6 共用同一个 Grid）
    e_grid, x_grid, s_grid = generate_grid_signals(
        cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
        **GRID_PARAMS,
    )
    n_grid_entries = int(e_grid.sum())

    # ── v5 (原版) ──
    e_sw5, x_sw5, s_sw5 = generate_grid_priority_switch_signals(
        cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
        e_grid, x_grid, ma20_arr, ma60_arr,
        high=hi_arr, low=lo_arr, volume=vol_arr,
        **SW_PARAMS_V5,
    )
    e_m5 = e_grid | e_sw5
    x_m5 = x_grid | x_sw5
    s_m5 = np.where(e_grid, s_grid, np.where(e_sw5, 0.99, 0.0))
    m5 = run_backtest(pd.Series(cl_arr, index=ind_test.index),
                      e_m5, x_m5, s_m5, open_=op_test)

    # ── v6 (OHLCV增强) ──
    result_v6 = generate_grid_priority_switch_signals_v6(
        cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
        e_grid, x_grid, ma20_arr, ma60_arr,
        high=hi_arr, low=lo_arr, open_=op_arr, volume=vol_arr,
        return_filter_stats=True,
        **SW_PARAMS_V6,
    )
    e_sw6 = result_v6["sw_entries"]
    x_sw6 = result_v6["sw_exits"]
    s_sw6 = result_v6["sw_sizes"]
    filter_stats = result_v6["filter_stats"]

    e_m6 = e_grid | e_sw6
    x_m6 = x_grid | x_sw6
    s_m6 = np.where(e_grid, s_grid, np.where(e_sw6, 0.99, 0.0))
    m6 = run_backtest(pd.Series(cl_arr, index=ind_test.index),
                      e_m6, x_m6, s_m6, open_=op_test)

    return {
        "n_grid": n_grid_entries,
        "n_sw5": int(e_sw5.sum()),
        "n_sw6": int(e_sw6.sum()),
        "sw5_reduced": int(e_sw5.sum()) - int(e_sw6.sum()),
        "filter_stats": filter_stats,
        "m5": m5,
        "m6": m6,
        "test_start": test_start_label,
        "n_bars": len(cl_arr),
    }


def main():
    print("=" * 90)
    print("  v5 (原版) vs v6 (OHLCV增强) Grid-priority Switch 对比")
    print("=" * 90)

    data = load_data("data/1d/512890.SH_hfq.parquet")
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    windows = generate_monthly_windows(
        close.index, train_months=22, test_months=12,
        step_months=3, warmup_months=12,
    )
    print(f"WF 窗口: {len(windows)} 个 (步长3月)\n")

    results = []
    total_filter = {"micro_up": 0, "long_upper_wick": 0, "normal_volume": 0,
                    "green_without_momentum": 0, "high_position": 0, "passed": 0}

    t0 = time.time()
    for wi, w in enumerate(windows):
        close_warmup_all = close.iloc[w.warmup_start:w.test_end]
        open_warmup_all = open_.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start

        r = evaluate_window(
            close_warmup_all, open_warmup_all, high, low, volume,
            test_offset, tw=10, vw=20,
        )
        if r is None:
            continue
        results.append(r)
        for k in total_filter:
            total_filter[k] += r["filter_stats"].get(k, 0)

        if (wi + 1) % 5 == 0 or wi == 0:
            elapsed = time.time() - t0
            print(f"  [{wi+1:>2d}/{len(windows)}] {elapsed:.0f}s  "
                  f"v5-Switch={sum(rr['n_sw5'] for rr in results):>3d}  "
                  f"v6-Switch={sum(rr['n_sw6'] for rr in results):>3d}  "
                  f"阻拦={sum(rr['sw5_reduced'] for rr in results):>3d}")

    elapsed = time.time() - t0
    n_win = len(results)
    print(f"\n  共 {n_win} 个有效窗口，耗时 {elapsed:.0f}s\n")

    # ── 聚合结果 ──
    total_grid = sum(r["n_grid"] for r in results)
    total_sw5 = sum(r["n_sw5"] for r in results)
    total_sw6 = sum(r["n_sw6"] for r in results)
    total_reduced = total_sw5 - total_sw6

    print("═" * 90)
    print("  入场过滤效果")
    print("═" * 90)
    print(f"  Grid 入场总数:     {total_grid:>5d}")
    print(f"  v5 Switch 入场:    {total_sw5:>5d}")
    print(f"  v6 Switch 入场:    {total_sw6:>5d}")
    print(f"  被阻拦:            {total_reduced:>5d}  ({total_reduced/max(total_sw5,1)*100:.1f}%)")
    print(f"")
    filter_total_blocked = sum(v for k, v in total_filter.items() if k != "passed")
    print(f"  ── 阻拦原因明细 ──")
    reason_labels = [
        ("micro_up", "微涨 0~+0.5% (胜率12.5%)"),
        ("long_upper_wick", "长上影≥30% (胜率18.4%)"),
        ("normal_volume", "正常量能0.8-1.5x (胜率13.7%)"),
        ("green_without_momentum", "阳线+连涨<4天 (阳线胜率23.8%)"),
        ("high_position", "高位≥70%+连涨<4天 (高位胜率32.1%)"),
    ]
    for key, label in reason_labels:
        count = total_filter.get(key, 0)
        print(f"  {label:<40s} {count:>5d}  ({count/max(filter_total_blocked,1)*100:5.1f}%)")
    print(f"  {'通过过滤':<40s} {total_filter['passed']:>5d}")

    # ── 回测对比 ──
    print(f"\n{'═' * 90}")
    print(f"  回测对比（{n_win} 窗口平均）")
    print(f"{'═' * 90}")

    agg_m5 = {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0,
              "num_trades": 0, "win_rate": 0}
    agg_m6 = {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0,
              "num_trades": 0, "win_rate": 0}
    v6_better = 0
    v6_worse = 0

    for r in results:
        for k in agg_m5:
            agg_m5[k] += r["m5"][k]
            agg_m6[k] += r["m6"][k]
        if r["m6"]["total_return"] > r["m5"]["total_return"]:
            v6_better += 1
        elif r["m6"]["total_return"] < r["m5"]["total_return"]:
            v6_worse += 1

    for k in agg_m5:
        agg_m5[k] /= n_win
        agg_m6[k] /= n_win

    print(f"  {'指标':<14s} {'v5 原版':>10s} {'v6 增强':>10s} {'变化':>10s} {'改善?':>8s}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for label, key, fmt, better_is_higher in [
        ("总收益/窗口", "total_return", "+.2%", True),
        ("Sharpe", "sharpe_ratio", ".3f", True),
        ("最大回撤", "max_drawdown", "+.2%", False),
        ("交易次数", "num_trades", ".1f", None),
        ("胜率", "win_rate", ".1%", True),
    ]:
        a, b = agg_m5[key], agg_m6[key]
        d = b - a
        if better_is_higher is True:
            improved = "✓" if d > 0 else ("✗" if d < 0 else "=")
        elif better_is_higher is False:
            improved = "✓" if d > 0 else ("✗" if d < 0 else "=")  # max_dd is negative, larger = better
        else:
            improved = ""
        print(f"  {label:<14s} {a:{fmt}}     {b:{fmt}}     {d:{fmt}}     {improved:>8s}")

    print(f"\n  v6 优于 v5 的窗口: {v6_better}/{n_win} ({v6_better/n_win*100:.0f}%)")
    print(f"  v6 劣于 v5 的窗口: {v6_worse}/{n_win} ({v6_worse/n_win*100:.0f}%)")
    print(f"  持平:              {n_win - v6_better - v6_worse}")

    # ── 按窗口明细 ──
    print(f"\n{'═' * 90}")
    print(f"  各窗口明细")
    print(f"{'═' * 90}")
    print(f"  {'测试起始':>12s} {'v5收益':>8s} {'v6收益':>8s} {'差异':>8s} "
          f"{'v5-SW':>6s} {'v6-SW':>6s} {'阻拦':>5s} {'v5胜率':>7s} {'v6胜率':>7s}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*5} {'─'*7} {'─'*7}")
    for r in results:
        diff = r["m6"]["total_return"] - r["m5"]["total_return"]
        print(f"  {str(r['test_start'])[:10]:>12s} "
              f"{r['m5']['total_return']:>+7.1%} {r['m6']['total_return']:>+7.1%} "
              f"{diff:>+7.1%} "
              f"{r['n_sw5']:>6d} {r['n_sw6']:>6d} {r['sw5_reduced']:>5d} "
              f"{r['m5']['win_rate']:>6.1%} {r['m6']['win_rate']:>6.1%}")

    # ── 总结 ──
    print(f"\n{'═' * 90}")
    print(f"  总结")
    print(f"{'═' * 90}")
    ret_diff = agg_m6["total_return"] - agg_m5["total_return"]
    sharpe_diff = agg_m6["sharpe_ratio"] - agg_m5["sharpe_ratio"]
    dd_diff = agg_m6["max_drawdown"] - agg_m5["max_drawdown"]
    print(f"  OHLCV增强版 (v6) 相比原版 (v5)：")
    print(f"    平均收益变化: {ret_diff:+.2%}")
    print(f"    Sharpe 变化:  {sharpe_diff:+.3f}")
    print(f"    最大回撤变化: {dd_diff:+.2%}")
    print(f"    Switch入场减少: {total_reduced} 笔 (过滤率 {total_reduced/max(total_sw5,1)*100:.1f}%)")
    print(f"    v6更优窗口占比: {v6_better}/{n_win} ({v6_better/n_win*100:.0f}%)")


if __name__ == "__main__":
    main()
