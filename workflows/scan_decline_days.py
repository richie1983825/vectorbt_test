#!/usr/bin/env python3
"""单独扫描 trend_decline_days 参数，找到 return 和 balanced 评分下的最优值。"""

import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from utils.backtest import run_backtest
from utils.scoring import select_by_return, select_balanced
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v7

warnings.filterwarnings("ignore")


V6_FIXED = dict(
    trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_vol_climax=2.5,
    enable_ohlcv_filter=True, enable_early_exit=True,
)

V7_BEST_TOP = dict(
    enable_top_avoidance=True, top_ret_5d=0.05,
    top_price_pos=0.80, top_amplitude=0.02, top_block_days=3,
)

# ── 加载数据 ──
data_path = "data/1d/512890.SH_hfq.parquet"
print(f"加载: {data_path}")
data = load_data(data_path)
close = data["Close"]
open_ = data["Open"]
high = data["High"]
low = data["Low"]
volume = data["Volume"]
print(f"  {len(data)} bars, {data.index[0].date()} → {data.index[-1].date()}")

# ── 取最新 WF 窗口的 Grid 最优参数 ──
cache = pd.read_csv("reports/grid_wf_cache.csv")
cache_return = cache[cache["selector"] == "return"]
cache_return["test_start_dt"] = pd.to_datetime(cache_return["test_start_date"])
latest_return = cache_return.nlargest(1, "test_start_dt").iloc[0]

grid_params = {}
for k in ["trend_window_days", "vol_window_days", "base_grid_pct",
           "volatility_scale", "trend_sensitivity", "max_grid_levels",
           "take_profit_grid", "stop_loss_grid", "max_holding_days",
           "cooldown_days", "min_signal_strength", "position_size",
           "position_sizing_coef"]:
    grid_params[k] = latest_return[k]

print(f"Grid params (return, {latest_return['test_start_date']}): "
      f"tw={int(grid_params['trend_window_days'])} "
      f"bgp={grid_params['base_grid_pct']:.3f} "
      f"ts={int(grid_params['trend_sensitivity'])}")

# ── 计算指标和 Grid 信号（与 predict.py 一致）──
tw = int(grid_params["trend_window_days"])
vw = int(grid_params["vol_window_days"])

ind = compute_polyfit_switch_indicators(
    close, fit_window_days=252, ma_windows=[20, 60],
    trend_window_days=tw, vol_window_days=vw,
)
com_idx = ind.index
cl_arr = close.loc[com_idx].values
op_arr = open_.reindex(com_idx).values
hi_arr = high.reindex(com_idx).values
lo_arr = low.reindex(com_idx).values
vol_arr = volume.reindex(com_idx).values

cl_s = close.loc[com_idx]
op_s = open_.reindex(com_idx)

e_grid, x_grid, s_grid = generate_grid_signals(
    cl_arr, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
    ind["RollingVolPct"].values, ind["PolyBasePred"].values,
    base_grid_pct=grid_params["base_grid_pct"],
    volatility_scale=grid_params["volatility_scale"],
    trend_sensitivity=grid_params["trend_sensitivity"],
    max_grid_levels=int(grid_params["max_grid_levels"]),
    take_profit_grid=grid_params["take_profit_grid"],
    stop_loss_grid=grid_params["stop_loss_grid"],
    max_holding_days=int(grid_params["max_holding_days"]),
    cooldown_days=int(grid_params["cooldown_days"]),
    min_signal_strength=grid_params["min_signal_strength"],
    position_size=grid_params["position_size"],
    position_sizing_coef=grid_params["position_sizing_coef"],
)

# ── 扫描 trend_decline_days ──
DECLINE_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

results = []
for tdd in DECLINE_VALS:
    result = generate_grid_priority_switch_signals_v7(
        cl_arr, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
        ind["RollingVolPct"].values, ind["PolyBasePred"].values,
        e_grid, x_grid, ind["MA20"].values, ind["MA60"].values,
        trend_entry_dp=V6_FIXED["trend_entry_dp"],
        trend_confirm_dp_slope=V6_FIXED["trend_confirm_dp_slope"],
        trend_atr_mult=V6_FIXED["trend_atr_mult"], trend_atr_window=14,
        trend_vol_climax=V6_FIXED["trend_vol_climax"],
        trend_decline_days=tdd,
        enable_ohlcv_filter=V6_FIXED["enable_ohlcv_filter"],
        enable_early_exit=V6_FIXED["enable_early_exit"],
        enable_top_avoidance=V7_BEST_TOP["enable_top_avoidance"],
        top_ret_5d=V7_BEST_TOP["top_ret_5d"],
        top_price_pos=V7_BEST_TOP["top_price_pos"],
        top_amplitude=V7_BEST_TOP["top_amplitude"],
        top_block_days=V7_BEST_TOP["top_block_days"],
        high=hi_arr, low=lo_arr, open_=op_arr, volume=vol_arr,
        return_filter_stats=True,
    )

    sw_entries = result["sw_entries"]
    sw_exits = result["sw_exits"]

    e_merged = e_grid | sw_entries
    x_merged = x_grid | sw_exits
    s_merged = np.where(e_grid, s_grid, np.where(sw_entries, 0.99, 0.0))

    total_entries = e_merged.sum()
    if total_entries == 0:
        results.append({
            "trend_decline_days": tdd,
            "total_return": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "calmar_ratio": 0.0,
            "num_trades": 0, "win_rate": 0.0,
        })
        continue

    m = run_backtest(cl_s, e_merged, x_merged, s_merged, open_=op_s)
    results.append({
        "trend_decline_days": tdd,
        "total_return": m["total_return"],
        "sharpe_ratio": m["sharpe_ratio"],
        "max_drawdown": m["max_drawdown"],
        "calmar_ratio": m["calmar_ratio"],
        "num_trades": m["num_trades"],
        "win_rate": m["win_rate"],
    })

df = pd.DataFrame(results)

# ── 输出 ──
print(f"\n{'='*80}")
print(f"trend_decline_days 扫描结果 (全量数据 {com_idx[0].date()} → {com_idx[-1].date()})")
print(f"{'='*80}")
print(f"{'tdd':>5}  {'Return':>10}  {'Sharpe':>8}  {'MaxDD':>8}  {'Calmar':>8}  {'Trades':>7}  {'WinRate':>8}")
print(f"{'-'*65}")
for _, r in df.iterrows():
    print(f"{int(r['trend_decline_days']):5d}  {r['total_return']:10.1%}  {r['sharpe_ratio']:8.3f}  "
          f"{r['max_drawdown']:8.1%}  {r['calmar_ratio']:8.3f}  {int(r['num_trades']):7d}  {r['win_rate']:8.1%}")

# 最优参数
best_return = select_by_return(df[df["num_trades"] > 0])
best_balanced = select_balanced(df[df["num_trades"] > 0])

print(f"\n{'='*80}")
print(f"  return  最优: trend_decline_days = {int(best_return['trend_decline_days'])}  "
      f"(Return={best_return['total_return']:.1%}, Sharpe={best_return['sharpe_ratio']:.3f}, "
      f"MaxDD={best_return['max_drawdown']:.1%})")
print(f"  balanced 最优: trend_decline_days = {int(best_balanced['trend_decline_days'])}  "
      f"(Return={best_balanced['total_return']:.1%}, Sharpe={best_balanced['sharpe_ratio']:.3f}, "
      f"MaxDD={best_balanced['max_drawdown']:.1%})")
print(f"\n  当前 V6_FIXED 值: trend_decline_days = 1  "
      f"(Return={df[df['trend_decline_days']==1]['total_return'].values[0]:.1%}, "
      f"MaxDD={df[df['trend_decline_days']==1]['max_drawdown'].values[0]:.1%})")
