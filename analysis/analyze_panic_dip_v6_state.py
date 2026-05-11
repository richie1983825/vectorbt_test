"""
统计：在基线之下发生大跌时，V6 策略的空仓/满仓概率，以及大跌后买入的概率。

大跌定义：单日跌幅 < -2%（可调整）
基线之下：PolyDevPct < 0
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6

# V6 固定最优参数
V6_GRID = dict(trend_window_days=10, vol_window_days=10, base_grid_pct=0.01,
    volatility_scale=0.0, trend_sensitivity=4, max_grid_levels=3, take_profit_grid=0.8,
    stop_loss_grid=1.6, max_holding_days=45, cooldown_days=1, min_signal_strength=0.3,
    position_size=0.99, position_sizing_coef=60)
V6_SWITCH = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_atr_window=14, trend_vol_climax=2.5, trend_decline_days=1,
    enable_ohlcv_filter=True, enable_early_exit=True)

DROP_THRESHOLD = -0.02  # 单日跌幅阈值

print("Loading data…")
data = load_data("data/1d/512890.SH_hfq.parquet")
close = data["Close"]
open_ = data["Open"]
high = data["High"]
low = data["Low"]
volume = data["Volume"]

# 计算指标
ind = compute_polyfit_switch_indicators(close, fit_window_days=252, ma_windows=[20, 60],
                                         trend_window_days=10, vol_window_days=10)
idx = ind.index
cl = close.loc[idx]
op = open_.reindex(idx)
hi = high.reindex(idx)
lo = low.reindex(idx)
vol = volume.reindex(idx)

dp = ind["PolyDevPct"].values
dt = ind["PolyDevTrend"].values
vp = ind["RollingVolPct"].values
pb = ind["PolyBasePred"].values

# V6 信号
e_g, x_g, s_g = generate_grid_signals(
    cl.values, dp, dt, vp, pb,
    base_grid_pct=V6_GRID["base_grid_pct"], volatility_scale=V6_GRID["volatility_scale"],
    trend_sensitivity=V6_GRID["trend_sensitivity"], max_grid_levels=int(V6_GRID["max_grid_levels"]),
    take_profit_grid=V6_GRID["take_profit_grid"], stop_loss_grid=V6_GRID["stop_loss_grid"],
    max_holding_days=int(V6_GRID["max_holding_days"]), cooldown_days=int(V6_GRID["cooldown_days"]),
    min_signal_strength=V6_GRID["min_signal_strength"], position_size=V6_GRID["position_size"],
    position_sizing_coef=V6_GRID["position_sizing_coef"],
)
e_s, x_s, s_s = generate_grid_priority_switch_signals_v6(
    cl.values, dp, dt, vp, pb, e_g, x_g,
    ind["MA20"].values, ind["MA60"].values,
    high=hi.values, low=lo.values, open_=op.values, volume=vol.values,
    **V6_SWITCH,
)

# 模拟仓位状态
n = len(cl)
grid_in = np.zeros(n, dtype=bool)
sw_in = np.zeros(n, dtype=bool)
g_in = False
s_in = False

for i in range(n):
    if g_in and x_g[i]:
        g_in = False
    if s_in and x_s[i]:
        s_in = False
    if e_g[i]:
        if s_in:
            s_in = False
        g_in = True
        grid_in[i] = True
        continue
    if not g_in:
        if e_s[i]:
            s_in = True
            sw_in[i] = True
    grid_in[i] = g_in
    sw_in[i] = s_in

in_position = grid_in | sw_in

# ── 统计：基线之下 + 大跌 ──
daily_ret = cl.pct_change().values
below_base = dp < 0
big_drop = daily_ret < DROP_THRESHOLD
big_drop_below_base = big_drop & below_base

drop_dates = idx[big_drop_below_base]
print(f"\n{'═' * 70}")
print(f"  V6 策略：基线之下发生大跌（<{DROP_THRESHOLD:.0%}）的仓位状态统计")
print(f"{'═' * 70}")
print(f"  总大跌日（基线之下）: {len(drop_dates)} 天")

# 大跌日的仓位状态
pos_on_drop = in_position[big_drop_below_base]
idle_on_drop = ~pos_on_drop
print(f"  大跌时已满仓: {pos_on_drop.sum()} 天 ({pos_on_drop.sum()/len(drop_dates)*100:.1f}%)")
print(f"  大跌时为空仓: {idle_on_drop.sum()} 天 ({idle_on_drop.sum()/len(drop_dates)*100:.1f}%)")

# 大跌时空仓中有多少是 Grid、多少是 Switch
grid_on_drop = grid_in[big_drop_below_base]
sw_on_drop = sw_in[big_drop_below_base]
print(f"    其中 Grid 持仓: {grid_on_drop.sum()} 天")
print(f"    其中 Switch 持仓: {sw_on_drop.sum()} 天")

# ── 大跌后 V6 是否买入 ──
print(f"\n{'═' * 70}")
print(f"  大跌后 V6 买入决策统计")
print(f"{'═' * 70}")

# 只看大跌当时空仓的情况
idle_drop_idx = np.where(big_drop_below_base & (~in_position))[0]

for look_forward in [1, 3, 5, 10]:
    bought = 0
    for i in idle_drop_idx:
        end = min(i + look_forward + 1, n)
        if e_g[i+1:end].any() or e_s[i+1:end].any():
            bought += 1
    pct = bought / max(len(idle_drop_idx), 1) * 100
    print(f"  大跌时空仓 → {look_forward}天内买入: {bought}/{len(idle_drop_idx)} ({pct:.1f}%)")

# ── 大跌时已满仓的情况：后续是否因大跌离场 ──
print(f"\n  大跌时已满仓的后续离场:")
pos_drop_idx = np.where(big_drop_below_base & in_position)[0]
for look_forward in [1, 3, 5]:
    exited = 0
    for i in pos_drop_idx:
        end = min(i + look_forward + 1, n)
        if x_g[i+1:end].any() or x_s[i+1:end].any():
            exited += 1
    pct = exited / max(len(pos_drop_idx), 1) * 100
    print(f"  大跌时满仓 → {look_forward}天内离场: {exited}/{len(pos_drop_idx)} ({pct:.1f}%)")

# ── 列举大跌日详细信息 ──
print(f"\n{'═' * 70}")
print(f"  大跌日详细列表（基线之下，跌幅<{DROP_THRESHOLD:.0%}）")
print(f"{'═' * 70}")
print(f"  {'日期':<12s} {'跌幅':>7s}  {'dp':>7s}  {'仓位':<8s} {'模式':<8s} {'次日涨跌':>7s} {'3日涨跌':>7s}")

for i in range(n):
    if not big_drop_below_base[i]:
        continue
    ret_1d = daily_ret[i]
    ret_next = daily_ret[min(i+1, n-1)] if i+1 < n else 0
    ret_3d = (cl[min(i+3, n-1)] / cl[i] - 1) if i+3 < n else 0

    if grid_in[i]:
        mode = "Grid"
    elif sw_in[i]:
        mode = "Switch"
    else:
        mode = "空仓"

    pos_str = "满仓" if in_position[i] else "空仓"
    print(f"  {str(idx[i].date()):<12s} {ret_1d:>+7.2%}  {dp[i]:>+7.1%}  {pos_str:<8s} {mode:<8s} {ret_next:>+7.2%} {ret_3d:>+7.2%}")

# ── 不同跌幅阈值下的统计 ──
print(f"\n{'═' * 70}")
print(f"  不同跌幅阈值下的空仓/满仓概率")
print(f"{'═' * 70}")
print(f"  {'阈值':>8s}  {'事件数':>6s}  {'满仓%':>7s}  {'空仓%':>7s}  {'空仓后5日买入%':>14s}")

for thresh in [-0.01, -0.015, -0.02, -0.025, -0.03]:
    big = (daily_ret < thresh) & below_base
    n_events = big.sum()
    if n_events == 0:
        continue
    pos_pct = in_position[big].sum() / n_events * 100
    idle_pct = (~in_position[big]).sum() / n_events * 100

    idle_idx = np.where(big & (~in_position))[0]
    bought_5d = 0
    for i in idle_idx:
        end = min(i + 6, n)
        if e_g[i+1:end].any() or e_s[i+1:end].any():
            bought_5d += 1
    buy_pct = bought_5d / max(len(idle_idx), 1) * 100

    print(f"  {thresh:>+8.1%}  {n_events:>6d}  {pos_pct:>6.1f}%  {idle_pct:>6.1f}%  {buy_pct:>13.1f}%")
