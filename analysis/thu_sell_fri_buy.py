"""分析: 持仓在周四收盘卖出、周五开盘买回 是否有利可图."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import vectorbt as vbt
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals

# ── 数据 ──────────────────────────────────────────────────
df = pd.read_parquet("data/1d/512890.SH_hfq.parquet")
df.index = pd.to_datetime(df.index)
close = df["Close"].loc["2019-01-01":"2026-04-30"]
open_ = df["Open"].loc["2019-01-01":"2026-04-30"]

BEST = dict(base_grid_pct=0.01, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=0.8, stop_loss_grid=1.6,
    max_holding_days=45, cooldown_days=1, min_signal_strength=0.3,
    position_size=0.99, position_sizing_coef=60)

ind = compute_polyfit_switch_indicators(
    close, fit_window_days=252, ma_windows=[],
    trend_window_days=10, vol_window_days=10)
idx = ind.index
cl = close.loc[idx].values; op = open_.reindex(idx).values
wd = pd.Series(idx.weekday, index=idx).values
e, x, s = generate_grid_signals(cl, ind["PolyDevPct"].values,
    ind["PolyDevTrend"].values, ind["RollingVolPct"].values,
    ind["PolyBasePred"].values, **BEST)

# ── 模拟持有状态 ──────────────────────────────────────────
n = len(cl)
in_pos = np.zeros(n, dtype=bool)
cur_in = False
for i in range(n):
    if e[i]: cur_in = True
    elif x[i]: cur_in = False
    in_pos[i] = cur_in

# ── 分析每个周四 ──────────────────────────────────────────
thu_mask = wd == 3
thu_idx = np.where(thu_mask)[0]
fri_mask = wd == 4
fri_idx_all = np.where(fri_mask)[0]

# 配对: 每个周四对应的下一个周五
paired = []  # (thu_i, fri_i, thu_close, fri_open, was_in_position, in_cooldown)
for ti in thu_idx:
    # 找下一个周五
    next_fri = fri_idx_all[fri_idx_all > ti]
    if len(next_fri) == 0:
        continue
    fi = next_fri[0]
    paired.append({
        "thu_i": ti, "fri_i": fi,
        "thu_close": cl[ti], "fri_open": op[fi],
        "was_in_pos": in_pos[ti],
    })

print(f"总周四数: {len(paired)}")
in_pos_thu = [p for p in paired if p["was_in_pos"]]
print(f"策略持仓的周四: {len(in_pos_thu)} 个 ({len(in_pos_thu)/len(paired):.0%})")

# ── 周四→周五 跳空分析 ────────────────────────────────────
gaps = np.array([p["fri_open"] / p["thu_close"] - 1 for p in paired])
pos_gaps = np.array([p["fri_open"] / p["thu_close"] - 1 for p in in_pos_thu])

print(f"\n═══════════════════════════════════════════════")
print(f"  周四收盘 → 周五开盘 跳空幅度")
print(f"═══════════════════════════════════════════════")
print(f"  全部周四 (N={len(gaps)}):")
print(f"    均值: {np.mean(gaps):+.4%}  中位: {np.median(gaps):+.4%}")
print(f"    下跌概率: {(gaps < 0).mean():.0%}")
print(f"    均值(仅下跌): {np.mean(gaps[gaps<0]):+.4%}")
print(f"    均值(仅上涨): {np.mean(gaps[gaps>0]):+.4%}")

print(f"\n  策略持仓周四 (N={len(pos_gaps)}):")
print(f"    均值: {np.mean(pos_gaps):+.4%}  中位: {np.median(pos_gaps):+.4%}")
print(f"    下跌概率: {(pos_gaps < 0).mean():.0%}")

# ── 模拟: 周四收盘卖出 → 周五开盘买回 ─────────────────────
# 假设: 每次操作都能成交，不改变策略状态
# 收益 = 周四收盘卖出 → 周五开盘买回的价差
# (忽略交易成本)
print(f"\n═══════════════════════════════════════════════")
print(f"  \"周四卖→周五买\" 的潜在收益")
print(f"═══════════════════════════════════════════════")

# 对比: 如果持仓不动(周四收盘持有到周五收盘) vs 周四卖周五买
scenarios = []
for p in in_pos_thu:
    ti, fi = p["thu_i"], p["fri_i"]
    gap = p["fri_open"] / p["thu_close"] - 1  # 周四卖 + 周五买的价差收益
    # 如果不动: 周四收盘→周五收盘
    buy_hold = cl[fi] / cl[ti] - 1
    # 如果做价差: 周四收盘卖 + 周五开盘买 = 节省了 gap 的损失
    # 即: 卖在 thu_close, 买在 fri_open, 持有到 fri_close
    # swing 收益 = (cl[fi] / op[fi] - 1) + (-gap) 近似
    swing = (cl[fi] / op[fi] - 1) - gap  # 日内 + 省下的跳空
    scenarios.append({"gap": gap, "hold": buy_hold, "swing": swing})

gaps_arr = np.array([s["gap"] for s in scenarios])
holds_arr = np.array([s["hold"] for s in scenarios])
swings_arr = np.array([s["swing"] for s in scenarios])

print(f"  持仓不动 (周四收→周五收):  均值 {np.mean(holds_arr):+.4%}  胜率 {(holds_arr>0).mean():.0%}")
print(f"  周四卖→周五买→周五收:     均值 {np.mean(swings_arr):+.4%}  胜率 {(swings_arr>0).mean():.0%}")
print(f"  改进: {np.mean(swings_arr-holds_arr):+.4%}")

# ── 累加效应 ──────────────────────────────────────────────
print(f"\n  若每次持仓周四都执行此操作 (N={len(in_pos_thu)}):")
cum_extra = np.sum(-gaps_arr)  # 每次省下的跳空, 累加
print(f"  累加减少的损失: {cum_extra:+.2%}")

# ── 关键障碍 ──────────────────────────────────────────────
print(f"\n═══════════════════════════════════════════════")
print(f"  执行层面的问题")
print(f"═══════════════════════════════════════════════")

# 1. 卖出后触发 cooldown，周五可能买不回来
print(f"\n  1. 冷却期冲突:")
print(f"     策略有 {BEST['cooldown_days']} 天冷却期")
print(f"     周四卖出 → 触发冷却 → 周五无法重新入场 → 到下周一才能买")
print(f"     如果跳过冷却 → 改变了策略的风险控制逻辑")

# 2. 周五可能没有入场信号
# Check: 在策略已持仓的周四之后，周五是否能触发入场信号？
reentry_ok = 0
reentry_fail = 0
for p in in_pos_thu:
    fi = p["fri_i"]
    if fi < len(e) and e[fi]:
        reentry_ok += 1
    else:
        reentry_fail += 1
print(f"\n  2. 周五入场信号:")
print(f"     若周四卖出，周五能重新触发入场: {reentry_ok}/{len(in_pos_thu)}")
print(f"     无法重新入场: {reentry_fail}/{len(in_pos_thu)}")
print(f"     → 若强制买回，{reentry_fail}次是在没有信号的情况下追入")

# 3. 交易成本
spread = 0.001  # 假设买卖价差 0.1%
round_trip = 2 * spread
print(f"\n  3. 交易成本:")
print(f"     单边成本 ~0.1% → 往返 ~0.2%")
print(f"     平均跳空 {np.mean(gaps_arr):+.3%}，扣除成本后仅剩 {np.mean(gaps_arr)+round_trip:+.3%}")

# 4. 跳空幅度分布
print(f"\n  4. 跳空幅度分布 (持仓周四):")
for lo, hi, label in [
    (-0.05, -1, "跌 >1.5%"),
    (-0.015, -0.005, "跌 0.5~1.5%"),
    (-0.005, 0, "跌 0~0.5%"),
    (0, 0.005, "涨 0~0.5%"),
    (0.005, 0.015, "涨 0.5~1.5%"),
    (0.015, 0.05, "涨 >1.5%"),
]:
    n = ((pos_gaps > lo) & (pos_gaps <= hi)).sum()
    print(f"  {label:<14} {n:>3}次 ({n/len(pos_gaps):>5.0%})")

# ── 净效果: 累计算 ───────────────────────────────────────
print(f"\n═══════════════════════════════════════════════")
print(f"  结论")
print(f"═══════════════════════════════════════════════")
avg_gap = np.mean(gaps_arr)
gross_benefit = -avg_gap  # 省下的跳空损失
net_after_cost = gross_benefit - round_trip
n_thu_holdings = len(in_pos_thu)
total_benefit = n_thu_holdings * net_after_cost

print(f"  持仓周四数: {n_thu_holdings}")
print(f"  平均跳空: {avg_gap:+.4%}")
print(f"  毛收益(省下的跳空): {gross_benefit:+.4%}/次")
print(f"  扣除交易成本: {net_after_cost:+.4%}/次")
print(f"  总净收益: {total_benefit:+.2%}")
print(f"  ──────────────────────────")
print(f"  vs 策略总收益 +167.7%, 这点收益{'值得投入' if total_benefit > 0.05 else '杯水车薪'}")

# ── 额外: 周四尾盘卖出 vs 正常持有 ────────────────────────
print(f"\n  额外检查: 持仓中遇到周四→周五跳空大跌的场景")
big_gap_downs = [(p, g) for p, g in zip(in_pos_thu, pos_gaps) if g < -0.01]
print(f"  周四→周五跳空 >1% 的次数: {len(big_gap_downs)}")
if big_gap_downs:
    print(f"  最大跳空: {min(pos_gaps):+.2%}")
    print(f"  这些是\"如果躲过就赚了\"的场景，但同时也是\"如果没躲过就亏了\"的反向")

print("\nDone.")
