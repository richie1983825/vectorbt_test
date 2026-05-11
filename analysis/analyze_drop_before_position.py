"""
分析：大跌前满仓情况 — 结合 OHLCV 寻找可规避的信号。

大跌定义：单日跌幅 < -2%，且当时 dp < 0（基线之下）
满仓定义：V6 策略大跌当日处于 Grid 或 Switch 持仓

目标：检查大跌前 1-5 天的 OHLCV 特征，寻找可预测大跌并提前离场的规律。
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6

V6_GRID = dict(trend_window_days=10, vol_window_days=10, base_grid_pct=0.01,
    volatility_scale=0.0, trend_sensitivity=4, max_grid_levels=3, take_profit_grid=0.8,
    stop_loss_grid=1.6, max_holding_days=45, cooldown_days=1, min_signal_strength=0.3,
    position_size=0.99, position_sizing_coef=60)
V6_SWITCH = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_atr_window=14, trend_vol_climax=2.5, trend_decline_days=1,
    enable_ohlcv_filter=True, enable_early_exit=True)

DROP_THRESHOLD = -0.02

print("Loading data…")
data = load_data("data/1d/512890.SH_hfq.parquet")
close = data["Close"]; open_ = data["Open"]
high = data["High"]; low = data["Low"]; volume = data["Volume"]

ind = compute_polyfit_switch_indicators(close, fit_window_days=252, ma_windows=[20, 60],
                                         trend_window_days=10, vol_window_days=10)
idx = ind.index
cl = close.loc[idx]; op = open_.reindex(idx)
hi = high.reindex(idx); lo = low.reindex(idx); vol = volume.reindex(idx)

dp = ind["PolyDevPct"].values; dt = ind["PolyDevTrend"].values
vp = ind["RollingVolPct"].values; pb = ind["PolyBasePred"].values

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

# 模拟仓位
n = len(cl)
grid_in = np.zeros(n, dtype=bool); sw_in = np.zeros(n, dtype=bool)
g_in = False; s_in = False
for i in range(n):
    if g_in and x_g[i]: g_in = False
    if s_in and x_s[i]: s_in = False
    if e_g[i]:
        if s_in: s_in = False
        g_in = True; grid_in[i] = True; continue
    if not g_in:
        if e_s[i]: s_in = True; sw_in[i] = True
    grid_in[i] = g_in; sw_in[i] = s_in
in_position = grid_in | sw_in

# 预计算滚动指标
daily_ret = cl.pct_change().values
vol_ema = vol.ewm(span=20, min_periods=1).mean().values
vol_ratio = vol.values / np.maximum(vol_ema, 1e-9)

# 20日高低位
h20 = hi.rolling(20).max().values
l20 = lo.rolling(20).min().values
price_pos_20 = np.where(h20 > l20, (cl.values - l20) / (h20 - l20), 0.5)

# 5日涨幅
ret_5d = np.zeros(n)
for i in range(5, n):
    ret_5d[i] = cl.values[i] / cl.values[i-5] - 1.0

below_base = dp < 0
big_drop = daily_ret < DROP_THRESHOLD
big_drop_below_pos = big_drop & below_base & in_position

drop_idx = np.where(big_drop_below_pos)[0]

print(f"\n{'═' * 80}")
print(f"  大跌（<{DROP_THRESHOLD:.0%}）+ 基线之下 + 满仓：{len(drop_idx)} 次")
print(f"{'═' * 80}")

# ── 逐次分析 ──
print(f"\n  {'日期':<12s} {'跌幅':>7s} {'dp':>7s} {'持仓':>5s} {'5d涨':>7s} {'价位':>6s} "
      f"{'量比':>6s} {'上影':>6s} {'连涨':>4s} {'前1d':>7s} {'前2d':>7s} {'后1d':>7s} {'后3d':>7s}")

records = []
for i in drop_idx:
    # 大跌前特征
    pre_ret_5d = ret_5d[i]  # 含大跌当日
    pre_ret_5d_ex = cl.values[max(0,i-1)] / cl.values[max(0,i-6)] - 1.0 if i >= 6 else 0  # 不含当日
    pre_price_pos = price_pos_20[max(0,i-1)]  # 前一日价位

    # 前1天 OHLCV
    if i >= 1:
        pre1_ret = daily_ret[i-1]
        pre1_upper_wick = (hi.values[i-1] - max(cl.values[i-1], op.values[i-1])) / max(hi.values[i-1] - lo.values[i-1], 1e-9)
        pre1_amplitude = (hi.values[i-1] - lo.values[i-1]) / cl.values[i-1]
        pre1_is_green = cl.values[i-1] >= op.values[i-1]
        pre1_vol_ratio = vol_ratio[i-1]
    else:
        pre1_ret = pre1_upper_wick = pre1_amplitude = pre1_is_green = pre1_vol_ratio = np.nan

    # 连涨天数（大跌前）
    cons_up = 0
    for j in range(i-1, max(0,i-10), -1):
        if cl.values[j] > cl.values[j-1]:
            cons_up += 1
        else:
            break

    # 大跌前振幅趋势
    amp_3d = np.mean([(hi.values[j]-lo.values[j])/cl.values[j] for j in range(max(0,i-3), i) if j > 0])

    # dp 趋势
    dp_change_3d = dp[i-1] - dp[max(0,i-4)] if i >= 4 else 0

    # 大跌日 OHLCV
    drop_upper_wick = (hi.values[i] - max(cl.values[i], op.values[i])) / max(hi.values[i] - lo.values[i], 1e-9)
    drop_lower_wick = (min(cl.values[i], op.values[i]) - lo.values[i]) / max(hi.values[i] - lo.values[i], 1e-9)
    drop_amplitude = (hi.values[i] - lo.values[i]) / cl.values[i]
    drop_vol_ratio = vol_ratio[i]

    # 后续
    next_ret = daily_ret[min(i+1, n-1)] if i+1 < n else 0
    ret_3d = cl.values[min(i+3, n-1)] / cl.values[i] - 1 if i+3 < n else 0

    mode = "Grid" if grid_in[i] else "Switch"
    holding = 0
    for j in range(i-1, -1, -1):
        if not in_position[j]:
            break
        holding += 1

    if np.isnan(pre1_upper_wick): pre1_upper_wick = 0
    if np.isnan(pre1_vol_ratio): pre1_vol_ratio = 1.0

    print(f"  {str(idx[i].date()):<12s} {daily_ret[i]:>+7.2%} {dp[i]:>+7.1%} {mode:>5s} "
          f"{pre_ret_5d_ex:>+7.1%} {pre_price_pos:>5.1%} {pre1_vol_ratio:>5.1f}x "
          f"{pre1_upper_wick:>5.0%} {cons_up:>4d} "
          f"{pre1_ret:>+7.2%} {'':>7s} {next_ret:>+7.2%} {ret_3d:>+7.2%}")

    records.append({
        "date": idx[i], "drop_ret": daily_ret[i], "dp": dp[i],
        "mode": mode, "holding_days": holding,
        "pre_ret_5d": pre_ret_5d_ex,
        "pre_price_pos": pre_price_pos,
        "pre1_ret": pre1_ret, "pre1_upper_wick": pre1_upper_wick,
        "pre1_amplitude": pre1_amplitude, "pre1_is_green": pre1_is_green,
        "pre1_vol_ratio": pre1_vol_ratio,
        "cons_up_before": cons_up,
        "amp_3d_mean": amp_3d,
        "dp_change_3d": dp_change_3d,
        "drop_upper_wick": drop_upper_wick,
        "drop_lower_wick": drop_lower_wick,
        "drop_amplitude": drop_amplitude,
        "drop_vol_ratio": drop_vol_ratio,
        "next_ret": next_ret, "ret_3d": ret_3d,
    })

df = pd.DataFrame(records)

# ── 汇总统计 ──
print(f"\n{'═' * 80}")
print(f"  大跌前 OHLCV 特征汇总")
print(f"{'═' * 80}")

print(f"\n  ── 大跌前 1 天特征 ──")
print(f"  前1天收阳: {(df['pre1_is_green']==True).sum()}/{len(df)} ({(df['pre1_is_green']==True).mean()*100:.0f}%)")
print(f"  前1天收阴: {(df['pre1_is_green']==False).sum()}/{len(df)} ({(df['pre1_is_green']==False).mean()*100:.0f}%)")
print(f"  前1天上影>30%: {(df['pre1_upper_wick']>=0.30).sum()}/{len(df)} ({(df['pre1_upper_wick']>=0.30).mean()*100:.0f}%)")
print(f"  前1天涨幅均值: {df['pre1_ret'].mean():+.2%}")
print(f"  前1天振幅均值: {df['pre1_amplitude'].mean():.2%}")

print(f"\n  ── 大跌前价位/涨幅 ──")
print(f"  前1天价位>80%: {(df['pre_price_pos']>0.8).sum()}/{len(df)} ({(df['pre_price_pos']>0.8).mean()*100:.0f}%)")
print(f"  前1天价位>70%: {(df['pre_price_pos']>0.7).sum()}/{len(df)} ({(df['pre_price_pos']>0.7).mean()*100:.0f}%)")
print(f"  前5天涨幅>5%: {(df['pre_ret_5d']>0.05).sum()}/{len(df)} ({(df['pre_ret_5d']>0.05).mean()*100:.0f}%)")
print(f"  前5天涨幅>3%: {(df['pre_ret_5d']>0.03).sum()}/{len(df)} ({(df['pre_ret_5d']>0.03).mean()*100:.0f}%)")
print(f"  连涨>=3天: {(df['cons_up_before']>=3).sum()}/{len(df)} ({(df['cons_up_before']>=3).mean()*100:.0f}%)")
print(f"  连涨>=2天: {(df['cons_up_before']>=2).sum()}/{len(df)} ({(df['cons_up_before']>=2).mean()*100:.0f}%)")

print(f"\n  ── 大跌前量能 ──")
print(f"  量比>1.5x: {(df['pre1_vol_ratio']>1.5).sum()}/{len(df)} ({(df['pre1_vol_ratio']>1.5).mean()*100:.0f}%)")
print(f"  量比>1.2x: {(df['pre1_vol_ratio']>1.2).sum()}/{len(df)} ({(df['pre1_vol_ratio']>1.2).mean()*100:.0f}%)")
print(f"  量比<0.8x: {(df['pre1_vol_ratio']<0.8).sum()}/{len(df)} ({(df['pre1_vol_ratio']<0.8).mean()*100:.0f}%)")

print(f"\n  ── 大跌前 dp 趋势 ──")
print(f"  dp 3日变化>+2%: {(df['dp_change_3d']>0.02).sum()}/{len(df)} ({(df['dp_change_3d']>0.02).mean()*100:.0f}%)")
print(f"  dp 3日变化>0: {(df['dp_change_3d']>0).sum()}/{len(df)} ({(df['dp_change_3d']>0).mean()*100:.0f}%)")
print(f"  dp 3日变化<0: {(df['dp_change_3d']<0).sum()}/{len(df)} ({(df['dp_change_3d']<0).mean()*100:.0f}%)")
print(f"  dp 3日变化均值: {df['dp_change_3d'].mean():+.2%}")

print(f"\n  ── 大跌日 OHLCV ──")
print(f"  上影>30%: {(df['drop_upper_wick']>=0.30).sum()}/{len(df)}")
print(f"  下影>30%: {(df['drop_lower_wick']>=0.30).sum()}/{len(df)}")
print(f"  振幅>3%: {(df['drop_amplitude']>0.03).sum()}/{len(df)}")
print(f"  量比>1.5x: {(df['drop_vol_ratio']>1.5).sum()}/{len(df)}")

print(f"\n  ── 大跌后走势 ──")
print(f"  次日反弹（涨）: {(df['next_ret']>0).sum()}/{len(df)} ({(df['next_ret']>0).mean()*100:.0f}%)")
print(f"  次日继续跌: {(df['next_ret']<0).sum()}/{len(df)} ({(df['next_ret']<0).mean()*100:.0f}%)")
print(f"  3日反弹（涨）: {(df['ret_3d']>0).sum()}/{len(df)} ({(df['ret_3d']>0).mean()*100:.0f}%)")
print(f"  3日继续跌: {(df['ret_3d']<0).sum()}/{len(df)} ({(df['ret_3d']<0).mean()*100:.0f}%)")

# ── 规避策略模拟 ──
print(f"\n{'═' * 80}")
print(f"  潜在规避策略效果模拟")
print(f"{'═' * 80}")

# 测试各种规避条件
strategies = [
    ("高位+连涨: 价位>70% & 连涨>=2天 & 前1天收阳",
     lambda d: (d['pre_price_pos'] > 0.70) & (d['cons_up_before'] >= 2) & (d['pre1_is_green'] == True)),
    ("高位+连涨: 价位>80% & 连涨>=2天 & 前1天收阳",
     lambda d: (d['pre_price_pos'] > 0.80) & (d['cons_up_before'] >= 2) & (d['pre1_is_green'] == True)),
    ("5d急涨+高位: 5日涨>3% & 价位>70%",
     lambda d: (d['pre_ret_5d'] > 0.03) & (d['pre_price_pos'] > 0.70)),
    ("5d急涨+高位+收阳: 5日涨>3% & 价位>70% & 前1天收阳",
     lambda d: (d['pre_ret_5d'] > 0.03) & (d['pre_price_pos'] > 0.70) & (d['pre1_is_green'] == True)),
    ("连涨+放量: 连涨>=2天 & 量比>1.2x",
     lambda d: (d['cons_up_before'] >= 2) & (d['pre1_vol_ratio'] > 1.2)),
    ("高位+上影: 价位>70% & 上影>30%",
     lambda d: (d['pre_price_pos'] > 0.70) & (d['pre1_upper_wick'] >= 0.30)),
    ("前1天收阴: 大跌前已收阴",
     lambda d: (d['pre1_is_green'] == False)),
    ("dp 3日下降: dp_change_3d < 0",
     lambda d: (d['dp_change_3d'] < 0)),
]

for name, condition in strategies:
    mask = condition(df)
    n_match = mask.sum()
    if n_match == 0:
        print(f"\n  {name}: 0/{len(df)} — 无法评估")
        continue

    # 如果提前离场，可以避免多少损失
    avg_drop = df.loc[mask, "drop_ret"].mean()
    avg_recovery = df.loc[mask, "next_ret"].mean()
    avg_recovery_3d = df.loc[mask, "ret_3d"].mean()

    # 规避收益 = 避免了大跌 + 可以在更低位置买回
    # 简化：如果次日反弹，则规避的价值 = 大跌 + 次日反弹（假设次日抄底）
    avoided_loss = -df.loc[mask, "drop_ret"].mean()  # 避免了当日的跌幅
    # 如果次日开盘买入，获得次日涨幅
    total_benefit = avoided_loss + df.loc[mask, "next_ret"].mean()

    print(f"\n  {name}")
    print(f"    命中: {n_match}/{len(df)} ({n_match/len(df)*100:.0f}%)")
    print(f"    平均大跌: {avg_drop:+.2%}  次日反弹均值: {avg_recovery:+.2%}  3日反弹均值: {avg_recovery_3d:+.2%}")
    print(f"    规避收益(避跌+次日买): {total_benefit:+.2%}")

    # 误报分析：在大跌前没有发生的满仓日，此条件触发的频率
    false_positive_days = 0
    for i in range(20, n):
        if in_position[i] and not big_drop_below_pos[i]:
            pre_price_pos_i = price_pos_20[max(0,i-1)]
            pre_ret_5d_i = ret_5d[max(0,i-1)]
            pre1_is_green_i = cl.values[max(0,i-1)] >= op.values[max(0,i-1)]
            pre1_upper_wick_i = 0
            if i >= 1:
                pre1_upper_wick_i = (hi.values[i-1] - max(cl.values[i-1], op.values[i-1])) / max(hi.values[i-1] - lo.values[i-1], 1e-9)
            pre1_vol_i = vol_ratio[max(0,i-1)]
            cons_up_i = 0
            for j in range(i-1, max(0,i-10), -1):
                if cl.values[j] > cl.values[j-1]: cons_up_i += 1
                else: break
            dp_chg = dp[max(0,i-1)] - dp[max(0,i-4)] if i >= 4 else 0

            # Check condition for this day
            try:
                row = pd.Series({
                    'pre_price_pos': pre_price_pos_i, 'pre_ret_5d': pre_ret_5d_i,
                    'pre1_is_green': pre1_is_green_i, 'pre1_upper_wick': pre1_upper_wick_i,
                    'pre1_vol_ratio': pre1_vol_i, 'cons_up_before': cons_up_i,
                    'dp_change_3d': dp_chg,
                })
                if condition(row):
                    false_positive_days += 1
            except:
                pass

    total_position_days = in_position[20:].sum()
    fp_rate = false_positive_days / max(total_position_days, 1)
    print(f"    误报率(满仓日触发%): {fp_rate*100:.1f}% ({false_positive_days}/{total_position_days})")
