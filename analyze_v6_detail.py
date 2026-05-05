"""
直接调用 v6 OHLCV增强版 Switch，逐笔输出被阻拦/通过的交易及盈亏。
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import (
    generate_grid_priority_switch_signals,
    generate_grid_priority_switch_signals_v6,
    _compute_entry_ohlcv_features,
    _check_entry_filters,
)

GRID_PARAMS = dict(
    base_grid_pct=0.008, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=1.0, stop_loss_grid=1.2,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=0.45, position_size=0.92, position_sizing_coef=30,
)

SW_PARAMS_V5 = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0003,
                    trend_atr_mult=2.5, trend_atr_window=14,
                    trend_vol_climax=3.0, trend_decline_days=3)
# v6 使用自己的优化默认值 (atr_mult=2.0, decline_days=2)，只传 entry 参数
SW_PARAMS_V6 = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0003)

data = load_data("data/1d/512890.SH_hfq.parquet")
close = data["Close"]; open_ = data["Open"]
high = data["High"]; low = data["Low"]; volume = data["Volume"]

windows = generate_monthly_windows(
    close.index, train_months=22, test_months=12, step_months=3, warmup_months=12)
print(f"数据: {len(close)} bars, WF窗口: {len(windows)} 个\n")

# ── 收集所有 Switch 候选 bar 的特征和结果 ──
all_checked = []   # 每个被检查的 bar
all_trades = []    # 实际执行的交易

for wi, w in enumerate(windows):
    cwa = close.iloc[w.warmup_start:w.test_end]
    owa = open_.iloc[w.warmup_start:w.test_end]
    toff = w.test_start - w.warmup_start
    tsl = cwa.index[toff]
    ind = compute_polyfit_switch_indicators(cwa, fit_window_days=252, ma_windows=[20,60],
                                             trend_window_days=10, vol_window_days=20)
    it = ind.loc[ind.index >= tsl]
    if len(it) < 10: continue

    cl = it.index; ca = cwa.loc[cl].values; oa = owa.reindex(cl).values
    ha = high.reindex(cl).values; la = low.reindex(cl).values
    va = volume.reindex(cl).values
    dpa = it["PolyDevPct"].values; dta = it["PolyDevTrend"].values
    vpa = it["RollingVolPct"].values; pba = it["PolyBasePred"].values
    m20a = it["MA20"].values; m60a = it["MA60"].values

    eg, xg, sg = generate_grid_signals(ca, dpa, dta, vpa, pba, **GRID_PARAMS)

    r5 = generate_grid_priority_switch_signals(
        ca, dpa, dta, vpa, pba, eg, xg, m20a, m60a,
        high=ha, low=la, volume=va, **SW_PARAMS_V5,
    )
    e_sw5 = r5[0]; x_sw5 = r5[1]

    r6 = generate_grid_priority_switch_signals_v6(
        ca, dpa, dta, vpa, pba, eg, xg, m20a, m60a,
        high=ha, low=la, open_=oa, volume=va,
        return_filter_stats=True, **SW_PARAMS_V6,
    )
    e_sw6 = r6["sw_entries"]; x_sw6 = r6["sw_exits"]

    # vol_ema for feature extraction
    vol_ema_arr = np.zeros(len(ca), dtype=np.float64)
    alpha_v = 2.0/21.0; ve = float(va[0]) if not np.isnan(va[0]) else 0.0
    vol_ema_arr[0] = ve
    for i in range(1, len(ca)):
        if not np.isnan(va[i]): ve = alpha_v*float(va[i]) + (1-alpha_v)*ve
        vol_ema_arr[i] = ve

    # ── 遍历所有 v5 Switch 入场 bar，检查 v6 的特征和过滤结果 ──
    v5_entry_bars = set(int(b) for b in np.where(e_sw5)[0])
    v6_entry_bars = set(int(b) for b in np.where(e_sw6)[0])

    for bar in sorted(v5_entry_bars):
        if bar < 10: continue
        feat = _compute_entry_ohlcv_features(ca, oa, ha, la, va, vol_ema_arr, bar)
        passed, reason = _check_entry_filters(feat, SW_PARAMS_V6["trend_entry_dp"],
                                               SW_PARAMS_V6["trend_confirm_dp_slope"])
        # find exit
        exit_bar = -1; exit_pnl = 0.0
        for j in range(bar+1, len(ca)):
            if x_sw5[j]:
                exit_bar = j; break
        if exit_bar < 0: exit_bar = len(ca)-1
        exit_pnl = ca[exit_bar]/ca[bar] - 1

        all_checked.append({
            "window": wi,
            "bar": bar,
            "entry_date": cl[bar],
            "close": ca[bar],
            "entry_day_ret": feat["entry_day_ret"],
            "consecutive_up": feat["consecutive_up"],
            "upper_wick_pct": feat["upper_wick_pct"],
            "rel_vol": feat["rel_vol"],
            "price_position": feat["price_position"],
            "is_green": feat["is_green"],
            "dp": dpa[bar],
            "passed": passed,
            "reject_reason": reason if not passed else "",
            "in_v6": bar in v6_entry_bars,
            "exit_bar": exit_bar,
            "pnl": exit_pnl,
        })

    # ── 收集 v6 实际交易 ──
    for bar in sorted(v6_entry_bars):
        exit_bar = -1
        for j in range(bar+1, len(ca)):
            if x_sw6[j]: exit_bar = j; break
        if exit_bar < 0: exit_bar = len(ca)-1
        pnl = ca[exit_bar]/ca[bar] - 1
        feat = _compute_entry_ohlcv_features(ca, oa, ha, la, va, vol_ema_arr, bar)
        all_trades.append({
            "window": wi, "entry_date": cl[bar], "entry_bar": bar,
            "exit_bar": exit_bar, "pnl": pnl, "holding": exit_bar-bar,
            "entry_day_ret": feat["entry_day_ret"],
            "consecutive_up": feat["consecutive_up"],
            "rel_vol": feat["rel_vol"], "price_position": feat["price_position"],
            "is_green": feat["is_green"], "dp": dpa[bar],
        })

df = pd.DataFrame(all_checked)
tdf = pd.DataFrame(all_trades)

# ══════════════════════════════════════════════════════════════════
print("═" * 100)
print("  v6 OHLCV 增强版 Switch — 逐笔分析")
print("═" * 100)

# ── 1. 过滤概览 ──
n_total = len(df)
n_blocked = (~df["passed"]).sum()
n_passed = df["passed"].sum()
n_in_v6 = df["in_v6"].sum()
n_blocked_but_in_v5 = n_total - n_in_v6

print(f"\n  v5 Switch 入场总数: {n_total}")
print(f"  v6 过滤通过:        {n_passed}")
print(f"  v6 过滤拒绝:        {n_blocked}  ({n_blocked/n_total*100:.1f}%)")
print(f"  v6 实际入场:        {n_in_v6}")
print(f"  (通过但因 Grid 状态未入场: {n_passed - n_in_v6})")
print(f"  有效阻拦:           {n_blocked_but_in_v5} 笔")

# ── 2. 被阻拦交易盈亏 ──
blocked = df[~df["passed"]]
print(f"\n{'─' * 100}")
print(f"  被阻拦的 {len(blocked)} 笔交易盈亏分布")
print(f"{'─' * 100}")
print(f"  盈利(>0):  {(blocked['pnl']>0).sum():>3d}  ({(blocked['pnl']>0).mean()*100:.1f}%)")
print(f"  亏本(<0):  {(blocked['pnl']<0).sum():>3d}  ({(blocked['pnl']<0).mean()*100:.1f}%)")
print(f"  平均盈亏:  {blocked['pnl'].mean():>+.2%}")
print(f"  中位盈亏:  {blocked['pnl'].median():>+.2%}")
print(f"  最亏:      {blocked['pnl'].min():>+.2%}")
print(f"  最赚:      {blocked['pnl'].max():>+.2%}")

# ── 3. 按阻拦原因分组 ──
print(f"\n{'─' * 100}")
print(f"  按阻拦原因分组 — 被阻拦交易的盈亏")
print(f"{'─' * 100}")
print(f"  {'阻拦原因':<30s} {'阻拦数':>6s} {'亏本':>6s} {'亏本率':>8s} {'平均盈亏':>9s} {'中位盈亏':>9s}")
print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*8} {'─'*9} {'─'*9}")

reason_labels = {
    "micro_up": "微涨0~+0.5% (胜率12.5%)",
    "long_upper_wick": "长上影≥30% (胜率18.4%)",
    "normal_volume": "正常量能0.8-1.5x (胜率13.7%)",
    "green_without_momentum": "阳线+连涨<4天 (胜率23.8%)",
    "high_position": "高位≥70%+连涨<4天 (胜率32.1%)",
}
for reason, label in reason_labels.items():
    sub = blocked[blocked["reject_reason"] == reason]
    if len(sub) == 0: continue
    print(f"  {label:<30s} {len(sub):>6d} {(sub['pnl']<0).sum():>6d} "
          f"{(sub['pnl']<0).mean()*100:>7.1f}% {sub['pnl'].mean():>+8.2%} {sub['pnl'].median():>+8.2%}")

# ── 4. 通过过滤的入场 vs 被阻拦的入场 ──
passed_df = df[df["passed"]]
print(f"\n{'─' * 100}")
print(f"  通过 vs 被阻拦 — 特征对比")
print(f"{'─' * 100}")
print(f"  {'':>20s} {'通过(n='+str(len(passed_df))+')':>18s} {'阻拦(n='+str(len(blocked))+')':>18s}")
print(f"  {'─'*20} {'─'*18} {'─'*18}")
for col, label, fmt in [
    ("entry_day_ret", "入场日涨跌", "+.2%"), ("consecutive_up", "连涨天数", ".1f"),
    ("upper_wick_pct", "上影线占比", ".1%"), ("rel_vol", "相对量能", ".2f"),
    ("price_position", "价格位置", ".1%"), ("dp", "偏离度dp", ".3f"),
]:
    print(f"  {label:<20s} {passed_df[col].mean():{fmt}}             {blocked[col].mean():{fmt}}")

# ── 5. 逐笔明细（前30笔）──
print(f"\n{'─' * 100}")
print(f"  被阻拦交易明细 (前30笔)")
print(f"{'─' * 100}")
print(f"  {'日期':>12s} {'收盘':>8s} {'涨跌':>7s} {'连涨':>4s} {'上影':>6s} "
      f"{'量比':>6s} {'价位':>6s} {'阳线':>4s} {'dp':>7s} {'阻拦原因':>22s} {'盈亏':>7s}")
print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*7} {'─'*22} {'─'*7}")
for _, r in blocked.head(30).iterrows():
    print(f"  {str(r['entry_date'])[:10]:>12s} {r['close']:>8.3f} {r['entry_day_ret']:>+6.2%} "
          f"{r['consecutive_up']:>4.0f} {r['upper_wick_pct']:>5.1%} {r['rel_vol']:>5.2f} "
          f"{r['price_position']:>5.1%} {'Y' if r['is_green'] else 'N':>4s} "
          f"{r['dp']:>+6.3f} {r['reject_reason']:<22s} {r['pnl']:>+6.2%}")

# ── 6. v6 实际交易表现 ──
print(f"\n{'═' * 100}")
print(f"  v6 实际执行的 {len(tdf)} 笔交易")
print(f"{'═' * 100}")
print(f"  盈利: {(tdf['pnl']>0).sum()} ({(tdf['pnl']>0).mean()*100:.1f}%)  "
      f"亏损: {(tdf['pnl']<0).sum()} ({(tdf['pnl']<0).mean()*100:.1f}%)")
print(f"  平均盈亏: {tdf['pnl'].mean():+.2%}  中位: {tdf['pnl'].median():+.2%}")
print(f"  平均持仓: {tdf['holding'].mean():.1f} 天")
print(f"  平均入场涨跌: {tdf['entry_day_ret'].mean():+.2%}")
print(f"  平均连涨: {tdf['consecutive_up'].mean():.1f} 天")

print(f"\n  ── 按连涨天数 ──")
for cons in sorted(tdf["consecutive_up"].unique()):
    sub = tdf[tdf["consecutive_up"]==cons]
    print(f"  连涨{int(cons):>1d}天: {len(sub):>3d}笔  胜率{(sub['pnl']>0).mean()*100:>5.1f}%  盈亏{sub['pnl'].mean():>+7.2%}  持仓{sub['holding'].mean():>5.1f}天")

print(f"\n  ── 按阳/阴线 ──")
for label, cond in [("阴线", ~tdf["is_green"]), ("阳线", tdf["is_green"])]:
    sub = tdf[cond]
    if len(sub)==0: continue
    print(f"  {label}: {len(sub):>3d}笔  胜率{(sub['pnl']>0).mean()*100:>5.1f}%  盈亏{sub['pnl'].mean():>+7.2%}  持仓{sub['holding'].mean():>5.1f}天")

print(f"\n  ── 按量能 ──")
for label, cond in [("缩量<0.8", tdf["rel_vol"]<0.8), ("放量>1.5", tdf["rel_vol"]>1.5)]:
    sub = tdf[cond]
    if len(sub)==0: continue
    print(f"  {label}: {len(sub):>3d}笔  胜率{(sub['pnl']>0).mean()*100:>5.1f}%  盈亏{sub['pnl'].mean():>+7.2%}")

# ── 7. 总结 ──
print(f"\n{'═' * 100}")
print(f"  总结")
print(f"{'═' * 100}")

# 如果被阻拦交易都执行了，v5收益
v5_total_pnl = df["pnl"].sum()
v6_total_pnl = tdf["pnl"].sum() if len(tdf) > 0 else 0
blocked_total_pnl = blocked["pnl"].sum()

print(f"  v5 全部 {n_total} 笔 Switch 交易合计盈亏: {v5_total_pnl:+.2%}")
print(f"  其中被阻拦 {len(blocked)} 笔合计盈亏:     {blocked_total_pnl:+.2%}")
print(f"  v6 实际 {len(tdf)} 笔 Switch 交易合计盈亏: {v6_total_pnl:+.2%}")
if len(tdf) > 0:
    print(f"  v6 每笔平均盈亏: {tdf['pnl'].mean():+.2%}  vs  阻拦每笔平均: {blocked['pnl'].mean():+.2%}")
print(f"")
print(f"  阻拦的 {len(blocked)} 笔中，亏本的 {(blocked['pnl']<0).sum()} 笔 ({(blocked['pnl']<0).mean()*100:.1f}%)")
print(f"  阻拦的 {len(blocked)} 笔中，赚钱的 {(blocked['pnl']>0).sum()} 笔 ({(blocked['pnl']>0).mean()*100:.1f}%)")
