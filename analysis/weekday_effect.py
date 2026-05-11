"""512890 星期效应：周五 vs 周一~周四 平均涨跌幅."""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/1d/512890.SH_hfq.parquet")
df.index = pd.to_datetime(df.index)
close = df["Close"].loc["2019-01-01":"2026-04-30"]

ret = np.log(close / close.shift(1)).dropna()
weekday = ret.index.dayofweek  # 0=Mon, 4=Fri

print(f"数据: {ret.index[0].date()} → {ret.index[-1].date()}, {len(ret)} bars\n")

# ── 各工作日统计 ──────────────────────────────
days = ["周一", "周二", "周三", "周四", "周五"]
print(f"{'':>6}  {'均值':>8}  {'中位数':>8}  {'Std':>8}  {'胜率':>8}  {'交易日':>6}")
print(f"{'':>6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

weekday_stats = {}
for i, name in enumerate(days):
    r = ret[weekday == i]
    mean_ret = r.mean()
    win_rate = (r > 0).mean()
    weekday_stats[i] = {"mean": mean_ret, "count": len(r), "win_rate": win_rate, "std": r.std()}
    print(f"  {name:>4}:  {mean_ret:>+8.4%}  {r.median():>+8.4%}  {r.std():>8.4%}  {win_rate:>7.1%}  {len(r):>6}")

# ── 周五 vs 周一~周四 对比 ─────────────────────
fri = ret[weekday == 4]
mon_thu = ret[weekday < 4]

print(f"\n{'='*50}")
print(f"  周五 vs 周一~周四 对比")
print(f"{'='*50}")
print(f"  周五均值:       {fri.mean():+.4%}")
print(f"  周一~周四均值:  {mon_thu.mean():+.4%}")
print(f"  差值:           {fri.mean() - mon_thu.mean():+.4%}")

# t 检验
from scipy import stats
t_stat, p_val = stats.ttest_ind(fri, mon_thu, equal_var=False)
print(f"\n  Welch t-test p-value: {p_val:.4f}")
print(f"  {'差异显著' if p_val < 0.05 else '差异不显著'} (α=0.05)")

# 概率: 周五跑输周一~周四的比例
print(f"\n  周五跑输周一~周四的天数 / 周五总天数: {(fri < mon_thu.mean()).sum()} / {len(fri)}")
print(f"  即 {(fri < mon_thu.mean()).mean():.0%} 的周五低于周一~周四的平均水平")

# ── 逐年对比 ───────────────────────────────────
print(f"\n{'='*60}")
print(f"  逐年: 周五 vs 周一~周四 均值差")
print(f"{'='*60}")
for yr in range(2019, 2027):
    ry = ret.loc[f"{yr}"]
    if len(ry) < 50:
        continue
    fry = ry[ry.index.dayofweek == 4]
    mty = ry[ry.index.dayofweek < 4]
    diff = fry.mean() - mty.mean()
    mark = " ★" if diff < 0 else ""
    print(f"  {yr}: 周五{fry.mean():>+8.4%}  周一~四{mty.mean():>+8.4%}  差值{diff:>+8.4%}{mark}")

# ── 隔夜 vs 日内拆分 (如果有 Open 数据) ────────
if "Open" in df.columns:
    op = df["Open"].loc["2019-01-01":"2026-04-30"]
    common = close.index.intersection(op.index)
    op, close = op.loc[common], close.loc[common]

    overnight = np.log(op / close.shift(1))  # 隔夜(含集合竞价)
    intraday = np.log(close / op)             # 日内

    overnight = overnight.dropna()
    intraday = intraday.dropna()

    print(f"\n{'='*60}")
    print(f"  拆分: 隔夜 vs 日内 (全体交易日)")
    print(f"{'='*60}")
    for label, ser in [("隔夜", overnight), ("日内", intraday)]:
        f_sub = ser[ser.index.dayofweek == 4]
        mt_sub = ser[ser.index.dayofweek < 4]
        print(f"\n  [{label}]")
        print(f"    周五:       {f_sub.mean():+.4%}   (胜率 {(f_sub>0).mean():.0%})")
        print(f"    周一~周四:  {mt_sub.mean():+.4%}   (胜率 {(mt_sub>0).mean():.0%})")
        print(f"    差值:       {f_sub.mean() - mt_sub.mean():+.4%}")

print("\nDone.")
