"""512890 vs 510880 相关性 & 波动率分析 (2019-2026)."""
import pandas as pd
import numpy as np

# ── 加载数据 ──────────────────────────────────────────────
s1 = pd.read_parquet("data/1d/512890.SH_hfq.parquet")
s2 = pd.read_parquet("data/1d/510880.SH_hfq.parquet")

s1.index = pd.to_datetime(s1.index)
s2.index = pd.to_datetime(s2.index)

cl1 = s1["Close"].loc["2019-01-01":"2026-04-30"]
cl2 = s2["Close"].loc["2019-01-01":"2026-04-30"]

# 对齐共同交易日
common_idx = cl1.index.intersection(cl2.index)
cl1, cl2 = cl1.loc[common_idx], cl2.loc[common_idx]

# 对数收益率
r1 = np.log(cl1 / cl1.shift(1)).dropna()
r2 = np.log(cl2 / cl2.shift(1)).dropna()

print(f"数据范围: {common_idx[0].date()} → {common_idx[-1].date()}")
print(f"共同交易日: {len(common_idx)} bars\n")

# ── 1. 全量相关性 ─────────────────────────────────────────
corr = r1.corr(r2)
print(f"═══════════════════════════════════════════════")
print(f"  全量相关性")
print(f"═══════════════════════════════════════════════")
print(f"  Pearson 相关系数 (日收益):  {corr:.4f}")
print(f"  R²:                         {corr**2:.4f}")

# ── 2. 波动率 ─────────────────────────────────────────────
vol1_ann = r1.std() * np.sqrt(252)
vol2_ann = r2.std() * np.sqrt(252)
print(f"\n═══════════════════════════════════════════════")
print(f"  年化波动率对比")
print(f"═══════════════════════════════════════════════")
print(f"  512890 (中证红利):  {vol1_ann:.2%}")
print(f"  510880 (上证红利):  {vol2_ann:.2%}")
print(f"  波动率差:           {vol1_ann - vol2_ann:+.2%}")
print(f"  波动率比值:         {vol1_ann / vol2_ann:.3f}x")

# ── 3. 逐年相关性 ─────────────────────────────────────────
print(f"\n═══════════════════════════════════════════════")
print(f"  逐年相关性 & 波动率")
print(f"═══════════════════════════════════════════════")
print(f"  {'Year':<6} {'Corr':>7}  {'512890 Vol':>10}  {'510880 Vol':>10}  {'Vol Diff':>9}  {'Vol Ratio':>9}")
print(f"  {'-'*6} {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}")

for yr in range(2019, 2027):
    r1y = r1.loc[f"{yr}"]
    r2y = r2.loc[f"{yr}"]
    if len(r1y) < 10:
        continue
    c = r1y.corr(r2y)
    v1 = r1y.std() * np.sqrt(252)
    v2 = r2y.std() * np.sqrt(252)
    print(f"  {yr:<6} {c:>7.4f}  {v1:>10.2%}  {v2:>10.2%}  {v1-v2:>+9.2%}  {v1/v2:>9.3f}x")

# ── 4. 滚动相关性 ─────────────────────────────────────────
roll_corr = r1.rolling(60).corr(r2)
print(f"\n═══════════════════════════════════════════════")
print(f"  滚动 60 日相关性统计 (2019-2026)")
print(f"═══════════════════════════════════════════════")
print(f"  Mean:   {roll_corr.mean():.4f}")
print(f"  Std:    {roll_corr.std():.4f}")
print(f"  Min:    {roll_corr.min():.4f}")
print(f"  Max:    {roll_corr.max():.4f}")
print(f"  Median: {roll_corr.median():.4f}")

# ── 5. 累计收益对比 ───────────────────────────────────────
cum1 = (1 + r1).cumprod()
cum2 = (1 + r2).cumprod()
print(f"\n═══════════════════════════════════════════════")
print(f"  累计收益 (2019-01 → 2026-04)")
print(f"═══════════════════════════════════════════════")
print(f"  512890:  {cum1.iloc[-1] - 1:+.2%}")
print(f"  510880:  {cum2.iloc[-1] - 1:+.2%}")
print(f"  收益差:  {(cum1.iloc[-1] - cum2.iloc[-1]):+.2%}")

# Sharpe
print(f"\n  512890 Sharpe: {(r1.mean()/r1.std())*np.sqrt(252):.3f}")
print(f"  510880 Sharpe: {(r2.mean()/r2.std())*np.sqrt(252):.3f}")

# 最大回撤
def max_dd(cum):
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return dd.min()

print(f"\n  512890 MaxDD:  {max_dd(cum1):.2%}")
print(f"  510880 MaxDD:  {max_dd(cum2):.2%}")

print("\nDone.")
