"""回测验证: 周四信号跳过周五Open执行 vs 正常执行.

比较三种方案:
  A. 默认 (周四信号 → 周五Open入场)
  B. 跳过周四 (周四不生成入场信号 → 周五重评 → 周一Open入场)
  C. 延迟执行 (周四信号 → 但改为周一Open入场，不重评)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import vectorbt as vbt
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals

# ── 加载数据 ──────────────────────────────────────────────
df = pd.read_parquet("data/1d/512890.SH_hfq.parquet")
df.index = pd.to_datetime(df.index)
close = df["Close"].loc["2019-01-01":"2026-04-30"]
open_ = df["Open"].loc["2019-01-01":"2026-04-30"]

# V6 最优 Grid 参数 (仅 generate_grid_signals 接受的参数)
BEST = dict(
    base_grid_pct=0.01, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=0.8, stop_loss_grid=1.6,
    max_holding_days=45, cooldown_days=1, min_signal_strength=0.3,
    position_size=0.99, position_sizing_coef=60,
)
# 指标参数 (传给 compute_polyfit_switch_indicators)
IND_PARAMS = dict(trend_window_days=10, vol_window_days=10)

# ── 指标 ──────────────────────────────────────────────────
ind = compute_polyfit_switch_indicators(
    close, fit_window_days=252, ma_windows=[],
    trend_window_days=IND_PARAMS["trend_window_days"],
    vol_window_days=IND_PARAMS["vol_window_days"],
)
idx = ind.index
cl = close.loc[idx].values
op = open_.reindex(idx).values
wd = pd.Series(idx.weekday, index=idx).values  # weekday array
dp = ind["PolyDevPct"].values
dt = ind["PolyDevTrend"].values
vp = ind["RollingVolPct"].values
pb = ind["PolyBasePred"].values

# ══════════════════════════════════════════════════════════════
# 方案 A: 默认 (不改)
# ══════════════════════════════════════════════════════════════
e_a, x_a, s_a = generate_grid_signals(cl, dp, dt, vp, pb, **BEST)

# ══════════════════════════════════════════════════════════════
# 方案 B: 跳过周四入场 (让其周五重评 → 周一Open成交)
# ══════════════════════════════════════════════════════════════
def generate_grid_skip_thursday(cl, dp, dt, vp, pb, weekday, **params):
    """Grid 信号，但周四(weekday=3)不生成入场信号."""
    n = len(cl)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)

    in_position = False
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    cooldown = 0

    bgp = params["base_grid_pct"]
    vs = params["volatility_scale"]
    ts = params["trend_sensitivity"]
    mgl = params["max_grid_levels"]
    tpg = params["take_profit_grid"]
    slg = params["stop_loss_grid"]
    mhd = params["max_holding_days"]
    cd_p = params["cooldown_days"]
    mss = params["min_signal_strength"]
    psize = params["position_size"]
    pscoef = params["position_sizing_coef"]

    for i in range(n):
        d = dp[i]; t = dt[i]; v = vp[i]; p = pb[i]
        if np.isnan(d) or np.isnan(t) or np.isnan(v) or np.isnan(p) or p <= 0 or cl[i] <= 0:
            continue
        if not in_position:
            entry_bar = -1; entry_level = 1; entry_grid_step = np.nan
        if cooldown > 0:
            cooldown -= 1

        vol_mult = 1.0 + vs * max(v, 0.0)
        dgs = bgp * (1.0 + ts * abs(t)) * vol_mult
        dgs = max(dgs, bgp * 0.3)

        if not in_position:
            if cooldown > 0:
                continue
            # ── 周四跳过 ──
            if weekday[i] == 3:  # Thursday → next bar is Friday
                continue

            sig = abs(d) / max(dgs, 1e-9)
            el = int(np.clip(np.floor(sig), 1, mgl))
            eth = -el * dgs
            if d <= eth and sig >= mss:
                sz = float(np.clip(abs(d) * (1.0 + max(v, 0.0)) * pscoef, 0.0, psize))
                if sz > 0:
                    entries[i] = True; sizes[i] = sz
                    in_position = True; entry_bar = i; entry_level = el
                    entry_grid_step = dgs
        else:
            hd = i - entry_bar
            ref_step = max(dgs, entry_grid_step) if not np.isnan(entry_grid_step) else dgs
            tp_thresh = entry_level * ref_step * tpg
            sl_thresh = entry_level * ref_step * slg
            if hd >= mhd or d <= -sl_thresh or d >= tp_thresh:
                exits[i] = True
                in_position = False
                cooldown = cd_p

    if in_position:
        exits[-1] = True
    return entries, exits, sizes

e_b, x_b, s_b = generate_grid_skip_thursday(cl, dp, dt, vp, pb, wd, **BEST)

# ══════════════════════════════════════════════════════════════
# 方案 C: 周四信号延迟到周一Open执行 (不重评，直接改fill price)
# ══════════════════════════════════════════════════════════════
# 思路: 用方案A的信号，但把周四entry的fill price从Friday Open改为Monday Open
# 实现: 周四(weekday=3)的entry → 改成在bar[i+1] (周五)上entry → fill = 周一Open
# 即: 将 entries[周四] 移到 entries[周五]

e_c = e_a.copy()
x_c = x_a.copy()
s_c = s_a.copy()

for i in range(len(e_c) - 1):
    if e_c[i] and wd[i] == 3:  # Thursday entry → would fill at Friday Open
        e_c[i] = False          # cancel Thursday entry
        if i + 1 < len(e_c):
            e_c[i + 1] = True   # move to Friday → fills at Monday Open

# ══════════════════════════════════════════════════════════════
# 回测
# ══════════════════════════════════════════════════════════════
fill_price = pd.Series(op, index=idx).shift(-1)

def bt(name, e, x, s):
    pf = vbt.Portfolio.from_signals(
        fill_price, entries=pd.Series(e, index=idx),
        exits=pd.Series(x, index=idx), size=pd.Series(s, index=idx),
        size_type="percent", init_cash=1.0, freq="D",
    )
    stats = pf.stats()
    # 兼容不同版本的 stats key 名称
    def _s(key_part):
        for k in stats.index:
            if key_part in k:
                return stats[k]
        return 0.0
    return {
        "name": name,
        "total_return": _s("Total Return") / 100,
        "sharpe": _s("Sharpe"),
        "max_dd": _s("Max Drawdown") / 100,
        "num_trades": int(_s("Total Trades")),
        "win_rate": _s("Win Rate") / 100,
    }

results = []
for name, e, x, s in [
    ("A: 默认(周五Open入场)", e_a, x_a, s_a),
    ("B: 跳过周四(周一Open入场)", e_b, x_b, s_b),
    ("C: 延迟执行(周一Open入场)", e_c, x_c, s_c),
]:
    r = bt(name, e, x, s)
    results.append(r)

# ── 打印对比 ──────────────────────────────────────────────
print(f"{'='*75}")
print(f"  周五入场策略对比 — Polyfit-Grid 全量回测 (2019-2026)")
print(f"{'='*75}")
print(f"  {'方案':<30} {'总收益':>8} {'Sharpe':>7} {'MaxDD':>7} {'交易数':>6} {'胜率':>6}")
print(f"  {'─'*30} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")
for r in results:
    print(f"  {r['name']:<30} {r['total_return']:>+7.1%} {r['sharpe']:>7.3f} {r['max_dd']:>+7.1%} {r['num_trades']:>6} {r['win_rate']:>5.0%}")

# ── 详细对比 ──────────────────────────────────────────────
print(f"\n{'─'*75}")
print(f"  周四信号统计 (方案A)")
print(f"{'─'*75}")

# 找到周四生成的入场
thu_entries_a = e_a & (wd == 3)
n_thu_signals = thu_entries_a.sum()
print(f"  周四生成的入场信号: {n_thu_signals} 个")
print(f"  占总交易数: {n_thu_signals/max(results[0]['num_trades'],1):.0%}")

# 周四入场 vs 非周四入场的表现
fill_arr = fill_price.values
trade_returns = []
thu_trade_returns = []
non_thu_trade_returns = []

in_pos = False
entry_price = 0; entry_size = 0; is_thu_entry = False
for i in range(len(e_a)):
    if e_a[i] and not in_pos:
        entry_price = fill_arr[i] if i+1 < len(fill_arr) else cl[i]
        entry_size = s_a[i]
        is_thu_entry = (wd[i] == 3)
        in_pos = True
    elif x_a[i] and in_pos:
        exit_price = fill_arr[i] if i+1 < len(fill_arr) else cl[i]
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            trade_returns.append(ret)
            if is_thu_entry:
                thu_trade_returns.append(ret)
            else:
                non_thu_trade_returns.append(ret)
        in_pos = False

if trade_returns:
    print(f"\n  周四入场 平均收益: {np.mean(thu_trade_returns):+.2%}  胜率: {(np.array(thu_trade_returns)>0).mean():.0%}  N={len(thu_trade_returns)}")
    print(f"  非周四入场 平均收益: {np.mean(non_thu_trade_returns):+.2%}  胜率: {(np.array(non_thu_trade_returns)>0).mean():.0%}  N={len(non_thu_trade_returns)}")

    from scipy import stats as sp_stats
    if len(thu_trade_returns) > 1 and len(non_thu_trade_returns) > 1:
        t, p = sp_stats.ttest_ind(thu_trade_returns, non_thu_trade_returns, equal_var=False)
        print(f"  t-test p={p:.4f}  {'差异显著' if p < 0.05 else '不显著'}")

# ── 逐年对比 ──────────────────────────────────────────────
print(f"\n{'─'*75}")
print(f"  逐年: 方案A vs 方案B")
print(f"{'─'*75}")
print(f"  {'Year':<6} {'A 收益':>8} {'B 收益':>8} {'差值':>8} {'B 胜':>6}")
for yr in range(2019, 2027):
    mask = (pd.Series(idx).dt.year == yr).values
    if mask.sum() < 50:
        continue
    cl_y = cl[mask]; op_y = op[mask]
    fy = pd.Series(op_y, index=idx[mask]).shift(-1)

    for label, e, x, s in [("A", e_a[mask], x_a[mask], s_a[mask]),
                             ("B", e_b[mask], x_b[mask], s_b[mask])]:
        pf = vbt.Portfolio.from_signals(
            fy, entries=pd.Series(e, index=idx[mask]),
            exits=pd.Series(x, index=idx[mask]),
            size=pd.Series(s, index=idx[mask]),
            size_type="percent", init_cash=1.0, freq="D",
        )
        st = pf.stats()
        ret = next(v for k, v in st.items() if "Total Return" in str(k)) / 100
        if label == "A": ret_a = ret
        else: ret_b = ret

    winner = "B ✓" if ret_b > ret_a else "A  "
    print(f"  {yr:<6} {ret_a:>+7.1%} {ret_b:>+7.1%} {ret_b-ret_a:>+7.1%} {winner:>6}")

print("\nDone.")
