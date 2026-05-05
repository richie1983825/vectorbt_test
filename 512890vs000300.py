"""512890 vs 000300 + Polyfit-Grid (return) equity curve."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import vectorbt as vbt

from utils.indicators import compute_polyfit_base_only, add_trend_vol_indicators
from strategies.polyfit_grid import generate_grid_signals

# ═══════════════════════════════════════════════════════
# Load price data
# ═══════════════════════════════════════════════════════
df_512890 = pd.read_parquet("data/1d/512890.SH_hfq.parquet")
df_000300 = pd.read_parquet("data/1d/000300.SH_hfq.parquet")
df_512890.index = pd.to_datetime(df_512890.index)
df_000300.index = pd.to_datetime(df_000300.index)

close_512890 = df_512890["Close"]
open_512890 = df_512890["Open"]

# ═══════════════════════════════════════════════════════
# Load best Grid-return params from last WF run
# ═══════════════════════════════════════════════════════
wf = pd.read_csv("reports/wf_comparison.csv")
best = wf[(wf["strategy"] == "Polyfit-Grid") &
          (wf["selector"] == "return")].nlargest(1, "test_return").iloc[0]

tw = int(best["trend_window_days"])
vw = int(best["vol_window_days"])
bgp = best["base_grid_pct"]
vs  = best["volatility_scale"]
ts  = best["trend_sensitivity"]
mgl = int(best["max_grid_levels"])
tpg = best["take_profit_grid"]
slg = best["stop_loss_grid"]
mss = best["min_signal_strength"]
ps  = best["position_size"]
psc = best["position_sizing_coef"]

# ═══════════════════════════════════════════════════════
# Generate Grid signals on full data
# ═══════════════════════════════════════════════════════
base = compute_polyfit_base_only(close_512890, fit_window_days=252)
ind  = add_trend_vol_indicators(base, close_512890, trend_window_days=tw,
                                 vol_window_days=vw)
grid_idx = ind.index                         # <- Grid 的有效时间轴
cl = close_512890.loc[grid_idx]
op = open_512890.reindex(grid_idx)

e, x, s = generate_grid_signals(
    cl.values,
    ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
    ind["RollingVolPct"].values, ind["PolyBasePred"].values,
    base_grid_pct=bgp, volatility_scale=vs, trend_sensitivity=ts,
    max_grid_levels=mgl, take_profit_grid=tpg, stop_loss_grid=slg,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=mss, position_size=ps, position_sizing_coef=psc,
)

# ═══════════════════════════════════════════════════════
# VectorBT backtest → extract equity curve
# ═══════════════════════════════════════════════════════
fill_price = op.shift(-1).reindex(grid_idx)
pf = vbt.Portfolio.from_signals(
    fill_price,
    entries=pd.Series(e, index=grid_idx),
    exits=pd.Series(x, index=grid_idx),
    size=pd.Series(s, index=grid_idx),
    size_type="percent",
    init_cash=1.0,
    freq="D",
)
eq = pf.value()  # portfolio value, indexed by grid_idx

# ═══════════════════════════════════════════════════════
# Normalise everything to start at 1.0 on grid_idx
# ═══════════════════════════════════════════════════════
norm_512890 = cl / cl.iloc[0]
norm_000300 = df_000300["Close"].reindex(grid_idx).dropna() / \
              df_000300["Close"].reindex(grid_idx).dropna().iloc[0]
eq_norm = eq / eq.iloc[0]

# ═══════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(18, 8))

c1, c2, c3 = "#1a5276", "#c0392b", "#27ae60"

ax1.plot(grid_idx, norm_512890, color=c1, linewidth=0.8, alpha=0.7,
         label="512890 Close (norm)")
ax1.plot(grid_idx, eq_norm, color=c3, linewidth=1.0,
         label="Polyfit-Grid return (norm)")

ax1.set_ylabel("Normalised price / equity", fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

ax2 = ax1.twinx()
ax2.plot(grid_idx, norm_000300, color=c2, linewidth=0.8,
         label="000300 Close (norm)")
ax2.set_ylabel("000300 Close (norm)", color=c2, fontsize=11)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

ax1.set_title("512890 vs 000300 + Polyfit-Grid (return) Equity Curve",
              fontsize=14, fontweight="bold")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax1.set_xlabel("")
ax1.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("reports/512890vs000300.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════
ret_512890 = cl.iloc[-1] / cl.iloc[0] - 1
ret_000300 = norm_000300.iloc[-1] - 1
ret_grid   = eq.iloc[-1] - 1
years      = (grid_idx[-1] - grid_idx[0]).days / 365.25
cagr_grid  = (1 + ret_grid) ** (1 / years) - 1

print(f"Period:     {grid_idx[0].date()} → {grid_idx[-1].date()}  ({len(grid_idx)} bars, {years:.1f}y)")
print(f"512890:    {ret_512890:+.1%}  (CAGR {(1+ret_512890)**(1/years)-1:+.1%})")
print(f"000300:    {ret_000300:+.1%}  (CAGR {(1+ret_000300)**(1/years)-1:+.1%})")
print(f"Grid(ret): {ret_grid:+.1%}  (CAGR {cagr_grid:+.1%})")
print(f"Sharpe: {pf.sharpe_ratio():.3f}  MaxDD: {pf.max_drawdown():+.1%}  "
      f"Trades: {pf.trades.count()}")
print(f"\nParams: tw={tw} vw={vw} bgp={bgp:.4f} mgl={mgl} tpg={tpg:.2f} slg={slg:.2f}")
print("Chart → reports/512890vs000300.png")
