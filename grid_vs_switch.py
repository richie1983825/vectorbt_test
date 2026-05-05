"""Grid vs Switch equity curves + 512890 price + trade markers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import vectorbt as vbt

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_polyfit_switch_signals

# ═══════════ Load data & best params ═══════════
df = load_data("data/1d/512890.SH_hfq.parquet")
close = df["Close"]; op = df["Open"]
hi = df["High"]; lo = df["Low"]; vol = df["Volume"]

wf = pd.read_csv("reports/wf_comparison.csv")
best_g = wf[(wf["strategy"] == "Polyfit-Grid") &
            (wf["selector"] == "balanced")].nlargest(1, "test_return").iloc[0]
best_s = wf[(wf["strategy"] == "Polyfit-Switch") &
            (wf["selector"] == "balanced")].nlargest(1, "test_return").iloc[0]

tw, vw = int(best_g["trend_window_days"]), int(best_g["vol_window_days"])
bgp, vs, ts = best_g["base_grid_pct"], best_g["volatility_scale"], best_g["trend_sensitivity"]
mgl, tpg, slg = int(best_g["max_grid_levels"]), best_g["take_profit_grid"], best_g["stop_loss_grid"]

# ═══════════ Indicators ═══════════
all_ma = [3, 5, 10, 20, 60]
ind = compute_polyfit_switch_indicators(close, 252, all_ma, tw, vw)
idx = ind.index; cl = close.loc[idx]; op_al = op.reindex(idx)
h_a = hi.reindex(idx); l_a = lo.reindex(idx); v_a = vol.reindex(idx)

# ═══════════ Pure Grid signals ═══════════
e_g, x_g, s_g = generate_grid_signals(
    cl.values, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
    ind["RollingVolPct"].values, ind["PolyBasePred"].values,
    base_grid_pct=bgp, volatility_scale=vs, trend_sensitivity=ts,
    max_grid_levels=mgl, take_profit_grid=tpg, stop_loss_grid=slg,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=best_g["min_signal_strength"],
    position_size=best_g["position_size"],
    position_sizing_coef=best_g["position_sizing_coef"],
)

# ═══════════ Switch signals ═══════════
e_s, x_s, sz_s, md_s, ho_s = generate_polyfit_switch_signals(
    cl.values, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
    ind["RollingVolPct"].values, ind["PolyBasePred"].values,
    ind["MA20"].values, ind["MA60"].values,
    base_grid_pct=bgp, volatility_scale=vs, trend_sensitivity=ts,
    max_grid_levels=mgl, take_profit_grid=tpg, stop_loss_grid=slg,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=best_g["min_signal_strength"],
    position_size=best_g["position_size"],
    position_sizing_coef=best_g["position_sizing_coef"],
    switch_deviation_m1=0.03, switch_deviation_m2=0.02,
    switch_trailing_stop=best_s["switch_trailing_stop"],
    trend_confirm_dp_slope=best_s.get("trend_confirm_dp_slope", 0.0003),
    trend_atr_mult=best_s.get("trend_atr_mult", 2.5),
    trend_atr_window=14,
    trend_vol_climax=best_s.get("trend_vol_climax", 3.0),
    high=h_a.values, low=l_a.values, volume=v_a.values,
    ma5=ind["MA5"].values, ma10=ind["MA10"].values, ma20=ind["MA20"].values,
    return_handovers=True,
)

# ═══════════ Equity curves via VectorBT ═══════════
fp = op_al.shift(-1).reindex(idx)
pf_g = vbt.Portfolio.from_signals(
    fp, entries=pd.Series(e_g, index=idx), exits=pd.Series(x_g, index=idx),
    size=pd.Series(s_g, index=idx), size_type="percent", init_cash=1.0, freq="D")
pf_s = vbt.Portfolio.from_signals(
    fp, entries=pd.Series(e_s, index=idx), exits=pd.Series(x_s, index=idx),
    size=pd.Series(sz_s, index=idx), size_type="percent", init_cash=1.0, freq="D")

eq_g = pf_g.value(); eq_s = pf_s.value()

# Normalise to 1.0
price_norm = cl / cl.iloc[0]
eq_g_norm = eq_g / eq_g.iloc[0]
eq_s_norm = eq_s / eq_s.iloc[0]

# ═══════════ Extract trades ═══════════
def get_trades(e, x, idx_arr):
    trades = []
    ex_idx = 0
    for ei in np.where(e)[0]:
        while ex_idx < len(np.where(x)[0]) and np.where(x)[0][ex_idx] <= ei:
            ex_idx += 1
        if ex_idx >= len(np.where(x)[0]): break
        xi = np.where(x)[0][ex_idx]
        trades.append((idx_arr[ei], idx_arr[xi]))
        ex_idx += 1
    return trades

trades_g = get_trades(e_g, x_g, idx)
trades_s = get_trades(e_s, x_s, idx)

# Mark Switch-handover trades (Grid entries that later transitioned to Switch)
handover_bars = set(np.where(ho_s)[0])
trades_ho = []
for ei in np.where(e_s)[0]:
    ex_idx = 0
    x_bars = np.where(x_s)[0]
    while ex_idx < len(x_bars) and x_bars[ex_idx] <= ei: ex_idx += 1
    if ex_idx >= len(x_bars): break
    xi = x_bars[ex_idx]
    # Check if handover happened between ei and xi
    if any(ho_s[ei:xi+1]):
        trades_ho.append((idx[ei], idx[xi]))
    ex_idx += 1

# ═══════════ Plot ═══════════
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(20, 11),
    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

# ── Top: equity curves + price ──
ax_top.plot(idx, price_norm, color="gray", linewidth=0.6, alpha=0.5, label="512890 Close (norm)")
ax_top.plot(idx, eq_g_norm, color="#1a5276", linewidth=1.0, label="Polyfit-Grid equity")
ax_top.plot(idx, eq_s_norm, color="#c0392b", linewidth=1.0, label="Polyfit-Switch equity")

# Grid trade markers
for en, ex in trades_g:
    ax_top.axvspan(en, ex, alpha=0.08, color="#1a5276", linewidth=0)

# Handover trade markers (highlighted)
for en, ex in trades_ho:
    ax_top.axvspan(en, ex, alpha=0.18, color="#e74c3c", linewidth=0)

# Entry/exit dots (sample to avoid overcrowding)
for en, ex in trades_g[::3]:
    ax_top.scatter(en, price_norm.loc[en], color="#1a5276", s=12, marker="^", zorder=5, alpha=0.7)
    ax_top.scatter(ex, price_norm.loc[ex], color="#1a5276", s=12, marker="v", zorder=5, alpha=0.7)

for en, ex in trades_ho:
    ax_top.scatter(en, price_norm.loc[en], color="#e74c3c", s=25, marker="^", zorder=6, alpha=0.9, edgecolors="black", linewidths=1)
    ax_top.scatter(ex, price_norm.loc[ex], color="#e74c3c", s=25, marker="v", zorder=6, alpha=0.9, edgecolors="black", linewidths=1)

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], color="gray", linewidth=0.8, alpha=0.5, label="512890 Close (norm)"),
    Line2D([0], [0], color="#1a5276", linewidth=1.5, label="Polyfit-Grid equity"),
    Line2D([0], [0], color="#c0392b", linewidth=1.5, label="Polyfit-Switch equity"),
    Patch(facecolor="#1a5276", alpha=0.08, label="Grid trade span (all)"),
    Patch(facecolor="#e74c3c", alpha=0.18, label="Switch / handover trade span"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c", markeredgecolor="black",
           markersize=10, label="Switch entry (handover from Grid TP)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c", markeredgecolor="black",
           markersize=10, label="Switch exit"),
]
ax_top.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=2)

ax_top.set_ylabel("Normalised (start = 1.0)", fontsize=11)
ax_top.set_title("Polyfit-Grid vs Polyfit-Switch — Equity Curves & Trade Markers", fontsize=14, fontweight="bold")
ax_top.grid(True, alpha=0.3)
ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

# Add annotation for key Switch trades
annotations = [
    ("2021-09-01", 2.05, "2021 Aug-Oct\n+9.5%"),
    ("2023-03-01", 2.45, "2023 Jan-Jun\n+11.9%"),
    ("2024-09-28", 2.82, "2024 Sep-Oct\n+7.5%"),
    ("2025-05-15", 2.75, "2025 Apr-Jul\n+8.0%"),
]
for dt, y, txt in annotations:
    ax_top.annotate(txt, xy=(pd.Timestamp(dt), y), fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", alpha=0.85))

# ── Bottom: drawdown ──
dd_g = (eq_g_norm / eq_g_norm.cummax() - 1)
dd_s = (eq_s_norm / eq_s_norm.cummax() - 1)
ax_bot.fill_between(idx, 0, dd_g, color="#1a5276", alpha=0.3, label="Grid DD")
ax_bot.fill_between(idx, 0, dd_s, color="#c0392b", alpha=0.3, label="Switch DD")
ax_bot.plot(idx, dd_g, color="#1a5276", linewidth=0.5)
ax_bot.plot(idx, dd_s, color="#c0392b", linewidth=0.5)
ax_bot.axhline(y=0, color="black", linewidth=0.5)
ax_bot.set_ylabel("Drawdown", fontsize=10)
ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax_bot.legend(loc="lower left", fontsize=8)
ax_bot.grid(True, alpha=0.3)

# X-axis
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

fig.tight_layout()
plt.savefig("reports/grid_vs_switch_trades.png", dpi=150, bbox_inches="tight")
plt.close()

# Stats
print(f"Grid:  final={eq_g_norm.iloc[-1]:.2f}  maxDD={dd_g.min():+.1%}  trades={len(trades_g)}")
print(f"Switch: final={eq_s_norm.iloc[-1]:.2f}  maxDD={dd_s.min():+.1%}  trades={len(trades_s)}  handovers={len(trades_ho)}")
print(f"Chart → reports/grid_vs_switch_trades.png")
