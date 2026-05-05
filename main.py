"""
VectorBT Walk-Forward — Polyfit-Grid vs Polyfit-Switch-v6 对比入口。

调用 workflows/polyfit_grid.py 和 workflows/polyfit_switch.py 执行 WF 扫描，
输出六组合对比 + HTML 报告。
"""

import os, time, warnings
import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore")

from utils.gpu import detect_gpu, print_gpu_info
from utils.data import load_data
from utils.indicators import compute_polyfit_base_only, add_trend_vol_indicators, compute_polyfit_switch_indicators
from utils.backtest import run_backtest
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6
from workflows.polyfit_grid import run_grid_wf, GRID_PARAMS_KEYS, load_grid_cache
from workflows.polyfit_switch import run_switch_wf, SWITCH_PARAMS_KEYS

REPORTS_DIR = "reports"

# 六组合定义
SIX_COMBOS = [
    ("Grid-return",               "Polyfit-Grid",        "return"),
    ("Grid-balanced",             "Polyfit-Grid",        "balanced"),
    ("Switch-return",             "Polyfit-Switch-v6",   "return"),
    ("Switch-balanced",           "Polyfit-Switch-v6",   "balanced"),
    ("Grid-ret+Switch-bal",       "Polyfit-Switch-v6",   "return-grid+balanced-switch"),
    ("Grid-bal+Switch-ret",       "Polyfit-Switch-v6",   "balanced-grid+return-switch"),
]


def _print_six_combo_table(all_wf: pd.DataFrame):
    """输出六组合 WF OOS 对比表。"""
    all_wf["excess"] = all_wf["test_return"] - all_wf["buy_hold_return"]

    print(f"\n{'═' * 100}")
    print(f"  六组合 Walk-Forward OOS 对比 (22m train / 12m test)")
    print(f"{'═' * 100}")
    print(f"  {'#':>2s} {'组合':<28s} {'OOS':>8s} {'α':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'pos%':>5s} {'>BH':>5s}")
    print(f"  {'─'*2} {'─'*28} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*5} {'─'*5}")

    best_name = None
    best_oos = -999
    combo_stats = {}

    for i, (label, strat, sel) in enumerate(SIX_COMBOS, 1):
        sub = all_wf[(all_wf["strategy"] == strat) & (all_wf["selector"] == sel)]
        if sub.empty:
            print(f"  {i:>2d} {label:<28s} (无数据)")
            continue
        oos = sub["test_return"].mean()
        alpha = sub["excess"].mean()
        sharpe = sub["test_sharpe"].mean()
        dd = sub["test_max_dd"].mean()
        pos = (sub["test_return"] > 0).mean()
        beat = (sub["excess"] > 0).mean()
        marker = ""
        if oos > best_oos:
            best_oos = oos
            best_name = label
            marker = " ★"
        print(f"  {i:>2d} {label:<28s} {oos:>+7.1%}  {alpha:>+7.1%}  {sharpe:>7.3f}  {dd:>+7.1%}  {pos:>.0%}  {beat:>.0%}{marker}")
        combo_stats[label] = {"oos": oos, "alpha": alpha, "sharpe": sharpe, "max_dd": dd,
                               "strat": strat, "sel": sel}

    print(f"\n  ★ 最优组合: {best_name} (OOS={best_oos:+.1%})")
    return best_name, combo_stats


def _generate_reports(close_hfq, open_raw, high_raw, low_raw, volume_raw,
                      data_hfq, all_wf, best_name, combo_stats):
    """基于最优组合生成全量回测 HTML 报告。"""
    from utils.reports import (
        generate_polyfit_grid_report,
        generate_polyfit_switch_report,
        build_index_html,
    )

    best_info = combo_stats[best_name]
    best_strat = best_info["strat"]
    best_sel = best_info["sel"]

    print(f"\n{'═' * 70}")
    print(f"  Generating HTML Reports — 最优: {best_name}")
    print(f"{'═' * 70}")

    base_ind_full = compute_polyfit_base_only(close_hfq, fit_window_days=252, ma_windows=[])
    common_idx_full = base_ind_full.index
    df_report = data_hfq.loc[common_idx_full].copy()
    df_report["PolyBasePred"] = base_ind_full["PolyBasePred"]
    reports_meta = []

    # 最优窗口参数
    best_sub = all_wf[(all_wf["strategy"] == best_strat) & (all_wf["selector"] == best_sel)]
    if best_sub.empty:
        best_sub = all_wf[(all_wf["strategy"] == "Polyfit-Grid") & (all_wf["selector"] == "balanced")]
    best_row = best_sub.nlargest(1, "test_return").iloc[0]

    tw = int(best_row["trend_window_days"])
    vw = int(best_row["vol_window_days"])
    grid_p = {
        "trend_window_days": tw, "vol_window_days": vw,
        "base_grid_pct": best_row["base_grid_pct"],
        "volatility_scale": best_row["volatility_scale"],
        "trend_sensitivity": best_row["trend_sensitivity"],
        "max_grid_levels": int(best_row["max_grid_levels"]),
        "take_profit_grid": best_row["take_profit_grid"],
        "stop_loss_grid": best_row["stop_loss_grid"],
        "max_holding_days": int(best_row["max_holding_days"]),
        "cooldown_days": int(best_row["cooldown_days"]),
        "min_signal_strength": best_row["min_signal_strength"],
        "position_size": best_row["position_size"],
        "position_sizing_coef": best_row["position_sizing_coef"],
    }

    # ── Grid 报告 ──
    grid_ind = add_trend_vol_indicators(base_ind_full, close_hfq, trend_window_days=tw, vol_window_days=vw)
    idx_g = grid_ind.index
    cl_g = close_hfq.loc[idx_g]
    op_g = open_raw.reindex(idx_g)
    e_g, x_g, s_g = generate_grid_signals(
        cl_g.values, grid_ind["PolyDevPct"].values, grid_ind["PolyDevTrend"].values,
        grid_ind["RollingVolPct"].values, grid_ind["PolyBasePred"].values,
        base_grid_pct=grid_p["base_grid_pct"], volatility_scale=grid_p["volatility_scale"],
        trend_sensitivity=grid_p["trend_sensitivity"], max_grid_levels=grid_p["max_grid_levels"],
        take_profit_grid=grid_p["take_profit_grid"], stop_loss_grid=grid_p["stop_loss_grid"],
        max_holding_days=grid_p["max_holding_days"], cooldown_days=grid_p["cooldown_days"],
        min_signal_strength=grid_p["min_signal_strength"],
        position_size=grid_p["position_size"], position_sizing_coef=grid_p["position_sizing_coef"],
    )
    grid_params_report = {k: best_row[k] for k in GRID_PARAMS_KEYS if k in best_row.index}
    grid_meta = generate_polyfit_grid_report(
        df_report, cl_g, e_g, x_g, s_g, params=grid_params_report,
        name="Polyfit-Grid", reports_dir=REPORTS_DIR, open_=op_g,
    )
    reports_meta.append(grid_meta)

    # ── Switch 报告（如果最优组合包含 Switch）──
    is_switch = best_strat == "Polyfit-Switch-v6"
    if is_switch:
        sw_ind = compute_polyfit_switch_indicators(
            close_hfq, fit_window_days=252, ma_windows=[20, 60],
            trend_window_days=tw, vol_window_days=vw,
        )
        idx_s = sw_ind.index
        cl_s = close_hfq.loc[idx_s]; op_s = open_raw.reindex(idx_s)
        hi_s = high_raw.reindex(idx_s); lo_s = low_raw.reindex(idx_s)
        vol_s = volume_raw.reindex(idx_s)

        dev_pct_s = sw_ind["PolyDevPct"].values
        dev_trend_s = sw_ind["PolyDevTrend"].values
        vol_pct_s = sw_ind["RollingVolPct"].values
        poly_base_s = sw_ind["PolyBasePred"].values

        e_grid_s, x_grid_s, s_grid_s = generate_grid_signals(
            cl_s.values, dev_pct_s, dev_trend_s, vol_pct_s, poly_base_s,
            base_grid_pct=grid_p["base_grid_pct"], volatility_scale=grid_p["volatility_scale"],
            trend_sensitivity=grid_p["trend_sensitivity"], max_grid_levels=grid_p["max_grid_levels"],
            take_profit_grid=grid_p["take_profit_grid"], stop_loss_grid=grid_p["stop_loss_grid"],
            max_holding_days=grid_p["max_holding_days"], cooldown_days=grid_p["cooldown_days"],
            min_signal_strength=grid_p["min_signal_strength"],
            position_size=grid_p["position_size"], position_sizing_coef=grid_p["position_sizing_coef"],
        )

        sw_e, sw_x, sw_sz = generate_grid_priority_switch_signals_v6(
            cl_s.values, dev_pct_s, dev_trend_s, vol_pct_s, poly_base_s,
            e_grid_s, x_grid_s, ma20=sw_ind["MA20"].values, ma60=sw_ind["MA60"].values,
            trend_entry_dp=float(best_row.get("trend_entry_dp", 0.0)),
            trend_confirm_dp_slope=float(best_row.get("trend_confirm_dp_slope", 0.0)),
            trend_atr_mult=float(best_row.get("trend_atr_mult", 2.0)),
            trend_atr_window=14,
            trend_vol_climax=float(best_row.get("trend_vol_climax", 2.5)),
            trend_decline_days=int(best_row.get("trend_decline_days", 2)),
            enable_ohlcv_filter=bool(best_row.get("enable_ohlcv_filter", True)),
            enable_early_exit=bool(best_row.get("enable_early_exit", True)),
            high=hi_s.values, low=lo_s.values, open_=op_s.values, volume=vol_s.values,
        )

        fill_price_s = op_s.shift(-1).reindex(idx_s)
        pf_grid_only = vbt.Portfolio.from_signals(
            fill_price_s, entries=pd.Series(e_grid_s, index=idx_s),
            exits=pd.Series(x_grid_s, index=idx_s),
            size=pd.Series(s_grid_s, index=idx_s),
            size_type="percent", init_cash=1.0, freq="D",
        )
        pf_switch_only = vbt.Portfolio.from_signals(
            fill_price_s, entries=pd.Series(sw_e, index=idx_s),
            exits=pd.Series(sw_x, index=idx_s),
            size=pd.Series(sw_sz, index=idx_s),
            size_type="percent", init_cash=1.0, freq="D",
        )
        e_merged = e_grid_s | sw_e
        x_merged = x_grid_s | sw_x
        s_merged = np.where(e_grid_s, s_grid_s, sw_sz)
        modes_merged = np.zeros(len(e_merged), dtype=int)
        modes_merged[e_grid_s] = 1
        modes_merged[sw_e] = 2

        switch_params = {
            **grid_p,
            "trend_entry_dp": float(best_row.get("trend_entry_dp", 0.0)),
            "trend_confirm_dp_slope": float(best_row.get("trend_confirm_dp_slope", 0.0)),
            "trend_atr_mult": float(best_row.get("trend_atr_mult", 2.0)),
            "trend_vol_climax": float(best_row.get("trend_vol_climax", 2.5)),
            "trend_decline_days": int(best_row.get("trend_decline_days", 2)),
            "enable_ohlcv_filter": bool(best_row.get("enable_ohlcv_filter", True)),
            "enable_early_exit": bool(best_row.get("enable_early_exit", True)),
        }
        sw_meta = generate_polyfit_switch_report(
            df_report, cl_s, e_merged, x_merged, s_merged, modes_merged,
            params=switch_params, name="Polyfit-Switch-v6",
            reports_dir=REPORTS_DIR, open_=op_s,
            pf_grid=pf_grid_only, pf_switch=pf_switch_only,
        )
        if sw_meta is not None:
            reports_meta.append(sw_meta)

    # ── 索引页 ──
    if reports_meta:
        index_html = build_index_html(reports_meta, REPORTS_DIR)
        index_path = f"{REPORTS_DIR}/index.html"
        with open(index_path, "w") as f:
            f.write(index_html)
        print(f"\n  Index → {index_path}")
        print(f"  Open: file://{os.path.abspath(index_path)}")


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)
    _t0_total = time.time()

    # ── 加载数据 ──
    print("Loading data…")
    data_hfq = load_data("data/1d/512890.SH_hfq.parquet")
    close_hfq = data_hfq["Close"]; open_raw = data_hfq["Open"]
    high_raw = data_hfq["High"]; low_raw = data_hfq["Low"]
    volume_raw = data_hfq["Volume"]
    print(f"  {len(data_hfq)} bars  {data_hfq.index[0].date()} → {data_hfq.index[-1].date()}")

    windows = generate_monthly_windows(
        close_hfq.index, train_months=22, test_months=12,
        step_months=3, warmup_months=12,
    )
    print(f"  WF 窗口: {len(windows)} 个")

    # ── Stage 1: Grid WF ──
    print(f"\n{'═' * 70}")
    print(f"  Stage 1/2: Polyfit-Grid Walk-Forward")
    print(f"{'═' * 70}")
    t1 = time.time()
    grid_df = run_grid_wf(close_hfq, open_raw, windows=windows, force_rescan=False)
    print(f"  Stage 1 done: {time.time()-t1:.0f}s")

    # ── Stage 2: Switch-v6 WF ──
    print(f"\n{'═' * 70}")
    print(f"  Stage 2/2: Polyfit-Switch-v6 Walk-Forward")
    print(f"{'═' * 70}")
    t2 = time.time()
    switch_df = run_switch_wf(close_hfq, open_raw, high_raw, low_raw, volume_raw,
                              windows=windows, grid_wf_df=grid_df)
    print(f"  Stage 2 done: {time.time()-t2:.0f}s")

    # ── 合并结果 ──
    all_wf = pd.concat([grid_df, switch_df], ignore_index=True)
    all_wf.to_csv(f"{REPORTS_DIR}/wf_comparison.csv", index=False)
    print(f"\n  Results → {REPORTS_DIR}/wf_comparison.csv")

    # ── 六组合输出 ──
    best_name, combo_stats = _print_six_combo_table(all_wf)

    # ── HTML 报告 ──
    _generate_reports(close_hfq, open_raw, high_raw, low_raw, volume_raw,
                      data_hfq, all_wf, best_name, combo_stats)

    # ── Done ──
    _dt_total = time.time() - _t0_total
    print(f"\n{'═' * 70}")
    print(f"  总耗时: {_dt_total:.0f}s  ({_dt_total/60:.1f}min)")
    print(f"{'═' * 70}")
