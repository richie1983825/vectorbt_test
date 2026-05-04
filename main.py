"""
VectorBT Walk-Forward — Polyfit-Switch v3 vs Polyfit-Grid 对比。

所有回测均使用 Walk-Forward（训练→OOS），禁止全量回测。
每次回测同时列出 return / balanced / robust 三种评分结果。
"""

import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from utils.gpu import detect_gpu, print_gpu_info
from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators, compute_polyfit_base_only, add_trend_vol_indicators
from utils.backtest import run_backtest
from utils.scoring import select_by_return, select_balanced, select_robust
from utils.walkforward import run_walk_forward_multi_selector
from strategies.polyfit_switch import scan_polyfit_switch_two_stage, generate_polyfit_switch_signals
from strategies.polyfit_grid import scan_polyfit_grid, generate_grid_signals

REPORTS_DIR = "reports"

# ══════════════════════════════════════════════════════════════
# 对比参数
# ══════════════════════════════════════════════════════════════
GRID_PARAMS_KEYS = [
    "trend_window_days", "vol_window_days",
    "base_grid_pct", "volatility_scale", "trend_sensitivity",
    "max_grid_levels", "take_profit_grid", "stop_loss_grid",
    "max_holding_days", "cooldown_days",
    "min_signal_strength", "position_size", "position_sizing_coef",
]
SWITCH_PARAMS_KEYS = GRID_PARAMS_KEYS + [
    "switch_deviation_m1", "switch_deviation_m2", "switch_trailing_stop",
    "switch_fast_ma", "switch_slow_ma",
]

SELECTORS = {"return": select_by_return, "balanced": select_balanced, "robust": select_robust}


def _make_grid_eval(open_raw):
    def _eval(close_warmup_all, test_offset, params):
        from utils.backtest import run_backtest as _bt
        base_ind = compute_polyfit_base_only(close_warmup_all, fit_window_days=252)
        common_idx = base_ind.index
        if len(common_idx) == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        tw = int(params.get("trend_window_days", 20))
        vw = int(params.get("vol_window_days", 20))
        ind_full = add_trend_vol_indicators(base_ind, close_warmup_all, tw, vw)
        test_start = close_warmup_all.index[test_offset]
        ind_test = ind_full.loc[ind_full.index >= test_start]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_raw.reindex(ind_test.index)
        e, x, s = generate_grid_signals(
            cl_test.values, ind_test["PolyDevPct"].values, ind_test["PolyDevTrend"].values,
            ind_test["RollingVolPct"].values, ind_test["PolyBasePred"].values,
            base_grid_pct=params.get("base_grid_pct", 0.012),
            volatility_scale=params.get("volatility_scale", 1.0),
            trend_sensitivity=params.get("trend_sensitivity", 8.0),
            max_grid_levels=int(params.get("max_grid_levels", 3)),
            take_profit_grid=params.get("take_profit_grid", 0.85),
            stop_loss_grid=params.get("stop_loss_grid", 1.6),
            max_holding_days=int(params.get("max_holding_days", 45)),
            cooldown_days=int(params.get("cooldown_days", 1)),
            min_signal_strength=params.get("min_signal_strength", 0.45),
            position_size=params.get("position_size", 0.5),
            position_sizing_coef=params.get("position_sizing_coef", 30.0),
        )
        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        m = _bt(cl_test, e, x, s, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"], "win_rate": m["win_rate"]}
    return _eval


def _make_switch_eval(open_raw):
    def _eval(close_warmup_all, test_offset, params):
        from utils.backtest import run_backtest as _bt
        ind = compute_polyfit_switch_indicators(
            close_warmup_all, fit_window_days=252, ma_windows=[5,10,20,60],
            trend_window_days=int(params.get("trend_window_days", 20)),
            vol_window_days=int(params.get("vol_window_days", 20)),
        )
        test_start = close_warmup_all.index[test_offset]
        ind_test = ind.loc[ind.index >= test_start]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_raw.reindex(ind_test.index)
        sw_fast = int(params.get("switch_fast_ma", 20))
        sw_slow = int(params.get("switch_slow_ma", 60))
        e, x, s, _m = generate_polyfit_switch_signals(
            cl_test.values,
            ind_test["PolyDevPct"].values, ind_test["PolyDevTrend"].values,
            ind_test["RollingVolPct"].values, ind_test["PolyBasePred"].values,
            ind_test[f"MA{sw_fast}"].values, ind_test[f"MA{sw_slow}"].values,
            base_grid_pct=params.get("base_grid_pct", 0.012),
            volatility_scale=params.get("volatility_scale", 1.0),
            trend_sensitivity=params.get("trend_sensitivity", 8.0),
            max_grid_levels=int(params.get("max_grid_levels", 3)),
            take_profit_grid=params.get("take_profit_grid", 0.85),
            stop_loss_grid=params.get("stop_loss_grid", 1.6),
            max_holding_days=int(params.get("max_holding_days", 45)),
            cooldown_days=int(params.get("cooldown_days", 1)),
            min_signal_strength=params.get("min_signal_strength", 0.45),
            position_size=params.get("position_size", 0.5),
            position_sizing_coef=params.get("position_sizing_coef", 30.0),
            switch_deviation_m1=params.get("switch_deviation_m1", 0.03),
            switch_deviation_m2=params.get("switch_deviation_m2", 0.02),
            switch_trailing_stop=params.get("switch_trailing_stop", 0.05),
        )
        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        m = _bt(cl_test, e, x, s, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"], "win_rate": m["win_rate"]}
    return _eval


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)

    print("Loading data…")
    data_hfq = load_data("data/1d/512890.SH_hfq.parquet")
    close_hfq = data_hfq["Close"]
    open_raw = data_hfq["Open"]  # 用 hfq Open 确保与 hfq Close 同一价格尺度
    print(f"  {len(data_hfq)} bars  {data_hfq.index[0].date()} → {data_hfq.index[-1].date()}")

    grid_eval = _make_grid_eval(open_raw)
    switch_eval = _make_switch_eval(open_raw)

    wf_all = []

    # ── Polyfit-Grid WF ──
    print(f"\n{'=' * 70}")
    print("  Polyfit-Grid — Walk-Forward (22m train, 12m test)")
    print(f"{'=' * 70}")
    for tm in [22]:
        wf = run_walk_forward_multi_selector(
            close_hfq, "Polyfit-Grid",
            lambda c: scan_polyfit_grid(c, open_=open_raw),
            grid_eval, GRID_PARAMS_KEYS,
            selectors=SELECTORS,
            train_months=tm, test_months=12, step_months=3,
            warmup_months=12, min_train_bars=252, verbose=False,
        )
        if not wf.empty:
            wf["excess"] = wf["test_return"] - wf["buy_hold_return"]
            wf_all.append(wf)
            for sel in ["return", "balanced", "robust"]:
                sub = wf[wf["selector"] == sel]
                if sub.empty: continue
                print(f"  Grid {sel:>10s}: OOS={sub['test_return'].mean():>+7.1%}  "
                      f"α={sub['excess'].mean():>+7.1%}  "
                      f"sharpe={sub['test_sharpe'].mean():>7.3f}  "
                      f"dd={sub['test_max_dd'].mean():>+7.1%}  "
                      f">BH={(sub['excess']>0).mean():>.0%}  w={len(sub)}")

    # ── Polyfit-Switch v3 WF ──
    print(f"\n{'=' * 70}")
    print("  Polyfit-Switch v3 — Walk-Forward (22m train, 12m test)")
    print(f"{'=' * 70}")
    for tm in [22]:
        wf = run_walk_forward_multi_selector(
            close_hfq, "Polyfit-Switch",
            lambda c: scan_polyfit_switch_two_stage(c, open_=open_raw),
            switch_eval, SWITCH_PARAMS_KEYS,
            selectors=SELECTORS,
            train_months=tm, test_months=12, step_months=3,
            warmup_months=12, min_train_bars=252, verbose=False,
        )
        if not wf.empty:
            wf["excess"] = wf["test_return"] - wf["buy_hold_return"]
            wf_all.append(wf)
            for sel in ["return", "balanced", "robust"]:
                sub = wf[wf["selector"] == sel]
                if sub.empty: continue
                print(f"  Switch {sel:>10s}: OOS={sub['test_return'].mean():>+7.1%}  "
                      f"α={sub['excess'].mean():>+7.1%}  "
                      f"sharpe={sub['test_sharpe'].mean():>7.3f}  "
                      f"dd={sub['test_max_dd'].mean():>+7.1%}  "
                      f">BH={(sub['excess']>0).mean():>.0%}  w={len(sub)}")

    # ── 汇总 ──
    if len(wf_all) >= 2:
        all_wf = pd.concat(wf_all, ignore_index=True)
        print(f"\n{'=' * 90}")
        print(f"  Walk-Forward 对比: Polyfit-Grid vs Polyfit-Switch v3 (22m)")
        print(f"{'=' * 90}")
        print(f"  {'策略':>22s} {'评分':>10s} {'OOS':>8s} {'α':>8s} "
              f"{'sharpe':>7s} {'max_dd':>7s} {'pos':>5s} {'>BH':>5s} {'w':>4s}")
        print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*4}")
        for strat in ["Polyfit-Grid", "Polyfit-Switch"]:
            for sel in ["return", "balanced", "robust"]:
                sub = all_wf[(all_wf["strategy"] == strat) & (all_wf["selector"] == sel)]
                if sub.empty: continue
                print(f"  {strat:>22s} {sel:>10s} "
                      f"{sub['test_return'].mean():>+7.1%}  "
                      f"{sub['excess'].mean():>+7.1%}  "
                      f"{sub['test_sharpe'].mean():>7.3f}  "
                      f"{sub['test_max_dd'].mean():>+7.1%}  "
                      f"{(sub['test_return']>0).mean():>.0%}  "
                      f"{(sub['excess']>0).mean():>.0%}  "
                      f"{len(sub):>4d}")

        all_wf.to_csv(f"{REPORTS_DIR}/wf_comparison.csv", index=False)
        print(f"\n  Results → {REPORTS_DIR}/wf_comparison.csv")

    elif len(wf_all) == 1:
        all_wf = wf_all[0]

    # ══════════════════════════════════════════════════════════════
    # HTML 报告生成
    # ══════════════════════════════════════════════════════════════
    if len(wf_all) >= 1:
        print(f"\n{'=' * 70}")
        print("  Generating HTML Reports")
        print(f"{'=' * 70}")

        from utils.reports import (
            generate_polyfit_grid_report,
            generate_polyfit_switch_report,
            build_index_html,
        )

        # 准备全量数据的 df（含 PolyBasePred 用于报告绘图）
        base_ind_full = compute_polyfit_base_only(close_hfq, fit_window_days=252, ma_windows=[])
        common_idx_full = base_ind_full.index
        df_report = data_hfq.loc[common_idx_full].copy()
        df_report["PolyBasePred"] = base_ind_full["PolyBasePred"]
        reports_meta = []

        # ── Polyfit-Grid 报告 ──
        grid_balanced = all_wf[(all_wf["strategy"] == "Polyfit-Grid") & (all_wf["selector"] == "balanced")]
        best_grid_params = None  # 供 Switch 报告复用
        if not grid_balanced.empty:
            best_g = grid_balanced.nlargest(1, "test_return").iloc[0]
            tw = int(best_g["trend_window_days"])
            vw = int(best_g["vol_window_days"])
            best_grid_params = {
                "trend_window_days": tw, "vol_window_days": vw,
                "base_grid_pct": best_g["base_grid_pct"],
                "volatility_scale": best_g["volatility_scale"],
                "trend_sensitivity": best_g["trend_sensitivity"],
                "max_grid_levels": int(best_g["max_grid_levels"]),
                "take_profit_grid": best_g["take_profit_grid"],
                "stop_loss_grid": best_g["stop_loss_grid"],
                "max_holding_days": int(best_g["max_holding_days"]),
                "cooldown_days": int(best_g["cooldown_days"]),
                "min_signal_strength": best_g["min_signal_strength"],
                "position_size": best_g["position_size"],
                "position_sizing_coef": best_g["position_sizing_coef"],
            }
            grid_ind = add_trend_vol_indicators(base_ind_full, close_hfq, tw, vw)
            idx_g = grid_ind.index
            cl_g = close_hfq.loc[idx_g]
            op_g = open_raw.reindex(idx_g)
            e_g, x_g, s_g = generate_grid_signals(
                cl_g.values,
                grid_ind["PolyDevPct"].values,
                grid_ind["PolyDevTrend"].values,
                grid_ind["RollingVolPct"].values,
                grid_ind["PolyBasePred"].values,
                base_grid_pct=best_g["base_grid_pct"],
                volatility_scale=best_g["volatility_scale"],
                trend_sensitivity=best_g["trend_sensitivity"],
                max_grid_levels=int(best_g["max_grid_levels"]),
                take_profit_grid=best_g["take_profit_grid"],
                stop_loss_grid=best_g["stop_loss_grid"],
                max_holding_days=int(best_g["max_holding_days"]),
                cooldown_days=int(best_g["cooldown_days"]),
                min_signal_strength=best_g["min_signal_strength"],
                position_size=best_g["position_size"],
                position_sizing_coef=best_g["position_sizing_coef"],
            )
            grid_params = {k: best_g[k] for k in GRID_PARAMS_KEYS if k in best_g.index}
            meta = generate_polyfit_grid_report(
                df_report, cl_g, e_g, x_g, s_g,
                params=grid_params, name="Polyfit-Grid",
                reports_dir=REPORTS_DIR, open_=op_g,
            )
            reports_meta.append(meta)

        # ── Polyfit-Switch v3 报告（复用 Polyfit-Grid 的 Grid 参数，确保 Grid-only 一致） ──
        switch_balanced = all_wf[(all_wf["strategy"] == "Polyfit-Switch") & (all_wf["selector"] == "balanced")]
        if not switch_balanced.empty and best_grid_params is not None:
            best_s = switch_balanced.nlargest(1, "test_return").iloc[0]
            tw_s = best_grid_params["trend_window_days"]
            vw_s = best_grid_params["vol_window_days"]
            sw_fast = int(best_s["switch_fast_ma"])
            sw_slow = int(best_s["switch_slow_ma"])
            sw_ind = compute_polyfit_switch_indicators(
                close_hfq, fit_window_days=252,
                ma_windows=sorted(set([5, 10, 20, 60] + [sw_fast, sw_slow])),
                trend_window_days=tw_s, vol_window_days=vw_s,
            )
            idx_s = sw_ind.index
            cl_s = close_hfq.loc[idx_s]
            op_s = open_raw.reindex(idx_s)
            e_s, x_s, sz_s, modes_s = generate_polyfit_switch_signals(
                cl_s.values,
                sw_ind["PolyDevPct"].values,
                sw_ind["PolyDevTrend"].values,
                sw_ind["RollingVolPct"].values,
                sw_ind["PolyBasePred"].values,
                sw_ind[f"MA{sw_fast}"].values,
                sw_ind[f"MA{sw_slow}"].values,
                base_grid_pct=best_grid_params["base_grid_pct"],
                volatility_scale=best_grid_params["volatility_scale"],
                trend_sensitivity=best_grid_params["trend_sensitivity"],
                max_grid_levels=best_grid_params["max_grid_levels"],
                take_profit_grid=best_grid_params["take_profit_grid"],
                stop_loss_grid=best_grid_params["stop_loss_grid"],
                max_holding_days=best_grid_params["max_holding_days"],
                cooldown_days=best_grid_params["cooldown_days"],
                min_signal_strength=best_grid_params["min_signal_strength"],
                position_size=best_grid_params["position_size"],
                position_sizing_coef=best_grid_params["position_sizing_coef"],
                switch_deviation_m1=best_s["switch_deviation_m1"],
                switch_deviation_m2=best_s["switch_deviation_m2"],
                switch_trailing_stop=best_s["switch_trailing_stop"],
            )
            switch_params = {**best_grid_params,
                             "switch_deviation_m1": best_s["switch_deviation_m1"],
                             "switch_deviation_m2": best_s["switch_deviation_m2"],
                             "switch_trailing_stop": best_s["switch_trailing_stop"],
                             "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow}
            meta = generate_polyfit_switch_report(
                df_report, cl_s, e_s, x_s, sz_s, modes_s,
                params=switch_params, name="Polyfit-Switch",
                reports_dir=REPORTS_DIR, open_=op_s,
            )
            if meta is not None:
                reports_meta.append(meta)

        # ── 构建索引页 ──
        if reports_meta:
            index_html = build_index_html(reports_meta, REPORTS_DIR)
            index_path = f"{REPORTS_DIR}/index.html"
            with open(index_path, "w") as f:
                f.write(index_html)
            print(f"\n  Index → {index_path}")
            print(f"  Open: file://{os.path.abspath(index_path)}")

    print(f"\nDone.")
