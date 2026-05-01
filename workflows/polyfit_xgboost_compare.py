"""Polyfit-Switch vs Polyfit-XGBoost 对比。

用法：
    uv run python workflows/polyfit_xgboost_compare.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu import detect_gpu
from utils.data import load_data
from utils.backtest import run_backtest
from utils.walkforward import run_walk_forward, print_walk_forward_summary
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_switch import generate_polyfit_switch_signals, scan_polyfit_switch_two_stage
from strategies.polyfit_xgboost import (
    train_xgb_filter, filter_signals, _find_matching_exits,
)

warnings.filterwarnings("ignore")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

PARAM_KEYS = [
    "fit_window_days", "trend_window_days", "vol_window_days",
    "base_grid_pct", "volatility_scale", "trend_sensitivity",
    "max_grid_levels", "take_profit_grid", "stop_loss_grid",
    "max_holding_days", "cooldown_days",
    "min_signal_strength", "position_size", "position_sizing_coef",
    "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
    "switch_trailing_stop",
    "switch_fast_ma", "switch_slow_ma",
]


def _make_polyfit_eval(open_):
    """Polyfit-Switch 评估函数（无 XGBoost，作为基准）。"""
    def _eval(close_warmup_all, test_offset, params):
        fw = int(params.get("fit_window_days", 252))
        tw = int(params.get("trend_window_days", 20))
        vw = int(params.get("vol_window_days", 20))
        sw_fast = int(params["switch_fast_ma"])
        sw_slow = int(params["switch_slow_ma"])
        ma_windows = [5, 10, 20, 60]

        ind = compute_polyfit_switch_indicators(
            close_warmup_all, fit_window_days=fw,
            ma_windows=ma_windows, trend_window_days=tw, vol_window_days=vw,
        )
        test_start = close_warmup_all.index[test_offset]
        ind_test = ind.loc[ind.index >= test_start]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_.reindex(ind_test.index) if open_ is not None else None

        e, x, s = generate_polyfit_switch_signals(
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
            flat_wait_days=int(params.get("flat_wait_days", 8)),
            switch_deviation_m1=params.get("switch_deviation_m1", 0.03),
            switch_deviation_m2=params.get("switch_deviation_m2", 0.02),
            switch_trailing_stop=params.get("switch_trailing_stop", 0.05),
        )
        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}

        m = run_backtest(cl_test, e, x, s, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"],
                "win_rate": m["win_rate"]}
    return _eval


def _make_xgb_eval(open_):
    """Polyfit-XGBoost 评估函数（训练 XGBoost 过滤训练入场，测试时过滤）。"""
    def _eval(close_warmup_all, test_offset, params):
        fw = int(params.get("fit_window_days", 252))
        tw = int(params.get("trend_window_days", 20))
        vw = int(params.get("vol_window_days", 20))
        sw_fast = int(params["switch_fast_ma"])
        sw_slow = int(params["switch_slow_ma"])
        ma_windows = [5, 10, 20, 60]

        # ── 指标计算（warmup + train + test）──
        ind_all = compute_polyfit_switch_indicators(
            close_warmup_all, fit_window_days=fw,
            ma_windows=ma_windows, trend_window_days=tw, vol_window_days=vw,
        )

        test_start_ts = close_warmup_all.index[test_offset]
        ind_train = ind_all.loc[ind_all.index < test_start_ts]
        ind_test = ind_all.loc[ind_all.index >= test_start_ts]

        if ind_train.empty or ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}

        cl_train = close_warmup_all.loc[ind_train.index]
        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_.reindex(ind_test.index) if open_ is not None else None

        # ── 训练集：生成信号 → 回测 → 提取标签 → 训练 XGBoost ──
        e_tr, x_tr, s_tr = generate_polyfit_switch_signals(
            cl_train.values,
            ind_train["PolyDevPct"].values, ind_train["PolyDevTrend"].values,
            ind_train["RollingVolPct"].values, ind_train["PolyBasePred"].values,
            ind_train[f"MA{sw_fast}"].values, ind_train[f"MA{sw_slow}"].values,
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
            flat_wait_days=int(params.get("flat_wait_days", 8)),
            switch_deviation_m1=params.get("switch_deviation_m1", 0.03),
            switch_deviation_m2=params.get("switch_deviation_m2", 0.02),
            switch_trailing_stop=params.get("switch_trailing_stop", 0.05),
        )

        model = train_xgb_filter(
            cl_train.values, e_tr, x_tr,
            ind_train["PolyDevPct"].values,
            ind_train["PolyDevTrend"].values,
            ind_train["RollingVolPct"].values,
            ind_train["PolyBasePred"].values,
            ind_train[f"MA{sw_fast}"].values,
            ind_train[f"MA{sw_slow}"].values,
            n_estimators=20, max_depth=3,
        )

        # ── 测试集：生成信号 → XGBoost 过滤 → 回测 ──
        e_ts, x_ts, s_ts = generate_polyfit_switch_signals(
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
            flat_wait_days=int(params.get("flat_wait_days", 8)),
            switch_deviation_m1=params.get("switch_deviation_m1", 0.03),
            switch_deviation_m2=params.get("switch_deviation_m2", 0.02),
            switch_trailing_stop=params.get("switch_trailing_stop", 0.05),
        )

        total_entries = int(e_ts.sum())

        if model is not None and e_ts.sum() > 0:
            e_ts_filtered, total, kept = filter_signals(
                e_ts,
                ind_test["PolyDevPct"].values, ind_test["PolyDevTrend"].values,
                ind_test["RollingVolPct"].values, ind_test["PolyBasePred"].values,
                cl_test.values,
                ind_test[f"MA{sw_fast}"].values, ind_test[f"MA{sw_slow}"].values,
                model, threshold=0.45,
            )
        else:
            e_ts_filtered = e_ts
        e = e_ts_filtered

        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}

        m = run_backtest(cl_test, e, x_ts, s_ts, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"],
                "num_trades": m["num_trades"],
                "win_rate": m["win_rate"]}
    return _eval


if __name__ == "__main__":
    detect_gpu()
    print()

    print("Loading data…")
    df = load_data()
    close = df["Close"]
    open_ = df.get("Open")
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    print()

    all_results = []

    # ── Polyfit-Switch (基准) ──
    print(f"{'=' * 70}")
    print("  Polyfit-Switch (baseline)")
    print(f"{'=' * 70}")
    t0 = time.time()
    pf_wf = run_walk_forward(
        close, "Polyfit-Switch",
        lambda c: scan_polyfit_switch_two_stage(c, open_=open_),
        _make_polyfit_eval(open_),
        param_keys=PARAM_KEYS,
        train_years=[2, 3],
    )
    elapsed = time.time() - t0

    if not pf_wf.empty:
        pf_wf["excess_return"] = pf_wf["test_return"] - pf_wf["buy_hold_return"]
        print_walk_forward_summary(pf_wf, "Polyfit-Switch")
        print(f"  Time: {elapsed:.0f}s")
    all_results.append(pf_wf)

    # ── Polyfit-XGBoost ──
    print(f"\n{'=' * 70}")
    print("  Polyfit-XGBoost (XGBoost signal filter)")
    print(f"{'=' * 70}")
    t0 = time.time()
    xgb_wf = run_walk_forward(
        close, "Polyfit-XGBoost",
        lambda c: scan_polyfit_switch_two_stage(c, open_=open_),
        _make_xgb_eval(open_),
        param_keys=PARAM_KEYS,
        train_years=[2, 3],
    )
    elapsed = time.time() - t0

    if not xgb_wf.empty:
        xgb_wf["excess_return"] = xgb_wf["test_return"] - xgb_wf["buy_hold_return"]
        print_walk_forward_summary(xgb_wf, "Polyfit-XGBoost")
        print(f"  Time: {elapsed:.0f}s")
    all_results.append(xgb_wf)

    # ── 对比 ──
    all_df = pd.concat(all_results, ignore_index=True)
    if not all_df.empty:
        all_df["excess_return"] = all_df["test_return"] - all_df["buy_hold_return"]

        print(f"\n{'=' * 80}")
        print("  Polyfit-Switch  vs  Polyfit-XGBoost")
        print(f"{'=' * 80}")

        comp = all_df.groupby(["strategy", "train_years"]).agg(
            avg_test_return=("test_return", "mean"),
            avg_excess=("excess_return", "mean"),
            avg_sharpe=("test_sharpe", "mean"),
            avg_dd=("test_max_dd", "mean"),
            pos_ratio=("test_return", lambda x: (x > 0).mean()),
            beat_bh=("excess_return", lambda x: (x > 0).mean()),
            windows=("test_return", "count"),
            test_returns=("test_return", list),
        ).reset_index()

        for _, r in comp.iterrows():
            rets = np.array(r["test_returns"])
            cv = rets.std() / (rets.mean() + 1e-9)
            ret_str = "  ".join(f"{v:+.1%}" for v in rets)
            print(f"\n  {r['strategy']:>20s}  {int(r['train_years'])}yr")
            print(f"    Avg OOS:      {r['avg_test_return']:+.1%}")
            print(f"    Avg excess:    {r['avg_excess']:+.1%}")
            print(f"    Avg Sharpe:    {r['avg_sharpe']:.3f}")
            print(f"    Avg MaxDD:     {r['avg_dd']:+.2%}")
            print(f"    Beat BH:       {r['beat_bh']:.0%}")
            print(f"    CV (↓更均衡):  {cv:.3f}")
            print(f"    Returns:       {ret_str}")

        all_df.to_csv(f"{REPORTS_DIR}/xgboost_comparison.csv", index=False)
        print(f"\n  Results → {REPORTS_DIR}/xgboost_comparison.csv")

    print("\nDone.")
