"""Polyfit-Switch 评分方法对比工作流。

对比「return（纯收益）」和「balanced（均衡）」两种评分标准
在 Walk-Forward 中的样本外表现差异。

用法：
    uv run python workflows/polyfit_switch.py
"""

import os
import sys
import time
import warnings

import pandas as pd
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu import detect_gpu, print_gpu_info
from utils.data import load_data
from utils.backtest import run_backtest
from utils.walkforward import run_walk_forward, print_walk_forward_summary
from utils.scoring import select_by_return, select_balanced
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_switch import (
    generate_polyfit_switch_signals,
    scan_polyfit_switch_two_stage,
)

warnings.filterwarnings("ignore")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Polyfit-Switch 评估函数 ──
def _make_eval_fn(open_):
    """创建评估函数（闭包捕获 open_）。"""
    def _eval(close_warmup_all, test_offset, params):
        from utils.indicators import compute_polyfit_switch_indicators

        fit_window = int(params.get("fit_window_days", 252))
        tw = int(params.get("trend_window_days", 20))
        vw = int(params.get("vol_window_days", 20))
        sw_fast = int(params["switch_fast_ma"])
        sw_slow = int(params["switch_slow_ma"])
        ma_windows = [5, 10, 20, 60]

        ind = compute_polyfit_switch_indicators(
            close_warmup_all, fit_window_days=fit_window,
            ma_windows=ma_windows, trend_window_days=tw, vol_window_days=vw,
        )
        test_start_date = close_warmup_all.index[test_offset]
        ind_test = ind.loc[ind.index >= test_start_date]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_.reindex(ind_test.index) if open_ is not None else None

        e, x, s = generate_polyfit_switch_signals(
            cl_test.values,
            ind_test["PolyDevPct"].values,
            ind_test["PolyDevTrend"].values,
            ind_test["RollingVolPct"].values,
            ind_test["PolyBasePred"].values,
            ind_test[f"MA{sw_fast}"].values,
            ind_test[f"MA{sw_slow}"].values,
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


if __name__ == "__main__":
    detect_gpu()
    print()

    print("Loading data…")
    df = load_data()
    close = df["Close"]
    open_ = df.get("Open")
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    print()

    param_keys = [
        "fit_window_days", "trend_window_days", "vol_window_days",
        "base_grid_pct", "volatility_scale", "trend_sensitivity",
        "max_grid_levels", "take_profit_grid", "stop_loss_grid",
        "max_holding_days", "cooldown_days",
        "min_signal_strength", "position_size", "position_sizing_coef",
        "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
        "switch_trailing_stop",
        "switch_fast_ma", "switch_slow_ma",
    ]

    eval_fn = _make_eval_fn(open_)

    methods = [
        ("return", select_by_return, "纯收益最大化"),
        ("balanced", select_balanced, "均衡评分（收益+Sharpe-回撤）"),
    ]

    all_results = []

    for method_name, selector, desc in methods:
        print(f"{'=' * 70}")
        print(f"  Polyfit-Switch 评分方法: {method_name} ({desc})")
        print(f"{'=' * 70}")

        t0 = time.time()
        wf_results = run_walk_forward(
            close, f"Polyfit-{method_name}",
            lambda c: scan_polyfit_switch_two_stage(c, open_=open_),
            eval_fn,
            param_keys=param_keys,
            train_years=[2, 3],  # 仅跑 2 年和 3 年训练
            best_selector=selector,
        )
        elapsed = time.time() - t0

        all_results.append(wf_results)

        if not wf_results.empty:
            wf_results["excess_return"] = (
                wf_results["test_return"] - wf_results["buy_hold_return"]
            )
            print_walk_forward_summary(wf_results, f"Polyfit-{method_name}")
            print(f"  Time: {elapsed:.0f}s")

    # ══════════════════════════════════════════════════════════════
    # 对比汇总
    # ══════════════════════════════════════════════════════════════
    all_df = pd.concat(all_results, ignore_index=True)
    if not all_df.empty:
        all_df["excess_return"] = (
            all_df["test_return"] - all_df["buy_hold_return"]
        )

        print(f"\n{'=' * 80}")
        print("  SCORING METHOD COMPARISON")
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
            returns_str = "  ".join(
                f"{v:+.1%}" for v in r["test_returns"]
            )
            returns_arr = np.array(r["test_returns"])
            # 计算收益的变异系数（越小越均衡）
            cv = returns_arr.std() / (returns_arr.mean() + 1e-9)

            print(f"\n  {r['strategy']:>20s}  {int(r['train_years'])}yr")
            print(f"    Avg OOS:      {r['avg_test_return']:+.1%}")
            print(f"    Avg excess:    {r['avg_excess']:+.1%}")
            print(f"    Avg Sharpe:    {r['avg_sharpe']:.3f}")
            print(f"    Avg MaxDD:     {r['avg_dd']:+.2%}")
            print(f"    Pos ratio:     {r['pos_ratio']:.0%}")
            print(f"    Beat BH:       {r['beat_bh']:.0%}")
            print(f"    CV (↓更均衡):  {cv:.3f}")
            print(f"    Returns:       {returns_str}")

        all_df.to_csv(f"{REPORTS_DIR}/scoring_comparison.csv", index=False)
        print(f"\n  Results → {REPORTS_DIR}/scoring_comparison.csv")

    print(f"\nDone.")
