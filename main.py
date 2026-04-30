"""
VectorBT walk-forward analysis — GPU accelerated.

Strategies: MA (grid), MA-Switch (grid + MA-crossover switch mode).
Default mode: rolling walk-forward (train N years, test next year).
"""

import os
import warnings

import pandas as pd

from utils.gpu import detect_gpu, print_gpu_info
from utils.data import load_data
from utils.backtest import run_backtest_batch
from utils.walkforward import run_walk_forward, print_walk_forward_summary
from strategies.ma_grid import (
    generate_grid_signals, scan_ma_strategy,
)
from strategies.ma_switch import (
    generate_switch_signals, scan_switch_two_stage,
)

warnings.filterwarnings("ignore")

REPORTS_DIR = "reports"


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)
    print()

    print("Loading data…")
    df = load_data()
    close = df["Close"]
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    print()

    print(f"{'=' * 70}")
    print("  Walk-Forward Analysis — Train N years → Test 1 year")
    print(f"{'=' * 70}")

    wf_results_all = []

    # --- MA Grid ---
    def _ma_eval(close_warmup_all, test_offset, params):
        from utils.indicators import compute_ma_indicators
        mw = int(params["ma_window"])
        ind = compute_ma_indicators(close_warmup_all, ma_window=mw)
        dev = ind["MADevPct"].values[test_offset:]
        trend = ind["MADevTrend"].values[test_offset:]
        vol = ind["RollingVolPct"].values[test_offset:]
        cl = close_warmup_all.values[test_offset:]

        e, x, s = generate_grid_signals(
            cl, dev, trend, vol,
            base_grid_pct=params["base_grid_pct"],
            volatility_scale=params["volatility_scale"],
            trend_sensitivity=params["trend_sensitivity"],
            take_profit_grid=params["take_profit_grid"],
            stop_loss_grid=params["stop_loss_grid"],
        )
        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}

        bt = run_backtest_batch(cl, e.reshape(1, -1), x.reshape(1, -1),
                                s.reshape(1, -1), n_combos=1)[0]
        return {"test_return": bt[0], "test_sharpe": bt[1],
                "test_max_dd": bt[2], "num_trades": int(bt[4]), "win_rate": bt[5]}

    print("\n── MA Grid ──")
    ma_wf = run_walk_forward(
        close, "MA", scan_ma_strategy, _ma_eval,
        param_keys=["ma_window", "base_grid_pct", "volatility_scale",
                     "trend_sensitivity", "take_profit_grid", "stop_loss_grid"],
        train_years=[1, 2, 3],
    )
    wf_results_all.append(ma_wf)

    # --- MA Switch (two-stage) ---
    def _switch_eval(close_warmup_all, test_offset, params):
        from utils.indicators import compute_ma_switch_indicators
        mw = int(params["ma_window"])
        sw_fast = int(params["switch_fast_ma"])
        sw_slow = int(params["switch_slow_ma"])
        ma_windows = [5, 10, 20, 60]

        ind = compute_ma_switch_indicators(close_warmup_all, ma_window=mw,
                                           ma_windows=ma_windows)
        dev = ind["MADevPct"].values[test_offset:]
        trend = ind["MADevTrend"].values[test_offset:]
        vol = ind["RollingVolPct"].values[test_offset:]
        cl = close_warmup_all.values[test_offset:]

        e, x, s = generate_switch_signals(
            cl, dev, trend, vol,
            ind["MABase"].values[test_offset:],
            ind[f"MA{sw_fast}"].values[test_offset:],
            ind[f"MA{sw_slow}"].values[test_offset:],
            base_grid_pct=params["base_grid_pct"],
            volatility_scale=params["volatility_scale"],
            trend_sensitivity=params["trend_sensitivity"],
            take_profit_grid=params["take_profit_grid"],
            stop_loss_grid=params["stop_loss_grid"],
            flat_wait_days=int(params["flat_wait_days"]),
            switch_deviation_m1=params["switch_deviation_m1"],
            switch_deviation_m2=params["switch_deviation_m2"],
            position_size=0.5, position_sizing_coef=30.0,
        )
        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}

        bt = run_backtest_batch(cl, e.reshape(1, -1), x.reshape(1, -1),
                                s.reshape(1, -1), n_combos=1)[0]
        return {"test_return": bt[0], "test_sharpe": bt[1],
                "test_max_dd": bt[2], "num_trades": int(bt[4]), "win_rate": bt[5]}

    print("\n── MA Switch (two-stage) ──")
    switch_wf = run_walk_forward(
        close, "MA-Switch", scan_switch_two_stage, _switch_eval,
        param_keys=["ma_window", "base_grid_pct", "volatility_scale",
                     "trend_sensitivity", "take_profit_grid", "stop_loss_grid",
                     "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
                     "switch_fast_ma", "switch_slow_ma"],
        train_years=[1, 2, 3],
    )
    wf_results_all.append(switch_wf)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════

    for df in wf_results_all:
        if not df.empty:
            print_walk_forward_summary(df, df["strategy"].iloc[0])

    all_wf = pd.concat(wf_results_all, ignore_index=True)
    if not all_wf.empty:
        print(f"\n{'=' * 80}")
        print("  CROSS-STRATEGY — Avg OOS Performance by Train Years")
        print(f"{'=' * 80}")
        wf_summary = all_wf.groupby(["strategy", "train_years"]).agg(
            avg_test_return=("test_return", "mean"),
            avg_test_sharpe=("test_sharpe", "mean"),
            pos_ratio=("test_return", lambda x: (x > 0).mean()),
            windows=("test_return", "count"),
            avg_trades=("num_trades", "mean"),
        ).reset_index()
        for _, r in wf_summary.iterrows():
            print(f"  {r['strategy']:>12s}  {int(r['train_years'])}yr train  "
                  f"avg_OOS={r['avg_test_return']:+.1%}  sharpe={r['avg_test_sharpe']:.3f}  "
                  f"pos={r['pos_ratio']:.0%}  windows={int(r['windows'])}  trades={r['avg_trades']:.0f}")

        all_wf.to_csv(f"{REPORTS_DIR}/walkforward_results.csv", index=False)
        print(f"\n  Results → {REPORTS_DIR}/walkforward_results.csv")

    print(f"\nDone.")
