"""对比 510880 和 512890 的 Polyfit-Switch 策略收益。

用法：
    uv run python workflows/compare_symbols.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu import detect_gpu
from utils.backtest import run_backtest
from utils.walkforward import run_walk_forward, print_walk_forward_summary
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_switch import generate_polyfit_switch_signals, scan_polyfit_switch_two_stage

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


def load_symbol(path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df["Close"], df["Open"]


def _make_eval(open_):
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

        e, x, s, _modes = generate_polyfit_switch_signals(
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


if __name__ == "__main__":
    detect_gpu()
    print()

    symbols = [
        ("512890", "data/1d/512890.SH_hfq.parquet"),
        ("510880", "data/1d/510880.SH_hfq.parquet"),
    ]

    all_wf = []

    for symbol, path in symbols:
        print(f"{'='*70}")
        print(f"  {symbol} — Polyfit-Switch Walk-Forward")
        print(f"{'='*70}")

        close, open_ = load_symbol(path)
        # 对齐到共同时间范围
        close = close.loc["2019-01-01":"2026-04-30"]
        open_ = open_.loc["2019-01-01":"2026-04-30"]

        print(f"  {len(close)} bars  |  {close.index[0].date()} → {close.index[-1].date()}")

        t0 = time.time()
        wf = run_walk_forward(
            close, symbol,
            lambda c: scan_polyfit_switch_two_stage(c, open_=open_),
            _make_eval(open_),
            param_keys=PARAM_KEYS,
            train_years=[2, 3],
        )
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s")

        if not wf.empty:
            wf["excess_return"] = wf["test_return"] - wf["buy_hold_return"]
            wf["symbol"] = symbol
            print_walk_forward_summary(wf, symbol)
        all_wf.append(wf)

    # ══════════════════════════════════════════════════════════════
    # 对比
    # ══════════════════════════════════════════════════════════════
    all_df = pd.concat(all_wf, ignore_index=True)
    if all_df.empty:
        print("No results.")
        sys.exit(1)

    all_df["excess_return"] = all_df["test_return"] - all_df["buy_hold_return"]
    # 提取测试年份
    all_df["test_year"] = pd.to_datetime(all_df["test_period"].str[:10]).dt.year

    print(f"\n{'='*80}")
    print(f"  510880 vs 512890 — Polyfit-Switch 逐年对比")
    print(f"{'='*80}")

    # 按 symbol + train_years + test_year 聚合
    yearly = all_df.groupby(["symbol", "train_years", "test_year"]).agg(
        test_return=("test_return", "mean"),
        bh_return=("buy_hold_return", "mean"),
        excess_return=("excess_return", "mean"),
        sharpe=("test_sharpe", "mean"),
        max_dd=("test_max_dd", "mean"),
        trades=("num_trades", "sum"),
    ).reset_index()

    # 打印每年的对比
    for yr in sorted(yearly["test_year"].unique()):
        print(f"\n  {'='*60}")
        print(f"  {int(yr)}")
        print(f"  {'='*60}")
        yr_data = yearly[yearly["test_year"] == yr]
        for ty in [2, 3]:
            sub = yr_data[yr_data["train_years"] == ty]
            if sub.empty:
                continue
            print(f"  --- {ty}年训练 ---")
            for _, r in sub.iterrows():
                arrow = " ▲" if r["excess_return"] > 0 else " ▼"
                print(f"    {r['symbol']}: ret={r['test_return']:+.1%}  "
                      f"BH={r['bh_return']:+.1%}  α={r['excess_return']:+.1%}{arrow}  "
                      f"sharpe={r['sharpe']:.3f}  dd={r['max_dd']:+.1%}  tr={int(r['trades'])}")

    # 汇总
    print(f"\n{'='*80}")
    print(f"  汇总对比")
    print(f"{'='*80}")

    summary = all_df.groupby(["symbol", "train_years"]).agg(
        avg_test=("test_return", "mean"),
        avg_bh=("buy_hold_return", "mean"),
        avg_excess=("excess_return", "mean"),
        avg_sharpe=("test_sharpe", "mean"),
        avg_dd=("test_max_dd", "mean"),
        pos_ratio=("test_return", lambda x: (x > 0).mean()),
        beat_bh=("excess_return", lambda x: (x > 0).mean()),
        windows=("test_return", "count"),
        test_returns=("test_return", list),
    ).reset_index()

    for _, r in summary.iterrows():
        rets = np.array(r["test_returns"])
        cv = rets.std() / (rets.mean() + 1e-9)
        ret_str = "  ".join(f"{v:+.1%}" for v in rets)
        print(f"\n  {r['symbol']}  {int(r['train_years'])}yr")
        print(f"    Avg OOS:   {r['avg_test']:+.1%}     Avg BH: {r['avg_bh']:+.1%}")
        print(f"    Avg α:     {r['avg_excess']:+.1%}     Sharpe: {r['avg_sharpe']:.3f}")
        print(f"    Avg MaxDD: {r['avg_dd']:+.1%}      Beat BH: {r['beat_bh']:.0%}")
        print(f"    CV:        {cv:.3f}")
        print(f"    Returns:   {ret_str}")

    all_df.to_csv(f"{REPORTS_DIR}/symbol_comparison.csv", index=False)
    print(f"\n  Results → {REPORTS_DIR}/symbol_comparison.csv")
    print("Done.")
