"""Polyfit-Grid Walk-Forward 工作流 + 最优参数缓存。

Grid 策略已固定，每次 WF 扫描后将最优参数和收益缓存到本地，
供 main.py 和 Switch 工作流直接调用，避免重复 GPU 扫描。

用法：
    from workflows.polyfit_grid import run_grid_wf, load_grid_cache
    grid_df = run_grid_wf(close, open_, windows)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu import gpu
from utils.indicators import compute_polyfit_base_only, add_trend_vol_indicators
from utils.backtest import run_backtest
from utils.scoring import select_by_return, select_balanced, select_robust
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import scan_polyfit_grid, generate_grid_signals

warnings.filterwarnings("ignore")
REPORTS_DIR = "reports"
GRID_CACHE_FILE = f"{REPORTS_DIR}/grid_wf_cache.csv"

GRID_PARAMS_KEYS = [
    "trend_window_days", "vol_window_days",
    "base_grid_pct", "volatility_scale", "trend_sensitivity",
    "max_grid_levels", "take_profit_grid", "stop_loss_grid",
    "max_holding_days", "cooldown_days",
    "min_signal_strength", "position_size", "position_sizing_coef",
]

SELECTORS = {"return": select_by_return, "balanced": select_balanced, "robust": select_robust}


def load_grid_cache() -> dict:
    """加载缓存的 Grid WF 结果。返回 {(test_start_date_str, selector): {params...}}"""
    cache = {}
    if os.path.exists(GRID_CACHE_FILE):
        df = pd.read_csv(GRID_CACHE_FILE)
        for _, r in df.iterrows():
            key = (str(r["test_start_date"])[:10], r["selector"])
            entry = {k: r[k] for k in GRID_PARAMS_KEYS if k in r.index}
            for col in ["train_return", "train_sharpe", "train_max_dd",
                        "test_return", "test_sharpe", "test_max_dd",
                        "num_trades", "win_rate", "buy_hold_return",
                        "n_train_bars", "test_start_date", "test_end_date"]:
                if col in r.index:
                    entry[col] = r[col]
            cache[key] = entry
    return cache


def _make_grid_eval(open_raw):
    def _eval(close_warmup_all, test_offset, params):
        base_ind = compute_polyfit_base_only(close_warmup_all, fit_window_days=252)
        common_idx = base_ind.index
        if len(common_idx) == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0,
                    "num_trades": 0, "win_rate": 0.0}
        tw = int(params.get("trend_window_days", 20))
        vw = int(params.get("vol_window_days", 20))
        ind_full = add_trend_vol_indicators(base_ind, close_warmup_all, tw, vw)
        test_start = close_warmup_all.index[test_offset]
        ind_test = ind_full.loc[ind_full.index >= test_start]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0,
                    "num_trades": 0, "win_rate": 0.0}
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
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0,
                    "num_trades": 0, "win_rate": 0.0}
        m = run_backtest(cl_test, e, x, s, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"],
                "win_rate": m["win_rate"]}
    return _eval


def run_grid_wf(close: pd.Series, open_: pd.Series | None = None,
                windows: list | None = None,
                train_months: int = 22, test_months: int = 12,
                step_months: int = 3, warmup_months: int = 12,
                force_rescan: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """运行 Polyfit-Grid Walk-Forward 扫描，缓存结果。

    Returns:
        DataFrame with columns: strategy, selector, train_return, test_return, ... + Grid params
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if windows is None:
        windows = generate_monthly_windows(
            close.index, train_months=train_months, test_months=test_months,
            step_months=step_months, warmup_months=warmup_months,
        )

    grid_cache = {} if force_rescan else load_grid_cache()
    grid_eval = _make_grid_eval(open_)
    wf_rows = []
    n_proc = 0
    t0 = time.time()

    for wi, w in enumerate(windows):
        if w.test_start - w.train_start < 252:
            continue

        close_train = close.iloc[w.train_start:w.test_start]
        close_warmup_all = close.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start
        test_close = close.iloc[w.test_start:w.test_end]
        bh_return = ((test_close.iloc[-1] - test_close.iloc[0]) / test_close.iloc[0]
                     if len(test_close) >= 2 else 0.0)
        test_start_str = str(test_close.index[0].date())
        n_train_bars = w.test_start - w.train_start

        for sel_name, sel_fn in SELECTORS.items():
            cache_key = (test_start_str, sel_name)

            if cache_key in grid_cache and not force_rescan:
                gc = grid_cache[cache_key]
                grid_params = {k: gc[k] for k in GRID_PARAMS_KEYS if k in gc}
                train_ret = gc.get("train_return", 0)
                train_sharpe = gc.get("train_sharpe", 0)
                train_dd = gc.get("train_max_dd", 0)
                oos_ret = gc.get("test_return", 0)
                oos_sharpe = gc.get("test_sharpe", 0)
                oos_dd = gc.get("test_max_dd", 0)
                n_trades = int(gc.get("num_trades", 0))
                win_rate = gc.get("win_rate", 0)
            else:
                gdf = scan_polyfit_grid(close_train, open_=open_)
                if gdf.empty:
                    continue
                try:
                    best = sel_fn(gdf)
                except Exception:
                    continue
                grid_params = {k: best[k] for k in GRID_PARAMS_KEYS if k in best.index}
                train_ret = best["total_return"]
                train_sharpe = best.get("sharpe_ratio", 0)
                train_dd = best.get("max_drawdown", 0)

                oos = grid_eval(close_warmup_all, test_offset, grid_params)
                oos_ret = oos["test_return"]
                oos_sharpe = oos["test_sharpe"]
                oos_dd = oos["test_max_dd"]
                n_trades = oos["num_trades"]
                win_rate = oos["win_rate"]

                # 写入缓存
                grid_cache[cache_key] = {
                    **grid_params,
                    "train_return": train_ret, "train_sharpe": train_sharpe,
                    "train_max_dd": train_dd,
                    "test_return": oos_ret, "test_sharpe": oos_sharpe,
                    "test_max_dd": oos_dd, "num_trades": n_trades,
                    "win_rate": win_rate, "buy_hold_return": bh_return,
                    "n_train_bars": n_train_bars,
                    "test_start_date": str(test_close.index[0].date()),
                    "test_end_date": str(test_close.index[-1].date()),
                }

            wf_rows.append({
                "strategy": "Polyfit-Grid", "train_months": train_months,
                "selector": sel_name, "n_train_bars": n_train_bars,
                "test_start_date": test_close.index[0],
                "test_end_date": test_close.index[-1],
                "train_return": train_ret, "train_sharpe": train_sharpe,
                "train_max_dd": train_dd, "buy_hold_return": bh_return,
                "test_return": oos_ret, "test_sharpe": oos_sharpe,
                "test_max_dd": oos_dd, "num_trades": n_trades,
                "win_rate": win_rate,
                **grid_params,
            })

        n_proc += 1
        if verbose and ((wi + 1) % 5 == 0 or wi == 0):
            elapsed = time.time() - t0
            print(f"  [Grid WF: {wi+1}/{len(windows)}] {elapsed:.0f}s")

    # 保存缓存
    cache_rows = []
    for (ts, sel), entry in grid_cache.items():
        cache_rows.append({"test_start_date": ts, "selector": sel, **entry})
    pd.DataFrame(cache_rows).to_csv(GRID_CACHE_FILE, index=False)

    wf_df = pd.DataFrame(wf_rows)
    if verbose:
        elapsed = time.time() - t0
        print(f"  Grid WF done: {len(wf_df)} rows, {elapsed:.0f}s, "
              f"cache → {GRID_CACHE_FILE}")

    return wf_df


# ══════════════════════════════════════════════════════════════
# 直接运行入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from utils.gpu import gpu
    from utils.data import load_data
    gpu()
    print("Loading data…")
    data = load_data("data/1d/512890.SH_hfq.parquet")
    close = data["Close"]
    open_ = data["Open"]
    print(f"  {len(data)} bars  {data.index[0].date()} → {data.index[-1].date()}")

    grid_df = run_grid_wf(close, open_, force_rescan=False)
    print(f"\nGrid WF results: {len(grid_df)} rows")
    for sel in ["return", "balanced", "robust"]:
        sub = grid_df[grid_df["selector"] == sel]
        if sub.empty: continue
        print(f"  {sel}: OOS={sub['test_return'].mean():+.2%}  "
              f"sharpe={sub['test_sharpe'].mean():.3f}  "
              f"dd={sub['test_max_dd'].mean():+.2%}  n={len(sub)}")
    print("\nDone.")
