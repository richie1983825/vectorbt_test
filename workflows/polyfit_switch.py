"""Polyfit-Switch-v7 (v6 + 顶部规避) Walk-Forward 工作流。

依赖 Grid WF 缓存（workflows/polyfit_grid.py 产出），
在每个 WF 窗口上用 Grid 固定参数 + CPU 扫描 Switch-v7 参数。

用法：
    from workflows.polyfit_switch import run_switch_wf
    switch_df = run_switch_wf(close, open_, high, low, volume, windows, grid_wf_df)
"""

import os, sys, time, warnings
from itertools import product
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import compute_polyfit_switch_indicators
from utils.backtest import run_backtest
from utils.scoring import select_by_return, select_balanced, select_robust
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v7
from workflows.polyfit_grid import GRID_PARAMS_KEYS

warnings.filterwarnings("ignore")

SWITCH_V7_PARAMS_KEYS = GRID_PARAMS_KEYS + [
    "trend_entry_dp", "trend_confirm_dp_slope",
    "trend_atr_mult", "trend_vol_climax", "trend_decline_days",
    "enable_ohlcv_filter", "enable_early_exit",
    "enable_top_avoidance", "top_ret_5d", "top_price_pos",
    "top_amplitude", "top_block_days",
]

SELECTORS = {"return": select_by_return, "balanced": select_balanced, "robust": select_robust}

# V6 固定最优 base 参数
V6_FIXED = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_decline_days=1, trend_vol_climax=2.5,
    enable_ohlcv_filter=True, enable_early_exit=True)

# V7 只扫描顶部规避参数（base 参数固定为 V6 最优）
SW_V7_TOP_SCAN = {
    "enable_top_avoidance": [True, False],
    "top_ret_5d": [0.03, 0.05, 0.07],
    "top_price_pos": [0.70, 0.80, 0.90],
    "top_amplitude": [0.02, 0.03, 0.04],
    "top_block_days": [2, 3, 5],
}
SW_V7_KEYS = list(SW_V7_TOP_SCAN.keys())
SW_V7_COMBOS = list(product(*SW_V7_TOP_SCAN.values()))


def _make_switch_v7_eval(open_raw, high_raw=None, low_raw=None, volume_raw=None):
    def _eval(close_warmup_all, test_offset, params):
        ind = compute_polyfit_switch_indicators(
            close_warmup_all, fit_window_days=252, ma_windows=[20, 60],
            trend_window_days=int(params.get("trend_window_days", 20)),
            vol_window_days=int(params.get("vol_window_days", 20)),
        )
        test_start = close_warmup_all.index[test_offset]
        ind_test = ind.loc[ind.index >= test_start]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0,
                    "num_trades": 0, "win_rate": 0.0}
        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_raw.reindex(ind_test.index)
        h_test = high_raw.reindex(ind_test.index).values if high_raw is not None else None
        l_test = low_raw.reindex(ind_test.index).values if low_raw is not None else None
        v_test = volume_raw.reindex(ind_test.index).values if volume_raw is not None else None

        e_grid, x_grid, s_grid = generate_grid_signals(
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

        e_sw, x_sw, s_sw = generate_grid_priority_switch_signals_v7(
            cl_test.values, ind_test["PolyDevPct"].values, ind_test["PolyDevTrend"].values,
            ind_test["RollingVolPct"].values, ind_test["PolyBasePred"].values,
            e_grid, x_grid, ind_test["MA20"].values, ind_test["MA60"].values,
            trend_entry_dp=V6_FIXED["trend_entry_dp"],
            trend_confirm_dp_slope=V6_FIXED["trend_confirm_dp_slope"],
            trend_atr_mult=V6_FIXED["trend_atr_mult"], trend_atr_window=14,
            trend_vol_climax=V6_FIXED["trend_vol_climax"],
            trend_decline_days=V6_FIXED["trend_decline_days"],
            enable_ohlcv_filter=V6_FIXED["enable_ohlcv_filter"],
            enable_early_exit=V6_FIXED["enable_early_exit"],
            enable_top_avoidance=bool(params.get("enable_top_avoidance", True)),
            top_ret_5d=params.get("top_ret_5d", 0.05),
            top_price_pos=params.get("top_price_pos", 0.80),
            top_amplitude=params.get("top_amplitude", 0.025),
            top_block_days=int(params.get("top_block_days", 3)),
            high=h_test, low=l_test, open_=op_test.values if op_test is not None else None,
            volume=v_test,
        )

        e_merged = e_grid | e_sw
        x_merged = x_grid | x_sw
        s_merged = np.where(e_grid, s_grid, np.where(e_sw, 0.99, 0.0))
        if e_merged.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0,
                    "num_trades": 0, "win_rate": 0.0}
        m = run_backtest(cl_test, e_merged, x_merged, s_merged, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"],
                "win_rate": m["win_rate"]}
    return _eval


def _scan_switch_v7_train(close_train, open_, high, low, volume, grid_params):
    """训练期 CPU 扫描 Switch-v7 参数。"""
    tw_s = int(grid_params["trend_window_days"])
    vw_s = int(grid_params["vol_window_days"])

    indicators = compute_polyfit_switch_indicators(
        close_train, fit_window_days=252, ma_windows=[20, 60],
        trend_window_days=tw_s, vol_window_days=vw_s,
    )
    com_idx = indicators.index
    if len(com_idx) == 0:
        return pd.DataFrame()

    cl_arr = close_train.loc[com_idx].values
    cl_s = close_train.loc[com_idx]
    op_s = open_.reindex(com_idx) if open_ is not None else None
    op_arr = op_s.values if op_s is not None else None
    h_train = high.reindex(com_idx).values if high is not None else None
    l_train = low.reindex(com_idx).values if low is not None else None
    v_train = volume.reindex(com_idx).values if volume is not None else None

    e_grid, x_grid, s_grid = generate_grid_signals(
        cl_arr, indicators["PolyDevPct"].values, indicators["PolyDevTrend"].values,
        indicators["RollingVolPct"].values, indicators["PolyBasePred"].values,
        base_grid_pct=grid_params.get("base_grid_pct", 0.012),
        volatility_scale=grid_params.get("volatility_scale", 1.0),
        trend_sensitivity=grid_params.get("trend_sensitivity", 8.0),
        max_grid_levels=int(grid_params.get("max_grid_levels", 3)),
        take_profit_grid=grid_params.get("take_profit_grid", 0.85),
        stop_loss_grid=grid_params.get("stop_loss_grid", 1.6),
        max_holding_days=int(grid_params.get("max_holding_days", 45)),
        cooldown_days=int(grid_params.get("cooldown_days", 1)),
        min_signal_strength=grid_params.get("min_signal_strength", 0.45),
        position_size=grid_params.get("position_size", 0.5),
        position_sizing_coef=grid_params.get("position_sizing_coef", 30.0),
    )

    results = []
    for combo in SW_V7_COMBOS:
        kw = dict(zip(SW_V7_KEYS, combo))
        e_sw, x_sw, s_sw = generate_grid_priority_switch_signals_v7(
            cl_arr, indicators["PolyDevPct"].values, indicators["PolyDevTrend"].values,
            indicators["RollingVolPct"].values, indicators["PolyBasePred"].values,
            e_grid, x_grid, indicators["MA20"].values, indicators["MA60"].values,
            trend_entry_dp=V6_FIXED["trend_entry_dp"],
            trend_confirm_dp_slope=V6_FIXED["trend_confirm_dp_slope"],
            trend_atr_mult=V6_FIXED["trend_atr_mult"], trend_atr_window=14,
            trend_vol_climax=V6_FIXED["trend_vol_climax"],
            trend_decline_days=V6_FIXED["trend_decline_days"],
            enable_ohlcv_filter=V6_FIXED["enable_ohlcv_filter"],
            enable_early_exit=V6_FIXED["enable_early_exit"],
            enable_top_avoidance=kw["enable_top_avoidance"],
            top_ret_5d=kw["top_ret_5d"], top_price_pos=kw["top_price_pos"],
            top_amplitude=kw["top_amplitude"], top_block_days=kw["top_block_days"],
            high=h_train, low=l_train, open_=op_arr, volume=v_train,
        )
        e_merged = e_grid | e_sw
        x_merged = x_grid | x_sw
        s_merged = np.where(e_grid, s_grid, np.where(e_sw, 0.99, 0.0))
        if e_merged.sum() == 0:
            continue
        m = run_backtest(cl_s, e_merged, x_merged, s_merged, open_=op_s)
        results.append({
            "total_return": m["total_return"], "sharpe_ratio": m["sharpe_ratio"],
            "max_drawdown": m["max_drawdown"], "calmar_ratio": m["calmar_ratio"],
            "num_trades": m["num_trades"], "win_rate": m["win_rate"],
            **kw, **grid_params,
        })

    return pd.DataFrame(results)


def run_switch_wf(close, open_=None, high=None, low=None, volume=None,
                  windows=None, grid_wf_df=None,
                  train_months=22, test_months=12, step_months=3, warmup_months=12,
                  verbose=True):
    """运行 Polyfit-Switch-v7 Walk-Forward 扫描。"""
    from utils.walkforward import generate_monthly_windows

    if windows is None:
        windows = generate_monthly_windows(
            close.index, train_months=train_months, test_months=test_months,
            step_months=step_months, warmup_months=warmup_months,
        )

    grid_lookup = {}
    if grid_wf_df is not None and not grid_wf_df.empty:
        for _, r in grid_wf_df.iterrows():
            key = (str(r["test_start_date"])[:10], r["selector"])
            grid_lookup[key] = {k: r[k] for k in GRID_PARAMS_KEYS if k in r.index}

    switch_eval = _make_switch_v7_eval(open_, high, low, volume)
    wf_rows = []
    n_proc = 0; t0 = time.time()

    for wi, w in enumerate(windows):
        if w.test_start - w.train_start < 252: continue

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
            if cache_key not in grid_lookup: continue
            grid_params = grid_lookup[cache_key]

            sw_df = _scan_switch_v7_train(close_train, open_, high, low, volume, grid_params)
            if sw_df.empty: continue

            try:
                best_sw = sel_fn(sw_df)
            except Exception:
                continue
            sw_params = {k: best_sw[k] for k in SWITCH_V7_PARAMS_KEYS if k in best_sw.index}
            sw_oos = switch_eval(close_warmup_all, test_offset, sw_params)
            wf_rows.append({
                "strategy": "Polyfit-Switch-v7", "train_months": train_months,
                "selector": sel_name, "n_train_bars": n_train_bars,
                "test_start_date": test_close.index[0],
                "test_end_date": test_close.index[-1],
                "train_return": best_sw["total_return"],
                "train_sharpe": best_sw["sharpe_ratio"],
                "train_max_dd": best_sw["max_drawdown"],
                "buy_hold_return": bh_return,
                **sw_oos, **sw_params,
            })

        n_proc += 1
        if verbose and ((wi + 1) % 5 == 0 or wi == 0):
            elapsed = time.time() - t0
            print(f"  [Switch-v7 WF: {wi+1}/{len(windows)}] {elapsed:.0f}s")

    wf_df = pd.DataFrame(wf_rows)
    if verbose:
        elapsed = time.time() - t0
        print(f"  Switch-v7 WF done: {len(wf_df)} rows, {elapsed:.0f}s")
    return wf_df
