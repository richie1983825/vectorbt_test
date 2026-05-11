"""
Polyfit-Grid vs Polyfit-Switch-v6 (OHLCV增强) Walk-Forward 对比。

Switch-v6 扫描参数：
  - trend_entry_dp, trend_confirm_dp_slope (入场)
  - trend_atr_mult, trend_decline_days, trend_vol_climax (离场)
  - enable_ohlcv_filter, enable_early_exit (v6 增强开关)
"""

import os, time, warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.gpu import gpu
from utils.indicators import (
    compute_polyfit_switch_indicators,
    compute_polyfit_base_only,
    add_trend_vol_indicators,
)
from utils.backtest import run_backtest
from utils.scoring import select_by_return, select_balanced, select_robust
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import scan_polyfit_grid, generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6

REPORTS_DIR = "reports"
SELECTORS = {"return": select_by_return, "balanced": select_balanced, "robust": select_robust}
GRID_PARAMS_KEYS = [
    "trend_window_days", "vol_window_days", "base_grid_pct", "volatility_scale",
    "trend_sensitivity", "max_grid_levels", "take_profit_grid", "stop_loss_grid",
    "max_holding_days", "cooldown_days", "min_signal_strength",
    "position_size", "position_sizing_coef",
]
SWITCH_PARAMS_KEYS = GRID_PARAMS_KEYS + [
    "trend_entry_dp", "trend_confirm_dp_slope",
    "trend_atr_mult", "trend_vol_climax", "trend_decline_days",
    "enable_ohlcv_filter", "enable_early_exit",
]

# ── v6 Switch 扫描空间 ──
SW_SCAN = {
    "trend_entry_dp": [0.0, 0.005, 0.01],
    "trend_confirm_dp_slope": [0.0, 0.0003, 0.001],
    "trend_atr_mult": [1.5, 2.0, 2.5],
    "trend_decline_days": [1, 2, 3],
    "trend_vol_climax": [2.5, 3.5, 5.0],
    "enable_ohlcv_filter": [True, False],
    "enable_early_exit": [True, False],
}


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
            cl_test.values, ind_test["PolyDevPct"].values,
            ind_test["PolyDevTrend"].values,
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


def _make_switch_v6_eval(open_raw, high_raw=None, low_raw=None, volume_raw=None):
    """Switch-v6 OOS 评估函数。"""
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

        cl_arr = cl_test.values
        dev_pct_arr = ind_test["PolyDevPct"].values
        dev_trend_arr = ind_test["PolyDevTrend"].values
        vol_pct_arr = ind_test["RollingVolPct"].values
        poly_base_arr = ind_test["PolyBasePred"].values
        ma20_arr = ind_test["MA20"].values
        ma60_arr = ind_test["MA60"].values

        # Grid signals
        e_grid, x_grid, s_grid = generate_grid_signals(
            cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
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

        # Switch v6 signals
        e_sw, x_sw, s_sw = generate_grid_priority_switch_signals_v6(
            cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
            e_grid, x_grid, ma20_arr, ma60_arr,
            trend_entry_dp=params.get("trend_entry_dp", 0.01),
            trend_confirm_dp_slope=params.get("trend_confirm_dp_slope", 0.0003),
            trend_atr_mult=params.get("trend_atr_mult", 2.0),
            trend_atr_window=14,
            trend_vol_climax=params.get("trend_vol_climax", 3.0),
            trend_decline_days=int(params.get("trend_decline_days", 2)),
            enable_ohlcv_filter=bool(params.get("enable_ohlcv_filter", True)),
            enable_early_exit=bool(params.get("enable_early_exit", True)),
            high=h_test, low=l_test, open_=op_test.values if op_test is not None else None,
            volume=v_test,
        )

        # Merge
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


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    t0_total = time.time()

    print("Loading data…")
    data_hfq = load_data("data/1d/512890.SH_hfq.parquet")
    close_hfq = data_hfq["Close"]
    open_raw = data_hfq["Open"]
    high_raw = data_hfq["High"]
    low_raw = data_hfq["Low"]
    volume_raw = data_hfq["Volume"]
    print(f"  {len(data_hfq)} bars  {data_hfq.index[0].date()} → {data_hfq.index[-1].date()}")

    grid_eval = _make_grid_eval(open_raw)
    switch_v6_eval = _make_switch_v6_eval(open_raw, high_raw, low_raw, volume_raw)

    # ── WF 窗口 ──
    windows = generate_monthly_windows(
        close_hfq.index, train_months=22, test_months=12,
        step_months=3, warmup_months=12,
    )
    print(f"\nWF 窗口: {len(windows)} 个 (22m train / 12m test / 3m step)")

    # ── 加载 Grid 缓存 ──
    prev_csv = f"{REPORTS_DIR}/wf_comparison.csv"
    grid_cache: dict = {}
    if os.path.exists(prev_csv):
        prev = pd.read_csv(prev_csv)
        prev_g = prev[prev["strategy"] == "Polyfit-Grid"]
        for _, r in prev_g.iterrows():
            key = (str(r["test_start_date"])[:10], r["selector"])
            grid_cache[key] = {k: r[k] for k in GRID_PARAMS_KEYS if k in r.index}
            # CSV 列名 → 缓存 key 映射
            for csv_col, cache_key_name in [
                ("test_return", "_test_return"), ("test_sharpe", "_test_sharpe"),
                ("test_max_dd", "_test_max_dd"), ("num_trades", "_num_trades"),
                ("win_rate", "_win_rate"), ("train_return", "_train_return"),
            ]:
                grid_cache[key][cache_key_name] = r[csv_col] if csv_col in r.index else 0
        print(f"Grid 缓存: {len(grid_cache)} 条")
    else:
        print(f"⚠ 无缓存，将运行完整 Grid 扫描")

    # ── 预计算 Switch 扫描组合 ──
    sw_keys = list(SW_SCAN.keys())
    sw_values = [SW_SCAN[k] for k in sw_keys]
    sw_combos_raw = list(product(*sw_values))
    sw_combos = []
    for combo in sw_combos_raw:
        d = dict(zip(sw_keys, combo))
        # 过滤无效组合：fast >= slow 此处不适用（v6用固定MA20/60）
        sw_combos.append(d)
    print(f"Switch-v6 扫描组合: {len(sw_combos)}")

    # ── WF 循环 ──
    wf_all = []
    t0_wf = time.time()
    n_proc = 0

    for wi, w in enumerate(windows):
        t0_win = time.time()
        n_train_bars = w.test_start - w.train_start
        if n_train_bars < 252:
            continue

        close_train = close_hfq.iloc[w.train_start:w.test_start]
        close_warmup_all = close_hfq.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start
        test_close = close_hfq.iloc[w.test_start:w.test_end]
        bh_return = ((test_close.iloc[-1] - test_close.iloc[0]) / test_close.iloc[0]
                     if len(test_close) >= 2 else 0.0)
        test_start_str = str(test_close.index[0].date())

        for sel_name, sel_fn in SELECTORS.items():
            # ── Grid: 缓存命中 or 完整扫描 ──
            cache_key = (test_start_str, sel_name)
            if cache_key in grid_cache:
                gc = grid_cache[cache_key]
                grid_params = {k: gc[k] for k in GRID_PARAMS_KEYS if k in gc}
                grid_oos = {
                    "test_return": gc["_test_return"],
                    "test_sharpe": gc["_test_sharpe"],
                    "test_max_dd": gc["_test_max_dd"],
                    "num_trades": int(gc["_num_trades"]),
                    "win_rate": gc["_win_rate"],
                }
                best_train_return = gc["_train_return"]
            else:
                gdf = scan_polyfit_grid(close_train, open_=open_raw)
                if gdf.empty: continue
                try:
                    best = sel_fn(gdf)
                except Exception: continue
                grid_params = {k: best[k] for k in GRID_PARAMS_KEYS if k in best.index}
                grid_oos = grid_eval(close_warmup_all, test_offset, grid_params)
                best_train_return = best["total_return"]

            wf_all.append({
                "strategy": "Polyfit-Grid", "train_months": 22,
                "selector": sel_name, "n_train_bars": n_train_bars,
                "test_start_date": test_close.index[0],
                "test_end_date": test_close.index[-1],
                "train_return": best_train_return,
                "train_sharpe": grid_oos.get("test_sharpe", 0),
                "train_max_dd": grid_oos.get("test_max_dd", 0),
                "buy_hold_return": bh_return,
                **grid_oos, **grid_params,
            })

            # ── Switch-v6 参数扫描（训练期）──
            tw_s = int(grid_params["trend_window_days"])
            vw_s = int(grid_params["vol_window_days"])
            best_bgp = grid_params["base_grid_pct"]
            best_vs = grid_params["volatility_scale"]
            best_ts = grid_params["trend_sensitivity"]
            best_max_gl = int(grid_params["max_grid_levels"])
            best_tpg = grid_params["take_profit_grid"]
            best_slg = grid_params["stop_loss_grid"]
            best_pos_sz = grid_params["position_size"]
            best_pos_coef = grid_params["position_sizing_coef"]
            best_min_ss = grid_params["min_signal_strength"]

            # 训练期指标
            indicators = compute_polyfit_switch_indicators(
                close_train, fit_window_days=252,
                ma_windows=[20, 60],
                trend_window_days=tw_s, vol_window_days=vw_s,
            )
            com_idx = indicators.index
            if len(com_idx) == 0: continue
            cl_train = close_train.loc[com_idx]
            op_train = open_raw.reindex(com_idx)
            h_train = high_raw.reindex(com_idx).values
            l_train = low_raw.reindex(com_idx).values
            v_train = volume_raw.reindex(com_idx).values
            cl_arr = cl_train.values
            dev_pct_arr = indicators["PolyDevPct"].values
            dev_trend_arr = indicators["PolyDevTrend"].values
            vol_pct_arr = indicators["RollingVolPct"].values
            poly_base_arr = indicators["PolyBasePred"].values
            ma20_arr = indicators["MA20"].values
            ma60_arr = indicators["MA60"].values

            # Grid 信号（训练期，固定参数）
            e_grid_tr, x_grid_tr, s_grid_tr = generate_grid_signals(
                cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, max_grid_levels=best_max_gl,
                take_profit_grid=best_tpg, stop_loss_grid=best_slg,
                max_holding_days=45, cooldown_days=1,
                min_signal_strength=best_min_ss, position_size=best_pos_sz,
                position_sizing_coef=best_pos_coef,
            )

            # ── CPU 扫描 Switch-v6 参数 ──
            sw_results_train = []
            for sw_combo in sw_combos:
                e_sw, x_sw, s_sw = generate_grid_priority_switch_signals_v6(
                    cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
                    e_grid_tr, x_grid_tr, ma20_arr, ma60_arr,
                    trend_entry_dp=sw_combo["trend_entry_dp"],
                    trend_confirm_dp_slope=sw_combo["trend_confirm_dp_slope"],
                    trend_atr_mult=sw_combo["trend_atr_mult"],
                    trend_atr_window=14,
                    trend_vol_climax=sw_combo["trend_vol_climax"],
                    trend_decline_days=sw_combo["trend_decline_days"],
                    enable_ohlcv_filter=sw_combo["enable_ohlcv_filter"],
                    enable_early_exit=sw_combo["enable_early_exit"],
                    high=h_train, low=l_train, open_=op_train.values, volume=v_train,
                )
                e_merged = e_grid_tr | e_sw
                x_merged = x_grid_tr | x_sw
                s_merged = np.where(e_grid_tr, s_grid_tr, np.where(e_sw, 0.99, 0.0))
                if e_merged.sum() == 0: continue
                m = run_backtest(cl_train, e_merged, x_merged, s_merged, open_=op_train)
                sw_results_train.append({
                    "total_return": m["total_return"],
                    "sharpe_ratio": m["sharpe_ratio"],
                    "max_drawdown": m["max_drawdown"],
                    "calmar_ratio": m["calmar_ratio"],
                    "num_trades": m["num_trades"],
                    "win_rate": m["win_rate"],
                    **sw_combo, **grid_params,
                })

            if not sw_results_train:
                continue
            sw_df = pd.DataFrame(sw_results_train)

            try:
                best_sw = sel_fn(sw_df)
            except Exception:
                continue

            sw_params_full = {k: best_sw[k] for k in SWITCH_PARAMS_KEYS
                              if k in best_sw.index}
            switch_v6_oos = switch_v6_eval(close_warmup_all, test_offset, sw_params_full)

            wf_all.append({
                "strategy": "Polyfit-Switch-v6", "train_months": 22,
                "selector": sel_name, "n_train_bars": n_train_bars,
                "test_start_date": test_close.index[0],
                "test_end_date": test_close.index[-1],
                "train_return": best_sw["total_return"],
                "train_sharpe": best_sw["sharpe_ratio"],
                "train_max_dd": best_sw["max_drawdown"],
                "buy_hold_return": bh_return,
                **switch_v6_oos, **sw_params_full,
            })

        n_proc += 1
        dt_win = time.time() - t0_win
        elapsed_wf = time.time() - t0_wf
        print(f"  [{wi+1:>2d}/{len(windows)}] {dt_win:.1f}s  "
              f"(train={close_hfq.index[w.train_start].date()} "
              f"test={close_hfq.index[w.test_start].date()}→"
              f"{close_hfq.index[w.test_end-1].date()})  "
              f"累计 {elapsed_wf:.0f}s")

    # ── 汇总 ──
    dt_wf = time.time() - t0_wf
    print(f"\n  WF 总耗时: {dt_wf:.0f}s  ({dt_wf/n_proc:.1f}s/window × {n_proc} windows)")

    if not wf_all:
        print("无结果")
        return

    all_wf = pd.DataFrame(wf_all)
    all_wf["excess"] = all_wf["test_return"] - all_wf["buy_hold_return"]

    # ══════════════════════════════════════════════════════════════
    # 结果输出
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 100}")
    print(f"  Polyfit-Grid vs Polyfit-Switch-v6 (OHLCV增强) — WF OOS 对比")
    print(f"{'═' * 100}")

    print(f"  {'策略':>22s} {'评分':>10s} {'OOS':>8s} {'α':>8s} "
          f"{'sharpe':>7s} {'max_dd':>7s} {'>0':>5s} {'>BH':>5s} {'w':>4s}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*4}")
    for strat in ["Polyfit-Grid", "Polyfit-Switch-v6"]:
        for sel in ["return", "balanced", "robust"]:
            sub = all_wf[(all_wf["strategy"] == strat) & (all_wf["selector"] == sel)]
            if sub.empty: continue
            print(f"  {strat:>22s} {sel:>10s} "
                  f"{sub['test_return'].mean():>+7.1%}  "
                  f"{sub['excess'].mean():>+7.1%}  "
                  f"{sub['test_sharpe'].mean():>7.3f}  "
                  f"{sub['test_max_dd'].mean():>+7.1%}  "
                  f"{(sub['test_return'] > 0).mean():>.0%}  "
                  f"{(sub['excess'] > 0).mean():>.0%}  "
                  f"{len(sub):>4d}")

    # 差异
    print(f"\n  ── Grid vs Switch-v6 差异 (balanced) ──")
    gb = all_wf[(all_wf["strategy"] == "Polyfit-Grid") & (all_wf["selector"] == "balanced")]
    sb = all_wf[(all_wf["strategy"] == "Polyfit-Switch-v6") & (all_wf["selector"] == "balanced")]
    if not gb.empty and not sb.empty:
        for label, key in [("OOS收益", "test_return"), ("α", "excess"),
                            ("Sharpe", "test_sharpe"), ("回撤", "test_max_dd"),
                            ("胜率", "win_rate")]:
            g_val = gb[key].mean()
            s_val = sb[key].mean()
            diff = s_val - g_val
            print(f"  {label}: Grid={g_val:+.2%}  Switch-v6={s_val:+.2%}  差异={diff:+.2%}")

    # 最优参数统计
    print(f"\n  ── Switch-v6 最优参数分布 (balanced) ──")
    if not sb.empty:
        for param in ["trend_entry_dp", "trend_confirm_dp_slope", "trend_atr_mult",
                       "trend_decline_days", "trend_vol_climax",
                       "enable_ohlcv_filter", "enable_early_exit"]:
            if param in sb.columns:
                if sb[param].dtype == bool:
                    vc = sb[param].value_counts()
                    print(f"  {param}: True={vc.get(True,0)} False={vc.get(False,0)}")
                else:
                    print(f"  {param}: mean={sb[param].mean():.4f}  "
                          f"median={sb[param].median():.4f}  "
                          f"top={sb[param].mode().values[:3]}")

    # 保存
    out_csv = f"{REPORTS_DIR}/wf_grid_vs_switch_v6.csv"
    all_wf.to_csv(out_csv, index=False)
    print(f"\n  Results → {out_csv}")

    dt_total = time.time() - t0_total
    print(f"\n  总耗时: {dt_total:.0f}s  ({dt_total/60:.1f}min)")
    print(f"    WF 扫描: {dt_wf:.0f}s ({dt_wf/dt_total*100:.0f}%)")


if __name__ == "__main__":
    main()
