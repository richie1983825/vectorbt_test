"""
VectorBT Walk-Forward — Grid vs Switch-v6 vs Switch-v7 对比。

Stage 1: Grid WF (GPU, 缓存)
Stage 2: Switch-v7 WF (CPU, V6固定base + 扫描顶部规避)
V6 使用历史最优固定参数，不参与 WF 扫描。

用法:
    uv run python main.py                          # 默认 sh512890
    uv run python main.py sh512890                 # 指定标的
    uv run python main.py sh563020 --force-rescan  # 强制重新扫描
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore")

from utils.gpu import detect_gpu, print_gpu_info
from utils.data import load_data
from utils.indicators import compute_polyfit_base_only, add_trend_vol_indicators, compute_polyfit_switch_indicators
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6, generate_grid_priority_switch_signals_v7
from workflows.polyfit_grid import run_grid_wf, GRID_PARAMS_KEYS
from workflows.polyfit_switch import run_switch_wf

REPORTS_DIR = "reports"

# V6 历史最优固定参数 (22.2% OOS)
V6_BEST_GRID = dict(trend_window_days=10, vol_window_days=10, base_grid_pct=0.01,
    volatility_scale=0.0, trend_sensitivity=4, max_grid_levels=3, take_profit_grid=0.8,
    stop_loss_grid=1.6, max_holding_days=45, cooldown_days=1, min_signal_strength=0.3,
    position_size=0.99, position_sizing_coef=60)
V6_BEST_SWITCH = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_atr_window=14, trend_vol_climax=2.5, trend_decline_days=1,
    enable_ohlcv_filter=True, enable_early_exit=True)

V7_BASE_SWITCH = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_atr_window=14, trend_vol_climax=2.5, trend_decline_days=1,
    enable_ohlcv_filter=True, enable_early_exit=True)

ALL_COMBOS = [
    ("Grid-return",       "Polyfit-Grid",      "return"),
    ("Grid-balanced",     "Polyfit-Grid",      "balanced"),
    ("Switch-v6-return",  "Polyfit-Switch-v6", "return"),
    ("Switch-v6-balanced","Polyfit-Switch-v6", "balanced"),
    ("Switch-v7-return",  "Polyfit-Switch-v7", "return"),
    ("Switch-v7-balanced","Polyfit-Switch-v7", "balanced"),
]


def _parse_symbol(sym: str) -> tuple:
    """解析股票代码，返回 (data_path, sym_name, display_name)。

    sh512890 → data/1d/512890.SH_hfq.parquet, 512890_SH, 512890.SH
    """
    sym = sym.lower().strip()
    if sym.startswith("sh"):
        code, market = sym[2:], "SH"
    elif sym.startswith("sz"):
        code, market = sym[2:], "SZ"
    else:
        parts = sym.split(".")
        code, market = parts[0], parts[1].upper()
    path = f"data/1d/{code}.{market}_hfq.parquet"
    sym_name = f"{code}_{market}"
    display = f"{code}.{market}"
    return path, sym_name, display


def _print_combo_table(all_wf, display_name):
    if all_wf.empty:
        print(f"\n  (无有效 WF 结果)")
        return None
    all_wf["excess"] = all_wf["test_return"] - all_wf["buy_hold_return"]
    print(f"\n{'═' * 100}")
    print(f"  {display_name} — Grid vs Switch-v6 vs Switch-v7 — WF OOS 对比")
    print(f"{'═' * 100}")
    print(f"  {'#':>2s} {'组合':<22s} {'OOS':>8s} {'α':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'pos%':>5s} {'>BH':>5s}")
    print(f"  {'─'*2} {'─'*22} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*5} {'─'*5}")
    best_name, best_oos = None, -999
    for i, (label, strat, sel) in enumerate(ALL_COMBOS, 1):
        sub = all_wf[(all_wf["strategy"] == strat) & (all_wf["selector"] == sel)]
        if sub.empty: continue
        oos = sub["test_return"].mean(); alpha = sub["excess"].mean()
        sharpe = sub["test_sharpe"].mean(); dd = sub["test_max_dd"].mean()
        pos = (sub["test_return"] > 0).mean(); beat = (sub["excess"] > 0).mean()
        marker = " ★" if oos > best_oos else ""
        if oos > best_oos: best_oos = oos; best_name = label
        print(f"  {i:>2d} {label:<22s} {oos:>+7.1%}  {alpha:>+7.1%}  {sharpe:>7.3f}  {dd:>+7.1%}  {pos:>.0%}  {beat:>.0%}{marker}")
    print(f"\n  ★ 最优: {best_name} (OOS={best_oos:+.1%})")
    return best_name


def _eval_v6_oos(close_warmup_all, open_, high, low, volume, test_offset):
    from utils.backtest import run_backtest as _bt

    ind = compute_polyfit_switch_indicators(
        close_warmup_all, fit_window_days=252, ma_windows=[20, 60],
        trend_window_days=V6_BEST_GRID["trend_window_days"],
        vol_window_days=V6_BEST_GRID["vol_window_days"],
    )
    test_start = close_warmup_all.index[test_offset]
    ind_test = ind.loc[ind.index >= test_start]
    if ind_test.empty:
        return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
    cl_test = close_warmup_all.loc[ind_test.index]; op_test = open_.reindex(ind_test.index)
    h_test = high.reindex(ind_test.index).values if high is not None else None
    l_test = low.reindex(ind_test.index).values if low is not None else None
    v_test = volume.reindex(ind_test.index).values if volume is not None else None

    e_grid, x_grid, s_grid = generate_grid_signals(
        cl_test.values, ind_test["PolyDevPct"].values, ind_test["PolyDevTrend"].values,
        ind_test["RollingVolPct"].values, ind_test["PolyBasePred"].values,
        base_grid_pct=V6_BEST_GRID["base_grid_pct"], volatility_scale=V6_BEST_GRID["volatility_scale"],
        trend_sensitivity=V6_BEST_GRID["trend_sensitivity"], max_grid_levels=int(V6_BEST_GRID["max_grid_levels"]),
        take_profit_grid=V6_BEST_GRID["take_profit_grid"], stop_loss_grid=V6_BEST_GRID["stop_loss_grid"],
        max_holding_days=int(V6_BEST_GRID["max_holding_days"]), cooldown_days=int(V6_BEST_GRID["cooldown_days"]),
        min_signal_strength=V6_BEST_GRID["min_signal_strength"], position_size=V6_BEST_GRID["position_size"],
        position_sizing_coef=V6_BEST_GRID["position_sizing_coef"],
    )
    e_sw, x_sw, s_sw = generate_grid_priority_switch_signals_v6(
        cl_test.values, ind_test["PolyDevPct"].values, ind_test["PolyDevTrend"].values,
        ind_test["RollingVolPct"].values, ind_test["PolyBasePred"].values,
        e_grid, x_grid, ind_test["MA20"].values, ind_test["MA60"].values,
        high=h_test, low=l_test, open_=op_test.values if op_test is not None else None, volume=v_test,
        **V6_BEST_SWITCH,
    )
    e_all = e_grid | e_sw; x_all = x_grid | x_sw
    s_all = np.where(e_grid, s_grid, np.where(e_sw, 0.99, 0.0))
    if e_all.sum() == 0:
        return {"test_return": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
    m = _bt(cl_test, e_all, x_all, s_all, open_=op_test)
    return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
            "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"], "win_rate": m["win_rate"]}


def _generate_v7_report(close_hfq, open_raw, high_raw, low_raw, volume_raw, switch_df, df_report, sym_name):
    from utils.backtest import run_backtest as _bt
    from utils.reports import generate_polyfit_switch_report

    v7_return = switch_df[(switch_df["selector"] == "return")]
    if v7_return.empty:
        return None
    br = v7_return.nlargest(1, "test_return").iloc[0]

    tw = int(br["trend_window_days"]); vw = int(br["vol_window_days"])
    sw_ind = compute_polyfit_switch_indicators(close_hfq, fit_window_days=252, ma_windows=[20, 60],
                                                trend_window_days=tw, vol_window_days=vw)
    idx_s = sw_ind.index; cl_s = close_hfq.loc[idx_s]; op_s = open_raw.reindex(idx_s)
    hi_s = high_raw.reindex(idx_s); lo_s = low_raw.reindex(idx_s); vol_s = volume_raw.reindex(idx_s)

    e_grid_s, x_grid_s, s_grid_s = generate_grid_signals(
        cl_s.values, sw_ind["PolyDevPct"].values, sw_ind["PolyDevTrend"].values,
        sw_ind["RollingVolPct"].values, sw_ind["PolyBasePred"].values,
        base_grid_pct=br.get("base_grid_pct", 0.01),
        volatility_scale=br.get("volatility_scale", 0),
        trend_sensitivity=br.get("trend_sensitivity", 4),
        max_grid_levels=int(br.get("max_grid_levels", 3)),
        take_profit_grid=br.get("take_profit_grid", 0.8),
        stop_loss_grid=br.get("stop_loss_grid", 1.6),
        max_holding_days=int(br.get("max_holding_days", 45)),
        cooldown_days=int(br.get("cooldown_days", 1)),
        min_signal_strength=br.get("min_signal_strength", 0.3),
        position_size=br.get("position_size", 0.99),
        position_sizing_coef=br.get("position_sizing_coef", 60),
    )

    v7_top_params = {}
    for k in ["enable_top_avoidance", "top_ret_5d", "top_price_pos", "top_amplitude", "top_block_days"]:
        if k in br.index:
            v7_top_params[k] = br[k]

    sw_e, sw_x, sw_sz = generate_grid_priority_switch_signals_v7(
        cl_s.values, sw_ind["PolyDevPct"].values, sw_ind["PolyDevTrend"].values,
        sw_ind["RollingVolPct"].values, sw_ind["PolyBasePred"].values,
        e_grid_s, x_grid_s, sw_ind["MA20"].values, sw_ind["MA60"].values,
        trend_entry_dp=V7_BASE_SWITCH["trend_entry_dp"],
        trend_confirm_dp_slope=V7_BASE_SWITCH["trend_confirm_dp_slope"],
        trend_atr_mult=V7_BASE_SWITCH["trend_atr_mult"],
        trend_atr_window=V7_BASE_SWITCH["trend_atr_window"],
        trend_vol_climax=V7_BASE_SWITCH["trend_vol_climax"],
        trend_decline_days=V7_BASE_SWITCH["trend_decline_days"],
        enable_ohlcv_filter=V7_BASE_SWITCH["enable_ohlcv_filter"],
        enable_early_exit=V7_BASE_SWITCH["enable_early_exit"],
        high=hi_s.values, low=lo_s.values, open_=op_s.values, volume=vol_s.values,
        **v7_top_params,
    )

    fill_price_s = op_s.shift(-1).reindex(idx_s)
    pf_grid = vbt.Portfolio.from_signals(fill_price_s, entries=pd.Series(e_grid_s, index=idx_s),
        exits=pd.Series(x_grid_s, index=idx_s), size=pd.Series(s_grid_s, index=idx_s),
        size_type="percent", init_cash=1.0, freq="D")
    pf_sw = vbt.Portfolio.from_signals(fill_price_s, entries=pd.Series(sw_e, index=idx_s),
        exits=pd.Series(sw_x, index=idx_s), size=pd.Series(sw_sz, index=idx_s),
        size_type="percent", init_cash=1.0, freq="D")
    e_merged = e_grid_s | sw_e; x_merged = x_grid_s | sw_x
    s_merged = np.where(e_grid_s, s_grid_s, sw_sz)
    modes_merged = np.zeros(len(e_merged), dtype=int); modes_merged[e_grid_s]=1; modes_merged[sw_e]=2

    all_params = {**{k: br[k] for k in GRID_PARAMS_KEYS if k in br.index},
                  **V7_BASE_SWITCH, **v7_top_params}
    name = f"Polyfit-Switch-v7-{sym_name}"
    return generate_polyfit_switch_report(df_report, cl_s, e_merged, x_merged, s_merged, modes_merged,
        params=all_params, name=name, reports_dir=REPORTS_DIR, open_=op_s,
        pf_grid=pf_grid, pf_switch=pf_sw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorBT Walk-Forward 回测")
    parser.add_argument("symbol", nargs="?", default="sh512890",
                        help="股票代码 (默认: sh512890)")
    parser.add_argument("--force-rescan", action="store_true",
                        help="强制重新扫描 Grid WF (忽略缓存)")
    args = parser.parse_args()

    data_path, sym_name, display_name = _parse_symbol(args.symbol)
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        print(f"可用文件:")
        for f in sorted(os.listdir("data/1d")):
            if f.endswith("_hfq.parquet"):
                print(f"  data/1d/{f}")
        sys.exit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    gpu_info = detect_gpu(); print_gpu_info(gpu_info)
    _t0_total = time.time()

    print(f"标的: {display_name}  ({data_path})")
    data_hfq = load_data(data_path)
    close_hfq = data_hfq["Close"]; open_raw = data_hfq["Open"]
    high_raw = data_hfq["High"]; low_raw = data_hfq["Low"]; volume_raw = data_hfq["Volume"]
    print(f"  {len(data_hfq)} bars  {data_hfq.index[0].date()} → {data_hfq.index[-1].date()}")

    windows = generate_monthly_windows(close_hfq.index, train_months=22, test_months=12,
                                        step_months=3, warmup_months=12)
    if len(windows) == 0:
        min_bars = 22 * 21 + 12 * 21 + 12 * 21  # ~ train + test + warmup
        print(f"\n  错误: 数据太短 ({len(data_hfq)} bars)，至少需要约 {min_bars} 条日线")
        print(f"  当前标的只有 {data_hfq.index[0].date()} → {data_hfq.index[-1].date()}，"
              f"跨度 {(data_hfq.index[-1] - data_hfq.index[0]).days} 天")
        sys.exit(1)
    print(f"  WF 窗口: {len(windows)} 个")

    # 按标的隔离缓存文件
    grid_cache_path = f"{REPORTS_DIR}/grid_wf_cache_{sym_name}.csv"

    # ── Stage 1: Grid WF ──
    print(f"\n{'═' * 70}\n  Stage 1/2: Polyfit-Grid Walk-Forward\n{'═' * 70}")
    t1 = time.time()
    grid_df = run_grid_wf(close_hfq, open_raw, windows=windows,
                           force_rescan=args.force_rescan, cache_path=grid_cache_path)
    print(f"  Stage 1 done: {time.time()-t1:.0f}s")

    # ── Stage 2: Switch-v7 WF ──
    print(f"\n{'═' * 70}\n  Stage 2/2: Polyfit-Switch-v7 Walk-Forward\n{'═' * 70}")
    t2 = time.time()
    switch_df = run_switch_wf(close_hfq, open_raw, high_raw, low_raw, volume_raw,
                               windows=windows, grid_wf_df=grid_df)
    print(f"  Stage 2 done: {time.time()-t2:.0f}s")

    # ── V6 OOS 评估 ──
    print(f"\n  Evaluating V6 with fixed best params…")
    v6_rows = []
    for w in windows:
        if w.test_start - w.train_start < 252: continue
        close_warmup_all = close_hfq.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start
        test_close = close_hfq.iloc[w.test_start:w.test_end]
        bh_return = ((test_close.iloc[-1] - test_close.iloc[0]) / test_close.iloc[0]
                     if len(test_close) >= 2 else 0.0)
        oos = _eval_v6_oos(close_warmup_all, open_raw, high_raw, low_raw, volume_raw, test_offset)
        for sel in ["return", "balanced"]:
            v6_rows.append({
                "strategy": "Polyfit-Switch-v6", "train_months": 22,
                "selector": sel, "n_train_bars": w.test_start - w.train_start,
                "test_start_date": test_close.index[0], "test_end_date": test_close.index[-1],
                "train_return": 0, "train_sharpe": 0, "train_max_dd": 0,
                "buy_hold_return": bh_return, **oos,
                **V6_BEST_GRID, **V6_BEST_SWITCH,
            })
    v6_df = pd.DataFrame(v6_rows)

    all_wf = pd.concat([grid_df, v6_df, switch_df], ignore_index=True)
    wf_csv = f"{REPORTS_DIR}/wf_comparison_{sym_name}.csv"
    all_wf.to_csv(wf_csv, index=False)
    print(f"  Results → {wf_csv}")

    best_name = _print_combo_table(all_wf, display_name)

    # ── HTML 报告 ──
    print(f"\n{'═' * 70}\n  Generating HTML Reports — V7 策略 ({display_name})\n{'═' * 70}")
    from utils.reports import generate_polyfit_grid_report, generate_polyfit_switch_report, build_index_html

    base_ind_full = compute_polyfit_base_only(close_hfq, fit_window_days=252, ma_windows=[])
    cidx = base_ind_full.index
    df_report = data_hfq.loc[cidx].copy()
    df_report["PolyBasePred"] = base_ind_full["PolyBasePred"]
    reports_meta = []

    # Grid 报告
    grid_balanced = all_wf[(all_wf["strategy"]=="Polyfit-Grid") & (all_wf["selector"]=="balanced")]
    if not grid_balanced.empty:
        br = grid_balanced.nlargest(1, "test_return").iloc[0]
        tw = int(br["trend_window_days"]); vw = int(br["vol_window_days"])
        grid_ind = add_trend_vol_indicators(base_ind_full, close_hfq, trend_window_days=tw, vol_window_days=vw)
        idx_g = grid_ind.index; cl_g = close_hfq.loc[idx_g]; op_g = open_raw.reindex(idx_g)
        e_g, x_g, s_g = generate_grid_signals(
            cl_g.values, grid_ind["PolyDevPct"].values, grid_ind["PolyDevTrend"].values,
            grid_ind["RollingVolPct"].values, grid_ind["PolyBasePred"].values,
            base_grid_pct=br.get("base_grid_pct",0.01), volatility_scale=br.get("volatility_scale",0),
            trend_sensitivity=br.get("trend_sensitivity",4), max_grid_levels=int(br.get("max_grid_levels",3)),
            take_profit_grid=br.get("take_profit_grid",0.8), stop_loss_grid=br.get("stop_loss_grid",1.6),
            max_holding_days=int(br.get("max_holding_days",45)), cooldown_days=int(br.get("cooldown_days",1)),
            min_signal_strength=br.get("min_signal_strength",0.3), position_size=br.get("position_size",0.99),
            position_sizing_coef=br.get("position_sizing_coef",60),
        )
        reports_meta.append(generate_polyfit_grid_report(df_report, cl_g, e_g, x_g, s_g,
            params={k:br[k] for k in GRID_PARAMS_KEYS if k in br.index}, name=f"Polyfit-Grid-{sym_name}",
            reports_dir=REPORTS_DIR, open_=op_g))

    # Switch-v6 报告
    sw_ind = compute_polyfit_switch_indicators(close_hfq, fit_window_days=252, ma_windows=[20,60],
                                                trend_window_days=10, vol_window_days=10)
    idx_s = sw_ind.index; cl_s = close_hfq.loc[idx_s]; op_s = open_raw.reindex(idx_s)
    hi_s = high_raw.reindex(idx_s); lo_s = low_raw.reindex(idx_s); vol_s = volume_raw.reindex(idx_s)
    e_grid_s, x_grid_s, s_grid_s = generate_grid_signals(
        cl_s.values, sw_ind["PolyDevPct"].values, sw_ind["PolyDevTrend"].values,
        sw_ind["RollingVolPct"].values, sw_ind["PolyBasePred"].values,
        base_grid_pct=V6_BEST_GRID["base_grid_pct"], volatility_scale=V6_BEST_GRID["volatility_scale"],
        trend_sensitivity=V6_BEST_GRID["trend_sensitivity"], max_grid_levels=int(V6_BEST_GRID["max_grid_levels"]),
        take_profit_grid=V6_BEST_GRID["take_profit_grid"], stop_loss_grid=V6_BEST_GRID["stop_loss_grid"],
        max_holding_days=int(V6_BEST_GRID["max_holding_days"]), cooldown_days=int(V6_BEST_GRID["cooldown_days"]),
        min_signal_strength=V6_BEST_GRID["min_signal_strength"], position_size=V6_BEST_GRID["position_size"],
        position_sizing_coef=V6_BEST_GRID["position_sizing_coef"],
    )
    sw_e, sw_x, sw_sz = generate_grid_priority_switch_signals_v6(
        cl_s.values, sw_ind["PolyDevPct"].values, sw_ind["PolyDevTrend"].values,
        sw_ind["RollingVolPct"].values, sw_ind["PolyBasePred"].values,
        e_grid_s, x_grid_s, sw_ind["MA20"].values, sw_ind["MA60"].values,
        high=hi_s.values, low=lo_s.values, open_=op_s.values, volume=vol_s.values, **V6_BEST_SWITCH,
    )
    fill_price_s = op_s.shift(-1).reindex(idx_s)
    pf_grid = vbt.Portfolio.from_signals(fill_price_s, entries=pd.Series(e_grid_s, index=idx_s),
        exits=pd.Series(x_grid_s, index=idx_s), size=pd.Series(s_grid_s, index=idx_s),
        size_type="percent", init_cash=1.0, freq="D")
    pf_sw = vbt.Portfolio.from_signals(fill_price_s, entries=pd.Series(sw_e, index=idx_s),
        exits=pd.Series(sw_x, index=idx_s), size=pd.Series(sw_sz, index=idx_s),
        size_type="percent", init_cash=1.0, freq="D")
    e_merged = e_grid_s | sw_e; x_merged = x_grid_s | sw_x
    s_merged = np.where(e_grid_s, s_grid_s, sw_sz)
    modes_merged = np.zeros(len(e_merged), dtype=int); modes_merged[e_grid_s]=1; modes_merged[sw_e]=2
    reports_meta.append(generate_polyfit_switch_report(df_report, cl_s, e_merged, x_merged, s_merged, modes_merged,
        params={**V6_BEST_GRID, **V6_BEST_SWITCH}, name=f"Polyfit-Switch-v6-{sym_name}",
        reports_dir=REPORTS_DIR, open_=op_s, pf_grid=pf_grid, pf_switch=pf_sw))

    # Switch-v7 报告
    v7_report = _generate_v7_report(close_hfq, open_raw, high_raw, low_raw, volume_raw, switch_df, df_report, sym_name)
    if v7_report:
        reports_meta.append(v7_report)

    if reports_meta:
        index_path = f"{REPORTS_DIR}/index_{sym_name}.html"
        with open(index_path, "w") as f:
            f.write(build_index_html(reports_meta, REPORTS_DIR))
        print(f"  Index → {index_path}")

    _dt_total = time.time() - _t0_total
    print(f"\n{'═' * 70}\n  总耗时: {_dt_total:.0f}s  ({_dt_total/60:.1f}min)\n{'═' * 70}")
