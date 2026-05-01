"""
VectorBT 量化回测系统 — 主入口。

策略：
  - MA Grid：均线动态网格均值回复策略
  - MA Switch：均线网格 + 均线交叉追踪止损双模式策略
  - Polyfit Switch：多项式拟合基线 + 追踪止损双模式策略（★ 最佳）

通过 Walk-Forward 滚动窗口分析评估策略在不同训练期下的样本外（OOS）表现。

工作流程：
  1. 加载行情数据，检测 GPU/CUDA 环境
  2. 对每个策略执行 Walk-Forward 分析（训练 N 年 → 测试下一年）
  3. 汇总跨策略对比
  4. 使用 Walk-Forward 最优参数生成完整回测报告（VectorBT 图表 + HTML）
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
from strategies.polyfit_switch import (
    generate_polyfit_switch_signals, scan_polyfit_switch_two_stage,
)

# 关闭第三方库的无害警告，保持输出整洁
warnings.filterwarnings("ignore")

REPORTS_DIR = "reports"


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── 检测并打印 GPU 信息 ──
    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)
    print()

    # ── 加载行情数据 ──
    print("Loading data…")
    df = load_data()
    close = df["Close"]
    open_ = df.get("Open")  # 用于 next-bar Open 执行模式
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    print()

    print(f"{'=' * 70}")
    print("  Walk-Forward Analysis — Train N years → Test 1 year")
    print(f"{'=' * 70}")

    wf_results_all = []

    # ══════════════════════════════════════════════════════════════
    # MA 网格策略 Walk-Forward
    # ══════════════════════════════════════════════════════════════
    def _ma_eval(close_warmup_all, test_offset, params):
        """MA Grid 评估函数：在测试集上生成信号并回测。

        在 walk-forward 流程中，此函数接收「预热+训练+测试」的完整数据，
        但仅对 test_offset 之后的部分生成信号——确保指标计算使用了足够的
        历史数据做预热，避免前视偏差。

        使用 next-bar Open 成交，消除前视偏差。
        """
        from utils.indicators import compute_ma_indicators
        mw = int(params["ma_window"])
        ind = compute_ma_indicators(close_warmup_all, ma_window=mw)
        dev = ind["MADevPct"].values[test_offset:]
        trend = ind["MADevTrend"].values[test_offset:]
        vol = ind["RollingVolPct"].values[test_offset:]
        cl = close_warmup_all.values[test_offset:]
        op_full = open_.loc[close_warmup_all.index] if open_ is not None else None
        op = op_full.values[test_offset:] if op_full is not None else None

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
                                s.reshape(1, -1), n_combos=1, open_=op)[0]
        return {"test_return": bt[0], "test_sharpe": bt[1],
                "test_max_dd": bt[2], "num_trades": int(bt[4]), "win_rate": bt[5]}

    print("\n── MA Grid ──")
    ma_wf = run_walk_forward(
        close, "MA", lambda c: scan_ma_strategy(c, open_=open_),
        _ma_eval,
        param_keys=["ma_window", "base_grid_pct", "volatility_scale",
                     "trend_sensitivity", "take_profit_grid", "stop_loss_grid"],
        train_years=[1, 2, 3],
    )
    wf_results_all.append(ma_wf)

    # ══════════════════════════════════════════════════════════════
    # MA-Switch 双模式策略 Walk-Forward
    # ══════════════════════════════════════════════════════════════
    def _switch_eval(close_warmup_all, test_offset, params):
        """MA-Switch 评估函数：在测试集上生成信号并回测。

        使用 next-bar Open 成交，消除前视偏差。
        """
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
        op_full = open_.loc[close_warmup_all.index] if open_ is not None else None
        op = op_full.values[test_offset:] if op_full is not None else None

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
            switch_trailing_stop=params["switch_trailing_stop"],
            position_size=0.5, position_sizing_coef=30.0,
        )
        if e.sum() == 0:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}

        bt = run_backtest_batch(cl, e.reshape(1, -1), x.reshape(1, -1),
                                s.reshape(1, -1), n_combos=1, open_=op)[0]
        return {"test_return": bt[0], "test_sharpe": bt[1],
                "test_max_dd": bt[2], "num_trades": int(bt[4]), "win_rate": bt[5]}

    print("\n── MA Switch (trailing stop) ──")
    switch_wf = run_walk_forward(
        close, "MA-Switch",
        lambda c: scan_switch_two_stage(c, open_=open_),
        _switch_eval,
        param_keys=["ma_window", "base_grid_pct", "volatility_scale",
                     "trend_sensitivity", "take_profit_grid", "stop_loss_grid",
                     "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
                     "switch_trailing_stop",
                     "switch_fast_ma", "switch_slow_ma"],
        train_years=[1, 2, 3],
    )
    wf_results_all.append(switch_wf)

    # ══════════════════════════════════════════════════════════════
    # Polyfit Switch 策略 Walk-Forward
    #
    # 核心差异（vs MA-Switch）：
    #   1. 基线用多项式拟合（Polyfit 252 天）替代简单均线
    #   2. Switch 离场用死叉（fast < slow）替代追踪止损
    #   3. 最大持仓 45 天，仓位系数更大（0.92-0.99）
    # ══════════════════════════════════════════════════════════════
    def _polyfit_eval(close_warmup_all, test_offset, params):
        """Polyfit-Switch 评估函数：在测试集上生成信号并回测。

        使用 Polyfit 基线 + 均线交叉（死叉离场）模式。
        使用 next-bar Open 执行（与 backtesting.py 一致）。

        注意：compute_polyfit_switch_indicators 内部会 dropna，
        导致返回的 DataFrame 比原始 close 短。因此不能用 test_offset 做
        位置索引，而是用测试起始日期做标签过滤。
        """
        from utils.indicators import compute_polyfit_switch_indicators
        from utils.backtest import run_backtest as _run_backtest_cpu
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
        # 用日期标签截取测试期（避免 NaN drop 导致的位置偏移）
        test_start_date = close_warmup_all.index[test_offset]
        ind_test = ind.loc[ind.index >= test_start_date]
        if ind_test.empty:
            return {"test_return": 0.0, "test_sharpe": 0.0,
                    "test_max_dd": 0.0, "num_trades": 0, "win_rate": 0.0}
        cl_test = close_warmup_all.loc[ind_test.index]
        # 获取匹配的 Open 价格用于 next-bar 执行
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

        m = _run_backtest_cpu(cl_test, e, x, s, open_=op_test)
        return {"test_return": m["total_return"], "test_sharpe": m["sharpe_ratio"],
                "test_max_dd": m["max_drawdown"], "num_trades": m["num_trades"],
                "win_rate": m["win_rate"]}

    print("\n── Polyfit Switch (trailing stop exit) ──")
    polyfit_wf = run_walk_forward(
        close, "Polyfit-Switch",
        lambda c: scan_polyfit_switch_two_stage(c, open_=open_),
        _polyfit_eval,
        param_keys=["fit_window_days", "trend_window_days", "vol_window_days",
                    "base_grid_pct", "volatility_scale", "trend_sensitivity",
                    "max_grid_levels", "take_profit_grid", "stop_loss_grid",
                    "max_holding_days", "cooldown_days",
                    "min_signal_strength", "position_size", "position_sizing_coef",
                    "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
                    "switch_trailing_stop",
                    "switch_fast_ma", "switch_slow_ma"],
        train_years=[1, 2, 3],
    )
    wf_results_all.append(polyfit_wf)

    # ══════════════════════════════════════════════════════════════
    # Walk-Forward 结果汇总
    # ══════════════════════════════════════════════════════════════

    # 逐个策略打印详细摘要
    for df in wf_results_all:
        if not df.empty:
            print_walk_forward_summary(df, df["strategy"].iloc[0])

    # 跨策略对比：按 strategy + train_years 聚合 OOS 表现
    all_wf = pd.concat(wf_results_all, ignore_index=True)
    if not all_wf.empty:
        # 超额收益 = 策略 OOS 收益 - 买入持有收益
        all_wf["excess_return"] = all_wf["test_return"] - all_wf["buy_hold_return"]

        print(f"\n{'=' * 80}")
        print("  CROSS-STRATEGY — Avg OOS Performance by Train Years")
        print(f"{'=' * 80}")
        wf_summary = all_wf.groupby(["strategy", "train_years"]).agg(
            avg_test_return=("test_return", "mean"),
            avg_bh_return=("buy_hold_return", "mean"),
            avg_excess=("excess_return", "mean"),
            avg_test_sharpe=("test_sharpe", "mean"),
            pos_ratio=("test_return", lambda x: (x > 0).mean()),       # 正收益窗口占比
            beat_bh=("excess_return", lambda x: (x > 0).mean()),       # 跑赢买入持有的窗口占比
            windows=("test_return", "count"),
            avg_trades=("num_trades", "mean"),
        ).reset_index()
        for _, r in wf_summary.iterrows():
            print(f"  {r['strategy']:>12s}  {int(r['train_years'])}yr  "
                  f"OOS={r['avg_test_return']:+.1%}  BH={r['avg_bh_return']:+.1%}  "
                  f"α={r['avg_excess']:+.1%}  sharpe={r['avg_test_sharpe']:.3f}  "
                  f"pos={r['pos_ratio']:.0%}  >BH={r['beat_bh']:.0%}  "
                  f"w={int(r['windows'])}  tr={r['avg_trades']:.0f}")

        # 保存全部 Walk-Forward 结果到 CSV
        all_wf.to_csv(f"{REPORTS_DIR}/walkforward_results.csv", index=False)
        print(f"\n  Results → {REPORTS_DIR}/walkforward_results.csv")

    # ══════════════════════════════════════════════════════════════
    # 使用 Walk-Forward 最优参数生成 VectorBT 完整报告
    #
    # 注意：这里使用全量数据生成报告（不是只测试期），目的是展示
    # 最优参数在整个历史区间上的表现，方便可视化分析。
    # ══════════════════════════════════════════════════════════════

    from utils.indicators import (compute_ma_indicators, compute_ma_switch_indicators,
                                   compute_polyfit_switch_indicators)
    from utils.reports import generate_portfolio_reports, build_index_html

    print(f"\n{'=' * 60}")
    print("Generating reports from best walk-forward parameters…")
    print(f"{'=' * 60}")

    reports_data = []

    def _best_params(wf_df, param_keys):
        """从 Walk-Forward 结果中选取 OOS 收益最高的参数组合。"""
        best = wf_df.nlargest(1, "test_return").iloc[0]
        return {k: best[k] for k in param_keys if k in best.index}, best

    # --- MA Grid 全量报告 ---
    if not ma_wf.empty:
        ma_params, ma_best = _best_params(ma_wf, [
            "ma_window", "base_grid_pct", "volatility_scale",
            "trend_sensitivity", "take_profit_grid", "stop_loss_grid",
        ])
        mw = int(ma_params["ma_window"])
        # 用全量数据计算指标并生成信号
        ind = compute_ma_indicators(close, ma_window=mw)
        e, x, s = generate_grid_signals(
            close.values, ind["MADevPct"].values, ind["MADevTrend"].values,
            ind["RollingVolPct"].values,
            base_grid_pct=ma_params["base_grid_pct"],
            volatility_scale=ma_params["volatility_scale"],
            trend_sensitivity=ma_params["trend_sensitivity"],
            take_profit_grid=ma_params["take_profit_grid"],
            stop_loss_grid=ma_params["stop_loss_grid"],
        )
        reports_data.append(generate_portfolio_reports(
            close, e, x, s, "MA",
            {**ma_params, "ma_window": mw,
             "train_period": ma_best.get("train_period", "N/A"),
             "test_period": ma_best.get("test_period", "N/A"),
             "test_return": f"{ma_best['test_return']:.2%}"},
            reports_dir=REPORTS_DIR,
            open_=open_,
        ))

    # --- MA Switch 全量报告 ---
    if not switch_wf.empty:
        sw_params, sw_best = _best_params(switch_wf, [
            "ma_window", "base_grid_pct", "volatility_scale",
            "trend_sensitivity", "take_profit_grid", "stop_loss_grid",
            "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
            "switch_trailing_stop", "switch_fast_ma", "switch_slow_ma",
        ])
        mw = int(sw_params["ma_window"])
        sw_fast = int(sw_params["switch_fast_ma"])
        sw_slow = int(sw_params["switch_slow_ma"])
        ma_windows = [5, 10, 20, 60]
        ind = compute_ma_switch_indicators(close, ma_window=mw, ma_windows=ma_windows)
        e, x, s = generate_switch_signals(
            close.values,
            ind["MADevPct"].values, ind["MADevTrend"].values,
            ind["RollingVolPct"].values, ind["MABase"].values,
            ind[f"MA{sw_fast}"].values, ind[f"MA{sw_slow}"].values,
            base_grid_pct=sw_params["base_grid_pct"],
            volatility_scale=sw_params["volatility_scale"],
            trend_sensitivity=sw_params["trend_sensitivity"],
            take_profit_grid=sw_params["take_profit_grid"],
            stop_loss_grid=sw_params["stop_loss_grid"],
            flat_wait_days=int(sw_params["flat_wait_days"]),
            switch_deviation_m1=sw_params["switch_deviation_m1"],
            switch_deviation_m2=sw_params["switch_deviation_m2"],
            switch_trailing_stop=sw_params["switch_trailing_stop"],
            position_size=0.5, position_sizing_coef=30.0,
        )
        reports_data.append(generate_portfolio_reports(
            close, e, x, s, "MA-Switch",
            {**sw_params, "ma_window": mw,
             "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
             "train_period": sw_best.get("train_period", "N/A"),
             "test_period": sw_best.get("test_period", "N/A"),
             "test_return": f"{sw_best['test_return']:.2%}"},
            reports_dir=REPORTS_DIR,
            open_=open_,
        ))

    # --- Polyfit Switch 全量报告 ---
    if not polyfit_wf.empty:
        pf_param_keys = [
            "fit_window_days", "trend_window_days", "vol_window_days",
            "base_grid_pct", "volatility_scale", "trend_sensitivity",
            "max_grid_levels", "take_profit_grid", "stop_loss_grid",
            "max_holding_days", "cooldown_days",
            "min_signal_strength", "position_size", "position_sizing_coef",
            "flat_wait_days", "switch_deviation_m1", "switch_deviation_m2",
            "switch_trailing_stop",
            "switch_fast_ma", "switch_slow_ma",
        ]
        pf_params, pf_best = _best_params(polyfit_wf, pf_param_keys)

        fit_w = int(pf_params.get("fit_window_days", 252))
        tw = int(pf_params.get("trend_window_days", 20))
        vw = int(pf_params.get("vol_window_days", 20))
        sw_fast = int(pf_params.get("switch_fast_ma", 20))
        sw_slow = int(pf_params.get("switch_slow_ma", 60))
        ma_windows = [5, 10, 20, 60]

        ind = compute_polyfit_switch_indicators(
            close, fit_window_days=fit_w,
            ma_windows=ma_windows, trend_window_days=tw, vol_window_days=vw,
        )
        # 对齐 close/open 和 indicators
        common_idx = ind.index
        cl = close.loc[common_idx]
        op_aligned = open_.reindex(common_idx) if open_ is not None else None
        e, x, s = generate_polyfit_switch_signals(
            cl.values,
            ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
            ind["RollingVolPct"].values, ind["PolyBasePred"].values,
            ind[f"MA{sw_fast}"].values, ind[f"MA{sw_slow}"].values,
            base_grid_pct=pf_params.get("base_grid_pct", 0.012),
            volatility_scale=pf_params.get("volatility_scale", 1.0),
            trend_sensitivity=pf_params.get("trend_sensitivity", 8.0),
            max_grid_levels=int(pf_params.get("max_grid_levels", 3)),
            take_profit_grid=pf_params.get("take_profit_grid", 0.85),
            stop_loss_grid=pf_params.get("stop_loss_grid", 1.6),
            max_holding_days=int(pf_params.get("max_holding_days", 45)),
            cooldown_days=int(pf_params.get("cooldown_days", 1)),
            min_signal_strength=pf_params.get("min_signal_strength", 0.45),
            position_size=pf_params.get("position_size", 0.5),
            position_sizing_coef=pf_params.get("position_sizing_coef", 30.0),
            flat_wait_days=int(pf_params.get("flat_wait_days", 8)),
            switch_deviation_m1=pf_params.get("switch_deviation_m1", 0.03),
            switch_deviation_m2=pf_params.get("switch_deviation_m2", 0.02),
            switch_trailing_stop=pf_params.get("switch_trailing_stop", 0.05),
        )
        reports_data.append(generate_portfolio_reports(
            cl, e, x, s, "Polyfit-Switch",
            {**pf_params,
             "train_period": pf_best.get("train_period", "N/A"),
             "test_period": pf_best.get("test_period", "N/A"),
             "test_return": f"{pf_best['test_return']:.2%}"},
            reports_dir=REPORTS_DIR,
            open_=op_aligned,
        ))

    # 生成汇总索引页面，链接到所有报告的详情页
    if reports_data:
        index_html = build_index_html(reports_data, reports_dir=REPORTS_DIR)
        index_path = f"{REPORTS_DIR}/index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_html)
        print(f"  Index → {index_path}")

    print(f"\nDone.")
