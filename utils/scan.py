"""参数扫描通用逻辑。

提供 indicator_and_scan 函数，封装了"计算指标 → 生成信号 → 回测 → 收集结果"
的完整扫描流程，支持 GPU 批量模式和 CPU 逐组合模式。

执行模型：默认使用次日开盘价成交（消除前视偏差）。
"""

from itertools import product

import numpy as np
import pandas as pd

from .gpu import gpu
from .backtest import run_backtest, run_backtest_batch


def indicator_and_scan(
    close: pd.Series,
    label: str,
    indicator_fn,
    window_param_name: str,
    windows: list[int],
    grid_pcts: list[float],
    vol_scales: list[float],
    trend_sens: list[float],
    tpg_values: list[float],
    slg_values: list[float],
    signal_fn,
    signal_batch_fn,
    open_: pd.Series | None = None,  # 开盘价，用于 next-bar Open 成交
) -> pd.DataFrame:
    """网格策略通用参数扫描。

    对给定的参数候选集做全排列扫描。当 MLX 可用且 signal_batch_fn 不为 None 时：
      - 同一 MA 窗口下，所有参数组合的指标数据相同
      - 批量信号生成 + 批量回测（GPU）

    成交执行：传入 open_ 则使用 next-bar Open 成交（消除前视偏差）。

    Args:
        close:              收盘价序列
        indicator_fn:       指标计算函数，签名为 (close, window) → DataFrame
        window_param_name:  窗口参数列名
        windows:            MA 窗口候选列表
        grid_pcts ~ slg_values: 网格策略各参数候选列表
        signal_fn:          CPU 信号生成函数
        signal_batch_fn:    GPU 批量信号生成函数
        open_:              可选，开盘价序列，用于 next-bar Open 执行

    Returns:
        DataFrame，每行包含一组参数的指标结果。
    """
    total = (
        len(windows) * len(grid_pcts) * len(vol_scales)
        * len(trend_sens) * len(tpg_values) * len(slg_values)
    )
    results = []
    count = 0
    use_gpu = gpu()["mlx_available"]

    # 准备 numpy 版本的 Open 数据（GPU 路径用）
    open_arr = open_.values if open_ is not None else None

    for w in windows:
        indicators = indicator_fn(close, w)
        close_arr = close.values
        dev_pct_arr = indicators.iloc[:, 1].values
        dev_trend_arr = indicators.iloc[:, 2].values
        vol_arr = indicators.iloc[:, 3].values

        param_combos = list(product(
            grid_pcts, vol_scales, trend_sens, tpg_values, slg_values
        ))

        if use_gpu and signal_batch_fn is not None:
            # ── GPU 路径：批量信号生成 + 批量回测（MLX）──
            bgp_a = np.array([p[0] for p in param_combos])
            vs_a = np.array([p[1] for p in param_combos])
            ts_a = np.array([p[2] for p in param_combos])
            tpg_a = np.array([p[3] for p in param_combos])
            slg_a = np.array([p[4] for p in param_combos])

            entries_b, exits_b, sizes_b = signal_batch_fn(
                dev_pct_arr, dev_trend_arr, vol_arr,
                bgp_a, vs_a, ts_a, tpg_a, slg_a,
            )

            bt_metrics = run_backtest_batch(
                close_arr, entries_b, exits_b, sizes_b,
                n_combos=len(param_combos),
                open_=open_arr,
            )

            for idx, (bgp, vs, ts, tpg, slg) in enumerate(param_combos):
                row = bt_metrics[idx]
                if int(row[4]) == 0:
                    continue

                metrics = {
                    "total_return": row[0],
                    "sharpe_ratio": row[1],
                    "max_drawdown": row[2],
                    "calmar_ratio": row[3],
                    "num_trades": int(row[4]),
                    "win_rate": row[5],
                    window_param_name: w,
                    "base_grid_pct": bgp,
                    "volatility_scale": vs,
                    "trend_sensitivity": ts,
                    "take_profit_grid": tpg,
                    "stop_loss_grid": slg,
                }
                results.append(metrics)
                count += 1
        else:
            # ── CPU 路径：逐组合循环 ──
            for bgp, vs, ts, tpg, slg in param_combos:
                entries, exits, sizes = signal_fn(
                    close_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                    base_grid_pct=bgp, volatility_scale=vs,
                    trend_sensitivity=ts, take_profit_grid=tpg,
                    stop_loss_grid=slg,
                )

                if entries.sum() == 0:
                    continue

                metrics = run_backtest(close, entries, exits, sizes, open_=open_)
                metrics[window_param_name] = w
                metrics["base_grid_pct"] = bgp
                metrics["volatility_scale"] = vs
                metrics["trend_sensitivity"] = ts
                metrics["take_profit_grid"] = tpg
                metrics["stop_loss_grid"] = slg
                results.append(metrics)
                count += 1

    return pd.DataFrame(results)


def print_top(results: pd.DataFrame, name: str, param_cols: list[str], top_n: int = 10):
    """格式化打印扫描结果的 Top N（按 Total Return 排序）。"""
    print(f"\n{'=' * 80}")
    print(f"  {name} — Top {top_n} by Total Return")
    print(f"{'=' * 80}")
    display_cols = param_cols + ["total_return", "sharpe_ratio", "max_drawdown",
                                  "calmar_ratio", "num_trades", "win_rate"]
    top = results.nlargest(top_n, "total_return")
    fmt = top[display_cols].copy()
    fmt["total_return"] = fmt["total_return"].apply("{:.2%}".format)
    fmt["max_drawdown"] = fmt["max_drawdown"].apply("{:.2%}".format)
    fmt["sharpe_ratio"] = fmt["sharpe_ratio"].apply("{:.3f}".format)
    fmt["calmar_ratio"] = fmt["calmar_ratio"].apply("{:.3f}".format)
    fmt["win_rate"] = fmt["win_rate"].apply("{:.2%}".format)
    print(fmt.to_string(index=False))
