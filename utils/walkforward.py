"""Walk-Forward 滚动窗口分析模块。

按日历年划分数据，生成训练/测试窗口，实现：
  训练 N 年 → 测试下一年 的滚动窗口回测流程。

核心概念：
  - 完整年（full year）：年初有数据（1/10 前开始）且年末有数据（12/20 后结束）
  - 不完整首年/末年：分别用作预热（warmup）和额外测试
  - Warmup：测试期之前的数据，用于指标计算的预热（避免 NaN）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .gpu import gpu


@dataclass
class WFWWindow:
    """Walk-Forward 窗口定义。

    每个窗口包含：
      - warmup:  训练期之前的一整年数据（用于指标预热，不参与信号生成）
      - train:   N 个完整年的训练数据
      - test:    紧随训练期之后的一整年测试数据
    """
    train_years: int            # 训练年数
    warmup_start: int           # 预热起始 iloc
    train_start: int            # 训练起始 iloc
    test_start: int             # 测试起始 iloc
    test_end: int               # 测试结束 iloc（不包含）
    warmup_label: str           # 预热年份标签
    train_label: str            # 训练年份标签
    test_label: str             # 测试年份标签


def _year_boundaries(index: pd.DatetimeIndex) -> dict[int, tuple[int, int]]:
    """返回 {年份: (起始iloc, 结束iloc_不包含)} 映射。

    用于快速定位每个日历年对应的数据位置。
    """
    years = {}
    for yr in sorted(set(index.year)):
        mask = index.year == yr
        idx = np.where(mask)[0]
        years[yr] = (int(idx[0]), int(idx[-1]) + 1)
    return years


def _is_full_year(index: pd.DatetimeIndex, year: int) -> bool:
    """判断某年是否为「完整年」。

    标准：该年有多条数据，且第一条在 1/10 前，最后一条在 12/20 后。
    这样排除了年初/年末数据稀疏的边界年份。
    """
    mask = index.year == year
    dates = index[mask]
    if len(dates) == 0:
        return False
    return dates[0].day <= 10 and dates[-1].month == 12 and dates[-1].day >= 20


def generate_windows(
    index: pd.DatetimeIndex,
    train_years: list[int] | None = None,
) -> list[WFWWindow]:
    """从 DatetimeIndex 生成 Walk-Forward 窗口定义。

    规则：
      - 第一个不完整年 → 仅用作 warmup（不参与训练或测试）
      - 最后一个不完整年 → 仅用作 test（不参与训练）
      - 完整年：可作训练或测试
      - 训练期 = N 个连续完整年
      - 测试期 = 训练期之后下一个完整年
      - 预热期 = 训练期之前一年（含可能的不完整首年）

    Args:
        index:       数据的 DatetimeIndex
        train_years: 训练窗口大小列表（如 [1, 2, 3]），默认 [1, 2, 3]

    Returns:
        WFWWindow 列表，按 train_years 和窗口起始年排列。
    """
    if train_years is None:
        train_years = [1, 2, 3]

    boundaries = _year_boundaries(index)
    year_list = sorted(boundaries.keys())
    first_yr = year_list[0]
    last_yr = year_list[-1]

    first_is_partial = not _is_full_year(index, first_yr)
    last_is_partial = not _is_full_year(index, last_yr)

    # 可用作训练的完整年范围
    trainable_start = first_yr + (1 if first_is_partial else 0)
    trainable_end = last_yr - (1 if last_is_partial else -1)  # 含

    windows: list[WFWWindow] = []

    for n in train_years:
        # t_start: 训练起始年，变化范围覆盖所有可能的训练窗口
        for t_start in range(trainable_start, trainable_end - n + 2):
            t_end = t_start + n  # 训练结束年（不包含）
            test_yr = t_end
            warmup_yr = t_start - 1

            # 测试年不能超出数据范围
            if test_yr > last_yr:
                continue

            # 预热年可能是不完整的首年
            warmup_start = boundaries[warmup_yr if warmup_yr >= first_yr else first_yr][0]
            train_start = boundaries[t_start][0]
            test_start = boundaries[test_yr][0]
            test_end = boundaries[test_yr][1]

            windows.append(WFWWindow(
                train_years=n,
                warmup_start=warmup_start,
                train_start=train_start,
                test_start=test_start,
                test_end=test_end,
                warmup_label=f"{warmup_yr}",
                train_label=f"{t_start}-{t_end - 1}" if n > 1 else str(t_start),
                test_label=f"{test_yr}",
            ))

    return windows


def _params_from_best(result_row: pd.Series, param_keys: list[str]) -> dict:
    """从扫描结果行中提取参数字典。"""
    return {k: result_row[k] for k in param_keys if k in result_row.index}


def run_walk_forward(
    close: pd.Series,
    strategy_name: str,
    scan_fn: Callable[[pd.Series], pd.DataFrame],          # 训练期扫描函数
    eval_fn: Callable[[pd.Series, int, dict], dict],       # 测试期评估函数
    param_keys: list[str],
    train_years: list[int] | None = None,
    best_selector: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """执行 Walk-Forward 滚动窗口分析。

    对每个窗口：
      1. 在训练集上运行参数扫描（scan_fn），选取最优参数
      2. 用最优参数在测试集上生成信号并回测（eval_fn）
      3. 记录训练/测试指标和参数

    Args:
        close:         完整收盘价序列
        strategy_name: 策略标签（如 "MA", "MA-Switch"）
        scan_fn:       参数扫描函数: (close_train_only) → DataFrame
        eval_fn:       测试评估函数: (close_warmup_all, test_offset, best_params) → dict
        param_keys:    要保存的参数列名
        train_years:   训练窗口大小列表，默认 [1, 2, 3]
        best_selector: 最优参数选取函数: (scan_results) → best_row。
                       默认 select_by_return（按 total_return 选取）。

    Returns:
        DataFrame，每行对应一个窗口的训练/测试结果。
    """
    if best_selector is None:
        from .scoring import select_by_return
        best_selector = select_by_return

    windows = generate_windows(close.index, train_years)

    rows = []
    for w in windows:
        # ── 训练阶段 ──
        # 传入 warmup+train 数据用于指标预热，scan 内部会自动处理
        close_warmup_train = close.iloc[w.warmup_start:w.test_start]
        close_train_only = close.iloc[w.train_start:w.test_start]
        scan_results = scan_fn(close_train_only)

        if scan_results.empty:
            continue

        # 按评分策略选取最优参数
        best = best_selector(scan_results)
        best_params = _params_from_best(best, param_keys)

        # ── 测试阶段 ──
        # eval_fn 接收完整的 warmup+train+test 数据，自行截取测试部分
        # 这样指标计算可以使用 train 的数据做预热
        close_warmup_all = close.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start
        test_metrics = eval_fn(close_warmup_all, test_offset, best_params)

        # 计算测试期的买入持有收益（基准）
        test_close = close.iloc[w.test_start:w.test_end]
        bh_return = ((test_close.iloc[-1] - test_close.iloc[0]) / test_close.iloc[0]
                     if len(test_close) >= 2 else 0.0)

        row = {
            "strategy": strategy_name,
            "train_years": w.train_years,
            "train_period": (
                f"{close_train_only.index[0].date()}→"
                f"{close_train_only.index[-1].date()}"
            ),
            "test_period": (
                f"{test_close.index[0].date()}→"
                f"{test_close.index[-1].date()}"
            ),
            "train_return": best["total_return"],
            "train_sharpe": best["sharpe_ratio"],
            "train_max_dd": best["max_drawdown"],
            "buy_hold_return": bh_return,
            **test_metrics,
            **best_params,
        }
        rows.append(row)

        # 逐窗口打印进度
        excess = test_metrics.get("test_return", 0) - bh_return
        print(f"  [{strategy_name}] train={w.train_label} test={w.test_label}  "
              f"train_ret={best['total_return']:.1%}  "
              f"test_ret={test_metrics.get('test_return', 0):.1%}  "
              f"BH={bh_return:.1%}  α={excess:+.1%}  "
              f"trades={test_metrics.get('num_trades', 0)}")

    return pd.DataFrame(rows)


def print_walk_forward_summary(results: pd.DataFrame, strategy_name: str) -> None:
    """打印 Walk-Forward 汇总统计。

    按训练年数分组展示：
      - 平均测试收益、平均买入持有收益、平均超额收益（α）
      - 跑赢买入持有的窗口比例
      - 平均 Sharpe 比率
      - 各窗口测试收益明细
    """
    if results.empty:
        print(f"\n  {strategy_name}: no results")
        return

    print(f"\n{'=' * 90}")
    print(f"  {strategy_name} — Walk-Forward Summary")
    print(f"{'=' * 90}")

    # 按训练年数分组
    for n, grp in results.groupby("train_years"):
        windows_n = len(grp)
        pos_test = (grp["test_return"] > 0).sum()
        pos_excess = (grp["test_return"] > grp["buy_hold_return"]).sum()
        avg_test = grp["test_return"].mean()
        avg_bh = grp["buy_hold_return"].mean()
        avg_excess = (grp["test_return"] - grp["buy_hold_return"]).mean()
        print(f"\n  Training = {n} year(s)  |  {windows_n} windows")
        print(f"    Avg test return:  {avg_test:+.1%}")
        print(f"    Avg buy & hold:   {avg_bh:+.1%}")
        print(f"    Avg excess (α):   {avg_excess:+.1%}")
        print(f"    Beat BH:          {pos_excess}/{windows_n} ({pos_excess/windows_n:.0%})")
        print(f"    Avg test Sharpe:  {grp['test_sharpe'].mean():.3f}")
        print(f"    Test returns:     {'  '.join(f'{v:+.1%}' for v in grp['test_return'])}")

    # 总体统计
    overall_excess = (results["test_return"] - results["buy_hold_return"]).mean()
    beat_bh = (results["test_return"] > results["buy_hold_return"]).sum()
    print(f"\n  Overall ({len(results)} windows):")
    print(f"    Mean test return:  {results['test_return'].mean():+.1%}")
    print(f"    Mean buy & hold:   {results['buy_hold_return'].mean():+.1%}")
    print(f"    Mean excess (α):   {overall_excess:+.1%}")
    print(f"    Beat BH:           {beat_bh}/{len(results)} ({beat_bh/len(results):.0%})")
    print(f"    Mean test Sharpe:  {results['test_sharpe'].mean():.3f}")
