"""Walk-forward analysis — rolling training/test split by calendar year.

Splits data into warmup (incomplete beginning year), training (N full years),
and test (1 year).  Incomplete ending year is used as test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .gpu import gpu


@dataclass
class WFWWindow:
    train_years: int
    warmup_start: int
    train_start: int
    test_start: int
    test_end: int  # exclusive
    warmup_label: str
    train_label: str
    test_label: str


def _year_boundaries(index: pd.DatetimeIndex) -> dict[int, tuple[int, int]]:
    """Return {year: (start_iloc, end_iloc_exclusive)} for each calendar year."""
    years = {}
    for yr in sorted(set(index.year)):
        mask = index.year == yr
        idx = np.where(mask)[0]
        years[yr] = (int(idx[0]), int(idx[-1]) + 1)
    return years


def _is_full_year(index: pd.DatetimeIndex, year: int) -> bool:
    """A year is 'full' if it starts by Jan 10 and ends after Dec 20."""
    mask = index.year == year
    dates = index[mask]
    if len(dates) == 0:
        return False
    return dates[0].day <= 10 and dates[-1].month == 12 and dates[-1].day >= 20


def generate_windows(
    index: pd.DatetimeIndex,
    train_years: list[int] | None = None,
) -> list[WFWWindow]:
    """Generate walk-forward window definitions from a DatetimeIndex.

    Rules:
    - First year (incomplete) → warmup only
    - Last year (incomplete) → test only
    - Training = N consecutive full years
    - Test = the full year immediately after training
    - Warmup = the year immediately before training (may be incomplete first year)

    Parameters
    ----------
    index : pd.DatetimeIndex
    train_years : list of training-window sizes (in years). Default [1, 2, 3].
    """
    if train_years is None:
        train_years = [1, 2, 3]

    boundaries = _year_boundaries(index)
    year_list = sorted(boundaries.keys())
    first_yr = year_list[0]
    last_yr = year_list[-1]

    first_is_partial = not _is_full_year(index, first_yr)
    last_is_partial = not _is_full_year(index, last_yr)

    # Full years available for training
    trainable_start = first_yr + (1 if first_is_partial else 0)
    trainable_end = last_yr - (1 if last_is_partial else -1)  # inclusive

    windows: list[WFWWindow] = []

    for n in train_years:
        for t_start in range(trainable_start, trainable_end - n + 2):
            t_end = t_start + n  # exclusive
            test_yr = t_end
            warmup_yr = t_start - 1

            if test_yr > last_yr:
                continue

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
    """Extract parameter dict from a scan result row."""
    return {k: result_row[k] for k in param_keys if k in result_row.index}


def run_walk_forward(
    close: pd.Series,
    strategy_name: str,
    scan_fn: Callable[[pd.Series], pd.DataFrame],
    eval_fn: Callable[[pd.Series, dict], dict],
    param_keys: list[str],
    train_years: list[int] | None = None,
) -> pd.DataFrame:
    """Run walk-forward analysis across multiple training-window sizes.

    Parameters
    ----------
    close : full price series
    strategy_name : label
    scan_fn : (close_train_only) → DataFrame of scan results
    eval_fn : (close_warmup_all, test_offset, best_params) → dict with keys
             test_return, test_sharpe, test_max_dd, num_trades, win_rate
    param_keys : columns to extract from best scan row as params
    train_years : training window sizes in years, default [1, 2, 3]

    Returns DataFrame with per-window train/test metrics + best parameters.
    """
    windows = generate_windows(close.index, train_years)

    rows = []
    for w in windows:
        # --- Training phase ---
        # Pass warmup+train data, scan will handle indicator warmup internally
        close_warmup_train = close.iloc[w.warmup_start:w.test_start]
        close_train_only = close.iloc[w.train_start:w.test_start]
        scan_results = scan_fn(close_train_only)

        if scan_results.empty:
            continue

        best = scan_results.nlargest(1, "total_return").iloc[0]
        best_params = _params_from_best(best, param_keys)

        # --- Test phase ---
        # eval_fn receives:
        #   close_warmup_all: warmup+train+test (for indicator warmup)
        #   test_offset: iloc within close_warmup_all where test begins
        close_warmup_all = close.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start
        test_metrics = eval_fn(close_warmup_all, test_offset, best_params)

        row = {
            "strategy": strategy_name,
            "train_years": w.train_years,
            "train_period": (
                f"{close_train_only.index[0].date()}→"
                f"{close_train_only.index[-1].date()}"
            ),
            "test_period": (
                f"{close.iloc[w.test_start:w.test_end].index[0].date()}→"
                f"{close.iloc[w.test_start:w.test_end].index[-1].date()}"
            ),
            "train_return": best["total_return"],
            "train_sharpe": best["sharpe_ratio"],
            "train_max_dd": best["max_drawdown"],
            **test_metrics,
            **best_params,
        }
        rows.append(row)

        print(f"  [{strategy_name}] train={w.train_label} test={w.test_label}  "
              f"train_ret={best['total_return']:.1%}  "
              f"test_ret={test_metrics.get('test_return', 0):.1%}  "
              f"trades={test_metrics.get('num_trades', 0)}")

    return pd.DataFrame(rows)


def print_walk_forward_summary(results: pd.DataFrame, strategy_name: str) -> None:
    """Print walk-forward summary statistics."""
    if results.empty:
        print(f"\n  {strategy_name}: no results")
        return

    print(f"\n{'=' * 90}")
    print(f"  {strategy_name} — Walk-Forward Summary")
    print(f"{'=' * 90}")

    # Per train_years summary
    for n, grp in results.groupby("train_years"):
        windows_n = len(grp)
        pos_test = (grp["test_return"] > 0).sum()
        avg_test = grp["test_return"].mean()
        avg_train = grp["train_return"].mean()
        print(f"\n  Training = {n} year(s)  |  {windows_n} windows")
        print(f"    Avg train return: {avg_train:+.1%}")
        print(f"    Avg test return:  {avg_test:+.1%}")
        print(f"    Test win rate:    {pos_test}/{windows_n} ({pos_test/windows_n:.0%})")
        print(f"    Avg test Sharpe:  {grp['test_sharpe'].mean():.3f}")
        print(f"    Test returns:     {'  '.join(f'{v:+.1%}' for v in grp['test_return'])}")

    print(f"\n  Overall ({len(results)} windows):")
    print(f"    Mean test return:  {results['test_return'].mean():+.1%}")
    print(f"    Mean test Sharpe:  {results['test_sharpe'].mean():.3f}")
    print(f"    Positive windows:  {(results['test_return'] > 0).sum()}/{len(results)}")
