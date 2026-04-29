"""
VectorBT parameter scanning for dynamic grid strategies.

Reimplements PolyfitDynamicGridStrategy and MovingAverageDynamicGridStrategy
logic using VectorBT for portfolio backtesting and parameter optimization.
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

DATA_PATH = "data/512890.SH_hfq.parquet"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ══════════════════════════════════════════════════════════════════
# Indicator computations
# ══════════════════════════════════════════════════════════════════

def rolling_linear_fit_pred(series: pd.Series, window: int) -> pd.Series:
    """Rolling degree-1 polynomial fit prediction at the last point of each window."""
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_diff = x - x_mean
    denom = np.sum(x_diff**2)

    def _pred(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        slope = np.sum(x_diff * (y - y_mean)) / denom
        return slope * (window - 1 - x_mean) + y_mean

    return series.rolling(window, min_periods=window).apply(_pred, raw=True)


def compute_rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window).std()


def compute_deviation_trend(dev_pct: pd.Series, window: int = 5) -> pd.Series:
    """Slope of deviation percentage over a short window."""
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_diff = x - x_mean
    denom = np.sum(x_diff**2)

    def _slope(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        return np.sum(x_diff * (y - y_mean)) / denom

    return dev_pct.rolling(window, min_periods=window).apply(_slope, raw=True)


def compute_polyfit_indicators(
    close: pd.Series, polyfit_window: int, vol_window: int = 20, trend_window: int = 5
) -> pd.DataFrame:
    poly_base = rolling_linear_fit_pred(close, polyfit_window)
    poly_dev_pct = (close - poly_base) / poly_base
    poly_dev_trend = compute_deviation_trend(poly_dev_pct, trend_window)
    rolling_vol_pct = compute_rolling_volatility(close, vol_window)
    return pd.DataFrame(
        {"PolyBasePred": poly_base, "PolyDevPct": poly_dev_pct,
         "PolyDevTrend": poly_dev_trend, "RollingVolPct": rolling_vol_pct},
        index=close.index,
    )


def compute_ma_indicators(
    close: pd.Series, ma_window: int, vol_window: int = 20, trend_window: int = 5
) -> pd.DataFrame:
    ma_base = close.rolling(ma_window, min_periods=ma_window).mean()
    ma_dev_pct = (close - ma_base) / ma_base
    ma_dev_trend = compute_deviation_trend(ma_dev_pct, trend_window)
    rolling_vol_pct = compute_rolling_volatility(close, vol_window)
    return pd.DataFrame(
        {"MABase": ma_base, "MADevPct": ma_dev_pct,
         "MADevTrend": ma_dev_trend, "RollingVolPct": rolling_vol_pct},
        index=close.index,
    )


# ══════════════════════════════════════════════════════════════════
# Signal generation (bar-by-bar, replicating original strategy logic)
# ══════════════════════════════════════════════════════════════════

def generate_grid_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate entry/exitsignals bar by bar, replicating the original strategy's
    `next()` logic.

    Returns entries, exits, sizes arrays.
    """
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)

    in_position = False
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    cooldown = 0

    for i in range(n):
        if (np.isnan(dev_pct[i]) or np.isnan(dev_trend[i])
                or np.isnan(rolling_vol_pct[i]) or close[i] <= 0):
            continue

        vol_mult = 1.0 + volatility_scale * max(rolling_vol_pct[i], 0.0)
        dynamic_grid_step = (
            base_grid_pct
            * (1.0 + trend_sensitivity * abs(dev_trend[i]))
            * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        if not in_position:
            if cooldown > 0:
                cooldown -= 1
                continue

            signal_strength = abs(dev_pct[i]) / max(dynamic_grid_step, 1e-9)
            entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
            entry_threshold = -entry_lvl * dynamic_grid_step

            if dev_pct[i] <= entry_threshold and signal_strength >= min_signal_strength:
                entries[i] = True
                sizes[i] = position_size
                in_position = True
                entry_bar = i
                entry_level = entry_lvl
                entry_grid_step = dynamic_grid_step
        else:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days

            ref_step = max(dynamic_grid_step, entry_grid_step) if not np.isnan(entry_grid_step) else dynamic_grid_step
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid

            if hold_limit or dev_pct[i] >= tp_threshold or dev_pct[i] <= -sl_threshold:
                exits[i] = True
                in_position = False
                cooldown = cooldown_days

    # Close any open position at the end
    if in_position:
        exits[-1] = True

    return entries, exits, sizes


def run_backtest(close: pd.Series, entries: np.ndarray, exits: np.ndarray,
                 sizes: np.ndarray, init_cash: float = 100_000.0) -> dict:
    """Run a single VectorBT backtest and return key metrics."""
    pf = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(entries, index=close.index),
        exits=pd.Series(exits, index=close.index),
        size=pd.Series(sizes, index=close.index),
        size_type="percent",
        init_cash=init_cash,
        freq="D",
    )
    dd = pf.max_drawdown()
    return {
        "total_return": pf.total_return(),
        "sharpe_ratio": pf.sharpe_ratio(),
        "max_drawdown": dd,
        "calmar_ratio": pf.total_return() / abs(dd) if dd != 0 else float("nan"),
        "num_trades": pf.trades.count(),
        "win_rate": pf.trades.win_rate() if pf.trades.count() > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════
# Parameter scanning
# ══════════════════════════════════════════════════════════════════

def _indicator_and_scan(
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
) -> pd.DataFrame:
    """Shared scan loop for both strategy types."""
    total = (
        len(windows) * len(grid_pcts) * len(vol_scales)
        * len(trend_sens) * len(tpg_values) * len(slg_values)
    )
    results = []
    count = 0

    for w in windows:
        indicators = indicator_fn(close, w)
        close_arr = close.values
        dev_pct_arr = indicators.iloc[:, 1].values
        dev_trend_arr = indicators.iloc[:, 2].values
        vol_arr = indicators.iloc[:, 3].values

        for bgp, vs, ts, tpg, slg in product(
            grid_pcts, vol_scales, trend_sens, tpg_values, slg_values
        ):
            entries, exits, sizes = generate_grid_signals(
                close_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                base_grid_pct=bgp, volatility_scale=vs,
                trend_sensitivity=ts, take_profit_grid=tpg,
                stop_loss_grid=slg,
            )

            if entries.sum() == 0:
                continue

            metrics = run_backtest(close, entries, exits, sizes)
            metrics[window_param_name] = w
            metrics["base_grid_pct"] = bgp
            metrics["volatility_scale"] = vs
            metrics["trend_sensitivity"] = ts
            metrics["take_profit_grid"] = tpg
            metrics["stop_loss_grid"] = slg
            results.append(metrics)

            count += 1
            if count % 200 == 0:
                print(f"  [{label}] {count}/{total}")

    return pd.DataFrame(results)


def scan_polyfit_strategy(close: pd.Series) -> pd.DataFrame:
    print("  Polyfit parameter scan…")
    return _indicator_and_scan(
        close, "Polyfit", compute_polyfit_indicators, "polyfit_window",
        windows=[500, 600, 750],
        grid_pcts=[0.008, 0.01, 0.012, 0.015, 0.02],
        vol_scales=[0.5, 1.0, 2.0],
        trend_sens=[4.0, 8.0, 12.0],
        tpg_values=[0.7, 0.85, 1.0],
        slg_values=[1.2, 1.6, 2.0],
    )


def scan_ma_strategy(close: pd.Series) -> pd.DataFrame:
    print("  MA parameter scan…")
    return _indicator_and_scan(
        close, "MA", compute_ma_indicators, "ma_window",
        windows=[20, 50, 100, 200],
        grid_pcts=[0.008, 0.01, 0.012, 0.015, 0.02],
        vol_scales=[0.5, 1.0, 2.0],
        trend_sens=[4.0, 8.0, 12.0],
        tpg_values=[0.7, 0.85, 1.0],
        slg_values=[1.2, 1.6, 2.0],
    )


# ══════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════

def print_top(results: pd.DataFrame, name: str, param_cols: list[str], top_n: int = 10):
    print(f"\n{'='*80}")
    print(f"  {name} — Top {top_n} by Total Return")
    print(f"{'='*80}")
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


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data…")
    df = load_data()
    close = df["Close"]
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")

    # --- Polyfit scan ---
    pf_results = scan_polyfit_strategy(close)
    pf_results.to_csv("polyfit_scan_results.csv", index=False)
    print_top(pf_results, "PolyfitDynamicGridStrategy",
              ["polyfit_window", "base_grid_pct", "volatility_scale",
               "trend_sensitivity", "take_profit_grid", "stop_loss_grid"])

    # --- MA scan ---
    ma_results = scan_ma_strategy(close)
    ma_results.to_csv("ma_scan_results.csv", index=False)
    print_top(ma_results, "MovingAverageDynamicGridStrategy",
              ["ma_window", "base_grid_pct", "volatility_scale",
               "trend_sensitivity", "take_profit_grid", "stop_loss_grid"])

    print("\nDone. Results saved to polyfit_scan_results.csv and ma_scan_results.csv")

