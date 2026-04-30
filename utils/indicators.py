"""Technical indicator computations with optional GPU acceleration."""

import numpy as np
import pandas as pd

from .gpu import gpu, xp


def compute_rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling volatility (std of returns).  GPU-accelerated via CuPy when available."""
    returns = close.pct_change()
    if gpu()["cupy_available"]:
        cp = xp()
        ret = cp.asarray(returns.values, dtype=cp.float64)
        n = len(ret)
        ret_clean = cp.nan_to_num(ret, nan=0.0)
        ret2_clean = ret_clean * ret_clean
        ones = cp.ones(window, dtype=cp.float64)
        sum_ret = cp.convolve(ret_clean, ones, mode="full")[:n]
        sum_ret2 = cp.convolve(ret2_clean, ones, mode="full")[:n]
        count = cp.clip(cp.arange(1, n + 1, dtype=cp.float64), 2.0, float(window))
        mean_ret = sum_ret / count
        pop_var = cp.maximum(sum_ret2 / count - mean_ret * mean_ret, 0.0)
        sample_var = pop_var * count / (count - 1.0)
        result_arr = cp.sqrt(sample_var)
        result_arr[:window] = cp.nan
        return pd.Series(cp.asnumpy(result_arr), index=returns.index)
    return returns.rolling(window).std()


def compute_deviation_trend(dev_pct: pd.Series, window: int = 5) -> pd.Series:
    """Slope of deviation percentage over a short window."""
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_diff = x - x_mean
    denom = np.sum(x_diff ** 2)

    def _slope(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        return np.sum(x_diff * (y - y_mean)) / denom

    return dev_pct.rolling(window, min_periods=window).apply(_slope, raw=True)


def compute_ma_indicators(
    close: pd.Series, ma_window: int, vol_window: int = 20, trend_window: int = 5,
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


def compute_ma_switch_indicators(
    close: pd.Series,
    ma_window: int,
    ma_windows: list[int] | None = None,
    vol_window: int = 20,
    trend_window: int = 5,
) -> pd.DataFrame:
    """MA indicators + extra MAs for switch strategy."""
    if ma_windows is None:
        ma_windows = [5, 10, 20, 60]
    df = compute_ma_indicators(close, ma_window, vol_window, trend_window)
    for mw in ma_windows:
        df[f"MA{mw}"] = close.rolling(mw, min_periods=mw).mean()
    return df
