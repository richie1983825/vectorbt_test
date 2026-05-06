"""技术指标计算模块。

计算各类策略需要的技术指标，可选 MLX（Apple Silicon GPU）加速。
主要指标：
  - 滚动波动率（Rolling Volatility）
  - 偏离趋势（Deviation Trend）—— 偏离率的短期斜率
  - MA 基准指标（MABase, MADevPct, MADevTrend, RollingVolPct）
  - MA-Switch 扩展指标（额外多根 MA）
  - Polyfit 基准指标（PolyBasePred, PolyDevPct, PolyDevTrend）
    —— 使用滑动窗口线性回归拟合基线，而非简单均线
"""

import numpy as np
import pandas as pd

from .gpu import gpu, get_mlx


def compute_rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """计算滚动波动率（收益率的标准差）。

    Apple Silicon 上使用 MLX 卷积加速，否则回退到 pandas rolling。
    使用样本标准差（除以 n-1），与 pandas .std() 行为一致。
    """
    returns = close.pct_change()
    info = gpu()

    if info["mlx_available"]:
        mx = get_mlx()
        ret = mx.array(returns.values.astype(np.float32))
        n = len(ret)
        ret_clean = mx.where(mx.isnan(ret), mx.array(0.0, dtype=mx.float32), ret)
        ret2_clean = ret_clean * ret_clean
        ones = mx.ones((window,), dtype=mx.float32)
        sum_ret = mx.convolve(ret_clean, ones, mode="full")[:n]
        sum_ret2 = mx.convolve(ret2_clean, ones, mode="full")[:n]
        count = mx.clip(mx.arange(1, n + 1, dtype=mx.float32),
                        mx.array(2.0, dtype=mx.float32),
                        mx.array(float(window), dtype=mx.float32))
        mean_ret = sum_ret / count
        pop_var = mx.maximum(sum_ret2 / count - mean_ret * mean_ret,
                             mx.array(0.0, dtype=mx.float32))
        sample_var = pop_var * count / (count - mx.array(1.0, dtype=mx.float32))
        result_arr = mx.sqrt(sample_var)
        result_arr = mx.where(
            mx.arange(n, dtype=mx.float32) < mx.array(float(window), dtype=mx.float32),
            mx.array(float('nan'), dtype=mx.float32), result_arr)
        mx.eval(result_arr)
        return pd.Series(np.array(result_arr).astype(np.float64), index=returns.index)

    return returns.rolling(window).std()


def compute_deviation_trend(dev_pct: pd.Series, window: int = 5) -> pd.Series:
    """计算偏离百分比的短期斜率（线性回归系数）。

    用于判断偏离是在扩大还是收敛：
      - 正斜率 → 价格在向均线回归
      - 负斜率 → 价格在进一步偏离均线
    """
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_diff = x - x_mean
    denom = np.sum(x_diff ** 2)  # 最小二乘分母，常量

    def _slope(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        return np.sum(x_diff * (y - y_mean)) / denom

    return dev_pct.rolling(window, min_periods=window).apply(_slope, raw=True)


def compute_ma_indicators(
    close: pd.Series, ma_window: int, vol_window: int = 20, trend_window: int = 5,
) -> pd.DataFrame:
    """计算 MA 网格策略所需的核心指标。

    指标说明：
      - MABase:       SMA(close, ma_window)，均线基准线
      - MADevPct:     (close - MABase) / MABase，价格偏离均线的百分比
      - MADevTrend:   MADevPct 的短期斜率，判断偏离方向
      - RollingVolPct: 滚动波动率，用于动态调整网格宽度

    Args:
        close:        收盘价序列
        ma_window:    均线周期
        vol_window:   波动率计算窗口
        trend_window: 偏离趋势计算窗口
    """
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
    """计算 MA-Switch 策略所需指标。

    在 MA 网格指标基础上，额外计算多根不同周期的均线，
    供 Switch 模式的均线交叉信号使用。

    Args:
        close:      收盘价序列
        ma_window:  MA 基准线周期
        ma_windows: 额外均线周期列表（如 [5, 10, 20, 60]）
        vol_window: 波动率窗口
        trend_window: 趋势窗口
    """
    if ma_windows is None:
        ma_windows = [5, 10, 20, 60]
    df = compute_ma_indicators(close, ma_window, vol_window, trend_window)
    for mw in ma_windows:
        df[f"MA{mw}"] = close.rolling(mw, min_periods=mw).mean()
    return df


# ══════════════════════════════════════════════════════════════════
# Polyfit 基准指标 — 使用滑动窗口线性回归拟合基线
#
# 与 MA 基准的区别：
#   - MA 基准在趋势行情中会滞后，且容易在震荡中频繁穿越
#   - Polyfit 基准用长期线性回归（通常 252 天≈1 年）预测当前值，
#     能更好地捕捉价格的长期运行中枢，减少噪音干扰
#
# PolyDevTrend 使用 EMA(diff) 而非斜率回归，对短期方向变化更敏感。
# ══════════════════════════════════════════════════════════════════

def _compute_polyfit_baseline_cpu(
    close: np.ndarray, fit_window_days: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU: 滑动窗口线性回归 (Python for-loop fallback)。"""
    n = len(close)
    pred = np.full(n, np.nan, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)

    effective_fit_window = min(max(int(fit_window_days), 5), n - 1)
    if n < effective_fit_window:
        return pred, slope

    x = np.arange(effective_fit_window, dtype=np.float64)
    x_mean = float(x.mean())
    x_center = x - x_mean
    x_var = float((x_center ** 2).sum())

    for i in range(effective_fit_window - 1, n):
        y = close[i - effective_fit_window + 1: i + 1]
        if np.isnan(y).any():
            continue
        y_mean = float(y.mean())
        beta = float(np.dot(x_center, y - y_mean) / max(x_var, 1e-12))
        alpha = y_mean - beta * x_mean
        pred[i] = alpha + beta * (effective_fit_window - 1)
        slope[i] = beta

    return pred, slope


def _compute_polyfit_baseline_mlx(
    close: np.ndarray, fit_window_days: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """MLX GPU: 用卷积实现滑动窗口线性回归。

    数学推导:
      x = [0, 1, ..., w-1]
      β = (w·Σxy - Σx·Σy) / (w·Σx² - (Σx)²)
      α = (Σy - β·Σx) / w
      pred = α + β·(w-1)

    Σy = conv(close, ones(w))       — 滚动和
    Σxy = conv(close, rev(x))       — 滚动加权和
    """
    import mlx.core as mx

    n = len(close)
    w = min(max(int(fit_window_days), 5), n - 1)
    if n < w:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

    if np.isnan(close).any():
        return _compute_polyfit_baseline_cpu(close, fit_window_days)

    sum_x = float(w * (w - 1)) / 2.0
    sum_x2 = float(w * (w - 1) * (2 * w - 1)) / 6.0
    denom = float(w) * sum_x2 - sum_x * sum_x

    close_f32 = mx.array(close.astype(np.float32))

    ones = mx.ones((w,), dtype=mx.float32)
    sum_y = mx.convolve(close_f32, ones, mode="full")[:n]

    x_rev = mx.array(np.arange(w - 1, -1, -1, dtype=np.float32))
    sum_xy = mx.convolve(close_f32, x_rev, mode="full")[:n]

    w_f = mx.array(float(w), dtype=mx.float32)
    sx_f = mx.array(sum_x, dtype=mx.float32)
    denom_f = mx.array(denom, dtype=mx.float32)
    wm1_f = mx.array(float(w - 1), dtype=mx.float32)

    beta = (w_f * sum_xy - sx_f * sum_y) / denom_f
    alpha = (sum_y - beta * sx_f) / w_f
    pred = alpha + beta * wm1_f

    valid = mx.arange(n, dtype=mx.float32) >= mx.array(float(w - 1), dtype=mx.float32)
    nan_f = mx.array(float("nan"), dtype=mx.float32)
    pred = mx.where(valid, pred, nan_f)
    slope = mx.where(valid, beta, nan_f)

    mx.eval(pred, slope)
    return np.array(pred).astype(np.float64), np.array(slope).astype(np.float64)


def compute_polyfit_baseline(
    close: np.ndarray, fit_window_days: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """用滑动窗口线性回归计算基准线和斜率。

    对每一天，用前 fit_window_days 根 bar 做最小二乘拟合 y=α+βx，
    取拟合线在当前位置的预测值作为 PolyBasePred。

    Apple Silicon 上使用 MLX GPU 卷积加速，否则回退到 CPU Python 循环。

    返回值：
      pred:  [n] float64，多项式预测基准线
      slope: [n] float64，拟合斜率（β）
    """
    n = len(close)
    if gpu()["mlx_available"] and n > 10000:
        try:
            return _compute_polyfit_baseline_mlx(close, fit_window_days)
        except Exception:
            pass
    return _compute_polyfit_baseline_cpu(close, fit_window_days)


def compute_polyfit_deviation_trend(
    dev_pct: pd.Series, trend_window_days: int = 20,
) -> pd.Series:
    """计算偏离率的趋势（EMA 平滑后的差分）。

    使用 EMA(diff(dev_pct), span=trend_window_days) 替代斜率回归：
      - 正值：偏离在扩大（价格在向基准线回归）
      - 负值：偏离在收敛（价格在进一步远离基准线）

    这种计算方式比滚动回归斜率更平滑，减少「锯齿」信号。
    """
    return dev_pct.diff().ewm(
        span=max(3, int(trend_window_days)),
        adjust=False,
        min_periods=max(3, int(trend_window_days) // 2),
    ).mean()


def compute_polyfit_volatility(
    close: pd.Series, vol_window_days: int = 20,
) -> pd.Series:
    """计算滚动波动率（总体标准差 ddof=0）。

    与 compute_rolling_volatility 的区别：使用总体标准差（除以 N），
    与 512890 项目的回测框架保持一致。
    """
    return close.pct_change().rolling(
        window=max(2, int(vol_window_days)),
        min_periods=max(2, int(vol_window_days)),
    ).std(ddof=0)


def compute_polyfit_indicators(
    close: pd.Series,
    fit_window_days: int = 252,
    trend_window_days: int = 20,
    vol_window_days: int = 20,
) -> pd.DataFrame:
    """计算 Polyfit 网格策略所需的核心指标。

    指标说明：
      - PolyBasePred:  滑动线性回归拟合的基准线预测值
      - PolySlope:     拟合斜率（正值=上升趋势，负值=下降趋势）
      - PolyDevPct:    (close - PolyBasePred) / PolyBasePred，
                       价格偏离基准线的百分比
      - PolyDevTrend:  偏离率的 EMA 趋势（diff 的指数平滑）
      - RollingVolPct: 滚动波动率（日收益率的总体标准差）

    Args:
        close:             收盘价序列
        fit_window_days:   线性回归拟合窗口（通常 252 = 1 年）
        trend_window_days: 偏离趋势的 EMA span
        vol_window_days:   波动率计算窗口
    """
    close_arr = close.values.astype(np.float64)
    pred, slope = compute_polyfit_baseline(close_arr, fit_window_days)

    data = pd.DataFrame(index=close.index)
    data["PolyBasePred"] = pred
    data["PolySlope"] = slope
    data["PolyDevPct"] = close / data["PolyBasePred"] - 1.0
    data["PolyDevTrend"] = compute_polyfit_deviation_trend(
        data["PolyDevPct"], trend_window_days
    )
    data["RollingVolPct"] = compute_polyfit_volatility(close, vol_window_days)

    feature_cols = [
        "PolyBasePred", "PolySlope", "PolyDevPct",
        "PolyDevTrend", "RollingVolPct",
    ]
    return data.dropna(subset=feature_cols)


def compute_polyfit_switch_indicators(
    close: pd.Series,
    fit_window_days: int = 252,
    ma_windows: list[int] | None = None,
    trend_window_days: int = 20,
    vol_window_days: int = 20,
) -> pd.DataFrame:
    """计算 Polyfit-Switch 策略所需指标。

    在 Polyfit 指标基础上，额外计算多根不同周期的均线，
    供 Switch 模式的均线交叉（金叉/死叉）信号使用。

    Args:
        close:             收盘价序列
        fit_window_days:   线性回归拟合窗口
        ma_windows:        额外均线周期列表（如 [5, 10, 20, 60]）
        trend_window_days: 偏离趋势的 EMA span
        vol_window_days:   波动率计算窗口
    """
    if ma_windows is None:
        ma_windows = [5, 10, 20, 60]
    df = compute_polyfit_indicators(close, fit_window_days, trend_window_days, vol_window_days)
    for mw in ma_windows:
        df[f"MA{mw}"] = close.rolling(mw, min_periods=mw).mean()
    feature_cols = (
        ["PolyBasePred", "PolySlope", "PolyDevPct",
         "PolyDevTrend", "RollingVolPct"]
        + [f"MA{mw}" for mw in ma_windows]
    )
    return df.dropna(subset=feature_cols)


def compute_polyfit_base_only(
    close: pd.Series,
    fit_window_days: int = 252,
    ma_windows: list[int] | None = None,
) -> pd.DataFrame:
    """只计算 Polyfit 基线 + 偏离度 + MA（不依赖 trend/vol 窗口的部分）。

    Polyfit 基线是扫描中最耗时的部分（CPU Python 循环，O(n×window)），
    但它只依赖 fit_window_days（固定 252），与 trend_window_days 和
    vol_window_days 无关。将此部分提取出来单独计算，在扫描中复用，避免重复计算。

    Returns:
        DataFrame with columns: PolyBasePred, PolyDevPct, MA{5,10,20,60}
    """
    if ma_windows is None:
        ma_windows = [5, 10, 20, 60]
    close_arr = close.values.astype(np.float64)
    pred, _slope = compute_polyfit_baseline(close_arr, fit_window_days)

    df = pd.DataFrame(index=close.index)
    df["PolyBasePred"] = pred
    df["PolyDevPct"] = close_arr / pred - 1.0
    for mw in ma_windows:
        df[f"MA{mw}"] = close.rolling(mw, min_periods=mw).mean()

    feature_cols = ["PolyBasePred", "PolyDevPct"] + [f"MA{mw}" for mw in ma_windows]
    return df.dropna(subset=feature_cols)


def add_trend_vol_indicators(
    base_df: pd.DataFrame,
    close: pd.Series,
    trend_window_days: int = 20,
    vol_window_days: int = 20,
) -> pd.DataFrame:
    """在已有基线指标上追加 PolyDevTrend 和 RollingVolPct。

    这两个指标依赖 trend_window_days 和 vol_window_days，
    在参数扫描中这两个参数会变化，而基线不变。
    """
    dev_pct = base_df["PolyDevPct"]
    df = base_df.copy()
    df["PolyDevTrend"] = compute_polyfit_deviation_trend(dev_pct, trend_window_days)
    df["RollingVolPct"] = compute_polyfit_volatility(close, vol_window_days)
    feature_cols = ["PolyDevTrend", "RollingVolPct"]
    return df.dropna(subset=feature_cols)
