"""
VectorBT parameter scanning for dynamic grid strategies — with CUDA/GPU support.

Reimplements PolyfitDynamicGridStrategy and MovingAverageDynamicGridStrategy
logic using VectorBT for portfolio backtesting and parameter optimization.
"""

from itertools import product
import os
import subprocess
from typing import Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# GPU / CUDA detection
# ══════════════════════════════════════════════════════════════════

_gpu_info: dict = {}


def detect_gpu() -> dict:
    """Detect NVIDIA GPU and CUDA capabilities (idempotent)."""
    global _gpu_info
    if _gpu_info:
        return _gpu_info

    info = {
        "gpu_detected": False,
        "gpu_name": "N/A",
        "memory_mb": 0,
        "compute_cap": "N/A",
        "cupy_available": False,
        "cupy_version": "N/A",
        "xp": np,  # default array module
    }

    # --- nvidia-smi detection ---
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            info["gpu_name"] = parts[0] if len(parts) > 0 else "N/A"
            mem_str = parts[1] if len(parts) > 1 else "0 MiB"
            info["memory_mb"] = int(mem_str.replace("MiB", "").strip()) if "MiB" in mem_str else 0
            info["compute_cap"] = parts[2] if len(parts) > 2 else "N/A"
            info["gpu_detected"] = True
    except Exception:
        pass

    # --- CuPy detection ---
    try:
        import cupy as cp
        # Test that JIT compilation actually works (not just import)
        _test = cp.arange(10, dtype=cp.float64)
        _ = cp.cumsum(_test)
        info["cupy_available"] = True
        info["cupy_version"] = cp.__version__
        info["xp"] = cp
    except Exception:
        pass

    _gpu_info = info
    return info


def gpu() -> dict:
    """Return cached GPU info (call detect_gpu first)."""
    return _gpu_info or detect_gpu()


def xp():
    """Return cupy if CUDA is available, else numpy."""
    return gpu()["xp"]


def print_gpu_info(info: dict) -> None:
    print(f"GPU: {info['gpu_name']}")
    print(f"  Memory:      {info['memory_mb']} MiB")
    print(f"  Compute Cap: {info['compute_cap']}")
    if info["cupy_available"]:
        print(f"  CuPy:        {info['cupy_version']}  ✓ (GPU acceleration enabled)")
    else:
        print(f"  CuPy:        not installed (CPU-only mode)")


# ══════════════════════════════════════════════════════════════════
# Data & output paths
# ══════════════════════════════════════════════════════════════════

DATA_PATH = "data/512890.SH_hfq.parquet"
REPORTS_DIR = "reports"


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
    """Rolling volatility (std of returns).  GPU-accelerated via CuPy when available."""
    returns = close.pct_change()
    if gpu()["cupy_available"]:
        cp = xp()
        ret = cp.asarray(returns.values, dtype=cp.float64)
        # GPU rolling std via convolution: std = sqrt(E[X²] - E[X]²)
        ones = cp.ones(window, dtype=cp.float64)
        sum_ret = cp.convolve(ret, ones, mode="same")
        sum_ret2 = cp.convolve(ret * ret, ones, mode="same")
        mask = cp.arange(len(ret)) >= window - 1
        mean_ret = cp.where(mask, sum_ret / window, cp.nan)
        mean_ret2 = cp.where(mask, sum_ret2 / window, cp.nan)
        var = cp.maximum(mean_ret2 - mean_ret * mean_ret, 0.0)
        result = pd.Series(cp.asnumpy(cp.sqrt(var)), index=returns.index)
        result.iloc[:window - 1] = np.nan
        return result
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
    Generate entry/exit signals bar by bar, replicating the original strategy's
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
# Report generation
# ══════════════════════════════════════════════════════════════════

def generate_portfolio_reports(
    close: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    name: str,
    params: dict,
) -> dict:
    """Generate VectorBT plots and stats for a single parameter set."""
    pf = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(entries, index=close.index),
        exits=pd.Series(exits, index=close.index),
        size=pd.Series(sizes, index=close.index),
        size_type="percent",
        init_cash=100_000.0,
        freq="D",
    )

    safe_name = name.replace(" ", "_")

    # --- Stats CSV ---
    stats = pf.stats()
    stats_path = f"{REPORTS_DIR}/{safe_name}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  Stats → {stats_path}")

    # --- Main overview plot (interactive HTML) ---
    fig = pf.plot()
    fig_path = f"{REPORTS_DIR}/{safe_name}_overview.html"
    fig.write_html(fig_path)
    print(f"  Overview plot → {fig_path}")

    # --- Individual plots ---
    plot_methods = [
        ("cum_returns", pf.plot_cum_returns),
        ("drawdowns", pf.plot_drawdowns),
        ("underwater", pf.plot_underwater),
        ("trades", pf.plot_trades),
        ("trade_pnl", pf.plot_trade_pnl),
        ("asset_value", pf.plot_asset_value),
    ]
    saved_plots = []
    for plot_name, plot_fn in plot_methods:
        try:
            f = plot_fn()
            p = f"{REPORTS_DIR}/{safe_name}_{plot_name}.html"
            f.write_html(p)
            saved_plots.append((plot_name, f"{safe_name}_{plot_name}.html"))
        except Exception:
            pass

    return {
        "name": name,
        "params": params,
        "stats": stats.to_dict(),
        "overview_html": f"{safe_name}_overview.html",
        "plots": saved_plots,
    }


def build_index_html(reports: list[dict]) -> str:
    """Build a simple index.html linking to all reports."""
    rows = []
    for r in reports:
        s = r["stats"]
        param_str = "  |  ".join(f"{k}={v}" for k, v in r["params"].items())
        rows.append(f"""
        <tr>
            <td><strong>{r['name']}</strong></td>
            <td>{param_str}</td>
            <td>{s.get('Total Return [%]', 0):.2f}%</td>
            <td>{s.get('Sharpe Ratio', 0):.3f}</td>
            <td>{s.get('Max Drawdown [%]', 0):.2f}%</td>
            <td>{s.get('Total Trades', 0)}</td>
            <td><a href="{r['overview_html']}">Overview</a></td>
            <td>{" | ".join(f'<a href="{p[1]}">{p[0]}</a>' for p in r['plots'])}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>VectorBT Strategy Reports — 512890.SH</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; }}
  a {{ color: #1a73e8; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  h1 {{ margin-bottom: 0.5em; }}
  .subtitle {{ color: #666; margin-bottom: 1.5em; }}
</style>
</head>
<body>
<h1>VectorBT Dynamic Grid Strategy Reports</h1>
<p class="subtitle">512890.SH 后复权  |  Best-parameter backtests from grid search</p>
<table>
<thead>
<tr><th>Strategy</th><th>Parameters</th><th>Total Return</th><th>Sharpe</th>
<th>Max DD</th><th>Trades</th><th>Overview</th><th>Detail Plots</th></tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════
# Scan-result display
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
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --- GPU detection ---
    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)
    print()

    # --- Load data ---
    print("Loading data…")
    df = load_data()
    close = df["Close"]
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")

    # --- Remove old root-level CSVs from previous runs ---
    for old_csv in ["polyfit_scan_results.csv", "ma_scan_results.csv"]:
        if os.path.exists(old_csv):
            os.remove(old_csv)
            print(f"  Removed stale {old_csv} from project root")

    # --- Polyfit scan ---
    pf_results = scan_polyfit_strategy(close)
    pf_results.to_csv(f"{REPORTS_DIR}/polyfit_scan_results.csv", index=False)
    print_top(pf_results, "PolyfitDynamicGridStrategy",
              ["polyfit_window", "base_grid_pct", "volatility_scale",
               "trend_sensitivity", "take_profit_grid", "stop_loss_grid"])

    # --- MA scan ---
    ma_results = scan_ma_strategy(close)
    ma_results.to_csv(f"{REPORTS_DIR}/ma_scan_results.csv", index=False)
    print_top(ma_results, "MovingAverageDynamicGridStrategy",
              ["ma_window", "base_grid_pct", "volatility_scale",
               "trend_sensitivity", "take_profit_grid", "stop_loss_grid"])

    # --- Generate VectorBT reports for best parameters ---
    print(f"\n{'='*60}")
    print("Generating VectorBT reports for best parameters…")
    print(f"{'='*60}")

    reports_data = []

    # Best Polyfit
    best_pf = pf_results.nlargest(1, "total_return").iloc[0]
    pf_indicators = compute_polyfit_indicators(close, polyfit_window=int(best_pf["polyfit_window"]))
    pf_e, pf_x, pf_s = generate_grid_signals(
        close.values, pf_indicators["PolyDevPct"].values,
        pf_indicators["PolyDevTrend"].values, pf_indicators["RollingVolPct"].values,
        base_grid_pct=best_pf["base_grid_pct"], volatility_scale=best_pf["volatility_scale"],
        trend_sensitivity=best_pf["trend_sensitivity"], take_profit_grid=best_pf["take_profit_grid"],
        stop_loss_grid=best_pf["stop_loss_grid"],
    )
    reports_data.append(generate_portfolio_reports(close, pf_e, pf_x, pf_s, "Polyfit", {
        "polyfit_window": int(best_pf["polyfit_window"]),
        "base_grid_pct": best_pf["base_grid_pct"],
        "volatility_scale": best_pf["volatility_scale"],
        "trend_sensitivity": best_pf["trend_sensitivity"],
        "take_profit_grid": best_pf["take_profit_grid"],
        "stop_loss_grid": best_pf["stop_loss_grid"],
    }))

    # Best MA
    best_ma = ma_results.nlargest(1, "total_return").iloc[0]
    ma_indicators = compute_ma_indicators(close, ma_window=int(best_ma["ma_window"]))
    ma_e, ma_x, ma_s = generate_grid_signals(
        close.values, ma_indicators["MADevPct"].values,
        ma_indicators["MADevTrend"].values, ma_indicators["RollingVolPct"].values,
        base_grid_pct=best_ma["base_grid_pct"], volatility_scale=best_ma["volatility_scale"],
        trend_sensitivity=best_ma["trend_sensitivity"], take_profit_grid=best_ma["take_profit_grid"],
        stop_loss_grid=best_ma["stop_loss_grid"],
    )
    reports_data.append(generate_portfolio_reports(close, ma_e, ma_x, ma_s, "MA", {
        "ma_window": int(best_ma["ma_window"]),
        "base_grid_pct": best_ma["base_grid_pct"],
        "volatility_scale": best_ma["volatility_scale"],
        "trend_sensitivity": best_ma["trend_sensitivity"],
        "take_profit_grid": best_ma["take_profit_grid"],
        "stop_loss_grid": best_ma["stop_loss_grid"],
    }))

    # Build index.html
    index_html = build_index_html(reports_data)
    index_path = f"{REPORTS_DIR}/index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"  Index → {index_path}")

    print(f"\nDone. Reports saved to {REPORTS_DIR}/")
