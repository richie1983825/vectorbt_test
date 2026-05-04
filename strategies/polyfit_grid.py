"""Polyfit Grid Strategy — 纯均值回复网格策略。

与 Polyfit-Switch 的区别：移除了 Switch（趋势追踪）模式，
仅保留 Grid（均值回复网格）模式，避免 Switch 的负贡献。

核心逻辑：
  - 基线：滑动窗口线性回归（252 天）预测价格中枢
  - 入场：价格偏离基线下方超过动态网格步长时，分批买入
  - 离场：止盈 / 止损 / 最大持仓天数
  - 执行：次日开盘价成交

GPU 加速：CuPy RawKernel 批量信号生成 + 批量回测。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import xp, gpu
from utils.indicators import compute_polyfit_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成（纯 Grid）
# ══════════════════════════════════════════════════════════════════

def generate_grid_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 45,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,
    position_sizing_coef: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成纯 Grid 策略的入场/离场信号。

    相比 polyfit_switch：移除了 Switch 模式的全部逻辑
    （flat_wait_days / switch_deviation / trailing_stop / MA cross）。

    Returns:
        entries, exits, sizes
    """
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)

    in_position = False
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    entry_close_price = np.nan
    cooldown = 0

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        pb = poly_base[i]

        if np.isnan(dp) or np.isnan(dt) or np.isnan(vp) or np.isnan(pb) or pb <= 0 or close[i] <= 0:
            continue

        if not in_position:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan

        if cooldown > 0:
            cooldown -= 1

        # 动态网格步长
        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        if not in_position:
            if cooldown > 0:
                continue
            signal_strength = abs(dp) / max(dynamic_grid_step, 1e-9)
            entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
            entry_threshold = -entry_lvl * dynamic_grid_step
            if dp <= entry_threshold and signal_strength >= min_signal_strength:
                size = float(np.clip(
                    abs(dp) * (1.0 + max(vp, 0.0)) * position_sizing_coef,
                    0.0, position_size,
                ))
                if size > 0:
                    entries[i] = True
                    sizes[i] = size
                    in_position = True
                    entry_bar = i
                    entry_level = entry_lvl
                    entry_grid_step = dynamic_grid_step
                    entry_close_price = close[i]
        else:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            # TP/SL 均以基线偏离度 dp 为锚
            if hold_limit or dev_pct[i] >= tp_threshold or dev_pct[i] <= -sl_threshold:
                exits[i] = True
                in_position = False
                cooldown = cooldown_days

    if in_position:
        exits[-1] = True

    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# GPU 批量信号生成 — CuPy RawKernel（纯 Grid）
# ══════════════════════════════════════════════════════════════════

_grid_kernel = None

_GRID_KERNEL_CODE = r"""
extern "C" __global__ void grid_signals_kernel(
    const double* close,
    const double* dev_pct,
    const double* dev_trend_all,
    const double* vol_pct_all,
    const int* indicator_idx,
    const double* poly_base,
    const double* base_grid_pct,
    const double* volatility_scale,
    const double* trend_sensitivity,
    const double* take_profit_grid,
    const double* stop_loss_grid,
    const int* max_grid_levels,
    const int* cooldown_days,
    const double* min_signal_strength,
    const double* position_size,
    const double* position_sizing_coef,
    bool* entries,
    bool* exits,
    double* sizes,
    int n_bars,
    int n_combos,
    int max_holding_days
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    int iset = indicator_idx[combo_idx];

    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];
    int mgl = max_grid_levels[combo_idx];
    int cd = cooldown_days[combo_idx];
    double mss = min_signal_strength[combo_idx];
    double ps = position_size[combo_idx];
    double psc = position_sizing_coef[combo_idx];

    bool in_position = false;
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    double entry_close_price = 0.0 / 0.0;  // NaN
    int cooldown = 0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        double dp = dev_pct[i];
        double dt = dev_trend_all[iset * n_bars + i];
        double vp = vol_pct_all[iset * n_bars + i];
        double pb = poly_base[i];

        if (isnan(cl) || isnan(dp) || isnan(dt) || isnan(vp)
            || isnan(pb) || pb <= 0.0 || cl <= 0.0) {
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        if (!in_position) {
            entry_bar = -1;
            entry_level = 1;
            entry_grid_step = -1.0;
        }
        if (cooldown > 0) cooldown--;

        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        entries[combo_idx * n_bars + i] = false;
        exits[combo_idx * n_bars + i] = false;
        sizes[combo_idx * n_bars + i] = 0.0;

        if (!in_position) {
            if (cooldown > 0) continue;
            double sig = fabs(dp) / fmax(dgs, 1e-9);
            int el = (int)floor(sig);
            el = el < 1 ? 1 : (el > mgl ? mgl : el);
            double eth = -(double)el * dgs;
            if (dp <= eth && sig >= mss) {
                double sz = fabs(dp) * (1.0 + fmax(vp, 0.0)) * psc;
                sz = sz > ps ? ps : (sz < 0.0 ? 0.0 : sz);
                if (sz > 0.0) {
                    entries[combo_idx * n_bars + i] = true;
                    sizes[combo_idx * n_bars + i] = sz;
                    in_position = true;
                    entry_bar = i;
                    entry_level = el;
                    entry_grid_step = dgs;
                    entry_close_price = cl;
                }
            }
        } else {
            int hd = i - entry_bar;
            bool hl = hd >= max_holding_days;
            double rs = fmax(dgs, entry_grid_step);
            if (entry_grid_step < 0.0) rs = dgs;
            double tp_threshold = entry_level * rs * tpg;
            double sl_threshold = entry_level * rs * slg;
            // TP/SL 均以基线偏离度 dp 为锚
            if (hl || dp >= tp_threshold || dp <= -sl_threshold) {
                exits[combo_idx * n_bars + i] = true;
                in_position = false;
                cooldown = cd;
            }
        }
    }
    if (in_position) {
        exits[combo_idx * n_bars + n_bars - 1] = true;
    }
}
"""


def _get_grid_kernel():
    global _grid_kernel
    if _grid_kernel is None:
        cp = xp()
        _grid_kernel = cp.RawKernel(_GRID_KERNEL_CODE, "grid_signals_kernel")
    return _grid_kernel


def generate_grid_signals_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend_all: np.ndarray,       # [n_indicator_sets, n_bars]
    rolling_vol_pct_all: np.ndarray, # [n_indicator_sets, n_bars]
    poly_base: np.ndarray,
    base_grid_pcts: np.ndarray,
    volatility_scales: np.ndarray,
    trend_sensitivities: np.ndarray,
    take_profit_grids: np.ndarray,
    stop_loss_grids: np.ndarray,
    indicator_idx: np.ndarray | None = None,
    max_grid_levels_arr: np.ndarray | None = None,
    max_holding_days: int = 45,
    cooldown_days_arr: np.ndarray | None = None,
    min_signal_strength_arr: np.ndarray | None = None,
    position_size_arr: np.ndarray | None = None,
    position_sizing_coef_arr: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU 批量生成纯 Grid 策略信号。

    支持多 indicator set 批量处理，通过 indicator_idx 指定组合归属。
    """
    cp = xp()
    n_bars = len(close)
    n_combos = len(base_grid_pcts)

    if dev_trend_all.ndim == 1:
        dev_trend_all = dev_trend_all.reshape(1, -1)
    if rolling_vol_pct_all.ndim == 1:
        rolling_vol_pct_all = rolling_vol_pct_all.reshape(1, -1)

    if indicator_idx is None:
        indicator_idx = np.zeros(n_combos, dtype=np.int32)

    block_size = 256
    grid_size = (n_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    def _pad_f64(a):
        result = cp.zeros(padded, dtype=cp.float64)
        result[:n_combos] = cp.asarray(a, dtype=cp.float64)
        if padded > n_combos:
            result[n_combos:] = cp.nan
        return result

    def _pad_i32(a):
        result = cp.zeros(padded, dtype=cp.int32)
        result[:n_combos] = cp.asarray(a, dtype=cp.int32)
        return result

    _mgl = max_grid_levels_arr if max_grid_levels_arr is not None else np.full(n_combos, 3, dtype=np.int32)
    _cd  = cooldown_days_arr if cooldown_days_arr is not None else np.full(n_combos, 1, dtype=np.int32)
    _mss = min_signal_strength_arr if min_signal_strength_arr is not None else np.full(n_combos, 0.45, dtype=np.float64)
    _ps  = position_size_arr if position_size_arr is not None else np.full(n_combos, 0.5, dtype=np.float64)
    _psc = position_sizing_coef_arr if position_sizing_coef_arr is not None else np.full(n_combos, 30.0, dtype=np.float64)

    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    trend_d = cp.asarray(dev_trend_all.ravel(), dtype=cp.float64)
    vol_d = cp.asarray(rolling_vol_pct_all.ravel(), dtype=cp.float64)
    iset_d = _pad_i32(indicator_idx)
    pb_d = cp.asarray(poly_base, dtype=cp.float64)

    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    kernel = _get_grid_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, dev_d, trend_d, vol_d, iset_d, pb_d,
            _pad_f64(base_grid_pcts), _pad_f64(volatility_scales),
            _pad_f64(trend_sensitivities), _pad_f64(take_profit_grids),
            _pad_f64(stop_loss_grids), _pad_i32(_mgl), _pad_i32(_cd),
            _pad_f64(_mss), _pad_f64(_ps), _pad_f64(_psc),
            entries_d, exits_d, sizes_d,
            n_bars, n_combos, max_holding_days,
        ),
    )

    entries = cp.asnumpy(entries_d).reshape(padded, n_bars)[:n_combos]
    exits = cp.asnumpy(exits_d).reshape(padded, n_bars)[:n_combos]
    sizes = cp.asnumpy(sizes_d).reshape(padded, n_bars)[:n_combos]
    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# 参数扫描
# ══════════════════════════════════════════════════════════════════

_grid_scan_cache: dict[tuple, pd.DataFrame] = {}


def _grid_cache_key(close: pd.Series, open_: pd.Series | None = None) -> tuple:
    return (close.index[0], close.index[-1], len(close), open_ is not None)


def clear_grid_cache() -> None:
    """清除 Grid 扫描缓存（用于跨品种对比等场景）。"""
    _grid_scan_cache.clear()


def scan_polyfit_grid(
    close: pd.Series, open_: pd.Series | None = None,
) -> pd.DataFrame:
    """Polyfit-Grid 策略参数扫描（GPU 批量）。

    扫描参数：
      - indicator: trend_window_days × vol_window_days
      - grid: base_grid_pct × vol_scale × trend_sens × max_grid_levels
              × tpg × slg × position_size × position_sizing_coef
              × min_signal_strength
    """
    cache_key = _grid_cache_key(close, open_)
    if cache_key in _grid_scan_cache:
        print("  [PolyfitGrid] Cache hit — reusing scan results")
        return _grid_scan_cache[cache_key].copy()

    use_gpu = gpu()["cupy_available"]
    open_arr = open_.values if open_ is not None else None
    all_ma_windows: list[int] = []  # Grid-only 不需要 MA

    trend_windows = [10, 15, 20]
    vol_windows = [10, 20, 30]
    grid_pcts = [0.008, 0.010, 0.012, 0.015]
    vol_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    trend_sens = [4.0, 6.0, 8.0, 10.0]
    max_grid_levels_vals = [2, 3, 4]
    tpg_values = [0.6, 0.8, 1.0]
    slg_values = [1.2, 1.6, 2.0]
    position_sizes = [0.92, 0.94, 0.97, 0.99]
    position_sizing_coefs = [20.0, 30.0, 60.0]
    min_signal_strengths = [0.30, 0.45, 0.60]

    print("  [PolyfitGrid] GPU-scanning grid params…")

    indicator_combos = list(product(trend_windows, vol_windows))
    grid_combos = list(product(
        grid_pcts, vol_scales, trend_sens, max_grid_levels_vals,
        tpg_values, slg_values, position_sizes, position_sizing_coefs,
        min_signal_strengths,
    ))
    n_ind = len(indicator_combos)
    n_grid = len(grid_combos)
    n_total = n_ind * n_grid
    print(f"    indicator combos={n_ind}  grid combos={n_grid}  total={n_total}")

    # 预计算基线（复用）
    from utils.indicators import compute_polyfit_base_only, add_trend_vol_indicators
    base_indicators = compute_polyfit_base_only(close, fit_window_days=252, ma_windows=[])

    indicator_cache = {}
    for tw, vw in indicator_combos:
        key = (tw, vw)
        if key not in indicator_cache:
            indicator_cache[key] = add_trend_vol_indicators(
                base_indicators, close, trend_window_days=tw, vol_window_days=vw,
            )

    common_idx = base_indicators.index
    if len(common_idx) == 0:
        return pd.DataFrame()
    cl_aligned = close.loc[common_idx]
    cl_arr = cl_aligned.values
    op_aligned = open_arr[close.index.get_indexer(common_idx)] if open_arr is not None else None

    poly_base_arr = base_indicators["PolyBasePred"].values
    dev_pct_arr = base_indicators["PolyDevPct"].values

    # 收集所有 indicator set 的 trend/vol
    all_dev_trend = []
    all_vol_pct = []
    ind_keys = []
    for tw, vw in indicator_combos:
        indicators = indicator_cache[(tw, vw)]
        all_dev_trend.append(indicators["PolyDevTrend"].reindex(common_idx).values)
        all_vol_pct.append(indicators["RollingVolPct"].reindex(common_idx).values)
        ind_keys.append((tw, vw))
    dev_trend_all = np.array(all_dev_trend)
    vol_pct_all = np.array(all_vol_pct)

    # 预构建 per-combo 数组
    s1_bgp = np.array([p[0] for p in grid_combos])
    s1_vs = np.array([p[1] for p in grid_combos])
    s1_ts = np.array([p[2] for p in grid_combos])
    s1_mgl = np.array([p[3] for p in grid_combos], dtype=np.int32)
    s1_tpg = np.array([p[4] for p in grid_combos])
    s1_slg = np.array([p[5] for p in grid_combos])
    s1_psz = np.array([p[6] for p in grid_combos])
    s1_psc = np.array([p[7] for p in grid_combos])
    s1_mss = np.array([p[8] for p in grid_combos])

    results = []

    if use_gpu:
        # Tile 所有 grid 参数 n_ind 次
        def _tile(arr):
            return np.tile(arr, n_ind).astype(arr.dtype)
        indicator_idx = np.repeat(np.arange(n_ind, dtype=np.int32), n_grid)

        entries_b, exits_b, sizes_b = generate_grid_signals_batch(
            cl_arr, dev_pct_arr, dev_trend_all, vol_pct_all, poly_base_arr,
            _tile(s1_bgp), _tile(s1_vs), _tile(s1_ts),
            _tile(s1_tpg), _tile(s1_slg),
            indicator_idx=indicator_idx,
            max_grid_levels_arr=_tile(s1_mgl),
            cooldown_days_arr=np.full(n_total, 1, dtype=np.int32),
            min_signal_strength_arr=_tile(s1_mss),
            position_size_arr=_tile(s1_psz),
            position_sizing_coef_arr=_tile(s1_psc),
        )
        bt = run_backtest_batch(cl_arr, entries_b, exits_b, sizes_b,
                                n_combos=n_total, open_=op_aligned)
        for ii, (tw, vw) in enumerate(ind_keys):
            for gi, (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss) in enumerate(grid_combos):
                idx = ii * n_grid + gi
                if int(bt[idx][4]) == 0:
                    continue
                results.append({
                    "total_return": bt[idx][0], "sharpe_ratio": bt[idx][1],
                    "max_drawdown": bt[idx][2], "calmar_ratio": bt[idx][3],
                    "num_trades": int(bt[idx][4]), "win_rate": bt[idx][5],
                    "trend_window_days": tw, "vol_window_days": vw,
                    "base_grid_pct": bgp, "volatility_scale": vs,
                    "trend_sensitivity": ts, "max_grid_levels": max_gl,
                    "take_profit_grid": tpg, "stop_loss_grid": slg,
                    "max_holding_days": 45, "cooldown_days": 1,
                    "min_signal_strength": min_ss,
                    "position_size": pos_sz, "position_sizing_coef": pos_coef,
                })
    else:
        for tw, vw in indicator_combos:
            indicators = indicator_cache[(tw, vw)]
            for (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss) in grid_combos:
                entries, exits, sizes = generate_grid_signals(
                    cl_arr, dev_pct_arr,
                    indicators["PolyDevTrend"].reindex(common_idx).values,
                    indicators["RollingVolPct"].reindex(common_idx).values,
                    poly_base_arr,
                    base_grid_pct=bgp, volatility_scale=vs,
                    trend_sensitivity=ts, max_grid_levels=max_gl,
                    take_profit_grid=tpg, stop_loss_grid=slg,
                    max_holding_days=45, cooldown_days=1,
                    min_signal_strength=min_ss,
                    position_size=pos_sz, position_sizing_coef=pos_coef,
                )
                if entries.sum() == 0:
                    continue
                m = run_backtest(cl_aligned, entries, exits, sizes, open_=op_aligned)
                m["trend_window_days"] = tw
                m["vol_window_days"] = vw
                m["base_grid_pct"] = bgp
                m["volatility_scale"] = vs
                m["trend_sensitivity"] = ts
                m["max_grid_levels"] = max_gl
                m["take_profit_grid"] = tpg
                m["stop_loss_grid"] = slg
                m["max_holding_days"] = 45
                m["cooldown_days"] = 1
                m["min_signal_strength"] = min_ss
                m["position_size"] = pos_sz
                m["position_sizing_coef"] = pos_coef
                results.append(m)

    result = pd.DataFrame(results)
    _grid_scan_cache[cache_key] = result
    return result
