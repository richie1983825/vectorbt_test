"""Polyfit Switch Strategy — 多项式拟合基线 + 均线交叉双模式策略。

与 MA-Switch 策略的核心区别：
  1. 基线：使用滑动窗口线性回归（Polyfit，通常 252 天）预测价格中枢，
     替代简单均线（SMA）。Polyfit 基线在趋势行情中滞后更小，
     且能更好地过滤震荡噪音。
  2. Switch 离场：使用最高价回撤止损（trailing stop），
     从持仓期间的最高收盘价回撤 switch_trailing_stop 比例后离场。
  3. 参数：最大持仓天数 45（vs 30），仓位系数更大（0.92-0.99 vs 0.5）。

双模式说明：
  - Grid 模式（默认）：均值回复网格交易，在 Polyfit 基线下方挂网格多单。
  - Switch 模式（趋势追踪）：当价格连续横盘后在基线上方且偏离超过阈值时激活，
    使用快慢均线金叉入场 + 最高价回撤追踪止损离场。

GPU 加速：通过 CuPy RawKernel 将多组参数组合的信号生成一次性在 GPU 上完成。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import xp, gpu
from utils.indicators import compute_polyfit_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成
# ══════════════════════════════════════════════════════════════════

def generate_polyfit_switch_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,       # (close - poly_base) / poly_base
    dev_trend: np.ndarray,     # EMA(diff(dev_pct))，偏离趋势
    rolling_vol_pct: np.ndarray, # 滚动波动率
    poly_base: np.ndarray,     # Polyfit 基线预测值
    ma_fast: np.ndarray,       # 快均线（Switch 金叉用）
    ma_slow: np.ndarray,       # 慢均线（Switch 死叉用）
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 45,    # 更长持有期（vs MA Grid 的 30）
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,
    position_sizing_coef: float = 30.0,
    flat_wait_days: int = 8,
    switch_deviation_m1: float = 0.03,
    switch_deviation_m2: float = 0.02,
    switch_trailing_stop: float = 0.05,  # Switch 追踪止损：从最高点回撤比例
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 Polyfit-Switch 策略的入场/离场信号。

    Switch 模式：金叉入场 + 最高价回撤追踪止损离场。
      持仓期间持续记录最高收盘价，当 close ≤ peak × (1 - trailing_stop) 时离场。
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
    switch_mode_active = False
    flat_days = 0
    peak_close = 0.0  # Switch 持仓期间的最高收盘价

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        pb = poly_base[i]
        cl = close[i]
        fma = ma_fast[i]
        sma = ma_slow[i]

        # 跳过无效数据
        nan_vars = (np.isnan(dp) or np.isnan(dt) or np.isnan(vp)
                    or np.isnan(pb) or np.isnan(fma) or np.isnan(sma))
        if nan_vars or pb <= 0 or cl <= 0:
            continue

        # 跟踪无持仓天数
        if in_position:
            flat_days = 0
        else:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan
            flat_days += 1

        if cooldown > 0:
            cooldown -= 1

        # 动态网格步长（与 MA 策略逻辑相同）
        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        # ── Switch 模式激活条件 ──
        if (not switch_mode_active and not in_position
                and flat_days >= flat_wait_days
                and cl > pb and dp > switch_deviation_m1):
            switch_mode_active = True

        # ── 更新 Switch 持仓期间的最高价 ──
        if in_position and switch_mode_active:
            peak_close = max(peak_close, cl)

        # ── Switch 模式逻辑 ──
        if switch_mode_active:
            if dp < switch_deviation_m2:
                switch_mode_active = False
                continue
            if cl <= pb or dp <= switch_deviation_m1:
                continue
            # 金叉入场：快线上穿慢线 + 无持仓
            if fma > sma and not in_position:
                full_size = float(np.nextafter(1.0, 0.0))
                entries[i] = True
                sizes[i] = full_size
                in_position = True
                entry_bar = i
                entry_level = 1
                entry_grid_step = max(base_grid_pct, 1e-9)
                flat_days = 0
                peak_close = cl
                continue
            # 追踪止损离场：从最高收盘价回撤超过 trailing_stop
            if in_position:
                if cl <= peak_close * (1.0 - switch_trailing_stop):
                    exits[i] = True
                    in_position = False
                    cooldown = cooldown_days
                    peak_close = 0.0
            continue

        # ── Grid 模式逻辑 ──
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
        else:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            if hold_limit or dp >= tp_threshold or dp <= -sl_threshold:
                exits[i] = True
                in_position = False
                cooldown = cooldown_days

    if in_position:
        exits[-1] = True

    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# GPU 批量信号生成 — CuPy RawKernel
#
# 与 MA-Switch kernel 的关键区别：
#   1. 基线从 MABase 改为 PolyBasePred
#   2. Switch 离场从 trailing_stop 改为死叉（fast_ma < slow_ma）
#   3. 移除了 switch_trailing_stop 参数
#   4. max_holding_days 默认 45
# ══════════════════════════════════════════════════════════════════

_polyfit_switch_kernel = None

_POLYFIT_SWITCH_KERNEL_CODE = r"""
extern "C" __global__ void polyfit_switch_signals_kernel(
    const double* close,
    const double* dev_pct,
    const double* dev_trend,
    const double* vol_pct,
    const double* poly_base,
    const double* ma_all,           // 展平的多根 MA: [n_ma_windows * n_bars]
    const int* fast_ma_idx,         // 每组合的快均线在 ma_all 中的索引
    const int* slow_ma_idx,         // 每组合的慢均线在 ma_all 中的索引
    const double* base_grid_pct,
    const double* volatility_scale,
    const double* trend_sensitivity,
    const double* take_profit_grid,
    const double* stop_loss_grid,
    const int* flat_wait_days,
    const double* switch_deviation_m1,
    const double* switch_deviation_m2,
    const double* switch_trailing_stop,  // [n_combos] — 追踪止损回撤比例
    const int* max_grid_levels,       // [n_combos] — 每个组合的最大网格层级
    const int* cooldown_days,         // [n_combos] — 每个组合的冷却天数
    const double* min_signal_strength,// [n_combos] — 每个组合的最小信号强度
    const double* position_size,      // [n_combos] — 每个组合的最大仓位比例
    const double* position_sizing_coef,//[n_combos] — 每个组合的仓位系数
    bool* entries,
    bool* exits,
    double* sizes,
    int n_bars,
    int n_combos,
    int max_holding_days
) {
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    // 读取当前组合的所有参数
    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];
    int fw = flat_wait_days[combo_idx];
    double sw_m1 = switch_deviation_m1[combo_idx];
    double sw_m2 = switch_deviation_m2[combo_idx];
    double sw_ts = switch_trailing_stop[combo_idx];
    int fi = fast_ma_idx[combo_idx];
    int si = slow_ma_idx[combo_idx];
    int mgl = max_grid_levels[combo_idx];
    int cd = cooldown_days[combo_idx];
    double mss = min_signal_strength[combo_idx];
    double ps = position_size[combo_idx];
    double psc = position_sizing_coef[combo_idx];

    // 状态机变量
    bool in_position = false;
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    int cooldown = 0;
    bool switch_mode_active = false;
    int flat_days = 0;
    double peak_close = 0.0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        double dp = dev_pct[i];
        double dt = dev_trend[i];
        double vp = vol_pct[i];
        double pb = poly_base[i];
        // 从展平的 ma_all 中索引对应 MA 值
        double fma = ma_all[fi * n_bars + i];
        double sma = ma_all[si * n_bars + i];

        // 跳过无效数据
        if (isnan(cl) || isnan(dp) || isnan(dt) || isnan(vp)
            || isnan(pb) || isnan(fma) || isnan(sma)
            || pb <= 0.0 || cl <= 0.0) {
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        // 更新持仓/空仓天数
        if (in_position) {
            flat_days = 0;
        } else {
            entry_bar = -1;
            entry_level = 1;
            entry_grid_step = -1.0;
            flat_days++;
        }
        if (cooldown > 0) cooldown--;

        // 动态网格步长
        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        // ── 更新 Switch 持仓期间的最高价 ──
        if (in_position && switch_mode_active) {
            peak_close = fmax(peak_close, cl);
        }

        // ── Switch 模式激活 ──
        if (!switch_mode_active && !in_position
                && flat_days >= fw && cl > pb && dp > sw_m1) {
            switch_mode_active = true;
        }

        // ── Switch 模式逻辑 ──
        if (switch_mode_active) {
            // 偏离回落到 m2 以下 → 关闭 Switch
            if (dp < sw_m2) {
                switch_mode_active = false;
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            // 偏离条件不满足 → 挂起
            if (cl <= pb || dp <= sw_m1) {
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            // 金叉入场（fast > slow && 无持仓）
            if (fma > sma && !in_position) {
                entries[combo_idx * n_bars + i] = true;
                sizes[combo_idx * n_bars + i] = 0.9999999999999999;
                in_position = true;
                entry_bar = i;
                entry_level = 1;
                entry_grid_step = fmax(bgp, 1e-9);
                flat_days = 0;
                peak_close = cl;
                continue;
            }
            // 追踪止损离场：从最高收盘价回撤超过 trailing_stop
            if (in_position) {
                if (cl <= peak_close * (1.0 - sw_ts)) {
                    exits[combo_idx * n_bars + i] = true;
                    in_position = false;
                    cooldown = cd;
                    peak_close = 0.0;
                } else {
                    exits[combo_idx * n_bars + i] = false;
                }
            }
            entries[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        // ── Grid 模式逻辑 ──
        entries[combo_idx * n_bars + i] = false;
        exits[combo_idx * n_bars + i] = false;
        sizes[combo_idx * n_bars + i] = 0.0;

        if (!in_position) {
            if (cooldown > 0) continue;
            double signal_strength = fabs(dp) / fmax(dgs, 1e-9);
            int entry_lvl = (int)floor(signal_strength);
            entry_lvl = entry_lvl < 1 ? 1 : (entry_lvl > mgl ? mgl : entry_lvl);
            double entry_threshold = -(double)entry_lvl * dgs;
            if (dp <= entry_threshold && signal_strength >= mss) {
                double size = fabs(dp) * (1.0 + fmax(vp, 0.0)) * psc;
                size = size > ps ? ps : (size < 0.0 ? 0.0 : size);
                if (size > 0.0) {
                    entries[combo_idx * n_bars + i] = true;
                    sizes[combo_idx * n_bars + i] = size;
                    in_position = true;
                    entry_bar = i;
                    entry_level = entry_lvl;
                    entry_grid_step = dgs;
                }
            }
        } else {
            int holding_days = i - entry_bar;
            bool hold_limit = holding_days >= max_holding_days;
            double ref_step = fmax(dgs, entry_grid_step);
            if (entry_grid_step < 0.0) ref_step = dgs;
            double tp_threshold = entry_level * ref_step * tpg;
            double sl_threshold = entry_level * ref_step * slg;
            if (hold_limit || dp >= tp_threshold || dp <= -sl_threshold) {
                exits[combo_idx * n_bars + i] = true;
                in_position = false;
                cooldown = cd;
            }
        }
    }

    // 末尾强制平仓
    if (in_position) {
        exits[combo_idx * n_bars + n_bars - 1] = true;
    }
}
"""


def _get_polyfit_switch_kernel():
    """延迟编译 CUDA kernel，避免在 CuPy 不可用时报错。"""
    global _polyfit_switch_kernel
    if _polyfit_switch_kernel is None:
        cp = xp()
        _polyfit_switch_kernel = cp.RawKernel(
            _POLYFIT_SWITCH_KERNEL_CODE, "polyfit_switch_signals_kernel"
        )
    return _polyfit_switch_kernel


def generate_polyfit_switch_signals_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,          # [n_bars]
    ma_all: np.ndarray,             # [n_ma_windows, n_bars]
    fast_ma_idx: np.ndarray,        # [n_combos] int32
    slow_ma_idx: np.ndarray,        # [n_combos] int32
    base_grid_pcts: np.ndarray,     # [n_combos]
    volatility_scales: np.ndarray,  # [n_combos]
    trend_sensitivities: np.ndarray,# [n_combos]
    take_profit_grids: np.ndarray,  # [n_combos]
    stop_loss_grids: np.ndarray,    # [n_combos]
    flat_wait_days_arr: np.ndarray, # [n_combos] int32
    switch_deviation_m1_arr: np.ndarray,  # [n_combos]
    switch_deviation_m2_arr: np.ndarray,  # [n_combos]
    switch_trailing_stop_arr: np.ndarray | None = None,  # [n_combos]
    max_grid_levels_arr: np.ndarray | None = None,   # [n_combos] int32
    max_holding_days: int = 45,
    cooldown_days_arr: np.ndarray | None = None,     # [n_combos] int32
    min_signal_strength_arr: np.ndarray | None = None,# [n_combos]
    position_size_arr: np.ndarray | None = None,      # [n_combos]
    position_sizing_coef_arr: np.ndarray | None = None,# [n_combos]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU 批量生成 Polyfit-Switch 策略信号。

    所有标记为 [n_combos] 的参数均为 per-combo 数组，支持不同组合使用不同值。
    若某数组为 None，则自动用标量默认值填充（兼容旧调用方式）。

    参数数组需 pad 到 256 的倍数以对齐 CUDA block。
    """
    cp = xp()
    n_bars = len(close)
    n_combos = len(base_grid_pcts)

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

    # 处理可选数组参数：None 时用标量默认值
    _mgl = max_grid_levels_arr if max_grid_levels_arr is not None else np.full(n_combos, 3, dtype=np.int32)
    _cd  = cooldown_days_arr if cooldown_days_arr is not None else np.full(n_combos, 1, dtype=np.int32)
    _mss = min_signal_strength_arr if min_signal_strength_arr is not None else np.full(n_combos, 0.45, dtype=np.float64)
    _str = switch_trailing_stop_arr if switch_trailing_stop_arr is not None else np.full(n_combos, 0.05, dtype=np.float64)
    _ps  = position_size_arr if position_size_arr is not None else np.full(n_combos, 0.5, dtype=np.float64)
    _psc = position_sizing_coef_arr if position_sizing_coef_arr is not None else np.full(n_combos, 30.0, dtype=np.float64)

    bgp_d = _pad_f64(base_grid_pcts)
    vs_d = _pad_f64(volatility_scales)
    ts_d = _pad_f64(trend_sensitivities)
    tpg_d = _pad_f64(take_profit_grids)
    slg_d = _pad_f64(stop_loss_grids)
    fw_d = _pad_i32(flat_wait_days_arr)
    m1_d = _pad_f64(switch_deviation_m1_arr)
    m2_d = _pad_f64(switch_deviation_m2_arr)
    str_d = _pad_f64(_str)
    fi_d = _pad_i32(fast_ma_idx)
    si_d = _pad_i32(slow_ma_idx)
    mgl_d = _pad_i32(_mgl)
    cd_d = _pad_i32(_cd)
    mss_d = _pad_f64(_mss)
    ps_d = _pad_f64(_ps)
    psc_d = _pad_f64(_psc)

    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    trend_d = cp.asarray(dev_trend, dtype=cp.float64)
    vol_d = cp.asarray(rolling_vol_pct, dtype=cp.float64)
    pb_d = cp.asarray(poly_base, dtype=cp.float64)
    ma_d = cp.asarray(ma_all.ravel(), dtype=cp.float64)

    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    kernel = _get_polyfit_switch_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, dev_d, trend_d, vol_d, pb_d, ma_d, fi_d, si_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d, fw_d, m1_d, m2_d,
            str_d, mgl_d, cd_d, mss_d, ps_d, psc_d,
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

# 扫描参数空间（与 global_best_switch_ma20_60_hold45 对齐）
def scan_polyfit_switch_strategy(close: pd.Series) -> pd.DataFrame:
    """Polyfit-Switch 策略参数网格扫描。

    对 polyfit 指标参数 + 网格参数 + Switch 参数进行全排列扫描。
    使用固定 fit_window_days=252（1 年回归窗口），
    fast_ma=20, slow_ma=60, max_holding_days=45（global_best 配置）。
    """
    print("  Polyfit-Switch parameter scan…")

    # 指标参数
    fit_window = 252
    trend_windows = [10, 15, 20]
    vol_windows = [10, 20, 30]

    # 网格参数
    grid_pcts = [0.008, 0.010, 0.012, 0.015]
    vol_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    trend_sens = [4.0, 6.0, 8.0, 10.0]
    max_grid_levels_vals = [2, 3, 4]
    tpg_values = [0.6, 0.8, 1.0]
    slg_values = [1.2, 1.6, 2.0]

    # Grid 持仓参数
    position_sizes = [0.92, 0.94, 0.97, 0.99]
    position_sizing_coefs = [20.0, 30.0, 60.0]
    min_signal_strengths = [0.30, 0.45, 0.60]

    # Switch 参数（global_best 固定值）
    flat_wait_days_vals = [8]
    switch_m1_vals = [0.03]
    switch_m2_vals = [0.02]
    switch_fast_vals = [20]
    switch_slow_vals = [60]

    all_ma_windows = sorted(set(switch_fast_vals + switch_slow_vals))
    ma_to_idx = {w: i for i, w in enumerate(all_ma_windows)}
    use_gpu = gpu()["cupy_available"]

    # 构建指标参数组合
    indicator_combos = list(product(trend_windows, vol_windows))
    # 构建网格参数组合
    grid_combos = list(product(
        grid_pcts, vol_scales, trend_sens, max_grid_levels_vals,
        tpg_values, slg_values, position_sizes, position_sizing_coefs,
        min_signal_strengths,
    ))
    switch_combos = list(product(
        flat_wait_days_vals, switch_m1_vals, switch_m2_vals,
        switch_fast_vals, switch_slow_vals,
    ))

    # 全排列：指标 × 网格 × switch
    all_combos = list(product(indicator_combos, grid_combos, switch_combos))
    total = len(all_combos)
    results = []
    count = 0

    # 缓存不同 (trend_window, vol_window) 的指标计算结果
    indicator_cache = {}
    for tw, vw in indicator_combos:
        if (tw, vw) not in indicator_cache:
            indicator_cache[(tw, vw)] = compute_polyfit_switch_indicators(
                close, fit_window_days=fit_window,
                ma_windows=all_ma_windows,
                trend_window_days=tw, vol_window_days=vw,
            )

    close_arr = close.values

    for (tw, vw), (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss), (fw, sw_m1, sw_m2, sw_fast, sw_slow) in all_combos:
        indicators = indicator_cache[(tw, vw)]
        # 确保 close 和 indicators 对齐（indicators 已 dropna）
        common_idx = indicators.index
        cl_aligned = close.loc[common_idx]
        poly_base_arr = indicators["PolyBasePred"].values
        dev_pct_arr = indicators["PolyDevPct"].values
        dev_trend_arr = indicators["PolyDevTrend"].values
        vol_arr = indicators["RollingVolPct"].values
        ma_fast_arr = indicators[f"MA{sw_fast}"].values
        ma_slow_arr = indicators[f"MA{sw_slow}"].values

        entries, exits, sizes = generate_polyfit_switch_signals(
            cl_aligned.values, dev_pct_arr, dev_trend_arr, vol_arr,
            poly_base_arr, ma_fast_arr, ma_slow_arr,
            base_grid_pct=bgp, volatility_scale=vs,
            trend_sensitivity=ts, max_grid_levels=max_gl,
            take_profit_grid=tpg, stop_loss_grid=slg,
            max_holding_days=45, cooldown_days=1,
            min_signal_strength=min_ss,
            position_size=pos_sz, position_sizing_coef=pos_coef,
            flat_wait_days=fw,
            switch_deviation_m1=sw_m1,
            switch_deviation_m2=sw_m2,
        )
        if entries.sum() == 0:
            continue

        # 用指标对齐后的 close 做回测
        m = run_backtest(cl_aligned, entries, exits, sizes, open_=open_)
        m["fit_window_days"] = fit_window
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
        m["flat_wait_days"] = fw
        m["switch_deviation_m1"] = sw_m1
        m["switch_deviation_m2"] = sw_m2
        m["switch_fast_ma"] = sw_fast
        m["switch_slow_ma"] = sw_slow
        results.append(m)
        count += 1
        if count % 50 == 0:
            print(f"  [PolyfitSwitch] {count}/{total}")

    return pd.DataFrame(results)


def scan_polyfit_switch_two_stage(close: pd.Series,
                                   open_: pd.Series | None = None) -> pd.DataFrame:
    """两阶段扫描：先 GPU 批量优化 Grid 参数，再 GPU 批量优化 Switch 参数。

    Stage 1: 固定 Switch 参数为 global_best 值，GPU 批量扫描 Grid 参数
    Stage 2: 固定最优 Grid 参数，GPU 批量扫描 Switch 相关参数

    GPU kernel 已支持所有参数为 per-combo 数组，实现全流程 GPU 加速。

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    use_gpu = gpu()["cupy_available"]
    open_arr = open_.values if open_ is not None else None

    fit_windows = [252]  # 必须固定为 252：短窗口导致严重过拟合
    all_ma_windows = [5, 10, 20, 60]
    ma_to_idx = {w: i for i, w in enumerate(all_ma_windows)}

    # ── Stage 1: Grid 参数 GPU 批量扫描（Switch 参数固定） ──
    print("  [PolyfitSwitch Stage 1] GPU-scanning grid params (switch fixed)…")

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

    flat_wait_days_fixed = 8
    switch_m1_fixed = 0.03
    switch_m2_fixed = 0.02
    switch_trailing_stop_fixed = 0.05
    switch_fast_fixed = 20
    switch_slow_fixed = 60

    indicator_combos = list(product(fit_windows, trend_windows, vol_windows))
    grid_combos = list(product(
        grid_pcts, vol_scales, trend_sens, max_grid_levels_vals,
        tpg_values, slg_values, position_sizes, position_sizing_coefs,
        min_signal_strengths,
    ))
    # ── Stage 1 全量参数扫描（GPU 约 10 秒完成 77760 组）──
    n_s1 = len(grid_combos)
    total_s1 = len(indicator_combos) * n_s1
    print(f"    indicator combos={len(indicator_combos)}  grid combos={n_s1}  total={total_s1}")

    # 预计算指标（缓存，key 包含 fit_window）
    indicator_cache = {}
    for fw, tw, vw in indicator_combos:
        key = (fw, tw, vw)
        if key not in indicator_cache:
            indicator_cache[key] = compute_polyfit_switch_indicators(
                close, fit_window_days=fw,
                ma_windows=all_ma_windows,
                trend_window_days=tw, vol_window_days=vw,
            )

    stage1_results = []

    # 预构建 per-combo 数组
    s1_bgp_a = np.array([p[0] for p in grid_combos])
    s1_vs_a  = np.array([p[1] for p in grid_combos])
    s1_ts_a  = np.array([p[2] for p in grid_combos])
    s1_mgl_a = np.array([p[3] for p in grid_combos], dtype=np.int32)
    s1_tpg_a = np.array([p[4] for p in grid_combos])
    s1_slg_a = np.array([p[5] for p in grid_combos])
    s1_psz_a = np.array([p[6] for p in grid_combos])
    s1_psc_a = np.array([p[7] for p in grid_combos])
    s1_mss_a = np.array([p[8] for p in grid_combos])
    # 固定的 switch 参数（全量 grid 数量）
    s1_fw_a  = np.full(n_s1, flat_wait_days_fixed, dtype=np.int32)
    s1_m1_a  = np.full(n_s1, switch_m1_fixed)
    s1_m2_a  = np.full(n_s1, switch_m2_fixed)
    s1_str_a = np.full(n_s1, switch_trailing_stop_fixed)
    s1_fi_a  = np.full(n_s1, ma_to_idx[switch_fast_fixed], dtype=np.int32)
    s1_si_a  = np.full(n_s1, ma_to_idx[switch_slow_fixed], dtype=np.int32)
    s1_cd_a  = np.full(n_s1, 1, dtype=np.int32)

    for (fw, tw, vw) in indicator_combos:
        indicators = indicator_cache[(fw, tw, vw)]
        common_idx = indicators.index
        if len(common_idx) == 0:
            continue  # 数据不足以计算 polyfit 指标（训练期太短）
        cl_aligned = close.loc[common_idx]
        cl_arr = cl_aligned.values
        op_aligned = open_arr[close.index.get_indexer(common_idx)] if open_arr is not None else None

        poly_base_arr = indicators["PolyBasePred"].values
        dev_pct_arr = indicators["PolyDevPct"].values
        dev_trend_arr = indicators["PolyDevTrend"].values
        vol_arr = indicators["RollingVolPct"].values
        ma_all = np.array([indicators[f"MA{mw}"].values for mw in all_ma_windows])

        if use_gpu:
            entries_b, exits_b, sizes_b = generate_polyfit_switch_signals_batch(
                cl_arr, dev_pct_arr, dev_trend_arr, vol_arr, poly_base_arr,
                ma_all, s1_fi_a, s1_si_a,
                s1_bgp_a, s1_vs_a, s1_ts_a, s1_tpg_a, s1_slg_a,
                s1_fw_a, s1_m1_a, s1_m2_a, s1_str_a,
                max_grid_levels_arr=s1_mgl_a,
                cooldown_days_arr=s1_cd_a,
                min_signal_strength_arr=s1_mss_a,
                position_size_arr=s1_psz_a,
                position_sizing_coef_arr=s1_psc_a,
            )
            bt = run_backtest_batch(cl_arr, entries_b, exits_b, sizes_b,
                                    n_combos=n_s1, open_=op_aligned)
            for si, (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss) in enumerate(grid_combos):
                if int(bt[si][4]) == 0:
                    continue
                stage1_results.append({
                    "total_return": bt[si][0], "sharpe_ratio": bt[si][1],
                    "max_drawdown": bt[si][2], "calmar_ratio": bt[si][3],
                    "num_trades": int(bt[si][4]), "win_rate": bt[si][5],
                    "fit_window_days": fw,
                    "trend_window_days": tw, "vol_window_days": vw,
                    "base_grid_pct": bgp, "volatility_scale": vs,
                    "trend_sensitivity": ts, "max_grid_levels": max_gl,
                    "take_profit_grid": tpg, "stop_loss_grid": slg,
                    "max_holding_days": 45, "cooldown_days": 1,
                    "min_signal_strength": min_ss,
                    "position_size": pos_sz, "position_sizing_coef": pos_coef,
                    "flat_wait_days": flat_wait_days_fixed,
                    "switch_deviation_m1": switch_m1_fixed,
                    "switch_deviation_m2": switch_m2_fixed,
                    "switch_trailing_stop": switch_trailing_stop_fixed,
                    "switch_fast_ma": switch_fast_fixed,
                    "switch_slow_ma": switch_slow_fixed,
                })
        else:
            # CPU 逐组合回退（使用采样后的参数）
            ma_fast_arr = indicators[f"MA{switch_fast_fixed}"].values
            ma_slow_arr = indicators[f"MA{switch_slow_fixed}"].values
            for (bgp, vs, ts, max_gl, tpg, slg, pos_sz, pos_coef, min_ss) in grid_combos:
                entries, exits, sizes = generate_polyfit_switch_signals(
                    cl_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                    poly_base_arr, ma_fast_arr, ma_slow_arr,
                    base_grid_pct=bgp, volatility_scale=vs,
                    trend_sensitivity=ts, max_grid_levels=max_gl,
                    take_profit_grid=tpg, stop_loss_grid=slg,
                    max_holding_days=45, cooldown_days=1,
                    min_signal_strength=min_ss,
                    position_size=pos_sz, position_sizing_coef=pos_coef,
                    flat_wait_days=flat_wait_days_fixed,
                    switch_deviation_m1=switch_m1_fixed,
                    switch_deviation_m2=switch_m2_fixed,
                    switch_trailing_stop=switch_trailing_stop_fixed,
                )
                if entries.sum() == 0:
                    continue
                m = run_backtest(cl_aligned, entries, exits, sizes, open_=open_)
                m["fit_window_days"] = fw
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
                m["flat_wait_days"] = flat_wait_days_fixed
                m["switch_deviation_m1"] = switch_m1_fixed
                m["switch_deviation_m2"] = switch_m2_fixed
                m["switch_fast_ma"] = switch_fast_fixed
                m["switch_slow_ma"] = switch_slow_fixed
                stage1_results.append(m)

    stage1_df = pd.DataFrame(stage1_results)
    if stage1_df.empty:
        return pd.DataFrame()

    best_grid = stage1_df.nlargest(1, "total_return").iloc[0]
    best_fw = int(best_grid["fit_window_days"])
    best_tw = int(best_grid["trend_window_days"])
    best_vw = int(best_grid["vol_window_days"])
    best_bgp = best_grid["base_grid_pct"]
    best_vs = best_grid["volatility_scale"]
    best_ts = best_grid["trend_sensitivity"]
    best_max_gl = int(best_grid["max_grid_levels"])
    best_tpg = best_grid["take_profit_grid"]
    best_slg = best_grid["stop_loss_grid"]
    best_pos_sz = best_grid["position_size"]
    best_pos_coef = best_grid["position_sizing_coef"]
    best_min_ss = best_grid["min_signal_strength"]

    print(f"  [PolyfitSwitch Stage 1] Best: fw={best_fw} tw={best_tw} vw={best_vw} "
          f"bgp={best_bgp:.4f} vs={best_vs:.1f} ts={best_ts:.0f} "
          f"max_gl={best_max_gl} tpg={best_tpg:.2f} slg={best_slg:.1f} "
          f"pos_sz={best_pos_sz:.2f} pos_coef={best_pos_coef:.0f} "
          f"min_ss={best_min_ss:.2f} ret={best_grid['total_return']:.1%}")

    # ── Stage 2: Switch 参数扫描（Grid 参数固定） ──
    print("  [PolyfitSwitch Stage 2] Scanning switch params (grid fixed)…")

    flat_wait_vals = [5, 8, 10, 15]
    switch_m1_vals = [0.02, 0.03, 0.04, 0.05]
    switch_m2_vals = [0.005, 0.01, 0.015, 0.02]
    switch_trailing_stop_vals = [0.02, 0.03, 0.05, 0.07, 0.10]
    switch_fast_vals = [5, 10, 20]
    switch_slow_vals = [10, 20, 60]

    all_ma_windows2 = sorted(set(switch_fast_vals + switch_slow_vals))
    ma_to_idx2 = {w: i for i, w in enumerate(all_ma_windows2)}

    switch_combos = []
    for fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow in product(
        flat_wait_vals, switch_m1_vals, switch_m2_vals,
        switch_trailing_stop_vals,
        switch_fast_vals, switch_slow_vals,
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_combos.append((fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow))

    indicators = compute_polyfit_switch_indicators(
        close, fit_window_days=best_fw,
        ma_windows=all_ma_windows2,
        trend_window_days=best_tw, vol_window_days=best_vw,
    )
    common_idx = indicators.index
    cl_aligned = close.loc[common_idx]
    cl_arr = cl_aligned.values
    poly_base_arr = indicators["PolyBasePred"].values
    dev_pct_arr = indicators["PolyDevPct"].values
    dev_trend_arr = indicators["PolyDevTrend"].values
    vol_arr = indicators["RollingVolPct"].values
    ma_all2 = np.array([indicators[f"MA{mw}"].values for mw in all_ma_windows2])

    n_switch = len(switch_combos)
    bgp_a = np.full(n_switch, best_bgp)
    vs_a = np.full(n_switch, best_vs)
    ts_a = np.full(n_switch, best_ts)
    tpg_a = np.full(n_switch, best_tpg)
    slg_a = np.full(n_switch, best_slg)
    fw_a = np.array([c[0] for c in switch_combos], dtype=np.int32)
    m1_a = np.array([c[1] for c in switch_combos])
    m2_a = np.array([c[2] for c in switch_combos])
    str_a = np.array([c[3] for c in switch_combos])
    fi_a = np.array([ma_to_idx2[c[4]] for c in switch_combos], dtype=np.int32)
    si_a = np.array([ma_to_idx2[c[5]] for c in switch_combos], dtype=np.int32)

    stage2_results = []

    # Stage 2 GPU 批量：Grid 参数固定，Switch 参数变化
    mgl_a = np.full(n_switch, best_max_gl, dtype=np.int32)
    cd_a = np.full(n_switch, 1, dtype=np.int32)
    mss_a = np.full(n_switch, best_min_ss)
    ps_a = np.full(n_switch, best_pos_sz)
    psc_a = np.full(n_switch, best_pos_coef)

    if use_gpu:
        entries_b, exits_b, sizes_b = generate_polyfit_switch_signals_batch(
            cl_arr, dev_pct_arr, dev_trend_arr, vol_arr, poly_base_arr,
            ma_all2, fi_a, si_a,
            bgp_a, vs_a, ts_a, tpg_a, slg_a,
            fw_a, m1_a, m2_a, str_a,
            max_grid_levels_arr=mgl_a,
            max_holding_days=45,
            cooldown_days_arr=cd_a,
            min_signal_strength_arr=mss_a,
            position_size_arr=ps_a,
            position_sizing_coef_arr=psc_a,
        )
        bt = run_backtest_batch(cl_arr, entries_b, exits_b, sizes_b, n_combos=n_switch,
                                open_=open_arr)
        for idx, (fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in enumerate(switch_combos):
            if int(bt[idx][4]) == 0:
                continue
            stage2_results.append({
                "total_return": bt[idx][0], "sharpe_ratio": bt[idx][1],
                "max_drawdown": bt[idx][2], "calmar_ratio": bt[idx][3],
                "num_trades": int(bt[idx][4]), "win_rate": bt[idx][5],
                "fit_window_days": best_fw,
                "trend_window_days": best_tw,
                "vol_window_days": best_vw,
                "base_grid_pct": best_bgp, "volatility_scale": best_vs,
                "trend_sensitivity": best_ts, "max_grid_levels": best_max_gl,
                "take_profit_grid": best_tpg, "stop_loss_grid": best_slg,
                "max_holding_days": 45, "cooldown_days": 1,
                "min_signal_strength": best_min_ss,
                "position_size": best_pos_sz,
                "position_sizing_coef": best_pos_coef,
                "flat_wait_days": fw, "switch_deviation_m1": sw_m1,
                "switch_deviation_m2": sw_m2,
                "switch_trailing_stop": sw_tr,
                "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
            })
    else:
        for (fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in switch_combos:
            ma_fast_arr = indicators[f"MA{sw_fast}"].values
            ma_slow_arr = indicators[f"MA{sw_slow}"].values
            entries, exits, sizes = generate_polyfit_switch_signals(
                cl_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                poly_base_arr, ma_fast_arr, ma_slow_arr,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, max_grid_levels=best_max_gl,
                take_profit_grid=best_tpg, stop_loss_grid=best_slg,
                max_holding_days=45, cooldown_days=1,
                min_signal_strength=best_min_ss,
                position_size=best_pos_sz,
                position_sizing_coef=best_pos_coef,
                flat_wait_days=fw,
                switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
            )
            if entries.sum() == 0:
                continue
            m = run_backtest(cl_aligned, entries, exits, sizes, open_=open_)
            m["fit_window_days"] = best_fw
            m["trend_window_days"] = best_tw
            m["vol_window_days"] = best_vw
            m["base_grid_pct"] = best_bgp
            m["volatility_scale"] = best_vs
            m["trend_sensitivity"] = best_ts
            m["max_grid_levels"] = best_max_gl
            m["take_profit_grid"] = best_tpg
            m["stop_loss_grid"] = best_slg
            m["max_holding_days"] = 45
            m["cooldown_days"] = 1
            m["min_signal_strength"] = best_min_ss
            m["position_size"] = best_pos_sz
            m["position_sizing_coef"] = best_pos_coef
            m["flat_wait_days"] = fw
            m["switch_deviation_m1"] = sw_m1
            m["switch_deviation_m2"] = sw_m2
            m["switch_trailing_stop"] = sw_tr
            m["switch_fast_ma"] = sw_fast
            m["switch_slow_ma"] = sw_slow
            stage2_results.append(m)

    return pd.DataFrame(stage2_results)
