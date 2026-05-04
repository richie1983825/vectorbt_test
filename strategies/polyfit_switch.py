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
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
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
    switch_deviation_m1: float = 0.03,      # Switch 激活偏离阈值
    switch_deviation_m2: float = 0.02,      # Switch 关闭偏离阈值
    switch_trailing_stop: float = 0.05,     # Switch 追踪止损回撤
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成 Polyfit-Switch 策略的入场/离场信号（v3：共享仓位 + 金叉入场）。

    v3 逻辑：
      - Grid 与 Switch 共享一个仓位（position_mode: 0=idle, 1=grid, 2=switch）
      - Switch 激活条件：偏离度 > m1 且 position_mode == 0
      - Switch 入场：金叉（fma > sma）且 Switch 已激活且金叉未被使用过
      - Switch 离场：追踪止损触发 或 偏离度 < m2
      - Switch 离场后：当前金叉标记为"已使用"，必须等下一根金叉才能重新入场
      - 模式切换（Grid↔Switch）时仓位必须为 0，否则打印 warning

    Returns:
        entries, exits, sizes, entry_modes
        entry_modes: 1=grid 入场, 2=switch 入场, 0=非入场 bar
    """
    import logging
    _log = logging.getLogger(__name__)

    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)
    entry_modes = np.zeros(n, dtype=np.int8)

    # 共享仓位
    position_mode = 0       # 0=idle, 1=grid, 2=switch
    entry_bar = -1
    entry_level = 1
    entry_grid_step = np.nan
    entry_close_price = np.nan  # 入场价，用于 TP/SL 价格计算
    cooldown = 0

    # Switch 状态
    switch_active = False         # Switch 已激活（偏离度满足，等待金叉）
    switch_peak = 0.0             # 持仓期间最高价
    cross_available = False       # 有可用的（未被使用过的）金叉
    prev_fma = 0.0
    prev_sma = 0.0

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        pb = poly_base[i]
        cl = close[i]
        fma = ma_fast[i]
        sma = ma_slow[i]

        if np.isnan(dp) or np.isnan(dt) or np.isnan(vp) or np.isnan(pb) or np.isnan(fma) or np.isnan(sma) or pb <= 0 or cl <= 0:
            prev_fma, prev_sma = fma, sma
            continue

        # 检测金叉（快线上穿慢线）
        new_golden_cross = (fma > sma and prev_fma <= prev_sma)
        if new_golden_cross:
            cross_available = True  # 新鲜金叉可用

        # 动态网格步长
        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        # 持仓冷却
        if cooldown > 0:
            cooldown -= 1

        # ── IDLE 状态：Grid 入场 或 Switch 激活 ──
        if position_mode == 0:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan

            # Grid 入场（均值回复）
            if cooldown <= 0:
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
                        entry_modes[i] = 1
                        position_mode = 1
                        entry_bar = i
                        entry_level = entry_lvl
                        entry_grid_step = dynamic_grid_step
                        entry_close_price = cl

            # Switch 激活（趋势追踪，等待金叉入场）
            if position_mode == 0 and not switch_active and dp > switch_deviation_m1:
                switch_active = True
                cross_available = new_golden_cross  # 激活时若已有金叉则立即可用

            # Switch 入场：激活中 + 有可用的金叉
            if position_mode == 0 and switch_active and cross_available:
                entries[i] = True
                sizes[i] = float(np.nextafter(1.0, 0.0))  # 接近全仓
                entry_modes[i] = 2
                position_mode = 2
                entry_bar = i
                entry_grid_step = max(base_grid_pct, 1e-9)
                switch_peak = cl
                cross_available = False  # 消耗此金叉

        # ── Grid 持仓状态 ──
        elif position_mode == 1:
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            # TP: 偏离度恢复到 tp_threshold 以上（接近基线）, SL: 偏离度扩大超过 sl_threshold
            if hold_limit or dp >= tp_threshold or dp <= -sl_threshold:
                exits[i] = True
                position_mode = 0
                cooldown = cooldown_days

        # ── Switch 持仓状态 ──
        elif position_mode == 2:
            switch_peak = max(switch_peak, cl)
            trailing_exit = cl <= switch_peak * (1.0 - switch_trailing_stop)
            deviation_exit = dp < switch_deviation_m2
            if trailing_exit or deviation_exit:
                exits[i] = True
                position_mode = 0
                switch_active = False
                switch_peak = 0.0
                cross_available = False  # 离场后必须等下一根金叉
                cooldown = cooldown_days

        # Switch 关闭条件（未持仓时，偏离度回落）
        if switch_active and position_mode != 2 and dp < switch_deviation_m2:
            switch_active = False
            cross_available = False

        prev_fma, prev_sma = fma, sma

    # 末尾强制平仓
    if position_mode != 0:
        exits[-1] = True

    return entries, exits, sizes, entry_modes


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
extern "C" __global__ void polyfit_switch_signals_kernel_v3(
    const double* close,
    const double* dev_pct,
    const double* dev_trend_all,     // [n_indicator_sets * n_bars] 展平
    const double* vol_pct_all,       // [n_indicator_sets * n_bars] 展平
    const int* indicator_idx,        // [n_combos] — 每个组合对应哪个 indicator set
    const double* poly_base,
    const double* ma_all,            // 展平的多根 MA: [n_ma_windows * n_bars]
    const int* fast_ma_idx,          // 每组合的快均线在 ma_all 中的索引
    const int* slow_ma_idx,          // 每组合的慢均线在 ma_all 中的索引
    const double* base_grid_pct,
    const double* volatility_scale,
    const double* trend_sensitivity,
    const double* take_profit_grid,
    const double* stop_loss_grid,
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

    int iset = indicator_idx[combo_idx];  // 该组合对应的 indicator set

    // 读取当前组合的所有参数
    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];
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

    // v3 状态机：共享仓位 + 金叉入场 + 新鲜金叉追踪
    int position_mode = 0;       // 0=idle, 1=grid, 2=switch
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    double entry_close_price = 0.0 / 0.0;  // NaN
    int cooldown = 0;
    bool switch_active = false;
    double switch_peak = 0.0;
    bool cross_available = false;
    double prev_fma = 0.0, prev_sma = 0.0;

    for (int i = 0; i < n_bars; i++) {
        double cl = close[i];
        double dp = dev_pct[i];
        double dt = dev_trend_all[iset * n_bars + i];
        double vp = vol_pct_all[iset * n_bars + i];
        double pb = poly_base[i];
        double fma = ma_all[fi * n_bars + i];
        double sma = ma_all[si * n_bars + i];

        entries[combo_idx * n_bars + i] = false;
        exits[combo_idx * n_bars + i] = false;
        sizes[combo_idx * n_bars + i] = 0.0;

        if (isnan(cl) || isnan(dp) || isnan(dt) || isnan(vp)
            || isnan(pb) || isnan(fma) || isnan(sma)
            || pb <= 0.0 || cl <= 0.0) {
            prev_fma = fma; prev_sma = sma;
            continue;
        }

        // 检测新金叉
        bool new_cross = (fma > sma && prev_fma <= prev_sma);
        if (new_cross) cross_available = true;

        if (cooldown > 0) cooldown--;

        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        // ── IDLE ──
        if (position_mode == 0) {
            entry_bar = -1; entry_level = 1; entry_grid_step = -1.0;
            entry_close_price = 0.0 / 0.0;

            // Grid 入场
            if (cooldown <= 0) {
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
                        position_mode = 1;
                        entry_bar = i; entry_level = el; entry_grid_step = dgs;
                        entry_close_price = cl;
                    }
                }
            }

            // Switch 激活
            if (position_mode == 0 && !switch_active && dp > sw_m1) {
                switch_active = true;
                cross_available = new_cross;
            }
            // Switch 入场：激活 + 有可用的新鲜金叉
            if (position_mode == 0 && switch_active && cross_available) {
                entries[combo_idx * n_bars + i] = true;
                sizes[combo_idx * n_bars + i] = 0.9999999999999999;
                position_mode = 2;
                entry_bar = i; entry_grid_step = fmax(bgp, 1e-9);
                switch_peak = cl;
                cross_available = false;
            }
        }

        // ── Grid 持仓 ──
        else if (position_mode == 1) {
            int hd = i - entry_bar;
            bool hl = hd >= max_holding_days;
            double rs = fmax(dgs, entry_grid_step);
            if (entry_grid_step < 0.0 || isnan(entry_grid_step)) rs = dgs;
            double tp_threshold = entry_level * rs * tpg;
            double sl_threshold = entry_level * rs * slg;
            // TP/SL 均以基线偏离度 dp 为锚
            if (hl || dp >= tp_threshold || dp <= -sl_threshold) {
                exits[combo_idx * n_bars + i] = true;
                position_mode = 0;
                cooldown = cd;
            }
        }

        // ── Switch 持仓 ──
        else if (position_mode == 2) {
            switch_peak = fmax(switch_peak, cl);
            bool trail_exit = cl <= switch_peak * (1.0 - sw_ts);
            bool dev_exit = dp < sw_m2;
            if (trail_exit || dev_exit) {
                exits[combo_idx * n_bars + i] = true;
                position_mode = 0;
                switch_active = false;
                switch_peak = 0.0;
                cross_available = false;  // 离场后等新鲜金叉
                cooldown = cd;
            }
        }

        // Switch 关闭（未持仓时偏离回落）
        if (switch_active && position_mode != 2 && dp < sw_m2) {
            switch_active = false;
            cross_available = false;
        }

        prev_fma = fma; prev_sma = sma;
    }

    if (position_mode != 0) {
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
            _POLYFIT_SWITCH_KERNEL_CODE, "polyfit_switch_signals_kernel_v3"
        )
    return _polyfit_switch_kernel


def generate_polyfit_switch_signals_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend_all: np.ndarray,      # [n_indicator_sets, n_bars] or [n_bars]
    rolling_vol_pct_all: np.ndarray,# [n_indicator_sets, n_bars] or [n_bars]
    poly_base: np.ndarray,          # [n_bars]
    ma_all: np.ndarray,             # [n_ma_windows, n_bars]
    fast_ma_idx: np.ndarray,        # [n_combos] int32
    slow_ma_idx: np.ndarray,        # [n_combos] int32
    base_grid_pcts: np.ndarray,     # [n_combos]
    volatility_scales: np.ndarray,  # [n_combos]
    trend_sensitivities: np.ndarray,# [n_combos]
    take_profit_grids: np.ndarray,  # [n_combos]
    stop_loss_grids: np.ndarray,    # [n_combos]
    switch_deviation_m1_arr: np.ndarray,  # [n_combos]
    switch_deviation_m2_arr: np.ndarray,  # [n_combos]
    switch_trailing_stop_arr: np.ndarray | None = None,  # [n_combos]
    flat_wait_days_arr: np.ndarray | None = None,  # [n_combos] int32 — v3: ignored, kept for compat
    indicator_idx: np.ndarray | None = None,  # [n_combos] int32 — 每个组合对应的 indicator set
    max_grid_levels_arr: np.ndarray | None = None,   # [n_combos] int32
    max_holding_days: int = 45,
    cooldown_days_arr: np.ndarray | None = None,     # [n_combos] int32
    min_signal_strength_arr: np.ndarray | None = None,# [n_combos]
    position_size_arr: np.ndarray | None = None,      # [n_combos]
    position_sizing_coef_arr: np.ndarray | None = None,# [n_combos]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU 批量生成 Polyfit-Switch 策略信号。

    支持多 indicator set 批量处理：通过 indicator_idx 指定每个参数组合
    使用哪一组 dev_trend/vol_pct 指标，避免多次 GPU kernel 调用。

    若 dev_trend_all/vol_pct_all 为 1D，自动视为单 indicator set（兼容旧调用）。

    参数数组需 pad 到 256 的倍数以对齐 CUDA block。
    """
    cp = xp()
    n_bars = len(close)
    n_combos = len(base_grid_pcts)

    # 兼容 1D 输入：转为 2D
    if dev_trend_all.ndim == 1:
        dev_trend_all = dev_trend_all.reshape(1, -1)
    if rolling_vol_pct_all.ndim == 1:
        rolling_vol_pct_all = rolling_vol_pct_all.reshape(1, -1)
    n_indicator_sets = dev_trend_all.shape[0]

    # 默认 indicator_idx：全零（单 indicator set）
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
    iset_d = _pad_i32(indicator_idx)

    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    trend_all_d = cp.asarray(dev_trend_all.ravel(), dtype=cp.float64)
    vol_all_d = cp.asarray(rolling_vol_pct_all.ravel(), dtype=cp.float64)
    pb_d = cp.asarray(poly_base, dtype=cp.float64)
    ma_d = cp.asarray(ma_all.ravel(), dtype=cp.float64)

    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    kernel = _get_polyfit_switch_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, dev_d, trend_all_d, vol_all_d, iset_d,
            pb_d, ma_d, fi_d, si_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d, m1_d, m2_d,
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
# 两阶段参数扫描（GPU）
# ══════════════════════════════════════════════════════════════════

def scan_polyfit_switch_two_stage(close: pd.Series,
                                   open_: pd.Series | None = None) -> pd.DataFrame:
    """两阶段扫描：Stage 1 复用 Polyfit-Grid 扫描，Stage 2 扫描 Switch 参数。

    Stage 1: 调用 scan_polyfit_grid() 获取纯 Grid 最优参数（与 Polyfit-Grid 策略一致）
    Stage 2: 固定最优 Grid 参数，GPU 批量扫描 Switch 参数（m1/m2/trailing_stop/MA）

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    use_gpu = gpu()["cupy_available"]
    open_arr = open_.values if open_ is not None else None

    # ── Stage 1: 复用 Polyfit-Grid 扫描（纯Grid，确保与Polyfit-Grid策略一致） ──
    print("  [PolyfitSwitch Stage 1] Reusing Polyfit-Grid scan…")
    from strategies.polyfit_grid import scan_polyfit_grid as _scan_grid
    stage1_df = _scan_grid(close, open_=open_)
    if stage1_df.empty:
        return pd.DataFrame()

    best_grid = stage1_df.nlargest(1, "total_return").iloc[0]
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

    print(f"  [PolyfitSwitch Stage 1] Best: tw={best_tw} vw={best_vw} "
          f"bgp={best_bgp:.4f} vs={best_vs:.1f} ts={best_ts:.0f} "
          f"max_gl={best_max_gl} tpg={best_tpg:.2f} slg={best_slg:.1f} "
          f"pos_sz={best_pos_sz:.2f} pos_coef={best_pos_coef:.0f} "
          f"min_ss={best_min_ss:.2f} ret={best_grid['total_return']:.1%}")

    # ── Stage 2: Switch 参数扫描（Grid 参数固定） ──
    print("  [PolyfitSwitch Stage 2] Scanning switch params (grid fixed)…")

    switch_m1_vals = [0.02, 0.03, 0.04, 0.05]
    switch_m2_vals = [0.005, 0.01, 0.015, 0.02]
    switch_trailing_stop_vals = [0.02, 0.03, 0.05, 0.07, 0.10]
    switch_fast_vals = [5, 10, 20]
    switch_slow_vals = [10, 20, 60]

    all_ma_windows2 = sorted(set(switch_fast_vals + switch_slow_vals))
    ma_to_idx2 = {w: i for i, w in enumerate(all_ma_windows2)}

    switch_combos = []
    for sw_m1, sw_m2, sw_tr, sw_fast, sw_slow in product(
        switch_m1_vals, switch_m2_vals,
        switch_trailing_stop_vals,
        switch_fast_vals, switch_slow_vals,
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_combos.append((sw_m1, sw_m2, sw_tr, sw_fast, sw_slow))

    indicators = compute_polyfit_switch_indicators(
        close, fit_window_days=252,
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
    m1_a = np.array([c[0] for c in switch_combos])
    m2_a = np.array([c[1] for c in switch_combos])
    str_a = np.array([c[2] for c in switch_combos])
    fi_a = np.array([ma_to_idx2[c[3]] for c in switch_combos], dtype=np.int32)
    si_a = np.array([ma_to_idx2[c[4]] for c in switch_combos], dtype=np.int32)

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
            m1_a, m2_a, str_a,
            max_grid_levels_arr=mgl_a,
            max_holding_days=45,
            cooldown_days_arr=cd_a,
            min_signal_strength_arr=mss_a,
            position_size_arr=ps_a,
            position_sizing_coef_arr=psc_a,
        )
        op_aligned = open_arr[close.index.get_indexer(common_idx)] if open_arr is not None else None
        bt = run_backtest_batch(cl_arr, entries_b, exits_b, sizes_b, n_combos=n_switch,
                                open_=op_aligned)
        for idx, (sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in enumerate(switch_combos):
            if int(bt[idx][4]) == 0:
                continue
            stage2_results.append({
                "total_return": bt[idx][0], "sharpe_ratio": bt[idx][1],
                "max_drawdown": bt[idx][2], "calmar_ratio": bt[idx][3],
                "num_trades": int(bt[idx][4]), "win_rate": bt[idx][5],
                "fit_window_days": 252,
                "trend_window_days": best_tw,
                "vol_window_days": best_vw,
                "base_grid_pct": best_bgp, "volatility_scale": best_vs,
                "trend_sensitivity": best_ts, "max_grid_levels": best_max_gl,
                "take_profit_grid": best_tpg, "stop_loss_grid": best_slg,
                "max_holding_days": 45, "cooldown_days": 1,
                "min_signal_strength": best_min_ss,
                "position_size": best_pos_sz,
                "position_sizing_coef": best_pos_coef,
                "switch_deviation_m1": sw_m1,
                "switch_deviation_m2": sw_m2,
                "switch_trailing_stop": sw_tr,
                "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
            })
    else:
        cl_s = close.loc[common_idx]
        op_s = open_.reindex(common_idx) if open_ is not None else None
        for (sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in switch_combos:
            ma_fast_arr = indicators[f"MA{sw_fast}"].values
            ma_slow_arr = indicators[f"MA{sw_slow}"].values
            entries, exits, sizes, _modes = generate_polyfit_switch_signals(
                cl_arr, dev_pct_arr, dev_trend_arr, vol_arr,
                poly_base_arr, ma_fast_arr, ma_slow_arr,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, max_grid_levels=best_max_gl,
                take_profit_grid=best_tpg, stop_loss_grid=best_slg,
                max_holding_days=45, cooldown_days=1,
                min_signal_strength=best_min_ss,
                position_size=best_pos_sz,
                position_sizing_coef=best_pos_coef,
                switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
            )
            if entries.sum() == 0:
                continue
            m = run_backtest(cl_s, entries, exits, sizes, open_=op_s)
            m["fit_window_days"] = 252
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
            m["switch_deviation_m1"] = sw_m1
            m["switch_deviation_m2"] = sw_m2
            m["switch_trailing_stop"] = sw_tr
            m["switch_fast_ma"] = sw_fast
            m["switch_slow_ma"] = sw_slow
            stage2_results.append(m)

    return pd.DataFrame(stage2_results)
