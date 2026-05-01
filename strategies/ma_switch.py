"""MA Switch Strategy — 均线网格 + 均线交叉双模式策略。

双模式说明：
  1. Grid 模式（默认）：均值回复网格交易，在 MA 基准线下方挂网格多单。
  2. Switch 模式（趋势追踪）：当价格连续 flat_wait_days 根 bar 在 MA 上方
     且偏离超过阈值时激活，使用快慢均线金叉入场 + 最高价回撤止损离场。

Switch 模式激活条件：flat_wait_days 后，close > MA 且 dev_pct > switch_deviation_m1
Switch 模式关闭条件：dev_pct < switch_deviation_m2（偏离回归到合理范围）

GPU 加速：Switch 策略的参数空间更大（包含 grid 参数 + switch 参数），
通过 CuPy RawKernel 将多组参数批量生成信号。
"""

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from utils.backtest import run_backtest, run_backtest_batch
from utils.gpu import xp, gpu
from utils.indicators import compute_ma_switch_indicators


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成
# ══════════════════════════════════════════════════════════════════

def generate_switch_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    ma_base: np.ndarray,       # MA 基准线
    ma_fast: np.ndarray,       # 快均线（Switch 模式用）
    ma_slow: np.ndarray,       # 慢均线（Switch 模式用）
    base_grid_pct: float = 0.012,
    volatility_scale: float = 1.0,
    trend_sensitivity: float = 8.0,
    max_grid_levels: int = 3,
    take_profit_grid: float = 0.85,
    stop_loss_grid: float = 1.6,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,          # 最大仓位比例
    position_sizing_coef: float = 30.0,   # 仓位系数（偏离越大仓位越重）
    flat_wait_days: int = 8,              # 横盘等待天数（触发 Switch 前需连续无持仓）
    switch_deviation_m1: float = 0.03,    # Switch 激活阈值（偏离需超过此值）
    switch_deviation_m2: float = 0.01,    # Switch 关闭阈值（偏离低于此值退出）
    switch_trailing_stop: float = 0.05,   # Switch 追踪止损（从最高点回撤 5%）
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 MA-Switch 策略的入场/离场信号。

    两种模式互斥：同一时间只能处于一种模式。
    Switch 模式仅在 Grid 模式闲置（无持仓）且满足激活条件时触发。

    返回值：
      entries/exits/sizes 均为 n_bars 长度数组。
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
    ending_position = 0.0
    switch_mode_active = False
    flat_days = 0       # 连续无持仓天数
    peak_close = 0.0    # Switch 持仓期间的最高收盘价（追踪止损用）

    for i in range(n):
        dp = dev_pct[i]
        dt = dev_trend[i]
        vp = rolling_vol_pct[i]
        mb = ma_base[i]
        cl = close[i]
        fma = ma_fast[i]
        sma = ma_slow[i]

        # 跳过无效数据
        nan_vars = (np.isnan(dp) or np.isnan(dt) or np.isnan(vp)
                    or np.isnan(mb) or np.isnan(fma) or np.isnan(sma))
        if nan_vars or mb <= 0 or cl <= 0:
            continue

        # Switch 模式持仓期间，持续更新最高价
        if in_position and switch_mode_active:
            peak_close = max(peak_close, cl)

        # 无持仓时累计 flat_days，用于判断横盘时间是否满足 Switch 激活条件
        if in_position:
            flat_days = 0
        else:
            entry_bar = -1
            entry_level = 1
            entry_grid_step = np.nan
            flat_days += 1

        if cooldown > 0:
            cooldown -= 1

        # 动态网格步长（同 MA Grid 策略逻辑）
        vol_mult = 1.0 + volatility_scale * max(vp, 0.0)
        dynamic_grid_step = (
            base_grid_pct * (1.0 + trend_sensitivity * abs(dt)) * vol_mult
        )
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        # ── Switch 模式激活条件 ──
        # 仅在 Grid 模式闲置（无持仓）时检查
        if (not switch_mode_active and not in_position
                and flat_days >= flat_wait_days
                and cl > mb and dp > switch_deviation_m1):
            switch_mode_active = True

        # ── Switch 模式逻辑 ──
        if switch_mode_active:
            # 偏离回落到 m2 以下 → 关闭 Switch 模式，回到 Grid
            if dp < switch_deviation_m2:
                switch_mode_active = False
                continue
            # 价格在均线下方或偏离不足 → Switch 模式暂时挂起
            if cl <= mb or dp <= switch_deviation_m1:
                continue
            # 快线上穿慢线（金叉）+ 无持仓 → 入场
            if fma > sma and not in_position:
                full_size = float(np.nextafter(1.0, 0.0))  # ≈ 0.999... 全仓
                entries[i] = True
                sizes[i] = full_size
                in_position = True
                entry_bar = i
                entry_level = 1
                entry_grid_step = max(base_grid_pct, 1e-9)
                flat_days = 0
                peak_close = cl
                continue
            # Switch 持仓中 → 追踪止损离场
            if in_position:
                if cl <= peak_close * (1.0 - switch_trailing_stop):
                    exits[i] = True
                    in_position = False
                    cooldown = cooldown_days
                    peak_close = 0.0
            continue

        # ── Grid 模式逻辑（与 MA Grid 策略相同） ──
        if not in_position:
            if cooldown > 0:
                continue
            signal_strength = abs(dp) / max(dynamic_grid_step, 1e-9)
            entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
            entry_threshold = -entry_lvl * dynamic_grid_step
            if dp <= entry_threshold and signal_strength >= min_signal_strength:
                # 仓位随偏离程度和波动率动态调整
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

    # 末尾强制平仓
    if in_position:
        exits[-1] = True

    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# GPU 批量 Switch 信号生成 — CuPy RawKernel
#
# 与 Grid kernel 不同，Switch kernel 额外接收多根 MA（快/慢均线）和
# Switch 专有参数（flat_wait_days, switch_deviation_m1/m2, trailing_stop）。
# 每个 CUDA 线程处理一组参数组合。
# ══════════════════════════════════════════════════════════════════

_switch_kernel = None

_SWITCH_KERNEL_CODE = r"""
extern "C" __global__ void switch_signals_kernel(
    const double* close,
    const double* dev_pct,
    const double* dev_trend,
    const double* vol_pct,
    const double* ma_base,
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
    const double* switch_trailing_stop,
    bool* entries,
    bool* exits,
    double* sizes,
    int n_bars,
    int n_combos,
    int max_grid_levels,
    int max_holding_days,
    int cooldown_days,
    double min_signal_strength,
    double position_size,
    double position_sizing_coef
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
        double mb = ma_base[i];
        // 从展平的 ma_all 中索引对应 MA 值
        double fma = ma_all[fi * n_bars + i];
        double sma = ma_all[si * n_bars + i];

        // 跳过无效数据
        if (isnan(cl) || isnan(dp) || isnan(dt) || isnan(vp)
            || isnan(mb) || isnan(fma) || isnan(sma)
            || mb <= 0.0 || cl <= 0.0) {
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        // 更新 Switch 模式持仓期间的最高价
        if (in_position && switch_mode_active) {
            peak_close = fmax(peak_close, cl);
        }

        // 更新持仓/空仓状态
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

        // Switch 模式激活
        if (!switch_mode_active && !in_position
                && flat_days >= fw && cl > mb && dp > sw_m1) {
            switch_mode_active = true;
        }

        // Switch 模式逻辑
        if (switch_mode_active) {
            if (dp < sw_m2) {  // 偏离回落 → 关闭 Switch
                switch_mode_active = false;
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            if (cl <= mb || dp <= sw_m1) {  // 条件不满足 → 挂起
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            if (fma > sma && !in_position) {  // 金叉入场
                entries[combo_idx * n_bars + i] = true;
                sizes[combo_idx * n_bars + i] = 0.9999999999999999;  // 接近全仓
                in_position = true;
                entry_bar = i;
                entry_level = 1;
                entry_grid_step = fmax(bgp, 1e-9);
                flat_days = 0;
                peak_close = cl;
                continue;
            }
            if (in_position) {
                // 追踪止损离场
                if (cl <= peak_close * (1.0 - sw_ts)) {
                    exits[combo_idx * n_bars + i] = true;
                    in_position = false;
                    cooldown = cooldown_days;
                    peak_close = 0.0;
                } else {
                    exits[combo_idx * n_bars + i] = false;
                }
            }
            entries[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        // Grid 模式逻辑
        entries[combo_idx * n_bars + i] = false;
        exits[combo_idx * n_bars + i] = false;
        sizes[combo_idx * n_bars + i] = 0.0;

        if (!in_position) {
            if (cooldown > 0) continue;
            double signal_strength = fabs(dp) / fmax(dgs, 1e-9);
            int entry_lvl = (int)floor(signal_strength);
            entry_lvl = entry_lvl < 1 ? 1 : (entry_lvl > max_grid_levels ? max_grid_levels : entry_lvl);
            double entry_threshold = -(double)entry_lvl * dgs;
            if (dp <= entry_threshold && signal_strength >= min_signal_strength) {
                double size = fabs(dp) * (1.0 + fmax(vp, 0.0)) * position_sizing_coef;
                size = size > position_size ? position_size : (size < 0.0 ? 0.0 : size);
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
                cooldown = cooldown_days;
            }
        }
    }

    // 末尾强制平仓
    if (in_position) {
        exits[combo_idx * n_bars + n_bars - 1] = true;
    }
}
"""


def _get_switch_kernel():
    """延迟编译 CUDA kernel，避免在 CuPy 不可用时报错。"""
    global _switch_kernel
    if _switch_kernel is None:
        cp = xp()
        _switch_kernel = cp.RawKernel(_SWITCH_KERNEL_CODE, "switch_signals_kernel")
    return _switch_kernel


def generate_switch_signals_batch(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    ma_base: np.ndarray,
    ma_all: np.ndarray,          # [n_ma_windows, n_bars] — 展平前
    fast_ma_idx: np.ndarray,     # [n_combos] int32
    slow_ma_idx: np.ndarray,     # [n_combos] int32
    base_grid_pcts: np.ndarray,
    volatility_scales: np.ndarray,
    trend_sensitivities: np.ndarray,
    take_profit_grids: np.ndarray,
    stop_loss_grids: np.ndarray,
    flat_wait_days_arr: np.ndarray,
    switch_deviation_m1_arr: np.ndarray,
    switch_deviation_m2_arr: np.ndarray,
    switch_trailing_stop_arr: np.ndarray,
    max_grid_levels: int = 3,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = 0.5,
    position_sizing_coef: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU 批量生成 Switch 策略信号。

    将所有参数组合一次性传入 CUDA kernel，每个线程处理一组。
    参数数组需 pad 到 256 的倍数以对齐 CUDA block。
    """
    cp = xp()
    n_bars = len(close)
    n_combos = len(base_grid_pcts)

    block_size = 256
    grid_size = (n_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    def _pad_f64(a):
        """Pad float64 数组，多余位置填 NaN 以避免产生信号。"""
        result = cp.zeros(padded, dtype=cp.float64)
        result[:n_combos] = cp.asarray(a, dtype=cp.float64)
        if padded > n_combos:
            result[n_combos:] = cp.nan
        return result

    def _pad_i32(a):
        """Pad int32 数组，多余位置填 0。"""
        result = cp.zeros(padded, dtype=cp.int32)
        result[:n_combos] = cp.asarray(a, dtype=cp.int32)
        return result

    # Pad 各参数到 block 对齐
    bgp_d = _pad_f64(base_grid_pcts)
    vs_d = _pad_f64(volatility_scales)
    ts_d = _pad_f64(trend_sensitivities)
    tpg_d = _pad_f64(take_profit_grids)
    slg_d = _pad_f64(stop_loss_grids)
    fw_d = _pad_i32(flat_wait_days_arr)
    m1_d = _pad_f64(switch_deviation_m1_arr)
    m2_d = _pad_f64(switch_deviation_m2_arr)
    tr_d = _pad_f64(switch_trailing_stop_arr)
    fi_d = _pad_i32(fast_ma_idx)
    si_d = _pad_i32(slow_ma_idx)

    # 上传指标数据到 GPU（所有组合共享）
    close_d = cp.asarray(close, dtype=cp.float64)
    dev_d = cp.asarray(dev_pct, dtype=cp.float64)
    trend_d = cp.asarray(dev_trend, dtype=cp.float64)
    vol_d = cp.asarray(rolling_vol_pct, dtype=cp.float64)
    mb_d = cp.asarray(ma_base, dtype=cp.float64)
    ma_d = cp.asarray(ma_all.ravel(), dtype=cp.float64)  # 展平为 1D

    # 分配输出缓冲区
    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    # 启动 CUDA kernel
    kernel = _get_switch_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            close_d, dev_d, trend_d, vol_d, mb_d, ma_d, fi_d, si_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d, fw_d, m1_d, m2_d, tr_d,
            entries_d, exits_d, sizes_d,
            n_bars, n_combos, max_grid_levels, max_holding_days, cooldown_days,
            min_signal_strength, position_size, position_sizing_coef,
        ),
    )

    # 取回结果，仅保留有效组合
    entries = cp.asnumpy(entries_d).reshape(padded, n_bars)[:n_combos]
    exits = cp.asnumpy(exits_d).reshape(padded, n_bars)[:n_combos]
    sizes = cp.asnumpy(sizes_d).reshape(padded, n_bars)[:n_combos]
    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# 单阶段参数扫描 — Grid + Switch 参数全排列
# ══════════════════════════════════════════════════════════════════

def scan_switch_strategy(close: pd.Series) -> pd.DataFrame:
    """平坦参数扫描（grid 参数 + switch 参数全排列）。保留作为参考。

    与 scan_switch_two_stage 的区别：
      - 此函数对 grid 和 switch 参数做全排列，组合数巨大
      - scan_switch_two_stage 分两阶段：先优化 grid，再在最优 grid 上优化 switch
    """
    print("  Switch parameter scan…")

    windows = [50, 75]
    grid_pcts = [0.01, 0.02]
    vol_scales = [0.5, 2.0]
    trend_sens = [4.0, 12.0]
    tpg_values = [0.7, 1.0]
    slg_values = [1.2, 2.0]

    flat_wait_days_vals = [5, 8, 15]
    switch_m1_vals = [0.02, 0.03, 0.05]
    switch_m2_vals = [0.005, 0.01, 0.02]
    switch_fast_vals = [5, 20]
    switch_slow_vals = [10, 60]

    # 构建 ma_window → index 映射，GPU 模式用索引访问 MA
    all_ma_windows = sorted(set(switch_fast_vals + switch_slow_vals))
    ma_to_idx = {w: i for i, w in enumerate(all_ma_windows)}
    use_gpu = gpu()["cupy_available"]

    grid_params = list(product(grid_pcts, vol_scales, trend_sens, tpg_values, slg_values))
    switch_params = []
    for fw, sw_m1, sw_m2, sw_fast, sw_slow in product(
        flat_wait_days_vals, switch_m1_vals, switch_m2_vals,
        switch_fast_vals, switch_slow_vals,
    ):
        # 过滤无效组合：close threshold 不能 >= activation threshold，快线必须短于慢线
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_params.append((fw, sw_m1, sw_m2, sw_fast, sw_slow))

    all_combos = list(product(windows, grid_params, switch_params))
    results = []
    count = 0

    for w, (bgp, vs, ts, tpg, slg), (fw, sw_m1, sw_m2, sw_fast, sw_slow) in all_combos:
        indicators = compute_ma_switch_indicators(close, w, ma_windows=all_ma_windows)
        close_arr = close.values
        ma_fast_arr = indicators[f"MA{sw_fast}"].values
        ma_slow_arr = indicators[f"MA{sw_slow}"].values

        entries, exits, sizes = generate_switch_signals(
            close_arr,
            indicators["MADevPct"].values,
            indicators["MADevTrend"].values,
            indicators["RollingVolPct"].values,
            indicators["MABase"].values,
            ma_fast_arr, ma_slow_arr,
            base_grid_pct=bgp, volatility_scale=vs,
            trend_sensitivity=ts, take_profit_grid=tpg,
            stop_loss_grid=slg,
            flat_wait_days=fw,
            switch_deviation_m1=sw_m1,
            switch_deviation_m2=sw_m2,
        )
        if entries.sum() == 0:
            continue
        m = run_backtest(close, entries, exits, sizes)
        m["ma_window"] = w
        m["base_grid_pct"] = bgp
        m["volatility_scale"] = vs
        m["trend_sensitivity"] = ts
        m["take_profit_grid"] = tpg
        m["stop_loss_grid"] = slg
        m["flat_wait_days"] = fw
        m["switch_deviation_m1"] = sw_m1
        m["switch_deviation_m2"] = sw_m2
        m["switch_fast_ma"] = sw_fast
        m["switch_slow_ma"] = sw_slow
        results.append(m)
        count += 1
        if count % 200 == 0:
            print(f"  [Switch] {count}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════
# 两阶段参数扫描 — 先优化 Grid 参数，再优化 Switch 参数
#
# 策略：先用 MA Grid 策略扫描找到最优的网格参数组合，
# 然后固定这些网格参数，扫描 Switch 模式的相关参数。
# 这样可以大幅减少组合数（从 O(G×S) 降到 O(G+S)）。
# ══════════════════════════════════════════════════════════════════

_SWITCH_PARAMS = dict(
    flat_wait_days_vals=[5, 8, 15],
    switch_m1_vals=[0.02, 0.03, 0.05],
    switch_m2_vals=[0.005, 0.01, 0.02],
    switch_trailing_stop_vals=[0.02, 0.03, 0.05, 0.07, 0.10],
    switch_fast_vals=[5, 20],
    switch_slow_vals=[10, 60],
)


def scan_switch_two_stage(close: pd.Series,
                          open_: pd.Series | None = None) -> pd.DataFrame:
    """两阶段扫描：先复用 MA Grid 最优参数，再优化 Switch 逻辑。

    Stage 1: 运行 MA Grid 扫描（scan_ma_strategy），找到最优网格参数
    Stage 2: 固定最优网格参数，扫描 switch 相关的入场/出场条件参数

    这种分层优化策略牺牲了全局最优性，但大幅降低了计算量，
    且 Grid 和 Switch 参数的交互效应通常较弱。

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    from .ma_grid import scan_ma_strategy

    # 准备 numpy 版本的 Open 数据
    open_arr = open_.values if open_ is not None else None

    # ── Stage 1: 复用 MA Grid 扫描结果 ──
    print("  [Switch Stage 1] Reusing MA Grid optimal params…")
    ma_results = scan_ma_strategy(close, open_=open_)

    if ma_results.empty:
        return pd.DataFrame()

    # 选取 Grid 扫描中收益最高的参数
    best_grid = ma_results.nlargest(1, "total_return").iloc[0]
    best_w = int(best_grid["ma_window"])
    best_bgp = best_grid["base_grid_pct"]
    best_vs = best_grid["volatility_scale"]
    best_ts = best_grid["trend_sensitivity"]
    best_tpg = best_grid["take_profit_grid"]
    best_slg = best_grid["stop_loss_grid"]

    print(f"  [Switch Stage 1] MA Grid best: w={best_w} bgp={best_bgp:.4f} "
          f"vs={best_vs:.1f} ts={best_ts:.0f} tpg={best_tpg:.2f} slg={best_slg:.1f} "
          f"ret={best_grid['total_return']:.1%}")

    # ── Stage 2: 固定 Grid 参数，扫描 Switch 参数 ──
    print("  [Switch Stage 2] Optimizing switch params (grid fixed)…")

    use_gpu = gpu()["cupy_available"]
    all_ma_windows = sorted(set(_SWITCH_PARAMS["switch_fast_vals"]
                                + _SWITCH_PARAMS["switch_slow_vals"]))
    ma_to_idx = {w: i for i, w in enumerate(all_ma_windows)}

    # 构建 Switch 参数组合（过滤无效组合）
    switch_combos = []
    for fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow in product(
        _SWITCH_PARAMS["flat_wait_days_vals"],
        _SWITCH_PARAMS["switch_m1_vals"],
        _SWITCH_PARAMS["switch_m2_vals"],
        _SWITCH_PARAMS["switch_trailing_stop_vals"],
        _SWITCH_PARAMS["switch_fast_vals"],
        _SWITCH_PARAMS["switch_slow_vals"],
    ):
        if sw_m2 >= sw_m1 or sw_fast >= sw_slow:
            continue
        switch_combos.append((fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow))

    n_switch = len(switch_combos)
    # 所有组合共享同一组 Grid 参数
    indicators = compute_ma_switch_indicators(close, best_w, ma_windows=all_ma_windows)
    close_arr = close.values
    ma_all = np.array([indicators[f"MA{mw}"].values for mw in all_ma_windows])

    bgp_a = np.full(n_switch, best_bgp)
    vs_a = np.full(n_switch, best_vs)
    ts_a = np.full(n_switch, best_ts)
    tpg_a = np.full(n_switch, best_tpg)
    slg_a = np.full(n_switch, best_slg)
    fw_a = np.array([c[0] for c in switch_combos], dtype=np.int32)
    m1_a = np.array([c[1] for c in switch_combos])
    m2_a = np.array([c[2] for c in switch_combos])
    tr_a = np.array([c[3] for c in switch_combos])
    fi_a = np.array([ma_to_idx[c[4]] for c in switch_combos], dtype=np.int32)
    si_a = np.array([ma_to_idx[c[5]] for c in switch_combos], dtype=np.int32)

    results = []

    if use_gpu:
        # GPU 路径：批量生成信号 + 批量回测
        entries_b, exits_b, sizes_b = generate_switch_signals_batch(
            close_arr,
            indicators["MADevPct"].values,
            indicators["MADevTrend"].values,
            indicators["RollingVolPct"].values,
            indicators["MABase"].values,
            ma_all, fi_a, si_a,
            bgp_a, vs_a, ts_a, tpg_a, slg_a,
            fw_a, m1_a, m2_a, tr_a,
            position_size=0.5, position_sizing_coef=30.0,
        )
        bt = run_backtest_batch(close_arr, entries_b, exits_b, sizes_b,
                                n_combos=n_switch, open_=open_arr)
        for idx, (fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in enumerate(switch_combos):
            if int(bt[idx][4]) == 0:  # 无交易 → 跳过
                continue
            results.append({
                "total_return": bt[idx][0], "sharpe_ratio": bt[idx][1],
                "max_drawdown": bt[idx][2], "calmar_ratio": bt[idx][3],
                "num_trades": int(bt[idx][4]), "win_rate": bt[idx][5],
                "ma_window": best_w,
                "base_grid_pct": best_bgp, "volatility_scale": best_vs,
                "trend_sensitivity": best_ts, "take_profit_grid": best_tpg,
                "stop_loss_grid": best_slg,
                "flat_wait_days": fw, "switch_deviation_m1": sw_m1,
                "switch_deviation_m2": sw_m2,
                "switch_trailing_stop": sw_tr,
                "switch_fast_ma": sw_fast, "switch_slow_ma": sw_slow,
            })
    else:
        # CPU 路径：逐组合生成信号 + 回测
        for (fw, sw_m1, sw_m2, sw_tr, sw_fast, sw_slow) in switch_combos:
            entries, exits, sizes = generate_switch_signals(
                close_arr,
                indicators["MADevPct"].values,
                indicators["MADevTrend"].values,
                indicators["RollingVolPct"].values,
                indicators["MABase"].values,
                indicators[f"MA{sw_fast}"].values,
                indicators[f"MA{sw_slow}"].values,
                base_grid_pct=best_bgp, volatility_scale=best_vs,
                trend_sensitivity=best_ts, take_profit_grid=best_tpg,
                stop_loss_grid=best_slg,
                flat_wait_days=fw, switch_deviation_m1=sw_m1,
                switch_deviation_m2=sw_m2,
                switch_trailing_stop=sw_tr,
                position_size=0.5, position_sizing_coef=30.0,
            )
            if entries.sum() == 0:
                continue
            m = run_backtest(close, entries, exits, sizes, open_=open_)
            m["ma_window"] = best_w
            m["base_grid_pct"] = best_bgp
            m["volatility_scale"] = best_vs
            m["trend_sensitivity"] = best_ts
            m["take_profit_grid"] = best_tpg
            m["stop_loss_grid"] = best_slg
            m["flat_wait_days"] = fw
            m["switch_deviation_m1"] = sw_m1
            m["switch_deviation_m2"] = sw_m2
            m["switch_fast_ma"] = sw_fast
            m["switch_slow_ma"] = sw_slow
            results.append(m)

    return pd.DataFrame(results)
