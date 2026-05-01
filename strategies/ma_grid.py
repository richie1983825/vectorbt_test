"""Moving Average Dynamic Grid Strategy — 均线动态网格策略。

核心思想：以 SMA 均线为基准线，当价格偏离均线超出动态阈值时入场（均值回复），
当价格回到止盈/止损区域时离场。

信号逻辑：
  - 入场：价格低于均线，且偏离程度（MADevPct）达到动态网格步长的整数倍
  - 出场：持仓达到最大持有天数，或偏离回归到止盈网格线以上，或跌破止损网格线
  - 网格步长随波动率和趋势强度动态调整（高波动 → 宽网格，强趋势 → 宽网格）

GPU 加速：通过 CuPy RawKernel 将多组参数组合的信号生成一次性在 GPU 上完成。
"""

from typing import Tuple

import numpy as np
import pandas as pd

from utils.gpu import xp
from utils.indicators import compute_ma_indicators
from utils.scan import indicator_and_scan


# ══════════════════════════════════════════════════════════════════
# CPU 信号生成 — 逐 bar 循环的简单实现
# ══════════════════════════════════════════════════════════════════

def generate_grid_signals(
    close: np.ndarray,
    dev_pct: np.ndarray,        # (close - MA) / MA，价格偏离均线的百分比
    dev_trend: np.ndarray,      # dev_pct 的短期斜率，反映趋势方向
    rolling_vol_pct: np.ndarray, # 滚动波动率（收益率的标准差）
    base_grid_pct: float = 0.012,     # 基础网格步长（如 1.2%）
    volatility_scale: float = 1.0,    # 波动率放大系数
    trend_sensitivity: float = 8.0,   # 趋势敏感度（趋势越强，网格越宽）
    max_grid_levels: int = 3,         # 最大网格层级（对应最多 3 倍步长的偏离）
    take_profit_grid: float = 0.85,   # 止盈：偏离回补到 entry_level * grid_step * 0.85
    stop_loss_grid: float = 1.6,      # 止损：偏离扩大到 entry_level * grid_step * 1.6
    max_holding_days: int = 30,       # 最大持仓天数
    cooldown_days: int = 1,           # 出场后的冷却期
    min_signal_strength: float = 0.45,# 最小信号强度（避免弱信号入场）
    position_size: float = np.inf,    # 单次仓位比例（inf = 全仓）
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成均线网格策略的入场/离场信号。

    返回值：
      entries: bool 数组 [n]，入场信号
      exits:   bool 数组 [n]，离场信号
      sizes:   float 数组 [n]，每次入场的仓位比例
    """
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sizes = np.zeros(n)

    in_position = False
    entry_bar = -1          # 入场 bar 索引
    entry_level = 1         # 入场网格层级（1~3）
    entry_grid_step = np.nan # 入场时的动态网格步长（用于出场阈值计算）
    cooldown = 0            # 剩余冷却天数

    for i in range(n):
        # 跳过无效数据（NaN 或价格为 0）
        if (np.isnan(dev_pct[i]) or np.isnan(dev_trend[i])
                or np.isnan(rolling_vol_pct[i]) or close[i] <= 0):
            continue

        # 动态网格步长 = 基础步长 × 趋势放大 × 波动放大
        # 趋势越强或波动越大 → 网格越宽 → 入场条件更严格
        vol_mult = 1.0 + volatility_scale * max(rolling_vol_pct[i], 0.0)
        dynamic_grid_step = (
            base_grid_pct
            * (1.0 + trend_sensitivity * abs(dev_trend[i]))
            * vol_mult
        )
        # 保证网格步长不低于基础值的 30%，避免极端情况下网格过密
        dynamic_grid_step = max(dynamic_grid_step, base_grid_pct * 0.3)

        if not in_position:
            if cooldown > 0:
                cooldown -= 1
                continue
            # 信号强度 = |偏离| / 动态网格步长，强度越高可以进入更深的网格层级
            signal_strength = abs(dev_pct[i]) / max(dynamic_grid_step, 1e-9)
            entry_lvl = int(np.clip(np.floor(signal_strength), 1, max_grid_levels))
            # 入场阈值：偏离必须超过 entry_lvl 倍网格步长（负方向）
            entry_threshold = -entry_lvl * dynamic_grid_step
            if dev_pct[i] <= entry_threshold and signal_strength >= min_signal_strength:
                entries[i] = True
                sizes[i] = position_size
                in_position = True
                entry_bar = i
                entry_level = entry_lvl
                entry_grid_step = dynamic_grid_step  # 冻结入场时的步长
        else:
            # 持仓中：检查出场条件
            holding_days = i - entry_bar
            hold_limit = holding_days >= max_holding_days
            # 参考步长取入场步长和当前步长中的较大值，避免出场条件过于宽松
            ref_step = (max(dynamic_grid_step, entry_grid_step)
                        if not np.isnan(entry_grid_step) else dynamic_grid_step)
            tp_threshold = entry_level * ref_step * take_profit_grid
            sl_threshold = entry_level * ref_step * stop_loss_grid
            # 三种出场条件：超时、止盈（偏离回补到 tp 以上）、止损（偏离继续扩大到 sl 以下）
            if hold_limit or dev_pct[i] >= tp_threshold or dev_pct[i] <= -sl_threshold:
                exits[i] = True
                in_position = False
                cooldown = cooldown_days

    # 最后一个 bar 强制平仓
    if in_position:
        exits[-1] = True

    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# GPU 批量信号生成 — CuPy RawKernel
#
# 将多组参数（base_grid_pct, volatility_scale, trend_sensitivity,
# take_profit_grid, stop_loss_grid）一次性传入 CUDA kernel，
# 每个 CUDA 线程独立处理一组参数组合，并行生成信号。
# ══════════════════════════════════════════════════════════════════

# lazy-init：首次调用时才编译 CUDA kernel
_generate_grid_signals_kernel = None

_GRID_KERNEL_CODE = r"""
extern "C" __global__ void generate_grid_signals_kernel(
    const double* dev_pct,
    const double* dev_trend,
    const double* vol_pct,
    const double* base_grid_pct,      // [n_combos]
    const double* volatility_scale,   // [n_combos]
    const double* trend_sensitivity,  // [n_combos]
    const double* take_profit_grid,   // [n_combos]
    const double* stop_loss_grid,     // [n_combos]
    bool* entries,                     // [n_combos * n_bars]
    bool* exits,                       // [n_combos * n_bars]
    double* sizes,                     // [n_combos * n_bars]
    int n_bars,
    int n_combos,
    int max_grid_levels,
    int max_holding_days,
    int cooldown_days,
    double min_signal_strength,
    double pos_size
) {
    // 每个线程处理一组参数组合
    int combo_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_idx >= n_combos) return;

    // 读取当前组合的参数
    double bgp = base_grid_pct[combo_idx];
    double vs = volatility_scale[combo_idx];
    double ts = trend_sensitivity[combo_idx];
    double tpg = take_profit_grid[combo_idx];
    double slg = stop_loss_grid[combo_idx];

    // 逐 bar 状态机（逻辑与 CPU 版本完全一致）
    bool in_position = false;
    int entry_bar = -1;
    int entry_level = 1;
    double entry_grid_step = -1.0;
    int cooldown = 0;

    for (int i = 0; i < n_bars; i++) {
        double dp = dev_pct[i];
        double dt = dev_trend[i];
        double vp = vol_pct[i];

        if (isnan(dp) || isnan(dt) || isnan(vp)) {
            entries[combo_idx * n_bars + i] = false;
            exits[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
            continue;
        }

        double vol_mult = 1.0 + vs * fmax(vp, 0.0);
        double dgs = bgp * (1.0 + ts * fabs(dt)) * vol_mult;
        dgs = fmax(dgs, bgp * 0.3);

        if (!in_position) {
            if (cooldown > 0) {
                cooldown--;
                entries[combo_idx * n_bars + i] = false;
                exits[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
                continue;
            }
            double signal_strength = fabs(dp) / fmax(dgs, 1e-9);
            int entry_lvl = (int)floor(signal_strength);
            entry_lvl = entry_lvl < 1 ? 1 : (entry_lvl > max_grid_levels ? max_grid_levels : entry_lvl);
            double entry_threshold = -(double)entry_lvl * dgs;
            if (dp <= entry_threshold && signal_strength >= min_signal_strength) {
                entries[combo_idx * n_bars + i] = true;
                sizes[combo_idx * n_bars + i] = pos_size;
                in_position = true;
                entry_bar = i;
                entry_level = entry_lvl;
                entry_grid_step = dgs;
            } else {
                entries[combo_idx * n_bars + i] = false;
                sizes[combo_idx * n_bars + i] = 0.0;
            }
            exits[combo_idx * n_bars + i] = false;
        } else {
            entries[combo_idx * n_bars + i] = false;
            sizes[combo_idx * n_bars + i] = 0.0;
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
            } else {
                exits[combo_idx * n_bars + i] = false;
            }
        }
    }

    // 末尾强制平仓
    if (in_position) {
        exits[combo_idx * n_bars + n_bars - 1] = true;
    }
}
"""


def _get_grid_kernel():
    """延迟编译 CUDA kernel，避免在 CuPy 不可用时报错。"""
    global _generate_grid_signals_kernel
    if _generate_grid_signals_kernel is None:
        cp = xp()
        _generate_grid_signals_kernel = cp.RawKernel(
            _GRID_KERNEL_CODE, "generate_grid_signals_kernel")
    return _generate_grid_signals_kernel


def generate_grid_signals_batch(
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    base_grid_pcts: np.ndarray,      # [n_combos]
    volatility_scales: np.ndarray,   # [n_combos]
    trend_sensitivities: np.ndarray, # [n_combos]
    take_profit_grids: np.ndarray,   # [n_combos]
    stop_loss_grids: np.ndarray,     # [n_combos]
    max_grid_levels: int = 3,
    max_holding_days: int = 30,
    cooldown_days: int = 1,
    min_signal_strength: float = 0.45,
    position_size: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU 批量信号生成：一个 CUDA 线程处理一组参数。

    将参数数组 pad 到 256 的倍数以满足 CUDA block 对齐要求，
    然后将指标数据和参数传入 kernel 并行计算。

    Returns:
        (entries, exits, sizes) 均为 [n_combos, n_bars] 的 numpy 数组。
    """
    cp = xp()
    n_bars = len(dev_pct)
    n_combos = len(base_grid_pcts)

    # CUDA 线程配置：每个 block 256 线程，block 数量覆盖所有参数组合
    block_size = 256
    grid_size = (n_combos + block_size - 1) // block_size
    padded = grid_size * block_size

    def _pad(a):
        """将数组 pad 到 padded 长度，多余位置填 NaN 使其不产生信号。"""
        result = cp.zeros(padded, dtype=cp.float64)
        result[:n_combos] = cp.asarray(a, dtype=cp.float64)
        if padded > n_combos:
            result[n_combos:] = cp.nan
        return result

    # pad 各参数数组并上传到 GPU
    bgp_d = _pad(base_grid_pcts)
    vs_d = _pad(volatility_scales)
    ts_d = _pad(trend_sensitivities)
    tpg_d = _pad(take_profit_grids)
    slg_d = _pad(stop_loss_grids)

    # 上传指标数据到 GPU（所有组合共享）
    dev_pct_d = cp.asarray(dev_pct, dtype=cp.float64)
    dev_trend_d = cp.asarray(dev_trend, dtype=cp.float64)
    vol_d = cp.asarray(rolling_vol_pct, dtype=cp.float64)

    # 分配 GPU 输出缓冲区
    entries_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    exits_d = cp.zeros(padded * n_bars, dtype=cp.bool_)
    sizes_d = cp.zeros(padded * n_bars, dtype=cp.float64)

    pos_size = float(position_size)

    # 启动 CUDA kernel
    kernel = _get_grid_kernel()
    kernel(
        (grid_size,), (block_size,),
        (
            dev_pct_d, dev_trend_d, vol_d,
            bgp_d, vs_d, ts_d, tpg_d, slg_d,
            entries_d, exits_d, sizes_d,
            n_bars, n_combos, max_grid_levels, max_holding_days, cooldown_days,
            min_signal_strength, pos_size,
        ),
    )

    # 将结果从 GPU 取回并 reshape，仅保留前 n_combos 组有效数据
    entries = cp.asnumpy(entries_d).reshape(padded, n_bars)[:n_combos]
    exits = cp.asnumpy(exits_d).reshape(padded, n_bars)[:n_combos]
    sizes = cp.asnumpy(sizes_d).reshape(padded, n_bars)[:n_combos]
    return entries, exits, sizes


# ══════════════════════════════════════════════════════════════════
# 参数扫描入口
# ══════════════════════════════════════════════════════════════════

def scan_ma_strategy(close: pd.Series,
                     open_: pd.Series | None = None) -> pd.DataFrame:
    """MA 网格策略参数网格扫描。

    对 ma_window（MA 周期）、base_grid_pct（基础网格步长）、
    volatility_scale（波动率系数）、trend_sensitivity（趋势敏感度）、
    take_profit_grid（止盈倍数）、stop_loss_grid（止损倍数）
    进行全组合扫描，返回按 total_return 排序的结果。

    GPU 模式下，同一 MA 窗口下的所有参数组合会批量生成信号并批量回测。

    Args:
        open_: 可选，开盘价序列，传入则使用 next-bar Open 成交
    """
    return indicator_and_scan(
        close, "MA", compute_ma_indicators, "ma_window",
        windows=[20, 50, 75, 100, 150, 200],
        grid_pcts=[0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02],
        vol_scales=[0.5, 1.0, 1.5, 2.0, 3.0],
        trend_sens=[3.0, 6.0, 9.0, 12.0, 15.0],
        tpg_values=[0.5, 0.7, 0.85, 1.0],
        slg_values=[1.0, 1.3, 1.6, 2.0, 2.5],
        signal_fn=generate_grid_signals,
        signal_batch_fn=generate_grid_signals_batch,
        open_=open_,
    )
