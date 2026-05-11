"""
分析：Switch 入场前一日下跌过滤的效果。

规则：如果 bar i 收盘价 < bar i-1 收盘价（即刚收盘的这根 bar 是跌的），
则 bar i+1 开盘不产生 Switch 买入信号。

对比：有过滤 vs 无过滤的 Switch 信号差异，重点关注：
  1. 被阻拦的买入数量及占比
  2. 被阻拦买入如果执行，亏本的比例
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import (
    compute_polyfit_switch_indicators,
    compute_polyfit_base_only,
    add_trend_vol_indicators,
)
from utils.backtest import run_backtest
from utils.scoring import select_balanced
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import scan_polyfit_grid, generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals


# ══════════════════════════════════════════════════════════════════
# 修改版 Switch 信号生成 — 支持下一天过滤 + 阻拦记录
# ══════════════════════════════════════════════════════════════════

def generate_switch_with_down_day_filter(
    close: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    grid_entries: np.ndarray,
    grid_exits: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    trend_entry_dp: float = 0.01,
    trend_confirm_dp_slope: float = 0.0003,
    trend_atr_mult: float = 2.5,
    trend_atr_window: int = 14,
    trend_vol_climax: float = 3.0,
    trend_decline_days: int = 3,
    enable_down_day_filter: bool = True,
) -> dict:
    """生成 Grid-priority Switch 信号，可选下一天过滤。

    Returns:
        dict with:
          - sw_entries, sw_exits, sw_sizes: Switch 信号数组
          - blocked_entries: 被阻拦的入场 bar 索引
          - blocked_simulations: 每个被阻拦入场的模拟交易信息
    """
    n = len(close)
    sw_entries = np.zeros(n, dtype=bool)
    sw_exits = np.zeros(n, dtype=bool)
    sw_sizes = np.ones(n) * 0.99

    # Pre-compute ATR
    _use_atr = high is not None and low is not None
    atr_arr = np.zeros(n, dtype=np.float64)
    if _use_atr:
        alpha_a = 2.0 / (trend_atr_window + 1)
        atr_ema = 0.0
        for i in range(1, n):
            if (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])
                or np.isnan(close[i-1])):
                continue
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            atr_ema = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema
            atr_arr[i] = atr_ema

    # Pre-compute vol EMA
    _use_vol = volume is not None
    vol_ema_arr = np.zeros(n, dtype=np.float64)
    if _use_vol:
        alpha_v = 2.0 / 21.0
        v_ema = float(volume[0]) if not np.isnan(volume[0]) else 0.0
        vol_ema_arr[0] = v_ema
        for i in range(1, n):
            if not np.isnan(volume[i]):
                v_ema = alpha_v * float(volume[i]) + (1.0 - alpha_v) * v_ema
            vol_ema_arr[i] = v_ema

    grid_in = False
    sw_in = False
    sw_peak = 0.0
    prev_dp = np.nan
    decline_count = 0
    prev_ma20 = 0.0
    prev_ma60 = 0.0

    # 记录被阻拦的入场
    blocked_entries = []      # list of bar indices
    blocked_reasons = []      # reason string

    for i in range(n):
        dp = dev_pct[i]; dt = dev_trend[i]; cl = close[i]
        m20 = ma20[i]; m60 = ma60[i]
        if np.isnan(dp) or np.isnan(dt) or np.isnan(cl) or cl <= 0:
            prev_ma20 = m20; prev_ma60 = m60; continue

        # Grid exits
        grid_just_exited = False
        if grid_in and grid_exits[i]:
            grid_in = False
            grid_just_exited = True

        # Grid entry = force Switch exit
        if grid_entries[i]:
            if sw_in:
                sw_exits[i] = True
                sw_in = False
                sw_peak = 0.0
            grid_in = True
            prev_ma20 = m20; prev_ma60 = m60
            continue

        # Only if Grid idle
        if not grid_in and not grid_just_exited:
            if not sw_in:
                # Check Switch entry: dp above threshold + momentum or MA
                if dp >= trend_entry_dp:
                    cond_s = dt > trend_confirm_dp_slope
                    cond_m = (not np.isnan(m20) and not np.isnan(m60) and m20 > m60)

                    if cond_s or cond_m:
                        # ── 下一天过滤 ──
                        if enable_down_day_filter and i >= 1:
                            if close[i] < close[i-1]:
                                blocked_entries.append(i)
                                blocked_reasons.append(
                                    f"close[{i}]={close[i]:.4f} < close[{i-1}]={close[i-1]:.4f}"
                                )
                                # 不生成入场信号，继续
                                prev_ma20 = m20; prev_ma60 = m60
                                continue

                        sw_entries[i] = True
                        sw_in = True
                        sw_peak = cl
                        decline_count = 0
            else:
                # Switch exit checks
                exit_now = False

                if trend_decline_days > 0 and decline_count >= trend_decline_days:
                    exit_now = True
                elif _use_atr and atr_arr[i] > 0:
                    if cl <= sw_peak - trend_atr_mult * atr_arr[i]:
                        exit_now = True
                elif _use_vol:
                    rv = volume[i] / max(vol_ema_arr[i], 1e-9)
                    if rv > trend_vol_climax and i > 0 and cl > close[i-1]:
                        exit_now = True
                elif ma20 is not None and ma60 is not None:
                    if (not np.isnan(m20) and not np.isnan(m60)
                        and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
                        and m20 < m60 and prev_ma20 >= prev_ma60):
                        exit_now = True

                if exit_now:
                    sw_exits[i] = True
                    sw_in = False
                    sw_peak = 0.0
                else:
                    sw_peak = max(sw_peak, cl)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp):
            if dp < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp
        prev_ma20 = m20; prev_ma60 = m60

    # Force exit at end
    if sw_in:
        sw_exits[-1] = True

    return {
        "sw_entries": sw_entries,
        "sw_exits": sw_exits,
        "sw_sizes": sw_sizes,
        "blocked_entries": np.array(blocked_entries, dtype=int),
        "blocked_reasons": blocked_reasons,
    }


# ══════════════════════════════════════════════════════════════════
# 模拟被阻拦交易的离场 — 用与 Switch 相同的离场逻辑
# ══════════════════════════════════════════════════════════════════

def simulate_blocked_trade_exit(
    entry_bar: int,
    close: np.ndarray,
    high: np.ndarray | None,
    low: np.ndarray | None,
    volume: np.ndarray | None,
    dev_pct: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    atr_arr: np.ndarray,
    vol_ema_arr: np.ndarray,
    trend_atr_mult: float = 2.5,
    trend_vol_climax: float = 3.0,
    trend_decline_days: int = 3,
    grid_entries: np.ndarray | None = None,
) -> dict | None:
    """模拟一笔被阻拦的 Switch 交易的完整生命周期。

    从 entry_bar 开始，用 Switch 离场规则向前扫描直至触发离场或数据结束。
    Grid 入场会强制 Switch 离场（Grid-priority 规则）。

    Returns:
        dict: {entry_bar, exit_bar, entry_price, exit_price, pnl, exit_reason}
        如果找不到离场则返回 None
    """
    n = len(close)
    sw_peak = close[entry_bar]
    decline_count = 0
    prev_dp = np.nan
    prev_ma20 = ma20[entry_bar]
    prev_ma60 = ma60[entry_bar]
    exit_bar = -1
    exit_reason = "unknown"

    for i in range(entry_bar + 1, n):
        cl = close[i]; dp_v = dev_pct[i]
        m20 = ma20[i]; m60 = ma60[i]

        if np.isnan(cl) or cl <= 0 or np.isnan(dp_v):
            continue

        # Grid entry forces Switch exit
        if grid_entries is not None and grid_entries[i]:
            exit_bar = i
            exit_reason = "grid_entry_force"
            break

        exit_now = False

        # dp decline
        if trend_decline_days > 0 and decline_count >= trend_decline_days:
            exit_bar = i
            exit_reason = "dp_decline"
            break

        # ATR trail
        if atr_arr[i] > 0 and cl <= sw_peak - trend_atr_mult * atr_arr[i]:
            exit_bar = i
            exit_reason = "atr_trail"
            break

        # Volume climax
        if volume is not None and vol_ema_arr[i] > 0 and i > 0:
            rv = volume[i] / max(vol_ema_arr[i], 1e-9)
            if rv > trend_vol_climax and cl > close[i-1]:
                exit_bar = i
                exit_reason = "vol_climax"
                break

        # MA death cross
        if (not np.isnan(m20) and not np.isnan(m60)
            and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
            and m20 < m60 and prev_ma20 >= prev_ma60):
            exit_bar = i
            exit_reason = "ma_death_cross"
            break

        sw_peak = max(sw_peak, cl)

        # dp decline tracking
        if not np.isnan(prev_dp) and not np.isnan(dp_v):
            if dp_v < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp_v
        prev_ma20 = m20; prev_ma60 = m60

    if exit_bar < 0:
        # 未找到自然离场 → 末尾强制平仓
        exit_bar = n - 1
        exit_reason = "force_close"

    # 成交价模型：次日开盘价
    entry_price = close[entry_bar]  # 用收盘价近似（open 通常接近 close）
    if exit_bar + 1 < n:
        exit_price = close[exit_bar]  # 近似
    else:
        exit_price = close[exit_bar]

    if entry_price <= 0:
        return None

    pnl = (exit_price - entry_price) / entry_price

    return {
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "exit_reason": exit_reason,
        "holding_days": exit_bar - entry_bar,
    }


# ══════════════════════════════════════════════════════════════════
# 主分析 — 全量数据上跨 WF 窗口聚合
# ══════════════════════════════════════════════════════════════════

# 固定 Grid 参数（基于项目最优：tw=10, vw=20）
GRID_PARAMS = dict(
    base_grid_pct=0.008, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=1.0, stop_loss_grid=1.2,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=0.45, position_size=0.92, position_sizing_coef=30,
)

SWITCH_PARAMS = dict(
    trend_entry_dp=0.0,          # 设 0 以获取更多 Switch 入场样本
    trend_confirm_dp_slope=0.0003,
    trend_atr_mult=2.5,
    trend_atr_window=14,
    trend_vol_climax=3.0,
    trend_decline_days=3,
)


def analyze_single_window(close_warmup_all, open_warmup_all, high, low, volume,
                          test_offset, tw=10, vw=20):
    """在单个 WF 测试窗口上运行过滤分析，返回统计。"""
    test_start_label = close_warmup_all.index[test_offset]
    ind_s = compute_polyfit_switch_indicators(
        close_warmup_all, fit_window_days=252, ma_windows=[20, 60],
        trend_window_days=tw, vol_window_days=vw,
    )
    ind_test = ind_s.loc[ind_s.index >= test_start_label]
    if len(ind_test) < 10:
        return None

    cl_test = close_warmup_all.loc[ind_test.index]
    op_test = open_warmup_all.loc[ind_test.index]

    cl_arr = cl_test.values
    dev_pct_arr = ind_test["PolyDevPct"].values
    dev_trend_arr = ind_test["PolyDevTrend"].values
    vol_pct_arr = ind_test["RollingVolPct"].values
    poly_base_arr = ind_test["PolyBasePred"].values
    ma20_arr = ind_test["MA20"].values
    ma60_arr = ind_test["MA60"].values
    h_arr = high.reindex(ind_test.index).values
    l_arr = low.reindex(ind_test.index).values
    v_arr = volume.reindex(ind_test.index).values

    # Grid 信号
    e_grid, x_grid, s_grid = generate_grid_signals(
        cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
        **GRID_PARAMS,
    )

    # Switch 信号：无过滤 + 有过滤
    r_no = generate_switch_with_down_day_filter(
        cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
        e_grid, x_grid, ma20_arr, ma60_arr,
        high=h_arr, low=l_arr, volume=v_arr,
        enable_down_day_filter=False, **SWITCH_PARAMS,
    )
    r_with = generate_switch_with_down_day_filter(
        cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
        e_grid, x_grid, ma20_arr, ma60_arr,
        high=h_arr, low=l_arr, volume=v_arr,
        enable_down_day_filter=True, **SWITCH_PARAMS,
    )

    e_sw_no = r_no["sw_entries"]
    e_sw_with = r_with["sw_entries"]
    blocked_bars = r_with["blocked_entries"]

    n_no = int(e_sw_no.sum())
    n_with = int(e_sw_with.sum())
    n_blocked = len(blocked_bars)

    # 模拟被阻拦交易
    n_test = len(cl_arr)
    atr_arr = np.zeros(n_test, dtype=np.float64)
    alpha_a = 2.0 / 15.0
    atr_ema_v = 0.0
    for i in range(1, n_test):
        tr = max(h_arr[i] - l_arr[i], abs(h_arr[i] - cl_arr[i-1]),
                 abs(l_arr[i] - cl_arr[i-1]))
        atr_ema_v = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema_v
        atr_arr[i] = atr_ema_v

    vol_ema_arr = np.zeros(n_test, dtype=np.float64)
    alpha_v = 2.0 / 21.0
    v_ema = float(v_arr[0]) if not np.isnan(v_arr[0]) else 0.0
    vol_ema_arr[0] = v_ema
    for i in range(1, n_test):
        if not np.isnan(v_arr[i]):
            v_ema = alpha_v * float(v_arr[i]) + (1.0 - alpha_v) * v_ema
        vol_ema_arr[i] = v_ema

    sims = []
    for bar in blocked_bars:
        sim = simulate_blocked_trade_exit(
            bar, cl_arr,
            high=h_arr, low=l_arr, volume=v_arr,
            dev_pct=dev_pct_arr, ma20=ma20_arr, ma60=ma60_arr,
            atr_arr=atr_arr, vol_ema_arr=vol_ema_arr,
            grid_entries=e_grid,
            trend_atr_mult=SWITCH_PARAMS["trend_atr_mult"],
            trend_vol_climax=SWITCH_PARAMS["trend_vol_climax"],
            trend_decline_days=SWITCH_PARAMS["trend_decline_days"],
        )
        if sim is not None:
            sims.append(sim)

    # 整体回测对比
    e_merged_no = e_grid | e_sw_no
    x_merged_no = x_grid | r_no["sw_exits"]
    s_merged_no = np.where(e_grid, s_grid, np.where(e_sw_no, 0.99, 0.0))

    e_merged_with = e_grid | e_sw_with
    x_merged_with = x_grid | r_with["sw_exits"]
    s_merged_with = np.where(e_grid, s_grid, np.where(e_sw_with, 0.99, 0.0))

    m_no = run_backtest(
        pd.Series(cl_arr, index=ind_test.index),
        e_merged_no, x_merged_no, s_merged_no, open_=op_test,
    )
    m_with = run_backtest(
        pd.Series(cl_arr, index=ind_test.index),
        e_merged_with, x_merged_with, s_merged_with, open_=op_test,
    )

    return {
        "n_switch_no_filter": n_no,
        "n_switch_with_filter": n_with,
        "n_blocked": n_blocked,
        "simulations": sims,
        "m_no": m_no,
        "m_with": m_with,
        "test_start": test_start_label,
        "test_end": cl_test.index[-1],
        "n_bars": n_test,
    }


def analyze_down_day_filter():
    print("=" * 80)
    print("  Switch 下一天过滤分析")
    print("  规则：close[i] < close[i-1] → bar i+1 Open 不产生 Switch 买入")
    print("=" * 80)

    data = load_data("data/1d/512890.SH_hfq.parquet")
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]
    print(f"\n数据: {len(data)} bars, {data.index[0].date()} → {data.index[-1].date()}")

    # 跨所有 WF 窗口聚合
    windows = generate_monthly_windows(
        close.index, train_months=22, test_months=12,
        step_months=3, warmup_months=12,
    )
    print(f"WF 窗口: {len(windows)} 个 (步长3月)")

    all_sims = []
    total_blocked = 0
    total_switch_no = 0
    total_switch_with = 0
    agg_m_no = {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0, "num_trades": 0, "win_rate": 0}
    agg_m_with = {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0, "num_trades": 0, "win_rate": 0}
    n_analyzed = 0

    for wi, w in enumerate(windows):
        close_warmup_all = close.iloc[w.warmup_start:w.test_end]
        open_warmup_all = open_.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start

        result = analyze_single_window(
            close_warmup_all, open_warmup_all, high, low, volume,
            test_offset, tw=10, vw=20,
        )
        if result is None:
            continue

        total_blocked += result["n_blocked"]
        total_switch_no += result["n_switch_no_filter"]
        total_switch_with += result["n_switch_with_filter"]
        all_sims.extend(result["simulations"])
        for k in agg_m_no:
            agg_m_no[k] += result["m_no"][k]
            agg_m_with[k] += result["m_with"][k]
        n_analyzed += 1

        if (wi + 1) % 5 == 0 or wi == 0:
            print(f"  [{wi+1}/{len(windows)}] 累计阻拦: {total_blocked}  "
                  f"累计Switch无过滤: {total_switch_no}  "
                  f"模拟交易: {len(all_sims)}")

    print(f"\n  共分析 {n_analyzed} 个窗口")

    if not all_sims:
        print("  没有被阻拦的交易，无法分析。")
        # 检查为何没有阻拦：看无过滤 Switch 入场有多少
        print(f"  所有窗口 Switch 入场(无过滤): {total_switch_no}")
        print(f"  所有窗口 Switch 入场(有过滤): {total_switch_with}")
        print(f"  被阻拦: {total_blocked}")
        return

    # ── 聚合被阻拦交易盈亏 ──
    pnls = np.array([s["pnl"] for s in all_sims])
    n_lose = int((pnls < 0).sum())
    n_win = int((pnls > 0).sum())
    n_even = int((pnls == 0).sum())
    lose_ratio = n_lose / len(pnls)

    from collections import Counter
    reason_counts = Counter(s["exit_reason"] for s in all_sims)
    reason_lose = Counter(s["exit_reason"] for s in all_sims if s["pnl"] < 0)

    print(f"\n{'═' * 70}")
    print(f"  被阻拦交易盈亏分析（{n_analyzed} 个 WF 窗口聚合）")
    print(f"{'═' * 70}")
    print(f"  无过滤 Switch 入场总数: {total_switch_no}")
    print(f"  有过滤 Switch 入场总数: {total_switch_with}")
    print(f"  被阻拦总数:             {total_blocked}  ({total_blocked/max(total_switch_no,1)*100:.1f}%)")
    print(f"  成功模拟:               {len(all_sims)}")
    print(f"")
    print(f"  盈利笔数:   {n_win}  ({n_win/len(pnls)*100:.1f}%)")
    print(f"  亏本笔数:   {n_lose}  ({lose_ratio*100:.1f}%)  ★")
    print(f"  持平笔数:   {n_even}  ({n_even/len(pnls)*100:.1f}%)")
    print(f"")
    print(f"  平均盈亏:   {pnls.mean():+.2%}")
    print(f"  中位盈亏:   {np.median(pnls):+.2%}")
    print(f"  最大盈利:   {pnls.max():+.2%}")
    print(f"  最大亏损:   {pnls.min():+.2%}")
    print(f"  盈亏标准差: {pnls.std():.2%}")
    print(f"  平均持仓:   {np.mean([s['holding_days'] for s in all_sims]):.1f} 天")

    print(f"\n  ── 离场原因分布 ──")
    print(f"  {'原因':<20s} {'总数':>8s} {'亏本':>8s} {'亏本率':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    for reason, count in reason_counts.most_common():
        lose_n = reason_lose.get(reason, 0)
        print(f"  {reason:<20s} {count:>8d} {lose_n:>8d} {lose_n/count*100:>7.1f}%")

    # ── 盈亏分布直方图（文本版）──
    print(f"\n  ── 盈亏分布 ──")
    bins = [(-0.15, -0.10), (-0.10, -0.05), (-0.05, -0.02), (-0.02, 0),
            (0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20)]
    max_count = 0
    counts = []
    for lo, hi in bins:
        c = int(((pnls >= lo) & (pnls < hi)).sum())
        counts.append(c)
        max_count = max(max_count, c)
    for (lo, hi), c in zip(bins, counts):
        bar = "█" * max(1, int(c / max(max_count, 1) * 40))
        print(f"  [{lo:>+5.0%} ~ {hi:>+5.0%}): {bar} {c}")

    # ── 整体回测影响（平均每窗口） ──
    print(f"\n{'═' * 70}")
    print(f"  整体回测影响（{n_analyzed} 窗口平均）")
    print(f"{'═' * 70}")
    for k in agg_m_no:
        agg_m_no[k] /= n_analyzed
        agg_m_with[k] /= n_analyzed

    for label, key, fmt in [
        ("总收益/窗口", "total_return", "+.2%"),
        ("Sharpe", "sharpe_ratio", ".3f"),
        ("最大回撤", "max_drawdown", "+.2%"),
        ("交易次数", "num_trades", ".1f"),
        ("胜率", "win_rate", ".1%"),
    ]:
        a, b = agg_m_no[key], agg_m_with[key]
        d = b - a
        print(f"  {label:>14s}: 无过滤 {a:{fmt}}  有过滤 {b:{fmt}}  变化 {d:{fmt}}")

    print(f"\n  ★ 结论：下一天过滤阻拦了 {total_blocked} 笔 Switch 入场，")
    print(f"    其中 {n_lose}/{len(pnls)} = {lose_ratio*100:.1f}% 如果执行会亏本。")
    print(f"    平均每窗口阻拦 {total_blocked/n_analyzed:.1f} 笔。")


if __name__ == "__main__":
    analyze_down_day_filter()
