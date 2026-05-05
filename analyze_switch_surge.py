"""
分析 Switch 入场前后的 OHLCV 特征：连续上涨 / 单日急涨。

对每笔 Switch 入场，提取入场前后 N 天的 OHLCV 数据，
按以下维度分类并寻找共性：
  1. 连续上涨天数（入场前连阳）
  2. 单日涨幅（入场当日涨跌幅）
  3. 量能特征（相对成交量）
  4. 价格位置（相对近期高低点）
  5. 盈亏结果
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
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals


# ══════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════

GRID_PARAMS = dict(
    base_grid_pct=0.008, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=1.0, stop_loss_grid=1.2,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=0.45, position_size=0.92, position_sizing_coef=30,
)

SWITCH_PARAMS = dict(
    trend_entry_dp=0.0,          # 设 0 获取更多样本
    trend_confirm_dp_slope=0.0003,
    trend_atr_mult=2.5,
    trend_atr_window=14,
    trend_vol_climax=3.0,
    trend_decline_days=3,
)

LOOKBACK = 10   # 入场前观察窗口
LOOKFWD = 15    # 入场后观察窗口


# ══════════════════════════════════════════════════════════════════
# 特征提取
# ══════════════════════════════════════════════════════════════════

def extract_entry_features(entry_bar, cl_arr, op_arr, hi_arr, lo_arr, vol_arr,
                           dev_pct_arr, ma20_arr, ma60_arr):
    """提取单笔 Switch 入场的 OHLCV 特征。"""
    n = len(cl_arr)

    # ── 入场前特征 ──
    pre_cl = cl_arr[max(0, entry_bar - LOOKBACK):entry_bar + 1]
    pre_vol = vol_arr[max(0, entry_bar - LOOKBACK):entry_bar + 1]

    # 连续上涨天数（入场前）
    consecutive_up = 0
    for j in range(entry_bar - 1, max(0, entry_bar - LOOKBACK) - 1, -1):
        if cl_arr[j] > cl_arr[j-1]:
            consecutive_up += 1
        else:
            break

    # 入场当日涨跌
    entry_day_ret = (cl_arr[entry_bar] / cl_arr[entry_bar-1] - 1) if entry_bar > 0 else 0

    # 入场前 N 日累计涨幅
    pre_n = min(LOOKBACK, entry_bar)
    pre_cum_ret = cl_arr[entry_bar] / cl_arr[entry_bar - pre_n] - 1 if pre_n > 0 else 0

    # 入场前最大单日涨幅
    pre_max_up = 0.0
    pre_max_up_day = -1
    for j in range(max(1, entry_bar - LOOKBACK), entry_bar + 1):
        r = cl_arr[j] / cl_arr[j-1] - 1
        if r > pre_max_up:
            pre_max_up = r
            pre_max_up_day = entry_bar - j

    # 入场前连阳最大涨幅
    pre_cons_up_ret = 0.0
    if consecutive_up > 0:
        start_bar = entry_bar - consecutive_up
        if start_bar >= 0:
            pre_cons_up_ret = cl_arr[entry_bar] / cl_arr[start_bar] - 1

    # 相对成交量（相对 20 日均量）
    vol_20_avg = np.mean(vol_arr[max(0, entry_bar-21):max(1, entry_bar-1)]) if entry_bar >= 21 else np.mean(vol_arr[:entry_bar])
    rel_vol = vol_arr[entry_bar] / vol_20_avg if vol_20_avg > 0 else 1.0

    # 成交量放大（入场日 / 前日均量）
    vol_ratio_1d = vol_arr[entry_bar] / vol_arr[entry_bar-1] if entry_bar > 0 and vol_arr[entry_bar-1] > 0 else 1.0

    # 价格位置：相对 N 日高低点
    pre_n_high = np.max(hi_arr[max(0, entry_bar - LOOKBACK):entry_bar + 1])
    pre_n_low = np.min(lo_arr[max(0, entry_bar - LOOKBACK):entry_bar + 1])
    price_position = ((cl_arr[entry_bar] - pre_n_low) / (pre_n_high - pre_n_low)
                      if pre_n_high > pre_n_low else 0.5)

    # 振幅 (High-Low 相对前收)
    amplitude = (hi_arr[entry_bar] - lo_arr[entry_bar]) / cl_arr[entry_bar-1] if entry_bar > 0 else 0

    # 上影线 / 下影线
    body = abs(cl_arr[entry_bar] - op_arr[entry_bar])
    upper_wick = hi_arr[entry_bar] - max(cl_arr[entry_bar], op_arr[entry_bar])
    lower_wick = min(cl_arr[entry_bar], op_arr[entry_bar]) - lo_arr[entry_bar]
    total_range = hi_arr[entry_bar] - lo_arr[entry_bar]
    upper_wick_pct = upper_wick / total_range if total_range > 0 else 0
    lower_wick_pct = lower_wick / total_range if total_range > 0 else 0

    # 实体相对大小
    body_pct = body / total_range if total_range > 0 else 0

    # 是阳线还是阴线
    is_green = cl_arr[entry_bar] >= op_arr[entry_bar]

    # dp 偏离度
    dp = dev_pct_arr[entry_bar]

    # MA 关系
    ma20_val = ma20_arr[entry_bar]
    ma60_val = ma60_arr[entry_bar]
    ma_diff = (ma20_val - ma60_val) / ma60_val if ma60_val > 0 else 0
    price_vs_ma20 = cl_arr[entry_bar] / ma20_val - 1 if ma20_val > 0 else 0
    price_vs_ma60 = cl_arr[entry_bar] / ma60_val - 1 if ma60_val > 0 else 0

    return {
        "entry_bar": entry_bar,
        "entry_close": cl_arr[entry_bar],
        "consecutive_up": consecutive_up,
        "entry_day_ret": entry_day_ret,
        "pre_cum_ret": pre_cum_ret,
        "pre_max_up": pre_max_up,
        "pre_max_up_days_ago": pre_max_up_day,
        "pre_cons_up_ret": pre_cons_up_ret,
        "rel_vol": rel_vol,
        "vol_ratio_1d": vol_ratio_1d,
        "price_position": price_position,
        "amplitude": amplitude,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "body_pct": body_pct,
        "is_green": is_green,
        "dp": dp,
        "ma_diff": ma_diff,
        "price_vs_ma20": price_vs_ma20,
        "price_vs_ma60": price_vs_ma60,
    }


def simulate_switch_trade(entry_bar, cl_arr, hi_arr, lo_arr, vol_arr,
                          dev_pct_arr, ma20_arr, ma60_arr, e_grid):
    """模拟单笔 Switch 交易的离场和盈亏。"""
    n = len(cl_arr)
    atr_arr = np.zeros(n, dtype=np.float64)
    alpha_a = 2.0 / 15.0
    atr_ema = 0.0
    for i in range(1, n):
        tr = max(hi_arr[i] - lo_arr[i], abs(hi_arr[i] - cl_arr[i-1]),
                 abs(lo_arr[i] - cl_arr[i-1]))
        atr_ema = tr if i == 1 else alpha_a * tr + (1.0 - alpha_a) * atr_ema
        atr_arr[i] = atr_ema

    vol_ema_arr = np.zeros(n, dtype=np.float64)
    alpha_v = 2.0 / 21.0
    v_ema = float(vol_arr[0]) if not np.isnan(vol_arr[0]) else 0.0
    vol_ema_arr[0] = v_ema
    for i in range(1, n):
        if not np.isnan(vol_arr[i]):
            v_ema = alpha_v * float(vol_arr[i]) + (1.0 - alpha_v) * v_ema
        vol_ema_arr[i] = v_ema

    sw_peak = cl_arr[entry_bar]
    decline_count = 0
    prev_dp = np.nan
    prev_ma20 = ma20_arr[entry_bar]
    prev_ma60 = ma60_arr[entry_bar]
    exit_bar = -1
    exit_reason = "unknown"

    for i in range(entry_bar + 1, n):
        cl = cl_arr[i]; dp_v = dev_pct_arr[i]
        m20 = ma20_arr[i]; m60 = ma60_arr[i]

        if np.isnan(cl) or cl <= 0 or np.isnan(dp_v):
            continue

        if e_grid is not None and e_grid[i]:
            exit_bar = i; exit_reason = "grid_force"; break

        exit_now = False
        if SWITCH_PARAMS["trend_decline_days"] > 0 and decline_count >= SWITCH_PARAMS["trend_decline_days"]:
            exit_bar = i; exit_reason = "dp_decline"; break
        if atr_arr[i] > 0 and cl <= sw_peak - SWITCH_PARAMS["trend_atr_mult"] * atr_arr[i]:
            exit_bar = i; exit_reason = "atr_trail"; break
        if vol_ema_arr[i] > 0 and i > 0:
            rv = vol_arr[i] / max(vol_ema_arr[i], 1e-9)
            if rv > SWITCH_PARAMS["trend_vol_climax"] and cl > cl_arr[i-1]:
                exit_bar = i; exit_reason = "vol_climax"; break
        if (not np.isnan(m20) and not np.isnan(m60)
            and not np.isnan(prev_ma20) and not np.isnan(prev_ma60)
            and m20 < m60 and prev_ma20 >= prev_ma60):
            exit_bar = i; exit_reason = "ma_death"; break

        sw_peak = max(sw_peak, cl)

        if not np.isnan(prev_dp) and not np.isnan(dp_v):
            if dp_v < prev_dp: decline_count += 1
            else: decline_count = 0
        prev_dp = dp_v; prev_ma20 = m20; prev_ma60 = m60

    if exit_bar < 0:
        exit_bar = n - 1; exit_reason = "force_close"

    pnl = cl_arr[exit_bar] / cl_arr[entry_bar] - 1
    return {"exit_bar": exit_bar, "exit_reason": exit_reason, "pnl": pnl,
            "holding_days": exit_bar - entry_bar}


# ══════════════════════════════════════════════════════════════════
# 主分析
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("  Switch 入场 OHLCV 特征分析：连续上涨 & 单日急涨")
    print("=" * 90)

    data = load_data("data/1d/512890.SH_hfq.parquet")
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    windows = generate_monthly_windows(
        close.index, train_months=22, test_months=12,
        step_months=3, warmup_months=12,
    )
    print(f"WF 窗口: {len(windows)} 个 (步长3月)")

    all_features = []
    all_trades = []

    for wi, w in enumerate(windows):
        close_warmup_all = close.iloc[w.warmup_start:w.test_end]
        open_warmup_all = open_.iloc[w.warmup_start:w.test_end]
        test_offset = w.test_start - w.warmup_start
        test_start_label = close_warmup_all.index[test_offset]

        ind_s = compute_polyfit_switch_indicators(
            close_warmup_all, fit_window_days=252, ma_windows=[20, 60],
            trend_window_days=10, vol_window_days=20,
        )
        ind_test = ind_s.loc[ind_s.index >= test_start_label]
        if len(ind_test) < 10:
            continue

        cl_test = close_warmup_all.loc[ind_test.index]
        op_test = open_warmup_all.loc[ind_test.index]

        cl_arr = cl_test.values
        op_arr = op_test.values
        hi_arr = high.reindex(ind_test.index).values
        lo_arr = low.reindex(ind_test.index).values
        vol_arr = volume.reindex(ind_test.index).values
        dev_pct_arr = ind_test["PolyDevPct"].values
        dev_trend_arr = ind_test["PolyDevTrend"].values
        vol_pct_arr = ind_test["RollingVolPct"].values
        poly_base_arr = ind_test["PolyBasePred"].values
        ma20_arr = ind_test["MA20"].values
        ma60_arr = ind_test["MA60"].values

        e_grid, x_grid, s_grid = generate_grid_signals(
            cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
            **GRID_PARAMS,
        )

        e_sw, x_sw, s_sw = generate_grid_priority_switch_signals(
            cl_arr, dev_pct_arr, dev_trend_arr, vol_pct_arr, poly_base_arr,
            e_grid, x_grid, ma20_arr, ma60_arr,
            high=hi_arr, low=lo_arr, volume=vol_arr,
            **SWITCH_PARAMS,
        )

        entry_bars = np.where(e_sw)[0]
        for bar in entry_bars:
            if bar < LOOKBACK:  # 需要足够的前置数据
                continue
            feat = extract_entry_features(
                bar, cl_arr, op_arr, hi_arr, lo_arr, vol_arr,
                dev_pct_arr, ma20_arr, ma60_arr,
            )
            trade = simulate_switch_trade(
                bar, cl_arr, hi_arr, lo_arr, vol_arr,
                dev_pct_arr, ma20_arr, ma60_arr, e_grid,
            )
            all_features.append(feat)
            all_trades.append(trade)

        if (wi + 1) % 5 == 0:
            print(f"  [{wi+1}/{len(windows)}] 累计 Switch 入场: {len(all_features)}")

    print(f"\n  共收集 {len(all_features)} 笔 Switch 入场\n")

    if not all_features:
        print("  无 Switch 入场信号")
        return

    df = pd.DataFrame(all_features)
    df["pnl"] = [t["pnl"] for t in all_trades]
    df["exit_reason"] = [t["exit_reason"] for t in all_trades]
    df["holding_days"] = [t["holding_days"] for t in all_trades]
    df["is_win"] = df["pnl"] > 0

    n_total = len(df)
    n_win = df["is_win"].sum()
    n_lose = n_total - n_win
    print(f"  总入场: {n_total}  盈利: {n_win} ({n_win/n_total*100:.1f}%)  "
          f"亏损: {n_lose} ({n_lose/n_total*100:.1f}%)")
    print(f"  平均盈亏: {df['pnl'].mean():+.2%}  中位盈亏: {df['pnl'].median():+.2%}")
    print(f"  平均持仓: {df['holding_days'].mean():.1f} 天\n")

    # ══════════════════════════════════════════════════════════════
    # 1. 按连续上涨天数分组
    # ══════════════════════════════════════════════════════════════
    print("═" * 90)
    print("  1. 入场前连续上涨天数分析")
    print("═" * 90)
    print(f"  {'连涨天':>6s} {'笔数':>6s} {'占比':>6s} {'盈利%':>7s} {'平均盈亏':>9s} "
          f"{'中位盈亏':>9s} {'平均持仓':>8s}")
    print(f"  {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*9} {'─'*9} {'─'*8}")

    for cons in range(0, 8):
        sub = df[df["consecutive_up"] == cons]
        if len(sub) == 0:
            continue
        win_r = sub["is_win"].mean()
        print(f"  {cons:>6d} {len(sub):>6d} {len(sub)/n_total:>5.1%} "
              f"{win_r:>6.1%} {sub['pnl'].mean():>+8.2%} {sub['pnl'].median():>+8.2%} "
              f"{sub['holding_days'].mean():>7.1f}")

    # 连涨 3+ 合并
    sub_high = df[df["consecutive_up"] >= 3]
    sub_low = df[df["consecutive_up"] < 3]
    if len(sub_high) > 0 and len(sub_low) > 0:
        print(f"\n  ── 连涨 >=3 天 vs <3 天 ──")
        for label, sub in [("连涨>=3天", sub_high), ("连涨<3天", sub_low)]:
            print(f"  {label}: {len(sub)}笔 胜率={sub['is_win'].mean():.1%}  "
                  f"平均盈亏={sub['pnl'].mean():+.2%} 中位盈亏={sub['pnl'].median():+.2%}  "
                  f"持仓={sub['holding_days'].mean():.1f}天")

    # ══════════════════════════════════════════════════════════════
    # 2. 按入场当日涨跌幅分组
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  2. 入场当日涨跌幅分析")
    print(f"{'═' * 90}")

    # 定义分组
    bins_ret = [(-0.05, -0.02), (-0.02, -0.01), (-0.01, 0), (0, 0.005),
                (0.005, 0.01), (0.01, 0.02), (0.02, 0.05)]
    print(f"  {'当日涨幅':>12s} {'笔数':>6s} {'胜率':>7s} {'平均盈亏':>9s} {'中位盈亏':>9s} "
          f"{'平均持仓':>8s} {'量比':>6s} {'连涨':>6s}")
    print(f"  {'─'*12} {'─'*6} {'─'*7} {'─'*9} {'─'*9} {'─'*8} {'─'*6} {'─'*6}")
    for lo, hi in bins_ret:
        sub = df[(df["entry_day_ret"] >= lo) & (df["entry_day_ret"] < hi)]
        if len(sub) == 0:
            continue
        win_r = sub["is_win"].mean()
        print(f"  [{lo:+.1%} ~ {hi:+.1%}) {len(sub):>5d} {win_r:>6.1%} "
              f"{sub['pnl'].mean():>+8.2%} {sub['pnl'].median():>+8.2%} "
              f"{sub['holding_days'].mean():>7.1f} {sub['rel_vol'].mean():>5.2f} "
              f"{sub['consecutive_up'].mean():>5.1f}")

    # ══════════════════════════════════════════════════════════════
    # 3. 单日急涨（涨幅>=1.5%）特征
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  3. 单日急涨入场特征（入场日涨幅 >= 1.5%）")
    print(f"{'═' * 90}")

    surge = df[df["entry_day_ret"] >= 0.015]
    non_surge = df[df["entry_day_ret"] < 0.015]

    if len(surge) > 0:
        print(f"  急涨入场: {len(surge)} 笔 ({len(surge)/n_total*100:.1f}%)")
        print(f"    胜率: {surge['is_win'].mean():.1%}")
        print(f"    平均盈亏: {surge['pnl'].mean():+.2%}")
        print(f"    中位盈亏: {surge['pnl'].median():+.2%}")
        print(f"    平均持仓: {surge['holding_days'].mean():.1f} 天")
        print(f"    平均连涨: {surge['consecutive_up'].mean():.1f} 天")
        print(f"    平均量比: {surge['rel_vol'].mean():.2f}")
        print(f"    均价位:   {surge['price_position'].mean():.2%} (N日高低点)")
        print(f"    均振幅:   {surge['amplitude'].mean():.2%}")

    if len(non_surge) > 0:
        print(f"\n  非急涨入场: {len(non_surge)} 笔 ({len(non_surge)/n_total*100:.1f}%)")
        print(f"    胜率: {non_surge['is_win'].mean():.1%}")
        print(f"    平均盈亏: {non_surge['pnl'].mean():+.2%}")

    # ══════════════════════════════════════════════════════════════
    # 4. OHLCV 形态特征
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  4. OHLCV 形态分析")
    print(f"{'═' * 90}")

    # K 线
    print(f"\n  ── K 线形态 ──")
    for label, sub in [("阳线(收>=开)", df[df["is_green"]]),
                        ("阴线(收<开)", df[~df["is_green"]])]:
        if len(sub) == 0:
            continue
        print(f"  {label}: {len(sub)}笔 ({len(sub)/n_total*100:.1f}%)  "
              f"胜率={sub['is_win'].mean():.1%} 盈亏={sub['pnl'].mean():+.2%}")

    # 上影线
    print(f"\n  ── 上影线比例（上影/振幅）──")
    for label, cond, threshold in [
        ("长上影(>=30%)", df["upper_wick_pct"] >= 0.30, 0.30),
        ("中上影(15-30%)", (df["upper_wick_pct"] >= 0.15) & (df["upper_wick_pct"] < 0.30), 0.15),
        ("短上影(<15%)", df["upper_wick_pct"] < 0.15, 0.15),
    ]:
        sub = df[cond]
        if len(sub) == 0:
            continue
        print(f"  {label}: {len(sub)}笔 ({len(sub)/n_total*100:.1f}%)  "
              f"胜率={sub['is_win'].mean():.1%} 盈亏={sub['pnl'].mean():+.2%}")

    # 实体大小
    print(f"\n  ── 实体比例（实体/振幅）──")
    for label, cond in [
        ("大实体(>=60%)", df["body_pct"] >= 0.60),
        ("中实体(30-60%)", (df["body_pct"] >= 0.30) & (df["body_pct"] < 0.60)),
        ("小实体(<30%)", df["body_pct"] < 0.30),
    ]:
        sub = df[cond]
        if len(sub) == 0:
            continue
        print(f"  {label}: {len(sub)}笔 ({len(sub)/n_total*100:.1f}%)  "
              f"胜率={sub['is_win'].mean():.1%} 盈亏={sub['pnl'].mean():+.2%}")

    # 成交量
    print(f"\n  ── 相对成交量（当日量 / 20日均量）──")
    for label, cond in [
        ("放量(>=1.5x)", df["rel_vol"] >= 1.5),
        ("正常(0.8-1.5x)", (df["rel_vol"] >= 0.8) & (df["rel_vol"] < 1.5)),
        ("缩量(<0.8x)", df["rel_vol"] < 0.8),
    ]:
        sub = df[cond]
        if len(sub) == 0:
            continue
        print(f"  {label}: {len(sub)}笔 ({len(sub)/n_total*100:.1f}%)  "
              f"胜率={sub['is_win'].mean():.1%} 盈亏={sub['pnl'].mean():+.2%}")

    # 价格位置
    print(f"\n  ── 价格位置（N日高低点百分位）──")
    for label, cond in [
        ("高位(>=70%)", df["price_position"] >= 0.70),
        ("中位(30-70%)", (df["price_position"] >= 0.30) & (df["price_position"] < 0.70)),
        ("低位(<30%)", df["price_position"] < 0.30),
    ]:
        sub = df[cond]
        if len(sub) == 0:
            continue
        print(f"  {label}: {len(sub)}笔 ({len(sub)/n_total*100:.1f}%)  "
              f"胜率={sub['is_win'].mean():.1%} 盈亏={sub['pnl'].mean():+.2%}")

    # ══════════════════════════════════════════════════════════════
    # 5. 盈利 vs 亏损的特征差异
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  5. 盈利 vs 亏损 — 特征均值对比")
    print(f"{'═' * 90}")

    win = df[df["is_win"]]
    lose = df[~df["is_win"]]

    features_to_compare = [
        ("consecutive_up", "入场前连涨天数"),
        ("entry_day_ret", "入场当日涨跌幅"),
        ("pre_max_up", "入场前最大单日涨幅"),
        ("rel_vol", "相对成交量(20日均)"),
        ("vol_ratio_1d", "量比(相对前日)"),
        ("price_position", "价格位置(N日%)"),
        ("amplitude", "当日振幅"),
        ("upper_wick_pct", "上影线占比"),
        ("lower_wick_pct", "下影线占比"),
        ("body_pct", "实体占比"),
        ("dp", "偏离度 dp"),
        ("ma_diff", "MA20-MA60 差"),
        ("price_vs_ma20", "价格vs MA20"),
    ]

    print(f"  {'特征':<22s} {'盈利均值':>10s} {'亏损均值':>10s} {'差异':>10s} {'方向':>8s}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for col, name in features_to_compare:
        w_mean = win[col].mean()
        l_mean = lose[col].mean()
        diff = w_mean - l_mean
        direction = "赢>亏" if diff > 0 else "赢<亏"
        print(f"  {name:<22s} {w_mean:>+10.4f} {l_mean:>+10.4f} "
              f"{diff:>+10.4f} {direction:>8s}")

    # ══════════════════════════════════════════════════════════════
    # 6. 交叉组合：连续上涨 + 单日急涨
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  6. 组合特征分析：连涨 + 急涨")
    print(f"{'═' * 90}")

    # 四种组合
    combo_labels = []
    combo_stats = []

    for cons_label, cons_cond in [
        ("连涨>=3天", df["consecutive_up"] >= 3),
        ("连涨0-2天", df["consecutive_up"] < 3),
    ]:
        for surge_label, surge_cond in [
            ("急涨>=1.5%", df["entry_day_ret"] >= 0.015),
            ("非急涨<1.5%", df["entry_day_ret"] < 0.015),
        ]:
            sub = df[cons_cond & surge_cond]
            label = f"{cons_label} + {surge_label}"
            if len(sub) > 0:
                combo_labels.append(label)
                combo_stats.append({
                    "label": label, "n": len(sub),
                    "win_rate": sub["is_win"].mean(),
                    "avg_pnl": sub["pnl"].mean(),
                    "med_pnl": sub["pnl"].median(),
                    "avg_hold": sub["holding_days"].mean(),
                    "avg_vol": sub["rel_vol"].mean(),
                    "avg_pos": sub["price_position"].mean(),
                })

    print(f"  {'组合':<28s} {'笔数':>5s} {'胜率':>7s} {'平均盈亏':>9s} "
          f"{'中位盈亏':>9s} {'持仓':>6s} {'量比':>6s} {'价位':>6s}")
    print(f"  {'─'*28} {'─'*5} {'─'*7} {'─'*9} {'─'*9} {'─'*6} {'─'*6} {'─'*6}")
    for s in combo_stats:
        print(f"  {s['label']:<28s} {s['n']:>5d} {s['win_rate']:>6.1%} "
              f"{s['avg_pnl']:>+8.2%} {s['med_pnl']:>+8.2%} {s['avg_hold']:>5.1f} "
              f"{s['avg_vol']:>5.2f} {s['avg_pos']:>.2%}")

    # ══════════════════════════════════════════════════════════════
    # 7. 离场原因分布
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  7. 离场原因分布")
    print(f"{'═' * 90}")
    reason_counts = df["exit_reason"].value_counts()
    print(f"  {'离场原因':<16s} {'笔数':>6s} {'占比':>6s} {'胜率':>7s} {'平均盈亏':>9s} {'持仓':>6s}")
    print(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*7} {'─'*9} {'─'*6}")
    for reason in reason_counts.index:
        sub = df[df["exit_reason"] == reason]
        print(f"  {reason:<16s} {len(sub):>6d} {len(sub)/n_total:>5.1%} "
              f"{sub['is_win'].mean():>6.1%} {sub['pnl'].mean():>+8.2%} "
              f"{sub['holding_days'].mean():>5.1f}")


if __name__ == "__main__":
    main()
