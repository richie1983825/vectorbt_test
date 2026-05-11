"""
分析 Switch-v6 趋势日的日内特征，基于模式推荐日内策略。

分析维度：
  1. 开盘方向：Switch日开盘相对前收的涨跌分布
  2. 日内路径：开盘→收盘的路径特征（高开高走/低开高走/冲高回落）
  3. 日内买卖时机：最优的入场时间窗口
  4. VWAP关系：价格相对VWAP的位置

推荐策略候选：
  A) 开盘直接买入（趋势日时间> timing）
  B) 早盘回调买入（30min低点附近挂单）
  C) VWAP回踩买入
  D) 分段买入（TWAP）
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from utils.walkforward import generate_monthly_windows
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6

# ══════════════════════════════════════════════════════════════════
GRID_PARAMS = dict(
    base_grid_pct=0.008, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=1.0, stop_loss_grid=1.2,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=0.45, position_size=0.92, position_sizing_coef=30,
)
SW_PARAMS = dict(trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
                 trend_atr_mult=1.5, trend_atr_window=14,
                 trend_vol_climax=2.5, trend_decline_days=1,
                 enable_ohlcv_filter=True, enable_early_exit=True)


def load_5m_data(path="data/5m/512890.SH_5min.parquet"):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def find_switch_entry_days(close_daily, open_daily, high_daily, low_daily, volume_daily):
    """找出所有 Switch-v6 入场日。"""
    windows = generate_monthly_windows(
        close_daily.index, train_months=22, test_months=12,
        step_months=3, warmup_months=12,
    )
    all_switch_dates = []

    for w in windows:
        cwa = close_daily.iloc[w.warmup_start:w.test_end]
        owa = open_daily.iloc[w.warmup_start:w.test_end]
        toff = w.test_start - w.warmup_start
        tsl = cwa.index[toff]

        ind = compute_polyfit_switch_indicators(
            cwa, fit_window_days=252, ma_windows=[20, 60],
            trend_window_days=10, vol_window_days=10,
        )
        it = ind.loc[ind.index >= tsl]
        if len(it) < 10: continue

        ca = cwa.loc[it.index].values; oa = owa.reindex(it.index).values
        ha = high_daily.reindex(it.index).values
        la = low_daily.reindex(it.index).values
        va = volume_daily.reindex(it.index).values

        eg, xg, sg = generate_grid_signals(
            ca, it["PolyDevPct"].values, it["PolyDevTrend"].values,
            it["RollingVolPct"].values, it["PolyBasePred"].values, **GRID_PARAMS,
        )
        esw, xsw, ssw = generate_grid_priority_switch_signals_v6(
            ca, it["PolyDevPct"].values, it["PolyDevTrend"].values,
            it["RollingVolPct"].values, it["PolyBasePred"].values,
            eg, xg, it["MA20"].values, it["MA60"].values,
            high=ha, low=la, open_=oa, volume=va, **SW_PARAMS,
        )

        for bar in np.where(esw)[0]:
            if bar < len(it.index):
                all_switch_dates.append(it.index[bar])

    return sorted(set(all_switch_dates))


def analyze_intraday(signal_dates, data_5m):
    """分析 Switch 信号日的日内特征。"""
    records = []
    for dt in signal_dates:
        day_bars = data_5m[data_5m.index.normalize() == dt.normalize()]
        if len(day_bars) < 8:  # 至少要有1小时数据
            continue

        op = day_bars["Open"].iloc[0]
        cl = day_bars["Close"].iloc[-1]
        hi = day_bars["High"].max()
        lo = day_bars["Low"].min()
        prev_cl = cl  # 用开盘作为近似

        # 前一日收盘（用前一天最后一根bar）
        prev_day = data_5m[data_5m.index.normalize() == (dt - pd.Timedelta(days=1)).normalize()]
        if len(prev_day) > 0:
            prev_cl = prev_day["Close"].iloc[-1]

        # 日内区间
        open_ret = op / prev_cl - 1
        close_ret = cl / op - 1
        full_ret = cl / prev_cl - 1
        high_ret = hi / op - 1   # 日内最大涨幅（相对开盘）
        low_ret = lo / op - 1    # 日内最大跌幅（相对开盘）
        range_pct = (hi - lo) / op

        # 分段收益
        n = len(day_bars)
        seg1_end = min(6, n)     # 30min
        seg2_end = min(12, n)    # 60min
        seg3_end = min(24, n)    # 120min

        seg1_ret = day_bars["Close"].iloc[seg1_end-1] / op - 1 if seg1_end > 0 else 0
        seg2_ret = day_bars["Close"].iloc[seg2_end-1] / op - 1 if seg2_end > 0 else 0
        seg3_ret = day_bars["Close"].iloc[seg3_end-1] / op - 1 if seg3_end > 0 else 0

        # 早盘是否有回调（30分钟内最低价 < 开盘-0.5%）
        early_low = day_bars["Low"].iloc[:min(12, n)].min()
        morning_dip = (early_low - op) / op

        # VWAP 计算
        typical = (day_bars["High"] + day_bars["Low"] + day_bars["Close"]) / 3
        vwap = (typical * day_bars["Volume"]).cumsum() / day_bars["Volume"].cumsum()
        close_vs_vwap = cl / vwap.iloc[-1] - 1 if vwap.iloc[-1] > 0 else 0

        # 价格在VWAP上方的时间占比
        above_vwap = (day_bars["Close"].values > vwap.values).mean()

        # 成交量分布
        morning_vol = day_bars["Volume"].iloc[:12].sum()    # 前60分钟
        afternoon_vol = day_bars["Volume"].iloc[24:].sum()  # 120分钟后
        total_vol = day_bars["Volume"].sum()
        morning_vol_pct = morning_vol / total_vol if total_vol > 0 else 0

        records.append({
            "date": dt, "n_bars": n,
            "open_ret": open_ret, "close_ret": close_ret, "full_ret": full_ret,
            "high_ret": high_ret, "low_ret": low_ret, "range_pct": range_pct,
            "seg1_30m": seg1_ret, "seg2_60m": seg2_ret, "seg3_120m": seg3_ret,
            "morning_dip": morning_dip, "close_vs_vwap": close_vs_vwap,
            "above_vwap": above_vwap, "morning_vol_pct": morning_vol_pct,
        })

    return pd.DataFrame(records)


def test_intraday_strategies(df, data_5m):
    """测试几种简单的日内策略在Switch信号日上的表现。"""
    results = {}

    for dt in df["date"]:
        day = data_5m[data_5m.index.normalize() == dt.normalize()]
        if len(day) < 8: continue
        n = len(day)
        op = day["Open"].iloc[0]
        cl = day["Close"].iloc[-1]

        # 策略 A: 开盘买入，收盘卖出
        strat_a = cl / op - 1

        # 策略 B: 等30分钟回调 -0.5%买入，否则开盘买入
        early_lo = day["Low"].iloc[:min(6, n)].min()
        if early_lo < op * 0.995:
            entry_b = op * 0.995  # 回调0.5%买入
        else:
            entry_b = op
        strat_b = cl / entry_b - 1

        # 策略 C: 等60分钟，如果仍上涨则追入
        seg2_cl = day["Close"].iloc[min(12, n)-1]
        if seg2_cl > op:
            entry_c = seg2_cl  # 确认上涨后追入
            # 后续到收盘的收益
            remaining_bars = day.iloc[min(12, n):]
            exit_c = remaining_bars["Close"].iloc[-1] if len(remaining_bars) > 0 else seg2_cl
            strat_c = exit_c / entry_c - 1
        else:
            strat_c = 0  # 未触发

        # 策略 D: VWAP + 趋势确认 (价格>VWAP时持有，<VWAP时空仓)
        typical = (day["High"] + day["Low"] + day["Close"]) / 3
        vwap = (typical * day["Volume"]).cumsum() / day["Volume"].cumsum()
        # 简化: 若收盘>VWAP，持仓；否则空仓
        if cl > vwap.iloc[-1]:
            strat_d = cl / op - 1
        else:
            strat_d = 0

        for name, ret in [("A_开盘买", strat_a), ("B_回调买", strat_b),
                           ("C_追涨", strat_c), ("D_VWAP过滤", strat_d)]:
            if name not in results:
                results[name] = []
            results[name].append(ret)

    return {k: np.array(v) for k, v in results.items()}


if __name__ == "__main__":
    print("=" * 80)
    print("  Switch-v6 趋势日 — 日内特征分析与策略推荐")
    print("=" * 80)

    # 加载数据
    print("\n[1/4] 加载数据…")
    daily = load_data("data/1d/512890.SH_hfq.parquet")
    close_d = daily["Close"]; open_d = daily["Open"]
    high_d = daily["High"]; low_d = daily["Low"]; vol_d = daily["Volume"]
    data_5m = load_5m_data()

    print(f"  日线: {len(daily)} bars  {daily.index[0].date()} → {daily.index[-1].date()}")
    print(f"  5分钟线: {len(data_5m)} bars  {data_5m.index[0]} → {data_5m.index[-1]}")

    # 找 Switch 信号日
    print("\n[2/4] 查找 Switch-v6 入场日…")
    switch_dates = find_switch_entry_days(close_d, open_d, high_d, low_d, vol_d)
    print(f"  找到 {len(switch_dates)} 个 Switch 入场日")
    print(f"  日期范围: {switch_dates[0].date() if switch_dates else 'N/A'} → "
          f"{switch_dates[-1].date() if switch_dates else 'N/A'}")

    # 日内分析
    print("\n[3/4] 日内特征分析…")
    df = analyze_intraday(switch_dates, data_5m)
    print(f"  有效样本: {len(df)} 天\n")

    # ── 特征统计 ──
    print("═" * 80)
    print("  日内特征统计")
    print("═" * 80)

    print(f"\n  ── 开盘方向 ──")
    gap_up = (df["open_ret"] > 0).sum()
    gap_down = (df["open_ret"] < 0).sum()
    gap_flat = (df["open_ret"] == 0).sum()
    print(f"  高开(>前收): {gap_up}天 ({gap_up/len(df)*100:.0f}%)")
    print(f"  低开(<前收): {gap_down}天 ({gap_down/len(df)*100:.0f}%)")
    print(f"  开平:       {gap_flat}天")
    print(f"  平均开盘涨跌: {df['open_ret'].mean():+.2%}")
    print(f"  开盘涨跌中位: {df['open_ret'].median():+.2%}")

    print(f"\n  ── 日内路径 ──")
    print(f"  日内涨跌(开→收): 均值 {df['close_ret'].mean():+.2%}  中位 {df['close_ret'].median():+.2%}")
    print(f"  全天涨跌(前收→收): 均值 {df['full_ret'].mean():+.2%}  中位 {df['full_ret'].median():+.2%}")
    print(f"  日内最大涨幅(相对开): 均值 {df['high_ret'].mean():+.2%}")
    print(f"  日内最大跌幅(相对开): 均值 {df['low_ret'].mean():+.2%}")
    print(f"  日内振幅: 均值 {df['range_pct'].mean():+.2%}")

    # 路径分类
    up_day = (df["close_ret"] > 0.002).sum()        # 收涨>0.2%
    down_day = (df["close_ret"] < -0.002).sum()      # 收跌<-0.2%
    flat_day = len(df) - up_day - down_day
    print(f"\n  收涨(>0.2%): {up_day}天 ({up_day/len(df)*100:.0f}%)")
    print(f"  收跌(<-0.2%): {down_day}天 ({down_day/len(df)*100:.0f}%)")
    print(f"  震荡: {flat_day}天 ({flat_day/len(df)*100:.0f}%)")

    print(f"\n  ── 分段收益 ──")
    print(f"  0-30min:  均值 {df['seg1_30m'].mean():+.2%}  中位 {df['seg1_30m'].median():+.2%}  正{(df['seg1_30m']>0).mean()*100:.0f}%")
    print(f"  0-60min:  均值 {df['seg2_60m'].mean():+.2%}  中位 {df['seg2_60m'].median():+.2%}  正{(df['seg2_60m']>0).mean()*100:.0f}%")
    print(f"  0-120min: 均值 {df['seg3_120m'].mean():+.2%}  中位 {df['seg3_120m'].median():+.2%}  正{(df['seg3_120m']>0).mean()*100:.0f}%")

    print(f"\n  ── 早盘回调 ──")
    print(f"  30min最低价相对开盘: 均值 {df['morning_dip'].mean():+.2%}")
    dip_gt_05 = (df["morning_dip"] < -0.005).sum()
    print(f"  回调>0.5%: {dip_gt_05}天 ({dip_gt_05/len(df)*100:.0f}%)")

    print(f"\n  ── VWAP 关系 ──")
    above_vwap_avg = df["above_vwap"].mean()
    print(f"  价格>VWAP 时间占比: {above_vwap_avg:.0%}")
    print(f"  收盘>VWAP: {(df['close_vs_vwap']>0).sum()}天 ({(df['close_vs_vwap']>0).mean()*100:.0f}%)")
    print(f"  收盘相对VWAP: 均值 {df['close_vs_vwap'].mean():+.3%}")

    print(f"\n  ── 成交量分布 ──")
    print(f"  前60分钟成交量占比: {df['morning_vol_pct'].mean():.0%}")

    # ── 策略对比 ──
    print(f"\n[4/4] 日内策略对比测试…")
    results = test_intraday_strategies(df, data_5m)

    print(f"\n{'═' * 80}")
    print(f"  策略对比（{len(df)} 个 Switch 信号日）")
    print(f"{'═' * 80}")
    print(f"  {'策略':<20s} {'均值':>8s} {'中位':>8s} {'胜率':>7s} {'夏普':>7s} {'最大盈':>8s} {'最大亏':>8s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")
    for name, rets in results.items():
        rets = rets[rets != 0]  # 排除未触发的
        if len(rets) < 3: continue
        mu = rets.mean()
        med = np.median(rets)
        win = (rets > 0).mean()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        print(f"  {name:<20s} {mu:>+7.2%} {med:>+7.2%} {win:>6.0%} {sharpe:>6.1f} {rets.max():>+7.2%} {rets.min():>+7.2%}")

    print(f"\n{'═' * 80}")
    print(f"  推荐")
    print(f"{'═' * 80}")
    # 找最佳策略
    best_name = max(results, key=lambda k: results[k][results[k]!=0].mean() if len(results[k][results[k]!=0])>0 else -999)
    best_rets = results[best_name][results[best_name]!=0]
    print(f"  基于 Switch 信号日 ({len(df)}天) 的日内特征：")
    print(f"    - 开盘涨跌均值 {df['open_ret'].mean():+.2%}，{gap_up/len(df)*100:.0f}% 高开")
    print(f"    - 日内收涨概率 {(df['close_ret']>0).mean()*100:.0f}%，均值 {df['close_ret'].mean():+.2%}")
    print(f"    - 早盘回调>0.5% 的概率 {dip_gt_05/len(df)*100:.0f}%")
    print(f"    - VWAP上方时间 {above_vwap_avg:.0%}")
    print(f"  ")
    print(f"  推荐策略: {best_name} (均值 {best_rets.mean():+.2%}, 胜率 {(best_rets>0).mean():.0%})")
