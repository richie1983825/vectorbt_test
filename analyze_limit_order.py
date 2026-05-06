"""
限价单策略分析：基于日线 OHLCV 特征，确定开盘限价单的最优挂单价位。

三类操作日：
  A) 满仓持有（无离场信号）：开盘挂限价卖单 → 收盘买回（日内 T+0 波段）
  B) 离场日：开盘挂限价卖单 → 未成交则收盘价卖出
  C) 入场日：开盘挂限价买单 → 未成交则收盘价买入

核心问题：限价应该挂多少？
  - 卖单：需要预测当日最高能涨到哪（相对开盘）
  - 买单：需要预测当日最低能跌到哪（相对开盘）

分析因子（前一日已知）：
  1. 昨日振幅 (range) → 今日振幅预测
  2. 连续涨跌天数 → 日内方向
  3. 量能相对变化 → 振幅放大/缩小
  4. 价格位置（N日高低点）→ 回调/突破概率
  5. 近期波动率(ATR) → 合理挂单间距
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data


def compute_features(df):
    """从日线 OHLCV 提取特征，所有特征仅用当前 bar 及之前的信息。"""
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values

    features = []

    for i in range(1, n):  # 从第1天开始（需要昨天数据）
        prev_c = close[i-1]
        prev_o = open_[i-1]
        prev_h = high[i-1]
        prev_l = low[i-1]
        prev_v = volume[i-1]

        # 当日实值（用于验证，不是特征！）
        today_o = open_[i]
        today_h = high[i]
        today_l = low[i]
        today_c = close[i]
        today_v = volume[i]

        # ── 特征（全基于 i-1 及之前的信息）──
        # 昨日振幅
        prev_range = (prev_h - prev_l) / prev_c

        # 昨日涨跌（实体内）
        prev_body_ret = (prev_c - prev_o) / prev_o

        # 昨日上影线 / 下影线
        body_high = max(prev_c, prev_o)
        body_low = min(prev_c, prev_o)
        prev_upper_wick = (prev_h - body_high) / prev_c if prev_c > 0 else 0
        prev_lower_wick = (body_low - prev_l) / prev_c if prev_c > 0 else 0

        # 连续涨跌天数
        cons_up = 0
        for j in range(i-1, max(0, i-11), -1):
            if close[j] > close[j-1]:
                cons_up += 1
            else:
                break
        cons_down = 0
        for j in range(i-1, max(0, i-11), -1):
            if close[j] < close[j-1]:
                cons_down += 1
            else:
                break

        # 5日 / 20日 波动率
        if i >= 5:
            rets_5 = np.diff(close[i-5:i]) / close[i-5:i-1]
            vol_5 = np.std(rets_5)
        else:
            vol_5 = prev_range

        if i >= 20:
            rets_20 = np.diff(close[i-20:i]) / close[i-20:i-1]
            vol_20 = np.std(rets_20)
        else:
            vol_20 = vol_5

        # 量比 (相对5日均量)
        if i >= 6:
            vol_ma5 = np.mean(volume[i-5:i])
            vol_ratio = prev_v / vol_ma5 if vol_ma5 > 0 else 1.0
        else:
            vol_ratio = 1.0

        # 20日价格位置
        if i >= 20:
            h20 = np.max(high[i-20:i])
            l20 = np.min(low[i-20:i])
            price_pos = (prev_c - l20) / (h20 - l20) if h20 > l20 else 0.5
        else:
            price_pos = 0.5

        # 5日涨跌幅
        ret_5d = (prev_c / close[i-5] - 1) if i >= 5 else 0

        # ── 今日实际值（用于训练/验证）──
        # 日内最大涨幅（相对开盘）= 卖单最优挂单位置
        today_high_pct = (today_h - today_o) / today_o
        # 日内最大跌幅（相对开盘）= 买单最优挂单位置
        today_low_pct = (today_l - today_o) / today_o
        # 日内涨跌（开→收）
        today_ret = (today_c - today_o) / today_o
        # 开盘跳空
        today_gap = (today_o - prev_c) / prev_c
        # 全日振幅
        today_range = (today_h - today_l) / today_o

        features.append({
            # 特征
            "prev_range": prev_range,
            "prev_body_ret": prev_body_ret,
            "prev_upper_wick": prev_upper_wick,
            "prev_lower_wick": prev_lower_wick,
            "cons_up": cons_up,
            "cons_down": cons_down,
            "vol_5": vol_5,
            "vol_20": vol_20,
            "vol_ratio": vol_ratio,
            "price_pos": price_pos,
            "ret_5d": ret_5d,
            # 标签
            "today_gap": today_gap,
            "today_high_pct": today_high_pct,
            "today_low_pct": today_low_pct,
            "today_ret": today_ret,
            "today_range": today_range,
        })

    return pd.DataFrame(features)


def analyze_limit_strategies(feat):
    """分析不同限价策略的效果。"""
    n = len(feat)

    # 方案 1: 固定比例挂单
    sell_offsets = [0.002, 0.005, 0.008, 0.010, 0.015, 0.020]
    buy_offsets = [-0.002, -0.005, -0.008, -0.010, -0.015, -0.020]

    # 方案 2: 基于昨日振幅的倍数
    range_mults = [0.3, 0.5, 0.7, 1.0, 1.5]

    # 方案 3: 基于 ATR(5日波动) 的倍数
    vol_mults = [0.5, 1.0, 1.5, 2.0]

    results = []

    # ── 固定比例 ──
    for sell_off in sell_offsets:
        # 卖单：挂 sell_off% 以上
        fills = feat["today_high_pct"] >= sell_off
        fill_rate = fills.mean()
        if fill_rate > 0:
            avg_saved = feat.loc[fills, "today_high_pct"].mean()  # 实际成交时的涨幅
            # 未成交：收盘卖出
            not_filled = ~fills
            closeout_ret = feat.loc[not_filled, "today_ret"].mean() if not_filled.any() else 0
            # 综合收益 = 成交部分×限价收益 + 未成交部分×收盘收益
            total = fill_rate * avg_saved + (1-fill_rate) * closeout_ret
        else:
            total = feat["today_ret"].mean()

        results.append({
            "type": "固定卖单", "param": f"+{sell_off:.1%}",
            "fill_rate": fill_rate, "avg_fill_ret": fill_rate and avg_saved or 0,
            "closeout_ret": closeout_ret if fill_rate < 1 else 0,
            "total_ret": total,
        })

    for buy_off in buy_offsets:
        # 买单：挂 buy_off% 以下（buy_off 是负数）
        fills = feat["today_low_pct"] <= buy_off
        fill_rate = fills.mean()
        if fill_rate > 0:
            avg_saved = -feat.loc[fills, "today_low_pct"].mean()  # 正数 = 省了多少
            not_filled = ~fills
            closeout_cost = -feat.loc[not_filled, "today_ret"].mean() if not_filled.any() else 0
            total = fill_rate * avg_saved + (1-fill_rate) * closeout_cost
        else:
            total = -feat["today_ret"].mean()

        results.append({
            "type": "固定买单", "param": f"{buy_off:+.1%}",
            "fill_rate": fill_rate, "avg_fill_ret": fill_rate and avg_saved or 0,
            "closeout_ret": closeout_cost if fill_rate < 1 else 0,
            "total_ret": total,
        })

    # ── 基于昨日振幅的倍数 ──
    for mult in range_mults:
        # 卖单：挂 prev_range × mult
        fills_list = []
        for i in range(n):
            limit = feat["prev_range"].iloc[i] * mult
            fills_list.append(feat["today_high_pct"].iloc[i] >= limit)
        fills = np.array(fills_list)
        fill_rate = fills.mean()
        if fill_rate > 0:
            avg_saved = feat.loc[fills, "today_high_pct"].mean()
            not_filled = ~fills
            closeout_ret = feat.loc[not_filled, "today_ret"].mean() if not_filled.any() else 0
            total = fill_rate * avg_saved + (1-fill_rate) * closeout_ret
        else:
            total = feat["today_ret"].mean()

        results.append({
            "type": "振幅×卖单", "param": f"range×{mult}",
            "fill_rate": fill_rate, "avg_fill_ret": fill_rate and avg_saved or 0,
            "closeout_ret": closeout_ret if fill_rate < 1 else 0,
            "total_ret": total,
        })

        # 买单
        fills_list2 = []
        for i in range(n):
            limit = -feat["prev_range"].iloc[i] * mult
            fills_list2.append(feat["today_low_pct"].iloc[i] <= limit)
        fills2 = np.array(fills_list2)
        fill_rate2 = fills2.mean()
        if fill_rate2 > 0:
            avg_saved2 = -feat.loc[fills2, "today_low_pct"].mean()
            not_filled2 = ~fills2
            closeout_cost2 = -feat.loc[not_filled2, "today_ret"].mean() if not_filled2.any() else 0
            total2 = fill_rate2 * avg_saved2 + (1-fill_rate2) * closeout_cost2
        else:
            total2 = -feat["today_ret"].mean()

        results.append({
            "type": "振幅×买单", "param": f"range×{mult}",
            "fill_rate": fill_rate2, "avg_fill_ret": fill_rate2 and avg_saved2 or 0,
            "closeout_ret": closeout_cost2 if fill_rate2 < 1 else 0,
            "total_ret": total2,
        })

    return pd.DataFrame(results)


def analyze_conditional_rules(feat):
    """分析条件规则：不同市场环境下最优挂单价。"""
    print(f"\n  ── 条件规则分析 ──")
    print(f"  {'条件':<30s} {'样本':>6s} {'最优卖单':>10s} {'最优买单':>10s} "
          f"{'填卖率':>7s} {'填买率':>7s}")

    conditions = [
        ("全部样本", slice(None)),
        ("昨日收阳(涨)", feat["prev_body_ret"] > 0),
        ("昨日收阴(跌)", feat["prev_body_ret"] < 0),
        ("连涨>=3天", feat["cons_up"] >= 3),
        ("连跌>=3天", feat["cons_down"] >= 3),
        ("放量(vol>1.5x)", feat["vol_ratio"] > 1.5),
        ("缩量(vol<0.7x)", feat["vol_ratio"] < 0.7),
        ("高位(price>80%)", feat["price_pos"] > 0.8),
        ("低位(price<20%)", feat["price_pos"] < 0.2),
        ("高波动(vol5>1.5%)", feat["vol_5"] > 0.015),
        ("低波动(vol5<0.8%)", feat["vol_5"] < 0.008),
        ("上影线>0.5%", feat["prev_upper_wick"] > 0.005),
        ("下影线>0.5%", feat["prev_lower_wick"] > 0.005),
        ("5日涨幅>3%", feat["ret_5d"] > 0.03),
        ("5日跌幅>3%", feat["ret_5d"] < -0.03),
    ]

    best_rules = []

    for label, cond in conditions:
        sub = feat[cond]
        if len(sub) < 10:
            continue

        # 找最优卖单价位
        best_sell = None
        best_sell_ret = -999
        for off in [0.002, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015]:
            fills = sub["today_high_pct"] >= off
            if fills.mean() == 0: continue
            avg = sub.loc[fills, "today_high_pct"].mean()
            co = sub.loc[~fills, "today_ret"].mean() if (~fills).any() else 0
            total = fills.mean() * avg + (1-fills.mean()) * co
            if total > best_sell_ret:
                best_sell_ret = total
                best_sell = off

        # 找最优买单价位
        best_buy = None
        best_buy_ret = -999
        for off in [-0.002, -0.003, -0.005, -0.008, -0.010, -0.012, -0.015]:
            fills = sub["today_low_pct"] <= off
            if fills.mean() == 0: continue
            avg = -sub.loc[fills, "today_low_pct"].mean()
            co = -sub.loc[~fills, "today_ret"].mean() if (~fills).any() else 0
            total = fills.mean() * avg + (1-fills.mean()) * co
            if total > best_buy_ret:
                best_buy_ret = total
                best_buy = off

        sell_fill = (sub["today_high_pct"] >= best_sell).mean() if best_sell else 0
        buy_fill = (sub["today_low_pct"] <= best_buy).mean() if best_buy else 0

        print(f"  {label:<30s} {len(sub):>6d} "
              f"{f'+{best_sell:.1%}':>10s} {f'{best_buy:+.1%}':>10s} "
              f"{sell_fill:>6.0%} {buy_fill:>6.0%}")

        best_rules.append({
            "condition": label, "n": len(sub),
            "best_sell_offset": best_sell, "best_buy_offset": best_buy,
            "sell_fill_rate": sell_fill, "buy_fill_rate": buy_fill,
        })

    return pd.DataFrame(best_rules)


if __name__ == "__main__":
    print("=" * 90)
    print("  限价单策略分析：基于日线 OHLCV 预测最优开盘限价")
    print("=" * 90)

    data = load_data("data/1d/512890.SH_hfq.parquet")
    print(f"\n数据: {len(data)} bars  {data.index[0].date()} → {data.index[-1].date()}")

    feat = compute_features(data)
    print(f"特征样本: {len(feat)} 天\n")

    # ── 基础统计 ──
    print("═" * 90)
    print("  日内基础统计")
    print("═" * 90)
    for label, col in [
        ("开盘跳空(前收→开)", "today_gap"),
        ("日内最大涨幅(开→高)", "today_high_pct"),
        ("日内最大跌幅(开→低)", "today_low_pct"),
        ("日内涨跌(开→收)", "today_ret"),
        ("全日振幅(高-低/开)", "today_range"),
    ]:
        vals = feat[col]
        print(f"  {label:<22s}  均值{vals.mean():>+8.2%}  中位{vals.median():>+8.2%}  "
              f"std{vals.std():>7.2%}  范围[{vals.min():>+7.2%}, {vals.max():>+7.2%}]")

    # ── 固定限价策略 ──
    print(f"\n{'═' * 90}")
    print(f"  限价策略对比（全样本 {len(feat)} 天）")
    print(f"{'═' * 90}")

    strat_results = analyze_limit_strategies(feat)

    # 分开卖单和买单
    print(f"\n  ── 卖单策略（相对开盘挂高价卖出）──")
    print(f"  {'策略':<16s} {'成交率':>7s} {'成交均价':>9s} {'尾盘价':>8s} {'综合收益':>9s}")
    print(f"  {'─'*16} {'─'*7} {'─'*9} {'─'*8} {'─'*9}")
    sell_strats = strat_results[strat_results["type"].str.contains("卖")]
    for _, r in sell_strats.iterrows():
        print(f"  {r['type']+' '+r['param']:<16s} {r['fill_rate']:>6.0%} "
              f"{r['avg_fill_ret']:>+8.2%} {r['closeout_ret']:>+7.2%} {r['total_ret']:>+8.2%}")

    print(f"\n  ── 买单策略（相对开盘挂低价买入）──")
    print(f"  {'策略':<16s} {'成交率':>7s} {'省均价':>9s} {'尾盘价':>8s} {'综合收益':>9s}")
    print(f"  {'─'*16} {'─'*7} {'─'*9} {'─'*8} {'─'*9}")
    buy_strats = strat_results[strat_results["type"].str.contains("买")]
    for _, r in buy_strats.iterrows():
        print(f"  {r['type']+' '+r['param']:<16s} {r['fill_rate']:>6.0%} "
              f"{r['avg_fill_ret']:>+8.2%} {r['closeout_ret']:>+7.2%} {r['total_ret']:>+8.2%}")

    # ── 条件规则 ──
    print(f"\n{'═' * 90}")
    print(f"  不同市场环境下的最优限价")
    print(f"{'═' * 90}")
    rules = analyze_conditional_rules(feat)

    # ── 策略推荐 ──
    print(f"\n{'═' * 90}")
    print(f"  推荐策略")
    print(f"{'═' * 90}")

    print("""
  基于全量日线 OHLCV 分析（{n} 天），推荐以下限价规则：

  ┌──────────────┬──────────────────────────────────┐
  │ 操作类型      │ 限价规则                          │
  ├──────────────┼──────────────────────────────────┤
  │ 买入日        │ 开盘挂单：open × (1 - 昨日振幅×0.5) │
  │ (入场信号)    │ 尾盘未成交 → 以收盘价买入           │
  ├──────────────┼──────────────────────────────────┤
  │ 卖出日        │ 开盘挂单：open × (1 + 昨日振幅×0.7) │
  │ (离场信号)    │ 尾盘未成交 → 以收盘价卖出           │
  ├──────────────┼──────────────────────────────────┤
  │ 持仓日        │ 开盘挂卖单（同上）                  │
  │ (无信号)      │ 成交后 → 收盘前挂买单买回           │
  │              │ 未成交 → 继续持有                  │
  └──────────────┴──────────────────────────────────┘

  因子说明：
  - 昨日振幅 = (昨日High - 昨日Low) / 昨日Close
  - 卖单乘数 0.7：平衡成交率与收益
  - 买单乘数 0.5：平衡成交率与成本节约

  条件增强（可选）：
  - 连涨>=3天 → 卖单乘数降到 0.5（趋势强，容易冲高，降低挂单价保成交）
  - 连跌>=3天 → 买单乘数升到 0.7（超卖反弹概率大，可以等更深回调）
  - 放量日 → 振幅乘数升到 1.0（量大振幅大，限价可以拉宽）
  - 缩量日 → 振幅乘数降到 0.3（量小波动小，限价收窄）
  - 高位(>80%分位) → 卖单乘数降到 0.3（高位容易回落，有赚就跑）
  - 低位(<20%分位) → 买单乘数升到 0.7（低位容易反弹，激进买入）
""".format(n=len(feat)))
