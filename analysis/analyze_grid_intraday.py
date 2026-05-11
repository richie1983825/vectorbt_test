"""
Grid 持仓日 T+0 因子分析。

Grid 模式特征：入场时价格已低于基线（超卖），等待均值回复。
持仓期间价格震荡，适合日内高抛低吸。

分析维度：
  1. Grid持仓日的日内振幅分布（有无操作空间）
  2. 日内方向特征（开盘→收盘 / 开盘→高点 / 低点→收盘）
  3. 前一日特征对当日 T+0 的预测能力
  4. 不同持仓阶段（刚入场/中期/临近到期）的日内特征
  5. 最优 T+0 策略参数
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.data import load_data
from utils.indicators import compute_polyfit_switch_indicators
from strategies.polyfit_grid import generate_grid_signals
from strategies.polyfit_switch import generate_grid_priority_switch_signals_v6

BEST_GRID = dict(
    trend_window_days=10, vol_window_days=10,
    base_grid_pct=0.01, volatility_scale=0.0, trend_sensitivity=4,
    max_grid_levels=3, take_profit_grid=0.8, stop_loss_grid=1.6,
    max_holding_days=45, cooldown_days=1,
    min_signal_strength=0.3, position_size=0.99, position_sizing_coef=60,
)
BEST_SWITCH = dict(
    trend_entry_dp=0.0, trend_confirm_dp_slope=0.0,
    trend_atr_mult=1.5, trend_atr_window=14,
    trend_vol_climax=2.5, trend_decline_days=1,
    enable_ohlcv_filter=True, enable_early_exit=True,
)


def analyze_grid_holding_days():
    data = load_data("data/1d/512890.SH_hfq.parquet")
    close = data["Close"]; open_ = data["Open"]
    high = data["High"]; low = data["Low"]; volume = data["Volume"]

    ind = compute_polyfit_switch_indicators(
        close, fit_window_days=252, ma_windows=[20, 60],
        trend_window_days=BEST_GRID["trend_window_days"],
        vol_window_days=BEST_GRID["vol_window_days"],
    )
    idx_f = ind.index
    cl = close.loc[idx_f]; op = open_.reindex(idx_f)
    hi = high.reindex(idx_f); lo = low.reindex(idx_f); vo = volume.reindex(idx_f)

    grid_sig_params = {k: v for k, v in BEST_GRID.items()
                       if k not in ("trend_window_days", "vol_window_days")}
    e_g, x_g, s_g = generate_grid_signals(
        cl.values, ind["PolyDevPct"].values, ind["PolyDevTrend"].values,
        ind["RollingVolPct"].values, ind["PolyBasePred"].values, **grid_sig_params,
    )

    # 遍历所有 Grid 交易，标记持仓日
    holding_days = []  # 所有持仓日（不含入场日和离场日）
    hold_features = []

    in_grid = False
    entry_bar = -1

    for i in range(len(cl)):
        if e_g[i]:
            in_grid = True
            entry_bar = i
            continue  # 入场日本身不算持仓日
        if x_g[i]:
            if in_grid and i > entry_bar + 1:
                # 标记持仓日（入场次日 → 离场前日）
                for j in range(entry_bar + 1, i):
                    holding_days.append(j)
            in_grid = False
            entry_bar = -1
            continue
        # Grid 入场当天即开始持仓，次日开始可以 T+0
        if in_grid and i > entry_bar:
            holding_days.append(i)

    print(f"Grid 持仓日: {len(holding_days)} 天  (总交易: {e_g.sum()} 笔)")

    # ── 提取持仓日的日内特征 ──
    records = []
    for i in holding_days:
        if i < 20: continue  # 需要足够的前置数据

        op_i = op.values[i]; cl_i = cl.values[i]
        hi_i = hi.values[i]; lo_i = lo.values[i]
        vo_i = vo.values[i]

        # 日内特征
        intraday_hi = (hi_i - op_i) / op_i      # 日内最大涨幅(相对开盘)
        intraday_lo = (lo_i - op_i) / op_i      # 日内最大跌幅(相对开盘)
        intraday_ret = (cl_i - op_i) / op_i     # 开→收
        intraday_range = (hi_i - lo_i) / op_i    # 全日振幅
        gap = (op_i - cl.values[i-1]) / cl.values[i-1]  # 开盘跳空

        # 前一日特征
        prev_range = (hi.values[i-1] - lo.values[i-1]) / cl.values[i-1]
        prev_ret = (cl.values[i-1] - op.values[i-1]) / op.values[i-1]
        prev_upper_wick = (hi.values[i-1] - max(cl.values[i-1], op.values[i-1])) / cl.values[i-1]
        prev_lower_wick = (min(cl.values[i-1], op.values[i-1]) - lo.values[i-1]) / cl.values[i-1]

        # 连涨/连跌
        cons_up = 0
        for j in range(i-1, max(0, i-11), -1):
            if cl.values[j] > cl.values[j-1]: cons_up += 1
            else: break

        # 量比 (5日)
        vol_ma5 = np.mean(vo.values[max(0,i-5):i])
        vol_ratio = vo_i / vol_ma5 if vol_ma5 > 0 else 1

        # 5日波动率
        if i >= 5:
            rets5 = np.diff(cl.values[i-5:i+1]) / cl.values[i-5:i]
            vol5 = np.std(rets5)
        else:
            vol5 = prev_range

        # 距基线偏离
        dp = ind["PolyDevPct"].values[i]
        dp_trend = ind["PolyDevTrend"].values[i]
        base = ind["PolyBasePred"].values[i]

        # 持仓天数
        # 找到当前持仓的入场日
        hold_days = 0
        for j in range(i-1, -1, -1):
            if e_g[j]:
                hold_days = i - j
                break

        records.append({
            "date": idx_f[i], "bar": i,
            "gap": gap,
            "intraday_hi": intraday_hi, "intraday_lo": intraday_lo,
            "intraday_ret": intraday_ret, "intraday_range": intraday_range,
            "prev_range": prev_range, "prev_ret": prev_ret,
            "prev_upper_wick": prev_upper_wick, "prev_lower_wick": prev_lower_wick,
            "cons_up": cons_up, "vol_ratio": vol_ratio, "vol5": vol5,
            "dp": dp, "dp_trend": dp_trend, "hold_days": hold_days,
        })

    df = pd.DataFrame(records)
    n = len(df)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  Grid 持仓日 ({n}天) 日内特征")
    print(f"{'═' * 80}")

    for label, col, fmt in [
        ("开盘跳空(前收→开)", "gap", "+.2%"),
        ("日内最大涨幅(开→高)", "intraday_hi", "+.2%"),
        ("日内最大跌幅(开→低)", "intraday_lo", "+.2%"),
        ("开→收", "intraday_ret", "+.2%"),
        ("全日振幅", "intraday_range", "+.2%"),
    ]:
        print(f"  {label:<22s} 均值{df[col].mean():>+7.2%}  中位{df[col].median():>+7.2%}  "
              f"std{df[col].std():>6.2%}  Q25={df[col].quantile(0.25):>+6.2%}  Q75={df[col].quantile(0.75):>+6.2%}")

    # ══════════════════════════════════════════════════════════════
    # T+0 策略测试
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  T+0 策略对比")
    print(f"{'═' * 80}")

    # 策略 A: 固定偏移卖单，收盘买回
    # 策略 B: 基于前日振幅的倍数卖单
    # 策略 C: 条件卖单（不同市场环境不同偏移）
    # 策略 D: 买入-卖出配对（先买后卖 vs 先卖后买）

    print(f"\n  ── 策略 A: 固定卖单偏移 + 收盘买回 ──")
    print(f"  {'卖单偏移':>10s} {'触发率':>7s} {'成交均价':>9s} {'年均触发':>8s} {'累计收益':>9s}")
    print(f"  {'─'*10} {'─'*7} {'─'*9} {'─'*8} {'─'*9}")

    for sell_off in [0.002, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015]:
        triggered = df["intraday_hi"] >= sell_off
        trig_rate = triggered.mean()
        if trig_rate > 0:
            avg_gain = df.loc[triggered, "intraday_hi"].mean()
            # 触发日的收益: sell_off - (close-open)/open  ≈ sell_off (因为卖在限价=开盘×1+offset)
            # 简化: 收益 = sell_off (卖在限价) + 收盘买回 = sell_off - intraday_ret
            # 但实际卖价 = op*(1+sell_off)，买回价 = close
            # 收益 = (op*(1+sell_off) - close) / op = sell_off - (close-op)/op = sell_off - intraday_ret
            profit_per_trig = sell_off - df.loc[triggered, "intraday_ret"].mean()
            annual_trig = trig_rate * 252  # 年均触发次数
            total_profit = profit_per_trig * trig_rate * n  # 全期累计
        else:
            profit_per_trig = 0; annual_trig = 0; total_profit = 0

        print(f"  {f'+{sell_off:.1%}':>10s} {trig_rate:>6.0%} "
              f"{f'+{sell_off:.1%}':>9s} {annual_trig:>7.0f}天 {total_profit:>+8.2%}")

    # ══════════════════════════════════════════════════════════════
    # 因子分析：什么条件下 T+0 更可能成功
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  T+0 因子分析（sell=+0.5% 为例）")
    print(f"{'═' * 80}")

    sell_target = 0.005
    df["t0_triggered"] = df["intraday_hi"] >= sell_target
    df["t0_profit"] = 0.0
    df.loc[df["t0_triggered"], "t0_profit"] = (
        sell_target - df.loc[df["t0_triggered"], "intraday_ret"]
    )

    trig_rate_all = df["t0_triggered"].mean()
    print(f"  整体触发率: {trig_rate_all:.1%}  条件触发时平均收益: {df.loc[df['t0_triggered'], 't0_profit'].mean():+.2%}")
    print(f"\n  {'因子条件':<32s} {'样本':>6s} {'触发率':>7s} {'触发收益':>8s} {'推荐偏移':>8s}")
    print(f"  {'─'*32} {'─'*6} {'─'*7} {'─'*8} {'─'*8}")

    factor_conditions = [
        ("全部持仓日", slice(None)),
        ("前日振幅>1.5%(高波动)", df["prev_range"] > 0.015),
        ("前日振幅<0.8%(低波动)", df["prev_range"] < 0.008),
        ("前日收阳", df["prev_ret"] > 0),
        ("前日收阴", df["prev_ret"] < 0),
        ("连涨>=3天", df["cons_up"] >= 3),
        ("放量(vol>1.5x)", df["vol_ratio"] > 1.5),
        ("缩量(vol<0.7x)", df["vol_ratio"] < 0.7),
        ("高波动(vol5>1.5%)", df["vol5"] > 0.015),
        ("低波动(vol5<0.8%)", df["vol5"] < 0.008),
        ("dp<-3%(深度超卖)", df["dp"] < -0.03),
        ("dp -1%~0%(接近基线)", (df["dp"] > -0.01) & (df["dp"] < 0)),
        ("dp_trend>0(回归中)", df["dp_trend"] > 0),
        ("dp_trend<0(继续偏离)", df["dp_trend"] < 0),
        ("持仓<5天(刚入场)", df["hold_days"] < 5),
        ("持仓>20天(长期持仓)", df["hold_days"] > 20),
        ("前日长上影>0.5%", df["prev_upper_wick"] > 0.005),
        ("前日长下影>0.5%", df["prev_lower_wick"] > 0.005),
        ("跳空高开>0.3%", df["gap"] > 0.003),
        ("跳空低开<-0.3%", df["gap"] < -0.003),
    ]

    best_rules = []
    for label, cond in factor_conditions:
        sub = df[cond]
        if len(sub) < 20: continue
        trig = sub["t0_triggered"].mean()
        prof = sub.loc[sub["t0_triggered"], "t0_profit"].mean() if trig > 0 else 0
        # 找此条件下的最优偏移
        best_off = 0.005
        best_score = -999
        for off in [0.002, 0.003, 0.005, 0.008, 0.010, 0.012]:
            t = sub["intraday_hi"] >= off
            if t.mean() == 0: continue
            p = off - sub.loc[t, "intraday_ret"].mean()
            score = t.mean() * p
            if score > best_score:
                best_score = score
                best_off = off
        print(f"  {label:<32s} {len(sub):>6d} {trig:>6.0%} {prof:>+7.2%} {f'+{best_off:.1%}':>8s}")
        best_rules.append({"condition": label, "n": len(sub), "trig_rate": trig,
                          "best_offset": best_off, "score": best_score})

    # ══════════════════════════════════════════════════════════════
    # 推荐
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  推荐 Grid T+0 策略")
    print(f"{'═' * 80}")

    # 找不同条件下最优偏移的规律
    rules_df = pd.DataFrame(best_rules)
    high_off = rules_df[rules_df["best_offset"] >= 0.008]
    low_off = rules_df[rules_df["best_offset"] <= 0.003]

    print(f"""
  Grid 持仓日特征（{n}天）：
    - 日内平均振幅 {df['intraday_range'].mean():.2%}，中位 {df['intraday_range'].median():.2%}
    - 日内平均最高涨幅(开→高) {df['intraday_hi'].mean():.2%}
    - 开→收平均 {df['intraday_ret'].mean():.2%}（均值回复特征：开盘后小幅上涨）

  推荐因子：

  1. 【波动率预测】前日振幅是当日振幅的最佳预测因子
     - 高波动日(前日振幅>1.5%) → 当天振幅更大 → 偏移可放宽到 +0.8%~+1.0%
     - 低波动日(前日振幅<0.8%) → 当天振幅更小 → 偏移收窄到 +0.2%~+0.3%

  2. 【超卖深度】dp 越深（越超卖），反弹动能越强
     - dp<-3%(深度超卖) → T+0 最有利，偏移可到 +0.8%
     - dp 接近基线(-1%~0) → 震荡为主，偏移 +0.3%

  3. 【趋势方向】dp_trend(偏离趋势)指示价格在回归还是继续偏离
     - dp_trend>0(回归中) → 日内继续回归概率大 → 可用 +0.5%
     - dp_trend<0(继续偏离) → 等待为主 → 偏移 +0.3%

  4. 【持仓阶段】刚入场 vs 长期持仓
     - 持仓<5天(刚入场) → 反弹动能最强 → +0.5%~+0.8%
     - 持仓>20天(长期) → 可能已接近离场 → 保守 +0.3%

  5. 【量能确认】缩量日更适合 T+0
     - 缩量(vol<0.7x) → 震荡市 → T+0 成功率高
     - 放量(vol>1.5x) → 趋势市 → T+0 风险大

  策略设计：
  ┌─────────────────────────────────────────────────────────┐
  │ Grid 持仓日的 T+0 偏移 = 基础偏移 × 波动率乘数           │
  │                                                         │
  │ 基础偏移 = +0.5% (固定)                                  │
  │                                                         │
  │ 波动率乘数：                                             │
  │   前日振幅 > 1.5%  → ×1.5  (+0.75% 偏移)               │
  │   前日振幅 < 0.8%  → ×0.5  (+0.25% 偏移)               │
  │   其他              → ×1.0  (+0.50% 偏移)               │
  │                                                         │
  │ 增强乘数（可选）：                                       │
  │   dp < -3%         → ×1.3  (深度超卖，反弹更强)        │
  │   持仓 < 5天        → ×1.2  (刚入场，动能最强)          │
  │   缩量(vol<0.7x)   → ×1.1  (震荡市有利)               │
  │   放量(vol>1.5x)   → ×0.5  (趋势市回避)               │
  │                                                         │
  │ 执行：                                                   │
  │   开盘挂限价卖单 open×(1 + 综合偏移)                    │
  │   成交 → 收盘买回 (T+0 完成)                            │
  │   未成交 → 继续持有                                     │
  └─────────────────────────────────────────────────────────┘
""")

    # 测试综合策略
    print(f"  ── 综合策略回测 ──")
    df["adj_offset"] = 0.005  # 基础

    # 波动率乘数
    df.loc[df["prev_range"] > 0.015, "adj_offset"] = 0.0075
    df.loc[df["prev_range"] < 0.008, "adj_offset"] = 0.0025

    # 深度超卖
    df.loc[df["dp"] < -0.03, "adj_offset"] *= 1.3

    # 刚入场
    df.loc[df["hold_days"] < 5, "adj_offset"] *= 1.2

    for label, strat_col in [
        ("固定+0.5%", "t0_triggered"),  # 已有的
    ]:
        pass

    # 综合：逐日计算
    total_profit_smart = 0.0
    n_trig_smart = 0
    for _, r in df.iterrows():
        off = r["adj_offset"]
        if r["intraday_hi"] >= off:
            profit = off - r["intraday_ret"]
            total_profit_smart += profit
            n_trig_smart += 1

    # 固定+0.5%
    fixed_05_trig = (df["intraday_hi"] >= 0.005).sum()
    fixed_05_profit = df.loc[df["intraday_hi"] >= 0.005, "t0_profit"].sum()

    print(f"  固定+0.5%: 触发{fixed_05_trig}次({fixed_05_trig/n:.0%})  累计收益{fixed_05_profit:+.2%}")
    print(f"  自适应偏移: 触发{n_trig_smart}次({n_trig_smart/n:.0%})  累计收益{total_profit_smart:+.2%}")


if __name__ == "__main__":
    analyze_grid_holding_days()
