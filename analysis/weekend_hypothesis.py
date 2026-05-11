"""验证周五效应假设：管理费 + 规避周末风险.

检验内容:
1. 管理费日损耗量级 vs 周五实际跌幅
2. 周五收盘→周一开盘 的跳空 (周末持有收益) — 检验"风险补偿"
3. 周四收盘→周一开盘 完整周末跨期收益
4. 周五日内分段: 早盘 vs 尾盘 (卖压集中在尾盘?)
5. 510880 对比: 如果管理费逻辑成立，510880 应有同样模式
"""
import pandas as pd
import numpy as np

s1 = pd.read_parquet("data/1d/512890.SH_hfq.parquet")
s2 = pd.read_parquet("data/1d/510880.SH_hfq.parquet")
s1.index = pd.to_datetime(s1.index)
s2.index = pd.to_datetime(s2.index)

for name, df in [("512890", s1), ("510880", s2)]:
    cl = df["Close"].loc["2019-01-01":"2026-04-30"]
    op = df["Open"].loc["2019-01-01":"2026-04-30"]
    common = cl.index.intersection(op.index)
    cl, op = cl.loc[common], op.loc[common]

    wd = cl.index.dayofweek

    # ── 管理费日损耗 ──────────────────────────
    daily_fee = 0.006 / 365
    weekend_fee = 3 * daily_fee  # 周五→周一 3 天

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  管理费 0.6%/年:")
    print(f"    每日: {daily_fee:.5%}")
    print(f"    周末 3 天: {weekend_fee:.5%}")

    # ── 核心: 跨周末的各种持有方式 ────────────
    # A. 周五日内 (Open→Close): 卖压
    fri_mask = wd == 4
    fri_intraday = np.log(cl / op)
    fri_intra_fri = fri_intraday[fri_mask]

    # B. 周五收盘 → 周一开盘: 周末持有的"被补偿"部分
    fri_close_to_mon_open = np.log(op.shift(-1) / cl)
    fri_to_mon_overnight = fri_close_to_mon_open[fri_mask]

    # C. 周一~周四 隔夜 (收盘→次日开盘)
    mon_to_thu_mask = (wd >= 0) & (wd <= 3)
    normal_overnight = fri_close_to_mon_open[mon_to_thu_mask]

    # D. 周四收盘 → 周五开盘: 周五开盘前的隔夜
    thu_mask = wd == 3
    thu_close_to_fri_open = fri_close_to_mon_open[thu_mask]

    # E. 周四收盘→下个周一开盘 (跳过周五，直接跨周末)
    thu_idx = cl.index[thu_mask]
    # 找到每个周四的"下一个周一开盘"：在 op 中找 >= 周四+3天 的第一个交易日开盘
    mon_open_vals = []
    thu_close_vals = []
    skipped = 0
    for ti in thu_idx:
        future_ops = op.loc[ti + pd.Timedelta(days=2):]
        if len(future_ops) >= 1 and future_ops.index[0] >= ti + pd.Timedelta(days=2):
            mon_open_vals.append(future_ops.iloc[0])
            thu_close_vals.append(cl.loc[ti])
        else:
            skipped += 1
    thu_cl_to_mon_op = np.log(np.array(mon_open_vals) / np.array(thu_close_vals))

    # F. 周三收盘 → 周四开盘 (普通隔夜，作为对照组)
    wed_mask = wd == 2
    wed_close = cl[wed_mask]
    thu_open = op.shift(-1).loc[wed_close.index]
    wed_cl_to_thu_op = np.log(thu_open.values / wed_close.values)

    print(f"\n  ── 跨周末各阶段收益 ──")
    print(f"  {'阶段':<28} {'均值':>8}  {'胜率':>6}  {'交易日':>6}")
    print(f"  {'─'*28} {'─'*8}  {'─'*6}  {'─'*6}")

    for label, ser in [
        ("周五日内 (开盘→收盘)", fri_intra_fri),
        ("周五收盘→周一开盘 (周末)", fri_to_mon_overnight),
        ("周一~周四 隔夜 (正常)", normal_overnight),
        ("周四收盘→周五开盘", thu_close_to_fri_open),
        ("周四收盘→周一开盘 (完整)", pd.Series(thu_cl_to_mon_op)),
    ]:
        print(f"  {label:<28} {ser.mean():>+8.4%}  {((ser>0).mean()):>5.0%}  {len(ser):>6}")

    # ── 关键对比 ─────────────────────────────
    print(f"\n  ── 关键对比 ──")
    fee_loss = weekend_fee

    # 如果假设成立: 周四收盘→周一开盘 ≈ 普通隔夜 × 3 (因为持有3天)
    avg_thu_cl_to_mon = thu_cl_to_mon_op.mean()
    avg_wed_cl_to_thu = wed_cl_to_thu_op.mean()
    print(f"  周四收→周一开 (持有3天):      {avg_thu_cl_to_mon:+.4%}")
    print(f"  周三收→周四开 (持有1天):      {avg_wed_cl_to_thu:+.4%}")
    print(f"  3× 普通隔夜 (作为参照):        {avg_wed_cl_to_thu*3:+.4%}")
    print(f"  周末实际 vs 理论3天:           {avg_thu_cl_to_mon - avg_wed_cl_to_thu*3:+.4%}")

    # 周五日内 vs 其他日内
    intraday_all = np.log(cl / op)
    week_intra = intraday_all[(wd >= 0) & (wd <= 3)]
    print(f"\n  周五日内均值:                  {fri_intra_fri.mean():+.4%}")
    print(f"  周一~周四日内均值:             {week_intra.mean():+.4%}")
    print(f"  日内差值 (周五-平日):          {fri_intra_fri.mean() - week_intra.mean():+.4%}")

    # 周五收盘→周一开盘 vs 其他隔夜
    print(f"\n  周五收→周一开 (周末隔夜):      {fri_to_mon_overnight.mean():+.4%}")
    print(f"  周一~周四 隔夜:                {normal_overnight.mean():+.4%}")
    print(f"  周末隔夜差值:                  {fri_to_mon_overnight.mean() - normal_overnight.mean():+.4%}")

    # ── 逐年: 周五日内 ───────────────────────
    print(f"\n  ── 逐年: 周五日内 vs 周五收→周一开 ──")
    print(f"  {'Yr':<6} {'周五日内':>9} {'周五收→周一开':>14}")
    for yr in range(2019, 2027):
        fy = fri_intraday[fri_intraday.index.year == yr]
        fom_y = fri_close_to_mon_open[fri_close_to_mon_open.index.year == yr]
        if len(fy) < 10:
            continue
        print(f"  {yr:<6} {fy.mean():>+9.4%} {fom_y.mean():>+14.4%}")

    # ── 结论 ─────────────────────────────────
    print(f"\n  ── 假设检验 ──")
    # H1: 管理费驱动 → 周五日内跌幅应 ≈ 3×日管理费 ≈ 0.005%
    #     实际周五日内 = -0.044% (来自上面全量分析)
    fee_contribution = weekend_fee / abs(fri_intra_fri.mean()) if fri_intra_fri.mean() != 0 else 0
    print(f"  管理费能解释的比例: {fee_contribution:.1%}")
    print(f"  → {'管理费解释力很弱' if fee_contribution < 0.3 else '管理费有一定解释力'}")

    # H2: 规避周末风险 → 周五收盘→周一开盘 应有正收益补偿
    #     因为有人周五卖出，周一要买回来
    print(f"  周五收盘→周一开盘 是否为正: {'是' if fri_to_mon_overnight.mean() > 0 else '否'}")

    # 净效果: 如果周四收盘持有到周一开盘
    print(f"  周四收盘→周一开盘: {avg_thu_cl_to_mon:+.4%}")
    print(f"  这相当于: 周四→周五隔夜 + 周五日内 + 周五→周一隔夜")

print("\nDone.")
