"""对比周五三种入场方式的执行价格 (以周四收盘为基准)."""
import pandas as pd
import numpy as np

for name in ["512890", "510880"]:
    df = pd.read_parquet(f"data/1d/{name}.SH_hfq.parquet")
    df.index = pd.to_datetime(df.index)
    cl = df["Close"].loc["2019-01-01":"2026-04-30"]
    op = df["Open"].loc["2019-01-01":"2026-04-30"]
    common = cl.index.intersection(op.index)
    cl, op = cl.loc[common], op.loc[common]
    wd = cl.index.dayofweek

    # 周五数据
    fri = wd == 4
    thu = wd == 3

    # 三种入场价格 (以周四收盘 = 1.0 为基准)
    # A: 周五 Open (当前方案)
    thu_cl = cl[thu].values
    fri_op = op[fri].values
    fri_cl = cl[fri].values

    # 对齐: 每个周四配对下一个周五
    thu_dates = cl.index[thu]
    fri_dates = cl.index[fri]
    paired_thu_cl = []
    paired_fri_op = []
    paired_fri_cl = []
    paired_mon_op = []

    for t in thu_dates:
        next_fri = fri_dates[fri_dates > t]
        if len(next_fri) == 0:
            continue
        nf = next_fri[0]
        paired_thu_cl.append(cl.loc[t])
        paired_fri_op.append(op.loc[nf])
        paired_fri_cl.append(cl.loc[nf])
        # 下一个周一 Open
        next_mon = op.index[(op.index > nf) & (op.index.dayofweek == 0)]
        if len(next_mon) > 0:
            paired_mon_op.append(op.loc[next_mon[0]])
        else:
            paired_mon_op.append(np.nan)

    paired_thu_cl = np.array(paired_thu_cl)
    paired_fri_op = np.array(paired_fri_op)
    paired_fri_cl = np.array(paired_fri_cl)
    paired_mon_op = np.array(paired_mon_op)

    # 以周四收盘为 1.0 的相对价格
    rel_fri_op = paired_fri_op / paired_thu_cl
    rel_fri_cl = paired_fri_cl / paired_thu_cl
    rel_mon_op = paired_mon_op / paired_thu_cl

    print(f"\n{'='*60}")
    print(f"  {name} — 周四收盘=1.0 基准下的入场价格 (N={len(paired_thu_cl)})")
    print(f"{'='*60}")

    for label, rel in [
        ("周五 Open 入场  (当前)", rel_fri_op),
        ("周五 Close 入场 (提议)", rel_fri_cl),
        ("周一 Open 入场  (延迟)", rel_mon_op),
    ]:
        valid = rel[~np.isnan(rel)]
        below_1 = (valid < 1.0).mean()
        print(f"  {label}")
        print(f"    均值: {np.mean(valid):.5f}  (相对周四收盘 {np.mean(valid)-1:+.4%})")
        print(f"    中位: {np.median(valid):.5f}")
        print(f"    最低: {np.min(valid):.5f}  ({np.min(valid)-1:+.3%})")
        print(f"    <1.0 (低于周四收盘): {below_1:.0%}")
        print()

    # 关键：周五 Close 比周五 Open 便宜的概率
    fri_cl_cheaper = (paired_fri_cl < paired_fri_op).mean()
    fri_cl_saving = np.mean(paired_fri_op / paired_fri_cl - 1)

    print(f"  ── 周五 Close vs 周五 Open ──")
    print(f"  周五 Close < 周五 Open 的比例: {fri_cl_cheaper:.0%}")
    print(f"  若 Close 更便宜时, 平均节省: {fri_cl_saving:+.4%}")

    # 周五 Open→Close 的分段分布
    fri_intra = paired_fri_cl / paired_fri_op - 1
    print(f"\n  ── 周五日内涨跌分布 (Open→Close) ──")
    for lo, hi, label in [
        (-1, -0.02, "跌 >2%"),
        (-0.02, -0.01, "跌 1~2%"),
        (-0.01, -0.005, "跌 0.5~1%"),
        (-0.005, 0, "跌 0~0.5%"),
        (0, 0.005, "涨 0~0.5%"),
        (0.005, 0.01, "涨 0.5~1%"),
        (0.01, 0.02, "涨 1~2%"),
        (0.02, 1, "涨 >2%"),
    ]:
        n = ((fri_intra > lo) & (fri_intra <= hi)).sum()
        pct = n / len(fri_intra)
        bar = "█" * max(1, int(pct * 50))
        print(f"  {label:<10}  {n:>3}天 ({pct:>4.0%})  {bar}")

    # 周五 Close 入场 → 周一 Open 的持有收益
    fri_cl_to_mon = paired_mon_op / paired_fri_cl - 1
    valid_f2m = fri_cl_to_mon[~np.isnan(fri_cl_to_mon)]
    print(f"\n  ── 周五 Close 入场后 → 周一 Open 持有收益 ──")
    print(f"  均值: {np.mean(valid_f2m):+.4%}")
    print(f"  胜率: {(valid_f2m > 0).mean():.0%}")

print("\nDone.")
