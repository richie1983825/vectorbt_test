"""VectorBT 报告生成模块。

使用 VectorBT 内置的绘图和统计功能，为指定参数组合生成：
  - 统计摘要 CSV
  - 交互式 HTML 图表（概览、收益曲线、回撤、交易明细等）
  - 汇总索引页面（index.html），链接到所有策略的报告
"""

import os

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_portfolio_reports(
    close: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    name: str,                   # 策略名称（用于目录命名）
    params: dict,                # 参数 dict（用于报告中显示）
    reports_dir: str = "reports",
    open_: pd.Series | None = None,  # 可选，开盘价用于 next-bar Open 执行
) -> dict:
    """为单组参数生成完整的 VectorBT 回测报告。

    生成内容：
      - 统计 CSV（stats）
      - 概览图（Overview）
      - 累积收益、回撤、水下、交易、交易盈亏、资产价值 等细分图表

    Args:
        open_: 传入则使用 next-bar Open 作为成交价
    Returns:
        dict: 包含报告路径和统计数据的元信息，供 build_index_html 使用。
    """
    idx = close.index
    if open_ is not None:
        fill_price = open_.shift(-1).reindex(idx)
    else:
        fill_price = close

    pf = vbt.Portfolio.from_signals(
        fill_price,
        entries=pd.Series(entries, index=idx),
        exits=pd.Series(exits, index=idx),
        size=pd.Series(sizes, index=idx),
        size_type="percent",
        init_cash=100_000.0,
        freq="D",
    )

    safe_name = name.replace(" ", "_")
    out_dir = f"{reports_dir}/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ── 统计 CSV ──
    stats = pf.stats()
    stats_path = f"{out_dir}/{safe_name}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  Stats → {stats_path}")

    # ── 概览图（交互式 HTML）──
    try:
        fig = pf.plot()
        fig_path = f"{out_dir}/{safe_name}_overview.html"
        fig.write_html(fig_path)
        print(f"  Overview plot → {fig_path}")
    except Exception as e:
        print(f"  Overview plot skipped: {e}")

    # ── 细分图表 ──
    plot_methods = [
        ("cum_returns", pf.plot_cum_returns),
        ("drawdowns", pf.plot_drawdowns),
        ("underwater", pf.plot_underwater),
        ("trades", pf.plot_trades),
        ("trade_pnl", pf.plot_trade_pnl),
        ("asset_value", pf.plot_asset_value),
    ]
    saved_plots = []
    for plot_name, plot_fn in plot_methods:
        try:
            f = plot_fn()
            p = f"{out_dir}/{safe_name}_{plot_name}.html"
            f.write_html(p)
            saved_plots.append((plot_name, f"{safe_name}_{plot_name}.html"))
        except Exception:
            pass  # 某些图表可能在无交易时失败，静默跳过

    return {
        "name": name,
        "params": params,
        "stats": stats.to_dict(),
        "overview_html": f"{safe_name}_overview.html",
        "plots": saved_plots,
        "out_dir": safe_name,
    }


def build_index_html(reports: list[dict], reports_dir: str = "reports") -> str:
    """构建汇总索引页面 HTML。

    将所有策略报告的关键指标和链接汇总到一个表格中，
    方便在浏览器中快速浏览和对比。

    Args:
        reports: generate_portfolio_reports 返回的 dict 列表
        reports_dir: 报告根目录（用于相对路径引用）

    Returns:
        完整的 HTML 字符串。
    """
    rows = []
    for r in reports:
        s = r["stats"]
        param_str = "  |  ".join(f"{k}={v}" for k, v in r["params"].items())
        out_dir = r.get("out_dir", r["name"].replace(" ", "_"))
        rows.append(f"""
        <tr>
            <td><strong>{r['name']}</strong></td>
            <td>{param_str}</td>
            <td>{s.get('Total Return [%]', 0):.2f}%</td>
            <td>{s.get('Sharpe Ratio', 0):.3f}</td>
            <td>{s.get('Max Drawdown [%]', 0):.2f}%</td>
            <td>{s.get('Total Trades', 0)}</td>
            <td><a href="{out_dir}/{r['overview_html']}">Overview</a></td>
            <td>{" | ".join(f'<a href="{out_dir}/{p[1]}">{p[0]}</a>' for p in r['plots'])}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>VectorBT Strategy Reports — 512890.SH</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; }}
  a {{ color: #1a73e8; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  h1 {{ margin-bottom: 0.5em; }}
  .subtitle {{ color: #666; margin-bottom: 1.5em; }}
</style>
</head>
<body>
<h1>VectorBT Dynamic Grid Strategy Reports</h1>
<p class="subtitle">512890.SH 后复权  |  Best-parameter backtests from grid search</p>
<table>
<thead>
<tr><th>Strategy</th><th>Parameters</th><th>Total Return</th><th>Sharpe</th>
<th>Max DD</th><th>Trades</th><th>Overview</th><th>Detail Plots</th></tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>"""
