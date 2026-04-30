"""VectorBT report generation — plots, stats, and index HTML."""

import os

import numpy as np
import pandas as pd
import vectorbt as vbt


def generate_portfolio_reports(
    close: pd.Series,
    entries: np.ndarray,
    exits: np.ndarray,
    sizes: np.ndarray,
    name: str,
    params: dict,
    reports_dir: str = "reports",
) -> dict:
    """Generate VectorBT plots and stats for a single parameter set."""
    pf = vbt.Portfolio.from_signals(
        close,
        entries=pd.Series(entries, index=close.index),
        exits=pd.Series(exits, index=close.index),
        size=pd.Series(sizes, index=close.index),
        size_type="percent",
        init_cash=100_000.0,
        freq="D",
    )

    safe_name = name.replace(" ", "_")
    out_dir = f"{reports_dir}/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Stats CSV ---
    stats = pf.stats()
    stats_path = f"{out_dir}/{safe_name}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  Stats → {stats_path}")

    # --- Main overview plot (interactive HTML) ---
    fig = pf.plot()
    fig_path = f"{out_dir}/{safe_name}_overview.html"
    fig.write_html(fig_path)
    print(f"  Overview plot → {fig_path}")

    # --- Individual plots ---
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
            pass

    return {
        "name": name,
        "params": params,
        "stats": stats.to_dict(),
        "overview_html": f"{safe_name}_overview.html",
        "plots": saved_plots,
        "out_dir": safe_name,
    }


def build_index_html(reports: list[dict], reports_dir: str = "reports") -> str:
    """Build a simple index.html linking to all reports."""
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
