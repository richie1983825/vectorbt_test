"""VectorBT 报告生成模块。

使用 VectorBT 内置的绘图和统计功能，为指定参数组合生成：
  - 统计摘要 CSV
  - 交互式 HTML 图表（概览、收益曲线、回撤、交易明细等）
  - 汇总索引页面（index.html），链接到所有策略的报告
  - Polyfit-Switch 增强报告（K线+基线+买卖标记+模式分析）
"""

import os

import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# ══════════════════════════════════════════════════════════════════
# Polyfit-Switch 增强报告
# ══════════════════════════════════════════════════════════════════

def _plot_price_with_baseline(df, entries, exits, entry_modes, title,
                             fill_price=None):
    """绘制 K 线 + Polyfit 基线 + 买卖点标记。

    买卖标记使用实际成交价（次日 Open）而非基线值。
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    idx = df.index
    if fill_price is None:
        fill_price = pd.Series(df["Close"].values, index=idx)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=(title, "成交量"),
    )

    fig.add_trace(go.Candlestick(
        x=idx, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="K线", increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
    ), row=1, col=1)

    if "PolyBasePred" in df.columns:
        fig.add_trace(go.Scatter(
            x=idx, y=df["PolyBasePred"], mode="lines",
            name="Polyfit基线(252d)",
            line=dict(color="#FF9800", width=1.5),
        ), row=1, col=1)

    fp = fill_price.reindex(idx).values

    entry_idx = np.where(entries)[0]
    if len(entry_idx) > 0:
        grid_mask = entry_modes[entry_idx] == 1
        if grid_mask.any():
            gi = entry_idx[grid_mask]
            fig.add_trace(go.Scatter(
                x=idx[gi], y=fp[gi], mode="markers",
                name="Grid入场", marker=dict(symbol="triangle-up", size=8,
                color="#4CAF50", line=dict(color="#2E7D32", width=1)),
                text=[f"成交价: {fp[i]:.4f}" for i in gi],
            ), row=1, col=1)
        sw_mask = entry_modes[entry_idx] == 2
        if sw_mask.any():
            si = entry_idx[sw_mask]
            fig.add_trace(go.Scatter(
                x=idx[si], y=fp[si], mode="markers",
                name="Switch入场", marker=dict(symbol="triangle-up", size=10,
                color="#2196F3", line=dict(color="#1565C0", width=1)),
                text=[f"成交价: {fp[i]:.4f}" for i in si],
            ), row=1, col=1)

    exit_idx = np.where(exits)[0]
    if len(exit_idx) > 0:
        fig.add_trace(go.Scatter(
            x=idx[exit_idx], y=fp[exit_idx], mode="markers",
            name="离场", marker=dict(symbol="triangle-down", size=8,
            color="#F44336", line=dict(color="#C62828", width=1)),
            text=[f"成交价: {fp[i]:.4f}" for i in exit_idx],
        ), row=1, col=1)

    colors = ["#ef5350" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#26a69a"
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=idx, y=df["Volume"], name="成交量",
        marker=dict(color=colors, opacity=0.4),
    ), row=2, col=1)

    fig.update_layout(height=700, showlegend=True,
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def generate_polyfit_switch_report(
    df, close, entries, exits, sizes, entry_modes,
    params, name="Polyfit-Switch", reports_dir="reports",
    open_=None,
):
    """Polyfit-Switch 增强报告：K线+基线+Full vs Grid-only 模式对比。"""
    import plotly.graph_objects as go

    idx = close.index
    fill_price = open_.shift(-1).reindex(idx) if open_ is not None else close

    safe_name = name.replace(" ", "_")
    out_dir = f"{reports_dir}/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    pf = vbt.Portfolio.from_signals(
        fill_price,
        entries=pd.Series(entries, index=idx),
        exits=pd.Series(exits, index=idx),
        size=pd.Series(sizes, index=idx),
        size_type="percent", init_cash=100_000.0, freq="D",
    )

    e_grid = entries.copy()
    e_grid[entry_modes == 2] = False
    pf_grid = vbt.Portfolio.from_signals(
        fill_price,
        entries=pd.Series(e_grid, index=idx),
        exits=pd.Series(exits, index=idx),
        size=pd.Series(sizes, index=idx),
        size_type="percent", init_cash=100_000.0, freq="D",
    )

    full_ret = pf.total_return()
    grid_ret = pf_grid.total_return()
    bh_ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    switch_marginal = full_ret - grid_ret
    grid_trades = int((entry_modes == 1).sum())
    switch_trades = int((entry_modes == 2).sum())

    print(f"\n  {'='*60}")
    print(f"  {name} — Full vs Grid-only 模式分析")
    print(f"  {'='*60}")
    print(f"  BH return:              {bh_ret:+.2%}")
    print(f"  Full (Grid+Switch):     {full_ret:+.2%}  (α={full_ret-bh_ret:+.2%})")
    print(f"  Grid-only:              {grid_ret:+.2%}  (α={grid_ret-bh_ret:+.2%})")
    print(f"  Switch 边际贡献:         {switch_marginal:+.2%}")
    print(f"  Grid 交易: {grid_trades}   Switch 交易: {switch_trades}")
    print(f"  Full MaxDD: {pf.max_drawdown():.2%}  Grid-only MaxDD: {pf_grid.max_drawdown():.2%}")

    stats = pf.stats()
    stats["Switch_marginal_return"] = switch_marginal
    stats["Grid_only_return"] = grid_ret
    stats["Grid_trades"] = grid_trades
    stats["Switch_trades"] = switch_trades
    stats_path = f"{out_dir}/{safe_name}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  Stats -> {stats_path}")

    saved_plots = []

    # ── 整合概览：baseline + cum_returns + trade_pnl 三图同步 ──
    try:
        from plotly.subplots import make_subplots

        recs = pf.trades.records_readable
        trade_returns_pct = []
        trade_dates = []
        # 构建 exit_fill_date → {entry info, exit_reason} 映射
        exit_info_map: dict = {}
        if len(recs) > 0:
            for i in range(len(recs)):
                r = float(recs["Return"].iloc[i])
                if not pd.notna(recs["Return"].iloc[i]):
                    continue
                trade_returns_pct.append(r * 100)
                ets = recs["Exit Timestamp"].iloc[i]
                exit_dt = pd.Timestamp(ets) if pd.notna(ets) else idx[-1]
                trade_dates.append(exit_dt)
                entry_dt = pd.Timestamp(recs["Entry Timestamp"].iloc[i]) if pd.notna(recs["Entry Timestamp"].iloc[i]) else None
                entry_px = float(recs["Avg Entry Price"].iloc[i])
                exit_px = float(recs["Avg Exit Price"].iloc[i])

                # 确定入场模式：在 idx 中找到入场日期对应的 entry_modes
                emode = 1  # default Grid
                if entry_dt is not None and entry_dt in idx:
                    e_pos = idx.get_loc(entry_dt)
                    if e_pos < len(entry_modes):
                        emode = int(entry_modes[e_pos])

                # 确定离场原因
                if emode == 2:  # Switch
                    if r < 0:
                        reason = "追踪止损"
                    else:
                        reason = "趋势结束"
                else:  # Grid
                    hd = (exit_dt - entry_dt).days if entry_dt else 0
                    if hd >= 45:
                        reason = "到期平仓"
                    elif r > 0:
                        reason = "止盈离场"
                    else:
                        reason = "止损离场"

                exit_info_map[exit_dt] = dict(
                    entry_dt=entry_dt, entry_px=entry_px, exit_px=exit_px,
                    return_pct=r * 100, mode=emode, reason=reason,
                )

        fig_ov = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.20, 0.15, 0.20],
            vertical_spacing=0.03,
            subplot_titles=(
                f"{name} - K线 + Polyfit基线 + 买卖标记",
                "累积收益 (Full vs Grid-only vs BH)",
                "持仓仓位",
                "逐笔交易盈亏 (%)",
            ),
        )

        # 对齐 df 到 idx，避免 df_report (base_ind) 和 idx_s (switch_ind) 长度不一致
        df_aligned = df.loc[idx]

        # Row 1: K线 + 基线 + 买卖标记
        fig_ov.add_trace(go.Candlestick(
            x=idx, open=df_aligned["Open"], high=df_aligned["High"],
            low=df_aligned["Low"], close=df_aligned["Close"],
            name="K线", increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
        ), row=1, col=1)
        if "PolyBasePred" in df.columns:
            fig_ov.add_trace(go.Scatter(
                x=idx, y=df_aligned["PolyBasePred"], mode="lines",
                name="Polyfit基线", line=dict(color="#FF9800", width=1.5),
            ), row=1, col=1)
        fp_vals = fill_price.reindex(idx).values
        n = len(idx)

        # ── 入场标记：放在成交日期（信号 bar + 1） ──
        e_idx = np.where(entries)[0]
        # Grid 入场
        e_grid = [(i, idx[i+1], fp_vals[i]) for i in e_idx
                  if entry_modes[i] == 1 and i+1 < n and not np.isnan(fp_vals[i])]
        if e_grid:
            fig_ov.add_trace(go.Scatter(
                x=[x[1] for x in e_grid], y=[x[2] for x in e_grid], mode="markers",
                name="Grid入场", marker=dict(symbol="triangle-up", size=8, color="#4CAF50"),
                text=[f"Grid入场<br>日期: {x[1].date()}<br>成交价: {x[2]:.4f}" for x in e_grid],
            ), row=1, col=1)
        # Switch 入场
        e_sw = [(i, idx[i+1], fp_vals[i]) for i in e_idx
                if entry_modes[i] == 2 and i+1 < n and not np.isnan(fp_vals[i])]
        if e_sw:
            fig_ov.add_trace(go.Scatter(
                x=[x[1] for x in e_sw], y=[x[2] for x in e_sw], mode="markers",
                name="Switch入场", marker=dict(symbol="triangle-up", size=10, color="#2196F3"),
                text=[f"Switch入场<br>日期: {x[1].date()}<br>成交价: {x[2]:.4f}" for x in e_sw],
            ), row=1, col=1)

        # ── 离场标记：放在成交日期，标注原因 ──
        x_idx = np.where(exits)[0]
        x_data = []
        for j in x_idx:
            if j+1 >= n or np.isnan(fp_vals[j]):
                continue
            fill_dt = idx[j+1]
            info = exit_info_map.get(fill_dt)
            if info is not None:
                x_data.append((fill_dt, fp_vals[j], info))
            else:
                x_data.append((fill_dt, fp_vals[j], None))
        if x_data:
            fig_ov.add_trace(go.Scatter(
                x=[d[0] for d in x_data], y=[d[1] for d in x_data], mode="markers",
                name="离场", marker=dict(symbol="triangle-down", size=8, color="#F44336"),
                text=[f"{d[2]['reason']}<br>日期: {d[0].date()}<br>成交价: {d[1]:.4f}<br>收益: {d[2]['return_pct']:+.2f}%"
                      if d[2] else f"离场<br>日期: {d[0].date()}<br>成交价: {d[1]:.4f}"
                      for d in x_data],
            ), row=1, col=1)

        # Row 2: 累积收益
        bh_cum = close / close.iloc[0] - 1
        eq_f = pf.value() / pf.value().iloc[0] - 1
        eq_g = pf_grid.value() / pf_grid.value().iloc[0] - 1
        fig_ov.add_trace(go.Scatter(
            x=idx, y=bh_cum, mode="lines", name=f"BH ({bh_ret:+.1%})",
            line=dict(color="gray", width=1, dash="dot"),
        ), row=2, col=1)
        fig_ov.add_trace(go.Scatter(
            x=eq_f.index, y=eq_f, mode="lines", name=f"Full ({full_ret:+.1%})",
            line=dict(color="#2196F3", width=2),
        ), row=2, col=1)
        fig_ov.add_trace(go.Scatter(
            x=eq_g.index, y=eq_g, mode="lines", name=f"Grid-only ({grid_ret:+.1%})",
            line=dict(color="#4CAF50", width=1.5, dash="dash"),
        ), row=2, col=1)

        # Row 3: 持仓仓位（Full 策略）
        pos_series = pf.gross_exposure().reindex(idx).fillna(0)
        pos_colors = ["#2196F3" if v > 0 else "rgba(0,0,0,0)" for v in pos_series.values]
        fig_ov.add_trace(go.Bar(
            x=idx, y=pos_series.values,
            marker_color=pos_colors, marker_line_width=0,
            name="持仓",
        ), row=3, col=1)

        # Row 4: 逐笔交易盈亏
        if trade_returns_pct:
            colors = ["#4CAF50" if r > 0 else "#F44336" for r in trade_returns_pct]
            fig_ov.add_trace(go.Bar(
                x=trade_dates, y=trade_returns_pct, marker_color=colors,
                name="Trade PnL %",
            ), row=4, col=1)

        fig_ov.update_layout(
            height=1100,
            hovermode="closest",
            legend=dict(orientation="h", y=1.15),
            title=f"{name} — 整合概览 (鼠标移动同步竖线)",
        )
        fig_ov.update_xaxes(rangeslider_visible=False, showspikes=True,
                            spikemode="across", spikethickness=1,
                            spikecolor="rgba(128,128,128,0.4)")
        fig_ov.update_yaxes(title_text="价格", row=1, col=1)
        fig_ov.update_yaxes(title_text="累积收益", tickformat=".0%", row=2, col=1)
        fig_ov.update_yaxes(title_text="仓位", tickformat=".0%", row=3, col=1)
        fig_ov.update_yaxes(title_text="盈亏 %", row=4, col=1)

        p = f"{out_dir}/{safe_name}_overview.html"
        fig_ov.write_html(p)
        saved_plots.append(("overview", f"{safe_name}_overview.html"))
        print(f"  Overview (3-panel sync) -> {p}")

        # 仍保留独立的 baseline 和 cum_returns（方便单图查看）
        try:
            fig_bl = _plot_price_with_baseline(
                df.loc[idx], entries, exits, entry_modes,
                title=f"{name} - Polyfit基线 + 买卖标记",
                fill_price=fill_price,
            )
            p = f"{out_dir}/{safe_name}_baseline.html"
            fig_bl.write_html(p)
            saved_plots.append(("baseline", f"{safe_name}_baseline.html"))
        except Exception:
            pass

        try:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=idx, y=bh_cum, mode="lines", name=f"BH ({bh_ret:+.1%})", line=dict(color="gray", width=1, dash="dot")))
            fig_cum.add_trace(go.Scatter(x=eq_f.index, y=eq_f, mode="lines", name=f"Full ({full_ret:+.1%})", line=dict(color="#2196F3", width=2)))
            fig_cum.add_trace(go.Scatter(x=eq_g.index, y=eq_g, mode="lines", name=f"Grid-only ({grid_ret:+.1%})", line=dict(color="#4CAF50", width=1.5, dash="dash")))
            fig_cum.update_layout(title=f"{name} - 累积收益", height=400, hovermode="x unified", yaxis_tickformat=".0%")
            p = f"{out_dir}/{safe_name}_cum_returns.html"
            fig_cum.write_html(p)
            saved_plots.append(("cum_returns", f"{safe_name}_cum_returns.html"))
        except Exception:
            pass

    except Exception as e:
        print(f"  Overview skipped: {e}")

    return {
        "name": name, "stats": stats, "params": params,
        "out_dir": safe_name, "overview_html": f"{safe_name}_overview.html",
        "plots": saved_plots,
    }


def generate_polyfit_grid_report(
    df, close, entries, exits, sizes,
    params, name="Polyfit-Grid", reports_dir="reports",
    open_=None,
):
    """Polyfit-Grid 增强报告：K线+基线+买卖标记+权益曲线+回撤。

    与 generate_polyfit_switch_report 格式一致，但不含 Switch 模式分解。
    """
    import plotly.graph_objects as go
    import vectorbt as vbt

    idx = close.index
    fill_price = open_.shift(-1).reindex(idx) if open_ is not None else close

    safe_name = name.replace(" ", "_")
    out_dir = f"{reports_dir}/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    pf = vbt.Portfolio.from_signals(
        fill_price,
        entries=pd.Series(entries, index=idx),
        exits=pd.Series(exits, index=idx),
        size=pd.Series(sizes, index=idx),
        size_type="percent", init_cash=100_000.0, freq="D",
    )

    total_ret = pf.total_return()
    bh_ret = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

    print(f"\n  {'='*60}")
    print(f"  {name} — 全量回测指标")
    print(f"  {'='*60}")
    print(f"  BH return:          {bh_ret:+.2%}")
    print(f"  Total Return:       {total_ret:+.2%}  (α={total_ret-bh_ret:+.2%})")
    print(f"  Max Drawdown:       {pf.max_drawdown():.2%}")
    print(f"  Sharpe Ratio:       {float(pf.stats().get('Sharpe Ratio', 0)):.3f}")
    print(f"  Total Trades:       {int(pf.stats().get('Total Trades', 0))}")

    stats = pf.stats()
    stats_path = f"{out_dir}/{safe_name}_stats.csv"
    stats.to_csv(stats_path)
    print(f"  Stats -> {stats_path}")

    saved_plots = []

    # 1) K线+基线+买卖标记（所有入场都是 Grid，统一绿色）
    modes_all_grid = np.ones_like(entries, dtype=np.int8)
    try:
        fig_bl = _plot_price_with_baseline(
            df.loc[idx], entries, exits, modes_all_grid,
            title=f"{name} - Polyfit基线 + 买卖标记 (Grid▲绿 离场▼红)",
            fill_price=fill_price,
        )
        p = f"{out_dir}/{safe_name}_baseline.html"
        fig_bl.write_html(p)
        saved_plots.append(("baseline", f"{safe_name}_baseline.html"))
        print(f"  Baseline chart -> {p}")
    except Exception as e:
        print(f"  Baseline chart skipped: {e}")

    # 2) 累积收益 vs BH
    try:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=idx, y=close / close.iloc[0] - 1, mode="lines",
            name=f"BH ({bh_ret:+.1%})", line=dict(color="gray", width=1, dash="dot"),
        ))
        eq = pf.value() / pf.value().iloc[0] - 1
        fig_cum.add_trace(go.Scatter(
            x=eq.index, y=eq, mode="lines",
            name=f"{name} ({total_ret:+.1%})", line=dict(color="#4CAF50", width=2),
        ))
        fig_cum.update_layout(title=f"{name} - 累积收益 vs BH",
                              height=400, hovermode="x unified", yaxis_tickformat=".0%")
        p = f"{out_dir}/{safe_name}_cum_returns.html"
        fig_cum.write_html(p)
        saved_plots.append(("cum_returns", f"{safe_name}_cum_returns.html"))
        print(f"  Cum returns -> {p}")
    except Exception as e:
        print(f"  Cum returns skipped: {e}")

    # 3) 回撤曲线
    try:
        fig_dd = go.Figure()
        dd = pf.drawdown()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd * 100, mode="lines",
            name=f"Drawdown (max={pf.max_drawdown():.1%})",
            line=dict(color="#FF9800", width=2), fill="tozeroy",
        ))
        fig_dd.update_layout(title=f"{name} - 回撤曲线",
                             height=300, hovermode="x unified", yaxis_title="回撤 %")
        p = f"{out_dir}/{safe_name}_drawdowns.html"
        fig_dd.write_html(p)
        saved_plots.append(("drawdowns", f"{safe_name}_drawdowns.html"))
        print(f"  Drawdowns -> {p}")
    except Exception as e:
        print(f"  Drawdowns skipped: {e}")

    # 4) 交易盈亏分布
    try:
        trade_returns = []
        for t in pf.trades.records_readable.itertuples():
            if hasattr(t, "Return") and pd.notna(t.Return):
                trade_returns.append(t.Return)
        if trade_returns:
            fig_pnl = go.Figure()
            colors = ["#4CAF50" if r > 0 else "#F44336" for r in trade_returns]
            fig_pnl.add_trace(go.Bar(
                x=list(range(len(trade_returns))), y=[r * 100 for r in trade_returns],
                marker_color=colors, name="Trade PnL",
            ))
            fig_pnl.update_layout(
                title=f"{name} - 逐笔交易盈亏 (%)", height=300,
                xaxis_title="Trade #", yaxis_title="Return %",
            )
            p = f"{out_dir}/{safe_name}_trade_pnl.html"
            fig_pnl.write_html(p)
            saved_plots.append(("trade_pnl", f"{safe_name}_trade_pnl.html"))
            print(f"  Trade PnL -> {p}")
    except Exception as e:
        print(f"  Trade PnL skipped: {e}")

    # 5) 水下曲线 (underwater)
    try:
        dd = pf.drawdown()
        fig_uw = go.Figure()
        fig_uw.add_trace(go.Scatter(
            x=dd.index, y=dd * 100, mode="lines",
            name="Underwater", line=dict(color="#FF9800", width=2), fill="tozeroy",
        ))
        fig_uw.update_layout(
            title=f"{name} - 水下曲线 (max DD={pf.max_drawdown():.1%})",
            height=300, yaxis_title="Drawdown %",
        )
        p = f"{out_dir}/{safe_name}_underwater.html"
        fig_uw.write_html(p)
        saved_plots.append(("underwater", f"{safe_name}_underwater.html"))
        print(f"  Underwater -> {p}")
    except Exception as e:
        print(f"  Underwater skipped: {e}")

    # 6) VectorBT 原生概览（尝试，失败则用简化版权益曲线）
    try:
        fig_ov = pf.plot()
        p = f"{out_dir}/{safe_name}_overview.html"
        fig_ov.write_html(p)
        saved_plots.append(("overview", f"{safe_name}_overview.html"))
        print(f"  Overview -> {p}")
    except Exception:
        # 回退：简化权益曲线+买卖标记
        try:
            fig_ov = go.Figure()
            eq = pf.value()
            fig_ov.add_trace(go.Scatter(
                x=eq.index, y=eq, mode="lines",
                name="Equity", line=dict(color="#2196F3", width=2),
            ))
            entry_dates = idx[entries.astype(bool)]
            exit_dates = idx[exits.astype(bool)]
            get_v = lambda d: eq.loc[d] if d in eq.index else float(eq.iloc[eq.index.get_indexer([d], method="ffill")[0]])
            fig_ov.add_trace(go.Scatter(
                x=entry_dates, y=[get_v(d) for d in entry_dates], mode="markers",
                name="买入", marker=dict(color="green", size=6, symbol="triangle-up"),
            ))
            fig_ov.add_trace(go.Scatter(
                x=exit_dates, y=[get_v(d) for d in exit_dates], mode="markers",
                name="卖出", marker=dict(color="red", size=6, symbol="triangle-down"),
            ))
            fig_ov.update_layout(title=f"{name} - 权益曲线", height=400, hovermode="x unified")
            p = f"{out_dir}/{safe_name}_overview.html"
            fig_ov.write_html(p)
            saved_plots.append(("overview", f"{safe_name}_overview.html"))
            print(f"  Overview (fallback) -> {p}")
        except Exception as e2:
            print(f"  Overview skipped: {e2}")

    return {
        "name": name, "stats": stats, "params": params,
        "out_dir": safe_name, "overview_html": f"{safe_name}_overview.html",
        "plots": saved_plots,
    }

