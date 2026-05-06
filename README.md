# VectorBT 量化回测系统

基于 VectorBT 的多策略 Walk-Forward 回测框架，GPU 加速（CuPy RawKernel）。

标的：512890.SH（中证红利 ETF），后复权数据。

## 当前最优策略

**Polyfit-Switch-v7（V6 + 顶部规避）**，22 月训练 / 12 月测试，步长 3 月，15 个滑动窗口。

### Grid vs Switch-v6 vs Switch-v7 — WF OOS 对比

```
  # 组合                    OOS        α   Sharpe   maxDD  pos%   >BH
  1 Grid-return             +18.3%    +7.7%    1.697    -7.2%  93%  73%
  2 Grid-balanced           +11.6%    +1.0%    1.434    -6.5%  87%  60%
  3 Switch-v6-return        +21.3%   +10.7%    1.721    -7.1%  93%  67%
  4 Switch-v6-balanced      +21.3%   +10.7%    1.721    -7.1%  93%  67%
  5 Switch-v7-return ★      +22.3%   +11.7%    1.971    -7.2%  100%  80%
  6 Switch-v7-balanced      +15.6%    +4.9%    1.791    -6.3%  93%  67%
```

**最优：Switch-v7-return (+22.3% OOS, Sharpe 1.971, 80% 跑赢 BH)**

---

## Polyfit-Switch-v7 顶部规避模块

基于 7 个历史顶部日（2021-07-02, 2021-09-23, 2021-10-25, 2022-02-14, 2023-05-09/10, 2024-10-08）的共性分析：

| 特征 | 规律 |
|------|------|
| 短期急涨 | 5/7 案例前 5 日涨幅 >5% |
| 高位运行 | 4/7 案例 20 日价位 >80% |
| 异常 K 线 | 全部振幅 >2.5%（正常 1.3%），4/7 长上影 >1% |
| 量能异常 | 3/7 案例量比 >1.5x |
| 后续走势 | 5/7 案例 3-5 天跌 3-5% |

### 规避规则

```
IF 5日涨幅 > top_ret_5d AND 价位 > top_price_pos AND 收阴 AND 振幅 > top_amplitude:
    → 强制离场 Switch 持仓
    → 禁止 Grid + Switch 新入场 top_block_days 天
```

### 最优参数（return 评分）

| 参数 | 值 | 说明 |
|------|-----|------|
| `enable_top_avoidance` | True | 启用顶部规避 |
| `top_ret_5d` | 0.05 | 5 日涨幅阈值 |
| `top_price_pos` | 0.80 | 20 日价位阈值 |
| `top_amplitude` | 0.02 | 异常振幅阈值 |
| `top_block_days` | 3 | 顶部后禁止入场天数 |

V7 相比 V6：return +1.0pp（+22.3% vs +21.3%），balanced +4.9pp α 且 maxDD 从 -7.1% 降到 -6.3%。

---

## Polyfit-Switch-v6 策略

OHLCV 增强版 Grid-priority Switch。V6 固定参数用于全量回测报告。

### V6 增强因子

**入场 OHLCV 过滤**（基于 109 笔 Switch 统计分析）：

| # | 条件 | 统计依据 |
|---|------|----------|
| 1 | 微涨 0~+0.5% → 不买 | 胜率 12.5% |
| 2 | 长上影 ≥30% → 不买 | 胜率 18.4% |
| 3 | 正常量能 0.8-1.5x → 不买 | 胜率 13.7% |
| 4 | 阳线 + 连涨<4 天 → 不买 | 阳线胜率 23.8% |
| 5 | 高位 ≥70% + 连涨<4 天 → 不买 | 高位胜率 32.1% |

**离场改进**：dp 连跌 1 天离场、dp<-0.5% 预警离场、ATR 乘数 1.5。

### 最优参数

| 参数 | return/balanced | 说明 |
|------|-----------------|------|
| `trend_entry_dp` | 0.0 | Switch 入场偏离阈值 |
| `trend_confirm_dp_slope` | 0.0 | 趋势确认斜率 |
| `trend_atr_mult` | 1.5 | ATR 追踪乘数 |
| `trend_decline_days` | 1 | dp 连续下跌离场天数 |
| `trend_vol_climax` | 2.5 | 量能衰竭阈值 |
| `enable_ohlcv_filter` | True | OHLCV 入场过滤 |
| `enable_early_exit` | True | 预警离场 |

全量回测：Grid+Switch 合并 +254%，Grid +167%，Switch +30%。

---

## Polyfit-Grid 策略

纯 Grid 均值回复网格策略，基线为滑动窗口线性回归（252 天固定）。

### 最优参数

| 参数 | return | balanced |
|------|--------|----------|
| OOS 均值 | +18.3% | +11.6% |
| `base_grid_pct` | 1.0% | 0.8% |
| `volatility_scale` | 0.0 | 0.0 |
| `trend_sensitivity` | 4.0 | 8.0 |
| `max_grid_levels` | 3 | 3 |
| `take_profit_grid` | 0.80 | 1.00 |
| `stop_loss_grid` | 1.60 | 1.20 |

全量回测：+167%，Sharpe 1.66，maxDD -9.1%，135 笔交易。

---

## 日内限价单策略（已废弃）

Grid 连跌入场时限价买入（-1.0% offset）在全量回测上贡献 +7.3% 增量，但在 Walk-Forward 验证中表现不稳定（Intra OOS 低于纯 Switch 3pp）。原因是每个 12 月 OOS 窗口仅 1-2 笔触发，参数选择噪声过大。**结论：日内限价策略无法通过 WF 验证，不可行。**

---

## 执行模型

所有策略使用 **次日开盘价成交**，消除前视偏差：

```
bar i Close → 计算指标 → 生成信号 → bar i+1 Open 成交
```

## 架构

```
main.py                         # 入口：Grid WF + Switch-v7 WF + V6固定OOS → 对比输出 + 报告
workflows/
├── polyfit_grid.py             # Grid WF + GPU 扫描 + 参数缓存
└── polyfit_switch.py           # Switch-v7 WF + CPU 扫描（固定V6 base，扫描顶部规避）
strategies/
├── polyfit_grid.py             # Grid 策略 + GPU 批量扫描
└── polyfit_switch.py           # Switch v6/v7 + OHLCV增强 + 顶部规避
utils/
├── backtest.py / data.py / gpu.py / indicators.py / reports.py
├── scoring.py                  # return/balanced/robust 评分
└── walkforward.py              # Walk-Forward 窗口生成
```

## 运行

```bash
uv run python main.py
```

## 评分方法

| 方法 | 逻辑 |
|------|------|
| `return` | 训练期总收益最高 |
| `balanced` | 收益(50%)+Sharpe(30%)-回撤(20%) |
| `robust` | 各训练子段均正收益，选最高 Sharpe（当前 fallback 到 balanced） |

## 约束

- `fit_window_days` 固定 252，放开扫描导致过拟合
- 禁止 ML 策略（数据量不足）
- 所有回测默认 Walk-Forward
- 每次回测同时列出 return/balanced/robust 三种评分
