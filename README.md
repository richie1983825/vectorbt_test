# VectorBT 量化回测系统

基于 VectorBT 的多策略 Walk-Forward 回测框架，GPU 加速（CuPy RawKernel）。

标的：512890.SH（中证红利 ETF），后复权数据。

## 当前最优策略

**Polyfit-Switch-v6（OHLCV 增强版 Grid-priority Switch）**，22 月训练 / 12 月测试，步长 3 月，15 个滑动窗口。

### 六组合 WF OOS 对比

```
  # 组合                            OOS        α   Sharpe   maxDD  pos%   >BH
  1 Grid-return                   +18.3%    +7.7%    1.697    -7.2%  93%  73%
  2 Grid-balanced                 +11.6%    +1.0%    1.434    -6.5%  87%  60%
  3 Switch-return ★              +22.2%   +11.5%    1.945    -7.4% 100%  87%
  4 Switch-balanced               +14.6%    +4.0%    1.581    -8.0%  87%  60%
  5 Grid-ret+Switch-bal           +22.2%   +11.5%    1.945    -7.4% 100%  87%
  6 Grid-bal+Switch-ret           +14.6%    +4.0%    1.581    -8.0%  87%  60%
```

**最优：Switch-return / Grid-ret+Switch-bal（并列），OOS +22.2%，Sharpe 1.945，87% 窗口跑赢 BH。**

---

## Polyfit-Grid 策略

纯 Grid 均值回复网格策略，不含 Switch。基线为滑动窗口线性回归（252 天固定）。

### 最优参数

| 参数 | return | balanced | 说明 |
|------|--------|----------|------|
| OOS 均值 | +18.3% | +11.6% | 15 窗口 WF |
| Sharpe | 1.697 | 1.434 | |
| maxDD | -7.2% | -6.5% | |
| `trend_window_days` | 10 | 10 | 偏离趋势 EMA |
| `vol_window_days` | 10 | 10 | 波动率窗口 |
| `base_grid_pct` | 1.0% | 0.8% | 基础网格步长 |
| `volatility_scale` | 0.0 | 0.0 | 波动率放大系数 |
| `trend_sensitivity` | 4.0 | 8.0 | 趋势敏感度 |
| `max_grid_levels` | 3 | 3 | 最大网格层级 |
| `take_profit_grid` | 0.80 | 1.00 | 止盈网格倍数 |
| `stop_loss_grid` | 1.60 | 1.20 | 止损网格倍数 |
| `max_holding_days` | 45 | 45 | 最大持仓 |
| `cooldown_days` | 1 | 1 | 出场冷却 |
| `min_signal_strength` | 0.30 | 0.30 | 最小信号强度 |
| `position_size` | 0.99 | 0.99 | 最大仓位 |
| `position_sizing_coef` | 60.0 | 60.0 | 仓位系数 |

> return 评分倾向更宽的网格步长和更高的止损容忍，牺牲回撤换收益。
> balanced 评分综合考量收益+Sharpe+回撤，网格更密、止盈更早、止损更紧。

**全量回测**（最优 return 参数）：总收益 +167%，α +55%，Sharpe 1.66，maxDD -9.1%，135 笔交易。

---

## Polyfit-Switch-v6 策略

OHLCV 增强版 Grid-priority Switch。在 Grid 基础上，Grid 空仓时 Switch 可入场。

### v6 增强因子（vs v5）

**入场 OHLCV 过滤**（基于 109 笔 Switch 统计分析）：

| # | 条件 | 统计依据 |
|---|------|----------|
| 1 | 微涨 0 ~ +0.5% → 不买 | 胜率 12.5% |
| 2 | 长上影 ≥30% → 不买 | 胜率 18.4% |
| 3 | 正常量能 0.8-1.5x → 不买 | 胜率 13.7% |
| 4 | 阳线 + 连涨<4天 → 不买 | 阳线胜率 23.8% |
| 5 | 高位 ≥70% + 连涨<4天 → 不买 | 高位胜率 32.1% |

**离场改进**：
- dp 连续下跌 1 天即离场（原 3 天）
- dp < -0.5% 预警离场（避免被 Grid 强制踢出，grid_force 胜率 0%）
- ATR 追踪乘数 1.5（原 2.5）

### 最优参数

| 参数 | return | balanced | 说明 |
|------|--------|----------|------|
| OOS 均值 | +22.2% | +14.6% | 15 窗口 WF |
| Sharpe | 1.945 | 1.581 | |
| maxDD | -7.4% | -8.0% | |
| `trend_entry_dp` | 0.0 | 0.0 | Switch 入场偏离阈值 |
| `trend_confirm_dp_slope` | 0.0 | 0.0 | 趋势确认斜率 |
| `trend_atr_mult` | 1.5 | 1.5 | ATR 追踪乘数 |
| `trend_decline_days` | 1 | 1 | dp 连续下跌离场天数 |
| `trend_vol_climax` | 2.5 | 2.5 | 量能衰竭阈值 |
| `enable_ohlcv_filter` | True | True | OHLCV 入场过滤 |
| `enable_early_exit` | True | True | 预警离场 |
| Grid 参数 | 同 Grid-return | 同 Grid-balanced | |

> Switch 最优参数高度一致：ATR=1.5、decline=1 天、OHLCV/early_exit 全开，说明这些因子稳定。

**全量回测**（最优 return 参数）：Grid+Switch 合并 +254%，纯 Grid +167%，纯 Switch +30%，53 笔 Switch 交易。

---

## 执行模型

所有策略使用 **次日开盘价成交**，消除前视偏差：

```
bar i Close → 计算指标 → 生成信号 → bar i+1 Open 成交
```

GPU kernel 分离 fill_price（成交价）和 close（NAV 估值价）。

---

## 架构

```
vectorbt_test/
├── main.py                         # 入口：调用 workflows，输出六组合对比 + HTML 报告
├── workflows/
│   ├── polyfit_grid.py             # Grid WF + GPU 扫描 + 参数缓存
│   │   └── reports/grid_wf_cache.csv  (最优 Grid 参数，return/balanced/robust)
│   └── polyfit_switch.py           # Switch-v6 WF + CPU 扫描（依赖 Grid 缓存）
├── strategies/
│   ├── polyfit_grid.py             # Grid 策略 + GPU 批量扫描
│   └── polyfit_switch.py           # Switch-v6 OHLCV 增强策略 (CPU + GPU kernel)
├── utils/
│   ├── backtest.py                 # 回测引擎（VectorBT CPU + CuPy GPU batch）
│   ├── data.py                     # 数据加载
│   ├── gpu.py                      # GPU/CUDA 检测
│   ├── indicators.py               # 技术指标（Polyfit / MA）
│   ├── reports.py                  # HTML 报告生成
│   ├── scoring.py                  # 参数评分（return/balanced/robust）
│   └── walkforward.py              # Walk-Forward 窗口生成
└── reports/                        # 输出目录
    ├── grid_wf_cache.csv           # Grid 参数缓存
    ├── wf_comparison.csv           # 六组合 WF 结果
    └── index.html                  # HTML 报告索引
```

## 运行

```bash
uv run python main.py
```

流程：
1. Stage 1 — Grid WF：`reports/grid_wf_cache.csv` 存在则跳过 GPU 扫描（秒级），否则逐窗口扫描（~3 分钟/15 窗口）
2. Stage 2 — Switch-v6 WF：逐窗口 CPU 扫描 162 组 Switch 参数（~1.5 分钟/15 窗口）
3. 输出六组合对比 + HTML 报告

总耗时：首次 ~5 分钟，后续 ~1.5 分钟（Grid 缓存命中）。

### 单独运行

```bash
uv run python workflows/polyfit_grid.py      # 仅 Grid WF + 更新缓存
uv run python workflows/polyfit_switch.py    # Grid + Switch-v6 WF
```

## 分析脚本

| 脚本 | 用途 |
|------|------|
| `analyze_down_day_filter.py` | Switch 下一天过滤效果分析 |
| `analyze_switch_surge.py` | Switch 入场 OHLCV 特征分析（连续上涨/单日急涨） |
| `compare_v5_v6.py` | v5 vs v6 固定参数对比 |
| `grid_vs_switch_v6.py` | Grid vs Switch-v6 WF 参数扫描对比 |

## 评分方法

| 方法 | 逻辑 | 适用场景 |
|------|------|----------|
| `return` | 训练期总收益最高 | 趋势明显、回撤容忍度高 |
| `balanced` | 收益(50%)+Sharpe(30%)-回撤(20%) | 均衡风险收益 |
| `robust` | 要求各训练子段均正收益，再选最高 Sharpe | 稳健性优先（当前 fallback 到 balanced） |

## 重要约束

- `fit_window_days` 必须固定为 252。放开扫描会导致训练期严重过拟合
- 禁止 ML 策略（XGBoost/神经网络）：数据量仅 1763 条日线，ML 极小样本必然过拟合
- 所有回测默认 Walk-Forward，禁止全量数据扫描后报告收益
- 每次回测必须同时列出 return/balanced/robust 三种评分结果
