# VectorBT 量化回测系统

基于 VectorBT 的多策略 Walk-Forward 回测框架，GPU 加速（CuPy RawKernel）。

标的：512890.SH（中证红利 ETF），后复权数据。

## 最佳历史收益

| 策略 | 训练期 | 平均 OOS 收益 | 平均 BH 收益 | 超额 α | Sharpe | 正收益窗口 | 跑赢 BH |
|------|--------|-------------|-------------|--------|--------|-----------|--------|
| **Polyfit-Switch** | 2 年 | **+20.5%** | +8.5% | +12.0% | 1.86 | 100% | 100% |
| **Polyfit-Switch** | 3 年 | **+15.1%** | +10.3% | +4.8% | 1.73 | 100% | 75% |
| MA Grid | 3 年 | +9.3% | +10.3% | -1.0% | 1.64 | 75% | 75% |
| MA-Switch | 2 年 | +0.7% | +8.5% | -7.8% | 0.29 | 60% | 20% |

**最佳单窗口：2024 年 +49.9%（Polyfit-Switch，2 年训练）。**  
**全量参数扫描：77,760 组 Grid × 9 组 Indicator × Stage 2 Switch，GPU 单窗口 ≈ 10s。**

### 最佳窗口参数（2024 +49.9%）

| 参数 | 值 | 说明 |
|------|-----|------|
| `fit_window_days` | 252 | 1 年线性回归窗口 |
| `trend_window_days` | 10 | 偏离趋势 EMA |
| `vol_window_days` | 10 | 波动率窗口 |
| `base_grid_pct` | 1.5% | 基础网格步长 |
| `volatility_scale` | 0.5 | 波动率放大系数 |
| `trend_sensitivity` | 10.0 | 趋势敏感度 |
| `max_grid_levels` | 2 | 最大网格层级 |
| `take_profit_grid` | 1.0 | 止盈网格倍数 |
| `stop_loss_grid` | 1.6 | 止损网格倍数 |
| `min_signal_strength` | 0.30 | 最小信号强度 |
| `position_size` | 0.99 | 最大仓位（99%） |
| `position_sizing_coef` | 60.0 | 仓位系数 |
| `flat_wait_days` | 5 | Switch 激活等待天数 |
| `switch_deviation_m1` | 0.02 | Switch 激活偏离阈值 |
| `switch_deviation_m2` | 0.005 | Switch 关闭偏离阈值 |
| `switch_trailing_stop` | 0.02 | 追踪止损回撤 2% |
| `switch_fast_ma` | 5 | 金叉快均线 |
| `switch_slow_ma` | 60 | 金叉慢均线 |

### 各窗口最优参数（2 年训练）

### Polyfit-Switch 各窗口明细（2 年训练）

| 测试年 | 收益 | BH 收益 | 超额 α | Sharpe | 交易次数 | 胜率 |
|--------|------|--------|--------|--------|---------|------|
| 2022 | +29.2% | +1.1% | +28.1% | — | — | — |
| 2023 | +12.1% | +11.2% | +0.9% | — | — | — |
| **2024** | **+49.9%** | +21.5% | +28.4% | — | — | — |
| 2025 | +8.1% | +6.8% | +1.3% | — | — | — |
| 2026 (YTD) | +3.2% | +1.9% | +1.4% | — | — | — |

#### 各窗口参数详情

| 窗口 | bgp | vs | ts | mgl | tpg | slg | sz | coef | mss | fw | m1 | m2 | tr | ma |
|------|-----|----|----|-----|-----|-----|-----|------|-----|-----|-----|-----|-----|-----|
| 2022 +29.2% | 0.008 | 0.0 | 10 | 3 | 1.00 | 1.2 | 0.99 | 60 | 0.30 | 5 | 0.020 | 0.015 | 0.03 | 5/10 |
| 2023 +12.1% | 0.010 | 2.0 | 4 | 3 | 0.60 | 1.2 | 0.99 | 60 | 0.30 | 5 | 0.030 | 0.020 | 0.02 | 5/10 |
| **2024 +49.9%** | 0.015 | 0.5 | 10 | 2 | 1.00 | 1.6 | 0.99 | 60 | 0.30 | 5 | 0.020 | 0.005 | 0.02 | 5/60 |
| 2025 +8.1% | 0.015 | 2.0 | 10 | 4 | 1.00 | 2.0 | 0.99 | 20 | 0.30 | 5 | 0.040 | 0.005 | 0.02 | 5/20 |
| 2026 +3.2% | 0.012 | 2.0 | 10 | 2 | 0.80 | 1.2 | 0.99 | 60 | 0.30 | 5 | 0.040 | 0.005 | 0.02 | 5/10 |

> 缩写：bgp=base_grid_pct, vs=volatility_scale, ts=trend_sensitivity, mgl=max_grid_levels,
> tpg=take_profit_grid, slg=stop_loss_grid, sz=position_size, coef=position_sizing_coef,
> mss=min_signal_strength, fw=flat_wait_days, m1=switch_deviation_m1, m2=switch_deviation_m2,
> tr=switch_trailing_stop, ma=switch_fast_ma/switch_slow_ma

### 所有策略整体对比

```
Polyfit-Switch  2yr  OOS=+20.5%  BH=+8.5%  α=+12.0%  sharpe=1.856  pos=100%  >BH=100%  w=5  tr=28
Polyfit-Switch  3yr  OOS=+15.1%  BH=+10.3%  α=+4.8%  sharpe=1.725  pos=100%  >BH=75%   w=4  tr=14
MA              3yr  OOS=+9.3%   BH=+10.3%  α=-1.1%  sharpe=1.643  pos=75%   >BH=75%   w=4  tr=12
MA-Switch       3yr  OOS=-2.4%   BH=+10.3%  α=-12.8% sharpe=-0.076 pos=75%   >BH=0%    w=4  tr=19
```

---

## Polyfit-Switch 策略详解

### 核心思想

用**滑动窗口线性回归**替代简单移动均线作为价格基准线，结合**双模式切换**机制：

- **Grid 模式（均值回复）**：在基线下方挂多单网格，偏离越大仓位越重，回归到止盈线离场
- **Switch 模式（趋势追踪）**：价格持续在基线上方且偏离较大时激活，用快慢均线金叉入场 + **最高价回撤追踪止损**离场

### 指标计算

```
1. PolyBasePred（基线）
   对前 252 根 bar（约 1 年）的收盘价做最小二乘拟合 y = α + βx
   取拟合线在当天的预测值作为基线。比 SMA 更能反映价格长期运行中枢。

2. PolyDevPct（偏离度）
   PolyDevPct = close / PolyBasePred - 1
   正值 = 价格高于基线，负值 = 价格低于基线

3. PolyDevTrend（偏离趋势）
   PolyDevTrend = EMA(diff(PolyDevPct), span=trend_window_days)
   正值 → 偏离在扩大（价格在向基线回归）
   负值 → 偏离在收敛（价格在继续远离基线）

4. RollingVolPct（滚动波动率）
   日收益率的滚动总体标准差（ddof=0），用于动态调整网格宽度
```

### 信号生成（逐 bar 状态机）

#### Grid 模式入场

```
条件：无持仓 且 cooldown == 0
  信号强度 = |PolyDevPct| / dynamic_grid_step
  入场等级 = clip(floor(信号强度), 1, max_grid_levels)
  入场阈值 = -入场等级 × dynamic_grid_step

  当 PolyDevPct ≤ 入场阈值 且 信号强度 ≥ min_signal_strength：
    仓位 = clip(|PolyDevPct| × (1 + max(RollingVolPct, 0)) × position_sizing_coef,
                0, position_size)
    入场
```

#### Grid 模式离场

```
持仓中，任一条件触发即离场：
  1. 持仓天数 ≥ max_holding_days（45 天）
  2. PolyDevPct ≥ 入场等级 × ref_step × take_profit_grid（止盈）
  3. PolyDevPct ≤ -入场等级 × ref_step × stop_loss_grid（止损）
```

#### Switch 模式激活/关闭

```
激活（三条件同时满足）：
  1. flat_days ≥ flat_wait_days（连续空仓天数达标）
  2. close > PolyBasePred（价格在基线上方）
  3. PolyDevPct > switch_deviation_m1（偏离超过激活阈值）

关闭：
  PolyDevPct < switch_deviation_m2（偏离回落到关闭阈值以下）
```

#### Switch 模式交易（★ 追踪止损）

```
入场：fast_ma > slow_ma（金叉）+ 无持仓 → 全仓买入
出：持仓期间持续追踪最高收盘价 peak_close
    当 close ≤ peak_close × (1 - switch_trailing_stop) 时离场

追踪止损 vs 死叉离场的优势：
  - 死叉容易被短期均线缠绕反复触发进出
  - 追踪止损只在价格从最高点回撤足够多时离场，持仓更稳
  - 实际测试：2024 年 +49.6%（追踪止损）vs +39.8%（死叉）
```

#### 动态网格步长

```
dynamic_grid_step = base_grid_pct
                  × (1 + trend_sensitivity × |PolyDevTrend|)
                  × (1 + volatility_scale × max(RollingVolPct, 0))
下限 = base_grid_pct × 0.3
```

高波动 / 强趋势 → 宽网格 → 更严格的入场条件，减少假信号。

### 执行模型

所有策略使用 **次日开盘价成交**，消除前视偏差：

```
bar i Close 数据到达 → 计算指标 → 生成 entry/exit 信号
    ↓
bar i+1 Open 成交（fill_price = open.shift(-1)）
    ↓
NAV 仍以 bar i Close 估值
```

### 参数说明

| 参数 | 含义 | 扫描范围 |
|------|------|---------|
| `fit_window_days` | 线性回归拟合窗口 | 252（固定） |
| `trend_window_days` | 偏离趋势 EMA 窗口 | 10-20 |
| `vol_window_days` | 波动率计算窗口 | 10-30 |
| `base_grid_pct` | 基础网格步长 | 0.8%-1.5% |
| `volatility_scale` | 波动率放大系数 | 0.0-2.0 |
| `trend_sensitivity` | 趋势敏感度 | 4.0-10.0 |
| `max_grid_levels` | 最大网格层级 | 2-4 |
| `take_profit_grid` | 止盈网格倍数 | 0.6-1.0 |
| `stop_loss_grid` | 止损网格倍数 | 1.2-2.0 |
| `max_holding_days` | 最大持仓天数 | 45（固定） |
| `cooldown_days` | 出场后冷却天数 | 1（固定） |
| `min_signal_strength` | 最小信号强度 | 0.30-0.60 |
| `position_size` | 最大仓位比例 | 0.92-0.99 |
| `position_sizing_coef` | 仓位系数 | 20-60 |
| `flat_wait_days` | Switch 激活等待天数 | 5-15 |
| `switch_deviation_m1` | Switch 激活偏离阈值 | 2%-5% |
| `switch_deviation_m2` | Switch 关闭偏离阈值 | 0.5%-2% |
| `switch_trailing_stop` | Switch 追踪止损回撤比例 | 2%-10% |
| `switch_fast_ma` | Switch 快均线周期 | 5-20 |
| `switch_slow_ma` | Switch 慢均线周期 | 10-60 |

### 扫描策略

采用**两阶段 GPU 批量扫描**，避免参数爆炸：

- **Stage 1**（GPU）：固定 Switch 参数为典型值，从 77,760 组 grid 参数中随机采样 2000 组，GPU 批量生成信号 + 批量回测
- **Stage 2**（GPU）：固定最优 Grid 参数，全排列扫描 Switch 参数（约 2000 组合），GPU 批量完成

## 工程结构

```
vectorbt_test/
├── main.py                      # 主入口，Walk-Forward 分析
├── strategies/
│   ├── ma_grid.py               # MA 网格策略（CPU + GPU）
│   ├── ma_switch.py             # MA-Switch 双模式策略（CPU + GPU）
│   └── polyfit_switch.py        # Polyfit-Switch 双模式策略（CPU + GPU）★ 最佳
├── utils/
│   ├── backtest.py              # 回测引擎（VectorBT CPU + CuPy GPU）
│   ├── data.py                  # 数据加载
│   ├── gpu.py                   # GPU/CUDA 检测
│   ├── indicators.py            # 技术指标计算（MA / Polyfit）
│   ├── reports.py               # 报告生成（VectorBT HTML 图表）
│   ├── scan.py                  # 参数扫描通用框架
│   └── walkforward.py           # Walk-Forward 窗口生成与分析
└── data/
    └── 512890.SH_hfq.parquet    # 后复权行情数据
```

## 运行

```bash
uv run python main.py
```

输出：
- `reports/walkforward_results.csv` — 所有窗口的训练/测试结果
- `reports/MA/` / `reports/MA-Switch/` / `reports/Polyfit-Switch/` — 最优参数 VectorBT 报告
- `reports/index.html` — 汇总索引页面
