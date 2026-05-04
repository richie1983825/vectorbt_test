# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

- 操作系统：WSL2 Debian (Linux 6.6.87-microsoft-standard-WSL2)
- Python 版本：3.13，通过 `.python-version` 管理
- 包管理：UV (`uv` / `uv pip` / `uv run`)，不要使用裸 `pip` 或 `python`
- 虚拟环境位于项目根目录 `.venv/`
- 注意：若 shell 环境设置了 `VIRTUAL_ENV` 指向其他 Python 版本，运行 `uv` 命令前需 `unset VIRTUAL_ENV`，否则会有 warning（不影响执行结果）

## 常用命令

```bash
# 运行 Python 脚本（自动使用 .venv）
uv run python main.py

# 安装/添加依赖
uv add <package>

# 运行测试（后续配置后使用）
uv run pytest -v

# 格式化/检查（后续配置后使用）
uv run ruff check .
uv run ruff format .
```

## 核心依赖框架

- **VectorBT** (`vectorbt`) — 量化回测核心框架
- **CuPy** (`cupy`) — GPU 加速（RawKernel 批量信号生成 + 批量回测）
- **NumPy / Pandas** — 数据处理

## 行情数据

数据源目录：`/home/zhanghs/tdx_converter/parquets/`

从通达信转换的 parquet 文件，文件命名规则：
- `{symbol}.{market}.parquet` — 未复权
- `{symbol}.{market}_hfq.parquet` — 后复权
- `{symbol}.{market}_qfq.parquet` — 前复权

数据表结构（以 trade_date 为索引）：
| 列名 | 类型 | 说明 |
|------|------|------|
| Open | float64 | 开盘价 |
| High | float64 | 最高价 |
| Low | float64 | 最低价 |
| Close | float64 | 收盘价 |
| Volume | int64 | 成交量 |
| Amount | float64 | 成交额 |
| adj_factor | float64 | 复权因子 |

项目数据文件：
- `data/512890.SH_hfq.parquet` — 后复权（默认加载）
- `data/512890.SH.parquet` — 未复权（前向复权计算用）

## 项目架构

```
vectorbt_test/
├── main.py                      # Polyfit-Switch vs Polyfit-Grid 对比入口
├── strategies/
│   ├── ma_grid.py               # MA 网格策略（CPU + GPU CuPy RawKernel）
│   ├── ma_switch.py             # MA-Switch 双模式策略（CPU + GPU）
│   ├── polyfit_grid.py          # Polyfit-Grid 纯网格策略（★ 当前最优）
│   ├── polyfit_switch.py        # Polyfit-Switch 双模式策略
│   └── intraday_execution.py    # 日内分批限价执行（实验性）
├── workflows/
│   ├── polyfit_switch.py        # 评分方法对比工作流
│   └── compare_symbols.py       # 跨品种对比
├── utils/
│   ├── backtest.py              # 回测引擎（VectorBT CPU + CuPy GPU batch）
│   ├── data.py                  # 数据加载
│   ├── gpu.py                   # GPU/CUDA 检测与工具
│   ├── indicators.py            # 技术指标（MA / Polyfit 基线）
│   ├── reports.py               # VectorBT HTML 报告生成
│   ├── scan.py                  # 参数扫描通用框架
│   ├── scoring.py               # 参数评分（return/balanced/robust）
│   └── walkforward.py           # Walk-Forward 窗口生成与分析
└── reports/                     # 回测报告输出
```

### 数据目录结构

```
data/
├── 1d/                          # [待迁移] 日线数据
├── 1m/                          # 1 分钟线
│   └── 512890.SH_1min.parquet   # 2026-01→2026-04, 65天, 15600条
├── 5m/                          # 5 分钟线
│   └── 512890.SH_5min.parquet   # 2024-06→2026-04, 450天, 21600条
├── 512890.SH_hfq.parquet        # 日线后复权（默认策略数据源）
├── 512890.SH.parquet            # 日线未复权（含 adj_factor，用于日内执行价格转换）
└── 510880.SH*.parquet           # 上证红利 ETF（跨品种对比用）
```

### 核心执行模型

所有策略使用 **次日开盘价成交**，消除前视偏差：
```
bar i Close → 计算指标 → 生成信号 → bar i+1 Open 成交
```
GPU kernel 分离 fill_price（成交价）和 close（NAV 估值价）。

### 回测路径

| 路径 | 引擎 | 用途 |
|------|------|------|
| `run_backtest` | VectorBT CPU | 单次回测、最终报告 |
| `run_backtest_batch` | CuPy GPU kernel | 批量参数扫描 |

### Walk-Forward 框架

- `utils/walkforward.py` — 按日历月/年生成训练/测试窗口
- **所有回测默认使用 Walk-Forward 方式**，禁止在全量数据上直接扫描最优参数后报告"收益"（会引入未来数据和过拟合偏差）
- 支持自定义评分函数（`best_selector` 参数）
- 每次回测完成后，必须同时列出 **return / balanced / robust** 三种评分方法的结果
- 训练窗口长度（月数）也是超参数，需纳入扫描范围（当前最优：22 个月）

### 三种评分方法说明

| 方法 | 函数 | 逻辑 | 适用场景 |
|------|------|------|---------|
| `return` | `select_by_return` | 选训练期总收益最高的参数 | 趋势明显、回撤容忍度高 |
| `balanced` | `select_balanced` | 综合收益(50%)+Sharpe(30%)-回撤(20%) | 需要均衡风险收益 |
| `robust` | `select_robust` | 要求各训练子段均正收益，再选最高 Sharpe | 需要稳健性（当前 fallback 到 balanced） |

### GPU 加速说明

- **所有参数扫描均使用 GPU**（CuPy RawKernel），每次调用处理 699,840 个参数组合（9 indicator sets × 77,760 grid combos）
- 单窗口扫描 ~10s（GPU），其中信号生成 ~8s，回测 ~2s
- 速度瓶颈：Walk-Forward 窗口数量 × 10s/窗口。例如 29 个窗口 ≈ 5 分钟
- 未走 GPU 的情况：仅当 CuPy 不可用时回退到 CPU（逐个参数组合循环，极慢）

## Polyfit-Grid 当前最优策略（★ 替代 Polyfit-Switch）

- **Polyfit-Grid** = 纯 Grid 均值回复网格策略，**已移除 Switch 模式**
- 基线：滑动窗口线性回归（252 天，固定），替代 SMA
- 最大持仓 45 天，仓位 0.92-0.99
- 全量回测：+206.5%（vs Polyfit-Switch +195.8%），Sharpe 1.88（vs 1.61），回撤 -8.8%（vs -15.8%）
- WF 22m return: OOS +11.2%，α +0.6%，87% 正收益，47% 跑赢 BH
- **Switch 模式确认是负贡献**：18 笔 Switch 交易边际收益 -9.5%，且增大了回撤
- 策略文件：`strategies/polyfit_grid.py`（纯 Grid + GPU 批量扫描）

### 策略设计约束

- `fit_window_days` 必须固定为 252。放开扫描（60/120/252）会导致训练期严重过拟合（训练收益 80-170%，OOS 崩到 2.9%）。252 天的固定窗口起到了正则化作用，防止基线过度拟合短期噪声。
- 两阶段 GPU 扫描：Stage 1 全量扫描 77,760 组 grid 参数（GPU batch，~10s/窗口），Stage 2 全排列 switch 参数（~1s/窗口）

## 重要设计约束

### 禁止引入复杂 ML 策略

**XGBoost、神经网络（MLP/LSTM/Transformer 等）不适用于本项目。** 原因：

1. **数据量太小**：总数据仅 1763 条日线 bar，每 Walk-Forward 训练窗口约 500 条。这是 ML 的极小样本。
2. **已验证的失败案例**：尝试用 XGBoost（20 棵树，depth=3）过滤 Polyfit-Switch 入场信号，结果 OOS 从 +20.5% 暴跌至 +6.7%，训练标签噪声过大、模型无法泛化。已删除相关代码（`strategies/polyfit_xgboost.py`、`workflows/polyfit_xgboost_compare.py`）。
3. **神经网络更不可行**：最小 MLP 也有数千参数，500 条数据训练必然过拟合。训练收益 200%，测试收益 -20%。
4. **参数化规则策略更适合**：当前最优的 Polyfit-Switch 仅有 18 个可解释参数，GPU 全量扫描可在 10 秒内完成。规则策略在小数据上比 ML 更稳健、更可解释、更容易调试。

如果将来数据量增长到 10 万条以上（约 400 年日线），可以重新评估 ML 方案的可行性。

### 回测规范（必须遵守）

1. **默认使用 Walk-Forward**：所有回测必须以 Walk-Forward 方式运行（训练→OOS 测试），禁止仅在全量数据上扫描最优参数后直接报告"收益"——这会引入未来数据和严重过拟合偏差。

2. **每次回测输出三种评分结果**：return / balanced / robust 必须同时列出，格式如下：
   ```
   训练  评分       OOS       α        BH     sharpe   max_dd    pos    >BH    w
   22    return     +X%       +X%      +X%     X.XXX    -X%      X%    X%     X
   22    balanced   +X%       +X%      +X%     X.XXX    -X%      X%    X%     X
   22    robust     +X%       +X%      +X%     X.XXX    -X%      X%    X%     X
   ```

3. **训练月数是超参数**：不仅 22m/24m，训练窗口长度本身也需扫描（12-36 月，步长 1 月）。当前最优：22 个月。

4. **严禁前视偏差**：
   - 所有信号基于 bar i Close → bar i+1 Open 执行
   - 指标计算使用 warmup 数据（训练期之前 12 个月），不偷看测试期
   - 参数选择仅在训练集上进行，测试集仅用于评估

### GPU 扫描耗时参考

| 扫描类型 | 组合数 | 耗时/窗口 | 说明 |
|---------|--------|----------|------|
| Polyfit-Grid | 699,840 (9×77,760) | ~10s | 一次 GPU kernel 调用 |
| Polyfit-Switch Stage1 | 699,840 (9×77,760) | ~10s | 同上 |
| Polyfit-Switch Stage2 | ~2000 | ~1s | Switch 参数排列 |
| Walk-Forward 全量 | N窗口 × 10s | ~5-60分钟 | 取决于窗口数量 |