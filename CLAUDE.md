# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境

- 操作系统：WSL2 Debian (Linux 6.6.87-microsoft-standard-WSL2)
- Python 版本：3.14，通过 `.python-version` 管理
- 包管理：UV (`uv` / `uv pip` / `uv run`)，不要使用裸 `pip` 或 `python`
- 虚拟环境位于项目根目录 `.venv/`

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

- **VectorBT** (`vectorbt`) — 量化回测核心框架。项目主要依赖此库进行策略回测、K线数据处理、技术指标计算、投资组合模拟和可视化。

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

读取数据使用 `pd.read_parquet(path)`，不需要额外的存储格式转换。

## 项目架构

当前处于初始化阶段。后续开发遵循：
- 策略文件放在独立模块中，与回测入口（main.py）分离
- 数据读取统一从 `/home/zhanghs/tdx_converter/parquets/` 加载
- 所有 Python 依赖通过 `uv add` 添加到 `pyproject.toml`
- 使用 VectorBT 的 `Data` / `Indicator` / `Portfolio` / `Strategy` 类构建回测流程
