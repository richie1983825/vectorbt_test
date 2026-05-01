"""参数评分与筛选模块。

提供多种最优参数选取策略，用于 Walk-Forward 训练阶段。
不同评分标准影响策略在样本外的表现均衡性。

用法：
    from utils.scoring import select_best
    best_row = select_best(scan_results, method='balanced')
"""

import numpy as np
import pandas as pd


def select_by_return(results: pd.DataFrame) -> pd.Series:
    """按总收益率选取（当前默认策略）。

    优点：最大化单窗口收益。
    缺点：容易过拟合到特定市场环境，导致窗口间收益不均衡。
    """
    return results.nlargest(1, "total_return").iloc[0]


def select_by_sharpe(results: pd.DataFrame) -> pd.Series:
    """按 Sharpe 比率选取。

    Sharpe = 收益 / 波动率，内在惩罚收益的不稳定性。
    偏好收益平稳的策略参数，但可能牺牲牛市中的爆发力。
    """
    return results.nlargest(1, "sharpe_ratio").iloc[0]


def select_by_calmar(results: pd.DataFrame) -> pd.Series:
    """按 Calmar 比率选取（= 总收益 / |最大回撤|）。

    强烈惩罚大幅回撤，偏好防守型参数。
    适合风险厌恶场景。
    """
    if "calmar_ratio" not in results.columns:
        results = results.copy()
        results["calmar_ratio"] = np.where(
            results["max_drawdown"].abs() > 1e-9,
            results["total_return"] / results["max_drawdown"].abs(),
            0.0,
        )
    return results.nlargest(1, "calmar_ratio").iloc[0]


def select_balanced(results: pd.DataFrame, return_weight: float = 0.5,
                    sharpe_weight: float = 0.3,
                    dd_penalty: float = 0.2) -> pd.Series:
    """均衡评分：综合收益、Sharpe、回撤三维度。

    Score = return_weight × total_return
          + sharpe_weight × sharpe_ratio
          - dd_penalty × |max_drawdown|

    设计思路：
      - total_return (50%)：保证绝对收益不差
      - sharpe_ratio (30%)：惩罚收益波动，偏好稳定策略
      - max_drawdown 惩罚 (20%)：惩罚大回撤，控制尾部风险

    相比纯 return 选取，balanced 倾向于选择在训练期内
    收益/回撤更均衡的参数组合，减少过拟合到特定行情。

    Args:
        results:        扫描结果 DataFrame
        return_weight:  收益权重
        sharpe_weight:  Sharpe 权重
        dd_penalty:     回撤惩罚权重
    """
    scored = results.copy()
    dd_abs = scored["max_drawdown"].abs()

    scored["balanced_score"] = (
        return_weight * scored["total_return"]
        + sharpe_weight * scored["sharpe_ratio"]
        - dd_penalty * dd_abs
    )
    # 额外惩罚：无交易或收益太低的组合
    scored.loc[scored["num_trades"] < 2, "balanced_score"] -= 0.5
    scored.loc[scored["total_return"] < 0, "balanced_score"] -= 0.3

    return scored.nlargest(1, "balanced_score").iloc[0]


def select_robust(results: pd.DataFrame, close_train: pd.Series | None = None,
                  n_segments: int = 4) -> pd.Series:
    """鲁棒评分：要求参数在训练期内各个子区间都表现良好。

    将训练期等分为 n_segments 段，要求：
      - 每段的收益至少为正（不能依赖某一特定行情）
      - 整体 Sharpe 较高

    逻辑：一个好的参数应该在各种行情下都能工作，
    而不是仅在某一段牛市中大赚特赚。

    Args:
        results:    扫描结果 DataFrame（需含各子段收益列或可计算）
        close_train: 训练期收盘价（用于划分子区间）
        n_segments:  子区间数量

    注意：此方法需要重新回测子区间，当前仅对已有的
    per-segment 指标列做筛选。如果 results 中没有子段指标，
    会回退到 balanced 评分。
    """
    # 尝试找子段收益列（如 segment_0, segment_1, ...）
    seg_cols = [c for c in results.columns if c.startswith("seg_")]
    if len(seg_cols) >= n_segments:
        scored = results.copy()
        # 所有子段收益必须 > -5%（不能有大亏的段）
        for col in seg_cols[:n_segments]:
            scored = scored[scored[col] > -0.05]
        if len(scored) == 0:
            # 如果没有组合满足条件，回退到 balanced
            return select_balanced(results)
        # 在满足条件的组合中，选 Sharpe 最高的
        return scored.nlargest(1, "sharpe_ratio").iloc[0]

    # 回退：用 balanced 评分
    return select_balanced(results)


# ── 评分方法注册表 ──
SCORING_METHODS = {
    "return": select_by_return,
    "sharpe": select_by_sharpe,
    "calmar": select_by_calmar,
    "balanced": select_balanced,
    "robust": select_robust,
}


def select_best(results: pd.DataFrame, method: str = "balanced",
                close_train: pd.Series | None = None) -> pd.Series:
    """统一入口：按指定的评分方法选取最优参数。

    Args:
        results:     扫描结果 DataFrame
        method:      评分方法名（"return"/"sharpe"/"calmar"/"balanced"/"robust"）
        close_train: 训练期收盘价（robust 方法需要）

    Returns:
        最优参数对应的行（pd.Series）
    """
    selector = SCORING_METHODS.get(method)
    if selector is None:
        raise ValueError(f"未知评分方法: {method}，可选: {list(SCORING_METHODS.keys())}")

    if method == "robust":
        return selector(results, close_train=close_train)
    return selector(results)
