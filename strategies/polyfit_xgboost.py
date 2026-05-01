"""Polyfit-XGBoost Strategy — Polyfit-Switch + XGBoost 信号过滤器。

在 Polyfit-Switch 信号基础上，训练一个轻量 XGBoost 分类器来过滤入场信号。
训练数据中盈利的入场 → label 1，亏损的入场 → label 0。
测试时只执行 model.predict_proba > threshold 的入场信号。

XGBoost 在小数据上的天然正则化使其比 NN 更适合这个场景：
  - 树模型参数少（20 棵树 × 深度 3 = 60 个叶子）
  - 天然处理特征交互
  - 输出概率可调节过滤阈值
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.backtest import run_backtest


def extract_entry_features(
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    close: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    entry_indices: np.ndarray,
) -> np.ndarray:
    """从入场 bar 提取特征矩阵 [n_entries, n_features]。

    特征选择原则：只用入场时刻已知的信息，无前视偏差。
    """
    if len(entry_indices) == 0:
        return np.zeros((0, 8))

    features = np.column_stack([
        dev_pct[entry_indices],                           # 偏离度
        dev_trend[entry_indices],                         # 偏离趋势
        rolling_vol_pct[entry_indices],                   # 波动率
        close[entry_indices] / poly_base[entry_indices] - 1,  # 同 dev_pct（冗余但可能有用）
        (ma_fast[entry_indices] - ma_slow[entry_indices])
        / (ma_slow[entry_indices] + 1e-9),               # MA 交叉状态
        np.clip(rolling_vol_pct[entry_indices] * 100, 0, 10),  # 波动率缩放
        np.abs(dev_pct[entry_indices]) / (rolling_vol_pct[entry_indices] + 1e-9),  # 偏离/波动比
        dev_trend[entry_indices] / (rolling_vol_pct[entry_indices] + 1e-9),  # 趋势/波动比
    ])
    return features


def _find_matching_exits(
    entries: np.ndarray, exits: np.ndarray,
) -> dict:
    """将入场映射到对应的离场。

    模拟简单订单匹配：入场后遇到的第一个离场信号即为对应离场。
    """
    entry_to_exit = {}
    pending_entries = []

    for i in range(len(entries)):
        if entries[i]:
            pending_entries.append(i)
        if exits[i] and pending_entries:
            # exit 信号 — 关闭最早未匹配的入场
            entry_idx = pending_entries.pop(0)
            entry_to_exit[entry_idx] = i

    return entry_to_exit


def train_xgb_filter(
    close_train: np.ndarray,
    entries_train: np.ndarray,
    exits_train: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    n_estimators: int = 20,
    max_depth: int = 3,
    min_child_weight: int = 3,
    subsample: float = 0.8,
) -> xgb.XGBClassifier | None:
    """在训练数据上训练 XGBoost 入场过滤器。

    Returns:
        XGBClassifier 或 None（训练数据不足时）
    """
    entry_to_exit = _find_matching_exits(entries_train, exits_train)
    if len(entry_to_exit) < 6:
        return None  # 数据太少，无法训练

    entry_indices = np.array(list(entry_to_exit.keys()))
    labels = np.array([
        1 if close_train[entry_to_exit[i]] > close_train[i] else 0
        for i in entry_indices
    ])
    # 检查标签是否两极分化（全是同一类无法学习）
    if labels.mean() < 0.05 or labels.mean() > 0.95:
        return None

    features = extract_entry_features(
        dev_pct, dev_trend, rolling_vol_pct,
        poly_base, close_train,
        ma_fast, ma_slow, entry_indices,
    )

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    model.fit(features, labels)
    return model


def filter_signals(
    entries: np.ndarray,
    dev_pct: np.ndarray,
    dev_trend: np.ndarray,
    rolling_vol_pct: np.ndarray,
    poly_base: np.ndarray,
    close: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    model: xgb.XGBClassifier,
    threshold: float = 0.45,
) -> tuple[np.ndarray, int, int]:
    """用 XGBoost 过滤入场信号。

    对每个入场 bar 提取特征，模型预测盈利概率。
    概率 < threshold 的信号被过滤掉。

    Returns:
        filtered_entries, total_entries, kept_entries
    """
    entry_indices = np.where(entries)[0]
    if len(entry_indices) == 0:
        return entries.copy(), 0, 0

    features = extract_entry_features(
        dev_pct, dev_trend, rolling_vol_pct,
        poly_base, close,
        ma_fast, ma_slow, entry_indices,
    )

    probs = model.predict_proba(features)[:, 1]
    keep_mask = probs >= threshold

    filtered = entries.copy()
    for i, idx in enumerate(entry_indices):
        if not keep_mask[i]:
            filtered[idx] = False

    return filtered, len(entry_indices), keep_mask.sum()
