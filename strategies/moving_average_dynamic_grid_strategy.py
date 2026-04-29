from __future__ import annotations

import pandas as pd
import numpy as np
from backtesting import Strategy


class MovingAverageDynamicGridStrategy(Strategy):
    """基于移动平均基准的动态阈值网格策略（仅做多）。"""

    base_grid_pct = 0.012
    volatility_scale = 1.0
    trend_sensitivity = 8.0
    max_grid_levels = 3
    take_profit_grid = 0.85
    stop_loss_grid = 1.6
    max_holding_days = 30
    cooldown_days = 1
    min_signal_strength = 0.45
    position_size = 0.5
    position_sizing_coef = 30.0
    initial_position = 0.0

    def init(self) -> None:
        self.entry_bar_index: int | None = None
        self.entry_price = np.nan
        self.entry_level = 1
        self.entry_grid_step = np.nan
        self.cooldown = 0
        self.initialized_position = False
        self.ending_position = 0.0
        self.current_entry_reason = ""
        self.trade_reason_records: list[dict[str, str | int | float]] = []

    def _build_entry_reason(self, close: float, ma_base: float, dev_pct: float, dynamic_grid_step: float, entry_level: int) -> str:
        return (
            f"mean_reversion_grid;base={ma_base:.4f};dev={dev_pct:.4%};"
            f"grid_step={dynamic_grid_step:.4%};level={entry_level};close={close:.4f}"
        )

    def _build_exit_reason(
        self,
        dev_pct: float,
        tp_threshold: float,
        sl_threshold: float,
        hold_limit: bool,
    ) -> str:
        reasons: list[str] = []
        if hold_limit:
            reasons.append(f"max_holding_days({int(self.max_holding_days)})")
        if dev_pct >= tp_threshold:
            reasons.append(f"take_profit_grid(dev={dev_pct:.4%}>=tp={tp_threshold:.4%})")
        if dev_pct <= -sl_threshold:
            reasons.append(f"stop_loss_grid(dev={dev_pct:.4%}<=-{sl_threshold:.4%})")
        return "; ".join(reasons) if reasons else "grid_exit_runtime_unknown"

    def _record_exit_reason(self, bar_index: int, close: float, portion: float, exit_reason: str) -> None:
        exit_time = self.data.index[-1]
        entry_time = self.data.index[self.entry_bar_index] if self.entry_bar_index is not None else exit_time
        self.trade_reason_records.append(
            {
                "EntryTime": pd.Timestamp(entry_time),
                "ExitTime": pd.Timestamp(exit_time),
                "EntryReason": self.current_entry_reason or "entry_runtime_unknown",
                "ExitReason": exit_reason,
                "ExitPortion": float(portion),
                "SignalBar": int(bar_index),
                "SignalClose": float(close),
            }
        )

    def _reset_state_if_flat(self) -> None:
        if not self.position:
            self.entry_bar_index = None
            self.entry_price = np.nan
            self.entry_level = 1
            self.entry_grid_step = np.nan

    def next(self) -> None:
        self._reset_state_if_flat()

        close = float(self.data.Close[-1])
        ma_base = float(self.data.MABase[-1])
        dev_pct = float(self.data.MADevPct[-1])
        dev_trend = float(self.data.MADevTrend[-1])
        rolling_vol_pct = float(self.data.RollingVolPct[-1])
        bar_index = len(self.data) - 1

        if any(np.isnan(v) for v in [close, ma_base, dev_pct, dev_trend, rolling_vol_pct]) or ma_base <= 0:
            return

        if not self.initialized_position:
            self.initialized_position = True
            init_size = float(np.clip(self.initial_position, 0.0, 1.0))
            if init_size > 0 and not self.position:
                self.buy(size=init_size)
                self.entry_bar_index = bar_index
                self.entry_price = close
                self.entry_level = 1
                self.entry_grid_step = max(float(self.base_grid_pct), 1e-9)
                self.ending_position = init_size
                self.current_entry_reason = "initial_position_carry"
                return

        if self.cooldown > 0:
            self.cooldown -= 1

        vol_multiplier = 1.0 + float(self.volatility_scale) * max(rolling_vol_pct, 0.0)
        dynamic_grid_step = (
            float(self.base_grid_pct)
            * (1.0 + float(self.trend_sensitivity) * abs(dev_trend))
            * vol_multiplier
        )
        dynamic_grid_step = max(dynamic_grid_step, float(self.base_grid_pct) * 0.3)

        if not self.position:
            if self.cooldown > 0:
                return

            signal_strength = abs(dev_pct) / max(dynamic_grid_step, 1e-9)
            entry_level = int(np.clip(np.floor(signal_strength), 1, int(self.max_grid_levels)))
            entry_threshold = -entry_level * dynamic_grid_step

            if dev_pct <= entry_threshold and signal_strength >= float(self.min_signal_strength):
                size = float(
                    np.clip(
                        abs(dev_pct) * (1.0 + max(rolling_vol_pct, 0.0)) * float(self.position_sizing_coef),
                        0.0,
                        float(np.clip(self.position_size, 0.0, 1.0)),
                    )
                )
                if size <= 0:
                    return
                self.buy(size=size)
                self.entry_bar_index = bar_index
                self.entry_price = close
                self.entry_level = entry_level
                self.entry_grid_step = dynamic_grid_step
                self.ending_position = size
                self.current_entry_reason = self._build_entry_reason(close, ma_base, dev_pct, dynamic_grid_step, entry_level)
            return

        if self.entry_bar_index is None:
            self.entry_bar_index = bar_index
        if np.isnan(self.entry_price):
            self.entry_price = float(self.trades[-1].entry_price) if self.trades else close
        if np.isnan(self.entry_grid_step):
            self.entry_grid_step = dynamic_grid_step

        holding_days = bar_index - self.entry_bar_index
        hold_limit = holding_days >= int(self.max_holding_days)

        ref_step = max(dynamic_grid_step, float(self.entry_grid_step))
        tp_threshold = self.entry_level * ref_step * float(self.take_profit_grid)
        sl_threshold = self.entry_level * ref_step * float(self.stop_loss_grid)

        take_profit = dev_pct >= tp_threshold
        stop_loss = dev_pct <= -sl_threshold

        if hold_limit or take_profit or stop_loss:
            exit_size = float(
                np.clip(
                    abs(dev_pct) * (1.0 + max(rolling_vol_pct, 0.0)) * float(self.position_sizing_coef),
                    0.0,
                    max(float(self.ending_position), 0.0),
                )
            )
            if exit_size <= 0:
                return
            portion = float(np.clip(exit_size / max(float(self.ending_position), 1e-9), 0.0, 1.0))
            exit_reason = self._build_exit_reason(dev_pct, tp_threshold, sl_threshold, hold_limit)
            self._record_exit_reason(bar_index, close, portion, exit_reason)
            self.position.close(portion=portion)
            self.ending_position = max(0.0, float(self.ending_position) * (1.0 - portion))
            if self.ending_position <= 1e-6:
                self.ending_position = 0.0
                self.cooldown = int(self.cooldown_days)
                self.current_entry_reason = ""