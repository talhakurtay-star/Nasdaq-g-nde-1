"""Olay-tabanlı basit backtest motoru.

Tasarım ilkeleri:
  - Sinyaller barın KAPANIŞINDA üretilir, bir SONRAKİ barın AÇILIŞINDA uygulanır
    (gerçekçi yürütme; ileriye dönük veri sızıntısı yok).
  - Long-only, tüm sermaye tek pozisyona (all-in) girer.
  - Komisyon ve slipaj her işlemde dikkate alınır.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    commission: float = 0.0005   # işlem başına oran (örn. 0.0005 = %0.05)
    slippage: float = 0.0002     # yürütme kayması oranı


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict

    def summary(self) -> str:
        from .metrics import format_metrics
        return format_metrics(self.metrics)


class Backtester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """Veri ve sinyallerle backtest çalıştır."""
        cfg = self.config
        data = data.copy()
        # Sinyali bir sonraki bara kaydır: t'de üretilen sinyal t+1'de uygulanır.
        target_pos = signals.reindex(data.index).fillna(0).shift(1).fillna(0)

        cash = cfg.initial_cash
        shares = 0.0
        position = 0  # 0 = nakit, 1 = long
        entry_price = 0.0
        entry_time = None

        equity_curve = []
        trades = []

        opens = data["open"]
        closes = data["close"]

        for i, ts in enumerate(data.index):
            desired = int(target_pos.iloc[i])
            price_open = opens.iloc[i]

            # Pozisyon değişimi gerekiyorsa açılış fiyatından işlem yap
            if desired == 1 and position == 0:
                # AL
                fill = price_open * (1 + cfg.slippage)
                invest = cash
                cost = invest * cfg.commission
                shares = (invest - cost) / fill
                cash = 0.0
                position = 1
                entry_price = fill
                entry_time = ts

            elif desired == 0 and position == 1:
                # SAT
                fill = price_open * (1 - cfg.slippage)
                proceeds = shares * fill
                cost = proceeds * cfg.commission
                cash = proceeds - cost
                pnl = cash - (shares * entry_price)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": fill,
                    "shares": shares,
                    "pnl": pnl,
                    "return_pct": (fill / entry_price - 1) * 100,
                })
                shares = 0.0
                position = 0

            # Bar kapanışında equity'yi işaretle (mark-to-market)
            mark = cash + shares * closes.iloc[i]
            equity_curve.append(mark)

        # Açık pozisyon kaldıysa son kapanışta kapat
        if position == 1:
            fill = closes.iloc[-1] * (1 - cfg.slippage)
            proceeds = shares * fill
            cost = proceeds * cfg.commission
            cash = proceeds - cost
            pnl = cash - (shares * entry_price)
            trades.append({
                "entry_time": entry_time,
                "exit_time": data.index[-1],
                "entry_price": entry_price,
                "exit_price": fill,
                "shares": shares,
                "pnl": pnl,
                "return_pct": (fill / entry_price - 1) * 100,
            })

        equity = pd.Series(equity_curve, index=data.index, name="equity")
        trades_df = pd.DataFrame(trades)
        metrics = compute_metrics(equity, trades_df)
        return BacktestResult(equity=equity, trades=trades_df, metrics=metrics)
