"""Alım-satım stratejileri.

Her strateji bir OHLCV DataFrame alır ve aynı index'te bir `signal` Series döndürür:
    +1 = long (pozisyonda ol / al)
     0 = nakit (pozisyon kapalı)

Sinyaller backtest motoru tarafından bir sonraki barın açılışında uygulanır
(ileriye dönük veri sızıntısını önlemek için).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Basit hareketli ortalama."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Üssel hareketli ortalama."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Göreceli Güç Endeksi (Wilder yöntemi)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@dataclass
class MACrossStrategy:
    """Hareketli ortalama kesişimi (opsiyonel RSI filtresi ile).

    Kurallar:
      - Hızlı MA, yavaş MA'nın üstündeyse trend yukarı kabul edilir.
      - RSI filtresi açıksa, aşırı alımda (RSI > rsi_overbought) yeni long açılmaz.
      - Hızlı MA yavaşın altına inince pozisyon kapatılır.

    Parametreler:
        fast: Hızlı MA periyodu.
        slow: Yavaş MA periyodu.
        use_rsi: RSI filtresini kullan.
        rsi_period: RSI periyodu.
        rsi_overbought: Bu seviyenin üstünde long açma.
        ma_type: "sma" ya da "ema".
    """

    fast: int = 20
    slow: int = 50
    use_rsi: bool = True
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    ma_type: str = "ema"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast periyodu slow'dan küçük olmalı.")
        if self.ma_type not in ("sma", "ema"):
            raise ValueError("ma_type 'sma' veya 'ema' olmalı.")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ma_func = ema if self.ma_type == "ema" else sma
        fast_ma = ma_func(close, self.fast)
        slow_ma = ma_func(close, self.slow)

        trend_up = fast_ma > slow_ma

        if self.use_rsi:
            rsi_vals = rsi(close, self.rsi_period)
            allow_entry = rsi_vals < self.rsi_overbought
        else:
            allow_entry = pd.Series(True, index=close.index)

        # Pozisyon mantığı: trend yukarıyken (ve giriş izinliyken) long, değilse nakit.
        # Bir kez girince trend bozulana kadar tutulur.
        signal = pd.Series(0, index=close.index, dtype=int)
        in_position = False
        for i in range(len(close)):
            if pd.isna(slow_ma.iloc[i]):
                signal.iloc[i] = 0
                continue
            if not in_position:
                if trend_up.iloc[i] and allow_entry.iloc[i]:
                    in_position = True
            else:
                if not trend_up.iloc[i]:
                    in_position = False
            signal.iloc[i] = 1 if in_position else 0

        return signal
