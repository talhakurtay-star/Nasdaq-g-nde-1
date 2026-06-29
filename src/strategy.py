"""Alım-satım stratejileri.

Her strateji bir OHLCV DataFrame alır ve aynı index'te bir `signal` Series döndürür:
    +1 = long  (al / yukarı pozisyon)
     0 = nakit (pozisyon kapalı)
    -1 = short (açığa sat / aşağı pozisyon)

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
class MeanReversionStrategy:
    """Ortalamaya dönüş (mean-reversion) — gün-içi endeksler için.

    Mantık: Fiyat ortalamasından çok uzaklaşınca geri döner.
      - z-skoru = (close - SMA) / std
      - z <= -entry_z  → çok düştü, LONG (geri toparlamasını bekle)
      - z >= +entry_z  → çok yükseldi, SHORT (geri çekilmesini bekle)
      - |z| <= exit_z  → ortalamaya döndü, pozisyonu kapat
    Opsiyonel RSI teyidi: long için RSI<rsi_long, short için RSI>rsi_short.

    Parametreler:
        lookback: ortalama/std penceresi (bar).
        entry_z: giriş eşiği (kaç standart sapma).
        exit_z: çıkış eşiği (ortalamaya bu kadar yaklaşınca kapat).
        use_rsi: RSI teyidi kullan.
        rsi_period, rsi_long, rsi_short: RSI parametreleri.
        allow_long / allow_short: yönleri aç/kapat.
    """

    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_long: float = 35.0
    rsi_short: float = 65.0
    allow_long: bool = True
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback >= 2 olmalı.")
        if self.entry_z <= self.exit_z:
            raise ValueError("entry_z, exit_z'den büyük olmalı.")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        mean = close.rolling(self.lookback, min_periods=self.lookback).mean()
        std = close.rolling(self.lookback, min_periods=self.lookback).std(ddof=0)
        z = ((close - mean) / std.replace(0, np.nan)).to_numpy()

        if self.use_rsi:
            rsi_vals = rsi(close, self.rsi_period).to_numpy()
        else:
            rsi_vals = np.full(len(close), np.nan)

        n = len(close)
        out = np.zeros(n, dtype=np.int8)
        pos = 0
        for i in range(n):
            zi = z[i]
            if np.isnan(zi):
                pos = 0
                continue
            if pos == 0:
                if self.allow_long and zi <= -self.entry_z:
                    if not self.use_rsi or rsi_vals[i] <= self.rsi_long:
                        pos = 1
                elif self.allow_short and zi >= self.entry_z:
                    if not self.use_rsi or rsi_vals[i] >= self.rsi_short:
                        pos = -1
            elif pos == 1:
                if zi >= -self.exit_z:        # ortalamaya döndü
                    pos = 0
            elif pos == -1:
                if zi <= self.exit_z:
                    pos = 0
            out[i] = pos

        return pd.Series(out, index=close.index, dtype=int)


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

        trend_up = (fast_ma > slow_ma).to_numpy()

        if self.use_rsi:
            rsi_vals = rsi(close, self.rsi_period)
            allow_entry = (rsi_vals < self.rsi_overbought).to_numpy()
        else:
            allow_entry = np.ones(len(close), dtype=bool)

        valid = (~slow_ma.isna()).to_numpy()

        # Pozisyon mantığı: trend yukarıyken (ve giriş izinliyken) long, değilse nakit.
        # Bir kez girince trend bozulana kadar tutulur. (numpy ile hızlı stateful döngü)
        n = len(close)
        out = np.zeros(n, dtype=np.int8)
        in_position = False
        for i in range(n):
            if not valid[i]:
                in_position = False
                continue
            if not in_position:
                if trend_up[i] and allow_entry[i]:
                    in_position = True
            elif not trend_up[i]:
                in_position = False
            out[i] = 1 if in_position else 0

        return pd.Series(out, index=close.index, dtype=int)
