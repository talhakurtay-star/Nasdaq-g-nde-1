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
class OpeningRangeBreakout:
    """Açılış Aralığı Kırılımı (Opening Range Breakout — ORB).

    Veri keşfinde NAS100'de kırılımların ~%76'sı devam ediyor (momentum).
    Mantık (her gün):
      - Açılış saatinden (open_hour) itibaren ilk `or_minutes` dakika "açılış aralığı"
        (OR) sayılır: bu pencerenin en yüksek/en düşüğü belirlenir.
      - Pencere bittikten sonra fiyat OR-üstünü kırarsa → LONG, OR-altını kırarsa → SHORT.
      - İlk kırılım yönü gün boyu KİLİTLENİR (whipsaw'da yön değiştirmez).
      - Çıkış motora bırakılır (gün-sonu kapanışı / günlük target-stop / işlem SL-TP).

    Parametreler (hepsi broker saati / dakika):
        open_hour: açılış saati (en hareketli saat; NAS100 için ~16).
        or_minutes: açılış aralığı süresi.
        allow_long / allow_short: yönleri aç/kapat.
        buffer_pct: kırılım için OR sınırına eklenen küçük tampon (yanlış kırılımı azaltır).
    """

    open_hour: int = 16
    or_minutes: int = 30
    allow_long: bool = True
    allow_short: bool = True
    buffer_pct: float = 0.0
    # Konviksiyon filtresi: açılış aralığı (OR) genişliği fiyatın %'si olarak
    # bu aralıkta değilse o gün işlem yapma. (Yalnız güçlü-momentum günleri.)
    min_or_range_pct: float = 0.0
    max_or_range_pct: float = 100.0
    # Kırılım gücü: bar sadece dokunmasın, KAPANIŞTA seviyeyi geçsin (sahte fitil eler).
    require_close_break: bool = False
    # Trend teyidi: EMA(periyot) yukarıdaysa sadece long, aşağıdaysa sadece short (0 = kapalı).
    trend_ema_period: int = 0
    # VWAP filtresi: kırılım VWAP'ın doğru tarafındaysa al (long ise close>VWAP). Yanlış taraf = tuzak.
    vwap_filter: bool = False
    # Gün filtresi: sadece bu hafta-içi günlerinde işlem (0=Pzt..4=Cum). None = hepsi.
    weekdays: tuple | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        idx = data.index
        minute = (idx.hour * 60 + idx.minute).to_numpy()
        day = idx.normalize().to_numpy()
        dow = idx.dayofweek.to_numpy()
        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        close = data["close"].to_numpy()
        if self.trend_ema_period > 0:
            ema_arr = ema(data["close"], self.trend_ema_period).to_numpy()
        else:
            ema_arr = None
        if self.vwap_filter:
            from . import indicators as _ind
            vwap_arr = _ind.vwap_daily(data["high"], data["low"], data["close"], data["volume"]).to_numpy()
        else:
            vwap_arr = None
        allowed = set(self.weekdays) if self.weekdays is not None else None
        n = len(idx)
        out = np.zeros(n, dtype=np.int8)
        open_min = self.open_hour * 60
        or_end = open_min + self.or_minutes
        buf = self.buffer_pct / 100.0

        start = 0
        for k in range(n + 1):
            if k == n or day[k] != day[start]:
                if allowed is None or dow[start] in allowed:
                    self._fill_day(out, minute, high, low, close, ema_arr, vwap_arr,
                                   start, k, open_min, or_end, buf)
                start = k
                if k == n:
                    break
        return pd.Series(out, index=idx, dtype=int)

    def _fill_day(self, out, minute, high, low, close, ema_arr, vwap_arr, s, e, open_min, or_end, buf):
        # 1) Açılış aralığını (OR) belirle
        or_high, or_low = -np.inf, np.inf
        has_or = False
        for j in range(s, e):
            if open_min <= minute[j] < or_end:
                or_high = max(or_high, high[j])
                or_low = min(or_low, low[j])
                has_or = True
            elif minute[j] >= or_end:
                break
        if not has_or:
            return
        # Konviksiyon filtresi: OR genişliği (fiyatın %'si)
        or_mid = (or_high + or_low) / 2
        or_range_pct = (or_high - or_low) / or_mid * 100 if or_mid > 0 else 0
        if not (self.min_or_range_pct <= or_range_pct <= self.max_or_range_pct):
            return
        up_level = or_high * (1 + buf)
        dn_level = or_low * (1 - buf)

        # 2) Pencere sonrası ilk kırılımı bul, yönü gün boyu kilitle
        day_dir = 0
        for j in range(s, e):
            if minute[j] < or_end:
                continue
            if day_dir == 0:
                # Kırılım: kapanış teyidi isteniyorsa close, yoksa high/low
                broke_up = (close[j] >= up_level) if self.require_close_break else (high[j] >= up_level)
                broke_dn = (close[j] <= dn_level) if self.require_close_break else (low[j] <= dn_level)
                # Trend teyidi
                if ema_arr is not None:
                    up_ok = close[j] > ema_arr[j]
                    dn_ok = close[j] < ema_arr[j]
                else:
                    up_ok = dn_ok = True
                # VWAP filtresi: long ise VWAP üstünde, short ise altında olmalı
                if vwap_arr is not None and not np.isnan(vwap_arr[j]):
                    up_ok = up_ok and close[j] > vwap_arr[j]
                    dn_ok = dn_ok and close[j] < vwap_arr[j]
                if self.allow_long and broke_up and up_ok:
                    day_dir = 1
                elif self.allow_short and broke_dn and dn_ok:
                    day_dir = -1
            if day_dir != 0:
                out[j] = day_dir


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
