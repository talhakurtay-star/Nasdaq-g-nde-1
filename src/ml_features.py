"""ML için nedensel (causal) özellik mühendisliği.

⛔ EN KRİTİK KURAL: HİÇBİR ÖZELLİK GELECEĞE BAKMAZ.
Her özellik yalnızca o barın KAPANIŞINA KADAR bilinen bilgiyle hesaplanır.
(Gün-içi kümülatifler cumsum/cummax/cummin ile nedenseldir.)
Hedef (target) geleceğe bakar ama SADECE eğitim etiketinde kullanılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """OHLCV'den nedensel özellik matrisi üret."""
    c, h, l, v = data["close"], data["high"], data["low"], data["volume"]
    day = data.index.normalize()
    f = pd.DataFrame(index=data.index)

    # Momentum / getiri (geçmiş)
    ret = c.pct_change()
    f["ret_1"] = ret
    f["ret_5"] = c.pct_change(5)
    f["ret_12"] = c.pct_change(12)
    f["ret_30"] = c.pct_change(30)

    # Volatilite
    f["atr_pct"] = ind.atr(h, l, c, 14) / c * 100
    f["rvol_20"] = ret.rolling(20).std() * 100

    # Osilatörler
    f["rsi_14"] = ind.rsi(c, 14)
    f["rsi_7"] = ind.rsi(c, 7)
    _, _, macd_hist = ind.macd(c)
    f["macd_hist"] = macd_hist / c * 100
    k, _ = ind.stochastic(h, l, c, 14, 3)
    f["stoch_k"] = k
    f["cci"] = ind.cci(h, l, c, 20)

    # Bollinger konumu (z-skoru)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    f["boll_z"] = (c - mid) / sd.replace(0, np.nan)

    # VWAP'tan uzaklık (gün-içi, nedensel)
    vwap = ind.vwap_daily(h, l, c, v)
    f["vwap_dist"] = (c - vwap) / c * 100

    # Hacim rejimi
    f["vol_ratio"] = v / v.rolling(50).mean().replace(0, np.nan)

    # Gün-içi zaman ve konum (nedensel: o ana kadarki gün yüksek/düşüğü)
    minute = data.index.hour * 60 + data.index.minute
    f["minute"] = minute
    f["hour_sin"] = np.sin(2 * np.pi * minute / 1440)
    f["hour_cos"] = np.cos(2 * np.pi * minute / 1440)
    day_high = h.groupby(day).cummax()
    day_low = l.groupby(day).cummin()
    rng = (day_high - day_low).replace(0, np.nan)
    f["day_range_pos"] = (c - day_low) / rng           # gün aralığında nerede (0..1)
    f["day_range_pct"] = rng / c * 100                 # günün şu ana dek genişliği

    # Mum gövdesi / fitil yapısı
    body = (c - data["open"]).abs()
    f["body_ratio"] = body / (h - l).replace(0, np.nan)

    return f


def make_target(data: pd.DataFrame, horizon: int = 12, deadband_pct: float = 0.0) -> pd.Series:
    """Hedef: `horizon` bar sonraki getirinin yönü (1=yukarı, 0=aşağı).

    deadband_pct > 0 ise küçük hareketler (gürültü) NaN ile elenir (eğitimde atılır).
    GELECEĞE BAKAR — sadece eğitim etiketi olarak kullanılır.
    """
    fwd = data["close"].shift(-horizon) / data["close"] - 1
    if deadband_pct > 0:
        b = deadband_pct / 100
        y = pd.Series(np.where(fwd > b, 1, np.where(fwd < -b, 0, np.nan)), index=data.index)
    else:
        y = (fwd > 0).astype(float)
    return y


FEATURE_COLS = [
    "ret_1", "ret_5", "ret_12", "ret_30", "atr_pct", "rvol_20",
    "rsi_14", "rsi_7", "macd_hist", "stoch_k", "cci", "boll_z",
    "vwap_dist", "vol_ratio", "minute", "hour_sin", "hour_cos",
    "day_range_pos", "day_range_pct", "body_ratio",
]
