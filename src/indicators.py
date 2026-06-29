"""Teknik gösterge kütüphanesi (vektörel).

Hepsi pandas Series/DataFrame alır, aynı index'te Series döndürür.
Gün-içi için VWAP gibi göstergeler her işlem günü başında sıfırlanır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stochastic(high, low, close, k_period=14, d_period=3):
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    md = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


def bollinger(close, period=20, num_std=2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return mid - num_std * sd, mid, mid + num_std * sd


def atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(high, low, close, period=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(high, low, close, period)
    pdi = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / a.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def vwap_daily(high, low, close, volume):
    """Her işlem günü başında sıfırlanan VWAP."""
    tp = (high + low + close) / 3
    day = close.index.normalize()
    pv = (tp * volume).groupby(day).cumsum()
    vv = volume.groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def supertrend(high, low, close, period=10, mult=3.0):
    """Supertrend yönü: +1 (yukarı trend) / -1 (aşağı trend)."""
    a = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    n = len(close)
    dir_ = np.ones(n, dtype=np.int8)
    fu = upper.to_numpy().copy()
    fl = lower.to_numpy().copy()
    c = close.to_numpy()
    for i in range(1, n):
        fu[i] = min(upper.iloc[i], fu[i - 1]) if c[i - 1] <= fu[i - 1] else upper.iloc[i]
        fl[i] = max(lower.iloc[i], fl[i - 1]) if c[i - 1] >= fl[i - 1] else lower.iloc[i]
        if c[i] > fu[i - 1]:
            dir_[i] = 1
        elif c[i] < fl[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
    return pd.Series(dir_, index=close.index)


def obv(close, volume):
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()
