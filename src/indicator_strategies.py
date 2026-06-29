"""Gösterge tabanlı sinyal üreticileri (hepsi -1/0/+1 Series döndürür).

Her üretici: generate(data, **params) -> pd.Series
İki tür var:
  - Trend takip (stateless): sinyal = göstergenin yönü.
  - Ortalamaya dönüş (stateful): koşula girince pozisyon aç, çıkış koşuluna kadar tut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _hold(entry_long, exit_long, entry_short, exit_short) -> np.ndarray:
    """Giriş/çıkış koşullarından (bool diziler) pozisyon dizisi (-1/0/+1) üret."""
    n = len(entry_long)
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    for i in range(n):
        if pos == 0:
            if entry_long[i]:
                pos = 1
            elif entry_short[i]:
                pos = -1
        elif pos == 1 and exit_long[i]:
            pos = 0
        elif pos == -1 and exit_short[i]:
            pos = 0
        out[i] = pos
    return out


# ───────────────── Trend takip (stateless: sinyal = yön) ─────────────────

def macd_trend(data, fast=12, slow=26, signal=9):
    line, sig, hist = ind.macd(data["close"], fast, slow, signal)
    s = np.sign(hist).fillna(0).astype(int)
    return pd.Series(s.to_numpy(), index=data.index)


def supertrend_trend(data, period=10, mult=3.0):
    d = ind.supertrend(data["high"], data["low"], data["close"], period, mult)
    return d.astype(int)


def vwap_trend(data, **_):
    v = ind.vwap_daily(data["high"], data["low"], data["close"], data["volume"])
    s = np.sign(data["close"] - v).fillna(0).astype(int)
    return pd.Series(s.to_numpy(), index=data.index)


def adx_macd_trend(data, adx_period=14, adx_min=25, fast=12, slow=26, signal=9):
    """MACD yönü, sadece ADX güçlü trend gösterirken."""
    line, sig, hist = ind.macd(data["close"], fast, slow, signal)
    a = ind.adx(data["high"], data["low"], data["close"], adx_period)
    s = np.sign(hist).fillna(0).to_numpy().astype(int).copy()
    s[(a.fillna(0).to_numpy() < adx_min)] = 0
    return pd.Series(s, index=data.index)


# ───────────────── Ortalamaya dönüş (stateful) ─────────────────

def rsi_revert(data, period=14, lo=30, hi=70, exit_mid=50):
    r = ind.rsi(data["close"], period).to_numpy()
    valid = ~np.isnan(r)
    el = valid & (r <= lo)
    es = valid & (r >= hi)
    xl = r >= exit_mid
    xs = r <= exit_mid
    return pd.Series(_hold(el, xl, es, xs), index=data.index)


def stoch_revert(data, k_period=14, d_period=3, lo=20, hi=80):
    k, d = ind.stochastic(data["high"], data["low"], data["close"], k_period, d_period)
    k = k.to_numpy(); valid = ~np.isnan(k)
    el = valid & (k <= lo); es = valid & (k >= hi)
    xl = k >= 50; xs = k <= 50
    return pd.Series(_hold(el, xl, es, xs), index=data.index)


def cci_revert(data, period=20, thr=100):
    c = ind.cci(data["high"], data["low"], data["close"], period).to_numpy()
    valid = ~np.isnan(c)
    el = valid & (c <= -thr); es = valid & (c >= thr)
    xl = c >= 0; xs = c <= 0
    return pd.Series(_hold(el, xl, es, xs), index=data.index)


def bollinger_revert(data, period=20, num_std=2.0):
    lo, mid, up = ind.bollinger(data["close"], period, num_std)
    c = data["close"].to_numpy(); lo=lo.to_numpy(); up=up.to_numpy(); mid=mid.to_numpy()
    valid = ~np.isnan(mid)
    el = valid & (c <= lo); es = valid & (c >= up)
    xl = c >= mid; xs = c <= mid
    return pd.Series(_hold(el, xl, es, xs), index=data.index)


def bollinger_breakout(data, period=20, num_std=2.0):
    lo, mid, up = ind.bollinger(data["close"], period, num_std)
    c = data["close"].to_numpy(); lo=lo.to_numpy(); up=up.to_numpy(); mid=mid.to_numpy()
    valid = ~np.isnan(mid)
    el = valid & (c >= up); es = valid & (c <= lo)
    xl = c <= mid; xs = c >= mid
    return pd.Series(_hold(el, xl, es, xs), index=data.index)


def vwap_revert(data, dist_pct=0.3):
    v = ind.vwap_daily(data["high"], data["low"], data["close"], data["volume"])
    c = data["close"].to_numpy(); vv = v.to_numpy()
    valid = ~np.isnan(vv)
    band = vv * dist_pct / 100
    el = valid & (c <= vv - band); es = valid & (c >= vv + band)
    xl = c >= vv; xs = c <= vv
    return pd.Series(_hold(el, xl, es, xs), index=data.index)


# Strateji kayıt defteri: ad -> (fonksiyon, param-ızgarası)
REGISTRY = {
    "macd_trend": (macd_trend, {"fast": [8, 12], "slow": [21, 26], "signal": [9]}),
    "supertrend": (supertrend_trend, {"period": [10, 14], "mult": [2.0, 3.0]}),
    "vwap_trend": (vwap_trend, {}),
    "adx_macd": (adx_macd_trend, {"adx_min": [20, 25, 30]}),
    "rsi_revert": (rsi_revert, {"period": [7, 14], "lo": [25, 30], "hi": [70, 75]}),
    "stoch_revert": (stoch_revert, {"k_period": [14, 21], "lo": [20], "hi": [80]}),
    "cci_revert": (cci_revert, {"period": [14, 20], "thr": [100, 150]}),
    "boll_revert": (bollinger_revert, {"period": [20, 30], "num_std": [2.0, 2.5]}),
    "boll_breakout": (bollinger_breakout, {"period": [20, 30], "num_std": [2.0, 2.5]}),
    "vwap_revert": (vwap_revert, {"dist_pct": [0.2, 0.4]}),
}
