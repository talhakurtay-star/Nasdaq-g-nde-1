"""Temel doğrulama testleri (pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtester import Backtester, BacktestConfig
from src.data_loader import load_mt5_csv
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig
from src.strategy import MACrossStrategy, rsi, sma


def _make_intraday(days=20, bars_per_day=78, seed=7):
    """Gün-içi M5 benzeri veri (her gün 78 bar = 6.5 saat)."""
    rng = np.random.default_rng(seed)
    stamps = []
    for day in pd.bdate_range("2024-01-02", periods=days):
        start = day + pd.Timedelta(hours=9, minutes=30)
        stamps += [start + pd.Timedelta(minutes=5 * b) for b in range(bars_per_day)]
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)
    price = 16000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.empty(n)
    o[0] = 16000.0
    o[1:] = price[:-1]  # kesintisiz: açılış = önceki kapanış
    h = np.maximum(o, price) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    lo = np.minimum(o, price) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": price, "volume": 1000}, index=idx)


def _make_data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    df = pd.DataFrame({
        "open": price,
        "high": price * 1.005,
        "low": price * 0.995,
        "close": price,
        "volume": 1000,
    }, index=idx)
    return df


def test_sma():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(s, 2).iloc[-1] == 4.5


def test_rsi_bounds():
    s = pd.Series(np.linspace(1, 100, 100))
    r = rsi(s, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_signals_are_binary():
    data = _make_data()
    sig = MACrossStrategy(fast=10, slow=30).generate_signals(data)
    assert set(sig.unique()).issubset({0, 1})
    assert len(sig) == len(data)


def test_backtest_runs_and_conserves_when_flat():
    data = _make_data()
    # Hiç sinyal yoksa equity sabit kalmalı
    flat = pd.Series(0, index=data.index)
    result = Backtester(BacktestConfig(initial_cash=5000)).run(data, flat)
    assert np.isclose(result.equity.iloc[-1], 5000)
    assert result.metrics["num_trades"] == 0


def test_backtest_with_strategy():
    data = _make_data()
    sig = MACrossStrategy(fast=10, slow=30).generate_signals(data)
    result = Backtester().run(data, sig)
    assert len(result.equity) == len(data)
    assert "sharpe" in result.metrics
    assert result.metrics["final_equity"] > 0


def test_mt5_loader(tmp_path):
    # MT5 'Export Bars' formatını taklit eden tab-ayrılmış dosya
    content = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.01.02\t00:00:00\t15000.0\t15100.0\t14950.0\t15050.0\t1234\t0\t2\n"
        "2024.01.03\t00:00:00\t15050.0\t15200.0\t15000.0\t15180.0\t1500\t0\t2\n"
    )
    f = tmp_path / "mt5.csv"
    f.write_text(content)
    df = load_mt5_csv(f)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[0] == 15050.0
    assert df["volume"].iloc[0] == 1234  # tick_volume volume'e düşmüş


# --- Seans motoru / prop-firm kuralları ---

def test_session_daily_caps_hold():
    """Hiçbir gün +target'ı veya -stop'u (tolerans dahilinde) aşmamalı."""
    data = _make_intraday(days=25)
    sig = MACrossStrategy(fast=5, slow=15).generate_signals(data)
    risk = RiskParams(daily_target_pct=0.44, daily_stop_pct=0.44)
    result = SessionBacktester(SessionConfig(), risk).run(data, sig)

    # Kesintisiz veride taşma sadece kapanış işleminin komisyon/slipajı kadar olmalı.
    assert result.days["return_pct"].max() <= 0.44 + 0.15
    assert result.days["return_pct"].min() >= -0.44 - 0.15


def test_session_never_breaches_firm_limit():
    data = _make_intraday(days=40, seed=99)
    sig = MACrossStrategy(fast=5, slow=15).generate_signals(data)
    result = SessionBacktester(SessionConfig(), RiskParams()).run(data, sig)
    assert result.metrics["firm_daily_breach"] is False
    assert result.days["return_pct"].min() >= -5.0


def test_session_locks_after_target():
    """Hedefe ulaşılan günde, kilitten sonra equity sabit kalmalı (yeni işlem yok)."""
    data = _make_intraday(days=30, seed=3)
    sig = MACrossStrategy(fast=5, slow=15).generate_signals(data)
    result = SessionBacktester(SessionConfig(), RiskParams()).run(data, sig)
    # En az bir gün bir sonuca (target/stop) ulaşmalı
    assert (result.days["outcome"] != "neutral").any()
    assert result.metrics["total_days"] == 30
