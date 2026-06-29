"""Temel doğrulama testleri (pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtester import Backtester, BacktestConfig
from src.data_loader import load_mt5_csv
from src.strategy import MACrossStrategy, rsi, sma


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
