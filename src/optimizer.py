"""Parametre tarama (grid search) — paralel, çok-stratejili.

Strateji + risk parametrelerini tarar, her kombinasyonu seans motorundan geçirir
ve seçilen amaç fonksiyonuna göre sıralar.

Desteklenen stratejiler: "orb", "meanrev" (REGISTRY).
Hız için Linux fork-paylaşımlı global veri kullanılır.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from .risk import RiskParams
from .session_backtester import SessionBacktester, SessionConfig
from .strategy import MeanReversionStrategy, OpeningRangeBreakout

_DATA: pd.DataFrame | None = None


@dataclass
class OptConfig:
    initial_cash: float = 10_000.0
    commission: float = 0.0002
    slippage: float = 0.0001
    # Risk şablonu (taranmayan risk parametreleri buradan gelir)
    daily_target_pct: float = 0.44
    daily_stop_pct: float = 0.44
    profit_lock_mode: str = "breakeven"
    leverage: float = 2.0
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 0.0
    risk_per_trade_pct: float = 0.30
    session_start_hour: int = 16
    session_end_hour: int = 22
    max_trades_per_day: int = 1


# Strateji adı -> (sınıf, taranabilir parametre anahtarları)
_STRATEGIES = {
    "orb": (OpeningRangeBreakout, {
        "open_hour", "or_minutes", "buffer_pct", "min_or_range_pct",
        "max_or_range_pct", "require_close_break", "trend_ema_period",
        "allow_long", "allow_short",
    }),
    "meanrev": (MeanReversionStrategy, {
        "lookback", "entry_z", "exit_z", "use_rsi", "allow_long", "allow_short",
    }),
}
_RISK_KEYS = {"stop_loss_pct", "take_profit_pct", "risk_per_trade_pct", "leverage",
              "profit_lock_mode", "session_start_hour", "session_end_hour",
              "max_trades_per_day", "daily_target_pct", "daily_stop_pct"}


def build_strategy(strategy: str, params: dict):
    cls, keys = _STRATEGIES[strategy]
    kw = {k: params[k] for k in keys if k in params}
    if "lookback" in kw:
        kw["lookback"] = int(kw["lookback"])
    if "open_hour" in kw:
        kw["open_hour"] = int(kw["open_hour"])
    if "or_minutes" in kw:
        kw["or_minutes"] = int(kw["or_minutes"])
    return cls(**kw)


def build_risk(params: dict, cfg: OptConfig) -> RiskParams:
    base = dict(
        daily_target_pct=cfg.daily_target_pct, daily_stop_pct=cfg.daily_stop_pct,
        profit_lock_mode=cfg.profit_lock_mode, leverage=cfg.leverage,
        stop_loss_pct=cfg.stop_loss_pct, take_profit_pct=cfg.take_profit_pct,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        session_start_hour=cfg.session_start_hour, session_end_hour=cfg.session_end_hour,
        max_trades_per_day=cfg.max_trades_per_day,
        firm_daily_dd_pct=5.0, monthly_dd_pct=10.0,
    )
    for k in _RISK_KEYS:
        if k in params:
            base[k] = params[k]
    return RiskParams(**base)


_KEEP_METRICS = [
    "total_return", "cagr", "sharpe", "max_drawdown", "num_trades",
    "target_days", "stop_days", "target_hit_rate", "avg_daily_return_pct",
    "worst_day_pct", "firm_daily_breach",
]


def evaluate(data: pd.DataFrame, strategy: str, params: dict, cfg: OptConfig) -> dict:
    strat = build_strategy(strategy, params)
    risk = build_risk(params, cfg)
    sig = strat.generate_signals(data)
    res = SessionBacktester(SessionConfig(cfg.initial_cash, cfg.commission, cfg.slippage), risk).run(data, sig)
    out = dict(params)
    for k in _KEEP_METRICS:
        out[k] = res.metrics.get(k)
    return out


def _worker(args):
    strategy, params, cfg = args
    return evaluate(_DATA, strategy, params, cfg)


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def grid_search(data, grid, cfg=None, strategy="orb", objective="total_return",
                n_jobs=4, min_trades=30) -> pd.DataFrame:
    global _DATA
    cfg = cfg or OptConfig()
    combos = expand_grid(grid)
    if n_jobs and n_jobs > 1:
        import multiprocessing as mp
        _DATA = data
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            rows = pool.map(_worker, [(strategy, c, cfg) for c in combos])
        _DATA = None
    else:
        rows = [evaluate(data, strategy, c, cfg) for c in combos]
    df = pd.DataFrame(rows)
    if "num_trades" in df:
        df = df[df["num_trades"] >= min_trades]
    if objective in df.columns and not df.empty:
        df = df.sort_values(objective, ascending=False).reset_index(drop=True)
    return df
