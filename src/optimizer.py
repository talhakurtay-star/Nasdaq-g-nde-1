"""Parametre tarama (grid search) — paralel.

Mean-reversion stratejisi + risk parametrelerini tarar, her kombinasyonu seans
motorundan geçirir ve seçilen amaç fonksiyonuna göre sıralar.

Hız için Linux fork-paylaşımlı global veri kullanılır (her görevde DataFrame'i
yeniden picklelemekten kaçınmak için).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from .risk import RiskParams
from .session_backtester import SessionBacktester, SessionConfig
from .strategy import MeanReversionStrategy

# Worker süreçlerinde fork ile paylaşılan veri (her görevde repickle yok)
_DATA: pd.DataFrame | None = None


@dataclass
class OptConfig:
    initial_cash: float = 10_000.0
    commission: float = 0.0005
    slippage: float = 0.0002
    daily_target_pct: float = 0.44
    daily_stop_pct: float = 0.44


# Taranabilir parametreler ve hangi nesneye ait oldukları
_STRAT_KEYS = {"lookback", "entry_z", "exit_z", "use_rsi", "allow_long", "allow_short"}
_RISK_KEYS = {"stop_loss_pct", "take_profit_pct", "risk_per_trade_pct", "leverage"}


def build_strategy(params: dict) -> MeanReversionStrategy:
    kw = {k: params[k] for k in _STRAT_KEYS if k in params}
    return MeanReversionStrategy(**kw)


def build_risk(params: dict, cfg: OptConfig) -> RiskParams:
    kw = {k: params[k] for k in _RISK_KEYS if k in params}
    return RiskParams(
        daily_target_pct=cfg.daily_target_pct,
        daily_stop_pct=cfg.daily_stop_pct,
        **kw,
    )


# İlgilendiğimiz metrikler (sonuç tablosuna girer)
_KEEP_METRICS = [
    "total_return", "cagr", "sharpe", "max_drawdown",
    "num_trades", "target_days", "stop_days", "target_hit_rate",
    "avg_daily_return_pct", "worst_day_pct", "firm_daily_breach",
]


def evaluate(data: pd.DataFrame, params: dict, cfg: OptConfig) -> dict:
    """Tek bir parametre kombinasyonunu çalıştır, metrikleri döndür."""
    strat = build_strategy(params)
    risk = build_risk(params, cfg)
    sig = strat.generate_signals(data)
    bt = SessionBacktester(
        SessionConfig(cfg.initial_cash, cfg.commission, cfg.slippage), risk
    )
    res = bt.run(data, sig)
    out = dict(params)
    for k in _KEEP_METRICS:
        out[k] = res.metrics.get(k)
    return out


def _worker(args):
    params, cfg = args
    return evaluate(_DATA, params, cfg)


def expand_grid(grid: dict) -> list[dict]:
    """{'a':[1,2],'b':[3]} -> [{'a':1,'b':3},{'a':2,'b':3}]"""
    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def grid_search(
    data: pd.DataFrame,
    grid: dict,
    cfg: OptConfig | None = None,
    objective: str = "total_return",
    n_jobs: int = 4,
    min_trades: int = 30,
) -> pd.DataFrame:
    """Grid'i tara, amaç fonksiyonuna göre sıralı DataFrame döndür.

    min_trades altındaki (istatistiksel anlamsız) kombinasyonlar elenir.
    """
    global _DATA
    cfg = cfg or OptConfig()
    combos = expand_grid(grid)

    if n_jobs and n_jobs > 1:
        import multiprocessing as mp
        _DATA = data  # fork ile worker'lara geçer
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            rows = pool.map(_worker, [(c, cfg) for c in combos])
        _DATA = None
    else:
        rows = [evaluate(data, c, cfg) for c in combos]

    df = pd.DataFrame(rows)
    if "num_trades" in df:
        df = df[df["num_trades"] >= min_trades]
    if objective in df.columns and not df.empty:
        df = df.sort_values(objective, ascending=False).reset_index(drop=True)
    return df
