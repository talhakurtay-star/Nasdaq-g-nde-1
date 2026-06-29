"""Çoklu-enstrüman portföy backtesti.

Aynı stratejiyi (varsayılan ORB) birkaç enstrümanda çalıştırır ve eşit-ağırlık
bir portföy equity'si oluşturur. Çeşitlendirme, korelasyonsuz piyasalarda riski
(DD) düşürür — tek tek enstrümandan daha düzgün equity.

Kullanım:
    python tools/portfolio.py
    python tools/portfolio.py --symbols NAS100_M5 NVDA_M5 --monthly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv
from src.metrics import compute_metrics, format_monthly
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig
from src.strategy import OpeningRangeBreakout


def main():
    ap = argparse.ArgumentParser(description="Çoklu-enstrüman portföy backtesti")
    ap.add_argument("--symbols", nargs="+",
                    default=["NAS100_M5", "NVDA_M5", "US500_M5", "GER40_M5"])
    ap.add_argument("--dir", default="data")
    ap.add_argument("--open-hour", type=int, default=16)
    ap.add_argument("--or-minutes", type=int, default=30)
    ap.add_argument("--lock", default="breakeven")
    ap.add_argument("--lev", type=float, default=2.0)
    ap.add_argument("--monthly", action="store_true")
    args = ap.parse_args()

    cfg = SessionConfig(10000, 0.0002, 0.0001)
    risk = RiskParams(daily_target_pct=0.44, daily_stop_pct=0.44, profit_lock_mode=args.lock,
                      stop_loss_pct=1.5, risk_per_trade_pct=0.30, leverage=args.lev,
                      session_start_hour=args.open_hour, session_end_hour=22, max_trades_per_day=1)

    curves = {}
    print(f"{'sembol':<14}{'getiri%':>9}{'yıllık%':>9}{'maxDD%':>8}{'sharpe':>8}")
    print("─" * 48)
    for sym in args.symbols:
        p = Path(args.dir) / f"{sym}.csv"
        if not p.exists():
            print(f"  {sym}: dosya yok, atlandı")
            continue
        d = load_mt5_csv(p)
        s = OpeningRangeBreakout(open_hour=args.open_hour, or_minutes=args.or_minutes).generate_signals(d)
        r = SessionBacktester(cfg, risk).run(d, s)
        m = r.metrics
        yrs = (d.index[-1] - d.index[0]).days / 365.25
        ann = ((1 + m["total_return"]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
        print(f"{sym:<14}{m['total_return']*100:>9.1f}{ann:>9.2f}{m['max_drawdown']*100:>8.1f}{m['sharpe']:>8.2f}")
        curves[sym] = r.equity / r.equity.iloc[0]   # normalize

    if len(curves) < 2:
        print("\nPortföy için en az 2 enstrüman gerekli.")
        return

    # Eşit-ağırlık portföy (günlük hizalanmış)
    allidx = sorted(set().union(*[c.index for c in curves.values()]))
    aligned = pd.concat([c.reindex(allidx).ffill() for c in curves.values()], axis=1).dropna()
    port = aligned.mean(axis=1)
    pm = compute_metrics(port * 10000, pd.DataFrame())
    yrs = (port.index[-1] - port.index[0]).days / 365.25
    ann = ((port.iloc[-1] / port.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0

    print("\n" + "═" * 48)
    print(f"  PORTFÖY (eşit ağırlık, {len(curves)} enstrüman)")
    print("═" * 48)
    print(f"  Getiri    : {(port.iloc[-1]/port.iloc[0]-1)*100:+.1f}%  (yıllık ~{ann:+.2f}%)")
    print(f"  Max düşüş : {pm['max_drawdown']*100:.2f}%   (çeşitlendirme etkisi)")
    print(f"  Sharpe    : {pm['sharpe']:.2f}")

    # Korelasyon (günlük getiriler)
    daily = aligned.resample("1D").last().pct_change().dropna()
    if len(daily) > 5 and daily.shape[1] >= 2:
        corr = daily.corr()
        print(f"\n  Ortalama ikili korelasyon: {corr.values[np.triu_indices_from(corr.values,1)].mean():.2f} "
              f"(düşük = iyi çeşitlendirme)")

    if args.monthly:
        print("\n" + format_monthly(port * 10000))


if __name__ == "__main__":
    main()
