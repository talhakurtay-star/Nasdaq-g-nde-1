"""Walk-forward ADAPTİF backtest — gerçek deploy simülasyonu.

Tek bir sabit parametre yerine, gerçekte nasıl çalıştırırsan onu taklit eder:
  - Bir eğitim penceresinde en iyi parametreyi bul,
  - Onu SONRAKİ (görülmemiş) test penceresinde uygula,
  - Pencereyi kaydır, parametreyi yeniden optimize et, tekrarla.
Sonuç: sürekli, dürüst, out-of-sample equity eğrisi.

Bu, "geçmişe en iyi tek parametre" yanılgısını ortadan kaldırır.

Kullanım:
    python tools/walkforward.py --data data/NAS100_M5.csv --strategy orb
    python tools/walkforward.py --data data/NAS100_M5.csv --train-days 250 --test-days 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv
from src.metrics import compute_metrics
from src.optimizer import OptConfig, build_risk, build_strategy, grid_search
from src.session_backtester import SessionBacktester, SessionConfig

GRIDS = {
    "orb": {"open_hour": [15, 16, 17], "or_minutes": [15, 30, 45], "stop_loss_pct": [1.0, 1.5, 2.0]},
    "meanrev": {"lookback": [15, 20, 30], "entry_z": [1.5, 2.0, 2.5], "stop_loss_pct": [0.8, 1.2]},
}


def main():
    ap = argparse.ArgumentParser(description="Walk-forward adaptif backtest")
    ap.add_argument("--data", default="data/NAS100_M5.csv")
    ap.add_argument("--strategy", choices=["orb", "meanrev"], default="orb")
    ap.add_argument("--train-days", type=int, default=250, help="Eğitim penceresi (işlem günü)")
    ap.add_argument("--test-days", type=int, default=60, help="Test penceresi (işlem günü)")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    cfg = OptConfig()
    scfg = SessionConfig(cfg.initial_cash, cfg.commission, cfg.slippage)
    d = load_mt5_csv(args.data)
    days = np.array(sorted(set(d.index.normalize())))
    print(f"Veri: {Path(args.data).stem} | {len(days)} işlem günü | "
          f"eğitim {args.train_days}g → test {args.test_days}g, kayan")

    grid = GRIDS[args.strategy]
    equity_parts = []
    cash = cfg.initial_cash
    rows = []
    start = 0
    while start + args.train_days + args.test_days <= len(days):
        tr_days = days[start: start + args.train_days]
        te_days = days[start + args.train_days: start + args.train_days + args.test_days]
        train = d[(d.index.normalize() >= tr_days[0]) & (d.index.normalize() <= tr_days[-1])]
        test = d[(d.index.normalize() >= te_days[0]) & (d.index.normalize() <= te_days[-1])]

        # Eğitimde optimize
        res = grid_search(train, grid, cfg, strategy=args.strategy, objective="total_return",
                          n_jobs=args.jobs, min_trades=10)
        if res.empty:
            start += args.test_days
            continue
        best = {k: res.iloc[0][k] for k in grid}
        for ik in ("open_hour", "or_minutes", "lookback"):
            if ik in best:
                best[ik] = int(best[ik])

        # Test penceresinde uygula (sermaye taşınır → bileşik)
        scfg2 = SessionConfig(cash, cfg.commission, cfg.slippage)
        strat = build_strategy(args.strategy, best)
        risk = build_risk(best, cfg)
        r = SessionBacktester(scfg2, risk).run(test, strat.generate_signals(test))
        equity_parts.append(r.equity)
        ret = r.equity.iloc[-1] / cash - 1
        cash = r.equity.iloc[-1]
        rows.append({"test": f"{te_days[0].date()}→{te_days[-1].date()}",
                     "getiri%": round(ret * 100, 2), "params": str(best)})
        print(f"  {te_days[0].date()}→{te_days[-1].date()}: {ret*100:+6.2f}%  | {best}")
        start += args.test_days

    if not equity_parts:
        print("Yeterli veri yok.")
        return
    eq = pd.concat(equity_parts)
    eq = eq[~eq.index.duplicated(keep="last")]
    m = compute_metrics(eq, pd.DataFrame())
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    pos = sum(1 for r in rows if r["getiri%"] > 0)
    print("\n" + "═" * 60)
    print(f"  WALK-FORWARD ADAPTİF SONUÇ ({len(rows)} test penceresi)")
    print("═" * 60)
    print(f"  Toplam OOS getiri : {(eq.iloc[-1]/eq.iloc[0]-1)*100:+.2f}%  (yıllık ~{ann:+.2f}%)")
    print(f"  Pozitif pencere   : {pos}/{len(rows)}")
    print(f"  Max düşüş         : {m['max_drawdown']*100:.2f}%")
    print(f"  → {'✓ Adaptif sistem OOS pozitif' if eq.iloc[-1] > eq.iloc[0] else '✗ Adaptif sistem bile OOS negatif (edge yok)'}")


if __name__ == "__main__":
    main()
