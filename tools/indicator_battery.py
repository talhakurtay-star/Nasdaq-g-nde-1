"""Gösterge bataryası — her tekniği OOS kapısından geçirir.

Disiplin (data-dredging'i önlemek için):
  1) Her strateji için parametreyi EĞİTİMDE (ilk %70) seç.
  2) OOS'ta (son %30) BİR KEZ doğrula.
  3) OOS-pozitif olanları SEMBOL-ARASI test et.
Sadece üç kapıdan da geçen "gerçek aday" sayılır.

Edge'i KEŞFETMEK için "kazananı koştur" yapısı kullanılır (günlük hedef kapağı yok,
günlük stop risk için açık). Aday bulunursa ayrıca 0.44 yapısında denenir.

Kullanım:
    python tools/indicator_battery.py
    python tools/indicator_battery.py --data data/NAS100_M5.csv --session 14 22
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv
from src.indicator_strategies import REGISTRY
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig

CROSS = ["NVDA_M5", "US500_M5", "US30_M5", "GER40_M5", "XAUUSD_M5"]


def make_risk(sess):
    # "Kazananı koştur": günlük hedef kapağı yok (4.9), günlük stop risk için 0.44,
    # per-trade SL 1.5, risk 0.30. Edge'i ölçer.
    return RiskParams(daily_target_pct=4.9, daily_stop_pct=0.44, firm_daily_dd_pct=99,
                      monthly_dd_pct=99, profit_lock_mode="hard",
                      stop_loss_pct=1.5, take_profit_pct=0.0, risk_per_trade_pct=0.30,
                      session_start_hour=sess[0], session_end_hour=sess[1],
                      max_trades_per_day=0)


def evaluate(data, fn, params, cfg, risk):
    sig = fn(data, **params)
    m = SessionBacktester(cfg, risk).run(data, sig).metrics
    return m


def expand(grid):
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/NAS100_M5.csv")
    ap.add_argument("--session", nargs=2, type=int, default=[14, 22],
                    help="İşlem saati penceresi (broker saati)")
    ap.add_argument("--split", type=float, default=0.70)
    args = ap.parse_args()

    cfg = SessionConfig(initial_cash=10000, commission=0.0002, slippage=0.0001)
    risk = make_risk(args.session)
    d = load_mt5_csv(args.data)
    cut = int(len(d) * args.split)
    train, test = d.iloc[:cut], d.iloc[cut:]
    print(f"Veri: {Path(args.data).stem} | {len(d)} bar | "
          f"eğitim {train.index[0].date()}→{train.index[-1].date()} | "
          f"test {test.index[0].date()}→{test.index[-1].date()}")
    print(f"Saat penceresi: {args.session[0]}-{args.session[1]} | "
          f"yapı: kazananı koştur (günlük stop 0.44, hedef kapağı yok)\n")

    print(f"{'strateji':>14} {'en iyi param':>34} {'EĞİTİM%':>9} {'OOS%':>8} {'OOSwin%':>8} {'işlem':>6}")
    print("─" * 86)
    survivors = []
    for name, (fn, grid) in REGISTRY.items():
        best = None
        for params in expand(grid):
            m = evaluate(train, fn, params, cfg, risk)
            if m.get("num_trades", 0) < 30:
                continue
            if best is None or m["total_return"] > best[0]:
                best = (m["total_return"], params, m)
        if best is None:
            print(f"{name:>14}  (yeterli işlem yok)")
            continue
        _, params, mtr = best
        mte = evaluate(test, fn, params, cfg, risk)
        flag = " ✓OOS+" if mte["total_return"] > 0 else ""
        print(f"{name:>14} {str(params):>34} {mtr['total_return']*100:>9.1f} "
              f"{mte['total_return']*100:>8.1f} {mte['win_rate']*100:>8.1f} {mte['num_trades']:>6}{flag}")
        if mte["total_return"] > 0:
            survivors.append((name, fn, params))

    # Sembol-arası kapı (sadece OOS-pozitif olanlar)
    print("\n" + "═" * 86)
    if not survivors:
        print("  OOS'tan geçen strateji YOK. (Hepsi geçmişe uydurulmuş / edge yok.)")
        return
    print(f"  OOS'tan geçen {len(survivors)} aday → SEMBOL-ARASI test:")
    for name, fn, params in survivors:
        print(f"\n  {name} {params}:")
        pos = 0
        for sym in CROSS:
            p = Path("data") / f"{sym}.csv"
            if not p.exists():
                continue
            m = evaluate(load_mt5_csv(p), fn, params, cfg, risk)
            ok = m["total_return"] > 0
            pos += ok
            print(f"    {sym:<12} {m['total_return']*100:+8.1f}%  win {m['win_rate']*100:.1f}%  {'✓' if ok else '✗'}")
        print(f"    → {pos}/{len(CROSS)} sembolde pozitif "
              f"{'✓ GERÇEK ADAY' if pos >= len(CROSS)*0.5 else '⚠ NAS100-özgü ya da zayıf'}")


if __name__ == "__main__":
    main()
