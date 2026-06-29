"""MT5 'Export Bars' formatında sentetik örnek veri üretir.

Gerçek MT5 verisi indirilene kadar botu test etmek için kullanılır.
Gün-içi (intraday) seans saatlerini taklit eder: 09:30–16:00 (6.5 saat).

Kullanım:
    python tools/generate_sample_data.py            # M5, ~60 gün
    python tools/generate_sample_data.py --tf M1 --days 30
    python tools/generate_sample_data.py --tf M15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Timeframe -> dakika
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}


def _session_index(days: int, minutes: int) -> pd.DatetimeIndex:
    """İş günleri için 09:30–16:00 arası bar zaman damgaları üret."""
    bars_per_day = int(6.5 * 60 / minutes)
    business_days = pd.bdate_range("2024-01-02", periods=days)
    stamps = []
    for day in business_days:
        start = day + pd.Timedelta(hours=9, minutes=30)
        for b in range(bars_per_day):
            stamps.append(start + pd.Timedelta(minutes=b * minutes))
    return pd.DatetimeIndex(stamps)


def generate(out_path: Path, timeframe: str = "M5", days: int = 60, seed: int = 42) -> None:
    minutes = TF_MINUTES[timeframe]
    idx = _session_index(days, minutes)
    n = len(idx)

    rng = np.random.default_rng(seed)
    # Bar başına volatiliteyi günlük ~%1'e ölçekle
    bar_vol = 0.01 / np.sqrt(6.5 * 60 / minutes)
    drift = 0.00002
    rets = rng.normal(drift, bar_vol, n)
    price = 16000 * np.exp(np.cumsum(rets))

    opens = price * (1 + rng.normal(0, bar_vol * 0.2, n))
    closes = price
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, bar_vol * 0.5, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, bar_vol * 0.5, n)))
    tickvol = rng.integers(500, 5000, n)

    df = pd.DataFrame({
        "<DATE>": idx.strftime("%Y.%m.%d"),
        "<TIME>": idx.strftime("%H:%M:%S"),
        "<OPEN>": np.round(opens, 2),
        "<HIGH>": np.round(highs, 2),
        "<LOW>": np.round(lows, 2),
        "<CLOSE>": np.round(closes, 2),
        "<TICKVOL>": tickvol,
        "<VOL>": 0,
        "<SPREAD>": rng.integers(1, 5, n),
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Örnek veri yazıldı: {out_path}  ({timeframe}, {days} gün, {n} bar)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="M5", choices=list(TF_MINUTES), help="Zaman dilimi")
    p.add_argument("--days", type=int, default=60, help="İş günü sayısı")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = root / "data" / f"SAMPLE_NDX_{args.tf}.csv"
    generate(out, timeframe=args.tf, days=args.days)
