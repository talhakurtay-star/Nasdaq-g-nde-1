"""Veri sağlık kontrolü — bozuk/şüpheli veriyi backtest'ten ÖNCE yakala.

Yanlış veri = yanlış sonuç. Bu araç her dosyayı şu açılardan denetler:
  - Eksik değer (NaN), tekrarlı zaman damgası
  - OHLC tutarlılığı (high>=low, high>=max(o,c), low<=min(o,c))
  - Bar aralığı tutarlılığı ve boşluklar
  - Aşırı fiyat sıçramaları (olası hatalı bar)
  - Sıfır/negatif fiyat, hafta sonu barları

Kullanım:
    python tools/data_health.py --data data/NAS100_M5.csv
    python tools/data_health.py            # data/ içindeki tüm dosyalar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv


def check(path: Path, jump_pct: float = 5.0) -> bool:
    try:
        d = load_mt5_csv(path)
    except Exception as exc:
        print(f"  ✗ {path.name}: YÜKLENEMEDİ — {exc}")
        return False

    issues = []
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]

    # Eksik değerler
    na = d[["open", "high", "low", "close"]].isna().sum().sum()
    if na:
        issues.append(f"{na} eksik (NaN) OHLC değeri")

    # Tekrarlı zaman damgaları
    dup = d.index.duplicated().sum()
    if dup:
        issues.append(f"{dup} tekrarlı zaman damgası")

    # Sıralı mı
    if not d.index.is_monotonic_increasing:
        issues.append("zaman damgaları sıralı değil")

    # OHLC tutarlılığı
    bad_hl = (h < l).sum()
    bad_h = (h < o.combine(c, max) - 1e-9).sum()
    bad_l = (l > o.combine(c, min) + 1e-9).sum()
    if bad_hl:
        issues.append(f"{bad_hl} barda high < low")
    if bad_h:
        issues.append(f"{bad_h} barda high < max(open,close)")
    if bad_l:
        issues.append(f"{bad_l} barda low > min(open,close)")

    # Sıfır / negatif fiyat
    nonpos = (d[["open", "high", "low", "close"]] <= 0).sum().sum()
    if nonpos:
        issues.append(f"{nonpos} sıfır/negatif fiyat")

    # Aşırı sıçrama (olası hatalı bar)
    ret = c.pct_change().abs() * 100
    jumps = (ret > jump_pct).sum()
    if jumps:
        issues.append(f"{jumps} bar > %{jump_pct} sıçrama (kontrol et)")

    # Bar aralığı / boşluklar
    deltas = pd.Series(d.index).diff().dropna()
    dom = deltas.mode().iloc[0] if len(deltas) else pd.Timedelta(0)
    gaps = (deltas > dom * 3).sum()  # baskın aralığın 3 katından büyük

    # Hafta sonu barları (Cumartesi/Pazar)
    wknd = ((d.index.dayofweek >= 5)).sum()

    status = "✓" if not issues else "⚠️"
    print(f"  {status} {path.name:<16} {len(d):>7} bar | {d.index[0].date()}→{d.index[-1].date()} "
          f"| baskın aralık {dom} | {gaps} büyük boşluk | {wknd} hafta-sonu barı")
    for it in issues:
        print(f"       → {it}")
    return not issues


def main():
    ap = argparse.ArgumentParser(description="Veri sağlık kontrolü")
    ap.add_argument("--data", default=None, help="Tek dosya (yoksa data/*.csv)")
    ap.add_argument("--dir", default="data")
    ap.add_argument("--jump", type=float, default=5.0, help="Sıçrama eşiği %%")
    args = ap.parse_args()

    if args.data:
        files = [Path(args.data)]
    else:
        files = [f for f in sorted(Path(args.dir).glob("*.csv")) if "SAMPLE" not in f.name]
    if not files:
        print("Dosya bulunamadı.")
        return

    print(f"VERİ SAĞLIK KONTROLÜ ({len(files)} dosya):\n")
    ok = 0
    for f in files:
        if check(f, args.jump):
            ok += 1
    print(f"\n  Temiz: {ok}/{len(files)}  "
          f"{'✓ hepsi sağlıklı' if ok == len(files) else '⚠️ bazı dosyalar kontrol gerektiriyor'}")


if __name__ == "__main__":
    main()
