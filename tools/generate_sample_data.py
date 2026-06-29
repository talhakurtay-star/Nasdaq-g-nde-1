"""MT5 'Export Bars' formatında sentetik örnek veri üretir.

Gerçek MT5 verisi indirilene kadar botu test etmek için kullanılır.
Çıktı: data/SAMPLE_NDX_D1.csv  (tab ile ayrılmış, MT5 başlıklı)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(out_path: Path, n: int = 750, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    # Geometrik Brownian hareket + hafif yukarı trend (Nasdaq benzeri)
    dates = pd.bdate_range("2022-01-03", periods=n)
    drift = 0.0004
    vol = 0.012
    rets = rng.normal(drift, vol, n)
    price = 15000 * np.exp(np.cumsum(rets))

    opens = price * (1 + rng.normal(0, 0.001, n))
    closes = price
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.003, n)))
    tickvol = rng.integers(50_000, 200_000, n)

    df = pd.DataFrame({
        "<DATE>": dates.strftime("%Y.%m.%d"),
        "<TIME>": "00:00:00",
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
    print(f"Örnek veri yazıldı: {out_path} ({n} bar)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    generate(root / "data" / "SAMPLE_NDX_D1.csv")
