"""data/ klasöründeki tüm MT5 dosyalarını tarar ve toplu backtest yapar.

Her dosyayı seans motorundan geçirir, sonuçları tek karşılaştırma tablosunda
sıralar. Hangi sembol/timeframe'de strateji en iyi sonucu veriyor hızlıca görülür.

Kullanım:
    python tools/batch_backtest.py
    python tools/batch_backtest.py --fast 10 --slow 30 --target 0.44 --stop 0.44
    python tools/batch_backtest.py --sort target_hit_rate --csv sonuclar.csv
    python tools/batch_backtest.py --glob "US100_*.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig
from src.strategy import MACrossStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="data/ klasörü için toplu backtest")
    p.add_argument("--dir", default="data", help="Veri klasörü")
    p.add_argument("--glob", default="*.csv", help="Dosya deseni (örn. 'US100_*.csv')")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--ma", choices=["sma", "ema"], default="ema")
    p.add_argument("--no-rsi", action="store_true")
    p.add_argument("--target", type=float, default=0.44)
    p.add_argument("--stop", type=float, default=0.44)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--sort", default="total_return",
                   help="Sıralama kolonu (total_return, target_hit_rate, sharpe, ...)")
    p.add_argument("--csv", default=None, help="Özet tabloyu bu CSV'ye yaz")
    return p.parse_args()


def run_one(path: Path, args) -> dict | None:
    try:
        data = load_mt5_csv(path)
    except Exception as exc:  # bozuk/uyumsuz dosyayı atla
        print(f"  ⚠️  {path.name} atlandı: {exc}")
        return None

    if len(data) < 100:
        print(f"  ⚠️  {path.name} atlandı: çok az bar ({len(data)})")
        return None

    strat = MACrossStrategy(fast=args.fast, slow=args.slow, ma_type=args.ma,
                            use_rsi=not args.no_rsi)
    signals = strat.generate_signals(data)
    risk = RiskParams(daily_target_pct=args.target, daily_stop_pct=args.stop,
                      leverage=args.leverage)
    result = SessionBacktester(SessionConfig(initial_cash=args.cash), risk).run(data, signals)
    m = result.metrics

    return {
        "dosya": path.name,
        "bar": len(data),
        "gün": m.get("total_days", 0),
        "getiri%": round(m.get("total_return", 0) * 100, 2),
        "hedef_gün": m.get("target_days", 0),
        "stop_gün": m.get("stop_days", 0),
        "hedef_oran%": round(m.get("target_hit_rate", 0) * 100, 1),
        "ort_gün%": round(m.get("avg_daily_return_pct", 0), 3),
        "sharpe": round(m.get("sharpe", 0), 2),
        "maxDD%": round(m.get("max_drawdown", 0) * 100, 2),
        "en_kötü_gün%": round(m.get("worst_day_pct", 0), 2),
        "firma_ihlal": "EVET" if m.get("firm_daily_breach") else "-",
    }


# Tabloda gösterim ile sıralama anahtarı eşlemesi
_SORT_ALIAS = {
    "total_return": "getiri%",
    "target_hit_rate": "hedef_oran%",
    "sharpe": "sharpe",
    "avg_daily_return_pct": "ort_gün%",
}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.dir)
    files = sorted(data_dir.glob(args.glob))
    if not files:
        print(f"'{data_dir}/{args.glob}' ile eşleşen dosya yok.")
        print("Önce MT5 verisi koyun ya da: python tools/generate_sample_data.py --tf M5")
        return

    print(f"{len(files)} dosya bulundu. Backtest çalışıyor "
          f"(fast={args.fast}, slow={args.slow}, target={args.target}, stop={args.stop})...\n")

    rows = []
    for f in files:
        row = run_one(f, args)
        if row:
            rows.append(row)
            print(f"  ✓ {row['dosya']:<22} getiri {row['getiri%']:>7.2f}%  "
                  f"hedef-oran {row['hedef_oran%']:>5.1f}%")

    if not rows:
        print("\nGeçerli sonuç üretilemedi.")
        return

    df = pd.DataFrame(rows)
    sort_col = _SORT_ALIAS.get(args.sort, args.sort)
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    print("\n" + "═" * 110)
    print("  TOPLU BACKTEST KARŞILAŞTIRMASI  (sıralama: " + sort_col + ")")
    print("═" * 110)
    print(df.to_string(index=False))
    print("═" * 110)

    # Özet yorum
    best = df.iloc[0]
    breaches = df[df["firma_ihlal"] == "EVET"]
    print(f"\n  En iyi: {best['dosya']}  →  getiri %{best['getiri%']}, "
          f"hedef-oran %{best['hedef_oran%']}")
    if not breaches.empty:
        print(f"  ⚠️  Firma günlük DD ihlali olan dosya(lar): {', '.join(breaches['dosya'])}")
    else:
        print("  ✓ Hiçbir dosyada firma günlük DD ihlali yok.")
    winners = df[df["hedef_oran%"] > 50]
    print(f"  Hedef-oran %50 üstü (avantajlı) sembol: {len(winners)}/{len(df)}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n  Özet kaydedildi: {args.csv}")


if __name__ == "__main__":
    main()
