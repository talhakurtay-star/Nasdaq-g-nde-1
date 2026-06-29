"""Detaylı OVERFITTING (aşırı uydurma) testi.

Bir stratejinin parametreleri "geçmişe uydurulmuş" mu, yoksa gerçek bir avantaj
mı taşıyor? Bunu dört bağımsız yöntemle sınar:

  1) IN-SAMPLE / OUT-OF-SAMPLE  : ilk %70'te optimize et, son %30'da (görülmemiş) test et.
  2) WALK-FORWARD               : kayan pencerelerde optimize→test, tekrar tekrar.
  3) SEMBOL-ARASI ROBUSTLUK     : bir sembolde bulunan parametreyi diğerlerinde dene.
  4) PARAMETRE DUYARLILIĞI      : optimum keskin tepe mi (overfit) geniş plato mu (sağlam)?

Kullanım:
    python tools/overfit_test.py
    python tools/overfit_test.py --data data/NAS100_M5.csv --objective total_return
    python tools/overfit_test.py --data data/NAS100_M15.csv --jobs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv
from src.optimizer import OptConfig, evaluate, grid_search

# Stratejiye göre taranan parametre ızgaraları.
GRIDS = {
    "orb": {
        "open_hour": [15, 16, 17],
        "or_minutes": [15, 30, 45],
        "stop_loss_pct": [1.0, 1.5, 2.0],
        "risk_per_trade_pct": [0.30],
        "min_or_range_pct": [0.0],
    },
    "meanrev": {
        "lookback": [15, 20, 30, 50],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.5],
        "stop_loss_pct": [0.6, 1.0, 1.5],
        "take_profit_pct": [1.0, 1.5, 2.5],
        "risk_per_trade_pct": [0.15, 0.35],
    },
}
SWEEP_KEYS = {
    "orb": ["open_hour", "or_minutes", "stop_loss_pct", "risk_per_trade_pct", "min_or_range_pct"],
    "meanrev": ["lookback", "entry_z", "exit_z", "stop_loss_pct", "take_profit_pct", "risk_per_trade_pct"],
}

# Sembol-arası test için aday dosyalar
CROSS_SYMBOLS = ["US500_M5", "US30_M5", "GER40_M5", "NVDA_M5", "XAUUSD_M5", "NAS100_M15"]


def optimize(data, grid, cfg, objective, jobs, strategy):
    """Grid'i tara; en iyi parametreyi (sadece sweep anahtarları) ve tabloyu döndür."""
    df = grid_search(data, grid, cfg, strategy=strategy, objective=objective, n_jobs=jobs)
    if df.empty:
        return None, df
    best = {k: df.iloc[0][k] for k in SWEEP_KEYS[strategy] if k in df.columns}
    for ik in ("lookback", "open_hour", "or_minutes"):
        if ik in best:
            best[ik] = int(best[ik])
    return best, df


def fmt_row(tag, m: dict) -> str:
    return (f"  {tag:<14} getiri {m.get('total_return',0)*100:8.2f}%  "
            f"hedef-oran {m.get('target_hit_rate',0)*100:5.1f}%  "
            f"sharpe {m.get('sharpe',0):6.2f}  "
            f"maxDD {m.get('max_drawdown',0)*100:7.2f}%  "
            f"işlem {int(m.get('num_trades',0)):>5}")


def section(title: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)


# ───────────────────────── 1) IN-SAMPLE / OUT-OF-SAMPLE ─────────────────────────
def test_is_oos(data, grid, cfg, objective, jobs, strategy, split=0.70):
    section("1) IN-SAMPLE / OUT-OF-SAMPLE (eğitim %70 / test %30)")
    cut = int(len(data) * split)
    train, test = data.iloc[:cut], data.iloc[cut:]
    print(f"  Eğitim: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} bar)")
    print(f"  Test  : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} bar)\n")

    best, df = optimize(train, grid, cfg, objective, jobs, strategy)
    if best is None:
        print("  Yeterli işlem üreten kombinasyon yok.")
        return None
    is_m = evaluate(train, strategy, best, cfg)
    oos_m = evaluate(test, strategy, best, cfg)

    print(f"  En iyi parametre (eğitimde bulundu): {best}\n")
    print(fmt_row("EĞİTİM (IS)", is_m))
    print(fmt_row("TEST (OOS)", oos_m))

    is_obj = is_m.get(objective, 0) or 0
    oos_obj = oos_m.get(objective, 0) or 0
    ratio = (oos_obj / is_obj) if is_obj not in (0, None) else float("nan")
    print(f"\n  OOS/IS oranı ({objective}): {ratio:.2f}  "
          f"→ {_verdict_ratio(ratio)}")
    return best


def _verdict_ratio(r):
    if np.isnan(r):
        return "değerlendirilemedi"
    if r >= 0.6:
        return "✓ SAĞLAM (test performansı korunmuş)"
    if r >= 0.2:
        return "⚠ ZAYIF (belirgin düşüş, kısmen overfit)"
    return "✗ OVERFIT (test'te çöküyor)"


# ───────────────────────────── 2) WALK-FORWARD ─────────────────────────────
def test_walk_forward(data, grid, cfg, objective, jobs, strategy, folds=5):
    section(f"2) WALK-FORWARD ({folds} kat: her blokta optimize → sonraki blokta test)")
    seg = np.array_split(np.arange(len(data)), folds)
    rows = []
    for k in range(len(seg) - 1):
        train = data.iloc[seg[k][0]: seg[k][-1] + 1]
        test = data.iloc[seg[k + 1][0]: seg[k + 1][-1] + 1]
        best, df = optimize(train, grid, cfg, objective, jobs, strategy)
        if best is None:
            continue
        is_m = evaluate(train, strategy, best, cfg)
        oos_m = evaluate(test, strategy, best, cfg)
        rows.append({
            "kat": k + 1,
            "test_dönem": f"{test.index[0].date()}→{test.index[-1].date()}",
            "IS_getiri%": round(is_m["total_return"] * 100, 2),
            "OOS_getiri%": round(oos_m["total_return"] * 100, 2),
            "OOS_hedef%": round(oos_m["target_hit_rate"] * 100, 1),
            "OOS_sharpe": round(oos_m["sharpe"], 2),
        })
        pstr = ", ".join(f"{k2}={best[k2]}" for k2 in best)
        print(f"  Kat {k+1}: {test.index[0].date()}→{test.index[-1].date()}  "
              f"IS {is_m['total_return']*100:7.2f}%  →  OOS {oos_m['total_return']*100:7.2f}%  ({pstr})")

    if not rows:
        print("  Yeterli veri yok.")
        return
    wf = pd.DataFrame(rows)
    avg_oos = wf["OOS_getiri%"].mean()
    pos = (wf["OOS_getiri%"] > 0).sum()
    print(f"\n  Ortalama OOS getiri/kat : {avg_oos:.2f}%")
    print(f"  Pozitif OOS kat sayısı  : {pos}/{len(wf)}")
    print(f"  → {_verdict_wf(avg_oos, pos, len(wf))}")


def _verdict_wf(avg, pos, total):
    if avg > 0 and pos >= total * 0.6:
        return "✓ SAĞLAM (katların çoğu görülmemiş veride pozitif)"
    if avg > 0:
        return "⚠ KARARSIZ (ortalama pozitif ama tutarsız)"
    return "✗ OVERFIT (görülmemiş veride ortalama negatif)"


# ───────────────────────── 3) SEMBOL-ARASI ROBUSTLUK ─────────────────────────
def test_cross_symbol(best, cfg, data_dir, primary, strategy):
    section("3) SEMBOL-ARASI ROBUSTLUK (aynı parametre, farklı piyasalar)")
    if best is None:
        print("  Parametre yok (IS/OOS başarısız).")
        return
    print(f"  Test edilen parametre: {best}\n")
    print(fmt_row(f"{primary} (kaynak)", evaluate(_load(Path(data_dir)/f'{primary}.csv'), strategy, best, cfg))
          if (Path(data_dir)/f'{primary}.csv').exists() else "  (kaynak yüklenemedi)")
    pos = 0
    tested = 0
    for sym in CROSS_SYMBOLS:
        p = Path(data_dir) / f"{sym}.csv"
        if not p.exists() or sym == primary:
            continue
        m = evaluate(_load(p), strategy, best, cfg)
        tested += 1
        if m["total_return"] > 0:
            pos += 1
        print(fmt_row(sym, m))
    if tested:
        print(f"\n  Pozitif sembol: {pos}/{tested}")
        print(f"  → {'✓ SAĞLAM (birden çok piyasada çalışıyor)' if pos >= tested*0.5 else '✗ OVERFIT (sadece kaynak sembole özgü)'}")


_cache: dict = {}
def _load(path: Path):
    key = str(path)
    if key not in _cache:
        _cache[key] = load_mt5_csv(path)
    return _cache[key]


# ───────────────────────── 4) PARAMETRE DUYARLILIĞI ─────────────────────────
def test_sensitivity(df, objective):
    section("4) PARAMETRE DUYARLILIĞI (optimum tepe mi, plato mu?)")
    if df is None or df.empty:
        print("  Tablo yok.")
        return
    top = df.head(10)
    print(f"  En iyi 10 kombinasyon ({objective} sıralı):\n")
    cols = ["open_hour", "or_minutes", "lookback", "entry_z", "stop_loss_pct",
            "take_profit_pct", "risk_per_trade_pct", "min_or_range_pct",
            "total_return", "target_hit_rate", "sharpe", "num_trades"]
    show = top[[c for c in cols if c in top.columns]].copy()
    if "total_return" in show:
        show["total_return"] = (show["total_return"] * 100).round(2)
    if "target_hit_rate" in show:
        show["target_hit_rate"] = (show["target_hit_rate"] * 100).round(1)
    if "sharpe" in show:
        show["sharpe"] = show["sharpe"].round(2)
    print(show.to_string(index=False))

    best_obj = df.iloc[0][objective]
    med_top = df.head(10)[objective].median()
    print(f"\n  En iyi {objective}: {best_obj:.4f} | top-10 medyan: {med_top:.4f}")
    if best_obj <= 0:
        verdict = "✗ AVANTAJSIZ (en iyi kombinasyon bile zararda — strateji bu veride çalışmıyor)"
    elif med_top < best_obj * 0.3:
        verdict = "⚠ KESKİN TEPE (tek kombinasyona bağlı, overfit riski yüksek)"
    else:
        verdict = "✓ GENİŞ PLATO (komşu parametreler de iyi → sağlam)"
    print(f"  → {verdict}")


def parse_args():
    p = argparse.ArgumentParser(description="Detaylı overfitting testi")
    p.add_argument("--data", default="data/NAS100_M5.csv", help="Birincil sembol dosyası")
    p.add_argument("--strategy", choices=["orb", "meanrev"], default="orb")
    p.add_argument("--dir", default="data", help="Sembol-arası test klasörü")
    p.add_argument("--objective", default="total_return",
                   help="Optimizasyon amacı (total_return, target_hit_rate, sharpe)")
    p.add_argument("--folds", type=int, default=5, help="Walk-forward kat sayısı")
    p.add_argument("--jobs", type=int, default=4, help="Paralel işlem sayısı")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OptConfig()
    strategy = args.strategy
    grid = GRIDS[strategy]
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    primary_path = Path(args.data)
    primary_name = primary_path.stem
    print(f"OVERFITTING TESTİ — strateji: {strategy} | birincil sembol: {primary_name}")
    print(f"Grid: {n_combos} kombinasyon | amaç: {args.objective} | paralel: {args.jobs}")

    data = _load(primary_path)
    print(f"Veri: {len(data)} bar | {data.index[0].date()} → {data.index[-1].date()}")

    best = test_is_oos(data, grid, cfg, args.objective, args.jobs, strategy)
    test_walk_forward(data, grid, cfg, args.objective, args.jobs, strategy, folds=args.folds)
    test_cross_symbol(best, cfg, args.dir, primary_name, strategy)

    # Duyarlılık için tüm veride bir kez daha optimize edip tabloyu göster
    _, full_df = optimize(data, grid, cfg, args.objective, args.jobs, strategy)
    test_sensitivity(full_df, args.objective)

    section("ÖZET")
    print("  Yukarıdaki 4 testin TAMAMI ✓ ise parametreye güvenilebilir.")
    print("  Herhangi biri ✗ ise: strateji geçmişe uydurulmuş, canlıda çökebilir.")


if __name__ == "__main__":
    main()
