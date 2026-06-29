"""Veri keşfi — edge'i TAHMİN etmeden VERİDE aramak için.

Strateji kodlamadan önce piyasanın gün-içi davranışını çıkarır:
  1) Saatlik hareketlilik ve yön eğilimi
  2) Gün-içi otokorelasyon (momentum mu mean-reversion mu?)
  3) Açılış-aralığı (opening range) davranışı: kırılım mı, geri dönüş mü?
  4) Haftanın günü etkisi
  5) İlk saatteki hareket vs günün geri kalanı

Kullanım:
    python tools/explore.py --data data/NAS100_M5.csv
    python tools/explore.py --data data/NAS100_M5.csv --open-hour 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv


def section(t):
    print("\n" + "═" * 70)
    print(f"  {t}")
    print("═" * 70)


def hourly_profile(d):
    section("1) SAATLİK PROFİL (hareketlilik ve yön)")
    ret = d["close"].pct_change() * 100
    g = pd.DataFrame({"abs": ret.abs(), "ret": ret, "hour": d.index.hour})
    prof = g.groupby("hour").agg(hareket=("abs", "mean"), yon=("ret", "mean"),
                                 bar=("ret", "size"))
    prof = prof.round(4)
    print(prof.to_string())
    top = prof["hareket"].sort_values(ascending=False).head(4)
    print(f"\n  → En hareketli saatler (edge potansiyeli): {list(top.index)}")
    print(f"  → Bu saatler ort. hareket: %{top.mean():.4f}  "
          f"(tüm gün ort: %{prof['hareket'].mean():.4f})")


def autocorrelation(d):
    section("2) OTOKORELASYON (momentum mu, mean-reversion mu?)")
    ret = d["close"].pct_change().dropna()
    print("  Bar-getirisi otokorelasyonu (lag = kaç bar sonrası):")
    print("  + değer = MOMENTUM (devam eder) | - değer = MEAN-REVERSION (geri döner)\n")
    for lag in [1, 2, 3, 5, 10, 20]:
        ac = ret.autocorr(lag)
        bar = "█" * int(abs(ac) * 200)
        sign = "MOM " if ac > 0 else "REV "
        print(f"   lag {lag:>2}: {ac:+.4f}  {sign}{bar}")
    # Genel eğilim
    short_ac = ret.autocorr(1)
    print(f"\n  → 1-bar otokorelasyon: {short_ac:+.4f} → "
          f"{'kısa vadede MOMENTUM' if short_ac > 0.01 else 'kısa vadede MEAN-REVERSION' if short_ac < -0.01 else 'nötr (rastgele)'}")


def opening_range(d, open_hour, or_minutes=30):
    section(f"3) AÇILIŞ ARALIĞI (saat {open_hour}:00 sonrası ilk {or_minutes}dk)")
    # Her gün: açılış aralığını (ilk or_minutes) belirle, sonra gün kapanışına kadar
    # fiyat aralığı YUKARI mı AŞAĞI mı kırıyor, ve kırılım DEVAM mı ediyor (breakout)
    # yoksa GERİ mi dönüyor (fade)?
    df = d.copy()
    df["day"] = df.index.normalize()
    df["hour"] = df.index.hour
    df["minute"] = df.index.hour * 60 + df.index.minute
    open_min = open_hour * 60
    breakout_cont = 0   # kırdı ve devam etti
    breakout_fade = 0   # kırdı ama geri döndü
    days = 0
    up_break = 0
    for day, g in df.groupby("day"):
        sess = g[(g["minute"] >= open_min) & (g["minute"] < open_min + or_minutes)]
        rest = g[g["minute"] >= open_min + or_minutes]
        if len(sess) < 2 or len(rest) < 3:
            continue
        or_high, or_low = sess["high"].max(), sess["low"].min()
        or_mid = (or_high + or_low) / 2
        # İlk kırılım yönü
        broke_up = (rest["high"] > or_high).any()
        broke_dn = (rest["low"] < or_low).any()
        close_px = rest["close"].iloc[-1]
        days += 1
        if broke_up and not broke_dn:
            up_break += 1
            # devam mı (kapanış OR üstünde) yoksa fade mi
            if close_px > or_high:
                breakout_cont += 1
            else:
                breakout_fade += 1
        elif broke_dn and not broke_up:
            if close_px < or_low:
                breakout_cont += 1
            else:
                breakout_fade += 1
    if days:
        print(f"  İncelenen gün: {days}")
        print(f"  Tek-yön kırılımlı gün: {breakout_cont + breakout_fade}")
        print(f"    → Kırıp DEVAM eden (breakout işe yarar): {breakout_cont}")
        print(f"    → Kırıp GERİ DÖNEN (fade işe yarar):     {breakout_fade}")
        tot = breakout_cont + breakout_fade
        if tot:
            pct = breakout_cont / tot * 100
            print(f"\n  → Kırılımların %{pct:.1f}'i devam ediyor → "
                  f"{'BREAKOUT eğilimi' if pct > 55 else 'FADE eğilimi' if pct < 45 else 'belirsiz'}")


def day_of_week(d):
    section("4) HAFTANIN GÜNÜ ETKİSİ")
    df = d.copy()
    df["day"] = df.index.normalize()
    daily = df.groupby("day")["close"].agg(["first", "last"])
    daily["ret"] = (daily["last"] / daily["first"] - 1) * 100
    daily["dow"] = daily.index.dayofweek
    names = {0: "Pzt", 1: "Sal", 2: "Çar", 3: "Per", 4: "Cum", 6: "Paz"}
    g = daily.groupby("dow")["ret"].agg(["mean", "std", "count"]).round(4)
    g.index = [names.get(i, str(i)) for i in g.index]
    print(g.to_string())


def first_hour(d, open_hour):
    section(f"5) İLK SAAT (≥{open_hour}:00) → GÜN SONU İLİŞKİSİ")
    df = d.copy()
    df["day"] = df.index.normalize()
    df["minute"] = df.index.hour * 60 + df.index.minute
    open_min = open_hour * 60
    rows = []
    for day, g in df.groupby("day"):
        fh = g[(g["minute"] >= open_min) & (g["minute"] < open_min + 60)]
        rest = g[g["minute"] >= open_min + 60]
        if len(fh) < 3 or len(rest) < 3:
            continue
        fh_ret = fh["close"].iloc[-1] / fh["open"].iloc[0] - 1
        rest_ret = rest["close"].iloc[-1] / rest["open"].iloc[0] - 1
        rows.append((fh_ret, rest_ret))
    if len(rows) > 20:
        arr = np.array(rows)
        corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
        print(f"  İlk-saat getirisi ile günün geri kalanı korelasyonu: {corr:+.3f}")
        print(f"  → {'POZİTİF: ilk saat yönü devam eder (momentum)' if corr > 0.05 else 'NEGATİF: ilk saat tersine döner (reversion)' if corr < -0.05 else 'İlişki yok'}")


def parse_args():
    p = argparse.ArgumentParser(description="Gün-içi veri keşfi")
    p.add_argument("--data", default="data/NAS100_M5.csv")
    p.add_argument("--open-hour", type=int, default=None,
                   help="NY açılış saati (broker saati). Verilmezse en hareketli saatten tahmin edilir.")
    return p.parse_args()


def main():
    args = parse_args()
    d = load_mt5_csv(args.data)
    print(f"Veri: {Path(args.data).stem} | {len(d)} bar | "
          f"{d.index[0].date()} → {d.index[-1].date()}")

    hourly_profile(d)
    autocorrelation(d)

    open_hour = args.open_hour
    if open_hour is None:
        # En hareketli saati açılış kabul et
        ret_abs = d["close"].pct_change().abs()
        open_hour = int(ret_abs.groupby(d.index.hour).mean().idxmax())
        print(f"\n  (Açılış saati otomatik seçildi: {open_hour}:00 — en hareketli saat)")

    opening_range(d, open_hour)
    day_of_week(d)
    first_hour(d, open_hour)

    section("SONUÇ")
    print("  Yukarıdaki kalıplara göre bir strateji hipotezi kuracağız,")
    print("  sonra MUTLAKA overfit_test.py'den geçireceğiz (altın kural).")


if __name__ == "__main__":
    main()
