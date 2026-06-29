# Nasdaq Prop-Firm Botu

MetaTrader 5 (MT5) verisiyle çalışan, Python tabanlı **gün-içi backtest / sinyal** botu.
Günde **%0.44**'e ulaşınca kendini kilitler; prop-firm (FTMO tarzı) risk disiplini ile çalışır.

---

> ## ⛔ ALTIN KURAL: OVERFITTING YAPMA
> **Hiçbir strateji/parametre, `tools/overfit_test.py`'den geçmeden "çalışıyor" sayılmaz.**
> Dört testin (IS/OOS, walk-forward, sembol-arası, duyarlılık) hepsi geçmeli; biri bile
> ✗ ise strateji **reddedilir**. Detaylar: [`CLAUDE.md`](CLAUDE.md)

---

## Çalışma Felsefesi
- Aylık kazancı **kovalamayız**. Her gün küçük hedef (%0.44) yakala, ulaşınca **o gün kilitlen**.
- Günlük stop %0.44 (1:1). Firma sınırları: günlük %5, aylık %10 DD (sert backstop).
- Gün-içi çalış, gün sonunda kapan (overnight risk yok).
- **Matematik:** %0.44 × ~22 gün ≈ aylık %10. (Gerçek edge'le ulaşılması ZOR — aşağıdaki bulgulara bak.)

## Kurulum
```bash
pip install -r requirements.txt
```

## MT5'ten Veri
Sembolü aç → **F2 → Export** → `data/SEMBOL_TF.csv` (örn. `NAS100_M5.csv`).
Format otomatik tanınır (Export Bars / copy_rates). Gerçek veriler `.gitignore`'da.

## Kullanım

**Backtest (varsayılan: ORB stratejisi + 0.44 kilit):**
```bash
python main.py --data data/NAS100_M5.csv
python main.py --data data/NAS100_M5.csv --lock breakeven --lev 2 --plot eq.png
python main.py --data data/NAS100_M5.csv --strategy meanrev
```

**Bugünkü kararı gör (canlı sinyal):**
```bash
python main.py --data data/NAS100_M5.csv --today
```

**Overfitting testi (ZORUNLU — her strateji için):**
```bash
python tools/overfit_test.py --strategy orb  --data data/NAS100_M5.csv
python tools/overfit_test.py --strategy meanrev --data data/NAS100_M15.csv
```

**Çoklu sembol karşılaştırma:**
```bash
python tools/batch_backtest.py --sort total_return
```

**Veri keşfi / gösterge bataryası / ML:**
```bash
python tools/explore.py --data data/NAS100_M5.csv
python tools/indicator_battery.py --data data/NAS100_M5.csv
python tools/ml_signal.py --data data/NAS100_M5.csv
```

## Stratejiler
- **ORB** (Opening Range Breakout, varsayılan) — açılış aralığı kırılımı, gün boyu yön kilidi, günde tek işlem.
- **MeanReversion** — z-score/Bollinger ortalamaya dönüş (long+short).
- **MACross** — hareketli ortalama kesişimi (referans/zayıf).

## Proje Yapısı
```
main.py                      # Bot çalıştırıcı (backtest + --today canlı sinyal)
src/
  data_loader.py             # MT5 CSV okuyucu
  strategy.py                # ORB, MeanReversion, MACross
  risk.py                    # Risk: 0.44 kilit, SL/TP, sizing, seans saati, firma sınırı
  session_backtester.py      # Gün-içi seans motoru (long/short, kilit modları)
  metrics.py                 # Performans + seans metrikleri
  optimizer.py               # Paralel grid-search (orb/meanrev)
  indicators.py              # RSI, MACD, ATR, ADX, VWAP, Supertrend...
  indicator_strategies.py    # 10 gösterge stratejisi
  ml_features.py             # ML için nedensel özellikler
tools/
  overfit_test.py            # ZORUNLU 4-testli overfitting kapısı
  batch_backtest.py          # Çoklu sembol
  explore.py                 # Veri keşfi
  indicator_battery.py       # Gösterge bataryası (OOS)
  ml_signal.py               # ML yön sinyali (walk-forward)
  generate_sample_data.py    # Sentetik MT5 verisi
tests/                       # pytest (14 test)
```

## Bulgular — KANITLANMIŞ (dürüst)
Sıfırdan kurulup titizlikle test edildi. Özet:
- **Gün-içi piyasa bu ölçekte verimli.** Naif göstergeler (10 adet test edildi) ve MA/mean-reversion edge taşımıyor.
- **Tek gerçek edge: ORB** (NAS100/NVDA) — ama küçük (~%2/yıl) ve walk-forward'da kırılgan.
- **ML** (gradient boosting, walk-forward): yön doğruluğu ~%52 (gerçek ama ince); maliyeti yenemiyor.
- **Günde %0.44 (≈%200/yıl)** tek-enstrümanlı basit botla **ulaşılamaz**. Gerçekçi tavan ~%1.5-3/yıl.
- Ayrıntı ve kanıtlar: [`CLAUDE.md`](CLAUDE.md).

## Test
```bash
python -m pytest tests/ -q
```

## Uyarı
Eğitim/araştırma aracıdır; yatırım tavsiyesi değildir. Geçmiş performans geleceği garanti etmez.
