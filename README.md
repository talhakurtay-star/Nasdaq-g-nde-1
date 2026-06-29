# Nasdaq Prop-Firm Backtest Botu

MetaTrader 5 (MT5) verisiyle çalışan, Python tabanlı bir **gün-içi backtest / simülasyon** botu.
Prop-firm (FTMO tarzı) disiplini ile çalışır; gerçek para riski yoktur.

## Çalışma Felsefesi

Aylık kazancı **kovalamayız**. Bunun yerine her gün küçük, ulaşılabilir bir hedefi yakalarız:

- 🎯 **Günlük hedef %0.44** — ulaşılınca bot **o gün kilitlenir**, başka işlem yapmaz (kârı geri vermez).
- 🛑 **Günlük stop %0.44** (1:1) — değince o gün kilitlenir (bir kötü gün = bir iyi gün).
- 🏦 **Firma sınırları** — günlük max DD %5, aylık max DD %10 sert backstop olarak korunur.

> **Matematik:** Günde %0.44 × ~22 işlem günü → bileşik `1.0044²² ≈ %10.1` aylık.
> Hedefi kovalamak yerine günlük küçük dilimlere bölmek çok daha güvenli.

## Özellikler

- 📥 **MT5 veri yükleyici** — "Export Bars" (tab-ayrılmış) ve `MetaTrader5.copy_rates` CSV biçimlerini otomatik tanır; her timeframe (M1/M5/M15…).
- 🗓️ **Seans motoru** — her işlem gününü bağımsız yönetir, gün-içi kümülatif P&L'e göre target/stop kilidi uygular, gün sonunda pozisyonu kapatır.
- 📈 **Strateji** — Hareketli ortalama kesişimi (SMA/EMA) + opsiyonel RSI filtresi.
- ⚙️ **Gerçekçi motor** — sinyaller bir sonraki bar açılışında uygulanır (veri sızıntısı yok); target/stop intrabar (high/low) tetiklenir; komisyon + slipaj.
- 📊 **Metrikler** — genel (Sharpe, Sortino, drawdown…) + seans (hedefe ulaşan gün oranı, stop günü, en iyi/en kötü gün, firma ihlal kontrolü).
- 🖼️ **Grafik** — equity eğrisi ve drawdown PNG çıktısı.

## Kurulum

```bash
pip install -r requirements.txt
```

## MT5'ten Veri İndirme

1. MetaTrader 5'te sembolü açın (örn. **NDX**, **USTEC**, **NAS100** — brokerınıza göre değişir).
2. **Görünüm → Semboller** (veya grafikte F2) → ilgili sembol → **Bars** sekmesi → zaman aralığını seçip **Export** / **Save**.
3. Oluşan CSV/TXT dosyasını bu projedeki `data/` klasörüne kopyalayın.

> Not: Gerçek veriler `.gitignore` ile repoya dahil edilmez (`SAMPLE_*.csv` hariç).

## Kullanım

Önce örnek (sentetik, gün-içi) veriyle deneyin:

```bash
python tools/generate_sample_data.py --tf M5          # M5 örnek veri üret
python main.py --data data/SAMPLE_NDX_M5.csv          # seans modu (varsayılan)
```

Farklı timeframe karşılaştırması:

```bash
python tools/generate_sample_data.py --tf M1
python tools/generate_sample_data.py --tf M15
python main.py --data data/SAMPLE_NDX_M1.csv  --fast 20 --slow 60
python main.py --data data/SAMPLE_NDX_M15.csv --fast 10 --slow 30
```

Gerçek MT5 verinizle:

```bash
python main.py --data data/NDX_M5.csv --target 0.44 --stop 0.44 --plot eq.png
python main.py --data data/NDX_M5.csv --mode simple   # klasik (kilitsiz) mod
```

### Parametreler

| Bayrak | Açıklama | Varsayılan |
|---|---|---|
| `--data` | MT5 CSV dosya yolu (zorunlu) | — |
| `--mode` | `session` (kilit + risk) / `simple` | `session` |
| `--target` | Günlük kâr hedefi % | 0.44 |
| `--stop` | Günlük kayıp stop'u % | 0.44 |
| `--monthly-dd` | Aylık max DD % (firma) | 10 |
| `--leverage` | Kaldıraç (1.0 = yok) | 1.0 |
| `--fast` / `--slow` | Hızlı / yavaş MA periyodu | 20 / 50 |
| `--ma` | `sma` veya `ema` | `ema` |
| `--no-rsi` | RSI filtresini kapat | açık |
| `--rsi-ob` | RSI aşırı alım seviyesi | 70 |
| `--cash` | Başlangıç sermayesi | 10000 |
| `--commission` / `--slippage` | Maliyet oranları | 0.0005 / 0.0002 |
| `--plot` | Equity grafiğini PNG'ye kaydet | — |

### Sonuçları yorumlama

Önemli olan **hedefe ulaşan gün oranının stop günlerini geçmesi**. Bot kazancı +%0.44,
zararı -%0.44 ile sabitler; işin matematiği şu: bu oranı %50'nin belirgin üstüne çıkaran
bir strateji/parametre + gerçek piyasa avantajı bulmak. Rastgele veride bu mümkün değildir.

## Proje Yapısı

```
.
├── main.py                 # Giriş noktası (CLI)
├── src/
│   ├── data_loader.py      # MT5 CSV okuyucu
│   ├── strategy.py         # Stratejiler + indikatörler (SMA/EMA/RSI)
│   ├── risk.py             # Prop-firm risk parametreleri
│   ├── session_backtester.py # Gün-içi seans motoru (target/stop kilidi)
│   ├── backtester.py       # Klasik backtest motoru
│   └── metrics.py          # Performans + seans metrikleri
├── tools/
│   └── generate_sample_data.py  # Sentetik MT5 verisi üretici
├── tests/
│   └── test_backtest.py    # Pytest testleri
├── data/                   # MT5 verileri buraya
└── requirements.txt
```

## Test

```bash
python -m pytest tests/ -q
```

## Yol Haritası (sonraki adımlar)

- [x] Gün-içi seans motoru + günlük target/stop kilidi
- [x] Firma risk sınırları (günlük %5, aylık %10)
- [ ] Parametre optimizasyonu (grid search / walk-forward) — en iyi hedef-tutturma oranını bulmak için
- [ ] Daha fazla strateji (breakout, VWAP, MACD, mean-reversion)
- [ ] Seans saati filtresi (örn. ilk 30 dk / haber saatlerinden kaçın)
- [ ] Kaldıraç + pozisyon boyutlandırma optimizasyonu
- [ ] MT5 canlı köprüsü (paper → canlı)

## Uyarı

Bu bir eğitim/araştırma aracıdır. Geçmiş performans gelecekteki sonuçların garantisi değildir.
Gerçek para ile işlem yapmadan önce kapsamlı test yapın.
