# Nasdaq Backtest Botu

MetaTrader 5 (MT5) verisiyle çalışan, Python tabanlı bir **backtest / simülasyon** botu.
Geçmiş fiyat verisi üzerinde alım-satım stratejilerini test eder; gerçek para riski yoktur.

## Özellikler

- 📥 **MT5 veri yükleyici** — "Export Bars" (tab-ayrılmış) ve `MetaTrader5.copy_rates` CSV biçimlerini otomatik tanır.
- 📈 **Strateji** — Hareketli ortalama kesişimi (SMA/EMA) + opsiyonel RSI filtresi.
- ⚙️ **Gerçekçi motor** — sinyaller bir sonraki barın açılışında uygulanır (veri sızıntısı yok), komisyon + slipaj hesaplanır.
- 📊 **Metrikler** — toplam getiri, CAGR, Sharpe, Sortino, max drawdown, kazanma oranı, kâr faktörü.
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

Önce örnek (sentetik) veriyle deneyin:

```bash
python tools/generate_sample_data.py
python main.py --data data/SAMPLE_NDX_D1.csv
```

Gerçek MT5 verinizle:

```bash
python main.py --data data/NDX_D1.csv --fast 20 --slow 50 --ma ema
python main.py --data data/NDX_D1.csv --no-rsi --plot equity.png
```

### Parametreler

| Bayrak | Açıklama | Varsayılan |
|---|---|---|
| `--data` | MT5 CSV dosya yolu (zorunlu) | — |
| `--fast` | Hızlı MA periyodu | 20 |
| `--slow` | Yavaş MA periyodu | 50 |
| `--ma` | `sma` veya `ema` | `ema` |
| `--no-rsi` | RSI filtresini kapat | açık |
| `--rsi-ob` | RSI aşırı alım seviyesi | 70 |
| `--cash` | Başlangıç sermayesi | 10000 |
| `--commission` | İşlem komisyon oranı | 0.0005 |
| `--slippage` | Slipaj oranı | 0.0002 |
| `--plot` | Equity grafiğini PNG'ye kaydet | — |

## Proje Yapısı

```
.
├── main.py                 # Giriş noktası (CLI)
├── src/
│   ├── data_loader.py      # MT5 CSV okuyucu
│   ├── strategy.py         # Stratejiler + indikatörler (SMA/EMA/RSI)
│   ├── backtester.py       # Backtest motoru
│   └── metrics.py          # Performans metrikleri
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

- [ ] Daha fazla strateji (Bollinger, MACD, breakout)
- [ ] Parametre optimizasyonu (grid search / walk-forward)
- [ ] Stop-loss / take-profit ve pozisyon boyutlandırma
- [ ] Çoklu sembol / portföy backtest
- [ ] Paper trading'e (Alpaca vb.) köprü

## Uyarı

Bu bir eğitim/araştırma aracıdır. Geçmiş performans gelecekteki sonuçların garantisi değildir.
Gerçek para ile işlem yapmadan önce kapsamlı test yapın.
