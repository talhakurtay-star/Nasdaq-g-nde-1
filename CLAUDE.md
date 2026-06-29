# Proje Kuralları — Nasdaq Prop-Firm Botu

## ⛔ EN ÖNEMLİ KURAL: OVERFITTING (AŞIRI UYDURMA) YAPMA

Bu projede **hiçbir strateji veya parametre, overfitting testinden geçmeden
"çalışıyor" sayılmaz.** Geçmiş veriye uydurulmuş güzel görünen sonuçlar
canlıda çöker. Kullanıcının net talimatı: **OVERFITTING YAPMA.**

### Zorunlu kurallar
1. **Asla tüm veride optimize edip o parametreye güvenme.** Mutlaka eğitim/test
   ayrımı (in-sample / out-of-sample) yap.
2. **Her yeni strateji/parametre `tools/overfit_test.py`'den geçirilecek.**
   Dört testin de geçmesi gerekir:
   - In-sample / Out-of-sample (OOS/IS oranı ≥ 0.6)
   - Walk-forward (katların çoğu OOS'ta pozitif)
   - Sembol-arası robustluk (birden çok piyasada pozitif)
   - Parametre duyarlılığı (geniş plato; en iyi sonuç pozitif olmalı)
3. **Bir test bile ✗ veriyorsa strateji REDDEDİLİR.** "Eğitimde güzeldi" gerekçe değildir.
4. **Sonuçları dürüst raporla.** Zarar varsa zarar de; süsleme.
5. **Az parametre, geniş plato** tercih et. Çok parametreli, tek-nokta-tepe
   sonuçlar overfitting işaretidir.

### Neden (kanıt)
NAS100 4.5 yıllık veride mean-reversion stratejisi eğitimde "ayarlanınca" makul
görünüyordu; ama overfitting testi onu eledi:
- Walk-forward: 0/4 kat OOS pozitif
- Sembol-arası: 0/5 sembol pozitif (US500/US30/GER40/NVDA/XAU hepsi -47%..-50%)
- Hedef-oran her yerde ~%0

Bu çerçeve, **tek kuruş riske atmadan** kötü stratejiyi yakaladı. Amaç budur.

---

## Botun Çalışma Felsefesi
- Aylık kazancı KOVALAMA. Günlük küçük hedef (varsayılan %0.44) yakala, ulaşınca
  o gün KİLİTLEN.
- Günlük stop %0.44 (1:1). Firma sınırları: günlük %5, aylık %10 DD (sert backstop).
- Gün-içi çalış, gün sonunda pozisyonu kapat (overnight risk yok).

## Mimari
- `src/data_loader.py` — MT5 CSV okuyucu (Export Bars + copy_rates)
- `src/strategy.py` — stratejiler (MACross, MeanReversion); sinyal -1/0/+1
- `src/risk.py` — risk parametreleri (target/stop, SL/TP, boyutlandırma, firma sınırı)
- `src/session_backtester.py` — gün-içi seans motoru (long+short, boyutlandırma)
- `src/metrics.py` — performans + seans metrikleri
- `src/optimizer.py` — paralel grid-search
- `tools/overfit_test.py` — **ZORUNLU overfitting testi**
- `tools/batch_backtest.py` — çoklu sembol karşılaştırma

## Test / Çalıştırma
```bash
python -m pytest tests/ -q                          # birim testleri
python tools/overfit_test.py --data data/NAS100_M15.csv   # ZORUNLU overfit testi
python tools/batch_backtest.py                      # çoklu sembol
```

## Doğrulanmış Gerçekler (tekrar keşfetme)
- Motor doğru: geleceği bilen oracle sinyali +622805% / %98.7 isabet veriyor.
- Naif stratejiler (MA-crossover, basit mean-reversion) gün-içinde maliyetleri
  yenemiyor — overfit testinden geçemiyorlar.
- Risk-bazlı boyutlandırmada pozisyonu `risk_per_trade_pct` belirler; `leverage`
  sadece tavandır (genelde bağlamaz).
- %0.44 günlük hedef güvenli boyutlandırmayla ulaşılması ZOR; gerçek avantaj şart.

## Git
- Geliştirme dalı: `claude/fervent-bell-6fpz34`
- Gerçek veri dosyaları `.gitignore` ile hariç (sadece `SAMPLE_*.csv` izinli).
