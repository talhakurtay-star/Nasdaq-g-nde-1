"""Prop-firm risk kuralları.

Botun çalışma felsefesi:
  - Aylık kazancı KOVALAMAYIZ. Bunun yerine her gün küçük, ulaşılabilir bir
    hedefi (varsayılan %0.44) yakalamaya çalışırız.
  - Günlük hedefe ulaşılınca bot O GÜN için KİLİTLENİR; gün boyu yeni işlem açmaz.
    (Kazancı geri vermemek için.)
  - Günlük kayıp stop'una değince de o gün için kilitlenir. (Tek kötü gün
    bir iyi günden fazlasını silmesin diye 1:1.)
  - Firma sınırları (günlük %5, aylık %10 DD) SERT backstop'tur; bot kendi
    daha sıkı sınırlarında zaten çok önce durur.

Matematik: günde %0.44, ayda ~22 işlem günü →
    bileşik (1.0044^22 - 1) ≈ %10.1 aylık.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskParams:
    # --- Botun kendi (yumuşak) sınırları ---
    daily_target_pct: float = 0.44   # gün başı equity'ye göre +%; ulaşınca günü kilitle
    daily_stop_pct: float = 0.44     # gün başı equity'ye göre -%; değince günü kilitle

    # --- Firma sınırları (sert backstop; normalde tetiklenmemeli) ---
    firm_daily_dd_pct: float = 5.0   # gün başı equity'ye göre izinli max günlük düşüş
    monthly_dd_pct: float = 10.0     # ay başı equity'ye göre izinli max aylık düşüş

    # --- İşlem başına risk / boyutlandırma ---
    # stop_loss_pct > 0 ise pozisyon, SL'e değdiğinde sermayenin risk_per_trade_pct
    # kadarını kaybedecek şekilde boyutlandırılır. 0 ise eski "all-in" davranışı.
    stop_loss_pct: float = 0.0        # işlem başına fiyat stop'u % (0 = kapalı)
    take_profit_pct: float = 0.0      # işlem başına fiyat hedefi % (0 = kapalı)
    risk_per_trade_pct: float = 0.15  # SL'e değince kaybedilecek sermaye %'si

    # --- Seans davranışı ---
    flat_at_session_end: bool = True  # gün sonunda pozisyonu kapat (overnight gap riski yok)
    leverage: float = 1.0             # max nominal = equity * leverage (boyut tavanı)

    def __post_init__(self) -> None:
        if self.daily_stop_pct > self.firm_daily_dd_pct:
            raise ValueError(
                f"Günlük stop (%{self.daily_stop_pct}) firma günlük DD sınırından "
                f"(%{self.firm_daily_dd_pct}) büyük olamaz."
            )
        if self.leverage <= 0:
            raise ValueError("leverage pozitif olmalı.")
        if self.stop_loss_pct < 0 or self.take_profit_pct < 0:
            raise ValueError("stop_loss_pct / take_profit_pct negatif olamaz.")

    # Oranlar (yüzde -> kesir)
    @property
    def daily_target(self) -> float:
        return self.daily_target_pct / 100.0

    @property
    def daily_stop(self) -> float:
        return self.daily_stop_pct / 100.0

    @property
    def firm_daily_dd(self) -> float:
        return self.firm_daily_dd_pct / 100.0

    @property
    def monthly_dd(self) -> float:
        return self.monthly_dd_pct / 100.0
