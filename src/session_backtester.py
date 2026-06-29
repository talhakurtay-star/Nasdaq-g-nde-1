"""Gün-içi (seans) backtest motoru — prop-firm kuralları ile.

Her işlem gününü bağımsız bir seans olarak yönetir:
  - Gün başında: gün-başı equity sabitlenir, kilit kalkar.
  - Gün içinde strateji sinyallerine göre long açılır/kapanır (bir sonraki bar açılışında).
  - Kümülatif gün P&L'i +daily_target'a ulaşınca → pozisyon kapanır, GÜN KİLİTLENİR.
  - Kümülatif gün P&L'i -daily_stop'a değince → pozisyon kapanır, GÜN KİLİTLENİR.
  - Gün sonunda açık pozisyon kapatılır (overnight risk yok).
  - Aylık DD sert sınırı aşılırsa o ay kilitlenir (backstop).

Target/stop intrabar (bar high/low) tetiklenir; aynı barda ikisi de değerse
kötümser kabulle ÖNCE stop uygulanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .risk import RiskParams


@dataclass
class SessionConfig:
    initial_cash: float = 10_000.0
    commission: float = 0.0005   # işlem başına oran
    slippage: float = 0.0002     # yürütme kayması oranı


@dataclass
class SessionResult:
    equity: pd.Series              # her bar sonunda işaretlenen equity
    trades: pd.DataFrame
    days: pd.DataFrame             # günlük seans özetleri
    metrics: dict

    def summary(self) -> str:
        from .metrics import format_metrics, format_session_summary
        return format_metrics(self.metrics) + "\n" + format_session_summary(self.days, self.metrics)


class SessionBacktester:
    def __init__(self, config: SessionConfig | None = None, risk: RiskParams | None = None) -> None:
        self.config = config or SessionConfig()
        self.risk = risk or RiskParams()

    def run(self, data: pd.DataFrame, signals: pd.Series,
            stop_dist_pct: pd.Series | None = None) -> SessionResult:
        """stop_dist_pct verilirse (her bar için fiyatın %'si olarak stop mesafesi),
        sabit stop_loss_pct yerine bu volatiliteye-uyarlı (ATR vb.) stop kullanılır;
        pozisyon, bu mesafede risk_per_trade kadar kaybedecek şekilde boyutlandırılır."""
        cfg, risk = self.config, self.risk
        data = data.copy()
        target_pos = signals.reindex(data.index).fillna(0).shift(1).fillna(0).astype(int)
        if stop_dist_pct is not None:
            sd_arr = stop_dist_pct.reindex(data.index).shift(1).to_numpy()
        else:
            sd_arr = None

        equity = cfg.initial_cash          # gerçekleşmiş (realized) hesap equity'si
        position = 0                       # -1 short, 0 flat, +1 long
        qty = 0.0                          # pozisyon büyüklüğü (adet)
        entry_fill = 0.0                   # giriş fiyatı (slipaj dahil)
        entry_notional = 0.0               # giriş nominali
        entry_time = None
        sl_price = 0.0                     # işlem başına stop fiyatı (0 = yok)
        tp_price = 0.0                     # işlem başına hedef fiyatı (0 = yok)

        cur_day = None
        day_start_equity = equity
        locked_today = False          # gün tamamen kilitli (pozisyon kapalı, yeni işlem yok)
        entries_locked = False        # sadece yeni işlem yok (pozisyon koşmaya devam edebilir)
        target_reached = False        # gün +target'a değdi mi
        day_floor_equity = 0.0        # etkin günlük zarar tabanı (yükselebilir)
        entries_today = 0
        target_equity = stop_equity = 0.0
        max_entries = risk.max_trades_per_day
        lock_mode = risk.profit_lock_mode
        trail_amt = risk.trail_pct / 100.0   # gün başı equity oranı

        cur_month = None
        month_start_equity = equity
        month_locked = False

        equity_curve: list[float] = []
        trades: list[dict] = []
        day_records: dict = {}

        idx = data.index
        n = len(idx)
        # Hız için tüm seri erişimlerini numpy dizilerine çevir (.iloc döngüde çok yavaş).
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        target_arr = target_pos.to_numpy()
        # Gün/ay anahtarlarını ve gün-sonu bayrağını önceden vektörel hesapla.
        day_key = idx.normalize().to_numpy()          # her bar için takvim günü
        month_key = (idx.year * 100 + idx.month).to_numpy()
        is_day_end_arr = np.empty(n, dtype=bool)
        if n:
            is_day_end_arr[-1] = True
            is_day_end_arr[:-1] = day_key[1:] != day_key[:-1]
        # İşlem saati penceresi (broker saati): pencere dışında pozisyon açılmaz.
        hours = idx.hour.to_numpy()
        in_session_arr = (hours >= risk.session_start_hour) & (hours < risk.session_end_hour)

        slip = cfg.slippage
        comm = cfg.commission

        sl_frac = risk.stop_loss_pct / 100.0
        tp_frac = risk.take_profit_pct / 100.0
        risk_frac = risk.risk_per_trade_pct / 100.0

        from collections import deque
        recent_results: deque = deque(maxlen=max(risk.cooldown_lookback, 1))

        def size_factor() -> float:
            """Soğuma modu: son N işlemde yeterli kazanç yoksa boyutu küçült."""
            if risk.cooldown_lookback <= 0 or len(recent_results) < risk.cooldown_lookback:
                return 1.0
            if sum(recent_results) < risk.cooldown_min_wins:
                return risk.cooldown_factor
            return 1.0

        def open_position(direction: int, price: float, ts, sl_now: float) -> bool:
            # Long girişte fiyat yukarı (alış), short girişte aşağı (satış) kayar.
            nonlocal position, qty, entry_fill, entry_notional, entry_time, sl_price, tp_price
            fill = price * (1 + direction * slip)
            sf = size_factor()
            if sf <= 0:
                return False                     # soğuma: bu işlemi tamamen atla
            max_notional = equity * risk.leverage * sf
            if sl_now > 0:
                # SL'e değince risk_frac kadar kayıp olacak şekilde boyutlandır.
                qty_risk = (equity * risk_frac * sf) / (sl_now * fill)
                qty = min(qty_risk, max_notional / fill)   # nominali tavanla sınırla
                sl_price = fill * (1 - direction * sl_now)
                tp_price = fill * (1 + direction * tp_frac) if tp_frac > 0 else 0.0
            else:
                qty = max_notional / fill                  # eski all-in davranışı
                sl_price = 0.0
                tp_price = 0.0
            entry_fill = fill
            entry_notional = qty * fill
            position = direction
            entry_time = ts
            return True

        def close_position(price: float, ts, reason: str) -> None:
            # Long çıkışta satış (fiyat aşağı), short çıkışta alış (fiyat yukarı) kayar.
            nonlocal position, qty, equity
            exit_fill = price * (1 - position * slip)
            pnl_gross = position * (exit_fill - entry_fill) * qty
            costs = (entry_notional + qty * exit_fill) * comm   # her iki bacak komisyonu
            equity = equity + pnl_gross - costs
            trades.append({
                "entry_time": entry_time, "exit_time": ts,
                "direction": "long" if position == 1 else "short",
                "entry_price": entry_fill, "exit_price": exit_fill,
                "qty": qty, "pnl": pnl_gross - costs,
                "return_pct": position * (exit_fill / entry_fill - 1) * 100,
                "reason": reason,
            })
            recent_results.append(1 if (pnl_gross - costs) > 0 else 0)
            qty = 0.0
            position = 0

        def mark_to_market(price: float) -> float:
            return equity if position == 0 else equity + position * (price - entry_fill) * qty

        for i in range(n):
            d = day_key[i]                     # gün anahtarı (numpy datetime64)
            m = month_key[i]
            is_day_end = is_day_end_arr[i]

            # --- Yeni ay ---
            if m != cur_month:
                cur_month = m
                month_start_equity = equity
                month_locked = False

            # --- Yeni gün ---
            if d != cur_day:
                cur_day = d
                day_start_equity = equity
                locked_today = False
                entries_locked = False
                target_reached = False
                entries_today = 0
                target_equity = day_start_equity * (1 + risk.daily_target)
                stop_equity = day_start_equity * (1 - risk.daily_stop)
                day_floor_equity = stop_equity
                day_records[d] = {
                    "date": d, "start_equity": day_start_equity,
                    "outcome": "neutral", "trades": 0,
                }

            # --- Aylık DD backstop ---
            if not month_locked and equity <= month_start_equity * (1 - risk.monthly_dd):
                month_locked = True
                locked_today = True

            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            desired = int(target_arr[i])       # -1 / 0 / +1
            # Kilitli, aylık kilitli, seans dışı ya da günlük işlem limiti dolduysa pozisyon yok.
            entries_full = max_entries > 0 and entries_today >= max_entries
            # entries_locked: +target sonrası yeni işlem yok ama mevcut pozisyon korunur;
            # bu yüzden yön DEĞİŞTİRMEYİ de engelle (sadece mevcut pozisyonu tut).
            if locked_today or month_locked or not in_session_arr[i]:
                desired = 0
            elif entries_locked:
                desired = position           # değişiklik yok, pozisyonu olduğu gibi tut

            # 1) Bar açılışında strateji kaynaklı yön değişimi (gerekirse ters çevir)
            if desired != position:
                if position != 0:
                    close_position(o, idx[i], "signal")
                    day_records[cur_day]["trades"] += 1
                # Yeni pozisyon: gün sonu barında değil ve günlük limit dolmadıysa
                if desired != 0 and not is_day_end and not entries_full:
                    # ATR-bazlı stop verildiyse onu, yoksa sabit stop_loss_pct'i kullan
                    if sd_arr is not None:
                        sd = sd_arr[i]
                        sl_now = (sd / 100.0) if (sd == sd and sd > 0) else sl_frac
                    else:
                        sl_now = sl_frac
                    if open_position(desired, o, idx[i], sl_now):
                        entries_today += 1

            # 2) Bar içi tetikleyiciler: işlem-bazlı SL/TP + günlük hesap taban/target.
            #    Zarar tarafı etkin günlük tabanı (day_floor_equity) kullanır; bu taban
            #    +target sonrası (breakeven/trail modunda) yükselebilir.
            if position != 0 and not locked_today:
                # mark(p) = equity + position*(p - entry_fill)*qty  →  hesap eşik fiyatları:
                daily_stop_p = entry_fill + position * (day_floor_equity - equity) / qty
                daily_target_p = entry_fill + position * (target_equity - equity) / qty

                if position == 1:
                    loss_p, loss_daily = daily_stop_p, True
                    if sl_price and sl_price > loss_p:
                        loss_p, loss_daily = sl_price, False
                    prof_p, prof_daily = daily_target_p, True
                    if tp_price and tp_price < prof_p:
                        prof_p, prof_daily = tp_price, False
                    hit_loss, hit_prof = l <= loss_p, h >= prof_p
                else:
                    loss_p, loss_daily = daily_stop_p, True
                    if sl_price and sl_price < loss_p:
                        loss_p, loss_daily = sl_price, False
                    prof_p, prof_daily = daily_target_p, True
                    if tp_price and tp_price > prof_p:
                        prof_p, prof_daily = tp_price, False
                    hit_loss, hit_prof = h >= loss_p, l <= prof_p

                if hit_loss:                           # kötümser: önce zarar
                    close_position(loss_p / (1 - position * slip), idx[i],
                                   "daily_stop" if loss_daily else "stop_loss")
                    day_records[cur_day]["trades"] += 1
                    if loss_daily:
                        # Günlük taban tetiklendi: target'a değdiyse "target" (artıda banklandı),
                        # değmediyse "stop".
                        day_records[cur_day]["outcome"] = "target" if target_reached else "stop"
                        locked_today = True
                elif hit_prof and not target_reached:
                    target_reached = True
                    day_records[cur_day]["outcome"] = "target"
                    if prof_daily and lock_mode == "hard":
                        close_position(prof_p / (1 - position * slip), idx[i], "daily_target")
                        day_records[cur_day]["trades"] += 1
                        locked_today = True
                    elif prof_daily:
                        # Yumuşak mod: KAPATMA, yeni işlem yok, tabanı yükselt.
                        entries_locked = True
                        if lock_mode == "breakeven":
                            day_floor_equity = max(day_floor_equity, day_start_equity)
                        else:  # trail: tepe - trail
                            day_floor_equity = max(day_floor_equity,
                                                   target_equity - day_start_equity * trail_amt)
                    else:
                        # işlem-bazlı TP (tp_price) hedefe değil, normal kapat
                        close_position(prof_p / (1 - position * slip), idx[i], "take_profit")
                        day_records[cur_day]["trades"] += 1
                        target_reached = False  # bu bir gün-hedefi değildi

            # 3) Gün sonu: açık pozisyonu kapat
            if is_day_end and position != 0 and risk.flat_at_session_end:
                close_position(c, idx[i], "session_end")
                day_records[cur_day]["trades"] += 1

            # 4) Bar kapanışında equity'yi işaretle
            mark = mark_to_market(c)
            equity_curve.append(mark)

            # 4b) Trail modu: hedefe değdikten sonra tabanı tepe-altı takip ettir
            if target_reached and lock_mode == "trail" and not locked_today:
                day_floor_equity = max(day_floor_equity, mark - day_start_equity * trail_amt)

            # 5) Kümülatif gün P&L'i eşik kontrolü (pozisyon dışıyken biriken drift dahil)
            if not locked_today:
                if mark <= day_floor_equity:
                    locked_today = True
                    if day_records[cur_day]["outcome"] == "neutral":
                        day_records[cur_day]["outcome"] = "target" if target_reached else "stop"
                elif not target_reached and mark >= target_equity:
                    target_reached = True
                    if day_records[cur_day]["outcome"] == "neutral":
                        day_records[cur_day]["outcome"] = "target"
                    if lock_mode == "hard":
                        locked_today = True
                    else:
                        entries_locked = True
                        if lock_mode == "breakeven":
                            day_floor_equity = max(day_floor_equity, day_start_equity)
                        else:
                            day_floor_equity = max(day_floor_equity,
                                                   target_equity - day_start_equity * trail_amt)

            # Gün sonu kaydını tamamla
            if is_day_end:
                rec = day_records[cur_day]
                end_eq = mark_to_market(c)
                rec["end_equity"] = end_eq
                rec["return_pct"] = (end_eq / rec["start_equity"] - 1) * 100

        equity_series = pd.Series(equity_curve, index=idx, name="equity")
        trades_df = pd.DataFrame(trades)
        days_df = pd.DataFrame(list(day_records.values())).set_index("date")
        metrics = compute_metrics(equity_series, trades_df)
        metrics.update(_session_metrics(days_df, self.risk))
        return SessionResult(equity=equity_series, trades=trades_df, days=days_df, metrics=metrics)


def _session_metrics(days: pd.DataFrame, risk: RiskParams) -> dict:
    """Seansa özel (prop-firm) metrikler."""
    if days.empty:
        return {}
    outcomes = days["outcome"].value_counts()
    worst_day = days["return_pct"].min()
    best_day = days["return_pct"].max()
    avg_day = days["return_pct"].mean()
    target_days = int(outcomes.get("target", 0))
    stop_days = int(outcomes.get("stop", 0))
    neutral_days = int(outcomes.get("neutral", 0))
    total_days = len(days)

    # Firma kuralı ihlali kontrolü
    firm_breach = bool(worst_day < -risk.firm_daily_dd_pct - 1e-9)

    return {
        "total_days": total_days,
        "target_days": target_days,
        "stop_days": stop_days,
        "neutral_days": neutral_days,
        "target_hit_rate": target_days / total_days if total_days else 0.0,
        "avg_daily_return_pct": avg_day,
        "best_day_pct": best_day,
        "worst_day_pct": worst_day,
        "firm_daily_breach": firm_breach,
    }
