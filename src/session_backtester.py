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

from dataclasses import dataclass, field

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

    def run(self, data: pd.DataFrame, signals: pd.Series) -> SessionResult:
        cfg, risk = self.config, self.risk
        data = data.copy()
        target_pos = signals.reindex(data.index).fillna(0).shift(1).fillna(0).astype(int)

        equity = cfg.initial_cash          # gerçekleşmiş (realized) hesap equity'si
        cash = cfg.initial_cash
        shares = 0.0
        position = 0
        entry_price = 0.0
        entry_time = None

        cur_day = None
        day_start_equity = equity
        locked_today = False
        target_equity = stop_equity = 0.0

        cur_month = None
        month_start_equity = equity
        month_locked = False

        equity_curve: list[float] = []
        trades: list[dict] = []
        day_records: dict = {}

        idx = data.index
        opens, highs, lows, closes = data["open"], data["high"], data["low"], data["close"]
        n = len(idx)

        def open_long(price: float, ts) -> None:
            nonlocal cash, shares, position, entry_price, entry_time
            fill = price * (1 + cfg.slippage)
            invest = cash * risk.leverage
            cost = invest * cfg.commission
            shares = (invest - cost) / fill
            cash = 0.0
            position = 1
            entry_price = fill
            entry_time = ts

        def close_long(price: float, ts, reason: str) -> None:
            nonlocal cash, shares, position, equity
            fill = price * (1 - cfg.slippage)
            proceeds = shares * fill
            cost = proceeds * cfg.commission
            # kaldıraçlı nominal: yatırılan öz sermaye + P&L
            pnl = (fill - entry_price) * shares - cost
            equity = equity + pnl
            cash = equity
            trades.append({
                "entry_time": entry_time, "exit_time": ts,
                "entry_price": entry_price, "exit_price": fill,
                "shares": shares, "pnl": pnl,
                "return_pct": (fill / entry_price - 1) * 100,
                "reason": reason,
            })
            shares = 0.0
            position = 0

        for i in range(n):
            ts = idx[i]
            d = ts.normalize()                 # gün anahtarı
            m = (ts.year, ts.month)
            is_last = (i == n - 1)
            is_day_end = is_last or idx[i + 1].normalize() != d

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
                target_equity = day_start_equity * (1 + risk.daily_target)
                stop_equity = day_start_equity * (1 - risk.daily_stop)
                day_records[d] = {
                    "date": d, "start_equity": day_start_equity,
                    "outcome": "neutral", "trades": 0,
                }

            # --- Aylık DD backstop ---
            if not month_locked and equity <= month_start_equity * (1 - risk.monthly_dd):
                month_locked = True
                locked_today = True

            o, h, l, c = opens.iloc[i], highs.iloc[i], lows.iloc[i], closes.iloc[i]
            desired = target_pos.iloc[i]
            if locked_today or month_locked:
                desired = 0

            # 1) Bar açılışında strateji kaynaklı giriş/çıkış
            if desired == 0 and position == 1:
                close_long(o, ts, "signal")
                day_records[cur_day]["trades"] += 1
            elif desired == 1 and position == 0 and not is_day_end:
                # gün sonu barında yeni pozisyon açma (kapatamadan kapanış olur)
                open_long(o, ts)

            # 2) Bar içi: kümülatif gün P&L'ine göre target/stop (sadece pozisyondayken)
            if position == 1 and not locked_today:
                # bu bardaki nakit 0 (all-in long), equity = shares*fiyat
                stop_price = stop_equity / shares
                target_price = target_equity / shares
                # kötümser: önce stop
                if l <= stop_price:
                    close_long(stop_price / (1 - cfg.slippage), ts, "daily_stop")
                    day_records[cur_day]["trades"] += 1
                    day_records[cur_day]["outcome"] = "stop"
                    locked_today = True
                elif h >= target_price:
                    close_long(target_price / (1 - cfg.slippage), ts, "daily_target")
                    day_records[cur_day]["trades"] += 1
                    day_records[cur_day]["outcome"] = "target"
                    locked_today = True

            # 3) Gün sonu: açık pozisyonu kapat
            if is_day_end and position == 1 and risk.flat_at_session_end:
                close_long(c, ts, "session_end")
                day_records[cur_day]["trades"] += 1

            # 4) Bar kapanışında equity'yi işaretle
            mark = equity if position == 0 else shares * c
            equity_curve.append(mark)

            # 5) Kümülatif gün P&L'i eşiği geçtiyse günü kilitle.
            #    (Pozisyon dışıyken sinyalle biriken küçük zararlar/kârlar da
            #     stop/target'ı aşmasın diye; yeni işlem açılmaz.)
            if not locked_today:
                day_pl = mark / day_start_equity - 1
                if day_pl <= -risk.daily_stop:
                    locked_today = True
                    if day_records[cur_day]["outcome"] == "neutral":
                        day_records[cur_day]["outcome"] = "stop"
                elif day_pl >= risk.daily_target:
                    locked_today = True
                    if day_records[cur_day]["outcome"] == "neutral":
                        day_records[cur_day]["outcome"] = "target"

            # Gün sonu kaydını tamamla
            if is_day_end:
                rec = day_records[cur_day]
                end_eq = equity if position == 0 else shares * c
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
