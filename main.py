"""Nasdaq Prop-Firm Botu — ana çalıştırıcı.

Günde %0.44'e ulaşınca KENDİNİ KİLİTLER (o gün başka işlem yapmaz).
En iyi doğrulanmış strateji varsayılan: Opening Range Breakout (ORB).

Kullanım:
    python main.py --data data/NAS100_M5.csv
    python main.py --data data/NAS100_M5.csv --strategy orb --lock hard
    python main.py --data data/NAS100_M5.csv --strategy meanrev --plot eq.png
    python main.py --data data/NAS100_M5.csv --target 0.44 --stop 0.44 --lev 2

Stratejiler: orb (varsayılan), meanrev, macross
Kilit modu (--lock): hard (klasik), breakeven, trail
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_mt5_csv
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig
from src.strategy import MACrossStrategy, MeanReversionStrategy, OpeningRangeBreakout


def build_strategy(args):
    if args.strategy == "orb":
        return OpeningRangeBreakout(
            open_hour=args.open_hour, or_minutes=args.or_minutes,
            require_close_break=args.close_break,
            trend_ema_period=args.trend_ema,
            min_or_range_pct=args.min_or,
        )
    if args.strategy == "meanrev":
        return MeanReversionStrategy(lookback=args.lookback, entry_z=args.entry_z)
    return MACrossStrategy(fast=args.fast, slow=args.slow, ma_type=args.ma)


def parse_args():
    p = argparse.ArgumentParser(description="Nasdaq prop-firm botu (0.44 günlük kilit)")
    p.add_argument("--data", required=True, help="MT5 CSV dosya yolu")
    p.add_argument("--strategy", choices=["orb", "meanrev", "macross"], default="orb")
    # Günlük kilit / risk
    p.add_argument("--target", type=float, default=0.44, help="Günlük kâr hedefi %% (kilit)")
    p.add_argument("--stop", type=float, default=0.44, help="Günlük kayıp stop'u %%")
    p.add_argument("--lock", choices=["hard", "breakeven", "trail"], default="hard",
                   help="Hedefe ulaşınca: hard=kapat+kilitle, breakeven/trail=koştur")
    p.add_argument("--lev", type=float, default=2.0, help="Kaldıraç (boyut tavanı)")
    p.add_argument("--sl", type=float, default=1.5, help="İşlem stop'u %% (0=all-in)")
    p.add_argument("--risk", type=float, default=0.30, help="İşlem başına risk %%")
    p.add_argument("--session", nargs=2, type=int, default=[16, 22],
                   help="İşlem saati penceresi (broker saati)")
    p.add_argument("--max-trades", type=int, default=1, help="Günde max işlem (0=sınırsız)")
    # ORB
    p.add_argument("--open-hour", type=int, default=16)
    p.add_argument("--or-minutes", type=int, default=30)
    p.add_argument("--close-break", action="store_true")
    p.add_argument("--trend-ema", type=int, default=0)
    p.add_argument("--min-or", type=float, default=0.0)
    # meanrev / macross
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--ma", choices=["sma", "ema"], default="ema")
    # Hesap / maliyet
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--commission", type=float, default=0.0002)
    p.add_argument("--slippage", type=float, default=0.0001)
    p.add_argument("--plot", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Veri yükleniyor: {args.data}")
    data = load_mt5_csv(args.data)
    yrs = (data.index[-1] - data.index[0]).days / 365.25
    print(f"  {len(data)} bar | {data.index[0]} → {data.index[-1]} ({yrs:.1f} yıl)")
    print(f"  Strateji: {args.strategy} | günlük kilit: +{args.target}% ({args.lock}) | "
          f"saat {args.session[0]}-{args.session[1]}\n")

    strategy = build_strategy(args)
    signals = strategy.generate_signals(data)

    risk = RiskParams(
        daily_target_pct=args.target, daily_stop_pct=args.stop,
        profit_lock_mode=args.lock, leverage=args.lev,
        stop_loss_pct=args.sl, risk_per_trade_pct=args.risk,
        session_start_hour=args.session[0], session_end_hour=args.session[1],
        max_trades_per_day=args.max_trades,
    )
    config = SessionConfig(args.cash, args.commission, args.slippage)
    result = SessionBacktester(config, risk).run(data, signals)

    print(result.summary())

    m = result.metrics
    if m.get("total_days"):
        ann = ((1 + m["total_return"]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
        print(f"\n  Yıllık (bileşik) : {ann:+.2f} %")
        print(f"  Ortalama günlük  : {m['avg_daily_return_pct']:+.4f} %  "
              f"(hedef: +{args.target} %)")
        print(f"  {args.cash:.0f}$ → {args.cash * (1 + m['total_return']):.0f}$")

    if args.plot:
        _plot(result, data, Path(args.plot))


def _plot(result, data, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib yok; grafik atlandı.")
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(result.equity.index, result.equity.values, color="#1f77b4", label="Bot equity")
    a1.set_title("Equity Eğrisi"); a1.legend(); a1.grid(alpha=0.3)
    dd = (result.equity / result.equity.cummax() - 1) * 100
    a2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    a2.set_title("Drawdown %"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"\nGrafik: {path}")


if __name__ == "__main__":
    main()
