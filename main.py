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
    p.add_argument("--lock", choices=["hard", "breakeven", "trail"], default="breakeven",
                   help="Hedefe ulaşınca: hard=kapat+kilitle (kazananı keser), "
                        "breakeven=stop'u başabaşa çek+koştur (kuyruğu yakalar, ÖNERİLEN), trail=trailing")
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
    p.add_argument("--today", action="store_true",
                   help="Backtest yerine: en güncel günün sinyalini/kararını göster")
    p.add_argument("--monthly", action="store_true", help="Aylık P&L dökümünü göster")
    p.add_argument("--export", default=None,
                   help="trade ve equity'yi <ÖNEK>_trades.csv / _equity.csv olarak kaydet")
    p.add_argument("--preset", choices=["fundingpips"], default=None,
                   help="Hazır konfig. fundingpips: eval-optimize (iç stop %2.5, risk %1.5, breakeven)")
    return p.parse_args()


def apply_preset(args):
    """Hazır konfigürasyonları uygula (CLI argümanlarını override eder)."""
    if args.preset == "fundingpips":
        # Dönem-robust optimize: iç günlük stop firma %5'in yarısı (tampon) → günlük
        # ihlal yapısal engelli; breakeven kazananı koşturur; risk %1.5 (düşük ihlal).
        args.strategy = "orb"
        args.target = 2.5        # iç günlük tavan (lock)
        args.stop = 2.5          # iç günlük stop (firma %5'in yarısı)
        args.lock = "breakeven"
        args.lev = 10.0
        args.sl = 1.5
        args.risk = 1.5
        args.session = [16, 22]
        args.max_trades = 1
        args.open_hour = 16
        args.or_minutes = 30
    return args


def show_today(data, signals, args) -> None:
    """En güncel günün sinyal durumunu ve botun kararını yazdır."""
    last_day = data.index.normalize()[-1]
    mask = data.index.normalize() == last_day
    day_sig = signals[mask]
    day_data = data[mask]
    cur = int(day_sig.iloc[-1]) if len(day_sig) else 0
    yon = {1: "🟢 LONG (AL)", -1: "🔴 SHORT (SAT)", 0: "⚪ BEKLE / pozisyon yok"}[cur]

    print("\n" + "═" * 50)
    print(f"  BUGÜNKÜ KARAR — {last_day.date()}")
    print("═" * 50)
    print(f"  Son bar      : {data.index[-1]}  fiyat {data['close'].iloc[-1]:.2f}")
    print(f"  Botun kararı : {yon}")
    # İlk sinyalin oluştuğu an
    nz = day_sig[day_sig != 0]
    if len(nz):
        first = nz.index[0]
        d0 = int(nz.iloc[0])
        print(f"  Sinyal saati : {first.strftime('%H:%M')} ({'yukarı kırılım' if d0==1 else 'aşağı kırılım'})")
    if args.strategy == "orb":
        from src.strategy import OpeningRangeBreakout  # OR seviyelerini göster
        open_min = args.open_hour * 60
        mins = day_data.index.hour * 60 + day_data.index.minute
        orw = day_data[(mins >= open_min) & (mins < open_min + args.or_minutes)]
        if len(orw):
            print(f"  Açılış aralığı: {orw['low'].min():.2f} – {orw['high'].max():.2f} "
                  f"(saat {args.open_hour}:00 sonrası {args.or_minutes}dk)")
    print("═" * 50)
    print("  Not: Bu eğitim/araştırma çıktısıdır; yatırım tavsiyesi değildir.")


def main():
    args = parse_args()
    args = apply_preset(args)
    if args.preset:
        print(f"[preset: {args.preset}]")
    print(f"Veri yükleniyor: {args.data}")
    data = load_mt5_csv(args.data)
    yrs = (data.index[-1] - data.index[0]).days / 365.25
    print(f"  {len(data)} bar | {data.index[0]} → {data.index[-1]} ({yrs:.1f} yıl)")
    print(f"  Strateji: {args.strategy} | günlük kilit: +{args.target}% ({args.lock}) | "
          f"saat {args.session[0]}-{args.session[1]}\n")

    strategy = build_strategy(args)
    signals = strategy.generate_signals(data)

    if args.today:
        show_today(data, signals, args)
        return

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

    if args.monthly:
        from src.metrics import format_monthly
        print("\n" + format_monthly(result.equity, 10.0))

    if args.export:
        result.trades.to_csv(f"{args.export}_trades.csv", index=False)
        result.equity.to_csv(f"{args.export}_equity.csv")
        result.days.to_csv(f"{args.export}_days.csv")
        print(f"\n  Dışa aktarıldı: {args.export}_trades.csv / _equity.csv / _days.csv "
              f"({len(result.trades)} işlem)")

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
