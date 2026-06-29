"""Nasdaq prop-firm backtest botu - ana giriş noktası.

Çalışma mantığı (seans modu):
  - Bot her gün küçük bir hedefi (varsayılan %0.44) yakalamaya çalışır.
  - Hedefe ulaşınca O GÜN kilitlenir, başka işlem yapmaz.
  - Günlük kayıp stop'una (%0.44) değince de o gün kilitlenir.
  - Aylık ~22 işlem günü → bileşik ~%10 (ama hedefi kovalamayız, günlük çalışırız).

Kullanım:
    # Önce örnek gün-içi veri üret:
    python tools/generate_sample_data.py --tf M5

    # Backtest (seans modu varsayılan):
    python main.py --data data/SAMPLE_NDX_M5.csv
    python main.py --data data/NDX_M5.csv --target 0.44 --stop 0.44 --plot eq.png

    # Klasik (gün-içi kilitsiz) mod:
    python main.py --data data/NDX_D1.csv --mode simple
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.backtester import Backtester, BacktestConfig
from src.data_loader import load_mt5_csv
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig
from src.strategy import MACrossStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MT5 verisiyle Nasdaq prop-firm backtest botu")
    p.add_argument("--data", required=True, help="MT5 CSV dosya yolu")
    p.add_argument("--mode", choices=["session", "simple"], default="session",
                   help="session: günlük kilit + risk kuralları; simple: klasik")
    # Strateji
    p.add_argument("--fast", type=int, default=20, help="Hızlı MA periyodu")
    p.add_argument("--slow", type=int, default=50, help="Yavaş MA periyodu")
    p.add_argument("--ma", choices=["sma", "ema"], default="ema", help="MA tipi")
    p.add_argument("--no-rsi", action="store_true", help="RSI filtresini kapat")
    p.add_argument("--rsi-ob", type=float, default=70.0, help="RSI aşırı alım seviyesi")
    # Risk / seans
    p.add_argument("--target", type=float, default=0.44, help="Günlük kâr hedefi %%")
    p.add_argument("--stop", type=float, default=0.44, help="Günlük kayıp stop'u %%")
    p.add_argument("--monthly-dd", type=float, default=10.0, help="Aylık max DD %% (firma)")
    p.add_argument("--leverage", type=float, default=1.0, help="Kaldıraç (1.0 = yok)")
    # Hesap
    p.add_argument("--cash", type=float, default=10_000.0, help="Başlangıç sermayesi")
    p.add_argument("--commission", type=float, default=0.0005, help="Komisyon oranı")
    p.add_argument("--slippage", type=float, default=0.0002, help="Slipaj oranı")
    p.add_argument("--plot", default=None, help="Equity eğrisini bu PNG'ye kaydet")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Veri yükleniyor: {args.data}")
    data = load_mt5_csv(args.data)
    print(f"  {len(data)} bar | {data.index[0]} → {data.index[-1]}")

    strategy = MACrossStrategy(
        fast=args.fast, slow=args.slow, ma_type=args.ma,
        use_rsi=not args.no_rsi, rsi_overbought=args.rsi_ob,
    )
    signals = strategy.generate_signals(data)

    if args.mode == "session":
        risk = RiskParams(
            daily_target_pct=args.target,
            daily_stop_pct=args.stop,
            monthly_dd_pct=args.monthly_dd,
            leverage=args.leverage,
        )
        config = SessionConfig(
            initial_cash=args.cash, commission=args.commission, slippage=args.slippage,
        )
        result = SessionBacktester(config, risk).run(data, signals)
    else:
        config = BacktestConfig(
            initial_cash=args.cash, commission=args.commission, slippage=args.slippage,
        )
        result = Backtester(config).run(data, signals)

    print()
    print(result.summary())

    bh_return = (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100
    print(f"\n  (Kıyas) Al & Tut getirisi: {bh_return:.2f} %")

    if args.plot:
        save_plot(result, data, Path(args.plot))


def save_plot(result, data, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib kurulu değil; grafik atlanıyor.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(result.equity.index, result.equity.values, label="Strateji equity", color="#1f77b4")
    bh = data["close"] / data["close"].iloc[0] * result.equity.iloc[0]
    ax1.plot(bh.index, bh.values, label="Al & Tut", color="gray", alpha=0.6, linestyle="--")
    ax1.set_title("Equity Eğrisi")
    ax1.legend()
    ax1.grid(alpha=0.3)

    running_max = result.equity.cummax()
    dd = (result.equity / running_max - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.set_title("Düşüş (Drawdown) %")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nGrafik kaydedildi: {path}")


if __name__ == "__main__":
    main()
