"""Nasdaq backtest botu - ana giriş noktası.

Kullanım:
    python main.py --data data/NDX_D1.csv
    python main.py --data data/NDX_D1.csv --fast 10 --slow 30 --ma sma --no-rsi
    python main.py --data data/NDX_D1.csv --plot equity.png

MT5'ten dosya indirip data/ klasörüne koyduktan sonra --data ile yolunu verin.
Örnek veriyle denemek için önce:
    python tools/generate_sample_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.backtester import Backtester, BacktestConfig
from src.data_loader import load_mt5_csv
from src.strategy import MACrossStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MT5 verisiyle Nasdaq backtest botu")
    p.add_argument("--data", required=True, help="MT5 CSV dosya yolu")
    p.add_argument("--fast", type=int, default=20, help="Hızlı MA periyodu")
    p.add_argument("--slow", type=int, default=50, help="Yavaş MA periyodu")
    p.add_argument("--ma", choices=["sma", "ema"], default="ema", help="MA tipi")
    p.add_argument("--no-rsi", action="store_true", help="RSI filtresini kapat")
    p.add_argument("--rsi-ob", type=float, default=70.0, help="RSI aşırı alım seviyesi")
    p.add_argument("--cash", type=float, default=10_000.0, help="Başlangıç sermayesi")
    p.add_argument("--commission", type=float, default=0.0005, help="Komisyon oranı")
    p.add_argument("--slippage", type=float, default=0.0002, help="Slipaj oranı")
    p.add_argument("--plot", default=None, help="Equity eğrisini bu PNG'ye kaydet")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Veri yükleniyor: {args.data}")
    data = load_mt5_csv(args.data)
    print(f"  {len(data)} bar | {data.index[0].date()} → {data.index[-1].date()}")

    strategy = MACrossStrategy(
        fast=args.fast,
        slow=args.slow,
        ma_type=args.ma,
        use_rsi=not args.no_rsi,
        rsi_overbought=args.rsi_ob,
    )
    signals = strategy.generate_signals(data)

    config = BacktestConfig(
        initial_cash=args.cash,
        commission=args.commission,
        slippage=args.slippage,
    )
    result = Backtester(config).run(data, signals)

    print()
    print(result.summary())

    # Buy & hold karşılaştırması
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
