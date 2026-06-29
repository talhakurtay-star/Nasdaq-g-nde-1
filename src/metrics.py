"""Backtest performans metrikleri."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bar aralığından yıllık periyot sayısını tahmin et (yıllıklaştırma için)."""
    if len(index) < 2:
        return 252.0
    median_delta = pd.Series(index).diff().median()
    if pd.isna(median_delta) or median_delta.total_seconds() <= 0:
        return 252.0
    seconds = median_delta.total_seconds()
    # Yaklaşık ticaret günü = 6.5 saat; basitlik için takvim bazlı yıllıklaştırma
    bars_per_day = 86400 / seconds
    return bars_per_day * 252.0 if seconds < 86400 else 252.0 * (86400 / seconds)


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Bir equity eğrisi ve işlem listesinden özet metrikler üret."""
    if equity.empty:
        return {}

    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    periods_per_year = _infer_periods_per_year(equity.index)
    n = len(returns)
    if n > 0 and equity.iloc[0] > 0:
        years = n / periods_per_year
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        cagr = 0.0

    vol = returns.std()
    sharpe = (returns.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0

    downside = returns[returns < 0].std()
    sortino = (returns.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    # İşlem istatistikleri
    if not trades.empty and "pnl" in trades:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / len(trades) if len(trades) else 0.0
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = trades["pnl"].mean()
    else:
        win_rate = profit_factor = avg_trade = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "num_trades": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade,
        "final_equity": float(equity.iloc[-1]),
    }


def format_metrics(metrics: dict) -> str:
    """Metrikleri okunabilir bir tabloya çevir."""
    if not metrics:
        return "Metrik üretilemedi (veri yok)."
    lines = [
        "─" * 40,
        "  BACKTEST SONUÇLARI",
        "─" * 40,
        f"  Toplam getiri      : {metrics['total_return'] * 100:>10.2f} %",
        f"  CAGR               : {metrics['cagr'] * 100:>10.2f} %",
        f"  Sharpe oranı       : {metrics['sharpe']:>10.2f}",
        f"  Sortino oranı      : {metrics['sortino']:>10.2f}",
        f"  Max düşüş (DD)     : {metrics['max_drawdown'] * 100:>10.2f} %",
        f"  İşlem sayısı       : {metrics['num_trades']:>10d}",
        f"  Kazanma oranı      : {metrics['win_rate'] * 100:>10.2f} %",
        f"  Kâr faktörü        : {metrics['profit_factor']:>10.2f}",
        f"  Ort. işlem K/Z     : {metrics['avg_trade_pnl']:>10.2f}",
        f"  Son equity         : {metrics['final_equity']:>10.2f}",
        "─" * 40,
    ]
    return "\n".join(lines)


def format_session_summary(days, metrics: dict) -> str:
    """Prop-firm seans metriklerini okunabilir tabloya çevir."""
    if "total_days" not in metrics:
        return ""
    breach = metrics["firm_daily_breach"]
    breach_txt = "⚠️  İHLAL VAR!" if breach else "✓ ihlal yok"
    lines = [
        "  GÜNLÜK SEANS ÖZETİ",
        "─" * 40,
        f"  Toplam işlem günü  : {metrics['total_days']:>10d}",
        f"  Hedefe ulaşan gün  : {metrics['target_days']:>10d}  (%{metrics['target_hit_rate'] * 100:.1f})",
        f"  Stop olan gün      : {metrics['stop_days']:>10d}",
        f"  Nötr gün           : {metrics['neutral_days']:>10d}",
        f"  Ort. günlük getiri : {metrics['avg_daily_return_pct']:>10.3f} %",
        f"  En iyi gün         : {metrics['best_day_pct']:>10.3f} %",
        f"  En kötü gün        : {metrics['worst_day_pct']:>10.3f} %",
        f"  Firma günlük DD    : {breach_txt:>15}",
        "─" * 40,
    ]
    return "\n".join(lines)
