"""FundingPips (ve benzeri prop-firma) EVALUATION geçme simülatörü.

Hedef artık "günde %0.44 sonsuza kadar" DEĞİL — bir kez kâr hedefine (örn. +%8)
ulaşıp DD ihlali yapmamak. Bu bir VARYANS problemidir: küçük edge + iyi risk
yönetimi + biraz şansla geçilebilir.

Anahtar fikir (kullanıcının 3-katman mimarisi, Katman 3):
  İç günlük stop'u firma limitinin YARISINA koy (örn. %2.5 < firma %5).
  Böylece motor günlük DD limitini YAPISAL olarak asla ihlal edemez (tampon).

Simülasyon: equity eğrisinde kayan pencerelerle "ulaştı mı / patladı mı" sayar.

Kullanım:
    python tools/fundingpips.py --data data/NAS100_M5.csv
    python tools/fundingpips.py --target 8 --daily 5 --max 10 --horizon 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mt5_csv
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig
from src.strategy import OpeningRangeBreakout


def daily_returns(d, risk_per_trade, daily_internal_stop, firm_daily, lev):
    """ORB'yi FundingPips risk konfigiyle çalıştır, günlük getiri serisi döndür.
    İç stop firma limitinin altında → motor günlük limiti asla ihlal etmez."""
    s = OpeningRangeBreakout(open_hour=16, or_minutes=30).generate_signals(d)
    risk = RiskParams(
        daily_target_pct=daily_internal_stop,      # iç stopa eşit hedef (simetrik günlük tavan)
        daily_stop_pct=daily_internal_stop,        # İÇ stop (firma limitinin yarısı)
        firm_daily_dd_pct=firm_daily, monthly_dd_pct=99,
        profit_lock_mode="breakeven", leverage=lev,
        stop_loss_pct=1.5, risk_per_trade_pct=risk_per_trade,
        session_start_hour=16, session_end_hour=22, max_trades_per_day=1,
    )
    res = SessionBacktester(SessionConfig(10000, 0.0002, 0.0001), risk).run(d, s)
    return res.days["return_pct"].to_numpy() / 100, res.metrics


def simulate(daily, target, max_dd, horizon, step=3):
    """Kayan pencere eval simülasyonu. Günlük DD iç stopla garantili → sadece
    +target'a ulaşma vs -max_dd (toplam) patlaması kontrol edilir."""
    n = len(daily)
    p = f = t = 0
    for s in range(0, max(1, n - horizon), step):
        eq = 1.0
        done = False
        for k in range(s, min(s + horizon, n)):
            eq *= (1 + daily[k])
            if eq <= 1 - max_dd:
                f += 1; done = True; break
            if eq >= 1 + target:
                p += 1; done = True; break
        if not done:
            t += 1
    tot = p + f + t
    return (p / tot * 100, f / tot * 100, t / tot * 100) if tot else (0, 0, 0)


def main():
    ap = argparse.ArgumentParser(description="FundingPips eval geçme simülatörü")
    ap.add_argument("--data", default="data/NAS100_M5.csv")
    ap.add_argument("--target", type=float, default=8.0, help="Kâr hedefi %%")
    ap.add_argument("--daily", type=float, default=5.0, help="Firma günlük DD limiti %%")
    ap.add_argument("--max", type=float, default=10.0, help="Firma toplam DD limiti %%")
    ap.add_argument("--horizon", type=int, default=200, help="Eval süresi (işlem günü)")
    args = ap.parse_args()

    d = load_mt5_csv(args.data)
    internal = args.daily / 2     # iç stop = firma limitinin yarısı (tampon)
    print(f"FundingPips eval simülasyonu — {Path(args.data).stem}")
    print(f"  Hedef +{args.target}% | firma DD: günlük {args.daily}% / toplam {args.max}% "
          f"| iç günlük stop {internal}% (tampon)")
    print(f"  Süre: {args.horizon} işlem günü\n")
    print(f"{'risk/işlem':>11}{'ort gün%':>10}{'GEÇTİ%':>9}{'İHLAL%':>9}{'süre doldu%':>12}")
    print("─" * 51)
    for rpt in [1.0, 2.0, 3.0, 5.0, 8.0]:
        daily, m = daily_returns(d, rpt, internal, args.daily, lev=10.0)
        p, f, t = simulate(daily, args.target / 100, args.max / 100, args.horizon)
        print(f"{rpt:>9.1f}% {daily.mean()*100:>10.4f}{p:>9.0f}{f:>9.0f}{t:>12.0f}")
    print("\n  Not: İç stop firma limitinin yarısında → günlük DD ihlali yapısal olarak engelli.")
    print("  İHLAL = toplam %{:.0f} DD'ye değme. GEÇTİ = hedefe ulaşma.".format(args.max))
    print("  Gerçek: tek-deneme varyans oyunu; %100 garanti yok, eval ücreti riski var.")


if __name__ == "__main__":
    main()
