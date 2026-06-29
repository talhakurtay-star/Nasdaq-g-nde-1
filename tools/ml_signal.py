"""ML tabanlı yön sinyali — eğit, OOS'ta doğrula, sembol-arası test et.

⛔ ANTİ-SIZINTI / ANTİ-OVERFIT DİSİPLİNİ:
  - Zaman-sıralı bölme (karıştırma YOK): eğitim ilk %70, test son %30.
  - Model SADECE eğitimde eğitilir; test'e BİR KEZ bakılır.
  - Özellikler nedensel (ml_features), hedef geleceğe bakar ama sadece etikette.
  - Sinyal bar t'de üretilir, motor t+1 açılışında girer (ekstra shift).
  - Gerçekçi maliyet + 0.44 günlük kilit motoru ile değerlendirilir.

Kullanım:
    python tools/ml_signal.py --data data/NAS100_M5.csv
    python tools/ml_signal.py --data data/NAS100_M5.csv --horizon 12 --session 14 22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import HistGradientBoostingClassifier

from src.data_loader import load_mt5_csv
from src.ml_features import FEATURE_COLS, build_features, make_target
from src.risk import RiskParams
from src.session_backtester import SessionBacktester, SessionConfig

CROSS = ["NVDA_M5", "US500_M5", "US30_M5", "GER40_M5", "XAUUSD_M5"]


def make_signals(proba_up: pd.Series, margin: float) -> pd.Series:
    """P(up)'tan yapışkan sinyal: güçlü tahminle yön değiştir, belirsizde tut."""
    n = len(proba_up)
    p = proba_up.to_numpy()
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    for i in range(n):
        if not np.isnan(p[i]):
            if p[i] >= 0.5 + margin:
                pos = 1
            elif p[i] <= 0.5 - margin:
                pos = -1
        out[i] = pos
    return pd.Series(out, index=proba_up.index)


def backtest(data, signals, sess, cfg):
    risk = RiskParams(daily_target_pct=0.44, daily_stop_pct=0.44, firm_daily_dd_pct=99,
                      monthly_dd_pct=99, profit_lock_mode="breakeven",
                      stop_loss_pct=1.5, risk_per_trade_pct=0.30, leverage=2.0,
                      session_start_hour=sess[0], session_end_hour=sess[1],
                      max_trades_per_day=0)
    return SessionBacktester(cfg, risk).run(data, signals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/NAS100_M5.csv")
    ap.add_argument("--horizon", type=int, default=12, help="Hedef: kaç bar sonraki yön")
    ap.add_argument("--deadband", type=float, default=0.05, help="Hedef gürültü bandı %%")
    ap.add_argument("--session", nargs=2, type=int, default=[14, 22])
    ap.add_argument("--split", type=float, default=0.70)
    args = ap.parse_args()

    cfg = SessionConfig(initial_cash=10000, commission=0.0002, slippage=0.0001)
    d = load_mt5_csv(args.data)
    print(f"Veri: {Path(args.data).stem} | {len(d)} bar | "
          f"{d.index[0].date()} → {d.index[-1].date()}")

    X = build_features(d)[FEATURE_COLS]
    y = make_target(d, horizon=args.horizon, deadband_pct=args.deadband)

    cut = int(len(d) * args.split)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr = y.iloc[:cut]

    # Eğitim: geçerli etiketli satırlar (NaN hedef = gürültü/son barlar atılır)
    fit_mask = ytr.notna()
    print(f"Eğitim örneği: {fit_mask.sum()} | özellik: {len(FEATURE_COLS)} | "
          f"yukarı oranı: {ytr[fit_mask].mean():.3f}")

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=200, random_state=42,
    )
    model.fit(Xtr[fit_mask], ytr[fit_mask].astype(int))

    # OOS sınıflandırma doğruluğu (sanity: >%50 mi?)
    yte = make_target(d, horizon=args.horizon, deadband_pct=args.deadband).iloc[cut:]
    pte = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index)
    acc_mask = yte.notna()
    acc = ((pte[acc_mask] > 0.5).astype(int) == yte[acc_mask].astype(int)).mean()
    print(f"OOS yön doğruluğu: {acc*100:.2f}%  (50%'nin anlamlı üstü mü?)")

    # Eğitim olasılıkları (eşik seçimi için)
    ptr = pd.Series(model.predict_proba(Xtr)[:, 1], index=Xtr.index)

    print(f"\n{'margin':>8}{'EĞİTİM%':>10}{'OOS%':>9}{'OOSwin%':>9}{'işlem':>7}")
    print("─" * 45)
    best = None
    for margin in [0.0, 0.02, 0.05, 0.08, 0.10]:
        sig_tr = make_signals(ptr, margin)
        mtr = backtest(d.iloc[:cut], sig_tr, args.session, cfg).metrics
        sig_te = make_signals(pte, margin)
        mte = backtest(d.iloc[cut:], sig_te, args.session, cfg).metrics
        flag = " ✓OOS+" if mte["total_return"] > 0 else ""
        print(f"{margin:>8.2f}{mtr['total_return']*100:>10.1f}{mte['total_return']*100:>9.1f}"
              f"{mte['win_rate']*100:>9.1f}{mte['num_trades']:>7}{flag}")
        if best is None or mtr["total_return"] > best[0]:
            best = (mtr["total_return"], margin, mte)

    # Eğitimde en iyi margin → OOS sonucu zaten yukarıda; özet + sembol-arası
    margin = best[1]
    oos = best[2]
    print(f"\nEğitimde seçilen margin={margin} → OOS getiri {oos['total_return']*100:+.1f}%")

    if oos["total_return"] > 0:
        print("\nOOS POZİTİF → SEMBOL-ARASI test (aynı model EĞİTİM sembolünde eğitildi,")
        print("diğer sembollerde TÜM veride sıfırdan eğitilip kendi OOS'unda test edilir):")
        for sym in CROSS:
            p = Path("data") / f"{sym}.csv"
            if not p.exists():
                continue
            ds = load_mt5_csv(p)
            Xs = build_features(ds)[FEATURE_COLS]
            ys = make_target(ds, args.horizon, args.deadband)
            cs = int(len(ds) * args.split)
            fm = ys.iloc[:cs].notna()
            mdl = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05,
                                                 l2_regularization=1.0, min_samples_leaf=200, random_state=42)
            mdl.fit(Xs.iloc[:cs][fm], ys.iloc[:cs][fm].astype(int))
            ps = pd.Series(mdl.predict_proba(Xs.iloc[cs:])[:, 1], index=Xs.iloc[cs:].index)
            ms = backtest(ds.iloc[cs:], make_signals(ps, margin), args.session, cfg).metrics
            print(f"    {sym:<12} OOS {ms['total_return']*100:+7.1f}%  win {ms['win_rate']*100:.1f}%  "
                  f"{'✓' if ms['total_return']>0 else '✗'}")
    else:
        print("\nOOS negatif → model gerçek yön avantajı taşımıyor (dürüst sonuç).")


if __name__ == "__main__":
    main()
