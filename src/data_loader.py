"""MT5'ten indirilen fiyat verisini okuyan yükleyici.

MetaTrader 5 verisini birkaç farklı formatta dışa aktarabilir:

1. Terminalden "Export Bars" (Sembol > Sağ tık > Save / F2):
   Tab ile ayrılmış, başlıklar şöyle olur:
       <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
       2024.01.02  00:00:00  16543.5  16550.0  16540.0  16548.0  1234  0  2

2. Python `MetaTrader5.copy_rates_*` çıktısı (pandas.to_csv ile):
       time,open,high,low,close,tick_volume,spread,real_volume
       2024-01-02 00:00:00,16543.5,...

Bu yükleyici her iki biçimi de (ve genel OHLCV CSV'lerini) otomatik tanır
ve standart bir DataFrame döndürür:
    index = datetime (tz-naive), kolonlar = open, high, low, close, volume
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

# Olası kolon adlarını standart adlara eşle
_COLUMN_ALIASES = {
    "open": "open",
    "<open>": "open",
    "high": "high",
    "<high>": "high",
    "low": "low",
    "<low>": "low",
    "close": "close",
    "<close>": "close",
    "volume": "volume",
    "<vol>": "volume",
    "real_volume": "volume",
    "tickvol": "tick_volume",
    "<tickvol>": "tick_volume",
    "tick_volume": "tick_volume",
    "spread": "spread",
    "<spread>": "spread",
}


def _detect_separator(sample: str) -> str:
    """İlk satıra bakarak ayırıcıyı (tab / virgül / noktalı virgül) tahmin et."""
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    if "\t" in first_line:
        return "\t"
    if ";" in first_line and "," not in first_line:
        return ";"
    return ","


def load_mt5_csv(path: str | Path) -> pd.DataFrame:
    """MT5 export dosyasını standart OHLCV DataFrame'e dönüştür.

    Returns:
        DatetimeIndex'li, [open, high, low, close, volume] kolonlu DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")

    text = path.read_text(encoding="utf-8-sig")  # BOM varsa temizle
    sep = _detect_separator(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")

    # Kolon adlarını normalize et (küçük harf, boşlukları temizle)
    df.columns = [str(c).strip().lower() for c in df.columns]

    df = _build_datetime_index(df)
    df = _rename_and_select(df)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def _build_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Tarih/saat kolonlarından bir DatetimeIndex kur."""
    cols = set(df.columns)

    # Format 1: ayrı <date> ve <time>
    date_col = next((c for c in ("<date>", "date") if c in cols), None)
    time_col = next((c for c in ("<time>", "time") if c in cols), None)

    if date_col and time_col and time_col != date_col:
        # "time" hem tarih hem ayrı saat kolonu olabilir; ayrı saat ise birleştir
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        idx = pd.to_datetime(combined, errors="coerce", format="mixed")
    elif "time" in cols:
        # Format 2: tek "time" kolonu (datetime ya da epoch saniye)
        time_series = df["time"]
        if pd.api.types.is_numeric_dtype(time_series):
            idx = pd.to_datetime(time_series, unit="s", errors="coerce")
        else:
            idx = pd.to_datetime(time_series, errors="coerce", format="mixed")
    elif date_col:
        idx = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    else:
        raise ValueError(
            "Tarih kolonu bulunamadı. Beklenen kolonlar: <DATE>/<TIME> ya da 'time'. "
            f"Görülen kolonlar: {list(df.columns)}"
        )

    if idx.isna().all():
        raise ValueError("Tarih kolonu ayrıştırılamadı; format beklenmedik.")

    df = df.copy()
    df.index = idx
    df.index.name = "datetime"
    return df


def _rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Kolonları standart adlara çevir ve OHLCV'yi seç."""
    renamed = {}
    for col in df.columns:
        if col in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[col]
    df = df.rename(columns=renamed)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Zorunlu OHLC kolonları eksik: {missing}. Mevcut: {list(df.columns)}"
        )

    # MT5'te <VOL> (real volume) endekslerde genelde 0'dır; asıl bilgi <TICKVOL>'dedir.
    # volume yoksa ya da tamamen sıfırsa tick_volume'a düş, o da yoksa 0 koy.
    has_tick = "tick_volume" in df.columns
    vol_missing = "volume" not in df.columns
    vol_all_zero = (not vol_missing) and pd.to_numeric(df["volume"], errors="coerce").fillna(0).eq(0).all()
    if vol_missing or vol_all_zero:
        df["volume"] = df["tick_volume"] if has_tick else 0

    # Sayısala çevir
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["open", "high", "low", "close", "volume"]]
