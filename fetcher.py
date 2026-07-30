"""データ取得層。

yfinance は非公式スクレイピングでいつ壊れてもおかしくないため、呼び出し側は
この層の関数しか触らない。将来 立花証券API / Finnhub に差し替える時はここだけ直す。

対策:
  - ローカルキャッシュ（既定6時間）。レート制限(429)と無駄な再取得を防ぐ
  - 空DataFrameを「エラー」として明示的に扱う（yfinanceはブロック時も空を返すため、
    データがない銘柄とブロックの区別がつかない = 最大の落とし穴）
  - 複数銘柄取得時はバッチ間にスリープ
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_TTL_HOURS = 6
FETCH_SLEEP_SEC = 1.0        # 銘柄間の待機。429回避
_last_fetch = 0.0


class DataUnavailable(RuntimeError):
    """データが取得できなかった。銘柄コード誤りかレート制限。"""


def market_of(symbol: str) -> str:
    """銘柄コードから市場を判定。日本株は末尾 .T が必須。"""
    return "JP" if symbol.upper().endswith(".T") else "US"


def _cache_path(symbol: str, period: str) -> Path:
    safe = symbol.replace("^", "IDX_").replace("=", "_").replace(".", "_")
    return CACHE_DIR / f"{safe}__{period}.csv"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age > timedelta(hours=CACHE_TTL_HOURS):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _throttle() -> None:
    global _last_fetch
    wait = FETCH_SLEEP_SEC - (time.monotonic() - _last_fetch)
    if wait > 0:
        time.sleep(wait)
    _last_fetch = time.monotonic()


def get_prices(symbol: str, period: str = "3y", use_cache: bool = True) -> pd.DataFrame:
    """日足の四本値+出来高を取得。indexはtz-naiveの日付に正規化する。

    tz正規化は必須。米国株(America/New_York)と日本株(Asia/Tokyo)を
    tz-awareのまま突き合わせると比較が壊れる。
    """
    import yfinance as yf

    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(symbol, period)

    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    _throttle()
    df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)

    if df.empty:
        # 空が返る原因は「存在しない銘柄」か「レート制限でブロック」。区別できない。
        stale = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
        if stale is not None and not stale.empty:
            print(f"  [警告] {symbol} の取得に失敗。期限切れキャッシュで代替します。")
            return stale
        raise DataUnavailable(
            f"{symbol} のデータが空です。銘柄コードの誤り"
            f"（日本株は末尾に .T が必要）か、レート制限の可能性があります。"
        )

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].dropna(subset=["Close"])

    df.to_csv(path)
    return df


FX_FALLBACK = 150.0


def get_usdjpy_series(period: str = "3y") -> Optional[pd.DataFrame]:
    """ドル円の時系列。過去日の判定では当時のレートを使う必要がある
    （現在のレートを過去に当てるとバックテスト結果が歪む）。"""
    for sym in ("USDJPY=X", "JPY=X"):
        try:
            df = get_prices(sym, period=period)
            if not df.empty and 50 < float(df["Close"].iloc[-1]) < 500:
                return df
        except Exception:
            continue
    return None


def get_usdjpy(as_of: Optional[pd.Timestamp] = None, period: str = "3y") -> float:
    """ドル円レート。as_of を渡すとその日以前の最終値を返す。"""
    df = get_usdjpy_series(period)
    if df is None:
        print(f"  [警告] ドル円が取得できません。{FX_FALLBACK} で仮計算します。")
        return FX_FALLBACK

    if as_of is not None:
        sub = df.loc[:as_of]
        if sub.empty:
            print(f"  [警告] {as_of.date()} 以前のドル円がありません。"
                  f"最古の値で代替します。")
            return float(df["Close"].iloc[0])
        return float(sub["Close"].iloc[-1])

    return float(df["Close"].iloc[-1])


def get_total_return_prices(symbol: str, period: str = "max",
                            use_cache: bool = True) -> pd.Series:
    """配当・分割込みの株価（トータルリターン）。

    長期保有の比較では配当を無視すると持ち続ける側を過小評価する。
    10年で年2%の配当なら約22%の差になり、比較の結論が変わりうる。
    auto_adjust=True で分割と配当の両方を調整した終値を得る。
    """
    import yfinance as yf

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"TR_{symbol.replace('^','IDX_').replace('=','_').replace('.','_')}__{period}.csv"

    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached.iloc[:, 0]

    _throttle()
    df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if df.empty or "Close" not in df.columns:
        raise DataUnavailable(f"{symbol} の配当込み株価が取得できません")

    s = df["Close"].dropna()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    s.name = symbol
    s.to_frame().to_csv(path)
    return s


def get_fundamentals(symbol: str) -> Optional[dict]:
    """財務データ（増収増益の判定用）。

    注意: これは「現在時点」のデータで、過去のある日に何が判明していたかは取れない。
    したがってバックテストで使うと未来参照（look-ahead bias）になる。
    買い判定には使わず、短期/長期の分類にのみ使用する。
    """
    import yfinance as yf

    try:
        _throttle()
        t = yf.Ticker(symbol)
        fin = t.quarterly_financials
        if fin is None or fin.empty:
            return None

        def trend(label: str) -> Optional[bool]:
            if label not in fin.index:
                return None
            # 列は新しい順。直近2四半期を比較
            vals = fin.loc[label].dropna()
            return bool(vals.iloc[0] > vals.iloc[1]) if len(vals) >= 2 else None

        rev = trend("Total Revenue")
        inc = trend("Net Income")
        if rev is None and inc is None:
            return None
        return {
            "growing": bool(rev) and bool(inc),
            "revenue_up": rev,
            "income_up": inc,
        }
    except Exception as e:
        print(f"  [警告] {symbol} の財務データ取得に失敗: {type(e).__name__}")
        return None


def get_news(symbol: str, limit: int = 15) -> list[dict]:
    """直近ニュース見出し。AI層に渡す。取得失敗は空リストで返し判定は止めない。"""
    import yfinance as yf

    try:
        _throttle()
        raw = yf.Ticker(symbol).news or []
    except Exception as e:
        print(f"  [警告] {symbol} のニュース取得に失敗: {type(e).__name__}")
        return []

    items = []
    for entry in raw[:limit]:
        # yfinance のニュース構造はバージョンで変わるため両形式に対応
        node = entry.get("content", entry) if isinstance(entry, dict) else {}
        title = node.get("title") or entry.get("title")
        if not title:
            continue
        publisher = ""
        provider = node.get("provider")
        if isinstance(provider, dict):
            publisher = provider.get("displayName", "")
        publisher = publisher or entry.get("publisher", "")
        items.append(
            {
                "title": title,
                "publisher": publisher,
                "published": str(node.get("pubDate") or entry.get("providerPublishTime", "")),
            }
        )
    return items
