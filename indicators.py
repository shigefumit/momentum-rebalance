"""テクニカル指標の計算。

すべてベクトル化した純粋関数。DataFrame を受けて指標列を足した DataFrame を返す。
Phase 1 のバックテストでも同じ関数を使うため、副作用と未来参照を一切入れない。

平滑化は Wilder 方式（ewm alpha=1/n, adjust=False）に統一。一般的なチャートツール
（TradingView 等）の RSI / ATR / ADX と一致させるため。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder平滑化（RSI・ATR・ADXの標準）。"""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    return _wilder(true_range(df), period)


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    # avg_loss=0 は「下落なし」= RSI 100。ゼロ除算を避けて明示的に扱う。
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0, 100.0)


def adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.DataFrame:
    """ADX と ±DI。トレンドの強さを測る。ADX<22 はもみ合いとして除外する。"""
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    atr_ = _wilder(true_range(df), period).replace(0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame(
        {"adx": _wilder(dx.fillna(0), period), "plus_di": plus_di, "minus_di": minus_di}
    )


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """価格データに全指標を付与する。この関数の出力がルール層の唯一の入力。"""
    if df.empty:
        return df
    out = df.copy()

    out["sma200"] = out["Close"].rolling(config.SMA_LONG).mean()
    out["sma50"] = out["Close"].rolling(config.SMA_MID).mean()
    out["ema20"] = out["Close"].ewm(span=config.EMA_PULLBACK, adjust=False).mean()

    out["rsi"] = rsi(out["Close"], config.RSI_PERIOD)
    out["rsi_fast"] = rsi(out["Close"], config.RSI_FAST)
    out["atr"] = atr(out, config.ATR_PERIOD)
    out["atr_pct"] = out["atr"] / out["Close"]

    adx_df = adx(out, config.ADX_PERIOD)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]

    out["vol_ma"] = out["Volume"].rolling(config.VOL_MA_PERIOD).mean()
    out["vol_ratio"] = out["Volume"] / out["vol_ma"]

    # 構造ストップ用の直近安値 / ブレイクアウト判定用の直近高値。
    # shift(1) で当日を除外し、未来参照を防ぐ。
    out["swing_low"] = out["Low"].rolling(config.SWING_LOW_LOOKBACK).min().shift(1)
    out["high20"] = out["High"].rolling(20).max().shift(1)
    out["trail_high"] = out["High"].rolling(config.TRAIL_LOOKBACK).max()

    # 上値抵抗（60日高値）。リスクリワード判定の基準。
    # ここを「R の定数倍」にすると R:R 判定が自己循環して無意味になるため、
    # 必ずチャート上の実際の抵抗を使う。
    out["high60"] = out["High"].rolling(60).max().shift(1)

    # ボラティリティ収縮（VCP）: 直近20日の値幅 ÷ 過去60日平均の値幅
    daily_range_pct = (out["High"] - out["Low"]) / out["Close"]
    out["range20"] = daily_range_pct.rolling(20).mean()
    out["range60"] = daily_range_pct.rolling(60).mean()
    out["squeeze"] = out["range20"] / out["range60"]

    # 200日線からの乖離率（短期スコアで「行き過ぎ」を測る）
    out["dist_sma200"] = (out["Close"] - out["sma200"]) / out["sma200"]

    # 実現ボラティリティ（年率）。日経VIの代替として指数側で使う。
    out["realized_vol"] = out["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # オーバーナイトギャップ。スイングは寄り付きで損切り値を飛び越えられるリスクがあり、
    # それが「1トレード1%」を守っていても実損が超過する主要因になる。
    # 過去1年の下方ギャップの5パーセンタイル（=悪い方から5%）を実測値として持つ。
    # 固定の想定値（例: 3%）だと銘柄のボラティリティと噛み合わず警告が無意味になる。
    prev_close = out["Close"].shift(1)
    out["overnight_gap"] = (out["Open"] - prev_close) / prev_close
    out["gap_down_p95"] = out["overnight_gap"].rolling(252, min_periods=120).quantile(0.05)

    return out
