"""判定ルール層。数式のみ。AIは一切関与しない。

設計上の最重要点: すべての関数は「任意のバー位置 i」で評価できる純粋関数。
live判定は i=-1（最新バー）、バックテストは i をループするだけで同じ関数が使える。
最終行を決め打ちにすると Phase 1 で全部書き直しになるため、この規約は崩さない。

未来参照の防止: enrich() 側で swing_low / high20 / high60 に shift(1) をかけてある。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config


# ============================================================ 戻り値の型
@dataclass
class Regime:
    score: int                       # 0-30
    state: str                       # RISK_ON / CAUTION / RISK_OFF
    details: dict = field(default_factory=dict)


@dataclass
class Quality:
    score: int                       # 0-30
    checks: dict = field(default_factory=dict)


@dataclass
class Setup:
    score: int                       # 0-40
    name: str                        # 成立したセットアップ名（"" = 不成立）
    style: str                       # short / swing / long
    entry_method: str
    passed: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)


@dataclass
class TradePlan:
    ok: bool
    reason: str = ""
    entry: float = 0.0
    stop: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    r_value: float = 0.0             # 1R = entry - stop（現地通貨）
    rr: float = 0.0                  # 抵抗までの現実的なリスクリワード
    shares: int = 0
    position_value_jpy: float = 0.0
    position_pct: float = 0.0
    risk_jpy: float = 0.0
    resistance: Optional[float] = None
    trail_stop: float = 0.0
    notes: list = field(default_factory=list)


@dataclass
class Horizon:
    long_score: int
    short_score: int
    label: str
    detail: dict = field(default_factory=dict)


# ============================================================ 補助
def _row_asof(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
    """ts 以前の最後の行を返す。市場ごとに休日が違うためインデックス一致を前提にしない。"""
    if df.empty:
        return None
    sub = df.loc[:ts]
    return None if sub.empty else sub.iloc[-1]


def _ok(v) -> bool:
    """NaN / None を False に落とす。指標の助走期間中の誤判定を防ぐ。"""
    return bool(v) and not (isinstance(v, float) and math.isnan(v))


# ============================================================ L1 市場環境（0-30）
def regime_score(spx: pd.DataFrame, vix: pd.DataFrame, nikkei: pd.DataFrame,
                 as_of: pd.Timestamp, market: str) -> Regime:
    """市場全体の状態を判定。

    根拠: S&P500が200日線より上の時の年率ボラは13.81%、下では29.93%（約2.2倍）。
    VIXが自身の200日線を上抜けると1〜5営業日先行して危機入りを示す。
    """
    s = _row_asof(spx, as_of)
    v = _row_asof(vix, as_of)
    n = _row_asof(nikkei, as_of)

    if s is None or v is None:
        return Regime(0, "UNKNOWN", {"error": "指数データが取得できず判定不能"})

    spx_above = _ok(s["Close"] > s["sma200"])
    vix_level = float(v["Close"])
    vix_hot = _ok(v["Close"] > v["sma200"])
    nikkei_above = _ok(n["Close"] > n["sma200"]) if n is not None else None

    if not spx_above or vix_level >= config.VIX_CRISIS:
        state = "RISK_OFF"
    elif vix_level >= config.VIX_CALM or vix_hot:
        state = "CAUTION"
    else:
        state = "RISK_ON"

    # 日本株は日経平均も条件に加える（米国が良くても日本が崩れていれば格下げ）
    if market == "JP" and state == "RISK_ON" and nikkei_above is False:
        state = "CAUTION"

    score = {"RISK_ON": 30, "CAUTION": 15, "RISK_OFF": 0}[state]

    return Regime(score, state, {
        "S&P500": f"{s['Close']:,.0f}（200日線 {s['sma200']:,.0f} を{'上回る' if spx_above else '下回る'}）",
        "VIX": f"{vix_level:.2f}（{_vix_zone(vix_level)}）"
               + ("・自身の200日線を上抜け中" if vix_hot else ""),
        "日経平均": (f"{n['Close']:,.0f}（200日線 {n['sma200']:,.0f} を"
                     f"{'上回る' if nikkei_above else '下回る'}）") if n is not None else "取得不可",
    })


def _vix_zone(v: float) -> str:
    if v < 15:
        return "低ボラ"
    if v < 20:
        return "通常"
    if v < 30:
        return "警戒"
    return "危機"


# ============================================================ L2 銘柄品質（0-30）
def quality_score(df: pd.DataFrame, i: int, market: str) -> Quality:
    """価格・出来高のみで判定。財務は使わない（過去検証で未来参照になるため）。

    財務指標（増収増益等）は horizon_scores() の長期判定に回している。
    こうすることで買い判定は100%バックテスト可能になる。
    """
    r = df.iloc[i]
    min_vol = config.MIN_AVG_VOLUME[market]
    lo, hi = config.ATR_PCT_RANGE

    checks = {
        f"終値が200日線より上": (_ok(r["Close"] > r["sma200"]), 8),
        f"50日線 > 200日線": (_ok(r["sma50"] > r["sma200"]), 7),
        f"平均出来高 {min_vol:,} 株以上": (_ok(r["vol_ma"] >= min_vol), 5),
        f"ATR率が {lo:.1%}〜{hi:.0%}": (_ok(lo <= r["atr_pct"] <= hi), 5),
        f"終値が50日線より上": (_ok(r["Close"] > r["sma50"]), 5),
    }
    score = sum(pts for passed, pts in checks.values() if passed)
    return Quality(score, {k: {"ok": p, "pts": pts} for k, (p, pts) in checks.items()})


# ============================================================ L3 セットアップ（0-40）
def setup_score(df: pd.DataFrame, i: int, regime_state: str) -> Setup:
    """3つのセットアップを評価し、最も高得点のものを採用する。"""
    candidates = [
        _setup_pullback(df, i),
        _setup_breakout(df, i),
        _setup_mean_reversion(df, i, regime_state),
    ]
    hit = [c for c in candidates if c.score > 0]
    if hit:
        return max(hit, key=lambda c: c.score)
    # 不成立: 最も惜しかったものの不足条件を返す
    return min(candidates, key=lambda c: len(c.failed))


def _grade(base: int, bonuses: list[tuple[bool, int]]) -> int:
    return base + sum(pts for ok, pts in bonuses if ok)


def _setup_pullback(df: pd.DataFrame, i: int) -> Setup:
    """A: 20EMA押し目買い。確認済みトレンド内で勝率55〜65%、R:R 2:1〜4:1。"""
    r, prev = df.iloc[i], df.iloc[i - 1]
    lo, hi = config.RSI_PULLBACK_RANGE

    req = {
        "長期上昇トレンド（終値>200日線 かつ 50日線>200日線）":
            _ok(r["Close"] > r["sma200"]) and _ok(r["sma50"] > r["sma200"]),
        f"ADX ≥ {config.ADX_MIN}（もみ合いでない）":
            _ok(r["adx"] >= config.ADX_MIN),
        f"20EMAまで押した（乖離 ≤ {config.PULLBACK_MAX_ATR}×ATR）":
            _ok(abs(r["Close"] - r["ema20"]) <= config.PULLBACK_MAX_ATR * r["atr"]),
        f"RSI が {lo}〜{hi}（暴落でも過熱でもない）":
            _ok(lo <= r["rsi"] <= hi),
        "反転確認（終値 > 前日高値）":
            _ok(r["Close"] > prev["High"]),
        "健全な押し（出来高 < 20日平均）":
            _ok(r["Volume"] < r["vol_ma"]),
    }
    failed = [k for k, v in req.items() if not v]
    if failed:
        return Setup(0, "20EMA押し目買い", "swing", "pullback", req, failed)

    score = _grade(28, [
        (_ok(r["adx"] >= 30), 4),                              # トレンドが特に強い
        (_ok(r["vol_ratio"] <= 0.7), 4),                       # 売り枯れが明確
        (_ok(45 <= r["rsi"] <= 55), 4),                        # RSIが中央付近
    ])
    return Setup(score, "20EMA押し目買い", "swing", "pullback", req, [])


def _setup_breakout(df: pd.DataFrame, i: int) -> Setup:
    """B: ボラティリティ収縮ブレイクアウト（VCP型）。"""
    r = df.iloc[i]
    req = {
        "長期上昇トレンド（終値>200日線 かつ 50日線>200日線）":
            _ok(r["Close"] > r["sma200"]) and _ok(r["sma50"] > r["sma200"]),
        "ボラティリティ収縮（直近20日の値幅 < 過去60日平均）":
            _ok(r["squeeze"] < 1.0),
        "出来高が20日平均の1.5倍以上":
            _ok(r["vol_ratio"] >= 1.5),
        "20日高値を上抜け":
            _ok(r["Close"] > r["high20"]),
    }
    failed = [k for k, v in req.items() if not v]
    if failed:
        return Setup(0, "ボラ収縮ブレイクアウト", "swing", "breakout", req, failed)

    score = _grade(28, [
        (_ok(r["squeeze"] < 0.8), 4),                          # 収縮が強い
        (_ok(r["vol_ratio"] >= 2.0), 4),                       # 出来高が突出
        (_ok(r["adx"] >= config.ADX_MIN), 4),
    ])
    return Setup(score, "ボラ収縮ブレイクアウト", "swing", "breakout", req, [])


def _setup_mean_reversion(df: pd.DataFrame, i: int, regime_state: str) -> Setup:
    """C: 短期平均回帰（RSI-2型）。RISK_ON時のみ。保有2〜5日想定。"""
    r = df.iloc[i]
    req = {
        "市場環境が RISK_ON": regime_state == "RISK_ON",
        "長期上昇トレンド（終値 > 200日線）": _ok(r["Close"] > r["sma200"]),
        f"RSI(2) < {config.RSI_FAST_OVERSOLD}（極端な短期売られすぎ）":
            _ok(r["rsi_fast"] < config.RSI_FAST_OVERSOLD),
    }
    failed = [k for k, v in req.items() if not v]
    if failed:
        return Setup(0, "短期平均回帰(RSI-2)", "short", "dip", req, failed)

    score = _grade(24, [
        (_ok(r["Close"] > r["sma50"]), 4),
        (_ok(r["rsi_fast"] < 5), 4),
        (_ok(r["atr_pct"] >= 0.02), 4),
    ])
    return Setup(score, "短期平均回帰(RSI-2)", "short", "dip", req, [])


# ============================================================ 指値・損切り・利確・株数
def plan_trade(df: pd.DataFrame, i: int, setup: Setup, market: str,
               usdjpy: float, capital_jpy: Optional[float] = None) -> TradePlan:
    """capital_jpy を渡すとその金額を基準に株数を計算する。
    live は config.CAPITAL_JPY（固定）、バックテストはその時点の資産額を渡す。"""
    capital = config.CAPITAL_JPY if capital_jpy is None else capital_jpy
    r = df.iloc[i]
    atr = float(r["atr"])
    close = float(r["Close"])
    if not _ok(atr) or atr <= 0:
        return TradePlan(False, "ATRが計算できません（データ不足）")

    notes: list[str] = []

    # ---- エントリー指値。セットアップの性質で方式を変える
    if setup.entry_method == "pullback":
        entry = float(r["ema20"]) + config.ENTRY_PULLBACK_ATR * atr
        if entry > close:
            # すでに指値より上にある = 今日の押しは終わっている。成行相当に切替
            entry = close
            notes.append("株価が押し目指値より上にあるため、当日終値を基準にしています")
    elif setup.entry_method == "breakout":
        entry = close * (1 + config.ENTRY_BREAKOUT_MARGIN)
        notes.append("ブレイクアウト追随のため逆指値買い（この価格を上抜けたら約定）")
    else:  # dip
        entry = close * 0.995
        notes.append("平均回帰狙いのため、もう一段の押しを指値で待ちます")

    # ---- 損切り: 構造とATRを併用し、損失幅を [MIN_STOP_ATR, mult] × ATR に必ず収める
    #
    # 上限(mult×ATR)がないと直近安値が遠い時に損失幅が無制限に広がる。
    # 下限(MIN_STOP_ATR×ATR)がないと直近安値が偶然エントリー直下に来た時に
    # 損失幅が数セントになり、ノイズで即刈られる「機能しない損切り」ができてしまう
    # （同時に R:R が計算上爆発して 39:1 のような非現実的な値になる）。
    mult = config.ATR_MULT[setup.style]
    far_limit = entry - mult * atr                      # これより遠くしない
    near_limit = entry - config.MIN_STOP_ATR * atr       # これより近くしない

    struct_stop = float(r["swing_low"]) - config.STRUCT_STOP_ATR_BUFFER * atr \
        if _ok(r["swing_low"]) else -math.inf

    stop = max(struct_stop, far_limit)     # 損失幅の上限を適用
    stop = min(stop, near_limit)           # 損失幅の下限を適用

    if stop >= entry:
        return TradePlan(False, "損切りラインがエントリー価格以上になり成立しません")

    if struct_stop > near_limit:
        which = f"最小損切り幅 {config.MIN_STOP_ATR}×ATR（直近安値が近すぎるため）"
    elif struct_stop < far_limit:
        which = f"最大損切り幅 ATR×{mult}（直近安値が遠すぎるため）"
    else:
        which = "直近安値ベース"
    notes.append(f"損切りは{which}を採用")

    r_value = entry - stop

    # ---- 現実的な上値目標。ここを R の定数倍にすると R:R 判定が自己循環するため、
    #      必ずチャート上の抵抗（60日高値）を基準にする
    high60 = float(r["high60"]) if _ok(r.get("high60")) else float("nan")
    at_new_high = math.isnan(high60) or close >= high60
    if at_new_high:
        resistance = None
        tp2 = entry + config.TP2_R * r_value
        rr = config.TP2_R
        notes.append("上値に直近60日の抵抗がないため、+2.5R を最終目標にしています")
    else:
        resistance = high60
        rr = (resistance - entry) / r_value
        tp2 = min(entry + config.TP2_R * r_value, resistance * 0.998)
        notes.append(f"直近60日高値 {resistance:,.2f} を上値抵抗として R:R を算出")

    tp1 = entry + config.TP1_R * r_value

    if rr < config.MIN_RR:
        return TradePlan(
            False,
            f"リスクリワードが {rr:.2f} で最低基準 {config.MIN_RR} に届きません"
            f"（上値抵抗が近すぎる）",
            entry=entry, stop=stop, r_value=r_value, rr=rr, resistance=resistance,
        )

    # ---- 株数
    risk_jpy = capital * config.RISK_PER_TRADE
    fx = usdjpy if market == "US" else 1.0
    risk_local = risk_jpy / fx
    lot = config.LOT_SIZE[market]

    raw = risk_local / r_value
    shares = int(math.floor(raw / lot) * lot)

    if shares <= 0:
        need = r_value * lot * fx
        return TradePlan(
            False,
            f"資金に対して株価・値動きが大きすぎます。1単元({lot}株)で "
            f"{need:,.0f}円 のリスクとなり、許容額 {risk_jpy:,.0f}円 を超えます",
            entry=entry, stop=stop, r_value=r_value, rr=rr, resistance=resistance,
        )

    # ---- 1銘柄への集中度上限
    pos_jpy = shares * entry * fx
    cap = capital * config.MAX_POSITION_PCT
    if pos_jpy > cap:
        shares = int(math.floor((cap / (entry * fx)) / lot) * lot)
        if shares <= 0:
            return TradePlan(
                False,
                f"1銘柄あたりの投資上限 {cap:,.0f}円 では1単元も買えません",
                entry=entry, stop=stop, r_value=r_value, rr=rr,
            )
        pos_jpy = shares * entry * fx
        notes.append(f"1銘柄上限（資金の{config.MAX_POSITION_PCT:.0%}）に合わせて株数を縮小しました")

    trail = float(r["trail_high"]) - config.TRAIL_ATR_MULT * atr if _ok(r["trail_high"]) else stop

    # ギャップリスクの警告。
    # スイングは寄り付きで損切り値を飛び越えられる。固定の想定値（例:3%）だと
    # 銘柄のボラティリティと噛み合わず警告が無意味になるため、その銘柄自身の
    # 過去1年の下方ギャップ実測値（5パーセンタイル）と損切り幅を比較する。
    expected_loss = shares * r_value * fx
    stop_pct = r_value / entry
    if _ok(r.get("gap_down_p95")):
        gap_pct = abs(float(r["gap_down_p95"]))
        if gap_pct > stop_pct:
            gap_loss = shares * entry * gap_pct * fx
            notes.append(
                f"【ギャップ注意】この銘柄の過去1年の逆ギャップは悪い方5%で "
                f"{gap_pct:.1%} あり、損切り幅 {stop_pct:.1%} を上回ります。"
                f"寄り付きで飛び越えられた場合の実損は約 {gap_loss:,.0f}円 "
                f"（想定 {expected_loss:,.0f}円 の {gap_loss / expected_loss:.1f}倍）"
                f"になり得ます"
            )

    return TradePlan(
        ok=True, entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        r_value=r_value, rr=rr, shares=shares,
        position_value_jpy=pos_jpy, position_pct=pos_jpy / capital,
        risk_jpy=shares * r_value * fx, resistance=resistance,
        trail_stop=trail, notes=notes,
    )


# ============================================================ 短期 or 長期
def horizon_scores(df: pd.DataFrame, i: int, market: str,
                   fundamentals: Optional[dict] = None) -> Horizon:
    r = df.iloc[i]
    sma200_prev = df["sma200"].iloc[i - 20] if i - 20 >= -len(df) else np.nan

    long_checks = {
        "終値が200日線より上": (_ok(r["Close"] > r["sma200"]), 20),
        "50日線 > 200日線": (_ok(r["sma50"] > r["sma200"]), 15),
        "200日線が上向き（20日前と比較）": (_ok(r["sma200"] > sma200_prev), 15),
        "値動きが落ち着いている（ATR率 < 3%）": (_ok(r["atr_pct"] < 0.03), 15),
        "流動性が十分": (_ok(r["vol_ma"] >= config.MIN_AVG_VOLUME[market]), 15),
    }
    # 財務は live 判定のみ（過去の時点データが取れず、バックテストでは未来参照になる）
    if fundamentals:
        long_checks["増収増益（直近決算）"] = (bool(fundamentals.get("growing")), 20)

    short_checks = {
        "値幅が大きい（ATR率 ≥ 3%）": (_ok(r["atr_pct"] >= 0.03), 25),
        "出来高急増（20日平均の1.5倍以上）": (_ok(r["vol_ratio"] >= 1.5), 25),
        "200日線から15%以上乖離（行き過ぎ）": (_ok(abs(r["dist_sma200"]) >= 0.15), 25),
        "短期RSIが極端": (_ok(r["rsi_fast"] < 10 or r["rsi_fast"] > 90), 25),
    }

    def tally(checks):
        got = sum(p for ok, p in checks.values() if ok)
        total = sum(p for _, p in checks.values())
        return int(round(100 * got / total)) if total else 0

    ls, ss = tally(long_checks), tally(short_checks)

    if ls >= 60 and ss < 60:
        label = "長期保有向き"
    elif ss >= 60 and ls < 60:
        label = "短期売買のみ"
    elif ls >= 60 and ss >= 60:
        label = "短期で入り、長期スコアが維持されれば持ち越し可"
    else:
        label = "どちらとも言えない（見送り推奨）"

    return Horizon(ls, ss, label, {
        "長期": {k: {"ok": o, "pts": p} for k, (o, p) in long_checks.items()},
        "短期": {k: {"ok": o, "pts": p} for k, (o, p) in short_checks.items()},
    })


# ============================================================ 保有中の売り判定
def exit_signal(df: pd.DataFrame, i: int, entry: float, stop: float,
                style: str) -> tuple[bool, str]:
    """保有ポジションの決済判定。Phase 1 のバックテストでも使う。"""
    r = df.iloc[i]
    if r["Low"] <= stop:
        return True, f"損切りライン {stop:,.2f} に到達"
    if config.USE_SMA200_EXIT and _ok(r["Close"] < r["sma200"]):
        return True, "終値が200日線を下回りトレンドが崩れました"
    if style == "short" and _ok(r["rsi_fast"] > 70):
        return True, "RSI(2) が70を超え短期の反発が完了しました"
    return False, ""
