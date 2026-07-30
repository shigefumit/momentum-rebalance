"""総合判定。各レイヤーを束ねて「買う / 売る / 何もしない」を出す。

スコア構成:
  市場環境(0-30) + 銘柄品質(0-30) + セットアップ(0-40) = ルールスコア(0-100)
  + AI補正(-20〜+20) = 総合スコア

AI層は買いを作れない。止めることと補正だけ。この非対称性が設計の中核。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import config
import fetcher
import indicators
import rules


@dataclass
class Judgement:
    symbol: str
    market: str
    as_of: pd.Timestamp
    close: float
    usdjpy: float

    regime: rules.Regime
    quality: rules.Quality
    setup: rules.Setup
    plan: rules.TradePlan
    horizon: rules.Horizon

    rule_score: int = 0
    ai_adjust: int = 0
    total_score: int = 0
    ai: Optional[dict] = None

    decision: str = "何もしない"
    reasons: list = field(default_factory=list)


def judge(symbol: str, use_ai: bool = False, period: str = "3y",
          i: int = -1, verbose: bool = True) -> Judgement:
    """1銘柄を判定する。

    i を変えれば過去の任意の日で判定できる（Phase 1 のバックテストで使う）。
    """
    market = fetcher.market_of(symbol)

    if verbose:
        print(f"  データ取得中: {symbol} …")
    px = indicators.enrich(fetcher.get_prices(symbol, period=period))
    if len(px) < config.SMA_LONG + 20:
        raise fetcher.DataUnavailable(
            f"{symbol} のデータが {len(px)} 本しかありません。"
            f"200日線の計算に最低 {config.SMA_LONG + 20} 本必要です"
        )

    if verbose:
        print("  指数データ取得中: S&P500 / VIX / 日経平均 …")
    spx = indicators.enrich(fetcher.get_prices("^GSPC", period=period))
    vix = indicators.enrich(fetcher.get_prices("^VIX", period=period))
    try:
        nikkei = indicators.enrich(fetcher.get_prices("^N225", period=period))
    except fetcher.DataUnavailable:
        nikkei = pd.DataFrame()

    row = px.iloc[i]
    as_of = px.index[i]

    # 為替は判定日時点のレートを使う。現在のレートを過去日に当てると
    # バックテストの株数・損益が実際と食い違う。
    usdjpy = fetcher.get_usdjpy(as_of=as_of, period=period) if market == "US" else 1.0

    # --- L1〜L3
    regime = rules.regime_score(spx, vix, nikkei, as_of, market)
    quality = rules.quality_score(px, i, market)
    setup = rules.setup_score(px, i, regime.state)

    # --- 指値・損切り・利確・株数
    if setup.score > 0:
        plan = rules.plan_trade(px, i, setup, market, usdjpy)
    else:
        plan = rules.TradePlan(False, "エントリー・セットアップが不成立")

    # --- 短期 or 長期
    fund = fetcher.get_fundamentals(symbol) if verbose else None
    horizon = rules.horizon_scores(px, i, market, fund)

    j = Judgement(
        symbol=symbol, market=market, as_of=as_of, close=float(row["Close"]),
        usdjpy=usdjpy, regime=regime, quality=quality, setup=setup,
        plan=plan, horizon=horizon,
    )
    j.rule_score = regime.score + quality.score + setup.score

    # --- L4 AI層（拒否権と補正のみ）
    if use_ai and setup.score > 0 and plan.ok:
        import ai_layer
        if verbose:
            print(f"  AI判断中（{config.AI_MODEL}）…")
        j.ai = ai_layer.evaluate(symbol, px, i, regime, quality, setup, plan, horizon)
        j.ai_adjust = int(j.ai.get("confidence_adjustment", 0))
        j.ai_adjust = max(-config.AI_ADJUST_RANGE, min(config.AI_ADJUST_RANGE, j.ai_adjust))

    j.total_score = j.rule_score + j.ai_adjust
    _decide(j)
    return j


def _decide(j: Judgement) -> None:
    r = j.reasons

    if j.regime.state == "RISK_OFF":
        r.append(f"市場環境が RISK_OFF（{'・'.join(j.regime.details.values())}）。"
                 f"新規買いは見送ります")

    if j.setup.score == 0:
        r.append(f"「{j.setup.name}」の条件を満たしません: " + "、".join(j.setup.failed))
        j.decision = "何もしない"
        return

    if not j.plan.ok:
        r.append(j.plan.reason)
        j.decision = "何もしない"
        return

    if j.ai and j.ai.get("veto"):
        r.append(f"AIが見送りを推奨: {j.ai.get('veto_reason', '理由不明')}")
        j.decision = "何もしない"
        return

    if j.total_score >= config.BUY_THRESHOLD:
        j.decision = "買う"
        r.append(f"総合スコア {j.total_score} 点が合格ライン "
                 f"{config.BUY_THRESHOLD} 点以上、R:R {j.plan.rr:.2f} が最低基準 "
                 f"{config.MIN_RR} 以上を満たしています")
    else:
        j.decision = "何もしない"
        r.append(f"総合スコア {j.total_score} 点が合格ライン "
                 f"{config.BUY_THRESHOLD} 点に届きません（あと "
                 f"{config.BUY_THRESHOLD - j.total_score} 点）")
