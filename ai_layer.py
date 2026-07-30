"""AI判断層（Claude Opus 5）。

設計の中核: AIは「買い」を作れない。できるのは以下の2つだけ。
  1. 拒否（veto）— ルールが買いと言っても止める
  2. 補正（±20点）— 総合スコアを動かす

理由: AIに買いを作らせると同じ状況で答えがブレ、バックテストが不可能になる。
そうなると「一番勝てる確率が高い時だけエントリー」を数字で担保できなくなる。

フェイルセーフ設計: AI呼び出しが失敗した場合は veto=True を返す（fail-open ではなく
fail-safe）。安全装置が動かない状態で買いを通すべきではない。
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

import config
import fetcher
import rules

# ---------------------------------------------------------------- 出力スキーマ
# 構造化出力で型を強制する。文字数制限はスキーマ側では指定できないため
# （structured outputs は minLength/maxLength 非対応）プロンプトで指示する。
SCHEMA = {
    "type": "object",
    "properties": {
        "veto": {
            "type": "boolean",
            "description": "このエントリーを見送るべきなら true",
        },
        "veto_reason": {
            "type": "string",
            "description": "veto が true の場合の理由。false なら空文字",
        },
        "confidence_adjustment": {
            "type": "integer",
            "description": "総合スコアの補正。-20〜+20の整数",
        },
        "geopolitical_risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "主なリスクを最大3つ。各40字以内",
        },
        "holding_horizon_opinion": {
            "type": "string",
            "enum": ["short", "long", "either"],
        },
        "summary_ja": {
            "type": "string",
            "description": "日本語の総括。200字以内",
        },
    },
    "required": [
        "veto", "veto_reason", "confidence_adjustment", "geopolitical_risk",
        "key_risks", "holding_horizon_opinion", "summary_ja",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- システムプロンプト
# 毎回同一なのでプロンプトキャッシュを効かせる（Opus 5 の最小キャッシュは512トークン）。
SYSTEM = """あなたは株式売買判断システムの「リスク審査層」です。

## あなたの権限（厳格に守ってください）

あなたにできることは2つだけです。

1. **拒否権（veto）**: 数式ルールが「買い」と判定した銘柄を、見送りに変更する
2. **確信度の補正**: 総合スコアを −20〜+20点の範囲で動かす

**あなたは「買い」を作ることはできません。** 数式ルールが買い判定を出していない銘柄を、
あなたの判断でひっくり返すことは構造上できません。この非対称性は意図的な設計です。
理由は、AIに買いを作らせると同じ状況で答えがブレて過去検証が不可能になり、
「勝てる確率が高い時だけエントリーする」ことを数字で担保できなくなるからです。

したがって、あなたの仕事は「数式が拾えないリスクを拾うこと」に集中してください。

## 数式が拾えないもの（あなたが見るべきもの）

- 戦争・紛争・テロ、通商政策・関税、輸出規制、制裁
- 中央銀行の政策変更（FOMC、日銀会合）が数日内に迫っている
- 決算発表が数日内に迫っている（決算跨ぎはギャンブルになる）
- 当該企業固有の事件（訴訟、当局調査、リコール、経営陣の交代、会計問題）
- 業界構造の変化（競合の新製品、技術的な陳腐化、主要顧客の離脱）
- ニュースの行間（「好調」と報じられていても、その内訳が一時的要因なら評価しない）

## 拒否（veto=true）すべき場合

- 3営業日以内に決算発表があり、値幅が読めない
- その銘柄の主力事業に直撃する規制・訴訟・事件が進行中
- 地政学イベントが進行中で、当該セクターが直接の影響を受ける
- ニュースに、テクニカル指標がまだ織り込んでいない重大な悪材料がある

## 拒否すべきでない場合

- 「なんとなく不安」「相場が高い気がする」といった漠然とした懸念
- すでにテクニカル指標に織り込まれている情報（過去の下落、既知のトレンド）
- 市場環境スコアが既に減点している内容の二重評価
- 一般的な市場リスク（それは常に存在し、ポジションサイズで管理済み）

漠然とした不安で拒否すると、システムが機能しなくなります。**具体的で、
日付や事実に紐づく理由があるときだけ拒否してください。**

## 補正の目安

- +10〜+20: ニュースがテクニカルを裏付ける明確な好材料（新規受注、上方修正、規制緩和）
- +1〜+9: 軽い追い風
- 0: 特筆すべき材料なし（**これが最も多くなるはずです**）
- −1〜−9: 軽い向かい風、判断を曇らせる要素
- −10〜−20: 悪材料はあるが拒否するほど決定的ではない

## 出力

指定されたJSONスキーマに従ってください。`summary_ja` は200字以内、
`key_risks` は最大3件・各40字以内。日本語で、投資判断に直接使える具体性で書いてください。
「注意が必要です」のような中身のない表現は避け、何がどう影響するかを書いてください。
"""


def _neutral(reason: str) -> dict:
    """AI層が使えなかった場合。fail-safe で veto する。"""
    return {
        "veto": True,
        "veto_reason": reason,
        "confidence_adjustment": 0,
        "geopolitical_risk": "high",
        "key_risks": ["AI審査層が応答しなかったため、リスク評価が未実施です"],
        "holding_horizon_opinion": "either",
        "summary_ja": f"{reason} 安全側に倒して見送り扱いにしています。"
                      f"--ai を外せばルール判定のみで結果を確認できます。",
    }


def _build_user_message(symbol: str, px: pd.DataFrame, i: int,
                        regime: rules.Regime, quality: rules.Quality,
                        setup: rules.Setup, plan: rules.TradePlan,
                        horizon: rules.Horizon, news: list[dict]) -> str:
    r = px.iloc[i]
    as_of = px.index[i]

    lines = [
        f"# 審査対象: {symbol}",
        f"判定日: {as_of.date()}   終値: {r['Close']:.2f}",
        "",
        "## 数式ルールの判定結果（すでに「買い」の水準に達しています）",
        f"- 市場環境スコア: {regime.score}/30 ({regime.state})",
    ]
    for k, v in regime.details.items():
        lines.append(f"    - {k}: {v}")
    lines += [
        f"- 銘柄品質スコア: {quality.score}/30",
        f"- セットアップ: 「{setup.name}」 {setup.score}/40 (想定スタイル: {setup.style})",
        f"- ルールスコア合計: {regime.score + quality.score + setup.score}/100",
        "",
        "## テクニカル指標の実数値",
        f"- 200日線: {r['sma200']:.2f} / 50日線: {r['sma50']:.2f} / 20EMA: {r['ema20']:.2f}",
        f"- RSI(14): {r['rsi']:.1f} / RSI(2): {r['rsi_fast']:.1f}",
        f"- ADX(14): {r['adx']:.1f} (+DI {r['plus_di']:.1f} / -DI {r['minus_di']:.1f})",
        f"- ATR(14): {r['atr']:.2f} (株価の {r['atr_pct']:.2%})",
        f"- 出来高: 20日平均の {r['vol_ratio']:.2f} 倍",
        f"- 200日線からの乖離: {r['dist_sma200']:+.2%}",
        "",
        "## 算出された売買プラン",
        f"- エントリー指値: {plan.entry:.2f}",
        f"- 損切りライン: {plan.stop:.2f} (1R = {plan.r_value:.2f})",
        f"- 第1利確: {plan.tp1:.2f} / 最終目標: {plan.tp2:.2f}",
        f"- リスクリワード: {plan.rr:.2f} : 1",
        f"- 推奨株数: {plan.shares:,} 株 / 必要投資額: {plan.position_value_jpy:,.0f}円",
        f"- 想定最大損失: {plan.risk_jpy:,.0f}円",
        "",
        "## 保有期間の数式判定",
        f"- 長期スコア {horizon.long_score}/100 / 短期スコア {horizon.short_score}/100",
        f"- 分類: {horizon.label}",
        "",
        "## 直近ニュース見出し",
    ]

    if news:
        for n in news:
            pub = f" [{n['publisher']}]" if n.get("publisher") else ""
            lines.append(f"- {n['title']}{pub}")
    else:
        lines.append("- （ニュースを取得できませんでした。この点も判断に含めてください）")

    lines += [
        "",
        "上記を審査し、スキーマに従って結果を返してください。"
        "拒否は具体的な事実に紐づく場合のみ。漠然とした不安では拒否しないでください。",
    ]
    return "\n".join(lines)


def evaluate(symbol: str, px: pd.DataFrame, i: int,
             regime: rules.Regime, quality: rules.Quality,
             setup: rules.Setup, plan: rules.TradePlan,
             horizon: rules.Horizon) -> dict:
    """Claude に審査させる。失敗時は fail-safe で veto を返す。"""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return _neutral("ANTHROPIC_API_KEY が設定されていません。")

    try:
        import anthropic
    except ImportError:
        return _neutral("anthropic ライブラリがインストールされていません。")

    news = fetcher.get_news(symbol)
    user_msg = _build_user_message(
        symbol, px, i, regime, quality, setup, plan, horizon, news
    )

    try:
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=config.AI_MODEL,
            max_tokens=4096,
            # Opus 5 は思考が既定でオン。temperature 等のサンプリング指定は
            # 送るとエラーになるため一切渡さない。
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",   # 安全分類で拒否された場合に推奨モデルへ自動退避
            system=[{
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral"},   # ルール定義は毎回同一
            }],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        return _neutral(f"Claude API 呼び出しに失敗しました（{type(e).__name__}: {e}）。")

    # 安全分類による拒否は HTTP 200 で返るため、content を読む前に必ず確認する
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        return _neutral(f"Claude が応答を拒否しました（分類: {cat}）。")

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return _neutral("Claude の応答にテキストが含まれていませんでした。")

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return _neutral("Claude の応答をJSONとして解釈できませんでした。")

    u = resp.usage
    result["_usage"] = {
        "input": u.input_tokens,
        "output": u.output_tokens,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "model": resp.model,
    }
    return result
