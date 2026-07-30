#!/usr/bin/env python3
"""株式売買判断ツール — Phase 0 CLI

使い方:
    .venv/bin/python main.py NVDA
    .venv/bin/python main.py NVDA --ai          # Claude 判断層を有効化
    .venv/bin/python main.py 7203.T             # 日本株（末尾 .T が必須）
    .venv/bin/python main.py NVDA --date 2026-06-15   # 過去の任意日で判定
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import config
import fetcher
import judge as judge_mod

W = 64
DECISION_MARK = {"買う": "◆ 買う", "売る": "◆ 売る", "何もしない": "・ 何もしない"}


def dwidth(s: str) -> int:
    """表示幅。日本語の全角文字は2カラム分を占めるため文字数では揃わない。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1 for c in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


def money(v: float, market: str) -> str:
    return f"${v:,.2f}" if market == "US" else f"{v:,.0f}円"


def bar(label: str, got: int, total: int, note: str = "") -> str:
    filled = int(round(12 * got / total)) if total else 0
    return (f"  {pad(label, 16)} {got:>3} / {total:<3} "
            f"[{'█' * filled}{'░' * (12 - filled)}] {note}")


def render(j: judge_mod.Judgement) -> None:
    m = j.market
    print()
    print("═" * W)
    print(f"  {j.symbol}   判定日 {j.as_of.date()}   終値 {money(j.close, m)}")
    if m == "US":
        print(f"  （ドル円 {j.usdjpy:.2f} で円換算）")
    print("═" * W)
    print()
    print(f"     {DECISION_MARK[j.decision]}")
    print()

    # ---- スコア内訳
    print(f"  総合スコア  {j.total_score} / 100   （合格ライン {config.BUY_THRESHOLD}）")
    print()
    print(bar("市場環境", j.regime.score, 30, j.regime.state))
    print(bar("銘柄品質", j.quality.score, 30))
    print(bar("セットアップ", j.setup.score, 40,
              j.setup.name if j.setup.score else f"{j.setup.name} 不成立"))
    if j.ai:
        sign = "+" if j.ai_adjust >= 0 else ""
        print(f"  {pad('AI補正', 16)} {sign}{j.ai_adjust} 点")
    print()

    # ---- 売買プラン
    p = j.plan
    print("─" * W)
    if p.ok:
        print("  売買プラン")
        print()
        print(f"    エントリー指値   {money(p.entry, m)}")
        print(f"    損切りライン     {money(p.stop, m)}   "
              f"（−{money(p.r_value, m)} = 1R / "
              f"{p.r_value / p.entry:.1%}）")
        print(f"    第1利確 (+1.0R)  {money(p.tp1, m)}   ← ここで半分売却")
        print(f"    最終目標         {money(p.tp2, m)}")
        print(f"    トレーリング     {money(p.trail_stop, m)}   "
              f"（直近{config.TRAIL_LOOKBACK}日高値 − {config.TRAIL_ATR_MULT}×ATR）")
        if p.resistance:
            print(f"    上値抵抗         {money(p.resistance, m)}（60日高値）")
        print()
        print(f"    リスクリワード   {p.rr:.2f} : 1   （最低基準 {config.MIN_RR}）")
        print(f"    推奨株数         {p.shares:,} 株")
        print(f"    必要投資額       {p.position_value_jpy:,.0f}円   "
              f"（資金の {p.position_pct:.1%}）")
        print(f"    想定最大損失     {p.risk_jpy:,.0f}円   "
              f"（資金の {p.risk_jpy / config.CAPITAL_JPY:.2%}）")
        print()
        for n in p.notes:
            print(f"    ※ {n}")
    else:
        print(f"  売買プラン: 算出せず")
        print(f"    理由: {p.reason}")
        if p.entry and p.stop:
            print(f"    （参考: 指値 {money(p.entry, m)} / "
                  f"損切り {money(p.stop, m)} / R:R {p.rr:.2f}）")
    print()

    # ---- 保有期間
    h = j.horizon
    print("─" * W)
    print(f"  保有期間の判定   {h.label}")
    print(f"    長期スコア {h.long_score} / 100     短期スコア {h.short_score} / 100")
    print()

    # ---- 市場環境の詳細
    print("─" * W)
    print(f"  市場環境   {j.regime.state}")
    for k, v in j.regime.details.items():
        print(f"    {k}: {v}")
    print()

    # ---- 銘柄品質の内訳
    print("─" * W)
    print("  銘柄品質の内訳")
    for k, v in j.quality.checks.items():
        print(f"    {'○' if v['ok'] else '×'} {k}  (+{v['pts'] if v['ok'] else 0})")
    print()

    # ---- セットアップ条件
    print("─" * W)
    print(f"  セットアップ条件   「{j.setup.name}」")
    for k, ok in j.setup.passed.items():
        print(f"    {'○' if ok else '×'} {k}")
    print()

    # ---- AIコメント
    if j.ai:
        print("─" * W)
        print(f"  AI判断（{config.AI_MODEL}）")
        print(f"    地政学リスク: {j.ai.get('geopolitical_risk', '-')}")
        print(f"    推奨保有期間: {j.ai.get('holding_horizon_opinion', '-')}")
        risks = j.ai.get("key_risks") or []
        if risks:
            print("    主なリスク:")
            for x in risks:
                print(f"      - {x}")
        summary = j.ai.get("summary_ja", "")
        if summary:
            print(f"    コメント: {summary}")
        print()

    # ---- 結論の根拠
    print("─" * W)
    print("  判定の根拠")
    for x in j.reasons:
        print(f"    ・{x}")
    print()
    print("═" * W)
    print("  ※ このツールは自分専用です。発注前に必ず証券会社の板で最終確認してください。")
    print("═" * W)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="株式売買判断ツール（Phase 0）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("symbol", help="銘柄コード。米国株=AAPL / 日本株=7203.T")
    ap.add_argument("--ai", action="store_true",
                    help=f"AI判断層を有効化（{config.AI_MODEL}・要 ANTHROPIC_API_KEY）")
    ap.add_argument("--date", default=None,
                    help="この日付時点で判定（YYYY-MM-DD）。過去検証用")
    ap.add_argument("--period", default="3y", help="取得期間（既定 3y）")
    ap.add_argument("--no-cache", action="store_true", help="キャッシュを使わず再取得")
    args = ap.parse_args()

    if args.no_cache:
        fetcher.CACHE_TTL_HOURS = 0

    symbol = args.symbol.strip().upper()

    # 日本株の .T 忘れは最も多いミス（エラーではなく空データが返るため気づきにくい）
    if symbol.isdigit() and len(symbol) == 4:
        print(f"\n  [注意] 「{symbol}」は日本株コードのようです。"
              f"末尾に .T が必要です → {symbol}.T に補正します")
        symbol += ".T"

    print(f"\n  判定開始: {symbol}")

    try:
        idx = -1
        if args.date:
            import pandas as pd
            import indicators
            px = indicators.enrich(fetcher.get_prices(symbol, period=args.period))
            target = pd.Timestamp(args.date)
            sub = px.loc[:target]
            if sub.empty:
                print(f"  [エラー] {args.date} 以前のデータがありません")
                return 1
            idx = px.index.get_loc(sub.index[-1]) - len(px)

        j = judge_mod.judge(symbol, use_ai=args.ai, period=args.period, i=idx)
    except fetcher.DataUnavailable as e:
        print(f"\n  [エラー] {e}\n")
        return 1
    except Exception as e:
        print(f"\n  [エラー] {type(e).__name__}: {e}\n")
        return 1

    render(j)
    return 0


if __name__ == "__main__":
    sys.exit(main())
