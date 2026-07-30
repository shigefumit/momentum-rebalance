#!/usr/bin/env python3
"""複数銘柄のスクリーニング / 過去シグナルの洗い出し。

Phase 1 のバックテストはこのスクリプトを拡張して作る。ルール層が
「任意のバー位置で評価できる純粋関数」になっているのでループするだけで済む。

使い方:
    # 今日のエントリー候補を探す
    .venv/bin/python scan.py NVDA AMD AVGO MSFT 7203.T

    # 1銘柄の過去シグナルを一覧（Phase 1 の下準備）
    .venv/bin/python scan.py NVDA --history
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import config
import fetcher
import indicators
import rules


def dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1 for c in s)


def pad(s: str, w: int) -> str:
    s = str(s)
    while dwidth(s) > w:
        s = s[:-1]
    return s + " " * max(0, w - dwidth(s))


class Context:
    """指数・為替は全銘柄で共通なので一度だけ取得して使い回す（レート制限対策）。"""

    def __init__(self, period: str):
        self.period = period
        print("  指数データ取得中: S&P500 / VIX / 日経平均 / ドル円 …")
        self.spx = indicators.enrich(fetcher.get_prices("^GSPC", period=period))
        self.vix = indicators.enrich(fetcher.get_prices("^VIX", period=period))
        try:
            self.nikkei = indicators.enrich(fetcher.get_prices("^N225", period=period))
        except fetcher.DataUnavailable:
            self.nikkei = pd.DataFrame()
        self.fx = fetcher.get_usdjpy_series(period)

    def fx_at(self, ts: pd.Timestamp, market: str) -> float:
        if market == "JP":
            return 1.0
        if self.fx is None:
            return fetcher.FX_FALLBACK
        sub = self.fx.loc[:ts]
        return float(sub["Close"].iloc[-1]) if not sub.empty else fetcher.FX_FALLBACK


def evaluate_bar(px: pd.DataFrame, i: int, market: str, ctx: Context) -> dict:
    """1銘柄・1バーを評価。ここが live と backtest の共通コア。"""
    ts = px.index[i]
    reg = rules.regime_score(ctx.spx, ctx.vix, ctx.nikkei, ts, market)
    qual = rules.quality_score(px, i, market)
    setup = rules.setup_score(px, i, reg.state)

    rule_score = reg.score + qual.score + setup.score
    plan = (rules.plan_trade(px, i, setup, market, ctx.fx_at(ts, market))
            if setup.score > 0 else rules.TradePlan(False, "セットアップ不成立"))

    decision = "何もしない"
    if setup.score > 0 and plan.ok and rule_score >= config.BUY_THRESHOLD:
        decision = "買う（AI審査前）"

    return {
        "date": ts.date(), "score": rule_score, "regime": reg.state,
        "setup": setup.name if setup.score else "-", "setup_score": setup.score,
        "plan_ok": plan.ok, "rr": round(plan.rr, 2) if plan.ok else None,
        "entry": round(plan.entry, 2) if plan.ok else None,
        "stop": round(plan.stop, 2) if plan.ok else None,
        "shares": plan.shares, "decision": decision,
        "note": "" if plan.ok else plan.reason,
    }


def scan_today(symbols: list[str], period: str) -> None:
    ctx = Context(period)
    rows = []
    for sym in symbols:
        market = fetcher.market_of(sym)
        print(f"  {sym} …")
        try:
            px = indicators.enrich(fetcher.get_prices(sym, period=period))
            if len(px) < config.SMA_LONG + 20:
                rows.append({"symbol": sym, "decision": "データ不足", "score": 0,
                             "setup": "-", "rr": None, "shares": 0, "note": f"{len(px)}本のみ"})
                continue
            r = evaluate_bar(px, -1, market, ctx)
            r["symbol"] = sym
            rows.append(r)
        except fetcher.DataUnavailable as e:
            rows.append({"symbol": sym, "decision": "取得失敗", "score": 0,
                         "setup": "-", "rr": None, "shares": 0, "note": str(e)[:50]})

    rows.sort(key=lambda x: -x.get("score", 0))

    print()
    print("═" * 100)
    hdr = (pad("銘柄", 10) + pad("判定", 18) + pad("スコア", 8) +
           pad("セットアップ", 22) + pad("R:R", 7) + pad("指値", 10) +
           pad("損切り", 10) + pad("株数", 7))
    print("  " + hdr)
    print("─" * 100)
    for r in rows:
        print("  " + pad(r["symbol"], 10) + pad(r.get("decision", "-"), 18)
              + pad(r.get("score", 0), 8) + pad(r.get("setup", "-"), 22)
              + pad(r.get("rr") or "-", 7) + pad(r.get("entry") or "-", 10)
              + pad(r.get("stop") or "-", 10) + pad(r.get("shares", 0), 7))
    print("═" * 100)

    buys = [r for r in rows if r.get("decision", "").startswith("買う")]
    print()
    if buys:
        print(f"  エントリー候補 {len(buys)} 件: "
              + "、".join(r["symbol"] for r in buys))
        print(f"  → 各銘柄を  main.py <銘柄> --ai  で個別に審査してください")
    else:
        print("  今日のエントリー候補はありません。")
        print("  （これが正常です。ほとんどの日は「何もしない」が正解です）")
    print()
    for r in rows:
        if r.get("note"):
            print(f"  {r['symbol']}: {r['note']}")
    print()


def scan_history(symbol: str, period: str) -> None:
    ctx = Context(period)
    market = fetcher.market_of(symbol)
    print(f"  {symbol} の履歴を走査中 …")
    px = indicators.enrich(fetcher.get_prices(symbol, period=period))

    start = config.SMA_LONG + 10
    if len(px) <= start:
        print(f"  [エラー] データが {len(px)} 本しかありません")
        return

    rows = [evaluate_bar(px, k - len(px), market, ctx) for k in range(start, len(px))]
    d = pd.DataFrame(rows)
    hits = d[d.setup_score > 0]
    buys = d[d.decision.str.startswith("買う")]
    bars = len(d)

    print()
    print("═" * 100)
    print(f"  {symbol}  過去 {bars} 営業日（{d.date.iloc[0]} 〜 {d.date.iloc[-1]}）")
    print("─" * 100)
    print(f"  セットアップ成立      {len(hits):>4} 件  （{len(hits)/bars:>5.1%}）")
    print(f"  プラン成立            {int(hits.plan_ok.sum()):>4} 件")
    print(f"  買いシグナル          {len(buys):>4} 件  （{len(buys)/bars:>5.1%}）")
    if len(buys):
        yrs = bars / 252
        print(f"  年間シグナル数        {len(buys)/yrs:>6.1f} 件/年")
        print(f"  R:R                   中央値 {buys.rr.median():.2f} / "
              f"範囲 {buys.rr.min():.2f}〜{buys.rr.max():.2f}")
        print()
        print("  セットアップ別:")
        for name, grp in buys.groupby("setup"):
            print(f"    {pad(name, 24)} {len(grp):>3} 件")
        print()
        print("  市場環境別:")
        for name, grp in buys.groupby("regime"):
            print(f"    {pad(name, 24)} {len(grp):>3} 件")
    print("═" * 100)
    print()
    if len(buys):
        print("  買いシグナル一覧:")
        print(buys[["date", "setup", "score", "regime", "rr", "entry",
                    "stop", "shares"]].to_string(index=False))
    print()
    print("  ※ ここに出るのは「シグナルが出た」だけで、勝ったかどうかは未検証です。")
    print("     勝率・期待値の測定は Phase 1 のバックテストで行います。")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="複数銘柄スクリーニング / 過去シグナル洗い出し")
    ap.add_argument("symbols", nargs="+", help="銘柄コード。日本株は末尾 .T")
    ap.add_argument("--history", action="store_true",
                    help="1銘柄の過去シグナルを一覧（Phase 1 の下準備）")
    ap.add_argument("--period", default="3y")
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols]
    syms = [s + ".T" if s.isdigit() and len(s) == 4 else s for s in syms]

    if args.history:
        if len(syms) != 1:
            print("  [エラー] --history は1銘柄のみ指定してください")
            return 1
        scan_history(syms[0], args.period)
    else:
        scan_today(syms, args.period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
