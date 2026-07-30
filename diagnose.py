#!/usr/bin/env python3
"""Phase 1 診断: エントリーに予測力があるのかを、決済ロジックから切り離して測る。

バックテストが負けた時、原因は2つに1つ。
  (A) エントリーのタイミングに予測力がない → 手法自体を作り直す必要がある
  (B) エントリーは良いが決済がリターンを壊している → 構造の修正で直る可能性がある

この2つを混ぜたまま「パラメータをいじる」と、2016〜2026年に曲線を当てはめるだけの
作業になり、バックテストでしか勝てないルールが出来上がる。だから先に切り分ける。

測るもの:
  1. シグナル翌日からのフォワードリターン（1/3/5/10/20/60営業日）
  2. 同じ銘柄・同じ期間の「全営業日」を母集団としたベースライン
     → シグナル日がベースラインを上回らなければ、エントリーに予測力はない
  3. MFE / MAE（最大含み益・最大含み損、R単位）
     → MFEが実際の獲得Rを大きく上回るなら、決済が利益を取り逃している

使い方:
    .venv/bin/python diagnose.py
    .venv/bin/python diagnose.py --symbols NVDA AMD --period 5y
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest as bt
import config
import fetcher
import indicators
import rules

HORIZONS = [1, 3, 5, 10, 20, 60]


def dwidth(s) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1
               for c in str(s))


def pad(s, w: int) -> str:
    s = str(s)
    return s + " " * max(0, w - dwidth(s))


def rpad(s, w: int) -> str:
    s = str(s)
    return " " * max(0, w - dwidth(s)) + s


def forward_returns(px: pd.DataFrame, k: int) -> dict:
    """バーk のシグナル → バーk+1 の寄り付きで買った場合のフォワードリターン。"""
    if k + 1 >= len(px):
        return {}
    base = float(px.iloc[k + 1]["Open"])
    if base <= 0:
        return {}
    out = {}
    for h in HORIZONS:
        j = k + 1 + h
        if j < len(px):
            out[h] = float(px.iloc[j]["Close"]) / base - 1.0
    return out


def excursions(px: pd.DataFrame, k: int, r_value: float, bars: int = 20) -> dict:
    """買った後 bars 日間の最大含み益(MFE)と最大含み損(MAE)をR単位で測る。"""
    if k + 1 >= len(px) or r_value <= 0:
        return {}
    entry = float(px.iloc[k + 1]["Open"])
    win = px.iloc[k + 1: k + 1 + bars]
    if win.empty:
        return {}
    return {
        "mfe_r": (float(win["High"].max()) - entry) / r_value,
        "mae_r": (float(win["Low"].min()) - entry) / r_value,
    }


def run(symbols: list[str], period: str) -> int:
    print(f"\n  Phase 1 診断   {len(symbols)} 銘柄 / 期間 {period}")
    ctx = bt.Context(period)

    sig_rows, base_rows = [], []

    for sym in symbols:
        market = fetcher.market_of(sym)
        try:
            px = indicators.enrich(fetcher.get_prices(sym, period=period))
        except fetcher.DataUnavailable as e:
            print(f"  {pad(sym, 9)} 取得失敗")
            continue
        if len(px) < config.SMA_LONG + 100:
            print(f"  {pad(sym, 9)} データ不足")
            continue

        n_sig = 0
        start = config.SMA_LONG + 10
        for k in range(start, len(px) - 1):
            ts = px.index[k]
            fr = forward_returns(px, k)
            if not fr:
                continue

            # --- ベースライン: 全営業日（ランダムに入った場合の母集団）
            base_rows.append({"symbol": sym, **{f"r{h}": v for h, v in fr.items()}})

            # --- シグナル日
            reg = ctx.regime_at(ts, market)
            setup = rules.setup_score(px, k - len(px), reg.state)
            if setup.score == 0:
                continue
            qual = rules.quality_score(px, k - len(px), market)
            score = reg.score + qual.score + setup.score
            if score < config.BUY_THRESHOLD:
                continue
            fx = ctx.fx_at(ts, market)
            plan = rules.plan_trade(px, k - len(px), setup, market, fx)
            if not plan.ok:
                continue

            ex = excursions(px, k, plan.r_value)
            sig_rows.append({
                "symbol": sym, "date": ts, "setup": setup.name,
                "regime": reg.state, "score": score,
                "r_pct": plan.r_value / plan.entry,
                **{f"r{h}": v for h, v in fr.items()}, **ex,
            })
            n_sig += 1
        print(f"  {pad(sym, 9)} シグナル {n_sig:>4} 件 / 全営業日 {len(px) - start:>5} 日")

    if not sig_rows:
        print("\n  シグナルが0件でした。診断できません。\n")
        return 1

    sig = pd.DataFrame(sig_rows)
    base = pd.DataFrame(base_rows)
    W = 78

    print()
    print("═" * W)
    print("  診断1: エントリーに予測力があるか（フォワードリターン比較）")
    print("═" * W)
    print(f"\n  シグナル {len(sig)} 件 vs 全営業日 {len(base)} 件（同じ銘柄・同じ期間）\n")
    print("    " + pad("保有日数", 10) + rpad("シグナル平均", 14) + rpad("全日平均", 12)
          + rpad("差", 10) + rpad("シグナル勝率", 14) + rpad("全日勝率", 12))
    print("    " + "─" * 70)

    cost = config.COST_PER_SIDE * 2      # 往復コスト。エッジはこれを超えて初めて意味を持つ
    edge_count = 0
    beats_cost = 0
    diffs = {}
    for h in HORIZONS:
        col = f"r{h}"
        if col not in sig or col not in base:
            continue
        s, b = sig[col].dropna(), base[col].dropna()
        if s.empty or b.empty:
            continue
        diff = s.mean() - b.mean()
        diffs[h] = diff
        if diff > 0:
            edge_count += 1
        if diff > cost:
            beats_cost += 1
        print("    " + pad(f"{h}日", 10)
              + rpad("{:+.2%}".format(s.mean()), 14)
              + rpad("{:+.2%}".format(b.mean()), 12)
              + rpad("{:+.2%}".format(diff), 10)
              + rpad("{:.1%}".format((s > 0).mean()), 14)
              + rpad("{:.1%}".format((b > 0).mean()), 12))

    best = max(diffs.values()) if diffs else 0.0
    print(f"\n  ベースラインを上回った期間: {edge_count} / {len(diffs)}")
    print(f"  往復コスト（手数料+スリッページ）: {cost:.2%}")
    print(f"  最大のエッジ: {best:+.2%}   → コスト比 {best / cost:.2f} 倍"
          if cost else "")
    print()
    # 判定はコストとの比較で行う。コストを超えないエッジは取引しても意味がない。
    if beats_cost >= 3 and best > cost * 2:
        print("  → エントリーに実用的な予測力がある。")
        print("     負けているのは決済側の問題である可能性が高い")
    elif best > cost:
        print("  → エッジは存在するが往復コストと同程度の大きさしかない。")
        print("     この差では手数料とスリッページに 食われ、利益として残らない。")
        print("     エントリー条件を強化するか、保有期間を延ばしてコスト比を改善する必要がある")
    else:
        print("  → エッジが往復コストに届かない。このエントリー条件では利益は出ない")

    print()
    print("═" * W)
    print("  診断2: 決済が利益を取り逃していないか（MFE / MAE）")
    print("═" * W)
    if "mfe_r" in sig:
        mfe, mae = sig["mfe_r"].dropna(), sig["mae_r"].dropna()
        print(f"\n  エントリー後20営業日の含み益・含み損（R単位）\n")
        print(f"    最大含み益(MFE) 中央値   {mfe.median():>7.2f} R")
        print(f"    最大含み益(MFE) 平均     {mfe.mean():>7.2f} R")
        print(f"    最大含み損(MAE) 中央値   {mae.median():>7.2f} R")
        print(f"    最大含み損(MAE) 平均     {mae.mean():>7.2f} R")
        print()
        print(f"    MFE ≥ 1.0R に到達        {(mfe >= 1.0).mean():>7.1%}")
        print(f"    MFE ≥ 2.0R に到達        {(mfe >= 2.0).mean():>7.1%}")
        print(f"    MFE ≥ 2.5R に到達        {(mfe >= 2.5).mean():>7.1%}")
        print(f"    MAE ≤ -1.0R（損切り圏）   {(mae <= -1.0).mean():>7.1%}")
        print()
        print("    ※ 実際のバックテストの平均利益は +0.94R でした。")
        print("       MFEの中央値がこれを大きく上回るなら、決済が早すぎて")
        print("       伸びる余地を捨てていることになります。")

    print()
    print("═" * W)
    print("  診断3: セットアップ別のフォワードリターン（10日）")
    print("═" * W)
    print()
    print("    " + pad("セットアップ", 26) + rpad("件数", 6) + rpad("10日平均", 12)
          + rpad("勝率", 9) + rpad("MFE中央値", 12))
    for name, g in sig.groupby("setup"):
        r10 = g["r10"].dropna()
        if r10.empty:
            continue
        m = g["mfe_r"].dropna()
        print("    " + pad(name, 26) + rpad(len(g), 6)
              + rpad("{:+.2%}".format(r10.mean()), 12)
              + rpad("{:.1%}".format((r10 > 0).mean()), 9)
              + rpad("{:.2f} R".format(m.median()) if not m.empty else "-", 12))

    print()
    print("═" * W)
    print("  診断4: 市場環境別のフォワードリターン（10日）")
    print("═" * W)
    print()
    print("    " + pad("市場環境", 26) + rpad("件数", 6) + rpad("10日平均", 12)
          + rpad("勝率", 9))
    for name, g in sig.groupby("regime"):
        r10 = g["r10"].dropna()
        if r10.empty:
            continue
        print("    " + pad(name, 26) + rpad(len(g), 6)
              + rpad("{:+.2%}".format(r10.mean()), 12)
              + rpad("{:.1%}".format((r10 > 0).mean()), 9))

    print()
    print("═" * W)
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 診断")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--period", default="10y")
    args = ap.parse_args()

    syms = args.symbols if args.symbols else bt.DEFAULT_UNIVERSE
    syms = [s.strip().upper() for s in syms]
    syms = [s + ".T" if s.isdigit() and len(s) == 4 else s for s in syms]
    return run(syms, args.period)


if __name__ == "__main__":
    sys.exit(main())
