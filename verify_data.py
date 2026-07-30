#!/usr/bin/env python3
"""生成データの健全性検査。自動実行が壊れたデータを公開するのを防ぐ。

なぜ必要か:
  yfinance は非公式スクレイピングで、失敗しても例外を投げず「空データ」を返す。
  そのため「取得に失敗した」と「その銘柄にデータがない」の区別がつかない。
  自動実行では人間が目視しないので、静かに壊れたまま公開されうる。

このプロジェクトで実際に踏んだバグを、それぞれ名指しで検査する:
  検査3  日米カレンダーのズレ（順位が片方の市場だけになる）
  検査4  価格取得の失敗（株数が計算できない）
  検査5  ドル円の異常値
  検査6  基準日が古すぎる（取得が止まっている）

異常があれば終了コード1を返す。GitHub Actions はここで止まり、
コミットせずに失敗通知を出す。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).parent / "app_data.json"

MIN_SYMBOLS = 60          # ユニバースは68銘柄。60未満なら取得が大きく欠けている
MAX_AGE_DAYS = 10         # 基準日がこれより古ければ取得が止まっている
FX_RANGE = (80.0, 300.0)  # ドル円の妥当な範囲


def main() -> int:
    if not DATA.exists():
        print(f"[NG] {DATA.name} がありません")
        return 1

    try:
        d = json.loads(DATA.read_text())
    except Exception as e:
        print(f"[NG] JSONとして読めません: {e}")
        return 1

    fails: list[str] = []
    warns: list[str] = []
    rank = d.get("ranking", [])
    top_n = d.get("top_n", 5)
    top = rank[:top_n]

    # ── 検査1: 銘柄数
    n = len(rank)
    ok = n >= MIN_SYMBOLS
    print(f"[{'OK' if ok else 'NG'}] 検査1 順位に載った銘柄数: {n}（{MIN_SYMBOLS}以上が必要）")
    if not ok:
        fails.append(f"銘柄数が {n} しかない。取得が大きく欠けている")

    # ── 検査2: リターンが妥当か
    rets = [r.get("return_12m") for r in rank if isinstance(r.get("return_12m"), (int, float))]
    if len(rets) != n:
        fails.append(f"12ヶ月リターンが計算できていない銘柄が {n - len(rets)} 件ある")
        print(f"[NG] 検査2 リターンの計算: {len(rets)}/{n} 件のみ")
    elif len(set(round(x, 6) for x in rets)) < max(2, n // 4):
        fails.append("リターンの値がほぼ同一。計算が壊れている可能性がある")
        print("[NG] 検査2 リターンの多様性: 値がほぼ同一")
    else:
        print(f"[OK] 検査2 リターンの計算: {len(rets)}/{n} 件（{min(rets):+.1%}〜{max(rets):+.1%}）")

    # ── 検査3: 日米カレンダーのズレ（実際に踏んだバグ）
    #
    # 最新日が日本の取引日で米国が休場だと、米国株が全部 NaN になり
    # 「上位5銘柄すべて日本株」に化ける。市場ごとの計上数で検知する。
    jp = [r for r in rank if r["symbol"].endswith(".T")]
    us = [r for r in rank if not r["symbol"].endswith(".T")]
    ok = len(jp) >= 5 and len(us) >= 40
    print(f"[{'OK' if ok else 'NG'}] 検査3 市場ごとの計上: 米国 {len(us)}銘柄 / 日本 {len(jp)}銘柄")
    if not ok:
        fails.append(
            f"片方の市場が欠落している（米国{len(us)}/日本{len(jp)}）。"
            f"日米カレンダーのズレで片側が NaN になっている疑いがある"
        )

    # ── 検査4: 上位銘柄の価格と株数
    bad = [r["symbol"] for r in top if not r.get("price_jpy")]
    if bad:
        fails.append(f"上位{top_n}銘柄のうち価格が取れていないものがある: {bad}")
        print(f"[NG] 検査4 上位の価格取得: {bad} が欠落")
    else:
        unbuildable = [r["symbol"] for r in top if not r.get("buildable")]
        print(f"[OK] 検査4 上位{top_n}銘柄すべてに価格あり"
              + (f"（うち買付不可: {unbuildable}）" if unbuildable else ""))
        if len(unbuildable) >= top_n:
            fails.append("上位銘柄がすべて買付不可。資金設定か価格取得に問題がある")
        elif unbuildable:
            warns.append(f"買付不可の銘柄がある: {unbuildable}（1単元が予算超過）")

    # ── 検査5: ドル円
    fx = d.get("usdjpy")
    ok = isinstance(fx, (int, float)) and FX_RANGE[0] < fx < FX_RANGE[1]
    print(f"[{'OK' if ok else 'NG'}] 検査5 ドル円: {fx}（{FX_RANGE[0]}〜{FX_RANGE[1]}が妥当）")
    if not ok:
        fails.append(f"ドル円が異常値: {fx}")

    # ── 検査6: 基準日の新しさ
    try:
        as_of = datetime.strptime(d["as_of"], "%Y-%m-%d")
        age = (datetime.now() - as_of).days
        ok = age <= MAX_AGE_DAYS
        print(f"[{'OK' if ok else 'NG'}] 検査6 基準日: {d['as_of']}（{age}日前 / "
              f"{MAX_AGE_DAYS}日以内が必要）")
        if not ok:
            fails.append(f"基準日が {age} 日前。データ取得が止まっている疑いがある")
    except Exception as e:
        fails.append(f"基準日が読めない: {e}")
        print(f"[NG] 検査6 基準日: 読めない")

    # ── 結果
    print()
    if warns:
        for w in warns:
            print(f"  警告: {w}")
        print()
    if fails:
        print("=" * 60)
        print("検査に失敗しました。データを公開しません。")
        print("=" * 60)
        for f in fails:
            print(f"  ・{f}")
        print()
        return 1

    sel = "、".join(r["symbol"] for r in top)
    print("=" * 60)
    print(f"全検査を通過しました。上位{top_n}銘柄: {sel}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
