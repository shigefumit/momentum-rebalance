#!/usr/bin/env python3
"""前回との差分を出す。自動実行の通知本文に使う。

入れ替えが必要な月だけ通知を出したいので、前回のデータと比較して
「変わったかどうか」を終了コードで返す。

    終了コード 0 … 上位銘柄に変化あり（通知する）
    終了コード 2 … 変化なし（通知しない）
    終了コード 1 … 比較できなかった

前回のデータは git から取る（git show HEAD:app_data.json）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CUR = Path(__file__).parent / "app_data.json"
OUT = Path(__file__).parent / "report.md"


def prev_data() -> dict | None:
    try:
        r = subprocess.run(["git", "show", "HEAD:app_data.json"],
                           capture_output=True, text=True, cwd=Path(__file__).parent)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def main() -> int:
    if not CUR.exists():
        print("app_data.json がありません")
        return 1

    cur = json.loads(CUR.read_text())
    prev = prev_data()

    n = cur.get("top_n", 5)
    rank = cur["ranking"]
    top = rank[:n]
    top_syms = [r["symbol"] for r in top]
    old_syms = [r["symbol"] for r in prev["ranking"][:prev.get("top_n", n)]] if prev else []

    sell = [s for s in old_syms if s not in top_syms]
    buy = [s for s in top_syms if s not in old_syms]
    keep = [s for s in old_syms if s in top_syms]

    name = {r["symbol"]: r["name"] for r in rank}
    if prev:
        name.update({r["symbol"]: r.get("name", r["symbol"]) for r in prev["ranking"]})

    L: list[str] = []
    L.append(f"基準日 **{cur['as_of']}** ／ ドル円 {cur['usdjpy']:.2f}")
    L.append("")

    if not prev:
        L.append(f"### 初回。上位{n}銘柄を買い付けてください")
    elif not sell and not buy:
        L.append("### 入れ替え不要。そのまま保有を継続してください")
    else:
        L.append(f"### {len(sell)}銘柄を売り、{len(buy)}銘柄を買う")
    L.append("")

    if sell:
        L.append("**売る**")
        for s in sell:
            rk = next((r["rank"] for r in rank if r["symbol"] == s), None)
            pos = f"{rk}位に転落" if rk else "順位圏外"
            L.append(f"- `{s}` {name.get(s, s)} — {pos}")
        L.append("")
    if buy:
        L.append("**買う**")
        for s in buy:
            r = next(r for r in rank if r["symbol"] == s)
            qty = (f"{r['shares']:,}株 / {r['cost_jpy']:,.0f}円"
                   if r.get("buildable") else "**買付不可**（1単元が予算超過）")
            L.append(f"- `{s}` {r['name']} — {r['rank']}位 {r['return_12m']:+.1%} — {qty}")
        L.append("")
    if keep:
        L.append("**継続保有**: " + "、".join(f"`{s}`" for s in keep))
        L.append("")

    L.append(f"### 今月の上位{n}銘柄")
    L.append("")
    L.append("| 順 | 銘柄 | セクター | 12ヶ月 | 株数 | 必要額 |")
    L.append("|---:|---|---|---:|---:|---:|")
    total = 0
    for r in top:
        if r.get("buildable"):
            q = f"{r['shares']:,}"
            c = f"{r['cost_jpy']:,.0f}円"
            total += r["cost_jpy"]
        else:
            q, c = "―", "買付不可"
        L.append(f"| {r['rank']} | `{r['symbol']}` {r['name']} | {r['sector']} | "
                 f"{r['return_12m']:+.1%} | {q} | {c} |")
    L.append(f"| | **合計** | | | | **{total:,.0f}円** |")
    L.append("")

    # セクター集中の警告
    cnt: dict[str, int] = {}
    for r in top:
        cnt[r["sector"]] = cnt.get(r["sector"], 0) + 1
    worst = max(cnt.items(), key=lambda x: x[1])
    if worst[1] >= max(2, (n * 3 + 4) // 5):
        L.append(f"> **注意**: {worst[1]}銘柄が「{worst[0]}」に集中しています。"
                 f"このルールはセクター分散を考慮しないため、{worst[0]}が崩れると"
                 f"同時に下落します。")
        L.append("")

    L.append("---")
    L.append("")
    L.append("アプリ: https://shigefumit.github.io/momentum-rebalance/")
    L.append("")
    L.append("発注は手作業で行ってください。自動発注は実装していません "
             "（データ源が非公式スクレイピングで、失敗時に例外ではなく空データを返すため）。")

    body = "\n".join(L)
    OUT.write_text(body)
    print(body)

    changed = bool(not prev or sell or buy)
    return 0 if changed else 2


if __name__ == "__main__":
    sys.exit(main())
