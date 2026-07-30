#!/usr/bin/env python3
"""PWA（iPhoneホーム画面用アプリ）のHTMLを生成する。

毎月の運用手順:
    1. .venv/bin/python momentum.py --json app_data.json
    2. .venv/bin/python build_app.py
    3. git push（GitHub Pages に反映される）

画面の中身は app_template.html にある。このスクリプトはそこに
app_data.json を埋め込んで2種類のHTMLを吐くだけ。

データを埋め込む方式にしている理由:
  - 月1回しか変わらないルールなので、常時通信する必要がない
  - 通信しないので API 障害・レート制限・CORS の影響を受けない
  - オフラインでも開ける（電車の中でも確認できる）
  - 保有情報は端末内（localStorage）にのみ残り、どこにも送信されない
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "app_data.json"
TEMPLATE = HERE / "app_template.html"
OUT = HERE / "app.html"
DOCS = HERE / "docs" / "index.html"


def standalone(body: str) -> str:
    """GitHub Pages 用に完全なHTML文書として出力する。

    Artifact は <head>/<body> を自動で付けてくれるが、GitHub Pages は
    素のHTMLをそのまま配信するため、こちらでは骨格を自分で書く必要がある。
    """
    head_part, rest = body.split("<style>", 1)
    css, tail = rest.split("</style>", 1)
    return (
        '<!doctype html>\n<html lang="ja">\n<head>\n'
        '<meta charset="utf-8" />\n'
        '<meta name="theme-color" content="#1e46c8" />\n'
        '<meta name="apple-mobile-web-app-capable" content="yes" />\n'
        '<meta name="apple-mobile-web-app-title" content="モメンタム" />\n'
        '<meta name="robots" content="noindex" />\n'
        + head_part
        + "<style>" + css + "</style>\n"
        "</head>\n<body>\n"
        + tail
        + "\n</body>\n</html>\n"
    )


def main() -> int:
    if not DATA.exists():
        print(f"\n  [エラー] {DATA.name} がありません。先にこれを実行してください:")
        print(f"    .venv/bin/python momentum.py --json {DATA.name}\n")
        return 1
    if not TEMPLATE.exists():
        print(f"\n  [エラー] {TEMPLATE.name} がありません\n")
        return 1

    data = json.loads(DATA.read_text())
    tpl = TEMPLATE.read_text()

    if "__DATA__" not in tpl:
        print(f"\n  [エラー] {TEMPLATE.name} に __DATA__ の差し込み位置がありません\n")
        return 1

    # </script> がJSON内に現れるとHTMLが壊れるのでエスケープ
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = tpl.replace("__DATA__", payload)

    OUT.write_text(html)                       # Artifact 用（head/body は自動付与される）
    DOCS.parent.mkdir(exist_ok=True)
    DOCS.write_text(standalone(html))          # GitHub Pages 用（骨格つき）

    sel = [r for r in data["ranking"] if r["selected"]]
    priced = sum(1 for r in data["ranking"] if r.get("price_jpy"))
    print(f"\n  {OUT.name:<16} {OUT.stat().st_size / 1024:>5.0f} KB")
    print(f"  docs/index.html  {DOCS.stat().st_size / 1024:>5.0f} KB（GitHub Pages 用）")
    print(f"  基準日 {data['as_of']} / 測定開始 {data.get('lookback_from', '?')}")
    print(f"  価格付き {priced}/{len(data['ranking'])} 銘柄"
          f" / 上位{data['top_n']}: {'、'.join(r['symbol'] for r in sel)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
