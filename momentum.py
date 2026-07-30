#!/usr/bin/env python3
"""モメンタム・リバランス判定（本番用）。

ルール:
    毎月1回、過去12ヶ月のリターン上位5銘柄を等ウェイトで持つ。それだけ。

このルールが選ばれた理由（research.py の検証結果）:
    2006-2015: +339%（何もしない場合 +178%）
    2016-2026: +1,956%（何もしない場合 +1,308%）
    リーマンショックを含む10年とAI相場の10年、両方で「何もしない」を上回った。
    最大ドローダウンは -58% / -42%。

意図的に入れていないもの（入れると壊れる）:
    - タイミング判定  検証で一貫して損だった。モメンタム単独 +1,956% に対し
                      「弱気相場で現金に逃げる」を足すと +636% に落ちた。
                      後知恵で勝ち組を選んだ場合でも +11,208% → +1,579% に落ちた。
    - 損切り          月1回の入れ替えが唯一の決済ルール。個別の損切りは持たない
    - AI判断          このルールは完全に機械的。AIを挟むと再現性が失われる
    - 指値            寄り付き成行で入れ替える想定

使い方:
    .venv/bin/python momentum.py                 # 今月の判定
    .venv/bin/python momentum.py --top 3         # 上位3銘柄版
    .venv/bin/python momentum.py --json out.json # PWA用のデータを書き出す
    .venv/bin/python momentum.py --set NVDA MU LLY NFLX AVGO   # 現在の保有を登録
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import fetcher
import research

HOLDINGS_FILE = Path(__file__).parent / "holdings.json"

LOOKBACK_MONTHS = 12
LOOKBACK_DAYS = LOOKBACK_MONTHS * 21
DEFAULT_TOP = 5

# 実勢価格は全銘柄で取得する。
# 保有株が順位を落ちても評価額を計算できるようにするため（上位だけでは足りない）。
# 月1回の処理なので取得時間は許容範囲。ローカルは6時間キャッシュされる。

# 銘柄の日本語名。判定結果を読む時に銘柄コードだけだと分かりにくいため。
NAMES = {
    "NVDA": "エヌビディア", "AMD": "AMD", "AVGO": "ブロードコム", "MU": "マイクロン",
    "AMAT": "アプライドマテリアルズ", "LRCX": "ラムリサーチ", "ADI": "アナログデバイセズ",
    "TXN": "テキサスインスツルメンツ", "QCOM": "クアルコム", "INTC": "インテル",
    "AAPL": "アップル", "MSFT": "マイクロソフト", "GOOGL": "アルファベット",
    "AMZN": "アマゾン", "META": "メタ", "NFLX": "ネットフリックス",
    "ADBE": "アドビ", "CRM": "セールスフォース", "ORCL": "オラクル", "CSCO": "シスコ",
    "TSLA": "テスラ", "V": "ビザ", "MA": "マスターカード",
    "JPM": "JPモルガン", "BAC": "バンクオブアメリカ", "GS": "ゴールドマンサックス",
    "LLY": "イーライリリー", "UNH": "ユナイテッドヘルス", "ABBV": "アッヴィ",
    "MRK": "メルク", "PFE": "ファイザー", "JNJ": "ジョンソン&ジョンソン",
    "XOM": "エクソンモービル", "CVX": "シェブロン",
    "WMT": "ウォルマート", "COST": "コストコ", "PG": "P&G", "KO": "コカコーラ",
    "PEP": "ペプシコ", "MCD": "マクドナルド", "HD": "ホームデポ", "NKE": "ナイキ",
    "CAT": "キャタピラー", "BA": "ボーイング", "MMM": "スリーエム",
    "IBM": "IBM", "GE": "GE", "DIS": "ディズニー", "VZ": "ベライゾン",
    "XLK": "米テクノロジーETF", "XLF": "米金融ETF", "XLE": "米エネルギーETF",
    "XLV": "米ヘルスケアETF", "XLP": "米生活必需品ETF", "XLI": "米資本財ETF",
    "XLY": "米一般消費財ETF", "XLU": "米公共ETF", "XLB": "米素材ETF",
    "7203.T": "トヨタ自動車", "8306.T": "三菱UFJ", "9432.T": "NTT",
    "6758.T": "ソニーグループ", "4063.T": "信越化学", "6861.T": "キーエンス",
    "9984.T": "ソフトバンクG", "8035.T": "東京エレクトロン", "4502.T": "武田薬品",
    "6367.T": "ダイキン工業",
}


# セクター分類。このルールはセクター分散を一切考慮しないため、
# 上位5銘柄が同一セクターに固まることがある（実際 2026年7月は5銘柄すべて半導体）。
# その場合はそのセクターが崩れると5銘柄同時に下落する。警告を出すために持つ。
SECTORS = {
    "NVDA": "半導体", "AMD": "半導体", "AVGO": "半導体", "MU": "半導体",
    "AMAT": "半導体", "LRCX": "半導体", "ADI": "半導体", "TXN": "半導体",
    "QCOM": "半導体", "INTC": "半導体", "8035.T": "半導体", "4063.T": "半導体",
    "AAPL": "テック", "MSFT": "テック", "GOOGL": "テック", "AMZN": "テック",
    "META": "テック", "NFLX": "テック", "ADBE": "テック", "CRM": "テック",
    "ORCL": "テック", "CSCO": "テック", "IBM": "テック", "XLK": "テック",
    "6758.T": "テック", "9984.T": "テック", "6861.T": "テック",
    "TSLA": "自動車", "7203.T": "自動車",
    "V": "金融", "MA": "金融", "JPM": "金融", "BAC": "金融", "GS": "金融",
    "XLF": "金融", "8306.T": "金融",
    "LLY": "ヘルスケア", "UNH": "ヘルスケア", "ABBV": "ヘルスケア",
    "MRK": "ヘルスケア", "PFE": "ヘルスケア", "JNJ": "ヘルスケア",
    "XLV": "ヘルスケア", "4502.T": "ヘルスケア",
    "XOM": "エネルギー", "CVX": "エネルギー", "XLE": "エネルギー",
    "WMT": "生活必需品", "COST": "生活必需品", "PG": "生活必需品",
    "KO": "生活必需品", "PEP": "生活必需品", "XLP": "生活必需品",
    "MCD": "消費", "HD": "消費", "NKE": "消費", "DIS": "消費", "XLY": "消費",
    "CAT": "資本財", "BA": "資本財", "MMM": "資本財", "GE": "資本財",
    "XLI": "資本財", "6367.T": "資本財",
    "VZ": "通信", "9432.T": "通信", "XLU": "公共", "XLB": "素材",
}


def sector_of(sym: str) -> str:
    return SECTORS.get(sym, "その他")


def dwidth(s) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1
               for c in str(s))


def pad(s, w: int) -> str:
    s = str(s)
    while dwidth(s) > w:
        s = s[:-1]
    return s + " " * max(0, w - dwidth(s))


def rpad(s, w: int) -> str:
    s = str(s)
    return " " * max(0, w - dwidth(s)) + s


def name_of(sym: str) -> str:
    return NAMES.get(sym, sym)


def load_holdings() -> dict:
    if HOLDINGS_FILE.exists():
        try:
            return json.loads(HOLDINGS_FILE.read_text())
        except Exception:
            pass
    return {"symbols": [], "updated": None}


def save_holdings(symbols: list) -> None:
    HOLDINGS_FILE.write_text(json.dumps(
        {"symbols": symbols, "updated": datetime.now().strftime("%Y-%m-%d")},
        ensure_ascii=False, indent=2))


def compute(top_n: int, capital: float, as_of_str: str | None = None) -> dict:
    """順位付けと株数を計算する。

    2種類の株価を使い分ける:
      - 順位付け     配当込み・円換算のトータルリターン（検証と同じ基準にするため）
      - 株数の計算   実際の市場価格（配当調整後の価格では実際に買えない）

    as_of_str を渡すとその日付時点で判定する。「2026年7月31日の終値までのデータで
    ルールは何と言ったか」を後から再現できる。バックテストと同じ手順
    （その日の終値で判定 → 翌営業日に売買）なので、検証結果と直接突き合わせられる。

    「12ヶ月」は 252営業日（12 × 21）で測る。カレンダーの1年とは数日ずれるが、
    検証と本番で同じ定義を使うことを優先している。
    """
    symbols = research.US + research.ETF + research.JP
    print(f"  データ取得中（{len(symbols)}銘柄）…")
    panel, _ = research.load_panel(symbols, quiet=True)

    if as_of_str:
        cut = pd.Timestamp(as_of_str)
        panel = panel.loc[:cut]
        if panel.empty:
            raise RuntimeError(f"{as_of_str} 以前のデータがありません")

    as_of = panel.index[-1]
    if len(panel) < LOOKBACK_DAYS + 5:
        raise RuntimeError("データが不足しています")

    # ---- 12ヶ月リターンで順位付け（カレンダー基準）
    #
    # 「営業日 × 21」で数えると誤差が出る。日米の営業日を1つの表にまとめると
    # 日付が両市場の和集合になり、1年あたりの行数が252より多くなるため、
    # 252行遡っても約11.6ヶ月しか戻らない。丸1年前の終値と比較する。
    pos = int(panel.index.searchsorted(as_of - pd.DateOffset(months=LOOKBACK_MONTHS)))
    if pos >= len(panel) - 5:
        raise RuntimeError(f"{LOOKBACK_MONTHS}ヶ月分のデータがありません")
    now = panel.iloc[-1]
    then = panel.iloc[pos]
    lookback_from = panel.index[pos]
    ret = ((now / then) - 1.0).dropna().sort_values(ascending=False)

    # 測定開始時点で実際にデータがある銘柄だけを対象にする
    valid = [s for s in ret.index
             if panel[s].loc[:lookback_from].dropna().size > 0]
    ret = ret.loc[valid]

    top = list(ret.head(top_n).index)

    # ---- 実際の市場価格を取る（株数計算用）
    print(f"  実勢価格を取得中（全{len(ret)}銘柄）…")
    fx = fetcher.get_usdjpy(as_of=as_of if as_of_str else None, period="max")
    rows = []
    per_pos = capital / top_n
    for rank, sym in enumerate(ret.index, 1):
        market = "JP" if sym.endswith(".T") else "US"
        lot = config.LOT_SIZE[market]
        rate = 1.0 if market == "JP" else fx

        try:
            px = fetcher.get_prices(sym, period="max" if as_of_str else "1mo")
            if as_of_str:
                px = px.loc[:as_of]
            price_local = float(px["Close"].iloc[-1])
        except Exception:
            price_local = float("nan")

        entry = {
            "rank": rank,
            "symbol": sym,
            "name": name_of(sym),
            "sector": sector_of(sym),
            "return_12m": float(ret[sym]),
            "selected": sym in top,
            "market": market,
            "lot": lot,
        }

        if not math.isnan(price_local) and price_local > 0:
            price_jpy = price_local * rate
            shares = int(math.floor((per_pos / price_jpy) / lot) * lot)
            entry.update({
                "price_local": price_local,
                "price_jpy": price_jpy,
                "shares": shares,
                "cost_jpy": shares * price_jpy,
                "buildable": shares > 0,
                "min_cost_jpy": lot * price_jpy,
            })
        else:
            entry.update({"price_local": None, "price_jpy": None,
                          "shares": 0, "buildable": False})

        rows.append(entry)

    held = load_holdings()
    cur = held.get("symbols", [])
    return {
        "as_of": str(as_of.date()),
        "lookback_from": str(lookback_from.date()),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "capital": capital,
        "top_n": top_n,
        "lookback_months": LOOKBACK_MONTHS,
        "usdjpy": fx,
        "ranking": rows,
        "top": top,
        "holdings": cur,
        "holdings_updated": held.get("updated"),
        "to_sell": [s for s in cur if s not in top],
        "to_buy": [s for s in top if s not in cur],
        "to_keep": [s for s in cur if s in top],
        "per_position_jpy": per_pos,
    }


def render(d: dict) -> None:
    W = 72
    print()
    print("═" * W)
    print(f"  モメンタム・リバランス判定")
    print(f"  基準日 {d['as_of']}（この日の終値で判定 → 翌営業日に売買）")
    print(f"  測定区間 {d.get('lookback_from', '?')} 〜 {d['as_of']}"
          f"（カレンダーで{d['lookback_months']}ヶ月）")
    print(f"  ルール: このリターン上位{d['top_n']}銘柄を等ウェイト")
    print("═" * W)

    print(f"\n  今月持つべき {d['top_n']} 銘柄")
    print("  " + "─" * (W - 2))
    print("  " + pad("順", 4) + pad("銘柄", 10) + pad("名称", 22)
          + rpad("12ヶ月", 9) + rpad("株数", 8) + rpad("必要額", 13))
    for r in d["ranking"]:
        if not r["selected"]:
            continue
        if r.get("buildable"):
            sh = "{:,}".format(r["shares"])
            cost = "{:,.0f}円".format(r["cost_jpy"])
        else:
            sh, cost = "―", "買付不可"
        print("  " + pad(r["rank"], 4) + pad(r["symbol"], 10)
              + pad(r["name"], 22) + rpad("{:+.1%}".format(r["return_12m"]), 9)
              + rpad(sh, 8) + rpad(cost, 13))

    total = sum(r.get("cost_jpy", 0) for r in d["ranking"] if r["selected"])
    print("  " + "─" * (W - 2))
    print("  " + pad("合計", 36) + rpad("", 9) + rpad("", 8)
          + rpad("{:,.0f}円".format(total), 13))
    print(f"  1銘柄あたりの予算 {d['per_position_jpy']:,.0f}円"
          f"（資金 {d['capital']:,.0f}円 ÷ {d['top_n']}）")

    # 買えない銘柄の警告
    bad = [r for r in d["ranking"] if r["selected"] and not r.get("buildable")]
    if bad:
        print()
        for r in bad:
            mc = r.get("min_cost_jpy")
            if mc:
                print(f"  [注意] {r['symbol']}（{r['name']}）は1単元 {r['lot']}株 = "
                      f"{mc:,.0f}円 で、1銘柄予算 {d['per_position_jpy']:,.0f}円 を超えます。"
                      f"単元未満株が使えないなら次順位に繰り下げてください")
            else:
                print(f"  [注意] {r['symbol']} の現在値が取得できませんでした")

    # ---- 乗り換え指示
    print()
    print("─" * W)
    if not d["holdings"]:
        print("  現在の保有が未登録です。")
        print("  初回は上記5銘柄を買い付け、その後こう登録してください:")
        print(f"    .venv/bin/python momentum.py --set {' '.join(d['top'])}")
    elif not d["to_sell"] and not d["to_buy"]:
        print("  今月の入れ替えは【不要】です。そのまま保有を継続してください。")
        print(f"  保有中: {'、'.join(name_of(s) for s in d['holdings'])}")
    else:
        print("  今月の入れ替え")
        print()
        if d["to_sell"]:
            print("    売る:")
            for s in d["to_sell"]:
                rk = next((r["rank"] for r in d["ranking"] if r["symbol"] == s), "?")
                print(f"      {pad(s, 9)} {pad(name_of(s), 20)} 現在{rk}位に転落")
        if d["to_buy"]:
            print("    買う:")
            for s in d["to_buy"]:
                r = next(r for r in d["ranking"] if r["symbol"] == s)
                sh = "{:,}株".format(r["shares"]) if r.get("buildable") else "買付不可"
                print(f"      {pad(s, 9)} {pad(name_of(s), 20)} {rk_fmt(r)}  {sh}")
        if d["to_keep"]:
            print("    継続保有:")
            for s in d["to_keep"]:
                print(f"      {pad(s, 9)} {name_of(s)}")
        print()
        print(f"    入れ替え後はこう登録してください:")
        print(f"      .venv/bin/python momentum.py --set {' '.join(d['top'])}")

    # ---- 順位表（上位20）
    print()
    print("─" * W)
    print("  全銘柄の12ヶ月リターン順位（上位20）")
    print()
    for r in d["ranking"][:20]:
        mark = " ★" if r["selected"] else "  "
        print("  " + rpad(r["rank"], 3) + mark + " " + pad(r["symbol"], 9)
              + pad(r["name"], 22) + rpad("{:+.1%}".format(r["return_12m"]), 9))

    print()
    print("═" * W)
    print("  このルールについて必ず覚えておくこと")
    print("═" * W)
    print("   ・過去の検証で最大 −42%〜−58% の下落局面がありました。")
    print("     資産が一時的に半分近くになる場面を通過する前提の戦略です。")
    print("   ・タイミング判定は意図的に入れていません。検証で一貫して損だったためです。")
    print("   ・月1回だけ実行してください。頻繁に見ても意味がなく、手数料と税金が増えます。")
    print("   ・検証は現在も上場している銘柄のみで行っており、実際より good に出ています。")
    print("   ・自分専用です。他人に有償で売買判断を提供すると投資助言業の登録が必要です。")
    print("═" * W)
    print()


def rk_fmt(r: dict) -> str:
    return "{:>2}位 {:+.1%}".format(r["rank"], r["return_12m"])


def main() -> int:
    ap = argparse.ArgumentParser(description="モメンタム・リバランス判定")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP, help="保有銘柄数（既定5）")
    ap.add_argument("--capital", type=float, default=config.CAPITAL_JPY)
    ap.add_argument("--json", default=None, help="PWA用データの書き出し先")
    ap.add_argument("--set", nargs="*", default=None, help="現在の保有銘柄を登録")
    ap.add_argument("--asof", default=None,
                    help="この日付の終値まででの判定を再現する（例 2026-06-30）")
    args = ap.parse_args()

    if args.set is not None:
        syms = [s.strip().upper() for s in args.set]
        syms = [s + ".T" if s.isdigit() and len(s) == 4 else s for s in syms]
        save_holdings(syms)
        print(f"\n  保有を登録しました: {'、'.join(name_of(s) for s in syms)}\n")
        return 0

    try:
        d = compute(args.top, args.capital, args.asof)
    except Exception as e:
        print(f"\n  [エラー] {type(e).__name__}: {e}\n")
        return 1

    render(d)

    if args.json:
        Path(args.json).write_text(json.dumps(d, ensure_ascii=False, indent=2))
        print(f"  データを書き出しました: {args.json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
