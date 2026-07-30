#!/usr/bin/env python3
"""Phase 1b: 期間分割による構造仮説の検証。

Phase 1 で判明したこと（FINDINGS.md）:
  - エントリーのエッジは最大 +0.37%（10日）で往復コスト0.20%と同程度
  - MAE中央値 −1.54R に対し損切りが −1.0R。63%が機能する前に振り落とされる
  - MFE中央値は +2.22R あり、放っておけば機能したトレードが多い
  - 「+1Rで半分売却＋建値撤退」が勝ちを +0.5R に切り詰めていた
  - ギャップが13.6%で発生し −1.995R（設計の2倍）

この診断から導いた設計仮説を、期間を分けて検証する。

═══ 検証プロトコル（走らせる前に確定。後から変更しない）═══

  1. 開発期間で5つの仮説を比較する
  2. 開発期間の期待値が最も高い「1つ」を選ぶ（トレード数100件以上を条件とする）
  3. 選んだ1つを検証期間で「一度だけ」測る ← これが本当のテスト
  4. 判定:
       検証期間の期待値 > 0 かつ 開発期間から大きく劣化していない → エッジは本物
       検証期間で崩れた                                          → 当てはめだった
  5. 参考として全仮説の検証期間成績も表示するが、判定には使わない
     （全部見てから選ぶと、それは検証ではなく2回目の開発になる）

仮説は5つだけ。走らせながら追加しない。試行を増やすほど「たまたま良かったもの」を
選ぶ確率が上がり、期間を分けた意味が失われる。

使い方:
    .venv/bin/python validate.py
    .venv/bin/python validate.py --period 10y
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest as bt
import config

# ═══════════════════════════════════════════════ 期間の定義（固定）
IS_START, IS_END = "2016-01-01", "2021-12-31"    # 開発期間: ここで仮説を選ぶ
OOS_START, OOS_END = "2022-01-01", "2026-07-31"  # 検証期間: 一度だけ測る

MIN_TRADES_FOR_SELECTION = 100   # これ未満の仮説は開発期間の選定対象にしない


# ═══════════════════════════════════════════════ 仮説の定義（事前確定）
@dataclass
class Variant:
    key: str
    name: str
    rationale: str
    overrides: dict = field(default_factory=dict)


VARIANTS = [
    Variant(
        "V0", "ベースライン（現行）",
        "PLAN.md の設計そのまま。比較の基準",
        {},
    ),
    Variant(
        "V1", "損切りを広げる",
        "MAE中央値 −1.54R（=約3.1×ATR）。損切りをノイズの外側に置く",
        {
            "MIN_STOP_ATR": 2.0,
            "ATR_MULT": {"short": 2.5, "swing": 3.0, "long": 4.0},
        },
    ),
    Variant(
        "V2", "半分売却を廃止",
        "+1R半分売却＋建値撤退が勝ちを +0.5R に切り詰めていた。目標か損切りまで持つ",
        {
            "USE_TP1_HALF": False,
            "USE_TRAILING": False,
        },
    ),
    Variant(
        "V3", "損切り拡大 ＋ 半分売却廃止",
        "V1 と V2 の組み合わせ。損切りを外に出し、勝ちを切らない",
        {
            "MIN_STOP_ATR": 2.0,
            "ATR_MULT": {"short": 2.5, "swing": 3.0, "long": 4.0},
            "USE_TP1_HALF": False,
            "USE_TRAILING": False,
        },
    ),
    Variant(
        "V4", "V3 ＋ 保有期間延長 ＋ 200日線決済を外す",
        "エッジ対コスト比は保有を延ばすと改善。200日線決済は31件で −0.584R の損失源だった",
        {
            "MIN_STOP_ATR": 2.0,
            "ATR_MULT": {"short": 2.5, "swing": 3.0, "long": 4.0},
            "USE_TP1_HALF": False,
            "USE_TRAILING": False,
            "USE_SMA200_EXIT": False,
            "MAX_HOLD_BARS": {"short": 20, "swing": 120, "long": 250},
        },
    ),
]


# ═══════════════════════════════════════════════ 補助
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


@contextmanager
def apply_overrides(ov: dict):
    """config を一時的に差し替える。全コードが呼び出し時に config を読むため、
    同じコードパスをそのまま通せる（live と検証でロジックが分岐しない）。"""
    old = {k: getattr(config, k) for k in ov}
    try:
        for k, v in ov.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def summarize(trades, curve, capital) -> dict:
    if not trades:
        return {"trades": 0, "expectancy_r": 0.0, "win_rate": 0.0,
                "total_return": 0.0, "max_dd": 0.0, "profit_factor": 0.0,
                "avg_bars": 0.0, "gap_rate": 0.0, "after_tax_return": 0.0}
    m = bt.metrics(trades, curve, capital)
    tax, gross, _ = bt.apply_tax(trades)
    m["after_tax_return"] = (capital + gross - tax) / capital - 1
    return m


def _curve_stats(curve: pd.Series) -> dict:
    """リターンだけでなくリスクも測る。ドローダウンを見ずにリターンを比べても
    判定にならない（本戦略は市場の外にいる時間が長く、当然リターンも小さくなる）。"""
    if len(curve) < 2:
        return {"ret": 0.0, "max_dd": 0.0, "sharpe": 0.0, "ret_dd": 0.0}
    ret = float(curve.iloc[-1]) / float(curve.iloc[0]) - 1
    peak = curve.cummax()
    dd = float(((curve - peak) / peak).min())
    r = curve.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    return {"ret": ret, "max_dd": dd, "sharpe": sharpe,
            "ret_dd": ret / abs(dd) if dd < 0 else float("inf")}


def bh_return(data: dict, ctx, start: str, end: str) -> dict:
    """等ウェイト・バイ&ホールドの資産曲線を実際に組み、リスクも含めて測る。"""
    st, en = pd.Timestamp(start), pd.Timestamp(end)

    norm = []
    for px in data.values():
        sub = px.loc[st:en]
        if len(sub) >= 2:
            norm.append(sub["Close"] / float(sub["Close"].iloc[0]))
    ew_curve = (pd.concat(norm, axis=1).ffill().dropna(how="all").mean(axis=1)
                if norm else pd.Series(dtype=float))

    spx = ctx.spx.loc[st:en, "Close"]

    return {
        "equal_weight": _curve_stats(ew_curve),
        "sp500": _curve_stats(spx),
    }


def table_header() -> str:
    return ("    " + pad("仮説", 6) + pad("内容", 30) + rpad("件数", 6)
            + rpad("勝率", 8) + rpad("期待値R", 10) + rpad("PF", 7)
            + rpad("税引後", 9) + rpad("最大DD", 9) + rpad("保有日", 7))


def table_row(v: Variant, m: dict) -> str:
    pf = m.get("profit_factor", 0)
    pf_s = "{:.2f}".format(pf) if np.isfinite(pf) else "∞"
    return ("    " + pad(v.key, 6) + pad(v.name, 30)
            + rpad(m.get("trades", 0), 6)
            + rpad("{:.1%}".format(m.get("win_rate", 0)), 8)
            + rpad("{:+.3f}".format(m.get("expectancy_r", 0)), 10)
            + rpad(pf_s, 7)
            + rpad("{:+.1%}".format(m.get("after_tax_return", 0)), 9)
            + rpad("{:.1%}".format(m.get("max_dd", 0)), 9)
            + rpad("{:.1f}".format(m.get("avg_bars", 0)), 7))


# ═══════════════════════════════════════════════ 本体
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1b 期間分割検証")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--period", default="max")
    ap.add_argument("--capital", type=float, default=config.CAPITAL_JPY)
    ap.add_argument("--fixed", default=None,
                    help="仮説選定を行わず、指定した仮説（例 V4）を両期間で測る。"
                         "戦略を確定した後に別ユニバースで測り直す用")
    args = ap.parse_args()

    syms = args.symbols if args.symbols else bt.DEFAULT_UNIVERSE
    syms = [s.strip().upper() for s in syms]
    syms = [s + ".T" if s.isdigit() and len(s) == 4 else s for s in syms]

    W = 96
    print()
    print("═" * W)
    print("  Phase 1b  期間分割による構造仮説の検証")
    print("═" * W)
    print(f"\n  開発期間（仮説を選ぶ）  {IS_START} 〜 {IS_END}")
    print(f"  検証期間（一度だけ測る）{OOS_START} 〜 {OOS_END}")
    print(f"  対象  {len(syms)} 銘柄 / 取得期間 {args.period}\n")

    data, ctx = bt.load_data(syms, args.period)
    if not data:
        print("  [エラー] データが取得できませんでした\n")
        return 1
    spans = [(px.index[0].date(), px.index[-1].date()) for px in data.values()]
    print(f"  読込完了 {len(data)} 銘柄  "
          f"（最古 {min(s[0] for s in spans)} / 最新 {max(s[1] for s in spans)}）")

    # ---------- 開発期間: 全仮説を比較
    print("\n" + "─" * W)
    print("  【1】開発期間での仮説比較  ← ここで1つ選ぶ")
    print("─" * W)
    print(table_header())
    is_results = {}
    for v in VARIANTS:
        with apply_overrides(v.overrides):
            tr, cv = bt.run_once(data, ctx, args.capital, IS_START, IS_END)
        m = summarize(tr, cv, args.capital)
        is_results[v.key] = m
        print(table_row(v, m))

    # ---------- 選定（事前に決めた基準のみで機械的に選ぶ）
    eligible = [v for v in VARIANTS
                if is_results[v.key]["trades"] >= MIN_TRADES_FOR_SELECTION]
    if not eligible:
        print(f"\n  [警告] トレード数 {MIN_TRADES_FOR_SELECTION} 件以上の仮説がありません。"
              f"件数条件を外して選定します")
        eligible = [v for v in VARIANTS if is_results[v.key]["trades"] > 0]
    if not eligible:
        print("\n  [エラー] 開発期間にトレードが発生した仮説がありません\n")
        return 1

    if args.fixed:
        match = [v for v in VARIANTS if v.key == args.fixed.upper()]
        if not match:
            print(f"\n  [エラー] 仮説 {args.fixed} は存在しません\n")
            return 1
        chosen = match[0]
        print(f"\n  [固定モード] 選定を行わず {chosen.key} を測ります"
              f"（戦略は既に確定済みのため）")
    else:
        chosen = max(eligible, key=lambda v: is_results[v.key]["expectancy_r"])
    cm = is_results[chosen.key]

    print("\n" + "─" * W)
    print("  【2】選定結果")
    print("─" * W)
    print(f"\n    選定 → {chosen.key}  {chosen.name}")
    print(f"    根拠   {chosen.rationale}")
    print(f"    選定基準  開発期間の期待値が最大（トレード数 {MIN_TRADES_FOR_SELECTION} 件以上）")
    print(f"\n    開発期間の成績  期待値 {cm['expectancy_r']:+.3f} R / "
          f"勝率 {cm['win_rate']:.1%} / {cm['trades']}件 / "
          f"税引後 {cm['after_tax_return']:+.1%}")
    if chosen.overrides:
        print("\n    変更したパラメータ:")
        for k, val in chosen.overrides.items():
            print(f"      {k} = {val}")

    # ---------- 検証期間: 選んだものを一度だけ
    print("\n" + "─" * W)
    print("  【3】検証期間での測定  ← 本当のテスト（一度だけ）")
    print("─" * W)
    with apply_overrides(chosen.overrides):
        oos_tr, oos_cv = bt.run_once(data, ctx, args.capital, OOS_START, OOS_END)
    om = summarize(oos_tr, oos_cv, args.capital)
    bh = bh_return(data, ctx, OOS_START, OOS_END)

    print()
    print("    " + pad("", 22) + rpad("開発期間", 14) + rpad("検証期間", 14)
          + rpad("変化", 12))
    print("    " + "─" * 60)
    for label, key, fmt in [
        ("トレード数", "trades", "{:.0f}"),
        ("勝率", "win_rate", "{:.1%}"),
        ("期待値R", "expectancy_r", "{:+.3f}"),
        ("プロフィットファクター", "profit_factor", "{:.2f}"),
        ("税引後リターン", "after_tax_return", "{:+.1%}"),
        ("最大ドローダウン", "max_dd", "{:.1%}"),
        ("平均保有日数", "avg_bars", "{:.1f}"),
        ("ギャップ損切り率", "gap_rate", "{:.1%}"),
    ]:
        a, b = cm.get(key, 0), om.get(key, 0)
        a_s = fmt.format(a) if np.isfinite(a) else "∞"
        b_s = fmt.format(b) if np.isfinite(b) else "∞"
        if key == "trades":
            d_s = "-"
        elif np.isfinite(a) and np.isfinite(b):
            d_s = "{:+.3f}".format(b - a) if "R" in label or key == "profit_factor" \
                else "{:+.1%}".format(b - a) if "%" in fmt else "{:+.1f}".format(b - a)
        else:
            d_s = "-"
        print("    " + pad(label, 22) + rpad(a_s, 14) + rpad(b_s, 14) + rpad(d_s, 12))

    # ---- ベンチマーク（リターンとリスクの両方）
    exposure_days = sum(t.bars_held for t in oos_tr)
    oos_bars = len(oos_cv)
    time_in_market = (exposure_days / (oos_bars * config.MAX_CONCURRENT_POSITIONS)
                      if oos_bars else 0.0)
    strat = {
        "ret": om["after_tax_return"],
        "max_dd": om["max_dd"],
        "sharpe": om.get("sharpe", 0.0),
        "ret_dd": (om["after_tax_return"] / abs(om["max_dd"])
                   if om["max_dd"] < 0 else float("inf")),
    }

    print("\n    " + "─" * 74)
    print("    検証期間のベンチマーク（リターンだけでなくリスクも比較する）")
    print()
    print("    " + pad("", 26) + rpad("リターン", 12) + rpad("最大DD", 11)
          + rpad("リターン/DD", 13) + rpad("シャープ", 11))
    print("    " + "─" * 74)
    for label, st_ in [("本戦略（税引後）", strat),
                       ("同銘柄 等ウェイト保有", bh["equal_weight"]),
                       ("S&P500 保有", bh["sp500"])]:
        rd = st_["ret_dd"]
        rd_s = "{:.2f}".format(rd) if np.isfinite(rd) else "∞"
        print("    " + pad("  " + label, 26)
              + rpad("{:+.1%}".format(st_["ret"]), 12)
              + rpad("{:.1%}".format(st_["max_dd"]), 11)
              + rpad(rd_s, 13)
              + rpad("{:.2f}".format(st_["sharpe"]), 11))
    print()
    print(f"    本戦略の市場滞在率  {time_in_market:.1%}"
          f"   （5枠のうち実際に埋まっていた割合）")
    print("    ※ 市場の外にいる時間が長いほどリターンは小さくなるが、"
          "その分ドローダウンも小さくなる。")

    # ---------- 判定
    print("\n" + "═" * W)
    print("  【4】判定")
    print("═" * W)
    print()
    e_is, e_oos = cm["expectancy_r"], om["expectancy_r"]
    ew = bh["equal_weight"]
    beats_ret = strat["ret"] > ew["ret"]
    beats_risk_adj = strat["ret_dd"] > ew["ret_dd"]
    caveat = ["", "ただしこれは1回の期間分割の結果です。実資金を入れる前に、",
              "ペーパートレードで前向きに数ヶ月確認することを強く推奨します。"]

    if om["trades"] < 30:
        verdict = "判定不能"
        msg = [f"検証期間のトレードが {om['trades']} 件しかなく、統計的に判断できません。",
               "銘柄数か期間を増やす必要があります。"]
    elif e_oos <= 0:
        verdict = "エッジは本物ではなかった"
        msg = [f"検証期間の期待値が {e_oos:+.3f} R でマイナスです。",
               f"開発期間では {e_is:+.3f} R だったので、これは 2016〜2021年に",
               "当てはまっていただけで、未知の期間では機能しませんでした。",
               "",
               "FINDINGS.md に書いたとおり、ここで3回目の調整に進むべきではありません。",
               "どんな理屈をつけても曲線当てはめになります。"]
    elif e_oos < e_is * 0.5:
        verdict = "エッジは存在するが大幅に劣化"
        msg = [f"検証期間の期待値 {e_oos:+.3f} R は開発期間 {e_is:+.3f} R の半分未満です。",
               "プラスではあるものの、開発期間への当てはめが相当含まれています。",
               "実運用に回すには根拠が弱すぎます。"]
    elif beats_ret:
        verdict = "エッジは本物の可能性が高い（絶対リターンでもバイ&ホールドを上回る）"
        msg = [f"検証期間の期待値 {e_oos:+.3f} R は開発期間 {e_is:+.3f} R を維持し、",
               f"税引後リターン {strat['ret']:+.1%} はバイ&ホールド {ew['ret']:+.1%} も",
               "上回りました。"] + caveat
    elif beats_risk_adj:
        verdict = "エッジは本物。ただし絶対リターンではバイ&ホールドに負ける"
        msg = [f"検証期間の期待値 {e_oos:+.3f} R は開発期間 {e_is:+.3f} R をほぼ維持しました。",
               "未知の期間で崩れなかったので、この優位は当てはめではありません。",
               "",
               f"絶対リターンでは負けます（{strat['ret']:+.1%} vs {ew['ret']:+.1%}）。",
               f"ただしリスク調整後では上回ります"
               f"（リターン/DD {strat['ret_dd']:.2f} vs {ew['ret_dd']:.2f}、",
               f"最大DD {strat['max_dd']:.1%} vs {ew['max_dd']:.1%}）。",
               "",
               "つまり「上昇相場でリターンを取りこぼすが、下落局面の被弾は小さい」",
               "という性質です。どちらを重視するかは投資方針の問題であり、",
               "数字だけでは決まりません。"] + caveat
    else:
        verdict = "エッジはあるがバイ&ホールドに及ばない"
        msg = [f"検証期間の期待値は {e_oos:+.3f} R でプラスを維持しました。",
               f"しかし絶対リターン（{strat['ret']:+.1%} vs {ew['ret']:+.1%}）でも",
               f"リスク調整後（{strat['ret_dd']:.2f} vs {ew['ret_dd']:.2f}）でも",
               "バイ&ホールドに届きません。売買する根拠が数字の上で存在しません。"]

    print(f"    ★ {verdict}")
    print()
    for line in msg:
        print(f"      {line}")

    # ---------- 参考: 全仮説の検証期間成績
    print("\n" + "─" * W)
    print("  【5】参考: 全仮説の検証期間成績")
    print("─" * W)
    print("\n    ※ これは判定に使いません。全部見てから選ぶと、それは検証ではなく")
    print("      2回目の開発になり、期間を分けた意味が失われます。")
    print("      ここに載せるのは「選定が運だったのか」を後から検証できるようにするためです。\n")
    print(table_header())
    for v in VARIANTS:
        if v.key == chosen.key:
            m = om
        else:
            with apply_overrides(v.overrides):
                tr, cv = bt.run_once(data, ctx, args.capital, OOS_START, OOS_END)
            m = summarize(tr, cv, args.capital)
        mark = "  ← 選定" if v.key == chosen.key else ""
        print(table_row(v, m) + mark)

    ranked = sorted(VARIANTS, key=lambda v: -is_results[v.key]["expectancy_r"])
    print(f"\n    開発期間での順位: " + " > ".join(v.key for v in ranked))

    print("\n" + "═" * W)
    print()

    # ---------- 選定仮説の詳細
    if oos_tr:
        print("─" * W)
        print(f"  【6】選定仮説（{chosen.key}）の検証期間内訳")
        print("─" * W)
        for title, keyfn, wid in [
            ("セットアップ別", lambda t: t.setup, 26),
            ("市場環境別", lambda t: t.regime, 16),
            ("決済理由別", lambda t: t.exit_reason, 26),
        ]:
            print(f"\n    {title}")
            print("      " + pad("", wid) + rpad("件数", 6) + rpad("勝率", 9)
                  + rpad("期待値R", 10) + rpad("損益(円)", 15))
            for k, v in bt.by_group(oos_tr, keyfn).items():
                print("      " + pad(k, wid) + rpad(v["n"], 6)
                      + rpad("{:.1%}".format(v["win_rate"]), 9)
                      + rpad("{:+.3f}".format(v["expectancy_r"]), 10)
                      + rpad("{:,.0f}".format(v["pnl"]), 15))
        print()
        print("═" * W)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
