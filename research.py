#!/usr/bin/env python3
"""後知恵つき戦略ラボ: 「このルールで売買していたら勝てたはず」を総当たりで探す。

ユーザーの依頼: 過去を遡ってカンニングしても良いので、勝てたルールを見つけてほしい。

この分析の価値は「勝てたルール」そのものではなく、**ズルの段階を分けて並べること**にある。

  段階0  ベンチマーク           — 何もしない（全銘柄を等ウェイトで持ち続ける／S&P500）
  段階1  ルールで銘柄を選ぶ     — 後知恵なし。過去の値動きだけを見て機械的に選ぶ
                                  （＝将来にも使える可能性がある）
  段階2  後知恵で銘柄を選ぶ     — 10年後に一番上がった銘柄を、上がったと知って選ぶ
                                  （＝将来には一切使えない。上限を知るためだけの数字）

段階1が段階2にどれだけ迫れるかが、この分析の答えになる。
迫れるなら「NVDAを手で選ぶ」代わりに「NVDAを自力で見つけるルール」が存在したことになる。

正確性のために守っていること:
  - 配当込みの株価（トータルリターン）を使う。長期保有比較で配当を無視すると
    持ち続ける側を過小評価し、結論が変わりうる
  - 米国株は円換算する（ユーザーの資金は円建てのため）
  - 判断は月末終値まで、売買は翌営業日終値。段階1の戦略に未来参照を入れない
  - 売買コスト0.2%と譲渡益税20.315%を反映。株数を保有し取得原価を追跡するので、
    「毎月入れ替える戦略は毎年課税され、持ち続ける戦略は課税が繰り延べられる」
    という現実の差がそのまま出る
  - 期間を2つに分ける（2006-2016 / 2016-2026）。両方で効いたルールだけが本物候補

既知の限界（結果を読む時に必ず考慮すること）:
  - ユニバースが生存者バイアス込み。現在も上場している銘柄しか入っていないため、
    段階1・段階2ともに実際より良く出る

使い方:
    .venv/bin/python research.py
    .venv/bin/python research.py --period-set recent
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import fetcher

# ═══════════════════════════════════════════════════════ ユニバース
# 「当時のテーマ株」を後知恵なしで拾えるかを試すため、テック偏重にせず
# 幅広いセクターを入れる。モメンタム戦略はこの中から自力で選ぶ。
US = [
    # 半導体・テック（2010年代後半〜のテーマ）
    "NVDA", "AMD", "AVGO", "MU", "AMAT", "LRCX", "ADI", "TXN", "QCOM", "INTC",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "ADBE", "CRM", "ORCL", "CSCO",
    "TSLA", "V", "MA",
    # 金融
    "JPM", "BAC", "GS",
    # ヘルスケア（2020年代前半のテーマ: 肥満症治療薬など）
    "LLY", "UNH", "ABBV", "MRK", "PFE", "JNJ",
    # エネルギー（2022年のテーマ）
    "XOM", "CVX",
    # 生活必需品・資本財・その他
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "HD", "NKE", "CAT", "BA", "MMM",
    "IBM", "GE", "DIS", "VZ",
]
ETF = ["XLK", "XLF", "XLE", "XLV", "XLP", "XLI", "XLY", "XLU", "XLB"]
JP = ["7203.T", "8306.T", "9432.T", "6758.T", "4063.T", "6861.T", "9984.T",
      "8035.T", "4502.T", "6367.T"]

PERIOD_SETS = {
    "both": [("2006-01-01", "2015-12-31", "前半10年 2006-2015"),
             ("2016-01-01", "2026-07-31", "後半10年 2016-2026")],
    "recent": [("2016-01-01", "2026-07-31", "後半10年 2016-2026")],
    "full": [("2006-01-01", "2026-07-31", "20年通し 2006-2026")],
}


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


# ═══════════════════════════════════════════════════════ 戦略の定義
@dataclass
class Strategy:
    key: str
    name: str
    note: str
    stage: int                  # 0=ベンチマーク 1=後知恵なし 2=後知恵あり
    rebalance: str              # once / monthly / quarterly
    weights: Callable = None    # (exec_date, past, full, ctx) -> {sym: weight}


def _avail(past: pd.DataFrame) -> list:
    last = past.iloc[-1]
    return [c for c in past.columns if pd.notna(last[c])]


def w_equal_all(exec_date, past, full, ctx):
    a = _avail(past)
    return {s: 1.0 / len(a) for s in a} if a else {}


def w_spx_only(exec_date, past, full, ctx):
    return {"__SPX__": 1.0}


def make_momentum(n: int, months: int, regime: bool = False):
    """過去Nヶ月のリターン上位n銘柄を持つ。後知恵は一切使わない。

    これが「当時のテーマ株を自力で拾う」systematic版。2016年ならクラウド、
    2020年なら半導体、2022年ならエネルギー、2023年以降ならAIを、
    人間が判断せずに値動きだけで拾えたかを試す。

    測定区間は**カレンダー基準**（12ヶ月なら丸1年前の終値と比較）。
    「営業日 × 21」で数えると誤差が出る。日米の営業日を1つの表にまとめると
    日付が両市場の和集合になり、1年あたりの行数が252より多くなるため
    （米国だけの営業日＋日本だけの営業日＋共通日）、252行遡っても
    約11.6ヶ月しか戻らない。
    """
    def f(exec_date, past, full, ctx):
        if len(past) < 260:
            return {}
        if regime:
            spx = ctx["spx"].loc[:past.index[-1]]
            if len(spx) >= 200 and spx.iloc[-1] < spx.tail(200).mean():
                return {}       # 弱気相場は全額現金
        end_ts = past.index[-1]
        pos = int(past.index.searchsorted(end_ts - pd.DateOffset(months=months)))
        if pos >= len(past) - 5:
            return {}
        ret = past.iloc[-1] / past.iloc[pos] - 1.0
        ret = ret.dropna()
        # 直近1ヶ月が急騰した銘柄は反落しやすいので除外しない（素朴なモメンタムを見る）
        if ret.empty:
            return {}
        top = ret.nlargest(min(n, len(ret)))
        return {s: 1.0 / len(top) for s in top.index}
    return f


def w_trend_all(exec_date, past, full, ctx):
    """自分の200日線より上にある銘柄だけを等ウェイトで持つ。下にあるものは現金。"""
    if len(past) < 200:
        return {}
    last, ma = past.iloc[-1], past.tail(200).mean()
    sel = [c for c in past.columns
           if pd.notna(last[c]) and pd.notna(ma[c]) and last[c] > ma[c]]
    return {s: 1.0 / len(sel) for s in sel} if sel else {}


def make_cheat_top(n: int):
    """【段階2・ズル】検証期間の全体を見て、最も上がった n 銘柄を最初から持つ。

    将来には一切使えない。段階1がどこまで迫れるかを測るための天井。
    """
    def f(exec_date, past, full, ctx):
        win = full.loc[ctx["start"]:ctx["end"]]
        rets = {}
        for c in win.columns:
            s = win[c].dropna()
            if len(s) > 250:
                rets[c] = s.iloc[-1] / s.iloc[0] - 1.0
        if not rets:
            return {}
        top = sorted(rets, key=rets.get, reverse=True)[:n]
        return {s: 1.0 / len(top) for s in top}
    return f


def make_cheat_top_with_regime(n: int):
    """【段階2・ズル】後知恵で選んだ勝ち組を、弱気相場では現金にして持つ。

    「銘柄選択のズル」に「タイミング」を足したら上乗せがあるのかを見る。
    """
    base = make_cheat_top(n)

    def f(exec_date, past, full, ctx):
        spx = ctx["spx"].loc[:past.index[-1]]
        if len(spx) >= 200 and spx.iloc[-1] < spx.tail(200).mean():
            return {}
        return base(exec_date, past, full, ctx)
    return f


STRATEGIES = [
    # ───────── 段階0: ベンチマーク（何もしない）
    Strategy("B1", "全銘柄を等ウェイトで持ち続ける", "売買しない。比較の基準", 0, "once", w_equal_all),
    Strategy("B2", "S&P500を持ち続ける", "売買しない。市場そのもの", 0, "once", w_spx_only),
    Strategy("B3", "全銘柄・毎月等ウェイトに戻す", "リバランスの効果だけを見る", 0, "monthly", w_equal_all),

    # ───────── 段階1: ルールで選ぶ（後知恵なし＝将来にも使える可能性）
    Strategy("R1", "12ヶ月モメンタム 上位5銘柄", "過去1年で上がった5銘柄を毎月入替", 1, "monthly", make_momentum(5, 12)),
    Strategy("R2", "12ヶ月モメンタム 上位3銘柄", "同じだが3銘柄に集中", 1, "monthly", make_momentum(3, 12)),
    Strategy("R3", "12ヶ月モメンタム 上位10銘柄", "同じだが10銘柄に分散", 1, "monthly", make_momentum(10, 12)),
    Strategy("R4", "6ヶ月モメンタム 上位5銘柄", "より短い期間の勢いで選ぶ", 1, "monthly", make_momentum(5, 6)),
    Strategy("R5", "12ヶ月モメンタム5＋弱気相場は現金", "S&P500が200日線割れなら全額現金", 1, "monthly", make_momentum(5, 12, regime=True)),
    Strategy("R6", "12ヶ月モメンタム5・四半期入替", "売買回数を減らして税とコストを抑える", 1, "quarterly", make_momentum(5, 12)),
    Strategy("R7", "200日線より上の銘柄を全部持つ", "上昇中の銘柄だけ保有、他は現金", 1, "monthly", w_trend_all),

    # ───────── 段階2: 後知恵で選ぶ（将来には使えない天井）
    Strategy("C1", "【ズル】期間中の勝ち組トップ5を持ち続ける", "10年後の結果を知って選ぶ", 2, "once", make_cheat_top(5)),
    Strategy("C2", "【ズル】勝ち組トップ3を持ち続ける", "さらに集中", 2, "once", make_cheat_top(3)),
    Strategy("C3", "【ズル】勝ち組トップ1を持ち続ける", "最強の1銘柄だけ。絶対的な天井", 2, "once", make_cheat_top(1)),
    Strategy("C4", "【ズル】勝ち組トップ5＋弱気相場は現金", "銘柄のズルにタイミングを足す", 2, "monthly", make_cheat_top_with_regime(5)),
]


# ═══════════════════════════════════════════════════════ シミュレーター
COST = config.COST_PER_SIDE          # 片道 0.1%
TAX = config.TAX_RATE                # 20.315%


def rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> list:
    """判断日と執行日のペアを返す。判断は月末終値まで、執行は翌営業日終値。
    こうしないと段階1の戦略に未来参照が入る。"""
    if freq == "once":
        return [(dates[0], dates[0])] if len(dates) else []
    ser = pd.Series(range(len(dates)), index=dates)
    rule = "ME" if freq == "monthly" else "QE"
    ends = ser.resample(rule).last().dropna().astype(int)
    out = []
    for i in ends.values:
        if i + 1 < len(dates):
            out.append((dates[i], dates[i + 1]))
    # 期首でも一度建てる
    if out and out[0][1] != dates[0]:
        out.insert(0, (dates[0], dates[0]))
    return out


def simulate(panel: pd.DataFrame, strat: Strategy, ctx: dict,
             start: str, end: str, capital: float) -> dict:
    """株数と取得原価を追跡する。これにより毎年の実現益に正しく課税でき、
    「毎月入替＝毎年課税」対「持ち続け＝課税繰延」の差が結果に出る。"""
    win = panel.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if win.empty:
        return {}
    dates = win.index
    plan = dict(rebalance_dates(dates, strat.rebalance))   # signal_date -> exec_date
    exec_map = {v: k for k, v in plan.items()}

    cash = capital
    shares: dict = defaultdict(float)
    basis: dict = defaultdict(float)      # 取得原価の合計
    realized = defaultdict(float)         # 年 -> 実現損益
    curve = {}
    turnover_total = 0.0
    n_trades = 0

    def price(sym, d):
        if sym == "__SPX__":
            v = ctx["spx"].loc[:d]
            return float(v.iloc[-1]) if len(v) else np.nan
        v = win[sym].loc[:d].dropna() if sym in win.columns else pd.Series(dtype=float)
        return float(v.iloc[-1]) if len(v) else np.nan

    # ---- 「一度だけ買って持ち続ける」戦略の対象を決める
    #
    # 期首日に全銘柄の株価が揃っている保証はない。前半10年の期首 2006-01-02 は
    # 米国市場が休場で、68銘柄中10銘柄（日本株）しか値が付いていなかった。
    # 素朴に「期首日に買う」と実装すると米国株を1銘柄も買えず、
    # ベンチマークが「日本株10銘柄だけ保有」に化けて比較の土台が壊れる。
    #
    # 正しい挙動: 対象銘柄ごとに資金を取り置き、**その銘柄に最初に値が付いた日**に買う。
    # 期間中に上場した銘柄（TSLA 2010年・AVGO 2009年など）も自然に扱える。
    pending: dict = {}
    if strat.rebalance == "once":
        # 対象の決定には期首1ヶ月分のデータを使う（「1月中に組む」という現実的な想定）
        head = dates[dates <= dates[0] + pd.Timedelta(days=31)]
        sig = head[-1] if len(head) else dates[0]
        tgt = strat.weights(sig, panel.loc[:sig], panel, ctx) or {}
        pending = {s: capital * w for s, w in tgt.items() if w > 0}

    prev_year = dates[0].year

    for d in dates:
        # ---- 取り置き資金の消化（値が付いた銘柄を順次買う）
        if pending:
            for s in [x for x in pending]:
                p = price(s, d)
                if pd.isna(p) or p <= 0:
                    continue
                amt = pending.pop(s)
                fee = amt * COST
                qty = (amt - fee) / p
                shares[s] += qty
                basis[s] += amt - fee
                cash -= amt
                turnover_total += amt
                n_trades += 1
        # ---- 年替わりで納税（損失は繰越控除しない簡易版。長期保有側に不利にならない）
        if d.year != prev_year:
            pnl = realized[prev_year]
            if pnl > 0:
                cash -= pnl * TAX
            prev_year = d.year

        # ---- リバランス（"once" は上の取り置き処理で完結するのでここは通らない）
        if strat.rebalance != "once" and d in exec_map:
            sig_date = exec_map[d]
            past = panel.loc[:sig_date]
            target = strat.weights(d, past, panel, ctx) or {}

            equity = cash + sum(sh * price(s, d) for s, sh in shares.items()
                                if sh and pd.notna(price(s, d)))
            if equity <= 0:
                curve[d] = 0.0
                continue

            held = set(shares) | set(target)
            for s in held:
                p = price(s, d)
                if pd.isna(p) or p <= 0:
                    continue
                cur_val = shares[s] * p
                tgt_val = equity * target.get(s, 0.0)
                delta = tgt_val - cur_val
                if abs(delta) < equity * 0.002:        # 微小な調整は売買しない
                    continue
                qty = delta / p
                if qty > 0:                            # 買い
                    cost = abs(delta) * COST
                    cash -= delta + cost
                    shares[s] += qty
                    basis[s] += delta
                else:                                  # 売り
                    sell_qty = min(-qty, shares[s])
                    if sell_qty <= 0:
                        continue
                    proceeds = sell_qty * p
                    avg = basis[s] / shares[s] if shares[s] else p
                    realized[d.year] += sell_qty * (p - avg)
                    basis[s] -= sell_qty * avg
                    shares[s] -= sell_qty
                    cost = proceeds * COST
                    cash += proceeds - cost
                turnover_total += abs(delta)
                n_trades += 1

        # ---- 日次の資産評価
        val = cash
        for s, sh in shares.items():
            if sh:
                p = price(s, d)
                if pd.notna(p):
                    val += sh * p
        curve[d] = val

    cv = pd.Series(curve).sort_index()

    # 期末の含み益にも課税（持ち続け戦略を有利にしすぎないため）
    final_unrealized = 0.0
    for s, sh in shares.items():
        if sh:
            p = price(s, dates[-1])
            if pd.notna(p):
                final_unrealized += sh * p - basis[s]
    last_year_tax = max(0.0, realized[dates[-1].year]) * TAX
    after_tax = float(cv.iloc[-1]) - last_year_tax - max(0.0, final_unrealized) * TAX

    peak = cv.cummax()
    dd = float(((cv - peak) / peak).min())
    r = cv.pct_change().dropna()
    years = max((dates[-1] - dates[0]).days / 365.25, 0.01)
    final = float(cv.iloc[-1])

    return {
        "ret": final / capital - 1.0,
        "after_tax_ret": after_tax / capital - 1.0,
        "cagr": (final / capital) ** (1 / years) - 1 if final > 0 else -1.0,
        "cagr_at": (after_tax / capital) ** (1 / years) - 1 if after_tax > 0 else -1.0,
        "max_dd": dd,
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
        "ret_dd": (final / capital - 1) / abs(dd) if dd < 0 else float("inf"),
        "final": final, "after_tax": after_tax,
        "trades": n_trades,
        "turnover": turnover_total / capital,
        "curve": cv,
    }


# ═══════════════════════════════════════════════════════ データ読込
def load_panel(symbols: list, quiet: bool = False) -> tuple:
    """配当込み株価を円換算して1枚のテーブルにする。"""
    fx = None
    try:
        fx = fetcher.get_total_return_prices("USDJPY=X", "max")
    except Exception:
        pass
    if fx is None or fx.empty:
        print("  [警告] ドル円が取れないため 150円 固定で換算します")

    cols = {}
    for i, sym in enumerate(symbols, 1):
        try:
            s = fetcher.get_total_return_prices(sym, "max")
        except Exception:
            if not quiet:
                print(f"  [{i}/{len(symbols)}] {pad(sym, 9)} 取得失敗")
            continue
        if sym.endswith(".T"):
            cols[sym] = s                       # 日本株はそのまま円
        else:
            rate = (fx.reindex(s.index).ffill().bfill() if fx is not None
                    else pd.Series(150.0, index=s.index))
            cols[sym] = s * rate                # 米国株は円換算
        if not quiet and i % 10 == 0:
            print(f"  読込 {i}/{len(symbols)} …")

    panel = pd.DataFrame(cols).sort_index()

    # 市場カレンダーのズレを埋める。
    #
    # 日本株と米国株を1枚のテーブルに並べると、日付は両市場の和集合になる。
    # 米国の祝日には米国株が全部NaN、日本の祝日には日本株が全部NaNになる。
    # この状態でリターンを計算すると、その日は片方の市場の銘柄しか順位に載らず、
    # 「上位5銘柄が全部日本株」のような明らかに誤った結果が出る。
    #
    # ffill で「直近に判明している株価」を使う。上場前は NaN のままなので、
    # 期間中に上場した銘柄を先に買ってしまうことはない。
    panel = panel.ffill()

    spx = fetcher.get_total_return_prices("^GSPC", "max")
    fxr = (fx.reindex(spx.index).ffill().bfill() if fx is not None
           else pd.Series(150.0, index=spx.index))
    ctx = {"spx": spx * fxr}                    # S&P500も円換算して条件を揃える
    return panel, ctx


# ═══════════════════════════════════════════════════════ レポート
def run_period(panel, ctx, start, end, label, capital) -> dict:
    W = 108
    ctx = dict(ctx)
    ctx["start"], ctx["end"] = pd.Timestamp(start), pd.Timestamp(end)

    print()
    print("═" * W)
    print(f"  {label}   （{start} 〜 {end}）  初期資金 {capital:,.0f}円")
    print("═" * W)

    results = {}
    stage_names = {0: "段階0  ベンチマーク（売買しない）",
                   1: "段階1  ルールで選ぶ（後知恵なし＝将来にも使える可能性）",
                   2: "段階2  後知恵で選ぶ（将来には使えない・天井を知るための数字）"}
    cur_stage = None

    for st in STRATEGIES:
        r = simulate(panel, st, ctx, start, end, capital)
        if not r:
            continue
        results[st.key] = (st, r)

        if st.stage != cur_stage:
            cur_stage = st.stage
            print()
            print(f"  {stage_names[st.stage]}")
            print("  " + "─" * (W - 2))
            print("  " + pad("", 6) + pad("戦略", 38) + rpad("税引後", 12)
                  + rpad("年率", 9) + rpad("最大DD", 9) + rpad("最終資産", 15)
                  + rpad("売買回数", 9))
        print("  " + pad(st.key, 6) + pad(st.name, 38)
              + rpad("{:+,.0%}".format(r["after_tax_ret"]), 12)
              + rpad("{:+.1%}".format(r["cagr_at"]), 9)
              + rpad("{:.0%}".format(r["max_dd"]), 9)
              + rpad("{:,.0f}円".format(r["after_tax"]), 15)
              + rpad(r["trades"], 9))

    return results


def compare(all_results: list, labels: list) -> None:
    """段階1が段階2にどれだけ迫れたかを出す。これがこの分析の答え。"""
    W = 108
    print()
    print("═" * W)
    print("  この分析の答え: 「自力で見つけるルール」は「後知恵で選ぶ」にどれだけ迫れたか")
    print("═" * W)

    for label, res in zip(labels, all_results):
        if not res:
            continue
        s0 = [r for st, r in res.values() if st.stage == 0]
        s1 = [(st, r) for st, r in res.values() if st.stage == 1]
        s2 = [(st, r) for st, r in res.values() if st.stage == 2]
        if not (s0 and s1 and s2):
            continue

        bench = max(x["after_tax_ret"] for x in s0)
        b1 = max(s1, key=lambda x: x[1]["after_tax_ret"])
        b2 = max(s2, key=lambda x: x[1]["after_tax_ret"])

        print(f"\n  【{label}】")
        print(f"    何もしない（最良のベンチマーク）      {bench:+,.0%}")
        print(f"    段階1 の最良  {pad(b1[0].name, 32)} {b1[1]['after_tax_ret']:+,.0%}"
              f"   最大DD {b1[1]['max_dd']:.0%}")
        print(f"    段階2 の最良  {pad(b2[0].name, 32)} {b2[1]['after_tax_ret']:+,.0%}"
              f"   最大DD {b2[1]['max_dd']:.0%}")

        beat = b1[1]["after_tax_ret"] - bench
        catch = (b1[1]["after_tax_ret"] / b2[1]["after_tax_ret"]
                 if b2[1]["after_tax_ret"] > 0 else 0)
        print()
        print(f"    → 段階1は「何もしない」を {beat:+,.0%} 上回った"
              + ("（勝てている）" if beat > 0 else "（勝てていない）"))
        print(f"    → 段階1は段階2（ズルの天井）の {catch:.0%} を捕まえた")

    # 両期間で効いたルールを特定
    if len(all_results) >= 2 and all(all_results):
        print()
        print("─" * W)
        print("  両期間で「何もしない」を上回った段階1ルール ← 本物候補")
        print("─" * W)
        keys = set.intersection(*[{k for k, (st, _) in r.items() if st.stage == 1}
                                  for r in all_results])
        benches = [max(x["after_tax_ret"] for st, x in r.values() if st.stage == 0)
                   for r in all_results]
        found = False
        for k in sorted(keys):
            rets = [all_results[i][k][1]["after_tax_ret"] for i in range(len(all_results))]
            if all(rt > bc for rt, bc in zip(rets, benches)):
                found = True
                name = all_results[0][k][0].name
                detail = " / ".join(
                    f"{lb}: {rt:+,.0%}（基準 {bc:+,.0%}）"
                    for lb, rt, bc in zip(labels, rets, benches))
                print(f"    ○ {k}  {name}")
                print(f"       {detail}")
        if not found:
            print("    該当なし。両方の期間で「何もしない」に勝ったルールは存在しませんでした。")
    print()
    print("═" * W)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="後知恵つき戦略ラボ")
    ap.add_argument("--period-set", default="both", choices=list(PERIOD_SETS))
    ap.add_argument("--capital", type=float, default=config.CAPITAL_JPY)
    ap.add_argument("--universe", default="all", choices=["all", "us", "etf", "jp"])
    args = ap.parse_args()

    syms = {"all": US + ETF + JP, "us": US, "etf": ETF, "jp": JP}[args.universe]

    print(f"\n  後知恵つき戦略ラボ   {len(syms)} 銘柄 / 全{len(STRATEGIES)}戦略")
    print("  配当込み・円換算・売買コスト0.2%・譲渡益税20.315% を反映")
    panel, ctx = load_panel(syms)
    print(f"  読込完了 {panel.shape[1]} 銘柄  "
          f"（{panel.index[0].date()} 〜 {panel.index[-1].date()}）")

    periods = PERIOD_SETS[args.period_set]
    all_res, labels = [], []
    for start, end, label in periods:
        res = run_period(panel, ctx, start, end, label, args.capital)
        all_res.append(res)
        labels.append(label)

    compare(all_res, labels)

    print("  ※ 注意: ユニバースは現在も上場している銘柄のみで構成されているため、")
    print("    生存者バイアスが入っています。段階1・段階2ともに実際より良く出ます。")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
