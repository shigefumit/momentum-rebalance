#!/usr/bin/env python3
"""Phase 1: バックテスト。

ここが分岐点。ルールが本当に勝てるかを数字で出す。勝てなければアプリを作る意味がない。

正確性のために守っていること:
  1. 未来参照の排除 — バーiの終値でシグナル、バーi+1で約定判定
  2. ギャップ処理 — 寄り付きが損切りを飛び越えたら寄り付き価格で決済
     （これを省くと結果が実際より良く出る。最も多い自己欺瞞）
  3. バー内は悲観側を仮定 — 高値と安値のどちらが先か分からないので損切りを先に判定
  4. 手数料・スリッページ・税金を全て含める
  5. ポートフォリオ制約（同時保有5銘柄・合計リスク6%）を実際に適用
  6. ベンチマーク比較 — 同じ銘柄のバイ&ホールドとS&P500に対して測る
  7. 期間別の安定性 — 6ヶ月ごとに区切って、特定期間のまぐれでないか確認

使い方:
    .venv/bin/python backtest.py                       # 既定ユニバース
    .venv/bin/python backtest.py --symbols NVDA AMD    # 銘柄指定
    .venv/bin/python backtest.py --period 10y --trades # 全トレード明細も表示
"""
from __future__ import annotations

import argparse
import math
import sys
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import fetcher
import indicators
import rules

# セクターを分散させたユニバース。全部テック銘柄にすると相関が高く、
# バックテストが「1つのトレンドに賭けた結果」になり過大評価される。
DEFAULT_UNIVERSE = [
    # 米国 半導体・テック
    "NVDA", "AMD", "AVGO", "MSFT", "AAPL", "GOOGL", "META", "AMZN",
    # 米国 非テック（分散のため必須）
    "JPM",   # 金融
    "XOM",   # エネルギー
    "UNH",   # ヘルスケア
    "WMT",   # 生活必需品
    "CAT",   # 資本財
    "KO",    # 飲料
    # 日本株
    "7203.T",  # トヨタ
    "8306.T",  # 三菱UFJ
    "9432.T",  # NTT
    "6758.T",  # ソニー
    "4063.T",  # 信越化学
]


def dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F", "A") else 1 for c in str(s))


def pad(s, w: int) -> str:
    s = str(s)
    while dwidth(s) > w:
        s = s[:-1]
    return s + " " * max(0, w - dwidth(s))


def rpad(s, w: int) -> str:
    s = str(s)
    return " " * max(0, w - dwidth(s)) + s


# ============================================================ データ型
@dataclass
class Signal:
    symbol: str
    market: str
    date: pd.Timestamp
    bar: int                 # px 上の整数位置
    setup: str
    style: str
    entry_method: str
    score: int
    regime: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    r_value: float
    rr: float
    fx: float


@dataclass
class Fill:
    date: pd.Timestamp
    shares: int
    price: float
    reason: str


@dataclass
class Trade:
    symbol: str
    market: str
    setup: str
    style: str
    regime: str
    score: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop: float
    tp1: float
    tp2: float
    r_value: float
    fx: float
    exits: list = field(default_factory=list)
    gapped_stop: bool = False
    took_tp1: bool = False
    bars_held: int = 0

    @property
    def exit_date(self):
        return self.exits[-1].date if self.exits else None

    @property
    def avg_exit(self) -> float:
        tot = sum(f.shares for f in self.exits)
        return sum(f.shares * f.price for f in self.exits) / tot if tot else 0.0

    @property
    def exit_reason(self) -> str:
        return "／".join(dict.fromkeys(f.reason for f in self.exits))

    @property
    def gross_local(self) -> float:
        return sum(f.shares * (f.price - self.entry_price) for f in self.exits)

    @property
    def cost_local(self) -> float:
        rate = config.COST_PER_SIDE
        buy = self.shares * self.entry_price * rate
        sell = sum(f.shares * f.price for f in self.exits) * rate
        return buy + sell

    @property
    def net_local(self) -> float:
        return self.gross_local - self.cost_local

    @property
    def net_jpy(self) -> float:
        return self.net_local * self.fx

    @property
    def r_multiple(self) -> float:
        """コスト込みのRマルチプル。系列に依存しないので edge の本当の指標。"""
        denom = self.shares * self.r_value
        return self.net_local / denom if denom else 0.0


# ============================================================ シグナル生成
class Context:
    def __init__(self, period: str):
        print("  指数・為替データ取得中 …")
        self.spx = indicators.enrich(fetcher.get_prices("^GSPC", period=period))
        self.vix = indicators.enrich(fetcher.get_prices("^VIX", period=period))
        try:
            self.nikkei = indicators.enrich(fetcher.get_prices("^N225", period=period))
        except fetcher.DataUnavailable:
            self.nikkei = pd.DataFrame()
        self.fx = fetcher.get_usdjpy_series(period)
        # 市場環境は日付だけで決まり、全銘柄が同じ日付を共有する。
        # 19銘柄×2500日で47,500回の重複計算になるためメモ化する。
        # 判定そのものは rules.regime_score をそのまま呼ぶので live と完全に同じ。
        self._regime_cache: dict = {}
        self._fx_cache: dict = {}

    def regime_at(self, ts: pd.Timestamp, market: str) -> rules.Regime:
        key = (ts, market)
        hit = self._regime_cache.get(key)
        if hit is None:
            hit = rules.regime_score(self.spx, self.vix, self.nikkei, ts, market)
            self._regime_cache[key] = hit
        return hit

    def fx_at(self, ts: pd.Timestamp, market: str) -> float:
        if market == "JP":
            return 1.0
        if self.fx is None:
            return fetcher.FX_FALLBACK
        hit = self._fx_cache.get(ts)
        if hit is None:
            sub = self.fx.loc[:ts]
            hit = float(sub["Close"].iloc[-1]) if not sub.empty else fetcher.FX_FALLBACK
            self._fx_cache[ts] = hit
        return hit


def generate_signals(symbol: str, px: pd.DataFrame, ctx: Context,
                     capital_for_sizing: float,
                     min_date: Optional[pd.Timestamp] = None) -> list[Signal]:
    """買いシグナルを抽出する。株数はここでは決めない（資産額が変動するため）。

    min_date を渡すとその日以降のバーだけを走査する。指標計算には全期間のデータが
    必要だが、シグナル探索は検証期間だけで足りるため、期間分割検証を高速化できる。
    """
    market = fetcher.market_of(symbol)
    out: list[Signal] = []
    start = config.SMA_LONG + 10
    if min_date is not None:
        pos = px.index.searchsorted(pd.Timestamp(min_date))
        start = max(start, int(pos))

    for k in range(start, len(px) - 1):     # 最終バーは翌日約定できないので除外
        i = k - len(px)
        ts = px.index[k]
        reg = ctx.regime_at(ts, market)
        setup = rules.setup_score(px, i, reg.state)
        if setup.score == 0:
            continue
        qual = rules.quality_score(px, i, market)
        score = reg.score + qual.score + setup.score
        if score < config.BUY_THRESHOLD:
            continue
        fx = ctx.fx_at(ts, market)
        plan = rules.plan_trade(px, i, setup, market, fx, capital_jpy=capital_for_sizing)
        if not plan.ok:
            continue
        out.append(Signal(
            symbol=symbol, market=market, date=ts, bar=k, setup=setup.name,
            style=setup.style, entry_method=setup.entry_method, score=score,
            regime=reg.state, entry=plan.entry, stop=plan.stop, tp1=plan.tp1,
            tp2=plan.tp2, r_value=plan.r_value, rr=plan.rr, fx=fx,
        ))
    return out


# ============================================================ ポートフォリオ・シミュレーション
@dataclass
class Position:
    trade: Trade
    px: pd.DataFrame
    remaining: int
    stop: float
    entry_bar: int


def simulate(signals: list[Signal], data: dict[str, pd.DataFrame],
             ctx: Context, initial_capital: float,
             start: Optional[pd.Timestamp] = None,
             end: Optional[pd.Timestamp] = None) -> tuple[list[Trade], pd.Series]:
    """日付順にポートフォリオを回す。start/end で期間を限定できる（期間分割検証用）。"""
    by_date: dict[pd.Timestamp, list[Signal]] = defaultdict(list)
    for s in signals:
        if start is not None and s.date < start:
            continue
        if end is not None and s.date > end:
            continue
        by_date[s.date].append(s)

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    if start is not None:
        all_dates = [d for d in all_dates if d >= start]
    if end is not None:
        all_dates = [d for d in all_dates if d <= end]
    equity = initial_capital
    cash_curve: dict[pd.Timestamp, float] = {}
    open_pos: list[Position] = []
    closed: list[Trade] = []
    pending: list[tuple[Signal, int]] = []   # (シグナル, 残り有効バー数)

    for d in all_dates:
        # ---------- 1. 保有ポジションの管理（決済判定が先。約定前に枠を空ける）
        still_open: list[Position] = []
        for p in open_pos:
            if d not in p.px.index:
                still_open.append(p)
                continue
            k = p.px.index.get_loc(d)
            r = p.px.iloc[k]
            t = p.trade
            t.bars_held = k - p.entry_bar
            closed_now = False

            # (a) ギャップで損切りを飛び越えた場合 → 寄り付きで決済
            if r["Open"] <= p.stop:
                t.exits.append(Fill(d, p.remaining, float(r["Open"]),
                                    "ギャップで損切り"))
                t.gapped_stop = True
                p.remaining = 0
                closed_now = True
            # (b) 通常の損切り（バー内は悲観側を仮定し、利確より先に判定）
            elif r["Low"] <= p.stop:
                t.exits.append(Fill(d, p.remaining, p.stop, "損切り"))
                p.remaining = 0
                closed_now = True
            else:
                # (c) 第1利確 → 半分売却し、損切りを建値へ
                if (config.USE_TP1_HALF and not t.took_tp1
                        and r["High"] >= t.tp1 and p.remaining > 1):
                    lot = config.LOT_SIZE[t.market]
                    half = max(lot, int(math.floor((p.remaining / 2) / lot) * lot))
                    half = min(half, p.remaining)
                    t.exits.append(Fill(d, half, t.tp1, "第1利確(+1R)"))
                    p.remaining -= half
                    t.took_tp1 = True
                    p.stop = max(p.stop, t.entry_price)
                # (d) 最終目標
                if p.remaining > 0 and r["High"] >= t.tp2:
                    t.exits.append(Fill(d, p.remaining, t.tp2, "最終目標"))
                    p.remaining = 0
                    closed_now = True
                # (e) トレーリング更新（第1利確後のみ）
                if (config.USE_TRAILING and p.remaining > 0 and t.took_tp1
                        and not math.isnan(r["trail_high"])):
                    p.stop = max(p.stop, float(r["trail_high"])
                                 - config.TRAIL_ATR_MULT * float(r["atr"]))
                # (f) ルールによる手仕舞い
                if p.remaining > 0:
                    hit, why = rules.exit_signal(p.px, k - len(p.px), t.entry_price,
                                                 p.stop, t.style)
                    if hit:
                        # 理由文に価格が入ると集計時に1件ずつ別グループになるため正規化
                        if why.startswith("損切りライン"):
                            label = "トレーリング損切り"
                        elif "200日線" in why:
                            label = "200日線割れ"
                        elif "RSI(2)" in why:
                            label = "RSI(2)>70で決済"
                        else:
                            label = why[:16]
                        t.exits.append(Fill(d, p.remaining, float(r["Close"]), label))
                        p.remaining = 0
                        closed_now = True
                # (g) 最大保有期間
                if p.remaining > 0 and t.bars_held >= config.MAX_HOLD_BARS[t.style]:
                    t.exits.append(Fill(d, p.remaining, float(r["Close"]), "期間満了"))
                    p.remaining = 0
                    closed_now = True

            if closed_now or p.remaining == 0:
                equity += t.net_jpy
                closed.append(t)
            else:
                still_open.append(p)
        open_pos = still_open

        # ---------- 2. 保留中の指値の約定判定（前日シグナル分）
        next_pending: list[tuple[Signal, int]] = []
        for s, life in pending:
            px = data[s.symbol]
            if d not in px.index:
                next_pending.append((s, life))
                continue
            k = px.index.get_loc(d)
            r = px.iloc[k]
            o, h, l = float(r["Open"]), float(r["High"]), float(r["Low"])

            fill: Optional[float] = None
            if s.entry_method == "breakout":
                # 逆指値買い: 指値以上に上昇したら約定。寄りが既に上なら寄りで約定（不利）
                if o >= s.entry:
                    fill = o
                elif h >= s.entry:
                    fill = s.entry
            else:
                # 指値買い: 指値以下に下落したら約定。寄りが既に下なら寄りで約定（有利）
                if o <= s.entry:
                    fill = o
                elif l <= s.entry:
                    fill = s.entry

            if fill is None:
                if life - 1 > 0:
                    next_pending.append((s, life - 1))
                continue

            # 枠と資金の確認
            if len(open_pos) >= config.MAX_CONCURRENT_POSITIONS:
                continue
            if any(p.trade.symbol == s.symbol for p in open_pos):
                continue      # 同一銘柄の重複建玉はしない

            heat = sum(p.remaining * (p.trade.entry_price - p.stop) * p.trade.fx
                       for p in open_pos) / equity
            if heat >= config.MAX_PORTFOLIO_HEAT:
                continue

            # 約定価格が動いたので株数を再計算（損切り幅が変わる）
            r_val = fill - s.stop
            if r_val <= 0:
                continue
            risk_jpy = equity * config.RISK_PER_TRADE
            lot = config.LOT_SIZE[s.market]
            shares = int(math.floor((risk_jpy / s.fx / r_val) / lot) * lot)
            if shares <= 0:
                continue
            pos_jpy = shares * fill * s.fx
            capn = equity * config.MAX_POSITION_PCT
            if pos_jpy > capn:
                shares = int(math.floor((capn / (fill * s.fx)) / lot) * lot)
            if shares <= 0:
                continue

            t = Trade(
                symbol=s.symbol, market=s.market, setup=s.setup, style=s.style,
                regime=s.regime, score=s.score, signal_date=s.date, entry_date=d,
                entry_price=fill, shares=shares, stop=s.stop,
                tp1=fill + config.TP1_R * r_val,
                tp2=fill + (s.tp2 - s.entry) / (s.entry - s.stop) * r_val,
                r_value=r_val, fx=s.fx,
            )
            open_pos.append(Position(trade=t, px=px, remaining=shares,
                                     stop=s.stop, entry_bar=k))
        pending = next_pending

        # ---------- 3. 当日のシグナルを翌日以降の保留に積む
        for s in by_date.get(d, []):
            pending.append((s, config.ENTRY_VALID_BARS))

        # ---------- 4. 資産曲線（含み損益も反映）
        unreal = 0.0
        for p in open_pos:
            if d in p.px.index:
                c = float(p.px.loc[d, "Close"])
                unreal += p.remaining * (c - p.trade.entry_price) * p.trade.fx
        cash_curve[d] = equity + unreal

    # 期末に残ったポジションは最終終値で強制決済
    for p in open_pos:
        last = p.px.index[-1]
        p.trade.exits.append(Fill(last, p.remaining, float(p.px.iloc[-1]["Close"]),
                                  "検証期間終了"))
        equity += p.trade.net_jpy
        closed.append(p.trade)

    closed.sort(key=lambda t: t.entry_date)
    return closed, pd.Series(cash_curve).sort_index()


# ============================================================ 税金
def apply_tax(trades: list[Trade]) -> tuple[float, float, dict]:
    """暦年ごとに譲渡益税を計算。損失は3年繰越控除。"""
    yearly: dict[int, float] = defaultdict(float)
    for t in trades:
        if t.exit_date is not None:
            yearly[t.exit_date.year] += t.net_jpy

    carry: list[tuple[int, float]] = []   # (発生年, 残損失額)
    total_tax = 0.0
    detail = {}
    for y in sorted(yearly):
        pnl = yearly[y]
        carry = [(fy, amt) for fy, amt in carry
                 if y - fy <= config.LOSS_CARRYFORWARD_YEARS]
        if pnl > 0:
            taxable = pnl
            new_carry = []
            for fy, amt in carry:
                used = min(taxable, amt)
                taxable -= used
                if amt - used > 0:
                    new_carry.append((fy, amt - used))
            carry = new_carry
            tax = taxable * config.TAX_RATE
            total_tax += tax
            detail[y] = {"損益": pnl, "課税対象": taxable, "税額": tax}
        else:
            carry.append((y, -pnl))
            detail[y] = {"損益": pnl, "課税対象": 0.0, "税額": 0.0}
    return total_tax, sum(yearly.values()), detail


# ============================================================ 指標
def metrics(trades: list[Trade], curve: pd.Series, initial: float) -> dict:
    if not trades:
        return {"trades": 0}

    rs = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.net_jpy for t in trades])
    wins, losses = rs[rs > 0], rs[rs <= 0]

    peak = curve.cummax()
    dd = (curve - peak) / peak
    ret = curve.pct_change().dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.01)
    final = float(curve.iloc[-1])

    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl <= 0].sum()

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(rs),
        "avg_win_r": wins.mean() if len(wins) else 0.0,
        "avg_loss_r": losses.mean() if len(losses) else 0.0,
        "expectancy_r": rs.mean(),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "total_pnl": pnl.sum(),
        "final_equity": final,
        "total_return": final / initial - 1,
        "cagr": (final / initial) ** (1 / years) - 1 if final > 0 else -1.0,
        "max_dd": float(dd.min()),
        "sharpe": (ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else 0.0,
        "years": years,
        "trades_per_year": len(trades) / years,
        "avg_bars": np.mean([t.bars_held for t in trades]),
        "gap_rate": sum(t.gapped_stop for t in trades) / len(trades),
        "tp1_rate": sum(t.took_tp1 for t in trades) / len(trades),
    }


def benchmark(data: dict[str, pd.DataFrame], ctx: Context,
              curve: pd.Series, initial: float) -> dict:
    """同じ銘柄を等ウェイトで買って持ち続けた場合と、S&P500。"""
    out = {}
    start, end = curve.index[0], curve.index[-1]

    rets = []
    for sym, px in data.items():
        sub = px.loc[start:end]
        if len(sub) < 2:
            continue
        rets.append(float(sub["Close"].iloc[-1]) / float(sub["Close"].iloc[0]) - 1)
    if rets:
        out["equal_weight_bh"] = float(np.mean(rets))

    spx = ctx.spx.loc[start:end]
    if len(spx) >= 2:
        out["sp500_bh"] = float(spx["Close"].iloc[-1]) / float(spx["Close"].iloc[0]) - 1
    return out


def by_group(trades: list[Trade], key) -> dict:
    g: dict = defaultdict(list)
    for t in trades:
        g[key(t)].append(t)
    out = {}
    for k, ts in sorted(g.items(), key=lambda x: -len(x[1])):
        rs = np.array([t.r_multiple for t in ts])
        out[k] = {
            "n": len(ts),
            "win_rate": float((rs > 0).mean()),
            "expectancy_r": float(rs.mean()),
            "pnl": sum(t.net_jpy for t in ts),
        }
    return out


def _half(ts: pd.Timestamp) -> str:
    return f"{ts.year}-{'H1' if ts.month <= 6 else 'H2'}"


def stability(trades: list[Trade]) -> list[dict]:
    """半年ごとの成績。特定期間のまぐれでないかを見る。"""
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[_half(t.entry_date)].append(t)

    out = []
    for label in sorted(groups):
        ts = groups[label]
        rs = np.array([t.r_multiple for t in ts])
        out.append({
            "period": label,
            "n": len(ts),
            "win_rate": float((rs > 0).mean()),
            "expectancy_r": float(rs.mean()),
            "pnl": sum(t.net_jpy for t in ts),
        })
    return out


# ============================================================ レポート
def report(trades: list[Trade], curve: pd.Series, initial: float,
           data: dict, ctx: Context, show_trades: bool) -> None:
    W = 78
    m = metrics(trades, curve, initial)

    print()
    print("═" * W)
    print("  Phase 1 バックテスト結果")
    print("═" * W)

    if not m.get("trades"):
        print("\n  トレードが1件も発生しませんでした。")
        print("  合格ラインが厳しすぎるか、対象期間に条件を満たす局面がありません。\n")
        return

    tax, gross_pnl, tax_detail = apply_tax(trades)
    after_tax = initial + gross_pnl - tax
    bm = benchmark(data, ctx, curve, initial)

    print(f"\n  検証期間   {curve.index[0].date()} 〜 {curve.index[-1].date()}"
          f"  （{m['years']:.1f}年）")
    print(f"  対象銘柄   {len(data)} 銘柄")
    print(f"  初期資金   {initial:,.0f}円")

    print("\n" + "─" * W)
    print("  【1】エッジの有無（系列に依存しないRマルチプル基準）")
    print("─" * W)
    print(f"    トレード数        {m['trades']:>10}      "
          f"（年 {m['trades_per_year']:.1f} 件）")
    print(f"    勝率              {m['win_rate']:>9.1%}")
    print(f"    平均利益          {m['avg_win_r']:>9.2f} R")
    print(f"    平均損失          {m['avg_loss_r']:>9.2f} R")
    print(f"    期待値            {m['expectancy_r']:>9.3f} R   "
          f"← {'プラス（エッジあり）' if m['expectancy_r'] > 0 else 'マイナス（エッジなし）'}")
    print(f"    プロフィットファクター {m['profit_factor']:>6.2f}")
    print(f"    平均保有           {m['avg_bars']:>9.1f} 営業日")
    print(f"    第1利確に到達      {m['tp1_rate']:>9.1%}")
    print(f"    ギャップで損切り   {m['gap_rate']:>9.1%}   "
          f"← 想定損失を超えた可能性のある回数")

    print("\n" + "─" * W)
    print("  【2】資金曲線（手数料・スリッページ込み / 税引前）")
    print("─" * W)
    print(f"    最終資産          {m['final_equity']:>13,.0f}円")
    print(f"    総リターン        {m['total_return']:>12.1%}")
    print(f"    年率(CAGR)        {m['cagr']:>12.1%}")
    print(f"    最大ドローダウン  {m['max_dd']:>12.1%}")
    print(f"    シャープレシオ    {m['sharpe']:>12.2f}")

    print("\n" + "─" * W)
    print("  【3】税引後（譲渡益税20.315% / 損失3年繰越控除）")
    print("─" * W)
    print(f"    納税額            {tax:>13,.0f}円")
    print(f"    税引後資産        {after_tax:>13,.0f}円")
    print(f"    税引後リターン    {after_tax / initial - 1:>12.1%}")

    print("\n" + "─" * W)
    print("  【4】ベンチマーク比較 ← これが最も重要")
    print("─" * W)
    print(f"    本戦略（税引前）  {m['total_return']:>12.1%}")
    print(f"    本戦略（税引後）  {after_tax / initial - 1:>12.1%}")
    if "equal_weight_bh" in bm:
        print(f"    同銘柄 等ウェイト保有 {bm['equal_weight_bh']:>8.1%}   "
              f"← 何もしないで持ち続けた場合")
    if "sp500_bh" in bm:
        print(f"    S&P500 保有       {bm['sp500_bh']:>12.1%}")
    if "equal_weight_bh" in bm:
        diff = (after_tax / initial - 1) - bm["equal_weight_bh"]
        verdict = "戦略が勝っている" if diff > 0 else "★ バイ&ホールドに負けている"
        print(f"\n    差（税引後 − 保有）{diff:>11.1%}   {verdict}")

    def table(title: str, groups: dict, label_head: str, label_w: int = 26) -> None:
        print("\n" + "─" * W)
        print(f"  {title}")
        print("─" * W)
        head = (pad(label_head, label_w) + rpad("件数", 6) + rpad("勝率", 9)
                + rpad("期待値R", 10) + rpad("損益(円)", 15))
        print("    " + head)
        for name, v in groups.items():
            wr = "{:.1%}".format(v["win_rate"])
            ex = "{:+.3f}".format(v["expectancy_r"])
            pl = "{:,.0f}".format(v["pnl"])
            print("    " + pad(name, label_w) + rpad(v["n"], 6)
                  + rpad(wr, 9) + rpad(ex, 10) + rpad(pl, 15))

    table("【5】セットアップ別", by_group(trades, lambda t: t.setup), "セットアップ")
    table("【6】市場環境別", by_group(trades, lambda t: t.regime), "市場環境", 16)
    table("【7】決済理由別", by_group(trades, lambda t: t.exit_reason), "決済理由")
    table("【8】銘柄別", by_group(trades, lambda t: t.symbol), "銘柄", 12)

    st = stability(trades)
    table("【9】期間別の安定性（半年ごと）← 特定期間のまぐれでないかの確認",
          {s["period"]: s for s in st}, "期間", 12)
    pos = sum(1 for s in st if s["expectancy_r"] > 0)
    print(f"\n    期待値プラスの期間: {pos} / {len(st)}"
          + ("   ← 安定している" if len(st) and pos / len(st) >= 0.6
             else "   ← ばらつきが大きい"))

    print("\n" + "─" * W)
    print("  【10】統計的な信頼性")
    print("─" * W)
    print(f"    トレード数 {m['trades']} 件")
    print("    経験則: 1パラメータあたり30トレード以上必要。")
    print(f"    本ルールの主要パラメータは概ね5個（ADX閾値/ATR倍率/RSI域/合格ライン/最小R:R）")
    need = 5 * 30
    if m["trades"] >= need:
        print(f"    → {need} 件以上あり、統計的に評価できる水準です")
    else:
        print(f"    → {need} 件には届きません。結果は参考値として扱ってください")
        print(f"       （銘柄数か期間を増やすとサンプルが増えます）")

    if show_trades:
        print("\n" + "─" * W)
        print("  全トレード明細")
        print("─" * W)
        rows = [{
            "entry": t.entry_date.date(), "sym": t.symbol,
            "setup": t.setup[:12], "score": t.score,
            "in": round(t.entry_price, 2), "out": round(t.avg_exit, 2),
            "sh": t.shares, "R": round(t.r_multiple, 2),
            "pnl": round(t.net_jpy), "bars": t.bars_held,
            "why": t.exit_reason[:16],
        } for t in trades]
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "═" * W)
    print()


# ============================================================ 再利用API（validate.py が使う）
def load_data(symbols: list[str], period: str,
              quiet: bool = False) -> tuple[dict, Context]:
    """価格データと指数コンテキストを読み込む。銘柄ごとに1回だけ実行する。"""
    ctx = Context(period)
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            px = indicators.enrich(fetcher.get_prices(sym, period=period))
        except fetcher.DataUnavailable as e:
            if not quiet:
                print(f"  {pad(sym, 9)} 取得失敗: {str(e)[:44]}")
            continue
        if len(px) < config.SMA_LONG + 30:
            if not quiet:
                print(f"  {pad(sym, 9)} データ不足 ({len(px)}本)")
            continue
        data[sym] = px
    return data, ctx


def run_once(data: dict, ctx: Context, capital: float,
             start=None, end=None) -> tuple[list[Trade], pd.Series]:
    """現在の config 設定でバックテストを1回実行する。
    設計仮説の切り替えは config の値を差し替えて行う（同じコードパスを通す）。"""
    signals: list[Signal] = []
    for sym, px in data.items():
        signals.extend(generate_signals(sym, px, ctx, capital, min_date=start))
    signals.sort(key=lambda x: x.date)
    st = pd.Timestamp(start) if start is not None else None
    en = pd.Timestamp(end) if end is not None else None
    return simulate(signals, data, ctx, capital, start=st, end=en)


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 バックテスト")
    ap.add_argument("--symbols", nargs="*", default=None, help="銘柄。既定は分散ユニバース")
    ap.add_argument("--period", default="10y", help="取得期間（既定 10y）")
    ap.add_argument("--capital", type=float, default=config.CAPITAL_JPY)
    ap.add_argument("--trades", action="store_true", help="全トレード明細を表示")
    args = ap.parse_args()

    syms = args.symbols if args.symbols else DEFAULT_UNIVERSE
    syms = [s.strip().upper() for s in syms]
    syms = [s + ".T" if s.isdigit() and len(s) == 4 else s for s in syms]

    print(f"\n  Phase 1 バックテスト   {len(syms)} 銘柄 / 期間 {args.period}")
    ctx = Context(args.period)

    data: dict[str, pd.DataFrame] = {}
    all_signals: list[Signal] = []
    for sym in syms:
        try:
            px = indicators.enrich(fetcher.get_prices(sym, period=args.period))
        except fetcher.DataUnavailable as e:
            print(f"  {pad(sym, 9)} 取得失敗: {str(e)[:44]}")
            continue
        if len(px) < config.SMA_LONG + 30:
            print(f"  {pad(sym, 9)} データ不足 ({len(px)}本)")
            continue
        data[sym] = px
        sigs = generate_signals(sym, px, ctx, args.capital)
        all_signals.extend(sigs)
        print(f"  {pad(sym, 9)} {len(px):>5}本  シグナル {len(sigs):>3} 件")

    if not data:
        print("\n  [エラー] 有効なデータが1銘柄もありません\n")
        return 1

    all_signals.sort(key=lambda s: s.date)
    print(f"\n  シグナル合計 {len(all_signals)} 件 → ポートフォリオ制約を適用してシミュレーション")

    trades, curve = simulate(all_signals, data, ctx, args.capital)
    print(f"  実際に約定したトレード {len(trades)} 件"
          f"（差分は同時保有5銘柄の枠・指値未約定・リスク上限による見送り）")

    report(trades, curve, args.capital, data, ctx, args.trades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
