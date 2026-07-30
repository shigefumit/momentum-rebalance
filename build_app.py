#!/usr/bin/env python3
"""PWA（iPhoneホーム画面用アプリ）のHTMLを生成する。

毎月の運用手順:
    1. .venv/bin/python momentum.py --json app_data.json
    2. .venv/bin/python build_app.py
    3. 生成された app.html を再公開する

データを埋め込む方式にしている理由:
  - 月1回しか変わらないルールなので、常時通信する必要がない
  - 通信しないので API 障害・レート制限・CORS の影響を受けない
  - オフラインでも開ける（電車の中でも確認できる）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "app_data.json"
OUT = HERE / "app.html"

TEMPLATE = """<title>モメンタム・リバランス</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<style>
  /* ── トークン。両テーマを同じ精度で設計する ───────────────────── */
  :root {
    --paper:   #f1f3f6;   /* 寒色に寄せた紙。純グレーは「選んでいない」色に見える */
    --surface: #ffffff;
    --sunken:  #e9edf2;
    --ink:     #14171d;
    --ink-mid: #414a58;
    --ink-soft:#5a6371;
    --rule:    #d3d8df;
    --rule-hi: #14171d;
    --signal:  #1e46c8;   /* 唯一のアクセント。保有と境界線にだけ使う */
    --signal-bg:#e7ecfb;
    --up:      #0b7048;
    --down:    #a82e1c;
    --warn:    #8f5a00;
    --warn-bg: #fdf3e0;

    --display: "Hiragino Mincho ProN", "Yu Mincho", YuMincho, "MS PMincho", serif;
    --body: "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", YuGothic,
            system-ui, -apple-system, sans-serif;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;

    --pad: clamp(16px, 4.5vw, 28px);
    --maxw: 720px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #0f1216; --surface: #171b21; --sunken: #1e242c;
      --ink: #e6eaf0; --ink-mid: #b4bcc8; --ink-soft: #8e97a4;
      --rule: #29303a; --rule-hi: #e6eaf0;
      --signal: #7b9bff; --signal-bg: #1a2338;
      --up: #35b37e; --down: #e8735c; --warn: #d69a2e; --warn-bg: #2a2214;
    }
  }
  /* 閲覧者のテーマ切替は media query に必ず勝つ */
  :root[data-theme="dark"] {
    --paper: #0f1216; --surface: #171b21; --sunken: #1e242c;
    --ink: #e6eaf0; --ink-mid: #b4bcc8; --ink-soft: #8e97a4;
    --rule: #29303a; --rule-hi: #e6eaf0;
    --signal: #7b9bff; --signal-bg: #1a2338;
    --up: #35b37e; --down: #e8735c; --warn: #d69a2e; --warn-bg: #2a2214;
  }
  :root[data-theme="light"] {
    --paper: #f1f3f6; --surface: #ffffff; --sunken: #e9edf2;
    --ink: #14171d; --ink-mid: #414a58; --ink-soft: #5a6371;
    --rule: #d3d8df; --rule-hi: #14171d;
    --signal: #1e46c8; --signal-bg: #e7ecfb;
    --up: #0b7048; --down: #a82e1c; --warn: #8f5a00; --warn-bg: #fdf3e0;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--paper); color: var(--ink);
    font-family: var(--body); line-height: 1.7;
    -webkit-text-size-adjust: 100%;
    font-feature-settings: "palt" 1;
  }
  .wrap { max-width: var(--maxw); margin: 0 auto; padding: var(--pad); }
  .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  /* ── ヘッダ ─────────────────────────────────────────────── */
  header { padding-block: clamp(20px, 6vw, 40px) 0; }
  .eyebrow {
    font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 .6em;
  }
  h1 {
    font-family: var(--display); font-weight: 600;
    font-size: clamp(1.7rem, 6.5vw, 2.5rem); line-height: 1.3;
    margin: 0; text-wrap: balance; letter-spacing: .01em;
  }
  .rule-line {
    margin: .9em 0 0; color: var(--ink-mid); font-size: .95rem;
    max-width: 34em;
  }
  .asof { margin: 1.4em 0 0; font-size: .82rem; color: var(--ink-soft); }

  /* ── カード ─────────────────────────────────────────────── */
  .card {
    background: var(--surface); border: 1px solid var(--rule);
    border-radius: 3px; padding: var(--pad); margin-block: 18px;
  }
  .card > :first-child { margin-top: 0; }
  .card > :last-child { margin-bottom: 0; }
  h2 {
    font-family: var(--display); font-weight: 600;
    font-size: 1.12rem; margin: 0 0 .2em; letter-spacing: .02em;
  }
  .sub { font-size: .82rem; color: var(--ink-soft); margin: 0 0 1.2em; }

  /* ── 今月の指示 ─────────────────────────────────────────── */
  .verdict { border-left: 3px solid var(--signal); }
  .verdict.hold { border-left-color: var(--up); }
  .action-h {
    font-family: var(--display); font-size: clamp(1.25rem, 5vw, 1.6rem);
    margin: 0 0 .5em; line-height: 1.35;
  }
  .moves { display: flex; flex-direction: column; gap: 14px; margin-top: 18px; }
  .move-grp { display: flex; flex-direction: column; gap: 6px; }
  .move-lbl {
    font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--ink-soft);
  }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    display: inline-flex; align-items: baseline; gap: .5em;
    border: 1px solid var(--rule); border-radius: 2px;
    padding: .35em .6em; font-size: .88rem; background: var(--paper);
  }
  .chip.sell { border-color: var(--down); color: var(--down); }
  .chip.buy  { border-color: var(--signal); color: var(--signal); }
  .chip .t { font-family: var(--mono); font-weight: 600; }
  .chip .q { font-size: .8em; color: var(--ink-soft); }

  /* ── 構成テーブル ───────────────────────────────────────── */
  .tw { overflow-x: auto; margin-inline: calc(var(--pad) * -1); padding-inline: var(--pad); }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; }
  th {
    text-align: left; font-weight: 500; font-size: .72rem;
    letter-spacing: .08em; text-transform: uppercase; color: var(--ink-soft);
    border-bottom: 1px solid var(--rule); padding: 0 .6em .5em 0; white-space: nowrap;
  }
  td { padding: .62em .6em .62em 0; border-bottom: 1px solid var(--rule); }
  th.r, td.r { text-align: right; padding-right: 0; }
  tr:last-child td { border-bottom: none; }
  .tick { font-family: var(--mono); font-weight: 600; }
  .nm { color: var(--ink-mid); font-size: .84em; }
  .sec {
    font-size: .7rem; color: var(--ink-soft); border: 1px solid var(--rule);
    border-radius: 2px; padding: .1em .4em; white-space: nowrap;
  }
  tfoot td { border-bottom: none; border-top: 2px solid var(--rule-hi);
             padding-top: .7em; font-weight: 600; }
  .pos { color: var(--up); } .neg { color: var(--down); }

  /* ── 順位表と境界線（この戦略の中心装置）─────────────────── */
  /* subgrid は Safari 16+ 限定なので使わず、各行を独立したグリッドにする。
     見た目は同じで、古い端末でも崩れない。 */
  .lrow {
    display: grid; grid-template-columns: 2.2em 1fr auto; gap: 0 .6em;
    align-items: baseline; padding: .5em .55em; margin-inline: -.55em;
    border-bottom: 1px solid var(--rule);
  }
  .lrow.in { background: var(--signal-bg); }
  .lrow .rk { font-family: var(--mono); font-size: .82rem; color: var(--ink-soft); }
  .lrow.in .rk { color: var(--signal); font-weight: 700; }
  .lrow .who { min-width: 0; }
  .lrow .who b { font-family: var(--mono); font-size: .92rem; }
  .lrow .who span { display: block; font-size: .78rem; color: var(--ink-soft); }
  .lrow .ret { font-family: var(--mono); font-variant-numeric: tabular-nums;
               font-size: .9rem; text-align: right; }
  .cut {
    display: flex; align-items: center; gap: .8em;
    padding: .5em 0; border-bottom: 2px solid var(--signal);
  }
  .cut span {
    font-size: .7rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--signal); font-weight: 700; white-space: nowrap;
  }
  .cut i { flex: 1; height: 0; }

  /* ── 操作 ───────────────────────────────────────────────── */
  .ctl { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-end; }
  .fld { display: flex; flex-direction: column; gap: .4em; }
  .fld > label {
    font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
    color: var(--ink-soft);
  }
  .seg { display: flex; border: 1px solid var(--rule); border-radius: 2px; overflow: hidden; }
  .seg button {
    font: inherit; font-size: .88rem; padding: .5em .9em; border: 0;
    background: var(--surface); color: var(--ink-mid); cursor: pointer;
    font-family: var(--mono);
  }
  /* 選択中は反転。--surface を文字色に使うことで両テーマでコントラストが保たれる */
  .seg button[aria-pressed="true"] { background: var(--signal); color: var(--surface); }
  input[type="number"] {
    font: inherit; font-family: var(--mono); font-size: .95rem;
    padding: .5em .6em; width: 11ch; color: var(--ink);
    background: var(--surface); border: 1px solid var(--rule); border-radius: 2px;
  }
  button.act {
    font: inherit; font-size: .92rem; padding: .7em 1.2em; cursor: pointer;
    background: var(--ink); color: var(--paper);
    border: 1px solid var(--ink); border-radius: 2px;
  }
  button.act.ghost { background: transparent; color: var(--ink-mid); border-color: var(--rule); }
  button:focus-visible, input:focus-visible, summary:focus-visible {
    outline: 2px solid var(--signal); outline-offset: 2px;
  }

  /* ── 警告 ───────────────────────────────────────────────── */
  .warn {
    background: var(--warn-bg); border: 1px solid var(--warn);
    border-radius: 3px; padding: 1em var(--pad); margin-block: 14px;
    font-size: .9rem; color: var(--ink);
  }
  .warn b { color: var(--warn); }
  .facts { display: grid; gap: 0; margin: 0; }
  .facts div {
    display: flex; justify-content: space-between; gap: 1em;
    padding: .55em 0; border-bottom: 1px solid var(--rule); font-size: .9rem;
  }
  .facts div:last-child { border-bottom: none; }
  .facts dt { color: var(--ink-soft); margin: 0; }
  .facts dd { margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums; }
  ul.notes { margin: 0; padding-left: 1.3em; font-size: .9rem; color: var(--ink-mid); }
  ul.notes li { margin-block: .5em; }
  ul.notes strong { color: var(--ink); }
  details { margin-top: 14px; }
  summary { cursor: pointer; font-size: .88rem; color: var(--signal); }
  footer {
    margin-block: 40px 24px; padding-top: 20px; border-top: 1px solid var(--rule);
    font-size: .78rem; color: var(--ink-soft);
  }
  .toast {
    position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%) translateY(120%);
    background: var(--ink); color: var(--paper); padding: .7em 1.3em;
    border-radius: 2px; font-size: .88rem; transition: transform .25s ease; z-index: 9;
  }
  .toast.on { transform: translateX(-50%) translateY(0); }
  @media (prefers-reduced-motion: reduce) { .toast { transition: none; } }
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">月次リバランス</p>
    <h1>上がっているものを、持つ。</h1>
    <p class="rule-line">
      丸1年前の終値と比べて最も上がっていた銘柄を、月に1度だけ入れ替えて持つ。
      売買タイミングは判定しない — 検証で一貫して損だったため。
    </p>
    <p class="asof">
      測定区間 <span class="num" id="from"></span> 〜 <span class="num" id="asof"></span>
      （この日の終値で判定し、翌営業日に売買）<br />
      対象 <span class="num" id="univ"></span> 銘柄 ／
      ドル円 <span class="num" id="fx"></span>
    </p>
  </header>

  <section class="card verdict" id="verdict">
    <h2>今月の指示</h2>
    <p class="sub" id="held-note"></p>
    <p class="action-h" id="action"></p>
    <div class="moves" id="moves"></div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:20px">
      <button class="act" id="btn-apply">この構成に入れ替えた</button>
      <button class="act ghost" id="btn-clear">保有を消す</button>
    </div>
  </section>

  <section class="card">
    <h2>保有すべき構成</h2>
    <p class="sub" id="budget"></p>
    <div class="tw">
      <table>
        <thead>
          <tr>
            <th>銘柄</th><th>セクター</th>
            <th class="r">12ヶ月</th><th class="r">株数</th><th class="r">必要額</th>
          </tr>
        </thead>
        <tbody id="alloc"></tbody>
        <tfoot id="alloc-foot"></tfoot>
      </table>
    </div>
    <div id="alerts"></div>
  </section>

  <section class="card">
    <h2>設定</h2>
    <p class="sub">変更するとこの端末に保存されます</p>
    <div class="ctl">
      <div class="fld">
        <label id="lbl-topn">保有銘柄数</label>
        <div class="seg" role="group" aria-labelledby="lbl-topn" id="seg-topn"></div>
      </div>
      <div class="fld">
        <label for="cap">投資資金（円）</label>
        <input type="number" id="cap" min="100000" step="100000" />
      </div>
    </div>
  </section>

  <section class="card">
    <h2>順位表</h2>
    <p class="sub">この線の上を持つ。線の下は持たない。それだけがルール。</p>
    <div class="ledger" id="ledger"></div>
    <details>
      <summary id="more-sum">下位の銘柄も見る</summary>
      <div class="ledger" id="ledger-rest" style="margin-top:10px"></div>
    </details>
  </section>

  <section class="card">
    <h2>このルールの検証結果</h2>
    <p class="sub">配当込み・売買コスト0.2%・譲渡益税20.315% を反映した後の数字</p>
    <dl class="facts">
      <div><dt>2006〜2015年（リーマンショックを含む）</dt><dd class="pos">+460%</dd></div>
      <div><dt>　同期間に全銘柄を持ち続けた場合</dt><dd>+174%</dd></div>
      <div><dt>2016〜2026年（AI相場）</dt><dd class="pos">+1,951%</dd></div>
      <div><dt>　同期間に全銘柄を持ち続けた場合</dt><dd>+1,308%</dd></div>
      <div><dt>最大の落ち込み</dt><dd class="neg">−57% / −41%</dd></div>
    </dl>
    <div class="warn" style="margin-top:18px">
      <b>途中で資産が半分近くになります。</b>
      増える道中で −57% の局面を通過しました。300万円が129万円に見える時期を
      持ち続けられるかが、この戦略の成否をすべて決めます。
    </div>
  </section>

  <section class="card">
    <h2>必ず覚えておくこと</h2>
    <ul class="notes">
      <li><strong>月1回だけ開く。</strong>頻繁に見ても意味がなく、売買が増えて手数料と税金だけが増えます。</li>
      <li><strong>タイミングを足さない。</strong>「相場が悪いから今月は現金で」をやると、検証では
          +1,951% が +636% に落ちました。後知恵で勝ち組を選んだ場合でも
          +11,208% → +1,579% に落ちています。悪手だと分かっている操作です。</li>
      <li><strong>損切りはありません。</strong>下がった銘柄は順位が落ちて自動的に外れます。
          それが唯一の撤退ルールです。</li>
      <li><strong>検証は今も上場している銘柄だけで行っています。</strong>この20年で潰れた会社は
          入っていないため、実際の成績はこれより悪くなります。</li>
      <li><strong>これは学術的に知られた現象です。</strong>「モメンタム効果」と呼ばれ、私の発明では
          ありません。同時に、壊滅的に失敗する時期があることも知られています
          （有名なのは2009年、暴落からの反発局面）。</li>
      <li><strong>自分専用です。</strong>他人に有償で売買判断を提供すると、金融商品取引法上の
          投資助言・代理業の登録が必要になります。</li>
    </ul>
  </section>

  <footer>
    データ生成 <span class="num" id="gen"></span>／
    順位は配当込み・円換算のトータルリターンで算出。株数は実勢価格ベース。
    日本株は100株単位のため、1単元が予算を超える銘柄は「買付不可」と表示されます。
    発注前に必ず証券会社の板で最終確認してください。
  </footer>
</div>

<div class="toast" id="toast"></div>

<script>
  const DATA = __DATA__;
  const KEY = "momentum-holdings-v1";
  const CFG = "momentum-config-v1";
  const TOPS = [3, 5, 10];

  const yen = n => Math.round(n).toLocaleString("ja-JP") + "円";
  const pct = n => (n >= 0 ? "+" : "") + (n * 100).toFixed(1) + "%";
  const el = id => document.getElementById(id);

  let cfg = { topN: DATA.top_n, capital: DATA.capital };
  try { Object.assign(cfg, JSON.parse(localStorage.getItem(CFG) || "{}")); } catch (e) {}
  if (!TOPS.includes(cfg.topN)) cfg.topN = 5;

  const readHold = () => {
    try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { return null; }
  };
  const saveHold = syms => localStorage.setItem(KEY,
    JSON.stringify({ symbols: syms, updated: new Date().toISOString().slice(0, 10) }));
  const saveCfg = () => localStorage.setItem(CFG, JSON.stringify(cfg));

  function shares(row, perPos) {
    if (!row.price_jpy || !row.lot) return null;
    const q = Math.floor((perPos / row.price_jpy) / row.lot) * row.lot;
    return { qty: q, cost: q * row.price_jpy, ok: q > 0, min: row.lot * row.price_jpy };
  }

  function toast(msg) {
    const t = el("toast");
    t.textContent = msg; t.classList.add("on");
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), 2200);
  }

  function ledgerRow(r, inside) {
    const d = document.createElement("div");
    d.className = "lrow" + (inside ? " in" : "");
    d.innerHTML =
      '<div class="rk">' + r.rank + '</div>' +
      '<div class="who"><b></b><span></span></div>' +
      '<div class="ret ' + (r.return_12m >= 0 ? "pos" : "neg") + '"></div>';
    d.querySelector("b").textContent = r.symbol;
    d.querySelector("span").textContent = r.name + " · " + r.sector;
    d.querySelector(".ret").textContent = pct(r.return_12m);
    return d;
  }

  function render() {
    el("asof").textContent = DATA.as_of;
    el("from").textContent = DATA.lookback_from || "―";
    el("univ").textContent = DATA.ranking.length;
    el("fx").textContent = DATA.usdjpy.toFixed(2);
    el("gen").textContent = DATA.generated;
    el("cap").value = cfg.capital;

    const n = cfg.topN;
    const perPos = cfg.capital / n;
    const top = DATA.ranking.slice(0, n);
    const topSyms = top.map(r => r.symbol);

    // ── 保有構成
    const body = el("alloc"); body.textContent = "";
    let total = 0; const bad = [];
    top.forEach(r => {
      const s = shares(r, perPos);
      if (s && s.ok) total += s.cost; else bad.push(r);
      const tr = document.createElement("tr");
      tr.innerHTML = '<td><span class="tick"></span><span class="nm"></span></td>' +
        '<td><span class="sec"></span></td>' +
        '<td class="r num"></td><td class="r num"></td><td class="r num"></td>';
      const td = tr.children;
      td[0].querySelector(".tick").textContent = r.symbol;
      td[0].querySelector(".nm").textContent = " " + r.name;
      td[1].querySelector(".sec").textContent = r.sector;
      td[2].textContent = pct(r.return_12m);
      td[2].className = "r num " + (r.return_12m >= 0 ? "pos" : "neg");
      td[3].textContent = s && s.ok ? s.qty.toLocaleString("ja-JP") : "―";
      td[4].textContent = s && s.ok ? yen(s.cost) : "買付不可";
      body.appendChild(tr);
    });
    el("alloc-foot").innerHTML =
      '<tr><td colspan="4">合計</td><td class="r num">' + yen(total) + "</td></tr>";
    el("budget").textContent =
      "1銘柄あたり " + yen(perPos) + "（資金 " + yen(cfg.capital) + " ÷ " + n + "銘柄）";

    // ── 警告（買付不可・セクター集中）
    const al = el("alerts"); al.textContent = "";
    bad.forEach(r => {
      const s = shares(r, perPos);
      const d = document.createElement("div"); d.className = "warn";
      d.innerHTML = "<b>" + r.symbol + " は買えません。</b>" +
        (s ? "1単元 " + r.lot + "株 = " + yen(s.min) + " で、1銘柄予算 " + yen(perPos) +
             " を超えます。単元未満株が使えないなら次順位に繰り下げてください。"
           : "現在値が取得できませんでした。");
      al.appendChild(d);
    });
    const cnt = {};
    top.forEach(r => cnt[r.sector] = (cnt[r.sector] || 0) + 1);
    const worst = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0];
    if (worst && worst[1] >= Math.ceil(n * 0.6)) {
      const d = document.createElement("div"); d.className = "warn";
      d.innerHTML = "<b>" + worst[1] + "銘柄が「" + worst[0] + "」に集中しています。</b>" +
        "このルールはセクター分散を考慮しません。" + worst[0] +
        "が崩れると同時に下落します。分散されていない前提で金額を決めてください。";
      al.appendChild(d);
    }

    // ── 今月の指示
    const held = readHold();
    const cur = held && Array.isArray(held.symbols) ? held.symbols : [];
    const sell = cur.filter(s => !topSyms.includes(s));
    const buy = topSyms.filter(s => !cur.includes(s));
    const keep = cur.filter(s => topSyms.includes(s));
    const vd = el("verdict");
    const mv = el("moves"); mv.textContent = "";

    el("held-note").textContent = cur.length
      ? "登録済みの保有 " + cur.length + "銘柄（" + (held.updated || "日付なし") + " 時点）"
      : "保有がまだ登録されていません";

    if (!cur.length) {
      vd.classList.remove("hold");
      el("action").textContent = "上位" + n + "銘柄を買い付けてください";
      addGroup(mv, "買う", topSyms, "buy", top, perPos);
    } else if (!sell.length && !buy.length) {
      vd.classList.add("hold");
      el("action").textContent = "入れ替え不要。そのまま保有を継続。";
      addGroup(mv, "継続保有", keep, "", top, perPos);
    } else {
      vd.classList.remove("hold");
      el("action").textContent =
        sell.length + "銘柄を売り、" + buy.length + "銘柄を買う";
      if (sell.length) addGroup(mv, "売る", sell, "sell", DATA.ranking, perPos);
      if (buy.length) addGroup(mv, "買う", buy, "buy", top, perPos);
      if (keep.length) addGroup(mv, "継続保有", keep, "", top, perPos);
    }

    // ── 順位表と境界線
    const lg = el("ledger"); lg.textContent = "";
    DATA.ranking.slice(0, Math.max(n + 5, 12)).forEach((r, i) => {
      if (i === n) {
        const c = document.createElement("div");
        c.className = "cut";
        c.innerHTML = '<span>ここから下は持たない</span><i></i>';
        lg.appendChild(c);
      }
      lg.appendChild(ledgerRow(r, i < n));
    });
    const rest = DATA.ranking.slice(Math.max(n + 5, 12));
    const lr = el("ledger-rest"); lr.textContent = "";
    rest.forEach(r => lr.appendChild(ledgerRow(r, false)));
    el("more-sum").textContent = "下位 " + rest.length + " 銘柄も見る";

    // ── セグメント
    const seg = el("seg-topn"); seg.textContent = "";
    TOPS.forEach(v => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = v;
      b.setAttribute("aria-pressed", v === n ? "true" : "false");
      b.setAttribute("aria-label", "上位" + v + "銘柄");
      b.onclick = () => { cfg.topN = v; saveCfg(); render(); };
      seg.appendChild(b);
    });
  }

  function addGroup(parent, label, syms, kind, source, perPos) {
    const g = document.createElement("div"); g.className = "move-grp";
    const l = document.createElement("div"); l.className = "move-lbl";
    l.textContent = label; g.appendChild(l);
    const c = document.createElement("div"); c.className = "chips";
    syms.forEach(sym => {
      const r = source.find(x => x.symbol === sym) ||
                DATA.ranking.find(x => x.symbol === sym) || { symbol: sym, name: sym };
      const s = kind === "buy" ? shares(r, perPos) : null;
      const sp = document.createElement("span");
      sp.className = "chip " + kind;
      const t = document.createElement("span"); t.className = "t"; t.textContent = sym;
      const q = document.createElement("span"); q.className = "q";
      q.textContent = r.name + (s && s.ok ? " " + s.qty.toLocaleString("ja-JP") + "株"
                       : (r.rank ? " " + r.rank + "位" : ""));
      sp.appendChild(t); sp.appendChild(q); c.appendChild(sp);
    });
    g.appendChild(c); parent.appendChild(g);
  }

  el("btn-apply").onclick = () => {
    saveHold(DATA.ranking.slice(0, cfg.topN).map(r => r.symbol));
    render(); toast("保有を更新しました");
  };
  el("btn-clear").onclick = () => {
    localStorage.removeItem(KEY); render(); toast("保有を消しました");
  };
  el("cap").onchange = e => {
    const v = Number(e.target.value);
    if (v >= 100000) { cfg.capital = v; saveCfg(); render(); }
    else { e.target.value = cfg.capital; toast("10万円以上を入れてください"); }
  };

  render();
</script>
"""


def _standalone(body: str) -> str:
    """GitHub Pages 用に完全なHTML文書として出力する。

    Artifact は <head>/<body> を自動で付けてくれるが、GitHub Pages は
    素のHTMLをそのまま配信するため、こちらでは自分で骨格を書く必要がある。
    """
    return (
        '<!doctype html>\n<html lang="ja">\n<head>\n'
        '<meta charset="utf-8" />\n'
        '<meta name="theme-color" content="#1e46c8" />\n'
        '<meta name="apple-mobile-web-app-capable" content="yes" />\n'
        '<meta name="apple-mobile-web-app-title" content="モメンタム" />\n'
        '<meta name="robots" content="noindex" />\n'
        + body.split("<style>")[0]
        + "<style>" + body.split("<style>", 1)[1].split("</style>")[0] + "</style>\n"
        "</head>\n<body>\n"
        + body.split("</style>", 1)[1]
        + "\n</body>\n</html>\n"
    )


def main() -> int:
    if not DATA.exists():
        print(f"\n  [エラー] {DATA.name} がありません。先にこれを実行してください:")
        print(f"    .venv/bin/python momentum.py --json {DATA.name}\n")
        return 1

    data = json.loads(DATA.read_text())
    # </script> がJSON内に現れるとHTMLが壊れるのでエスケープ
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)

    # 単体ファイル（Artifact公開用）
    OUT.write_text(html)

    # GitHub Pages 用。docs/ を公開ディレクトリにすると
    # https://<user>.github.io/<repo>/ で開ける
    docs = HERE / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(_standalone(html))

    sel = [r for r in data["ranking"] if r["selected"]]
    print(f"\n  {OUT.name} を生成しました（{OUT.stat().st_size / 1024:.0f} KB）")
    print(f"  docs/index.html も生成しました（GitHub Pages 用）")
    print(f"  基準日 {data['as_of']} / 上位{data['top_n']}銘柄: "
          f"{'、'.join(r['symbol'] for r in sel)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
