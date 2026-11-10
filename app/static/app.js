// AMD 10-minute indicator: four pages (Live, History, Performance, How to read) on one HTML page.
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const secs = iso => Date.parse(iso) / 1000;
const ET_FMT = new Intl.DateTimeFormat('en-GB', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit' });
const etTime = iso => ET_FMT.format(new Date(iso));
const localTime = ms => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const money = x => '$' + x.toFixed(2);
const pct = (x, d = 2) => (x >= 0 ? '+' : '') + (x * 100).toFixed(d) + '%';
const usd = x => (x >= 0 ? '+$' : '−$') + Math.abs(x).toFixed(2);
const share = (a, b) => b ? `${Math.round(100 * a / b)}%` : '–';
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const dirOf = g => g.lean === 'weak up' ? 1 : g.lean === 'weak down' ? -1 : 0;
const confOf = g => { const s = g.p_up + g.p_down; return s > 0 ? Math.max(g.p_up, g.p_down) / s : 0.5; };
const NAMES = { indicator: 'Our indicator', always_up: 'Always Up', rsi_rule: 'RSI rule' };
const REVIEW = 'Review of a past session: every signal is scored. Each was recomputed after the close from the data available at its own minute only (the same results as live), but none was shown live.';
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
const getJSON = async url => { const r = await fetch(url); if (!r.ok) throw new Error(r.status); return r.json(); };

// ── chart (one per page) ───────────────────────────────────────
function makeChart(el, onPick) {
  const chart = LightweightCharts.createChart(el, { autoSize: true,
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 4 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: { mouseWheel: false }, handleScale: { mouseWheel: false } });   // wheel handled below
  const candles = chart.addCandlestickSeries({ borderVisible: false });
  const line = (style, width) => chart.addLineSeries({ lineWidth: width, lineStyle: style,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  const bandHi = line(LightweightCharts.LineStyle.Dashed, 2), bandLo = line(LightweightCharts.LineStyle.Dashed, 2);
  const typHi = line(LightweightCharts.LineStyle.Dotted, 1), typLo = line(LightweightCharts.LineStyle.Dotted, 1);
  const strip = chart.addHistogramSeries({ priceScaleId: 'dir', priceLineVisible: false, lastValueVisible: false });
  chart.priceScale('dir').applyOptions({ scaleMargins: { top: 0.9, bottom: 0 }, visible: false });
  candles.priceScale().applyOptions({ scaleMargins: { top: 0.05, bottom: 0.14 } });
  let offset = 0, last = null, autoFit = true, session = null;
  ['mousedown', 'touchstart'].forEach(e => el.addEventListener(e, () => { autoFit = false; }, { passive: true }));
  const ts = chart.timeScale();
  function zoom(factor, x) {   // factor < 1 zooms in, around pixel x (default: the middle)
    const r = ts.getVisibleLogicalRange(); if (!r) return;
    const a = x == null ? (r.from + r.to) / 2 : ts.coordinateToLogical(x);
    if (a == null) return;
    const from = a - (a - r.from) * factor, to = a + (r.to - a) * factor;
    if (to - from < 8) return;
    autoFit = false; ts.setVisibleLogicalRange({ from, to });
  }
  el.addEventListener('wheel', e => {   // the plain wheel scrolls the page; pinch or Ctrl/Cmd + wheel zooms; sideways swipe pans
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault(); zoom(Math.exp(e.deltaY * 0.01), e.offsetX);
    } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      e.preventDefault(); const r = ts.getVisibleLogicalRange(); if (!r) return;
      const shift = e.deltaX / ts.options().barSpacing; autoFit = false;
      ts.setVisibleLogicalRange({ from: r.from + shift, to: r.to + shift });
    }
  }, { passive: false });
  const tools = document.createElement('div');
  tools.className = 'charttools';
  tools.innerHTML = '<button type="button" data-z="in" aria-label="Zoom in">+</button><button type="button" data-z="out" aria-label="Zoom out">−</button><button type="button" data-z="fit">Fit</button>';
  tools.onclick = e => { const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.z === 'fit') { autoFit = true; ts.fitContent(); } else zoom(b.dataset.z === 'in' ? 0.6 : 1.6); };
  el.parentElement.style.position = 'relative'; el.parentElement.appendChild(tools);
  const t = iso => secs(iso) + offset;
  chart.subscribeClick(p => {
    if (!p.time || !last) return;
    const g = last.signals.find(x => t(x.ts) === p.time);
    onPick(g ? g.ts : null);
  });
  function theme() {
    chart.applyOptions({ layout: { background: { color: 'transparent' }, textColor: css('--ink-2'), fontSize: 11 },
      grid: { vertLines: { color: css('--line') }, horzLines: { color: css('--line') } },
      rightPriceScale: { borderColor: css('--line') }, timeScale: { borderColor: css('--line') } });
    candles.applyOptions({ upColor: css('--up'), downColor: css('--down'), wickUpColor: css('--up'), wickDownColor: css('--down') });
    [bandHi, bandLo].forEach(x => x.applyOptions({ color: css('--band') }));
    [typHi, typLo].forEach(x => x.applyOptions({ color: css('--typ') }));
    if (last) render(last);
  }
  function render(s) {
    last = s; offset = s.et_offset_seconds;
    if (s.session !== session) { session = s.session; autoFit = true; }
    const bars = s.bars.map(b => ({ time: t(b.time), open: b.open, high: b.high, low: b.low, close: b.close,
      ...(b.filled ? { color: css('--ink-2'), wickColor: css('--ink-2') } : {}) }));
    candles.setData(bars);
    const h = s.horizon_minutes * 60;
    const at = k => s.signals.map(g => ({ time: t(g.ts) + h, value: g[k] }));
    bandHi.setData(at('band_hi')); bandLo.setData(at('band_lo'));
    typHi.setData(at('typical_hi')); typLo.setData(at('typical_lo'));
    const withP = s.signals.filter(g => g.p_up != null);
    strip.setData(withP.map(g => { const d = dirOf(g);
      return { time: t(g.ts), value: Math.max(confOf(g) - 0.5, 0.01),
        color: d > 0 ? css('--up') : d < 0 ? css('--down') : css('--line') }; }));
    const times = new Set(bars.map(b => b.time));
    const arrows = withP.filter(g => dirOf(g) !== 0).map(g => dirOf(g) > 0
      ? { time: t(g.ts), position: 'belowBar', color: css('--up'), shape: 'arrowUp' }
      : { time: t(g.ts), position: 'aboveBar', color: css('--down'), shape: 'arrowDown' });
    const news = s.news.map(n => ({ time: t(n.time), position: 'aboveBar', color: css('--news'), shape: 'circle', text: 'N' }));
    candles.setMarkers(arrows.concat(news).filter(m => times.has(m.time)).sort((a, b) => a.time - b.time));
    if (bars.length) fit();
  }
  function fit() { if (autoFit) setTimeout(() => chart.timeScale().fitContent(), 80); }
  new ResizeObserver(() => { if (last) fit(); }).observe(el);   // a chart drawn while hidden had no width
  function clear() { last = null; session = null;
    [candles, bandHi, bandLo, typHi, typLo, strip].forEach(x => x.setData([])); candles.setMarkers([]); }
  theme();
  return { render, theme, clear, fit };
}

// ── shared pieces ──────────────────────────────────────────────
const pickSig = (s, ts) => (ts && s.signals.find(g => g.ts === ts)) || s.signals[s.signals.length - 1];

function dirText(sig) {
  if (!sig || sig.p_up == null) return { cls: 'none', text: 'warming up', sub: 'needs 30 minutes of data' };
  const d = dirOf(sig), conf = Math.round(100 * confOf(sig));
  const pu = Math.round(100 * sig.p_up / (sig.p_up + sig.p_down));
  return { cls: d > 0 ? 'up' : d < 0 ? 'down' : 'none',
    text: d > 0 ? `▲ UP · ${conf}%` : d < 0 ? `▼ DOWN · ${conf}%` : 'No clear signal',
    sub: d ? 'weak hint, about a coin flip in testing' : `up/down ${pu}/${100 - pu}: too close to call` };
}

function renderKpis(root, sig) {
  if (!sig) { root.innerHTML = ''; return; }
  const d = dirText(sig);
  root.innerHTML = `
    <div class="kpi"><span class="lab">Typical move</span><span class="val">±${money(sig.close * sig.typical_bps / 1e4)}</span>
      <span class="sub">${(sig.typical_bps / 100).toFixed(2)}% · half of moves are smaller</span></div>
    <div class="kpi"><span class="lab">80% range</span><span class="val">${money(sig.band_lo)} – ${money(sig.band_hi)}</span>
      <span class="sub">close at ${etTime(new Date(secs(sig.ts) * 1000 + 600000).toISOString())} ET, 8 times in 10</span></div>
    <div class="kpi"><span class="lab">Activity</span><span class="val level ${sig.level}">${sig.level}</span>
      <span class="sub">${sig.activity.toFixed(1)}× normal for this time${sig.warning ? ' · bigger moves likely' : ''}</span></div>
    <div class="kpi"><span class="lab">Direction</span><span class="val dir ${d.cls}">${d.text}</span><span class="sub">${d.sub}</span></div>`;
}

function renderPanel(root, s, sig, pinned, onLatest) {
  if (!sig) { root.innerHTML = '<div class="card muted">No signals yet for this session.</div>'; return; }
  const out = s.outcomes.find(o => o.ts === sig.ts);
  const d = dirText(sig);
  const isLatest = sig === s.signals[s.signals.length - 1];
  const viewing = isLatest ? '' : ` <span class="viewing">· at ${etTime(sig.ts)} ET</span> <button class="linkbtn" data-latest>latest</button>`;
  let happened = 'Outcome known 10 minutes after this signal.';
  if (out) {
    const mv = out.ret_bps / 100, dd = dirOf(sig);
    happened = `What happened: <b class="${mv >= 0 ? 'pos' : 'neg'}">${mv >= 0 ? '+' : ''}${mv.toFixed(2)}%</b> by ${etTime(out.target_ts)} ET, ` +
      (out.in_band ? 'inside' : 'outside') + ' the 80% range' +
      (dd ? (Math.sign(out.ret_bps) === dd ? '; direction call right.' : '; direction call wrong.') : '.');
  }
  const p = k => Math.round(100 * sig[k]);
  const recent = s.news.filter(n => secs(n.time) < secs(sig.ts) + 60).slice(-4).reverse();
  root.innerHTML = `
    <div class="card">
      <h2>Next 10 minutes${viewing}</h2>
      <div class="row"><span>Price at this minute</span><span class="v">${money(sig.close)}</span></div>
      <div class="row"><span>Typical move</span><span class="v big">±${money(sig.close * sig.typical_bps / 1e4)}</span></div>
      <div class="row"><span>80% range</span><span class="v">${money(sig.band_lo)} – ${money(sig.band_hi)}</span></div>
      <div class="row"><span>Activity</span><span class="v"><span class="level ${sig.level}">${sig.level}</span> <span class="muted">${sig.activity.toFixed(1)}×</span></span></div>
      ${sig.warning ? '<div class="warn">Bigger moves than usual are likely</div>' : ''}
      <p class="note">${happened}${sig.late && !s.record.review ? ' <i>Made late (after a restart): not counted in the live score.</i>' : ''}</p>
    </div>
    <div class="card">
      <h2>Direction</h2>
      <div class="row"><span class="dir ${d.cls}">${d.text}</span><span class="v muted">${d.cls === 'none' ? '' : 'confident signal'}</span></div>
      ${sig.p_up == null ? '' : `
      <div class="pbar"><span style="width:${p('p_down')}%;background:var(--down)"></span><span style="width:${p('p_flat')}%;background:var(--line)"></span><span style="width:${p('p_up')}%;background:var(--up)"></span></div>
      <div class="row muted small" style="border:0"><span>down ${p('p_down')}%</span><span>flat ${p('p_flat')}%</span><span>up ${p('p_up')}%</span></div>`}
      <p class="note">In testing on 2023–2026 data, confident signals were right about 50% of the time. Treat them as a weak hint and rely on the range.</p>
    </div>
    <div class="card">
      <h2>Context</h2>
      <div class="row"><span>QQQ, last 10 min</span><span class="v">${sig.qqq_ret_10 == null ? '–' : pct(sig.qqq_ret_10)}</span></div>
      <div class="row"><span>NVDA, last 10 min</span><span class="v">${sig.nvda_ret_10 == null ? '–' : pct(sig.nvda_ret_10)}</span></div>
      <div class="row"><span>AMD headlines, last 30 min</span><span class="v">${sig.news_30}</span></div>
      <div class="row"><span>Minutes without IEX trades (last 30)</span><span class="v">${Math.round(sig.filled_30 * 30)}</span></div>
      ${recent.length ? `<div class="news">${recent.map(n => `<div><span class="muted">${etTime(n.time)}</span> ${esc(n.headline)}</div>`).join('')}</div>` : ''}
    </div>`;
  const b = root.querySelector('[data-latest]'); if (b) b.onclick = onLatest;
}

function scoreRows(sc, lr) {
  const rows = [['indicator', sc.indicator, NAMES.indicator]];
  if (lr && lr.scorecard.indicator) rows.push(['live', lr.scorecard.indicator, `↳ shown live (${etTime(lr.first)}–${etTime(lr.last)} ET)`]);
  rows.push(['always_up', sc.always_up, NAMES.always_up], ['rsi_rule', sc.rsi_rule, NAMES.rsi_rule]);
  return rows.map(([k, r, name]) => {
    const acc = r.signals ? `${r.correct} (${Math.round(100 * r.accuracy)}%)` : '–';
    const pnl = r.trades ? `<span class="${r.pnl_usd >= 0 ? 'pos' : 'neg'}">${usd(r.pnl_usd)}</span>` : '–';
    return `<tr class="${k === 'indicator' ? 'ours' : k === 'live' ? 'sub' : ''}"><td>${name}</td><td>${r.signals}</td><td>${acc}</td><td>${r.trades}</td><td>${pnl}</td></tr>`;
  }).join('');
}

function renderScore(root, s, title) {
  const sc = s.scorecard || {}, lr = s.live_record, c = s.costs;
  root.innerHTML = `<h2>${title}</h2>` + (sc.indicator ? `
    <div class="tablewrap"><table class="data"><thead><tr><th></th><th>Signals</th><th>Right</th><th>Trades</th><th>Profit</th></tr></thead>
    <tbody>${scoreRows(sc, lr)}</tbody></table></div>
    <p class="note">“Right” = the price 10 minutes later moved the way the signal said. Profit after costs on paper trades: ${c.shares} shares, held 10 minutes, one at a time, $${(2 * c.per_share_each_way).toFixed(2)}/share round trip. Scored on IEX closes. ${s.record.review ? REVIEW : 'Signals made late are not counted.'}${lr ? ' The “shown live” row counts only what the app showed on time that day.' : ''} One day is a small sample.</p>`
    : '<p class="muted">No scored signals yet: a signal is scored 10 minutes after it is made.</p>');
}

function renderRange(root, s, title) {
  const r = s.record, lr = s.live_record;
  const line = (a, b, target) => b ? `${a} of ${b} <span class="muted">(${share(a, b)}, target ${target})</span>` : 'none yet';
  root.innerHTML = `<h2>${title}</h2>
    <div class="row"><span>Inside the 80% range</span><span class="v">${line(r.in_band, r.scored, '80%')}</span></div>
    <div class="row"><span>Inside the typical move</span><span class="v">${line(r.in_typical, r.scored, '50%')}</span></div>
    ${lr && lr.record.scored ? `<div class="row"><span>Shown live, ${etTime(lr.first)}–${etTime(lr.last)} ET</span><span class="v">${line(lr.record.in_band, lr.record.scored, '80%')}</span></div>` : ''}
    <p class="note">Scored on IEX closes 10 minutes later. ${r.review ? REVIEW : (r.late_signals ? `${r.late_signals} signals made late (computed after the fact) are not counted.` : '')}</p>`;
}

// ── status and clocks (sidebar) ────────────────────────────────
let market = null;
function renderStatus(st) {
  const kind = st.error ? 'error' : st.mode === 'replay' ? 'review' : st.review ? 'review' : 'live';
  $('statusdot').className = 'dot ' + kind;
  $('statustext').textContent = st.error ? 'Data error, retrying' : st.mode === 'replay' ? 'REPLAY' : st.review ? 'REVIEW' : 'LIVE';
}
function renderClock() {
  const now = Date.now();
  $('clocks').textContent = `New York ${ET_FMT.format(new Date(now))} · You ${localTime(now)}`;
  if (!market) return;
  const span = ms => { const m = Math.max(0, Math.round(ms / 60000)); return m >= 60 ? `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m` : `${m}m`; };
  $('market').textContent = market.open_now ? `Market open · closes in ${span(Date.parse(market.closes) - now)}`
    : `Market closed · opens in ${span(Date.parse(market.next_open) - now)} (${localTime(Date.parse(market.next_open))} your time)`;
}

// ── LIVE page ──────────────────────────────────────────────────
let live = null, livePinned = null;
let liveChart = null;   // charts are created when their page is first shown: a chart made while hidden has no width
function renderLive(s) {
  live = s;
  const st = s.status || {};
  renderStatus(st); market = s.market || market; renderClock();
  $('live-msg').textContent = st.error ? st.error : (st.message || '');
  if (!s.session) return;
  if (!liveChart && !$('page-live').hidden) liveChart = makeChart($('live-chart'), ts => { livePinned = ts; renderLive(live); });
  if (liveChart) liveChart.render(s);
  const sig = s.signals.length ? pickSig(s, livePinned) : null;
  renderKpis($('live-kpis'), sig);
  renderPanel($('live-panel'), s, sig, livePinned, () => { livePinned = null; renderLive(live); });
  renderScore($('live-score'), s, s.record.review ? `Direction scorecard · review of ${s.session}` : `Direction scorecard · live, ${s.session}`);
  renderRange($('live-range'), s, s.record.review ? `Range track record · review of ${s.session}` : `Range track record · live, ${s.session}`);
  if (s.bars.length) {
    const first = s.bars[0], last = s.bars[s.bars.length - 1], ch = last.close / first.open - 1;
    $('live-price').textContent = money(last.close);
    $('live-chg').innerHTML = `<span class="${ch >= 0 ? 'pos' : 'neg'}">${pct(ch)}</span> <span class="muted small">since the open</span>`;
    $('live-asof').textContent = `${s.session} · bar ${etTime(last.time)} ET`;
  }
  localTimes(s);
}
async function pollLive() {
  try { renderLive(await getJSON('/api/state')); }
  catch (e) { $('statusdot').className = 'dot error'; $('statustext').textContent = 'offline'; }
  setTimeout(pollLive, live && live.status && live.status.mode === 'replay' ? 1000 : 5000);
}

// ── HISTORY page ───────────────────────────────────────────────
let hist = null, histPinned = null, histToken = 0, daysLoaded = false;
let histChart = null;
async function loadDays() {
  const { days } = await getJSON('/api/days');
  const sel = $('hist-day'), keep = sel.value;
  sel.innerHTML = days.length ? days.map(d => `<option value="${d.day}">${d.day}${d.live ? ' · was live' : ''}</option>`).join('')
    : '<option value="">no past sessions yet</option>';
  if (keep && days.some(d => d.day === keep)) sel.value = keep;
  daysLoaded = true;
}
async function loadHist() {
  const day = $('hist-day').value; if (!day) return;
  const token = ++histToken; histPinned = null;
  $('hist-msg').hidden = false; $('hist-msg').className = 'banner';
  $('hist-msg').textContent = `Loading ${day}… the first time a day is opened it is recomputed, which can take up to a minute.`;
  try {
    const s = await getJSON('/api/state?day=' + day);
    if (token !== histToken) return;
    if (!s.session) throw new Error(s.status && s.status.error || 'not available');
    hist = s; $('hist-msg').hidden = true; $('hist-body').hidden = false;
    if (!histChart) histChart = makeChart($('hist-chart'), ts => { histPinned = ts; renderHistPick(true); });
    histChart.clear(); histChart.render(s);
    renderHistKpis(s);
    renderScore($('hist-score'), s, `Direction scorecard · ${s.session}`);
    renderRange($('hist-range'), s, `Range track record · ${s.session}`);
    renderHistPick(false);
  } catch (e) {
    if (token !== histToken) return;
    $('hist-msg').className = 'banner error'; $('hist-msg').textContent = `Could not load ${day}: ${e.message}. Try again in a minute.`;
  }
}
function renderHistKpis(s) {
  const r = s.record, sc = s.scorecard || {}, ours = sc.indicator, up = sc.always_up, rsi = sc.rsi_rule;
  const first = s.bars[0], last = s.bars[s.bars.length - 1], ch = last.close / first.open - 1;
  const lr = s.live_record;
  $('hist-kpis').innerHTML = `
    <div class="kpi"><span class="lab">AMD that day</span><span class="val ${ch >= 0 ? 'pos' : 'neg'}">${pct(ch)}</span><span class="sub">${money(first.open)} → ${money(last.close)}</span></div>
    <div class="kpi"><span class="lab">Inside 80% range</span><span class="val">${share(r.in_band, r.scored)}</span><span class="sub">${r.in_band} of ${r.scored} · target 80%</span></div>
    <div class="kpi"><span class="lab">Inside typical move</span><span class="val">${share(r.in_typical, r.scored)}</span><span class="sub">${r.in_typical} of ${r.scored} · target 50%</span></div>
    ${ours ? `<div class="kpi"><span class="lab">Direction calls right</span><span class="val">${ours.signals ? share(ours.correct, ours.signals) : '–'}</span><span class="sub">${ours.correct} of ${ours.signals} · always Up ${share(up.correct, up.signals)}</span></div>
    <div class="kpi"><span class="lab">Paper profit</span><span class="val ${ours.pnl_usd >= 0 ? 'pos' : 'neg'}">${usd(ours.pnl_usd)}</span><span class="sub">always Up ${usd(up.pnl_usd)} · RSI ${usd(rsi.pnl_usd)}</span></div>` : ''}
    ${lr ? `<div class="kpi"><span class="lab">Was live</span><span class="val">${etTime(lr.first)}–${etTime(lr.last)}</span><span class="sub">ET · ${lr.signals} signals on time</span></div>` : ''}`;
}
function renderHistPick(scroll) {
  const s = hist; if (!s) return;
  const sig = s.signals.length ? pickSig(s, histPinned) : null;
  renderPanel($('hist-panel'), s, sig, histPinned, () => { histPinned = null; renderHistPick(false); });
  renderHistTable(scroll);
}
function rowsFor(s) {
  const outs = new Map(s.outcomes.map(o => [o.ts, o]));
  return s.signals.map(g => { const o = outs.get(g.ts), d = dirOf(g);
    return { g, o, d, right: o && d ? Math.sign(o.ret_bps) === d : null }; });
}
function renderHistTable(scroll) {
  const s = hist, only = $('hist-only').checked;
  const rows = rowsFor(s).filter(r => !only || r.d);
  const sel = histPinned || (s.signals.length && s.signals[s.signals.length - 1].ts);
  $('hist-table').innerHTML = `<thead><tr><th>Time ET</th><th>Price</th><th>Typical ±</th><th>80% range</th><th class="l">Activity</th>
    <th class="l">Direction</th><th>Actual move</th><th>In range</th><th>Call</th></tr></thead><tbody>` +
    rows.map(({ g, o, d, right }) => {
      const dt = dirText(g);
      return `<tr class="click${g.ts === sel ? ' sel' : ''}" data-ts="${g.ts}"><td>${etTime(g.ts)}${g.late && !s.record.review ? ' <span class="muted">late</span>' : ''}</td>
      <td>${money(g.close)}</td><td>${(g.typical_bps / 100).toFixed(2)}%</td><td>${money(g.band_lo)} – ${money(g.band_hi)}</td>
      <td class="l"><span class="level ${g.level}">${g.level}</span></td><td class="l"><span class="dir ${dt.cls}">${d ? dt.text : '–'}</span></td>
      <td>${o ? `<span class="${o.ret_bps >= 0 ? 'pos' : 'neg'}">${(o.ret_bps / 100 >= 0 ? '+' : '') + (o.ret_bps / 100).toFixed(2)}%</span>` : '–'}</td>
      <td>${o ? (o.in_band ? 'yes' : '<b class="neg">no</b>') : '–'}</td>
      <td>${right == null ? '–' : right ? '<span class="pos">right</span>' : '<span class="neg">wrong</span>'}</td></tr>`; }).join('') + '</tbody>';
  $('hist-table').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => { histPinned = tr.dataset.ts; renderHistPick(false); });
  if (scroll) { const tr = $('hist-table').querySelector('tr.sel'); if (tr) tr.scrollIntoView({ block: 'nearest' }); }
}
function downloadCsv() {
  const s = hist; if (!s) return;
  const head = ['time_et', 'ts_utc', 'late', 'close', 'typical_pct', 'band_lo', 'band_hi', 'activity', 'level', 'p_up', 'p_flat', 'p_down',
    'direction', 'close_after', 'move_pct', 'in_typical', 'in_band', 'direction_right'];
  const lines = [head.join(',')].concat(rowsFor(s).map(({ g, o, d, right }) => [etTime(g.ts), g.ts, g.late, g.close, (g.typical_bps / 100).toFixed(4),
    g.band_lo.toFixed(2), g.band_hi.toFixed(2), g.activity.toFixed(3), g.level, g.p_up, g.p_flat, g.p_down, d > 0 ? 'up' : d < 0 ? 'down' : '',
    o ? o.close_after : '', o ? (o.ret_bps / 100).toFixed(4) : '', o ? o.in_typical : '', o ? o.in_band : '', right == null ? '' : right ? 1 : 0].join(',')));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  a.download = `amd-signals-${s.session}.csv`; a.click(); URL.revokeObjectURL(a.href);
}
$('hist-day').onchange = loadHist;
$('hist-only').onchange = () => renderHistTable(false);
$('hist-csv').onclick = downloadCsv;

// ── PERFORMANCE page ───────────────────────────────────────────
let runs = null, model = null, perf = null, expFilter = 'all';
const WEIGHT_NAMES = { lrv_30: 'last 30 minutes', lrv_sess: 'so far today', lrv_prev_day: 'yesterday', lrv_10: 'last 10 minutes', lrv_seas: 'usual for this time' };
const firstPct = s => { const m = /([\d.]+)%/.exec(s || ''); return m ? m[1] + '%' : '–'; };
const run = id => (runs || []).find(r => r.run_id === id) || {};
async function loadPerf() {
  try {
    [perf, runs, model] = await Promise.all([getJSON('/api/performance'), runs ? runs : getJSON('/api/experiments').then(x => x.runs),
      model ? model : getJSON('/api/model')]);
  } catch (e) { $('perf-note').textContent = 'Could not load the track record. Try again in a minute.'; return; }
  renderPerf(); renderClaims(); renderExperiments(); renderGuideWeights();
}
function renderPerf() {
  const t = perf.total;
  if (!t) { $('perf-kpis').innerHTML = ''; $('perf-days').innerHTML = ''; $('perf-note').textContent = 'No live sessions recorded yet.'; return; }
  const sc = t.scorecard, c = perf.costs;
  $('perf-kpis').innerHTML = `
    <div class="kpi"><span class="lab">Days live</span><span class="val">${perf.days.length}</span><span class="sub">${t.signals} on-time signals</span></div>
    <div class="kpi"><span class="lab">Inside 80% range</span><span class="val">${share(t.in_band, t.scored)}</span><span class="sub">${t.in_band} of ${t.scored} · target 80%</span></div>
    <div class="kpi"><span class="lab">Inside typical move</span><span class="val">${share(t.in_typical, t.scored)}</span><span class="sub">${t.in_typical} of ${t.scored} · target 50%</span></div>
    <div class="kpi"><span class="lab">Direction calls right</span><span class="val">${share(sc.indicator.correct, sc.indicator.signals)}</span><span class="sub">always Up ${share(sc.always_up.correct, sc.always_up.signals)} · RSI ${share(sc.rsi_rule.correct, sc.rsi_rule.signals)}</span></div>
    <div class="kpi"><span class="lab">Paper profit</span><span class="val ${sc.indicator.pnl_usd >= 0 ? 'pos' : 'neg'}">${usd(sc.indicator.pnl_usd)}</span><span class="sub">always Up ${usd(sc.always_up.pnl_usd)} · RSI ${usd(sc.rsi_rule.pnl_usd)}</span></div>`;
  const cell = r => r.signals ? `${r.correct}/${r.signals} (${Math.round(100 * r.accuracy)}%)` : '–';
  const money_ = r => r.trades ? `<span class="${r.pnl_usd >= 0 ? 'pos' : 'neg'}">${usd(r.pnl_usd)}</span>` : '–';
  const row = (d, cls, label) => { const s = d.scorecard;
    return `<tr class="${cls}"><td>${label}</td><td>${d.signals}</td><td>${d.in_band}/${d.scored} (${share(d.in_band, d.scored)})</td>
      <td>${d.in_typical}/${d.scored} (${share(d.in_typical, d.scored)})</td><td>${cell(s.indicator)}</td><td>${money_(s.indicator)}</td>
      <td>${cell(s.always_up)}</td><td>${money_(s.always_up)}</td><td>${cell(s.rsi_rule)}</td><td>${money_(s.rsi_rule)}</td></tr>`; };
  $('perf-days').innerHTML = `<thead><tr><th>Day</th><th>Signals</th><th>In 80% range</th><th>In typical</th><th>Ours right</th><th>Ours profit</th>
    <th>Always Up right</th><th>Always Up profit</th><th>RSI right</th><th>RSI profit</th></tr></thead><tbody>` +
    perf.days.map(d => row(d, '', `<a href="#history" data-day="${d.day}">${d.day}</a> <span class="muted small">${etTime(d.first)}–${etTime(d.last)} ET</span>`)).join('') +
    (perf.days.length > 1 ? row(t, 'total', 'All live days') : '') + '</tbody>';
  $('perf-days').querySelectorAll('a[data-day]').forEach(a => a.onclick = () => { pendingDay = a.dataset.day; });
  $('perf-note').textContent = `Only signals the app showed on time count (no late or recomputed ones). Scored on IEX closes 10 minutes later. Paper trades: ${c.shares} shares, held 10 minutes, one at a time, $${(2 * c.per_share_each_way).toFixed(2)}/share round trip. A handful of days is a small sample: one strong rally day can make “always Up” look brilliant.`;
}
function renderClaims() {
  const v0 = run('V0').result || '', v2 = run('V2').result || '';
  const num = (re, s) => { const m = re.exec(s); return m ? m.slice(1) : []; };
  const [r2f, r2c] = num(/R² ([\d.]+) \/ ([\d.]+)/, v2), [bf, bc] = num(/band ([\d.]+)% \/ ([\d.]+)%/, v2);
  const [q2] = num(/Fresh ([\d.]+)/, v2), [q0] = num(/Fresh ([\d.]+)/, v0);
  const t = perf && perf.total;
  const names = model.move.columns.map((c, i) => [WEIGHT_NAMES[c] || c, model.move.coef[i]]).sort((a, b) => b[1] - a[1]);
  const top = names[0][1];
  $('perf-move').innerHTML = `<h2>Move size · the main forecast</h2>
    <p class="claim"><b>How big the next 10-minute move will be is predictable</b>, and the ranges hold.</p>
    <div class="facts">
      <div class="fact"><div class="lab">80% range held (unseen 2023–24 / recent)</div><div class="val">${bf || '–'}% / ${bc || '–'}%</div></div>
      <div class="fact"><div class="lab">Share of move size explained (R²)</div><div class="val">${r2f || '–'} / ${r2c || '–'}</div></div>
      <div class="fact"><div class="lab">Error vs “same as last 10 min”</div><div class="val">${q2 && q0 ? `${Math.round(100 * (1 - q2 / q0))}% lower` : '–'}</div></div>
      <div class="fact"><div class="lab">80% range held, live</div><div class="val">${t ? share(t.in_band, t.scored) : '–'}</div></div>
    </div>
    <p class="note">Model: HAR (Corsi, 2009) with time of day, fitted on ${model.move.rows.toLocaleString()} minutes (${model.move.first_day} → ${model.move.last_day}). Learned weights on how jumpy AMD was:</p>
    <div class="weights">${names.map(([n, w]) => `<div class="wrow"><span>${n}</span><span class="wbar" style="width:${Math.max(3, 100 * w / top)}%"></span><span class="n">${w.toFixed(2)}</span></div>`).join('')}</div>
    <p class="note">Fancier models (ridge, LightGBM with news and peer stocks) did not beat it on unseen years, so by a rule set in advance the simple one is used.</p>`;
  const nDir = (runs || []).filter(r => /^E/.test(r.run_id)).length;
  const e56 = run('E5.6'), e0 = run('E5e.0'), e5d = run('E5d.3');
  $('perf-dir').innerHTML = `<h2>Direction · the weak hint</h2>
    <p class="claim"><b>Up or down in the next 10 minutes is not predictable</b> from public data, and the page says so.</p>
    <div class="facts">
      <div class="fact"><div class="lab">Direction models tested</div><div class="val">${nDir}</div></div>
      <div class="fact"><div class="lab">Passed the pre-set bar</div><div class="val">0</div></div>
      <div class="fact"><div class="lab">Used live (E5.6), unseen years</div><div class="val">${firstPct(e0.result)}</div></div>
      <div class="fact"><div class="lab">Always Up on the same rows</div><div class="val">${(/base ([\d.]+)%/.exec(e0.bar || '') || [, '–'])[1]}%</div></div>
    </div>
    <ul class="ticks small">
      <li><b>E5.6</b>, logistic regression on 32 inputs: ${esc(e56.result || '')}; it needed ${esc(e56.bar || '')}.</li>
      <li><b>E5d.3</b> seemed to pass (${firstPct(e5d.result)}), but the edge vanished when entering one minute later or on full-market prices: rejected.</li>
      <li>Research stopped by a rule written before the last round. The arrow is kept only as a clearly labelled, weak hint.</li>
    </ul>
    ${t ? `<p class="note"><b>Live so far</b> (on-time signals only):</p>
    <div class="tablewrap"><table class="data"><thead><tr><th></th><th>Signals</th><th>Right</th><th>Trades</th><th>Profit</th></tr></thead>
    <tbody>${scoreRows(t.scorecard, null)}</tbody></table></div>
    <p class="note">Too few days to judge: a single rally day lifts “always Up”, a falling day sinks it.</p>` : ''}`;
}
function renderExperiments() {
  const kind = r => /^[BE]/.test(r.run_id) ? 'dir' : /^V/.test(r.run_id) ? 'move' : 'live';
  const tag = o => { const t = o.replace(/[✅❌⚠️]\s*/gu, '').trim();
    const cls = /✅|Pass|Chosen/.test(o) ? 'pass' : /❌|Fail/.test(o) ? 'fail' : /⚠/.test(o) ? 'warn' : /Baseline/.test(o) ? 'base' : 'other';
    return `<span class="tag ${cls}">${esc(t)}</span>`; };
  const list = runs.filter(r => expFilter === 'all' || kind(r) === expFilter);
  $('exp-count').textContent = `${list.length} of ${runs.length} runs`;
  $('exp-table').innerHTML = `<thead><tr><th>Run</th><th class="l">Date</th><th class="l">What was tried</th><th class="l">Result</th><th class="l">Needed</th><th class="l">Outcome</th></tr></thead><tbody>` +
    list.map(r => `<tr><td><b>${esc(r.run_id)}</b></td><td class="l">${esc(r.date)}</td><td class="l what">${esc(r.tried)}<div class="muted small">${esc(r.split)} · ${esc(r.scoring)}</div></td>
      <td class="l res">${esc(r.result)}</td><td class="l">${esc(r.bar)}</td><td class="l">${tag(r.outcome)}</td></tr>`).join('') + '</tbody>';
}
$('exp-filter').onclick = e => { const b = e.target.closest('button'); if (!b) return;
  expFilter = b.dataset.f; $('exp-filter').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  if (runs) renderExperiments(); };

// ── GUIDE page ─────────────────────────────────────────────────
function localTimes(s) {
  if (!s || !s.session) return;
  const [y, m, d] = s.session.split('-').map(Number);
  document.querySelectorAll('.localtime').forEach(el => { const [hh, mm] = el.dataset.et.split(':').map(Number);
    el.textContent = localTime(Date.UTC(y, m - 1, d, hh, mm) - s.et_offset_seconds * 1000); });
}
async function renderGuideWeights() {
  try { model = model || await getJSON('/api/model'); } catch (e) { return; }
  const top = model.move.columns.map((c, i) => [WEIGHT_NAMES[c] || c, model.move.coef[i]]).sort((a, b) => b[1] - a[1]);
  $('guide-weights').textContent = 'Learned weights: ' + top.map(([n, w]) => `${n} ${w.toFixed(2)}`).join(', ') + '.';
}

// ── pages, theme, start ────────────────────────────────────────
let pendingDay = null;
const PAGES = ['live', 'history', 'performance', 'guide'];
async function route() {
  const page = PAGES.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'live';
  PAGES.forEach(p => { $('page-' + p).hidden = p !== page; });
  document.querySelectorAll('.nav a').forEach(a => a.classList.toggle('on', a.dataset.page === page));
  window.scrollTo(0, 0);
  if (page === 'live' && live) renderLive(live);
  if (page === 'history' && histChart) histChart.fit();
  if (page === 'history') {
    try { if (!daysLoaded || pendingDay) await loadDays(); } catch (e) { return; }
    if (pendingDay) { $('hist-day').value = pendingDay; pendingDay = null; loadHist(); }
    else if (!hist) loadHist();
  }
  if (page === 'performance') loadPerf();
  if (page === 'guide') renderGuideWeights();
}
window.addEventListener('hashchange', route);

function setTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  if (liveChart) liveChart.theme();
  if (histChart) histChart.theme();
}
$('theme').onclick = () => {
  const dark = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() === '#0f1114';
  const t = dark ? 'light' : 'dark'; store.set('theme', t); setTheme(t);
};
if (store.get('theme')) setTheme(store.get('theme'));
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => setTheme(store.get('theme')));

setInterval(renderClock, 20000);
setInterval(() => { if (!$('page-performance').hidden) loadPerf(); }, 60000);
route();
pollLive();
