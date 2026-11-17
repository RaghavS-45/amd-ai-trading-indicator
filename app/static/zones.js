// Zones page: support/resistance zones from earlier days, what the price did at each zone today,
// alerts and model-made levels. Uses the same /api/state as the Live page (helpers from app.js).
// Prices here are IEX closes: an idea starts at the close of the candle that reached the zone.

// ── zones drawn as shaded boxes behind the candles ──
class ZonesLayer {
  constructor() { this.zones = []; this.series = null; this.chart = null; this.req = null; this.range = null;
    const self = this; this.view = { zOrder: () => 'bottom', renderer: () => ({ draw: () => {}, drawBackground: t => self.draw(t) }) }; }
  attached(p) { this.chart = p.chart; this.series = p.series; this.req = p.requestUpdate; }
  detached() { this.req = null; }
  paneViews() { return [this.view]; }
  updateAllViews() {}
  autoscaleInfo() { return this.range ? { priceRange: this.range } : null; }   // keep the nearest zones in view
  set(zones, range) { this.zones = zones; this.range = range || null; if (this.req) this.req(); }
  draw(target) {
    if (!this.series || !this.zones.length) return;
    target.useBitmapCoordinateSpace(sc => {
      const ctx = sc.context, hr = sc.horizontalPixelRatio, vr = sc.verticalPixelRatio, W = sc.bitmapSize.width;
      ctx.font = `${Math.round(11 * vr)}px system-ui, sans-serif`; ctx.textBaseline = 'top'; ctx.textAlign = 'right';
      for (const z of this.zones) {
        const y1 = this.series.priceToCoordinate(z.hi), y2 = this.series.priceToCoordinate(z.lo);
        if (y1 == null || y2 == null) continue;
        let x0 = 0;
        if (z.from != null) { const x = this.chart.timeScale().timeToCoordinate(z.from); if (x == null) continue; x0 = x * hr; }
        const top = Math.min(y1, y2) * vr, h = Math.max(Math.abs(y2 - y1) * vr, 2 * vr), lw = Math.max(1, vr);
        ctx.fillStyle = z.fill; ctx.fillRect(x0, top, W - x0, h);
        ctx.fillStyle = z.edge; ctx.fillRect(x0, top, W - x0, lw); ctx.fillRect(x0, top + h - lw, W - x0, lw);
        ctx.fillText(z.text, W - 8 * hr, top + 2 * vr);
      }
    });
  }
}

// ── zones known at a time ──
function znOpeningRange(s) {   // high/low of the first 30 minutes, known from 10:00
  const end = secs(s.open) + 1800, first = s.bars.filter(b => secs(b.time) < end && !b.filled);
  if (!first.length || !s.bars.some(b => secs(b.time) >= end)) return null;
  return { lo: Math.min(...first.map(b => b.low)), hi: Math.max(...first.map(b => b.high)), from: end };
}
function znZones(s, upto, price) {   // upto: seconds (UTC) or null for the whole day; price: sets support/resistance
  const z = (s.zones || []).map(x => ({ ...x, kind: price == null ? x.kind : (x.lo + x.hi) / 2 > price ? 'resistance' : 'support' }));
  const or = znOpeningRange(s);
  if (or && (upto == null || upto >= or.from)) z.push({ lo: or.lo, hi: or.hi, kind: 'range', label: 'Opening range', strength: 0, from: or.from });
  return z;
}
const znName = z => z.kind === 'range' ? 'opening range' : z.kind;

// ── chance of touching a price within 10 minutes, measured on the training days ──
// (Nov 2024 – Jun 2026, walk-forward forecasts: how often the 10-minute high/low reached a level k forecast
// standard deviations away; sd = this minute's 80% range / 1.2816)
const REACH_K = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0];
const REACH_P = [0.9226, 0.777, 0.6333, 0.4998, 0.3837, 0.2889, 0.2124, 0.1531, 0.1086, 0.0761, 0.0532, 0.037, 0.0258,
                 0.018, 0.0128, 0.0093, 0.0068, 0.0052, 0.0039, 0.0031, 0.0025];
function reach(sig, level) {
  const sd = (sig.band_bps / 1e4) / 1.2816;
  if (!(sd > 0)) return 0;
  const k = Math.abs(Math.log(level / sig.close)) / sd;
  if (k >= 4) return REACH_P[REACH_P.length - 1];
  const i = Math.floor(k / 0.2), f = (k - REACH_K[i]) / 0.2;
  return REACH_P[i] + f * (REACH_P[i + 1] - REACH_P[i]);
}
const pc = x => `${Math.round(100 * x)}%`;

// ── what the price did at each zone today ──
function znTouches(s) {
  const bars = s.bars, out = [], last = {};
  const zs = znZones(s, null, null);
  for (let i = 1; i < bars.length; i++) {
    const b = bars[i], prev = bars[i - 1], t = secs(b.time);
    if (b.filled) continue;
    zs.forEach((z, k) => {
      if (z.from != null && t < z.from) return;
      if (i - (last[k] ?? -999) < 30) return;
      const fromAbove = prev.close > z.hi && b.low <= z.hi, fromBelow = prev.close < z.lo && b.high >= z.lo;
      if (!fromAbove && !fromBelow) return;
      last[k] = i;
      const side = fromAbove ? 1 : -1, ev = { i, time: b.time, zone: z, side, entry: b.close, done: false };
      if (i + 10 < bars.length) {
        const next = bars.slice(i + 1, i + 11);
        ev.broke = side === 1 ? next.some(x => x.close < z.lo) : next.some(x => x.close > z.hi);
        ev.move = side * (bars[i + 10].close - b.close);
        ev.done = true;
      }
      out.push(ev);
    });
  }
  return out;
}

// ── alerts: shown only when the model is sure. Curves measured on 2023–2024 (walk-forward forecasts);
// accuracy checked on 372 unseen days in 2025–2026 (numbers in TESTED). IEX closes, highs and lows.
const EXC_K = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5];
const EXC_P = [1, 0.9992, 0.9497, 0.7755, 0.5622, 0.3817, 0.2475, 0.1565, 0.0979, 0.0613, 0.0386, 0.0243, 0.0157,
               0.0105, 0.0072, 0.0051, 0.0038, 0.0027, 0.0022, 0.0017, 0.0013];   // P(biggest move either way ≥ k sd)
const HOLD_W = [0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875, 2.125, 2.375, 2.875];
const HOLD_P = [0.304, 0.518, 0.673, 0.748, 0.894, 0.900, 0.929, 0.955, 0.962, 0.97];  // P(zone holds 10 min) by zone width in sd
const BIG = 0.0075, CALM = 0.003, P_BIG = 0.5, P_CALM = 0.9, P_HOLD = 0.95;
const TESTED = { big: { hit: 66, normal: 9 }, calm: { hit: 91, normal: 48 } };
const interp = (x, xs, ys) => { if (x <= xs[0]) return ys[0]; for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (x - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1]); return ys[ys.length - 1]; };
const sdOf = sig => (sig.band_bps / 1e4) / 1.2816;

function znAlerts(s) {   // every alert of the day, with its outcome once 10 minutes have passed
  const at = new Map(s.bars.map((b, i) => [b.time, i])), bars = s.bars, out = [];
  const outcome = (i, test) => (i + 10 < bars.length ? test(bars.slice(i + 1, i + 11), bars[i]) : null);
  for (const g of s.signals) {
    const i = at.get(g.ts); if (i == null) continue;
    const sd = sdOf(g); if (!(sd > 0)) continue;
    const pBig = interp(BIG / sd, EXC_K, EXC_P), pCalm = 1 - interp(CALM / sd, EXC_K, EXC_P);
    const exc = (nx, b) => Math.max(...nx.map(x => Math.max(Math.abs(Math.log(x.high / b.close)), Math.abs(Math.log(x.low / b.close)))));
    if (pBig >= P_BIG) out.push({ type: 'big', ts: g.ts, i, p: pBig, price: g.close, ok: outcome(i, (nx, b) => exc(nx, b) >= BIG) });
    if (pCalm >= P_CALM) out.push({ type: 'calm', ts: g.ts, i, p: pCalm, price: g.close, lo: g.close * Math.exp(-CALM), hi: g.close * Math.exp(CALM),
                                    ok: outcome(i, (nx, b) => exc(nx, b) < CALM) });
  }
  return out;
}

function znScenarios(s, sig) {   // chance of reaching the nearest resistance / support in 10 minutes
  const p = sig.close, zs = znZones(s, secs(sig.ts), p).filter(x => !(p >= x.lo && p <= x.hi));
  const up = zs.filter(x => x.lo > p).sort((a, b) => a.lo - b.lo)[0], dn = zs.filter(x => x.hi < p).sort((a, b) => b.hi - a.hi)[0];
  const pu = up ? reach(sig, up.lo) : 0, pd = dn ? reach(sig, dn.hi) : 0;
  return { up, dn, pu, pd, neither: Math.max(0, 1 - pu - pd) };
}

// ── page ──
let znChart = null, znCandles = null, znLayer = null, znPinned = null, znState = null, znAuto = true, znSession = null;
function znMakeChart() {
  const el = $('zn-chart');
  znChart = LightweightCharts.createChart(el, { autoSize: true, timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 4 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal } });
  znCandles = znChart.addCandlestickSeries({ borderVisible: false });
  znLayer = new ZonesLayer(); znCandles.attachPrimitive(znLayer);
  ['mousedown', 'touchstart'].forEach(e => el.addEventListener(e, () => { znAuto = false; }, { passive: true }));
  znChart.subscribeClick(p => {
    if (!p.time || !znState) return;
    const off = znState.et_offset_seconds, g = znState.signals.find(x => secs(x.ts) + off === p.time);
    if (g) { znPinned = g.ts; znAuto = true; window.renderZonesPage(znState); }
  });
  window.zonesTheme();
}
window.zonesTheme = function () {
  if (!znChart) return;
  znChart.applyOptions({ layout: { background: { color: 'transparent' }, textColor: css('--ink-2'), fontSize: 11 },
    grid: { vertLines: { color: css('--line') }, horzLines: { color: css('--line') } },
    rightPriceScale: { borderColor: css('--line') }, timeScale: { borderColor: css('--line') } });
  znCandles.applyOptions({ upColor: css('--up'), downColor: css('--down'), wickUpColor: css('--up'), wickDownColor: css('--down') });
  if (znState) window.renderZonesPage(znState);
};

function znDrawChart(s, sig, alerts) {
  const off = s.et_offset_seconds, t = iso => secs(iso) + off;
  if (s.session !== znSession) { znSession = s.session; znAuto = true; }
  znCandles.setData(s.bars.map(b => ({ time: t(b.time), open: b.open, high: b.high, low: b.low, close: b.close,
    ...(b.filled ? { color: css('--ink-2'), wickColor: css('--ink-2') } : {}) })));
  const price = sig ? sig.close : null, zs = znZones(s, sig ? secs(sig.ts) : null, price), real = s.bars.filter(b => !b.filled);
  let range = null;
  if (zs.length && real.length) {
    const hi = Math.max(...real.map(b => b.high)), lo = Math.min(...real.map(b => b.low));
    const up = zs.filter(z => z.lo > hi).sort((a, b) => a.lo - b.lo)[0], dn = zs.filter(z => z.hi < lo).sort((a, b) => b.hi - a.hi)[0];
    range = { minValue: dn ? dn.lo : lo, maxValue: up ? up.hi : hi };
  }
  const colour = { resistance: [css('--down-soft'), css('--down')], support: [css('--up-soft'), css('--up')], range: ['rgba(0,0,0,0)', css('--news')] };
  const fixed = zs.map(z => ({ lo: z.lo, hi: z.hi, from: z.from == null ? null : z.from + off, fill: colour[z.kind][0], edge: colour[z.kind][1],
    text: z.kind === 'range' ? 'Opening range (09:30–10:00)' : `${z.kind === 'support' ? 'Support' : 'Resistance'} · ${z.label}` }));
  const ex = sig ? znDayExtremes(s, sig) : null, ml = [];
  if (ex) {
    if (ex.high.pNew >= 0.05) ml.push({ lo: ex.high.lo, hi: ex.high.hi, from: secs(sig.ts) + off, fill: 'rgba(224,122,40,0.16)', edge: css('--vwap'),
      text: `Model: likely day high if a new high comes · ${pc(ex.high.pNew)} chance` });
    if (ex.low.pNew >= 0.05) ml.push({ lo: ex.low.lo, hi: ex.low.hi, from: secs(sig.ts) + off, fill: 'rgba(40,150,200,0.16)', edge: 'rgb(60,160,210)',
      text: `Model: likely day low if a new low comes · ${pc(ex.low.pNew)} chance` });
    if (range) { for (const z of ml) { range.minValue = Math.min(range.minValue, z.lo); range.maxValue = Math.max(range.maxValue, z.hi); } }
  }
  znLayer.set((store.get('zn_fixed') === 'off' ? [] : fixed).concat(ml), range);
  // one marker per run of the same alert (consecutive minutes), coloured by whether it came true
  const runs = []; let prev = {};
  for (const a of alerts) { if (prev[a.type] !== a.i - 1) runs.push(a); prev[a.type] = a.i; }
  const col = { big: css('--vwap'), calm: css('--up') };
  znCandles.setMarkers(runs.map(a => ({ time: t(a.ts), position: a.type === 'calm' ? 'belowBar' : 'aboveBar', shape: 'circle',
    text: a.type, color: a.ok === false ? css('--ink-2') : col[a.type] })).sort((x, y) => x.time - y.time));
  if (znAuto) { znAuto = false; setTimeout(() => znChart.timeScale().fitContent(), 80); }
}

function znAlertCards(s, sig, alerts) {
  const now = alerts.filter(a => a.ts === sig.ts);
  const until = iso => etTime(new Date(secs(iso) * 1000 + 600000).toISOString());
  const card = (cls, title, body, tested) => `<div class="alertcard ${cls}"><div><b>${title}</b><br>${body}</div><div class="muted small">${tested}</div></div>`;
  const cards = [];
  for (const a of now) {
    if (a.type === 'big') cards.push(card('big', '⚠ Big move likely in the next 10 minutes',
      `AMD is likely to move at least ${(100 * BIG).toFixed(2)}% (about ${money(a.price * BIG)}) up <i>or</i> down before ${until(a.ts)} ET. Model: ${pc(a.p)}. Widen stops, trade smaller, or wait.`,
      `In testing this came true ${TESTED.big.hit}% of the time (normally ${TESTED.big.normal}%).`));
    if (a.type === 'calm') cards.push(card('calm', '✓ Calm: AMD likely to stay in a small range',
      `Very likely to stay between <b>${money(a.lo)}</b> and <b>${money(a.hi)}</b> until ${until(a.ts)} ET. Model: ${pc(a.p)}. A breakout is unlikely right now.`,
      `In testing this came true ${TESTED.calm.hit}% of the time (normally ${TESTED.calm.normal}%).`));
  }
  $('zn-alerts').innerHTML = cards.length ? `<div class="alertrow">${cards.join('')}</div>`
    : `<div class="alertrow"><div class="alertcard none"><div><b>No alert right now</b><br>The model is not sure about anything unusual, so it stays quiet. A normal 10-minute move from here is about ±${money(sig.close * sig.typical_bps / 1e4)}.</div></div></div>`;
}

function znPanel(s, sig, alerts) {
  const root = $('zn-panel');
  if (!sig) { root.innerHTML = '<div class="card muted">No forecasts yet for this session (they start at 10:03 ET).</div>'; return; }
  const sc = znScenarios(s, sig), bar = (w, c) => `<span style="display:inline-block;height:10px;width:${Math.round(100 * w)}%;background:${c};border-radius:3px"></span>`;
  const row = (label, p, c) => `<div class="row"><span>${label}</span><span class="v">${pc(p)}</span></div><div style="margin:-4px 0 8px">${bar(p, c)}</div>`;
  const typ = sig.close * sig.typical_bps / 1e4, close = lvl => Math.abs(lvl - sig.close) <= 0.25 * typ;   // within a quarter of a normal move
  const atLevel = (kind, lvl) => `<div class="row"><span><b>Price is right at ${kind}</b> ${money(lvl)} <span class="muted small">${usd(lvl - sig.close)} away</span></span><span class="v muted small">almost certain to be touched</span></div>
      <p class="note" style="margin-top:-4px">It is closer than a quarter of a normal move, so whether it gets touched is not the question: watch whether it holds.</p>`;
  const done = t => alerts.filter(a => a.type === t && a.ok !== null && secs(a.ts) <= secs(sig.ts) - 600);
  const rec = t => { const d = done(t), ok = d.filter(a => a.ok).length; return d.length ? `${ok} of ${d.length} came true` : 'none yet'; };
  const ex = znDayExtremes(s, sig);
  const exCard = ex ? `
    <div class="card">
      <h2>Model levels for the rest of today</h2>
      <div class="row"><span>High so far</span><span class="v">${money(ex.hs)}</span></div>
      <div class="row"><span><b style="color:var(--vwap)">Chance of a new high today</b></span><span class="v big">${pc(ex.high.pNew)}</span></div>
      <div class="row"><span>If it comes, most likely around</span><span class="v">${money(ex.high.mid)} <span class="muted small">(50%: ${money(ex.high.lo)}–${money(ex.high.hi)})</span></span></div>
      <div class="row"><span>Low so far</span><span class="v">${money(ex.ls)}</span></div>
      <div class="row"><span><b style="color:rgb(60,160,210)">Chance of a new low today</b></span><span class="v big">${pc(ex.low.pNew)}</span></div>
      <div class="row"><span>If it comes, most likely around</span><span class="v">${money(ex.low.mid)} <span class="muted small">(50%: ${money(ex.low.lo)}–${money(ex.low.hi)})</span></span></div>
      <div class="row"><span>Show fixed levels too</span><span class="v"><button type="button" class="btn${store.get('zn_fixed') === 'off' ? '' : ' on'}" data-fixed>${store.get('zn_fixed') === 'off' ? 'Off' : 'On'}</button></span></div>
      <p class="note">Built from the current price, today’s high and low so far, how jumpy AMD is right now and the time left. Tested on 372 unseen days: the chance of a new high/low matched reality (said 37% → happened 40%; 64% → 65%), and when a new high came it landed in the 50% band 56% of the time. These levels missed the real day high by 0.5% on average, against 1.9% for yesterday’s high.</p>
    </div>` : '';
  root.innerHTML = exCard + `
    <div class="card">
      <h2>What could happen in the next 10 minutes</h2>
      ${sc.up ? (close(sc.up.lo) ? atLevel('resistance', sc.up.lo) : row(`Reaches resistance ${money(sc.up.lo)} <span class="muted small">${usd(sc.up.lo - sig.close)}</span>`, sc.pu, css('--down'))) : ''}
      ${sc.dn ? (close(sc.dn.hi) ? atLevel('support', sc.dn.hi) : row(`Reaches support ${money(sc.dn.hi)} <span class="muted small">${usd(sc.dn.hi - sig.close)}</span>`, sc.pd, css('--up'))) : ''}
      ${row('Touches neither (stays between them)', sc.neither, css('--ink-2'))}
      <p class="note">Both sides are shown on purpose: the model does not know the direction. These chances matched reality within 2 points on unseen days.</p>
    </div>
    <div class="card">
      <h2>How reliable are the alerts?</h2>
      <div class="row"><span>⚠ Big move</span><span class="v">tested ${TESTED.big.hit}% (normally ${TESTED.big.normal}%) · today ${rec('big')}</span></div>
      <div class="row"><span>✓ Calm range</span><span class="v">tested ${TESTED.calm.hit}% (normally ${TESTED.calm.normal}%) · today ${rec('calm')}</span></div>
      <p class="note">“Tested”: rules set on 2023–2024, checked on 372 unseen days in 2025–2026. “Today”: each alert minute, checked 10 minutes later.</p>
    </div>
`;
  const fx = root.querySelector('[data-fixed]'); if (fx) fx.onclick = () => { store.set('zn_fixed', store.get('zn_fixed') === 'off' ? 'on' : 'off'); window.renderZonesPage(znState); };
}

// time controls: step through the session minute by minute (signals only exist from 10:03)
function znIndex() { const n = znState.signals.length; return znPinned ? Math.max(0, znState.signals.findIndex(g => g.ts === znPinned)) : n - 1; }
function znGo(i) {
  const n = znState.signals.length; if (!n) return;
  znPinned = i >= n - 1 ? null : znState.signals[Math.max(0, i)].ts; znAuto = true; window.renderZonesPage(znState);
}
document.querySelectorAll('#zn-time [data-step]').forEach(b => b.onclick = () => {
  if (!znState) return;
  if (b.dataset.step === 'all') { znPinned = null; znAuto = true; window.renderZonesPage(znState); return; }
  znGo(znIndex() + Number(b.dataset.step));
});
$('zn-slider').oninput = e => { if (znState) znGo(Number(e.target.value)); };

// ── reach before the close (used by the model's day high / low levels) ──
const RD_K = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5,
              2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4];
const RD_P = [0.9828, 0.9169, 0.8478, 0.782, 0.7156, 0.651, 0.5955, 0.5459, 0.5003, 0.458, 0.4132, 0.379, 0.3421, 0.3098, 0.2845,
              0.2607, 0.2378, 0.2171, 0.1998, 0.1841, 0.1709, 0.1579, 0.1468, 0.1358, 0.1274, 0.1177, 0.1108, 0.1029, 0.0964, 0.0899,
              0.0843, 0.0793, 0.0746, 0.0713, 0.0679, 0.0651, 0.0613, 0.0582, 0.0558, 0.0525, 0.0493];   // reach before the close, measured on 2023–2024
// ── model-made levels: where today's high and low will probably end up, from the current situation ──
// chance of a new high / low later today by distance (in rest-of-day sd), learned on 2023–2024; checked on 2025–2026
const NK = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3];
const NEW_HI = [0.8974, 0.8203, 0.7652, 0.6898, 0.6293, 0.5265, 0.4585, 0.3744, 0.3317, 0.2694, 0.2202, 0.2202, 0.1765, 0.1373, 0.1138, 0.0833,
                0.0517, 0.0353, 0.0353, 0.0353, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const NEW_LO = [0.9257, 0.8854, 0.7798, 0.6321, 0.5485, 0.5256, 0.4726, 0.3843, 0.3081, 0.2818, 0.2456, 0.2147, 0.2027, 0.2027, 0.0992, 0.0992,
                0.0992, 0.0992, 0.0614, 0.0614, 0.0614, 0.0614, 0.0345, 0.0345, 0, 0, 0, 0, 0, 0, 0];
function znDayExtremes(s, sig) {
  const real = s.bars.filter(b => !b.filled && secs(b.time) <= secs(sig.ts)); if (!real.length) return null;
  const p = sig.close, hs = Math.max(...real.map(b => b.high)), ls = Math.min(...real.map(b => b.low));
  const left = Math.max(10, (secs(s.close) - secs(sig.ts)) / 60), sd = (sig.band_bps / 1e4) / 1.2816 * Math.sqrt(left / 10);
  const kAt = q => interp(-q, RD_P.map(x => -x), RD_K);   // distance k (sd) with P(reach ≥ k) = q
  const side = (need, table, sgn) => {
    const pNew = interp(need, NK, table), pr = interp(need, RD_K, RD_P);
    const a = kAt(pr * 0.75), b = kAt(pr * 0.25), m = kAt(pr * 0.5);   // where the new extreme lands, given there is one
    const px = k => p * Math.exp(sgn * k * sd);
    return { pNew, lo: Math.min(px(a), px(b)), hi: Math.max(px(a), px(b)), mid: px(m) };
  };
  return { hs, ls, high: side(Math.log(hs / p) / sd, NEW_HI, 1), low: side(Math.log(p / ls) / sd, NEW_LO, -1) };
}

window.renderZonesPage = function (s) {
  if (!s || !s.session) return;
  znState = s;
  if (!znChart) znMakeChart();
  const sig = s.signals.length ? ((znPinned && s.signals.find(g => g.ts === znPinned)) || s.signals[s.signals.length - 1]) : null;
  const v = sig && znPinned ? { ...s, bars: s.bars.filter(b => secs(b.time) <= secs(sig.ts)), signals: s.signals.filter(g => secs(g.ts) <= secs(sig.ts)) } : s;
  const alerts = znAlerts(v);
  znDrawChart(v, sig, alerts);
  if (sig) znAlertCards(v, sig, alerts); else $('zn-alerts').innerHTML = '';
  znPanel(v, sig, alerts);
  const last = v.bars[v.bars.length - 1];
  if (last) { $('zn-price').textContent = money(last.close); $('zn-asof').textContent = `${s.session} · bar ${etTime(last.time)} ET`; }
  const sl = $('zn-slider'); sl.max = Math.max(0, s.signals.length - 1); sl.value = znIndex();
  $('zn-when').textContent = sig ? (znPinned ? `showing ${etTime(sig.ts)} ET` : 'showing now') : '';
};
if (!$('page-zones').hidden && typeof live !== 'undefined' && live) window.renderZonesPage(live);
