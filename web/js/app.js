// Embodied Fly Lab replay viewer: run picker, synchronised body + brain views, pipeline panels.
import { FlyView } from './flyview.js';
import { BrainView, CLASS_COLORS, ROLE_COLORS } from './brainview.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(+x) ? '–' : (+x).toFixed(d));
const label = (s) => String(s ?? '–').replace(/_/g, ' ');

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = { t: 0, playing: !reducedMotion, speed: 0.25, duration: 1, run: null, index: null, manifest: null, lastTs: null, scrubbing: false };
let fly, brain;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function verdictPill(v) {
  if (!v) return '<span class="pill">not verified</span>';
  return `<span class="pill ${esc(v)}">verifier: ${esc(v)}</span>`;
}

function renderRunChips() {
  const nav = $('#runs');
  nav.innerHTML = '';
  for (const r of state.index.runs) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'run-chip';
    b.dataset.id = r.id;
    b.setAttribute('aria-pressed', 'false');
    b.innerHTML = `<span>${esc(r.title)}</span><span class="meta">
      <span class="pill ${r.mode === 'flight' ? 'flight' : ''}">${esc(r.mode)}</span>
      ${r.kind === 'agent_hypothesis' ? '<span class="pill agent">agent hypothesis</span>' : ''}
      ${r.kind === 'surprise' ? '<span class="pill agent">surprise</span>' : ''}
      ${r.kind === 'control' ? '<span class="pill">control</span>' : ''}
      ${r.kind === 'prediction' ? '<span class="pill">model prediction</span>' : ''}
      ${verdictPill(r.verdict)}</span>`;
    b.addEventListener('click', () => selectRun(r.id));
    nav.appendChild(b);
  }
}

function renderLegend() {
  const items = Object.entries(CLASS_COLORS).filter(([c]) => (brain.meta.class_counts[c] || 0) > 0)
    .map(([c, col]) => `<span><i style="background:${col}"></i>${esc(label(c))}</span>`).join('');
  $('#brain-legend').innerHTML = `${items}<span class="sep"></span>
    <span><i style="background:${ROLE_COLORS.stimulated}"></i>stimulated</span>
    <span><i style="background:${ROLE_COLORS.readout}"></i>bridge read-out neuron (active)</span>
    <span><i style="background:${ROLE_COLORS.descending}"></i>descending neuron (active)</span>
    <span><i style="background:#ffd36b"></i>other active, brighter = higher mean rate</span>
    <span><i style="background:${ROLE_COLORS.silenced}"></i>silenced</span>`;
}

function bar(v, max = 1, center = false) {
  if (center) {
    const x = Math.max(-1, Math.min(1, v / max));
    const left = x < 0 ? 50 + 50 * x : 50, w = Math.abs(50 * x);
    return `<div class="bar center"><span style="left:${left}%;width:${w}%"></span></div>`;
  }
  return `<div class="bar"><span style="left:0;width:${Math.max(0, Math.min(100, (100 * v) / max))}%"></span></div>`;
}

function card(step, title, body, cls = '') {
  return `<article class="panel card ${cls}"><h3><span class="step">${step}</span>${esc(title)}</h3>${body}</article>`;
}

function gtPhrase(gt) {
  const eff = { induce: 'induces', reduce: 'reduces / abolishes', increase: 'increases' }[gt.effect];
  return `${label(gt.manipulation)} ${gt.target_group} → ${eff ? eff + ' ' : ''}${label(gt.expected_behavior)}`;
}

function runQuestion(run) {
  const s = run.stimulus;
  const drive = s.excite.length ? `driving ${s.excite.join(' + ')}` : 'a baseline without any stimulus';
  const sil = s.silence.length ? ` while silencing ${s.silence.join(', ')}` : '';
  return `Does ${drive}${sil} make the simulated fly show "${label(run.expected_behavior)}"?`;
}

function shortCite(c) {
  if (!c || !c.authors) return '';
  const first = String(c.authors).split(',')[0].trim();
  return `${first} et al. ${c.year}`;
}

function renderSide(run) {
  const a = run.agent, s = run.stimulus, b = run.brain, br = run.bridge, m = run.body.metrics || {};
  const ver = run.verifier, gt = run.ground_truth, c = gt?.citation;
  const kindPill = { agent_hypothesis: '<span class="pill agent">agent hypothesis</span>', surprise: '<span class="pill agent">surprising result</span>',
    control: '<span class="pill">control</span>', validation: '<span class="pill">validation</span>',
    prediction: '<span class="pill">model prediction</span>' }[run.kind] || '';
  const rows = [];

  rows.push(['Question', `<p style="margin:0">${esc(a?.question || runQuestion(run))}</p>`]);
  if (a && a.hypotheses && a.hypotheses.length) {
    rows.push(['Hypothesis <span class="pill agent">agent-generated</span>', `<p style="margin:0">${esc(a.hypotheses[0].text)}</p>
      <p class="muted" style="margin:.2rem 0 0">Written by agent <code>${esc(a.hypotheses[0].agent)}</code>; unconfirmed in the wet lab. Expected here: ${esc(label(run.expected_behavior))}.</p>`]);
  } else if (gt) {
    rows.push(['Published expectation', `<p style="margin:0"><b>${esc(gtPhrase(gt))}</b>
      ${c ? `<span class="muted">(${esc(shortCite(c))})</span>` : ''}</p>`]);
  }
  rows.push(['Stimulus (in silico)', `<p style="margin:0">${esc(s.excite.join(' + ') || 'none')}${s.silence.length ? `, silenced: <b>${esc(s.silence.join(', '))}</b>` : ''}
    <span class="muted">· ${s.n_excited} neurons at ${fmt(s.excite_rate_hz, 0)} Hz</span></p>`]);

  const gr = Object.entries(b.group_rates_hz || {}).filter(([, v]) => v > 0).map(([k, v]) => `${esc(k)} ${fmt(v, 0)} Hz`).join(', ');
  const bv = br.values || {};
  const bridgeTxt = Object.entries(bv).map(([k, x]) => `${esc(k)} ${fmt(x)}`).join(' · ');
  rows.push(['Brain → bridge output', `<p style="margin:0">${b.n_active.toLocaleString('en-US')} of 138,639 neurons active${gr ? `; read-outs: ${gr}` : '; no bridge read-out neuron active'}.</p>
    <p style="margin:.2rem 0 0"><b>${esc(label(br.type))}:</b> ${bridgeTxt}</p>
    ${br.note ? `<p class="muted" style="margin:.2rem 0 0">${esc(br.note)}</p>` : ''}
    ${(br.explain || {}).connectome_decides ? `<p class="muted" style="margin:.2rem 0 0">Connectome decides: ${esc(br.explain.connectome_decides)}. Hand-designed: ${esc(br.explain.hand_designed)}.</p>` : ''}`]);

  rows.push(['Behaviour', `<p style="margin:0">Body: <b>${esc(label(m.behavior))}</b> <span class="muted">(expected ${esc(label(run.expected_behavior))})</span></p>`]);

  const vv = ver && ver.available ? ver.final_verdict : null;
  const verdictBox = vv
    ? `<div class="verdict ${esc(vv)}"><b>Movement verifier: ${esc(vv)}</b><br><span class="muted">${esc(ver.reason || ver.kinematic?.reason || '')}</span></div>`
    : `<div class="verdict"><b>Movement verifier:</b> not available for this export</div>`;
  const gtBox = gt && c
    ? `<p style="margin:.5rem 0 0">Published result: ${esc(c.authors?.split(',')[0] || '')} et al. ${esc(c.year)}, <a href="https://doi.org/${esc(c.doi)}" target="_blank" rel="noopener">doi:${esc(c.doi)}</a>.
       <span class="muted">Compared in silico, not a replication.</span></p>${gt.note ? `<p class="muted" style="margin:.3rem 0 0">${esc(gt.note)}</p>` : ''}`
    : '';

  $('#side').innerHTML = `<h2>${esc(run.title)}</h2>
    <div class="pills"><span class="pill ${run.mode === 'flight' ? 'flight' : ''}">${esc(run.mode)}</span>${kindPill}</div>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
    ${verdictBox}${gtBox}`;
}

function renderPipeline(run) {
  const out = [];
  const a = run.agent;
  if (a) {
    const hyps = (a.hypotheses || []).map((h) => `<div class="agent-box"><b>${esc(h.agent)}:</b> ${esc(h.text)}</div>`).join('');
    const an = (a.analyses || []).slice(0, 2).map((h) => `<p class="quote">${esc(h.agent)}: ${esc(h.text)}</p>`).join('');
    const dec = a.decision ? `<p><b>Next decision (${esc(a.decision.agent)}):</b> ${esc(a.decision.text)}</p>` : '';
    out.push(card('Q', 'Agent lab context (live Omnigent run)', `
      <p><b>Question:</b> ${esc(a.question)}</p>
      <p><span class="pill agent">agent-generated</span> ${esc(a.label)}</p>${hyps}${an}${dec}
      <p class="muted">Source: <code>${esc(a.source)}</code></p>`, 'wide'));
  }
  const s = run.stimulus;
  out.push(card(1, 'Stimulus (in silico)', `
    <p class="big">${esc(s.excite.join(' + ') || 'none')}</p>
    <p>${s.n_excited} neurons driven at ${fmt(s.excite_rate_hz, 0)} Hz (Poisson input)${s.silence.length ? `; silenced: <b>${esc(s.silence.join(', '))}</b> (${s.n_silenced})` : ''}.</p>
    <p class="muted">Expected from literature: <b>${esc(label(run.expected_behavior))}</b></p>`));

  const b = run.brain;
  const gr = Object.entries(b.group_rates_hz || {}).filter(([, v]) => v > 0);
  const grRows = gr.length ? gr.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${bar(v, 200)}</td><td class="num">${fmt(v, 0)} Hz</td></tr>`).join('')
    : '<tr><td colspan="3" class="muted">no bridge read-out neuron active</td></tr>';
  const ro = Object.entries(b.readouts || {}).filter(([, v]) => v && typeof v === 'object' && 'active' in v)
    .map(([k, v]) => `<span class="pill ${v.active ? 'correct' : ''}">${esc(k)} (${esc(v.group)}): ${fmt(v.rate_hz, 0)} Hz</span>`).join(' ');
  const top = (b.top || []).filter((x) => x.role !== 'stimulated').slice(0, 6)
    .map((x) => `<tr><td>${esc(x.cell_type || '?')}</td><td class="muted">${esc(label(x.super_class))} ${esc(x.side || '')}</td><td class="num">${fmt(x.rate_hz, 0)} Hz</td></tr>`).join('');
  out.push(card(2, 'Brain: whole-brain LIF model', `
    <p><span class="big">${b.n_active.toLocaleString('en-US')}</span> neurons active (mean rate &gt; 0) of 138,639</p>
    <table><tbody>${grRows}</tbody></table>
    <p style="margin-top:.5rem">${ro}</p>
    ${top ? `<p class="muted" style="margin:.5rem 0 .2rem">Strongest downstream neurons</p><table><tbody>${top}</tbody></table>` : ''}
    <p class="muted">${esc(b.note)} ${b.duration_ms} ms, ${b.n_trials} trials, seed ${b.seed}.</p>`));

  const br = run.bridge, v = br.values || {};
  const rows = br.type === 'walk_drive'
    ? `<tr><td>forward</td><td>${bar(v.forward)}</td><td class="num">${fmt(v.forward)}</td></tr>
       <tr><td>backward</td><td>${bar(v.backward)}</td><td class="num">${fmt(v.backward)}</td></tr>
       <tr><td>turn (− left / + right)</td><td>${bar(v.turn, 1, true)}</td><td class="num">${fmt(v.turn)}</td></tr>`
    : Object.entries(v).map(([k, x]) => `<tr><td>${esc(k)}</td><td>${bar(x, 1, k === 'yaw' || k === 'pitch')}</td><td class="num">${fmt(x)}</td></tr>`).join('');
  const refRate = (br.explain || {}).ref_rate_hz;
  out.push(card(3, br.type === 'walk_drive' ? 'Bridge: descending drive (frozen adapter)'
    : br.note ? 'Flight command (direct, not the bridge)' : 'Bridge: flight command (frozen adapter)', `
    <table><tbody>${rows}</tbody></table>
    <p class="muted">${esc(br.frozen)}.${refRate !== undefined && refRate !== null ? ` Ref. rate ${fmt(refRate, 1)} Hz.` : ''}</p>
    ${br.note ? `<p class="agent-box">${esc(br.note)}</p>` : ''}
    ${(br.explain || {}).connectome_decides ? `<p><b>Connectome decides:</b> ${esc(br.explain.connectome_decides)}.</p>
      <p class="muted"><b>Hand-designed (not from the connectome):</b> ${esc(br.explain.hand_designed)}.</p>` : ''}
    ${(br.explain || {}).frozen ? `<p class="muted">Adapter <code>${esc(br.explain.frozen.version)}</code>, parameter hash <code>${esc(br.explain.frozen.parameter_hash)}</code>, frozen ${esc(br.explain.frozen.frozen_at)}.</p>` : ''}`));

  const m = run.body.metrics || {};
  const bodyRows = run.mode === 'flight'
    ? [['airborne', m.airborne], ['takeoff time', `${fmt(m.takeoff_time_s)} s`], ['flight time', `${fmt(m.flight_time_s)} s`], ['max height', `${fmt(m.max_height_mm)} mm`]]
    : [['forward displacement', `${fmt(m.forward_disp_mm)} mm`], ['lateral displacement (+ left)', `${fmt(m.lateral_disp_mm)} mm`],
       ['heading change (+ left)', `${fmt(m.heading_change_deg, 1)}°`], ['mean speed', `${fmt(m.mean_speed_mm_s, 1)} mm/s`]];
  out.push(card(4, `Body: ${run.body.model === 'flybody' ? 'FlyBody flight (MuJoCo)' : 'NeuroMechFly walking (flygym / MuJoCo)'}`, `
    <p>Body classifier: <span class="big">${esc(label(m.behavior))}</span></p>
    <table><tbody>${bodyRows.map(([k, x]) => `<tr><td>${esc(k)}</td><td class="num">${esc(x)}</td></tr>`).join('')}</tbody></table>
    <p class="muted">${fmt(run.body.duration_s, 1)} s simulated; poses at ${esc(run.body.poses?.fps)} fps. Legs/wings are moved by the body's controller, not by individual neurons.</p>`));

  const ver = run.verifier;
  let vbody;
  if (ver && ver.available) {
    const kin = ver.kinematic || {};
    const checks = (kin.checks || []).map((c) => `<tr><td>${esc(label(c.name))}</td><td class="num">${esc(typeof c.value === 'number' ? fmt(c.value) : c.value)}</td><td class="muted">${esc(c.threshold)}</td><td>${c.pass ? '<span class="pill correct">pass</span>' : '<span class="pill incorrect">fail</span>'}</td></tr>`).join('');
    vbody = `<p>${verdictPill(ver.final_verdict)} expected <b>${esc(label(run.expected_behavior))}</b></p>
      <p class="muted">${esc(ver.reason || kin.reason || '')}</p>
      <table><thead><tr><th>check</th><th>value</th><th>threshold</th><th></th></tr></thead><tbody>${checks}</tbody></table>
      <p class="muted">Recomputed from the raw trajectory, independent of the body classifier. Vision check: ${esc(ver.vision)}.</p>
      ${ver.benchmark ? `<p class="muted">Verifier benchmark on ${ver.benchmark.n_runs} committed embodied-fly videos (<code>${esc(ver.benchmark.file)}</code>): kinematic check correct on ${ver.benchmark.kinematic_correct_on_known_label}/${ver.benchmark.n_runs}, rejects the opposite label ${ver.benchmark.kinematic_incorrect_on_opposite_label}/${ver.benchmark.n_runs}; blind Claude-vision check exact on ${ver.benchmark.vision_exact_on_unique_videos}/${ver.benchmark.vision_unique_videos} unique videos (labels = body classifier, not independent ground truth).</p>` : ''}`;
  } else {
    vbody = `<p class="muted">${esc(ver?.note || 'No verifier verdict in this export.')}</p>`;
  }
  out.push(card(5, 'Movement verifier agent', vbody));

  const gt = run.ground_truth;
  if (gt) {
    const c = gt.citation || {};
    out.push(card(6, 'Published experiment (ground truth)', `
      <p><b>${esc(gtPhrase(gt))}</b></p>
      ${gt.note ? `<p class="agent-box">${esc(gt.note)}</p>` : ''}
      <p class="quote">${esc(gt.evidence)}</p>
      <p>${esc(c.authors)} (${esc(c.year)}). ${esc(c.title)}. <a href="https://doi.org/${esc(c.doi)}" target="_blank" rel="noopener">doi:${esc(c.doi)}</a></p>
      <p class="muted">DOI verified via ${esc((c.verified_by || []).join(' + ') || 'n/a')}.</p>`));
  }
  const val = run.validation;
  if (val && !val.error) {
    const pillFor = (v) => (v === 'consistent' ? 'correct' : v === 'inconsistent' ? 'incorrect' : 'uncertain');
    const checks = (val.checks || []).map((c) => `<tr><td>${esc(c.gt_id)}</td><td>${esc(label(c.observed))}</td><td><span class="pill ${pillFor(c.verdict)}">${esc(label(c.verdict || 'n/a'))}</span></td></tr>`).join('');
    const nec = val.necessity_check;
    const necTxt = nec ? (nec.text || nec.reason || nec.summary || nec.conclusion || '') : '';
    out.push(card('7', 'Flight validation (recorded benchmark)', `
      <p><span class="pill ${val.as_expected ? 'correct' : 'incorrect'}">${val.as_expected ? 'as expected' : 'not as expected'}</span> expected <b>${esc(label(val.expected))}</b></p>
      <p class="muted">${esc(val.expected_source)}</p>
      <p class="agent-box">${esc(val.result_type_text)}</p>
      ${checks ? `<table><thead><tr><th>published check</th><th>observed</th><th></th></tr></thead><tbody>${checks}</tbody></table>` : '<p class="muted">No literature check for this condition.</p>'}
      ${necTxt ? `<p class="muted">${esc(necTxt)}</p>` : ''}
      ${val.note_walk ? `<p class="muted">${esc(val.note_walk)}</p>` : ''}
      <p class="muted">Source: <code>${esc(val.file)}</code>, condition <code>${esc(val.condition)}</code>, adapter hash <code>${esc(val.frozen_hash || '?')}</code>.
      This export re-simulated the run and ${val.export_reproduces_validation_run ? 'reproduces' : '<b>does not reproduce</b>'} the benchmark row (same behaviour and peak height).</p>`));
  }
  const f = state.manifest?.files?.find((x) => x.path === `data/runs/${run.id}.json`);
  out.push(card('i', 'Provenance', `
    <p>Reproduce: <code>${esc(run.reproduce)}</code></p>
    <p class="muted">Exported ${esc(run.created)} · git ${esc(run.git_rev || '?')} · ${f ? `sha256 <span class="mono">${esc(f.sha256.slice(0, 16))}…</span>` : ''}</p>
    <p class="muted">${esc(run.label)}.</p>`));
  $('#pipeline').innerHTML = out.join('');
}

async function selectRun(id) {
  const entry = state.index.runs.find((r) => r.id === id) || state.index.runs[0];
  for (const b of document.querySelectorAll('.run-chip')) {
    const on = b.dataset.id === entry.id;
    b.setAttribute('aria-pressed', String(on));
    if (on) { try { b.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (e) { /* ignore */ } }
  }
  if (location.hash !== `#${entry.id}`) history.replaceState(null, '', `#${entry.id}`);
  const run = await getJSON(entry.file);
  state.run = run;
  await fly.setRun(run, run.body.geometry);
  brain.setRun(run);
  state.duration = fly.duration || run.body.duration_s || 1;
  state.t = 0;
  $('#body-model').textContent = run.body.model === 'flybody' ? 'FlyBody flight · MuJoCo' : 'NeuroMechFly walking · flygym 2.1 / MuJoCo';
  $('#body-label').innerHTML = `${esc(label(run.body.metrics?.behavior))} ${verdictPill(run.verifier?.final_verdict)}`;
  $('#brain-label').textContent = `${run.brain.n_active.toLocaleString('en-US')} active · mean rate, open loop`;
  renderSide(run);
  renderPipeline(run);
}

function updateTimeUI() {
  $('#time').textContent = `${fmt(state.t)} / ${fmt(state.duration)} s`;
  if (!state.scrubbing) $('#scrub').value = String(Math.round((1000 * state.t) / Math.max(1e-6, state.duration)));
}

function loop(ts) {
  if (state.lastTs !== null && state.playing && state.run) {
    const dt = Math.min(0.1, (ts - state.lastTs) / 1000);
    state.t += dt * state.speed;
    if (state.t > state.duration) state.t = 0;
  }
  state.lastTs = ts;
  if (state.run) { fly.setTime(state.t); updateTimeUI(); }
  fly.render();
  brain.render();
  requestAnimationFrame(loop);
}

function wireControls() {
  const play = $('#play');
  play.textContent = state.playing ? 'Pause' : 'Play';
  play.addEventListener('click', () => { state.playing = !state.playing; play.textContent = state.playing ? 'Pause' : 'Play'; });
  const scrub = $('#scrub');
  scrub.addEventListener('input', () => {
    state.scrubbing = true;
    state.t = (Number(scrub.value) / 1000) * state.duration;
  });
  scrub.addEventListener('change', () => { state.scrubbing = false; });
  $('#speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); });
  $('#follow').addEventListener('change', (e) => { fly.follow = e.target.checked; if (fly.follow) fly.resetCamera(); });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); play.click(); }
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { fly.applyTheme(); brain.applyTheme(); });
  const themeBtn = $('#theme');
  const showTheme = () => {
    const mode = document.documentElement.dataset.theme || 'auto';
    themeBtn.textContent = mode[0].toUpperCase() + mode.slice(1);
    themeBtn.setAttribute('aria-label', `Colour theme: ${mode}`);
  };
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || 'auto';
    const next = { auto: 'light', light: 'dark', dark: 'auto' }[cur];
    if (next === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = next;
    try { if (next === 'auto') localStorage.removeItem('flylab-theme'); else localStorage.setItem('flylab-theme', next); } catch (e) { /* storage unavailable */ }
    showTheme();
    fly.applyTheme(); brain.applyTheme();
  });
  showTheme();
  window.addEventListener('hashchange', () => { const id = location.hash.slice(1); if (id && id !== state.run?.id) selectRun(id); });
}

async function main() {
  try {
    fly = new FlyView($('#fly-canvas'));
    brain = new BrainView($('#brain-canvas'));
    wireControls();
    requestAnimationFrame(loop);
    const [index, manifest] = await Promise.all([
      getJSON('data/runs/index.json'), getJSON('data/manifest.json').catch(() => null)]);
    state.index = index;
    state.manifest = manifest;
    if (!index.runs.length) throw new Error('no runs exported yet (python -m flylab.export3d --all)');
    renderRunChips();
    await brain.load('data/brain_points.json');
    renderLegend();
    const wanted = location.hash.slice(1);
    await selectRun(index.runs.some((r) => r.id === wanted) ? wanted : index.runs[0].id);
    if (manifest) $('#prov').textContent = `data manifest: ${manifest.files.length} files, ${(manifest.total_bytes / 1e6).toFixed(1)} MB, git ${manifest.git_rev || '?'}`;
    window.__flylab = { state, fly, brain, ready: true };
  } catch (err) {
    console.error(err);
    $('#pipeline').innerHTML = `<div class="panel error">Could not load the replay data: ${esc(err.message)}</div>`;
    window.__flylab = { error: String(err) };
  }
}

main();
