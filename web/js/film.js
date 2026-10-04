// Cinematic film of the Embodied Fly Lab (web/film.html). Deterministic: renderAt(t) draws frame t (seconds),
// so tools/render_film.py can capture it frame by frame. All body motion = recorded simulation poses,
// all lit neurons = recorded brain-model activity. The sugar cube is an illustrative prop (labelled on screen).
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const UNIT = { mm: 1, cm: 10, m: 1000, model: 1 };
const DURATION = 53;
const CAPTURE = new URLSearchParams(location.search).has('capture');

// ---------- helpers ----------
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
const ramp = (t, a, b) => smooth((t - a) / (b - a));
const win = (t, a, b, f = 0.6) => Math.min(ramp(t, a, a + f), 1 - ramp(t, b - f, b));
const lerp = (a, b, x) => a + (b - a) * x;
const V = (x, y, z) => new THREE.Vector3(x, y, z);
async function json(u) { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); }

// ---------- renderer ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(CAPTURE ? 1 : Math.min(2, window.devicePixelRatio || 1));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.FogExp2(0x000000, 0.0045);
const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.01, 4000);
scene.add(new THREE.AmbientLight(0x8899bb, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(40, 80, 60); scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 1.4); rim.position.set(-60, 30, -80); scene.add(rim);
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.7, 0.5, 0.42);
composer.addPass(bloom);
composer.addPass(new OutputPass());
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight); composer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
});

// MuJoCo world (z up, mm) inside three (y up)
const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);

// floor: soft radial disc
function radialTexture(inner, outer) {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d'); const gr = g.createRadialGradient(256, 256, 0, 256, 256, 256);
  gr.addColorStop(0, inner); gr.addColorStop(1, outer); g.fillStyle = gr; g.fillRect(0, 0, 512, 512);
  return new THREE.CanvasTexture(c);
}
const floor = new THREE.Mesh(new THREE.CircleGeometry(16, 96),
  new THREE.MeshBasicMaterial({ map: radialTexture('rgba(34,40,56,0.55)', 'rgba(0,0,0,0)'), transparent: true, depthWrite: false }));
world.add(floor);
const ring = new THREE.Mesh(new THREE.RingGeometry(5.5, 5.62, 128), new THREE.MeshBasicMaterial({ color: 0x3a4766, transparent: true, opacity: 0.25 }));
ring.position.z = 0.01; world.add(ring);

// ---------- glass fly ----------
const glassVS = `varying vec3 vN; varying vec3 vV;
  void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`;
const glassFS = `uniform vec3 uBase; uniform vec3 uRim; uniform float uAlpha; varying vec3 vN; varying vec3 vV;
  void main(){ float ln = length(vN); vec3 n = ln > 1e-6 ? vN / ln : vec3(0.0, 0.0, 1.0); float lv = length(vV); vec3 v = lv > 1e-6 ? vV / lv : vec3(0.0, 0.0, 1.0); float f = pow(clamp(1.0 - abs(dot(n, v)), 0.0, 1.0), 2.2);
    gl_FragColor = vec4(mix(uBase, uRim, f), uAlpha * (0.03 + 0.55 * f)); }`;
function glassMat(base, rimC) {
  return new THREE.ShaderMaterial({ uniforms: { uBase: { value: new THREE.Color(base) }, uRim: { value: new THREE.Color(rimC) }, uAlpha: { value: 1 } },
    vertexShader: glassVS, fragmentShader: glassFS, transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
}

async function loadGeom(url) {
  const meta = await json(url);
  const buf = await (await fetch(url.replace(/[^/]+$/, meta.bin))).arrayBuffer();
  const verts = new Float32Array(buf, 0, meta.vertex_floats);
  const idx = meta.index_type === 'uint32' ? new Uint32Array(buf, meta.index_byte_offset, meta.index_count)
    : new Uint16Array(buf, meta.index_byte_offset, meta.index_count);
  return { meta, verts, idx };
}
function primitive(g) {
  const s = g.size; let geo;
  if (g.type === 'sphere') geo = new THREE.SphereGeometry(s[0], 16, 12);
  else if (g.type === 'ellipsoid') { geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(s[0], s[1], s[2]); }
  else if (g.type === 'capsule') { geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 4, 12); geo.rotateX(Math.PI / 2); }
  else if (g.type === 'cylinder') { geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 16); geo.rotateX(Math.PI / 2); }
  else if (g.type === 'box') geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  return geo || null;
}
function buildFly(geom) {
  const root = new THREE.Group(); root.scale.setScalar(UNIT[geom.meta.units] ?? 1); world.add(root);
  const bodies = new Map();
  const body = glassMat(0x1b2740, 0xcfe0ff), wing = glassMat(0x0d1424, 0x9fb8ff), eye = glassMat(0x2a0c0c, 0xff8a7a);
  for (const g of geom.meta.geoms) {
    let bg = bodies.get(g.body);
    if (!bg) { bg = new THREE.Group(); bg.name = g.body; bodies.set(g.body, bg); root.add(bg); }
    let geo;
    if (g.type === 'mesh' && g.mesh) {
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(geom.verts.subarray(g.mesh.vOff, g.mesh.vOff + 3 * g.mesh.vCount), 3));
      geo.setIndex(new THREE.BufferAttribute(geom.idx.subarray(g.mesh.iOff, g.mesh.iOff + g.mesh.iCount), 1));
      geo.computeVertexNormals();
    } else geo = primitive(g);
    if (!geo) continue;
    const n = (g.body + ' ' + (g.name || '')).toLowerCase();
    const m = new THREE.Mesh(geo, /wing/.test(n) ? wing : /eye/.test(n) ? eye : body);
    m.position.fromArray(g.pos); m.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); m.updateMatrix();
    bg.add(m);
  }
  return { root, bodies, mats: [body, wing, eye], unit: UNIT[geom.meta.units] ?? 1 };
}
function setAlpha(fly, a) { fly.mats.forEach((m, i) => { m.uniforms.uAlpha.value = a * (i === 1 ? 0.8 : 1); }); fly.root.visible = a > 0.002; }

// pose access (MuJoCo frame, mm)
function frameAt(P, t) {
  const T = P.t, n = T.length; let f = 0;
  while (f < n - 1 && T[f + 1] <= t) f++;
  const f1 = Math.min(n - 1, f + 1);
  const a = f1 === f ? 0 : clamp((t - T[f]) / (T[f1] - T[f]));
  return { f, f1, a };
}
const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion();
function applyPose(fly, P, t) {
  const { f, f1, a } = frameAt(P, t);
  const p0 = P.p[f], p1 = P.p[f1], q0 = P.q[f], q1 = P.q[f1];
  P.bodies.forEach((name, i) => {
    const g = fly.bodies.get(name); if (!g) return;
    const j = 3 * i, k = 4 * i;
    g.position.set(lerp(p0[j], p1[j], a), lerp(p0[j + 1], p1[j + 1], a), lerp(p0[j + 2], p1[j + 2], a));
    _q0.set(q0[k + 1], q0[k + 2], q0[k + 3], q0[k]); _q1.set(q1[k + 1], q1[k + 2], q1[k + 3], q1[k]);
    g.quaternion.slerpQuaternions(_q0, _q1, a);
  });
  for (const [name, g] of fly.bodies) g.visible = P.bodies.includes(name);
}
function bodyPose(P, name, t, u) {
  const i = P.bodies.indexOf(name); if (i < 0) return null;
  const { f, f1, a } = frameAt(P, t), j = 3 * i, k = 4 * i;
  const p = V(lerp(P.p[f][j], P.p[f1][j], a), lerp(P.p[f][j + 1], P.p[f1][j + 1], a), lerp(P.p[f][j + 2], P.p[f1][j + 2], a)).multiplyScalar(u);
  const q0 = new THREE.Quaternion(P.q[f][k + 1], P.q[f][k + 2], P.q[f][k + 3], P.q[f][k]);
  const q1 = new THREE.Quaternion(P.q[f1][k + 1], P.q[f1][k + 2], P.q[f1][k + 3], P.q[f1][k]);
  return { p, q: q0.slerp(q1, a) };
}

// ---------- brain ----------
const CLASS_COL = { central: 0x8fb3ff, optic: 0x2f4a8a, visual_projection: 0x6fd3ff, visual_centrifugal: 0x4f7bd0, sensory: 0x7af0c2,
  ascending: 0xb59cff, descending: 0xffd27a, motor: 0xff8a65, endocrine: 0xff9ad5, unknown: 0x8890a8 };
const brain = { group: new THREE.Group(), base: null, act: null, local: null, n: 0, width: 815 };
world.add(brain.group);
async function loadBrain() {
  const meta = await json('data/brain_points.json');
  const buf = await (await fetch('data/' + meta.bin)).arrayBuffer();
  const n = meta.n, raw = new Float32Array(buf, 0, 3 * n), cls = new Uint8Array(buf, 12 * n, n);
  const [cx, cy, cz] = meta.center;
  brain.local = new Float32Array(3 * n);
  const pos = [], col = [];
  const pal = meta.classes.map((c) => new THREE.Color(CLASS_COL[c] ?? 0x8890a8));
  for (let i = 0; i < n; i++) {
    const x = raw[3 * i], y = raw[3 * i + 1], z = raw[3 * i + 2];
    // FlyWire (x: left->right, y: ventral+, z: posterior+, um) -> head-local MuJoCo frame (x fwd, y left, z up), um
    const X = -(z - cz), Y = -(x - cx), Z = -(y - cy);
    brain.local[3 * i] = X; brain.local[3 * i + 1] = Y; brain.local[3 * i + 2] = Z;
    if (!Number.isFinite(x)) continue;
    pos.push(X, Y, Z); const c = pal[cls[i]] || pal[pal.length - 1]; const k = cls[i] === 1 ? 0.35 : 0.8; col.push(c.r * k, c.g * k, c.b * k);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  brain.base = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.0035, sizeAttenuation: true, vertexColors: true, transparent: true,
    opacity: 0.3, depthWrite: false, blending: THREE.AdditiveBlending }));
  brain.group.add(brain.base);
  brain.n = n; brain.width = meta.bbox_max[0] - meta.bbox_min[0];
}
const activeSets = {};
function makeActive(run, color) {
  const b = run.brain, idx = b.active_idx || [], rate = b.active_rate_hz || [];
  const pos = [], col = [], c = new THREE.Color(color), stim = new Set(b.stimulated_idx || []);
  idx.forEach((i, j) => {
    const X = brain.local[3 * i]; if (!Number.isFinite(X)) return;
    pos.push(X, brain.local[3 * i + 1], brain.local[3 * i + 2]);
    const k = stim.has(i) ? 1.6 : 0.45 + 0.9 * clamp(rate[j] / 150);
    col.push(c.r * k, c.g * k, c.b * k);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.014, sizeAttenuation: true, vertexColors: true, transparent: true,
    opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
  brain.group.add(pts);
  return pts;
}
function placeBrain(pos, quat, widthMm) {
  brain.group.position.copy(pos); brain.group.quaternion.copy(quat); brain.group.scale.setScalar(widthMm / brain.width);
}

// ---------- signal pulses (brain -> legs / wings) ----------
const PULSE_K = 42;
const pulse = { pts: null, curves: [] };
function initPulses(maxCurves = 8) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxCurves * PULSE_K * 3), 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(maxCurves * PULSE_K * 3), 3));
  pulse.pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.09, sizeAttenuation: true, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending }));
  pulse.pts.frustumCulled = false; world.add(pulse.pts);
}
function drawPulses(starts, ends, color, t, alpha, speed = 0.9) {
  const P = pulse.pts.geometry.attributes.position.array, C = pulse.pts.geometry.attributes.color.array;
  P.fill(0); C.fill(0);
  const c = new THREE.Color(color), tmp = new THREE.Vector3();
  ends.forEach((e, ci) => {
    const s = starts; const mid = s.clone().add(e).multiplyScalar(0.5); mid.z -= 0.25 * s.distanceTo(e);
    const curve = new THREE.QuadraticBezierCurve3(s, mid, e);
    for (let k = 0; k < PULSE_K; k++) {
      const u = k / (PULSE_K - 1); curve.getPoint(u, tmp);
      const o = 3 * (ci * PULSE_K + k);
      P[o] = tmp.x; P[o + 1] = tmp.y; P[o + 2] = tmp.z;
      const ph = (t * speed + ci * 0.07) % 1;
      const glow = 0.18 + 1.6 * Math.exp(-((u - ph) ** 2) / 0.006);
      C[o] = c.r * glow * alpha; C[o + 1] = c.g * glow * alpha; C[o + 2] = c.b * glow * alpha;
    }
  });
  pulse.pts.geometry.attributes.position.needsUpdate = true; pulse.pts.geometry.attributes.color.needsUpdate = true;
}

// ---------- text ----------
const $ = (id) => document.getElementById(id);
const cues = [];
function cue(id, text, a, b, cls = '') { cues.push({ id, text, a, b, cls }); }
function drawText(t) {
  for (const id of ['title', 'sub', 'note']) {
    const c = cues.find((q) => q.id === id && t >= q.a && t <= q.b);
    const el = $(id);
    if (!c) { el.style.opacity = 0; continue; }
    if (el.textContent !== c.text) el.textContent = c.text;
    el.className = 't ' + c.cls;
    const o = win(t, c.a, c.b, 0.7);
    el.style.opacity = o; el.style.transform = `translateY(${(1 - o) * 14}px)`;
  }
}
// story (seconds)
cue('title', '138,639 neurons.', 0.8, 6.6);
cue('sub', 'The complete wiring of a fruit-fly brain.', 1.6, 6.6);
cue('title', 'Now it has a body.', 8.2, 13.6);
cue('sub', 'A physics-simulated fly around a simulated brain.', 9.0, 13.6);
cue('title', 'Neurons, wired to muscles.', 14.4, 19.4);
cue('sub', 'Descending neurons connect the brain to legs and wings.', 15.0, 19.4);
cue('title', 'Test 1: steering.', 20.6, 24.4);
cue('sub', 'Left steering neurons fire. The adapter is wired wrong on purpose.', 21.2, 24.4);
cue('title', 'Wrong turn.', 24.8, 28.0, 'red');
cue('sub', 'The verifier expected left. The fly went right.', 25.2, 28.0);
cue('title', 'Detected. Rewired.', 28.2, 30.2, 'red');
cue('title', 'Verified.', 30.6, 33.6, 'green');
cue('sub', 'Same neurons, left turn, as published.', 31.0, 33.6);
cue('title', 'Test 2: escape.', 34.6, 38.4);
cue('sub', 'Giant fiber and DNg02 neurons fire.', 35.2, 38.4);
cue('title', 'Lift-off.', 38.8, 42.6, 'green');
cue('title', 'Taste.', 43.0, 46.2, 'grad');
cue('sub', 'Sugar neurons light up the feeding neuron MN9.', 43.4, 46.2);
cue('title', 'Embodied Fly Lab', 47.0, 53.0, 'grad');
cue('sub', 'Our agents test, simulate and check every result against published research.', 48.0, 53.0);
cue('note', 'Recorded simulations: FlyWire brain model, NeuroMechFly and FlyBody in MuJoCo. Brain shown scaled inside the head.', 0.5, 33.6);
cue('note', 'Recorded flight simulation. Sugar cube is an illustration.', 34.0, 46.4);
cue('note', 'FlyWire · Shiu et al. 2024 · NeuroMechFly · FlyBody · Omnigent · github.com/maximiliankahl/embodied-fly-lab', 47.6, 53.0);

// ---------- load ----------
const S = {};
async function init() {
  const [gFB, gNMF, rMis, rOk, rFly, rSug] = await Promise.all([
    loadGeom('data/geometry/flybody.json'), loadGeom('data/geometry/neuromechfly.json'),
    json('data/runs/story_miswired_dna02l.json'), json('data/runs/dna02l_turn_left.json'),
    json('data/runs/gf_dng02_climb.json'), json('data/runs/story_sugar_feeding.json'), loadBrain()]);
  S.fb = buildFly(gFB); S.nmf = buildFly(gNMF);
  S.mis = rMis.body.poses; S.ok = rOk.body.poses; S.fly = rFly.body.poses;
  S.uFly = UNIT[S.fly.units] ?? 1; S.uWalk = UNIT[S.mis.units] ?? 1;
  S.actMis = makeActive(rMis, 0xff4d6d); S.actOk = makeActive(rOk, 0x30d158); S.actFly = makeActive(rFly, 0xffb340); S.actSug = makeActive(rSug, 0xff7ad9);
  // FlyBody head: centre of the head meshes in head-local coordinates (mm)
  const hb = new THREE.Box3(); const head = S.fb.bodies.get('head');
  head.children.forEach((m) => { m.geometry.computeBoundingBox(); hb.union(m.geometry.boundingBox.clone().applyMatrix4(m.matrix)); });
  S.headOff = hb.getCenter(new THREE.Vector3()).multiplyScalar(S.fb.unit);
  const hs = hb.getSize(new THREE.Vector3()); S.headW = Math.max(hs.x, hs.y, hs.z) * S.fb.unit;
  initPulses(8);
  S.trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffb340, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending }));
  const ti = S.fly.bodies.indexOf('thorax'); const tp = [];
  S.fly.p.forEach((p) => tp.push(p[3 * ti] * S.uFly, p[3 * ti + 1] * S.uFly, p[3 * ti + 2] * S.uFly));
  S.trail.geometry.setAttribute('position', new THREE.Float32BufferAttribute(tp, 3)); S.trail.frustumCulled = false; world.add(S.trail);
  const end = V(tp[tp.length - 3], tp[tp.length - 2], tp[tp.length - 1]);
  S.sugar = new THREE.Mesh(new RoundedBoxGeometry(1.7, 1.7, 1.7, 4, 0.28),
    new THREE.MeshStandardMaterial({ color: 0xf6efe6, roughness: 0.35, metalness: 0.0, emissive: 0xffc27a, emissiveIntensity: 0.12 }));
  S.sugar.position.copy(end).add(V(7, 1.5, 0.6)); world.add(S.sugar);
}

// ---------- frame ----------
const tmpV = new THREE.Vector3();
function camOrbit(target, r, azDeg, elDeg) {
  // spherical around target in MuJoCo frame (z up) -> three coords
  const az = THREE.MathUtils.degToRad(azDeg), el = THREE.MathUtils.degToRad(elDeg);
  const m = V(target.x + r * Math.cos(el) * Math.cos(az), target.y + r * Math.cos(el) * Math.sin(az), target.z + r * Math.sin(el));
  const toThree = (v) => V(v.x, v.z, -v.y);
  camera.position.copy(toThree(m)); camera.lookAt(toThree(target));
}
const LEGS_FB = ['coxa_T1_left', 'coxa_T2_left', 'coxa_T3_left', 'coxa_T1_right', 'coxa_T2_right', 'coxa_T3_right'];
const WINGS_FB = ['wing_left', 'wing_right'];
const LEGS_NMF = ['nmf/lf_coxa', 'nmf/lm_coxa', 'nmf/lh_coxa', 'nmf/rf_coxa', 'nmf/rm_coxa', 'nmf/rh_coxa'];

function renderAt(t) {
  t = clamp(t, 0, DURATION);
  const walk = t >= 20 && t < 34;
  // ---- choose body + time
  let simT = 0, P = S.fly, fly = S.fb, u = S.uFly;
  if (walk) {
    fly = S.nmf; u = S.uWalk;
    if (t < 28) { P = S.mis; simT = clamp((t - 20.8) * 0.2, 0, 0.99); }
    else if (t < 29.4) { P = S.mis; simT = clamp(0.99 * (1 - (t - 28) / 1.2), 0, 0.99); }
    else { P = S.ok; simT = clamp((t - 29.6) * 0.22, 0, 0.99); }
  } else if (t >= 34) {
    simT = t < 36.2 ? 0 : clamp((t - 36.2) / 6.8 * 0.99, 0, 0.99);
  }
  setAlpha(S.fb, walk ? 0 : (t < 7.5 ? 0 : ramp(t, 7.5, 11.5)));
  setAlpha(S.nmf, walk ? 1 : 0);
  applyPose(fly, P, simT);

  // ---- brain placement
  if (walk) {
    const l = bodyPose(P, 'nmf/l_eye', simT, u), r = bodyPose(P, 'nmf/r_eye', simT, u), th = bodyPose(P, 'nmf/c_thorax', simT, u);
    placeBrain(l.p.clone().add(r.p).multiplyScalar(0.5), th.q, 0.8 * l.p.distanceTo(r.p));
  } else {
    const h = bodyPose(P, 'head', simT, u), th = bodyPose(P, 'thorax', simT, u);
    placeBrain(h.p.clone().add(S.headOff.clone().applyQuaternion(h.q)), th.q, 0.72 * S.headW);
  }
  const thorax = bodyPose(P, walk ? 'nmf/c_thorax' : 'thorax', simT, u).p;
  const bc = brain.group.position.clone();

  // ---- brain styling
  brain.base.material.opacity = t < 7 ? 0.3 * ramp(t, 0.2, 2.2) : lerp(0.3, 0.22, ramp(t, 7, 11));
  brain.base.material.size = lerp(0.0032, 0.0045, ramp(t, 0, 7));
  S.actMis.material.opacity = walk && t < 29.4 ? ramp(t, 21.0, 22.0) : 0;
  S.actOk.material.opacity = walk && t >= 29.4 ? ramp(t, 29.6, 30.4) : 0;
  S.actFly.material.opacity = t >= 34 && t < 43 ? ramp(t, 35.0, 36.0) * (1 - ramp(t, 42.4, 43.0)) : 0;
  S.actSug.material.opacity = t >= 43 && t < 47 ? ramp(t, 43.0, 43.8) * (1 - ramp(t, 46.2, 47)) : 0;

  // ---- pulses
  let ends = [], col = 0x7fb2ff, a = 0;
  if (t >= 14 && t < 20) { ends = [...LEGS_FB, ...WINGS_FB]; a = win(t, 14.2, 19.8, 0.8); }
  else if (walk && t < 28) { ends = LEGS_NMF; col = 0xff4d6d; a = win(t, 21.0, 27.8, 0.6); }
  else if (walk) { ends = LEGS_NMF; col = 0x30d158; a = win(t, 29.6, 33.6, 0.6); }
  else if (t >= 34 && t < 43) { ends = [...WINGS_FB, 'coxa_T2_left', 'coxa_T2_right']; col = 0xffb340; a = win(t, 35.0, 42.8, 0.6); }
  const endPos = ends.map((n) => { const bp = bodyPose(P, n, simT, u); return bp ? bp.p : bc; });
  drawPulses(bc, endPos, col, t, a);

  // ---- flight extras
  const fl = t >= 34 && t < 47;
  S.trail.visible = fl && simT > 0;
  if (S.trail.visible) { const { f } = frameAt(S.fly, simT); S.trail.geometry.setDrawRange(0, f + 1); }
  S.sugar.visible = fl && t > 41.5;
  S.sugar.scale.setScalar(smooth((t - 41.5) / 1.2) || 0.0001);
  S.sugar.rotation.z = t * 0.4; S.sugar.rotation.x = 0.4;
  floor.visible = !fl || simT < 0.3; ring.visible = floor.visible;
  if (walk) { floor.position.set(thorax.x, thorax.y, 0); ring.position.set(thorax.x, thorax.y, 0.01); }
  else { floor.position.set(0, 0, 0); ring.position.set(0, 0, 0.01); }

  // ---- camera
  if (t < 7) camOrbit(bc, lerp(1.25, 1.6, ramp(t, 0, 7)), lerp(10, -25, t / 7), lerp(8, 14, t / 7));
  else if (t < 14) { const x = ramp(t, 7, 13.5); camOrbit(bc.clone().lerp(thorax, x), lerp(1.6, 7.5, x), lerp(-25, -55, x), lerp(14, 24, x)); }
  else if (t < 20) camOrbit(thorax, lerp(7.5, 6.8, ramp(t, 14, 20)), lerp(-55, -95, ramp(t, 14, 20)), 26);
  else if (walk) camOrbit(thorax, lerp(10.5, 9.5, ramp(t, 20, 34)), lerp(-120, -150, ramp(t, 20, 34)), 52);
  else if (t < 47) {
    const look = thorax.clone(); if (t > 43) look.lerp(S.sugar.position, 0.3 * ramp(t, 43, 45.5));
    camOrbit(look, lerp(9, 16, ramp(t, 36, 43)), lerp(-60, -25, ramp(t, 34, 46)), lerp(18, 10, ramp(t, 36, 43)));
  } else camOrbit(thorax, 20, -25 + (t - 47) * 3, 12);

  // ---- fades + text
  const fade = Math.max(win(t, 19.4, 20.6, 0.6), win(t, 33.4, 34.6, 0.6), ramp(t, 46.3, 47.2));
  $('fade').style.opacity = fade;
  drawText(t);
  composer.render();
}

init().then(() => {
  renderAt(0);
  window.__film = { ready: true, duration: DURATION, renderAt };
  window.__dbg = { camera, S, brain };
  if (!CAPTURE) {
    const t0 = performance.now();
    const loop = () => { renderAt(((performance.now() - t0) / 1000) % DURATION); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
}).catch((e) => { console.error(e); document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f66">${e}</pre>`); window.__film = { error: String(e) }; });
