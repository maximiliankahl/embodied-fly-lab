// 3D replay of the physics fly body (NeuroMechFly walking or FlyBody flight).
// Geometry: data/geometry/<model>.json/.bin (flylab.export3d); poses: run.body.poses (flylab.poses format).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const UNIT_TO_MM = { mm: 1, cm: 10, m: 1000, model: 1 };

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function partColor(name) {
  const n = name.toLowerCase();
  if (/eye/.test(n)) return { color: 0x9b2b1f, roughness: 0.5 };
  if (/wing/.test(n)) return { color: 0xc9d6e6, opacity: 0.38, roughness: 0.2 };
  if (/haltere/.test(n)) return { color: 0xc2ab7d };
  if (/arista|funiculus|pedicel|antenn/.test(n)) return { color: 0x7d5b33 };
  if (/abdomen|abdom/.test(n)) {
    const k = parseInt((n.match(/(\d+)$/) || [0, 0])[1], 10);
    return { color: k % 2 ? 0x8a5a26 : 0x6d451c };
  }
  if (/coxa|femur|tibia|tarsus|claw|leg/.test(n)) return { color: /tarsus|claw/.test(n) ? 0x4a3420 : 0x6e4b2a };
  if (/thorax|head|rostrum|haustellum|proboscis|labrum/.test(n)) return { color: 0xa8743a };
  return { color: 0x9a7b55 };
}

function primitive(g) {
  const s = g.size;
  let geo;
  switch (g.type) {
    case 'sphere': geo = new THREE.SphereGeometry(s[0], 16, 12); break;
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(s[0], s[1], s[2]); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 4, 12); geo.rotateX(Math.PI / 2); break;
    case 'cylinder': geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 16); geo.rotateX(Math.PI / 2); break;
    case 'box': geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    default: return null;
  }
  return geo;
}

export class FlyView {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 2000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x6b5a45, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(30, 60, 40);
    this.scene.add(sun);

    // MuJoCo world is z-up; three.js is y-up. Everything from the simulation lives in `world`.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2; // (x, y, z)_mujoco -> (x, z, -y)_three
    this.scene.add(this.world);

    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600),
      new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 1 }));
    this.world.add(this.ground);
    this.grid1 = new THREE.GridHelper(200, 200, 0xaaaaaa, 0xaaaaaa); // 1 mm
    this.grid10 = new THREE.GridHelper(200, 20, 0x888888, 0x888888); // 10 mm
    for (const g of [this.grid1, this.grid10]) {
      g.rotation.x = Math.PI / 2; // grid in the MuJoCo xy plane
      g.position.z = 0.002;
      g.material.transparent = true;
      this.world.add(g);
    }
    this.grid1.material.opacity = 0.35;
    this.grid10.material.opacity = 0.7;

    this.fly = new THREE.Group();
    this.world.add(this.fly);
    this.bodyGroups = new Map();
    this.geomCache = new Map();
    this.trail = null;
    this.shadowTrail = null;
    this.follow = true;
    this._q0 = new THREE.Quaternion();
    this._q1 = new THREE.Quaternion();
    this._tmp = new THREE.Vector3();
    this._lastTarget = null;
    this.applyTheme();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  applyTheme() {
    this.scene.background = new THREE.Color(cssVar('--scene-bg', '#eef0f2'));
    this.ground.material.color.set(cssVar('--ground', '#e4e1da'));
    this.grid1.material.color.set(cssVar('--grid', '#c9c3b6'));
    this.grid10.material.color.set(cssVar('--grid-2', '#a9a194'));
    if (this.trail) this.trail.material.color.set(cssVar('--trail', '#d0442b'));
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async loadGeometry(url) {
    if (this.geomCache.has(url)) return this.geomCache.get(url);
    const meta = await (await fetch(url)).json();
    const binUrl = url.replace(/[^/]+$/, meta.bin);
    const buf = await (await fetch(binUrl)).arrayBuffer();
    const verts = new Float32Array(buf, 0, meta.vertex_floats);
    const idx = meta.index_type === 'uint32'
      ? new Uint32Array(buf, meta.index_byte_offset, meta.index_count)
      : new Uint16Array(buf, meta.index_byte_offset, meta.index_count);
    const entry = { meta, verts, idx };
    this.geomCache.set(url, entry);
    return entry;
  }

  buildBody(geom) {
    for (const g of this.bodyGroups.values()) this.fly.remove(g);
    this.bodyGroups.clear();
    const mats = new Map();
    const scale = UNIT_TO_MM[geom.meta.units] ?? 1;
    this.fly.scale.setScalar(scale);
    for (const g of geom.meta.geoms) {
      let bg = this.bodyGroups.get(g.body);
      if (!bg) { bg = new THREE.Group(); bg.name = g.body; this.bodyGroups.set(g.body, bg); this.fly.add(bg); }
      let geo;
      if (g.type === 'mesh' && g.mesh) {
        geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(geom.verts.subarray(g.mesh.vOff, g.mesh.vOff + 3 * g.mesh.vCount), 3));
        geo.setIndex(new THREE.BufferAttribute(geom.idx.subarray(g.mesh.iOff, g.mesh.iOff + g.mesh.iCount), 1));
        geo.computeVertexNormals();
      } else {
        geo = primitive(g);
        if (!geo) continue;
      }
      const look = partColor(g.body + ' ' + (g.name || ''));
      const key = JSON.stringify(look);
      if (!mats.has(key)) {
        mats.set(key, new THREE.MeshStandardMaterial({
          color: look.color, roughness: look.roughness ?? 0.65, metalness: 0.05,
          transparent: look.opacity !== undefined, opacity: look.opacity ?? 1,
          side: look.opacity !== undefined ? THREE.DoubleSide : THREE.FrontSide,
          depthWrite: look.opacity === undefined,
        }));
      }
      const mesh = new THREE.Mesh(geo, mats.get(key));
      mesh.position.fromArray(g.pos);
      mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      bg.add(mesh);
    }
  }

  async setRun(run, geometryUrl) {
    const geom = await this.loadGeometry(geometryUrl);
    if (this.currentGeometry !== geometryUrl) { this.buildBody(geom); this.currentGeometry = geometryUrl; }
    const poses = run.body.poses;
    this.poses = poses;
    this.unit = UNIT_TO_MM[poses.units] ?? 1;
    this.bodyIndex = poses.bodies.map((b) => this.bodyGroups.get(b) || null);
    for (const [name, g] of this.bodyGroups) g.visible = poses.bodies.includes(name);
    this.duration = poses.t[poses.t.length - 1] || 0;
    const ti = poses.bodies.findIndex((b) => /thorax$/i.test(b));
    this.thoraxIdx = ti >= 0 ? ti : 0;
    // flight poses are stroboscopic (one frame every ~3 wingbeats): show recorded frames, do not interpolate
    this.interpolate = run.mode !== 'flight';
    this.buildTrail(run.mode === 'flight');
    this.setTime(0);
    this.resetCamera();
  }

  buildTrail(flight) {
    for (const t of [this.trail, this.shadowTrail]) if (t) { this.world.remove(t); t.geometry.dispose(); }
    const n = this.poses.t.length, i3 = this.thoraxIdx * 3;
    const pts = new Float32Array(n * 3), sh = new Float32Array(n * 3);
    for (let f = 0; f < n; f++) {
      const p = this.poses.p[f];
      const u = this.unit;
      pts[3 * f] = p[i3] * u; pts[3 * f + 1] = p[i3 + 1] * u; pts[3 * f + 2] = flight ? p[i3 + 2] * u : 0.03;
      sh[3 * f] = p[i3] * u; sh[3 * f + 1] = p[i3 + 1] * u; sh[3 * f + 2] = 0.02;
    }
    const mk = (arr, color, opacity) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
      line.frustumCulled = false;
      this.world.add(line);
      return line;
    };
    this.trail = mk(pts, cssVar('--trail', '#d0442b'), 1);
    this.shadowTrail = flight ? mk(sh, cssVar('--muted', '#888888'), 0.6) : null;
  }

  thoraxAt(f) {
    const p = this.poses.p[f], i3 = this.thoraxIdx * 3, u = this.unit;
    // MuJoCo (x, y, z) -> three (x, z, -y)
    return new THREE.Vector3(p[i3] * u, p[i3 + 2] * u, -p[i3 + 1] * u);
  }

  resetCamera() {
    const c = this.thoraxAt(0);
    this.controls.target.copy(c);
    const flight = this.shadowTrail !== null;
    this.camera.position.set(c.x + (flight ? 14 : 5.5), c.y + (flight ? 9 : 4.5), c.z + (flight ? 18 : 7));
    this._lastTarget = c.clone();
    this.controls.update();
  }

  setTime(t) {
    const P = this.poses;
    if (!P) return;
    const T = P.t, n = T.length;
    let f = Math.min(n - 1, Math.max(0, Math.round(t * (P.fps || 60))));
    while (f > 0 && T[f] > t) f--;
    while (f < n - 1 && T[f + 1] <= t) f++;
    const f1 = Math.min(n - 1, f + 1);
    const a = !this.interpolate || f1 === f ? 0 : Math.min(1, Math.max(0, (t - T[f]) / (T[f1] - T[f])));
    const p0 = P.p[f], p1 = P.p[f1], q0 = P.q[f], q1 = P.q[f1];
    for (let i = 0; i < this.bodyIndex.length; i++) {
      const g = this.bodyIndex[i];
      if (!g) continue;
      const j = 3 * i, k = 4 * i;
      g.position.set(p0[j] + (p1[j] - p0[j]) * a, p0[j + 1] + (p1[j + 1] - p0[j + 1]) * a, p0[j + 2] + (p1[j + 2] - p0[j + 2]) * a);
      this._q0.set(q0[k + 1], q0[k + 2], q0[k + 3], q0[k]);
      this._q1.set(q1[k + 1], q1[k + 2], q1[k + 3], q1[k]);
      g.quaternion.slerpQuaternions(this._q0, this._q1, a);
    }
    if (this.trail) this.trail.geometry.setDrawRange(0, f + 1);
    if (this.shadowTrail) this.shadowTrail.geometry.setDrawRange(0, f + 1);
    if (this.follow && this._lastTarget) {
      const c = this.thoraxAt(f);
      this._tmp.subVectors(c, this._lastTarget);
      this.camera.position.add(this._tmp);
      this.controls.target.add(this._tmp);
      this._lastTarget.copy(c);
    }
    this.frame = f;
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
