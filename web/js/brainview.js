// 3D point cloud of all 138,639 neurons of the whole-brain model, coloured by FlyWire super_class,
// with the run's active neurons glowing by mean firing rate (flylab.export3d brain_points + run.brain).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const CLASS_COLORS = {
  central: '#8d96a8', optic: '#5f9aa0', visual_projection: '#4f74c9', visual_centrifugal: '#3fb6c9',
  sensory: '#5aa765', ascending: '#9a6cc4', descending: '#d08a2e', motor: '#c4504a', endocrine: '#c46aa0',
  unknown: '#9a9a9a',
};
export const ROLE_COLORS = { stimulated: '#ff3fd2', readout: '#00e0ff', silenced: '#7a7a7a' };

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// rate -> warm colour (dim orange .. bright yellow-white)
function heat(rate, out) {
  const x = Math.min(1, Math.log1p(rate) / Math.log1p(200));
  out.setRGB(1.0, 0.35 + 0.6 * x, 0.08 + 0.75 * x * x);
  return out;
}

const glowVS = `
  attribute float size;
  attribute vec3 color;
  varying vec3 vColor;
  uniform float uScale;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(2.0, size * uScale / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const glowFS = `
  varying vec3 vColor;
  uniform float uDark;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float core = smoothstep(0.5, 0.0, d);
    float a = mix(core, pow(core, 1.5), uDark);
    gl_FragColor = vec4(vColor * (0.75 + 0.5 * core), a);
  }`;

export class BrainView {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 1, 20000);
    this.camera.position.set(0, 60, 1500);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.autoRotate = false;
    this.glowMat = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 400 }, uDark: { value: 0 } },
      vertexShader: glowVS, fragmentShader: glowFS, transparent: true, depthWrite: false,
    });
    this.active = null;
    this.applyTheme();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  isDark() {
    const t = document.documentElement.dataset.theme;
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  applyTheme() {
    const dark = this.isDark();
    this.scene.background = new THREE.Color(cssVar('--brain-bg', dark ? '#0d0e11' : '#f3f4f7'));
    this.glowMat.blending = dark ? THREE.AdditiveBlending : THREE.NormalBlending;
    this.glowMat.uniforms.uDark.value = dark ? 1 : 0;
    this.glowMat.needsUpdate = true;
    if (this.base) {
      this.base.material.opacity = dark ? 0.28 : 0.3;
      this.base.material.blending = dark ? THREE.AdditiveBlending : THREE.NormalBlending;
      this.base.material.needsUpdate = true;
    }
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const px = h * this.renderer.getPixelRatio();
    this.glowMat.uniforms.uScale.value = px / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    if (this.base) this.base.material.size = Math.max(1, 1.2 * this.renderer.getPixelRatio());
  }

  async load(metaUrl) {
    const meta = await (await fetch(metaUrl)).json();
    const binUrl = metaUrl.replace(/[^/]+$/, meta.bin);
    const buf = await (await fetch(binUrl)).arrayBuffer();
    const n = meta.n;
    const raw = new Float32Array(buf, 0, 3 * n);
    const cls = new Uint8Array(buf, 12 * n, n);
    const [cx, cy, cz] = meta.center;
    // FlyWire frame (x: fly's left -> right, y: dorsal -> ventral, z: anterior -> posterior, um)
    // to a frontal view: anterior towards the camera (+z), dorsal up (+y), fly's left on screen right (+x).
    const pos = new Float32Array(3 * n);
    const col = new Float32Array(3 * n);
    const palette = meta.classes.map((c) => new THREE.Color(CLASS_COLORS[c] || CLASS_COLORS.unknown));
    let k = 0;
    const keep = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const x = raw[3 * i], y = raw[3 * i + 1], z = raw[3 * i + 2];
      if (!Number.isFinite(x)) continue;
      pos[3 * k] = -(x - cx); pos[3 * k + 1] = -(y - cy); pos[3 * k + 2] = -(z - cz);
      const c = palette[cls[i]] || palette[palette.length - 1];
      col[3 * k] = c.r; col[3 * k + 1] = c.g; col[3 * k + 2] = c.b;
      keep[k] = i; k++;
    }
    this.meta = meta;
    this.n = n;
    this.cls = cls;
    // full-index position lookup (NaN for neurons without a position)
    this.posByIdx = new Float32Array(3 * n).fill(NaN);
    for (let j = 0; j < k; j++) {
      const i = keep[j];
      this.posByIdx[3 * i] = pos[3 * j]; this.posByIdx[3 * i + 1] = pos[3 * j + 1]; this.posByIdx[3 * i + 2] = pos[3 * j + 2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, 3 * k), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, 3 * k), 3));
    this.base = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 1.2, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.4, depthWrite: false,
    }));
    this.scene.add(this.base);
    this.applyTheme();
    this.resize();
    return meta;
  }

  setRun(run) {
    if (this.active) { this.scene.remove(this.active); this.active.geometry.dispose(); this.active = null; }
    const b = run.brain || {};
    const role = new Map();
    for (const i of b.readout_idx || []) role.set(i, 'readout');
    for (const i of b.silenced_idx || []) role.set(i, 'silenced');
    for (const i of b.stimulated_idx || []) role.set(i, 'stimulated');
    const idx = b.active_idx || [], rates = b.active_rate_hz || [];
    const entries = [];
    for (let j = 0; j < idx.length; j++) entries.push([idx[j], rates[j]]);
    // stimulated / silenced neurons are always drawn (silenced ones have rate 0)
    const seen = new Set(idx);
    for (const [i, r] of role) if (!seen.has(i) && r !== 'readout') entries.push([i, 0]);
    const m = entries.length;
    const pos = new Float32Array(3 * m), col = new Float32Array(3 * m), size = new Float32Array(m);
    const c = new THREE.Color();
    let k = 0;
    for (const [i, rate] of entries) {
      const x = this.posByIdx[3 * i];
      if (!Number.isFinite(x)) continue;
      pos[3 * k] = x; pos[3 * k + 1] = this.posByIdx[3 * i + 1]; pos[3 * k + 2] = this.posByIdx[3 * i + 2];
      const r = role.get(i) || 'downstream';
      if (r === 'stimulated') c.set(ROLE_COLORS.stimulated);
      else if (r === 'readout') c.set(ROLE_COLORS.readout);
      else if (r === 'silenced') c.set(ROLE_COLORS.silenced);
      else heat(rate, c);
      col[3 * k] = c.r; col[3 * k + 1] = c.g; col[3 * k + 2] = c.b;
      const boost = r === 'readout' ? 22 : r === 'stimulated' ? 14 : 0;
      size[k] = 9 + 16 * Math.min(1, rate / 150) + boost;
      k++;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, 3 * k), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, 3 * k), 3));
    geo.setAttribute('size', new THREE.BufferAttribute(size.subarray(0, k), 1));
    this.active = new THREE.Points(geo, this.glowMat);
    this.active.renderOrder = 2;
    this.scene.add(this.active);
    return { drawn: k };
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
