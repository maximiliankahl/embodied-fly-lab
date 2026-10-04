// Story mode for the demo video (open index.html?story). Plays RECORDED runs in a fixed order with captions.
// Nothing here is animated by hand: every body motion and every lit neuron comes from an exported simulation run.

const STEPS = [
  { run: 'story_sugar_feeding', layout: 'brain', ms: 9000, speed: 0.25, verdict: null,
    title: '138,639 neurons',
    sub: 'The complete wiring diagram of a fruit-fly brain (FlyWire v783, open data), running as a spiking whole-brain model (after Shiu et al. 2024).' },
  { run: 'story_miswired_dna02l', layout: 'both', ms: 7000, speed: 0.25, ghost: 0.28, paused: true, verdict: null,
    title: 'A body around the brain',
    sub: 'A physics fly (NeuroMechFly / FlyBody in MuJoCo), connected to the brain through descending neurons.' },
  { run: 'story_miswired_dna02l', layout: 'both', ms: 9000, speed: 0.35, verdict: null,
    title: 'Test 1: steering neurons → legs',
    sub: 'The lab activates DNa02 on the left side. Adapter v0 was wired with the wrong sign on purpose.' },
  { run: 'story_miswired_dna02l', layout: 'both', ms: 7000, speed: 0.35, verdict: 'bad',
    title: 'Movement verifier: incorrect',
    sub: 'Expected a LEFT turn (Rayshubskiy et al. 2025, eLife). The body turned right (−328°). Adapter v0 is rejected and reverted.' },
  { run: 'dna02l_turn_left', layout: 'both', ms: 9000, speed: 0.35, verdict: 'good',
    title: 'Reverted → correct',
    sub: 'Frozen bridge restored: same neurons, the fly now turns left. Kinematics and the published result agree.' },
  { run: 'gf_dng02_climb', layout: 'both', ms: 11000, speed: 0.25, verdict: 'good',
    title: 'Test 2: escape neurons → wings',
    sub: 'Giant fiber + DNg02 fire in the brain model → takeoff and climb (FlyBody, quasi-steady aerodynamics). Verified: takeoff.' },
  { run: 'story_sugar_feeding', layout: 'brain', ms: 9000, speed: 0.25, verdict: 'good',
    title: 'Sugar → feeding neuron',
    sub: 'Sugar taste neurons drive the feeding motor neuron MN9 (60–80 Hz), as published by Shiu et al. 2024.' },
  { run: 'gf_dng02_climb', layout: 'both', ms: 9000, speed: 0.25, verdict: null,
    title: 'Embodied Fly Lab',
    sub: 'Our agents solve scientific problems by testing, validating through simulation and comparing with published research.' },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function setGhost(fly, opacity) {
  fly.scene.traverse((o) => {
    if (!o.isMesh || o === fly.ground || !o.material || o.material.isLineBasicMaterial) return;
    const m = o.material;
    if (m.userData.baseOpacity === undefined) {
      m.userData.baseOpacity = m.opacity;
      m.userData.baseTransparent = m.transparent;
      m.userData.baseDepthWrite = m.depthWrite;
    }
    if (opacity === null) {
      m.opacity = m.userData.baseOpacity; m.transparent = m.userData.baseTransparent; m.depthWrite = m.userData.baseDepthWrite;
    } else {
      m.opacity = Math.min(m.userData.baseOpacity, opacity); m.transparent = true; m.depthWrite = false;
    }
    m.needsUpdate = true;
  });
}

export async function runStory({ selectRun, state, fly, brain }) {
  document.body.classList.add('story');
  const cap = document.createElement('div');
  cap.id = 'story-caption';
  cap.innerHTML = '<div class="story-badge">Recorded simulation replay · not live</div><h2></h2><p></p>';
  document.body.appendChild(cap);
  const h = cap.querySelector('h2'), p = cap.querySelector('p');
  if (brain.controls) { brain.controls.autoRotate = true; brain.controls.autoRotateSpeed = 0.8; }
  const loop = new URLSearchParams(location.search).has('loop');
  do {
    for (const s of STEPS) {
      document.body.classList.toggle('story-brain', s.layout === 'brain');
      cap.className = s.verdict ? `verdict-${s.verdict}` : '';
      cap.classList.remove('show');
      if (!state.run || state.run.id !== s.run) await selectRun(s.run);
      state.t = 0;
      state.speed = s.speed;
      state.playing = !s.paused;
      setGhost(fly, s.ghost ?? null);
      window.dispatchEvent(new Event('resize'));
      h.textContent = s.title;
      p.textContent = s.sub;
      await sleep(250);
      cap.classList.add('show');
      await sleep(s.ms);
    }
  } while (loop);
  cap.classList.remove('show');
}
