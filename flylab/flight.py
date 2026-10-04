"""Physics-based flight for the Embodied Fly Lab: FlyBody (Vaxenburg et al. 2025) in MuJoCo.

Contract (docs/CONTRACTS.md, Phase 3):
    simulate_flight(command, duration_s=1.0, render_path=None, seed=0, *, record_poses=False, camera="chase") -> dict
    classify_flight(metrics) -> "no_takeoff" | "takeoff_fall" | "hover" | "climb" | "forward_flight"
                                | "flight_turn_left" | "flight_turn_right"
    command = {"takeoff": 0..1, "thrust": 0..1, "yaw": -1..1 (neg = left), "pitch": -1..1 (neg = backward)}

BODY
----
The original FlyBody MJCF (``fruitfly.xml`` as shipped inside flygym 2.1, cm-g-s units) with its
full-size meshes (lazily downloaded by flygym into ``data/raw/flygym_assets``, gitignored).
FlyBody: Vaxenburg et al. 2025, Nature, doi:10.1038/s41586-025-09029-4 (github.com/TuragaLab/flybody).
The fly is a free MuJoCo rigid-body tree (thorax free joint, legs, head, abdomen, 2 wings x 3 DOF)
standing on a ground plane; gravity -981 cm/s^2; all motion comes from integrating forces.

WHAT IS PHYSICS vs. HAND-DESIGNED (G1/G4)  -> also returned as result["model_notes"]
--------------------------------------------------------------------------------
Approach B of the plan (wingbeat-cycle-averaged quasi-steady aerodynamics). Approach A (MuJoCo's
own fluid model on the flapping wing ellipsoids) was NOT attempted to completion in the time box;
MuJoCo's fluid model is switched off (density = viscosity = 0) so forces are not double-counted.

Physics (MuJoCo integrates): rigid-body dynamics of the whole fly, gravity, ground contact, the
jump impulse, the cycle-averaged aerodynamic forces below, wing-joint servo dynamics.

Quasi-steady aerodynamics (per wing w, per control step; formula after the blade-element /
quasi-steady framework, coefficients from Dickinson, Lehmann & Sane 1999, Science,
doi:10.1126/science.284.5422.1954):
    mean lift   L_w = 0.5 * rho * C_L(alpha) * S * <U_2^2>,   <U_2^2> = (pi * f * Phi_w * r_2)^2 / 2
    mean drag   D_w = 0.5 * rho * C_D(alpha) * S * <U_2^2>    (cancels over a symmetric stroke)
    C_L(a) = 0.225 + 1.58 sin(2.13 a - 7.2 deg),  C_D(a) = 1.92 - 1.55 cos(2.04 a - 9.82 deg), a = 45 deg
    S, R from the FlyBody wing-fluid ellipsoid (semi-axes 0.0551 x 0.114 cm): S = pi*a*b, R = 2*0.114 cm,
    r_2 = 0.559 R (second-moment radius of an elliptical wing hinged at its tip), rho = 0.00128 g/cm^3
    (FlyBody XML value), f = 200 Hz (hand-chosen typical Drosophila wingbeat frequency).
    Lift acts along the stroke-plane normal (FlyBody site ``hover_up_dir``), applied at a centre of
    pressure = wing hinge + 0.6 r_2 laterally + r_2 sin(phi_0) fore/aft (phi_0 = mean stroke angle).
    Flapping counter-force damping (Hedrick et al. 2009 framework): F = -c_t v, tau = -c_t r_2^2 omega,
    c_t = 2 rho C_D S <|U_2|>, <|U_2|> = 2 Phi f r_2.  (Isotropic: a simplification.)

Hand-designed (NOT from the connectome; stands in for halteres, visual feedback and the
flight-motor / steering-muscle system of the ventral nerve cord):
  * Trim: baseline stroke amplitude Phi_hover solved so that quasi-steady lift = model weight.
  * thrust (0..1) -> stroke amplitude  Phi = Phi_hover * (1 + THRUST_GAIN * thrust), clipped to PHI_MAX.
  * Attitude stabiliser (PD on the stroke-plane normal, ~10 Hz bandwidth) -> left/right amplitude
    asymmetry (roll), mean stroke angle phi_0 (pitch), left/right drag asymmetry (yaw); outputs are
    clipped to physical limits and the forces are re-computed from the clipped wing parameters.
  * pitch (-1..1) -> target forward tilt of the stroke plane (TILT_MAX); yaw (-1..1) -> target yaw rate.
  * takeoff >= 0.5 -> giant-fiber-like escape sequence at T_TRIGGER: leg-extension jump impulse
    (force m*A_JUMP for T_JUMP at the centre of mass, elevation JUMP_ELEV, T2 leg extensors driven to
    full extension), wing beat starts T_WING_DELAY later. Impulse size is hand-chosen, not measured.
  * Wing kinematics (joint targets for the wing servos) are a sinusoidal stroke with rotation flips;
    they are for realism/visualisation only - the aerodynamic forces come from the formula above.
Not modelled: unsteady effects (rotational lift, wake capture, added mass), halteres as sensors,
sensory feedback to the brain (open loop), leg retraction in flight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform.startswith("win"):
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "spikes" / "flight" / "out"
ASSET_DIR = ROOT / "assets" / "flight"
MESH_CACHE = ROOT / "data" / "raw" / "flygym_assets"  # gitignored
FLYBODY_MESH_SET = "flybody_fullsize_meshes_20260623a"
MODEL_CACHE = OUT_DIR / "flybody_flight_model.mjb"

LABELS = ("no_takeoff", "takeoff_fall", "hover", "climb", "forward_flight", "flight_turn_left", "flight_turn_right")
CMD_KEYS = ("takeoff", "thrust", "yaw", "pitch")
CM_TO_MM = 10.0

# ---------------------------------------------------------------- aerodynamic constants (see docstring)
RHO = 0.00128            # g/cm^3, FlyBody XML option density
WING_A, WING_B = 0.0551, 0.114  # cm, FlyBody wing-fluid ellipsoid semi-axes (chord/2, span/2)
WING_S = math.pi * WING_A * WING_B   # cm^2 (~2 mm^2)
WING_R = 2 * WING_B                  # cm
R2 = 0.559 * WING_R                  # cm
FREQ_HZ = 200.0
ALPHA_DEG = 45.0
C_L = 0.225 + 1.58 * math.sin(math.radians(2.13 * ALPHA_DEG - 7.20))
C_D = 1.92 - 1.55 * math.cos(math.radians(2.04 * ALPHA_DEG - 9.82))
K_LIFT = 0.5 * RHO * C_L * WING_S * (math.pi * FREQ_HZ * R2) ** 2 / 2.0  # L = K_LIFT * Phi^2  (dyn/rad^2)
K_DRAG = K_LIFT * C_D / C_L
CP_LAT = 0.6 * R2        # lateral centre-of-pressure offset from the hinge (cm)

# ---------------------------------------------------------------- hand-designed controller constants
PHI_MAX = math.radians(170.0)   # max peak-to-peak stroke amplitude (joint range of the FlyBody wing-yaw DOF ~172 deg)
THRUST_GAIN = 0.12              # thrust=1 -> +12 % amplitude (~ +25 % lift)
TILT_MAX = math.radians(25.0)   # |pitch|=1 -> stroke plane tilted 25 deg (forward/backward)
PHI0_MAX = math.radians(40.0)   # mean stroke angle limit (pitch torque)
ETA_MAX = 0.35                  # max drag asymmetry (fraction of mean drag) for yaw torque
YAW_RATE_MAX = math.radians(400.0)  # |yaw|=1 -> target yaw rate 400 deg/s
ATT_BW_HZ, ATT_ZETA = 10.0, 0.9
YAW_BW_HZ = 4.0                 # heading-hold bandwidth (optomotor-like, hand-designed)
V_FWD_MAX = 30.0                # cm/s: |pitch|=1 -> target horizontal speed 300 mm/s along the heading
K_VEL = 1.0 / 0.15              # 1/s: horizontal velocity regulation (visual ground-speed control stand-in)
K_VEL_I = 2.0 * K_VEL           # 1/s^2: integral term (removes steady tilt offsets, e.g. from wing-inertia torques)
CONTROL_DT = 0.001              # s, controller period (physics dt 0.1 ms)
T_TRIGGER = 0.05                # s, escape trigger time after the start of the run
T_JUMP = 0.006                  # s, leg-extension impulse duration
A_JUMP = 6000.0                 # cm/s^2 (~6 g) during T_JUMP -> ~0.35 m/s take-off speed (hand-chosen)
JUMP_ELEV = math.radians(70.0)  # elevation of the jump direction (forward-up)
T_WING_DELAY = 0.004            # s, wing beat starts after the trigger
WING_RAMP = 0.010               # s, amplitude ramp-up
SAMPLE_DT = 0.01                # trajectory sampling
_DEBUG: list | None = None      # set to [] to collect controller diagnostics

# ---------------------------------------------------------------- classification thresholds
AIRBORNE_MM = 0.5            # thorax this much above its standing height = airborne
SETTLE_AFTER_TAKEOFF_S = 0.3  # flight-phase metrics are computed after this transient
FALL_FINAL_MM = 0.5          # airborne once but final height below this = fell back / landed
TURN_MIN_DEG = 45.0
TURN_RATE_MIN = 60.0         # deg/s over the flight phase
FWD_MIN_SPEED = 40.0         # mm/s horizontal along the initial heading, flight phase
CLIMB_MIN_RATE = 15.0        # mm/s mean vertical speed, flight phase
TUMBLE_COS = 0.0             # stroke normal pointing below horizontal = tumbling


def _num(v: Any, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def sanitize_command(command: dict | None) -> dict:
    c = command or {}
    return {"takeoff": _num(c.get("takeoff", 0.0), 0.0, 1.0), "thrust": _num(c.get("thrust", 0.0), 0.0, 1.0),
            "yaw": _num(c.get("yaw", 0.0), -1.0, 1.0), "pitch": _num(c.get("pitch", 0.0), -1.0, 1.0)}


# ---------------------------------------------------------------- model
def _mesh_dir() -> Path:
    os.environ.setdefault("FLYGYM_ASSET_CACHE_DIR", str(MESH_CACHE))
    from flygym.utils.assets_lazy_loading import lazy_load_asset_dir
    return Path(lazy_load_asset_dir(FLYBODY_MESH_SET))


def _flybody_xml() -> Path:
    import flygym
    return Path(flygym.__file__).resolve().parent / "assets" / "model" / "flybody" / "fruitfly.xml"


def build_model(use_cache: bool = True):
    """FlyBody MJCF + ground + chase camera, fluid model off, implicitfast integrator. Returns MjModel."""
    import mujoco as mj

    if use_cache and MODEL_CACHE.exists():
        try:
            return mj.MjModel.from_binary_path(str(MODEL_CACHE))
        except Exception:  # noqa: BLE001
            pass
    from flygym.flybody.parse_flybody import translate_mesh_name

    mesh_dir = _mesh_dir()
    spec = mj.MjSpec.from_file(str(_flybody_xml()))
    spec.meshdir = str(mesh_dir)
    for mesh in spec.meshes:
        stem = Path(mesh.file).stem
        new = f"{translate_mesh_name(stem)}.obj"
        if not (mesh_dir / new).exists() and (mesh_dir / f"{stem}.obj").exists():
            new = f"{stem}.obj"
        mesh.file = new
    spec.option.density = 0.0     # Approach B: MuJoCo fluid model OFF, quasi-steady forces applied instead
    spec.option.viscosity = 0.0
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    # ground + light + checker texture
    tex = spec.add_texture(name="grid", type=mj.mjtTexture.mjTEXTURE_2D, builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.86, 0.88, 0.9], rgb2=[0.72, 0.75, 0.8], width=300, height=300)
    mat = spec.add_material(name="grid")
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [8, 8]
    mat.reflectance = 0.05
    spec.add_texture(name="sky", type=mj.mjtTexture.mjTEXTURE_SKYBOX, builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.7, 0.9], rgb2=[0.95, 0.97, 1.0], width=256, height=256)
    floor = spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1], material="grid")
    floor.contype = 1
    floor.conaffinity = 1
    spec.worldbody.add_light(name="sun", pos=[0, 0, 6], dir=[0, 0, -1], type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.6, 0.6, 0.6], castshadow=0)
    thorax = spec.body("thorax")
    # chase camera: follows the centre of mass, world-fixed orientation, behind-left-above the start heading
    cam_pos = np.array([-1.1, 0.75, 0.55])
    fwd = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    thorax.add_camera(name="chase", mode=mj.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=cam_pos.tolist(),
                      xyaxes=list(right) + list(up), fovy=45)
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    spec.visual.quality.shadowsize = 0
    m = spec.compile()
    _tune_wing_servos(m)
    try:
        MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        mj.mj_saveModel(m, str(MODEL_CACHE), None)
    except Exception:  # noqa: BLE001
        pass
    return m


def _tune_wing_servos(m) -> None:
    """Turn the 6 FlyBody wing torque actuators into stiff position servos (natural freq ~800 Hz)."""
    import mujoco as mj

    d = mj.MjData(m)
    mj.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mj.mj_fullM(m, M, d.qM)
    for side in ("left", "right"):
        for ax in ("yaw", "roll", "pitch"):
            aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"wing_{ax}_{side}")
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"wing_{ax}_{side}")
            dof = m.jnt_dofadr[jid]
            inertia = float(M[dof, dof])
            wn = 2 * math.pi * 800.0
            kp = inertia * wn ** 2
            kv = 2 * 1.0 * math.sqrt(kp * inertia)
            m.actuator_gaintype[aid] = mj.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[aid] = mj.mjtBias.mjBIAS_AFFINE
            m.actuator_gainprm[aid, :] = 0
            m.actuator_biasprm[aid, :] = 0
            m.actuator_gainprm[aid, 0] = kp
            m.actuator_biasprm[aid, 1] = -kp
            m.actuator_biasprm[aid, 2] = -kv
            m.actuator_ctrlrange[aid] = m.jnt_range[jid]
            m.actuator_ctrllimited[aid] = 1
            m.actuator_forcelimited[aid] = 0


class _Ids:
    def __init__(self, m):
        import mujoco as mj
        n2 = lambda t, n: mj.mj_name2id(m, t, n)  # noqa: E731
        B, S, A, J = mj.mjtObj.mjOBJ_BODY, mj.mjtObj.mjOBJ_SITE, mj.mjtObj.mjOBJ_ACTUATOR, mj.mjtObj.mjOBJ_JOINT
        self.thorax = n2(B, "thorax")
        self.wing = {"left": n2(B, "wing_left"), "right": n2(B, "wing_right")}
        self.hover_site = n2(S, "hover_up_dir")
        self.wing_act = {(s, a): n2(A, f"wing_{a}_{s}") for s in ("left", "right") for a in ("yaw", "roll", "pitch")}
        self.wing_jnt = {(s, a): n2(J, f"wing_{a}_{s}") for s in ("left", "right") for a in ("yaw", "roll", "pitch")}
        names = [mj.mj_id2name(m, A, i) or "" for i in range(m.nu)]
        self.adhesion = [i for i, n in enumerate(names) if n.startswith("adhere")]
        # mesothoracic (T2) leg extensors used for the jump (TTM-like leg extension)
        self.jump_act = [i for i, n in enumerate(names) if "T2" in n and (n.startswith("femur") or n.startswith("tibia"))]
        self.names = names


def _composite_inertia_body(m, d, root: int) -> np.ndarray:
    """3x3 inertia of the whole fly about its CoM, expressed in the thorax frame (computed once)."""
    com = d.subtree_com[root].copy()
    Rt = d.xmat[root].reshape(3, 3)
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        mass = m.body_mass[b]
        if mass <= 0:
            continue
        Rb = d.ximat[b].reshape(3, 3)
        Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
        r = d.xipos[b] - com
        I += Ib + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return Rt.T @ I @ Rt


def _euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """roll, pitch (+ = nose up), yaw (heading, CCW +) of the thorax frame (+x anterior, +y left, +z dorsal)."""
    x = R[:, 0]
    yaw = math.atan2(x[1], x[0])
    pitch = math.atan2(x[2], math.hypot(x[0], x[1]))
    # roll: angle of the body y axis below the horizontal plane around x
    y = R[:, 1]
    roll = math.atan2(-y[2], R[2, 2])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def phi_hover(mass_g: float, g: float = 981.0) -> float:
    """Peak-to-peak stroke amplitude (rad) at which the quasi-steady lift of 2 wings equals the weight."""
    return math.sqrt(mass_g * g / (2 * K_LIFT))


def model_notes(mass_g: float | None = None) -> dict:
    ph = math.degrees(phi_hover(mass_g)) if mass_g else None
    return {
        "body": "FlyBody (Vaxenburg et al. 2025, Nature, doi:10.1038/s41586-025-09029-4), original fruitfly.xml from flygym 2.1 "
                "with full-size meshes; MuJoCo rigid-body physics, ground contact, gravity.",
        "approach": "B: wingbeat-cycle-averaged quasi-steady aerodynamics applied as forces/torques (mj_applyFT); "
                    "MuJoCo fluid model off. Approach A (MuJoCo fluid forces on flapping wings) not completed in the time box.",
        "physics": ["rigid-body dynamics of the whole fly (MuJoCo, dt 0.1 ms)", "gravity and ground contact",
                    "quasi-steady mean lift L = 0.5 rho C_L S <U2^2> per wing at a centre of pressure (roll/pitch torques "
                    "emerge from left/right lift difference and fore/aft centre-of-pressure shift)",
                    "flapping counter-force damping (translational and rotational)", "jump impulse integration",
                    "wing joint servo dynamics (wings flap at 200 Hz; visual realism, no fluid force from them)"],
        "aero_parameters": {"rho_g_cm3": RHO, "wing_area_mm2": round(WING_S * 100, 3), "wing_length_mm": round(WING_R * 10, 2),
                            "r2_mm": round(R2 * 10, 3), "freq_hz": FREQ_HZ, "alpha_deg": ALPHA_DEG, "C_L": round(C_L, 3),
                            "C_D": round(C_D, 3), "C_L_C_D_source": "Dickinson, Lehmann & Sane 1999, Science, doi:10.1126/science.284.5422.1954",
                            "phi_hover_deg": round(ph, 1) if ph else None, "mass_mg": round(mass_g * 1000, 3) if mass_g else None},
        "hand_designed": ["trim: baseline amplitude solved for lift = weight",
                          f"thrust -> amplitude: Phi = Phi_hover * (1 + {THRUST_GAIN} * thrust), max {math.degrees(PHI_MAX):.0f} deg",
                          f"attitude stabiliser (PD, {ATT_BW_HZ} Hz) acting through amplitude asymmetry (roll), mean stroke angle (pitch), "
                          "drag asymmetry (yaw); stands in for halteres + steering muscles (not connectome)",
                          f"pitch -> stroke-plane tilt up to {math.degrees(TILT_MAX):.0f} deg; yaw -> target yaw rate up to "
                          f"{math.degrees(YAW_RATE_MAX):.0f} deg/s",
                          f"takeoff >= 0.5 -> escape sequence: jump impulse {A_JUMP / 981:.1f} g for {T_JUMP * 1000:.0f} ms "
                          f"(elevation {math.degrees(JUMP_ELEV):.0f} deg) + T2 leg extension, wings start after "
                          f"{T_WING_DELAY * 1000:.0f} ms (all values hand-chosen)",
                          "wing joint kinematics (sinusoidal stroke + rotation flips) are for visualisation"],
        "connectome_decides": "only the command (takeoff trigger, thrust, yaw, pitch) via flylab.bridge.rates_to_flight_command; "
                              "a body that can fly is not evidence of connectome-controlled flight (G4).",
        "not_modelled": ["unsteady aerodynamics (rotational lift, wake capture, added mass)", "haltere sensing",
                         "sensory feedback to the brain (open loop)", "leg retraction in flight", "landing"],
    }


# ---------------------------------------------------------------- simulation
def simulate_flight(command: dict, duration_s: float = 1.0, render_path: str | None = None, seed: int = 0, *,
                    record_poses: bool = False, camera: str = "chase", camera_res: tuple[int, int] = (360, 480),
                    pose_fps: float | None = None) -> dict:
    """Simulate FlyBody standing on the ground, then (if commanded) escape takeoff and flight.

    Returns the contract dict: trajectory [[t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg], ...]
    (x, y relative to the start, z = thorax height gain above the initial standing height, yaw unwrapped,
    CCW + = left), airborne, takeoff_time_s, flight_time_s, max_height_mm, net_displacement_mm,
    heading_change_deg, mean_speed_mm_s, behavior, video, poses (if record_poses), runtime_s, model_notes,
    plus wing_amplitude_deg (mean commanded stroke amplitude during flight) and diagnostics.
    """
    import mujoco as mj

    t_start = time.perf_counter()
    cmd = sanitize_command(command)
    warnings: list[str] = []
    unknown = sorted(set((command or {}).keys()) - set(CMD_KEYS))
    if unknown:
        warnings.append(f"unknown command keys ignored: {unknown}")
    duration_s = min(5.0, max(0.1, _num(duration_s, 0.1, 5.0) or 1.0))
    rng = np.random.default_rng(seed)

    m = build_model()
    d = mj.MjData(m)
    ids = _Ids(m)
    mass = float(m.body_subtreemass[ids.thorax])
    g = -float(m.opt.gravity[2])
    weight = mass * g
    ph_hover = phi_hover(mass, g)
    dt = float(m.opt.timestep)
    ctrl_every = max(1, int(round(CONTROL_DT / dt)))

    # initial state: standing, wings folded (servo targets = joint spring rest angles)
    mj.mj_resetData(m, d)
    d.qpos[2] = 0.25  # cm, drop onto the ground
    if seed:
        d.qpos[2] += float(rng.uniform(0, 0.01))
    wing_rest = {k: float(m.qpos_spring[m.jnt_qposadr[j]]) for k, j in ids.wing_jnt.items()}
    for k, a in ids.wing_act.items():
        d.ctrl[a] = wing_rest[k]
    mj.mj_forward(m, d)
    I_body = _composite_inertia_body(m, d, ids.thorax)
    kp_att = (2 * math.pi * ATT_BW_HZ) ** 2
    kd_att = 2 * ATT_ZETA * 2 * math.pi * ATT_BW_HZ
    kp_yaw = (2 * math.pi * YAW_BW_HZ) ** 2
    kd_yaw = 2 * ATT_ZETA * 2 * math.pi * YAW_BW_HZ
    psi_des = None
    v_int = np.zeros(3)
    omega_buf = np.zeros((max(1, int(round(1.0 / (FREQ_HZ * float(m.opt.timestep))))), 3))
    # settle on the ground (not counted)
    for _ in range(int(0.15 / dt)):
        mj.mj_step(m, d)
    d.time = 0.0
    p0 = d.subtree_com[ids.thorax].copy()
    z0 = float(d.xpos[ids.thorax][2])
    xy0 = d.xpos[ids.thorax][:2].copy()
    _, _, yaw0 = _euler_deg(d.xmat[ids.thorax].reshape(3, 3))

    renderer = None
    frames: list[np.ndarray] = []
    render_error = None
    # stroboscopic rendering: one frame every 2.1 wingbeats -> the 200 Hz wingbeat appears slowed down
    frame_dt = 2.1 / FREQ_HZ
    if render_path:
        try:
            renderer = mj.Renderer(m, height=camera_res[0], width=camera_res[1])
        except Exception as exc:  # noqa: BLE001
            render_error = f"{type(exc).__name__}: {exc}"
    recorder = None
    if record_poses:
        from flylab.poses import PoseRecorder
        recorder = PoseRecorder(m, fps=pose_fps or FREQ_HZ / 3.1, units="cm")

    takeoff = cmd["takeoff"] >= 0.5
    n_steps = int(round(duration_s / dt))
    sample_every = max(1, int(round(SAMPLE_DT / dt)))
    traj: list[list[float]] = []
    yaw_unwrapped, prev_yaw = yaw0, yaw0
    stroke = {"left": 0.0, "right": 0.0}
    phi0 = 0.0
    eta = 0.0
    wings_on = False
    applied = np.zeros(m.nv)
    amp_log: list[float] = []
    min_up_cos = 1.0
    t_next_frame = 0.0
    sat_count = 0
    ctrl_count = 0

    def _record_traj(t: float) -> None:
        nonlocal yaw_unwrapped, prev_yaw
        R = d.xmat[ids.thorax].reshape(3, 3)
        roll, pitch, yaw = _euler_deg(R)
        dyaw = (yaw - prev_yaw + 180.0) % 360.0 - 180.0
        yaw_unwrapped += dyaw
        prev_yaw = yaw
        pos = d.xpos[ids.thorax]
        traj.append([round(t, 4), round(float(pos[0] - xy0[0]) * CM_TO_MM, 3), round(float(pos[1] - xy0[1]) * CM_TO_MM, 3),
                     round(float(pos[2] - z0) * CM_TO_MM, 3), round(roll, 2), round(pitch, 2), round(yaw_unwrapped, 2)])

    _record_traj(0.0)
    t_loop = time.perf_counter()
    for i in range(n_steps):
        t = i * dt
        if i % ctrl_every == 0:
            applied[:] = 0.0
            mj.mj_subtreeVel(m, d)
            com = d.subtree_com[ids.thorax].copy()
            vcom = d.subtree_linvel[ids.thorax].copy()
            # thorax angular velocity averaged over the last wingbeat (the 200 Hz wing-inertia oscillation
            # would otherwise alias into the 1 kHz controller; flies' halteres also integrate over the cycle)
            omega = omega_buf.mean(axis=0) if i >= len(omega_buf) else d.cvel[ids.thorax, :3].copy()
            Rt = d.xmat[ids.thorax].reshape(3, 3)
            n = d.site_xmat[ids.hover_site].reshape(3, 3)[:, 2].copy()  # stroke-plane normal (world)
            min_up_cos = min(min_up_cos, float(n[2])) if wings_on else min_up_cos
            # ---- escape: jump impulse + T2 leg extension
            if takeoff and T_TRIGGER <= t < T_TRIGGER + T_JUMP:
                h = Rt[:, 0].copy(); h[2] = 0.0
                h = h / (np.linalg.norm(h) + 1e-12)
                jdir = math.cos(JUMP_ELEV) * h + math.sin(JUMP_ELEV) * np.array([0, 0, 1.0])
                mj.mj_applyFT(m, d, mass * A_JUMP * jdir, np.zeros(3), com, ids.thorax, applied)
                for a in ids.jump_act:
                    d.ctrl[a] = m.actuator_ctrlrange[a, 0] if "femur" in ids.names[a] else m.actuator_ctrlrange[a, 1]
            elif takeoff and t >= T_TRIGGER + T_JUMP:
                for a in ids.jump_act:
                    d.ctrl[a] = 0.0
            wings_on = takeoff and t >= T_TRIGGER + T_WING_DELAY
            if wings_on:
                ctrl_count += 1
                ramp = min(1.0, (t - T_TRIGGER - T_WING_DELAY) / WING_RAMP)
                e_fwd = Rt[:, 0] - np.dot(Rt[:, 0], n) * n
                e_fwd /= np.linalg.norm(e_fwd) + 1e-12
                e_lat = np.cross(n, e_fwd)  # fly's left
                # target stroke-plane normal: world up tilted by pitch command toward the current heading
                hh = e_fwd.copy(); hh[2] = 0.0
                hh /= np.linalg.norm(hh) + 1e-12
                # horizontal velocity regulation: tilt the stroke plane to accelerate toward v_des (pitch command)
                v_h = vcom.copy(); v_h[2] = 0.0
                v_des = cmd["pitch"] * V_FWD_MAX * hh
                c_t_now = 2 * RHO * C_D * WING_S * (2 * ph_hover * FREQ_HZ * R2)
                v_int += (v_des - v_h) * ctrl_every * dt
                a_h = K_VEL * (v_des - v_h) + K_VEL_I * v_int + (c_t_now / mass) * v_des
                a_norm = float(np.linalg.norm(a_h))
                a_lim = g * math.tan(TILT_MAX)
                if a_norm > a_lim:
                    a_h *= a_lim / a_norm
                dvec = np.array([a_h[0], a_h[1], g]); dvec /= np.linalg.norm(dvec)
                tilt = math.acos(max(-1.0, min(1.0, float(dvec[2]))))
                # heading hold: target heading integrates the commanded yaw rate (yaw<0 = left = CCW +)
                psi = math.atan2(hh[1], hh[0])
                if psi_des is None:
                    psi_des = psi
                psi_des += -cmd["yaw"] * YAW_RATE_MAX * CONTROL_DT * ctrl_every * dt / CONTROL_DT
                psi_err = (psi_des - psi + math.pi) % (2 * math.pi) - math.pi
                w_n_des = -cmd["yaw"] * YAW_RATE_MAX
                w_n = float(np.dot(omega, n))
                omega_perp = omega - w_n * n
                alpha = kp_att * np.cross(n, dvec) - kd_att * omega_perp
                alpha += (kp_yaw * psi_err + kd_yaw * (w_n_des - w_n)) * n
                I_w = Rt @ I_body @ Rt.T
                tau = I_w @ alpha
                # total lift: trim x thrust (amplitude), tilt compensation
                phi_mean = ph_hover * (1.0 + THRUST_GAIN * cmd["thrust"]) / math.sqrt(max(0.6, math.cos(tilt)))
                phi_mean = min(PHI_MAX, phi_mean) * ramp
                L_tot = 2 * K_LIFT * phi_mean ** 2
                # roll (about e_fwd): y_cp (L_L - L_R)
                hinge = {s: d.xpos[ids.wing[s]].copy() for s in ("left", "right")}
                y_cp = 0.5 * float(np.dot(hinge["left"] - hinge["right"], e_lat)) + CP_LAT
                tau_roll = float(np.dot(tau, e_fwd))
                dL = tau_roll / max(y_cp, 1e-4)
                L_w = {"left": 0.5 * (L_tot + dL), "right": 0.5 * (L_tot - dL)}
                for s in ("left", "right"):
                    ph = math.sqrt(max(0.0, L_w[s]) / K_LIFT)
                    if ph > PHI_MAX:
                        sat_count += 1
                    stroke[s] = min(PHI_MAX, ph)
                L_w = {s: K_LIFT * stroke[s] ** 2 for s in stroke}
                L_sum = L_w["left"] + L_w["right"]
                # pitch (about e_lat): centre-of-pressure shift x_s = R2 sin(phi0)
                tau_pitch_des = float(np.dot(tau, e_lat))
                base_pts = {s: hinge[s] + (CP_LAT if s == "left" else -CP_LAT) * e_lat for s in hinge}
                tau0 = sum(np.cross(base_pts[s] - com, L_w[s] * n) for s in hinge)
                tau0_lat = float(np.dot(tau0, e_lat))
                x_s = (tau0_lat - tau_pitch_des) / max(L_sum, 1e-9) if L_sum > 1e-9 else 0.0
                phi0 = math.asin(max(-math.sin(PHI0_MAX), min(math.sin(PHI0_MAX), x_s / R2)))
                x_s = R2 * math.sin(phi0)
                # yaw (about n): drag asymmetry +-F on the two wings
                D_mean = K_DRAG * (0.5 * (stroke["left"] + stroke["right"])) ** 2
                tau_yaw_des = float(np.dot(tau, n))
                F = tau_yaw_des / (2 * max(y_cp, 1e-4))
                Fmax = ETA_MAX * D_mean
                F = max(-Fmax, min(Fmax, F))
                eta = F / D_mean if D_mean > 0 else 0.0
                for s in hinge:
                    pt = base_pts[s] + x_s * e_fwd
                    f_vec = L_w[s] * n + (F if s == "right" else -F) * e_fwd
                    mj.mj_applyFT(m, d, f_vec, np.zeros(3), pt, ids.thorax, applied)
                # flapping counter-force damping
                U_abs = 2 * phi_mean * FREQ_HZ * R2
                c_t = 2 * RHO * C_D * WING_S * U_abs
                mj.mj_applyFT(m, d, -c_t * vcom, -c_t * R2 ** 2 * omega, com, ids.thorax, applied)
                amp_log.append(math.degrees(0.5 * (stroke["left"] + stroke["right"])))
                if _DEBUG is not None and ctrl_count % 20 == 0:
                    _DEBUG.append({"t": round(t, 3), "n": np.round(n, 3).tolist(), "d": np.round(dvec, 3).tolist(),
                                   "v_h": np.round(v_h, 2).tolist(), "phi0": round(math.degrees(phi0), 1),
                                   "strokeLR": [round(math.degrees(stroke['left']), 1), round(math.degrees(stroke['right']), 1)],
                                   "F/Fmax": round(F / max(Fmax, 1e-12), 2), "tau": np.round(tau * 1e3, 3).tolist(),
                                   "tau_pitch_des": round(tau_pitch_des * 1e3, 4), "tau0_lat": round(tau0_lat * 1e3, 4),
                                   "psi_err": round(math.degrees(psi_err), 1), "w": np.round(omega, 2).tolist()})
            d.qfrc_applied[:] = applied
        # wing kinematics (servo targets), every physics step
        if wings_on:
            ph = 2 * math.pi * FREQ_HZ * (t - T_TRIGGER - T_WING_DELAY)
            c, s_ = math.cos(ph), math.sin(ph)
            flip = math.tanh(3.0 * s_)
            for side in ("left", "right"):
                yaw_t = phi0 + 0.5 * stroke[side] * c
                d.ctrl[ids.wing_act[(side, "yaw")]] = yaw_t
                d.ctrl[ids.wing_act[(side, "roll")]] = 0.0
                d.ctrl[ids.wing_act[(side, "pitch")]] = 0.8 + 0.8 * flip
        mj.mj_step(m, d)
        omega_buf[i % len(omega_buf)] = d.cvel[ids.thorax, :3]
        tn = (i + 1) * dt
        if (i + 1) % sample_every == 0 or i == n_steps - 1:
            _record_traj(tn)
        if recorder is not None:
            recorder.maybe_record(tn, m, d)
        if renderer is not None and tn + 1e-12 >= t_next_frame:
            try:
                renderer.update_scene(d, camera=camera if camera in ("chase",) else "chase")
                frames.append(renderer.render().copy())
            except Exception as exc:  # noqa: BLE001
                render_error = f"{type(exc).__name__}: {exc}"
                renderer = None
            t_next_frame += frame_dt
        if not np.all(np.isfinite(d.qpos)):
            warnings.append(f"physics became non-finite at t={tn:.4f}s; run stopped")
            break
    loop_s = time.perf_counter() - t_loop

    video = None
    if render_path and frames:
        try:
            import imageio.v2 as iio
            out = Path(render_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with iio.get_writer(str(out), fps=30, codec="libx264", macro_block_size=8,
                                ffmpeg_params=["-crf", "30", "-preset", "veryfast"]) as w:
                for fr in frames:
                    w.append_data(fr)
            video = out.resolve().as_posix()
        except Exception as exc:  # noqa: BLE001
            render_error = f"{type(exc).__name__}: {exc}"
    if renderer is not None:
        try:
            renderer.close()
        except Exception:  # noqa: BLE001
            pass

    tr = np.asarray(traj, dtype=float)
    t_arr, z = tr[:, 0], tr[:, 3]
    air = z > AIRBORNE_MM
    takeoff_time = float(t_arr[np.argmax(air)]) if air.any() else None
    flight_time = float(np.sum(air) * SAMPLE_DT) if air.any() else 0.0
    net = (tr[-1, 1:4] - tr[0, 1:4]).tolist()
    heading_change = float(tr[-1, 6] - tr[0, 6])
    seg = np.linalg.norm(np.diff(tr[:, 1:4], axis=0), axis=1)
    sim_t = float(t_arr[-1]) or duration_s
    res: dict[str, Any] = {
        "command": cmd,
        "trajectory": traj,
        "airborne": bool(air.any()),
        "takeoff_time_s": round(takeoff_time, 4) if takeoff_time is not None else None,
        "flight_time_s": round(flight_time, 3),
        "max_height_mm": round(float(z.max()), 3),
        "final_height_mm": round(float(z[-1]), 3),
        "net_displacement_mm": [round(float(v), 3) for v in net],
        "heading_change_deg": round(heading_change, 2),
        "mean_speed_mm_s": round(float(seg.sum()) / sim_t, 3),
        "wing_amplitude_deg": round(float(np.mean(amp_log)), 2) if amp_log else 0.0,
        "phi_hover_deg": round(math.degrees(ph_hover), 2),
        "stroke_saturation_fraction": round(sat_count / max(1, 2 * ctrl_count), 4),
        "min_stroke_normal_up_cos": round(min_up_cos, 3),
        "mass_mg": round(mass * 1000, 4),
        "weight_dyn": round(weight, 5),
        "sim_duration_s": round(sim_t, 4),
        "video": video,
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "wall_per_sim_s": round(loop_s / sim_t, 2),
        "conventions": ("trajectory [t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]: thorax position relative to the "
                        "start (world frame, start heading ~ +x), z = height gain above the standing height; pitch + = nose up; "
                        "yaw unwrapped, CCW + = LEFT turn (same as heading_change_deg). Video/poses are stroboscopic "
                        "(wingbeat appears slowed)."),
        "model_notes": model_notes(mass),
        "seed": seed,
    }
    if record_poses and recorder is not None:
        res["poses"] = recorder.to_dict()
    if render_error:
        res["render_error"] = render_error
    if warnings:
        res["warnings"] = warnings
    res["metrics_flight_phase"] = flight_phase_metrics(res)
    res["behavior"] = classify_flight(res)
    return res


def flight_phase_metrics(metrics: dict) -> dict:
    """Metrics over the flight phase (takeoff + SETTLE_AFTER_TAKEOFF_S .. end), recomputed from the trajectory."""
    tr = np.asarray(metrics.get("trajectory") or [], dtype=float)
    if tr.size == 0 or metrics.get("takeoff_time_s") is None:
        return {"window_s": None}
    t0 = float(metrics["takeoff_time_s"]) + SETTLE_AFTER_TAKEOFF_S
    w = tr[tr[:, 0] >= t0]
    if len(w) < 3:
        w = tr[tr[:, 0] >= float(metrics["takeoff_time_s"])]
    if len(w) < 2:
        return {"window_s": None}
    T = float(w[-1, 0] - w[0, 0]) or 1e-9
    h0 = math.radians(tr[0, 6])
    dxy = w[-1, 1:3] - w[0, 1:3]
    fwd = float(dxy[0] * math.cos(h0) + dxy[1] * math.sin(h0))
    return {"window_s": [round(float(w[0, 0]), 3), round(float(w[-1, 0]), 3)],
            "climb_rate_mm_s": round(float(w[-1, 3] - w[0, 3]) / T, 3),
            "forward_speed_mm_s": round(fwd / T, 3),
            "horizontal_speed_mm_s": round(float(np.linalg.norm(dxy)) / T, 3),
            "yaw_rate_deg_s": round(float(w[-1, 6] - w[0, 6]) / T, 3),
            "heading_change_deg": round(float(w[-1, 6] - w[0, 6]), 2)}


def classify_flight(metrics: dict) -> str:
    """Flight label from metrics (thresholds are module constants).

    1. never above AIRBORNE_MM (0.5 mm) -> no_takeoff
    2. final height < FALL_FINAL_MM or stroke-plane normal tipped below horizontal (tumble) -> takeoff_fall
    3. flight phase (takeoff + 0.3 s .. end): |heading change| >= 45 deg and |yaw rate| >= 60 deg/s
       -> flight_turn_left (CCW, +) / flight_turn_right
    4. forward speed along the initial heading >= 40 mm/s -> forward_flight
    5. climb rate >= 15 mm/s -> climb
    6. otherwise -> hover
    """
    if not metrics.get("airborne"):
        return "no_takeoff"
    if float(metrics.get("final_height_mm", 0.0)) < FALL_FINAL_MM or float(metrics.get("min_stroke_normal_up_cos", 1.0)) < TUMBLE_COS:
        return "takeoff_fall"
    fp = metrics.get("metrics_flight_phase") or flight_phase_metrics(metrics)
    if not fp.get("window_s"):
        return "takeoff_fall"
    if abs(fp["heading_change_deg"]) >= TURN_MIN_DEG and abs(fp["yaw_rate_deg_s"]) >= TURN_RATE_MIN:
        return "flight_turn_left" if fp["heading_change_deg"] > 0 else "flight_turn_right"
    if fp["forward_speed_mm_s"] >= FWD_MIN_SPEED:
        return "forward_flight"
    if fp["climb_rate_mm_s"] >= CLIMB_MIN_RATE:
        return "climb"
    return "hover"


# ---------------------------------------------------------------- demo
DEMO_CONDITIONS: dict[str, tuple[dict, str]] = {
    "no_takeoff": ({"takeoff": 0.0}, "no_takeoff"),
    "hover": ({"takeoff": 1.0}, "hover"),
    "climb": ({"takeoff": 1.0, "thrust": 1.0}, "climb"),
    "forward_flight": ({"takeoff": 1.0, "pitch": 0.8}, "forward_flight"),
    "flight_turn_left": ({"takeoff": 1.0, "yaw": -0.6}, "flight_turn_left"),
    "flight_turn_right": ({"takeoff": 1.0, "yaw": 0.6}, "flight_turn_right"),
}


def _slim(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("trajectory", "poses", "model_notes")}


def run_demo(duration_s: float = 1.0, render: bool = True, only: list[str] | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, (cmd, expected) in DEMO_CONDITIONS.items():
        if only and name not in only:
            continue
        rp = str(OUT_DIR / "demo" / f"{name}.mp4") if render else None
        r = simulate_flight(cmd, duration_s=duration_s, render_path=rp)
        ok = r["behavior"] == expected
        fp = r["metrics_flight_phase"]
        print(f"{name:<18} -> {r['behavior']:<18} {'ok' if ok else 'NO':<3} h_max {r['max_height_mm']:6.1f} mm  "
              f"net {r['net_displacement_mm']}  dH {r['heading_change_deg']:7.1f}  amp {r['wing_amplitude_deg']:5.1f}  "
              f"fp {fp}  wall {r['runtime_s']}s", flush=True)
        rows[name] = {"command": cmd, "expected": expected, "ok": ok, **_slim(r), "trajectory": r["trajectory"]}
    summary = {"n_ok": sum(v["ok"] for v in rows.values()), "n": len(rows), "rows": rows, "model_notes": model_notes()}
    (OUT_DIR / "demo.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{summary['n_ok']}/{summary['n']} demo commands classified as intended; saved {OUT_DIR / 'demo.json'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FlyBody flight (quasi-steady aerodynamics) driven by a 4-number command")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--command", type=str, default=None, help='e.g. "takeoff=1,thrust=0.5,yaw=-0.5"')
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--render", type=str, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached compiled model")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.rebuild and MODEL_CACHE.exists():
        MODEL_CACHE.unlink()
    if args.demo:
        run_demo(duration_s=args.duration, render=not args.no_render, only=args.only)
    elif args.command:
        cmd = {k.strip(): float(v) for k, v in (p.split("=") for p in args.command.split(",") if "=" in p)}
        r = simulate_flight(cmd, duration_s=args.duration, render_path=args.render)
        print(json.dumps(_slim(r), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
