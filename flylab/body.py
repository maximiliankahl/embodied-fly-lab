"""Physics body for the Embodied Fly Lab: NeuroMechFly (flygym 2.1, MuJoCo) driven by a
low-dimensional descending command.

Contract (docs/CONTRACTS.md):
    simulate_walk(drive, duration_s=1.5, render_path=None, seed=0) -> dict
    classify(metrics) -> "forward" | "backward" | "turn_left" | "turn_right" | "stop"

How a descending command becomes walking
----------------------------------------
We reuse flygym's own *hybrid turning controller* (package ``flygym_demo.complex_terrain``,
shipped with flygym 2.1.0): a network of 6 coupled phase oscillators (CPG, tripod gait,
12 Hz) drives *preprogrammed single-leg step kinematics* extracted from experimentally
recorded fly walking (flygym asset ``single_steps_untethered.pkl``; provenance per
NeuroMechFly v2, not re-checked here), plus sensory-feedback
corrections (leg retraction / stumbling). The controller takes a 2-element
"descending signal" [left, right]:

* |signal| sets the oscillator amplitude on that side (step size),
* sign(signal) sets the direction in which that side's oscillators run
  (negative = step kinematics replayed in reverse).

Sources:
* NeuroMechFly v2 (hybrid turning controller, descending drive):
  Wang-Chen et al. 2024, Nature Methods, doi:10.1038/s41592-024-02497-y
* NeuroMechFly body model: Lobato-Rios et al. 2022, Nature Methods,
  doi:10.1038/s41592-022-01466-7
* Code: https://github.com/NeLy-EPFL/flygym (v2.1.0, tutorial 4d_turning_controller)

Our mapping from the contract's ``drive`` to [left, right] (differential steering):
    net   = forward - backward                       (-1..1)
    s_L   = net + turn,  s_R = net - turn            (turn < 0 = left -> left side weaker)
    [L,R] = AMP_MAX * [s_L, s_R] / max(1, |s_L|, |s_R|)
With AMP_MAX = 1.2 this reproduces the flygym tutorial values: forward=1, turn=-0.5 ->
[0.4, 1.2] (left turn). forward=0, turn=-1 -> [-1.2, 1.2] = turning in place.

APPROXIMATION (backward walking): flygym has no recorded backward step kinematics.
Backward walking here = the *forward* step kinematics replayed in reverse phase order on
both sides (negative descending signal, the same mechanism flygym uses for the inner
side of sharp turns). Real backward walking in Drosophila (e.g. moonwalker descending
neurons, Bidaye et al. 2014) uses different leg kinematics; treat our backward gait as
a qualitative stand-in only.

Sign conventions (all relative to the fly at t=0, viewed from above, right-handed world
frame, z up; in NeuroMechFly +x = anterior, +y = fly's LEFT):
* heading_deg / heading_change_deg: mathematical convention, counter-clockwise
  positive. heading_change_deg > 0  ==  the fly turned LEFT.
* turn_index = -heading_change_deg / 90 (clipped to -1..1), i.e. the *drive* convention:
  turn_index < 0 == left. Verified in simulation: drive turn<0 gives heading_change>0.
* lateral_disp_mm > 0 == displaced toward the fly's initial LEFT side (+y body frame).
* drive.turn < 0 rotates the body counter-clockwise (nose to the left) both when walking
  forward and when walking backward (differential "tank" steering, not car-like
  reversing). Checked: backward=1, turn=-0.5 -> heading_change > 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

# MuJoCo picks its OpenGL backend at import time. On Windows only GLFW (offscreen
# hidden window) is available; must be set before mujoco / flygym are imported.
if sys.platform.startswith("win"):
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "spikes" / "body" / "out"

LABELS = ("forward", "backward", "turn_left", "turn_right", "stop")

# --- controller / mapping parameters -------------------------------------------------
AMP_MAX = 1.2  # max CPG amplitude per side (flygym tutorial 4d uses 1.2 / 0.4)
MIN_SIGNAL = 0.05  # |side signal| below this -> treated as 0 (no drive)
SAMPLE_DT_S = 0.01  # trajectory sampling interval
# Tripod phase-locked initial CPG phases (legs lf, lm, lh, rf, rm, rh). Tripod groups:
# {lf, lh, rm} and {lm, rf, rh}, see flygym_demo.complex_terrain.get_cpg_biases("tripod").
# Starting phase-locked (instead of random phases) avoids an asymmetric start transient
# that otherwise produces a left/right drift during straight walking.
TRIPOD_INIT_PHASES = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi])

# --- classification thresholds (documented, see classify()) -------------------------
TURN_RATE_THRESH_DEG_S = 40.0  # |mean yaw rate| above this -> turn_left/turn_right
# ...and |total heading change| must also exceed this. Stride-locked heading wobble is a
# few degrees, so very short runs (< ~0.25 s) would otherwise be labelled turns from
# wobble alone (seen in review: 0.05 s forward run, -3 deg -> "turn_right"). For runs
# >= 0.25 s the rate threshold (40 deg/s * 0.25 s = 10 deg) dominates, so no change.
TURN_MIN_ABS_DEG = 10.0
STOP_SPEED_THRESH_MM_S = 1.5  # net displacement speed below this -> stop
# A run counts as straight forward/backward if |net speed| >= STOP_SPEED_THRESH and
# |yaw rate| < TURN_RATE_THRESH; direction from the sign of forward_disp_mm.


DRIVE_KEYS = ("forward", "turn", "backward")
MIN_DURATION_S = 0.05  # shorter runs are clamped up (avoids empty trajectories / div by 0)
MAX_DURATION_S = 10.0  # longer runs are clamped down (1 sim-s costs ~7-25 s wall)
UPRIGHT_MIN_COS = 0.5  # thorax z-axis . world z below this (tilt > 60 deg) = fell over


def _num(value: Any, lo: float, hi: float) -> float:
    """Robust float conversion for drive values: None/NaN/garbage -> 0, then clip."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def sanitize_drive(drive: dict | None) -> dict:
    """Return {"forward", "turn", "backward"} as clipped floats (NaN/None/str-safe)."""
    drive = drive or {}
    return {
        "forward": _num(drive.get("forward", 0.0), 0.0, 1.0),
        "turn": _num(drive.get("turn", 0.0), -1.0, 1.0),
        "backward": _num(drive.get("backward", 0.0), 0.0, 1.0),
    }


def drive_to_descending(drive: dict | None) -> list[float]:
    """Map the contract drive dict to flygym's [left, right] descending signal."""
    d = sanitize_drive(drive)
    fwd, bwd, turn = d["forward"], d["backward"], d["turn"]
    net = fwd - bwd
    s_l, s_r = net + turn, net - turn
    norm = max(1.0, abs(s_l), abs(s_r))
    sig = [AMP_MAX * s_l / norm, AMP_MAX * s_r / norm]
    return [0.0 if abs(s) < MIN_SIGNAL else round(float(s), 4) for s in sig]


def _build(camera: str | None, control_every: int = 1):
    """Build fly + flat world + simulation + hybrid turning controller."""
    from flygym import Simulation
    from flygym.anatomy import BodySegment, ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        HybridTurningController,
        PreprogrammedSteps,
        make_locomotion_fly,
    )

    fly = make_locomotion_fly(name="nmf", add_adhesion=True, colorize=camera is not None)
    cam = None
    if camera == "top":
        # "track" mode: follows the thorax position, world-fixed orientation, looking
        # straight down (+x to the right of the image, +y = fly's left = image up).
        cam = fly.add_tracking_camera(
            name="topcam",
            pos_offset=(-0.5, 0.0, 13.0),
            rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0)),
            fovy=40.0,
        )
    elif camera == "side":
        cam = fly.add_tracking_camera(name="sidecam")  # flygym default 3/4 view
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    ctrl = HybridTurningController(
        timestep=sim.timestep * control_every,
        preprogrammed_steps=steps,
        output_dof_order=dof_order,
        # persistence is counted in controller steps: keep its duration in seconds
        retraction_persistence_steps=max(
            1, round(HybridTurningController.retraction_persistence_steps / control_every)
        ),
    )
    thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    return fly, cam, sim, steps, dof_order, ctrl, thorax_idx


def _heading_rad(sim, fly_name: str, thorax_idx: int) -> float:
    # thorax body frame: +x = anterior, +y = fly's left, +z = dorsal (checked against
    # coxa / abdomen positions in spikes/body/review_checks.py)
    body_id = sim._internal_bodyids_by_fly[fly_name][thorax_idx]
    x_axis = sim.mj_data.xmat[body_id].reshape(3, 3)[:, 0]
    return math.atan2(float(x_axis[1]), float(x_axis[0]))


def _upright_cos(sim, fly_name: str, thorax_idx: int) -> float:
    """Cosine between the thorax dorsal (+z body) axis and world up (1 = upright)."""
    body_id = sim._internal_bodyids_by_fly[fly_name][thorax_idx]
    return float(sim.mj_data.xmat[body_id].reshape(3, 3)[2, 2])


def plot_trajectory(results: dict[str, dict] | dict, path: str | os.PathLike) -> str:
    """Top-down plot of one result or {label: result} trajectories. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "trajectory" in results:
        results = {results.get("behavior", "run"): results}
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    for label, res in results.items():
        tr = np.asarray(res["trajectory"], dtype=float)
        if tr.size == 0:
            continue
        line = ax.plot(tr[:, 1], tr[:, 2], label=f"{label} -> {res.get('behavior')}")[0]
        # heading arrow at the end
        h = math.radians(tr[-1, 3])
        ax.annotate(
            "",
            xy=(tr[-1, 1] + 1.5 * math.cos(h), tr[-1, 2] + 1.5 * math.sin(h)),
            xytext=(tr[-1, 1], tr[-1, 2]),
            arrowprops=dict(arrowstyle="->", color=line.get_color()),
        )
    ax.plot(0, 0, "ko", ms=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (mm, initial heading = +x)")
    ax.set_ylabel("y (mm, + = fly's left)")
    ax.set_title("NeuroMechFly thorax trajectories (top view)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def simulate_walk(
    drive: dict,
    duration_s: float = 1.5,
    render_path: str | None = None,
    seed: int = 0,
    *,
    camera: str = "top",
    camera_res: tuple[int, int] = (352, 480),
    playback_speed: float = 0.25,
    sample_dt_s: float = SAMPLE_DT_S,
    phase_noise_rad: float = 0.0,
    control_every: int = 1,
) -> dict:
    """Simulate NeuroMechFly walking on flat ground under a constant descending drive.

    Args:
        drive: {"forward": 0..1, "turn": -1..1 (neg = left), "backward": 0..1}.
        duration_s: simulated time (after a 0.05 s posture warm-up that is not counted).
        render_path: optional .mp4 path. If MuJoCo rendering fails, a trajectory PNG is
            written instead (same stem, .png) and returned in "video"; the error is in
            "render_error".
        seed: seeds the CPG RNG and the optional initial-phase noise. With
            phase_noise_rad=0 (default) initial phases are tripod-locked and the run is
            deterministic regardless of seed.
        camera: "top" (top-down follow camera, best for turns) or "side" (3/4 view).
        phase_noise_rad: std of Gaussian noise (seeded by ``seed``) added to the
            tripod initial CPG phases; 0 = deterministic. Use >0 for trial variability.
        control_every: run the (pure-Python, ~80% of runtime) controller only every
            N physics steps (CPG integrated with N*dt, actuator targets held in
            between). 1 = flygym tutorial setting (default). See spikes/body notes.

    Returns the contract dict (see module docstring for sign conventions) plus extras:
        descending_signal, turn_index, yaw_rate_deg_s, net_speed_mm_s, sim_duration_s,
        wall_per_sim_s, drive (sanitised), conventions, upright_min_cos, fell_over,
        warnings (only if any: unknown drive keys, clamped duration, fall-over, NaN).
    Drive values are sanitised (None/NaN/non-numeric -> 0, then clipped);
    duration_s is clamped to [MIN_DURATION_S, MAX_DURATION_S].
    """
    from flygym_demo.complex_terrain import (
        HybridControllerObservation,
        LocomotionAction,
        apply_locomotion_action,
    )

    t_start = time.perf_counter()
    warnings: list[str] = []
    unknown = sorted(set((drive or {}).keys()) - set(DRIVE_KEYS))
    if unknown:
        warnings.append(f"unknown drive keys ignored: {unknown} (use {list(DRIVE_KEYS)})")
    clean_drive = sanitize_drive(drive)
    try:
        req_dur = float(duration_s)
    except (TypeError, ValueError):
        req_dur = 1.0
    if math.isnan(req_dur):
        req_dur = 1.0
    duration_s = min(MAX_DURATION_S, max(MIN_DURATION_S, req_dur))
    if duration_s != req_dur:
        warnings.append(f"duration_s clamped from {req_dur} to {duration_s}")
    signal = np.asarray(drive_to_descending(clean_drive), dtype=float)

    want_video = render_path is not None
    render_error = None
    control_every = max(1, int(control_every))
    fly, cam, sim, steps, dof_order, ctrl, thorax_idx = _build(
        camera if want_video else None, control_every
    )
    if want_video:
        try:
            sim.set_renderer(
                [cam], camera_res=camera_res, playback_speed=playback_speed, output_fps=25
            )
        except Exception as exc:  # noqa: BLE001 - fall back to PNG
            render_error = f"{type(exc).__name__}: {exc}"
            sim.renderer = None

    sim.reset()
    init_phases = TRIPOD_INIT_PHASES.copy()
    if phase_noise_rad > 0:
        init_phases += np.random.RandomState(seed).normal(0.0, phase_noise_rad, 6)
    ctrl.reset(seed=seed, init_phases=init_phases)
    init_action = LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool),
    )
    apply_locomotion_action(sim, fly.name, init_action)
    sim.warmup()

    dt = sim.timestep
    n_steps = int(round(duration_s / dt))
    sample_every = max(1, int(round(sample_dt_s / dt)))

    p0 = sim.get_body_positions(fly.name)[thorax_idx].copy()
    h0 = _heading_rad(sim, fly.name, thorax_idx)
    traj: list[list[float]] = [[0.0, 0.0, 0.0, math.degrees(h0)]]
    unwrapped = h0
    prev_h = h0
    upright_min = _upright_cos(sim, fly.name, thorax_idx)
    t_loop = time.perf_counter()
    for i in range(1, n_steps + 1):
        if (i - 1) % control_every == 0:  # controller rate = physics rate / control_every
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            action = ctrl.step(signal, obs)
            apply_locomotion_action(sim, fly.name, action)
        sim.step()
        if sim.renderer is not None:
            try:
                sim.render_as_needed()
            except Exception as exc:  # noqa: BLE001
                render_error = f"{type(exc).__name__}: {exc}"
                sim.renderer = None
        if i % sample_every == 0 or i == n_steps:
            pos = sim.get_body_positions(fly.name)[thorax_idx]
            h = _heading_rad(sim, fly.name, thorax_idx)
            dh = (h - prev_h + math.pi) % (2 * math.pi) - math.pi
            unwrapped += dh
            prev_h = h
            upright_min = min(upright_min, _upright_cos(sim, fly.name, thorax_idx))
            traj.append(
                [
                    round(i * dt, 4),
                    round(float(pos[0] - p0[0]), 4),
                    round(float(pos[1] - p0[1]), 4),
                    round(math.degrees(unwrapped), 3),
                ]
            )
    loop_s = time.perf_counter() - t_loop

    tr = np.asarray(traj, dtype=float)
    # displacement in the fly's initial body frame
    dx, dy = tr[-1, 1], tr[-1, 2]
    c, s = math.cos(h0), math.sin(h0)
    forward_disp = dx * c + dy * s
    lateral_disp = -dx * s + dy * c  # + = fly's initial left
    heading_change = float(tr[-1, 3] - tr[0, 3])
    sim_t = float(tr[-1, 0]) if tr[-1, 0] > 0 else float(duration_s)
    # path length on 20 ms samples (smooths stride wobble)
    stride = max(1, int(round(0.02 / sample_dt_s)))
    sub = tr[::stride, 1:3]
    if len(sub) < 2 or not np.allclose(sub[-1], tr[-1, 1:3]):
        sub = np.vstack([sub, tr[-1, 1:3]])
    path_len = float(np.sum(np.linalg.norm(np.diff(sub, axis=0), axis=1)))

    video = None
    if want_video:
        out = Path(render_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if sim.renderer is not None and render_error is None:
            try:
                sim.renderer.save_video(out)
                video = out.resolve().as_posix()
            except Exception as exc:  # noqa: BLE001
                render_error = f"{type(exc).__name__}: {exc}"
        if video is None:
            png = out.with_suffix(".png")
            metrics_tmp = {"trajectory": traj, "behavior": "?"}
            video = Path(plot_trajectory({"run": metrics_tmp}, png)).resolve().as_posix()
    try:
        sim.close()
    except Exception:  # noqa: BLE001
        pass

    result: dict[str, Any] = {
        "trajectory": traj,
        "forward_disp_mm": round(float(forward_disp), 3),
        "lateral_disp_mm": round(float(lateral_disp), 3),
        "heading_change_deg": round(heading_change, 2),
        "mean_speed_mm_s": round(path_len / sim_t, 3),
        "net_speed_mm_s": round(math.hypot(dx, dy) / sim_t, 3),
        "yaw_rate_deg_s": round(heading_change / sim_t, 2),
        "turn_index": round(float(np.clip(-heading_change / 90.0, -1, 1)), 3),
        "video": video,
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "wall_per_sim_s": round(loop_s / sim_t, 2),
        "sim_duration_s": round(sim_t, 3),
        "control_every": control_every,
        "drive": clean_drive,
        "descending_signal": [float(x) for x in signal],
        "upright_min_cos": round(upright_min, 3),
        "fell_over": bool(upright_min < UPRIGHT_MIN_COS),
        "conventions": (
            "heading_change_deg>0 = counter-clockwise = LEFT turn; turn_index<0 = left "
            "(same sign as drive.turn); lateral_disp_mm>0 = toward fly's initial left; "
            "x,y in mm relative to start, world frame."
        ),
    }
    if render_error:
        result["render_error"] = render_error
    if result["fell_over"]:
        warnings.append(
            f"fly tilted > 60 deg (min upright cos {upright_min:.2f}); behaviour label unreliable"
        )
    if not np.all(np.isfinite(tr)):
        warnings.append("non-finite values in trajectory (physics instability)")
    if warnings:
        result["warnings"] = warnings
    result["behavior"] = classify(result)
    return result


def classify(metrics: dict) -> str:
    """Map walk metrics to a behaviour label.

    Rules (applied in order; thresholds are module constants):
      1. |yaw_rate| >= TURN_RATE_THRESH_DEG_S (40 deg/s) and |heading change| >=
         TURN_MIN_ABS_DEG (10 deg) -> turn_left if heading increased
         (counter-clockwise), else turn_right. Turning in place counts as turn.
      2. net displacement speed < STOP_SPEED_THRESH_MM_S (1.5 mm/s) -> stop.
      3. forward_disp_mm > 0 -> forward, else backward.
    Reference values from our own runs (1 s, full drive): forward ~15.5 mm/s net,
    backward ~10.5 mm/s, turns ~150 deg/s, stop ~0 mm/s, straight-walk drift ~7 deg/s.
    """
    dur = float(metrics.get("sim_duration_s") or 0.0)
    if dur <= 0:
        tr = metrics.get("trajectory") or []
        dur = float(tr[-1][0]) if tr else 1.0
        dur = dur or 1.0
    dh = float(metrics.get("heading_change_deg", 0.0))
    yaw_rate = metrics.get("yaw_rate_deg_s")
    yaw_rate = float(yaw_rate) if yaw_rate is not None else dh / dur
    fwd = float(metrics.get("forward_disp_mm", 0.0))
    lat = float(metrics.get("lateral_disp_mm", 0.0))
    net_speed = metrics.get("net_speed_mm_s")
    net_speed = float(net_speed) if net_speed is not None else math.hypot(fwd, lat) / dur

    if abs(yaw_rate) >= TURN_RATE_THRESH_DEG_S and abs(dh) >= TURN_MIN_ABS_DEG:
        return "turn_left" if yaw_rate > 0 else "turn_right"
    if net_speed < STOP_SPEED_THRESH_MM_S:
        return "stop"
    return "forward" if fwd > 0 else "backward"


# --------------------------------------------------------------------------------------
DEMO_CONDITIONS: dict[str, dict] = {
    "forward": {"forward": 1.0, "turn": 0.0, "backward": 0.0},
    "turn_left": {"forward": 1.0, "turn": -0.5, "backward": 0.0},
    "turn_right": {"forward": 1.0, "turn": 0.5, "backward": 0.0},
    "stop": {"forward": 0.0, "turn": 0.0, "backward": 0.0},
    "backward": {"forward": 0.0, "turn": 0.0, "backward": 1.0},
}


def _run_one(args: tuple) -> tuple[str, dict]:
    name, drive, duration, render_path, seed, kwargs = args
    return name, simulate_walk(
        drive, duration_s=duration, render_path=render_path, seed=seed, **kwargs
    )


def simulate_many(
    drives: dict[str, dict],
    duration_s: float = 1.0,
    render_dir: str | os.PathLike | None = None,
    seed: int = 0,
    parallel: bool = True,
    max_workers: int | None = None,
    **sim_kwargs: Any,
) -> dict[str, dict]:
    """Run several drives (optionally in parallel processes). Returns {name: result}.

    Extra keyword args (camera, phase_noise_rad, control_every, ...) go to simulate_walk.
    """
    jobs = []
    for name, drive in drives.items():
        # names may come from an agent ("MDN activation / L"): make them file-safe
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))[:80]
        rp = str(Path(render_dir) / f"{safe or 'run'}.mp4") if render_dir else None
        jobs.append((name, drive, duration_s, rp, seed, sim_kwargs))
    if not parallel or len(jobs) == 1:
        return dict(_run_one(j) for j in jobs)
    workers = max_workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(_run_one, jobs))
    except (BrokenProcessPool, OSError, pickle.PicklingError) as exc:  # e.g. spawn issues
        print(f"[flylab.body] process pool failed ({type(exc).__name__}: {exc}); running serially")
        return dict(_run_one(j) for j in jobs)


def run_demo(
    out_dir: str | os.PathLike = OUT_DIR,
    duration_s: float = 1.0,
    render: bool = True,
    parallel: bool = True,
    control_every: int = 1,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = simulate_many(
        DEMO_CONDITIONS,
        duration_s=duration_s,
        render_dir=out_dir if render else None,
        parallel=parallel,
        control_every=control_every,
    )
    wall = time.perf_counter() - t0
    plot = plot_trajectory(results, out_dir / "demo_trajectories.png")

    header = f"{'condition':<11} {'L,R signal':>13} {'fwd mm':>7} {'lat mm':>7} {'dHead deg':>9} {'speed':>6} {'label':<10} {'ok':<3} {'wall/sim-s':>10}"
    print(header)
    print("-" * len(header))
    n_ok = 0
    for name, r in results.items():
        ok = r["behavior"] == name
        n_ok += ok
        sig = ",".join(f"{x:+.2f}" for x in r["descending_signal"])
        print(
            f"{name:<11} {sig:>13} {r['forward_disp_mm']:>7.2f} {r['lateral_disp_mm']:>7.2f} "
            f"{r['heading_change_deg']:>9.1f} {r['mean_speed_mm_s']:>6.1f} {r['behavior']:<10} "
            f"{'yes' if ok else 'NO':<3} {r['wall_per_sim_s']:>10.1f}"
        )
    print(f"\n{n_ok}/{len(results)} conditions classified as intended; total wall {wall:.1f}s")
    print(f"trajectory plot: {plot}")
    summary = {
        "conditions": DEMO_CONDITIONS,
        "results": results,
        "n_correct": n_ok,
        "total_wall_s": round(wall, 1),
        "trajectory_plot": plot,
        "parallel": parallel,
        "control_every": control_every,
    }
    with open(out_dir / "demo.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {out_dir / 'demo.json'}")
    return summary


def _parse_drive(text: str) -> dict:
    drive = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            drive[k.strip()] = float(v)
    return drive


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="NeuroMechFly body: descending drive -> walking")
    ap.add_argument("--demo", action="store_true", help="run 5 canonical drives")
    ap.add_argument("--drive", type=str, help='e.g. "forward=1,turn=-0.5"')
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--render", type=str, default=None, help="mp4 path for --drive")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--serial", action="store_true", help="no multiprocessing")
    ap.add_argument(
        "--control-every", type=int, default=1,
        help="controller every N physics steps (1 = flygym default; 5 = ~3.5x faster)",
    )
    args = ap.parse_args(argv)

    if args.demo:
        run_demo(
            duration_s=args.duration or 1.0,
            render=not args.no_render,
            parallel=not args.serial,
            control_every=args.control_every,
        )
    elif args.drive:
        res = simulate_walk(
            _parse_drive(args.drive),
            duration_s=args.duration or 1.5,
            render_path=args.render,
            control_every=args.control_every,
        )
        res = {k: v for k, v in res.items() if k != "trajectory"}
        print(json.dumps(res, indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
