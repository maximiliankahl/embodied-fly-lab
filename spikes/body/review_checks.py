"""Reviewer checks for flylab/body.py (sign conventions, robustness).

Run from 02_App:  uv run python spikes/body/review_checks.py
"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from flylab import body  # noqa: E402


def frame_check():
    """Is the thorax x-axis anterior? Is gravity -z? Where is the fly's left?"""
    from flygym.anatomy import BodySegment

    fly, cam, sim, steps, dof_order, ctrl, thorax_idx = body._build(None)
    sim.reset()
    sim.warmup()
    order = fly.get_bodysegs_order()
    names = [str(getattr(b, "name", b)) for b in order]
    pos = sim.get_body_positions(fly.name)
    th = pos[thorax_idx]
    print("gravity:", sim.mj_model.opt.gravity, "timestep:", sim.timestep)
    tid = sim._internal_bodyids_by_fly[fly.name][thorax_idx]
    xm = sim.mj_data.xmat[tid].reshape(3, 3)
    print("thorax x-axis (world):", np.round(xm[:, 0], 3), " y-axis:", np.round(xm[:, 1], 3))
    for want in ("c_head", "c_abdomen12", "c_abdomen1", "lf_coxa", "rf_coxa", "lm_coxa", "rm_coxa"):
        for i, n in enumerate(names):
            if n.endswith(want) or n == want:
                print(f"{n:>14}: rel to thorax {np.round(pos[i] - th, 3)}")
                break
    sim.close()


def robustness_check():
    print("drive_to_descending tests")
    cases = [
        {"forward": 1},
        {"forward": float("nan"), "turn": -0.5},
        {"forward": "0.5"},
        {"forward": None, "turn": None},
        {"turn": float("inf")},
        {},
        None,
    ]
    for c in cases:
        try:
            print(" ", c, "->", body.drive_to_descending(c))
        except Exception as exc:  # noqa: BLE001
            print(" ", c, "-> EXC", type(exc).__name__, exc)
    print("classify on contract-only metrics (no extras):")
    m = {"trajectory": [[0, 0, 0, 0], [1.0, 10, 0, 5]], "forward_disp_mm": 10.0,
         "lateral_disp_mm": 0.0, "heading_change_deg": 5.0, "mean_speed_mm_s": 10.0}
    print("  ", body.classify(m))
    m2 = {"forward_disp_mm": 0.0, "lateral_disp_mm": 0.0, "heading_change_deg": 0.0}
    print("  empty-ish:", body.classify(m2))


def short_sim(drive, dur=0.3):
    t = time.perf_counter()
    r = body.simulate_walk(drive, duration_s=dur)
    keys = ["forward_disp_mm", "lateral_disp_mm", "heading_change_deg", "net_speed_mm_s",
            "yaw_rate_deg_s", "behavior", "descending_signal", "wall_per_sim_s"]
    print(drive, {k: r[k] for k in keys}, f"{time.perf_counter() - t:.1f}s")
    return r


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("frame", "all"):
        frame_check()
    if what in ("robust", "all"):
        robustness_check()
    if what in ("sims",):
        for d in [{"forward": 1, "turn": -0.5}, {"forward": float("nan")},
                  {"forward": 1, "turn": 0, "backward": 0}]:
            try:
                short_sim(d)
            except Exception as exc:  # noqa: BLE001
                print(d, "EXC", type(exc).__name__, exc)
