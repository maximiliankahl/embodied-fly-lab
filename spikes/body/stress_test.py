"""Reviewer stress test for flylab/body.py. Run from 02_App:
    uv run python spikes/body/stress_test.py
"""
import json, sys, time
sys.path.insert(0, ".")
from flylab import body

DRIVES = {
    "turn_in_place_L": {"turn": -1.0},
    "turn_in_place_R": {"turn": 1.0},
    "back_left": {"backward": 1.0, "turn": -0.5},
    "weak_fwd_0.1": {"forward": 0.1},
    "fwd0.6_turnR0.3": {"forward": 0.6, "turn": 0.3},
    "nan_forward": {"forward": float("nan"), "turn": -0.5},
    "str_forward": {"forward": "0.8"},
    "unknown_key / agent": {"fwd": 1.0},
}

if __name__ == "__main__":
    t = time.perf_counter()
    res = body.simulate_many(DRIVES, duration_s=1.0, parallel=True, max_workers=8)
    res["zero_duration"] = body.simulate_walk({"forward": 1}, duration_s=0)
    keys = ["descending_signal", "forward_disp_mm", "lateral_disp_mm", "heading_change_deg",
            "net_speed_mm_s", "behavior", "upright_min_cos", "fell_over", "wall_per_sim_s", "warnings"]
    for n, r in res.items():
        print(f"{n:<20}", json.dumps({k: r.get(k) for k in keys}))
    print(f"total {time.perf_counter() - t:.1f}s")
