"""control_every 1 vs 5 tolerance check + backward-label robustness (writes spikes/bridge/out/body_check.json)."""
import json, sys, time
sys.path.insert(0, ".")
from flylab import body

DRIVES = dict(body.DEMO_CONDITIONS)
DRIVES["p9L_like"] = {"forward": 0.49, "turn": -0.61, "backward": 0.0}
DRIVES["backward_small_turn"] = {"forward": 0.0, "turn": 0.1, "backward": 1.0}
DRIVES["backward_half"] = {"forward": 0.0, "turn": 0.0, "backward": 0.5}
KEYS = ["forward_disp_mm", "lateral_disp_mm", "heading_change_deg", "yaw_rate_deg_s", "net_speed_mm_s", "behavior",
        "wall_per_sim_s", "fell_over"]

if __name__ == "__main__":
    out = {}
    for ce in (5, 1):
        for noise, seeds in ((0.0, [0]), (0.5, [1, 2, 3])):
            for seed in seeds:
                t = time.perf_counter()
                res = body.simulate_many(DRIVES, duration_s=1.0, parallel=True, max_workers=4, control_every=ce,
                                         phase_noise_rad=noise, seed=seed)
                for n, r in res.items():
                    out[f"ce{ce}|noise{noise}|seed{seed}|{n}"] = {k: r.get(k) for k in KEYS}
                print(f"ce={ce} noise={noise} seed={seed}: {time.perf_counter()-t:.1f}s", flush=True)
                for n, r in res.items():
                    print(f"   {n:<20} " + " ".join(f"{k[:8]}={r.get(k)}" for k in KEYS), flush=True)
    json.dump(out, open("spikes/bridge/out/body_check.json", "w"), indent=1)
