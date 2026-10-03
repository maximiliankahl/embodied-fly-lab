"""Condense spikes/brain/out/validation.json (flylab.brain --validate) into data/benchmarks/brain_validation.json.
Key numbers only; values are copied, not recomputed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = json.loads((ROOT / "spikes/brain/out/validation.json").read_text(encoding="utf-8"))
mn9_keys = [k for k in src["table"][0] if k.startswith("MN9")]
dose = []
conditions = []
for row in src["table"]:
    r = {"condition": row["condition"], "n_active": row["n_active"], "runtime_s": row["runtime_s"]}
    for k in mn9_keys:
        short = "MN9_a" if k.startswith("MN9_a") else "MN9_b"
        r[f"{short}_hz"] = row[k]["mean_hz"]
        r[f"{short}_std_hz"] = row[k]["std_hz"]
        r[f"{short}_trials_active"] = row[k]["trials_active"]
    conditions.append(r)
    if row["condition"].startswith("sugar GRNs ") and row["condition"].endswith("Hz"):
        dose.append({"sugar_rate_hz": float(row["condition"].split()[2]),
                     "MN9_a_hz": r["MN9_a_hz"], "MN9_b_hz": r["MN9_b_hz"], "n_active": row["n_active"]})
bx = src["brian2_crosscheck_subnetwork"]
out = {
    "_meta": {
        "source": "spikes/brain/out/validation.json (uv run python -m flylab.brain --validate), condensed by "
                  "spikes/dashboard/make_brain_validation.py; numbers copied, not recomputed",
        "paper": src["paper"],
        "connectome": src["connectome"],
        "n_neurons": src["n_neurons"],
        "n_trials": src["n_trials"],
        "duration_ms": src["duration_ms"],
        "params": src["params"],
        "mn9_ids": src["stimulus_sets"]["mn9"],
        "side_note": src["stimulus_sets"]["side_convention_note"],
        "silencing_semantics": src["silencing_semantics"],
    },
    "checks": src["checks"],
    "conditions": conditions,
    "sugar_dose_response": dose,
    "speed_s_per_trial_per_sim_s": src["speed_s_per_trial_per_sim_s"],
    "brian2_crosscheck": {
        "what": "same LIF equations simulated with Brian2 (reference implementation of Shiu et al.) on a "
                "sugar-GRN subnetwork, 100 Hz stimulation",
        "n_neurons": bx["n_neurons"], "n_connections": bx["n_connections"], "n_trials": bx["n_trials"],
        "pearson_r_nonstim_rates": round(bx["pearson_r_nonstim_rates"], 4),
        "n_active_nonstim": bx["n_active_nonstim"],
        "total_rate_nonstim_hz": bx["total_rate_nonstim"],
        "mn9": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in bx["mn9"].items()},
    },
}
dst = ROOT / "data/benchmarks/brain_validation.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", dst, dst.stat().st_size, "bytes")
