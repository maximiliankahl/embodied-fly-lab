"""Reviewer integration check: do atlas stimulus groups drive the readout groups named in ground_truth (brain model only)?"""
import sys, time, json
sys.stdout.reconfigure(encoding="utf-8")
from flylab import atlas, brain
t0 = time.time()
brain.load_connectome()
print("connectome loaded", round(time.time() - t0, 1), "s", flush=True)
def mean(res, g):
    r = brain.rates_for(res, atlas.group_ids(g))
    vals = [float(v) for v in r.values()]
    return round(sum(vals) / max(len(atlas.group_ids(g)), 1), 1), {k[-6:]: round(float(v), 1) for k, v in r.items()}
tests = [
    ("sugar_GRN_shiu2024", None, ["MN9_L", "MN9_R"]),
    ("bitter_GRN_shiu2024", None, ["MN9_L", "MN9_R"]),
    ("water_GRN_shiu2024", None, ["MN9_L", "MN9_R"]),
    ("JO_CE_shiu2024", None, ["aBN1_L", "aBN1_R", "aDN1_shiu2024", "aDN2_shiu2024"]),
    ("LPLC2", None, ["GF_L", "GF_R"]),
    ("LPLC2_L", None, ["GF_L", "GF_R"]),
    ("aBN1", None, ["aDN1_shiu2024", "aDN2_shiu2024"]),
    ("MDN", None, ["MDN", "P9"]),
]
out = []
for exc, sil, reads in tests:
    t = time.time()
    res = brain.simulate(atlas.group_ids(exc), silence=sil, excite_rate_hz=150.0, duration_ms=500.0, n_trials=2)
    row = {"excite": exc, "runtime_s": round(time.time() - t, 1), "n_active": res["n_active"]}
    for g in reads:
        row[g] = mean(res, g)
    print(json.dumps(row), flush=True)
    out.append(row)
json.dump(out, open("spikes/atlas/out/brain_readouts.json", "w"), indent=1)
