"""Bridge calibration: rate each readout/steering group reaches under direct 150 Hz Poisson drive,
plus cross-activation, plus natural-stimulus conditions (brain only). Output: spikes/bridge/out/calibration.json"""
import json, sys, time
sys.stdout.reconfigure(encoding="utf-8")
from flylab import atlas, brain

READ = ["MDN_L", "MDN_R", "P9_L", "P9_R", "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R", "GF_L", "GF_R",
        "aDN1_shiu2024", "aDN2_shiu2024", "MN9_L", "MN9_R", "DNg07", "DNg11", "aBN1_L", "aBN1_R"]
def grp(res, g):
    ids = atlas.group_ids(g)
    r = brain.rates_for(res, ids)
    return round(sum(r.values()) / len(ids), 2)

conds = [(g, None) for g in ["MDN", "P9", "P9_L", "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R", "GF", "aDN1_shiu2024",
                             "aDN2_shiu2024", "MN9"]]
conds += [("LPLC2", None), ("LPLC2", "MDN"), ("sugar_GRN_shiu2024", None), ("JO_CE_shiu2024", None)]
brain.load_connectome()
out = []
for exc, sil in conds:
    t = time.time()
    res = brain.simulate(atlas.group_ids(exc), silence=atlas.group_ids(sil) if sil else None,
                         excite_rate_hz=150.0, duration_ms=1000.0, n_trials=3, n_threads=3)
    row = {"excite": exc, "silence": sil, "runtime_s": round(time.time() - t, 1), "n_active": res["n_active"],
           "rates": {g: grp(res, g) for g in READ}}
    print(json.dumps(row), flush=True)
    out.append(row)
json.dump(out, open("spikes/bridge/out/calibration.json", "w"), indent=1)
