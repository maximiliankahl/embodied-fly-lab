"""Cross-check flylab.brain against the ORIGINAL Brian2 model.py (Shiu et al. 2024) on a subnetwork.

The full brain is too slow in Brian2's numpy runtime, so we cut out the subnetwork of neurons that are
active when the sugar GRNs are driven (plus MN9) and run *both* simulators on exactly that subnetwork
with the original model.py code path (create_model / poi / run_trial).

Run (Brian2 only in an ephemeral overlay, not added to the project):
    uv run --with brian2 --with joblib python spikes/brain/crosscheck_brian2.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flylab import brain  # noqa: E402

OUT = ROOT / "spikes" / "brain" / "out" / "crosscheck"
OUT.mkdir(parents=True, exist_ok=True)
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 16

full = brain.load_connectome()
sugar = [i for i in brain.SUGAR_GRNS_V630 if i in full.id2idx]
mn9 = list(brain.MN9.values())

# 1) subnetwork = neurons active in a full-brain sugar run at 200 Hz (+ MN9)
res = brain.simulate(sugar, excite_rate_hz=200.0, n_trials=3, duration_ms=1000.0, seed=3)
keep_ids = sorted(set(int(k) for k in res["rates"]) | set(sugar) | set(mn9))
keep = np.array([full.id2idx[i] for i in keep_ids])
old2new = -np.ones(full.N, np.int64)
old2new[keep] = np.arange(keep.size)
pre_l, post_l, w_l = [], [], []
for new_pre, old_pre in enumerate(keep):
    a, b = full.indptr[old_pre], full.indptr[old_pre + 1]
    posts = old2new[full.indices[a:b]]
    m = posts >= 0
    pre_l.append(np.full(m.sum(), new_pre)); post_l.append(posts[m]); w_l.append(full.syn[a:b][m])
pre = np.concatenate(pre_l); post = np.concatenate(post_l); w = np.concatenate(w_l).astype(np.int64)
ids = np.array(keep_ids, np.int64)
print(f"subnetwork: {ids.size} neurons, {pre.size} connections")

comp_p, con_p = OUT / "sub_completeness.csv", OUT / "sub_connectivity.parquet"
pd.DataFrame({"Completed": True}, index=pd.Index(ids)).to_csv(comp_p)
pd.DataFrame({"Presynaptic_ID": ids[pre], "Postsynaptic_ID": ids[post], "Presynaptic_Index": pre,
              "Postsynaptic_Index": post, "Connectivity": np.abs(w), "Excitatory": np.sign(w),
              "Excitatory x Connectivity": w}).to_parquet(con_p)

# 2) our simulator on the subnetwork
order = np.argsort(pre, kind="stable")
indptr = np.zeros(ids.size + 1, np.int64)
np.cumsum(np.bincount(pre, minlength=ids.size), out=indptr[1:])
brain._CONN = brain.Connectome(ids=ids, indptr=indptr, indices=post[order].astype(np.int32),
                               syn=w[order].astype(np.float32), id2idx={int(r): i for i, r in enumerate(ids)})
t = time.time()
ours = brain.simulate(sugar, excite_rate_hz=RATE, n_trials=N_TRIALS, duration_ms=1000.0, seed=11)
print(f"ours: {time.time() - t:.1f}s")

# 3) original Brian2 model.py on the same subnetwork
sys.path.insert(0, str(ROOT / "data" / "raw"))
from brian2 import Hz, prefs  # noqa: E402
prefs.codegen.target = "numpy"
import model  # noqa: E402  (original model.py from the reference repo)
from joblib import Parallel, delayed  # noqa: E402

params = dict(model.default_params)
params["r_poi"] = RATE * Hz
id2i = {int(r): i for i, r in enumerate(ids)}
exc = [id2i[i] for i in sugar]


RAW_DIR = str(ROOT / "data" / "raw")


def one(seed):
    import sys as _s
    if RAW_DIR not in _s.path:
        _s.path.insert(0, RAW_DIR)
    import model as _m
    from brian2 import seed as bseed, prefs as p2
    p2.codegen.target = "numpy"
    bseed(seed)
    trn = _m.run_trial(exc, [], [], comp_p, con_p, params)
    return {int(k): len(v) for k, v in trn.items()}


t = time.time()
trials = Parallel(n_jobs=min(N_TRIALS, 12))(delayed(one)(s) for s in range(N_TRIALS))
print(f"brian2: {time.time() - t:.1f}s")
b2 = np.zeros((N_TRIALS, ids.size))
for k, d in enumerate(trials):
    for i, n in d.items():
        b2[k, i] = n
b2_mean = b2.mean(0)
our_mean = np.array([ours["rates"].get(str(int(r)), 0.0) for r in ids])

nonstim = np.ones(ids.size, bool); nonstim[exc] = False
r = np.corrcoef(b2_mean[nonstim], our_mean[nonstim])[0, 1]
rep = {
    "rate_hz": RATE, "n_trials": N_TRIALS, "n_neurons": int(ids.size), "n_connections": int(pre.size),
    "pearson_r_nonstim_rates": float(r),
    "total_rate_nonstim": {"brian2": float(b2_mean[nonstim].sum()), "ours": float(our_mean[nonstim].sum())},
    "n_active_nonstim": {"brian2": int((b2_mean[nonstim] > 0).sum()), "ours": int((our_mean[nonstim] > 0).sum())},
    "stim_mean_rate": {"brian2": float(b2_mean[exc].mean()), "ours": float(our_mean[exc].mean())},
    "mn9": {str(i): {"brian2_mean": float(b2_mean[id2i[i]]), "brian2_std": float(b2[:, id2i[i]].std()),
                     "ours_mean": float(our_mean[id2i[i]]), "ours_std": float(ours["std"].get(str(i), 0.0))}
            for i in mn9},
}
print(json.dumps(rep, indent=2))
(OUT / f"crosscheck_{int(RATE)}Hz.json").write_text(json.dumps(rep, indent=2))
