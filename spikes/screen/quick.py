import time
from flylab import brain, atlas
df = atlas.load_annotations()
conn = brain.load_connectome()
tg = {g: atlas.group_ids(g) for g in ["MDN", "GF", "P9", "DNa02", "MN9"]}
for ct in ["LC16", "LPLC2", "LC4", "LC12", "LC10a", "LC6"]:
    ids = [int(i) for i in df[df.cell_type == ct].root_id if int(i) in conn.id2idx]
    t = time.perf_counter()
    r = brain.simulate(ids, excite_rate_hz=150, duration_ms=500, n_trials=2, seed=0, n_threads=2)
    rt = {g: round(sum(brain.rates_for(r, v).values()) / len(v), 1) for g, v in tg.items()}
    print(ct, len(ids), rt, "active", r["n_active"], f"{time.perf_counter()-t:.1f}s", flush=True)
