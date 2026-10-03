"""Reviewer check: re-simulate a few screen experiments without the cache and compare to the cached values."""
import json, sys, time
from flylab import screen as S
cache = S._load_cache()
types = sys.argv[1].split(",")
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
t0 = time.perf_counter()
rows = S.brain_screen(types, ["MDN", "GF"], seed=seed, n_threads=2, use_cache=False)
for r in rows:
    c = cache.get(S._cache_key(r["cell_type"], 150.0, 500.0, 2, 0))
    print(f"{r['cell_type']:10s} seed{seed} MDN {r['target_rates']['MDN']:6.2f} (cache {c['group_rates']['MDN']:6.2f})  "
          f"GF {r['target_rates']['GF']:6.2f} (cache {c['group_rates']['GF']:6.2f})  {r['runtime_s']:.2f}s (cache {c['runtime_s']})")
print("total", round(time.perf_counter() - t0, 1))
