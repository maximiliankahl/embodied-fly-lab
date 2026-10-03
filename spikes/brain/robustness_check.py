"""Reviewer robustness checks for flylab.brain (API edge cases, silencing semantics, stress speed).
    uv run python spikes/brain/robustness_check.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flylab import brain  # noqa: E402

conn = brain.load_connectome()
sugar = [i for i in brain.SUGAR_GRNS_V630 if i in conn.id2idx]
mn9 = list(brain.MN9.values())


def expect_error(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError as e:
        print("  OK raises ValueError:", str(e)[:90])
        return
    raise AssertionError("expected ValueError")


print("1) bad inputs")
expect_error(brain.simulate, sugar, duration_ms=0)
expect_error(brain.simulate, sugar, n_trials=0)
expect_error(brain.simulate, sugar, excite_rate_hz=float("nan"))
expect_error(brain.simulate, sugar, params={"w_syn": 0.3})

print("2) numpy / pandas / str / unknown IDs")
import pandas as pd  # noqa: E402
r1 = brain.simulate(np.array(sugar), duration_ms=200, n_trials=2, seed=4)
r2 = brain.simulate([str(i) for i in sugar] + ["not_an_id", 123], duration_ms=200, n_trials=2, seed=4)
r3 = brain.simulate(pd.Series(sugar), silence=np.array([], dtype=np.int64), duration_ms=200, n_trials=2, seed=4)
assert r1["rates"] == r2["rates"] == r3["rates"], "same IDs in different containers must give same result"
print("  OK identical results; unknown_ids =", r2["unknown_ids"])
assert brain.rates_for(r1, None) == {}

print("3) silencing semantics")
base = brain.simulate(sugar, excite_rate_hz=100, duration_ms=500, n_trials=3, seed=2)
sil = brain.simulate(sugar, excite_rate_hz=100, silence=mn9[:1], duration_ms=500, n_trials=3, seed=2)
print("  MN9 rates (no silencing):", brain.rates_for(base, mn9))
print("  MN9 rates (MN9_a silenced):", brain.rates_for(sil, mn9), " raw spikes of silenced:", sil["silenced_spiking_rates"])
assert brain.rates_for(sil, mn9[:1])[str(mn9[0])] == 0.0
assert set(sil["silenced_spiking_rates"]) <= {str(mn9[0])}

print("4) determinism")
a = brain.simulate(sugar, excite_rate_hz=100, duration_ms=300, n_trials=3, seed=9)
b = brain.simulate(sugar, excite_rate_hz=100, duration_ms=300, n_trials=3, seed=9)
assert a["rates"] == b["rates"]
print("  OK same seed -> same rates")

print("5) stress: 2000 random neurons at 150 Hz, 2 trials x 300 ms")
rng = np.random.default_rng(0)
big = conn.ids[rng.choice(conn.N, 2000, replace=False)].tolist()
t = time.perf_counter()
s = brain.simulate(big, excite_rate_hz=150, duration_ms=300, n_trials=2, seed=0)
print(f"  {time.perf_counter() - t:.1f}s, n_active={s['n_active']}, "
      f"=> {s['runtime_s'] / 2 / 0.3:.1f} s per trial per simulated second")
print("ALL OK")
