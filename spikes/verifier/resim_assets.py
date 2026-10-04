"""Re-simulate the committed embodied validation videos' body runs (same drive, seed 0,
control_every 5, 1 s) to get raw trajectories for the kinematic verifier test.
Output: spikes/verifier/out/body_resim.json  (no video rendering; videos come from assets/embodied)."""
import json
import time
from pathlib import Path
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flylab import body  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
bench = json.loads((ROOT / "data/benchmarks/embodied_validation.json").read_text(encoding="utf-8"))
proto = bench["protocol"]
out = {}
for row in bench["rows"]:
    t0 = time.time()
    res = body.simulate_walk(row["drive"], duration_s=float(proto["body_duration_s"]), render_path=None,
                             seed=int(proto["seed"]))
    out[row["condition"]] = {"drive": row["drive"], "benchmark_behavior": row["behavior"],
                             "benchmark_metrics": {k: row["body_metrics"].get(k) for k in
                                                   ("forward_disp_mm", "heading_change_deg", "behavior")},
                             "result": res, "video": row["video"], "gt_id": row.get("gt_id"), "wall_s": round(time.time() - t0, 1)}
    print(row["condition"], res["behavior"], res["forward_disp_mm"], res["heading_change_deg"], round(time.time() - t0, 1), flush=True)
p = ROOT / "spikes/verifier/out/body_resim.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out), encoding="utf-8")
print("wrote", p)
