"""Real (non-mock) check of run_experiments_parallel: 2 brain runs + 1 embodied run in a scratch run (deleted after)."""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flylab import record, tools  # noqa: E402
rid = record.new_run("SCRATCH parallel test (deleted)", set_current=False)
try:
    t0 = time.time()
    out = tools.run_experiments_parallel([
        {"kind": "brain", "excite_groups": ["MDN"], "duration_ms": 500, "n_trials": 2, "seed": 0},
        {"kind": "brain", "excite_groups": ["P9"], "duration_ms": 500, "n_trials": 2, "seed": 0},
        {"kind": "embodied", "excite_groups": ["MDN"], "duration_ms": 500, "n_trials": 2, "duration_s": 1.0, "seed": 0}],
        max_workers=3, run_id=rid)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    for r in out["results"]:
        print(r.get("kind"), r.get("ok"), r.get("artifact"), r.get("runtime_s"), r.get("brain_runtime_s"), r.get("body_runtime_s"), r.get("behavior"), r.get("error"))
    emb = [r for r in out["results"] if r.get("kind") == "embodied"][0]
    v = tools.verify_movement(emb["artifact"], ground_truth_id="gt01_mdn_activate_backward", run_id=rid)
    print("VERIFY", v.get("final_verdict"), v.get("reason"), v.get("contact_sheet"), v.get("error"))
    Path(ROOT / "spikes/verifier/out/parallel_test.json").write_text(json.dumps({"parallel": {k: v2 for k, v2 in out.items() if k != "results"}, "verify": v}, indent=1, default=str))
    shutil.copy(record.artifacts_dir(rid) / Path(v["contact_sheet"]).name, ROOT / "spikes/verifier/out/parallel_embodied_contact.jpg") if v.get("contact_sheet") else None
finally:
    shutil.rmtree(record.RUNS_DIR / rid, ignore_errors=True)
