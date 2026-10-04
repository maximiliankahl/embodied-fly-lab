"""Test flylab.verify on the committed embodied videos (assets/embodied/*.mp4) with known labels.
Kinematics: raw trajectories re-simulated with the same drive/seed (spikes/verifier/out/body_resim.json,
bit-identical metrics to data/benchmarks/embodied_validation.json). Vision: blind Claude call per unique video
(cached by SHA-256). Reference label = body-classifier label of the benchmark run.
Writes data/benchmarks/movement_verifier.json."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flylab import verify  # noqa: E402

resim = json.loads((ROOT / "spikes/verifier/out/body_resim.json").read_text(encoding="utf-8"))
rows, spend_in, spend_out = [], 0, 0
OPPOSITE = {"forward": "backward", "backward": "forward", "turn_left": "turn_right", "turn_right": "turn_left", "stop": "forward"}
for cond, r in resim.items():
    res = dict(r["result"])
    res["video"] = r["video"]
    label = r["benchmark_behavior"]
    sheet = str(ROOT / f"spikes/verifier/out/{cond}_contact.jpg")
    t0 = time.time()
    v = verify.verify_movement(res, label, mode="walk", use_vision=True, sheet_path=sheet)
    neg = verify.verify_movement(res, OPPOSITE[label], mode="walk", use_vision=True, sheet_path=sheet)
    vis = v.get("vision") or {}
    if vis and not vis.get("cached"):
        spend_in += (vis.get("usage") or {}).get("input_tokens") or 0
        spend_out += (vis.get("usage") or {}).get("output_tokens") or 0
    rows.append({"condition": cond, "video": r["video"], "known_label": label, "gt_id": r.get("gt_id"),
                 "kinematic_verdict": v["kinematic"]["verdict"], "kinematic_reason": v["kinematic"]["reason"],
                 "recomputed": v["kinematic"]["recomputed"],
                 "vision_observed": vis.get("observed_behavior"), "vision_confidence": vis.get("confidence"),
                 "vision_correct": vis.get("observed_behavior") == label if vis else None,
                 "vision_observations": vis.get("observations"), "vision_model": vis.get("model"),
                 "final_verdict": v["final_verdict"], "agreement": v["agreement"],
                 "negative_control": {"expected": OPPOSITE[label], "kinematic": neg["kinematic"]["verdict"],
                                      "final": neg["final_verdict"]},
                 "video_sha256": vis.get("video_sha256"), "wall_s": round(time.time() - t0, 1),
                 "vision_error": v.get("vision_error")})
    print(cond, label, "| kin", v["kinematic"]["verdict"], "| vis", vis.get("observed_behavior"), vis.get("confidence"),
          "| final", v["final_verdict"], "| neg", neg["kinematic"]["verdict"], neg["final_verdict"], flush=True)

uniq = {}
for row in rows:
    uniq.setdefault(row["video_sha256"], row)
u = list(uniq.values())
summary = {
    "n_runs": len(rows), "n_unique_videos": len(u),
    "kinematic_correct_on_known_label": sum(r["kinematic_verdict"] == "correct" for r in rows),
    "kinematic_incorrect_on_opposite_label": sum(r["negative_control"]["kinematic"] == "incorrect" for r in rows),
    "vision_exact_on_unique_videos": sum(bool(r["vision_correct"]) for r in u),
    "vision_unique_videos": len(u),
    "final_correct_on_known_label": sum(r["final_verdict"] == "correct" for r in rows),
    "final_uncertain_on_known_label": sum(r["final_verdict"] == "uncertain" for r in rows),
    "final_correct_on_opposite_label": sum(r["negative_control"]["final"] == "correct" for r in rows),
    "new_api_tokens_this_run": {"input": spend_in, "output": spend_out},
}
out = {"what": "Movement verifier test on committed embodied videos (known labels = body-classifier labels)",
       "created": time.strftime("%Y-%m-%d %H:%M:%S"), "verifier": "flylab/verify.py",
       "prompt_version": verify.PROMPT_VERSION, "vision_min_confidence": verify.VISION_MIN_CONFIDENCE,
       "caveats": ["5 of 10 videos are byte-identical (zero drive -> fly stands still), so vision accuracy is reported "
                   "on unique videos", "known labels come from flylab.body.classify on the same trajectories; the "
                   "verifier's kinematic check uses its own formulas/thresholds but the same physics, so agreement is a "
                   "consistency check, not ground truth", "top-down tracking camera + periodic checkerboard floor makes "
                   "translation direction hard to read from keyframes"],
       "summary": summary, "rows": rows}
(ROOT / "data/benchmarks/movement_verifier.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
