"""Audit: are the headline numbers in README.md / docs/CHALLENGE_COMPLIANCE.md equal to the committed results?

  uv run python spikes/audit/check_numbers.py

Every row compares a number that the README prints with the value read from data/benchmarks/*.json, data/*.json,
the live-run record or the Omnigent session file. Exit code 1 if any row differs. Also checks that relative links
in README.md and docs/*.md point to existing files.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RUN_F = ROOT / "runs/20261004-093251-which-visual-neurons-trigger-esc-9460"


def J(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


mdn, gf = J("data/benchmarks/screen_mdn.json"), J("data/benchmarks/screen_gf.json")
fl, emb = J("data/benchmarks/flight_validation.json"), J("data/benchmarks/embodied_validation.json")
mv, bv = J("data/benchmarks/movement_verifier.json"), J("data/benchmarks/brain_validation.json")
gt = J("data/ground_truth.json")["entries"]
groups = J("data/neurons.json")["groups"]
rec = [json.loads(line) for line in (RUN_F / "record.jsonl").read_text(encoding="utf-8").splitlines()]
sess = J("runs/20261004-093251-which-visual-neurons-trigger-esc-9460/omnigent/sessions.json")
frow = {r["condition"]: r for r in fl["rows"]}
gfm = fl["summary"]["gf_rate_hz_mean"]
dose = {r["lplc2_rate_hz"]: r for r in fl["lplc2_dose_response"]["rows"]}

rows = []  # (label, printed in README, value from files)


def chk(label, printed, actual, tol=0.0):
    ok = (abs(float(printed) - float(actual)) <= tol) if isinstance(printed, (int, float)) and not isinstance(printed, bool) else printed == actual
    rows.append((label, printed, actual, ok))


# discovery screen
chk("MDN hits", 12, mdn["search"]["n_hits"])
chk("GF hits", 17, gf["search"]["n_hits"])
chk("MDN guided first", 1.0, mdn["search"]["guided"]["experiments_to_first_hit"])
chk("MDN random first", 25.2, mdn["search"]["random"]["experiments_to_first_hit_expected"], 0.05)
chk("MDN reduction first", 25.15, mdn["search"]["random"]["experiments_to_first_hit_expected"] / mdn["search"]["guided"]["experiments_to_first_hit"], 0.01)
chk("MDN largest-first", 5.0, mdn["baselines"]["largest_type_first"]["guided_speedup_vs_baseline_first"])
chk("GF random first", 18.2, gf["search"]["random"]["experiments_to_first_hit_expected"], 0.05)
chk("GF reduction first", 18.17, gf["search"]["random"]["experiments_to_first_hit_expected"] / gf["search"]["guided"]["experiments_to_first_hit"], 0.01)
chk("GF largest-first", 3.0, gf["baselines"]["largest_type_first"]["guided_speedup_vs_baseline_first"])
chk("GF all-hits speed-up 0.97", 0.97, round(gf["search"]["random"]["experiments_to_all_hits_expected"] / gf["search"]["guided"]["experiments_to_all_hits"], 2), 0.005)
chk("MDN robust hits (second seed)", 11, mdn["seed_robustness"]["n_robust_hits"])
chk("GF robust hits (second seed)", 16, gf["seed_robustness"]["n_robust_hits"])
chk("exhaustive screen wall s", 404.5, mdn["search"]["exhaustive"]["wall_s"])
chk("MDN literature headline has 12 vs 109 (9.08x)", True, "12.0 experiments vs 109.0" in mdn["literature_headline"] and "9.08x" in mdn["literature_headline"])
chk("MDN confirmed driver LC16 16 vs 163.5 (10.2x)", True, "LC16 after 16 guided experiments vs 163.5" in mdn["literature_headline"] and "10.2x" in mdn["literature_headline"])
chk("GF literature headline 2 vs 109 (54.5x)", True, "2.0 experiments vs 109.0" in gf["literature_headline"] and "54.5x" in gf["literature_headline"])
chk("MDN 20 experiments find 12/12 vs 0.7", True, "finds 12/12 hits vs 0.7" in mdn["headline"])
chk("GF 20 experiments find 14/17 vs 1.0", True, "finds 14/17 hits vs 1.0" in gf["headline"])
# validation
chk("embodied comparable", 9, emb["summary"]["n_comparable"])
chk("embodied consistent", 8, emb["summary"]["n_consistent"])
chk("brian2 r", 0.9987, bv["brian2_crosscheck"]["pearson_r_nonstim_rates"])
chk("brian2 n_neurons", 426, bv["brian2_crosscheck"]["n_neurons"])
chk("brian2 n_trials", 16, bv["brian2_crosscheck"]["n_trials"])
chk("walking seed robustness all 1.0", True, all(r["body_seed_robustness"]["fraction_same_as_main"] == 1.0 for r in emb["rows"]))
# flight
chk("flight conditions as expected", 8, fl["summary"]["n_as_expected"])
chk("flight gt checks consistent", 6, fl["summary"]["n_consistent"])
chk("flight adapter hash", "02ef23b1b5d6dca2", fl["protocol"]["adapter_parameter_hash"])
for name, v in (("control", 0.0), ("GF_bilateral", 169.3), ("LPLC2_10Hz", 36.2), ("LPLC2_30Hz", 84.3), ("LPLC2_bilateral", 160.5),
                ("LPLC2_GF_silenced", 0.0), ("DNg02_only", 0.0), ("GF_DNg02", 160.3)):
    chk(f"GF mean {name}", v, gfm[name], 0.05)
for name, v in (("GF_bilateral", 18.2), ("GF_DNg02", 183.4), ("control", 0.0)):
    chk(f"max height {name}", v, abs(frow[name]["body"]["max_height_mm"]), 0.05)
chk("LPLC2_bilateral verifier uncertain", "uncertain", fl["summary"]["verification"]["LPLC2_bilateral"])
chk("other 7 verifier correct", 7, sum(v == "correct" for v in fl["summary"]["verification"].values()))
chk("takeoff threshold LPLC2 Hz", 23.7, fl["lplc2_dose_response"]["takeoff_threshold_lplc2_rate_hz_interpolated"], 0.05)
chk("dose 20 Hz activation", 0.45, dose[20.0]["takeoff_activation"], 0.005)
chk("dose 25 Hz activation", 0.52, dose[25.0]["takeoff_activation"], 0.005)
chk("flight validate wall s (about 6 min)", 370.1, fl["summary"]["total_wall_s"], 0.05)
# verifier benchmark
chk("verifier kinematic known 10/10", 10, mv["summary"]["kinematic_correct_on_known_label"])
chk("verifier kinematic opposite 10/10", 10, mv["summary"]["kinematic_incorrect_on_opposite_label"])
chk("verifier vision 4/6", (4, 6), (mv["summary"]["vision_exact_on_unique_videos"], mv["summary"]["vision_unique_videos"]))
chk("verifier final 8 correct / 2 uncertain", (8, 2), (mv["summary"]["final_correct_on_known_label"], mv["summary"]["final_uncertain_on_known_label"]))
# atlas / ground truth
chk("neuron groups", 63, len(groups))
chk("ground-truth entries", 26, len(gt))
chk("ground-truth distinct DOIs", 15, len({e["citation"]["doi"].lower() for e in gt}))
# live run F
chk("run F record events", 33, len(rec))
chk("run F root cost", 1.11, sess["root_total_cost_usd"], 0.005)
chk("run F sessions", 12, len(sess["sessions"]))
txt = " ".join(str(e.get("content")) for e in rec)
chk("run F parallel flights 21.0 s vs 30.58", True, "wall 21.0 s vs sum of runtimes 30.58 s" in txt)
chk("run F verifier flight_01 uncertain", True, "FINAL UNCERTAIN" in txt)
dec = [e for e in rec if e["type"] == "decision" and e["agent"] == "supervisor"]
chk("run F supervisor reopens an assumption", True, bool(dec[0]["data"].get("reopens_assumption")))

# links
bad_links = []
for md in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
    for m in re.finditer(r"\]\((?!https?://|#|mailto:)([^)#\s]+)", md.read_text(encoding="utf-8")):
        if not (md.parent / m.group(1)).resolve().exists():
            bad_links.append(f"{md.relative_to(ROOT)} -> {m.group(1)}")

rows = [r for r in rows if r]
width = max(len(r[0]) for r in rows)
for label, printed, actual, ok in rows:
    print(("ok   " if ok else "FAIL ") + label.ljust(width), "| README:", printed, "| files:", actual)
print(f"\n{sum(r[3] for r in rows)}/{len(rows)} numbers match; broken relative links: {bad_links or 'none'}")
sys.exit(0 if all(r[3] for r in rows) and not bad_links else 1)
