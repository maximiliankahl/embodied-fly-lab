"""Render the screen section with a SYNTHETIC benchmark doc (schema of flylab.screen.benchmark_search) - test only."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from pathlib import Path
out = Path(ROOT) / "spikes/dashboard/out/synthetic_bench"; out.mkdir(parents=True, exist_ok=True)
table = [{"cell_type": f"LC{i}", "n": 50, "connectome_rank": i, "score": 1.0 / i, "direct_syn": 100 - i, "rate_hz": 60.0 / i,
          "hit": i in (1, 3, 7), "literature_known": i == 3, "runtime_s": 2.5} for i in range(1, 41)]
doc = {"name": "screen_synthetic", "target_group": "MDN", "headline": "SYNTHETIC headline", "literature_headline": "SYNTHETIC lit",
       "search": {"n_candidates": 40, "n_hits": 3, "guided": {"experiments_to_first_hit": 1.0, "experiments_to_all_hits": 7.0,
                  "wall_s_to_first_hit": 4.2}, "random": {"experiments_to_first_hit_expected": 10.25, "experiments_to_all_hits_expected": 30.75,
                  "wall_s_to_first_hit_expected": 25.6}, "exhaustive": {"experiments": 40, "wall_s": 100.0},
                  "reduction_factor_first_hit": 10.25, "reduction_factor_all_hits": 4.39, "reduction_factor_vs_exhaustive_all_hits": 5.71},
       "ranking": {"spearman_score_vs_rate": 0.8}, "definitions": {"hit": "x"}, "limitations": ["a"],
       "sensitivity_threshold": {"1Hz": {"n_hits": 5, "guided_first": 1}, "5Hz": {"n_hits": 3, "guided_first": 1}},
       "literature": {"known_hits": [{"cell_type": "LC3", "in_candidates": True, "in_silico_rate_hz": 20.0, "in_silico_hit": True,
                      "connectome_rank": 3, "connectome_score": 0.33, "evidence": {"gt": ["gt20"], "why": "synthetic"}}],
                      "recall_in_silico": 1.0, "precision": "n/a"},
       "novel_in_silico_hits": {"note": "agent-generated", "cell_types": ["LC1", "LC7"], "rates_hz": {"LC1": 60, "LC7": 8.6}},
       "timing": {"wall_s_per_experiment_mean": 2.5}, "table": table}
(out / "screen_synthetic.json").write_text(json.dumps(doc), encoding="utf-8")
from streamlit.testing.v1 import AppTest
def run_page():
    from pathlib import Path
    from flylab import ui, ui_validation
    d = Path(ui.ROOT) / "spikes/dashboard/out/synthetic_bench"
    ui.benchmark_files = lambda prefix="": sorted(d.glob(f"{prefix}*.json"))
    ui_validation._screen_section()
    ui_validation._speed_section()
at = AppTest.from_function(run_page, default_timeout=60).run()
print("exc", [str(e.value)[:1500] for e in at.exception])
print("metrics", [(m.label, m.value) for m in at.metric])
print("markdown", [m.value[:120] for m in at.markdown])
print("captions", [c.value[:150] for c in at.caption])
