"""Notebook page with a SYNTHETIC run (scratch dir) whose embodied result points to an existing video - test only."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from pathlib import Path
runs = Path(ROOT) / "spikes/dashboard/out/synthetic_runs"; (runs / "synthetic_real_like").mkdir(parents=True, exist_ok=True)
v = json.load(open("data/benchmarks/embodied_validation.json"))
row = next(r for r in v["rows"] if r["condition"] == "MDN_bilateral")
ev = [{"seq": 0, "agent": "human", "type": "question", "content": "Which DNs drive backward walking?", "data": None, "citations": []},
      {"seq": 1, "agent": "runner", "type": "experiment_result", "content": "Embodied sim: excite=['MDN'] -> backward",
       "data": {"kind": "embodied", "excite": ["MDN"], "silence": [], "descending_group_rates_hz": row["key_group_rates_hz"],
                "drive": {**row["drive"], "explain": {"formula": "x"}}, "behavior": row["behavior"],
                "forward_disp_mm": row["body_metrics"]["forward_disp_mm"], "heading_change_deg": row["body_metrics"]["heading_change_deg"],
                "mean_speed_mm_s": row["body_metrics"]["mean_speed_mm_s"], "video": row["video"], "artifact": None, "mock": False,
                "bridge": {"bridge_input": "neuron_rates"}}, "citations": []},
      {"seq": 2, "agent": "analysis", "type": "analysis", "content": "gt01 -> consistent",
       "data": {"ground_truth_id": "gt01_mdn_activate_backward", "verdict": "consistent", "expected": "backward", "observed": "backward",
                "effect": "induce", "surprise": False}, "citations": ["10.1126/science.1249964"]}]
(runs / "synthetic_real_like" / "record.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
from streamlit.testing.v1 import AppTest
def run_page():
    from pathlib import Path
    from flylab import ui, ui_notebook
    ui.RUNS = Path(ui.ROOT) / "spikes/dashboard/out/synthetic_runs"
    ui_notebook.page()
at = AppTest.from_function(run_page, default_timeout=60).run()
print("exc", [str(e.value)[:1500] for e in at.exception])
print("videos", len(at.get("video")), "metrics", [(m.label, m.value) for m in at.metric][:12])
print("mock banner?", any("MOCK DATA" in m.value for m in at.markdown))
