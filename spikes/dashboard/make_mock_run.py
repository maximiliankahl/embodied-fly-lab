"""Create runs/demo_mock_<name>/ : a MOCK research run (FLYLAB_MOCK=1) for dashboard testing.

Everything in it is synthetic (flylab.tools mock mode); the dashboard badges it as MOCK.
Usage (from 02_App): uv run python spikes/dashboard/make_mock_run.py
"""
import os
import shutil
import sys

os.environ["FLYLAB_MOCK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flylab import record, tools  # noqa: E402

RID = "demo_mock_backward_walking"
shutil.rmtree(record.RUNS_DIR / RID, ignore_errors=True)
Q = "MOCK DEMO: Which descending neurons drive backward walking, and does steering follow P9 laterality?"
record.log_event(RID, "human", "question", Q)
record.log_event(RID, "supervisor", "note", "Run opened by orchestrator (MOCK demo run for dashboard testing).",
                 {"question": Q, "mock": True})
tools.search_literature("moonwalker descending neurons backward walking", n=2, agent="literature", run_id=RID)
tools.log_note("MOCK evidence summary: literature reports MDN activation evokes backward walking; P9 drives "
               "forward walking with ipsilateral turning. Ground truth: gt_mock_mdn_activate, gt_mock_p9l_activate.",
               agent="literature", event_type="evidence", run_id=RID)
tools.log_hypothesis("Activating MDN in the connectome model produces backward walking in the body.", "activate",
                     ["MDN"], "backward", "MOCK rationale: MDN is the canonical backward-walking command.", 0.7,
                     agent="hypothesis", run_id=RID)
tools.log_hypothesis("Activating left P9 produces a left turn (ipsilateral steering).", "activate", ["P9_L"],
                     "turn_left", "MOCK rationale: P9 drives forward walking with ipsilateral turning.", 0.55,
                     agent="hypothesis", run_id=RID)
tools.log_experiment_plan([
    {"id": "E1", "kind": "embodied", "excite_groups": ["MDN"], "tests_hypothesis": "H1",
     "expected_information_gain": "high", "est_cost_units": 3.0, "why": "closes the brain->body loop for H1"},
    {"id": "E2", "kind": "brain", "excite_groups": ["MDN"], "tests_hypothesis": "H1",
     "expected_information_gain": "medium", "est_cost_units": 0.8, "why": "cheap, but no behavior readout"},
    {"id": "E3", "kind": "embodied", "excite_groups": ["P9_L"], "tests_hypothesis": "H2",
     "expected_information_gain": "high", "est_cost_units": 3.0, "why": "tests steering laterality"},
], "E1", "MOCK: E1 gives a behavior-level answer for H1 at moderate cost; E3 follows within budget.", 10.0,
    agent="planner", run_id=RID)
tools.request_approval("2 embodied runs: MDN activation (E1), P9_L activation (E3)", "tests H1 and H2", 6.0,
                       agent="safety", run_id=RID)
tools.run_brain_experiment(["MDN"], agent="runner", run_id=RID)
tools.run_embodied_experiment(["MDN"], agent="runner", run_id=RID)
tools.run_embodied_experiment(["P9_L"], agent="runner", run_id=RID)
tools.compare_to_ground_truth("backward", "gt_mock_mdn_activate", "embodied_01", agent="analysis", run_id=RID)
tools.compare_to_ground_truth("turn_right", "gt_mock_p9l_activate", "embodied_02", agent="analysis", run_id=RID)
tools.log_decision("Surprise on P9_L: run P9_R and check the bridge's turn sign before cycle 2",
                   "MOCK: embodied_02 turned right although the literature predicts an ipsilateral (left) turn.",
                   "embodied P9_R activation + inspect bridge sign convention", "laterality of P9 -> turn mapping",
                   agent="supervisor", run_id=RID)
tools.log_note("MOCK final report: 1 consistent, 1 inconsistent (laterality). All numbers are MOCK.",
               agent="record_keeper", run_id=RID)
print(record.summarize(RID))
