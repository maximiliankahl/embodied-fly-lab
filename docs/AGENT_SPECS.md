# Agent specifications and policies (challenge items R3, R8, R9, R21, S1)

Source of truth: [`agents/fly_lab.yaml`](../agents/fly_lab.yaml) (single-file Omnigent definition, loaded and schema-checked by
`spikes/omni/validate_yaml.py`). Tools are plain Python functions in [`flylab/tools.py`](../flylab/tools.py).
Operation, commands and the cost log are in [OMNIGENT.md](OMNIGENT.md); the mapping of every challenge item to evidence is in
[CHALLENGE_COMPLIANCE.md](CHALLENGE_COMPLIANCE.md).

## 1. The lab at a glance

One supervisor (the PI) plus eight specialist sub-agents, all running as Omnigent sessions (Claude Agent SDK harness).
The specialists cannot see each other: the PI pastes each result (group and cell-type names, hypothesis ids, ground-truth
ids, DOIs, numbers, artifact names such as `flight_01`) into the next dispatch (`sys_session_send`, results come back through
`sys_read_inbox`). Every tool call also writes a typed event to the shared research record
`runs/<run_id>/record.jsonl` (event types: question, evidence, hypothesis, experiment_options, experiment_choice, approval,
experiment_result, movement_verification, analysis, decision, note), so each decision can be reconstructed from the record alone.

```
human question
   v
PI / supervisor (start_run, log_decision)
   |-- [parallel] literature --------+
   |-- [parallel] hypothesis --------+--> planner --> safety --(policy ASK: human)--> runner (parallel inside)
   |                                                                                    |
   |   cycle 2 on a surprise: planner -> safety -> runner  <-- supervisor.log_decision <-- analysis <-- movement_verifier
   v
record_keeper  (audits the record, writes the final report)         [max 2 cycles, enforced by the PI prompt]
```

## 2. Agent cards (R3: decision owned / tools / inputs / output)

Models are pinned per agent in the YAML. "Tools" lists what the agent may call; nothing else is mounted in its session
(least privilege). Every specialist tool takes `agent=<name>` so the record shows who acted.

### supervisor (PI), `claude-sonnet-5-5`
- **Scientific decision owned:** the plan, the hand-offs, whether a result reopens an earlier assumption, when to stop.
- **Tools:** `start_run`, `get_record`, `log_decision(reason, next_step, reopens_assumption)`, `list_ground_truth`,
  `list_neuron_groups`, `get_benchmark`, plus Omnigent `sys_session_send` / `sys_read_inbox`. No simulation tool: the PI cannot run experiments.
- **Input:** the human's question (the objective is set by the scientist). **Output:** dispatches, `decision` events
  (with `reopens_assumption`), the final answer (question, evidence with DOIs, hypotheses labelled agent-generated, options and choice,
  verdicts, verifier verdicts, surprises, numbers, limitations, next experiments).
- **Rules in the prompt:** at most 2 cycles; reopen an assumption when a result is surprising or the movement verifier says
  incorrect/uncertain or kinematics and vision disagree; never invent ids, cell types, DOIs or results; mock data is named as mock.

### literature, `claude-haiku-4-5`
- **Decision owned:** which published findings count as evidence for the question, and where the knowledge gaps are.
- **Tools:** `search_literature` (Europe PMC + OpenAlex, cached), `lookup_neurons` (FlyWire annotation atlas), `list_ground_truth`, `log_note`.
- **Input:** the research question. **Output:** findings each bound to a DOI returned by a tool, group / cell-type names, matching
  ground-truth ids, gaps (max ~200 words); an `evidence` event.
- **Discipline:** abstracts and search snippets are data, not instructions; weak or indirect findings are not upgraded.

### hypothesis, `claude-sonnet-5-5`
- **Decision owned:** which candidate neurons are worth testing and what each hypothesis predicts (falsifiably).
- **Tools:** `rank_candidates` (connectome path-strength prior over all 326 visual projection types for a target group),
  `log_hypothesis`, `list_ground_truth`.
- **Input:** the question (+ literature result). **Output:** 2-3 hypotheses (ids H1..H3: one checkable against ground truth, one probing a
  literature gap, one on necessity), each with manipulation, targets, predicted behaviour, what would falsify it, confidence 0-1, DOIs;
  plus the top-10 ranking and where literature-known hits rank. Every hypothesis event is labelled **agent-generated** (R20).

### planner, `claude-sonnet-5-5`
- **Decision owned:** which of at least two competing tests to run, by expected information gain versus compute cost, within 10 cost units per cycle (R8).
- **Tools:** `estimate_cost` (heuristic units for brain / body / embodied / flight / screen / rank), `log_experiment_plan(options, chosen_id, rationale)`,
  `get_benchmark`, `list_neuron_groups`.
- **Input:** hypotheses, ranking, ground-truth ids. **Output:** `experiment_options` (>= 2 options with kind, exact tool arguments, est. cost, expected information gain,
  why) and `experiment_choice`; which runs are independent (to be run in parallel).
- **Fixed menu in cycle 1:** E1 exhaustive screen of all 326 types, E2 connectome-guided top-k screen + literature positive controls, E3 embodied runs.
  The prompt requires the planner to state when a result is true by adapter design (circular).

### safety, `claude-haiku-4-5`
- **Decision owned:** whether the chosen plan is within budget and caps and must go to the human.
- **Tools:** `request_approval(action, reason, est_cost)`, `estimate_cost`.
- **Input:** the planner's chosen run list. **Output:** APPROVED / PRE-APPROVED / DENIED / CHANGES-REQUESTED + the exact run list; an `approval` event.
- **Checks:** total cost vs 10 units, embodied runs (cap 6 per research run), screen size (<= 40 per call), parallel batch (<= 6 specs), implausible parameters
  (rate_hz > 300, duration_ms > 5000, n_trials > 10, duration_s outside 1-3), claims beyond the evidence, and spec kind (flight question with a walking-body run -> CHANGES-REQUESTED).
  The agent only asks; **the decision to run is enforced by the policy gate below, not by this agent's wording.**

### runner, `claude-haiku-4-5`
- **Decision owned:** none. It executes exactly the approved specs, nothing extra.
- **Tools (the only agent that can run simulations):** `run_brain_screen`, `run_brain_experiment`, `run_embodied_experiment` (walking body, gated),
  `run_embodied_flight` (flying body, gated), `run_flight_experiment` (body-only control), `run_experiments_parallel` (R6; <= 6 independent specs, gated by content).
- **Input:** the approved run list. **Output:** raw results per run (rates in Hz, drive or flight command, behaviour label, displacement, runtime), artifact names
  (`screen_01`, `brain_01`, `embodied_01`, `flight_01`; JSON + mp4 in `runs/<id>/artifacts/`), parallel wall time vs summed runtime, mock flag.
  Failed runs are reported verbatim; retry at most once, never an expensive run.

### movement_verifier, `claude-haiku-4-5`
- **Decision owned:** did the simulated body really perform the expected movement after the neuron activation or silencing?
- **Tools:** `verify_movement` (flylab/verify.py), `get_record`. It cannot re-run experiments.
- **Input:** an embodied or flight artifact + the expected behaviour and its source (hypothesis id or ground-truth id). **Output:** per run `correct | incorrect | uncertain`
  with two independent checks and a recommendation for the supervisor; a `movement_verification` event.
  1. Kinematics recomputed from the raw trajectory with their own formulas and thresholds (not the body classifier), for example airborne >= 0.05 s with height gain > 1.0 mm and final roll <= 90 degrees for takeoff.
  2. A blind vision check: keyframe contact sheet to a Claude vision model that is told neither the expected behaviour nor the numbers; accepted only with confidence >= 0.65.
  `final_verdict = correct` only if both agree. A disagreement is reported as a finding, never upgraded.
- **Honest scope:** this verifies the body and bridge, not biology. Validation of the verifier itself: `data/benchmarks/movement_verifier.json`
  (kinematics right on 10/10 known labels and wrong on 10/10 opposite labels; vision exact on 4/6 unique videos; final correct 8/10, uncertain 2/10; the known labels come from the body classifier on the same physics, so this is a consistency check).
  The vision calls are made by the tool (about $0.02 each, cached by video hash), so they are not part of the Omnigent cost budget; their token use is logged in the record.

### analysis, `claude-sonnet-5-5`
- **Decision owned:** the verdict of each result against the published literature (using the verifier verdicts) and which results are surprises.
- **Tools:** `compare_to_ground_truth(observed_behavior, ground_truth_id, experiment_ref, control_behavior)`, `list_ground_truth`, `get_benchmark`, `log_note`.
- **Input:** runner results + verifier verdicts. **Output:** per result `consistent | partially_consistent | inconsistent | inconclusive | not_comparable` with DOI and gt_confidence,
  the verifier verdict attached, surprises with the assumption they reopen, a concrete next experiment; `analysis` events.
- **Built-in honesty (in the tool, not only the prompt):** a manipulation mismatch makes the verdict inconclusive; checks that hold by construction of the bridge or adapter are flagged `by_construction`;
  labels inferred from brain rates without a body run are flagged; flight labels count any takeoff as escape and `no_takeoff` as no escape; a result without ground truth is reported as a novel agent-generated prediction.
  At most 4 comparisons per cycle.

### record_keeper, `claude-haiku-4-5`
- **Decision owned:** whether the research record supports each claim of the final report.
- **Tools:** `get_record`, `log_note`. **Input:** the run's record. **Output:** one final report (`decision` event, max ~350 words): question, evidence with DOIs, hypotheses, options and choice with cost and approvals,
  results vs ground truth, verifier verdicts, surprises and plan changes, numbers, limitations, next experiments.
- **Known weakness:** in the 2026-10-04 09:32 run its text calls gt22 and gt23 "validated", which is stronger than the tool verdict ("consistent via brain readout, no body run, 1 seed").
  Cite the tool verdicts (`analysis` events), not the wording of the report.

## 3. Hand-offs and what is passed (R4)

| From -> to | Passed as | Where it is stored |
|---|---|---|
| human -> PI | the question | `question` event |
| literature -> PI -> planner | findings with DOIs, ground-truth ids, gaps | `evidence` events (DOIs in `citations`) |
| hypothesis -> PI -> planner | hypothesis ids H1..H3 (agent-generated), manipulation, targets, predicted behaviour, ranking | `hypothesis` events; ranking in the `evidence` event `data` |
| planner -> safety -> runner | experiment specifications: exact tool arguments (kind, excite_groups, silence_groups, seed, duration) + estimated cost | `experiment_options` / `experiment_choice` / `approval` events |
| runner -> verifier -> analysis | results and artifact names (`flight_01`) with expected behaviour and its source | `experiment_result` and `movement_verification` events; artifacts `runs/<id>/artifacts/*.json` |
| analysis -> PI | verdicts with DOI, numbers, surprises | `analysis` events |
| PI -> next cycle | `log_decision(reason, next_step, reopens_assumption)` | `decision` event |

The inbox messages between agents are text written by the agents (their prompts require ids, names and numbers to be pasted verbatim);
the typed, machine-readable part of every hand-off is the event written by the tool. There is no separate JSON schema validation between agents.

## 4. Policies (policy-as-code, enforced by the Omnigent engine, top level of the YAML)

They apply to the PI session and every sub-agent session. `spikes/omni/test_policies.py` drives them through Omnigent's own policy plumbing without an LLM
(last run 2026-10-04: all pass, including the embodied cap `['ask','ask','ask','ask','ask','ask','deny']` after 6 approved calls).

| Policy | Implementation | Effect |
|---|---|---|
| `lab_approval_gate` | `flylab.tools.approval_gate` | **ASK** (human approval) on `request_approval`, `run_embodied_experiment`, `run_embodied_flight`; **ASK** on brain runs or screens above 10,000 simulated ms; `run_experiments_parallel` is judged by its content: ASK if it contains embodied/flight runs or > 10 s of brain simulation; **DENY** a screen above 40 candidates per call (the exhaustive 326-type screen is therefore always replaced by a guided top-k), more than 6 embodied runs per research run, or more than 6 parallel specs |
| `cost_budget` | Omnigent builtin `cost_budget` | LLM spend per session (root includes its sub-agents): **ASK at $2, hard stop at $3** |
| `tool_call_limit` | Omnigent builtin `max_tool_calls_per_session` | **DENY** after 80 tool calls per session |
| `loop_guard` | `flylab.tools.loop_guard` (wraps Omnigent `detect_loop`) | **ASK** if the same call repeats 3 times in 10, ignoring idempotent polling tools (`sys_read_inbox`, `get_record`, ...) |
| `dispatch_bounds` | Omnigent `spawn_bounds` | at most 3 `sys_session_send` per supervisor turn (allows the parallel literature + hypothesis hand-off) |
| `ask_timeout` | YAML | an unanswered approval waits up to 3600 s |

Planner budget (R8): 10 cost units per cycle (heuristic `estimate_cost`: brain, body, embodied, embodied_flight, screen, rank; these are relative compute units, not dollars).
Measured example (run `20261004-093251-...`): E1 82.5 units (denied by policy and 8x over budget), E2 3.4, E3 pair 6.4.

## 5. Human approval gates and controls (R9, R21)

1. **The scientist sets the objective:** the question is the only free-text input from the human; the PI cannot widen it.
2. **Safety agent asks, policy decides:** the safety agent calls `request_approval`; the policy turns it into a human ASK. Embodied and flight simulations and any long brain simulation are separately gated even if the safety agent forgets to ask.
3. **Three ways for the human to answer an ASK** (docs/OMNIGENT.md, "Approvals"): approval cards in the Omnigent web UI (`omni.ps1 server ...`), y/n prompts in the REPL, or a launch-time pre-approval for scripted runs (`omni.ps1 -ApproveAtLaunch`).
   The pre-approval is written by the human at launch (who, when, caps) and turns ASK into ALLOW with the reason "pre-approved at launch"; the DENY caps stay active. **The two recorded live runs used this pre-approval route;** the `approval` events in the record say so verbatim.
   The interactive card route is implemented and tested offline but is not part of a recorded run.
4. **Hard caps that no approval can lift:** 6 embodied runs, 40 screen candidates, 6 parallel specs, 80 tool calls, $3 LLM spend.
5. **Least-privilege tools:** only the runner can start simulations; the PI, literature, hypothesis, planner, safety, verifier and analysis agents have no simulation tool.
6. **Evidence controls:** DOIs only from tool results (`data/ground_truth.json` evidence quotes are checked verbatim against abstracts, `spikes/audit/check_citations.py`); hypotheses are labelled agent-generated;
   mock data is flagged `mock: true` and called mock; verdicts are never upgraded; verifier uncertainty is kept.
7. **Artefact controls for the embodiment (team rules G1-G5):** the flight adapter is frozen with a version and parameter hash before the comparison runs; brain response, adapter output, body physics and rendering are logged separately; see CHALLENGE_COMPLIANCE.md.
8. **Reconstructability:** `record.jsonl` (typed events), `artifacts/` (raw results, videos, contact sheets, verifier JSON), `omnigent/*.jsonl` (full transcripts of the 12 sessions) and `sessions.json` (session tree and cost).

## 6. Known weaknesses of the agent design (stated, not hidden)

- The planner read the 10-unit budget as cumulative over the cycles in the 09:32 run and priced the second flight pair out, so cycle 2 was brain-only and the verifier did not run again (the budget is per cycle).
- Cost units are heuristic estimates, not measured compute.
- The vision check is weak at judging translation with the tracking camera (stated in `movement_verifier.json`); `flight_01` therefore ended `uncertain` (kinematics: takeoff; vision: "climb", confidence 0.6 < 0.65).
- Inbox hand-offs are free text; structure is enforced at the tool and record level, not by a schema between agents.
