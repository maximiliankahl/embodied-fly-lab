# Omnigent in the Embodied Fly Lab

Omnigent 0.16 (Databricks' open-source agent meta-harness) orchestrates the live discovery
loop: a PI/supervisor agent dispatches 7 specialist sub-agents, they call typed Python tools
(`flylab/tools.py`), every step lands in a shared research record, and **policies enforced by
Omnigent's policy engine** (not just prompts) gate cost and expensive simulations behind
human approval.

| File | What |
|---|---|
| `agents/fly_lab.yaml` | Lab definition: supervisor + sub-agents + tools + policies (single-file Omnigent YAML) |
| `agents/omni.ps1` | Windows launcher (env, telemetry off, claude.exe, mock flag) for any `omnigent` command |
| `flylab/tools.py` | 16 function tools + `approval_gate` policy factory; `FLYLAB_MOCK` support |
| `flylab/record.py` | Research record `runs/<run_id>/record.jsonl` + `artifacts/` |
| `spikes/omni/validate_yaml.py` | Loads/validates the YAML with Omnigent's own loader, checks runtime tool schemas |
| `spikes/omni/test_policies.py` | Runs the YAML policies through Omnigent's policy plumbing (no LLM needed) |

## Run it (PowerShell, from `02_App`)

```powershell
cd C:\Users\mkahl\Hackathon\HackNation-7\02_App

# 0) Offline checks (no credentials, ~10 s)
$env:FLYLAB_MOCK = "1"; uv run python -m flylab.tools --selftest; Remove-Item Env:FLYLAB_MOCK
uv run python spikes/omni/validate_yaml.py
uv run python spikes/omni/test_policies.py

# 1) Short end-to-end smoke test with mocked science modules
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 -Mock -EnvModel run agents/fly_lab.yaml -p "SMOKE TEST: call start_run, dispatch only the literature specialist (list_ground_truth + lookup_neurons MDN, max 60 words), then log_decision and reply SMOKE OK."

# 2) Real research session (REPL; approvals appear as prompts)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 -EnvModel run agents/fly_lab.yaml -p "Which descending neurons drive backward walking, and does the embodied model reproduce it?"

# 3) Web UI instead of REPL: open http://localhost:6767, pick agent fly_lab, approve gates in the browser
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 server --agent agents/fly_lab.yaml

# Stop everything Omnigent started (server, host daemon)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 stop
```

`-Mock` = science modules (atlas/literature/brain/body/bridge) return clearly labelled mock data
(`"mock": true`). `-EnvModel` = pass `--model $env:ANTHROPIC_MODEL` from `.env` (only the
supervisor; sub-agents use the provider default). Plain `uv run omnigent ...` also works but
then you must set the env vars from `omni.ps1` yourself.

### Credentials (Max does this; never paste keys into chat)
- `ANTHROPIC_API_KEY` in `02_App\.env` is loaded by `omni.ps1` (value never printed). Omnigent
  passes it to `claude.exe` through an `apiKeyHelper` (`printf %s <key>`), which is why
  `omni.ps1` appends Git's `usr\bin` (printf.exe) to PATH.
- **Status 2026-10-03 23:15:** the smoke test reached the Anthropic API but got
  `400 This API key is not scoped to a workspace ... must include the anthropic-workspace-id header`.
  Fix: in console.anthropic.com create a key **inside a workspace** (e.g. "Default") and put it into
  `.env`. Then rerun step 1.
- Alternative: Claude subscription. Run `powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 setup`
  in your own terminal and pick the Claude subscription provider (interactive login).
- OpenAI instead: harness `openai-agents` (`--harness openai-agents --model <gpt model>`), untested here.

## Agent architecture and hand-offs

```
human question
   |
   v
fly_lab (PI / supervisor) --start_run--> runs/<run_id>/record.jsonl  <-- every tool logs here
   |  sys_session_send(agent=..., args.input=<previous result>)   /   sys_read_inbox
   +--> literature     search_literature, lookup_neurons, list_neuron_groups, list_ground_truth -> evidence + gaps + DOIs
   +--> hypothesis     log_hypothesis (labelled agent-generated, confidence, falsifier)
   +--> planner        estimate_cost, log_experiment_plan (>= 2 options, pick by info gain / cost, budget 10 units)
   +--> safety         request_approval  ==> POLICY ASK -> human approves/denies
   +--> runner         run_brain_experiment | run_body_experiment | run_embodied_experiment (==> POLICY ASK each)
   +--> analysis       compare_to_ground_truth (verdict + DOI), flags surprises
   |      surprise? --> log_decision(reopens_assumption=...) --> cycle 2 (hypothesis/planner)   [max 2 cycles]
   +--> record_keeper  get_record audit + final report (log_note type decision)
```

- Specialists get **only the tools of their role** (least privilege); the PI cannot run simulations.
- Hand-offs are visible twice: in Omnigent (sub-agent sessions in the web UI / REPL) and in the
  record (`agent` field of every event: human, supervisor, literature, hypothesis, planner, safety,
  runner, analysis, record_keeper).
- Tools resolve the run via explicit `run_id` > env `FLYLAB_RUN_ID` > `runs/CURRENT` (written by
  `start_run`). The file fallback matters because Omnigent's host passes only an env allowlist to
  the runner process that executes the tools.

## Policies (top level of `fly_lab.yaml`, apply to the PI and every sub-agent session)

| Policy | Implementation | Effect |
|---|---|---|
| `lab_approval_gate` | `flylab.tools.approval_gate` | **ASK** (human) on `request_approval` and every `run_embodied_experiment`; ASK on brain runs > 10 000 simulated ms (duration_ms x n_trials); **DENY** once 6 embodied runs exist, counted per Omnigent session *and* in the active research record (so new runner sub-sessions cannot reset it) |
| `cost_budget` | `omnigent.policies.builtins.cost.cost_budget` | ASK at $1 and $2.5 LLM spend, hard stop at $5 per session |
| `tool_call_limit` | `...builtins.safety.max_tool_calls_per_session` | DENY after 80 tool calls per session |
| `loop_guard` | `...builtins.safety.detect_loop` | ASK when the same call repeats 3x in 10 |
| `dispatch_bounds` | `omnigent.inner.nessie.policies.spawn_bounds` | max 3 `sys_session_send` per supervisor turn |

`ask_timeout: 3600` s. Approvals show up as prompts in the REPL or as approval cards in the web
UI. ASK state updates (the embodied-run counter, an `increment`) are applied only on approve.
The `on: [tool_call]` lines in the YAML are documentation only: Omnigent ignores `on:` for
`type: function` policies (the callable self-selects by returning `None`).
`cost_budget` reads the session's own LLM usage; whether a sub-agent session's spend is
counted in the parent's budget was not verified offline.
`request_approval` writes an `approval` event only after the human approved (the ASK gate runs
before the tool body). Called outside Omnigent (selftest, scripts) there is no human gate.

## Where things are stored

| What | Where |
|---|---|
| Research record (dashboard source) | `runs/<run_id>/record.jsonl`, videos/JSON in `runs/<run_id>/artifacts/` |
| Active run pointer | `runs/CURRENT` |
| Omnigent config, sessions (`chat.db`), logs | `C:\Users\mkahl\Hackathon\HackNation-7\.omnigent\` (set by `omni.ps1` via `OMNIGENT_CONFIG_HOME` + `OMNIGENT_DATA_DIR`; outside the git repo on purpose, may hold credentials) |
| Runner / server / host logs | `...\.omnigent\logs\{runner,server,host,cli}\` |
| Full Omnigent transcript of a session | `powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 session export --id <conv_id> --output runs\<run_id>\omnigent_transcript.jsonl` (conv id = last part of the session URL) |

Record event schema: `{ts, seq, run_id, agent, type, content, data, citations}`;
types: question, evidence, hypothesis, experiment_options, experiment_choice, approval,
experiment_result, analysis, decision, note.
CLI: `uv run python -m flylab.record --list` / `--show <run_id>`.

## Scientific semantics of the tools (review 2026-10-03)

- `compare_to_ground_truth` verdicts: `consistent` | `partially_consistent` | `inconsistent` |
  `inconclusive`; `surprise` = inconsistent or partially_consistent.
  - Turning in the **opposite** direction to the paper = `inconsistent` (same rule as `atlas.evaluate`).
  - `effect: reduce` entries (silencing, e.g. gt02 MDN) need `control_behavior`: behavior present in the
    manipulated run = inconsistent; absent with a control that shows it = consistent; absent without a
    control = partially_consistent (weak).
  - escape / groom / feed cannot be produced by the walking body: a body label against such an entry
    gives `inconclusive` with `comparable: false`. Compare the entry's `readout_group` firing instead
    (`run_brain_experiment` now returns `other_group_rates_hz`, e.g. MN9 for sugar -> feeding).
  - Unknown observed labels give `inconclusive`. Output also carries `gt_confidence` and `gt_note`.
- Body runs shorter than 1 s get a `label_warning`: in review, backward drive 1.0 for 0.5 s drifted
  -20 deg in heading and was classified `turn_right` by `body.classify`. Use `duration_s >= 1`.
- `run_embodied_experiment` passes per-neuron rates `{root_id: Hz}` (brain.simulate()["rates"]) to
  `bridge.rates_to_drive`; if the bridge fails on them or returns an all-zero drive while descending
  groups fire, it retries with `{group_name: mean Hz}`. `bridge` in the result says which input was used.
- `estimate_cost` is a heuristic calibrated on this laptop (brain: ~3 s per simulated second per trial
  + 5 s; body: 25 s per simulated second). 1 cost unit = 10 s wall. An embodied run with defaults is
  ~3.9 units, so the 10-unit cycle budget allows about 2 embodied runs.
- Artifacts: `brain_NN.json`, `body_NN.mp4`, `embodied_NN.mp4` + `embodied_NN.json` (same NN).
- `python -m flylab.tools --selftest` no longer touches `runs/CURRENT` (safe during a live session).

## Known limits / to do
- `flylab/bridge.py` missing: real-mode `run_embodied_experiment` returns `ok=false` (ImportError hint)
  until it exists. Mock mode works.
- Mock flag file `data/cache/FLYLAB_MOCK`: if an `omni.ps1 -Mock server` window is killed, the file can
  stay behind. Every tool output then says `"mock": true`. Delete the file, or start any
  `omni.ps1 run|server` without `-Mock` (it removes the file).
- `.gitignore`: `.omnigent/` lives outside the repo (fine). If it is ever moved into `02_App`, add
  `.omnigent/` to `.gitignore`. `runs/` is not ignored (videos are, via `*.mp4`).
- The record lock is per process. Concurrent writes from two processes (e.g. Omnigent runner and
  Streamlit) can give duplicate `seq` numbers or, rarely, a garbled line (skipped by `record.load`).

## Windows notes (degraded mode: web UI + SDK harnesses, no tmux/bwrap)
- `claude-sdk` needs a native `claude.exe`; the pip wheel bundles none. `omni.ps1` puts the
  `claude.exe` of the Claude desktop app on PATH (nothing installed).
- `PYTHONUTF8=1` is required, otherwise the host daemon crashes on a `✓` glyph ("host is offline").
- Telemetry off: `DO_NOT_TRACK=1` and `OMNIGENT_ANALYTICS=0` (set by `omni.ps1`).
- Always start Omnigent from `02_App` (cwd goes on `sys.path`, so `flylab.tools` imports).
- `omnigent run` uses a local background server on port 6767; `omni.ps1 stop` ends it.
