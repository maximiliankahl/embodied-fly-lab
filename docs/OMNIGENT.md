# Omnigent in the Embodied Fly Lab

Omnigent 0.16 (Databricks' open-source agent meta-harness) orchestrates the live discovery
loop: a PI/supervisor agent dispatches 7 specialist sub-agents, they call typed Python tools
(`flylab/tools.py`), every step lands in a shared research record, and **policies enforced by
Omnigent's policy engine** (not just prompts) gate cost and expensive simulations behind
human approval.

| File | What |
|---|---|
| `agents/fly_lab.yaml` | Lab definition: supervisor + 7 sub-agents + tools + policies + models (single-file Omnigent YAML) |
| `agents/omni.ps1` | Windows launcher for any `omnigent` command + `lab` (scripted demo run); env, workspace proxy, claude.exe, mock / pre-approval flags |
| `agents/anthropic_ws_proxy.py` | Local proxy (127.0.0.1:8788) that adds the `anthropic-workspace-id` header (see Credentials) |
| `agents/run_lab.py` | Scripted full-lab run: stock `omnigent run -p`, but follows the whole sub-agent tree, then exports transcripts |
| `agents/export_transcript.py` | Exports supervisor + all sub-agent sessions to `runs/<run_id>/omnigent/` |
| `flylab/tools.py` | 19 function tools + policy factories `approval_gate`, `loop_guard`; `FLYLAB_MOCK` support |
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

# 1) DEMO: full live discovery loop, scripted (Max pre-approves the gated actions by launching it)
#    ~10-20 min, real brain screen + embodied run; prints the session URL (watch it live in the browser)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 -ApproveAtLaunch lab "Which visual projection neurons drive backward walking (retreat) in Drosophila, and is the moonwalker pathway necessary?"

# 2) Same lab in the WEB UI with real human approval cards (best for the video)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 server --agent agents/fly_lab.yaml
#    -> open http://localhost:6767, start a chat with fly_lab, paste the question,
#       approve each gate (request_approval, embodied run) on its card.

# 3) Interactive REPL (approvals appear as prompts in the terminal)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 run agents/fly_lab.yaml

# Short smoke test with mocked science modules (~1 min, a few cents)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 -Mock run agents/fly_lab.yaml -p "SMOKE TEST: call start_run, dispatch only the literature specialist (list_ground_truth only, max 60 words), then log_decision and reply SMOKE OK."

# Export a session's transcript later (server must run; conv id = last part of the session URL)
uv run python agents/export_transcript.py <conv_id> --run <run_id>

# Stop everything (Omnigent server, host daemon, workspace proxy)
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 stop
```

Switches of `omni.ps1`: `-ApproveAtLaunch` = human pre-approval for this one command (see Approvals),
`-Mock` = all science modules return labelled mock data, `-MockOnly screen` = mock only listed
components, `-EnvModel` = override the supervisor model with `ANTHROPIC_MODEL` from `.env`.

### Why `lab` and not plain `omnigent run -p`
Omnigent 0.16's headless `run -p` decides an async orchestrator is finished when the supervisor's
snapshot reads `idle`; our PI ends its turn while a specialist works, so the CLI exited after the
first hand-off and stopped the session (observed 00:40). `agents/run_lab.py` runs the same CLI
in-process and only patches the follow logic (session counts as running while any sub-agent is busy,
`SessionsChat.tree_busy`, plus a 15 s grace period for the wake-up turn). Server, runner, policies
and approvals are stock Omnigent. At the end it exports all transcripts and prints the record summary.

## Credentials (Max's key from `.env`, nothing else, no login)
- `.env` holds `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_WORKSPACE_ID`; `omni.ps1` loads them
  (values never printed). Omnigent hands the key to `claude.exe` via an `apiKeyHelper` (`printf %s <key>`),
  so `omni.ps1` appends Git's `usr\bin` (printf.exe) to PATH.
- The key is not workspace-scoped: every request needs `anthropic-workspace-id`. `ANTHROPIC_CUSTOM_HEADERS`
  does **not** survive Omnigent's env allowlists (CLI -> host daemon in `omnigent/cli.py
  _build_host_daemon_env`, daemon -> runner in `omnigent/host/connect.py _RUNNER_ENV_ALLOWLIST`), but
  `ANTHROPIC_BASE_URL` does. So `omni.ps1` starts `agents/anthropic_ws_proxy.py` on 127.0.0.1:8788
  and sets `ANTHROPIC_BASE_URL` to it; the proxy forwards to api.anthropic.com and adds the header.
  It logs only method, path, status (log: `..\.omnigent\logs\ws_proxy.log`).
- If a daemon was started earlier without the proxy, runs fail with `400 ... anthropic-workspace-id`:
  run `omni.ps1 stop` once and retry.

## Models (pinned per agent in `fly_lab.yaml`)
| Agent | Model |
|---|---|
| fly_lab (PI / supervisor) | `claude-sonnet-5-5` |
| hypothesis, planner, analysis | `claude-sonnet-5-5` |
| literature, safety, runner, record_keeper | `claude-haiku-4-5-20251001` |

## Agent architecture and hand-offs

```
human question
   |
   v
fly_lab (PI / supervisor) --start_run--> runs/<run_id>/record.jsonl  <-- every tool logs here
   |  sys_session_send(agent=..., args.input=<previous result>)   /   sys_read_inbox
   +--> literature     search_literature, lookup_neurons, list_ground_truth -> evidence + gaps + DOIs
   +--> hypothesis     rank_candidates (connectome prior over 326 VPN types -> MDN), log_hypothesis (agent-generated)
   +--> planner        estimate_cost, get_benchmark, log_experiment_plan (>= 2 options: exhaustive vs guided screen ...)
   +--> safety         request_approval  ==> POLICY ASK -> human approves/denies
   +--> runner         run_brain_screen | run_brain_experiment | run_embodied_experiment (==> POLICY ASK)
   +--> analysis       compare_to_ground_truth (verdict + DOI), get_benchmark, flags surprises
   |      log_decision(reopens_assumption=...) --> cycle 2: necessity (hit + MDN silenced vs control)   [max 2 cycles]
   +--> record_keeper  get_record audit + final report (log_note type decision)
```

- Specialists get **only the tools of their role** (least privilege); the PI cannot run simulations.
- Hand-offs are visible three times: Omnigent web UI (sub-agent sessions), the exported transcripts
  `runs/<run_id>/omnigent/*.jsonl` + `sessions.json` (tree), and the record (`agent` field of every event).
- Tools resolve the run via explicit `run_id` > env `FLYLAB_RUN_ID` > `runs/CURRENT` (written by `start_run`).

## Approvals

| Route | How the human approves |
|---|---|
| Web UI (`omni.ps1 server ...`) | approval cards in the browser, per gated call |
| REPL (`omni.ps1 run agents/fly_lab.yaml`) | y/n prompts in the terminal |
| Scripted (`omni.ps1 -ApproveAtLaunch lab "..."`) | Max approves **at launch**: `omni.ps1` writes `data/cache/FLYLAB_PREAPPROVE` (who, when, caps); `approval_gate` turns its ASKs into ALLOW with reason "pre-approved at launch"; the flag is deleted when the command ends. DENY caps stay active. |

Headless `omnigent run -p` alone cannot answer an ASK (the session just waits), hence the pre-approval route.

## Policies (top level of `fly_lab.yaml`, apply to the PI and every sub-agent session)

| Policy | Implementation | Effect |
|---|---|---|
| `lab_approval_gate` | `flylab.tools.approval_gate` | **ASK** on `request_approval` and every `run_embodied_experiment`; ASK on brain runs / screens > 10 000 simulated ms; **DENY** screens > 40 candidates per call (forces guided top-k instead of an exhaustive 326-type screen) and more than 6 embodied runs per research run |
| `cost_budget` | `omnigent.policies.builtins.cost.cost_budget` | per session (root includes its sub-agents): ASK at $2 LLM spend, hard stop at $3 |
| `tool_call_limit` | `...builtins.safety.max_tool_calls_per_session` | DENY after 80 tool calls per session |
| `loop_guard` | `flylab.tools.loop_guard` (wraps Omnigent `detect_loop`) | ASK when the same call repeats 3x in 10, ignoring idempotent polling (`sys_read_inbox`, `get_record`, ...) |
| `dispatch_bounds` | `omnigent.inner.nessie.policies.spawn_bounds` | max 3 `sys_session_send` per supervisor turn |

`loop_guard` exists because stock `detect_loop` ASKed on the PI's 3rd identical `sys_read_inbox()`
and stalled the live run (00:48). `ask_timeout: 3600` s.

## Cost
Measured (Omnigent `total_cost_usd` of the root session, includes all sub-agents):
- Full demo run `runs/20261004-005152-which-visual-projection-neurons-908e`: **$1.03**, 466 s wall,
  12 Omnigent sessions (PI + 11 specialist sessions), 2 cycles, 32 record events.
- Review re-run after the tool fixes `runs/20261004-010600-which-visual-projection-neurons-ebe8`
  (current `runs/CURRENT`): **$1.21**, 552 s wall, 12 sessions, 2 cycles, 37 record events.
- All live tests on 2026-10-04 together: about $2.61 (smoke $0.05, two aborted runs $0.33, demo $1.03, review run $1.21).

Note: the root session's spend crossed $1 in the final step of that run. The ASK threshold was therefore
raised from $1 to $2 (review 01:20), otherwise a scripted run stalls at the end; the $3 hard stop stays.

## Demo runs (reference)
**Current (`runs/CURRENT`): `runs/20261004-010600-which-visual-projection-neurons-ebe8/`** - after the review
fixes (3-hop prior, per-target hits, experiment_ref checks). Connectome prior ranks LPC1, LT82b, LC6 top and
LC16 16/326; screen MDN hits: LPC1, LT82b, LC6, LC18, LC4, LC9, LC16; LPLC2 drives GF 159 Hz, MDN 0 Hz
(gt19 inconsistent = surprise; gt08 escape consistent at brain level); the LC16 embodied run moves
-1.22 mm but the body classifier says `stop`, so gt20 at body level is honestly logged inconsistent;
gt02 (MDN silencing) is flagged by construction. Caveat in this run: cycle-2 "seed replicates" were
identical because the tools had no seed parameter then (fixed afterwards: `seed` on brain/embodied/screen tools).

Older run `runs/20261004-005152-which-visual-projection-neurons-908e/` (before the review fixes; its record
logs gt01/gt18 "MDN activation -> backward: consistent" from an LC16 run and "observed backward" where the body
classifier said `stop` - do not cite those verdicts): `record.jsonl` (hand-offs human ->
supervisor -> literature -> hypothesis -> planner -> safety -> runner -> analysis -> supervisor ->
planner -> safety -> runner -> analysis -> supervisor -> record_keeper), `artifacts/` (screen, brain,
embodied JSON + mp4), `omnigent/` (12 transcripts + `sessions.json`). Highlights: connectome prior
ranked LC16 295/326 yet the screen hit it (MDN 32 Hz, embodied backward drift -1.22 mm); LPLC2 drove
GF (159 Hz) not MDN -> surprise -> plan update -> cycle 2 necessity test (LC16 +/- MDN silencing).

## Where things are stored

| What | Where |
|---|---|
| Research record (dashboard source) | `runs/<run_id>/record.jsonl`, JSON/videos in `runs/<run_id>/artifacts/` |
| Omnigent transcripts of a run | `runs/<run_id>/omnigent/<conv_id>.jsonl` (one per session) + `sessions.json` |
| Active run pointer | `runs/CURRENT` |
| Omnigent config, sessions (`chat.db`), logs | `C:\Users\mkahl\Hackathon\HackNation-7\.omnigent\` (outside the git repo on purpose) |
| Runner / server / host / proxy logs | `...\.omnigent\logs\{runner,server,host,cli}\`, `ws_proxy.log` |

Record event schema: `{ts, seq, run_id, agent, type, content, data, citations}`;
types: question, evidence, hypothesis, experiment_options, experiment_choice, approval,
experiment_result, analysis, decision, note. CLI: `uv run python -m flylab.record --list` / `--show <run_id>`.

## Scientific semantics of the tools

- `rank_candidates`: mean-field connectome score over 3 hops (default; the primary score of the
  committed `data/benchmarks/screen_mdn.json`, so the agents' prior and the cited acceleration use the
  same ranking) of all 326 visual projection types onto a target group. A **prior from the same
  connectome the brain model uses**, not independent evidence. (The reference run below still used the
  older 2-hop default, which ranks LC16 295/326; with 3 hops LC16 is rank 16.)
- `run_brain_screen`: activates each candidate type (Poisson 150 Hz, 500 ms, 2 trials by default) and
  reads target group rates; hit = >= 5 Hz (model baseline 0 Hz). `hits` = hit on ANY target (contract);
  use `hits_by_target` (MDN = backward, P9 = forward, GF = escape). In-silico predictions only.
- `compare_to_ground_truth` verdicts: `consistent` | `partially_consistent` | `inconsistent` |
  `inconclusive`; `surprise` = inconsistent or partially_consistent. Silencing (`effect: reduce`) needs a control run.
  It loads the artifacts named in `experiment_ref` (e.g. `embodied_01.json`) and records
  `manipulation_match` (a run that did not manipulate the published target -> `inconclusive`, e.g. an LC16
  run is not a test of "activate MDN"), `label_source` (`body_classifier` | `agent_override` |
  `brain_readout_inference` | `unverified`) and `by_construction` (MDN -> backward, P9 -> forward are wired
  into the bridge; agreement there checks the bridge, it is not independent validation).
- Bridge: backward drive is read only from MDN, so an embodied run with MDN silenced cannot walk backward
  by construction; necessity is judged from brain-level descending rates.
- Body runs < 1 s get a `label_warning` (short backward runs can be mislabelled as turns).
- `estimate_cost`: heuristic, 1 cost unit = 10 s wall on this laptop.

## Known limits
- Mock flag file `data/cache/FLYLAB_MOCK` can stay behind if an `omni.ps1 -Mock server` window is killed;
  start any `omni.ps1 run|server|lab` without `-Mock` to remove it.
- `cost_budget` ASKs at $2 per session; in a scripted run nobody answers it, so a session that
  exceeds $2 waits (watch the session URL and approve in the web UI if that happens).
- `run_lab.py` does not detect a pending ASK: a session waiting for approval looks idle, and after the
  15 s grace period the CLI exits and stops the session. Pre-approval covers the lab gates, not cost ASKs.
- The record lock is per process; concurrent writers (runner + Streamlit) can rarely duplicate `seq`.

## Windows notes (degraded mode: web UI + SDK harnesses, no tmux/bwrap)
- `claude-sdk` needs a native `claude.exe`; `omni.ps1` puts the one of the Claude desktop app on PATH.
- `PYTHONUTF8=1` is required, otherwise the host daemon crashes on a glyph ("host is offline").
- Telemetry off: `DO_NOT_TRACK=1` and `OMNIGENT_ANALYTICS=0`.
- Always start Omnigent from `02_App` (cwd goes on `sys.path`, so `flylab.tools` imports).
