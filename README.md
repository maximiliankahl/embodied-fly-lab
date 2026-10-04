# Embodied Fly Lab

> An agentic AI lab that wires a 139,000-neuron fruit-fly brain model (FlyWire connectome) to a physics-simulated fly body (walking and flying) and lets specialist agents generate, test and check hypotheses against published experiments. A brain screen experiment takes about a second of compute.

Built in 24 hours at the **7th Hack-Nation Global AI Hackathon** (Munich Hub, October 3-4, 2026).
Challenge 03: **Agentic Scientific Discovery** (Databricks, built with **Omnigent**).

| | |
|---|---|
| **Repository** | https://github.com/maximiliankahl/embodied-fly-lab |
| **Demo video (2 min)** | LINK (to be added before submission) |
| **3D replay viewer** | https://maximiliankahl.github.io/embodied-fly-lab/ (live after GitHub Pages is enabled; recorded runs, never live computation). Local: `uv run python -m http.server 8777 --directory web`, then http://localhost:8777 |
| **Agent specs and policies** | [docs/AGENT_SPECS.md](docs/AGENT_SPECS.md) (decision owned, tools, inputs, output per agent; budget, caps, approval gate) |
| **Challenge compliance audit** | [docs/CHALLENGE_COMPLIANCE.md](docs/CHALLENGE_COMPLIANCE.md) (every challenge item mapped to evidence, with gaps) |
| **Omnigent guide and run log** | [docs/OMNIGENT.md](docs/OMNIGENT.md) |
| **Data inputs (sources, versions, licences, SHA256)** | [data/manifest.json](data/manifest.json) |

---

## The question and the bottleneck we attack

*Which visual neurons trigger an escape (or a retreat), and are the descending neurons they use necessary for it?*
Two live runs asked this: backward walking via the "moonwalker" neurons, and escape takeoff via the giant fiber.

**Bottleneck:** deciding **which of the 326 visual projection cell types to test**. In the wet lab each candidate needs its own genetic driver line and a behavioural experiment. With the full adult fly connectome published (FlyWire, 2024), the candidates can be ranked and screened in silico first, and only the best ones need an expensive test (a physics body run, an agent cycle, or a wet-lab line).
The lab compresses that step: a connectome-guided prior, a fast whole-brain screen, a physics body to check the behavioural consequence, an independent movement check, and a comparison against 26 published results.

## What we built

```
 Question ─► Literature ─┐
 (human)     (Europe PMC, ├─► Planner (>= 2 designs, cost vs. information gain, 10-unit budget)
              OpenAlex)   │        │
          Hypothesis ─────┘        ▼
          (connectome ranking)  Safety agent ─► POLICY GATE (human approval, hard caps)
                                                   │
 Supervisor (PI) ◄─ Analysis ◄─ Movement verifier ◄┴─ Runner (independent runs in parallel)
   │  reopens an assumption     (kinematics + blind       │
   │  after a surprise)          vision on the video)     ▼
   └─► cycle 2 (necessity)      whole-brain LIF model ─► bridge (descending neurons) ─► physics body (MuJoCo)
                                                                                       walking | flying
```

| Layer | What it is | Evidence it works |
|---|---|---|
| **Brain** (`flylab/brain.py`) | Event-driven numpy re-implementation of the whole-brain leaky-integrate-and-fire model of Shiu et al. 2024 (*Nature*): 138,639 neurons, 15.1 M connections, FlyWire v783 | Rates of the non-stimulated neurons match the original Brian2 model on a 426-neuron sugar-GRN subnetwork (16 trials, r = 0.9987). Reproduces the paper's sugar-GRN -> MN9 result, bitter suppression and contralateral bias. About 0.9 s wall time per trial per simulated second. [`brain_validation.json`](data/benchmarks/brain_validation.json) |
| **Bridge** (`flylab/bridge.py`) | Hand-designed interface standing in for the ventral nerve cord. Walking: P9 -> forward, MDN -> backward, DNa01/DNa02 asymmetry -> turning. Flight: giant fiber -> takeoff, DNg02 -> wingbeat amplitude, DNa01/DNa02 asymmetry -> yaw. Reference rate measured, not assumed (148.3 Hz). Every term cites its paper. | Walking and brain read-outs: [`embodied_validation.json`](data/benchmarks/embodied_validation.json). Flight adapter frozen as `flight-adapter-v1` (hash `02ef23b1b5d6dca2`) before the comparison runs. |
| **Walking body** (`flylab/body.py`) | flygym 2.1 / NeuroMechFly v2 fly in MuJoCo, CPG + preprogrammed steps, behaviour classifier, video | Body labels identical for 3 phase-noise seeds in all 10 conditions (`body_seed_robustness`, same file). |
| **Flying body** (`flylab/flight.py`) | FlyBody (Vaxenburg et al. 2025) in MuJoCo with cycle-averaged quasi-steady wing forces, gravity, ground contact, a takeoff jump and a hand-designed attitude stabiliser. With video rendering one simulated second takes about 16-18 s of wall time (`wall_per_sim_s` in the benchmark file). | 8 conditions simulated, all behave as expected (table below); [`flight_validation.json`](data/benchmarks/flight_validation.json) |
| **Movement verifier** (`flylab/verify.py`, agent `movement_verifier`) | Checks that the body really did what the brain readout promised. (1) Kinematics recomputed from the raw trajectory with its own formulas. (2) A **blind** Claude vision check of a keyframe contact sheet (it never sees the expected behaviour or the numbers). Final verdict is `correct` only if both agree. | On the 10 committed walking videos kinematics were right on 10/10 known labels and wrong on 10/10 opposite labels; vision was exact on 4 of 6 unique videos; final 8 correct, 2 uncertain. A consistency check, not ground truth. [`movement_verifier.json`](data/benchmarks/movement_verifier.json) |
| **Atlas + literature** (`flylab/atlas.py`, `literature.py`) | 63 neuron groups mapped to FlyWire v783 IDs; 26 published neuron -> behaviour relations with 15 distinct DOIs | `spikes/audit/check_citations.py`: all 42 distinct DOIs found anywhere in the repo resolve (Europe PMC / OpenAlex, at audit time); 26 of 26 ground-truth evidence quotes occur verbatim in the abstracts. |
| **Discovery engine** (`flylab/screen.py`) | Connectome path-strength ranking of candidate cell types + fast whole-brain screen | See results. |
| **Agent lab** (`agents/fly_lab.yaml`) | Omnigent supervisor + 8 specialist sub-agents (literature, hypothesis, planner, safety, runner, movement verifier, analysis, record keeper) with typed tools and policies-as-code | Latest live run: 2 cycles, 12 agent sessions, 33 record events, $1.11, 550 s. |
| **3D replay viewer** (`web/`) | Static Three.js page (no build step): 12 recorded runs (6 walking, 6 flight) with the fly's physics poses, a 138,639-point brain coloured by mean firing rate, the bridge output, the verifier verdict, the ground-truth DOI. Header: "Recorded simulation replay, not live". | Smoke test `spikes/viewer3d/check_site.py` (all files HTTP 200); flight runs reproduce their benchmark rows. |
| **Dashboard** (`app.py`) | Streamlit: lab notebook with agent hand-offs, experiment bench, validation and speed, flight page, method and limits | |

Every agent decision is written to a shared research record (`runs/<run_id>/record.jsonl`), so each step can be reconstructed afterwards.

### Flight: what the connectome decides and what we designed by hand

A body that can fly is not evidence that the connectome controls flight. Exactly:

| Part | Who decides | Status |
|---|---|---|
| Whether visual input makes the giant fiber (GF) fire, and how strongly (graded, from the 138,639-neuron model) | **connectome model** | partly emergent. The only non-circular part. |
| GF firing -> takeoff trigger (threshold 0.5 = half the 148.3 Hz reference, about 74 Hz) | adapter design from Lima & Miesenboeck 2005 and von Reyn et al. 2014 | plumbing check (the same papers are the ground truth), threshold hand-chosen |
| DNg02 population rate -> wingbeat-amplitude increment | adapter design from Namiki et al. 2022 (25 DNg02 neurons in v783, paper: at least 15 pairs; identified by cell-type name) | plumbing check, gain hand-set (+12 %) |
| Yaw from the DNa01/DNa02 left-right asymmetry | transfer assumption from walking steering, not validated for flight | unvalidated |
| Pitch / forward flight | none (pitch = 0): no forward flight comes from the connectome | |
| Jump impulse, wing kinematics, attitude stabiliser, hover trim, aerodynamics | hand-designed physics model (`flylab/flight.py`) | not from the connectome |
| Open loop | constant command from 1 s mean rates (3 trials); the stimulus is Poisson activation of LPLC2 neurons, not a rendered looming stimulus; the video is stroboscopic (200 Hz wingbeat in a 30 fps video) | |

## Results (measured, not claimed)

**1. Faster triage: 25x fewer experiments to the first hit (MDN), 18x (GF), against random order. Against a stronger baseline 5x and 3x.**
Targets: backward-walking command neurons (MDN, `screen_mdn.json`) and the giant fiber (GF, `screen_gf.json`). All 326 visual projection types were screened in the whole-brain model (1.24 s per experiment on average; exhaustive screen 404.5 s on the development machine).

| Target | in-silico hits (>= 5 Hz) | Guided: first hit after | Random order (expected) | Baseline "largest type first" | Literature-known drivers |
|---|---|---|---|---|---|
| MDN | 12 | 1.0 experiment | 25.2 (**25.15x**) | 5.0 (**5x**) | first known driver LPLC2 (connectome rank 12, the model does *not* confirm it as a hit) after 12 vs 109 random (9.08x); first known driver the model also confirms, LC16, after 16 vs 163.5 (10.2x) |
| GF | 17 | 1.0 experiment | 18.2 (**18.17x**) | 3.0 (**3x**) | first known driver LPLC2 (rank 2) after 2 vs 109 random (54.5x) |

A budget of 20 experiments (6 % of the screen) finds 12 of 12 MDN hits and 14 of 17 GF hits, against 0.7 and 1.0 expected for random order.
**Caveats, all in the JSON files:** the speed-up counts in-silico experiments (the exhaustive screen takes only 404.5 s, so the gain matters when each follow-up test is expensive); the all-hits speed-up is *not* robust (MDN: 15x for the primary score variant but 0.94x to 0.96x for the others; GF: 0.97x); the in-silico "hits" come from the same connectome the ranking uses (partly circular); the literature checks rest on 2 known drivers per target; 11 of 12 MDN hits and 16 of 17 GF hits replicate with a second seed.

**2. Validation against published experiments.**
- Walking and brain read-outs: **8 of 9 comparable checks agree** (MDN -> backward, P9 -> forward, unilateral P9 / DNa02 -> ipsilateral turning, sugar GRNs -> feeding motor neuron, Johnston's organ -> grooming pathway, LPLC2 -> escape read-out). The one disagreement is LPLC2 -> backward walking (see 4). The direct descending-neuron checks are partly circular because the bridge was built from the same papers; the sensory ones (LPLC2, sugar, JO) and the P9 -> DNa02 recruitment test the connectome model. [`embodied_validation.json`](data/benchmarks/embodied_validation.json)
- Flight: **6 of 6 comparable ground-truth checks consistent**, all 8 conditions behave as the validation script expected (the expectations are listed per row; those for LPLC2 at 10 Hz and 30 Hz were fixed after a dose probe, so they are not blind). Most rows are circular by design, as labelled in the file (table below). [`flight_validation.json`](data/benchmarks/flight_validation.json)

| Condition (brain 1 s, 3 trials; body 1 s) | GF rate, mean L/R (Hz) | Body result | Max height (mm) | Type of result | Movement verifier |
|---|---|---|---|---|---|
| control (no stimulus) | 0 | no_takeoff | 0 | control | correct |
| GF driven directly | 169.3 | hover | 18.2 | circular (adapter check) | correct |
| LPLC2 at 10 Hz | 36.2 | no_takeoff | 0 | model prediction, partly emergent | correct |
| LPLC2 at 30 Hz | 84.3 | hover | 18.2 | model prediction, partly emergent | correct |
| LPLC2 at 150 Hz | 160.5 | hover | 18.2 | partly emergent | **uncertain** (kinematics: takeoff; vision: "climb", confidence 0.6 < 0.65) |
| LPLC2 at 150 Hz, GF silenced | 0 | no_takeoff | 0 | circular by design | correct (kinematics + vision) |
| DNg02 only | 0 | no_takeoff | 0 | circular by design | correct |
| GF + DNg02 | 160.3 | climb | 183.4 | circular | correct |

LPLC2 dose response (11 rates, 5-150 Hz, brain + adapter): GF firing is graded, not all-or-none; the adapter's takeoff threshold is crossed at about 23.7 Hz LPLC2 drive (interpolated; 20 Hz gives activation 0.45, 25 Hz gives 0.52). The threshold 0.5 is hand-chosen and the Poisson rate is not calibrated to looming strength. The expected outcomes for the 10 Hz and 30 Hz rows were fixed after this dose probe, so they are not blind. One brain seed per condition.

**3. The latest live run: escape takeoff through the giant fiber** (`runs/20261004-093251-which-visual-neurons-trigger-esc-9460/`, real data, nothing mocked; the second example is the walking run `runs/20261004-010600-...`).
Question: *Which visual neurons trigger escape takeoff through the giant fiber, does the simulated body actually take off, and is the giant fiber necessary?*
Hand-offs as recorded: literature and hypothesis agent in parallel -> planner -> safety agent -> (human pre-approval) -> runner -> movement verifier -> analysis -> supervisor -> cycle 2 -> record keeper.
- Literature: LPLC2 and LC4 synapse onto the GF (doi:10.1016/j.cub.2019.01.079); several visual projection neurons evoke jumping (doi:10.7554/eLife.21022); GF spike timing and takeoff (doi:10.1038/nn.3741).
- Hypotheses H1-H3, all labelled **agent-generated** (LPLC2 -> GF -> takeoff; a gap probe on LC15 and others; GF silencing abolishes it).
- Planner: 3 options priced against 10 units (exhaustive 326-type screen 82.5 units, rejected and denied by policy above 40 candidates; guided 10-type screen 3.4; flight pair 6.4). Chosen: guided screen + flight pair = 9.8 units.
- Runner: both flights in **one parallel call, 21.0 s wall vs 30.6 s summed (1.46x)**; the body renders are serialised, only the brain parts overlap.
- Result: LPLC2 drives the GF at 158 Hz, takeoff command 0.992, the body takes off (max height 18.2 mm, label `hover`); with GF silenced: GF 0 Hz, no takeoff.
- Verifier: `flight_01` **uncertain** (kinematics say takeoff, blind vision says "climb" at confidence 0.6, below the preset 0.65); `flight_02` **correct** against `no_takeoff`. The verdict was not upgraded.
- Supervisor: the verifier's uncertainty and the saturated screen (all 10 candidates drove the GF at >= 5 Hz) **reopened** the assumption that the bridge turns a GF rate into a reliably labelled escape movement and that "GF >= 5 Hz" is a meaningful hit criterion. Cycle 2 became a brain-level necessity replicate (seed 2): GF 157 Hz with LPLC2, 0 Hz with GF silenced, other descending neurons unchanged.

**4. A surprise that changed the next decision (walking run).**
Wu et al. 2016 report that LPLC2 activation produces backward walking about as often as jumping. In the model LPLC2 drives the giant fiber (escape) at 159 Hz but **MDN at 0 Hz**, while LC16 drives MDN at 32 Hz. The agents flagged this as inconsistent, reopened the assumption that LPLC2 retreat runs through MDN, and planned a necessity test (LC16 with MDN silenced: MDN 32.5 Hz -> 0 Hz). Two predictions for the wet lab follow, both **agent-generated and untested**: LPLC2-evoked retreat should not depend on MDN (or depends on pathways outside the brain connectome), and LPC1 and LC6 (they drive MDN in silico but are not linked to retreat in our literature set) are new candidates to test.

## Acceleration: what was attacked, what the agents learned, what comes next, what 10x needs

**Bottleneck attacked:** choosing which candidate cell types deserve an expensive test (326 visual projection types; see above).

**Measured improvement, with baselines:** first in-silico hit after 1 experiment instead of 25.2 (MDN) and 18.2 (GF) for random order (25x, 18x); 5x and 3x against the stronger "largest cell type first" baseline; about 9x for the first literature-known MDN driver. Not an improvement: finding *all* hits (0.97x for GF, not robust for MDN). In the latest live run the 10-candidate screen did not discriminate at all (10 of 10 hits). Parallel runs: 1.46x (flights) and 1.94x (brain runs) in wall time.

**What the agents learned:**
- A connectome-based prior puts a first hit on top, so it saves experiments when each test is expensive; it does not order the hits by strength. At 150 Hz Poisson drive the top-10 types all saturate the giant fiber, so the binary hit criterion is too coarse (the supervisor reopened exactly this assumption).
- LPLC2 drives the giant fiber strongly and MDN not at all in the model, which is in tension with a published backward-walking result.
- GF firing depends on the visual drive in a graded way; the takeoff threshold of the frozen adapter is crossed at about 23.7 Hz LPLC2 drive.
- A single body label is fragile: the verifier could not confirm "hover" versus "climb" from video, and said so.
- "GF necessary for takeoff" is true here by design of the adapter; it checks the pipeline and is not a biological finding (real flies also take off through non-GF circuits, von Reyn et al. 2014).

**Next experiment (proposed, not run):**
1. Re-screen at lower drive rates (for example 10-30 Hz) and rank candidates by the rate needed to cross the takeoff threshold, which removes the ceiling that hid differences in the live run.
2. Give the adapter a second, non-GF takeoff route so that silencing the GF lowers the takeoff probability instead of abolishing it, and compare that prediction with von Reyn et al. 2014.
3. Test the agent-generated candidates in the wet lab, starting with LC6 and LPC1, and the in-silico GF drivers beyond the literature set (`screen_gf.json`, `novel_in_silico_hits`, for example LC17, LC31a, LT1b; all untested).

**What would be needed to approach 10x at scale (our assessment, not a measurement):**
- **Known precision against biology.** 25x fewer in-silico experiments is not 25x faster discovery until the hit list is checked against a larger set of known positives and negatives (we have 2 known drivers per target, and no negatives). That is the first thing to buy.
- **A search space where triage matters.** At 326 candidates the exhaustive screen takes 404.5 s. The 10x claim only applies when a test is expensive (wet-lab line, body run at about 17 s, agent cycle at about 9 min and $1.1), or the space is far larger than the visual projection types.
- **Removing the hand-designed layers:** the ventral nerve cord connectome instead of the bridge, a rendered looming stimulus with sensory feedback instead of Poisson drive, calibrated flight control instead of the attitude stabiliser.
- **Statistics:** several brain seeds per condition (we have one), blind pre-registered expectations, an ensemble over model variants.
- **Agent loop cost and planning:** cheaper, batched experiments (the body renders are serial today), and a planner whose budget logic is checked (in cycle 2 it read the 10-unit budget as cumulative and priced out the body run).

## Controls and human approval gates

Full detail in [docs/AGENT_SPECS.md](docs/AGENT_SPECS.md) and [docs/OMNIGENT.md](docs/OMNIGENT.md).
- **The scientist sets the objective** (the question). The supervisor cannot widen it and has no simulation tool.
- **Safety agent asks, the Omnigent policy decides.** `lab_approval_gate` turns every embodied or flight run, every `request_approval` and every long brain simulation into a human **ASK**, and **DENIES** screens above 40 candidates per call, more than 6 embodied runs per research run and more than 6 parallel specs. Other policies: LLM spend (ASK at $2, hard stop at $3), 80 tool calls per session, loop guard, at most 3 dispatches per supervisor turn. `uv run python spikes/omni/test_policies.py` checks them without an LLM (all pass).
- **Least-privilege tools:** only the runner can start a simulation.
- **Honest note on the recorded runs:** both used the launch-time pre-approval (`omni.ps1 -ApproveAtLaunch`); the human decision is recorded in the `approval` events and the deny caps stay active. The interactive approval cards (Omnigent web UI) are implemented but not part of a recorded run.
- **Evidence controls:** DOIs only from tool results, hypotheses labelled agent-generated, mock data flagged, verdicts never upgraded, circular checks labelled, the flight adapter frozen with a version and hash before the comparison runs.

## Validation still needed before real-world use

This is a computational hypothesis generator. Nothing here is validated biology, and it must not be used to draw conclusions about real animals, to guide an intervention, or as a flight-control model without the following.
- **Wet-lab tests** of every agent-generated hypothesis: optogenetic activation and silencing of the proposed cell types with behavioural tracking (retreat, takeoff), including a dose-response test of the predicted graded GF recruitment.
- **Electrophysiology or imaging** of the predicted LPLC2 -> GF drive under a calibrated looming stimulus; the Poisson rates in the model are not calibrated to stimulus strength.
- **A ventral nerve cord model** (the FlyWire v783 brain has none) and validation of the body against high-speed video of real takeoffs, including the short versus long takeoff modes; flight yaw is a transfer assumption.
- **Cell-type identification checks** (for example DNg02: 25 neurons in v783 versus at least 15 pairs in the paper) and the few v630 -> v783 ID substitutions.
- **Replication:** multiple brain seeds, larger known-positive and known-negative sets, independent review of the adapter choices.

## Limitations (honest)
- FlyWire covers the **brain only**. The ventral nerve cord is replaced by a hand-designed, documented bridge; all runs are open loop (no sensory feedback) with whole-cell-type Poisson activation and mean rates over 1 s.
- Backward walking is approximated by reversed forward-step kinematics. Slow backward drift can be classified as "stop".
- The in-silico "hits" come from the same connectome the ranking uses (partly circular). The literature checks cover only 2 known drivers per target.
- Takeoff is one label (short and long modes are not distinguished); post-takeoff stability comes from a hand-built attitude controller.
- A few v630 -> v783 ID substitutions (1 of 21 sugar and 1 of 21 bitter GRNs dropped; one MN9 matched by cell type).
- The record keeper's final report in the latest live run calls two brain-readout checks "validated"; the tool verdict is only "consistent (brain-readout inference, no body run, 1 seed)". Cite the `analysis` events.
- No wet-lab validation. Every agent-generated hypothesis is labelled as such and needs experimental confirmation.

## Run it locally (Windows / macOS / Linux)
Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12. On first use the connectome data (~104 MB) downloads automatically; the FlyBody meshes (~134 MB) are fetched by flygym on the first flight run. Inputs and checksums: [data/manifest.json](data/manifest.json) (`uv run python spikes/audit/make_manifest.py --verify`).

```bash
uv sync
uv run streamlit run app.py                 # dashboard on http://localhost:8501
uv run python -m flylab.brain --validate    # reproduce Shiu et al. 2024 checks
uv run python -m flylab.bridge --validate   # brain -> body validation vs. literature
uv run python -m flylab.screen --benchmark  # discovery screen + speed-up
uv run python -m flylab.flight --demo       # flying body demo (video in spikes/flight/out/)
uv run python -m flylab.flight --validate   # brain -> frozen adapter -> flight validation (about 6 min)
uv run python -m flylab.export3d --all      # rebuild the 3D viewer data in web/
uv run python -m http.server 8777 --directory web   # 3D replay viewer on http://localhost:8777
uv run python spikes/audit/check_citations.py       # DOIs resolve? evidence quotes in the abstracts?
```

Agent lab (needs `ANTHROPIC_API_KEY` in `.env`, see `.env.example`; about $1.1 and 9 minutes per run); full guide in [docs/OMNIGENT.md](docs/OMNIGENT.md):

```powershell
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 -ApproveAtLaunch lab "Which visual neurons trigger escape takeoff through the giant fiber, does the simulated body actually take off, and is the giant fiber necessary?"
```

## Built on (please cite)
- Shiu P.K. et al. (2024) A Drosophila computational brain model reveals sensorimotor processing. *Nature*. doi:10.1038/s41586-024-07763-9
- Dorkenwald S. et al. (2024) and Schlegel P. et al. (2024) FlyWire whole-brain connectome and annotations, *Nature* (connectivity data v783: Zenodo 10.5281/zenodo.10676866, CC-BY-4.0)
- Wang-Chen S. et al. (2024) NeuroMechFly v2. *Nature Methods*. doi:10.1038/s41592-024-02497-y
- Vaxenburg R. et al. (2025) FlyBody, *Nature*. doi:10.1038/s41586-025-09029-4
- Bidaye S.S. et al. (2014) *Science* (MDN); Bidaye S.S. et al. (2020) *Neuron* (P9); Rayshubskiy A. et al. (2025) *eLife* (DNa02); Wu M. et al. (2016) *eLife*; Sen R. et al. (2017) *Curr Biol*; Lima & Miesenboeck (2005) *Cell*; von Reyn C.R. et al. (2014) *Nat Neurosci*; Ache J.M. et al. (2019) *Curr Biol*; Namiki S. et al. (2022) *Curr Biol*. Full list with verified DOIs in `data/ground_truth.json`.
- [Omnigent](https://github.com/omnigent-ai/omnigent) (Databricks, Apache 2.0)

## Team and credits
Maximilian Kahl: product, story, demo. Code written with Claude (Anthropic) as an AI pair programmer.
Teammate repository [JonasMayerDev/FlyBrainLab](https://github.com/JonasMayerDev/FlyBrainLab): we took over its replay-viewer plan (a Three.js page on GitHub Pages showing recorded runs only), its artefact-control rules for the embodiment (no scripted animations, separate logs for brain, adapter, body and rendering, a frozen and disclosed adapter, "a body that flies is not evidence of connectome control", whole-brain visualisation is not whole-brain dynamics; our checklist G1-G5 in [docs/CHALLENGE_REQUIREMENTS.md](docs/CHALLENGE_REQUIREMENTS.md)) and the idea of a data manifest with checksums ([data/manifest.json](data/manifest.json)). Its Shiu-file checksums equal ours.
