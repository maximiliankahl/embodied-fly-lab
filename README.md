# Embodied Fly Lab

> An agentic AI lab that wires a 139,000-neuron fruit-fly brain (FlyWire connectome) to a physics-simulated fly body and lets specialist agents generate, test and check hypotheses against published experiments. Each experiment takes seconds instead of weeks.

Built in 24 hours at the **7th Hack-Nation Global AI Hackathon** (Munich Hub, October 3–4, 2026).
Challenge 03: **Agentic Scientific Discovery** (Databricks, built with **Omnigent**).

**Demo video:** LINK · **Repository:** https://github.com/maximiliankahl/embodied-fly-lab

---

## The question

*Which visual neurons make a fly walk backward (retreat), and is the "moonwalker" descending pathway necessary for it?*

In the lab, answering a question like this means building a genetic driver line for each candidate cell type, then running optogenetic activation and behavioural tracking. That takes weeks per cell type, and the fly visual system alone has **326 visual projection cell types** (FlyWire v783).
With the full adult fly connectome now published (FlyWire, 2024), these experiments can be run *in silico* first, and the wet lab only has to test the best candidates.

## What we built

```
 Question ─► Literature agent ─► Hypothesis agent ─► Planner (≥2 designs, cost vs. information gain)
                (Europe PMC, OpenAlex,   (connectome ranking)        │
                 DOIs verified)                                      ▼
 Updated decision ◄─ Supervisor ◄─ Analysis agent ◄─ Runner ◄─ Safety agent (human approval gate)
        │                          (vs. literature      │
        └── next cycle             ground truth)        ▼
                               whole-brain LIF model ─► descending-neuron bridge ─► NeuroMechFly body (MuJoCo)
```

| Layer | What it is | Evidence it works |
|---|---|---|
| **Brain** (`flylab/brain.py`) | Event-driven numpy re-implementation of the whole-brain leaky-integrate-and-fire model of Shiu et al. 2024 (*Nature*), 138,639 neurons, 15.1 M connections, FlyWire v783 | Matches the original Brian2 model (per-neuron rates r = 0.9987). Reproduces the paper's sugar-GRN → MN9 feeding result, bitter suppression and contralateral bias. About 1 s wall time per simulated second. |
| **Bridge** (`flylab/bridge.py`) | Hand-designed interface standing in for the ventral nerve cord: descending neurons P9 → forward, MDN → backward, DNa01/DNa02 asymmetry → turning. Giant fiber, MN9 and aDN read out escape, feeding and grooming. | Reference rate measured, not assumed (148 Hz). Every term cites its paper. |
| **Body** (`flylab/body.py`) | flygym 2.1 / NeuroMechFly v2 fly in MuJoCo with a CPG + preprogrammed-step controller, plus behaviour classifier and video | 5/5 commanded behaviours classified correctly; labels stable across seeds |
| **Atlas + literature** (`flylab/atlas.py`, `literature.py`) | 51 neuron groups mapped to FlyWire v783 IDs; 23 published neuron → behaviour relations | Every DOI resolved via Europe PMC and OpenAlex; every evidence quote checked word-for-word against the abstract |
| **Discovery engine** (`flylab/screen.py`) | Connectome path-strength ranking of candidate cell types + fast whole-brain screen | See results below |
| **Agent lab** (`agents/fly_lab.yaml`) | Omnigent supervisor + 7 specialist sub-agents (literature, hypothesis, planner, safety, runner, analysis, record keeper) with typed tools and policies-as-code (cost budget, tool-call limits, loop guard, human approval gate for embodied runs) | Live run: 2 discovery cycles, 12 agent sessions, $1.03 |
| **Dashboard** (`app.py`) | Streamlit: lab notebook with agent hand-offs, experiment bench, validation & speed, method & limits | |

Every agent decision is written to a shared research record (`runs/<run_id>/record.jsonl`), so each step can be reconstructed afterwards.

## Results (measured, not claimed)

**1. Faster discovery: 25× fewer experiments to the first hit.**
Target: backward-walking command neurons (MDN). We screened all 326 visual projection types in the whole-brain model (1.24 s per experiment) and counted 12 in-silico hits.
- Ranked by connectome path strength, the **first hit comes after 1 experiment, vs. 25.2 expected for random order (25×)**. Against a stronger naive baseline ("largest cell type first") the gain is 5× to the first hit and 7× to all hits.
- A 20-experiment budget (6 % of the screen) finds 12 of 12 hits, vs. 0.7 expected for random order.
- **Independent check against literature:** the first *literature-verified* hit (LC16; Wu et al. 2016, Sen et al. 2017) comes after **12 guided experiments vs. 109 random (9×)**.
- Caveats: the all-hits speed-up (15×) holds only for one score variant (3-hop mean-field), so we lead with the first-hit number. For the escape target (giant fiber) the first hit is also 18× faster, but finding *all* hits is not faster (0.97×).

**2. Validation against published experiments: 8 of 9 comparable checks agree.**
Examples: MDN → backward walking, P9 → forward, unilateral P9 / DNa02 → ipsilateral turning, sugar GRNs → feeding motor neuron (contralateral bias as published), Johnston's organ → grooming pathway, LPLC2 → escape.
The direct descending-neuron checks are partly circular, because the bridge was built from the same papers. They mostly test wiring and signs. The checks that really test the connectome model are the sensory ones (LPLC2, sugar, JO) and the P9 → DNa02 recruitment.

**3. A surprise that changed the next decision.**
Wu et al. 2016 report that LPLC2 activation produces backward walking about as often as jumping. In the model, LPLC2 drives the giant fiber (escape) at 159 Hz but **MDN at 0 Hz**, while LC16 drives MDN at 32 Hz. The agents flagged this as inconsistent and reopened the assumption that LPLC2 retreat runs through MDN. They then planned a necessity test (LC16 with MDN silenced).
Two predictions for the wet lab follow: LPLC2-evoked retreat should **not** depend on MDN (or depends on pathways outside the brain connectome), and **LPC1 and LC6** (which drive MDN in silico but are not linked to retreat in our literature set) are new candidates to test.

## Limitations (honest)
- FlyWire covers the **brain only**. The ventral nerve cord is replaced by a hand-designed, documented bridge.
- Backward walking is approximated by reversed forward-step kinematics. Slow backward drift can be classified as "stop".
- The in-silico "hits" come from the same connectome the ranking uses (partly circular). The literature check covers only 2 known MDN hits.
- A few v630 → v783 ID substitutions (1 of 21 sugar and 1 of 21 bitter GRNs dropped; one MN9 matched by cell type).
- No wet-lab validation. Every agent-generated hypothesis is labelled as such and needs experimental confirmation.

## Run it locally (Windows / macOS / Linux)
Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12. On first use the connectome data (~104 MB) downloads automatically.

```bash
uv sync
uv run streamlit run app.py                 # dashboard on http://localhost:8501
uv run python -m flylab.brain --validate    # reproduce Shiu et al. 2024 checks
uv run python -m flylab.bridge --validate   # brain -> body validation vs. literature
uv run python -m flylab.screen --benchmark  # discovery screen + speed-up
```

Agent lab (needs `ANTHROPIC_API_KEY` in `.env`, see `.env.example`); full guide in [docs/OMNIGENT.md](docs/OMNIGENT.md):

```powershell
powershell -ExecutionPolicy Bypass -File .\agents\omni.ps1 -ApproveAtLaunch lab "Which visual projection neurons drive backward walking (retreat) in Drosophila, and is the moonwalker pathway necessary?"
```

## Built on (please cite)
- Shiu P.K. et al. (2024) A Drosophila computational brain model reveals sensorimotor processing. *Nature*. doi:10.1038/s41586-024-07763-9
- Dorkenwald S. et al. / Schlegel P. et al. (2024) FlyWire whole-brain connectome and annotations, *Nature*
- Wang-Chen S. et al. (2024) NeuroMechFly v2. *Nature Methods*. doi:10.1038/s41592-024-02497-y
- Bidaye S.S. et al. (2014) *Science* (MDN); Bidaye S.S. et al. (2020) *Neuron* (P9); Rayshubskiy A. et al. (2025) *eLife* (DNa02); Wu M. et al. (2016) *eLife*; Sen R. et al. (2017) *Curr Biol*. Full list with verified DOIs in `data/ground_truth.json`.
- [Omnigent](https://github.com/omnigent-ai/omnigent) (Databricks, Apache 2.0)

## Team
Maximilian Kahl: product, story, demo. Code written with Claude (Anthropic) as an AI pair programmer.
