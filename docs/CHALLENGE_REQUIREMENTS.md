# Challenge 03 requirements (Hack-Nation 7 × Databricks, "Agentic Scientific Discovery")

Source: official challenge brief (C3 PDF, 4 pages) and the team submission plan (JonasMayerDev/FlyBrainLab, HACKATHON_PLAN.md §9).
This file is the checklist every part of the project is measured against. `docs/CHALLENGE_COMPLIANCE.md` maps each item to evidence in the repo.

## Mandatory platform
- R1 **Build with Omnigent** (managed on Databricks or open source). Omnigent must **orchestrate the live discovery workflow**.
- R2 Show **multiple specialist agents exchanging outputs, using tools and adapting their plan after an experimental result**.

## Agent design ("Design the handoffs that accelerate discovery")
- R3 For **each agent** specify: the **scientific decision it owns**, the **tools** it can use, its **inputs** and its **output**.
- R4 Pass **structured evidence, candidate IDs, experiment specifications and results** between agents.
- R5 Keep a **shared research record** so every decision can be reconstructed.
- R6 Run **independent searches or experiments in parallel**.
- R7 Let **surprising results reopen an earlier assumption**.
- R8 Give the **planner a budget** and make it **choose between competing tests**.
- R9 **Human approval**: scientists set the objective and approve consequential actions; a **safety agent** flags risks and requests approval; the boundary is **enforced through tool permissions and Omnigent policies** (not only prompts).

## Experiment ("Run an experiment that changes the next decision")
- R10 Clear **question** and a **measurable scientific outcome**.
- R11 Use literature, structured data, simulations, models or generated data to **develop a hypothesis**.
- R12 Design **at least two possible tests** and choose one using **expected learning, feasibility and cost**.
- R13 **Run** the test, **interpret** the result and use it to **determine the next step**; the evidence must **change the next scientific decision**.
- R14 Experiment = reproducible computational test (simulation, screening, benchmark, model comparison, sensitivity/counterfactual analysis, …).

## Acceleration ("Show progress toward 10× faster discovery")
- R15 Identify **one bottleneck** in scientific discovery and show how the lab compresses/automates/improves it.
- R16 **Report the improvement actually observed** (1.5×, 3×, … honest numbers; strength of evidence > size of claim).
- R17 **One complete discovery loop**: Question → Evidence → Hypothesis → Experiment → Result → Updated decision.
- R18 Explain **what bottleneck** was attacked, **what the agents learned**, **what should happen next**, and **what would be needed to approach 10× at scale**.

## Rigor and responsibility
- R19 **Citations for factual claims**; attach source evidence or run records.
- R20 **Label agent-generated hypotheses**; **preserve uncertainty**.
- R21 Document **controls** and **human approval gates**.
- R22 State the **validation still needed before real-world use**.

## Judging weights
Omnigent orchestration 30 % · breakthrough potential 25 % · discovery acceleration and learning 20 % · scientific rigor 15 % · creativity and responsibility 10 %.

## Submission (C3 brief + team plan)
- S1 **Repository** (public) with **agent specifications and policies**, **cited evidence**, **experiment code and results**, **measured improvement**, **next experiment**.
- S2 **Two-minute demo** showing: the question, agent hand-offs, the experiment, the result, what the lab learned, what it would investigate next.
- S3 Hackathon platform: team photo (JPG/PNG/WebP ≤ 10 MB); three videos ≤ 60 s each (team introduction, product demo, technical walkthrough; MP4/MOV ≤ 1 GB); submit on **HackOS and the Google Form** (both confirmations).
- S4 Public demo link (team plan: Three.js replay viewer on GitHub Pages showing **recorded** simulation runs, never claiming live re-computation).

## Team ground rules for the embodiment (FlyBrainLab plan, "Artefaktkontrollen")
- G1 No "if neuron X fires, play an animation": behaviour must come from body physics driven by a documented adapter.
- G2 Log brain response, adapter output, body physics and rendering separately.
- G3 Freeze and disclose the adapter (read-out neurons, parameters) before comparison runs.
- G4 A body controller that can walk/fly is not evidence of connectome control: state exactly what the connectome decides.
- G5 Whole-brain visualisation ≠ whole-brain dynamics; label sub-networks and open-loop (no sensory feedback) honestly.
