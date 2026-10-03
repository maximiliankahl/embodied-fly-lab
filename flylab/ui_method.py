"""Dashboard page 4: Method & limits - architecture, what is validated, honest limitations, citations."""

from __future__ import annotations

import streamlit as st

from flylab import ui

ARCH_DOT = r"""
digraph A {
  rankdir=LR; bgcolor="transparent"; nodesep=0.18; ranksep=0.5; newrank=true;
  node [shape=box, style="rounded,filled", fillcolor="#f4f7fa", color="#1F6F8B", fontname="Helvetica", fontsize=10, margin="0.12,0.05"];
  edge [color="#7d8c9a", fontname="Helvetica", fontsize=8, fontcolor="#4a5a6a", arrowsize=0.6];

  human [label="Human PI\nquestion + approvals", fillcolor="#fff3e0", color="#E07B39"];

  subgraph cluster_omni {
    label="Omnigent lab  (agents/fly_lab.yaml)"; fontname="Helvetica"; fontsize=10; color="#1F6F8B"; style="rounded";
    supervisor [label="supervisor (PI agent)\ndelegates, adapts the plan", fillcolor="#e6f1f5"];
    policies [label="policies: approval gate (ASK),\nembodied-run cap (DENY), cost budget,\ntool-call limit, loop guard", shape=note, fillcolor="#fff8ec", color="#E07B39"];
    { rank=same; literature; hypothesis; planner; safety; runner; analysis; record_keeper; }
  }

  tools [label="flylab.tools\n__NTOOLS__ function tools\nlog every step", fillcolor="#eef4f7"];

  subgraph cluster_sci {
    label="Science stack  (flylab/)"; fontname="Helvetica"; fontsize=10; color="#7d8c9a"; style="rounded,dashed";
    { rank=same;
      lit [label="literature: Europe PMC + OpenAlex"];
      atlas [label="atlas: FlyWire v783 groups + ground truth (DOIs)"];
      screen [label="screen: connectome ranking of candidates\n+ fast whole-brain screen"];
      brain [label="brain: whole-brain LIF model, 138,639 neurons\n(after Shiu et al. 2024)"];
      bridge [label="bridge: descending-neuron rates -> forward / turn / backward\n(hand-designed, replaces the VNC)", fillcolor="#fff8ec", color="#E07B39"];
      body [label="body: NeuroMechFly v2 (flygym, MuJoCo)\nbehavior label + video"];
    }
  }

  record [label="research record\nruns/<id>/record.jsonl", shape=cylinder, fillcolor="#eef4f7"];
  dash [label="dashboard\n(this app)", fillcolor="#e6f1f5"];

  human -> supervisor [label="question"];
  supervisor -> literature [label="1"]; supervisor -> hypothesis [label="2"]; supervisor -> planner [label="3"];
  supervisor -> safety [label="4"]; supervisor -> runner [label="5"]; supervisor -> analysis [label="6"];
  supervisor -> record_keeper [label="7"];
  human -> policies [label="approve / deny", style=dashed, color="#E07B39", dir=both];
  literature -> tools; runner -> tools [label="experiments"]; analysis -> tools;
  policies -> tools [label="gate each call", style=dashed, color="#E07B39"];
  tools -> lit; tools -> atlas; tools -> brain; tools -> screen; screen -> brain [label="1 sim / candidate"];
  brain -> bridge [label="DN rates"]; bridge -> body [label="drive"];
  tools -> record -> dash;
}
"""

VALIDATED = [
    ("Brain model", "Re-implementation of the Shiu et al. 2024 LIF model reproduces the published sugar GRN -> MN9 "
                    "activation (10/10 trials), bitter suppression, 0 Hz baseline; Brian2 cross-check on a "
                    "426-neuron subnetwork gives r = 0.999 (Validation & speed, Brain-model fidelity)."),
    ("Neuron atlas", "Every group maps to FlyWire v783 root ids from the public annotation release; DNa01/DNa02 "
                     "left/right ids match the ids published by Rayshubskiy et al. 2025."),
    ("Ground truth", "{n_gt} published activation/silencing results; every DOI resolved through Europe PMC/OpenAlex "
                     "and every evidence string is a verbatim quote from the paper."),
    ("Body", "flygym's hybrid turning controller; forward, turning, backward and stop drives were checked to "
             "produce the expected behavior labels."),
    ("Agent loop", "Mock-mode self-test of the tool chain (question to decision) and of the approval-gate policy (ASK; DENY after the embodied-run cap)."),
]

LIMITS = [
    "The ventral nerve cord (VNC) is not simulated. A hand-designed bridge maps descending-neuron rates to a "
    "3-number locomotor drive; this is where most of the behaviour is decided, and it is our design, not biology.",
    "Backward walking is approximated by replaying forward step kinematics in reverse; real backward stepping "
    "uses different leg kinematics.",
    "Short runs drift (straight walking turns about 7 deg/s to the left), so runs shorter than 1 s can be mislabelled as turns.",
    "The brain model is a leaky integrate-and-fire network with synapse counts as weights: no neuromodulation, "
    "plasticity, gap junctions or graded transmission, and neurotransmitter signs are predicted, not measured.",
    "Some neuron groups substitute cell types or IDs (e.g. Shiu-era v630 sets reused in v783, left/right "
    "convention change); every group carries a confidence level and mapping source in data/neurons.json.",
    "Escape, grooming and feeding cannot be expressed by the walking body; they are read out at brain level only.",
    "No wet-lab validation: agreement with published experiments is necessary, not sufficient. "
    "Hypotheses are agent-generated and are labelled as such.",
    "Speed-up numbers are measured on one laptop for this pipeline and a small candidate set; they are not a "
    "general claim about lab experiments. The in-silico hits are defined by the same connectome model that ranks "
    "the candidates (partly circular), the primary ranking score was chosen after a pilot, and the literature "
    "check rests on only 2 known inputs per target (see Validation & speed, Acceleration).",
    "The end-to-end agreement counts include direct descending-neuron activations whose outcome the bridge was "
    "designed to produce (same papers), so they test plumbing and sign conventions, not discovery.",
]

CORE_REFS = [
    ("Shiu PK et al. (2024) A Drosophila computational brain model reveals sensorimotor processing. Nature.",
     "10.1038/s41586-024-07763-9"),
    ("Dorkenwald S et al. (2024) Neuronal wiring diagram of an adult brain. Nature.", "10.1038/s41586-024-07558-y"),
    ("Schlegel P et al. (2024) Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature.",
     "10.1038/s41586-024-07686-5"),
    ("Stürner T et al. (2025) Comparative connectomics of Drosophila descending and ascending neurons. Nature.",
     "10.1038/s41586-025-08925-z"),
    ("Wang-Chen S et al. (2024) NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila. "
     "Nature Methods.", "10.1038/s41592-024-02497-y"),
]


def _n_tools() -> str:
    """Number of Omnigent function tools, read from flylab/tools.py source (no import)."""
    import re

    try:
        src = (ui.ROOT / "flylab" / "tools.py").read_text(encoding="utf-8")
        m = re.search(r"^ALL_TOOLS\s*=\s*\[(.*?)\]", src, re.S | re.M)
        if m:
            return str(len([x for x in m.group(1).replace("\n", " ").split(",") if x.strip()]))
    except OSError:
        pass
    return "the"


def _table(rows: list) -> "object":
    import pandas as pd

    df = pd.DataFrame([r for r in rows if isinstance(r, dict)])
    for c in df.columns:
        df[c] = df[c].map(lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple)) else x)
    return df


def _bridge_table() -> None:
    st.subheader("The brain-body bridge (hand-designed, replaces the ventral nerve cord)")
    if not ui.module_available("bridge"):
        st.caption("flylab/bridge.py is not available in this build yet.")
        return
    try:
        from flylab import bridge

        desc = bridge.describe()
    except Exception as exc:  # the dashboard must not die on a bridge problem
        st.caption(f"bridge.describe() is not available ({type(exc).__name__}: {exc}).")
        return
    if not isinstance(desc, dict):
        st.json(desc)
        return
    if isinstance(desc.get("scope"), str):
        st.markdown(desc["scope"])
    if desc.get("activation"):
        st.markdown(f"Activation: `{desc['activation']}` with reference rate **{desc.get('ref_rate_hz', '?')} Hz** "
                    "(calibrated: the rate a descending neuron reaches under direct 150 Hz drive in the brain model).")
    terms = desc.get("drive_terms") or desc.get("mapping") or desc.get("rows")
    if isinstance(terms, list) and terms:
        st.markdown("**Drive terms** (what the body receives)")
        df = _table(terms)
        st.dataframe(df[[c for c in ("output", "formula", "sign", "citations", "description") if c in df.columns]
                        or list(df.columns)], hide_index=True, width="stretch")
    reads = desc.get("readouts")
    if isinstance(reads, list) and reads:
        st.markdown("**Brain-level readouts** (behaviours the walking body cannot show)")
        st.dataframe(_table(reads), hide_index=True, width="stretch")
    lims = desc.get("limitations")
    if isinstance(lims, list) and lims:
        with st.expander("Bridge limitations"):
            for x in lims:
                st.markdown(f"- {x}")
    with st.expander("bridge.describe() (raw)"):
        st.json(desc, expanded=False)


def page() -> None:
    st.title("Method & limits")
    st.caption("What the system is, what has been checked, and where it can be wrong.")
    st.subheader("Architecture")
    st.graphviz_chart(ARCH_DOT.replace("__NTOOLS__", _n_tools()), width="stretch")
    st.markdown(
        "One discovery loop: **question -> evidence -> hypothesis -> at least two experiment designs, chosen by "
        "expected information gain vs. compute cost -> human approval -> simulation -> comparison with the "
        "published result -> updated decision**. A surprise reopens an assumption and starts a second cycle. "
        "Every step is written to the research record, which this dashboard replays."
    )
    _bridge_table()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is validated")
        n_gt = len(ui.ground_truth())
        for title, text in VALIDATED:
            st.markdown(f"**{title}.** {text.replace('{n_gt}', str(n_gt))}")
        vdoc = ui.load_json(ui.BENCH / "embodied_validation.json")
        vc = ui.validation_counts(vdoc.get("summary") if isinstance(vdoc, dict) else None)
        if vc:
            st.markdown(f"**End to end.** Brain -> bridge -> body is consistent with {vc['n_ok']} of {vc['n']} "
                        f"informative published checks ({vc['breakdown']})"
                        + (f"; {vc['n_uninformative']} further check was uninformative" if vc["n_uninformative"] else "")
                        + ". Direct descending-neuron rows are partly circular (bridge designed from the same "
                          "papers); details and videos: Validation & speed.")
    with c2:
        st.subheader("Limitations")
        for text in LIMITS:
            st.markdown(f"- {text}")
    st.subheader("Citations")
    st.markdown("**Models and data**")
    for text, doi in CORE_REFS:
        st.markdown(f"- {text} {ui.doi_md(doi)}")
    seen = set()
    refs = []
    for e in ui.ground_truth():
        cit = e.get("citation") or {}
        doi = cit.get("doi")
        if doi and doi.lower() not in seen and doi not in {d for _, d in CORE_REFS}:
            seen.add(doi.lower())
            refs.append((cit.get("authors") or "", cit.get("year") or "", cit.get("title") or "", doi))
    if refs:
        st.markdown("**Published experiments used as ground truth**")
        for a, y, t, doi in sorted(refs, key=lambda r: (str(r[1]), r[0])):
            st.markdown(f"- {a} ({y}) {t} {ui.doi_md(doi)}")
    st.markdown("**Software**: Omnigent (agent orchestration and policies), Claude (agent model), flygym / MuJoCo, "
                "Streamlit.")
