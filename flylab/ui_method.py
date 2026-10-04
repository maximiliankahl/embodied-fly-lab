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
    { rank=same; __AGENTS__ }
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
      body [label="body: NeuroMechFly v2 (flygym, MuJoCo)\nwalking: behavior label + video"];
      __FLIGHT_NODES__
    }
  }

  record [label="research record\nruns/<id>/record.jsonl", shape=cylinder, fillcolor="#eef4f7"];
  dash [label="dashboard\n(this app)", fillcolor="#e6f1f5"];

  human -> supervisor [label="question"];
  __AGENT_EDGES__
  human -> policies [label="approve / deny", style=dashed, color="#E07B39", dir=both];
  __TOOL_EDGES__
  policies -> tools [label="gate each call", style=dashed, color="#E07B39"];
  tools -> lit; tools -> atlas; tools -> brain; tools -> screen; screen -> brain [label="1 sim / candidate"];
  brain -> bridge [label="DN rates"]; bridge -> body [label="drive"];
  __FLIGHT_EDGES__
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
    "the candidates (partly circular), the primary ranking score was chosen after a pilot, random order is a weak "
    "baseline (a no-connectome 'largest types first' order shrinks the gain considerably), several near-threshold "
    "hits do not replicate with a second seed, and the literature check rests on only 2 known inputs per target "
    "(see Validation & speed, Acceleration).",
    "The end-to-end agreement counts include direct descending-neuron activations whose outcome the bridge was "
    "designed to produce (same papers), so they test plumbing and sign conventions, not discovery.",
]

FLIGHT_LIMITS = [
    "Flight: the connectome model only sets the adapter inputs (takeoff trigger from the giant fiber, thrust and "
    "steering from flight descending neurons). Wing kinematics and flight stabilisation come from the body "
    "controller, so a flying body is not by itself evidence of connectome-controlled flight (ground rule G4).",
    "All runs are open loop: the body does not feed vision or mechanosensation back into the brain model.",
    "The 3D viewer replays recorded runs; a whole-brain point cloud is a visualisation, not a re-simulation (G5).",
    "The movement verifier checks the body's movement against the expected behaviour; its vision verdict comes "
    "from a language model looking at keyframes and can be wrong, so kinematic and vision verdicts are shown "
    "side by side and disagreements are flagged.",
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


DEFAULT_AGENTS = ["literature", "hypothesis", "planner", "safety", "runner", "analysis", "record_keeper"]


def _agents() -> list[str]:
    """Specialist sub-agents declared in agents/fly_lab.yaml (type: agent), else the default list."""
    try:
        import yaml

        doc = yaml.safe_load((ui.ROOT / "agents" / "fly_lab.yaml").read_text(encoding="utf-8"))
        tools = doc.get("tools") if isinstance(doc, dict) else None
        names = [k for k, v in (tools or {}).items() if isinstance(v, dict) and v.get("type") == "agent"]
        return names or DEFAULT_AGENTS
    except Exception:
        return DEFAULT_AGENTS


def arch_dot() -> str:
    agents = _agents()
    q = lambda x: '"' + str(x).replace('"', "") + '"'  # noqa: E731
    agent_edges = " ".join(f"supervisor -> {q(a)} [label=\"{i}\"];" for i, a in enumerate(agents, 1))
    callers = [a for a in agents if any(k in a for k in ("literature", "runner", "analysis", "verif"))]
    tool_edges = " ".join(f"{q(a)} -> tools" + (' [label="experiments"]' if "runner" in a else "") + ";"
                          for a in callers)
    has_flight = ui.module_available("flight") or (ui.BENCH / "flight_validation.json").exists()
    has_verify = ui.module_available("verify")
    has_web = (ui.WEB / "index.html").exists() or ui.module_available("export3d")
    nodes, edges = [], []
    if has_flight:
        nodes.append('flight [label="flight: winged body in MuJoCo\\ntakeoff / thrust / yaw command"];')
        edges.append('bridge -> flight [label="flight command"];')
    if has_verify:
        nodes.append('verify [label="verify: kinematics recomputed from trajectory\\n+ vision check of keyframes", '
                     'fillcolor="#e9f5ec", color="#2e7d4f"];')
        edges.append("body -> verify;" + (" flight -> verify;" if has_flight else "") + " tools -> verify;")
    if has_web:
        nodes.append('viewer [label="3D replay viewer (web/, Three.js)\\nrecorded poses, not live", fillcolor="#e6f1f5"];')
        edges.append("body -> viewer [style=dashed];" + (" flight -> viewer [style=dashed];" if has_flight else ""))
    return (ARCH_DOT.replace("__NTOOLS__", _n_tools()).replace("__AGENTS__", "; ".join(q(a) for a in agents) + ";")
            .replace("__AGENT_EDGES__", agent_edges).replace("__TOOL_EDGES__", tool_edges)
            .replace("__FLIGHT_NODES__", " ".join(nodes)).replace("__FLIGHT_EDGES__", " ".join(edges)))


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
    try:
        st.graphviz_chart(arch_dot(), width="stretch")
    except Exception as exc:  # a broken diagram must not take the page down
        st.caption(f"Architecture diagram unavailable ({type(exc).__name__}).")
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
        fs = ui.flight_summary(ui.flight_rows(ui.load_json(ui.BENCH / "flight_validation.json")))
        if fs["n_conditions"]:
            st.markdown(f"**Flight and movement verification.** {fs['n_conditions']} flight conditions; the movement "
                        f"verifier judged {fs['n_verified_correct']} of {fs['n_verified']} movements correct"
                        + (f", kinematic and vision verdicts agree in {fs['n_agree']} of {fs['n_both_verifiers']}"
                           if fs["n_both_verifiers"] else "")
                        + (f"; {fs['n_gt_consistent']} of {fs['n_gt']} published flight checks consistent"
                           if fs["n_gt"] else "") + " (details: Flight & 3D).")
    with c2:
        st.subheader("Limitations")
        for text in LIMITS:
            st.markdown(f"- {text}")
        if ui.module_available("flight") or (ui.BENCH / "flight_validation.json").exists() or ui.WEB.exists():
            for text in FLIGHT_LIMITS:
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
    _checklist()


# --------------------------------------------------------------------------- challenge checklist

# Where each requirement can be seen in THIS dashboard (pointers, not a self-assessment; the evidence and status
# columns come from docs/CHALLENGE_COMPLIANCE.md when it exists).
DASH_PAGES = {"notebook": "Lab notebook", "bench": "Experiment bench", "flight": "Flight & 3D",
              "validation": "Validation & speed", "method": "Method & limits"}
DASH_MAP = {
    "R1": ["notebook", "method"], "R2": ["notebook"], "R3": ["method"], "R4": ["notebook"], "R5": ["notebook"],
    "R6": ["notebook"], "R7": ["notebook"], "R8": ["notebook"], "R9": ["notebook", "method"],
    "R10": ["notebook"], "R11": ["notebook"], "R12": ["notebook"], "R13": ["notebook"],
    "R14": ["bench", "validation"], "R15": ["validation"], "R16": ["validation"], "R17": ["notebook"],
    "R18": ["validation", "method"], "R19": ["validation", "method"], "R20": ["notebook", "validation"],
    "R21": ["notebook", "method"], "R22": ["method"], "S1": [], "S2": [], "S3": [], "S4": ["flight"],
    "G1": ["flight", "method"], "G2": ["notebook", "flight"], "G3": ["flight", "method"], "G4": ["flight", "method"],
    "G5": ["flight", "method"],
}
_ID = r"(?:R|S|G)\d{1,2}"


def requirements() -> list[dict]:
    """Items of docs/CHALLENGE_REQUIREMENTS.md: [{id, group, text}] in file order."""
    import re

    path = ui.DOCS / "CHALLENGE_REQUIREMENTS.md"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out, group = [], ""
    for line in lines:
        if line.startswith("## "):
            group = re.sub(r"\s*\(.*\)\s*$", "", line[3:].strip())
            continue
        m = re.match(rf"^\s*[-*]\s+({_ID})\s+(.*)$", line)
        if m:
            out.append({"id": m.group(1), "group": group, "text": m.group(2).strip()})
    return out


def _repo_links(md: str) -> str:
    """Relative markdown links (from docs/) -> GitHub URLs, so they work inside the dashboard."""
    import re

    def fix(m):
        text, target = m.group(1), m.group(2).strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        path, _, anchor = target.partition("#")
        for base in (ui.DOCS, ui.ROOT / "docs", ui.ROOT):
            cand = (base / path).resolve()
            try:
                rel = cand.relative_to(ui.ROOT.resolve())
            except ValueError:
                continue
            if cand.exists():
                kind = "tree" if cand.is_dir() else "blob"
                return f"[{text}]({ui.REPO_URL}/{kind}/main/{rel.as_posix()}" + (f"#{anchor}" if anchor else "") + ")"
        return m.group(0)

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix, md)


_STATUS_WORDS = ("done", "met", "yes", "partial", "partly", "missing", "open", "todo", "planned", "n/a", "pending",
                 "in progress", "not met", "fulfilled", "demonstrated")


def compliance() -> dict[str, dict]:
    """id -> {evidence, status} from docs/CHALLENGE_COMPLIANCE.md (tables, list items or headings)."""
    import re

    path = ui.DOCS / "CHALLENGE_COMPLIANCE.md"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out: dict[str, dict] = {}
    cur = None
    for line in lines:
        st_ = line.strip()
        if st_.startswith("|"):
            cells = [c.strip() for c in st_.strip("|").split("|")]
            idx = next((i for i, c in enumerate(cells) if re.fullmatch(rf"[*`]*({_ID})[*`]*", c)), None)
            if idx is None:
                continue
            rid = re.sub(r"[*`]", "", cells[idx])
            rest = [c for i, c in enumerate(cells) if i != idx]
            status = ""
            for c in list(rest):
                low = re.sub(r"[*`_]", "", c).strip().lower()
                if low and (any(low.startswith(w) for w in _STATUS_WORDS) and len(low) < 40
                            or any(ch in c for ch in "\u2705\u274c\u26a0\U0001f7e1\U0001f7e2\U0001f534")):
                    status = c
                    rest.remove(c)
                    break
            evidence = " · ".join(rest[1:]) if len(rest) >= 2 else (rest[0] if rest else "")
            out[rid] = {"evidence": evidence, "status": status}
            cur = None
            continue
        m = re.match(rf"^\s*[-*]\s+[*`]*({_ID})[*`]*[\s:.)\u2013-]*(.*)$", line)
        if m:
            out[m.group(1)] = {"evidence": m.group(2).strip(), "status": ""}
            cur = None
            continue
        m = re.match(rf"^#+\s+[*`]*({_ID})\b[*`]*[\s:.)\u2013-]*(.*)$", line)
        if m:
            cur = m.group(1)
            out[cur] = {"evidence": "", "status": ""}
            continue
        if cur and st_ and not st_.startswith("#"):
            ev = out[cur]["evidence"]
            if len(ev) < 400:
                out[cur]["evidence"] = (ev + " " + st_).strip()
        elif st_.startswith("#"):
            cur = None
    return out


def _cell(x: str) -> str:
    return str(x or "").replace("|", "\\|").replace("\n", " ")


def _checklist() -> None:
    st.subheader("Challenge checklist")
    reqs = requirements()
    if not reqs:
        st.caption("docs/CHALLENGE_REQUIREMENTS.md is not available in this build.")
        return
    comp = compliance()
    src = ("Evidence and status from [docs/CHALLENGE_COMPLIANCE.md]"
           f"({ui.REPO_URL}/blob/main/docs/CHALLENGE_COMPLIANCE.md)." if comp else
           "docs/CHALLENGE_COMPLIANCE.md is not written yet, so only the requirements and the dashboard pages that "
           "show them are listed (no self-assessed status).")
    st.caption("Requirements of Challenge 03 (Databricks, Agentic Scientific Discovery) from "
               f"[docs/CHALLENGE_REQUIREMENTS.md]({ui.REPO_URL}/blob/main/docs/CHALLENGE_REQUIREMENTS.md). "
               "'Shown in' links open the dashboard page where the item can be seen. " + src)
    groups: dict[str, list[dict]] = {}
    for r in reqs:
        groups.setdefault(r["group"] or "Other", []).append(r)
    short = {"Mandatory platform": "Platform", "Agent design": "Agents", "Experiment": "Experiment",
             "Acceleration": "Acceleration", "Rigor and responsibility": "Rigor", "Submission": "Submission"}
    names = list(groups)
    tabs = st.tabs([next((v for k, v in short.items() if g.startswith(k)), "Embodiment rules" if "embodiment" in g.lower()
                         else g[:20]) for g in names])
    for tab, g in zip(tabs, names):
        with tab:
            head = "| ID | Requirement | Shown in |" + (" Evidence | Status |" if comp else "")
            sep = "|---|---|---|" + ("---|---|" if comp else "")
            lines = [head, sep]
            for r in groups[g]:
                shown = ", ".join(f"[{DASH_PAGES[p]}](./{p})" for p in DASH_MAP.get(r["id"], []) if p in DASH_PAGES)
                if r["id"] == "S1":
                    shown = f"[GitHub repository]({ui.REPO_URL})"
                row = f"| **{r['id']}** | {_cell(r['text'])} | {shown or 'outside the dashboard'} |"
                if comp:
                    c = comp.get(r["id"], {})
                    row += f" {_cell(_repo_links(c.get('evidence', '')))} | {_cell(c.get('status', ''))} |"
                lines.append(row)
            st.markdown("\n".join(lines))
