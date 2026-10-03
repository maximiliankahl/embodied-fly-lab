"""Dashboard page 1: Lab notebook - replay one research run (runs/<run_id>/record.jsonl)."""

from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from flylab import ui


# --------------------------------------------------------------------------- graph


def _dot_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def handoff_dot(summary: dict) -> str:
    """Graphviz DOT of the agent hand-off chain (edge labels = step numbers)."""
    chain = summary.get("handoffs", [])
    by_agent = summary.get("by_agent", {})
    edges: dict[tuple[str, str], list[int]] = {}
    for i in range(len(chain) - 1):
        edges.setdefault((chain[i], chain[i + 1]), []).append(i + 1)
    lines = [
        "digraph G {",
        'rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;',
        'node [shape=box, style="rounded,filled", fillcolor="#eef4f7", color="#1F6F8B", '
        'fontname="Helvetica", fontsize=11, penwidth=1.2];',
        'edge [color="#7d8c9a", fontname="Helvetica", fontsize=9, fontcolor="#4a5a6a", arrowsize=0.7];',
    ]
    for agent in dict.fromkeys(chain):
        fill = "#fff3e0" if agent == "human" else ("#e6f1f5" if agent == "supervisor" else "#f4f7fa")
        lines.append(f'"{_dot_escape(agent)}" [label="{_dot_escape(agent)}\\n{by_agent.get(agent, 0)} events", '
                     f'fillcolor="{fill}"];')
    for (a, b), steps in edges.items():
        lab = ", ".join(str(s) for s in steps[:6]) + ("..." if len(steps) > 6 else "")
        lines.append(f'"{_dot_escape(a)}" -> "{_dot_escape(b)}" [label="{lab}"];')
    lines.append("}")
    return "\n".join(lines)


def loop_strip(by_type: dict) -> str:
    parts = []
    for i, (t, label) in enumerate(ui.LOOP_STAGES):
        n = by_type.get(t, 0)
        if t == "experiment_options":
            n = max(n, by_type.get("experiment_choice", 0))
        cls = "fl-step done" if n else "fl-step"
        parts.append(f'<span class="{cls}">{label}<span class="n">{n}</span></span>')
        if i < len(ui.LOOP_STAGES) - 1:
            parts.append('<span class="fl-arrow">&rarr;</span>')
    return '<div class="fl-loop">' + "".join(parts) + "</div>"


# --------------------------------------------------------------------------- event renderers


def _citations(e: dict) -> None:
    cits = [c for c in (e.get("citations") or []) if c]
    if cits:
        links = []
        for c in cits:
            c = str(c)
            links.append(ui.doi_md(c) if ("/" in c and c.startswith("10.")) or c.startswith("http") else c)
        st.markdown("Cited: " + " · ".join(links))


def _rates_chart(rates: dict | None, title: str, only_positive: bool = True, height: int = 400) -> None:
    import pandas as pd

    if rates is None:
        st.markdown(f"**{title}**")
        st.caption("No rates recorded.")
        return
    rates = {k: float(v) for k, v in rates.items() if v is not None}
    shown = {k: v for k, v in rates.items() if v > 0} if only_positive else rates
    st.markdown(f"**{title}**")
    if not shown:
        st.caption("No group fired (all 0 Hz).")
        return
    import altair as alt

    df = pd.DataFrame({"group": list(shown), "rate (Hz)": [round(v, 2) for v in shown.values()]})
    df = df.sort_values("rate (Hz)", ascending=False).head(20)
    chart = alt.Chart(df).mark_bar(color="#1F6F8B").encode(
        x=alt.X("rate (Hz):Q"), y=alt.Y("group:N", sort="-x", title=None), tooltip=["group", "rate (Hz)"]
    ).properties(height=min(height, 30 + 20 * len(df)))
    st.altair_chart(chart, width="stretch")


def _drive_block(drive: dict | None) -> None:
    if not isinstance(drive, dict):
        return
    c = st.columns(3)
    c[0].metric("forward drive", ui.fmt_num(drive.get("forward"), 2))
    c[1].metric("turn drive (neg = left)", ui.fmt_num(drive.get("turn"), 2))
    c[2].metric("backward drive", ui.fmt_num(drive.get("backward"), 2))
    if drive.get("explain"):
        with st.expander("Bridge: how rates became this drive"):
            st.json(drive["explain"], expanded=False)


def _body_block(d: dict) -> None:
    beh = d.get("behavior")
    if beh:
        st.markdown(f"Behavior: {ui.badge(beh, 'blue')}"
                    + (f" {ui.badge('fell over', 'red')}" if d.get("fell_over") else ""))
    c = st.columns(3)
    c[0].metric("forward displacement", ui.fmt_num(d.get("forward_disp_mm"), 2, " mm"))
    c[1].metric("heading change (+ = left)", ui.fmt_num(d.get("heading_change_deg"), 1, " deg"))
    c[2].metric("mean speed", ui.fmt_num(d.get("mean_speed_mm_s"), 1, " mm/s"))
    if d.get("label_warning"):
        st.caption(f"Label warning: {d['label_warning']}")


def _video(rel: str | None, mock: bool) -> None:
    p = ui.resolve(rel)
    if p is not None and p.suffix.lower() in (".mp4", ".webm", ".mov"):
        st.video(str(p))
    elif rel:
        st.caption(f"Video `{rel}` not found in this deployment.")
    else:
        st.caption("No video (mock run or rendering disabled)." if mock else "No video recorded.")


def _screen_result(d: dict) -> None:
    """run_brain_screen result: one whole-brain simulation per candidate type, target-group rates."""
    import pandas as pd

    rows = []
    for r in d.get("rows") or []:
        if not isinstance(r, dict):
            continue
        row = {"candidate": r.get("cell_type"), "n stimulated": r.get("n_stimulated")}
        for g, v in (r.get("target_rates") or {}).items():
            row[f"{g} (Hz)"] = v
        row["hit"] = ", ".join(r.get("hit_targets") or []) or ""
        row["wall s"] = r.get("runtime_s")
        row["cached"] = "yes" if r.get("cached") else ""
        rows.append(row)
    hits = d.get("hits") or []
    st.markdown(f"Targets: **{', '.join(map(str, d.get('target_groups') or []))}** · "
                f"{len(rows)} candidates · hits (>= 5 Hz): **{', '.join(map(str, hits)) or 'none'}**")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _ranking_evidence(d: dict) -> None:
    """rank_candidates evidence: a connectome prediction, not a result."""
    import pandas as pd

    st.markdown(ui.badge("computed prediction, not a result", "orange")
                + f" target **{d.get('target_group', '?')}** · {d.get('n_candidates', '?')} "
                  f"{d.get('candidate_kind', '')} candidates ranked in {ui.fmt_num(d.get('runtime_s'), 1)} s")
    top = [r for r in (d.get("top") or []) if isinstance(r, dict)]
    if top:
        cols = [c for c in ("rank", "cell_type", "n", "score", "direct_syn", "two_hop_score", "sign_note")
                if any(c in r for r in top)]
        st.dataframe(pd.DataFrame(top)[cols], hide_index=True, width="stretch")
    known = d.get("literature_known_hits") or {}
    if isinstance(known, dict) and known:
        st.caption("Literature-known inputs and their connectome rank: "
                   + "; ".join(f"{ct} rank {v.get('rank', '?')}" for ct, v in known.items() if isinstance(v, dict)))


def _result(e: dict, mock: bool) -> None:
    d = e.get("data") or {}
    kind = d.get("kind", "?")
    ex, si = d.get("excite") or [], d.get("silence") or []
    bits = [ui.badge(kind, "blue")]
    if ex:
        bits.append(f"stimulate **{', '.join(map(str, ex))}**")
    if si:
        bits.append(f"silence **{', '.join(map(str, si))}**")
    if d.get("n_active") is not None:
        bits.append(f"{d['n_active']} active neurons")
    if d.get("runtime_s") is not None:
        bits.append(f"{ui.fmt_num(d['runtime_s'], 1)} s wall")
    st.markdown(" · ".join(bits))
    art = ui.load_json(d["artifact"]) if d.get("artifact") else None
    if kind == "embodied" and isinstance(art, dict):
        b_rt = art.get("brain_runtime_s")
        body_rt = (art.get("body") or {}).get("runtime_s") if isinstance(art.get("body"), dict) else None
        if b_rt is not None or body_rt is not None:
            st.caption(f"Wall time: brain {ui.fmt_num(b_rt, 1)} s, body {ui.fmt_num(body_rt, 1)} s")
    if kind == "body":
        cols = st.columns([1, 1])
        with cols[0]:
            _drive_block(d.get("drive"))
            _body_block(d)
        with cols[1]:
            _video(d.get("video"), mock)
        return
    if kind == "embodied":
        cols = st.columns([1, 1])
        with cols[0]:
            _rates_chart(d.get("descending_group_rates_hz"), "Descending-neuron group rates (brain)")
            _drive_block(d.get("drive"))
            _body_block(d)
            bridge = d.get("bridge")
            if isinstance(bridge, dict) and bridge.get("warning"):
                st.warning(bridge["warning"])
        with cols[1]:
            _video(d.get("video"), mock)
        return
    if kind == "screen":
        _screen_result(d)
        return
    # brain (or unknown)
    cols = st.columns([1, 1])
    with cols[0]:
        _rates_chart(d.get("descending_group_rates_hz"), "Descending-neuron group rates")
    with cols[1]:
        _rates_chart(d.get("other_group_rates_hz"), "Other groups that fired (brain readouts)")
    if d.get("unknown_groups"):
        st.caption(f"Unknown groups (not simulated): {d['unknown_groups']}")


def _analysis(e: dict, gt_by_id: dict) -> None:
    d = e.get("data") or {}
    verdict = str(d.get("verdict", "?"))
    st.markdown(
        f"Verdict {ui.badge(verdict.replace('_', ' '), ui.VERDICT_COLOR.get(verdict, 'gray'))} "
        f"· `{d.get('ground_truth_id', '?')}` · expected **{d.get('expected', '?')}** "
        f"({d.get('effect', 'induce')}) vs observed **{d.get('observed', '?')}**"
        + (f" (control: {d['control']})" if d.get("control") else "")
    )
    if d.get("note"):
        st.caption(d["note"])
    if d.get("surprise"):
        st.warning("Surprise: the result disagrees (partly) with the published experiment, so assumptions are reopened.")
    gt = gt_by_id.get(str(d.get("ground_truth_id")))
    if gt:
        cit = gt.get("citation") or {}
        st.markdown(f'Published evidence: "{gt.get("evidence", "")}" ({cit.get("authors", "")} '
                    f'{cit.get("year", "")}, {ui.doi_md(cit.get("doi"))})')
    else:
        _citations(e)


def _options(e: dict, chosen_id: str | None) -> None:
    import pandas as pd

    d = e.get("data") or {}
    opts = d.get("options") or []
    rows = []
    for o in opts:
        if not isinstance(o, dict):
            rows.append({"id": "?", "description": str(o)})
            continue
        r = {}
        for k, v in o.items():
            r[k] = (", ".join(map(str, v)) if isinstance(v, list)
                    else json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
        r["chosen"] = "CHOSEN" if chosen_id is not None and str(o.get("id")) == str(chosen_id) else ""
        rows.append(r)
    if rows:
        df = pd.DataFrame(rows)
        front = [c for c in ("chosen", "id", "kind", "excite_groups", "silence_groups", "tests_hypothesis",
                             "expected_information_gain", "est_cost_units", "why") if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]
        for c in df.columns:  # agent-written options may mix numbers and text in one column
            if df[c].dtype == object:
                df[c] = df[c].map(lambda x: "" if x is None else str(x))
        st.dataframe(df, hide_index=True, width="stretch")
    if d.get("budget_units") is not None:
        st.caption(f"Remaining budget: {d['budget_units']} cost units")
    if d.get("warning"):
        st.warning(d["warning"])


def render_event(e: dict, ctx: dict) -> None:
    t = e.get("type", "note")
    d = e.get("data") if isinstance(e.get("data"), dict) else {}
    agent = str(e.get("agent", "?"))
    mock = ctx["mock"] or ui.event_is_mock(e)
    head = [f"`#{e.get('seq', '?')}`", ui.badge(agent, ui.AGENT_COLOR.get(agent, "gray")),
            f"**{ui.TYPE_LABEL.get(t, t)}**"]
    if t == "hypothesis":
        head.append(ui.badge("agent-generated", "orange"))
    if mock:
        head.append(ui.badge("MOCK", "orange"))
    ts = str(e.get("ts", ""))[11:19]
    with st.container(border=True):
        st.markdown(" ".join(head) + (f" <span class='fl-small'>{ts}</span>" if ts else ""),
                    unsafe_allow_html=True)
        content = str(e.get("content", ""))
        if t == "question":
            st.markdown(f'<div class="fl-quote">{html.escape(content)}</div>', unsafe_allow_html=True)
        elif t == "hypothesis":
            st.markdown(d.get("statement") or content)
            meta = []
            if d.get("manipulation"):
                meta.append(f"{d['manipulation']} {', '.join(map(str, d.get('target_groups') or []))}")
            if d.get("predicted_behavior"):
                meta.append(f"predicts **{d['predicted_behavior']}**")
            if d.get("confidence") is not None:
                meta.append(f"prior confidence {ui.fmt_num(d['confidence'], 2)} (agent-assigned)")
            if meta:
                st.markdown(" · ".join(meta))
            if d.get("rationale"):
                st.caption(f"Rationale: {d['rationale']}")
            _citations(e)
        elif t == "experiment_options":
            st.markdown(content)
            _options(e, ctx["chosen_after"].get(e.get("seq")))
        elif t == "experiment_choice":
            st.markdown(f"Chosen **{d.get('chosen_id', '?')}**: {d.get('rationale') or content}")
        elif t == "approval":
            status = str(d.get("status") or "")
            sb = {"approved": ("approved by a human (Omnigent ASK gate)", "green"),
                  "pre-approved": ("pre-approved at launch (scripted run)", "orange"),
                  "denied": ("denied", "red")}.get(status, (status, "gray") if status else None)
            st.markdown((ui.badge(*sb) + " " if sb else "") + f"**{d.get('action') or content}**")
            if d.get("reason"):
                st.markdown(f"Reason: {d['reason']} · est. cost {d.get('est_cost_units', '?')} units")
            if d.get("gate"):
                st.caption(d["gate"])
        elif t == "experiment_result":
            st.markdown(content)
            _result(e, mock)
        elif t == "analysis":
            st.markdown(content)
            _analysis(e, ctx["gt"])
        elif t == "decision":
            st.markdown(f"**{content}**")
            if d.get("reason"):
                st.markdown(f"Reason: {d['reason']}")
            if d.get("next_step"):
                st.markdown(f"Next step: {d['next_step']}")
            if d.get("reopens_assumption"):
                st.warning(f"Reopened assumption: {d['reopens_assumption']}")
        elif t == "evidence":
            st.markdown(content)
            if d.get("kind") == "connectome_ranking":
                _ranking_evidence(d)
            hits = d.get("hits") or []
            if hits:
                for h in hits[:8]:
                    doi = h.get("doi")
                    st.markdown(f"- {h.get('title') or '(untitled)'} ({h.get('year') or 'n.d.'})"
                                + (f" · {ui.doi_md(doi)}" if doi else "")
                                + (f" · {h.get('source')}" if h.get("source") else ""))
            _citations(e)
        else:
            st.markdown(content)
            _citations(e)
            if d and t not in ("note",):
                with st.expander("data"):
                    st.json(d, expanded=False)


# --------------------------------------------------------------------------- page


def _outcome_panel(events: list[dict]) -> None:
    st.markdown("**Outcome so far**")
    analyses = [e for e in events if e.get("type") == "analysis"]
    decisions = [e for e in events if e.get("type") == "decision"]
    if not analyses and not decisions:
        st.caption("No analysis or decision recorded yet.")
    for e in analyses:
        d = e.get("data") or {}
        v = str(d.get("verdict", "?"))
        st.markdown(f"{ui.badge(v.replace('_', ' '), ui.VERDICT_COLOR.get(v, 'gray'))} "
                    f"`{d.get('ground_truth_id', '?')}`: expected {d.get('expected', '?')}, "
                    f"observed {d.get('observed', '?')}")
    for e in decisions:
        d = e.get("data") or {}
        st.markdown(f"Decision `#{e.get('seq')}`: {e.get('content', '')}")
        if d.get("reopens_assumption"):
            st.caption(f"Reopened: {d['reopens_assumption']}")


def page() -> None:
    st.title("Lab notebook")
    st.caption("Replay of one research run: every agent action is one entry in runs/<run_id>/record.jsonl, "
               "written by the Omnigent tools while the lab works.")
    runs = ui.list_runs()
    if not runs:
        ui.placeholder("A research run", "Start the Omnigent lab (agents/omni.ps1) or create the mock run.")
        return
    mock_flags = {r: ui.run_is_mock(r) for r in runs}
    # Default: the real (non-mock) run that covers most discovery-loop stages; ties -> most recent.
    # (A 2-event smoke test must not hide a complete run.)
    stages = [t for t, _ in ui.LOOP_STAGES]

    def _coverage(r: str) -> int:
        bt_ = ui.summarize(ui.load_events(r))["by_type"]
        return sum(1 for t in stages if bt_.get(t))

    cov = {r: _coverage(r) for r in runs}
    real = [i for i, r in enumerate(runs) if not mock_flags[r]]
    default = max(real, key=lambda i: (cov[runs[i]], -i)) if real else 0
    rid = st.selectbox("Research run", runs, index=default,
                       format_func=lambda r: f"{r}  ({cov[r]}/{len(stages)} loop stages)"
                                             + ("  [MOCK]" if mock_flags[r] else ""))
    events = ui.load_events(rid)
    if not events:
        st.warning("This run has no readable events.")
        return
    mock = mock_flags[rid]
    if mock:
        ui.mock_banner("This run")
    summary = ui.summarize(events)
    question = next((e.get("content") for e in events if e.get("type") == "question"), "(no question logged)")
    st.markdown(f'<div class="fl-quote">{html.escape(str(question))}</div>', unsafe_allow_html=True)
    st.markdown(loop_strip(summary["by_type"]), unsafe_allow_html=True)

    bt = summary["by_type"]
    m = st.columns(6)
    m[0].metric("Events", summary["n_events"])
    m[1].metric("Agents", len(summary["by_agent"]))
    m[2].metric("Hypotheses", bt.get("hypothesis", 0))
    m[3].metric("Experiments run", bt.get("experiment_result", 0))
    appr = [str((e.get("data") or {}).get("status") or "") for e in events if e.get("type") == "approval"]
    m[4].metric("Approval gates", len(appr),
                help=f"approved by a human in Omnigent: {appr.count('approved')}; pre-approved at launch "
                     f"(scripted run): {appr.count('pre-approved')}; other: "
                     f"{len(appr) - appr.count('approved') - appr.count('pre-approved')}")
    m[5].metric("Decisions", bt.get("decision", 0))

    st.markdown("**Agent hand-offs** (Omnigent supervisor and specialists; edge labels = hand-off order)")
    st.graphviz_chart(handoff_dot(summary), width="stretch")
    with st.container(border=True):
        _outcome_panel(events)

    st.subheader("Timeline")
    agents = list(summary["by_agent"])
    types = [t for t in ui.TYPE_LABEL if t in bt] + [t for t in bt if t not in ui.TYPE_LABEL]
    f1, f2 = st.columns(2)
    sel_agents = f1.multiselect("Agents", agents, default=agents)
    sel_types = f2.multiselect("Event types", types, default=types,
                               format_func=lambda t: ui.TYPE_LABEL.get(t, t))
    chosen_after: dict[Any, str] = {}
    pending = None
    for e in events:
        if e.get("type") == "experiment_options":
            pending = e.get("seq")
        elif e.get("type") == "experiment_choice" and pending is not None:
            chosen_after[pending] = str((e.get("data") or {}).get("chosen_id"))
            pending = None
    ctx = {"mock": mock, "gt": ui.ground_truth_by_id(), "chosen_after": chosen_after}
    for e in events:
        if e.get("agent", "?") in sel_agents and e.get("type", "?") in sel_types:
            render_event(e, ctx)
    with st.expander("Raw record (JSONL)"):
        st.code("\n".join(json.dumps(e, ensure_ascii=False) for e in events[-200:]), language="json")
