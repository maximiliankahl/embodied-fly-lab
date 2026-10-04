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


def _flight_block(d: dict, art: dict | None = None) -> None:
    """Flight result: adapter command (takeoff / thrust / yaw / pitch) + physics outcome."""
    art = art if isinstance(art, dict) else {}
    cmd = d.get("command") if isinstance(d.get("command"), dict) else {}
    if str(d.get("kind")) == "flight_body":
        st.caption("Body-only control: the flight command was set by hand, no brain involved (G4: a body that can "
                   "fly is not evidence of connectome control).")
    if cmd:
        c = st.columns(4)
        c[0].metric("takeoff", ui.fmt_num(cmd.get("takeoff"), 2))
        c[1].metric("thrust", ui.fmt_num(cmd.get("thrust"), 2))
        c[2].metric("yaw (neg = left)", ui.fmt_num(cmd.get("yaw"), 2))
        c[3].metric("pitch", ui.fmt_num(cmd.get("pitch"), 2))
        binfo = d.get("bridge") if isinstance(d.get("bridge"), dict) else {}
        explain = cmd.get("explain") or binfo.get("explain")
        if binfo.get("mock"):
            st.warning(str(binfo.get("bridge") or "MOCK flight adapter"))
        if explain:
            with st.expander("Adapter: how rates became this flight command"):
                st.json(explain, expanded=False)
    if d.get("mock_parts"):
        st.markdown("Mocked parts: " + " ".join(ui.badge(str(x), "orange") for x in d["mock_parts"]))
    full = art.get("flight") if isinstance(art.get("flight"), dict) else {}
    fl = {**full, **{k: v for k, v in d.items() if v is not None}}
    beh = fl.get("behavior") or d.get("behavior")
    if beh:
        st.markdown(f"Behavior: {ui.badge(beh, 'blue')}"
                    + (f" {ui.badge('airborne', 'green')}" if fl.get("airborne") else ""))
    c = st.columns(3)
    c[0].metric("max height", ui.fmt_num(fl.get("max_height_mm"), 2, " mm"))
    c[1].metric("flight time", ui.fmt_num(fl.get("flight_time_s"), 2, " s"))
    c[2].metric("heading change (+ = left)", ui.fmt_num(fl.get("heading_change_deg"), 1, " deg"))


def _verification_block(v: dict, compact: bool = False) -> None:
    """flylab.verify result: kinematic verdict (recomputed from the raw trajectory) vs. vision verdict."""
    vs = ui.verification_summary(v)
    if not vs:
        return
    final = str(vs.get("final") or "?")
    line = [f"Movement verifier {ui.badge(final, ui.VERIFY_COLOR.get(final, 'gray'))}"]
    if vs.get("expected"):
        line.append(f"expected **{vs['expected']}**" + (f" ({vs['mode']})" if vs.get("mode") else ""))
    if vs.get("agreement") is not None:
        line.append("kinematic and vision " + (ui.badge("agree", "green") if vs["agreement"]
                                                else ui.badge("disagree", "red")))
    st.markdown(" · ".join(line))
    if compact:
        return
    import pandas as pd

    c1, c2 = st.columns([1, 1])
    with c1:
        kv = str(vs.get("kinematic") or "n/a")
        st.markdown(f"**Kinematic check** {ui.badge(kv, ui.VERIFY_COLOR.get(kv, 'gray'))}")
        if vs.get("checks"):
            df = pd.DataFrame(vs["checks"])
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].map(lambda x: "" if x is None else str(x))
            st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Recomputed from the raw trajectory, independent of the body's own behaviour classifier.")
    with c2:
        if vs.get("vision_used"):
            vv = str(vs.get("vision") or "n/a")
            st.markdown(f"**Vision check** {ui.badge(vv, ui.VERIFY_COLOR.get(vv, 'gray'))}"
                        + (f" <span class='fl-small'>{vs.get('model')}, {vs.get('frames_used')} frames</span>"
                           if vs.get("model") else ""), unsafe_allow_html=True)
            obs = vs.get("observations")
            if obs:
                st.caption(obs if isinstance(obs, str) else "; ".join(map(str, obs)) if isinstance(obs, list)
                           else str(obs))
        else:
            st.markdown("**Vision check** not used for this movement.")
        cs = ui.resolve(vs.get("contact_sheet"))
        if cs is not None and cs.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            st.image(str(cs), caption="Keyframe contact sheet", width="stretch")
        elif vs.get("contact_sheet"):
            st.caption(f"Contact sheet `{vs['contact_sheet']}` not found in this deployment.")
    if vs.get("error"):
        st.caption(f"Verifier note: {vs['error']}")


def _is_flight(d: dict) -> bool:
    return (str(d.get("kind", "")).lower() in ("flight", "embodied_flight", "flight_body")
            or str(d.get("mode", "")).lower() == "flight")


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
    if _is_flight(d):
        fl = d.get("flight") if isinstance(d.get("flight"), dict) else {}
        cols = st.columns([1, 1])
        with cols[0]:
            _rates_chart(d.get("descending_group_rates_hz") or d.get("key_group_rates_hz"),
                         "Descending-neuron group rates (brain)")
            _flight_block(d, art if isinstance(art, dict) else None)
        with cols[1]:
            _video(d.get("video") or fl.get("video"), mock)
        if isinstance(d.get("verification"), dict):
            _verification_block(d["verification"], compact=True)
        return
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
    tags = []
    if d.get("by_construction"):
        tags.append(ui.badge("by construction: the bridge maps this group to this behaviour (circular)", "orange"))
    ls = d.get("label_source")
    if ls == "agent_override":
        tags.append(ui.badge("label set by the agent, not the body classifier", "red"))
    elif ls == "brain_readout_inference":
        tags.append(ui.badge("label inferred from brain readout", "gray"))
    elif ls == "body_classifier":
        tags.append(ui.badge("label from the body classifier", "blue"))
    if d.get("manipulation_match") is False:
        tags.append(ui.badge("not a test of this entry (manipulation mismatch)", "red"))
    if d.get("movement_verified") is True:
        tags.append(ui.badge("movement confirmed by the verifier", "green"))
    elif d.get("movement_verified") is False:
        tags.append(ui.badge("movement NOT confirmed by the verifier", "red"))
    if tags:
        st.markdown(" ".join(tags))
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
        elif t == "movement_verification":
            st.markdown(content)
            _verification_block(ui.verification_of(d))
        elif t == "experiment_batch" or "batch" in t or d.get("parallel_batch"):
            st.markdown(content)
            _batch_event(d)
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


# --------------------------------------------------------------------------- parallel batches


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _batch_id(d: dict) -> str | None:
    b = d.get("batch_id")
    if not b and isinstance(d.get("batch"), dict):
        b = d["batch"].get("id")
    return str(b) if b else None


def batches(events: list[dict]) -> list[dict]:
    """Parallel experiment batches: batch events plus experiment results that carry a batch_id."""
    out: dict[str, dict] = {}
    for e in events:
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        t = str(e.get("type", ""))
        if t == "experiment_batch" or "batch" in t or d.get("parallel_batch"):
            bid = _batch_id(d) or f"#{e.get('seq')}"
            b = out.setdefault(bid, {"batch": bid, "seq": e.get("seq"), "results": []})
            exps = d.get("experiments") or d.get("specs") or d.get("results") or d.get("kinds") or []
            b.update({"n_planned": d.get("n_experiments") or d.get("n") or (len(exps) if isinstance(exps, list) else None),
                      "kinds": d.get("kinds"), "speedup": _num(d.get("speedup")),
                      "workers": d.get("workers") or d.get("n_workers") or d.get("max_workers") or d.get("parallel_workers"),
                      "wall_s": _num(d.get("wall_s") or d.get("batch_wall_s") or d.get("runtime_s")),
                      "serial_s": _num(d.get("sum_runtime_s") or d.get("serial_s") or d.get("sum_experiment_s"))})
        elif t == "experiment_result":
            bid = _batch_id(d)
            if bid:
                out.setdefault(bid, {"batch": bid, "seq": e.get("seq"), "results": []})["results"].append(d)
    rows = []
    for b in out.values():
        res = b["results"]
        rts = [_num(r.get("runtime_s")) for r in res if _num(r.get("runtime_s")) is not None]
        serial = b.get("serial_s") or (sum(rts) if rts else None)
        wall = b.get("wall_s")
        speed = b.get("speedup") or (serial / wall if serial and wall else None)
        kinds = b.get("kinds") if isinstance(b.get("kinds"), list) else [r.get("kind", "?") for r in res]
        rows.append({"batch": b["batch"], "event": b.get("seq"),
                     "experiments": len(res) or b.get("n_planned"), "parallel workers": b.get("workers"),
                     "kinds": ", ".join(sorted({str(k) for k in kinds})),
                     "batch wall s": round(wall, 1) if wall else None,
                     "sum of experiment wall s": round(serial, 1) if serial else None,
                     "observed parallel speed-up": f"{speed:.1f}x" if speed else ""})
    return rows


def _batch_event(d: dict) -> None:
    import pandas as pd

    exps = d.get("experiments") or d.get("specs") or d.get("results") or []
    bits = []
    for k, lab in (("workers", "workers"), ("n_workers", "workers"), ("wall_s", "wall s"),
                   ("sum_runtime_s", "sum of experiment s"), ("speedup", "speed-up x")):
        if d.get(k) is not None:
            bits.append(f"{lab} **{d[k]}**")
    if bits:
        st.markdown(" · ".join(bits))
    rows = [x for x in exps if isinstance(x, dict)] if isinstance(exps, list) else []
    if rows:
        df = pd.DataFrame(rows)
        for c in df.columns:
            df[c] = df[c].map(lambda x: "" if x is None else (", ".join(map(str, x)) if isinstance(x, list)
                                                             else str(x)))
        st.dataframe(df, hide_index=True, width="stretch")
    elif isinstance(d.get("artifacts"), list) and d["artifacts"]:
        st.caption("Artifacts: " + ", ".join(f"`{a}`" for a in d["artifacts"] if a))
    elif d:
        with st.expander("data"):
            st.json(d, expanded=False)


def _batches_panel(events: list[dict]) -> None:
    import pandas as pd

    rows = batches(events)
    if not rows:
        return
    st.markdown("**Parallel experiment batches** (independent experiments run at the same time; speed-up = sum of "
                "the single experiments' wall times / wall time of the batch, as measured)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _verification_panel(events: list[dict]) -> None:
    import pandas as pd

    rows = []
    for e in events:
        if e.get("type") != "movement_verification":
            continue
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        vs = ui.verification_summary(ui.verification_of(d))
        rows.append({"event": e.get("seq"),
                     "experiment": str(d.get("artifact") or d.get("condition") or d.get("experiment") or ""),
                     "mode": vs.get("mode") or "", "expected": vs.get("expected") or "",
                     "kinematic": vs.get("kinematic") or "", "vision": vs.get("vision") or "not used",
                     "final": vs.get("final") or "",
                     "agree": "" if vs.get("agreement") is None else ("yes" if vs["agreement"] else "NO")})
    if rows:
        st.markdown("**Movement verification** (did the body do what the brain commanded? kinematic check vs. "
                    "vision check of keyframes)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


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
    m = st.columns(7)
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
    mv = [ui.verification_summary(ui.verification_of(e.get("data")))
          for e in events if e.get("type") == "movement_verification"]
    m[6].metric("Movement checks", f"{sum(v.get('final') == 'correct' for v in mv)} / {len(mv)}" if mv else "0",
                help="movement-verifier verdicts 'correct' / all movement checks in this run")

    st.markdown("**Agent hand-offs** (Omnigent supervisor and specialists; edge labels = hand-off order)")
    st.graphviz_chart(handoff_dot(summary), width="stretch")
    with st.container(border=True):
        _outcome_panel(events)
    _batches_panel(events)
    _verification_panel(events)

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
