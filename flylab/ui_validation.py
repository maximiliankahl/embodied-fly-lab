"""Dashboard page 3: Validation & speed.

Reads (each may be missing - other agents produce them):
  data/ground_truth.json, data/benchmarks/embodied_validation.json,
  data/benchmarks/screen_*.json, data/benchmarks/brain_validation.json, runs/*/record.jsonl
Benchmark schemas are read defensively (several key names accepted); unknown content is
shown raw instead of guessed.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from flylab import ui


# --------------------------------------------------------------------------- helpers


def _first(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _items(doc: Any) -> list[dict]:
    """List of entry dicts from a benchmark doc (list, {results|entries|rows|...: [...]}, or {id: {...}})."""
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    if isinstance(doc, dict):
        for k in ("results", "entries", "rows", "validation", "runs", "items", "experiments"):
            if isinstance(doc.get(k), list):
                return [x for x in doc[k] if isinstance(x, dict)]
            if isinstance(doc.get(k), dict) and all(isinstance(v, dict) for v in doc[k].values()):
                return [{"ground_truth_id": kk, **vv} for kk, vv in doc[k].items()]
        vals = {k: v for k, v in doc.items() if not str(k).startswith("_")}
        if vals and all(isinstance(v, dict) for v in vals.values()):
            return [{"ground_truth_id": k, **v} for k, v in vals.items()]
    return []


def _validation_by_gt(doc: Any) -> dict[str, dict]:
    """gt_id -> {observed, verdict, reason, wall_s, video, condition, mock, raw}.

    Native schema (flylab.bridge.validate): {"rows": [{"condition", "behavior", "checks": [{"gt_id", "observed",
    "verdict", "reason", "comparison"}], "runtimes_s": {...}, "video"}], "summary": {...}}. Flat lists of
    {ground_truth_id|gt_id|id, verdict, observed} are accepted too.
    """
    out: dict[str, dict] = {}
    for it in _items(doc):
        rts = it.get("runtimes_s") if isinstance(it.get("runtimes_s"), dict) else {}
        body = it.get("body") if isinstance(it.get("body"), dict) else {}
        rt = _first(it, "wall_s", "runtime_s", "total_runtime_s", default=rts.get("total"))
        if rt is None and (it.get("brain_runtime_s") is not None or it.get("body_runtime_s") is not None):
            rt = float(it.get("brain_runtime_s") or 0) + float(it.get("body_runtime_s") or 0)
        base = {"wall_s": rt, "video": _first(it, "video", "video_path", default=body.get("video")),
                "condition": it.get("condition"), "mock": bool(it.get("mock")), "raw": it,
                "drive": it.get("drive"), "behavior": it.get("behavior")}
        checks = it.get("checks") if isinstance(it.get("checks"), list) else None
        if checks:
            for c in checks:
                if isinstance(c, dict) and c.get("gt_id"):
                    out[str(c["gt_id"])] = {**base, "observed": c.get("observed"), "verdict": c.get("verdict"),
                                            "reason": c.get("reason"), "comparison": c.get("comparison"),
                                            "informative": c.get("informative", True)}
            continue
        gid = _first(it, "ground_truth_id", "gt_id", "gt", "id")
        if not gid:
            continue
        ev = it.get("evaluation") if isinstance(it.get("evaluation"), dict) else {}
        out[str(gid)] = {**base,
                         "observed": _first(it, "observed", "observed_behavior", "behavior", default=body.get("behavior")),
                         "verdict": _first(it, "verdict", default=ev.get("verdict")),
                         "reason": _first(it, "reason", "note", default=ev.get("reason"))}
    return out


# --------------------------------------------------------------------------- sections


def _ground_truth_section() -> None:
    import pandas as pd

    st.subheader("Published experiments vs. the embodied model")
    gts = ui.ground_truth()
    if not gts:
        ui.placeholder("data/ground_truth.json", "Build it with `uv run python -m flylab.atlas --build`.")
        return
    vdoc = ui.load_json(ui.BENCH / "embodied_validation.json")
    val = _validation_by_gt(vdoc)
    rows = []
    for e in gts:
        cit = e.get("citation") or {}
        v = val.get(str(e.get("id")), {})
        rows.append({
            "id": e.get("id"),
            "manipulation": e.get("manipulation"),
            "target": e.get("target_group"),
            "published effect": f"{e.get('effect', 'induce')} {e.get('expected_behavior')}",
            "readout": e.get("readout_group") or "body",
            "simulated condition": v.get("condition") or "",
            "observed": v.get("observed") or "",
            "verdict": (v.get("verdict") or "not run").replace("_", " ")
                       + (" (uninformative)" if v and v.get("informative") is False else ""),
            "compared at": v.get("comparison") or "",
            "paper": ui.doi_url(cit.get("doi")),
            "reference": f"{str(cit.get('authors') or '').split(',')[0]} {cit.get('year') or ''}".strip(),
            "gt confidence": e.get("confidence"),
            "evidence (verbatim)": e.get("evidence"),
        })
    df = pd.DataFrame(rows)
    informative = [v for v in val.values() if v.get("verdict") and v.get("informative") is not False]
    verdicts = [str(v.get("verdict")) for v in informative]
    comparable = [v for v in verdicts if v not in ("not_comparable", "inconclusive", "None")]
    n_cons = sum(v == "consistent" for v in verdicts)
    n_uninf = sum(1 for v in val.values() if v.get("informative") is False)
    m = st.columns(5)
    m[0].metric("Published", len(gts), help="published results in data/ground_truth.json")
    m[1].metric("Simulated", len(val), help="ground-truth entries tested end to end"
                + (f"; {n_uninf} uninformative (excluded from the counts on the right)" if n_uninf else ""))
    m[2].metric("Consistent", n_cons, help="informative checks only")
    m[3].metric("Partial / incons.",
                f"{sum(v == 'partially_consistent' for v in verdicts)} / {sum(v == 'inconsistent' for v in verdicts)}")
    m[4].metric("Agreement", help="consistent / comparable informative checks (not_comparable, inconclusive and "
                "uninformative checks excluded)", value=
                f"{n_cons}/{len(comparable)} ({100 * n_cons / len(comparable):.0f}%)" if comparable else "n/a")
    cav = vdoc.get("caveats") if isinstance(vdoc, dict) else None
    if isinstance(cav, list) and cav:
        st.info("**How to read these numbers**\n" + "\n".join(f"- {c}" for c in cav))
    if vdoc is None:
        ui.placeholder("data/benchmarks/embodied_validation.json",
                       "The table shows the published ground truth; simulated verdicts appear once the "
                       "validation sweep has written this file.")
    elif any(v.get("mock") for v in val.values()):
        ui.mock_banner("Part of the validation file")
    if isinstance(vdoc, dict) and isinstance(vdoc.get("protocol"), dict):
        pr = vdoc["protocol"]
        st.caption(f"Protocol: brain {pr.get('n_trials')} x {pr.get('brain_duration_ms')} ms at "
                   f"{pr.get('excite_rate_hz')} Hz, body {pr.get('body_duration_s')} s, seed {pr.get('seed')}"
                   + (f" · created {vdoc.get('created')}" if vdoc.get("created") else ""))

    colors = {"consistent": "#d8f0de", "partially consistent": "#fff0d6", "inconsistent": "#fbdcdc"}

    def _style(col):
        return [f"background-color: {colors.get(str(x), '')}" if colors.get(str(x)) else "" for x in col]

    st.dataframe(
        df.style.apply(_style, subset=["verdict"]),
        hide_index=True,
        width="stretch",
        column_config={
            "paper": st.column_config.LinkColumn("paper (DOI)", display_text=r"https://doi\.org/(.*)"),
            "evidence (verbatim)": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption("Verdict rules (flylab.atlas.evaluate): 'reduce' entries are consistent only if the behavior is "
               "absent; opposite turn direction is inconsistent; escape/groom/feed cannot be produced by the "
               "walking body and are compared at brain level (readout group firing) instead.")
    vids = {k: v for k, v in val.items() if ui.resolve(v.get("video"))}
    if vids:
        pick = st.selectbox("Watch a validation run", sorted(vids),
                            format_func=lambda k: f"{k}  ({vids[k].get('condition') or '?'})")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.video(str(ui.resolve(vids[pick]["video"])))
        with c2:
            v = vids[pick]
            st.markdown(f"Observed **{v.get('observed')}** · verdict "
                        f"{ui.badge(str(v.get('verdict')).replace('_', ' '), ui.VERDICT_COLOR.get(str(v.get('verdict')), 'gray'))}")
            if v.get("reason"):
                st.caption(v["reason"])
            drv = v.get("drive") if isinstance(v.get("drive"), dict) else {}
            if drv:
                st.markdown(f"Drive: forward {ui.fmt_num(drv.get('forward'), 2)}, turn {ui.fmt_num(drv.get('turn'), 2)}"
                            f" (neg = left), backward {ui.fmt_num(drv.get('backward'), 2)}")
            if v.get("wall_s") is not None:
                st.caption(f"Wall time brain + body: {ui.fmt_num(v['wall_s'], 1)} s")
    if vdoc is not None:
        with st.expander("Raw embodied_validation.json"):
            st.json(vdoc, expanded=False)


def _screen_summary_row(path, doc: dict) -> dict:
    s = doc.get("search") if isinstance(doc.get("search"), dict) else {}
    g = s.get("guided") or {}
    r = s.get("random") or {}
    return {"benchmark": path.stem, "target": doc.get("target_group"), "candidates": s.get("n_candidates"),
            "in-silico hits": s.get("n_hits"),
            "guided: exp. to 1st hit": g.get("experiments_to_first_hit"),
            "random: exp. to 1st hit": r.get("experiments_to_first_hit_expected"),
            "reduction 1st hit": s.get("reduction_factor_first_hit"),
            "reduction all hits": s.get("reduction_factor_all_hits")}


def _fx(v) -> str:
    return f"{v}x" if v is not None else "n/a"


def _score_variants(doc: dict) -> None:
    """Robustness of the speed-up to the choice of connectome score (sensitivity_score_variants)."""
    import pandas as pd

    sv = doc.get("sensitivity_score_variants")
    if not isinstance(sv, dict) or not sv:
        return
    primary = (doc.get("ranking") or {}).get("primary") if isinstance(doc.get("ranking"), dict) else None
    rows = []
    for name, v in sv.items():
        if not isinstance(v, dict):
            continue
        ins = v.get("vs_in_silico_hits") or {}
        lit = v.get("vs_literature_hits") or {}
        rows.append({"score variant": name + (" (primary)" if name == primary else ""),
                     "in silico: 1st hit": _fx(ins.get("reduction_first")),
                     "in silico: all hits": _fx(ins.get("reduction_all")),
                     "literature: 1st": _fx(lit.get("reduction_first")),
                     "literature: all": _fx(lit.get("reduction_all"))})
    if rows:
        st.markdown("**Robustness: speed-up vs. random order for every connectome score variant** "
                    "(the primary score was chosen after a pilot, see 'Read with care')")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _screen_section() -> None:
    import altair as alt
    import pandas as pd

    st.subheader("Discovery acceleration: connectome-guided screen")
    st.caption("Question: which upstream cell types drive a target group? Each experiment = one whole-brain "
               "simulation activating one candidate type. The connectome ranks candidates first (milliseconds), "
               "so the expensive simulations start with the most promising ones.")
    files = ui.benchmark_files("screen_")
    docs = {p: ui.load_json(p) for p in files}
    docs = {p: d for p, d in docs.items() if isinstance(d, dict)}
    if not docs:
        ui.placeholder("A screen benchmark (data/benchmarks/screen_*.json)",
                       "It is written by `uv run python -m flylab.screen --benchmark`: connectome ranking, an "
                       "exhaustive brain screen, and experiments-to-first-hit for connectome-guided vs. random "
                       "vs. exhaustive search.")
        return
    summ = pd.DataFrame([_screen_summary_row(p, d) for p, d in docs.items()])
    st.dataframe(summ, hide_index=True, width="stretch")
    paths = list(docs)
    pref = next((i for i, p in enumerate(paths) if str(docs[p].get("target_group")) == "MDN"), 0)
    pick = (st.selectbox("Benchmark", paths, index=pref, format_func=lambda p: p.stem)
            if len(paths) > 1 else paths[0])
    doc = docs[pick]
    if doc.get("mock"):
        ui.mock_banner(pick.name)
    for key in ("headline", "literature_headline"):
        if doc.get(key):
            st.markdown(f"- {doc[key]}")
    lims = [x for x in (doc.get("limitations") or []) if isinstance(x, str)]
    key_lims = [x for x in lims if x.lower().startswith(("in-silico ground truth", "score choice"))]
    if key_lims:
        st.info("**Read with care**\n" + "\n".join(f"- {x}" for x in key_lims))
    s = doc.get("search") if isinstance(doc.get("search"), dict) else {}
    g, rnd, ex = s.get("guided") or {}, s.get("random") or {}, s.get("exhaustive") or {}
    bl = (doc.get("baselines") or {}).get("largest_type_first") if isinstance(doc.get("baselines"), dict) else None
    bl = bl if isinstance(bl, dict) else {}
    bl_ins = bl.get("vs_in_silico_hits") if isinstance(bl.get("vs_in_silico_hits"), dict) else {}
    if s:
        if bl.get("guided_speedup_vs_baseline_first") is not None:
            st.markdown(f"Against a **stronger baseline without the connectome** (largest cell types first): "
                        f"**{_fx(bl.get('guided_speedup_vs_baseline_first'))}** fewer experiments to the first hit, "
                        f"**{_fx(bl.get('guided_speedup_vs_baseline_all'))}** to all hits. Random order is the "
                        "weaker reference used in the numbers below.")
        m = st.columns(4)
        m[0].metric("Fewer experiments to 1st hit", _fx(s.get("reduction_factor_first_hit")),
                    help="vs. random order (exact expectation)")
        m[1].metric("Fewer experiments to all hits", _fx(s.get("reduction_factor_all_hits")),
                    help="vs. random order (exact expectation)")
        m[2].metric("vs. exhaustive (all hits)", _fx(s.get("reduction_factor_vs_exhaustive_all_hits")))
        m[3].metric("Candidates / in-silico hits", f"{s.get('n_candidates', '?')} / {s.get('n_hits', '?')}")
        rows = []
        for milestone, gk, rk in (("first hit", "experiments_to_first_hit", "experiments_to_first_hit_expected"),
                                  ("all hits", "experiments_to_all_hits", "experiments_to_all_hits_expected")):
            if g.get(gk) is not None:
                rows.append({"milestone": milestone, "strategy": "connectome-guided", "experiments": float(g[gk])})
            if rnd.get(rk) is not None:
                rows.append({"milestone": milestone, "strategy": "random order (expected)",
                             "experiments": float(rnd[rk])})
            bk = "guided_first" if milestone == "first hit" else "guided_all"
            if isinstance(bl_ins.get(bk), (int, float)):
                rows.append({"milestone": milestone, "strategy": "largest type first (no connectome)",
                             "experiments": float(bl_ins[bk])})
            if ex.get("experiments") is not None:
                rows.append({"milestone": milestone, "strategy": "exhaustive", "experiments": float(ex["experiments"])})
        if rows:
            df = pd.DataFrame(rows)
            order = ["connectome-guided", "largest type first (no connectome)", "random order (expected)",
                     "exhaustive"]
            base = alt.Chart(df).encode(
                y=alt.Y("strategy:N", title=None, sort=order, axis=alt.Axis(labelLimit=260)),
                x=alt.X("experiments:Q", title="experiments (whole-brain simulations)"))
            bars = base.mark_bar().encode(
                color=alt.Color("strategy:N", legend=None,
                                scale=alt.Scale(domain=order,
                                                range=["#1F6F8B", "#7FA7B8", "#9aa8b5", "#c9d4de"])),
                tooltip=["milestone", "strategy", alt.Tooltip("experiments:Q", format=".1f")])
            text = base.mark_text(align="left", dx=4, color="#1B2430").encode(
                text=alt.Text("experiments:Q", format=".1f"))
            st.altair_chart((bars + text).properties(height=120).facet(row=alt.Row("milestone:N", title=None, sort=["first hit", "all hits"])))
        wt = {"connectome-guided (incl. ranking)": g.get("wall_s_to_first_hit"),
              "random order (expected)": rnd.get("wall_s_to_first_hit_expected"),
              "exhaustive screen": ex.get("wall_s")}
        if any(isinstance(v, (int, float)) for v in wt.values()):
            st.caption("Wall time to first hit: " + " · ".join(f"{k} {v:.0f} s" for k, v in wt.items()
                                                                 if isinstance(v, (int, float))))
    _score_variants(doc)
    table = doc.get("table") if isinstance(doc.get("table"), list) else []
    if table:
        df = pd.DataFrame(table)
        if "connectome_rank" in df.columns:
            df = df.sort_values("connectome_rank")
        top = df.head(25).copy()

        def _cat(r):
            if r.get("literature_known"):
                return "literature-known"
            return "in-silico hit" if r.get("hit") else "no hit"

        top["class"] = top.apply(_cat, axis=1)
        st.markdown("**Top 25 candidates by connectome score** (color = outcome of the simulation)")
        tips = [c for c in ("cell_type", "connectome_rank", "score", "direct_syn", "rate_hz", "hit",
                            "literature_known", "n") if c in top.columns]
        sort_field = "connectome_rank" if "connectome_rank" in top.columns else "score"
        chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("score:Q", title="connectome score"),
            y=alt.Y("cell_type:N", sort=alt.EncodingSortField(sort_field), title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=200)),
            color=alt.Color("class:N", scale=alt.Scale(domain=["literature-known", "in-silico hit", "no hit"],
                                                        range=["#E07B39", "#1F6F8B", "#c9d4de"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=tips).properties(height=24 * len(top) + 40)
        st.altair_chart(chart, width="stretch")
        rk = doc.get("ranking") or {}
        if isinstance(rk, dict) and isinstance(rk.get("spearman_score_vs_rate"), (int, float)):
            st.caption("Spearman correlation of connectome score vs. simulated target rate over all candidates: "
                       f"{rk['spearman_score_vs_rate']:.2f}")
    lit = doc.get("literature") if isinstance(doc.get("literature"), dict) else {}
    if lit.get("known_hits"):
        st.markdown("**Literature check** (independent of the connectome)")
        rows = []
        for r in lit["known_hits"]:
            ev = r.get("evidence")
            rows.append({**{k: v for k, v in r.items() if k != "evidence"},
                         "why": ev.get("why") if isinstance(ev, dict) else ev,
                         "ground truth": ", ".join(ev.get("gt", [])) if isinstance(ev, dict) else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if isinstance(lit.get("recall_in_silico"), (int, float)):
            st.caption(f"In-silico recall of literature-known hits: {lit['recall_in_silico']:.0%}. "
                       f"Precision: {lit.get('precision', '')}")
    nov = doc.get("novel_in_silico_hits")
    if isinstance(nov, dict) and nov.get("cell_types"):
        st.markdown(f"**New candidates** {ui.badge('agent-generated hypotheses, untested', 'orange')}")
        st.caption(nov.get("note", ""))
        rates = nov.get("rates_hz") or {}
        st.markdown(", ".join(f"{c} ({rates.get(c, '?')} Hz)" for c in nov["cell_types"][:15]))
    if doc.get("definitions") or doc.get("limitations") or doc.get("sensitivity_threshold"):
        with st.expander("Definitions, sensitivity and limitations"):
            for k, v in (doc.get("definitions") or {}).items():
                st.markdown(f"**{k}**: {v}")
            if isinstance(doc.get("sensitivity_threshold"), dict):
                st.markdown("**Sensitivity to the hit threshold**")
                st.dataframe(pd.DataFrame(doc["sensitivity_threshold"]).T, width="stretch")
            for lim in doc.get("limitations") or []:
                st.markdown(f"- {lim}")
    with st.expander(f"Raw {pick.name}"):
        st.json(doc, expanded=False)


def _walltime_rows(include_mock: bool) -> list[dict]:
    rows = []
    for rid in ui.list_runs():
        events = ui.load_events(rid)
        mock_run = ui.run_is_mock(rid, events)
        if mock_run and not include_mock:
            continue
        for e in events:
            if e.get("type") != "experiment_result" or not isinstance(e.get("data"), dict):
                continue
            d = e["data"]
            kind = d.get("kind", "?")
            brain_s = body_s = None
            if kind in ("brain", "screen"):
                brain_s = d.get("runtime_s")
            elif kind == "body":
                body_s = d.get("runtime_s")
            elif kind == "embodied":
                art = ui.load_json(d["artifact"]) if d.get("artifact") else None
                if isinstance(art, dict):
                    brain_s = art.get("brain_runtime_s")
                    body_s = (art.get("body") or {}).get("runtime_s") if isinstance(art.get("body"), dict) else None
            total = sum(float(x) for x in (brain_s, body_s) if isinstance(x, (int, float)))
            rows.append({"source": f"run {rid} #{e.get('seq')}", "kind": kind, "brain_s": brain_s,
                         "body_s": body_s,
                         "total_s": round(total, 2) if (brain_s is not None or body_s is not None) else None,
                         "mock": mock_run or ui.event_is_mock(e)})
    vdoc = ui.load_json(ui.BENCH / "embodied_validation.json")
    if isinstance(vdoc, dict):
        for r in vdoc.get("rows") or []:
            rts = r.get("runtimes_s") if isinstance(r, dict) else None
            if isinstance(rts, dict):
                rows.append({"source": f"validation sweep: {r.get('condition')}", "kind": "embodied",
                             "brain_s": rts.get("brain"), "body_s": rts.get("body"), "total_s": rts.get("total"),
                             "mock": bool(r.get("mock"))})
    # Several screen benchmarks (e.g. MDN and GF targets) replay the SAME cached whole-brain simulations
    # (one simulation per candidate, all target groups read out). Count each simulation once.
    seen: dict[tuple, dict] = {}
    for p in ui.benchmark_files("screen_"):
        doc = ui.load_json(p)
        if not isinstance(doc, dict):
            continue
        bp = doc.get("brain_params") if isinstance(doc.get("brain_params"), dict) else {}
        bkey = tuple(sorted((k, str(v)) for k, v in bp.items()))
        for r in doc.get("table") or []:
            if isinstance(r, dict) and isinstance(r.get("runtime_s"), (int, float)):
                key = (bkey, str(r.get("cell_type")), r["runtime_s"])
                if key in seen:
                    seen[key]["source"] += f", {doc.get('target_group') or p.stem}"
                    continue
                seen[key] = {"source": f"screen ({doc.get('target_group') or p.stem}): {r.get('cell_type')}",
                             "kind": "brain (screen)", "brain_s": r["runtime_s"], "body_s": None,
                             "total_s": r["runtime_s"], "mock": bool(doc.get("mock"))}
    rows.extend(seen.values())
    return rows


def _speed_section() -> None:
    import pandas as pd

    st.subheader("Wall time per experiment")
    include_mock = st.toggle("Include mock runs", value=False)
    rows = _walltime_rows(include_mock)
    bv = ui.load_json(ui.BENCH / "brain_validation.json") or {}
    c = st.columns(3)
    if bv.get("speed_s_per_trial_per_sim_s") is not None:
        c[0].metric("Brain model speed", f"{bv['speed_s_per_trial_per_sim_s']:.2f} s",
                    help="wall seconds per trial per simulated second, 138,639 neurons, 4 threads")
    df = pd.DataFrame(rows)
    if not df.empty and df["total_s"].notna().any():
        agg = df.dropna(subset=["total_s"]).groupby("kind")["total_s"].agg(["count", "median", "max"]).reset_index()
        c[1].metric("Experiments with timing", int(agg["count"].sum()))
        emb = agg[agg["kind"] == "embodied"]
        if not emb.empty:
            c[2].metric("Median embodied experiment", f"{float(emb['median'].iloc[0]):.1f} s")
        st.dataframe(agg.rename(columns={"count": "n", "median": "median wall s", "max": "max wall s"}),
                     hide_index=True)
        st.caption("Screen experiments are brain-only (shorter, 2 threads each, shared CPU); embodied experiments "
                   "add the physics body and video rendering.")
        with st.expander("All timed experiments"):
            st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("No timed experiments in the recorded runs yet"
                   + (" (mock runs excluded)." if not include_mock else "."))


def _brain_section() -> None:
    import pandas as pd

    st.subheader("Brain-model fidelity (vs. Shiu et al. 2024 and Brian2)")
    bv = ui.load_json(ui.BENCH / "brain_validation.json")
    if not isinstance(bv, dict):
        ui.placeholder("data/benchmarks/brain_validation.json", "Run `uv run python -m flylab.brain --validate`.")
        return
    meta = bv.get("_meta", {})
    paper = meta.get("paper", {})
    nn = meta.get("n_neurons")
    st.caption(f"{meta.get('connectome', '')}, {f'{nn:,}' if isinstance(nn, int) else '?'} neurons, {meta.get('n_trials')} trials x "
               f"{meta.get('duration_ms')} ms per condition. Paper: {ui.doi_md(paper.get('doi'))}. "
               f"Reference code: {paper.get('code', '')}")
    c1, c2 = st.columns([1, 1])
    with c1:
        dose = bv.get("sugar_dose_response") or []
        if dose:
            df = pd.DataFrame(dose).rename(columns={"MN9_a_hz": "MN9 contralateral (Hz)",
                                                    "MN9_b_hz": "MN9 ipsilateral (Hz)"})
            st.markdown("**Sugar GRN stimulation -> MN9 (proboscis motor neuron) firing**")
            st.line_chart(df, x="sugar_rate_hz", y=["MN9 contralateral (Hz)", "MN9 ipsilateral (Hz)"], height=260,
                          x_label="sugar GRN stimulation (Hz)", y_label="MN9 rate (Hz)")
    with c2:
        bx = bv.get("brian2_crosscheck") or {}
        if bx:
            st.markdown("**Cross-check against Brian2** (reference simulator)")
            st.caption(bx.get("what", ""))
            m = st.columns(2)
            m[0].metric("Pearson r, non-stimulated rates", ui.fmt_num(bx.get("pearson_r_nonstim_rates"), 4),
                        help=f"{bx.get('n_neurons', '?')}-neuron subnetwork, {bx.get('n_trials', '?')} trials")
            na = bx.get("n_active_nonstim") or {}
            m[1].metric("Active neurons (ours / Brian2)", f"{na.get('ours', '?')} / {na.get('brian2', '?')}")
            rows = []
            for nid, v in (bx.get("mn9") or {}).items():
                rows.append({"MN9 root id": nid, "ours (Hz)": f"{v.get('ours_mean')} +/- {v.get('ours_std')}",
                             "Brian2 (Hz)": f"{v.get('brian2_mean')} +/- {v.get('brian2_std')}"})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    checks = bv.get("checks") or []
    if checks:
        st.markdown("**Published claims checked**")
        rows = []
        for ch in checks:
            agree = ch.get("agree")
            rows.append({"claim": ch.get("claim"),
                         "our model": ch.get("ours") if isinstance(ch.get("ours"), str) else str(ch.get("ours")),
                         "agrees": "yes" if agree is True else ("no" if agree is False else str(agree))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"claim": st.column_config.TextColumn(width="large"),
                                    "our model": st.column_config.TextColumn(width="large")})


def page() -> None:
    st.title("Validation & speed")
    st.caption("How far can the in-silico lab be trusted, and how much faster does it get to an answer?")
    tabs = st.tabs(["Ground truth", "Acceleration", "Wall time", "Brain-model fidelity"])
    with tabs[0]:
        _ground_truth_section()
    with tabs[1]:
        _screen_section()
    with tabs[2]:
        _speed_section()
    with tabs[3]:
        _brain_section()
