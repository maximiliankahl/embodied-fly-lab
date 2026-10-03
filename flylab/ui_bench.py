"""Dashboard page 2: Experiment bench - one hand-driven brain -> bridge -> body experiment.

All simulation imports (flylab.brain / bridge / body / atlas, flygym) happen only when the
user presses "Run", so this page renders in replay mode too (controls disabled).
"""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from flylab import ui

ROLE_ORDER = {"descending": 0, "sensory": 1, "interneuron": 2, "motor": 3}
# Wall-time constants for the estimate shown before a run (measured on the dev laptop):
BRAIN_S_PER_TRIAL_S = 1.0     # data/benchmarks/brain_validation.json: 0.93 s per trial per simulated second (4 threads)
BRAIN_SETUP_S = 5.0           # connectome load / setup (first run in a process takes longer)
BODY_S_PER_SIM_S = 2.7        # flylab/body.py with control_every=5: ~2-2.7 s wall per simulated second


def _group_label(name: str, g: dict) -> str:
    return f"{name}  ({g.get('role', '?')}, n={len(g.get('root_ids', []))})"


def _gt_label(gt_by_id: dict, gid: str) -> str:
    if gid == "(none)":
        return "(no comparison)"
    e = gt_by_id.get(gid, {})
    cit = e.get("citation") or {}
    return (f"{gid}: {e.get('manipulation')} {e.get('target_group')} -> {e.get('effect', 'induce')} "
            f"{e.get('expected_behavior')} ({str(cit.get('authors') or '').split(',')[0]} {cit.get('year') or ''})")


def _run(excite: list[str], silence: list[str], rate: float, brain_ms: float, n_trials: int, body_s: float,
         gt_id: str | None) -> dict:
    """Run the chain; returns a result dict (also partial results if a stage fails)."""
    groups = ui.groups()
    ex_ids = sorted({int(i) for n in excite for i in groups[n]["root_ids"]})
    si_ids = sorted({int(i) for n in silence for i in groups[n]["root_ids"]})
    out: dict = {"inputs": {"excite": excite, "silence": silence, "rate_hz": rate, "duration_ms": brain_ms,
                            "n_trials": n_trials, "duration_s": body_s, "gt_id": gt_id},
                 "ts": datetime.now().strftime("%H:%M:%S")}
    t_all = time.time()
    with st.status("Running experiment...", expanded=True) as status:
        st.write(f"1/4 Brain: {len(ex_ids)} neurons stimulated at {rate:.0f} Hz, {len(si_ids)} silenced, "
                 f"{n_trials} x {brain_ms:.0f} ms (FlyWire v783 LIF model)")
        from flylab import brain

        t0 = time.time()
        res = brain.simulate(ex_ids, si_ids or None, excite_rate_hz=float(rate), duration_ms=float(brain_ms),
                             n_trials=int(n_trials), n_threads=4)
        rates = {str(k): float(v) for k, v in (res.get("rates") or {}).items()}
        out["brain"] = {"n_active": res.get("n_active"), "runtime_s": round(time.time() - t0, 1),
                        "dn_rates": ui.group_rates(rates, "descending"),
                        "other_rates": {k: v for k, v in ui.group_rates(rates, None).items()
                                        if v > 0 and groups.get(k, {}).get("role") != "descending"}}
        st.write("2/4 Bridge: descending-neuron rates -> locomotor drive")
        try:
            from flylab import bridge
        except ImportError as exc:
            out["error"] = f"flylab.bridge is not available yet ({exc}); showing brain results only."
            status.update(label="Brain done, bridge missing", state="error")
            return out
        drive = bridge.rates_to_drive(rates)
        out["drive"] = drive
        if hasattr(bridge, "brain_readouts"):
            try:
                out["readouts"] = bridge.brain_readouts(rates)
            except Exception as exc:  # readouts are optional
                out["readouts_error"] = str(exc)
        st.write(f"3/4 Body: NeuroMechFly walks for {body_s:.1f} s (MuJoCo physics, video)")
        from flylab import body

        ui.BENCH_OUT.mkdir(parents=True, exist_ok=True)
        video = ui.BENCH_OUT / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        t0 = time.time()
        b = body.simulate_walk({k: float(drive.get(k, 0.0) or 0.0) for k in ("forward", "turn", "backward")},
                               duration_s=float(body_s), render_path=str(video), control_every=5)
        out["body"] = {k: b.get(k) for k in ("behavior", "forward_disp_mm", "lateral_disp_mm", "heading_change_deg",
                                             "mean_speed_mm_s", "fell_over", "warnings")}
        out["body"]["runtime_s"] = round(time.time() - t0, 1)
        out["video"] = b.get("video") or (str(video) if video.exists() else None)
        if gt_id:
            st.write(f"4/4 Compare with published result {gt_id}")
            from flylab import atlas

            # Same rule as flylab.bridge.validate: escape / groom / feed cannot be shown by the walking body,
            # so those entries are compared with the brain-level readout (readout group firing), not the gait.
            exp = str(ui.ground_truth_by_id().get(gt_id, {}).get("expected_behavior", ""))
            observed, comparison = str(b.get("behavior")), "body behavior"
            ro = out.get("readouts")
            if exp in ("escape", "groom", "feed") and isinstance(ro, dict) and hasattr(bridge, "readout_label"):
                observed, comparison = bridge.readout_label(ro, exp), "brain readout"
            out["verdict"] = {**atlas.evaluate(observed, gt_id), "comparison": comparison}
        out["wall_s"] = round(time.time() - t_all, 1)
        status.update(label=f"Done in {out['wall_s']:.0f} s", state="complete", expanded=False)
    return out


def _readouts(ro: dict) -> None:
    """bridge.brain_readouts -> small table (behaviours the walking body cannot show)."""
    import pandas as pd

    rows = []
    for beh, v in ro.items():
        if isinstance(v, dict) and "active" in v:
            rows.append({"behavior": beh, "readout group": v.get("group"), "max rate (Hz)": v.get("rate_hz"),
                         "threshold (Hz)": v.get("threshold_hz"), "active": "yes" if v.get("active") else "no"})
    if rows:
        st.markdown("**Brain-level readouts** (escape, feeding, grooming cannot be shown by the walking body)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _show(r: dict) -> None:
    from flylab.ui_notebook import _body_block, _drive_block, _rates_chart

    inp = r.get("inputs", {})
    st.subheader("Result")
    st.caption(f"Run at {r.get('ts')} · stimulate {inp.get('excite')} · silence {inp.get('silence') or 'none'} · "
               f"{inp.get('rate_hz')} Hz · brain {inp.get('n_trials')} x {inp.get('duration_ms')} ms · "
               f"body {inp.get('duration_s')} s"
               + (f" · total wall time {r['wall_s']} s" if r.get("wall_s") else ""))
    if r.get("error"):
        st.warning(r["error"])
    c1, c2 = st.columns([1, 1])
    br = r.get("brain") or {}
    with c1:
        st.markdown(f"Brain: {br.get('n_active', '?')} active neurons · {br.get('runtime_s', '?')} s wall")
        _rates_chart(br.get("dn_rates"), "Descending-neuron group rates")
        _rates_chart(br.get("other_rates"), "Other groups that fired (brain readouts)")
        if r.get("readouts"):
            _readouts(r["readouts"])
    with c2:
        if r.get("drive"):
            _drive_block(r["drive"])
        if r.get("body"):
            _body_block(r["body"])
            st.caption(f"Body wall time {r['body'].get('runtime_s')} s")
        vp = ui.resolve(r.get("video"))
        if vp is not None:
            st.video(str(vp))
    v = r.get("verdict")
    if v:
        verdict = str(v.get("verdict"))
        gt = ui.ground_truth_by_id().get(str(v.get("id")), {})
        cit = gt.get("citation") or {}
        with st.container(border=True):
            st.markdown(f"Comparison with `{v.get('id')}`: "
                        f"{ui.badge(verdict.replace('_', ' '), ui.VERDICT_COLOR.get(verdict, 'gray'))} "
                        f"· expected **{v.get('expected_behavior')}** ({v.get('effect')}) vs observed "
                        f"**{v.get('observed')}**")
            st.caption(f"Compared at: {v.get('comparison', 'body behavior')}. {v.get('reason', '')}")
            if v.get("effect") == "reduce" and v.get("verdict") == "consistent":
                st.caption("A 'reduce' result is only informative if the same stimulus WITHOUT silencing shows the "
                           "behaviour: run that control too (validation sweep: gt02 was uninformative for this reason).")
            if gt:
                st.markdown(f'Published: "{gt.get("evidence", "")}" ({cit.get("authors", "")} {cit.get("year", "")}, '
                            f'{ui.doi_md(cit.get("doi"))})')


def page() -> None:
    st.title("Experiment bench")
    st.caption("Run one in-silico experiment by hand: stimulate or silence neuron groups in the whole-brain "
               "connectome model, translate descending-neuron activity into a locomotor drive, and let the "
               "physics body walk. The agents use the same chain through flylab.tools.")
    replay, reasons = ui.replay_status()
    if replay:
        st.info("**Replay mode**: live experiments are disabled here (" + "; ".join(reasons) + "). "
                "Recorded runs are in *Lab notebook* and *Validation & speed*. To run live: install the "
                "project environment (uv), keep data/raw/ and unset FLYLAB_REPLAY.")
    bridge_ok = ui.module_available("bridge")
    if not replay and not bridge_ok:
        st.warning("flylab/bridge.py is not available yet: a run will stop after the brain stage.")

    groups = ui.groups()
    if not groups:
        ui.placeholder("data/neurons.json", "Build it with `uv run python -m flylab.atlas --build`.")
        return
    names = sorted(groups, key=lambda n: (ROLE_ORDER.get(groups[n].get("role"), 9), n.lower()))
    gt_by_id = ui.ground_truth_by_id()

    c1, c2 = st.columns(2)
    excite = c1.multiselect("Stimulate (Poisson input)", names, default=["MDN"] if "MDN" in groups else [],
                            format_func=lambda n: _group_label(n, groups[n]), disabled=replay)
    silence = c2.multiselect("Silence (outgoing synapses set to 0)", names, default=[],
                             format_func=lambda n: _group_label(n, groups[n]), disabled=replay)
    c3, c4, c5, c6 = st.columns(4)
    rate = c3.slider("Stimulation rate (Hz)", 10, 300, 150, 10, disabled=replay)
    brain_ms = c4.select_slider("Brain time per trial (ms)", [250, 500, 1000, 2000], 500, disabled=replay)
    n_trials = c5.slider("Brain trials", 1, 5, 2, disabled=replay)
    body_s = c6.select_slider("Body time (s)", [1.0, 1.5, 2.0, 3.0], 1.5, disabled=replay)

    ids = list(gt_by_id)
    # Silencing entries match the silenced groups (the stimulus only provides activity); activation entries
    # match the stimulated groups.
    match = [g for g in ids if gt_by_id[g].get("manipulation") == "silence"
             and gt_by_id[g].get("target_group") in set(silence)]
    match += [g for g in ids if gt_by_id[g].get("manipulation") == "activate"
              and gt_by_id[g].get("target_group") in set(excite) and g not in match]
    ordered = ["(none)"] + match + [g for g in ids if g not in match]
    gt_choice = st.selectbox("Compare with a published result (ground truth)", ordered,
                             index=1 if match else 0, format_func=lambda g: _gt_label(gt_by_id, g),
                             disabled=replay)
    est = BRAIN_SETUP_S + BRAIN_S_PER_TRIAL_S * n_trials * brain_ms / 1000 + BODY_S_PER_SIM_S * body_s
    st.caption(f"Estimated wall time about {est:.0f} s (brain about {BRAIN_S_PER_TRIAL_S:.0f} s per trial-second, "
               f"body about {BODY_S_PER_SIM_S} s per simulated second; the first run also loads the connectome).")
    if silence and not excite:
        st.caption("Note: the brain model has 0 Hz baseline activity, so silencing alone needs a stimulus "
                   "(e.g. a sensory group) to show an effect.")
    run = st.button("Run experiment", type="primary", disabled=replay or not excite)
    if run:
        try:
            st.session_state["bench_result"] = _run(excite, silence, rate, brain_ms, n_trials, body_s,
                                                    None if gt_choice == "(none)" else gt_choice)
        except Exception as exc:  # show the error, keep the page alive
            st.session_state["bench_result"] = None
            st.error(f"Experiment failed: {type(exc).__name__}: {exc}")
    r = st.session_state.get("bench_result")
    if r:
        _show(r)
    elif not replay:
        st.caption("Choose groups and press Run. Example: stimulate MDN and compare with "
                   "gt01_mdn_activate_backward (Bidaye et al. 2014).")
