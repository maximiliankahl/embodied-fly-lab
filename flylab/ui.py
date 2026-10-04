"""Shared helpers for the Streamlit dashboard (app.py + flylab/ui_*.py).

Light on purpose: only stdlib + streamlit at import time. Simulation modules (brain, body,
bridge, flygym) are imported lazily inside the Experiment bench, so the dashboard also runs
in "replay" mode (recorded runs and benchmark files only).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATA = ROOT / "data"
BENCH = DATA / "benchmarks"
ASSETS = ROOT / "assets"
BENCH_OUT = ROOT / "spikes" / "dashboard" / "out" / "bench"  # gitignored scratch for bench videos
WEB = ROOT / "web"  # static Three.js replay viewer (GitHub Pages source)
FLIGHT_ASSETS = ASSETS / "flight"
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/maximiliankahl/embodied-fly-lab"
PAGES_URL = "https://maximiliankahl.github.io/embodied-fly-lab/"

VERDICT_COLOR = {
    "consistent": "green",
    "partially_consistent": "orange",
    "inconsistent": "red",
    "inconclusive": "gray",
    "not_comparable": "gray",
}
VERIFY_COLOR = {"correct": "green", "incorrect": "red", "uncertain": "gray"}
AGENT_COLOR = {
    "human": "gray",
    "supervisor": "violet",
    "literature": "blue",
    "hypothesis": "orange",
    "planner": "blue",
    "safety": "red",
    "runner": "green",
    "analysis": "violet",
    "record_keeper": "gray",
    "movement_verifier": "green",
    "verifier": "green",
}
TYPE_LABEL = {
    "question": "Question",
    "evidence": "Evidence",
    "hypothesis": "Hypothesis",
    "experiment_options": "Experiment options",
    "experiment_choice": "Experiment choice",
    "approval": "Human approval gate",
    "experiment_result": "Result",
    "movement_verification": "Movement verification",
    "experiment_batch": "Parallel experiment batch",
    "analysis": "Analysis",
    "decision": "Decision",
    "note": "Note",
}
LOOP_STAGES = [
    ("question", "Question"),
    ("evidence", "Evidence"),
    ("hypothesis", "Hypothesis"),
    ("experiment_options", "Experiment designs"),
    ("approval", "Human approval"),
    ("experiment_result", "Result"),
    ("movement_verification", "Movement check"),
    ("analysis", "Analysis"),
    ("decision", "Updated decision"),
]


# --------------------------------------------------------------------------- styling


def inject_css() -> None:
    st.markdown(
        """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.fl-loop {display:flex; flex-wrap:wrap; gap:6px; margin: 0.2rem 0 0.8rem 0;}
.fl-step {border:1px solid #c9d4de; border-radius:6px; padding:4px 10px; font-size:0.82rem;
          background:#f7f9fb; color:#3b4a5a;}
.fl-step.done {border-color:#1F6F8B; background:#e6f1f5; color:#0f4a5e; font-weight:600;}
.fl-step .n {opacity:0.65; font-weight:400; margin-left:4px;}
.fl-arrow {color:#9aa8b5; align-self:center; font-size:0.8rem;}
.fl-mock {border:1px solid #e0a43a; background:#fff6e5; color:#7a4b00; border-radius:6px;
          padding:6px 10px; font-size:0.85rem; margin-bottom:0.6rem;}
.fl-quote {border-left:3px solid #1F6F8B; padding:2px 0 2px 12px; color:#2b3a48; font-size:1.15rem;}
.fl-small {font-size:0.8rem; color:#5b6b7b;}
</style>
""",
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = "gray") -> str:
    """Markdown badge (Streamlit >= 1.45 directive)."""
    safe = str(text).replace("[", "(").replace("]", ")")
    return f":{color}-badge[{safe}]"


def mock_banner(what: str = "This run") -> None:
    st.markdown(
        f'<div class="fl-mock"><b>MOCK DATA.</b> {what} was produced with <code>FLYLAB_MOCK</code>: '
        "synthetic placeholders for testing the agent pipeline. Not a simulation or literature result.</div>",
        unsafe_allow_html=True,
    )


def placeholder(title: str, how: str) -> None:
    st.info(f"**{title}** is not available yet. {how}")


def doi_url(doi: str | None) -> str | None:
    if not doi:
        return None
    d = str(doi).strip()
    if d.lower().startswith("http"):
        return d
    return f"https://doi.org/{d}"


def doi_md(doi: str | None) -> str:
    url = doi_url(doi)
    return f"[{doi}]({url})" if url else ""


# --------------------------------------------------------------------------- data access


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, mtime: float) -> Any:  # mtime only invalidates the cache
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: str | Path) -> Any | None:
    """JSON file content, or None if missing/unreadable (cached per modification time)."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return _read_json_cached(str(p), p.stat().st_mtime)
    except (OSError, ValueError):
        return None


def resolve(rel: str | None) -> Path | None:
    """Project-relative path -> existing absolute Path (or None)."""
    if not rel:
        return None
    p = Path(str(rel))
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return p
    # A path recorded on another machine (absolute, or with backslashes): retry from runs/ or assets/.
    s = str(rel).replace("\\", "/")
    for anchor in ("runs/", "assets/", "spikes/"):
        i = s.find(anchor)
        if i >= 0 and (ROOT / s[i:]).exists():
            return ROOT / s[i:]
    return None


def groups() -> dict[str, dict]:
    doc = load_json(DATA / "neurons.json") or {}
    return doc.get("groups", {}) if isinstance(doc, dict) else {}


def ground_truth() -> list[dict]:
    doc = load_json(DATA / "ground_truth.json") or {}
    return list(doc.get("entries", [])) if isinstance(doc, dict) else []


def ground_truth_by_id() -> dict[str, dict]:
    return {str(e.get("id")): e for e in ground_truth()}


def list_runs() -> list[str]:
    """Run ids with a record, most recently modified first."""
    if not RUNS.exists():
        return []
    paths = [p for p in RUNS.glob("*/record.jsonl") if p.is_file()]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.parent.name for p in paths]


@st.cache_data(show_spinner=False)
def _load_events_cached(path: str, mtime: float) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_events(run_id: str) -> list[dict]:
    p = RUNS / run_id / "record.jsonl"
    try:
        return _load_events_cached(str(p), p.stat().st_mtime)
    except OSError:
        return []


def event_is_mock(e: dict) -> bool:
    d = e.get("data")
    if isinstance(d, dict) and d.get("mock"):
        return True
    return "[MOCK]" in str(e.get("content", "")) or str(e.get("content", "")).startswith("MOCK")


def run_is_mock(run_id: str, events: list[dict] | None = None) -> bool:
    rid = run_id.lower()
    if rid.startswith("demo_mock") or "mock" in rid or "selftest" in rid:
        return True
    events = events if events is not None else load_events(run_id)
    return any(event_is_mock(e) for e in events)


def summarize(events: list[dict]) -> dict:
    """Same shape as flylab.record.summarize, computed from loaded events."""
    by_type: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    handoffs: list[str] = []
    last = None
    for e in events:
        t, a = e.get("type", "?"), e.get("agent", "?")
        by_type[t] = by_type.get(t, 0) + 1
        by_agent[a] = by_agent.get(a, 0) + 1
        if a != last:
            handoffs.append(a)
            last = a
    return {"n_events": len(events), "by_type": by_type, "by_agent": by_agent, "handoffs": handoffs}


def benchmark_files(prefix: str = "") -> list[Path]:
    if not BENCH.exists():
        return []
    return sorted(BENCH.glob(f"{prefix}*.json"))


# --------------------------------------------------------------------------- run mode


def replay_status() -> tuple[bool, list[str]]:
    """(replay, reasons). Replay = recorded data only, no live simulation."""
    reasons: list[str] = []
    if os.environ.get("FLYLAB_REPLAY", "").strip().lower() in ("1", "true", "yes", "on"):
        reasons.append("FLYLAB_REPLAY=1 is set")
    if importlib.util.find_spec("flygym") is None:
        reasons.append("flygym (physics body) is not installed")
    raw = DATA / "raw"
    if not ((raw / "brain_csr_v783.npz").exists() or (raw / "Connectivity_783.parquet").exists()):
        reasons.append("connectome files (data/raw/) are not present")
    elif not (raw / "Completeness_783.csv").exists():
        reasons.append("data/raw/Completeness_783.csv is missing")
    return bool(reasons), reasons


def module_available(name: str) -> bool:
    """True if flylab.<name> exists as a file (no import)."""
    return (ROOT / "flylab" / f"{name}.py").exists()


# --------------------------------------------------------------------------- small formatters


def fmt_num(x: Any, nd: int = 1, unit: str = "") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:.{nd}f}{unit}"


def group_rates(rates: dict[str, float], role: str | None = "descending") -> dict[str, float]:
    """Mean rate per atlas group (silent neurons count as 0); mirrors flylab.tools._group_rates."""
    out: dict[str, float] = {}
    for name, g in groups().items():
        if role and g.get("role") != role:
            continue
        rids = [str(r) for r in g.get("root_ids", [])]
        if rids:
            out[name] = round(sum(float(rates.get(r, 0.0)) for r in rids) / len(rids), 3)
    return out


def _fx(v: Any) -> str:
    try:
        return f"{float(v):g}x"
    except (TypeError, ValueError):
        return "n/a"


def validation_counts(summ: dict | None) -> dict | None:
    """Honest headline from embodied_validation.json 'summary': informative checks only (uninformative
    rows, e.g. a silencing test whose control lacks the behaviour, are excluded), split by comparison type."""
    if not isinstance(summ, dict) or not summ.get("n_comparable"):
        return None
    n_inf = summ.get("n_informative")
    n_ok = summ.get("n_consistent_informative")
    if n_inf is None or n_ok is None:  # older schema: fall back to all comparable checks
        n_inf, n_ok = summ.get("n_comparable"), summ.get("n_consistent")
    parts = []
    for k, v in (summ.get("by_comparison") or {}).items():
        if isinstance(v, dict) and v.get("n") is not None:
            parts.append(f"{v.get('consistent')}/{v.get('n')} {k}")
    return {"n_ok": n_ok, "n": n_inf, "breakdown": ", ".join(parts),
            "n_uninformative": int(summ["n_inconclusive"]) if summ.get("n_inconclusive") is not None
                               else max(0, (summ.get("n_comparable") or 0) - (summ.get("n_informative") or 0))}


# --------------------------------------------------------------------------- flight + movement verification


def dig(d: Any, *paths: str, default: Any = None) -> Any:
    """First non-None value among dotted paths, e.g. dig(row, "verification.final_verdict", "verdict")."""
    for path in paths:
        cur = d
        for k in path.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return default


def as_list(x: Any) -> list:
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple)):
        return [y for y in x if y is not None and y != ""]
    return [x]


def stimulus_type(stim_groups: list, explicit: Any = None) -> str:
    """'direct DN' (an adapter input group is stimulated: partly circular, the adapter was designed from the
    same papers) vs 'upstream' (stimulus enters the brain upstream of the descending neurons, so the
    behaviour is emergent from the connectome model). An explicit label in the data wins."""
    if isinstance(explicit, str) and explicit:
        return explicit
    if explicit is True:
        return "direct DN (partly circular)"
    if not stim_groups:
        return "no stimulus (control)"
    gs = groups()
    roles = {str(gs.get(str(g), {}).get("role", "")) for g in stim_groups}
    if roles and roles <= {"descending"}:
        return "direct DN (partly circular)"
    if "descending" in roles:
        return "mixed (DN + upstream)"
    return "upstream (emergent via connectome)"


def verification_summary(v: Any) -> dict:
    """Normalise a flylab.verify.verify_movement() result (or a looser dict) for display."""
    if not isinstance(v, dict):
        return {}
    kin = v.get("kinematic") if isinstance(v.get("kinematic"), dict) else {}
    vis = v.get("vision") if isinstance(v.get("vision"), dict) else None
    final = v.get("final_verdict") or v.get("verdict") or kin.get("verdict")
    agree = v.get("agreement")
    if agree is None and vis and kin.get("verdict") and vis.get("verdict"):
        agree = kin.get("verdict") == vis.get("verdict")
    return {"final": final, "kinematic": kin.get("verdict"), "vision": (vis or {}).get("verdict"),
            "vision_used": vis is not None, "agreement": agree, "contact_sheet": v.get("contact_sheet"),
            "checks": [c for c in (kin.get("checks") or []) if isinstance(c, dict)],
            "observations": (vis or {}).get("observations"), "model": (vis or {}).get("model"),
            "frames_used": (vis or {}).get("frames_used"), "expected": v.get("expected_behavior") or v.get("expected"),
            "mode": v.get("mode"), "reason": v.get("reason") or kin.get("reason"),
            "vision_observed": (vis or {}).get("observed_behavior"), "vision_confidence": (vis or {}).get("confidence"),
            "expected_source": v.get("expected_source"),
            "error": v.get("error") or v.get("vision_error") or (vis or {}).get("error")}


def verification_of(d: Any) -> dict:
    """The verify_movement() result inside an event's data: data["verification"] when it is a dict (it can
    also be the path of the saved verification file), else the data itself."""
    if not isinstance(d, dict):
        return {}
    return d["verification"] if isinstance(d.get("verification"), dict) else d


def flight_rows(doc: Any) -> list[dict]:
    """Normalised rows of data/benchmarks/flight_validation.json (schema read defensively).

    Each: {condition, stim, silenced, rates, command, behavior, expected, verify (verification_summary),
           checks: [{gt_id, verdict, informative, comparison, observed}], stim_type, video, wall_s, mock, raw}."""
    items: list[dict] = []
    if isinstance(doc, list):
        items = [x for x in doc if isinstance(x, dict)]
    elif isinstance(doc, dict):
        for k in ("rows", "results", "conditions", "entries", "runs", "experiments"):
            if isinstance(doc.get(k), list):
                items = [x for x in doc[k] if isinstance(x, dict)]
                break
    out = []
    for it in items:
        stim = as_list(dig(it, "stimulus_groups", "excite_groups", "excite", "stimulus.excite", "stimulus"))
        if len(stim) == 1 and isinstance(stim[0], dict):
            stim = as_list(stim[0].get("excite") or stim[0].get("groups"))
        sil = as_list(dig(it, "silenced_groups", "silence_groups", "silence", "stimulus.silence"))
        rates = dig(it, "key_group_rates_hz", "group_rates_hz", "rates_hz", "brain.group_rates_hz",
                    "descending_group_rates_hz", "brain_rates_hz", default={})
        cmd = dig(it, "command", "flight_command", "bridge.command", "adapter_output", default={})
        ver = dig(it, "verification", "verify", "verifier", "movement_verification", default=None)
        checks = []
        for c in as_list(it.get("checks")):
            if isinstance(c, dict) and (c.get("gt_id") or c.get("ground_truth_id")):
                checks.append({"gt_id": str(c.get("gt_id") or c.get("ground_truth_id")), "verdict": c.get("verdict"),
                               "informative": c.get("informative", True), "comparison": c.get("comparison"),
                               "observed": c.get("observed"), "reason": c.get("reason")})
        gid = dig(it, "gt_id", "ground_truth_id", "ground_truth.id")
        if not checks and gid:
            checks.append({"gt_id": str(gid), "verdict": dig(it, "gt_verdict", "verdict", "ground_truth.verdict"),
                           "informative": it.get("informative", True), "comparison": it.get("comparison"),
                           "observed": None, "reason": it.get("reason")})
        rts = it.get("runtimes_s") if isinstance(it.get("runtimes_s"), dict) else {}
        out.append({
            "condition": dig(it, "condition", "name", "id", default="?"),
            "stim": [str(x) for x in stim if not isinstance(x, dict)], "silenced": [str(x) for x in sil],
            "rates": rates if isinstance(rates, dict) else {},
            "command": cmd if isinstance(cmd, dict) else {},
            "behavior": dig(it, "behavior", "flight.behavior", "body.behavior", "result.behavior", "observed"),
            "expected": dig(it, "expected_behavior", "expected", "verification.expected_behavior"),
            "verify": verification_summary(ver),
            "checks": checks,
            "stim_type": stimulus_type([str(x) for x in stim if not isinstance(x, dict)],
                                       dig(it, "stimulus_type", "test_type", "circular")),
            "video": dig(it, "video", "flight.video", "body.video", "video_path"),
            "wall_s": dig(it, "runtimes_s.total", "runtime_s", "wall_s", default=rts.get("total")),
            "runtimes": rts,
            "mock": bool(it.get("mock")),
            "raw": it,
        })
    return out


def flight_summary(rows: list[dict]) -> dict:
    """Counts for the flight checks, kept separate from the walking validation."""
    ver = [r["verify"] for r in rows if r["verify"].get("final")]
    both = [v for v in ver if v.get("kinematic") and v.get("vision")]
    checks = [(c, r["stim_type"]) for r in rows for c in r["checks"] if c.get("verdict")]
    inf = [(c, t) for c, t in checks if c.get("informative") is not False
           and str(c.get("verdict")) not in ("not_comparable", "inconclusive", "None")]
    by_type: dict[str, dict] = {}
    for c, t in inf:
        b = by_type.setdefault(t, {"n": 0, "consistent": 0})
        b["n"] += 1
        b["consistent"] += int(c.get("verdict") == "consistent")
    return {"n_conditions": len(rows), "n_verified": len(ver),
            "n_verified_correct": sum(v.get("final") == "correct" for v in ver),
            "n_both_verifiers": len(both), "n_agree": sum(bool(v.get("agreement")) for v in both),
            "n_gt": len(inf), "n_gt_consistent": sum(c.get("verdict") == "consistent" for c, _ in inf),
            "by_stim_type": by_type}


def key_results() -> list[tuple[str, str]]:
    """Headline numbers for the sidebar, read from the benchmark files (missing files are skipped).
    Labels state what each number measures; the less flattering companion numbers are shown too."""
    out: list[tuple[str, str]] = []
    v = load_json(BENCH / "embodied_validation.json")
    vc = validation_counts(v.get("summary") if isinstance(v, dict) else None)
    if vc:
        out.append((f"informative published checks consistent ({vc['breakdown']}); direct descending-neuron "
                    "rows are partly circular (the bridge was designed from the same papers)",
                    f"{vc['n_ok']} / {vc['n']}"))
    for p in sorted(benchmark_files("screen_"), key=lambda q: (q.stem.lower() != "screen_mdn", q.stem)):
        d = load_json(p)
        s = d.get("search") if isinstance(d, dict) else None
        tg = d.get("target_group", p.stem) if isinstance(d, dict) else p.stem
        if isinstance(s, dict) and s.get("reduction_factor_first_hit") is not None:
            bl = ((d.get("baselines") or {}).get("largest_type_first") or {}) if isinstance(d, dict) else {}
            bl_txt = (f"; vs. the stronger 'largest type first' baseline only "
                      f"{_fx(bl.get('guided_speedup_vs_baseline_first'))} / {_fx(bl.get('guided_speedup_vs_baseline_all'))}"
                      if isinstance(bl, dict) and bl.get("guided_speedup_vs_baseline_first") is not None else "")
            out.append((f"{tg}: fewer simulations to the first / to all {s.get('n_hits', '?')} in-silico hits "
                        "(connectome-guided vs. random order; hits defined by the same connectome model"
                        f"{bl_txt})",
                        f"{_fx(s['reduction_factor_first_hit'])} / {_fx(s.get('reduction_factor_all_hits'))}"))
        lit = (d.get("literature") or {}).get("search_for_literature_hits") if isinstance(d, dict) else None
        if isinstance(lit, dict) and lit.get("reduction_factor_first_hit") is not None:
            out.append((f"{tg}: fewer experiments until the first literature-known candidate is tested "
                        f"(only {lit.get('n_hits', '?')} known candidates)", _fx(lit["reduction_factor_first_hit"])))
    fs = flight_summary(flight_rows(load_json(BENCH / "flight_validation.json")))
    if fs["n_gt"]:
        out.append(("flight checks consistent with published experiments (separate from the walking checks; "
                    + ", ".join(f"{v['consistent']}/{v['n']} {k}" for k, v in fs["by_stim_type"].items()) + ")",
                    f"{fs['n_gt_consistent']} / {fs['n_gt']}"))
    if fs["n_verified"]:
        out.append(("flight movements judged correct by the movement verifier (kinematics recomputed from the raw "
                    "trajectory" + (f"; vision agrees in {fs['n_agree']}/{fs['n_both_verifiers']}"
                                    if fs["n_both_verifiers"] else "") + ")",
                    f"{fs['n_verified_correct']} / {fs['n_verified']}"))
    b = load_json(BENCH / "brain_validation.json")
    bx = b.get("brain_crosscheck") if isinstance(b, dict) else None
    bx = bx or (b.get("brian2_crosscheck") if isinstance(b, dict) else None)
    if isinstance(bx, dict) and bx.get("pearson_r_nonstim_rates") is not None:
        out.append((f"brain model vs. Brian2 reference on a {bx.get('n_neurons', '?')}-neuron subnetwork "
                    "(Pearson r of firing rates)", f"{bx['pearson_r_nonstim_rates']:.3f}"))
    return out
