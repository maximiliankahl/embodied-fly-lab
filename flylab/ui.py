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

VERDICT_COLOR = {
    "consistent": "green",
    "partially_consistent": "orange",
    "inconsistent": "red",
    "inconclusive": "gray",
    "not_comparable": "gray",
}
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
}
TYPE_LABEL = {
    "question": "Question",
    "evidence": "Evidence",
    "hypothesis": "Hypothesis",
    "experiment_options": "Experiment options",
    "experiment_choice": "Experiment choice",
    "approval": "Human approval gate",
    "experiment_result": "Result",
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
    b = load_json(BENCH / "brain_validation.json")
    bx = b.get("brain_crosscheck") if isinstance(b, dict) else None
    bx = bx or (b.get("brian2_crosscheck") if isinstance(b, dict) else None)
    if isinstance(bx, dict) and bx.get("pearson_r_nonstim_rates") is not None:
        out.append((f"brain model vs. Brian2 reference on a {bx.get('n_neurons', '?')}-neuron subnetwork "
                    "(Pearson r of firing rates)", f"{bx['pearson_r_nonstim_rates']:.3f}"))
    return out
