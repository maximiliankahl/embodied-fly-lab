"""Omnigent function tools for the Embodied Fly Lab.

Thin, typed wrappers around the science modules (atlas, literature, brain,
body, bridge). Every tool:
  * returns JSON-serialisable data (dict),
  * logs what it did to the shared research record (flylab.record), using the
    explicit ``run_id`` argument, else env ``FLYLAB_RUN_ID``, else ``runs/CURRENT``,
  * accepts an optional ``agent`` argument (the calling specialist's role name)
    so the dashboard can show who handed what to whom.

Mock mode (for testing the agent pipeline before the science modules exist):
  FLYLAB_MOCK=1            -> every component is mocked
  FLYLAB_MOCK=brain,body   -> only the listed components (atlas, literature,
                              brain, body, bridge, screen) are mocked
Mock outputs always carry ``"mock": true`` and a ``"MOCK"`` notice. They are
NOT scientific results. Without mock mode a missing module returns an
``error`` (never a silent fallback to fake data).

This module also contains ``approval_gate`` - an Omnigent *policy* factory
(used in agents/fly_lab.yaml) that enforces human approval for expensive
simulations at the harness level, not only via prompt.

NOTE: no ``from __future__ import annotations`` here on purpose: Omnigent
derives JSON schemas from real annotation objects.
"""

import json
import re
import os
import time
import traceback
from pathlib import Path
from typing import Any

from flylab import record

MOCK_NOTICE = "MOCK DATA (FLYLAB_MOCK) - plausible placeholder, NOT a simulation or literature result."
_COMPONENTS = ("atlas", "literature", "brain", "body", "bridge", "screen")
BEHAVIORS = ("forward", "backward", "turn_left", "turn_right", "stop", "escape", "groom", "feed")
BODY_BEHAVIORS = ("forward", "backward", "turn_left", "turn_right", "stop")  # what flylab.body can classify
TURNS = {"turn_left", "turn_right"}


# =========================================================================== helpers


MOCK_FLAG_FILE = record.ROOT / "data" / "cache" / "FLYLAB_MOCK"  # gitignored; written by agents/omni.ps1 -Mock


def _mock_setting() -> str:
    """FLYLAB_MOCK env var, else the flag file. The file exists because Omnigent's host only
    passes an allowlist of env vars to the runner process that executes these tools."""
    raw = os.environ.get("FLYLAB_MOCK", "").strip()
    if raw:
        return raw
    try:
        return MOCK_FLAG_FILE.read_text(encoding="utf-8").strip() if MOCK_FLAG_FILE.exists() else ""
    except OSError:
        return ""


def _mock(component: str) -> bool:
    raw = _mock_setting().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on", "all"):
        return True
    return component in {p.strip() for p in raw.split(",")}


PREAPPROVE_FLAG_FILE = record.ROOT / "data" / "cache" / "FLYLAB_PREAPPROVE"  # gitignored; agents/omni.ps1 -ApproveAtLaunch


def _preapproval() -> dict | None:
    """Human pre-approval for a SCRIPTED (headless) run, or None.

    Headless ``omnigent run -p`` declines every ASK (no human can answer), so for a scripted demo
    Max pre-approves the lab's gated actions AT LAUNCH with ``agents/omni.ps1 -ApproveAtLaunch``, which
    writes this flag file (deleted again when the command ends). approval_gate then ALLOWs what it
    would otherwise ASK (hard caps still DENY), and every approval event in the research record says
    "pre-approved at launch" instead of "approved in the UI". Env FLYLAB_PREAPPROVE=1 works too
    (only in-process; the Omnigent runner does not inherit it)."""
    raw = os.environ.get("FLYLAB_PREAPPROVE", "").strip()
    try:
        if not raw and PREAPPROVE_FLAG_FILE.exists():
            raw = PREAPPROVE_FLAG_FILE.read_text(encoding="utf-8").strip() or "1"
    except OSError:
        return None
    if raw.lower() in ("", "0", "false", "no", "off"):
        return None
    try:
        info = json.loads(raw)
        return info if isinstance(info, dict) else {"by": "human at launch"}
    except json.JSONDecodeError:
        return {"by": "human at launch (FLYLAB_PREAPPROVE)"}


def _rid(run_id: str = "") -> str:
    return record.resolve_run_id(run_id or None)


def _as_list(x: Any) -> list:
    """Accept list, JSON-list string, comma-separated string or None."""
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if s.startswith("["):
        try:
            return [str(v).strip() for v in json.loads(s) if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [p.strip() for p in s.split(",") if p.strip()]


def _err(tool: str, exc: BaseException, run_id: str | None = None, agent: str = "") -> dict:
    msg = f"{type(exc).__name__}: {exc}"
    hint = ""
    if isinstance(exc, ImportError):
        hint = "Module not available yet. Set FLYLAB_MOCK=1 (or FLYLAB_MOCK=<component>) to test the pipeline."
    out = {"ok": False, "tool": tool, "error": msg, "hint": hint,
           "trace_tail": traceback.format_exc(limit=3)[-800:]}
    if run_id:
        try:
            record.log_event(run_id, agent or "system", "note", f"{tool} failed: {msg}", {"error": msg, "hint": hint})
        except Exception:
            pass
    return out


def _next_artifact(run_id: str, prefix: str, ext: str) -> Path:
    """Next free ``<prefix>_NN.<ext>``. NN is free only if NO file with that stem exists (any
    extension), so the .mp4 and .json of one run share the same number even when a run
    produced no video (mock mode, render failure)."""
    d = record.artifacts_dir(run_id)
    i = 1
    while any(d.glob(f"{prefix}_{i:02d}.*")):
        i += 1
    return d / f"{prefix}_{i:02d}.{ext}"


def _save_json(path: Path, obj: Any) -> str:
    path.write_text(json.dumps(record._jsonable(obj), indent=1), encoding="utf-8")
    return str(path.relative_to(record.ROOT)).replace("\\", "/")


# =========================================================================== mock data
# Everything below is clearly synthetic. Group names mirror commonly discussed
# Drosophila descending-neuron classes so the pipeline can be exercised, but
# root ids are fake (negative) and no citation is attached.
_MOCK_GROUPS = {
    "MDN": {"root_ids": [-101, -102, -103, -104], "cell_type": "MDN", "side": "both", "role": "descending",
            "description": "MOCK: moonwalker descending neurons (backward walking in the literature)."},
    "P9": {"root_ids": [-201, -202], "cell_type": "P9", "side": "both", "role": "descending",
           "description": "MOCK: P9 descending neurons (forward walking / steering in the literature)."},
    "P9_L": {"root_ids": [-201], "cell_type": "P9", "side": "L", "role": "descending",
             "description": "MOCK: left P9."},
    "P9_R": {"root_ids": [-202], "cell_type": "P9", "side": "R", "role": "descending",
             "description": "MOCK: right P9."},
    "sugar_GRN": {"root_ids": [-301, -302, -303], "cell_type": "sugar GRN", "side": "both", "role": "sensory",
                  "description": "MOCK: sugar-sensing gustatory receptor neurons."},
    "MN9": {"root_ids": [-401, -402], "cell_type": "MN9", "side": "both", "role": "motor",
            "description": "MOCK: proboscis motor neuron 9."},
}
for _g in _MOCK_GROUPS.values():
    _g["citations"] = []
    _g["mock"] = True
_MOCK_GT = [
    {"id": "gt_mock_mdn_activate", "manipulation": "activate", "target_group": "MDN", "expected_behavior": "backward",
     "evidence": "MOCK entry. Real entries come from data/ground_truth.json.",
     "citation": {"doi": None, "title": "MOCK - replace with data/ground_truth.json", "year": None, "authors": None},
     "mock": True},
    {"id": "gt_mock_p9_activate", "manipulation": "activate", "target_group": "P9", "expected_behavior": "forward",
     "evidence": "MOCK entry. Real entries come from data/ground_truth.json.",
     "citation": {"doi": None, "title": "MOCK - replace with data/ground_truth.json", "year": None, "authors": None},
     "mock": True},
    {"id": "gt_mock_p9l_activate", "manipulation": "activate", "target_group": "P9_L", "expected_behavior": "turn_left",
     "evidence": "MOCK entry; direction mirrors gt04 in data/ground_truth.json (ipsilateral turning, Bidaye et al. 2020).",
     "citation": {"doi": None, "title": "MOCK - replace with data/ground_truth.json", "year": None, "authors": None},
     "mock": True},
]


def _mock_brain(excite: list[str], silence: list[str], rate_hz: float, duration_ms: float) -> dict:
    rates: dict[str, float] = {}
    gscale = min(rate_hz / 150.0, 2.0)
    for name, g in _MOCK_GROUPS.items():
        base = 0.5
        if name in excite or any(name.startswith(e + "_") for e in excite):
            base = 40.0 * gscale
        if name in silence or any(name.startswith(s + "_") for s in silence):
            base = 0.0
        for i, rid in enumerate(g["root_ids"]):
            if base > 0:
                rates[str(rid)] = round(base * (0.85 + 0.1 * i), 2)
    if "sugar_GRN" in excite and "MN9" not in silence:
        for rid in _MOCK_GROUPS["MN9"]["root_ids"]:
            rates[str(rid)] = round(25.0 * gscale, 2)
    return {"rates": rates, "std": {k: round(v * 0.1, 2) for k, v in rates.items()}, "n_active": len(rates),
            "runtime_s": 0.01, "params": {"excite_rate_hz": rate_hz, "duration_ms": duration_ms, "mock": True}}


def _mock_drive(group_rates: dict[str, float]) -> dict:
    fwd = min(group_rates.get("P9", 0.0) / 40.0, 1.0)
    back = min(group_rates.get("MDN", 0.0) / 40.0, 1.0)
    turn = max(-1.0, min(1.0, (group_rates.get("P9_L", 0.0) - group_rates.get("P9_R", 0.0)) / 40.0))
    return {"forward": round(fwd, 3), "turn": round(turn, 3), "backward": round(back, 3), "mock": True}


def _mock_body(drive: dict, duration_s: float) -> dict:
    f, t, b = float(drive.get("forward", 0)), float(drive.get("turn", 0)), float(drive.get("backward", 0))
    speed = 12.0 * (f - b)  # mm/s, synthetic
    heading = 90.0 * t * duration_s
    if abs(speed) < 1 and abs(heading) < 10:
        beh = "stop"
    elif abs(heading) >= 30:
        beh = "turn_right" if heading > 0 else "turn_left"
    else:
        beh = "forward" if speed > 0 else "backward"
    traj = [[round(duration_s * k / 4, 3), round(speed * duration_s * k / 4, 2), 0.0, round(heading * k / 4, 1)]
            for k in range(5)]
    return {"trajectory": traj, "forward_disp_mm": round(speed * duration_s, 2), "lateral_disp_mm": 0.0,
            "heading_change_deg": round(heading, 1), "mean_speed_mm_s": round(abs(speed), 2), "behavior": beh,
            "video": None, "runtime_s": 0.01, "mock": True}


# =========================================================================== science-module access


def _groups() -> dict[str, dict]:
    if _mock("atlas"):
        return _MOCK_GROUPS
    from flylab import atlas
    return atlas.groups()


def _ground_truth() -> list[dict]:
    if _mock("atlas"):
        return _MOCK_GT
    from flylab import atlas
    return atlas.ground_truth()


def _resolve_groups(names: list[str]) -> tuple[list[int], dict[str, int], list[str]]:
    """Group names (or raw root ids) -> root ids. Returns (ids, per-group count, unknown names)."""
    groups = _groups()
    ids: list[int] = []
    counts: dict[str, int] = {}
    unknown: list[str] = []
    for n in names:
        if n.lstrip("-").isdigit():
            ids.append(int(n))
            counts[n] = 1
            continue
        if n in groups:
            gids = [int(x) for x in groups[n].get("root_ids", [])]
        else:
            match = [k for k in groups if k.lower() == n.lower()]
            if not match and not _mock("atlas"):
                try:  # atlas aliases, e.g. "moonwalker" -> MDN, "giant fiber" -> GF
                    from flylab import atlas
                    match = [atlas.resolve_group(n)]
                except Exception:
                    match = []
            gids = [int(x) for x in groups[match[0]].get("root_ids", [])] if match and match[0] in groups else []
            if not gids and not _mock("atlas") and not _mock("screen"):
                try:  # any FlyWire v783 cell type, e.g. a screen hit such as "LC16" or "LPLC2"
                    from flylab import screen
                    gids = [int(x) for x in screen._resolve_ids(n)]
                except Exception:
                    gids = []
        if not gids:
            unknown.append(n)
        ids.extend(gids)
        counts[n] = len(gids)
    return sorted(set(ids)), counts, unknown


def _group_rates(rates: dict[str, float], role: str | None = "descending") -> dict[str, float]:
    """Mean rate (Hz) per atlas group (over all its neurons, silent ones count as 0)."""
    out: dict[str, float] = {}
    for name, g in _groups().items():
        if role and g.get("role") != role:
            continue
        rids = [str(r) for r in g.get("root_ids", [])]
        if not rids:
            continue
        out[name] = round(sum(float(rates.get(r, 0.0)) for r in rids) / len(rids), 3)
    return out


def _simulate_brain(excite_names: list[str], silence_names: list[str], rate_hz: float,
                    duration_ms: float, n_trials: int, seed: int = 0) -> tuple[dict, dict]:
    ex_ids, ex_counts, ex_unknown = _resolve_groups(excite_names)
    si_ids, si_counts, si_unknown = _resolve_groups(silence_names)
    meta = {"excite_groups": ex_counts, "silence_groups": si_counts,
            "unknown_groups": ex_unknown + si_unknown, "n_excited": len(ex_ids), "n_silenced": len(si_ids)}
    if not ex_ids:
        raise ValueError(f"no neurons resolved for excite_groups={excite_names} (unknown: {ex_unknown}). "
                         "Use list_neuron_groups / lookup_neurons first.")
    if _mock("brain"):
        res = _mock_brain(excite_names, silence_names, rate_hz, duration_ms)
    else:
        from flylab import brain
        res = brain.simulate(ex_ids, si_ids or None, excite_rate_hz=float(rate_hz),
                             duration_ms=float(duration_ms), n_trials=int(n_trials), seed=int(seed))
    return res, meta


def _drive_from_rates(rates: dict[str, float]) -> tuple[dict, dict]:
    """Per-neuron rates -> body drive via flylab.bridge.rates_to_drive. Returns (drive, info).

    docs/CONTRACTS.md types the argument as dict[str, float] but is ambiguous about the keys
    (per-neuron root ids as in brain.simulate()['rates'] vs. group names). We pass per-neuron
    rates first; if the bridge fails on them, or returns an all-zero drive although descending
    groups fire, we retry with {group_name: mean_rate_hz}. ``info`` says which input was used,
    so a silent all-zero drive (every run classified as 'stop') cannot go unnoticed.
    """
    if _mock("bridge"):
        return _mock_drive(_group_rates(rates, role=None)), {"bridge_input": "group_rates (mock bridge)"}
    from flylab import bridge
    group = _group_rates(rates, role=None)
    dn_active = any(v > 0 for v in _group_rates(rates, "descending").values())

    def _zero(d: dict) -> bool:
        return all(abs(float(d.get(k, 0.0) or 0.0)) < 1e-9 for k in ("forward", "turn", "backward"))

    try:
        drive = bridge.rates_to_drive(rates)
        if not (dn_active and _zero(drive)):
            return drive, {"bridge_input": "neuron_rates"}
        first_err = "all-zero drive from per-neuron rates although descending groups fire"
    except (KeyError, TypeError, ValueError) as exc:
        first_err = f"{type(exc).__name__}: {exc}"
    drive = bridge.rates_to_drive(group)
    info = {"bridge_input": "group_rates", "bridge_retry_reason": first_err}
    if dn_active and _zero(drive):
        info["warning"] = "bridge returned an all-zero drive although descending groups fire - check flylab.bridge"
    return drive, info


def _simulate_body(drive: dict, duration_s: float, render_path: str | None, seed: int = 0) -> dict:
    clean = {"forward": float(drive.get("forward", 0.0)), "turn": float(drive.get("turn", 0.0)),
             "backward": float(drive.get("backward", 0.0))}
    if _mock("body"):
        return _mock_body(clean, duration_s)
    from flylab import body
    return body.simulate_walk(clean, duration_s=float(duration_s), render_path=render_path, seed=int(seed))


def _rel(p: str | None) -> str | None:
    if not p:
        return None
    try:
        return str(Path(p).resolve().relative_to(record.ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


SHORT_BODY_RUN_S = 1.0  # below this, gait-cycle heading drift can cross body.classify's turn threshold


def _body_summary(res: dict, duration_s: float | None = None) -> dict:
    keys = ("behavior", "forward_disp_mm", "lateral_disp_mm", "heading_change_deg", "mean_speed_mm_s", "runtime_s")
    out = {k: res.get(k) for k in keys}
    if duration_s is not None and float(duration_s) < SHORT_BODY_RUN_S:
        # Observed in review: backward drive 1.0 for 0.5 s -> heading drift -20 deg -> labelled turn_right.
        out["label_warning"] = (f"duration_s={duration_s} < {SHORT_BODY_RUN_S}: heading drift within a few gait "
                                "cycles can be classified as a turn; use duration_s >= 1 for validation runs.")
    out["video"] = _rel(res.get("video"))
    if "fell_over" in res:  # a fallen fly makes the behavior label meaningless
        out["fell_over"] = bool(res.get("fell_over"))
    traj = res.get("trajectory") or []
    out["trajectory_points"] = len(traj)
    if traj:
        out["trajectory_end"] = traj[-1]
    return out


# =========================================================================== run management tools


def start_run(question: str, agent: str = "supervisor") -> dict:
    """Open a new research run for a scientific question and make it the active run.
    Call this ONCE at the start of a session. All later tool calls log into this
    run (runs/<run_id>/record.jsonl) unless they pass an explicit run_id.
    :param question: The research question, e.g. "Which descending neurons drive backward walking?"
    :param agent: Calling agent role name.
    :returns: {"run_id", "record", "artifacts"}
    """
    rid = record.new_run(question, set_current=True)
    if agent and agent != "human":
        record.log_event(rid, agent, "note", "Run opened by orchestrator.", {"question": question})
    return {"ok": True, "run_id": rid, "record": f"runs/{rid}/record.jsonl", "artifacts": f"runs/{rid}/artifacts/",
            "mock_mode": _mock_setting() or None}


def get_record(run_id: str = "", last_n: int = 30) -> dict:
    """Read the shared research record (most recent events) so agents can see each other's outputs.
    :param run_id: Run to read; default = active run.
    :param last_n: Number of most recent events to return.
    :returns: {"run_id", "summary", "events": [...]} with long data fields truncated.
    """
    rid = _rid(run_id)
    events = record.load(rid)[-max(1, int(last_n)):]
    slim = []
    for e in events:
        d = e.get("data")
        ds = json.dumps(d) if d is not None else ""
        slim.append({"seq": e.get("seq"), "agent": e.get("agent"), "type": e.get("type"),
                     "content": e.get("content", "")[:600], "citations": e.get("citations", []),
                     "data": d if len(ds) <= 1500 else ds[:1500] + "...(truncated)"})
    return {"ok": True, "run_id": rid, "summary": record.summarize(rid), "events": slim}


def log_note(content: str, agent: str = "record_keeper", event_type: str = "note", run_id: str = "") -> dict:
    """Log a free-form entry to the research record (e.g. evidence summary, analysis, gap).
    :param content: The text to record. Cite sources (DOI) inline where relevant.
    :param agent: Calling agent role name.
    :param event_type: One of question, evidence, hypothesis, experiment_options, experiment_choice,
        approval, experiment_result, analysis, decision, note.
    :param run_id: Run id; default = active run.
    """
    rid = _rid(run_id)
    et = event_type if event_type in record.EVENT_TYPES else "note"
    ev = record.log_event(rid, agent, et, content)
    return {"ok": True, "run_id": rid, "seq": ev["seq"], "type": et}


# =========================================================================== knowledge tools


def search_literature(query: str, n: int = 5, agent: str = "literature", run_id: str = "") -> dict:
    """Search published literature (Europe PMC + OpenAlex) for evidence.
    :param query: Free-text query, e.g. "moonwalker descending neurons backward walking Drosophila".
    :param n: Max results per source (1-10).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: {"results": [{"title","year","doi","pmid","authors","abstract","url","source"}], ...}
        Abstracts are truncated. Only cite DOIs that appear here or in list_ground_truth.
    """
    rid = _rid(run_id)
    n = max(1, min(int(n), 10))
    try:
        if _mock("literature"):
            items = [{"title": f"MOCK paper {i + 1} for '{query}'", "year": None, "doi": None, "pmid": None,
                      "authors": None, "abstract": MOCK_NOTICE, "url": None, "source": "mock", "mock": True}
                     for i in range(min(n, 2))]
        else:
            from flylab import literature
            items = []
            for fn in (literature.search_europepmc, literature.search_openalex):
                try:
                    items.extend(fn(query, n=n))
                except Exception as exc:  # one source failing must not kill the search
                    items.append({"source": getattr(fn, "__name__", "?"), "error": str(exc)})
        seen, uniq = set(), []
        for it in items:
            key = (it.get("doi") or "").lower() or (it.get("title") or "").lower()[:80]
            if key and key in seen:
                continue
            seen.add(key)
            it = dict(it)
            if it.get("abstract"):
                it["abstract"] = str(it["abstract"])[:700]
            uniq.append(it)
        dois = [it["doi"] for it in uniq if it.get("doi")]
        record.log_event(rid, agent, "evidence", f"Literature search: {query!r} -> {len(uniq)} hits",
                         {"query": query, "hits": [{k: it.get(k) for k in ("title", "year", "doi", "source")}
                                                    for it in uniq], "mock": _mock("literature")}, dois)
        return {"ok": True, "query": query, "n_results": len(uniq), "results": uniq,
                "mock": _mock("literature"), **({"notice": MOCK_NOTICE} if _mock("literature") else {})}
    except Exception as exc:
        return _err("search_literature", exc, rid, agent)


def list_neuron_groups(role: str = "", agent: str = "", run_id: str = "") -> dict:
    """List the curated FlyWire v783 neuron groups that experiments can target.
    :param role: Optional filter, e.g. "descending", "sensory", "motor". Empty = all.
    :param agent: Calling agent role name (not logged; read-only).
    :param run_id: Unused (read-only tool), accepted for uniformity.
    :returns: {"groups": [{"name","cell_type","side","role","n_neurons","description","citations"}]}
    """
    try:
        out = []
        for name, g in _groups().items():
            if role and g.get("role") != role:
                continue
            out.append({"name": name, "cell_type": g.get("cell_type"), "side": g.get("side"), "role": g.get("role"),
                        "n_neurons": len(g.get("root_ids", [])), "description": g.get("description", "")[:300],
                        "citations": g.get("citations", [])})
        return {"ok": True, "n_groups": len(out), "groups": out, "mock": _mock("atlas"),
                **({"notice": MOCK_NOTICE} if _mock("atlas") else {})}
    except Exception as exc:
        return _err("list_neuron_groups", exc)


def lookup_neurons(query: str, agent: str = "literature", run_id: str = "") -> dict:
    """Search the FlyWire annotation atlas by cell type / hemibrain type (e.g. "MDN", "DNa02", "P9").
    :param query: Cell-type query string.
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: {"matches": [...]} (at most 50) - each with root ids and annotations.
    """
    try:
        if _mock("atlas"):
            q = query.lower()
            matches = [{"name": k, **v} for k, v in _MOCK_GROUPS.items()
                       if q in k.lower() or q in str(v.get("cell_type", "")).lower()]
        else:
            from flylab import atlas
            matches = atlas.find_cell_type(query)
        return {"ok": True, "query": query, "n_matches": len(matches), "matches": record._jsonable(matches[:50]),
                "truncated": len(matches) > 50, "mock": _mock("atlas"),
                **({"notice": MOCK_NOTICE} if _mock("atlas") else {})}
    except Exception as exc:
        return _err("lookup_neurons", exc)


def list_ground_truth(agent: str = "", run_id: str = "") -> dict:
    """List published experimental results (activation/silencing -> behavior) used for validation.
    :param agent: Calling agent role name (read-only, not logged).
    :param run_id: Unused, accepted for uniformity.
    :returns: {"entries": [{"id","manipulation","target_group","expected_behavior","evidence","citation"}]}
    """
    try:
        gt = _ground_truth()
        return {"ok": True, "n_entries": len(gt), "entries": gt, "mock": _mock("atlas"),
                **({"notice": MOCK_NOTICE} if _mock("atlas") else {})}
    except Exception as exc:
        return _err("list_ground_truth", exc)


# =========================================================================== hypothesis / planning tools


def log_hypothesis(statement: str, manipulation: str, target_groups: list, predicted_behavior: str,
                   rationale: str = "", confidence: float = 0.5, citations: list = None,
                   agent: str = "hypothesis", run_id: str = "") -> dict:
    """Record a testable, AGENT-GENERATED hypothesis in the research record.
    :param statement: One-sentence hypothesis.
    :param manipulation: "activate" or "silence" (or "activate+silence").
    :param target_groups: Neuron group names (see list_neuron_groups).
    :param predicted_behavior: One of forward, backward, turn_left, turn_right, stop, escape, groom, feed.
    :param rationale: Why - which evidence supports it, and what would falsify it.
    :param confidence: Prior confidence 0..1 (subjective, agent-assigned).
    :param citations: DOIs supporting the rationale (only DOIs actually retrieved).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    """
    rid = _rid(run_id)
    groups = _as_list(target_groups)
    pb = str(predicted_behavior).strip().lower()
    hid = f"H{sum(1 for e in record.load(rid) if e.get('type') == 'hypothesis') + 1}"
    data = {"hypothesis_id": hid, "statement": statement, "manipulation": manipulation, "target_groups": groups,
            "predicted_behavior": pb, "rationale": rationale, "confidence": float(confidence),
            "provenance": "agent-generated (not established fact)",
            "warning": None if pb in BEHAVIORS else f"predicted_behavior not in {BEHAVIORS}"}
    record.log_event(rid, agent, "hypothesis", f"[{hid}, agent-generated] {statement}", data, _as_list(citations))
    return {"ok": True, "run_id": rid, "hypothesis_id": hid, **data}


def estimate_cost(kind: str, duration_ms: float = 1000.0, n_trials: int = 3, duration_s: float = 1.0,
                  n_candidates: int = 1) -> dict:
    """Rough compute-cost estimate for an experiment, used by the planner to trade off information vs cost.
    :param kind: "brain", "body", "embodied", "screen" (run_brain_screen over n_candidates cell types) or
        "rank" (rank_candidates, connectome only).
    :param duration_ms: Simulated brain time per trial (ms) (screen default in run_brain_screen: 500).
    :param n_trials: Brain trials (screen default: 2).
    :param duration_s: Simulated body time (s).
    :param n_candidates: Number of candidate cell types for kind="screen" (e.g. 326 = all visual projection types).
    :returns: {"est_wall_s", "cost_units", "needs_approval"} - heuristic, NOT measured; refine with runtime_s
        returned by real runs.
    """
    # Heuristic constants, calibrated on this laptop (refine with runtime_s returned by real runs):
    # brain: spikes/brain/out/validation.json, 10 trials x 1000 ms took 6-61 s wall (depends on how much of
    #        the brain becomes active) -> ~3 s per simulated second per trial + ~5 s connectome load/setup.
    # body:  flylab/body.py notes "1 sim-s costs ~7-25 s wall" -> 25 s/sim-s (conservative, incl. video).
    # screen: flylab.screen.brain_screen with 4 threads, ~1 s wall per trial-second + ~1 s setup per candidate,
    #         + ~10 s one-off connectome load (Phase 1 note: ~1 s wall per trial-second); rank: ~15 s incl. load.
    brain_s = 5.0 + 3.0 * duration_ms / 1000.0 * n_trials
    body_s = 25.0 * duration_s
    n_c = max(1, int(n_candidates))
    screen_s = 10.0 + n_c * (1.0 + 1.5 * duration_ms / 1000.0 * n_trials)
    k = kind.strip().lower()
    est = {"brain": brain_s, "body": body_s, "embodied": brain_s + body_s, "screen": screen_s, "rank": 15.0}.get(k)
    if est is None:
        return {"ok": False, "error": f"unknown kind {kind!r}; use brain, body, embodied, screen or rank"}
    units = round(est / 10.0, 2)
    return {"ok": True, "kind": k, "est_wall_s": round(est, 1), "cost_units": units,
            "n_candidates": n_c if k == "screen" else None,
            "needs_approval": k == "embodied" or (k == "brain" and duration_ms * n_trials > 10000)
                              or (k == "screen" and n_c * duration_ms * n_trials > 10000),
            "denied_by_policy": k == "screen" and n_c > 40,
            "basis": "heuristic, calibrated on this laptop (brain validation runs, body.py notes); "
                     "compare with runtime_s from real runs"}


def log_experiment_plan(options: list, chosen_id: str, rationale: str, budget_units: float = 10.0,
                        agent: str = "planner", run_id: str = "") -> dict:
    """Record >=2 candidate experiments and the chosen one (expected information gain vs cost).
    :param options: List of candidate experiments, each an object like
        {"id": "E1", "kind": "brain|body|embodied", "excite_groups": [...], "silence_groups": [...],
         "tests_hypothesis": "H1", "expected_information_gain": "high|medium|low" or 0..1,
         "est_cost_units": 1.2, "why": "..."}.
    :param chosen_id: id of the chosen option.
    :param rationale: Why this option maximises information gain per cost within budget.
    :param budget_units: Remaining compute budget in cost units.
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    """
    rid = _rid(run_id)
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            options = [{"id": "?", "description": options}]
    opts = list(options or [])
    warn = None if len(opts) >= 2 else "fewer than 2 candidate experiments - planner contract expects >= 2"
    chosen = next((o for o in opts if str(o.get("id")) == str(chosen_id)), None) if opts and isinstance(opts[0], dict) else None
    record.log_event(rid, agent, "experiment_options", f"{len(opts)} candidate experiments proposed",
                     {"options": opts, "budget_units": budget_units, "warning": warn})
    record.log_event(rid, agent, "experiment_choice", f"Chose {chosen_id}: {rationale}",
                     {"chosen_id": chosen_id, "chosen": chosen, "rationale": rationale, "budget_units": budget_units})
    return {"ok": True, "run_id": rid, "n_options": len(opts), "chosen": chosen or chosen_id, "warning": warn}


def log_decision(decision: str, reason: str, next_step: str = "", reopens_assumption: str = "",
                 agent: str = "supervisor", run_id: str = "") -> dict:
    """Record a decision (e.g. revise plan after a surprising result) in the research record.
    :param decision: What was decided, e.g. "Switch to silencing experiment".
    :param reason: Evidence-based reason (cite result seq numbers / DOIs).
    :param next_step: The next concrete action.
    :param reopens_assumption: Assumption that a surprise has reopened (if any).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    """
    rid = _rid(run_id)
    ev = record.log_event(rid, agent, "decision", decision, {"reason": reason, "next_step": next_step,
                                                              "reopens_assumption": reopens_assumption or None})
    return {"ok": True, "run_id": rid, "seq": ev["seq"]}


def request_approval(action: str, reason: str, est_cost: float = 0.0, agent: str = "safety", run_id: str = "") -> dict:
    """Ask the human PI to approve a risky or expensive action.
    Under Omnigent this call is gated by the ``approval_gate`` policy (ASK): the body only
    runs after a human clicked "approve" in the Omnigent UI/REPL; a denial blocks it.
    :param action: What will be run, e.g. "3 embodied runs: MDN activation, P9 activation, MDN silencing".
    :param reason: Why it is worth the cost (expected information gain).
    :param est_cost: Estimated cost units (see estimate_cost).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    """
    rid = _rid(run_id)
    # This body only executes after the Omnigent ASK gate was approved (a denial blocks the call).
    # Called outside Omnigent (selftest, scripts) there is no human gate - the record says so.
    pre = _preapproval()
    if pre:
        status = "pre-approved"
        gate = (f"omnigent policy lab_approval_gate: PRE-APPROVED by {pre.get('by', 'human at launch')} for this "
                f"scripted run (agents/omni.ps1 -ApproveAtLaunch; caps still enforced: {pre.get('caps', 'see policy')})")
        content = f"Pre-approved at launch (scripted run): {action}"
    else:
        status = "approved"
        gate = ("omnigent policy lab_approval_gate (ASK): this entry is only written after a human approved "
                "in the Omnigent UI/REPL; outside Omnigent no human gate exists")
        content = f"Approved via approval gate: {action}"
    data = {"action": action, "reason": reason, "est_cost_units": est_cost, "status": status, "gate": gate}
    ev = record.log_event(rid, agent, "approval", content, data)
    return {"ok": True, "approved": True, "status": status, "run_id": rid, "seq": ev["seq"]}


# =========================================================================== experiment tools


def run_brain_experiment(excite_groups: list, silence_groups: list = None, rate_hz: float = 150.0,
                         duration_ms: float = 1000.0, n_trials: int = 3, seed: int = 0, agent: str = "runner",
                         run_id: str = "") -> dict:
    """In-silico activation/silencing in the whole-brain LIF connectome model (FlyWire v783, after Shiu et al. 2024).
    :param excite_groups: Group names (or root ids) driven with Poisson input, e.g. ["MDN"].
    :param silence_groups: Group names (or root ids) silenced, e.g. ["P9"]. Optional.
    :param rate_hz: Poisson rate of the excitation (Hz). Default 150.
    :param duration_ms: Simulated time per trial (ms). Default 1000.
    :param n_trials: Number of trials (each costs compute). Default 3.
    :param seed: Random seed of the Poisson input; the same seed + parameters gives an IDENTICAL result, so use
        different seeds for independent replicates.
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: descending-neuron group rates (Hz), top responding neurons, n_active, runtime_s,
        path of the full result JSON in runs/<id>/artifacts.
    """
    rid = _rid(run_id)
    ex, si = _as_list(excite_groups), _as_list(silence_groups)
    try:
        t0 = time.time()
        res, meta = _simulate_brain(ex, si, rate_hz, duration_ms, n_trials, seed)
        rates = {str(k): float(v) for k, v in (res.get("rates") or {}).items()}
        top = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:15]
        dn = _group_rates(rates, "descending")
        # Non-descending groups (motor, sensory, ...) that fire: needed for brain-level readouts such as
        # MN9 for feeding (Shiu et al. 2024) - ground-truth entries name these as "readout_group".
        other = {k: v for k, v in _group_rates(rates, None).items() if k not in dn and v > 0}
        other = dict(sorted(other.items(), key=lambda kv: kv[1], reverse=True)[:25])
        path = _save_json(_next_artifact(rid, "brain", "json"), {"inputs": {"excite": ex, "silence": si,
                          "rate_hz": rate_hz, "duration_ms": duration_ms, "n_trials": n_trials, "seed": seed},
                          **meta, "result": res})
        out = {"ok": True, "run_id": rid, "kind": "brain", "excite": ex, "silence": si, "seed": seed, **meta,
               "descending_group_rates_hz": dn, "other_group_rates_hz": other,
               "top_neurons": [{"root_id": k, "rate_hz": round(v, 2)} for k, v in top],
               "n_active": res.get("n_active", len(rates)), "runtime_s": res.get("runtime_s", round(time.time() - t0, 2)),
               "n_unknown_ids": len(res.get("unknown_ids") or []),
               "params": res.get("params"), "artifact": path, "mock": _mock("brain")}
        if _mock("brain"):
            out["notice"] = MOCK_NOTICE
        record.log_event(rid, agent, "experiment_result",
                         f"Brain sim: excite={ex} silence={si} -> {out['n_active']} active neurons"
                         + (" [MOCK]" if _mock("brain") else ""),
                         {k: out[k] for k in ("kind", "excite", "silence", "seed", "descending_group_rates_hz",
                                              "other_group_rates_hz", "n_active", "n_unknown_ids",
                                              "runtime_s", "artifact", "mock", "unknown_groups")})
        return out
    except Exception as exc:
        return _err("run_brain_experiment", exc, rid, agent)


def run_body_experiment(forward: float = 1.0, turn: float = 0.0, backward: float = 0.0, duration_s: float = 1.0,
                        render: bool = True, agent: str = "runner", run_id: str = "") -> dict:
    """Drive the NeuroMechFly (flygym) physics body directly with a descending command (no brain).
    :param forward: Forward drive 0..1.
    :param turn: Turn drive -1..1 (negative = left, positive = right).
    :param backward: Backward drive 0..1.
    :param duration_s: Simulated seconds (keep <= 2 for speed).
    :param render: Save an mp4 into runs/<id>/artifacts.
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: behavior label, displacements, heading change, mean speed, video path, runtime_s.
    """
    rid = _rid(run_id)
    drive = {"forward": forward, "turn": turn, "backward": backward}
    try:
        video = str(_next_artifact(rid, "body", "mp4")) if render else None
        res = _simulate_body(drive, duration_s, video)
        summ = _body_summary(res, duration_s)
        out = {"ok": True, "run_id": rid, "kind": "body", "drive": drive, **summ, "mock": _mock("body")}
        if _mock("body"):
            out["notice"] = MOCK_NOTICE
        record.log_event(rid, agent, "experiment_result",
                         f"Body sim: drive={drive} -> {summ['behavior']}" + (" [MOCK]" if _mock("body") else ""),
                         {"kind": "body", "drive": drive, **summ, "mock": _mock("body")})
        return out
    except Exception as exc:
        return _err("run_body_experiment", exc, rid, agent)


def run_embodied_experiment(excite_groups: list, silence_groups: list = None, rate_hz: float = 150.0,
                            duration_ms: float = 1000.0, n_trials: int = 3, duration_s: float = 1.0,
                            seed: int = 0, agent: str = "runner", run_id: str = "") -> dict:
    """Full closed chain: connectome brain -> descending-neuron rates -> bridge -> physics body -> behavior + video.
    EXPENSIVE: requires human approval (Omnigent policy approval_gate).
    :param excite_groups: Group names (or root ids) to activate, e.g. ["MDN"].
    :param silence_groups: Group names (or root ids) to silence. Optional.
    :param rate_hz: Poisson excitation rate (Hz).
    :param duration_ms: Brain simulation time per trial (ms).
    :param n_trials: Brain trials.
    :param duration_s: Body simulation time (s).
    :param seed: Random seed of brain input and body (same seed + parameters = identical result).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: descending rates, drive, behavior, body metrics, video path (runs/<id>/artifacts), runtimes.
    """
    rid = _rid(run_id)
    ex, si = _as_list(excite_groups), _as_list(silence_groups)
    try:
        bres, meta = _simulate_brain(ex, si, rate_hz, duration_ms, n_trials, seed)
        rates = {str(k): float(v) for k, v in (bres.get("rates") or {}).items()}
        dn = _group_rates(rates, "descending")
        drive, bridge_info = _drive_from_rates(rates)
        video_path = _next_artifact(rid, "embodied", "mp4")  # .json below shares the number
        body_res = _simulate_body(drive, duration_s, str(video_path), seed)
        summ = _body_summary(body_res, duration_s)
        mock = any(_mock(c) for c in ("brain", "bridge", "body"))
        path = _save_json(video_path.with_suffix(".json"),
                          {"inputs": {"excite": ex, "silence": si, "rate_hz": rate_hz, "duration_ms": duration_ms,
                                      "n_trials": n_trials, "duration_s": duration_s, "seed": seed}, **meta,
                           "descending_group_rates_hz": dn, "drive": drive, "bridge": bridge_info, "body": body_res,
                           "brain_runtime_s": bres.get("runtime_s"), "mock": mock})
        out = {"ok": True, "run_id": rid, "kind": "embodied", "excite": ex, "silence": si, **meta,
               "descending_group_rates_hz": dn, "drive": drive, "bridge": bridge_info, **summ,
               "brain_runtime_s": bres.get("runtime_s"), "body_runtime_s": body_res.get("runtime_s"),
               "artifact": path, "mock": mock}
        if body_res.get("warnings"):
            out["body_warnings"] = body_res.get("warnings")
        if mock:
            out["notice"] = MOCK_NOTICE
        record.log_event(rid, agent, "experiment_result",
                         f"Embodied sim: excite={ex} silence={si} -> behavior={summ['behavior']}"
                         + (" [MOCK]" if mock else ""),
                         {k: out.get(k) for k in ("kind", "excite", "silence", "descending_group_rates_hz", "drive",
                                                  "behavior", "forward_disp_mm", "heading_change_deg",
                                                  "mean_speed_mm_s", "fell_over", "label_warning", "video", "artifact", "mock",
                                                  "unknown_groups", "bridge")})
        return out
    except Exception as exc:
        return _err("run_embodied_experiment", exc, rid, agent)


# =========================================================================== analysis tools

# Mirrors flylab.bridge.rates_to_drive (backward = MDN, forward = P9): comparing these groups' manipulation with
# the behavior they are wired to is a check of the hand-designed bridge + body, not an independent test.
_BRIDGE_SOURCE = {"backward": ("MDN",), "forward": ("P9",)}
_ARTIFACT_RE = re.compile(r"\b((?:embodied|brain|screen|body)_\d+)", re.IGNORECASE)


def _same_group(a: str, b: str) -> bool:
    a, b = str(a).strip().lower(), str(b).strip().lower()
    return a == b or a.startswith(b + "_") or b.startswith(a + "_")


def _check_experiment_ref(rid: str, experiment_ref: str, entry: dict, obs: str) -> dict:
    """Load the artifacts named in experiment_ref (e.g. "embodied_01.json + screen_01") and check
    (a) that they apply the published manipulation to the published target group, (b) where the observed
    label comes from (body classifier vs agent override vs brain-readout inference). Never raises."""
    out = {"manipulation_match": None, "manipulated": None, "label_source": "unverified", "body_label": None,
           "by_construction": False, "note": ""}
    tgt = str(entry.get("target_group") or "")
    exp = str(entry.get("expected_behavior") or "").lower()
    if any(_same_group(tgt, g) for g in _BRIDGE_SOURCE.get(exp, ())):
        out["by_construction"] = True
        out["note"] = (f"{tgt} -> {exp} is wired into the bridge by construction; agreement checks the bridge/body, "
                       "it is not independent evidence.")
    try:
        names = sorted({m.lower() for m in _ARTIFACT_RE.findall(str(experiment_ref or ""))})
        arts = []
        for n in names:
            p = record.artifacts_dir(rid) / f"{n}.json"
            if p.exists():
                arts.append((n, json.loads(p.read_text(encoding="utf-8"))))
        if not arts:
            out["note"] = (out["note"] + " " if out["note"] else "") + \
                "experiment_ref names no artifact of this run - manipulation and label source unverified."
            return out
        manip, body_labels = {}, []
        for n, a in arts:
            inp = a.get("inputs") or {}
            ex = _as_list(inp.get("candidates") if n.startswith("screen") else inp.get("excite"))
            si = _as_list(inp.get("silence"))
            manip[n] = {"excite": ex, "silence": si}
            if n.startswith("embodied") and isinstance(a.get("body"), dict) and a["body"].get("behavior"):
                body_labels.append(str(a["body"]["behavior"]))
        out["manipulated"] = manip
        key = "silence" if str(entry.get("manipulation")) == "silence" else "excite"
        out["manipulation_match"] = any(_same_group(tgt, g) for m in manip.values() for g in m[key])
        if not out["manipulation_match"]:
            out["note"] = (out["note"] + " " if out["note"] else "") + (
                f"manipulation mismatch: this entry is '{entry.get('manipulation')} {tgt}', but the referenced "
                f"experiment(s) {manip} did not {entry.get('manipulation')} {tgt} - not a test of this entry.")
        if body_labels:
            out["body_label"] = body_labels[0] if len(set(body_labels)) == 1 else body_labels
            if obs in body_labels:
                out["label_source"] = "body_classifier"
            else:
                out["label_source"] = "agent_override"
                out["note"] = (out["note"] + " " if out["note"] else "") + (
                    f"observed '{obs}' is the agent's label; the body classifier labelled the run {body_labels}.")
        else:
            out["label_source"] = "brain_readout_inference"
    except Exception as exc:  # the check is advisory; never break the comparison
        out["note"] = (out["note"] + " " if out["note"] else "") + f"experiment_ref check failed: {exc}"
    return out


def compare_to_ground_truth(observed_behavior: str, ground_truth_id: str, experiment_ref: str = "",
                            control_behavior: str = "", agent: str = "analysis", run_id: str = "") -> dict:
    """Compare a simulated behavior with a published experimental result (activation: behavior should appear; silencing: behavior should be reduced vs. a control).
    :param observed_behavior: Behavior label from the manipulated body/embodied run (forward, backward, turn_left, ...).
    :param ground_truth_id: id from list_ground_truth.
    :param experiment_ref: Which run/artifact produced the observation (e.g. artifact path or record seq).
    :param control_behavior: For silencing entries: behavior of the matching control run WITHOUT the silencing (strongly recommended).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: {"verdict": "consistent"|"partially_consistent"|"inconsistent"|"inconclusive", "surprise": bool, "comparable": bool,
        "expected", "effect", "readout_group", "gt_confidence", "citation"}. Opposite turn direction = inconsistent;
        escape/groom/feed vs. a walking-body label = inconclusive (compare the readout group's brain firing instead).
    """
    rid = _rid(run_id)
    try:
        gt = {str(e.get("id")): e for e in _ground_truth()}
        entry = gt.get(str(ground_truth_id))
        if entry is None:
            return {"ok": False, "error": f"unknown ground_truth_id {ground_truth_id!r}", "known_ids": sorted(gt)[:50]}
        obs = str(observed_behavior).strip().lower().replace(" ", "_")
        ctrl = str(control_behavior or "").strip().lower().replace(" ", "_")
        exp = str(entry.get("expected_behavior", "")).strip().lower()
        effect = str(entry.get("effect") or ("reduce" if entry.get("manipulation") == "silence" else "induce")).lower()
        readout = entry.get("readout_group")
        note = ""
        comparable = True
        if obs not in BEHAVIORS:
            verdict, comparable = "inconclusive", False
            note = f"observed label {obs!r} is not one of {BEHAVIORS}; pass the body's behavior label."
        elif exp not in BODY_BEHAVIORS and obs in BODY_BEHAVIORS:
            # The NeuroMechFly walking body cannot express escape / groom / feed (same rule as atlas.evaluate).
            verdict, comparable = "inconclusive", False
            note = (f"'{exp}' cannot be produced by the walking body - not comparable at body level. Compare "
                    f"brain-level firing of {readout or 'a readout group'} (run_brain_experiment: "
                    "descending_group_rates_hz / other_group_rates_hz) against a control run instead.")
        elif effect == "reduce":
            # Published result: the manipulation REDUCES/abolishes `exp`.
            if obs == exp:
                verdict = "inconsistent"
                note = f"{exp} still present despite the manipulation that should reduce it."
            elif ctrl and ctrl != exp:
                verdict = "inconclusive"
                note = f"control run did not show {exp} either ({ctrl}); the comparison cannot test the reduction."
            elif ctrl:
                verdict = "consistent"
                note = f"control shows {exp}, manipulated run does not ({obs})."
            else:
                verdict = "partially_consistent"
                note = f"{exp} absent, but no control run was given - weak evidence."
        else:
            if obs == exp:
                verdict = "consistent"
            elif obs in TURNS and exp in TURNS:
                # Wrong laterality contradicts the published result (same rule as atlas.evaluate).
                verdict = "inconsistent"
                note = "turning in the OPPOSITE direction to the published result (laterality mismatch)."
            elif {obs, exp} <= {"forward"} | TURNS:
                verdict = "partially_consistent"
                note = "locomotion matches, but the steering component differs."
            else:
                verdict = "inconsistent"
                note = f"expected {exp}, observed {obs}."
        if comparable and exp not in BODY_BEHAVIORS:
            note = (note + " " if note else "") + (f"NOTE: '{exp}' is not a body-model behavior; the observed label "
                                                    "is an agent interpretation of brain readouts.")
        # Check the referenced experiment(s) against the published manipulation and the label provenance
        # (review 2026-10-04: an LC16 run was logged as 'consistent' with "MDN activation -> backward", and an
        # agent-chosen 'backward' label was logged although the body classifier said 'stop').
        chk = _check_experiment_ref(rid, experiment_ref, entry, obs)
        if chk["manipulation_match"] is False:
            verdict, comparable = "inconclusive", False
            note = (note + " " if note else "") + chk["note"]
        elif chk["note"]:
            note = (note + " " if note else "") + chk["note"]
        surprise = verdict in ("inconsistent", "partially_consistent")
        cit = entry.get("citation") or {}
        doi = cit.get("doi") if isinstance(cit, dict) else None
        out = {"ok": True, "run_id": rid, "ground_truth_id": ground_truth_id, "manipulation": entry.get("manipulation"),
               "target_group": entry.get("target_group"), "effect": effect, "expected": exp, "observed": obs,
               "control": ctrl or None, "verdict": verdict, "note": note, "surprise": surprise,
               "comparable": comparable, "readout_group": readout,
               "gt_confidence": entry.get("confidence"), "gt_note": entry.get("note"),
               "evidence": entry.get("evidence"), "citation": cit, "experiment_ref": experiment_ref,
               "manipulation_match": chk["manipulation_match"], "manipulated": chk["manipulated"],
               "label_source": chk["label_source"], "body_label": chk["body_label"],
               "by_construction": chk["by_construction"], "mock": bool(entry.get("mock"))}
        tags = []
        if chk["label_source"] == "agent_override":
            tags.append(f"agent label; body classifier said {chk['body_label']}")
        elif chk["label_source"] == "brain_readout_inference":
            tags.append("label inferred from brain readout, no body run")
        if chk["by_construction"]:
            tags.append("bridge maps this group to this behavior by construction")
        if chk["manipulation_match"] is False:
            tags.append("NOT a test of this entry: manipulation mismatch")
        record.log_event(rid, agent, "analysis",
                         f"{ground_truth_id} ({effect} {exp}): observed {obs}"
                         + (f", control {ctrl}" if ctrl else "") + f" -> {verdict}"
                         + (f" [{'; '.join(tags)}]" if tags else "")
                         + (" (SURPRISE - reopen assumptions)" if surprise else ""),
                         {k: out[k] for k in ("ground_truth_id", "manipulation", "target_group", "effect", "expected",
                                              "observed", "control", "verdict", "note", "surprise", "comparable",
                                              "readout_group", "gt_confidence", "experiment_ref", "manipulation_match",
                                              "manipulated", "label_source", "body_label", "by_construction", "mock")},
                         [doi] if doi else [])
        return out
    except Exception as exc:
        return _err("compare_to_ground_truth", exc, rid, agent)


# =========================================================================== discovery-engine tools (flylab.screen)
# flylab.screen (Phase 2 contract) does connectome-guided ranking + a fast in-silico brain screen. It is
# imported lazily; with FLYLAB_MOCK=screen (or =1) the tools return clearly labelled MOCK rows, and in
# real mode a missing/broken module returns ok=false (never silent fake data).

SCREEN_HIT_HZ = 5.0  # same pre-registered hit threshold as flylab.screen.benchmark_search
_MOCK_SCREEN_TYPES = ["MOCK_LC16", "MOCK_LPLC2", "MOCK_LC4", "MOCK_LC9", "MOCK_LPC1", "MOCK_LT86"]


def _screen_known_hits(target_group: str) -> dict:
    try:
        from flylab import screen
        return {k: v.get("gt", []) for k, v in screen.KNOWN_HITS.get(target_group, {}).items()}
    except Exception:
        return {}


def rank_candidates(target_group: str, candidate_kind: str = "visual_projection", top_k: int = 10,
                    max_hops: int = 3, agent: str = "hypothesis", run_id: str = "") -> dict:
    """Connectome-guided prior: rank ALL candidate cell types (e.g. the 326 FlyWire visual projection types) by how
    strongly the signed FlyWire v783 connectome predicts they excite a target group (e.g. MDN = moonwalker / backward
    walking). Cheap (seconds, no simulation). A high score is a PREDICTION to be tested with run_brain_screen, not a result.
    :param target_group: Target neuron group, e.g. "MDN" (backward walking) or "GF" (escape).
    :param candidate_kind: "visual_projection", "descending", "ascending", "sensory" or a regex on cell types.
    :param top_k: How many top-ranked candidates to return (1-40).
    :param max_hops: Path length considered (1 = direct synapses only; default 3 = the primary score of the committed
        screen benchmark, data/benchmarks/screen_mdn.json; 2 hops misses LC16 -> MDN).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: {"n_candidates", "top": [{rank, cell_type, n, score, direct_syn, two_hop_score, sign_note}],
        "literature_known_hits": {cell_type: {"rank", "score", "gt_ids"}}, "runtime_s", "method"}
    """
    rid = _rid(run_id)
    top_k = max(1, min(int(top_k), 40))
    t0 = time.time()
    try:
        if _mock("screen"):
            ranked = [{"rank": i + 1, "cell_type": c, "n": 10 + i, "score": round(0.003 / (i + 1), 6), "direct_syn": 0,
                       "two_hop_score": round(0.003 / (i + 1), 6), "sign_note": "MOCK"} for i, c in enumerate(_MOCK_SCREEN_TYPES)]
            n_total, known, method_used = len(ranked), {}, "MOCK"
        else:
            from flylab import screen
            full = screen.rank_by_connectome(target_group, candidate_kind, max_hops=int(max_hops))
            n_total = len(full)
            ranked = [{k: r.get(k) for k in ("rank", "cell_type", "n", "score", "pred_rate_hz", "direct_syn",
                                             "two_hop_score", "sign_note")}
                      for r in full[:top_k]]
            method_used = (full[0].get("method") if full else None) or "meanfield"
            pos = {r["cell_type"]: r for r in full}
            known = {ct: {"rank": pos[ct]["rank"] if ct in pos else None,
                          "score": pos[ct]["score"] if ct in pos else None, "gt_ids": gts}
                     for ct, gts in _screen_known_hits(target_group).items()}
        for r in ranked:
            if isinstance(r.get("score"), float):
                r["score"] = round(r["score"], 6)
        out = {"ok": True, "run_id": rid, "target_group": target_group, "candidate_kind": candidate_kind,
               "n_candidates": n_total, "top_k": top_k, "top": ranked, "literature_known_hits": known,
               "runtime_s": round(time.time() - t0, 2), "max_hops": max_hops,
               "method": (f"{'MOCK' if _mock('screen') else method_used} connectome score over {max_hops} hops "
                          "(flylab.screen.rank_by_connectome); a prior from the SAME connectome the brain model uses"),
               "mock": _mock("screen")}
        if _mock("screen"):
            out["notice"] = MOCK_NOTICE
        record.log_event(rid, agent, "evidence",
                         f"Connectome ranking of {n_total} {candidate_kind} types -> {target_group}: top "
                         + ", ".join(str(r["cell_type"]) for r in ranked[:5]) + (" [MOCK]" if _mock("screen") else ""),
                         {"kind": "connectome_ranking", "target_group": target_group, "candidate_kind": candidate_kind,
                          "n_candidates": n_total, "top": ranked, "literature_known_hits": known,
                          "runtime_s": out["runtime_s"], "provenance": "computed connectome prior (prediction, not a result)",
                          "mock": _mock("screen")})
        return out
    except Exception as exc:
        return _err("rank_candidates", exc, rid, agent)


def run_brain_screen(candidates: list, target_groups: list, rate_hz: float = 150.0, duration_ms: float = 500.0,
                     n_trials: int = 2, seed: int = 0, agent: str = "runner", run_id: str = "") -> dict:
    """In-silico activation screen: activate each candidate cell type (all its neurons, Poisson input) in the whole-brain
    LIF model (FlyWire v783, Shiu et al. 2024) and read the mean firing rate of each target group. A candidate is a HIT
    when a target fires >= 5 Hz (model baseline is 0 Hz). Cost ~1 s wall per candidate per trial-second.
    Gated by policy: > 40 candidates per call is denied; > 10 000 simulated ms in total needs approval.
    :param candidates: Cell types to activate, e.g. ["LC16", "LPLC2", "LC9"] (from rank_candidates).
    :param target_groups: Groups to read out, e.g. ["MDN", "GF", "P9"].
    :param rate_hz: Poisson activation rate (Hz).
    :param duration_ms: Simulated time per trial (ms).
    :param n_trials: Trials per candidate.
    :param seed: Random seed (default 0 = the seed of the committed benchmark, so rows may come from its cache).
    :param agent: Calling agent role name.
    :param run_id: Run id; default = active run.
    :returns: {"rows": [{cell_type, n_stimulated, target_rates, hit_targets, runtime_s, cached}], "hits", "runtime_s"}
    """
    rid = _rid(run_id)
    cands, targets = _as_list(candidates), _as_list(target_groups)
    t0 = time.time()
    try:
        if not cands or not targets:
            raise ValueError("candidates and target_groups must be non-empty")
        if _mock("screen"):
            rows = [{"cell_type": c, "n_stimulated": 10, "cached": False, "runtime_s": 0.01,
                     "target_rates": {g: (12.0 if (i == 0 and g in ("MDN", "MDN_L", "MDN_R")) else 0.0) for g in targets}}
                    for i, c in enumerate(cands)]
        else:
            from flylab import screen
            rows = screen.brain_screen(cands, targets, rate_hz=float(rate_hz), duration_ms=float(duration_ms),
                                       n_trials=int(n_trials), seed=int(seed), n_threads=4)
        for r in rows:
            r["hit_targets"] = [g for g, v in (r.get("target_rates") or {}).items() if float(v) >= SCREEN_HIT_HZ]
        hits = [r["cell_type"] for r in rows if r["hit_targets"]]
        # Per-target hits: with several targets (MDN, P9, GF) "hit on any target" mixes behaviours
        # (MDN = backward, P9 = forward, GF = escape); report them separately.
        hits_by_target = {g: [r["cell_type"] for r in rows if g in r["hit_targets"]] for g in targets}
        wall = round(time.time() - t0, 2)
        mock = _mock("screen")
        path = _save_json(_next_artifact(rid, "screen", "json"),
                          {"inputs": {"candidates": cands, "target_groups": targets, "rate_hz": rate_hz,
                                      "duration_ms": duration_ms, "n_trials": n_trials, "seed": seed},
                           "hit_threshold_hz": SCREEN_HIT_HZ, "rows": rows, "hits": hits, "hits_by_target": hits_by_target,
                           "wall_s": wall, "mock": mock})
        out = {"ok": True, "run_id": rid, "kind": "screen", "n_candidates": len(cands), "target_groups": targets,
               "rows": rows, "hits": hits, "hits_by_target": hits_by_target, "hit_threshold_hz": SCREEN_HIT_HZ,
               "runtime_s": wall,
               "sim_runtime_s_sum": round(sum(float(r.get("runtime_s") or 0) for r in rows), 2),
               "n_cached": sum(1 for r in rows if r.get("cached")), "artifact": path, "mock": mock,
               "note": "cached rows were simulated earlier with identical parameters (runtime_s = original measured time)"}
        if mock:
            out["notice"] = MOCK_NOTICE
        record.log_event(rid, agent, "experiment_result",
                         f"Brain screen: {len(cands)} candidates; hits (>= {SCREEN_HIT_HZ} Hz) per target: "
                         + "; ".join(f"{g}: {', '.join(v) or 'none'}" for g, v in hits_by_target.items())
                         + (" [MOCK]" if mock else ""),
                         {"kind": "screen", "candidates": cands, "target_groups": targets, "rate_hz": rate_hz,
                          "duration_ms": duration_ms, "n_trials": n_trials, "rows": rows, "hits": hits,
                          "hits_by_target": hits_by_target,
                          "runtime_s": wall, "artifact": path, "mock": mock})
        return out
    except Exception as exc:
        return _err("run_brain_screen", exc, rid, agent)


def get_benchmark(name: str = "", agent: str = "", run_id: str = "") -> dict:
    """Read a committed benchmark from data/benchmarks/ (e.g. "screen_MDN": measured acceleration of the
    connectome-guided screen vs random order vs exhaustive; "brain_validation": brain model vs paper).
    :param name: Benchmark name without .json; empty = list the available benchmarks.
    :param agent: Calling agent role name (read-only, not logged).
    :param run_id: Unused, accepted for uniformity.
    :returns: {"available": [...], "benchmark": {...}} (long lists truncated).
    """
    bdir = record.ROOT / "data" / "benchmarks"
    avail = sorted(p.stem for p in bdir.glob("*.json")) if bdir.exists() else []
    try:
        if not name:
            return {"ok": True, "available": avail}
        stem = Path(str(name)).stem
        stem = next((a for a in avail if a.lower() == stem.lower()), stem)  # "screen_MDN" -> screen_mdn.json (Linux too)
        path = bdir / f"{stem}.json"
        if not path.exists():
            if _mock("screen"):
                return {"ok": True, "available": avail, "mock": True, "notice": MOCK_NOTICE,
                        "benchmark": {"name": stem, "experiments_to_first_hit": {"guided": 2, "random_expected": 40.0},
                                      "reduction_factor_first_hit": 20.0}}
            return {"ok": False, "available": avail,
                    "error": f"benchmark {stem!r} not found (not computed yet?) - use one of {avail}"}
        data = json.loads(path.read_text(encoding="utf-8"))

        def slim(x: Any, depth: int = 0) -> Any:
            if isinstance(x, dict):
                return {k: slim(v, depth + 1) for k, v in list(x.items())[:40]}
            if isinstance(x, list):
                return [slim(v, depth + 1) for v in x[:12]] + ([f"... {len(x) - 12} more"] if len(x) > 12 else [])
            if isinstance(x, str) and len(x) > 600:
                return x[:600] + "..."
            return x
        s = slim(data)
        txt = json.dumps(s, default=str)
        if len(txt) > 12000:
            s = {k: (v if len(json.dumps(v, default=str)) < 2500 else "(large - see file)") for k, v in s.items()} \
                if isinstance(s, dict) else txt[:12000]
        return {"ok": True, "available": avail, "name": stem, "file": f"data/benchmarks/{stem}.json", "benchmark": s}
    except Exception as exc:
        return _err("get_benchmark", exc)


# =========================================================================== Omnigent policy


_EXPENSIVE_DEFAULT = ("run_embodied_experiment", "request_approval")
_PREFIXES = ("mcp__omnigent__", "functions.", "omnigent__")


def _bare(name: str) -> str:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


def _embodied_runs_in_record() -> int:
    """Embodied experiment results logged in the active run (0 if none / unreadable). Never raises."""
    try:
        rid = record.current_run_id()
        if not rid:
            return 0
        return sum(1 for e in record.load(rid) if e.get("type") == "experiment_result"
                   and isinstance(e.get("data"), dict) and e["data"].get("kind") == "embodied")
    except Exception:
        return 0


def approval_gate(ask_tools: list = None, brain_ms_threshold: float = 10000.0, max_embodied_runs: int = 6,
                  max_screen_candidates: int = 40):
    """Omnigent policy factory: human approval for expensive / risky lab actions.

    * tool calls to any of ``ask_tools`` -> ASK (human approves in the Omnigent UI/REPL)
    * run_brain_experiment with duration_ms * n_trials > brain_ms_threshold -> ASK
    * run_brain_screen with n_candidates * duration_ms * n_trials > brain_ms_threshold -> ASK;
      more than ``max_screen_candidates`` candidates in one call -> DENY (e.g. an exhaustive
      326-type screen must be split or replaced by a connectome-guided top-k screen)
    * more than ``max_embodied_runs`` embodied runs (per Omnigent session, and per research run as
      counted in the active record) -> DENY (hard cap)
    Scripted runs: if a human pre-approved at launch (``agents/omni.ps1 -ApproveAtLaunch`` -> flag file,
    see ``_preapproval``), every ASK above becomes ALLOW; the DENY caps stay.
    Abstains (None) on everything else so other policies (cost budget, tool-call cap) decide.
    """
    ask = set(ask_tools or _EXPENSIVE_DEFAULT)
    key = "_flylab_embodied_runs"

    def _ask(resp: dict) -> dict:
        if _preapproval():
            out = {"result": "ALLOW", "reason": "pre-approved at launch (omni.ps1 -ApproveAtLaunch): " + resp.get("reason", "")}
            if resp.get("state_updates"):
                out["state_updates"] = resp["state_updates"]
            return out
        return resp

    def evaluate(event: dict):
        resp = _evaluate(event)
        return _ask(resp) if resp and resp.get("result") == "ASK" else resp

    def _evaluate(event: dict):
        if event.get("type") != "tool_call":
            return None
        data = event.get("data") or {}
        name = _bare(str(data.get("name") or data.get("tool") or ""))
        args = data.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        state = event.get("session_state") or {}
        if name == "run_embodied_experiment":
            # session_state is per Omnigent session (each runner sub-agent session has its own), so
            # also count embodied results already in the active research record (cap per research run).
            n = max(int(state.get(key, 0) or 0), _embodied_runs_in_record())
            if n >= max_embodied_runs:
                return {"result": "DENY", "reason": f"Embodied-run cap reached ({n}/{max_embodied_runs} per session or research run). "
                                                    "Analyse existing results or ask the PI to raise the cap."}
            # increment (not set n+1): ASK defers the write until approval, and parallel pending
            # approvals must not overwrite each other (same reasoning as omnigent spawn_bounds).
            upd = [{"key": key, "action": "increment", "value": 1}]
            if name in ask:
                return {"result": "ASK", "state_updates": upd,
                        "reason": f"Embodied simulation #{n + 1} (brain + physics + video): "
                                  f"excite={args.get('excite_groups')} silence={args.get('silence_groups')}. Approve?"}
            return {"result": "ALLOW", "state_updates": upd}
        if name == "run_brain_experiment":
            try:
                cost = float(args.get("duration_ms", 1000)) * float(args.get("n_trials", 3))
            except (TypeError, ValueError):
                cost = 0.0
            if cost > brain_ms_threshold or name in ask:
                return {"result": "ASK", "reason": f"Long brain simulation ({cost:.0f} simulated ms in total, "
                                                  f"excite={args.get('excite_groups')}). Approve?"}
            return None
        if name == "run_brain_screen":
            cands = _as_list(args.get("candidates"))
            if len(cands) > max_screen_candidates:
                return {"result": "DENY", "reason": f"Screen of {len(cands)} candidates exceeds the cap of "
                                                    f"{max_screen_candidates} per call. Use rank_candidates and screen "
                                                    "the connectome-guided top-k, or split the screen."}
            try:
                cost = len(cands) * float(args.get("duration_ms", 500)) * float(args.get("n_trials", 2))
            except (TypeError, ValueError):
                cost = 0.0
            if cost > brain_ms_threshold or name in ask:
                return {"result": "ASK", "reason": f"Brain screen of {len(cands)} candidate types ({cost:.0f} simulated "
                                                  f"ms in total, targets={args.get('target_groups')}). Approve?"}
            return None
        if name in ask:
            if name == "request_approval":
                return {"result": "ASK", "reason": f"Lab approval request: {args.get('action')} | reason: "
                                                  f"{args.get('reason')} | est. cost units: {args.get('est_cost')}"}
            return {"result": "ASK", "reason": f"{name} requires PI approval. Approve?"}
        return None
    return evaluate


_LOOP_IGNORE_DEFAULT = ("sys_read_inbox", "sys_session_status", "sys_list_sessions", "get_record", "list_ground_truth")


def loop_guard(window: int = 10, threshold: int = 3, ignore_tools: list = None):
    """Omnigent policy factory: Omnigent's ``detect_loop`` (ASK when the same tool call repeats ``threshold``
    times in ``window`` calls), except for idempotent polling/read tools. Needed because the PI calls
    ``sys_read_inbox()`` with identical (empty) arguments after every specialist; stock ``detect_loop`` then
    ASKs on the 3rd hand-off and the lab stalls waiting for a human (observed live 2026-10-04 00:48).
    """
    from omnigent.policies.builtins.safety import detect_loop
    inner = detect_loop(window=window, threshold=threshold)
    ignore = set(ignore_tools or _LOOP_IGNORE_DEFAULT)

    def evaluate(event: dict):
        if event.get("type") == "tool_call":
            data = event.get("data") or {}
            if _bare(str(data.get("name") or data.get("tool") or "")) in ignore:
                return None
        return inner(event)
    return evaluate


# =========================================================================== self-test


ALL_TOOLS = [start_run, get_record, log_note, search_literature, list_neuron_groups, lookup_neurons,
             list_ground_truth, log_hypothesis, estimate_cost, log_experiment_plan, log_decision,
             request_approval, run_brain_experiment, run_body_experiment, run_embodied_experiment,
             compare_to_ground_truth, rank_candidates, run_brain_screen, get_benchmark]


def _selftest(keep: bool = False) -> int:
    """Exercise the full loop in mock mode (python -m flylab.tools --selftest [--keep])."""
    os.environ["FLYLAB_MOCK"] = "1"
    # Never touch the real runs/CURRENT: a live Omnigent session may be logging into it right now.
    real_current = record.CURRENT_FILE
    record.CURRENT_FILE = record.RUNS_DIR / ".selftest_CURRENT"
    prev_env = os.environ.pop("FLYLAB_RUN_ID", None)
    rid = None
    try:
        rid = start_run("SELFTEST (mock): which descending neurons drive backward walking?")["run_id"]
        steps = [
            search_literature("moonwalker descending neurons backward walking", n=2),
            list_neuron_groups(role="descending"),
            lookup_neurons("MDN"),
            list_ground_truth(),
            log_hypothesis("Activating MDN produces backward walking.", "activate", ["MDN"], "backward",
                           "MOCK rationale", 0.7),
            estimate_cost("embodied"),
            log_experiment_plan([{"id": "E1", "kind": "embodied", "excite_groups": ["MDN"]},
                                 {"id": "E2", "kind": "brain", "excite_groups": ["MDN"]}], "E1", "MOCK rationale"),
            request_approval("1 embodied run (MDN)", "tests H1", 2.5),
            run_brain_experiment(["MDN"]),
            run_body_experiment(0.0, 0.0, 1.0),
            run_embodied_experiment(["MDN"]),
            run_embodied_experiment(["P9_L"]),
            compare_to_ground_truth("backward", "gt_mock_mdn_activate"),
            compare_to_ground_truth("turn_right", "gt_mock_p9l_activate"),  # opposite laterality -> inconsistent
            log_decision("Surprise on P9_L: test P9_R next", "turn direction mismatch", "run P9_R", "laterality of P9"),
            get_record(last_n=5),
            rank_candidates("MDN", top_k=3),
            run_brain_screen(["MOCK_LC16", "MOCK_LC9"], ["MDN"]),
            get_benchmark(""),
            estimate_cost("screen", duration_ms=500, n_trials=2, n_candidates=326),
        ]
        bad = [s for s in steps if not s.get("ok", True)]
        for s in steps:
            json.dumps(s)  # must be JSON-serialisable
        assert steps[12]["verdict"] == "consistent", steps[12]
        assert steps[13]["verdict"] == "inconsistent" and steps[13]["surprise"] is True, steps[13]
        emb = [steps[10], steps[11]]  # the two embodied runs
        assert [Path(s["artifact"]).stem for s in emb] == ["embodied_01", "embodied_02"], [s["artifact"] for s in emb]
        pol = approval_gate()
        assert pol({"type": "tool_call", "data": {"name": "mcp__omnigent__run_embodied_experiment",
                                                   "arguments": {"excite_groups": ["MDN"]}}})["result"] == "ASK"
        assert pol({"type": "tool_call", "data": {"name": "search_literature", "arguments": {}}}) is None
        assert pol({"type": "tool_call", "session_state": {"_flylab_embodied_runs": 6},
                    "data": {"name": "run_embodied_experiment", "arguments": {}}})["result"] == "DENY"
        assert steps[17]["hits"] == ["MOCK_LC16"], steps[17]
        assert steps[17]["hits_by_target"] == {"MDN": ["MOCK_LC16"]}, steps[17]
        # experiment_ref check: embodied_01 activated MDN -> matches gt MDN-activate, not gt P9-activate
        c_ok = compare_to_ground_truth(steps[10]["behavior"], "gt_mock_mdn_activate", "embodied_01.json")
        assert c_ok["manipulation_match"] is True and c_ok["label_source"] == "body_classifier", c_ok
        assert c_ok["by_construction"] is True, c_ok  # MDN -> backward is wired into the bridge
        c_bad = compare_to_ground_truth("forward", "gt_mock_p9_activate", "embodied_01.json")
        assert c_bad["manipulation_match"] is False and c_bad["verdict"] == "inconclusive", c_bad
        assert c_bad["label_source"] == "agent_override" and c_bad["surprise"] is False, c_bad
        assert steps[19]["denied_by_policy"] is True and steps[19]["cost_units"] > 50, steps[19]
        big = {"type": "tool_call", "data": {"name": "run_brain_screen",
                                             "arguments": {"candidates": [f"T{i}" for i in range(41)], "target_groups": ["MDN"]}}}
        assert pol(big)["result"] == "DENY"
        mid = {"type": "tool_call", "data": {"name": "run_brain_screen",
                                             "arguments": {"candidates": [f"T{i}" for i in range(12)], "target_groups": ["MDN"]}}}
        assert pol(mid)["result"] == "ASK"
        os.environ["FLYLAB_PREAPPROVE"] = '{"by": "selftest"}'
        try:
            assert pol(mid)["result"] == "ALLOW" and pol(big)["result"] == "DENY"
            assert request_approval("x", "y", 1.0)["status"] == "pre-approved"
        finally:
            os.environ.pop("FLYLAB_PREAPPROVE", None)
        print(json.dumps(record.summarize(rid), indent=1))
        print("failed steps:", [b.get("tool") for b in bad])
        return 1 if bad else 0
    finally:
        record.CURRENT_FILE.unlink(missing_ok=True)
        record.CURRENT_FILE = real_current
        os.environ.pop("FLYLAB_RUN_ID", None)
        if prev_env:
            os.environ["FLYLAB_RUN_ID"] = prev_env
        if rid and not keep:  # do not leave mock runs in runs/ (it may be committed)
            import shutil
            shutil.rmtree(record.RUNS_DIR / rid, ignore_errors=True)
            print("selftest run removed (use --keep to keep it); runs/CURRENT untouched")
        elif rid:
            print(f"kept runs/{rid} (runs/CURRENT untouched)")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(_selftest(keep="--keep" in sys.argv))
    print(__doc__)
