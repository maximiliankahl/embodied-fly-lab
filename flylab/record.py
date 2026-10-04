"""Shared research record for the Embodied Fly Lab.

Every agent action that matters scientifically (question, evidence, hypothesis,
experiment options/choice, approval, result, analysis, decision) is appended as
one JSON line to ``runs/<run_id>/record.jsonl``. Artifacts (videos, plots, raw
results) live in ``runs/<run_id>/artifacts/``. The Streamlit dashboard reads
these files to show the agent hand-offs.

Contract (docs/CONTRACTS.md):
    new_run(question) -> run_id
    log_event(run_id, agent, type, content, data=None, citations=None) -> dict
    load(run_id) -> list[dict]
    list_runs() -> list[str]

CLI:
    uv run python -m flylab.record --list
    uv run python -m flylab.record --show <run_id>
"""

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
CURRENT_FILE = RUNS_DIR / "CURRENT"  # pointer to the active run (plain text run_id)

EVENT_TYPES = (
    "question",
    "evidence",
    "hypothesis",
    "experiment_options",
    "experiment_choice",
    "approval",
    "experiment_result",
    "analysis",
    "decision",
    "note",
    "movement_verification",  # flylab.verify via tools.verify_movement (Phase 3)
)

_LOCK = threading.Lock()
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


# --------------------------------------------------------------------------- paths


def _check_run_id(run_id: str) -> str:
    run_id = str(run_id).strip()
    if not _RUN_ID_RE.match(run_id) or run_id in {".", ".."}:
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


def run_dir(run_id: str) -> Path:
    """Directory of one run (created on demand)."""
    d = RUNS_DIR / _check_run_id(run_id)
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(run_id: str) -> Path:
    """``runs/<run_id>/artifacts`` (created on demand)."""
    return run_dir(run_id) / "artifacts"


def record_path(run_id: str) -> Path:
    return run_dir(run_id) / "record.jsonl"


def _slug(text: str, n: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-")) or "run"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serialisable data (numpy, Paths, sets...)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    for attr in ("item", "tolist"):  # numpy scalars / arrays
        if hasattr(obj, attr):
            try:
                return _jsonable(getattr(obj, attr)())
            except Exception:
                pass
    return str(obj)


# --------------------------------------------------------------------------- API


def new_run(question: str, set_current: bool = True) -> str:
    """Create a new run folder, log the research question, return the run_id.

    run_id format: ``YYYYMMDD-HHMMSS-<slug>-<4hex>`` (local time).
    When ``set_current`` the run becomes the active run (``runs/CURRENT`` and
    the ``FLYLAB_RUN_ID`` env var of this process).
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{_slug(question)}-{secrets.token_hex(2)}"
    run_dir(run_id)
    log_event(run_id, "human", "question", question)
    if set_current:
        set_current_run(run_id)
    return run_id


def log_event(
    run_id: str,
    agent: str,
    type: str,  # noqa: A002 - name fixed by contract
    content: str,
    data: dict | list | None = None,
    citations: list | None = None,
) -> dict:
    """Append one event to ``runs/<run_id>/record.jsonl`` and return it.

    ``type`` should be one of EVENT_TYPES; unknown types are stored as given
    but flagged with ``"nonstandard_type": true`` so the dashboard can still
    show them.
    """
    path = record_path(run_id)
    event: dict[str, Any] = {
        "ts": _now(),
        "run_id": run_id,
        "agent": str(agent or "unknown"),
        "type": str(type),
        "content": str(content),
        "data": _jsonable(data) if data is not None else None,
        "citations": _jsonable(citations) if citations else [],
    }
    if event["type"] not in EVENT_TYPES:
        event["nonstandard_type"] = True
    with _LOCK:
        seq = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                seq = sum(1 for line in f if line.strip())
        event["seq"] = seq
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def load(run_id: str) -> list[dict]:
    """All events of a run, in order. Malformed lines are skipped."""
    path = RUNS_DIR / _check_run_id(run_id) / "record.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def list_runs() -> list[str]:
    """Run ids that have a record, newest first."""
    if not RUNS_DIR.exists():
        return []
    runs = [p.parent.name for p in RUNS_DIR.glob("*/record.jsonl")]
    return sorted(runs, reverse=True)


# --------------------------------------------------------------------------- active run


def set_current_run(run_id: str) -> None:
    """Mark ``run_id`` as the active run for tools that get no explicit run_id."""
    run_id = _check_run_id(run_id)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_FILE.write_text(run_id, encoding="utf-8")
    os.environ["FLYLAB_RUN_ID"] = run_id


def current_run_id() -> str | None:
    """Active run: env FLYLAB_RUN_ID, else ``runs/CURRENT``, else None."""
    env = os.environ.get("FLYLAB_RUN_ID", "").strip()
    if env:
        return _check_run_id(env)
    if CURRENT_FILE.exists():
        rid = CURRENT_FILE.read_text(encoding="utf-8").strip()
        if rid and _RUN_ID_RE.match(rid):
            return rid
    return None


def resolve_run_id(run_id: str | None = None, question_if_new: str = "ad-hoc session") -> str:
    """Explicit run_id > FLYLAB_RUN_ID > runs/CURRENT > a freshly created run."""
    if run_id and str(run_id).strip():
        rid = _check_run_id(run_id)
        run_dir(rid)
        return rid
    rid = current_run_id()
    if rid:
        run_dir(rid)
        return rid
    return new_run(question_if_new)


def summarize(run_id: str) -> dict:
    """Compact summary: counts per type and per agent, plus hand-off sequence."""
    events = load(run_id)
    by_type: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    handoffs: list[str] = []
    last_agent = None
    for e in events:
        by_type[e.get("type", "?")] = by_type.get(e.get("type", "?"), 0) + 1
        a = e.get("agent", "?")
        by_agent[a] = by_agent.get(a, 0) + 1
        if a != last_agent:
            handoffs.append(a)
            last_agent = a
    return {"run_id": run_id, "n_events": len(events), "by_type": by_type,
            "by_agent": by_agent, "handoffs": handoffs}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Embodied Fly Lab research record")
    ap.add_argument("--list", action="store_true", help="list runs")
    ap.add_argument("--show", metavar="RUN_ID", help="print the events of a run")
    ap.add_argument("--selftest", action="store_true", help="write + read a test run")
    args = ap.parse_args()
    if args.list:
        for r in list_runs():
            print(r)
    elif args.show:
        for e in load(args.show):
            print(f"[{e['seq']:03d}] {e['ts']} {e['agent']:>12} {e['type']:<18} {e['content'][:110]}")
    elif args.selftest:
        rid = new_run("selftest: does record.py work?", set_current=False)
        log_event(rid, "hypothesis", "hypothesis", "test hypothesis", {"x": 1}, ["test-citation"])
        ev = load(rid)
        assert len(ev) == 2 and ev[1]["seq"] == 1, ev
        print("ok", rid, summarize(rid))
        import shutil

        shutil.rmtree(RUNS_DIR / rid, ignore_errors=True)  # keep runs/ clean (it is committed)
    else:
        ap.print_help()
