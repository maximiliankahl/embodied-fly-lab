"""Export the Omnigent transcript of a lab session (supervisor + every sub-agent session) next to its
research record.

    uv run python agents/export_transcript.py <conv_id> [--run <run_id>] [--server http://127.0.0.1:6767]

Writes runs/<run_id>/omnigent/<conv_id>.jsonl for the supervisor and each sub-agent session (same
JSONL format as `omnigent session export`: line 1 = session_meta, then one line per item) plus
runs/<run_id>/omnigent/sessions.json (tree: id, parent, agent, title, item count, usage if reported).
<run_id> defaults to runs/CURRENT. Needs the local Omnigent server running (omni.ps1 starts it).
Start it through agents/omni.ps1 (sets OMNIGENT_CONFIG_HOME) or with the same env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = {"Origin": "omnigent-internal"}


def _get(c: httpx.Client, url: str, **params) -> dict:
    r = c.get(url, params=params or None)
    r.raise_for_status()
    return r.json()


def _items(c: httpx.Client, base: str, sid: str) -> list[dict]:
    out, after = [], None
    for _ in range(200):
        params = {"limit": 100, "order": "asc"}
        if after:
            params["after"] = after
        body = _get(c, f"{base}/v1/sessions/{sid}/items", **params)
        data = body.get("data") or []
        out.extend(data)
        if not body.get("has_more") or not data:
            break
        after = body.get("last_id") or data[-1].get("id")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("conv_id")
    ap.add_argument("--run", default="")
    ap.add_argument("--server", default="http://127.0.0.1:6767")
    a = ap.parse_args(argv)
    run_id = a.run or (ROOT / "runs" / "CURRENT").read_text(encoding="utf-8").strip()
    out_dir = ROOT / "runs" / run_id / "omnigent"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = a.server.rstrip("/")
    tree = []
    with httpx.Client(headers=ORIGIN, timeout=60.0, trust_env=False) as c:
        todo = [(a.conv_id, None)]
        seen = set()
        while todo:
            sid, parent = todo.pop(0)
            if sid in seen:
                continue
            seen.add(sid)
            meta = _get(c, f"{base}/v1/sessions/{sid}")
            items = _items(c, base, sid)
            with (out_dir / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"record_type": "session_meta", **meta}, default=str) + "\n")
                for it in items:
                    f.write(json.dumps({"record_type": "item", **it}, default=str) + "\n")
            children = (_get(c, f"{base}/v1/sessions/{sid}/child_sessions", limit=200).get("data") or [])
            tree.append({"id": sid, "parent": parent, "agent": meta.get("agent_name") or meta.get("title"),
                         "title": meta.get("title"), "n_items": len(items), "model": meta.get("llm_model"),
                         "total_cost_usd": meta.get("total_cost_usd"), "usage_by_model": meta.get("usage_by_model"),
                         "n_children": len(children)})
            todo.extend((ch.get("id"), sid) for ch in children if ch.get("id"))
    root_cost = tree[0].get("total_cost_usd") if tree else None  # root session cost includes its sub-agents
    (out_dir / "sessions.json").write_text(json.dumps({"root": a.conv_id, "run_id": run_id,
                                                      "root_total_cost_usd": root_cost, "sessions": tree},
                                                      indent=1, default=str), encoding="utf-8")
    print(f"exported {len(tree)} sessions -> {out_dir.relative_to(ROOT)} (root total_cost_usd={root_cost})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
