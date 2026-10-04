"""Knowledge base export (sources / claims / runs as JSONL) in the FlyBrainLab schema.

Schema adopted from the teammate repo JonasMayerDev/FlyBrainLab (scripts/kb_store.py):
- sources: url, title, source_kind, retrieval_status (candidate | metadata_only | retrieved), ...
- claims:  text, claim_kind (reported_finding | extraction | hypothesis | model_assumption), evidence_type,
           source_location, conditions, dataset_version, review_status (unreviewed | checked | rejected), created_by, source_ids
- runs:    run_kind, status (planned | completed | failed | fixture), agent, model_version, dataset_version, claim_ids, source_ids

Built from what this repo already verified: data/ground_truth.json (DOIs resolved, quotes checked verbatim against the
abstract), the frozen bridge (model assumptions), agent hypotheses in runs/*/record.jsonl (unreviewed, agent-generated)
and the benchmark files. Run:  uv run python -m flylab.knowledge   ->  data/knowledge/*.jsonl
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "knowledge"
DATASET = "FlyWire FAFB v783 (Shiu et al. 2024 model files)"


def _id(prefix: str, text: str) -> str:
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def build() -> dict:
    gt = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))
    entries = gt["entries"] if isinstance(gt, dict) else gt
    sources, claims, runs = {}, [], []

    for e in entries:
        c = e.get("citation") or {}
        doi = c.get("doi")
        if not doi:
            continue
        sid = _id("src", doi)
        sources.setdefault(sid, {
            "source_id": sid, "url": f"https://doi.org/{doi}", "title": c.get("title"), "year": c.get("year"),
            "authors": c.get("authors"), "source_kind": "peer_reviewed_article", "retrieval_status": "retrieved",
            "retrieved_part": "abstract (Europe PMC / OpenAlex, cached locally in data/cache)",
            "verified_by": c.get("verified_by")})
        claims.append({
            "claim_id": _id("clm", e["id"]), "ground_truth_id": e["id"],
            "text": f"{e['manipulation']} {e['target_group']} -> {e.get('effect', 'induce')} {e['expected_behavior']}",
            "claim_kind": "reported_finding",
            "evidence_type": "activation/silencing experiment" if e["manipulation"] in ("activate", "silence") else "reported finding",
            "source_location": f"abstract: \"{e.get('evidence', '')}\"" + (" (+ full-text quote)" if e.get("evidence_fulltext") else ""),
            "conditions": e.get("context") or "Drosophila melanogaster, conditions as in the cited paper",
            "dataset_version": DATASET, "neuron_group": e["target_group"],
            "review_status": "checked" if e.get("evidence_in_abstract") else "unreviewed",
            "confidence": e.get("confidence"), "limitations": e.get("note"),
            "created_by": "atlas.build (DOI resolved + verbatim abstract check)", "source_ids": [sid]})

    try:
        from flylab import bridge
        desc = bridge.describe()
        for term in desc.get("drive_terms", []) or []:
            txt = json.dumps(term, ensure_ascii=False)
            claims.append({
                "claim_id": _id("asm", txt), "text": f"Bridge term: {term.get('output', term)} <- {term.get('groups', '')}",
                "claim_kind": "model_assumption", "evidence_type": "hand-designed adapter (ventral nerve cord not simulated)",
                "source_location": "flylab/bridge.py describe()", "conditions": "frozen before validation runs",
                "dataset_version": DATASET, "review_status": "checked", "created_by": "flylab.bridge", "source_ids": [],
                "detail": term})
    except Exception as exc:  # noqa: BLE001
        claims.append({"claim_id": "asm_unavailable", "text": f"bridge.describe() unavailable: {exc}",
                       "claim_kind": "model_assumption", "evidence_type": "n/a", "source_location": "n/a",
                       "conditions": "n/a", "dataset_version": DATASET, "review_status": "unreviewed",
                       "created_by": "flylab.knowledge", "source_ids": []})

    for rec in sorted((ROOT / "runs").glob("*/record.jsonl")):
        for line in rec.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "hypothesis":
                claims.append({
                    "claim_id": _id("hyp", rec.parent.name + str(ev.get("seq"))), "text": str(ev.get("content"))[:600],
                    "claim_kind": "hypothesis", "evidence_type": "agent-generated hypothesis",
                    "source_location": f"runs/{rec.parent.name}/record.jsonl seq {ev.get('seq')}",
                    "conditions": "in silico", "dataset_version": DATASET, "review_status": "unreviewed",
                    "created_by": f"agent:{ev.get('agent')}", "source_ids": [_id('src', d) for d in (ev.get('citations') or []) if _id('src', d) in sources]})
        runs.append({"run_id": rec.parent.name, "run_kind": "omnigent_discovery_loop", "status": "completed",
                     "agent": "omnigent:fly_lab", "model_version": "agents/fly_lab.yaml", "dataset_version": DATASET,
                     "record": f"runs/{rec.parent.name}/record.jsonl", "claim_ids": [], "source_ids": []})

    for f in sorted((ROOT / "data" / "benchmarks").glob("*.json")):
        runs.append({"run_id": f.stem, "run_kind": "benchmark", "status": "completed", "agent": "flylab",
                     "model_version": "flylab (git)", "dataset_version": DATASET,
                     "record": f"data/benchmarks/{f.name}", "claim_ids": [], "source_ids": []})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sources", list(sources.values())), ("claims", claims), ("runs", runs)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"sources": len(sources), "claims": len(claims), "runs": len(runs), "dir": str(OUT)}


if __name__ == "__main__":
    print(build())
