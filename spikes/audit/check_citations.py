"""Audit: do the DOIs cited in the repo resolve (flylab.literature.get_by_doi: Europe PMC / OpenAlex, free, cached)?

  uv run python spikes/audit/check_citations.py            # writes spikes/audit/citations.json (committed)

Sources scanned: data/ground_truth.json, data/neurons.json, the live-run records under runs/, README.md,
docs/*.md, data/benchmarks/*.json. A DOI "resolves" if get_by_doi returns an item with a title.
For ground-truth entries the stored evidence quote is also checked against the abstract (it must appear verbatim,
ignoring case / whitespace) when the entry says evidence_in_abstract = true.
No paid API is called.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flylab import literature  # noqa: E402

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>\]\)\\,;]+", re.I)


def clean(d: str) -> str:
    return d.rstrip(".*_`").lower()


def scan_text(txt: str) -> set[str]:
    return {clean(m.group(0)) for m in DOI_RE.finditer(txt)}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip().lower()


def main() -> None:
    sources: dict[str, set[str]] = {}
    gt = json.loads((ROOT / "data/ground_truth.json").read_text(encoding="utf-8"))["entries"]
    sources["data/ground_truth.json"] = {clean(e["citation"]["doi"]) for e in gt if e.get("citation", {}).get("doi")}
    sources["data/neurons.json"] = scan_text((ROOT / "data/neurons.json").read_text(encoding="utf-8"))
    for run in sorted((ROOT / "runs").glob("2026*")):
        sources[f"{run.relative_to(ROOT).as_posix()}/record.jsonl"] = scan_text((run / "record.jsonl").read_text(encoding="utf-8"))
    for f in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        sources[f.relative_to(ROOT).as_posix()] = scan_text(f.read_text(encoding="utf-8"))
    for f in sorted((ROOT / "data/benchmarks").glob("*.json")):
        sources[f.relative_to(ROOT).as_posix()] = scan_text(f.read_text(encoding="utf-8"))

    all_dois = sorted(set().union(*sources.values()))
    res = {}
    for d in all_dois:
        try:
            item = literature.get_by_doi(d)
        except Exception as e:  # noqa: BLE001
            item = None
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
        res[d] = {"resolved": bool(item and item.get("title")), "title": (item or {}).get("title"), "year": (item or {}).get("year"),
                  "source": (item or {}).get("source"), "error": err,
                  "cited_in": sorted(s for s, v in sources.items() if d in v)}
    # evidence quotes of ground truth vs abstract
    quotes = []
    for e in gt:
        doi = clean(e.get("citation", {}).get("doi", ""))
        if not doi or not e.get("evidence_in_abstract"):
            continue
        item = literature.get_by_doi(doi) or {}
        found = norm(e.get("evidence", "")) in norm(item.get("abstract", ""))
        quotes.append({"id": e["id"], "doi": doi, "quote_in_abstract": found})

    n_ok = sum(v["resolved"] for v in res.values())
    out = {"n_dois": len(res), "n_resolved": n_ok, "unresolved": [d for d, v in res.items() if not v["resolved"]],
           "ground_truth": {"n_entries": len(gt), "n_unique_dois": len(sources["data/ground_truth.json"]),
                            "n_quote_checked": len(quotes), "n_quote_in_abstract": sum(q["quote_in_abstract"] for q in quotes),
                            "quote_not_found": [q["id"] for q in quotes if not q["quote_in_abstract"]]},
           "dois": res, "quotes": quotes}
    dest = ROOT / "spikes/audit/citations.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"DOIs found: {len(res)}, resolved via get_by_doi: {n_ok}, unresolved: {out['unresolved']}")
    print("ground truth:", out["ground_truth"])
    for s in sources:
        print(f"  {s}: {len(sources[s])} DOIs, unresolved {[d for d in sources[s] if not res[d]['resolved']]}")


if __name__ == "__main__":
    main()
