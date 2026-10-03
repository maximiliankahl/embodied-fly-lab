"""Reviewer check: every ground-truth evidence string is verbatim in the DOI's abstract (or the cached PMC full text)."""
import json, re, html, sys
from pathlib import Path
from flylab import literature
ROOT = Path(__file__).resolve().parents[3]
gt = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))
items = gt["entries"]
ft = {}
for f in (ROOT / "spikes" / "screen" / "out").glob("PMC*.xml"):
    t = re.sub(r"<[^>]+>", " ", f.read_text(encoding="utf-8"))
    ft[f.stem] = html.unescape(re.sub(r"\s+", " ", t))
def norm(s): return re.sub(r"\s+", " ", s or "").strip().lower()
for e in items:
    if not str(e.get("id", "")).startswith(("gt19", "gt2")) and "--all" not in sys.argv:
        continue
    e["doi"] = e["citation"]["doi"]; rec = literature.get_by_doi(e["doi"])
    ab = norm(rec["abstract"]) if rec else ""
    ev = norm(e.get("evidence"))
    print(e["id"], e["doi"], "| title:", (rec or {}).get("title", "")[:90], "| year", (rec or {}).get("year"))
    print("   evidence in abstract:", ev in ab, "| words:", len(ev.split()))
    eft = e.get("evidence_fulltext")
    if eft:
        hit = [k for k, t in ft.items() if norm(eft) in norm(t)]
        print("   fulltext in cached PMC xml:", hit, "| words:", len(eft.split()))
    for k in ("note",):
        if e.get(k):
            for q in re.findall(r"'([^']{15,})'", e[k]):
                print(f"   note quote in abstract={norm(q) in ab} fulltext={[k2 for k2,t in ft.items() if norm(q) in norm(t)]}: {q[:70]}")
