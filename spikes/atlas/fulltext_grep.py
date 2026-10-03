"""Fetch Europe PMC full text (OA only) for a DOI and grep sentences. Scratch tool for curating ground truth."""
import re, sys, json, requests
from pathlib import Path
OUT = Path("spikes/atlas/out"); OUT.mkdir(parents=True, exist_ok=True)
def fulltext(doi):
    f = OUT / ("ft_" + re.sub(r"\W", "_", doi) + ".txt")
    if f.exists():
        return f.read_text(encoding="utf-8")
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"}, timeout=20).json()
    pmcs = [x.get("pmcid") for x in r["resultList"]["result"] if x.get("pmcid")]
    if not pmcs:
        return ""
    t = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcs[0]}/fullTextXML", timeout=40).text
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))
    f.write_text(t, encoding="utf-8")
    return t
doi, pat = sys.argv[1], sys.argv[2]
t = fulltext(doi)
print(doi, "chars:", len(t))
seen = set()
for s in re.split(r"(?<=[.!?])\s+", t):
    if re.search(pat, s) and s not in seen:
        seen.add(s); print("-", s[:350])
