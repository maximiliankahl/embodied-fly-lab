"""Fetch Europe PMC full text for a DOI and print sentences matching a regex (for verbatim evidence quotes)."""
import json, re, sys, html
from pathlib import Path
import requests
OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
doi, pat = sys.argv[1], sys.argv[2]
r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                 params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"}, timeout=30).json()
res = r["resultList"]["result"]
pmcid = next((x.get("pmcid") for x in res if x.get("pmcid")), None)
print("pmcid", pmcid)
if not pmcid:
    sys.exit()
f = OUT / f"{pmcid}.xml"
if not f.exists():
    x = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML", timeout=60)
    print("status", x.status_code)
    if x.status_code != 200:
        sys.exit()
    f.write_text(x.text, encoding="utf-8")
txt = re.sub(r"<[^>]+>", " ", f.read_text(encoding="utf-8"))
txt = html.unescape(re.sub(r"\s+", " ", txt))
sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", txt)
for s in sents:
    if re.search(pat, s):
        print("-", s[:400])
