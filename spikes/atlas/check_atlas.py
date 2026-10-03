"""Sanity checks for the atlas: brain-model membership, evidence_fulltext verbatim in Europe PMC full text, API smoke test."""
import re, json, requests
from pathlib import Path
from flylab import atlas
OUT = Path("spikes/atlas/out")

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

norm = lambda s: re.sub(r"\s+", " ", s.replace("&gt;", ">").replace("&lt;", "<")).lower()
for e in atlas.ground_truth():
    assert e["evidence_in_abstract"], e["id"]
    if "evidence_fulltext" in e:
        ft = fulltext(e["citation"]["doi"])
        print(e["id"], "fulltext verbatim:", norm(e["evidence_fulltext"]) in norm(ft), "(chars", len(ft), ")")
gs = atlas.groups()
bad = {k: g.get("excluded_not_in_brain_model") for k, g in gs.items() if g.get("excluded_not_in_brain_model")}; assert all("brain_model_check" in g for g in gs.values())
print("groups with excluded IDs (not in brain model):", {k: len(v) for k, v in bad.items()})
print("group_ids('mdn_l') =", atlas.group_ids("mdn_l"))
print("find DNa02:", [(r["cell_type"], r["side"], r["n"], r["match"]) for r in atlas.find_cell_type("DNa02")])
print("find giant fiber:", [(r["cell_type"], r["side"], r["root_ids"]) for r in atlas.find_cell_type("giant fiber")])
print("neuron_info:", atlas.neuron_info(720575940660219265))
try:
    atlas.group_ids("moonwalker")
except KeyError as ex:
    print("KeyError ok:", ex)
json.dumps(atlas.groups()); json.dumps(atlas.ground_truth()); print("JSON-serialisable: ok")
