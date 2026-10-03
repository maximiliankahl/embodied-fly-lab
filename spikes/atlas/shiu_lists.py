"""Extract neuron ID lists from Shiu et al. (Drosophila_brain_model) figures.ipynb and check against FlyWire v783 annotations."""
import json, re
import pandas as pd
nb = json.load(open("data/raw/figures.ipynb", encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
lists = {}
for m in re.finditer(r"(\w+)\s*=\s*\[([\s\d,#\w\-\(\)]*?)\]", src):
    ids = [int(x) for x in re.findall(r"\b7205759\d{11}\b", m.group(2))]
    if ids:
        lists.setdefault(m.group(1), ids)
for m in re.finditer(r"(\w+)\s*=\s*(7205759\d{11})", src):
    lists.setdefault(m.group(1), [int(m.group(2))])
df = pd.read_csv("data/raw/flywire_annotations_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False).set_index("root_id")
for k, v in lists.items():
    present = [i for i in v if i in df.index]
    types = df.loc[present].groupby(["cell_type", "cell_sub_class", "side"], dropna=False).size().to_dict() if present else {}
    print(k, len(v), "in783:", len(present), types)
json.dump(lists, open("spikes/atlas/out/shiu_lists.json", "w"), indent=1)
