"""Reviewer checks: every DOI cited anywhere in neurons.json / ground_truth.json resolves; group IDs exist in annotation + brain model."""
import json, sys
import pandas as pd
from flylab import atlas, literature
sys.stdout.reconfigure(encoding="utf-8")
gs = atlas.groups()
dois = set()
for g in gs.values():
    dois.update(g["citations"])
dois.update(atlas._neurons_doc()["_meta"].get("annotation_citations", []))
for e in atlas.ground_truth():
    dois.add(e["citation"]["doi"])
for d in sorted(dois, key=str.lower):
    r = literature.get_by_doi(d)
    print(f"{d:34s} ->", (f"{r['year']} {r['authors'][:40]} | {r['title'][:90]} | {r['verified_by']}" if r else "NOT RESOLVED"))
df = atlas.load_annotations().set_index("root_id")
comp = set(pd.read_csv("data/raw/Completeness_783.csv").iloc[:, 0].astype("int64"))
for k, g in gs.items():
    ids = g["root_ids"]
    miss_a = [i for i in ids if i not in df.index]
    miss_c = [i for i in ids if i not in comp]
    dup = len(ids) - len(set(ids))
    sides = df.loc[[i for i in ids if i in df.index], "side"].value_counts().to_dict()
    if miss_a or miss_c or dup or g["n"] != len(ids):
        print("PROBLEM", k, len(miss_a), len(miss_c), dup)
    print(f"{k:22s} n={len(ids):4d} side={g['side']:4s} annot_sides={sides}")
