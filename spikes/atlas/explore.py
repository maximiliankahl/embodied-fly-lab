"""Exploration helper for the atlas (scratch). Usage: python spikes/atlas/explore.py <regex> [column]"""
import sys
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 70)
df = pd.read_csv("data/raw/flywire_annotations_Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False)
pat = sys.argv[1]
cols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cell_type", "hemibrain_type", "synonyms"]
mask = False
for c in cols:
    mask = mask | df[c].fillna("").astype(str).str.contains(pat, regex=True)
s = df[mask]
g = s.groupby(["super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side"], dropna=False).size().reset_index(name="n")
print(g.to_string(index=False))
syn = s.dropna(subset=["synonyms"]).drop_duplicates("cell_type")[["cell_type", "synonyms"]]
if len(syn):
    print(syn.to_string(index=False))
