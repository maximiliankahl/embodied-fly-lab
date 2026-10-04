"""Regression: walking bridge + atlas.evaluate unchanged vs git HEAD (additive flight changes only)."""
import importlib.util, json, subprocess, sys, tempfile, random
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from flylab import atlas, bridge

def load_head(name):
    src = subprocess.check_output(["git", "show", f"HEAD:flylab/{name}.py"], text=True, encoding="utf-8")
    root = Path(atlas.__file__).resolve().parents[1].as_posix()
    src = src.replace("Path(__file__).resolve().parents[1]", f"Path({root!r})")
    tmp = Path(tempfile.gettempdir()) / f"head_{name}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"head_{name}", tmp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

hb = load_head("bridge")
ha = load_head("atlas")
random.seed(1)
cases = [{}, {"P9": 150.0}, {"MDN": 100.0, "P9_L": 20.0}, {"DNa02_L": 80.0, "DNa01_R": 30.0, "P9": 50.0},
         {"GF": 160.0, "MDN_L": 12.0}, {"MN9": 5.0, "aDN1_shiu2024": 3.0}]
ids = [i for g in ("P9_L", "P9_R", "MDN_L", "MDN_R", "DNa02_L", "DNa02_R", "GF_L", "GF_R") for i in atlas.group_ids(g)]
for _ in range(5):
    cases.append({str(i): random.choice([0, 0, 5, 40, 150, 170]) for i in ids})
bad = 0
for c in cases:
    a = hb.rates_to_drive(c); b = bridge.rates_to_drive(c)
    ra = hb.brain_readouts(c); rb = bridge.brain_readouts(c)
    if a != b or ra != rb:
        bad += 1
        print("DIFF", list(c)[:3])
print("walking bridge cases", len(cases), "diffs", bad)
# describe(): old keys unchanged
da, db = hb.describe(), bridge.describe()
print("describe keys added:", sorted(set(db) - set(da)), "removed:", sorted(set(da) - set(db)),
      "old values equal:", all(da[k] == db[k] for k in da))
# evaluate on all old GT ids x all body labels
old_ids = [e["id"] for e in ha.ground_truth() if e["id"] <= "gt23_zzz"]
labels = ["forward", "backward", "turn_left", "turn_right", "stop", "escape", "no_escape", "groom", "feed", "no_feed"]
nd = 0
for gid in [e["id"] for e in ha.ground_truth() if int(e["id"][2:4]) <= 23]:
    for l in labels:
        x, y = ha.evaluate(l, gid), atlas.evaluate(l, gid)
        if x != y:
            nd += 1; print("EVAL DIFF", l, gid)
print("evaluate diffs on old labels x old GT:", nd)
