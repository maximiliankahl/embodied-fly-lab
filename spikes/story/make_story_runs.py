"""Real simulation runs for the demo-video story (web/?story=1).

1. NEGATIVE CONTROL: adapter v0 deliberately mis-wired (steering sign flipped = 'contralateral' hypothesis).
   DNa02_L activation -> the body turns RIGHT; the movement verifier must flag it as incorrect
   (published: ipsilateral turning, Rayshubskiy et al. 2025, gt05). The frozen bridge is restored afterwards.
2. Sugar GRN activation -> feeding motor neuron MN9 (Shiu et al. 2024) for the closing brain shot.
Both are real brain -> adapter -> body simulations; nothing is animated by hand.
"""
import json
from pathlib import Path

from flylab import bridge, export3d

RUNS = Path(__file__).resolve().parents[2] / "web" / "data" / "runs"

_orig = bridge.rates_to_drive


def _miswired(rates, *a, **k):
    d = dict(_orig(rates, *a, **k))
    d["turn"] = -float(d.get("turn", 0.0))
    ex = dict(d.get("explain") or {})
    ex["MISWIRED"] = "adapter v0: steering sign flipped (contralateral hypothesis) - negative control, not the frozen bridge"
    d["explain"] = ex
    return d


bridge.rates_to_drive = _miswired
try:
    p = export3d.export_run(
        "story_miswired_dna02l", ["DNa02_L"], mode="walk", expected="turn_left",
        gt="gt05_dna02L_activate_turn_left", kind="negative_control",
        title="Adapter v0 (mis-wired on purpose): DNa02 left -> fly turns the WRONG way")
finally:
    bridge.rates_to_drive = _orig

run = json.loads(Path(p).read_text(encoding="utf-8"))
run["bridge"]["frozen"] = ("NOT the frozen bridge: adapter v0 with the steering sign flipped on purpose "
                           "(negative control for the movement verifier). The frozen bridge maps DNa02 to ipsilateral turning.")
run["story_note"] = ("Negative control: shows that the movement verifier catches a wrongly wired adapter. "
                     "After the verifier flags it, the frozen (ipsilateral) bridge is used again (run dna02l_turn_left).")
Path(p).write_text(json.dumps(run, separators=(",", ":")), encoding="utf-8")
print("miswired:", run["body"]["metrics"].get("behavior"), "| verifier:", (run.get("verifier") or {}).get("final_verdict"))

p2 = export3d.export_run(
    "story_sugar_feeding", ["sugar_GRN_shiu2024"], mode="walk", expected="stop",
    gt="gt10_sugar_activate_feed", kind="validation",
    title="Sugar taste neurons -> feeding motor neuron MN9 fires (brain read-out)")
run2 = json.loads(Path(p2).read_text(encoding="utf-8"))
print("sugar:", run2["brain"]["readouts"].get("feed"), "| behavior:", run2["body"]["metrics"].get("behavior"))
print(export3d.write_index())
