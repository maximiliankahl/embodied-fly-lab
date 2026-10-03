"""Fast checks for flylab.bridge (no brain/body simulation). Run: uv run python spikes/bridge/test_bridge.py"""
import json, math, sys
sys.path.insert(0, ".")
from flylab import atlas, bridge

def neuron_rates(group_hz: dict) -> dict:
    out = {}
    for g, hz in group_hz.items():
        for i in atlas.group_ids(g):
            out[str(i)] = hz
    return out

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

d = bridge.rates_to_drive({})
check("empty -> zero drive", d["forward"] == d["turn"] == d["backward"] == 0.0)
d = bridge.rates_to_drive({"MDN": 150.0})
check("MDN group -> backward 1", d["backward"] == 1.0 and d["forward"] == 0 and d["turn"] == 0)
d2 = bridge.rates_to_drive(neuron_rates({"MDN": 150.0}))
check("MDN neuron mode == group mode", d2["backward"] == d["backward"] and d2["explain"]["input_mode"] == "neuron_rates")
d = bridge.rates_to_drive({"P9": 148.3})
check("P9 -> forward 1, no turn", abs(d["forward"] - 1) < 1e-3 and d["turn"] == 0)
d = bridge.rates_to_drive({"P9_L": 148.3})
check("P9_L -> forward 0.5, turn < 0 (left, ipsilateral)", abs(d["forward"] - 0.5) < 1e-3 and d["turn"] < 0)
d = bridge.rates_to_drive({"DNa02_L": 148.3})
check("DNa02_L -> turn -1 (left)", d["turn"] == -1.0)
d = bridge.rates_to_drive({"DNa02_R": 148.3})
check("DNa02_R -> turn +1 (right)", d["turn"] == 1.0)
d = bridge.rates_to_drive({"DNa01_R": 74.15})
check("DNa01_R half rate -> turn +0.25", abs(d["turn"] - 0.25) < 1e-3)
d = bridge.rates_to_drive({"moonwalker": 74.15})
check("alias moonwalker -> backward 0.5", abs(d["backward"] - 0.5) < 1e-3)
d = bridge.rates_to_drive({"MDN": float("nan"), "P9": -5, "DNa02_L": None, "bogus_group": 100})
check("NaN/neg/None/unknown -> zero drive, unknown listed",
      d["forward"] == d["backward"] == d["turn"] == 0 and "bogus_group" in d["explain"].get("ignored_keys", []))
d = bridge.rates_to_drive({"MDN": 500})
check("saturation at 1", d["backward"] == 1.0)
ro = bridge.brain_readouts({"GF": 3.0, "MN9_R": 0.5, "aDN2_shiu2024": 20})
check("readouts escape on, feed off (<1 Hz), groom on", ro["escape"]["active"] and not ro["feed"]["active"] and ro["groom"]["active"])
check("readout_label", bridge.readout_label(ro, "feed") == "no_feed" and bridge.readout_label(ro, "groom") == "groom")
desc = bridge.describe()
check("describe JSON-serialisable", bool(json.dumps(desc)))
# tools integration (real bridge, no sim): tools._drive_from_rates with per-neuron input
from flylab import tools
drv, info = tools._drive_from_rates(neuron_rates({"P9_R": 148.3}))
check("tools._drive_from_rates uses neuron_rates and turns right for P9_R", info.get("bridge_input") == "neuron_rates" and drv["turn"] > 0)
# body.classify (no sim): backward-first rule only for backward displacement
from flylab import body
bst = dict(forward_disp_mm=-10.1, lateral_disp_mm=2.5, heading_change_deg=-44.2, yaw_rate_deg_s=-44.2,
           net_speed_mm_s=10.4, sim_duration_s=1.0)
check("classify: backward with small curve -> backward", body.classify(bst) == "backward")
check("classify: same with dH -46 (old 45-deg knife edge) -> backward",
      body.classify({**bst, "heading_change_deg": -46.4, "yaw_rate_deg_s": -46.4}) == "backward")
fwd_arc = dict(forward_disp_mm=7.0, lateral_disp_mm=-2.0, heading_change_deg=-44, yaw_rate_deg_s=-88,
               net_speed_mm_s=14.6, sim_duration_s=0.5)
check("classify: forward 0.5 s arc at 88 deg/s -> turn_right (yaw rule, unchanged from Phase 1)",
      body.classify(fwd_arc) == "turn_right")
check("classify: stop", body.classify(dict(forward_disp_mm=0.03, lateral_disp_mm=0.0, heading_change_deg=0.2,
                                           yaw_rate_deg_s=0.2, net_speed_mm_s=0.03, sim_duration_s=1.0)) == "stop")
print("FAILS:", fails or "none")
sys.exit(1 if fails else 0)
