"""Brain -> body bridge: descending-neuron firing rates -> NeuroMechFly drive (+ brain-level readouts).

SCOPE / WHAT THIS IS NOT
------------------------
FlyWire v783 (and therefore the Shiu et al. 2024 LIF model in flylab.brain) covers the
*brain only*. In a real fly, descending neurons (DNs) project into the ventral nerve cord
(VNC), whose premotor and motor circuits turn DN activity into leg movements. That VNC is
NOT simulated here. This module REPLACES the VNC with a small, hand-designed, documented
interface: a handful of identified DN groups are read out, normalised, and combined into
the 3-number command that the body controller (flylab.body, flygym hybrid turning
controller) understands:

    drive = {"forward": 0..1, "turn": -1..1 (neg = LEFT), "backward": 0..1}

Every term, weight and threshold below is a modelling choice (literature-motivated, not
fitted to behavioural data). Behaviours the walking body cannot produce (escape takeoff,
grooming, feeding) are read out at brain level instead (brain_readouts()).

MAPPING (see describe() for the machine-readable table with citations)
-------
  a(g)      = min(1, rate_hz(g) / REF_RATE_HZ)                    activation of group g (0..1)
  forward   = mean(a(P9_L), a(P9_R))                              DNp09 "P9", Bidaye et al. 2020
  backward  = mean(a(MDN_L), a(MDN_R))                            moonwalker DNs, Bidaye et al. 2014
  turn      = clip( 1.0 * (a(DNa02_R) - a(DNa02_L))               steering DNs, Rayshubskiy et al. 2025
                  + 0.5 * (a(DNa01_R) - a(DNa01_L))               (DNa02 high gain, DNa01 low gain)
                  + 0.5 * (a(P9_R)    - a(P9_L)), -1, 1)          P9: forward walking + ipsilateral turning
  Sign: left-dominant activity -> turn < 0 -> the body turns LEFT (ipsilateral turning).
  rate_hz(g) = mean firing rate over all root IDs of atlas group g (silent neurons count as 0).

REFERENCE RATE (calibrated on the brain model, spikes/bridge/calibrate.py)
  REF_RATE_HZ = 148.3 Hz = median rate that a DN / readout neuron reaches when it is itself
  driven by 150 Hz Poisson input in flylab.brain (Shiu et al. 2024 activation protocol;
  1000 ms, 3 trials, seed 0; 15 group-sides: MDN, P9, DNa01, DNa02, GF, aDN1, aDN2, MN9;
  range 146.3-179.0 Hz, mean 153.5 Hz). So a = 1 means "as active as under direct
  optogenetic-style 150 Hz drive"; a DN reached only via synapses at 15 Hz gives a = 0.10.
  The reference scales with the stimulation protocol: direct activation at 50 Hz -> a ~ 0.33.

BRAIN-LEVEL READOUTS (not expressible by the walking body)
  escape <- GF (DNp01, giant fiber)           Lima & Miesenboeck 2005; von Reyn et al. 2014
  feed   <- MN9 (proboscis motor neuron)      Gordon & Scott 2009; used as proxy by Shiu et al. 2024
  groom  <- aDN1 / aDN2 (antennal grooming)   Hampel et al. 2015; as used in Shiu et al. 2024 Fig. 5
  active  = max rate over the readout neurons >= READOUT_MIN_HZ (1 Hz). Shiu et al. 2024 define
  an activated neuron as one with "greater than 0 Hz firing" (the model has 0 Hz baseline);
  we require >= 1 Hz (on average >= 1 spike per 1-s trial) to ignore isolated single spikes.

LIMITATIONS
  * Hand-designed VNC replacement; weights 1.0/0.5/0.5 and the linear-saturating activation
    are choices, not fits. Only 4 DN types drive the body (out of ~1,300 DNs in FlyWire).
  * Rates are time-averaged over the brain run; the body receives a constant command (no
    closed loop, no sensory feedback from body to brain).
  * MDN asymmetry (backward turning) is not mapped; backward gait = reversed forward
    kinematics (see flylab.body).
  * A steering DN active without a locomotor DN produces turning in place (the body has no
    "resting fly ignores steering" state); the papers activated steering DNs in walking flies.

FLIGHT (Phase 3, frozen adapter; see the block "flight adapter" below and describe()["flight"])
  rates_to_flight_command(rates) -> {"takeoff","thrust","yaw","pitch","explain"} for flylab.flight:
    takeoff <- GF (escape trigger), thrust <- DNg02 population (Namiki et al. 2022), yaw <- DNa01/DNa02 asymmetry
    (transfer assumption), pitch = 0. rates_to_behavior_command(rates) -> {"mode": "walk"|"flight", ...} picks flight
    when the GF takeoff trigger is active. The walking mapping above is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "spikes" / "bridge" / "out"
BENCH_JSON = ROOT / "data" / "benchmarks" / "embodied_validation.json"
ASSET_DIR = ROOT / "assets" / "embodied"

# --------------------------------------------------------------------------- parameters
REF_RATE_HZ = 148.3  # calibrated, see module docstring / CALIBRATION
CALIBRATION = {
    "ref_rate_hz": REF_RATE_HZ,
    "statistic": "median over 15 group-sides of the self-rate under direct drive",
    "protocol": "flylab.brain.simulate(excite=<group>, excite_rate_hz=150, duration_ms=1000, n_trials=3, seed=0)",
    "groups": ["MDN", "P9", "P9_L", "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R", "GF", "aDN1_shiu2024",
               "aDN2_shiu2024", "MN9"],
    "self_rates_hz": [146.33, 146.33, 146.33, 146.33, 146.33, 146.33, 146.67, 148.33, 148.33, 150.0,
                      159.67, 163.5, 164.67, 164.67, 179.0],
    "range_hz": [146.33, 179.0],
    "mean_hz": 153.5,
    "script": "spikes/bridge/calibrate.py (raw output spikes/bridge/out/calibration.json)",
    "note": "Values above 150 Hz come from recurrent excitation between co-activated partners "
            "(e.g. both GFs); a single directly driven neuron reaches ~146 Hz (Poisson input on a 0.1 ms grid).",
}
READOUT_MIN_HZ = 1.0

# Locomotor terms: (drive key, atlas base group, weight, citations, description)
LOCOMOTOR_TERMS = [
    ("forward", "P9", 1.0, ["10.1016/j.neuron.2020.07.032"],
     "DNp09 (P9): bilateral activation initiates forward walking (Bidaye et al. 2020, Neuron)."),
    ("backward", "MDN", 1.0, ["10.1126/science.1249964"],
     "Moonwalker DNs: activation is sufficient for backward walking (Bidaye et al. 2014, Science)."),
]
# Steering terms: (atlas base group, weight on (a_R - a_L), citations, description)
STEERING_TERMS = [
    ("DNa02", 1.0, ["10.7554/elife.102230"],
     "DNa02: unilateral activation influences ipsilateral rotation; large/transient (high-gain) steering "
     "(Rayshubskiy et al. 2025, eLife)."),
    ("DNa01", 0.5, ["10.7554/elife.102230"],
     "DNa01: predicts sustained, lower-gain steering (Rayshubskiy et al. 2025, eLife); half weight (our choice)."),
    ("P9", 0.5, ["10.1016/j.neuron.2020.07.032"],
     "P9: 'forward walking with ipsilateral turning' (Bidaye et al. 2020); unilateral -> ipsilateral turn. "
     "Half weight (our choice)."),
]
# Brain-level readouts: behaviour -> (atlas groups, citations, description)
READOUTS = {
    "escape": (["GF_L", "GF_R"], ["10.1016/j.cell.2005.02.004", "10.1038/nn.3741"],
               "Giant fiber (DNp01): activation elicits escape takeoff/jump."),
    "feed": (["MN9_L", "MN9_R"], ["10.1016/j.neuron.2008.12.033", "10.1038/s41586-024-07763-9"],
             "MN9 proboscis motor neuron: proxy for proboscis extension / feeding initiation (as in Shiu et al. 2024)."),
    "groom": (["aDN1_shiu2024", "aDN2_shiu2024"], ["10.7554/eLife.08758", "10.1038/s41586-024-07763-9"],
              "Antennal grooming DNs aDN1/aDN2 (Hampel et al. 2015), IDs as used in Shiu et al. 2024 Fig. 5."),
}
SHIU_ACTIVE_DEF = ("Shiu et al. 2024 (doi:10.1038/s41586-024-07763-9): activated neurons are those with "
                   "'greater than 0 Hz firing'; we use >= 1 Hz to ignore isolated single spikes.")

SIDED = ("P9", "MDN", "DNa02", "DNa01")
NEEDED_GROUPS = ([f"{b}_{s}" for b in SIDED for s in "LR"]
                 + sorted({g for groups, _, _ in READOUTS.values() for g in groups}))


# --------------------------------------------------------------------------- helpers
def _clean_rate(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x) or x < 0:
        return 0.0
    return x


def activation(rate_hz: float, ref_rate_hz: float = REF_RATE_HZ) -> float:
    """Linear-saturating normalisation: min(1, rate / ref_rate_hz), 0 for invalid input."""
    r = _clean_rate(rate_hz)
    ref = float(ref_rate_hz) if ref_rate_hz and ref_rate_hz > 0 else REF_RATE_HZ
    return min(1.0, r / ref)


def _is_neuron_key(k: Any) -> bool:
    return str(k).strip().isdigit()


def group_rates(rates: dict | None, needed: list[str] | None = None) -> tuple[dict[str, float], dict]:
    """Rates of the groups the bridge needs, from per-neuron OR group-level input.

    Auto-detect: if every key is all-digit -> per-neuron rates {root_id: Hz} (as
    brain.simulate()["rates"]); group rate = mean over the atlas group's root IDs, silent
    neurons counted as 0 (same convention as flylab.tools._group_rates).
    Otherwise -> group rates {group_name: Hz}; names/aliases resolved with
    atlas.resolve_group; a side group missing from the input falls back to its bilateral
    group (e.g. "MDN" for MDN_L and MDN_R); digit / unknown keys are ignored and listed.
    Returns (group -> Hz, info). `needed` (default NEEDED_GROUPS, the walking groups) selects other groups,
    e.g. FLIGHT_NEEDED_GROUPS for the flight command.
    """
    from flylab import atlas

    needed = list(needed) if needed else NEEDED_GROUPS
    rates = dict(rates or {})
    info: dict[str, Any] = {}
    if not rates:
        info["input_mode"] = "empty"
        return {g: 0.0 for g in needed}, info
    if all(_is_neuron_key(k) for k in rates):
        info["input_mode"] = "neuron_rates"
        clean = {str(k).strip(): _clean_rate(v) for k, v in rates.items()}
        out = {}
        for g in needed:
            ids = atlas.group_ids(g)
            out[g] = (sum(clean.get(str(i), 0.0) for i in ids) / len(ids)) if ids else 0.0
        return out, info

    info["input_mode"] = "group_rates"
    canon: dict[str, float] = {}
    ignored = []
    for k, v in rates.items():
        if _is_neuron_key(k):
            ignored.append(str(k))
            continue
        try:
            canon[atlas.resolve_group(str(k))] = _clean_rate(v)
        except KeyError:
            ignored.append(str(k))
    out, fallbacks = {}, []
    for g in needed:
        if g in canon:
            out[g] = canon[g]
            continue
        base = g.rsplit("_", 1)[0] if g.endswith(("_L", "_R")) else None
        if base and base in canon:
            out[g] = canon[base]
            fallbacks.append(f"{g}<-{base}")
        else:
            out[g] = 0.0
    if ignored:
        info["ignored_keys"] = ignored[:20] + (["..."] if len(ignored) > 20 else [])
    if fallbacks:
        info["side_fallbacks"] = fallbacks
    return out, info


def brain_readouts(rates: dict | None, min_hz: float = READOUT_MIN_HZ, *, _grp: dict | None = None) -> dict:
    """Brain-level readouts for behaviours the walking body cannot show.

    Accepts the same input as rates_to_drive (per-neuron or group rates). Returns
    {"escape": {"group": "GF", "groups": [...], "rates_hz": {...}, "rate_hz": max, "active": bool,
                "threshold_hz", "citations"}, "feed": {...}, "groom": {...},
     "active_behaviors": [...]}
    """
    grp = _grp if _grp is not None else group_rates(rates)[0]
    out: dict[str, Any] = {}
    for beh, (groups, cites, _desc) in READOUTS.items():
        rs = {g: round(float(grp.get(g, 0.0)), 3) for g in groups}
        mx = max(rs.values()) if rs else 0.0
        label = {"escape": "GF", "feed": "MN9", "groom": "aDN1/aDN2"}[beh]
        out[beh] = {"group": label, "groups": groups, "rates_hz": rs, "rate_hz": round(mx, 3),
                    "active": bool(mx >= min_hz), "threshold_hz": float(min_hz), "citations": cites}
    out["active_behaviors"] = [b for b in READOUTS if out[b]["active"]]
    return out


def readout_label(readouts: dict, behavior: str) -> str:
    """Observed label for a brain-level ground-truth entry: behavior if active else 'no_<behavior>'."""
    return behavior if readouts.get(behavior, {}).get("active") else f"no_{behavior}"


def rates_to_drive(rates: dict | None, ref_rate_hz: float = REF_RATE_HZ) -> dict:
    """Descending-neuron rates -> body drive (see module docstring for the formulas).

    rates: per-neuron {root_id_str: Hz} or group {group_name: Hz} (auto-detected).
    Returns {"forward": 0..1, "turn": -1..1 (neg = left), "backward": 0..1, "explain": {...}}.
    "explain" lists the group rates used, activations, each term, the reference rate, the
    formula and the brain-level readouts (escape / feed / groom) for transparency.
    """
    grp, info = group_rates(rates)
    a = {g: activation(grp[g], ref_rate_hz) for g in NEEDED_GROUPS if g.split("_")[0] in SIDED}
    terms: dict[str, Any] = {}
    drive = {"forward": 0.0, "backward": 0.0}
    for key, base, w, _c, _d in LOCOMOTOR_TERMS:
        val = w * (a[f"{base}_L"] + a[f"{base}_R"]) / 2.0
        drive[key] = min(1.0, max(0.0, val))
        terms[key] = f"{w} * mean(a({base}_L)={a[f'{base}_L']:.3f}, a({base}_R)={a[f'{base}_R']:.3f}) = {val:.3f}"
    turn_raw = 0.0
    turn_terms = {}
    for base, w, _c, _d in STEERING_TERMS:
        c = w * (a[f"{base}_R"] - a[f"{base}_L"])
        turn_raw += c
        turn_terms[base] = round(c, 4)
    turn = max(-1.0, min(1.0, turn_raw))
    readouts = brain_readouts(None, _grp=grp)
    explain = {
        **info,
        "group_rates_hz": {g: round(float(v), 3) for g, v in grp.items()},
        "activation": {g: round(v, 4) for g, v in a.items()},
        "ref_rate_hz": float(ref_rate_hz),
        "terms": {**terms, "turn_components (w*(a_R-a_L))": turn_terms, "turn_raw": round(turn_raw, 4)},
        "formula": ("a=min(1,rate/ref); forward=mean(a(P9_L),a(P9_R)); backward=mean(a(MDN_L),a(MDN_R)); "
                    "turn=clip(1.0*dDNa02+0.5*dDNa01+0.5*dP9,-1,1), d=a_R-a_L (neg=left)"),
        "brain_readouts": {b: {"active": readouts[b]["active"], "rate_hz": readouts[b]["rate_hz"]} for b in READOUTS},
        "note": "hand-designed VNC replacement (FlyWire covers the brain only); see flylab.bridge.describe()",
    }
    return {"forward": round(drive["forward"], 4), "turn": round(turn, 4),
            "backward": round(drive["backward"], 4), "explain": explain}


def describe() -> dict:
    """Machine-readable mapping table (groups, signs, weights, reference rate, citations)."""
    rows = []
    for key, base, w, cites, desc in LOCOMOTOR_TERMS:
        rows.append({"output": key, "groups": [f"{base}_L", f"{base}_R"], "weight": w,
                     "formula": f"{key} = {w} * mean(a({base}_L), a({base}_R))", "sign": "+", "citations": cites,
                     "description": desc})
    for base, w, cites, desc in STEERING_TERMS:
        rows.append({"output": "turn", "groups": [f"{base}_L", f"{base}_R"], "weight": w,
                     "formula": f"turn += {w} * (a({base}_R) - a({base}_L))",
                     "sign": "left-dominant -> negative (left turn, ipsilateral)", "citations": cites,
                     "description": desc})
    readouts = [{"behavior": b, "groups": g, "threshold_hz": READOUT_MIN_HZ, "rule": "max rate >= threshold",
                 "citations": c, "description": d} for b, (g, c, d) in READOUTS.items()]
    return {
        "name": "flylab.bridge (hand-designed brain->body interface)",
        "scope": ("FlyWire v783 covers the brain only; the ventral nerve cord is NOT simulated. This bridge replaces it "
                  "with a transparent mapping from 4 descending-neuron types to the body's 3-number drive."),
        "activation": "a(g) = min(1, mean_rate_hz(g) / ref_rate_hz)",
        "ref_rate_hz": REF_RATE_HZ,
        "calibration": CALIBRATION,
        "drive_terms": rows,
        "turn_clip": [-1.0, 1.0],
        "readouts": readouts,
        "readout_threshold_note": SHIU_ACTIVE_DEF,
        "body": "flylab.body (NeuroMechFly / flygym 2.1 hybrid turning controller), doi:10.1038/s41592-024-02497-y",
        "flight": describe_flight(),
        "limitations": [
            "Weights (1.0 / 0.5 / 0.5) and the linear-saturating activation are hand-chosen, not fitted.",
            "Only P9, MDN, DNa01, DNa02 drive the body; all other descending neurons are ignored.",
            "Open loop: time-averaged brain rates -> constant body command; no body->brain feedback.",
            "Steering DNs without a locomotor DN give turning in place; papers tested walking flies.",
            "MDN asymmetry (backward turning) not mapped; backward gait = reversed forward kinematics.",
            "Escape / feeding / grooming are brain-level readouts only (the body cannot jump, groom or feed).",
        ],
    }


# --------------------------------------------------------------------------- flight adapter (Phase 3)
# Frozen before the flight comparison runs (G3): parameters below + FLIGHT_FROZEN["parameter_hash"].
#
#   takeoff = mean(a(GF_L), a(GF_R))                    giant fiber (DNp01): escape takeoff trigger
#                                                       Lima & Miesenboeck 2005; von Reyn et al. 2014
#             flylab.flight starts the escape sequence (jump + wing start) when takeoff >= 0.5
#   thrust  = BASELINE + (1-BASELINE) * a(DNg02 pop)    DNg02 population code -> wingbeat amplitude
#             pop = neuron-count-weighted mean of DNg02_L / DNg02_R  (Namiki et al. 2022, Curr Biol)
#             BASELINE = 0.0: thrust 0 = the hand-designed hover trim of flylab.flight (constant baseline
#             flight-motor drive once airborne; the connectome adds amplitude on top of it)
#   yaw     = clip(1.0*d(DNa02) + 0.5*d(DNa01), -1, 1)  d = a_R - a_L (neg = left). TRANSFER ASSUMPTION: these are
#             WALKING steering DNs (Rayshubskiy et al. 2025); no literature-verified flight-steering DN pair
#             was identified here and flight yaw is not part of the flight validation.
#   pitch   = 0.0                                       no connectome input (documented default: no forward drive)
#   a(g) = min(1, rate/REF_RATE_HZ) with the same measured reference as the walking bridge (148.3 Hz).
FLIGHT_TAKEOFF_THRESHOLD = 0.5   # == flylab.flight escape trigger (cmd["takeoff"] >= 0.5), i.e. half the reference rate
FLIGHT_BASELINE_THRUST = 0.0
FLIGHT_STEERING_TERMS = [("DNa02", 1.0), ("DNa01", 0.5)]
FLIGHT_NEEDED_GROUPS = ["GF_L", "GF_R", "DNg02_L", "DNg02_R", "DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R"]
FLIGHT_CITATIONS = {
    "takeoff": ["10.1016/j.cell.2005.02.004", "10.1038/nn.3741"],
    "thrust": ["10.1016/j.cub.2022.01.008"],
    "yaw": ["10.7554/elife.102230"],
}
FLIGHT_PARAMS = {"takeoff_group": "GF", "takeoff_rule": "mean(a(GF_L), a(GF_R))",
                 "takeoff_threshold": FLIGHT_TAKEOFF_THRESHOLD, "thrust_group": "DNg02",
                 "thrust_rule": "baseline + (1-baseline)*a(n-weighted mean rate of DNg02_L, DNg02_R)",
                 "thrust_baseline": FLIGHT_BASELINE_THRUST, "yaw_terms": FLIGHT_STEERING_TERMS, "pitch": 0.0,
                 "ref_rate_hz": REF_RATE_HZ}


def _flight_param_hash() -> str:
    import hashlib
    return hashlib.sha256(json.dumps(FLIGHT_PARAMS, sort_keys=True).encode()).hexdigest()[:16]


FLIGHT_FROZEN = {"version": "flight-adapter-v1", "frozen_at": "2026-10-04 08:55 (before the flight_validation.json runs)",
                 "parameter_hash": _flight_param_hash(),
                 "note": "Parameters are frozen before the comparison runs (team rule G3). The validation JSON records the hash; "
                         "a changed hash means the adapter was modified after the comparison."}


def rates_to_flight_command(rates: dict | None, ref_rate_hz: float = REF_RATE_HZ) -> dict:
    """Descending-neuron rates -> flight command {"takeoff","thrust","yaw","pitch","explain"} (see block comment above).

    rates: per-neuron {root_id_str: Hz} or group {group_name: Hz} (auto-detected, same as rates_to_drive).
    What the connectome decides: whether takeoff is triggered (GF), how much wingbeat amplitude is added (DNg02) and
    the left/right steering asymmetry. NOT decided by it: the jump impulse, wing kinematics, attitude stabilisation
    (halteres / visual feedback stand-ins) and the baseline flight-motor drive (all hand-designed, flylab.flight).
    """
    from flylab import atlas

    grp, info = group_rates(rates, FLIGHT_NEEDED_GROUPS)
    a = {g: activation(grp[g], ref_rate_hz) for g in FLIGHT_NEEDED_GROUPS}
    takeoff = min(1.0, max(0.0, (a["GF_L"] + a["GF_R"]) / 2.0))
    n_l, n_r = len(atlas.group_ids("DNg02_L")), len(atlas.group_ids("DNg02_R"))
    pop_hz = (n_l * grp["DNg02_L"] + n_r * grp["DNg02_R"]) / max(1, n_l + n_r)
    a_pop = activation(pop_hz, ref_rate_hz)
    thrust = min(1.0, max(0.0, FLIGHT_BASELINE_THRUST + (1.0 - FLIGHT_BASELINE_THRUST) * a_pop))
    yaw_terms = {}
    yaw_raw = 0.0
    for base, w in FLIGHT_STEERING_TERMS:
        c = w * (a[f"{base}_R"] - a[f"{base}_L"])
        yaw_raw += c
        yaw_terms[base] = round(c, 4)
    yaw = max(-1.0, min(1.0, yaw_raw))
    triggered = takeoff >= FLIGHT_TAKEOFF_THRESHOLD
    explain = {
        **info,
        "group_rates_hz": {g: round(float(v), 3) for g, v in grp.items()},
        "activation": {g: round(v, 4) for g, v in a.items()},
        "ref_rate_hz": float(ref_rate_hz),
        "terms": {
            "takeoff": f"mean(a(GF_L)={a['GF_L']:.3f}, a(GF_R)={a['GF_R']:.3f}) = {takeoff:.3f}",
            "thrust": (f"{FLIGHT_BASELINE_THRUST} + {1.0 - FLIGHT_BASELINE_THRUST} * a(DNg02 population "
                       f"{pop_hz:.2f} Hz; n_L={n_l}, n_R={n_r}) = {thrust:.3f}"),
            "yaw_components (w*(a_R-a_L))": yaw_terms, "yaw_raw": round(yaw_raw, 4), "pitch": "0.0 (no connectome input)",
        },
        "takeoff_threshold": FLIGHT_TAKEOFF_THRESHOLD, "takeoff_triggered": bool(triggered),
        "thrust_applies_only_when_airborne": "flylab.flight beats the wings only after a takeoff trigger; "
                                             "without GF-driven takeoff the thrust value has no effect",
        "formula": ("a=min(1,rate/ref); takeoff=mean(a(GF_L),a(GF_R)) [flight if >= 0.5]; thrust=a(DNg02 population); "
                    "yaw=clip(1.0*dDNa02+0.5*dDNa01,-1,1) [transfer assumption]; pitch=0"),
        "connectome_decides": "takeoff trigger (GF), wingbeat-amplitude increment (DNg02), left/right steering asymmetry",
        "hand_designed": "jump impulse, wing kinematics, attitude/heading stabilisation, baseline flight-motor drive, "
                         "quasi-steady aerodynamics (flylab.flight.model_notes())",
        "frozen": dict(FLIGHT_FROZEN),
        "note": "hand-designed VNC replacement (FlyWire covers the brain only); open loop, constant command from time-averaged rates",
    }
    return {"takeoff": round(takeoff, 4), "thrust": round(thrust, 4), "yaw": round(yaw, 4), "pitch": 0.0, "explain": explain}


def rates_to_behavior_command(rates: dict | None, ref_rate_hz: float = REF_RATE_HZ) -> dict:
    """Choose the body mode from brain rates: flight if the takeoff trigger (GF) is active, else walking.

    Returns {"mode": "walk", "drive": {...}, "explain": {...}} or {"mode": "flight", "command": {...}, "explain": {...}}.
    Design choice (not from a paper): an active GF escape takeoff pre-empts walking commands; below the threshold
    the walking drive of rates_to_drive() is returned unchanged.
    """
    cmd = rates_to_flight_command(rates, ref_rate_hz)
    ex = cmd["explain"]
    rule = (f"mode = flight if takeoff = mean(a(GF_L), a(GF_R)) >= {FLIGHT_TAKEOFF_THRESHOLD} "
            f"(= {FLIGHT_TAKEOFF_THRESHOLD * ref_rate_hz:.0f} Hz mean GF rate), else walk")
    if ex["takeoff_triggered"]:
        return {"mode": "flight", "command": {k: cmd[k] for k in ("takeoff", "thrust", "yaw", "pitch")},
                "explain": {"mode_rule": rule, **ex}}
    drive = rates_to_drive(rates, ref_rate_hz)
    return {"mode": "walk", "drive": {k: drive[k] for k in ("forward", "turn", "backward")},
            "explain": {"mode_rule": rule, "takeoff_activation": cmd["takeoff"],
                        "takeoff_threshold": FLIGHT_TAKEOFF_THRESHOLD, **drive["explain"]}}


def describe_flight() -> dict:
    """Machine-readable table of the (frozen) flight adapter."""
    return {
        "name": "flylab.bridge flight adapter (hand-designed, frozen)",
        "frozen": dict(FLIGHT_FROZEN),
        "params": FLIGHT_PARAMS,
        "terms": [
            {"output": "takeoff", "groups": ["GF_L", "GF_R"], "formula": "takeoff = mean(a(GF_L), a(GF_R)); escape sequence if >= "
             f"{FLIGHT_TAKEOFF_THRESHOLD}", "citations": FLIGHT_CITATIONS["takeoff"],
             "description": "Giant fiber (DNp01): GF activation elicits escape takeoff / wing beating / flight (Lima & Miesenboeck 2005); "
                            "GF spike timing selects short vs. long takeoff (von Reyn et al. 2014). Threshold = half the measured "
                            "reference rate (hand-chosen)."},
            {"output": "thrust", "groups": ["DNg02_L", "DNg02_R"], "formula": "thrust = a(n-weighted mean DNg02 rate)",
             "citations": FLIGHT_CITATIONS["thrust"],
             "description": "DNg02 population code for wingbeat amplitude (Namiki et al. 2022). 25 DNg02 neurons in FlyWire v783 vs "
                            "'at least 15 pairs' in the paper (identification by cell-type name, medium confidence). "
                            "thrust 0 = hand-designed hover trim (constant baseline flight-motor drive once airborne)."},
            {"output": "yaw", "groups": ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R"],
             "formula": "yaw = clip(1.0*(a(DNa02_R)-a(DNa02_L)) + 0.5*(a(DNa01_R)-a(DNa01_L)), -1, 1)",
             "citations": FLIGHT_CITATIONS["yaw"],
             "description": "Walking-steering DNs reused for flight yaw: a TRANSFER ASSUMPTION, not literature-verified for flight."},
            {"output": "pitch", "groups": [], "formula": "pitch = 0", "citations": [],
             "description": "No connectome input: no forward-flight drive."},
        ],
        "mode_rule": "flight if takeoff >= threshold (rates_to_behavior_command), else the walking drive",
        "scope": "FlyWire covers the brain only; the VNC, flight muscles, halteres and wing hinge are replaced by this adapter "
                 "and by flylab.flight (quasi-steady aerodynamics + attitude stabiliser).",
        "limitations": [
            "G4: a body controller that can fly is not evidence of connectome-controlled flight. The connectome decides only the "
            "takeoff trigger (GF), the wingbeat-amplitude increment (DNg02) and a steering asymmetry; everything else is hand-designed.",
            "GF is the only takeoff trigger. In real flies parallel (non-GF) circuits also produce takeoffs (long takeoff that initiates "
            "stable flight, von Reyn et al. 2014), so 'GF silenced -> no takeoff' is a consequence of this design, not a finding.",
            "The model has a single takeoff label; the short (GF-driven, flight-unstable) vs. long takeoff modes are not distinguished.",
            "Post-takeoff flight stability comes from the hand-designed attitude stabiliser, not from the connectome.",
            "Open loop: constant command from time-averaged 1-s brain rates; no visual or haltere feedback to the brain.",
            "Yaw reuses walking-steering DNs (assumption); flight turning is not validated against literature.",
            "A single GF at full rate gives takeoff = 0.5 = the threshold (borderline by construction); unilateral GF activation "
            "is not validated.",
        ],
    }


# --------------------------------------------------------------------------- calibration
def calibrate(groups: list[str] | None = None, rate_hz: float = 150.0, duration_ms: float = 1000.0,
              n_trials: int = 3, n_threads: int = 3) -> dict:
    """Re-measure the reference rate: self-rate of each group (per side) under direct Poisson drive."""
    import statistics

    from flylab import atlas, brain

    groups = groups or CALIBRATION["groups"]
    rows, vals = [], []
    for g in groups:
        res = brain.simulate(atlas.group_ids(g), excite_rate_hz=rate_hz, duration_ms=duration_ms,
                             n_trials=n_trials, n_threads=n_threads)
        sides = [g] if g.endswith(("_L", "_R")) or g in ("aDN1_shiu2024", "aDN2_shiu2024") else [f"{g}_L", f"{g}_R"]
        for s in sides:
            ids = atlas.group_ids(s)
            r = sum(brain.rates_for(res, ids).values()) / len(ids)
            rows.append({"group": s, "rate_hz": round(r, 2)})
            vals.append(r)
    return {"rows": rows, "median_hz": round(statistics.median(vals), 2), "mean_hz": round(statistics.mean(vals), 2),
            "protocol": {"rate_hz": rate_hz, "duration_ms": duration_ms, "n_trials": n_trials}}


# --------------------------------------------------------------------------- end-to-end validation
# (condition, excite groups, silence groups, gt_id, comparison kind, extra gt checks)
VALIDATION_CONDITIONS = [
    ("control", [], [], None, "body", []),
    ("MDN_bilateral", ["MDN"], [], "gt01_mdn_activate_backward", "body", []),
    ("P9_bilateral", ["P9"], [], "gt03_p9_activate_forward", "body", []),
    ("P9_L", ["P9_L"], [], "gt04_p9L_activate_turn_left", "body", []),
    ("DNa02_L", ["DNa02_L"], [], "gt05_dna02L_activate_turn_left", "body", []),
    ("DNa02_R", ["DNa02_R"], [], "gt06_dna02R_activate_turn_right", "body", []),
    ("LPLC2_bilateral", ["LPLC2"], [], "gt19_lplc2_activate_backward", "body", [("gt08_lplc2_activate_escape", "escape")]),
    ("LPLC2_MDN_silenced", ["LPLC2"], ["MDN"], "gt02_mdn_silence_backward", "body", []),
    ("sugar_GRN", ["sugar_GRN_shiu2024"], [], "gt10_sugar_activate_feed", "readout:feed", []),
    ("JO_CE", ["JO_CE_shiu2024"], [], "gt14_jo_activate_groom", "readout:groom", []),
]
CONDITION_NOTES = {
    "control": "No stimulation. The Shiu et al. model has 0 Hz baseline, so the brain is silent (not simulated) "
               "and the body receives a zero drive.",
    "DNa02_L": "Steering DN alone, no locomotor DN active -> turning in place. Rayshubskiy et al. tested walking flies.",
    "DNa02_R": "Steering DN alone, no locomotor DN active -> turning in place. Rayshubskiy et al. tested walking flies.",
    "LPLC2_bilateral": "Upstream visual stimulus: direct 150 Hz activation of the looming-sensitive LPLC2 neurons "
                       "(analogous to the optogenetic activation of Wu et al. 2016, not a simulated looming stimulus). In the brain model LPLC2 drives the giant fiber "
                       "strongly but MDN stays at 0 Hz, so the body does not walk backward. Wu et al. 2016 report "
                       "jumping and backward walking with about equal penetrance: the escape half is reproduced, "
                       "the backward half is not. Unverified hypothesis: the backward component uses DNs other than "
                       "MDN/P9/DNa01/DNa02 or pathways outside the brain connectome (VNC).",
    "LPLC2_MDN_silenced": "Silencing control for MDN. Uninformative here: MDN was already at 0 Hz without silencing, "
                          "so the check is reported as 'inconclusive' (raw verdict kept) and excluded from the hit rate.",
    "P9_bilateral": "P9 recruits DNa02 (R > L) via the connectome -> small rightward turn term; label stays forward.",
    "P9_L": "Turn direction matches (ipsilateral, recruited DNa02_L adds to it), but with forward 0.49 and turn -0.62 "
            "the body turns almost in place (net forward displacement ~0 mm): the 'forward walking' part of "
            "'forward walking with ipsilateral turning' (Bidaye et al. 2020) is not reproduced by this mapping.",
    "MDN_bilateral": "MDN_R fires above MDN_L in the brain model, but MDN asymmetry is not mapped to turning; the "
                     "heading drift comes from the reversed-kinematics backward gait of the body model.",
}
SEED_ROBUSTNESS_SEEDS = (1, 2, 3)  # body re-runs with phase_noise_rad=0.5 (same drive) for label stability
KEY_GROUPS = ["P9_L", "P9_R", "MDN_L", "MDN_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "GF_L", "GF_R",
              "MN9_L", "MN9_R", "aDN1_shiu2024", "aDN2_shiu2024"]


def _run_condition(name, excite, silence, duration_s, brain_ms, n_trials, rate_hz, render, control_every, seed):
    from flylab import atlas, body, brain

    ex_ids = [i for g in excite for i in atlas.group_ids(g)]
    si_ids = [i for g in silence for i in atlas.group_ids(g)]
    t0 = time.perf_counter()
    if ex_ids:
        bres = brain.simulate(ex_ids, si_ids or None, excite_rate_hz=rate_hz, duration_ms=brain_ms,
                              n_trials=n_trials, seed=seed, n_threads=min(4, n_trials))
        rates, brain_rt, n_active = bres["rates"], bres["runtime_s"], bres["n_active"]
    else:  # no stimulation: the Shiu model has 0 Hz baseline, nothing to simulate
        rates, brain_rt, n_active = {}, 0.0, 0
    drive = rates_to_drive(rates)
    readouts = brain_readouts(rates)
    rp = str(OUT_DIR / "videos" / f"{name}.mp4") if render else None
    bres_body = body.simulate_walk({k: drive[k] for k in ("forward", "turn", "backward")}, duration_s=duration_s,
                                   render_path=rp, seed=seed, control_every=control_every)
    return {
        "rates_n_active": n_active, "brain_runtime_s": brain_rt, "group_rates_hz": drive["explain"]["group_rates_hz"],
        "drive": {k: drive[k] for k in ("forward", "turn", "backward")}, "drive_explain": drive["explain"],
        "brain_readouts": {b: {k: readouts[b][k] for k in ("rate_hz", "active", "rates_hz")} for b in READOUTS},
        "body": {k: bres_body.get(k) for k in ("behavior", "forward_disp_mm", "lateral_disp_mm", "heading_change_deg",
                                                "yaw_rate_deg_s", "net_speed_mm_s", "mean_speed_mm_s", "fell_over",
                                                "warnings", "descending_signal", "runtime_s", "control_every")},
        "video_tmp": bres_body.get("video"), "total_runtime_s": round(time.perf_counter() - t0, 2),
    }


def validate(duration_s: float = 1.0, brain_ms: float = 1000.0, n_trials: int = 3, rate_hz: float = 150.0,
             render: bool = True, control_every: int | None = None, seed: int = 0,
             only: list[str] | None = None, save: bool = True, seed_robustness: bool = True) -> dict:
    """Run brain -> bridge -> body for VALIDATION_CONDITIONS and compare with atlas ground truth."""
    from flylab import atlas, body

    ce = control_every if control_every is not None else getattr(body, "DEFAULT_CONTROL_EVERY", 5)
    rows = []
    t_all = time.perf_counter()
    for name, excite, silence, gt_id, kind, extra in VALIDATION_CONDITIONS:
        if only and name not in only:
            continue
        r = _run_condition(name, excite, silence, duration_s, brain_ms, n_trials, rate_hz, render, ce, seed)
        beh = r["body"]["behavior"]
        checks = []
        for gid, k in ([(gt_id, kind)] if gt_id else []) + [(g, f"readout:{b}") for g, b in extra]:
            obs = readout_label(r["brain_readouts"], k.split(":")[1]) if k.startswith("readout:") else beh
            ev = atlas.evaluate(obs, gid)
            checks.append({"gt_id": gid, "comparison": "brain readout" if k.startswith("readout:") else "body behavior",
                           "observed": obs, "expected_behavior": ev["expected_behavior"], "effect": ev["effect"],
                           "verdict": ev["verdict"], "reason": ev["reason"]})
        video = None
        if r["video_tmp"] and Path(r["video_tmp"]).exists():
            src = Path(r["video_tmp"])
            if save and not only:  # only a full, saved run replaces the committed videos
                ASSET_DIR.mkdir(parents=True, exist_ok=True)
                dst = ASSET_DIR / f"{name}{src.suffix}"
                shutil.copyfile(src, dst)
            else:
                dst = src
            try:
                video = dst.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                video = dst.as_posix()
        row = {"condition": name, "stimulus_groups": excite, "silenced_groups": silence,
               "key_group_rates_hz": {g: r["group_rates_hz"].get(g, 0.0) for g in KEY_GROUPS},
               "n_active_neurons": r["rates_n_active"], "drive": r["drive"], "behavior": beh,
               "body_metrics": r["body"], "brain_readouts": r["brain_readouts"], "checks": checks,
               "gt_id": gt_id, "verdict": checks[0]["verdict"] if checks else "no_ground_truth",
               "runtimes_s": {"brain": r["brain_runtime_s"], "body": r["body"]["runtime_s"],
                              "total": r["total_runtime_s"]},
               "video": video}
        if name in CONDITION_NOTES:
            row["note"] = CONDITION_NOTES[name]
        rows.append(row)
        print(f"{name:<20} drive f={r['drive']['forward']:.2f} t={r['drive']['turn']:+.2f} b={r['drive']['backward']:.2f}"
              f" -> {beh:<10} readouts={[b for b in READOUTS if r['brain_readouts'][b]['active']]}"
              f" | " + "; ".join(f"{c['gt_id']}: {c['verdict']}" for c in checks)
              + f" | brain {r['brain_runtime_s']}s body {r['body']['runtime_s']}s", flush=True)
    # silencing checks are only informative if the unsilenced control showed the behaviour
    by_stim = {tuple(r["stimulus_groups"]): r for r in rows if not r["silenced_groups"]}
    for row in rows:
        for c in row["checks"]:
            c["informative"] = True
            if c["effect"] == "reduce" and row["silenced_groups"]:
                ctrl = by_stim.get(tuple(row["stimulus_groups"]))
                c["control_condition"] = ctrl["condition"] if ctrl else None
                c["control_behavior"] = ctrl["behavior"] if ctrl else None
                if ctrl is None or ctrl["behavior"] != c["expected_behavior"]:
                    # a silencing effect can only be tested if the behaviour occurs without silencing;
                    # report 'inconclusive' (dashboard excludes it from agreement), keep the raw verdict
                    c["informative"] = False
                    c["raw_verdict"] = c["verdict"]
                    c["verdict"] = "inconclusive"
                    c["reason"] += ("; INCONCLUSIVE: the unsilenced control did not show "
                                    f"{c['expected_behavior']} either, so a reduction cannot be tested")
        if row["checks"]:
            row["verdict"] = row["checks"][0]["verdict"]
    # label stability of the body under initial-phase noise (same drive, no render)
    if seed_robustness and rows:
        from flylab import body as _body
        drives = {r["condition"]: r["drive"] for r in rows}
        labels = {r["condition"]: [] for r in rows}
        for sd in SEED_ROBUSTNESS_SEEDS:
            res = _body.simulate_many(drives, duration_s=duration_s, seed=sd, parallel=True, max_workers=4,
                                      control_every=ce, phase_noise_rad=0.5)
            for k, v in res.items():
                labels[k].append(v["behavior"])
        for r in rows:
            ls = labels[r["condition"]]
            r["body_seed_robustness"] = {"phase_noise_rad": 0.5, "seeds": list(SEED_ROBUSTNESS_SEEDS), "labels": ls,
                                         "fraction_same_as_main": round(sum(x == r["behavior"] for x in ls) / len(ls), 3)}
    all_checks = [c for row in rows for c in row["checks"]]
    comparable = [c for c in all_checks if c["verdict"] not in ("not_comparable", "inconclusive")]
    n_cons = sum(c["verdict"] == "consistent" for c in comparable)
    informative = [c for c in comparable if c.get("informative", True)]
    n_cons_inf = sum(c["verdict"] == "consistent" for c in informative)
    # readout specificity: where does each brain-level readout fire? (false positives would show up here)
    expected_ro = {"escape": {"LPLC2"}, "feed": {"sugar_GRN_shiu2024"}, "groom": {"JO_CE_shiu2024"}}
    specificity = {}
    for b in READOUTS:
        act = [r["condition"] for r in rows if r["brain_readouts"][b]["active"]]
        unexpected = [r["condition"] for r in rows if r["brain_readouts"][b]["active"]
                      and not (set(r["stimulus_groups"]) & expected_ro[b])]
        specificity[b] = {"active_in": act, "unexpected_active_in": unexpected,
                          "n_conditions": len(rows)}
    out = {
        "what": "End-to-end embodied validation: FlyWire v783 LIF brain -> flylab.bridge -> NeuroMechFly body",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {"brain_duration_ms": brain_ms, "n_trials": n_trials, "excite_rate_hz": rate_hz,
                     "body_duration_s": duration_s, "body_control_every": ce, "seed": seed,
                     "bridge_ref_rate_hz": REF_RATE_HZ, "readout_threshold_hz": READOUT_MIN_HZ},
        "bridge": describe(),
        "caveats": [
            "Partly circular: the bridge maps P9->forward, MDN->backward, DNa01/DNa02 asymmetry->turn using the same "
            "papers that define gt01/gt03-gt06, so direct-DN rows mainly test the plumbing and sign conventions "
            "(brain -> bridge -> body), not a discovery.",
            "Non-trivial (emergent from the connectome model) rows: LPLC2 (GF escape yes, MDN/backward no), sugar GRN "
            "-> MN9, JO-C/E -> aDN1/aDN2, P9 -> DNa02 recruitment. The brain-level readout results reproduce "
            "Shiu et al. 2024 model behaviour rather than new findings.",
            "Single seed per condition for brain and main body run; body label stability checked with 3 phase-noise "
            "seeds (body_seed_robustness). The rest of the VNC is replaced by a hand-designed mapping.",
            "The MDN-silencing row is uninformative (MDN at 0 Hz without silencing): its verdict is 'inconclusive' "
            "(raw_verdict kept) and it is excluded from hit_rate.",
            "Body labels come from flylab.body.classify thresholds (yaw rate, displacement); 'turn' rows for DNa02 are "
            "turning in place because no locomotor DN is active.",
        ],
        "rows": rows,
        "summary": {"n_checks": len(all_checks), "n_comparable": len(comparable), "n_consistent": n_cons,
                    "n_inconclusive": sum(c["verdict"] == "inconclusive" for c in all_checks),
                    "hit_rate": round(n_cons / len(comparable), 3) if comparable else None,
                    "hit_rate_note": "consistent / comparable; not_comparable and inconclusive checks excluded",
                    "readout_specificity": specificity,
                    "n_informative": len(informative), "n_consistent_informative": n_cons_inf,
                    "hit_rate_informative": round(n_cons_inf / len(informative), 3) if informative else None,
                    "by_comparison": {k: {"n": sum(c["comparison"] == k for c in informative),
                                          "consistent": sum(c["comparison"] == k and c["verdict"] == "consistent"
                                                            for c in informative)}
                                      for k in ("body behavior", "brain readout")},
                    "verdicts": {c["gt_id"]: c["verdict"] for c in all_checks},
                    "total_wall_s": round(time.perf_counter() - t_all, 1)},
    }
    sm = out["summary"]
    print(f"final verdicts (after silencing-control check): {sm['verdicts']}")
    print(f"hit rate {sm['n_consistent']}/{sm['n_comparable']} comparable = {sm['hit_rate']} "
          f"(inconclusive: {sm['n_inconclusive']}; not_comparable excluded)")
    if save:
        # a subset run (--only) must not overwrite the full committed benchmark the dashboard reads
        target = BENCH_JSON if not only else OUT_DIR / "embodied_validation_subset.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"saved {target.relative_to(ROOT)}")
    return out


def _print_describe() -> None:
    d = describe()
    print(d["scope"])
    print(f"activation: {d['activation']}  (ref_rate_hz = {d['ref_rate_hz']})")
    for r in d["drive_terms"]:
        print(f"  {r['output']:<9} {r['formula']:<45} {', '.join(r['citations'])}")
    for r in d["readouts"]:
        print(f"  readout {r['behavior']:<7} {'/'.join(r['groups']):<30} >= {r['threshold_hz']} Hz  {', '.join(r['citations'])}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="flylab.bridge: brain rates -> body drive")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="re-measure the reference rate (brain sims)")
    ap.add_argument("--validate", action="store_true", help="end-to-end validation (brain+body, renders videos)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of validation conditions (saved to spikes/bridge/out/embodied_validation_subset.json)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--control-every", type=int, default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.calibrate:
        print(json.dumps(calibrate(), indent=1))
    elif args.validate:
        validate(render=not args.no_render, control_every=args.control_every, only=args.only, save=not args.no_save)
    else:
        _print_describe()


if __name__ == "__main__":
    main()
