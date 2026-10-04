"""FlyWire v783 neuron atlas + literature ground truth for the Embodied Fly Lab.

Public API (docs/CONTRACTS.md):
    groups() -> dict[str, dict]          name -> {"root_ids","cell_type","side","role","description","citations", ...}
    group_ids(name) -> list[int]
    find_cell_type(query) -> list[dict]  search FlyWire annotations by cell_type / hemibrain_type / synonyms
    ground_truth() -> list[dict]         literature-verified neuron -> behaviour relations
Extras: neuron_info(root_id), build(), groups_for_behavior(label).

Data:
    data/neurons.json       curated groups (committed, small) - written by `python -m flylab.atlas --build`
    data/ground_truth.json  curated literature relations (committed) - written by --build (DOIs verified online)
    data/raw/flywire_annotations_Supplemental_file1_neuron_annotations.tsv  (downloaded on demand, gitignored)

Root IDs: the `root_id` column of the FlyWire annotation file is the materialisation-783 ID
(see flyconnectome/flywire_annotations supplemental_files/README.md), i.e. the same ID space as the
Shiu et al. 2024 brain model files Completeness_783.csv / Connectivity_783.parquet.

CLI:
    python -m flylab.atlas --summary
    python -m flylab.atlas --find DNa02
    python -m flylab.atlas --group MDN_L
    python -m flylab.atlas --build           (re-curate both JSON files; needs internet for DOI checks)
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
NEURONS_JSON = DATA / "neurons.json"
GT_JSON = DATA / "ground_truth.json"
ANNOT_TSV = RAW / "flywire_annotations_Supplemental_file1_neuron_annotations.tsv"
ANNOT_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
             "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
ANNOT_SOURCE = "https://github.com/flyconnectome/flywire_annotations (Supplemental_file1_neuron_annotations.tsv)"
SHIU_FIGURES_URL = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/figures.ipynb"
SHIU_REPO = "https://github.com/philshiu/Drosophila_brain_model"
COMPLETENESS_783 = RAW / "Completeness_783.csv"   # brain-model neuron list (owned by the brain component)

DOI_SCHLEGEL_2024 = "10.1038/s41586-024-07686-5"   # FlyWire whole-brain annotation (cell types, sides)
DOI_STURNER_2025 = "10.1038/s41586-025-08925-z"    # DN/AN matching FlyWire <-> light microscopy
DOI_SHIU_2024 = "10.1038/s41586-024-07763-9"       # whole-brain LIF model

BEHAVIORS = ("forward", "backward", "turn_left", "turn_right", "stop", "escape", "groom", "feed", "flight_power")
# flight labels of flylab.flight.classify_flight; for 'escape' ground truth any takeoff counts as escape (see evaluate())
FLIGHT_LABELS = ("no_takeoff", "takeoff_fall", "hover", "climb", "forward_flight", "flight_turn_left", "flight_turn_right")

__all__ = ["groups", "group_ids", "find_cell_type", "ground_truth", "neuron_info", "groups_for_behavior",
           "load_annotations", "build", "resolve_group", "evaluate", "FLIGHT_LABELS"]


# ----------------------------------------------------------------------------- loading
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _neurons_doc() -> dict:
    if not NEURONS_JSON.exists():
        raise FileNotFoundError(f"{NEURONS_JSON} missing - run `python -m flylab.atlas --build`")
    return _read_json(NEURONS_JSON)


def groups() -> dict[str, dict]:
    """All curated neuron groups: name -> group dict (contract fields + confidence/mapping metadata)."""
    return copy.deepcopy(_neurons_doc()["groups"])


_ALIASES = {
    "moonwalker": "MDN", "mdns": "MDN", "dnp09": "P9", "dnp09_l": "P9_L", "dnp09_r": "P9_R",
    "giant_fiber": "GF", "giant fiber": "GF", "giantfiber": "GF", "dnp01": "GF", "dnp01_l": "GF_L", "dnp01_r": "GF_R",
    "sugar": "sugar_GRN_shiu2024", "bitter": "bitter_GRN_shiu2024", "water": "water_GRN_shiu2024",
    "jo": "JO_CE_shiu2024", "johnston": "JO_CE_shiu2024", "jon": "JO_CE_shiu2024", "looming": "LPLC2",
    "sad093": "aBN1", "cb0701": "MN9", "adn1": "aDN1_shiu2024", "adn2": "aDN2_shiu2024",
}


def resolve_group(name: str) -> str:
    """Canonical group name for a name/alias (case-insensitive). Raises KeyError."""
    g = _neurons_doc()["groups"]
    if name in g:
        return name
    low = {k.lower(): k for k in g}
    key = name.strip().lower()
    if key in low:
        return low[key]
    if key in _ALIASES and _ALIASES[key] in g:
        return _ALIASES[key]
    sugg = [k for k in g if key.split("_")[0] in k.lower()]
    raise KeyError(f"unknown group {name!r}; similar: {sugg[:10]}; aliases: {sorted(_ALIASES)}")


def group_ids(name: str) -> list[int]:
    """Root IDs (FlyWire v783) of one group. Case-insensitive, accepts aliases (e.g. 'moonwalker'); KeyError otherwise."""
    return [int(i) for i in _neurons_doc()["groups"][resolve_group(name)]["root_ids"]]


@functools.lru_cache(maxsize=1)
def _ground_truth_doc() -> dict:
    if not GT_JSON.exists():
        raise FileNotFoundError(f"{GT_JSON} missing - run `python -m flylab.atlas --build`")
    return _read_json(GT_JSON)


def ground_truth() -> list[dict]:
    """Literature-verified neuron->behaviour relations (see data/ground_truth.json)."""
    return copy.deepcopy(_ground_truth_doc()["entries"])


BODY_BEHAVIORS = ("forward", "backward", "turn_left", "turn_right", "stop")


def evaluate(observed_behavior: str, ground_truth_id: str) -> dict:
    """Effect-aware comparison of an observed behaviour label with one ground-truth entry.

    IMPORTANT: entries with effect='reduce' (e.g. gt02 MDN silencing) predict that the behaviour is ABSENT/reduced,
    so observing expected_behavior there is INCONSISTENT. A naive `observed == expected_behavior` check gets these wrong.
    Returns {"verdict": "consistent"|"partially_consistent"|"inconsistent"|"not_comparable", "reason", "expected_behavior",
             "effect", "observed", "id", "readout_group"}.
    """
    gt = {e["id"]: e for e in ground_truth()}
    if ground_truth_id not in gt:
        raise KeyError(f"unknown ground_truth_id {ground_truth_id!r}; known: {sorted(gt)}")
    e = gt[ground_truth_id]
    obs = str(observed_behavior).strip().lower()
    exp = e["expected_behavior"]
    effect = e.get("effect", "induce")
    out = {"id": ground_truth_id, "observed": obs, "expected_behavior": exp, "effect": effect,
           "readout_group": e.get("readout_group")}
    if obs in FLIGHT_LABELS:
        # flight body labels are only comparable with 'escape' ground truth: takeoff (any label except
        # no_takeoff, including a fall after takeoff) counts as escape, no_takeoff as no escape.
        if exp != "escape":
            return {**out, "verdict": "not_comparable",
                    "reason": f"flight label '{obs}' can only be compared with escape (takeoff) ground truth, not '{exp}'"}
        obs = "escape" if obs != "no_takeoff" else "no_escape"
        out["observed_class"] = obs
    if exp not in BODY_BEHAVIORS and obs in BODY_BEHAVIORS:
        return {**out, "verdict": "not_comparable",
                "reason": f"'{exp}' cannot be produced by the walking body; compare brain-level firing of "
                          f"{e.get('readout_group') or 'a readout group'} instead"}
    if effect == "reduce":
        if obs == exp:
            return {**out, "verdict": "inconsistent", "reason": f"paper: manipulation reduces {exp}, but {exp} was observed"}
        return {**out, "verdict": "consistent", "reason": f"{exp} absent as predicted (observed {obs})"}
    if obs == exp:
        return {**out, "verdict": "consistent", "reason": "observed behaviour matches"}
    turns = {"turn_left", "turn_right"}
    if obs in turns and exp in turns:
        return {**out, "verdict": "inconsistent", "reason": "turning in the opposite direction"}
    if {obs, exp} <= {"forward"} | turns:
        return {**out, "verdict": "partially_consistent", "reason": "locomotion matches but the steering component differs"}
    return {**out, "verdict": "inconsistent", "reason": f"expected {exp}, observed {obs}"}


def groups_for_behavior(label: str) -> list[dict]:
    """Ground-truth entries whose expected_behavior == label (e.g. 'backward')."""
    return [e for e in ground_truth() if e["expected_behavior"] == label]


def _download(url: str, dest: Path, timeout: int = 180) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


@functools.lru_cache(maxsize=1)
def load_annotations():
    """FlyWire v783 neuron annotations as a pandas DataFrame (downloads ~32 MB on first use)."""
    import pandas as pd
    if not ANNOT_TSV.exists() or ANNOT_TSV.stat().st_size < 1_000_000:
        _download(ANNOT_URL, ANNOT_TSV)
    df = pd.read_csv(ANNOT_TSV, sep="\t", low_memory=False)
    df["root_id"] = df["root_id"].astype("int64")
    return df


def _side_short(s: Any) -> str:
    return {"left": "L", "right": "R", "center": "C"}.get(str(s), str(s))


def _clean(v: Any) -> Any:
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    except Exception:
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def find_cell_type(query: str, limit: int = 50) -> list[dict]:
    """Search annotations by cell_type / hemibrain_type / synonyms (case-insensitive).

    Exact matches come first, then prefix, then substring matches. One row per (cell_type, side):
    {"cell_type","hemibrain_type","synonyms","super_class","cell_class","cell_sub_class","side","n",
     "root_ids","top_nt","match"}.
    """
    q = query.strip().lower()
    if not q:
        return []
    curated: list[dict] = []
    try:
        gname = resolve_group(query.strip())
    except (KeyError, FileNotFoundError):
        gname = None
    if gname is not None:
        doc = _neurons_doc()["groups"]
        for name in [gname] + [f"{gname}_L", f"{gname}_R"]:
            if name in doc:
                g = doc[name]
                curated.append({"cell_type": g["cell_type"], "hemibrain_type": None, "synonyms": None, "super_class": None,
                                "cell_class": None, "cell_sub_class": None, "side": g["side"], "n": len(g["root_ids"]),
                                "root_ids": list(g["root_ids"]), "top_nt": None, "match": "curated_group", "group": name,
                                "confidence": g.get("confidence"), "description": g.get("description")})
    df = load_annotations()
    cols = ["cell_type", "hemibrain_type", "synonyms"]
    low = {c: df[c].fillna("").astype(str).str.lower() for c in cols}
    exact = (low["cell_type"] == q) | (low["hemibrain_type"] == q)
    prefix = low["cell_type"].str.startswith(q) | low["hemibrain_type"].str.startswith(q)
    sub = False
    for c in cols:
        sub = sub | low[c].str.contains(q, regex=False)
    hits = df[sub | exact | prefix].copy()
    if hits.empty:
        return curated[:limit]
    hits["_rank"] = 2
    hits.loc[prefix[hits.index], "_rank"] = 1
    hits.loc[exact[hits.index], "_rank"] = 0
    out = []
    keys = ["cell_type", "side"]
    for (ct, side), sdf in hits.groupby(keys, dropna=False, sort=False):
        first = sdf.iloc[0]
        out.append({
            "cell_type": _clean(ct), "hemibrain_type": _clean(first["hemibrain_type"]),
            "synonyms": _clean(first["synonyms"]), "super_class": _clean(first["super_class"]),
            "cell_class": _clean(first["cell_class"]), "cell_sub_class": _clean(first["cell_sub_class"]),
            "side": _side_short(side), "n": int(len(sdf)),
            "root_ids": [int(i) for i in sdf["root_id"]], "top_nt": _clean(first["top_nt"]),
            "match": ["exact", "prefix", "substring"][int(sdf["_rank"].min())],
        })
    out.sort(key=lambda r: (["exact", "prefix", "substring"].index(r["match"]), str(r["cell_type"]), r["side"]))
    return (curated + out)[:limit]


def neuron_info(root_id: int) -> dict | None:
    """Annotation row for one root ID (v783) or None."""
    df = load_annotations()
    row = df[df["root_id"] == int(root_id)]
    if row.empty:
        return None
    r = row.iloc[0]
    keys = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side",
            "top_nt", "nerve", "synonyms", "status"]
    return {k: _clean(r[k]) for k in keys}


# ----------------------------------------------------------------------------- curation (build)
# Each group spec: selector over the annotation table OR explicit Shiu et al. ID list, plus provenance.
_GROUP_SPECS: list[dict] = [
    # ---- locomotion descending neurons
    dict(name="MDN", sel={"cell_type": "MDN"}, role="descending", function="backward walking",
         description="Moonwalker descending neurons; activation drives backward walking.",
         citations=["10.1126/science.1249964", DOI_SCHLEGEL_2024],
         confidence=("high", "FlyWire cell_type and hemibrain_type are both literally 'MDN' (Schlegel et al. 2024 annotation).")),
    dict(name="P9", sel={"cell_type": "DNp09"}, role="descending", function="forward walking (with ipsilateral turning)",
         description="P9 = DNp09; activation initiates forward walking with ipsilateral turning.",
         citations=["10.1016/j.neuron.2020.07.032", "10.1038/s41586-024-07854-7", "10.1038/s41586-024-07523-9"],
         confidence=("high", "FlyWire cell_type 'DNp09'; P9 = DNp09 stated in Sapkal et al. 2024 Nature ('P9 (DNp09) that drives forward turning'). "
                     "Two DNp71 neurons whose hemibrain_type is 'DNp09' are deliberately excluded (FlyWire cell_type takes precedence).")),
    dict(name="DNa01", sel={"cell_type": "DNa01"}, role="descending", function="steering (sustained, low-gain)",
         description="DNa01 steering descending neuron (predicts sustained low-gain steering).",
         citations=["10.7554/elife.102230", DOI_STURNER_2025],
         confidence=("high", "FlyWire cell_type 'DNa01'. Root IDs are identical to those Rayshubskiy et al. 2025 (eLife, Methods, Europe PMC PMC12279373) "
                     "list for 'DNa01R 720575940644438551, and DNa01L 720575940627787609' in FAFB/FlyWire v783; the paper notes this pair is "
                     "distinct from the neurons '(incorrectly) labeled as DNa01 in hemibrain:v1.2.1' - which explains why FlyWire DNa01 has "
                     "hemibrain_type 'VES006' and FlyWire 'DNae001' carries the hemibrain label 'DNa01'. L/R sides also match the paper.")),
    dict(name="DNa02", sel={"cell_type": "DNa02"}, role="descending", function="steering (transient, high-gain)",
         description="DNa02 steering descending neuron; unilateral activation influences ipsilateral rotation.",
         citations=["10.7554/elife.102230", "10.1016/j.cell.2024.08.033", DOI_STURNER_2025],
         confidence=("high", "FlyWire cell_type and hemibrain_type both 'DNa02'; root IDs and L/R sides identical to Rayshubskiy et al. 2025 "
                     "(eLife, Methods: 'DNa02R 720575940604737708, DNa02L 720575940629327659' in FAFB/FlyWire v783).")),
    dict(name="GF", sel={"cell_type": "DNp01"}, role="descending", function="escape / takeoff",
         description="Giant fiber (DNp01); activation elicits jump/escape takeoff.",
         citations=["10.1016/j.cell.2005.02.004", "10.1038/nn.3741"],
         confidence=("high", "FlyWire cell_type 'DNp01' with hemibrain_type 'Giant Fiber'.")),
    dict(name="DNg02", sel={"cell_type_regex": r"^DNg02"}, role="descending", function="flight motor regulation (wingbeat amplitude, population code)",
         description="DNg02 descending neurons (cell types DNg02_a..h); projecting to the dorsal flight neuropil of the VNC; "
                     "activating more cells raises wingbeat amplitude (Namiki et al. 2022, Curr Biol).",
         citations=["10.1016/j.cub.2022.01.008", DOI_SCHLEGEL_2024],
         confidence=("medium", "FlyWire v783 cell_type 'DNg02_a' ... 'DNg02_h' (25 neurons, 13 left / 12 right, super_class descending, all in the "
                     "brain model). Namiki et al. 2022 (abstract) describe 'a population of at least 15 DNg02 cell pairs'; the FlyWire count is "
                     "lower, and we identify the population by cell-type name only (no morphological matching here).")),
    # ---- grooming descending neurons
    dict(name="DNg11", sel={"cell_type": "DNg11"}, role="descending", function="anterior grooming",
         description="DNg11; named among DNs for anterior grooming sequences (Stürner et al. 2025 intro, citing prior work).",
         citations=[DOI_STURNER_2025, "10.1038/s41586-024-07523-9"],
         confidence=("medium", "FlyWire cell_type 'DNg11' (3 per side). Behavioural role taken from a literature summary, not verified in a primary paper here.")),
    dict(name="DNg07", sel={"cell_type": "DNg07"}, role="descending", function="head grooming",
         description="DNg07; optogenetic activation produced head grooming in the Cande et al. 2018 DN screen.",
         citations=["10.7554/eLife.34275"],
         confidence=("medium", "FlyWire cell_type 'DNg07' (8 per side); Cande et al. used a split-GAL4 line named for DNg07, so the line may not cover all 8 FlyWire neurons.")),
    dict(name="aDN1_shiu2024", shiu_var="id_DN1_1", role="descending", function="antennal grooming",
         description="Antennal grooming descending neuron aDN1 exactly as used in Shiu et al. 2024 (Fig. 5c, figures.ipynb `id_DN1_1`).",
         citations=[DOI_SHIU_2024, "10.7554/eLife.08758", "10.1038/s41586-024-07523-9"],
         confidence=("medium", "ID copied from the Shiu et al. model code (v630 ID unchanged in v783; FlyWire cell_type DNg62). "
                     "The aDN1 label is Shiu's; Shiu notes 'exactly which aDNs are not known' for JON input.")),
    dict(name="aDN2_shiu2024", shiu_var="id_DN2_l", role="descending", function="antennal grooming",
         description="Antennal grooming descending neuron aDN2 as used in Shiu et al. 2024 (figures.ipynb `id_DN2_l`).",
         citations=[DOI_SHIU_2024, "10.1038/s41586-024-07523-9"],
         confidence=("medium", "ID copied from the Shiu et al. model code (unchanged in v783; FlyWire cell_type DNge078); aDN2 label is Shiu's.")),
    # ---- sensory groups usable as natural stimuli
    dict(name="sugar_GRN_shiu2024", shiu_var="neu_sugar", role="sensory", function="sugar taste (labellar GRNs)",
         description="Labellar sugar GRNs exactly as stimulated in Shiu et al. 2024 (20 of 21 v630 IDs persist in v783).",
         citations=[DOI_SHIU_2024],
         confidence=("high", "IDs copied from the Shiu et al. model code. Note: Shiu called these 'right hemisphere'; the v783 annotation "
                     "(nerve-entry side) says 'left' and sub-classifies them as LB3c sugar (9), LB3b sugar/low_salt (2), LB3d high_salt (5), LB4b putative_attractive (4).")),
    dict(name="sugar_GRN", sel={"cell_sub_class": "sugar", "cell_class": "gustatory"}, role="sensory", function="sugar taste",
         description="All GRNs annotated cell_sub_class 'sugar' (LB3c) in FlyWire v783.",
         citations=[DOI_SCHLEGEL_2024, DOI_SHIU_2024],
         confidence=("medium", "Annotation-based (cell_sub_class == 'sugar'); differs from the Shiu et al. stimulation set.")),
    dict(name="bitter_GRN_shiu2024", shiu_var="neu_bitter", role="sensory", function="bitter taste (labellar GRNs)",
         description="Labellar bitter GRNs as used in Shiu et al. 2024 Fig. 3 (20 of 21 IDs persist in v783).",
         citations=[DOI_SHIU_2024],
         confidence=("high", "IDs copied from the Shiu et al. model code; all annotated as bitter (LB1a-d) in v783.")),
    dict(name="bitter_GRN", sel={"cell_sub_class": "bitter", "cell_class": "gustatory"}, role="sensory", function="bitter taste",
         description="All GRNs annotated cell_sub_class 'bitter' in FlyWire v783.",
         citations=[DOI_SCHLEGEL_2024],
         confidence=("high", "Annotation-based (cell_sub_class == 'bitter').")),
    dict(name="water_GRN_shiu2024", shiu_var="neu_water", role="sensory", function="water taste (labellar GRNs)",
         description="Labellar water GRNs as used in Shiu et al. 2024 Fig. 4 (18/18 IDs persist in v783).",
         citations=[DOI_SHIU_2024],
         confidence=("medium", "IDs copied from the Shiu et al. model code, but the v783 annotation labels 11 as water (LB3a) and 7 as sugar (LB3c).")),
    dict(name="water_GRN", sel={"cell_sub_class": "water", "cell_class": "gustatory"}, role="sensory", function="water taste",
         description="All GRNs annotated cell_sub_class 'water' (LB3a) in FlyWire v783.",
         citations=[DOI_SCHLEGEL_2024],
         confidence=("high", "Annotation-based (cell_sub_class == 'water').")),
    dict(name="Ir94e_GRN_shiu2024", shiu_var="neu_ir94e", role="sensory", function="Ir94e taste (labellar GRNs)",
         description="Ir94e GRNs as used in Shiu et al. 2024 Fig. 3b.",
         citations=[DOI_SHIU_2024],
         confidence=("medium", "IDs copied from the Shiu et al. model code; v783 annotation labels them LB1e (glutamate) and LB2a-c (putative_aversive).")),
    dict(name="JO_CE_shiu2024", shiu_var="neu_JON_CE", role="sensory", function="antennal mechanosensation (Johnston's organ C/E)",
         description="Johnston's organ JO-C/E neurons as used in Shiu et al. 2024 Fig. 5 (69 of 70 IDs persist).",
         citations=[DOI_SHIU_2024, "10.7554/eLife.08758"],
         confidence=("high", "IDs copied from the Shiu et al. model code; v783 annotation: JO-C*/JO-E* wind_gravity.")),
    dict(name="JO_F_shiu2024", shiu_var="neu_JON_F", role="sensory", function="antennal mechanosensation (Johnston's organ F)",
         description="Johnston's organ JO-F neurons as used in Shiu et al. 2024 Fig. 5.",
         citations=[DOI_SHIU_2024, "10.7554/eLife.08758"],
         confidence=("high", "IDs copied from the Shiu et al. model code; v783 annotation: JO-F* (cell_sub_class 'grooming').")),
    dict(name="JO_CE", sel={"cell_type_regex": r"^JO-[CE]"}, role="sensory", function="antennal mechanosensation (JO-C/E, wind/gravity)",
         description="All JO-C* and JO-E* neurons in the FlyWire v783 annotation.",
         citations=[DOI_SCHLEGEL_2024, "10.7554/eLife.08758"],
         confidence=("high", "Annotation-based (cell_type starts with JO-C or JO-E).")),
    dict(name="JO_all", sel={"cell_type_regex": r"^JO-"}, role="sensory", function="antennal mechanosensation (all Johnston's organ neurons)",
         description="All Johnston's organ neurons (JO-*) in the FlyWire v783 annotation.",
         citations=[DOI_SCHLEGEL_2024],
         confidence=("high", "Annotation-based (cell_type starts with 'JO-').")),
    dict(name="LPLC2", sel={"cell_type": "LPLC2"}, role="sensory", function="looming detection (visual projection neurons)",
         description="LPLC2 lobula plate/lobula columnar neurons; looming detectors upstream of the giant fiber.",
         citations=["10.1016/j.cub.2019.01.079", "10.7554/eLife.21022"],
         confidence=("high", "FlyWire cell_type and hemibrain_type both 'LPLC2' (super_class visual_projection).")),
    dict(name="LC16", sel={"cell_type": "LC16"}, role="sensory", function="looming-responsive visual projection neurons (backward walking)",
         description="Lobula columnar 16 visual projection neurons; respond to looming, bilateral activation evokes backward walking "
                     "(Wu et al. 2016), proposed to act via MDN (Sen et al. 2017).",
         citations=["10.7554/eLife.21022", "10.1016/j.cub.2017.02.008"],
         confidence=("high", "FlyWire cell_type and hemibrain_type both 'LC16' (super_class visual_projection, 151 neurons).")),
    dict(name="LC4", sel={"cell_type": "LC4"}, role="sensory", function="looming-responsive visual projection neurons (escape / GF input)",
         description="Lobula columnar 4 visual projection neurons; activation evokes jumping (Wu et al. 2016); "
                     "a visual projection input to the giant fiber escape circuit (von Reyn et al. 2017).",
         citations=["10.7554/eLife.21022", "10.1016/j.neuron.2017.05.036", "10.1016/j.cub.2019.01.079"],
         confidence=("high", "FlyWire cell_type and hemibrain_type both 'LC4' (super_class visual_projection, 104 neurons). "
                     "LC4 -> GF: Ache et al. 2019 abstract (verbatim) 'show that LPLC2 and LC4 synapse directly onto the GF' and "
                     "attributes looming-velocity input to LC4 (their earlier work, von Reyn et al. 2017, whose abstract only says "
                     "'a visual projection neuron type').")),
    dict(name="LC6", sel={"cell_type": "LC6"}, role="sensory", function="looming-responsive visual projection neurons (jumping)",
         description="Lobula columnar 6 visual projection neurons; respond to looming, activation evokes highly penetrant jumping (Wu et al. 2016).",
         citations=["10.7554/eLife.21022"],
         confidence=("high", "FlyWire cell_type and hemibrain_type both 'LC6' (super_class visual_projection, 125 neurons).")),
    # ---- interneurons / motor
    dict(name="aBN1", sel={"cell_type": "SAD093"}, role="interneuron", function="antennal grooming command interneuron",
         description="aBN1 antennal grooming brain interneuron (FlyWire SAD093; one per hemisphere).",
         citations=["10.7554/eLife.08758", DOI_SHIU_2024],
         confidence=("medium", "Shiu et al. figures.ipynb `id_aBN1` (720575940630907434) is cell_type SAD093 in v783; we include the contralateral SAD093 "
                     "homologue (flagged status 'outlier_seg' in the annotation).")),
    dict(name="MN9", sel={"cell_type": "CB0701"}, role="motor", function="proboscis (rostrum) extension motor neuron",
         description="MN9 proboscis motor neuron (controls rostrum lifting); readout for feeding initiation in Shiu et al. 2024.",
         citations=[DOI_SHIU_2024, "10.1016/j.neuron.2008.12.033", "10.7554/elife.79887"],
         confidence=("high", "Shiu et al. MN9 ID 720575940660219265 is unchanged in v783 and annotated as CB0701 (ingestion_motor_neuron, PhN); "
                     "the other CB0701 is its contralateral homologue. Shiu called 720575940660219265 'left'; v783 annotation says 'right'.")),
]

# Groups that are split into _L / _R in addition to the bilateral group.
_SPLIT_SIDES = {"MDN", "P9", "DNa01", "DNa02", "GF", "DNg02", "DNg11", "DNg07", "sugar_GRN", "bitter_GRN", "water_GRN",
                "JO_CE", "LPLC2", "LC16", "LC4", "LC6", "aBN1", "MN9"}


def _shiu_lists() -> dict[str, list[int]]:
    """Neuron ID lists from the Shiu et al. Drosophila_brain_model figures.ipynb (local copy or GitHub)."""
    local = RAW / "figures.ipynb"
    cached = CACHE / "shiu_figures.ipynb"
    if local.exists():
        nb = _read_json(local)
    else:
        if not cached.exists():
            _download(SHIU_FIGURES_URL, cached, timeout=60)
        nb = _read_json(cached)
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    lists: dict[str, list[int]] = {}
    for m in re.finditer(r"(\w+)\s*=\s*\[([^\]]*?)\]", src):
        ids = [int(x) for x in re.findall(r"\b7205759\d{11}\b", m.group(2))]
        if ids:
            lists.setdefault(m.group(1), ids)
    for m in re.finditer(r"(\w+)\s*=\s*(7205759\d{11})\b", src):
        lists.setdefault(m.group(1), [int(m.group(2))])
    return lists


def _select(df, sel: dict):
    m = df["root_id"].notna()
    for k, v in sel.items():
        if k == "cell_type_regex":
            m &= df["cell_type"].fillna("").str.contains(v, regex=True)
        else:
            m &= df[k] == v
    return df[m]


def _brain_model_ids() -> set[int] | None:
    if COMPLETENESS_783.exists():
        import pandas as pd
        c = pd.read_csv(COMPLETENESS_783)
        return set(c.iloc[:, 0].astype("int64"))
    return None


def _build_groups() -> dict:
    df = load_annotations()
    shiu = _shiu_lists()
    model_ids = _brain_model_ids()
    out: dict[str, dict] = {}
    for spec in _GROUP_SPECS:
        if "shiu_var" in spec:
            wanted = shiu[spec["shiu_var"]]
            sub = df[df["root_id"].isin(wanted)]
            source = f"{SHIU_REPO} figures.ipynb variable `{spec['shiu_var']}` ({len(wanted)} IDs; {len(sub)} present in v783 annotation)"
        else:
            sub = _select(df, spec["sel"])
            source = f"{ANNOT_SOURCE}; selector {spec['sel']}"
        if sub.empty:
            print(f"WARNING: group {spec['name']} is empty", file=sys.stderr)
            continue
        variants = [("both", sub, spec["name"])]
        if spec["name"] in _SPLIT_SIDES:
            for side, short in (("left", "L"), ("right", "R")):
                s = sub[sub["side"] == side]
                if not s.empty:
                    variants.append((short, s, f"{spec['name']}_{short}"))
        for side, s, name in variants:
            excluded: list[int] = []
            if model_ids is not None:
                excluded = sorted(int(i) for i in s["root_id"] if int(i) not in model_ids)
                s = s[s["root_id"].isin(model_ids)]
            ids = sorted(int(i) for i in s["root_id"])
            cts = s["cell_type"].fillna("NA").value_counts().to_dict()
            sides = sorted(set(_side_short(x) for x in s["side"]))
            g = {
                "root_ids": ids,
                "cell_type": "/".join(sorted(cts)) if len(cts) <= 4 else f"{len(cts)} types",
                "side": side if side != "both" else ("both" if len(sides) > 1 else sides[0]),
                "role": spec["role"],
                "function": spec["function"],
                "description": spec["description"] + ("" if side == "both" else f" ({'left' if side == 'L' else 'right'} side only)"),
                "citations": spec["citations"],
                "n": len(ids),
                "cell_type_counts": {k: int(v) for k, v in cts.items()},
                "sides_present": sides,
                "confidence": {"level": spec["confidence"][0], "reason": spec["confidence"][1]},
                "mapping_source": source,
                "side_convention": "FlyWire v783 annotation 'side' (soma side for brain neurons, nerve-entry side for sensory neurons).",
            }
            if model_ids is not None:
                g["brain_model_check"] = "all root_ids are present in Completeness_783.csv (Shiu et al. model neuron list)"
                if excluded:
                    g["excluded_not_in_brain_model"] = excluded
            out[name] = g
    return out


# Ground-truth specs. 'evidence' MUST be a verbatim fragment of the abstract (checked at build time).
_GT_SPECS: list[dict] = [
    dict(id="gt01_mdn_activate_backward", manipulation="activate", target_group="MDN", expected_behavior="backward", effect="induce",
         doi="10.1126/science.1249964",
         evidence="sufficient to trigger backward walking under conditions in which flies would otherwise walk forward",
         confidence="high"),
    dict(id="gt02_mdn_silence_backward", manipulation="silence", target_group="MDN", expected_behavior="backward", effect="reduce",
         context="fly encounters an impassable barrier (would normally back up)",
         doi="10.1126/science.1249964",
         evidence="MDN activity is required for flies to walk backward when they encounter an impassable barrier",
         confidence="high"),
    dict(id="gt03_p9_activate_forward", manipulation="activate", target_group="P9", expected_behavior="forward", effect="induce",
         doi="10.1016/j.neuron.2020.07.032",
         evidence="P9 drives forward walking with ipsilateral turning",
         confidence="high"),
    dict(id="gt04_p9L_activate_turn_left", manipulation="activate", target_group="P9_L", expected_behavior="turn_left", effect="induce",
         context="unilateral activation; forward walking with ipsilateral turning",
         doi="10.1016/j.neuron.2020.07.032",
         evidence="P9 drives forward walking with ipsilateral turning",
         confidence="medium",
         note="Abstract says 'ipsilateral turning'; that unilateral (left-only) activation yields a left turn is our inference."),
    dict(id="gt05_dna02L_activate_turn_left", manipulation="activate", target_group="DNa02_L", expected_behavior="turn_left", effect="induce",
         doi="10.7554/elife.102230",
         evidence="descending neurons in the Drosophila brain that predict and influence orientation (heading) during walking",
         evidence_fulltext="unilateral activation of DNa02 influences ipsilateral rotational movements",
         confidence="medium",
         note="Ipsilateral direction from the full text (Europe PMC PMC12279373); reviewers note activation does not always reliably evoke turning."),
    dict(id="gt06_dna02R_activate_turn_right", manipulation="activate", target_group="DNa02_R", expected_behavior="turn_right", effect="induce",
         doi="10.7554/elife.102230",
         evidence="descending neurons in the Drosophila brain that predict and influence orientation (heading) during walking",
         evidence_fulltext="unilateral activation of DNa02 influences ipsilateral rotational movements",
         confidence="medium",
         note="Mirror of gt05."),
    dict(id="gt07_gf_activate_escape", manipulation="activate", target_group="GF", expected_behavior="escape", effect="induce",
         doi="10.1016/j.cell.2005.02.004",
         evidence="Photostimulation of neurons in the giant fiber system elicited the characteristic escape behaviors of jumping, wing beating, and flight",
         confidence="high",
         note="Lima & Miesenboeck targeted the giant fiber system (GF plus partners), not GF alone."),
    dict(id="gt08_lplc2_activate_escape", manipulation="activate", target_group="LPLC2", expected_behavior="escape", effect="induce",
         doi="10.7554/eLife.21022",
         evidence="for several types, optogenetic activation in freely moving flies evokes specific behaviors",
         evidence_fulltext="LPLC2 activation elicited both jumping and backward walking behaviors with about equal penetrance in the arena assay.",
         confidence="high",
         readout_group="GF",
         note="LPLC2-specific statement is in the full text (jumping = escape takeoff). The abstract's 'two LC types' resembling loom avoidance "
              "are LC6/LC16, not LPLC2 (full text: not known 'whether LPLC1, LPLC2, LC4 and LC15 are indeed sensitive to looming'). "
              "Brain-model readout suggestion: giant fiber (DNp01) firing; LPLC2 synapses directly onto GF (Ache et al. 2019)."),
    dict(id="gt19_lplc2_activate_backward", manipulation="activate", target_group="LPLC2", expected_behavior="backward", effect="induce",
         doi="10.7554/eLife.21022",
         evidence="for several types, optogenetic activation in freely moving flies evokes specific behaviors",
         evidence_fulltext="LPLC2 activation elicited both jumping and backward walking behaviors with about equal penetrance in the arena assay.",
         confidence="medium",
         readout_group="MDN",
         note="Body-level counterpart of gt08 (the physics body cannot jump). Backward walking and jumping occurred with about equal "
              "penetrance, so either outcome is consistent with the paper; the MDN readout is our suggestion, not a claim of the paper."),
    dict(id="gt09_lplc2_silence_escape", manipulation="silence", target_group="LPLC2", expected_behavior="escape", effect="reduce",
         context="looming stimulus / GF-mediated escape",
         doi="10.1016/j.cub.2019.01.079",
         evidence="We find LPLC2 neurons to be necessary for GF-mediated escape",
         confidence="high"),
    dict(id="gt10_sugar_activate_feed", manipulation="activate", target_group="sugar_GRN_shiu2024", expected_behavior="feed", effect="induce",
         doi="10.1016/j.neuron.2008.12.033",
         evidence="The motor neurons are activated by sugar stimulation of gustatory neurons and inhibited by bitter compounds",
         confidence="high",
         readout_group="MN9",
         note="Model readout: MN9 firing (Shiu et al. 2024 use MN9 as proxy for proboscis extension). Laterality: Shiu et al. 2024 "
              "(doi:10.1038/s41586-024-07763-9, full text) report that 'unilateral sugar GRN activation activates the contralateral MN9 "
              "more strongly compared with the ipsilateral MN9'. With v783 sides, sugar_GRN_shiu2024 is L, so expect MN9_R > MN9_L "
              "(reviewer smoke test, 150 Hz: MN9_R 98 Hz vs MN9_L 62 Hz)."),
    dict(id="gt11_bitter_activate_reduce_feed", manipulation="activate", target_group="bitter_GRN_shiu2024", expected_behavior="feed", effect="reduce",
         context="co-activation with sugar GRNs; bitter suppresses sugar-evoked feeding initiation",
         doi="10.7554/elife.79887",
         evidence="the feeding initiation circuit is inhibited by a bitter taste pathway that impinges on premotor neurons",
         confidence="high",
         readout_group="MN9"),
    dict(id="gt12_mn9_activate_feed", manipulation="activate", target_group="MN9", expected_behavior="feed", effect="induce",
         doi="10.1016/j.neuron.2008.12.033",
         evidence="By silencing and activating subsets of the defined cell population, we identify the neurons involved in the taste behavior as a pair of motor neurons",
         confidence="medium",
         note="Identity of these motor neurons with FlyWire MN9 (CB0701) relies on Shiu et al. 2024, which cites MN9 as controlling rostrum lifting."),
    dict(id="gt13_water_activate_feed", manipulation="activate", target_group="water_GRN_shiu2024", expected_behavior="feed", effect="induce",
         context="thirsty fly",
         doi=DOI_SHIU_2024,
         evidence="activation of sugar-sensing or water-sensing gustatory neurons in the computational model accurately predicts neurons that respond to tastes and are required for feeding initiation",
         confidence="medium",
         readout_group="MN9",
         note="Supporting statement is about the model's agreement with experiments, not a direct behavioural experiment."),
    dict(id="gt14_jo_activate_groom", manipulation="activate", target_group="JO_CE_shiu2024", expected_behavior="groom", effect="induce",
         doi="10.7554/eLife.08758",
         evidence="This multilayered circuit is organized such that neurons within each layer are sufficient to specifically elicit antennal grooming",
         confidence="medium",
         readout_group="aDN1_shiu2024",
         note="The first circuit layer are the antennal mechanosensory chordotonal (Johnston's organ) neurons ('Mechanosensory chordotonal neurons "
              "detect displacements of the antennae and excite three different classes of functionally connected interneurons'). "
              "That the JO-C/E subset used by Shiu et al. is the sufficient population is our mapping (medium)."),
    dict(id="gt15_abn1_activate_groom", manipulation="activate", target_group="aBN1", expected_behavior="groom", effect="induce",
         doi="10.7554/eLife.08758",
         evidence="neurons within each layer are sufficient to specifically elicit antennal grooming",
         confidence="medium",
         readout_group="aDN1_shiu2024",
         note="aBN1 = SAD093 mapping via Shiu et al. figures.ipynb `id_aBN1`."),
    dict(id="gt16_dng07_activate_groom", manipulation="activate", target_group="DNg07", expected_behavior="groom", effect="induce",
         doi="10.7554/eLife.34275",
         evidence="activation of most of the descending neurons drove stereotyped behaviors",
         evidence_fulltext="we found descending neurons that produced different types of grooming, such as head grooming (DNg07 and DNg08, and DNg12)",
         confidence="medium",
         note="Cande et al. 2018 DN screen (head grooming); split-GAL4 line vs. all 8 FlyWire DNg07 per side."),
    dict(id="gt17_p9_activate_forward_braun", manipulation="activate", target_group="P9", expected_behavior="forward", effect="induce",
         doi="10.1038/s41586-024-07523-9",
         evidence="command-like neurons that are sufficient to drive behaviours",
         evidence_fulltext="Flies reliably (i) walk forward upon DNp09 stimulation for stimuli ≥ 21 μ W",
         confidence="high",
         note="Independent replication (Braun et al. 2024, Ramdya lab) of DNp09->forward and MDN->backward."),
    dict(id="gt18_mdn_activate_backward_braun", manipulation="activate", target_group="MDN", expected_behavior="backward", effect="induce",
         doi="10.1038/s41586-024-07523-9",
         evidence="command-like neurons that are sufficient to drive behaviours",
         evidence_fulltext="(iii) walk backward upon MDN stimulation for stimuli ≥ 10.5 μ W",
         confidence="high"),
    # ---- visual projection neurons (added for the discovery screen, flylab/screen.py)
    dict(id="gt20_lc16_activate_backward", manipulation="activate", target_group="LC16", expected_behavior="backward", effect="induce",
         doi="10.7554/eLife.21022",
         evidence="The activation phenotypes of two LC types closely resemble natural avoidance behaviors triggered by a visual loom",
         evidence_fulltext="this variability does not change our conclusion that LC16 activation results in a strong backward walking response",
         confidence="high",
         readout_group="MDN",
         note="Full text (Europe PMC PMC5293491): bilateral activation; 'Unilateral LC16 activation produced far less backward walking than "
              "bilateral activation' and instead turning. The MDN readout is supported by Sen et al. 2017 (gt21)."),
    dict(id="gt21_lc16_activate_backward_via_mdn", manipulation="activate", target_group="LC16", expected_behavior="backward", effect="induce",
         doi="10.1016/j.cub.2017.02.008",
         evidence="LC16 and MDNs are critical components of the neural circuit that transduces threatening visual stimuli into directional locomotor output",
         confidence="high",
         readout_group="MDN",
         note="Abstract: 'we hypothesized that LC16 neurons induce backward walking via MDNs' and reports 'functional imaging, behavioral "
              "epistasis, and unilateral activation experiments that support these hypotheses'. Mechanistic prediction for the brain model: "
              "LC16 activation should drive MDN firing."),
    dict(id="gt22_lc4_activate_escape", manipulation="activate", target_group="LC4", expected_behavior="escape", effect="induce",
         doi="10.7554/eLife.21022",
         evidence="for several types, optogenetic activation in freely moving flies evokes specific behaviors",
         evidence_fulltext="five different cell types (LC4, LC6, LC15, LPLC1 and LPLC2) drove highly penetrant jumping in at least one of the two assays",
         confidence="medium",
         readout_group="GF",
         note="Jumping = escape takeoff; Wu et al. did not test whether the jumps are GF-mediated (full text: 'LC4 neurons ... might "
              "evoke a jumping response via activation of the Giant Fiber (GF) cells'). Supporting GF link: Ache et al. 2019 "
              "(doi:10.1016/j.cub.2019.01.079) abstract: 'show that LPLC2 and LC4 synapse directly onto the GF' (anatomy, EM); "
              "von Reyn et al. 2017 (Neuron, doi:10.1016/j.neuron.2017.05.036) abstract: 'we identify a visual projection neuron type "
              "that conveys predator approach information to the Drosophila giant fiber (GF) escape circuit'. GF readout is our suggestion."),
    dict(id="gt23_lc6_activate_escape", manipulation="activate", target_group="LC6", expected_behavior="escape", effect="induce",
         doi="10.7554/eLife.21022",
         evidence="The activation phenotypes of two LC types closely resemble natural avoidance behaviors triggered by a visual loom",
         evidence_fulltext="three LC neuron driver lines that produced robust and highly penetrant activation phenotypes in both assays: LC6 (jumping)",
         confidence="high",
         readout_group="GF",
         note="The two loom-like LC types of the abstract are LC6 (jumping) and LC16 (backward walking) per the full text. GF readout is our "
              "suggestion; Wu et al. did not test GF dependence."),
    # ---- flight (added for the embodied flight validation, flylab/flight.py)
    dict(id="gt24_gf_activate_takeoff_vonreyn", manipulation="activate", target_group="GF", expected_behavior="escape", effect="induce",
         context="looming-evoked escape takeoff (head-fixed GF recordings)",
         doi="10.1038/nn.3741",
         evidence="the GF circuit has a higher activation threshold than the parallel circuits, but can override ongoing behavior to force a short takeoff",
         confidence="medium",
         readout_group="GF",
         note="von Reyn et al. 2014 recorded the GF during looming-evoked escape (spike timing selects short vs. long takeoff); it is not a direct "
              "GF-activation experiment (that is Lima & Miesenboeck 2005, gt07). In the paper the GF-driven short takeoff sacrifices flight "
              "stability and the parallel (non-GF) circuits give the long takeoff that initiates stable flight - the model's single 'takeoff' "
              "label does not distinguish the two modes."),
    dict(id="gt25_lplc2_gf_input_ache", manipulation="activate", target_group="LPLC2", expected_behavior="escape", effect="induce",
         context="anatomical GF input (EM); behavioural LPLC2 activation -> jumping is gt08 (Wu et al. 2016)",
         doi="10.1016/j.cub.2019.01.079",
         evidence="LPLC2 and LC4 synapse directly onto the GF",
         confidence="medium",
         readout_group="GF",
         note="Ache et al. 2019 abstract: anatomical (EM reconstruction) direct LPLC2 -> GF synapses and LPLC2 necessary for GF-mediated escape "
              "(gt09); the abstract contains no LPLC2 activation experiment. The mechanistic prediction for the brain model is: LPLC2 activation "
              "drives GF firing and thereby takeoff; the behavioural activation result (jumping) is Wu et al. 2016 (gt08)."),
    dict(id="gt26_dng02_activate_wingbeat_amplitude", manipulation="activate", target_group="DNg02", expected_behavior="flight_power", effect="induce",
         context="flying fly (optogenetic activation of different numbers of DNg02 cells)",
         doi="10.1016/j.cub.2022.01.008",
         evidence="these neurons regulate wingbeat amplitude over a wide dynamic range via a population code",
         confidence="medium",
         note="Namiki et al. 2022 (Curr Biol): population of DNg02 cells; more activated cells -> larger wingbeat amplitude. Our bridge uses exactly "
              "this population code (mean activation over all DNg02 neurons), so a body check against this entry is partly circular. "
              "FlyWire v783 holds 25 DNg02 neurons vs. 'at least 15 cell pairs' in the paper (medium confidence in the identification)."),
]


_BEHAVIOR_TEXT = {"forward": "forward walking", "backward": "backward walking", "turn_left": "left turning",
                  "turn_right": "right turning", "stop": "stopping", "escape": "escape (takeoff/jump)",
                  "groom": "grooming", "feed": "feeding initiation (proboscis extension)",
                  "flight_power": "increased wingbeat amplitude (flight power)"}


def _outcome_text(manipulation: str, behavior: str, effect: str) -> str:
    b = _BEHAVIOR_TEXT.get(behavior, behavior)
    verb = "activating" if manipulation == "activate" else "silencing"
    if effect == "reduce":
        return f"{verb} the target group REDUCES/abolishes {b} (compare against a control without the manipulation)"
    return f"{verb} the target group evokes/increases {b}"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("‐", "-")).strip()


def _build_ground_truth(group_names: set[str]) -> list[dict]:
    from flylab import literature
    out = []
    for spec in _GT_SPECS:
        assert spec["expected_behavior"] in BEHAVIORS, spec["id"]
        assert spec["target_group"] in group_names, f"{spec['id']}: unknown group {spec['target_group']}"
        assert len(spec["evidence"].split()) <= 25, f"{spec['id']}: evidence > 25 words"
        rec = literature.get_by_doi(spec["doi"])
        if rec is None:
            print(f"WARNING: DOI {spec['doi']} not verified - skipping {spec['id']}", file=sys.stderr)
            continue
        in_abstract = _norm_text(spec["evidence"]) in _norm_text(rec.get("abstract", ""))
        if not in_abstract:
            print(f"WARNING: evidence for {spec['id']} not found verbatim in abstract", file=sys.stderr)
        e = {
            "id": spec["id"],
            "manipulation": spec["manipulation"],
            "target_group": spec["target_group"],
            "expected_behavior": spec["expected_behavior"],
            "effect": spec["effect"],
            "expected_outcome": _outcome_text(spec["manipulation"], spec["expected_behavior"], spec["effect"]),
            "evidence": spec["evidence"],
            "evidence_in_abstract": bool(in_abstract),
            "citation": {
                "doi": spec["doi"],
                "title": re.sub(r"<[^>]+>", "", rec["title"]).rstrip("."),
                "year": rec["year"],
                "authors": rec["authors"],
                "url": f"https://doi.org/{spec['doi']}",
                "pmid": rec.get("pmid") or None,
                "verified_by": rec.get("verified_by", []),
            },
            "confidence": spec["confidence"],
        }
        for k in ("context", "evidence_fulltext", "readout_group", "note"):
            if k in spec:
                e[k] = spec[k]
        out.append(e)
    return out


def build() -> dict:
    """Re-curate data/neurons.json and data/ground_truth.json from the raw annotation + verified literature."""
    t0 = time.time()
    g = _build_groups()
    now = time.strftime("%Y-%m-%d %H:%M")
    neurons_doc = {
        "_meta": {
            "description": "Curated FlyWire v783 neuron groups for the Embodied Fly Lab (built by flylab/atlas.py --build).",
            "root_id_version": "FlyWire materialization 783 (same ID space as Shiu et al. 2024 model Completeness_783.csv)",
            "annotation_source": ANNOT_SOURCE,
            "annotation_citations": [DOI_SCHLEGEL_2024, "10.1038/s41586-024-07558-y", DOI_STURNER_2025],
            "shiu_model_code": SHIU_REPO,
            "built": now,
            "notes": [
                "Groups ending in _shiu2024 reuse the exact neuron IDs from the Shiu et al. model code (v630 IDs that persist unchanged in v783).",
                "Left/right: Shiu et al. (v630 era) label the sugar/bitter/water sets 'right hemisphere'; the v783 annotation 'side' for the same IDs is 'left'. "
                "We always report the v783 annotation side.",
                "Every group carries confidence {level, reason} and mapping_source.",
                "Side cross-check: DNa01/DNa02 _L/_R root IDs equal the v783 IDs that Rayshubskiy et al. 2025 (eLife) give for DNa01L/R and "
                "DNa02L/R, so our L/R labels agree with that paper. The Shiu-era 'right' vs v783 'left' discrepancy is consistent for every "
                "Shiu set (sugar, bitter, water, JON, MN9, aDN2 'id_DN2_l'), i.e. a global convention change between v630-era labels and the "
                "v783 annotation, not a per-neuron error (probable cause: FAFB left/right inversion; not verified here). The published "
                "Shiu et al. 2024 paper describes Fig. 1c as 'unilateral left hemisphere sugar GRN activation' - presumably the 21-GRN main "
                "set that the notebook calls 'right' and v783 labels left (not verified figure-by-figure).",
                "Brain-model smoke test (spikes/atlas/brain_readouts.py, 150 Hz, 2x500 ms): sugar_GRN_shiu2024 -> MN9 fires (both sides), "
                "bitter_GRN_shiu2024 -> MN9 silent, LPLC2 -> GF fires, JO_CE_shiu2024 -> aBN1/aDN1/aDN2 fire, aBN1 -> aDN1/aDN2 fire.",
            ],
        },
        "groups": g,
    }
    gt = _build_ground_truth(set(g))
    gt_doc = {
        "_meta": {
            "description": "Literature-verified neuron -> behaviour relations. Each DOI was resolved via Europe PMC/OpenAlex "
                           "(flylab.literature.get_by_doi); 'evidence' is a verbatim abstract fragment (<=25 words), "
                           "'evidence_fulltext' (optional) a verbatim full-text sentence fragment.",
            "behaviors": list(BEHAVIORS),
            "effect": "'induce' = manipulation evokes/increases the behaviour; 'reduce' = manipulation abolishes/decreases it. "
                      "For 'reduce' entries, observing expected_behavior is INCONSISTENT with the paper - use atlas.evaluate().",
            "built": now,
        },
        "entries": gt,
    }
    NEURONS_JSON.write_text(json.dumps(neurons_doc, indent=1, ensure_ascii=False), encoding="utf-8")
    GT_JSON.write_text(json.dumps(gt_doc, indent=1, ensure_ascii=False), encoding="utf-8")
    _neurons_doc.cache_clear()
    _ground_truth_doc.cache_clear()
    return {"groups": len(g), "ground_truth": len(gt), "runtime_s": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- CLI
def _summary() -> None:
    gs = groups()
    print(f"Neuron groups ({len(gs)})  [FlyWire v783 root IDs]")
    print(f"{'name':24s} {'n':>4s} {'side':4s} {'role':11s} {'conf':6s} cell_type")
    for name, g in gs.items():
        conf = g.get("confidence", {}).get("level", "?")
        print(f"{name:24s} {g['n'] if 'n' in g else len(g['root_ids']):>4d} {g['side']:4s} {g['role']:11s} {conf:6s} {g['cell_type']}")
    gt = ground_truth()
    print(f"\nGround truth ({len(gt)} entries)")
    for e in gt:
        c = e["citation"]
        first = (c.get("authors") or "").split(",")[0]
        eff = "REDUCES" if e.get("effect") == "reduce" else "induces"
        print(f"{e['id']:34s} {e['manipulation']:8s} {e['target_group']:20s} -> {eff:7s} {e['expected_behavior']:10s} "
              f"[{first} {c.get('year')}, doi:{c['doi']}] conf={e['confidence']}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="FlyWire v783 neuron atlas + literature ground truth")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--build", action="store_true", help="rebuild data/neurons.json + data/ground_truth.json")
    ap.add_argument("--find", metavar="QUERY")
    ap.add_argument("--group", metavar="NAME")
    a = ap.parse_args()
    if a.build:
        print(build())
    if a.find:
        for r in find_cell_type(a.find):
            print(f"{str(r['cell_type']):14s} {r['side']} n={r['n']:<4d} hb={r['hemibrain_type']} class={r['super_class']}/{r['cell_class']}/{r['cell_sub_class']} "
                  f"syn={r['synonyms']} [{r['match']}]")
    if a.group:
        print(json.dumps(groups()[resolve_group(a.group)], indent=1, ensure_ascii=False))
    if a.summary or not (a.build or a.find or a.group):
        _summary()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    _main()
