"""Export recorded simulation runs for the static Three.js replay viewer in ``web/``.

Contract (docs/CONTRACTS.md, Phase 3):
    export_geometry(model="neuromechfly"|"flybody") -> path   (once per body model)
    export_brain_points() -> path                             (all neurons of the brain model)
    export_run(run_id, excite, ...) -> path                   (web/data/runs/<id>.json)

CLI:
    uv run python -m flylab.export3d --all            # geometry + brain points + all runs
    uv run python -m flylab.export3d --runs mdn_backward p9_forward
    uv run python -m flylab.export3d --geometry --brain --index

What the viewer shows is a *recorded* simulation, never a live re-computation (S4).
Each run file keeps the four layers separate (team ground rule G2):
    brain   - flylab.brain LIF model, mean firing rate per neuron over the brain run
    bridge  - flylab.bridge adapter output (drive for walking / command for flight)
    body    - MuJoCo physics: per-frame world poses of every body (flylab.poses format)
    verify  - flylab.verify movement check (if the module is available), recomputed
              from the raw trajectory, independent of the body classifier.

Files written (all relative to ``web/``):
    data/geometry/<model>.json + .bin  per-geom placement + mesh vertices (Float32, mm)
                                       and triangle indices (Uint16/Uint32), little endian
    data/brain_points.json + .bin      Float32 xyz (um) for every model neuron, then uint8
                                       super_class index; order = flylab.brain model index
    data/runs/<id>.json                one recorded run (see export_run)
    data/runs/index.json               run list for the picker
    data/manifest.json                 SHA-256 + size of every data file (provenance)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DATA = WEB / "data"
RUNS = DATA / "runs"
GEOM = DATA / "geometry"
RAW = ROOT / "data" / "raw"

EXCITE_RATE_HZ = 150.0
FLYBODY_CLUSTER_CM = 0.0012  # 12 um grid for the visual FlyBody meshes (fly length ~0.25 cm)

# ---------------------------------------------------------------------------------------
# Run catalogue: real experiments that are exported for the public viewer.
# expected/gt: literature expectation used by the verifier (ground_truth.json ids).
# ---------------------------------------------------------------------------------------
RUN_SPECS: dict[str, dict] = {
    "mdn_backward": {
        "title": "MDN (moonwalker) activation -> backward walking",
        "excite": ["MDN"], "silence": [], "mode": "walk", "expected": "backward",
        "gt": "gt01_mdn_activate_backward", "kind": "validation",
    },
    "p9_forward": {
        "title": "P9 activation -> forward walking",
        "excite": ["P9"], "silence": [], "mode": "walk", "expected": "forward",
        "gt": "gt03_p9_activate_forward", "kind": "validation",
    },
    "dna02l_turn_left": {
        "title": "DNa02 left activation -> left turn",
        "excite": ["DNa02_L"], "silence": [], "mode": "walk", "expected": "turn_left",
        "gt": "gt05_dna02L_activate_turn_left", "kind": "validation",
    },
    "dna02r_turn_right": {
        "title": "DNa02 right activation -> right turn",
        "excite": ["DNa02_R"], "silence": [], "mode": "walk", "expected": "turn_right",
        "gt": "gt06_dna02R_activate_turn_right", "kind": "validation",
    },
    "lc16_retreat": {
        "title": "LC16 activation (agent hypothesis H1) -> retreat?",
        "excite": ["LC16"], "silence": [], "mode": "walk", "expected": "backward",
        "gt": "gt20_lc16_activate_backward", "kind": "agent_hypothesis",
    },
    "lplc2_walk": {
        "title": "LPLC2 looming detectors -> giant fiber escape readout (walking body)",
        "excite": ["LPLC2"], "silence": [], "mode": "walk", "expected": "backward",
        "gt": "gt19_lplc2_activate_backward", "kind": "surprise",
    },
    # flight runs (need flylab.flight; use the bridge flight command when it exists, else a labelled direct command)
    "gf_takeoff": {
        "title": "Giant fiber activation -> escape takeoff and flight",
        "excite": ["GF"], "silence": [], "mode": "flight", "expected": "escape",
        "gt": "gt07_gf_activate_escape", "kind": "validation", "condition": "GF_bilateral",
    },
    "lplc2_takeoff": {
        "title": "LPLC2 looming detectors -> giant fiber -> escape takeoff",
        "excite": ["LPLC2"], "silence": [], "mode": "flight", "expected": "escape",
        "gt": "gt08_lplc2_activate_escape", "kind": "validation", "condition": "LPLC2_bilateral",
    },
    "gf_dng02_climb": {
        "title": "Giant fiber + DNg02 population -> takeoff with added wingbeat amplitude (climb)",
        "excite": ["GF", "DNg02"], "silence": [], "mode": "flight", "expected": "escape",
        "gt": "gt26_dng02_activate_wingbeat_amplitude", "kind": "validation", "condition": "GF_DNg02",
        "gt_note": "Partly circular: the bridge maps the DNg02 population to wingbeat amplitude using the same paper "
                   "that defines the expectation, and the amplitude gain is a design constant (plumbing check, not a discovery).",
    },
    "lplc2_30hz_takeoff": {
        "title": "LPLC2 at 30 Hz (weaker looming stand-in) -> takeoff (model prediction)",
        "excite": ["LPLC2"], "silence": [], "mode": "flight", "expected": "escape", "rate": 30.0,
        "gt": "gt08_lplc2_activate_escape", "kind": "prediction", "condition": "LPLC2_30Hz",
        "gt_note": "Dose-response point of the model, not a literature result: the cited paper activated LPLC2 "
                   "optogenetically at an unknown, unmatched strength.",
    },
    "lplc2_10hz_no_takeoff": {
        "title": "LPLC2 at 10 Hz (sub-threshold) -> no takeoff (model prediction)",
        "excite": ["LPLC2"], "silence": [], "mode": "flight", "expected": "no_takeoff", "rate": 10.0,
        "gt": "gt08_lplc2_activate_escape", "kind": "prediction", "condition": "LPLC2_10Hz",
        "gt_note": "Dose-response point of the model, not a literature result: weak drive is predicted to stay below the "
                   "giant-fiber takeoff threshold. The cited paper reports activation at a stronger, unmatched drive.",
    },
    "lplc2_gf_silenced": {
        "title": "LPLC2 activation with giant fiber silenced -> no takeoff (control)",
        "excite": ["LPLC2"], "silence": ["GF"], "mode": "flight", "expected": "no_takeoff",
        "gt": "gt09_lplc2_silence_escape", "kind": "control", "condition": "LPLC2_GF_silenced",
        "gt_note": "In-silico control: the giant fiber (downstream of LPLC2) is silenced. The cited paper silenced "
                   "LPLC2 itself; this control tests the same pathway one synapse later and is not a replication.",
    },
}

AGENT_RUN_DIR = ROOT / "runs" / "20261004-010600-which-visual-projection-neurons-ebe8"


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _r(x: float, d: int = 4) -> float:
    x = float(x)
    return 0.0 if not math.isfinite(x) else round(x, d)


def _rel(p: Path) -> str:
    return p.resolve().relative_to(WEB.resolve()).as_posix()


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str | None:
    try:
        import subprocess

        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------
_GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
               5: "cylinder", 6: "box", 7: "mesh"}


def _simplify_mesh(verts: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Weld + vertex-clustering simplification: snap vertices to a grid of size ``cell``, merge
    each occupied cell into its mean vertex, drop collapsed and duplicate triangles."""
    keys = np.floor(verts / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    k = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=k).astype(np.float64)
    new_v = np.stack([np.bincount(inv, weights=verts[:, c], minlength=k) / cnt for c in range(3)], axis=1)
    f = inv[faces.reshape(-1, 3)]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    used = np.unique(f)
    remap = -np.ones(k, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return new_v[used].astype(np.float32), remap[f]


def geometry_from_mujoco(mj_model, model_name: str, body_filter=None, geom_filter=None,
                         units: str = "mm", source: str = "", license_note: str = "",
                         cluster: float | None = None) -> Path:
    """Write ``data/geometry/<model_name>.json/.bin`` from a compiled MuJoCo model.

    ``cluster`` (model units): if set, meshes are welded and simplified by vertex clustering
    on a grid of that cell size (purely visual; physics used the original meshes).
    Uses the *compiled* model: mesh vertices are in the (re-centred) mesh frame and
    geom_pos/geom_quat place them in the parent body frame, so in the browser
    world = body_pose * geom_pose * vertex, with body poses from flylab.poses.
    """
    import mujoco as mj

    m = mj_model
    bnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(m.nbody)]
    vert_chunks: list[np.ndarray] = []
    idx_chunks: list[np.ndarray] = []
    geoms: list[dict] = []
    v_off = 0  # in floats
    i_off = 0  # in indices
    big_index = False
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        bname = bnames[b]
        if b == 0 or (body_filter is not None and not body_filter(bname)):
            continue
        if geom_filter is not None and not geom_filter(m, g):
            continue
        gtype = _GEOM_TYPES.get(int(m.geom_type[g]), "other")
        gname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        entry = {"name": gname, "body": bname, "type": gtype,
                 "pos": [_r(v, 6) for v in m.geom_pos[g]], "quat": [_r(v, 6) for v in m.geom_quat[g]],
                 "size": [_r(v, 6) for v in m.geom_size[g]], "rgba": [_r(v, 3) for v in m.geom_rgba[g]]}
        if gtype == "mesh":
            mid = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            verts = np.asarray(m.mesh_vert[va:va + vn], dtype=np.float32)
            faces = np.asarray(m.mesh_face[fa:fa + fn], dtype=np.int64)
            if cluster:
                verts, faces = _simplify_mesh(verts, faces, cluster)
            faces = faces.reshape(-1)
            vn = int(len(verts))
            if vn > 65535:
                big_index = True
            vert_chunks.append(verts.reshape(-1))
            idx_chunks.append(faces)
            entry["mesh"] = {"vOff": v_off, "vCount": vn, "iOff": i_off, "iCount": int(faces.size)}
            v_off += vn * 3
            i_off += int(faces.size)
        elif gtype in ("plane", "hfield", "other"):
            continue
        geoms.append(entry)

    verts_all = np.concatenate(vert_chunks) if vert_chunks else np.zeros(0, np.float32)
    idx_dtype = np.uint32 if big_index else np.uint16
    idx_all = np.concatenate(idx_chunks).astype(idx_dtype) if idx_chunks else np.zeros(0, idx_dtype)
    GEOM.mkdir(parents=True, exist_ok=True)
    bin_path = GEOM / f"{model_name}.bin"
    with open(bin_path, "wb") as f:
        f.write(verts_all.astype("<f4").tobytes())
        f.write(idx_all.astype("<u4" if big_index else "<u2").tobytes())
    meta = {
        "format": "flylab-geometry-v1", "model": model_name, "units": units,
        "up_axis": "z (MuJoCo world)", "quat_order": "w,x,y,z",
        "bin": bin_path.name, "vertex_floats": int(verts_all.size),
        "index_type": "uint32" if big_index else "uint16", "index_count": int(idx_all.size),
        "index_byte_offset": int(verts_all.size * 4),
        "n_geoms": len(geoms), "n_triangles": int(idx_all.size // 3),
        "bodies": sorted({g["body"] for g in geoms}), "geoms": geoms,
        "source": source, "license": license_note,
        "note": (f"Meshes simplified for the browser by vertex clustering (cell {cluster} {units}); purely visual, "
                 "the physics used the original meshes.") if cluster else
                "Meshes copied from the compiled MuJoCo model without decimation.",
    }
    path = _write_json(GEOM / f"{model_name}.json", meta)
    return path


def export_geometry(model: str = "neuromechfly") -> str:
    """Export per-body geometry for ``neuromechfly`` (flygym 2.1) or ``flybody`` (flight)."""
    model = model.lower()
    if model == "neuromechfly":
        from flylab import body

        fly, cam, sim, *_ = body._build(None, 5)
        try:
            prefix = f"{fly.name}/"
            path = geometry_from_mujoco(
                sim.mj_model, "neuromechfly", body_filter=lambda n: n.startswith(prefix),
                source="NeuroMechFly v2 body model via flygym 2.1 (Lobato-Rios et al. 2022, "
                       "doi:10.1038/s41592-022-01466-7; Wang-Chen et al. 2024, doi:10.1038/s41592-024-02497-y)",
                license_note="flygym: Apache-2.0 (https://github.com/NeLy-EPFL/flygym)")
        finally:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
        return path.as_posix()
    if model == "flybody":
        from flylab import flight  # written by the flight workstream

        mj_model = None
        for fn in ("get_mj_model", "load_model", "build_model", "_build_model", "_build", "_load_model"):
            f = getattr(flight, fn, None)
            if f is None:
                continue
            obj = f()
            if isinstance(obj, tuple):
                obj = next((o for o in obj if hasattr(o, "nbody") or hasattr(o, "mj_model") or hasattr(o, "model")), obj[0])
            mj_model = getattr(obj, "mj_model", None) or getattr(obj, "model", None) or obj
            if hasattr(mj_model, "ptr"):  # dm_control Physics.model wrapper
                mj_model = mj_model.ptr
            if hasattr(mj_model, "nbody"):
                break
        if mj_model is None or not hasattr(mj_model, "nbody"):
            raise RuntimeError("flylab.flight exposes no model builder (expected get_mj_model/load_model/_build)")

        def visual(m, g):  # FlyBody: visual meshes only (no collision / wing-fluid / inertial primitives)
            return int(m.geom_type[g]) == 7

        path = geometry_from_mujoco(
            mj_model, "flybody", geom_filter=visual, units="cm", cluster=FLYBODY_CLUSTER_CM,
            source="FlyBody fruit-fly model (Vaxenburg et al. 2025, Nature, doi:10.1038/s41586-025-09029-4) "
                   "as used by flylab.flight",
            license_note="flybody: Apache-2.0 (https://github.com/TuragaLab/flybody)")
        return path.as_posix()
    raise ValueError(f"unknown model {model!r} (neuromechfly | flybody)")


# ---------------------------------------------------------------------------------------
# brain point cloud
# ---------------------------------------------------------------------------------------
SUPER_CLASSES = ["central", "optic", "visual_projection", "visual_centrifugal", "sensory",
                 "ascending", "descending", "motor", "endocrine", "unknown"]
_VOXEL_NM = np.array([4.0, 4.0, 40.0])  # FlyWire annotation anchor coords: 4x4x40 nm voxels


def _model_ids() -> np.ndarray:
    """Root IDs in flylab.brain model order (rows of Completeness_783.csv)."""
    import pandas as pd

    comp = pd.read_csv(RAW / "Completeness_783.csv", index_col=0)
    return comp.index.to_numpy(np.int64)


_ID2IDX: dict[int, int] | None = None


def _id2idx() -> dict[int, int]:
    global _ID2IDX
    if _ID2IDX is None:
        _ID2IDX = {int(r): i for i, r in enumerate(_model_ids().tolist())}
    return _ID2IDX


def export_brain_points() -> str:
    """Write 3D positions + super_class of every neuron of the brain model (138,639).

    Position = FlyWire v783 annotation anchor point (pos_x/y/z, a point on the neuron's
    backbone; Schlegel et al. 2024 supplemental file 1), converted from 4x4x40 nm voxels
    to micrometres in the FAFB14 / FlyWire coordinate frame (x: medio-lateral,
    y: dorso-ventral, increasing ventrally; z: antero-posterior, increasing posteriorly).
    """
    import pandas as pd

    ids = _model_ids()
    ann = pd.read_csv(RAW / "flywire_annotations_Supplemental_file1_neuron_annotations.tsv", sep="\t",
                      usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class"],
                      dtype={"root_id": np.int64, "super_class": "string"})
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    sub = ann.reindex(ids)
    xyz_vox = sub[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64)
    missing = ~np.isfinite(xyz_vox).all(axis=1)
    xyz = (xyz_vox * _VOXEL_NM / 1000.0).astype(np.float32)  # um
    xyz[missing] = np.nan
    sc = sub["super_class"].fillna("unknown").astype(str).to_numpy()
    cls_index = {c: i for i, c in enumerate(SUPER_CLASSES)}
    cls = np.array([cls_index.get(s, len(SUPER_CLASSES) - 1) for s in sc], dtype=np.uint8)
    ok = ~missing
    lo, hi = np.nanmin(xyz[ok], axis=0), np.nanmax(xyz[ok], axis=0)
    center = np.nanmean(xyz[ok], axis=0)
    DATA.mkdir(parents=True, exist_ok=True)
    bin_path = DATA / "brain_points.bin"
    with open(bin_path, "wb") as f:
        f.write(np.nan_to_num(xyz, nan=np.float32(np.nan)).astype("<f4").tobytes())
        f.write(cls.tobytes())
    counts = {c: int((cls == i).sum()) for i, c in enumerate(SUPER_CLASSES)}
    meta = {
        "format": "flylab-brain-points-v1", "n": int(len(ids)), "bin": bin_path.name,
        "layout": "Float32 xyz[n*3] (um, little endian; NaN = no position), then uint8 class[n]",
        "order": "flylab.brain model index = row order of data/raw/Completeness_783.csv (Shiu et al. 2024 model)",
        "units": "um",
        "frame": "FlyWire FAFB14 v783: x medio-lateral, y dorso-ventral (+ = ventral), z antero-posterior (+ = posterior)",
        "position_source": "anchor point pos_x/y/z (on the neuron backbone) from FlyWire v783 annotations, "
                           "Schlegel et al. 2024 Nature doi:10.1038/s41586-024-07686-5, voxel size 4x4x40 nm",
        "classes": SUPER_CLASSES, "class_counts": counts, "n_missing_position": int(missing.sum()),
        "bbox_min": [_r(v, 2) for v in lo], "bbox_max": [_r(v, 2) for v in hi], "center": [_r(v, 2) for v in center],
    }
    path = _write_json(DATA / "brain_points.json", meta)
    return path.as_posix()


# ---------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------
def _compact_poses(poses: dict | None) -> dict | None:
    if not poses or not poses.get("frames"):
        return None
    frames = poses["frames"]
    return {
        "format": "flylab-poses-v1-compact", "units": poses.get("units", "model"), "fps": poses.get("fps"),
        "bodies": poses["bodies"],
        "t": [_r(f["t"], 4) for f in frames],
        "p": [[_r(v, 4) for xyz in f["p"] for v in xyz] for f in frames],
        "q": [[_r(v, 3) for wxyz in f["q"] for v in wxyz] for f in frames],
    }


def _annotations_lookup(root_ids: list[int]) -> dict[int, dict]:
    try:
        from flylab import brain

        ann = brain.annotations()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    try:
        df = ann.set_index("root_id") if "root_id" in getattr(ann, "columns", []) else ann
        for r in root_ids:
            if r in df.index:
                row = df.loc[r]
                if hasattr(row, "ndim") and row.ndim > 1:
                    row = row.iloc[0]
                out[r] = {k: (None if str(row.get(k)) in ("nan", "<NA>", "None") else str(row.get(k)))
                          for k in ("cell_type", "super_class", "side") if k in row.index}
    except Exception:  # noqa: BLE001
        return out
    return out


def _agent_context(spec: dict) -> dict | None:
    """Pull the agent hypothesis / analysis / decision for this run from the live Omnigent record."""
    rec = AGENT_RUN_DIR / "record.jsonl"
    if spec.get("kind") not in ("agent_hypothesis", "surprise") or not rec.exists():
        return None
    events = [json.loads(line) for line in rec.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [g.lower() for g in spec["excite"]]

    def hit(e):
        c = e.get("content")
        c = c if isinstance(c, str) else json.dumps(c)
        return any(k in c.lower() for k in keys)

    pick = lambda typ: [e for e in events if e.get("type") == typ and hit(e)]  # noqa: E731
    q = next((e for e in events if e.get("type") == "question"), None)
    hyps = pick("hypothesis")
    analyses = pick("analysis")
    decisions = [e for e in events if e.get("type") == "decision" and e.get("content")]

    def short(e, n=600):
        c = e.get("content")
        c = c if isinstance(c, str) else json.dumps(c)
        return {"agent": e.get("agent"), "type": e.get("type"), "text": c[:n], "ts": e.get("ts") or e.get("time")}

    return {
        "source": f"runs/{AGENT_RUN_DIR.name}/record.jsonl (live Omnigent run, shared research record)",
        "question": q.get("content") if q else None,
        "hypotheses": [short(e) for e in hyps[:2]],
        "analyses": [short(e) for e in analyses[:3]],
        "decision": short(decisions[0]) if decisions else None,
        "label": "Hypotheses are AGENT-GENERATED and unconfirmed in the wet lab.",
    }


def _ground_truth(gt_id: str | None) -> dict | None:
    if not gt_id:
        return None
    from flylab import atlas

    for e in atlas.ground_truth():
        if e.get("id") == gt_id:
            return {k: e.get(k) for k in ("id", "manipulation", "target_group", "expected_behavior", "effect",
                                           "evidence", "citation")}
    return None


def _verifier_benchmark() -> dict | None:
    """Headline numbers of the committed verifier benchmark (data/benchmarks/movement_verifier.json)."""
    p = ROOT / "data" / "benchmarks" / "movement_verifier.json"
    try:
        s = json.loads(p.read_text(encoding="utf-8"))["summary"]
        return {"file": "data/benchmarks/movement_verifier.json", "n_runs": s["n_runs"],
                "kinematic_correct_on_known_label": s["kinematic_correct_on_known_label"],
                "kinematic_incorrect_on_opposite_label": s["kinematic_incorrect_on_opposite_label"],
                "vision_exact_on_unique_videos": s["vision_exact_on_unique_videos"],
                "vision_unique_videos": s["vision_unique_videos"]}
    except Exception:  # noqa: BLE001
        return None


def _verify(result: dict, expected: str | None, mode: str) -> dict | None:
    if not expected:
        return None
    try:
        from flylab import verify  # written by the verifier workstream
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"flylab.verify not available at export time ({type(exc).__name__})"}
    try:
        v = verify.verify_movement(result, expected, mode=mode, use_vision=False)
        keep = {k: v.get(k) for k in ("kinematic", "final_verdict", "agreement") if k in v}
        keep["available"] = True
        keep["benchmark"] = _verifier_benchmark()
        keep["vision"] ="not run for this export (kinematic check only; vision check needs a rendered video and an API call)"
        return keep
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"verify_movement failed: {type(exc).__name__}: {exc}"[:300]}


def _flight_available() -> bool:
    try:
        from flylab import flight

        return hasattr(flight, "simulate_flight")
    except Exception:  # noqa: BLE001
        return False


DIRECT_GF_MIN_HZ = 20.0
FLIGHT_BENCH = ROOT / "data" / "benchmarks" / "flight_validation.json"


def _vision_summary(vis: dict | None, err: str | None = None) -> str:
    if not vis:
        return err or "not run for this condition (kinematic check only)"
    bits = []
    for k in ("observed_behavior", "verdict", "confidence", "matches_expected"):
        if vis.get(k) is not None:
            bits.append(f"{k.replace('_', ' ')}: {vis[k]}")
    model = vis.get("model")
    return "blind Claude-vision check (" + (model or "model n/a") + "; sees neither the expected behaviour nor the kinematics): " + ", ".join(bits)


def _flight_validation(condition: str | None, res: dict) -> dict | None:
    """Attach the flight_validation.json row of ``condition`` and check this export reproduces it."""
    if not condition or not FLIGHT_BENCH.exists():
        return None
    try:
        bench = json.loads(FLIGHT_BENCH.read_text(encoding="utf-8"))
        rows = next((v for v in bench.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "condition" in v[0]), [])
        row = next((r for r in rows if r.get("condition") == condition), None)
        if row is None:
            return None
        b = row.get("body", {})
        same = (b.get("behavior") == res.get("behavior") and bool(b.get("airborne")) == bool(res.get("airborne"))
                and abs(float(b.get("max_height_mm", 0)) - float(res.get("max_height_mm", 0))) < 0.05)
        ver = row.get("verification") or {}
        verifier = {"available": True, "final_verdict": ver.get("final_verdict"),
                    "kinematic": ver.get("kinematic"), "agreement": ver.get("agreement"), "reason": ver.get("reason"),
                    "benchmark": _verifier_benchmark(),
                    "vision": _vision_summary(ver.get("vision"), ver.get("vision_error")),
                    "vision_detail": ver.get("vision")}
        return {"verifier": verifier, "validation": {
            "file": "data/benchmarks/flight_validation.json", "condition": condition,
            "expected": row.get("expected"), "expected_source": row.get("expected_source"),
            "as_expected": row.get("as_expected"), "result_type": row.get("result_type"),
            "result_type_text": row.get("result_type_text"), "stimulus_text": row.get("stimulus"),
            "checks": row.get("checks"), "necessity_check": row.get("necessity_check"),
            "note_walk": row.get("note_walk"), "frozen_hash": (row.get("adapter") or {}).get("frozen_hash"),
            "export_reproduces_validation_run": bool(same)}}
    except Exception as exc:  # noqa: BLE001
        return {"validation": {"file": "data/benchmarks/flight_validation.json", "condition": condition,
                               "error": f"{type(exc).__name__}: {exc}"[:200]}}


def _direct_flight_command(rates: dict, atlas) -> tuple[dict, dict]:
    """Fallback when flylab.bridge has no flight mapping: takeoff trigger straight from the GF rate."""
    ids = atlas.group_ids("GF")
    vals = [float(rates.get(i, rates.get(str(i), 0.0)) or 0.0) for i in ids]
    gf = float(np.mean(vals)) if vals else 0.0
    on = gf >= DIRECT_GF_MIN_HZ
    command = {"takeoff": 1.0 if on else 0.0, "thrust": 0.5 if on else 0.0, "yaw": 0.0, "pitch": 0.0}
    return command, {"group_rates_hz": {"GF": _r(gf, 2)}, "rule": f"takeoff = GF mean rate >= {DIRECT_GF_MIN_HZ:.0f} Hz",
                     "direct_command": True}


def export_run(run_id: str, excite: list[str], silence: list[str] | None = None, *, mode: str = "walk",
               expected: str | None = None, gt: str | None = None, title: str | None = None,
               kind: str = "validation", duration_s: float = 1.0, brain_ms: float = 1000.0,
               n_trials: int = 3, seed: int = 0, top_n: int = 25, gt_note: str | None = None,
               precomputed: dict | None = None, excite_rate_hz: float | None = None,
               condition: str | None = None) -> str:
    """Run brain -> bridge -> body (walk) or flight, record poses and write web/data/runs/<run_id>.json.

    ``precomputed``: an already simulated ``flylab.flight.simulate_flight`` result (with poses) to export
    instead of simulating again. ``excite_rate_hz``: Poisson drive of the excited groups (default 150 Hz).
    ``condition``: name of the matching row in data/benchmarks/flight_validation.json (flight runs); the row's
    verification, result type and ground-truth checks are attached under ``validation``."""
    from flylab import atlas, brain, bridge

    silence = list(silence or [])
    rate_hz = float(EXCITE_RATE_HZ if excite_rate_hz is None else excite_rate_hz)
    t0 = time.perf_counter()
    ex_ids = [i for g in excite for i in atlas.group_ids(g)]
    si_ids = [i for g in silence for i in atlas.group_ids(g)]
    if ex_ids:
        bres = brain.simulate(ex_ids, si_ids or None, excite_rate_hz=rate_hz, duration_ms=brain_ms,
                              n_trials=n_trials, seed=seed, n_threads=min(4, n_trials))
        rates = bres["rates"]
        brain_rt = bres.get("runtime_s")
    else:
        rates, brain_rt = {}, 0.0
    readouts = bridge.brain_readouts(rates)

    bridge_note = None
    if mode == "flight":
        from flylab import flight

        if hasattr(bridge, "rates_to_flight_command"):
            cmd_full = bridge.rates_to_flight_command(rates)
            command = {k: cmd_full[k] for k in ("takeoff", "thrust", "yaw", "pitch") if k in cmd_full}
            explain = cmd_full.get("explain")
        else:
            command, explain = _direct_flight_command(rates, atlas)
            bridge_note = ("DIRECT command, NOT the connectome bridge: flylab.bridge has no flight mapping in this "
                           "export. takeoff = 1 if the mean giant-fiber (GF) rate is at least "
                           f"{DIRECT_GF_MIN_HZ:.0f} Hz, else 0; thrust fixed. Only the takeoff trigger comes from the brain model.")
        res = precomputed if precomputed is not None else flight.simulate_flight(
            command, duration_s=duration_s, seed=seed, record_poses=True)
        adapter = {"type": "flight_command", "values": command, "explain": explain}
        if bridge_note:
            adapter["note"] = bridge_note
        geometry = "flybody"
    else:
        drv_full = bridge.rates_to_drive(rates)
        drive = {k: drv_full[k] for k in ("forward", "turn", "backward")}
        from flylab import body

        res = body.simulate_walk(drive, duration_s=duration_s, seed=seed, record_poses=True)
        adapter = {"type": "walk_drive", "values": drive, "explain": drv_full.get("explain")}
        geometry = "neuromechfly"

    verdict = _verify(res, expected, mode)

    # --- brain activity, indexed into brain_points order --------------------------------
    id2idx = _id2idx()
    items = sorted(((int(k), float(v)) for k, v in rates.items()), key=lambda kv: -kv[1])
    act_idx, act_rate = [], []
    for rid, hz in items:
        i = id2idx.get(rid)
        if i is not None and hz > 0:
            act_idx.append(i)
            act_rate.append(_r(hz, 2))
    top_ids = [rid for rid, _ in items[:top_n]]
    ann = _annotations_lookup(top_ids)
    ex_set, si_set = set(ex_ids), set(si_ids)
    top = [{"root_id": str(rid), "idx": id2idx.get(rid), "rate_hz": _r(hz, 2),
            "role": "stimulated" if rid in ex_set else "downstream", **ann.get(rid, {})}
           for rid, hz in items[:top_n]]
    dn_idx = []
    try:
        for name in ("MDN", "P9", "DNa01", "DNa02", "GF", "aDN1_shiu2024", "aDN2_shiu2024", "MN9"):
            for rid in atlas.group_ids(name):
                if rid in id2idx:
                    dn_idx.append(id2idx[rid])
    except Exception:  # noqa: BLE001
        pass

    trajectory = res.get("trajectory") or []
    body_metrics = {k: res.get(k) for k in (
        "behavior", "forward_disp_mm", "lateral_disp_mm", "heading_change_deg", "mean_speed_mm_s",
        "net_speed_mm_s", "yaw_rate_deg_s", "fell_over", "airborne", "takeoff_time_s", "flight_time_s",
        "max_height_mm", "net_displacement_mm", "sim_duration_s", "runtime_s", "warnings", "model_notes",
        "control_every", "descending_signal") if k in res}

    spec_gt = _ground_truth(gt)
    if spec_gt is not None and gt_note:
        spec_gt["note"] = gt_note
    out = {
        "format": "flylab-run-v1",
        "id": run_id, "title": title or run_id, "kind": kind, "mode": mode,
        "label": "RECORDED simulation replay - not a live re-computation",
        "created": datetime.now().isoformat(timespec="seconds"), "git_rev": _git_rev(),
        "reproduce": f"uv run python -m flylab.export3d --runs {run_id}",
        "stimulus": {"excite": excite, "silence": silence, "excite_rate_hz": rate_hz,
                     "n_excited": len(ex_ids), "n_silenced": len(si_ids)},
        "brain": {
            "model": "flylab.brain: whole-brain LIF after Shiu et al. 2024 (Nature, doi:10.1038/s41586-024-07763-9), FlyWire v783",
            "duration_ms": brain_ms, "n_trials": n_trials, "seed": seed, "runtime_s": brain_rt,
            "n_active": len(act_idx),
            "note": "Mean firing rate per neuron over the brain run (not time-resolved). Open loop: the brain run "
                    "precedes the body run and receives no sensory feedback.",
            "active_idx": act_idx, "active_rate_hz": act_rate,
            "stimulated_idx": sorted(id2idx[i] for i in ex_set if i in id2idx),
            "silenced_idx": sorted(id2idx[i] for i in si_set if i in id2idx),
            "readout_idx": sorted(set(dn_idx)),
            "top": top,
            "group_rates_hz": (adapter.get("explain") or {}).get("group_rates_hz"),
            "readouts": {b: ({k: v for k, v in r.items() if k in ("group", "rate_hz", "active", "rates_hz")}
                             if isinstance(r, dict) and "rate_hz" in r else r)
                         for b, r in (readouts or {}).items()},
        },
        "bridge": {**adapter, "frozen": ("flylab.bridge mapping is hand-designed and fixed before these runs "
                                         "(ventral nerve cord not simulated)") if not bridge_note else
                   "direct command, not the frozen bridge (ventral nerve cord not simulated)"},
        "body": {"model": geometry, "geometry": f"data/geometry/{geometry}.json", "duration_s": duration_s,
                 "metrics": body_metrics,
                 "trajectory": [[_r(v, 4) for v in row] for row in trajectory],
                 "poses": _compact_poses(res.get("poses"))},
        "expected_behavior": expected,
        "verifier": verdict,
        "ground_truth": spec_gt,
        "agent": _agent_context({"kind": kind, "excite": excite}),
        "export_runtime_s": round(time.perf_counter() - t0, 2),
    }
    val = _flight_validation(condition, res) if mode == "flight" else None
    if val:
        out["validation"] = val["validation"]
        if val.get("verifier"):
            out["verifier"] = val["verifier"]
    path = _write_json(RUNS / f"{run_id}.json", out)
    return path.as_posix()


def write_index() -> str:
    """(Re)build data/runs/index.json from the run files and data/manifest.json (SHA-256)."""
    runs = []
    for p in sorted(RUNS.glob("*.json")):
        if p.name == "index.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("kind") == "test" or str(d.get("id", "")).startswith("zz_"):
            continue  # spike / test runs are never listed in the public viewer
        g = d.get("ground_truth")
        if g and "effect" not in g and g.get("id"):  # runs exported before the effect field existed
            fresh = _ground_truth(g["id"])
            if fresh:
                d["ground_truth"] = {**fresh, **{k: v for k, v in g.items() if k == "note"}}
                _write_json(p, d)
        v = d.get("verifier") or {}
        if v.get("available") and "benchmark" not in v:  # runs exported before the benchmark pointer existed
            v["benchmark"] = _verifier_benchmark()
            d["verifier"] = v
            _write_json(p, d)
        runs.append({"id": d["id"], "title": d.get("title"), "mode": d.get("mode"), "kind": d.get("kind"),
                     "file": _rel(p), "behavior": (d.get("body") or {}).get("metrics", {}).get("behavior"),
                     "expected": d.get("expected_behavior"), "verdict": v.get("final_verdict"),
                     "created": d.get("created")})
    order = list(RUN_SPECS)
    runs.sort(key=lambda r: order.index(r["id"]) if r["id"] in order else 99)
    idx = _write_json(RUNS / "index.json", {"format": "flylab-run-index-v1", "runs": runs,
                                            "label": "Recorded simulation replays (not live)"})
    files = []
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files.append({"path": _rel(p), "bytes": p.stat().st_size, "sha256": _sha256(p)})
    _write_json(DATA / "manifest.json", {
        "format": "flylab-manifest-v1", "generated": datetime.now().isoformat(timespec="seconds"),
        "git_rev": _git_rev(), "generator": "uv run python -m flylab.export3d --all", "files": files,
        "total_bytes": sum(f["bytes"] for f in files)})
    return idx.as_posix()


def export_all(run_ids: list[str] | None = None, *, geometry: bool = True, brain_points: bool = True) -> dict:
    out: dict[str, Any] = {"runs": {}, "skipped": {}}
    if geometry:
        out["geometry_neuromechfly"] = export_geometry("neuromechfly")
        if _flight_available():
            try:
                out["geometry_flybody"] = export_geometry("flybody")
            except Exception as exc:  # noqa: BLE001
                out["skipped"]["geometry_flybody"] = f"{type(exc).__name__}: {exc}"
    if brain_points:
        out["brain_points"] = export_brain_points()
    fl = _flight_available()
    for rid in (run_ids or list(RUN_SPECS)):
        spec = RUN_SPECS[rid]
        if spec["mode"] == "flight" and not fl:
            out["skipped"][rid] = "flylab.flight / bridge.rates_to_flight_command not available yet"
            continue
        t = time.perf_counter()
        try:
            out["runs"][rid] = export_run(rid, spec["excite"], spec.get("silence"), mode=spec["mode"],
                                          expected=spec.get("expected"), gt=spec.get("gt"),
                                          title=spec.get("title"), kind=spec.get("kind", "validation"),
                                          gt_note=spec.get("gt_note"), excite_rate_hz=spec.get("rate"),
                                          condition=spec.get("condition"))
            print(f"[export3d] {rid}: {time.perf_counter() - t:.1f} s", flush=True)
        except Exception as exc:  # noqa: BLE001
            out["skipped"][rid] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"[export3d] {rid} FAILED: {out['skipped'][rid]}", flush=True)
    out["index"] = write_index()
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="geometry + brain points + all runs + index")
    ap.add_argument("--runs", nargs="*", help=f"run ids ({', '.join(RUN_SPECS)})")
    ap.add_argument("--geometry", nargs="?", const="neuromechfly", help="export geometry (neuromechfly|flybody)")
    ap.add_argument("--brain", action="store_true", help="export brain point cloud")
    ap.add_argument("--index", action="store_true", help="rebuild runs/index.json + manifest.json")
    a = ap.parse_args(argv)
    if a.all:
        print(json.dumps(export_all(), indent=1))
        return
    if a.geometry:
        print(export_geometry(a.geometry))
    if a.brain:
        print(export_brain_points())
    if a.runs:
        print(json.dumps(export_all(a.runs, geometry=False, brain_points=False), indent=1))
    elif a.index or a.geometry or a.brain:
        print(write_index())


if __name__ == "__main__":
    main()
