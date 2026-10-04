"""Build / verify data/manifest.json (idea: teammate repo JonasMayerDev/FlyBrainLab, data/brain/manifest.json).

  uv run python spikes/audit/make_manifest.py            # (re)write data/manifest.json from the local files
  uv run python spikes/audit/make_manifest.py --verify   # check data/raw against the committed manifest

Raw data (data/raw, ~353 MB) is gitignored and never committed. The manifest records where each file comes from,
which version/licence applies, and its SHA256, so a fresh clone can re-download and verify the inputs.
Everything that is not computed here (URLs, licences) is copied from files in this repo or from the upstream
metadata that is cited in the entry; entries we could not verify say so.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifest.json"

SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"  # pinned by the teammate manifest; all 7 hashes below equal theirs
SHIU_RAW = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(rel: str, url: str | None = None, note: str | None = None) -> dict:
    p = ROOT / rel
    e = {"path": rel.replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}
    if url:
        e["url"] = url
    if note:
        e["note"] = note
    return e


def build() -> dict:
    shiu_files = [
        ("Connectivity_783.parquet", "model-ready connectivity (neuron pairs, synapse counts, signs)"),
        ("Completeness_783.csv", "neuron list (root IDs) of the model, 138,639 neurons"),
        ("LICENSE", "MIT licence of the Shiu et al. code repository"),
        ("Readme.md", None), ("model.py", "reference Brian2 implementation (used only for the cross-check)"),
        ("utils.py", None), ("example.ipynb", None),
    ]
    shiu_entries = [entry(f"data/raw/{n}", SHIU_RAW + n, note) for n, note in shiu_files]
    # figures.ipynb / sez_neurons.pickle come from the same repo, branch main (downloaded by flylab.brain / flylab.atlas)
    shiu_entries += [
        entry("data/raw/figures.ipynb", "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/figures.ipynb",
              "branch main (not commit-pinned); source of the Shiu sugar/bitter/water/JO neuron ID sets used in the atlas"),
        entry("data/raw/sez_neurons.pickle", None,
              "downloaded together with the Shiu files (file time 2026-10-03 22:56); the download source is not recorded in the flylab code and no flylab module reads it, so its origin is NOT VERIFIED here"),
    ]
    mesh_dir = RAW / "flygym_assets" / "flybody_fullsize_meshes_20260623a"
    meshes = {}
    for p in sorted(mesh_dir.rglob("*")):
        if p.is_file():
            meshes[p.relative_to(mesh_dir).as_posix()] = [p.stat().st_size, sha256(p)]
    tree_hash = hashlib.sha256("\n".join(f"{k} {v[1]}" for k, v in meshes.items()).encode()).hexdigest()

    committed = ["data/neurons.json", "data/ground_truth.json"] + sorted(
        str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "data" / "benchmarks").glob("*.json"))

    return {
        "schema_version": 1,
        "project": "Embodied Fly Lab (Hack-Nation 7, Challenge 03)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "spikes/audit/make_manifest.py (SHA256 and sizes computed from the local files; nothing here is downloaded)",
        "policy": "data/raw/ is gitignored and never committed. Re-download with the URLs below (flylab.brain / flylab.atlas do it on first use, flygym lazily) and verify with --verify.",
        "verify": "uv run python spikes/audit/make_manifest.py --verify",
        "idea_credit": "manifest structure follows JonasMayerDev/FlyBrainLab data/brain/manifest.json; its Shiu-file hashes equal ours (cross-checked), and its Zenodo metadata is the source of the connectome licence below",
        "datasets": [
            {
                "id": "flywire_v783_connectivity_shiu",
                "title": "FlyWire v783 model-ready connectivity and neuron list, as packaged by the Shiu et al. 2024 repository",
                "used_for": "whole-brain LIF model (flylab/brain.py): 138,639 neurons, 15.1 M connections",
                "source_repository": "https://github.com/philshiu/Drosophila_brain_model",
                "source_commit": SHIU_COMMIT,
                "source_commit_note": "commit taken from the teammate manifest (JonasMayerDev/FlyBrainLab). The SHA256 of all 7 files below equal that manifest, so our copies are byte-identical to that commit. flylab.brain downloads from branch main, which may move.",
                "version": "FlyWire FAFB materialization 783",
                "underlying_connectome": {
                    "doi": "10.5281/zenodo.10676866",
                    "url": "https://zenodo.org/records/10676866",
                    "version": "783.0",
                    "license": "CC-BY-4.0 (Zenodo record 'rights' field, as stored in the teammate repo data/brain/flywire783/zenodo_metadata.json)",
                    "attribution": "FlyWire Consortium; Dorkenwald et al. 2024 (doi:10.1038/s41586-024-07558-y); Schlegel et al. 2024 (doi:10.1038/s41586-024-07686-5)",
                },
                "code_license": "MIT (data/raw/LICENSE, Copyright 2023 Philip Shiu and Nico Spiller)",
                "paper": "Shiu et al. 2024, Nature, doi:10.1038/s41586-024-07763-9",
                "limitations": [
                    "model-ready transformed tables from the Shiu repository, not the raw synapse archive",
                    "brain only: FlyWire v783 has no ventral nerve cord, so the descending-to-body bridge is hand-designed",
                ],
                "files": shiu_entries,
            },
            {
                "id": "flywire_v783_connectivity_derived_csr",
                "title": "Sparse CSR cache of the connectivity (derived locally by flylab.brain, not an upstream file)",
                "used_for": "fast event-driven simulation",
                "files": [entry("data/raw/brain_csr_v783.npz", None,
                                "built by flylab.brain from Connectivity_783.parquet; hash is of OUR build (depends on numpy/scipy versions), informative only; delete to rebuild")],
            },
            {
                "id": "flywire_v783_annotations",
                "title": "FlyWire neuron annotations (cell types, classes, soma side, neurotransmitter) for root IDs of release 783",
                "used_for": "atlas (flylab/atlas.py: 63 curated groups, candidate cell types for the screen), brain-point positions of the 3D viewer",
                "source_repository": "https://github.com/flyconnectome/flywire_annotations",
                "url": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
                "url_note": "branch main (not commit-pinned); copy of flylab.atlas.ANNOT_URL. The upstream README says the content may have been updated and extended with Berg et al. 2025 relative to Schlegel et al. 2024.",
                "version": "root_id column = FlyWire release 783",
                "citations": ["10.1038/s41586-024-07686-5", "10.1038/s41586-024-07558-y", "10.1038/s41586-025-08925-z"],
                "license": "NOT VERIFIED here. Cite the three papers and see the upstream repository before redistributing; the file is gitignored and not redistributed by this repo.",
                "files": [
                    entry("data/raw/flywire_annotations_Supplemental_file1_neuron_annotations.tsv"),
                    entry("data/raw/flywire_annotations_README.md"),
                ],
            },
            {
                "id": "flygym_flybody_meshes",
                "title": "FlyBody full-size meshes (.obj) fetched lazily by flygym 2.1 for the flight body (flylab/flight.py)",
                "used_for": "physics flight body (MuJoCo) and the 3D viewer geometry",
                "source": "https://datasets.epfl.ch/nely-public-share/flygym_assets/flybody_fullsize_meshes_20260623a/<file>",
                "source_note": "public S3-compatible bucket 'nely-public-share', key prefix 'flygym_assets/flybody_fullsize_meshes_20260623a' (constants in flygym/utils/assets_lazy_loading.py and flygym/compose/fly/flybody.py)",
                "version": "mesh set 20260623a, flygym 2.1.0",
                "model_papers": ["FlyBody: Vaxenburg et al. 2025, Nature, doi:10.1038/s41586-025-09029-4", "NeuroMechFly v2: Wang-Chen et al. 2024, Nature Methods, doi:10.1038/s41592-024-02497-y"],
                "license": "flygym package: Apache-2.0 (package metadata). Licence of the mesh files themselves NOT VERIFIED here; the original FlyBody model is at https://github.com/TuragaLab/flybody.",
                "n_files": len(meshes),
                "total_bytes": sum(v[0] for v in meshes.values()),
                "tree_sha256": tree_hash,
                "tree_sha256_definition": "sha256 of the lines '<relative file name> <file sha256>' (sorted by name, joined by LF)",
                "files": {k: {"bytes": v[0], "sha256": v[1]} for k, v in meshes.items()},
            },
        ],
        "software_versions": {
            "note": "from the uv environment (uv.lock) at generation time",
            "flygym": "2.1.0 (Apache-2.0)", "mujoco": "3.9.0 (Apache-2.0)", "omnigent": "0.16.0 (Apache Software License)",
            "anthropic_sdk": "1.11.0 (MIT)",
        },
        "committed_derived_data": {
            "note": "small result files that ARE committed; hashes are of the files at generation time (they change when an experiment is re-run)",
            "files": [entry(p) for p in committed],
        },
    }


def verify() -> int:
    m = json.loads(OUT.read_text(encoding="utf-8"))
    bad = missing = ok = 0
    for ds in m["datasets"]:
        files = ds["files"]
        items = ([(f["path"], f["sha256"]) for f in files] if isinstance(files, list)
                 else [(f"data/raw/flygym_assets/flybody_fullsize_meshes_20260623a/{k}", v["sha256"]) for k, v in files.items()])
        for path, want in items:
            p = ROOT / path
            if not p.exists():
                missing += 1
                print("MISSING", path)
            elif sha256(p) != want:
                bad += 1
                print("MISMATCH", path)
            else:
                ok += 1
    print(f"verified {ok} files, {missing} missing, {bad} mismatched")
    return 0 if not (bad or missing) else 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")
