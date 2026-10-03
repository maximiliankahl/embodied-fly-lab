# Module contracts (shared by all builders)

Package: `flylab/` (Python 3.12, run with `uv run python -m flylab.<module>` from `02_App`).
All public functions return JSON-serialisable data (dicts/lists/str/float/int) unless noted.
Paths are relative to the `02_App` root; use `flylab.paths` helpers if present, else `Path(__file__).resolve().parents[1]`.

## flylab/brain.py  (whole-brain LIF model after Shiu et al. 2024, Nature)
- `load_connectome() -> object` cached; reads `data/raw/` (download on first use).
- `simulate(excite: list[int], silence: list[int] | None = None, excite_rate_hz: float = 150.0,
            duration_ms: float = 1000.0, n_trials: int = 5, seed: int = 0) -> dict`
  returns `{"rates": {root_id(str): mean_rate_hz}, "std": {...}, "n_active": int, "runtime_s": float, "params": {...}}`
  (only neurons with rate > 0 in `rates`).
- `rates_for(result: dict, ids: list[int]) -> dict[str, float]`
- CLI `python -m flylab.brain --validate` reproduces a published Shiu et al. result (sugar GRNs -> MN9).

## flylab/atlas.py  (FlyWire v783 neuron atlas + literature ground truth)
- `groups() -> dict[str, dict]`: name -> `{"root_ids": [int], "cell_type": str, "side": "L"|"R"|"both", "role": "descending"|"sensory"|"motor"|..., "description": str, "citations": [doi]}`
  Data file: `data/neurons.json` (committed, small).
- `group_ids(name: str) -> list[int]`
- `find_cell_type(query: str) -> list[dict]` (search annotations by cell type / hemibrain type)
- `ground_truth() -> list[dict]` from `data/ground_truth.json`: each
  `{"id", "manipulation": "activate"|"silence", "target_group", "expected_behavior": "forward"|"backward"|"turn_left"|"turn_right"|"stop"|"escape"|"groom"|"feed", "evidence": str, "citation": {"doi","title","year","authors"}}`
## flylab/literature.py
- `search_europepmc(query: str, n: int = 5) -> list[dict]`, `search_openalex(query: str, n: int = 5) -> list[dict]`,
  `get_by_doi(doi: str) -> dict | None`; each item `{"title","year","doi","pmid","authors","abstract","url","source"}`. Cached in `data/cache/`.

## flylab/body.py  (NeuroMechFly / flygym 2.x physics body)
- `simulate_walk(drive: dict, duration_s: float = 1.5, render_path: str | None = None, seed: int = 0) -> dict`
  `drive = {"forward": 0..1, "turn": -1..1 (neg = left, pos = right), "backward": 0..1}`
  returns `{"trajectory": [[t, x_mm, y_mm, heading_deg], ...], "forward_disp_mm", "lateral_disp_mm", "heading_change_deg",
            "mean_speed_mm_s", "behavior": "forward"|"backward"|"turn_left"|"turn_right"|"stop", "video": str|None, "runtime_s"}`
- `classify(metrics) -> str` (same labels).

## flylab/bridge.py  (brain -> body interface; written at integration)
- `rates_to_drive(rates: dict[str, float]) -> dict` maps descending-neuron group rates to `drive`.

## flylab/record.py  (shared research record)
- `new_run(question: str) -> str` (run_id), `log_event(run_id, agent, type, content, data=None, citations=None) -> dict`,
  `load(run_id) -> list[dict]`, `list_runs() -> list[str]`. File: `runs/<run_id>/record.jsonl`, artifacts in `runs/<run_id>/artifacts/`.
  Event types: question, evidence, hypothesis, experiment_options, experiment_choice, approval, experiment_result, analysis, decision, note.

## flylab/tools.py  (Omnigent function tools; thin wrappers that also log to the record)
## agents/*.yaml   (Omnigent lab definition: supervisor + specialist sub-agents + policies)

---
# Phase 2 contracts (added Sun 00:20)

## flylab/bridge.py  (brain -> body; hand-designed, documented interface)
- `rates_to_drive(rates: dict[str, float]) -> dict` accepts EITHER per-neuron rates `{root_id_str: Hz}` (as `brain.simulate()["rates"]`)
  OR group rates `{group_name: Hz}` (auto-detect: all-digit keys = neurons). Returns `{"forward": 0..1, "turn": -1..1 (neg = left), "backward": 0..1, "explain": {...}}`.
  `tools._drive_from_rates` reads only forward/turn/backward; `explain` is for transparency (group rates used, reference rate, formula).
- `brain_readouts(rates) -> dict`: non-walking behaviours read at brain level, e.g. `{"escape": {"group": "GF", "rate_hz", "active": bool}, "feed": {"group": "MN9", ...}, "groom": {...}}`.
- `describe() -> dict`: mapping table (groups, signs, reference rates, citations) for docs/dashboard.

## flylab/screen.py  (discovery engine: connectome-guided candidate ranking + fast brain screen + acceleration benchmark)
- `candidate_types(kind: str = "visual_projection", min_n: int = 1) -> list[dict]` (`{cell_type, n, root_ids, super_class, cell_class}`), kinds include
  "visual_projection", "descending", "ascending", "sensory", or a regex on cell_type.
- `rank_by_connectome(target_group: str, candidates: list[str] | str, max_hops: int = 2) -> list[dict]` sorted by `score` desc
  (`{cell_type, n, score, direct_syn, two_hop_score, sign_note}`), score documented in the docstring.
- `brain_screen(candidates: list[str], target_groups: list[str], rate_hz=150.0, duration_ms=500.0, n_trials=2) -> list[dict]`
  (`{cell_type, n_stimulated, target_rates: {group: Hz}, runtime_s}`).
- `benchmark_search(target_group, candidates, known_hits: list[str], ...) -> dict` with experiments-to-first-hit for
  connectome-guided order vs random order (expected value) vs exhaustive, and the resulting reduction factor. Saves to `data/benchmarks/<name>.json` (committed).

## Dashboard (app.py, Streamlit)
- Reads `runs/<run_id>/record.jsonl` (+ artifacts), `data/benchmarks/*.json`, `data/neurons.json`, `data/ground_truth.json`, `assets/**`.
- Must render without any simulation dependency installed at runtime path (lazy imports), so it can be deployed in "replay" mode.
