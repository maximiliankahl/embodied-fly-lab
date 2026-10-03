"""Discovery engine: connectome-guided candidate ranking + fast in-silico brain screen + acceleration benchmark.

Question the engine answers: "Which upstream cell types (e.g. visual projection neurons) drive a target
neuron group (e.g. the moonwalker descending neurons, MDN = backward walking)?"

Pipeline (docs/CONTRACTS.md, Phase 2):
    candidate_types(kind)                    all FlyWire v783 cell types of a class (e.g. 326 visual projection types)
    rank_by_connectome(target, candidates)   cheap prior from the signed connectome (milliseconds per candidate)
    brain_screen(candidates, targets)        expensive test: activate each candidate type in the whole-brain LIF
                                             model (Shiu et al. 2024) and read the target group's firing rate
    benchmark_search(target, candidates, known_hits)
                                             exhaustive screen = in-silico ground truth; then measures how many
                                             experiments a connectome-guided order needs vs a random order
                                             (exact expectation) to find the first / all hits, plus agreement
                                             with literature-known hits. Saves data/benchmarks/screen_<target>.json

Honesty notes (also written into every benchmark file):
  * The in-silico ground truth comes from the SAME connectome that the ranking uses, so ranking-vs-screen agreement
    measures how well a cheap linear connectome prior predicts the expensive nonlinear simulation - not biology.
    The non-circular check is the literature agreement (do literature-known hits rank / fire highly?).
  * Literature negatives are not known for most cell types, so precision vs literature cannot be computed; we
    report recall over the literature-known hits and the in-silico hits that are NOT in the literature as
    agent-generated, untested hypotheses.
  * Brain model: zero baseline activity, Poisson activation of every neuron of a type at one rate; real
    optogenetic split-GAL4 activation covers a subset of a type with unknown effective rates.

CLI:
    uv run python -m flylab.screen --benchmark                     (MDN and GF, shared exhaustive screen)
    uv run python -m flylab.screen --benchmark --target MDN
    uv run python -m flylab.screen --rank MDN [--kind visual_projection] [--hops 2]
    uv run python -m flylab.screen --types visual_projection
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmarks"
OUT = ROOT / "spikes" / "screen" / "out"
CACHE_FILE = OUT / "screen_cache.jsonl"

SUPER_CLASSES = ("visual_projection", "descending", "ascending", "sensory", "optic", "central",
                 "visual_centrifugal", "motor", "sensory_ascending", "endocrine")

# Literature-known hits for the benchmark targets. Every entry points to data/ground_truth.json ids
# (DOIs verified via flylab.literature.get_by_doi, verbatim evidence there).
KNOWN_HITS: dict[str, dict[str, dict]] = {
    "MDN": {
        "LC16": {"gt": ["gt20_lc16_activate_backward", "gt21_lc16_activate_backward_via_mdn"],
                 "why": "LC16 activation -> backward walking (Wu 2016); LC16 acts via MDN (Sen 2017)"},
        "LPLC2": {"gt": ["gt19_lplc2_activate_backward"],
                  "why": "LPLC2 activation -> backward walking and jumping with about equal penetrance (Wu 2016 full text); "
                         "that backward walking runs via MDN is our assumption"},
    },
    "GF": {
        "LPLC2": {"gt": ["gt08_lplc2_activate_escape", "gt09_lplc2_silence_escape"],
                  "why": "LPLC2 necessary for GF-mediated escape (Ache 2019); activation -> jumping (Wu 2016)"},
        "LC4": {"gt": ["gt22_lc4_activate_escape"],
                "why": "LC4 activation -> jumping (Wu 2016); LC4 synapses directly onto the GF and conveys looming velocity "
                       "(Ache 2019 abstract; von Reyn 2017). Jumping was not shown to be GF-mediated"},
    },
}
# Broader, weaker literature set for GF: all types with 'highly penetrant jumping' in Wu et al. 2016 (full text).
# Jumping is not necessarily GF-mediated, so this is reported separately.
KNOWN_JUMPING_WU2016 = ["LC4", "LC6", "LC15", "LPLC1", "LPLC2"]

__all__ = ["candidate_types", "rank_by_connectome", "brain_screen", "benchmark_search", "KNOWN_HITS"]


# ----------------------------------------------------------------------------- candidates
@functools.lru_cache(maxsize=1)
def _type_index() -> dict:
    """cell_type -> {ids, super_class, cell_class, n_left, n_right} over the whole annotation (brain-model IDs only)."""
    from flylab import atlas, brain
    df = atlas.load_annotations()
    conn = brain.load_connectome()
    df = df[df["cell_type"].notna()]
    df = df[df["root_id"].isin(np.fromiter(conn.id2idx.keys(), dtype=np.int64))]
    out = {}
    for ct, sdf in df.groupby("cell_type", sort=True):
        sc = sdf["super_class"].mode()
        cc = sdf["cell_class"].dropna().mode()
        out[str(ct)] = {
            "ids": [int(i) for i in sdf["root_id"]],
            "super_class": str(sc.iloc[0]) if len(sc) else None,
            "super_classes": sorted(set(str(x) for x in sdf["super_class"].dropna())),
            "cell_class": str(cc.iloc[0]) if len(cc) else None,
            "n_left": int((sdf["side"] == "left").sum()),
            "n_right": int((sdf["side"] == "right").sum()),
        }
    return out


def candidate_types(kind: str = "visual_projection", min_n: int = 1) -> list[dict]:
    """Cell types of one FlyWire v783 super_class (e.g. 'visual_projection' = LC/LPLC/LLPC/MeTu/... types),
    or, if `kind` is not a super_class, all cell types matching `kind` as a regex (e.g. '^LC1[0-9]').
    Only neurons present in the brain model are counted. Returns
    [{cell_type, n, root_ids, super_class, cell_class, n_left, n_right}] sorted by cell_type."""
    import re
    idx = _type_index()
    out = []
    for ct, d in idx.items():
        if kind in SUPER_CLASSES:
            if d["super_class"] != kind:
                continue
        elif not re.search(kind, ct):
            continue
        if len(d["ids"]) < min_n:
            continue
        out.append({"cell_type": ct, "n": len(d["ids"]), "root_ids": d["ids"], "super_class": d["super_class"],
                    "cell_class": d["cell_class"], "n_left": d["n_left"], "n_right": d["n_right"]})
    return out


def _resolve_ids(name: str) -> list[int]:
    """Root IDs of a cell type (annotation) or, failing that, of an atlas group (e.g. 'LC16_L')."""
    idx = _type_index()
    if name in idx:
        return list(idx[name]["ids"])
    from flylab import atlas
    return atlas.group_ids(name)


def _names(candidates: list | str) -> list[str]:
    if isinstance(candidates, str):
        return [c["cell_type"] for c in candidate_types(candidates)]
    return [c["cell_type"] if isinstance(c, dict) else str(c) for c in candidates]


# ----------------------------------------------------------------------------- connectome ranking
_MATRIX_BUILD_S: list[float] = []   # measured one-time build cost, reported in every benchmark


@functools.lru_cache(maxsize=1)
def _matrices():
    """(Wt, At): Wt = signed synapse counts as CSR (post x pre); At = input-fraction version
    A[i, j] = signed synapses(i -> j) / sum_i |synapses(i -> j)|, transposed (post x pre)."""
    import scipy.sparse as sp
    from flylab import brain
    c = brain.load_connectome()   # shared with the simulator, not counted as ranking cost
    t0 = time.perf_counter()
    # COPY indices/indptr: scipy shares the arrays passed in and sorts the column indices in place during later
    # operations (sum_duplicates/sort_indices), which silently scrambled the simulator's connectome (indices permuted,
    # syn not) for every brain.simulate() run later in the same process. Found in review, 2026-10-04 00:45.
    W = sp.csr_matrix((c.syn.astype(np.float32), c.indices.copy(), c.indptr.copy()), shape=(c.N, c.N))
    in_abs = np.asarray(abs(W).sum(axis=0)).ravel()
    inv = np.zeros_like(in_abs, dtype=np.float32)
    inv[in_abs > 0] = 1.0 / in_abs[in_abs > 0]
    A = W @ sp.diags(inv)
    out = (W.T.tocsr(), A.T.tocsr())
    _MATRIX_BUILD_S.append(time.perf_counter() - t0)
    return out


def _lif_rate(V: np.ndarray) -> np.ndarray:
    """Deterministic LIF transfer function of the brain model (Hz) for a constant mean depolarisation V (mV above
    rest): r = 1 / (t_rfc + t_mbr * ln(V / (V - theta))) for V > theta (theta = v_th - v_0 = 7 mV), else 0."""
    from flylab import brain
    p = brain.DEFAULT_PARAMS
    th = p["v_th_mV"] - p["v_0_mV"]
    V = np.asarray(V, dtype=np.float64)
    out = np.zeros_like(V)
    m = V > th
    out[m] = 1000.0 / (p["t_rfc_ms"] + p["t_mbr_ms"] * np.log(V[m] / (V[m] - th)))
    return out


def _candidate_matrix(names: list[str]):
    import scipy.sparse as sp
    from flylab import brain
    c = brain.load_connectome()
    rows, cols = [], []
    for k, nm in enumerate(names):
        ix, _ = c.index_of(_resolve_ids(nm))
        rows.append(ix)
        cols.append(np.full(ix.size, k))
    r_idx = np.concatenate(rows) if rows else np.empty(0, np.int64)
    c_idx = np.concatenate(cols) if cols else np.empty(0, np.int64)
    X = sp.csr_matrix((np.ones(r_idx.size, np.float32), (r_idx, c_idx)), shape=(c.N, len(names)))
    return X, [int(r.size) for r in rows]


def rank_by_connectome(target_group: str, candidates: list | str = "visual_projection", max_hops: int = 3,
                       method: str = "meanfield", rate_hz: float = 150.0) -> list[dict]:
    """Rank candidate cell types by how strongly the signed connectome predicts they excite `target_group`.
    All neurons of a candidate type are activated together (as in brain_screen). Two documented scores:

    method="meanfield" (default; derived from the brain model's equations, no fitted parameters):
        candidate neurons fire at r0 = rate_hz (each Poisson input spike makes them spike)
        mean depolarisation of neuron j:  V_j = w_syn * tau * sum_i syn_ij * r_i      (mV; syn signed)
        intermediate rates:               r_j = LIF transfer function f(V_j)   (0 below theta = 7 mV)
        (max_hops - 1) fixed-point steps starting from the stimulated set, candidates clamped at r0
        score = mean over target neurons of V_target in mV (direct + multi-hop); theta = 7 mV is the firing threshold,
        pred_rate_hz = mean f(V_target). Signed: inhibitory paths lower V; net-inhibited intermediates fire 0 Hz
        (zero-baseline model, so disinhibition cannot create spikes).
    method="fraction" (input-fraction graph score, cheaper and model-agnostic):
        A[i, j] = signed syn_ij / sum_i |syn_ij|; h1 = A^T x; h_{k+1} = A^T max(h_k, 0) (candidates removed as
        intermediates); score = sum_k mean_t h_k[t]  (direct signed input fraction + normalised 2-hop paths).
    Default max_hops=3 (contract said 2): visual projection -> descending neuron pathways are mostly >= 2 synapses
    long, and a 2-hop mean-field misses LC16 -> MDN, which fires MDN in the brain model (pilot run); all variants
    are reported side by side in the benchmark files.
    `direct_syn` = raw signed synapse count candidate type -> target group.
    Returns [{cell_type, n, score, direct_syn, direct_score, two_hop_score, pred_rate_hz, method, rank, sign_note}]
    sorted by score desc.
    """
    from flylab import atlas, brain
    c = brain.load_connectome()
    Wt, At = _matrices()
    tgt, _ = c.index_of(atlas.group_ids(target_group))
    names = _names(candidates)
    X, ns = _candidate_matrix(names)
    max_hops = max(1, int(max_hops))
    Wt_t = Wt[tgt]
    direct_syn = np.asarray((Wt_t @ X).sum(axis=0)).ravel()
    pred_rate = np.zeros(len(names))
    if method == "meanfield":
        p = brain.DEFAULT_PARAMS
        kmv = p["w_syn_mV"] * p["tau_ms"] / 1000.0          # mV per (synapse x Hz)
        S0 = X * float(rate_hz)
        direct = np.asarray((Wt_t @ S0).toarray() * kmv)     # |T| x K
        R = None
        for _ in range(max_hops - 1):
            V = (Wt @ (S0 if R is None else S0 + R)).tocsr() * kmv
            V.data = _lif_rate(V.data).astype(np.float32)
            V = (V - V.multiply(X)).tocsr()                  # candidates stay clamped at r0
            V.eliminate_zeros()
            R = V
        VT = direct if R is None else np.asarray((Wt_t @ (S0 + R)).toarray() * kmv)
        score = VT.mean(axis=0)
        d_score = direct.mean(axis=0)
        pred_rate = _lif_rate(VT).mean(axis=0)
    elif method == "fraction":
        At_t = At[tgt]
        hop_scores, H = [], X
        for hop in range(1, max_hops + 1):
            hop_scores.append(np.asarray((At_t @ H).mean(axis=0)).ravel())
            if hop < max_hops:
                Hn = (At @ H).tocsr()
                Hn.data = np.maximum(Hn.data, 0)
                Hn = (Hn - Hn.multiply(X)).tocsr()
                Hn.eliminate_zeros()
                H = Hn
        score = np.sum(hop_scores, axis=0)
        d_score = hop_scores[0]
    else:
        raise ValueError("method must be 'meanfield' or 'fraction'")
    out = []
    for k, nm in enumerate(names):
        s, d = float(score[k]), float(d_score[k])
        if direct_syn[k] > 0:
            note = "direct excitatory"
        elif direct_syn[k] < 0:
            note = "direct net inhibitory" + (" + indirect excitatory" if s > d else "")
        elif s > 0:
            note = f"indirect only (<= {max_hops} hops)"
        else:
            note = f"no excitatory path <= {max_hops} hops"
        out.append({"cell_type": nm, "n": ns[k], "score": s, "direct_syn": int(round(float(direct_syn[k]))),
                    "direct_score": d, "two_hop_score": s - d, "pred_rate_hz": float(pred_rate[k]), "method": method,
                    "max_hops": max_hops, "sign_note": note})
    out.sort(key=lambda r: (-r["score"], r["cell_type"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# ----------------------------------------------------------------------------- brain screen
def _cache_key(ct: str, rate_hz: float, duration_ms: float, n_trials: int, seed: int, n_threads: int = 0) -> str:
    """Cache key without n_threads: the thread split only changes the random streams (statistically identical
    experiment); the n_threads actually used is stored in the cached record's params."""
    return f"{ct}|{rate_hz:g}|{duration_ms:g}|{n_trials}|{seed}"


def _load_cache() -> dict:
    out = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                k = d["key"]
                if k.count("|") == 5:            # older records included n_threads in the key
                    k = k.rsplit("|", 1)[0]
                # keep the record with the most group readouts (a re-simulation appended because an older record lacked
                # a target group must win, otherwise that experiment would be re-run on every call)
                if k not in out or len(d.get("group_rates", {})) > len(out[k].get("group_rates", {})):
                    out[k] = d
            except Exception:
                pass
    return out


def brain_screen(candidates: list | str, target_groups: list[str], rate_hz: float = 150.0, duration_ms: float = 500.0,
                 n_trials: int = 2, seed: int = 0, n_threads: int = 2, use_cache: bool = True,
                 progress: bool = False) -> list[dict]:
    """Activate each candidate cell type (all its neurons, Poisson at rate_hz) in the whole-brain LIF model and read
    the mean firing rate of every target group. Returns
    [{cell_type, n_stimulated, target_rates: {group: mean Hz}, target_max: {group: max Hz}, n_active, runtime_s, cached}].
    runtime_s is the measured wall time of the simulation (also when the row is served from the on-disk cache
    spikes/screen/out/screen_cache.jsonl, flagged cached=True). Rates for ALL atlas groups are stored in the cache."""
    from flylab import atlas, brain
    gids = {g: atlas.group_ids(g) for g in atlas.groups()}
    for g in target_groups:
        gids.setdefault(g, atlas.group_ids(g))
    cache = _load_cache() if use_cache else {}
    out = []
    names = _names(candidates)
    t_all = time.perf_counter()
    for i, ct in enumerate(names):
        key = _cache_key(ct, rate_hz, duration_ms, n_trials, seed, n_threads)
        rec = cache.get(key)
        cached = rec is not None and all(g in rec["group_rates"] for g in target_groups)
        if not cached:
            ids = _resolve_ids(ct)
            t0 = time.perf_counter()
            res = brain.simulate(ids, excite_rate_hz=rate_hz, duration_ms=duration_ms, n_trials=n_trials, seed=seed,
                                 n_threads=n_threads)
            wall = time.perf_counter() - t0
            gr, gm = {}, {}
            for g, v in gids.items():
                r = brain.rates_for(res, v)
                vals = list(r.values())
                gr[g] = float(np.mean(vals)) if vals else 0.0
                gm[g] = float(np.max(vals)) if vals else 0.0
            rec = {"key": key, "cell_type": ct, "n_stimulated": len(ids), "group_rates": gr, "group_max": gm,
                   "n_active": res["n_active"], "runtime_s": round(wall, 3), "sim_runtime_s": res["runtime_s"],
                   "params": {"rate_hz": rate_hz, "duration_ms": duration_ms, "n_trials": n_trials, "seed": seed,
                              "n_threads": n_threads}}
            if use_cache:
                OUT.mkdir(parents=True, exist_ok=True)
                with CACHE_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        row = {"cell_type": ct, "n_stimulated": rec["n_stimulated"],
               "target_rates": {g: round(rec["group_rates"][g], 3) for g in target_groups},
               "target_max": {g: round(rec["group_max"][g], 3) for g in target_groups},
               "n_active": rec["n_active"], "runtime_s": rec["runtime_s"], "cached": bool(cached)}
        out.append(row)
        if progress:
            el = time.perf_counter() - t_all
            print(f"[screen] {i + 1}/{len(names)} {ct:12s} n={rec['n_stimulated']:4d} "
                  + " ".join(f"{g}={row['target_rates'][g]:.1f}" for g in target_groups)
                  + f"  {rec['runtime_s']:.1f}s{' (cache)' if cached else ''}  elapsed {el:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------- benchmark maths
def _expected_random_first(N: int, K: int) -> float:
    """E[position of the first hit] for a uniformly random order of N candidates with K hits = (N+1)/(K+1)."""
    return (N + 1) / (K + 1) if K > 0 else float("nan")


def _expected_random_all(N: int, K: int) -> float:
    """E[position of the last hit] (= experiments to find all K hits) = K(N+1)/(K+1)."""
    return K * (N + 1) / (K + 1) if K > 0 else float("nan")


def _p_random_within(N: int, K: int, k: int) -> float:
    """P(a random order finds >= 1 hit within the first k experiments) = 1 - C(N-K, k)/C(N, k)."""
    if K <= 0:
        return 0.0
    k = int(math.floor(k))
    if k >= N - K + 1:
        return 1.0
    return 1.0 - math.comb(N - K, k) / math.comb(N, k)


def _guided_positions(scores: list[float], is_hit: list[bool]) -> dict:
    """Tie-aware expected positions for an order sorted by score desc (ties = random order within the tie block)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    blocks, i = [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        blocks.append(order[i:j + 1])
        i = j + 1
    first = last = None
    s = 0
    for b in blocks:
        h = sum(is_hit[x] for x in b)
        m = len(b)
        if h and first is None:
            first = s + (m + 1) / (h + 1)
        if h:
            last = s + h * (m + 1) / (h + 1)
        s += m
    return {"first": first, "all": last}


def _avg_precision(order_hits: list[bool]) -> float:
    K = sum(order_hits)
    if not K:
        return float("nan")
    s, c = 0.0, 0
    for i, h in enumerate(order_hits, 1):
        if h:
            c += 1
            s += c / i
    return s / K


def _search_stats(names: list[str], scores: dict[str, float], hits: set[str], runtimes: dict[str, float],
                  ranking_s: float) -> dict:
    N, K = len(names), len(hits)
    det = sorted(names, key=lambda n: (-scores[n], n))
    det_hits = [n in hits for n in det]
    k_first_det = det_hits.index(True) + 1 if K else None
    k_all_det = (max(i for i, h in enumerate(det_hits, 1) if h) if K else None)
    tie = _guided_positions([scores[n] for n in names], [n in hits for n in names])
    e_first, e_all = _expected_random_first(N, K), _expected_random_all(N, K)
    mean_rt = float(np.mean([runtimes[n] for n in names]))
    out = {
        "n_candidates": N, "n_hits": K,
        "guided": {
            "experiments_to_first_hit": tie["first"], "experiments_to_all_hits": tie["all"],
            "experiments_to_first_hit_deterministic": k_first_det, "experiments_to_all_hits_deterministic": k_all_det,
            "tie_note": "primary numbers are tie-aware expectations (candidates with equal score tested in random order); "
                        "'deterministic' breaks ties alphabetically",
            "wall_s_to_first_hit": (round(ranking_s + sum(runtimes[n] for n in det[:k_first_det]), 2) if K else None),
            "wall_s_to_all_hits": (round(ranking_s + sum(runtimes[n] for n in det[:k_all_det]), 2) if K else None),
            "average_precision": _avg_precision(det_hits),
            "recall_at": {str(k): (sum(det_hits[:k]) / K if K else None) for k in (5, 10, 20, 50)},
        },
        "random": {
            "experiments_to_first_hit_expected": e_first, "experiments_to_all_hits_expected": e_all,
            "formula": "first hit: (N+1)/(K+1); all hits: K(N+1)/(K+1) (exact expectations over all orders)",
            "wall_s_to_first_hit_expected": round(e_first * mean_rt, 2) if K else None,
            "wall_s_to_all_hits_expected": round(e_all * mean_rt, 2) if K else None,
            "p_random_finds_hit_within_guided_k": (_p_random_within(N, K, tie["first"]) if K else None),
            "p_note": "P(random order finds >= 1 hit within floor(guided experiments_to_first_hit)) = 1 - C(N-K,k)/C(N,k)",
            "average_precision_expected": (K / N if N else None),
            "recall_at_expected": {str(k): (k / N if N else None) for k in (5, 10, 20, 50)},
        },
        "exhaustive": {"experiments": N, "wall_s": round(sum(runtimes[n] for n in names), 1)},
        "discovery_curve": {"note": "cumulative hits found after k experiments (guided = deterministic order; random = "
                                    "exact expectation k*K/N)",
                            "guided_hits": [int(sum(det_hits[:k])) for k in range(1, N + 1)],
                            "random_expected_hits": [round(k * K / N, 3) for k in range(1, N + 1)]},
    }
    if K:
        out["reduction_factor_first_hit"] = round(e_first / tie["first"], 2)
        out["reduction_factor_all_hits"] = round(e_all / tie["all"], 2)
        out["reduction_factor_vs_exhaustive_all_hits"] = round(N / tie["all"], 2)
    return out


def benchmark_search(target_group: str, candidates: list | str = "visual_projection", known_hits: list[str] | None = None,
                     threshold_hz: float = 5.0, max_hops: int = 3, method: str = "meanfield", rate_hz: float = 150.0, duration_ms: float = 500.0,
                     n_trials: int = 2, seed: int = 0, n_threads: int = 2, name: str | None = None, save: bool = True,
                     extra_known: dict[str, list[str]] | None = None, progress: bool = True,
                     seed_check: bool = True) -> dict:
    """Measured acceleration of discovery: connectome-guided vs random order vs exhaustive in-silico screen.

    1. rank all candidates with rank_by_connectome (primary: meanfield, max_hops=3; hit threshold 5 Hz mean target
       rate, fixed before the full screen); all score variants (meanfield/fraction x 1-3 hops) reported as sensitivity
    2. exhaustive brain_screen of every candidate -> in-silico hits (target mean rate >= threshold_hz;
       the model has 0 Hz baseline, so any firing is stimulus-evoked)
    3. experiments needed to the first / all hits for guided order vs the exact random expectation
    4. agreement with literature-known hits (non-circular check)
    5. naive no-connectome baseline (largest cell types first) and, if seed_check, a second-seed replicate of every
       candidate with target rate >= threshold/2 -> 'robust' hits (hit in both seeds) and search stats on them
    Saves data/benchmarks/<name>.json (default screen_<target>.json)."""
    from flylab import atlas, brain
    t_start = time.perf_counter()
    cand_label = candidates if isinstance(candidates, str) else "custom list"
    names = _names(candidates)
    target_group = atlas.resolve_group(target_group)
    known = list(known_hits if known_hits is not None else KNOWN_HITS.get(target_group, {}).keys())

    _matrices()
    build_s = _MATRIX_BUILD_S[0] if _MATRIX_BUILD_S else 0.0
    rankings, rank_s = {}, {}
    variants = sorted({(m, h) for m in ("meanfield", "fraction") for h in (1, 2, 3)} | {(method, max_hops)})
    for m, h in variants:
        t0 = time.perf_counter()
        rankings[f"{m}_{h}hop"] = rank_by_connectome(target_group, names, max_hops=h, method=m, rate_hz=rate_hz)
        rank_s[f"{m}_{h}hop"] = time.perf_counter() - t0
    pkey = f"{method}_{max_hops}hop"
    primary = rankings[pkey]
    score = {r["cell_type"]: r["score"] for r in primary}
    rank = {r["cell_type"]: r["rank"] for r in primary}

    screen = brain_screen(names, [target_group], rate_hz=rate_hz, duration_ms=duration_ms, n_trials=n_trials,
                          seed=seed, n_threads=n_threads, progress=progress)
    rate = {r["cell_type"]: r["target_rates"][target_group] for r in screen}
    runtimes = {r["cell_type"]: float(r["runtime_s"]) for r in screen}
    hits = {n for n in names if rate[n] >= threshold_hz}
    ranking_s = build_s + rank_s[pkey]

    stats = _search_stats(names, score, hits, runtimes, ranking_s)
    def _brief(st: dict) -> dict:
        return {"n_hits": st["n_hits"], "guided_first": st["guided"]["experiments_to_first_hit"],
                "guided_all": st["guided"]["experiments_to_all_hits"],
                "random_first_expected": st["random"]["experiments_to_first_hit_expected"],
                "random_all_expected": st["random"]["experiments_to_all_hits_expected"],
                "reduction_first": st.get("reduction_factor_first_hit"), "reduction_all": st.get("reduction_factor_all_hits"),
                "average_precision": st["guided"]["average_precision"], "recall_at_10": st["guided"]["recall_at"]["10"]}

    known_in = [k for k in known if k in names]
    sens_hops = {}
    for h, rk in rankings.items():
        sc = {r["cell_type"]: r["score"] for r in rk}
        rr = {r["cell_type"]: r["rank"] for r in rk}
        sens_hops[h] = {"vs_in_silico_hits": _brief(_search_stats(names, sc, hits, runtimes, build_s + rank_s[h])),
                        "vs_literature_hits": (_brief(_search_stats(names, sc, set(known_in), runtimes, build_s + rank_s[h]))
                                               if known_in else None),
                        "ranks_of_literature_hits": {k: rr.get(k) for k in known_in},
                        "top10": [r["cell_type"] for r in rk[:10]]}
    sens_thr = {}
    for th in (1.0, 5.0, 10.0, 20.0, 50.0):
        hs = {n for n in names if rate[n] >= th}
        st = _search_stats(names, score, hs, runtimes, ranking_s)
        sens_thr[f"{th:g}Hz"] = {"n_hits": st["n_hits"], "guided_first": st["guided"]["experiments_to_first_hit"],
                                 "random_first_expected": st["random"]["experiments_to_first_hit_expected"],
                                 "guided_all": st["guided"]["experiments_to_all_hits"],
                                 "random_all_expected": st["random"]["experiments_to_all_hits_expected"],
                                 "reduction_first": st.get("reduction_factor_first_hit"),
                                 "reduction_all": st.get("reduction_factor_all_hits")}
    try:
        from scipy.stats import spearmanr
        rho = spearmanr([score[n] for n in names], [rate[n] for n in names]).correlation
        rho = float(rho) if rho == rho else None
    except Exception:
        rho = None

    # ---- naive baseline without connectome: test the largest cell types first (bigger types inject more spikes)
    size = {r["cell_type"]: float(r["n_stimulated"]) for r in screen}
    b_in = _brief(_search_stats(names, size, hits, runtimes, 0.0))
    b_lit = _brief(_search_stats(names, size, set(known_in), runtimes, 0.0)) if known_in else None
    baselines = {"largest_type_first": {
        "note": "order by number of neurons in the type (desc), no connectome; ties tested in random order. A stronger "
                "baseline than random: larger types inject more spikes, so they are more likely to drive anything.",
        "vs_in_silico_hits": b_in, "vs_literature_hits": b_lit,
        "guided_speedup_vs_baseline_first": (round(b_in["guided_first"] / stats["guided"]["experiments_to_first_hit"], 2)
                                             if stats["n_hits"] else None),
        "guided_speedup_vs_baseline_all": (round(b_in["guided_all"] / stats["guided"]["experiments_to_all_hits"], 2)
                                           if stats["n_hits"] else None)}}

    # ---- seed robustness: replicate every candidate near or above threshold with a second seed
    robust = None
    if seed_check and names:
        recheck = [n for n in names if rate[n] >= threshold_hz / 2]
        scr2 = brain_screen(recheck, [target_group], rate_hz=rate_hz, duration_ms=duration_ms, n_trials=n_trials,
                            seed=seed + 1, n_threads=n_threads, progress=False) if recheck else []
        rate2 = {r["cell_type"]: r["target_rates"][target_group] for r in scr2}
        hits2 = {n for n in recheck if rate2[n] >= threshold_hz}
        robust_hits = hits & hits2
        flips = [{"cell_type": n, f"rate_hz_seed{seed}": round(rate[n], 2), f"rate_hz_seed{seed + 1}": round(rate2[n], 2),
                  f"hit_seed{seed}": n in hits, f"hit_seed{seed + 1}": n in hits2} for n in sorted(hits ^ hits2)]
        r_stats = _search_stats(names, score, robust_hits, runtimes, ranking_s) if robust_hits else None
        robust = {
            "note": f"every candidate with target rate >= {threshold_hz / 2:g} Hz (seed {seed}) re-simulated with seed {seed + 1}; "
                    f"candidates below that are assumed non-hits. robust hit = hit in both seeds.",
            "second_seed": seed + 1, "n_rechecked": len(recheck),
            "rechecked_wall_s": round(sum(float(r["runtime_s"]) for r in scr2), 1),
            "hits_seed_a": len(hits), "hits_seed_b": len(hits2), "n_robust_hits": len(robust_hits),
            "robust_hits": sorted(robust_hits, key=lambda n: -rate[n]), "unstable": flips,
            "search_robust_hits": (_brief(r_stats) if r_stats else None),
            "variants_robust_hits": {h: _brief(_search_stats(names, {r["cell_type"]: r["score"] for r in rk}, robust_hits,
                                                             runtimes, build_s + rank_s[h])) if robust_hits else None
                                     for h, rk in rankings.items()},
        }

    # ---- literature (non-circular) check
    lit_rows = []
    for k in known:
        lit_rows.append({"cell_type": k, "in_candidates": k in rate, "in_silico_rate_hz": rate.get(k),
                         "in_silico_hit": k in hits, "connectome_rank": rank.get(k), "connectome_score": score.get(k),
                         "evidence": KNOWN_HITS.get(target_group, {}).get(k)})
    lit_stats = None
    if known_in:
        lit_stats = _search_stats(names, score, set(known_in), runtimes, ranking_s)
        scr_order = {n: rate[n] for n in names}
        lit_stats["screen_rank_of_known"] = {k: 1 + sum(scr_order[n] > scr_order[k] for n in names) for k in known_in}
    recall = (sum(1 for k in known_in if k in hits) / len(known_in)) if known_in else None
    extra = {}
    for label, lst in (extra_known or {}).items():
        li = [k for k in lst if k in rate]
        extra[label] = {"types": lst, "in_silico_hits": [k for k in li if k in hits],
                        "recall": (sum(k in hits for k in li) / len(li)) if li else None,
                        "connectome_ranks": {k: rank.get(k) for k in li},
                        "rates_hz": {k: rate.get(k) for k in li},
                        "guided_vs_random": (lambda s: {"guided_first": s["guided"]["experiments_to_first_hit"],
                                                        "random_first_expected": s["random"]["experiments_to_first_hit_expected"],
                                                        "reduction_first": s.get("reduction_factor_first_hit"),
                                                        "guided_all": s["guided"]["experiments_to_all_hits"],
                                                        "random_all_expected": s["random"]["experiments_to_all_hits_expected"],
                                                        "reduction_all": s.get("reduction_factor_all_hits")})(
                            _search_stats(names, score, set(li), runtimes, ranking_s)) if li else None}

    extra_types = {k for lst in (extra_known or {}).values() for k in lst}
    novel = sorted([n for n in hits if n not in known and n not in extra_types], key=lambda n: -rate[n])
    table = sorted(({"cell_type": n, "n": next(r["n_stimulated"] for r in screen if r["cell_type"] == n),
                     "connectome_rank": rank[n], "score": round(score[n], 6),
                     "direct_syn": next(r["direct_syn"] for r in primary if r["cell_type"] == n),
                     "pred_rate_hz": round(next(r["pred_rate_hz"] for r in primary if r["cell_type"] == n), 2),
                     "rate_hz": round(rate[n], 2), "hit": n in hits, "literature_known": n in known,
                     "runtime_s": runtimes[n]} for n in names), key=lambda r: r["connectome_rank"])
    mean_rt = float(np.mean(list(runtimes.values())))
    g, rnd = stats["guided"], stats["random"]
    headline = None
    if stats["n_hits"]:
        # robustness of the all-hits factor across score variants (>= 2 hops) - reported so the headline is not cherry-picked
        other_all = [v["vs_in_silico_hits"]["reduction_all"] for h, v in sens_hops.items()
                     if h != pkey and not h.endswith("_1hop") and v["vs_in_silico_hits"]["reduction_all"] is not None]
        other_first = [v["vs_in_silico_hits"]["reduction_first"] for h, v in sens_hops.items()
                       if h != pkey and not h.endswith("_1hop") and v["vs_in_silico_hits"]["reduction_first"] is not None]
        headline = (f"{target_group}: among {stats['n_candidates']} candidate types ({stats['n_hits']} in-silico hits at "
                    f">= {threshold_hz:g} Hz), connectome-guided ranking finds the first hit after "
                    f"{g['experiments_to_first_hit']:.1f} experiments vs {rnd['experiments_to_first_hit_expected']:.1f} expected for "
                    f"random order ({stats['reduction_factor_first_hit']}x"
                    + (f"; other >=2-hop score variants {min(other_first)}-{max(other_first)}x" if other_first else "")
                    + f"; naive 'largest type first' order: {b_in['guided_first']:.1f}) and all hits after "
                    f"{g['experiments_to_all_hits']:.1f} vs {rnd['experiments_to_all_hits_expected']:.1f} "
                    f"({stats['reduction_factor_all_hits']}x"
                    + (f"; other >=2-hop variants {min(other_all)}-{max(other_all)}x" if other_all else "")
                    + f"); exhaustive = {stats['n_candidates']} experiments.")
    budget = 20
    if stats["n_hits"]:
        found = stats["discovery_curve"]["guided_hits"][min(budget, stats["n_candidates"]) - 1]
        headline += (f" With a budget of {budget} experiments ({100 * budget / stats['n_candidates']:.0f}% of the screen) the "
                     f"guided order finds {found}/{stats['n_hits']} hits vs {budget * stats['n_hits'] / stats['n_candidates']:.1f} "
                     f"expected for random order.")
        if robust is not None:
            rs = robust["search_robust_hits"]
            headline += (f" Second-seed replicate: {robust['n_robust_hits']}/{stats['n_hits']} hits replicate"
                         + (f"; all robust hits found after {rs['guided_all']:.1f} vs {rs['random_all_expected']:.1f} random "
                            f"({rs['reduction_all']}x)." if rs else "."))
    lit_headline = None
    if lit_stats and lit_stats["n_hits"]:
        one_hop = sens_hops.get(f"{method}_1hop", {}).get("vs_literature_hits") or {}
        lit_headline = (f"Literature check: first literature-verified hit ({', '.join(known_in)}) after "
                        f"{lit_stats['guided']['experiments_to_first_hit']:.1f} guided experiments vs "
                        f"{lit_stats['random']['experiments_to_first_hit_expected']:.1f} expected random "
                        f"({lit_stats.get('reduction_factor_first_hit')}x"
                        + (f"; direct-synapse-only ranking: {one_hop['guided_first']:.1f}" if one_hop.get("guided_first") else "")
                        + f"); in-silico recall of literature hits {sum(k in hits for k in known_in)}/{len(known_in)}"
                        + (f" (missed: {', '.join(k for k in known_in if k not in hits)})" if any(k not in hits for k in known_in) else "")
                        + ".")
    result = {
        "name": name or f"screen_{target_group.lower()}",
        "target_group": target_group,
        "target_root_ids": atlas.group_ids(target_group),
        "candidates": cand_label,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "headline": headline,
        "literature_headline": lit_headline,
        "definitions": {
            "hit": f"mean firing rate of the {target_group} group >= {threshold_hz:g} Hz when all neurons of the candidate type "
                   f"are driven with Poisson input at {rate_hz:g} Hz ({n_trials} trials x {duration_ms:g} ms). Model baseline "
                   "is 0 Hz, so any firing is evoked. Threshold chosen by us before the full screen (not externally "
                   "pre-registered); sensitivity in 'sensitivity_threshold', seed replicate in 'seed_robustness'.",
            "experiment": "one whole-brain LIF simulation (flylab.brain.simulate) activating one candidate cell type",
            "connectome_score": "see flylab.screen.rank_by_connectome docstring. meanfield = predicted mean depolarisation "
                                "(mV, threshold 7 mV) of the target neurons from (max_hops-1) fixed-point steps of the "
                                "brain model's own LIF mean-field equations (no fitted parameters); fraction = direct signed "
                                "input fraction + normalised multi-hop excitatory paths",
            "random_baseline": "exact expectation over uniformly random test orders, not a sampled run",
        },
        "brain_params": {"rate_hz": rate_hz, "duration_ms": duration_ms, "n_trials": n_trials, "seed": seed,
                         "n_threads": n_threads, "model": "LIF after Shiu et al. 2024 (doi:10.1038/s41586-024-07763-9), FlyWire v783"},
        "ranking": {"primary": pkey, "method": method, "max_hops": max_hops, "matrix_build_s": round(build_s, 2),
                    "ranking_s": {h: round(s, 3) for h, s in rank_s.items()},
                    "ranking_note": "matrix_build_s is a one-time cost (sparse connectome matrices), included in the guided "
                                    "wall time; ranking all candidates afterwards takes ranking_s",
                    "spearman_score_vs_rate": rho},
        "search": stats,
        "sensitivity_score_variants": sens_hops,
        "sensitivity_threshold": sens_thr,
        "baselines": baselines,
        "seed_robustness": robust,
        "literature": {"known_hits": lit_rows, "recall_in_silico": recall,
                       "precision": "not computable: literature negatives are unknown for most of the candidate types; "
                                    "lower bound = literature-known fraction of in-silico hits",
                       "precision_lower_bound": (sum(1 for k in known_in if k in hits) / len(hits)) if hits else None,
                       "search_for_literature_hits": lit_stats, "extra_sets": extra},
        "novel_in_silico_hits": {"note": "agent-generated hypotheses: in-silico hits not among the literature-known hits used "
                                         "here (incl. extra_sets) - may be known elsewhere, or model artefacts; untested",
                                 "cell_types": novel[:40], "rates_hz": {n: round(rate[n], 2) for n in novel[:40]}},
        "timing": {"wall_s_per_experiment_mean": round(mean_rt, 3),
                   "wall_s_per_experiment_median": round(float(np.median(list(runtimes.values()))), 3),
                   "wall_s_exhaustive_screen": round(sum(runtimes.values()), 1),
                   "wall_s_benchmark_total": round(time.perf_counter() - t_start, 1),
                   "n_experiments_replayed_from_cache": sum(bool(r["cached"]) for r in screen),
                   "cache_note": "per-experiment wall times are the measured simulation times (also for rows replayed from "
                                 "spikes/screen/out/screen_cache.jsonl); wall_s_benchmark_total is this run's own wall time "
                                 "and is small when the screen is replayed from the cache",
                   "hardware_note": "shared 16-thread Windows laptop CPU, 2 threads per experiment, other jobs running"},
        "limitations": [
            "In-silico ground truth uses the same connectome as the ranking (partly circular); literature agreement is the "
            "independent check, and it covers only a few known hits.",
            "Literature negatives are unknown, so precision vs biology cannot be computed.",
            "All neurons of a type are activated at one Poisson rate; split-GAL4 lines label subsets with unknown rates.",
            "Model has no spontaneous activity, no neuromodulation, no plasticity; only firing of the target group is read out "
            "(not behaviour).",
            "Wall times are for this shared machine; the speed-up in experiment counts is hardware independent.",
            "Score choice: the primary score (meanfield, 3 hops) was chosen after a rank-only check showed that 2-hop scores "
            "miss LC16 -> MDN, whose MDN drive we had seen in a 6-type pilot simulation. The contract's input-fraction score and "
            "all hop counts are reported in 'sensitivity_score_variants'; the first-hit speed-up is robust across >= 2-hop "
            "variants, the all-hits speed-up is not (see there).",
            "The hit threshold (5 Hz mean target rate) is a convention; see 'sensitivity_threshold'.",
            f"Hits near the threshold depend on the random seed ({n_trials} trials x {duration_ms:g} ms; the {target_group} "
            f"group has {len(atlas.group_ids(target_group))} neurons, so the mean-rate resolution is "
            f"{1000.0 / (n_trials * duration_ms * max(1, len(atlas.group_ids(target_group)))):.2g} Hz); see 'seed_robustness' "
            "for which hits replicate with a second seed.",
            "Random order is a weak baseline; 'baselines.largest_type_first' (no connectome, bigger types first) is a "
            "stronger one and also beats random, so only part of the speed-up is due to connectome structure.",
        ],
        "table": table,
    }
    if save:
        BENCH.mkdir(parents=True, exist_ok=True)
        (BENCH / f"{result['name']}.json").write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    return result


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _print_bench(res: dict) -> None:
    s = res["search"]
    print(f"\n=== {res['name']}  target={res['target_group']}  candidates={s['n_candidates']}  hits={s['n_hits']} ===")
    print(res["headline"])
    print(res["literature_headline"])
    for r in res["literature"]["known_hits"]:
        print(f"  literature {r['cell_type']:8s} rate={r['in_silico_rate_hz']} Hz hit={r['in_silico_hit']} "
              f"connectome_rank={r['connectome_rank']}")
    print(f"  wall/experiment {res['timing']['wall_s_per_experiment_mean']} s, exhaustive {res['timing']['wall_s_exhaustive_screen']} s")
    print("  top 10 by connectome:", [(r["cell_type"], r["rate_hz"]) for r in res["table"][:10]])
    for k, v in res["sensitivity_score_variants"].items():
        a, b = v["vs_in_silico_hits"], v["vs_literature_hits"] or {}
        print(f"  variant {k:15s} in-silico: first {a['guided_first']} all {a['guided_all']} red {a['reduction_first']}/{a['reduction_all']} "
              f"AP {a['average_precision']:.2f} | literature: first {b.get('guided_first')} red {b.get('reduction_first')} "
              f"ranks {v['ranks_of_literature_hits']}")
    print("  sensitivity (threshold):", {k: (v["n_hits"], v["reduction_first"], v["reduction_all"])
                                         for k, v in res["sensitivity_threshold"].items()})
    b = res.get("baselines", {}).get("largest_type_first")
    if b:
        print(f"  baseline largest-type-first: first {b['vs_in_silico_hits']['guided_first']} all "
              f"{b['vs_in_silico_hits']['guided_all']} -> guided is {b['guided_speedup_vs_baseline_first']}x / "
              f"{b['guided_speedup_vs_baseline_all']}x faster than this baseline")
    rb = res.get("seed_robustness")
    if rb:
        print(f"  seed replicate: {rb['n_robust_hits']}/{rb['hits_seed_a']} hits replicate; unstable "
              f"{[u['cell_type'] for u in rb['unstable']]}; robust-hit search {rb['search_robust_hits']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Discovery engine: connectome ranking + brain screen + benchmark")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--target", default=None, help="target group (default: MDN and GF)")
    ap.add_argument("--kind", default="visual_projection")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--method", default="meanfield", choices=["meanfield", "fraction"])
    ap.add_argument("--rank", metavar="TARGET")
    ap.add_argument("--types", metavar="KIND")
    ap.add_argument("--limit", type=int, default=0, help="only the first N candidates (smoke test)")
    ap.add_argument("--no-seed-check", action="store_true", help="skip the second-seed replicate of near-threshold hits")
    a = ap.parse_args(argv)
    if a.types:
        ts = candidate_types(a.types)
        print(f"{len(ts)} cell types, {sum(t['n'] for t in ts)} neurons")
        print(", ".join(f"{t['cell_type']}({t['n']})" for t in ts[:80]))
    if a.rank:
        for r in rank_by_connectome(a.rank, a.kind, a.hops, method=a.method)[:25]:
            print(f"{r['rank']:4d} {r['cell_type']:12s} n={r['n']:4d} score={r['score']:.5f} direct_syn={r['direct_syn']:6d} "
                  f"2hop={r['two_hop_score']:.5f} {r['sign_note']}")
    if a.benchmark:
        cands: list | str = a.kind
        if a.limit:
            cands = _names(a.kind)[: a.limit]
        targets = [a.target] if a.target else ["MDN", "GF"]
        for t in targets:
            extra = {"jumping_wu2016": KNOWN_JUMPING_WU2016} if t.upper().startswith("GF") else None
            res = benchmark_search(t, cands, threshold_hz=a.threshold, max_hops=a.hops, method=a.method, extra_known=extra,
                                   save=not a.limit, name=None, seed_check=not a.no_seed_check)
            _print_bench(res)
    if not (a.types or a.rank or a.benchmark):
        ap.print_help()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    main()
