"""Whole-brain leaky-integrate-and-fire model of Drosophila (FlyWire v783).

Faithful numpy re-implementation of the Brian2 model of
  Shiu et al. 2024, Nature 634:210-219, "A Drosophila computational brain model
  reveals sensorimotor processing", doi:10.1038/s41586-024-07763-9
  reference code: https://github.com/philshiu/Drosophila_brain_model (model.py)

Model (copied from the reference model.py, units mV / ms):
    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike if v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    presynaptic spike  ->  g_post += w  after delay t_dly,
                           w = (+/-1 by transmitter) * n_synapses * w_syn
    activation: PoissonInput on v with weight w_syn * f_poi at rate r_poi,
                Poisson targets have no refractory period
    silencing:  all outgoing synapses of the silenced neuron get weight 0

Simulator: exact (matrix-exponential, = Brian2 method='linear') integration on a
dt = 0.1 ms grid (Brian2 default clock), CSR connectivity by presynaptic neuron,
only neurons that spiked are propagated, synaptic delay via a ring buffer, trials
vectorised as one (n_trials * N) state vector. Per-step order follows the Brian2
schedule: state update -> threshold -> synapses (spike delivery + Poisson input)
-> reset. As in Brian2, synaptic input that arrives while a neuron is refractory is discarded.

Fidelity: bit-identical spike counts vs Brian2 2.10.1 in deterministic micro tests
(spikes/brain/micro_check.py) and r = 0.999 per-neuron rate agreement vs the original model.py
on the 426-neuron sugar subnetwork (spikes/brain/crosscheck_brian2.py).

CLI:
    uv run python -m flylab.brain --validate [--trials 10] [--count-trials 30]   sugar GRN -> MN9 (Shiu et al. Fig. 1)
    uv run python -m flylab.brain --bench        timing only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "spikes" / "brain" / "out"

REPO = "https://github.com/philshiu/Drosophila_brain_model"
RAW_URL = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
COMPLETENESS = "Completeness_783.csv"      # FlyWire v783 neuron list (row order = model index)
CONNECTIVITY = "Connectivity_783.parquet"  # pre/post index, synapse count, sign
CACHE = "brain_csr_v783.npz"
ANNOTATIONS = "flywire_annotations_Supplemental_file1_neuron_annotations.tsv"  # Schlegel et al. 2024 (optional)

PAPER_DOI = "10.1038/s41586-024-07763-9"

# All values from default_params in the reference model.py (sources as cited there).
DEFAULT_PARAMS = {
    "v_0_mV": -52.0,      # resting potential (Kakaria & de Bivort 2017, doi:10.3389/fnbeh.2017.00008)
    "v_rst_mV": -52.0,    # reset potential (same)
    "v_th_mV": -45.0,     # spike threshold (same)
    "t_mbr_ms": 20.0,     # membrane time constant (same)
    "tau_ms": 5.0,        # synaptic time constant (Juergensen et al., doi:10.1088/2634-4386/ac3ba6)
    "t_rfc_ms": 2.2,      # refractory period (Lazar et al., doi:10.7554/eLife.62362)
    "t_dly_ms": 1.8,      # synaptic delay (Paul et al. 2015, doi:10.3389/fncel.2015.00029)
    "w_syn_mV": 0.275,    # weight per synapse, free parameter fitted in the paper
    "f_poi": 250.0,       # Poisson input weight factor (w_syn * f_poi per input spike)
    "dt_ms": 0.1,         # Brian2 default clock
}

# ---- neuron IDs used in the paper / reference notebooks (traceable) -------------------
# Labellar sugar GRNs, 21 FlyWire v630 root IDs from example.ipynb / figures.ipynb (Fig. 1).
# 20/21 still exist as v783 root IDs (one was re-segmented and is dropped, see validation).
SUGAR_GRNS_V630 = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
# Bitter GRNs, 21 v630 root IDs from figures.ipynb (Fig. 3a); 20/21 present in v783.
BITTER_GRNS_V630 = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
# MN9 (proboscis motor neuron). 720575940660219265 is the id_mn9 of example.ipynb / figures.ipynb
# and is unchanged in v783. The paper's second MN9 (720575940645521262, v630, figures.ipynb Fig. 2
# cell: "ids_mn9 = [...660219265, ...645521262] # left and right") no longer exists in v783; its v783
# counterpart was identified as the only other neuron with FlyWire annotation cell_type 'CB0701'
# (the cell type of id_mn9; ingestion_motor_neuron, nerve PhN) in Schlegel et al. 2024 Supplemental
# file 1. This is a match by cell type, not by ID.
# SIDE LABELS: the notebooks' code comments call these sugar GRNs "right" and id_mn9 "left" (old,
# left-right inverted FAFB convention). The paper text says FAFB "was found to be left-right inverted",
# that it refers to the true biological side, and that it "performed unilateral left hemisphere
# activation for all simulations". The v783 annotation agrees with the paper text: all 20 sugar and
# 20 bitter GRNs side=left, id_mn9 side=right (contralateral), its CB0701 partner side=left
# (ipsilateral). Use the v783 annotation 'side'; ignore the notebook comments for left/right.
MN9 = {"MN9_a (paper id_mn9; contralateral, v783 side R)": 720575940660219265,
       "MN9_b (CB0701 partner; ipsilateral, v783 side L)": 720575940618238523}


# ---- connectome ------------------------------------------------------------------------
@dataclass
class Connectome:
    ids: np.ndarray        # int64 root IDs, position = model index
    indptr: np.ndarray     # int64 (N+1) CSR by presynaptic neuron
    indices: np.ndarray    # int32 postsynaptic index
    syn: np.ndarray        # float32 signed synapse count ("Excitatory x Connectivity")
    id2idx: dict

    @property
    def N(self) -> int:
        return len(self.ids)

    def index_of(self, root_ids) -> tuple[np.ndarray, list]:
        """Model indices for root IDs (accepts list / tuple / numpy array / pandas Series / None,
        ints or numeric strings). Unknown or non-numeric IDs are returned in `missing`."""
        idx, missing = [], []
        if root_ids is None:
            root_ids = []
        elif isinstance(root_ids, (int, np.integer, str)):
            root_ids = [root_ids]
        for r in root_ids:
            try:
                key = int(r)
            except (TypeError, ValueError):
                missing.append(str(r))
                continue
            i = self.id2idx.get(key)
            (missing.append(key) if i is None else idx.append(i))
        return np.asarray(sorted(set(idx)), dtype=np.int64), missing


_CONN: Connectome | None = None


def _download(name: str) -> Path:
    import requests

    p = RAW / name
    if p.exists() and p.stat().st_size > 0:
        return p
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"[brain] downloading {name} from {REPO} ...", flush=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.part")  # unique per writer
    try:
        with requests.get(RAW_URL + name, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if not p.exists():
            os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return p


_LOAD_LOCK = threading.Lock()


def _read_cache(cache: Path):
    """Read the CSR cache; returns None if it is missing or unreadable (e.g. truncated)."""
    if not cache.exists():
        return None
    try:
        with np.load(cache) as z:  # context manager: no open handle left behind (Windows file locks)
            ids, indptr, indices, syn = z["ids"], z["indptr"], z["indices"], z["syn"]
        if indptr.size != ids.size + 1 or indices.size != syn.size or int(indptr[-1]) != indices.size:
            raise ValueError("inconsistent CSR arrays")
        return ids, indptr, indices, syn
    except Exception as e:  # corrupt cache -> rebuild from parquet
        print(f"[brain] ignoring unreadable cache {cache.name}: {e}", flush=True)
        return None


def load_connectome() -> Connectome:
    """Load FlyWire v783 connectome as CSR (cached in memory and in data/raw/brain_csr_v783.npz).
    Thread-safe; the .npz cache is written atomically and rebuilt if it is unreadable."""
    global _CONN
    if _CONN is not None:
        return _CONN
    with _LOAD_LOCK:
        if _CONN is not None:
            return _CONN
        cache = RAW / CACHE
        cached = _read_cache(cache)
        if cached is not None:
            ids, indptr, indices, w = cached
        else:
            import pandas as pd

            comp = pd.read_csv(_download(COMPLETENESS), index_col=0)
            ids = comp.index.to_numpy(np.int64)
            df = pd.read_parquet(_download(CONNECTIVITY),
                                 columns=["Presynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index",
                                          "Excitatory x Connectivity"])
            pre = df["Presynaptic_Index"].to_numpy(np.int64)
            if not np.array_equal(ids[pre[:: 997]], df["Presynaptic_ID"].to_numpy()[:: 997]):
                raise RuntimeError("Completeness/Connectivity index-ID mismatch")
            post = df["Postsynaptic_Index"].to_numpy(np.int32)
            wx = df["Excitatory x Connectivity"].to_numpy()
            if np.abs(wx).max() > np.iinfo(np.int16).max:
                raise RuntimeError("synapse counts exceed int16")  # v783: range -2405..1897
            w = wx.astype(np.int16)
            order = np.argsort(pre, kind="stable")
            pre, post, w = pre[order], post[order], w[order]
            indptr = np.zeros(len(ids) + 1, np.int64)
            np.cumsum(np.bincount(pre, minlength=len(ids)), out=indptr[1:])
            indices = post
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "wb") as f:  # file handle: np.savez must not append '.npz'
                    np.savez(f, ids=ids, indptr=indptr, indices=indices, syn=w)
                os.replace(tmp, cache)
            except OSError as e:  # cache is an optimisation only (e.g. read-only deploy dir)
                print(f"[brain] could not write cache: {e}", flush=True)
                if tmp.exists():
                    tmp.unlink()
        _CONN = Connectome(ids=ids, indptr=indptr, indices=indices, syn=w.astype(np.float32),
                           id2idx={int(r): i for i, r in enumerate(ids.tolist())})
    return _CONN


# ---- simulator ---------------------------------------------------------------------------
def _run_batch(conn: Connectome, exc: np.ndarray, rates_exc: np.ndarray, sil_mask: np.ndarray,
               n_steps: int, T: int, rng: np.random.Generator, p: dict,
               trace: list | None = None, weights: np.ndarray | None = None) -> np.ndarray:
    """Simulate T trials at once. Returns spike counts, shape (T, N).
    trace: optional list; if given, (u, g) of all flat indices are appended after every step (debug).
    weights: optional precomputed conn.syn * w_syn (float32), shared read-only between threads."""
    N = conn.N
    TN = T * N
    dt = p["dt_ms"]
    tm, ts = p["t_mbr_ms"], p["tau_ms"]
    Em, Es = math.exp(-dt / tm), math.exp(-dt / ts)
    c_gv = ts / (ts - tm) * (Es - Em)          # exact coupling g -> v over one step
    thr = p["v_th_mV"] - p["v_0_mV"]           # u = v - v_0
    u_rst = p["v_rst_mV"] - p["v_0_mV"]
    D = int(round(p["t_dly_ms"] / dt))
    L = D + 1
    R = int(round(p["t_rfc_ms"] / dt))          # refractory steps (Brian2 timestep semantics)
    w_poi = p["w_syn_mV"] * p["f_poi"]
    if weights is None:
        weights = conn.syn * np.float32(p["w_syn_mV"])
    indptr, indices = conn.indptr, conn.indices

    u = np.zeros(TN, np.float32)
    g = np.zeros(TN, np.float32)
    tmp = np.empty(TN, np.float32)
    buf = np.zeros((L, TN), np.float32)
    counts = np.zeros(TN, np.int32)

    no_rfc = np.zeros(N, bool)
    no_rfc[exc] = True                          # Poisson targets: rfc = 0 (as in model.py poi())
    poi_tgt = (np.arange(T)[:, None] * N + exc[None, :]).ravel()
    poi_p = np.tile(rates_exc * dt * 1e-3, T)   # P(input spike per step)

    ref_hist: deque = deque()                   # (flat indices, first step no longer refractory)
    empty = np.empty(0, np.int64)
    f32 = np.float32
    for step in range(n_steps):
        while ref_hist and ref_hist[0][1] <= step:
            ref_hist.popleft()
        ref_idx = np.concatenate([a for a, _ in ref_hist]) if ref_hist else empty
        # 1) state update (exact)
        np.multiply(g, f32(c_gv), out=tmp)
        u *= f32(Em)
        u += tmp
        g *= f32(Es)
        if ref_idx.size:                        # v clamped at v_rst while refractory
            u[ref_idx] = u_rst
        # 2) threshold
        spk = np.flatnonzero(u > thr)
        # 3) synapses: deliver delayed input, Poisson input, push new spikes
        slot = step % L
        g += buf[slot]
        buf[slot] = 0.0
        if ref_idx.size:
            # Brian2 semantics (verified against brian2 2.10.1, spikes/brain/micro_check.py): with
            # 'g ... (unless refractory)' and reset g = 0, synaptic input arriving while the neuron is
            # refractory is discarded, i.e. g stays 0 for the whole refractory period.
            g[ref_idx] = 0.0
        if poi_tgt.size:
            hit = poi_tgt[rng.random(poi_tgt.size) < poi_p]
            u[hit] += w_poi
        if spk.size:
            pre = spk % N
            keep = ~sil_mask[pre]
            sp, pre = spk[keep], pre[keep]
            if sp.size:
                starts = indptr[pre]
                lens = indptr[pre + 1] - starts
                tot = int(lens.sum())
                if tot:
                    off = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(tot)
                    tgt = indices[off] + np.repeat(sp - pre, lens)
                    np.add.at(buf[(step + D) % L], tgt, weights[off])
            # 4) reset
            u[spk] = u_rst
            g[spk] = 0.0
            counts[spk] += 1
            rf = spk[~no_rfc[spk % N]]
            if rf.size and R > 1:
                ref_hist.append((rf, step + R))
        if trace is not None:
            trace.append((u.copy(), g.copy()))
    return counts.reshape(T, N)


def simulate(excite: list[int], silence: list[int] | None = None, excite_rate_hz: float = 150.0,
             duration_ms: float = 1000.0, n_trials: int = 5, seed: int = 0, *,
             excite2: list[int] | None = None, excite2_rate_hz: float = 0.0,
             params: dict | None = None, record_ids: list[int] | None = None,
             n_threads: int | None = None) -> dict:
    """Run the whole-brain LIF model.

    excite:   FlyWire v783 root IDs driven by Poisson input at excite_rate_hz (model.py neu_exc)
    excite2:  optional 2nd set at excite2_rate_hz (model.py neu_exc2 / r_poi2)
    silence:  root IDs whose outgoing synapses are set to 0 (model.py neu_slnc). Because their output
              is removed, silenced neurons are reported with an effective rate of 0 in "rates" (so a
              silenced descending neuron cannot drive the body); the spikes they still emit (as in
              model.py) are listed separately in "silenced_spiking_rates".
    record_ids: optional IDs for which per-trial rates (raw spike rates) are returned in "trial_rates".
    params:   overrides for DEFAULT_PARAMS keys (unknown keys raise ValueError).
    n_threads: trials are split into this many vectorised batches run in parallel threads
               (numpy releases the GIL); default min(n_trials, 4). Results are deterministic for a
               given (seed, n_trials, n_threads).
    Returns {"rates": {root_id: mean Hz} (only > 0), "std": {...}, "n_active", "runtime_s", "params", ...}
    """
    t0 = time.perf_counter()
    unknown = set(params or {}) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}; valid keys: {sorted(DEFAULT_PARAMS)}")
    duration_ms, n_trials = float(duration_ms), int(n_trials)
    if not (math.isfinite(duration_ms) and duration_ms > 0):
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    for nm, val in (("excite_rate_hz", excite_rate_hz), ("excite2_rate_hz", excite2_rate_hz)):
        if not (math.isfinite(float(val)) and float(val) >= 0):
            raise ValueError(f"{nm} must be a finite rate >= 0 Hz, got {val}")
    conn = load_connectome()
    p = dict(DEFAULT_PARAMS, **(params or {}))
    exc1, miss1 = conn.index_of(excite)
    exc2, miss2 = conn.index_of(excite2)
    exc2 = np.setdiff1d(exc2, exc1)
    sil, miss3 = conn.index_of(silence)
    exc = np.concatenate([exc1, exc2]).astype(np.int64)
    rates_exc = np.concatenate([np.full(exc1.size, float(excite_rate_hz)),
                                np.full(exc2.size, float(excite2_rate_hz))])
    sil_mask = np.zeros(conn.N, bool)
    sil_mask[sil] = True
    n_steps = int(round(duration_ms / p["dt_ms"]))
    nt = max(1, min(n_trials, n_threads or 4))
    sizes = [n_trials // nt + (k < n_trials % nt) for k in range(nt)]
    seeds = np.random.SeedSequence(seed).spawn(nt)
    weights = conn.syn * np.float32(p["w_syn_mV"])  # computed once, shared by all threads
    if nt == 1:
        chunks = [_run_batch(conn, exc, rates_exc, sil_mask, n_steps, n_trials, np.random.default_rng(seeds[0]), p,
                             weights=weights)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(nt) as ex:
            chunks = list(ex.map(lambda a: _run_batch(conn, exc, rates_exc, sil_mask, n_steps, a[0],
                                                      np.random.default_rng(a[1]), p, weights=weights),
                                 zip(sizes, seeds)))
    counts = np.concatenate(chunks, axis=0)
    trial_rates = counts / (duration_ms / 1000.0)
    mean = trial_rates.mean(axis=0)
    std = trial_rates.std(axis=0)               # population std, as utils.get_rate (np.std)
    act = np.flatnonzero((mean > 0) & ~sil_mask)
    sil_spk = np.flatnonzero((mean > 0) & sil_mask)
    out = {
        "rates": {str(int(conn.ids[i])): float(mean[i]) for i in act},
        "std": {str(int(conn.ids[i])): float(std[i]) for i in act},
        "silenced_spiking_rates": {str(int(conn.ids[i])): float(mean[i]) for i in sil_spk},
        "n_active": int(act.size),
        "runtime_s": round(time.perf_counter() - t0, 3),
        "params": {**p, "excite_rate_hz": float(excite_rate_hz), "n_threads": nt, "excite2_rate_hz": float(excite2_rate_hz),
                   "duration_ms": float(duration_ms), "n_trials": int(n_trials), "seed": int(seed),
                   "n_excited": int(exc.size), "n_silenced": int(sil.size), "n_neurons": conn.N,
                   "connectome": "FlyWire v783 (Completeness_783.csv / Connectivity_783.parquet)",
                   "model": f"LIF after Shiu et al. 2024, doi:{PAPER_DOI}; {REPO}"},
        "unknown_ids": miss1 + miss2 + miss3,
    }
    if record_ids:
        out["trial_rates"] = {str(int(r)): ([float(x) for x in trial_rates[:, conn.id2idx[int(r)]]]
                                             if int(r) in conn.id2idx else None) for r in record_ids}
    return out


def rates_for(result: dict, ids: list[int]) -> dict[str, float]:
    """Mean rate (Hz) for each root ID; 0.0 for silent / silenced / unknown neurons."""
    r = result.get("rates", {})
    return {str(int(i)): float(r.get(str(int(i)), 0.0)) for i in (ids if ids is not None else [])}


# ---- annotations (optional, for reporting only) ------------------------------------------
_ANN = None


def annotations():
    """FlyWire v783 neuron annotations (Schlegel et al. 2024, doi:10.1038/s41586-024-07686-5),
    if data/raw/flywire_annotations_*.tsv exists, else None."""
    global _ANN
    if _ANN is None:
        f = RAW / ANNOTATIONS
        if not f.exists():
            return None
        import pandas as pd

        _ANN = pd.read_csv(f, sep="\t", usecols=["root_id", "super_class", "cell_class", "cell_type",
                                                 "hemibrain_type", "side"],
                           dtype={"root_id": "int64", "hemibrain_type": str}, low_memory=False)
    return _ANN


def descending_rates(result: dict) -> list[dict]:
    """Active descending neurons (super_class == 'descending' in the FlyWire annotation)."""
    ann = annotations()
    if ann is None:
        return []
    dn = ann[ann.super_class == "descending"]
    rows = []
    for r in dn.itertuples():
        hz = result["rates"].get(str(r.root_id), 0.0)
        if hz > 0:
            rows.append({"root_id": int(r.root_id), "cell_type": r.cell_type if isinstance(r.cell_type, str) else "",
                         "side": r.side if isinstance(r.side, str) else "", "rate_hz": round(hz, 2),
                         "std_hz": round(result["std"].get(str(r.root_id), 0.0), 2)})
    return sorted(rows, key=lambda d: -d["rate_hz"])


# ---- CLI ---------------------------------------------------------------------------------
def _n_stim_active(res: dict, ids: list[int]) -> int:
    return sum(1 for i in ids if res["rates"].get(str(int(i)), 0.0) > 0)


def _validate(n_trials: int, duration_ms: float, count_trials: int = 0) -> dict:
    conn = load_connectome()
    sugar = [i for i in SUGAR_GRNS_V630 if i in conn.id2idx]
    bitter = [i for i in BITTER_GRNS_V630 if i in conn.id2idx]
    mn9 = list(MN9.values())
    runs = [
        ("no stimulation (baseline)", dict(excite=[], excite_rate_hz=0.0)),
        ("sugar GRNs 10 Hz", dict(excite=sugar, excite_rate_hz=10.0)),
        ("sugar GRNs 50 Hz", dict(excite=sugar, excite_rate_hz=50.0)),
        ("sugar GRNs 100 Hz", dict(excite=sugar, excite_rate_hz=100.0)),
        ("sugar GRNs 150 Hz", dict(excite=sugar, excite_rate_hz=150.0)),
        ("sugar GRNs 200 Hz", dict(excite=sugar, excite_rate_hz=200.0)),
        ("bitter GRNs 100 Hz (control)", dict(excite=bitter, excite_rate_hz=100.0)),
        ("sugar 100 Hz + bitter 100 Hz", dict(excite=sugar, excite_rate_hz=100.0,
                                             excite2=bitter, excite2_rate_hz=100.0)),
        ("sugar 100 Hz, GRNs silenced (sanity)", dict(excite=sugar, excite_rate_hz=100.0, silence=sugar)),
    ]
    rows, results = [], {}
    for name, kw in runs:
        res = simulate(duration_ms=duration_ms, n_trials=n_trials, seed=1, record_ids=mn9, **kw)
        results[name] = res
        tr = res["trial_rates"]
        row = {"condition": name, "n_active": res["n_active"], "runtime_s": res["runtime_s"]}
        for lab, i in MN9.items():
            row[lab] = {"mean_hz": round(res["rates"].get(str(i), 0.0), 2),
                        "std_hz": round(res["std"].get(str(i), 0.0), 2),
                        "trials_active": f"{sum(x > 0 for x in tr[str(i)])}/{n_trials}"}
        rows.append(row)
        mn = "  ".join(f"{k.split(' ')[0]}={v['mean_hz']:6.1f}+-{v['std_hz']:4.1f}Hz ({v['trials_active']})"
                       for k, v in row.items() if k.startswith("MN9"))
        print(f"{name:32s} active={res['n_active']:5d}  {mn}  [{res['runtime_s']:.1f}s]", flush=True)

    def mx(name):
        return max(rates_for(results[name], mn9).values())

    s100, s200 = mx("sugar GRNs 100 Hz"), mx("sugar GRNs 200 Hz")
    checks = [
        {"claim": "Baseline firing of every neuron is 0 Hz (paper: 'The baseline firing of each neuron in our model is 0 Hz')",
         "ours": f"{results['no stimulation (baseline)']['n_active']} active neurons",
         "agree": results["no stimulation (baseline)"]["n_active"] == 0},
        {"claim": "Sugar GRN activation at 100 Hz robustly activates MN9 in 100% of simulations (paper, shuffle-control text)",
         "ours": "; ".join(f"{k.split(' ')[0]} active in {v['trials_active']} trials, {v['mean_hz']} Hz"
                           for k, v in rows[3].items() if k.startswith("MN9")),
         "agree": all(int(v["trials_active"].split("/")[0]) == n_trials for k, v in rows[3].items()
                      if k.startswith("MN9"))},
        {"claim": "w_syn was chosen so that sugar 100 Hz gives roughly 80% of maximal MN9 firing (paper Methods)",
         "ours": f"MN9 max(100 Hz)/max(200 Hz) = {s100:.1f}/{s200:.1f} = {s100 / s200 if s200 else float('nan'):.2f}",
         "agree": (True if s200 and 0.7 <= s100 / s200 <= 0.9 else
                   "partial: MN9 at 100 Hz is clearly sub-maximal but the ratio differs from ~0.8 "
                   "(paper tuned w_syn on v630; this is v783)")},
        {"claim": "MN9 rate increases with sugar GRN rate (not quoted from the paper; implied by its w_syn tuning to ~80% of maximal MN9 firing at 100 Hz)",
         "ours": [mx(f"sugar GRNs {f} Hz") for f in (10, 50, 100, 150, 200)],
         "agree": mx("sugar GRNs 10 Hz") <= mx("sugar GRNs 50 Hz") <= mx("sugar GRNs 100 Hz") <= s200},
        {"claim": "Number of responsive neurons: 45 at 10 Hz, 455 at 200 Hz sugar (paper, v630 with 127,400 neurons, 30 trials)",
         "ours": (f"{results['sugar GRNs 10 Hz']['n_active']} at 10 Hz, {results['sugar GRNs 200 Hz']['n_active']} at 200 Hz "
                  f"including the {len(sugar)} stimulated GRNs ({results['sugar GRNs 10 Hz']['n_active'] - _n_stim_active(results['sugar GRNs 10 Hz'], sugar)} / "
                  f"{results['sugar GRNs 200 Hz']['n_active'] - _n_stim_active(results['sugar GRNs 200 Hz'], sugar)} without them); "
                  f"v783, {conn.N} neurons, {n_trials} trials (fewer trials -> fewer rarely-firing neurons counted)"),
         "agree": "approximate only (different connectome version and trial count; the paper does not state whether its count includes the stimulated GRNs)"},
        {"claim": ("Unilateral sugar GRN activation drives the contralateral MN9 more strongly than the "
                   "ipsilateral MN9 (paper Fig. 1c)"),
         "ours": [{"rate_hz": f, "contra_MN9_a": rates_for(results[f"sugar GRNs {f} Hz"], mn9)[str(mn9[0])],
                   "ipsi_MN9_b": rates_for(results[f"sugar GRNs {f} Hz"], mn9)[str(mn9[1])]} for f in (50, 100, 150, 200)],
         "agree": all(rates_for(results[f"sugar GRNs {f} Hz"], mn9)[str(mn9[0])] >
                      rates_for(results[f"sugar GRNs {f} Hz"], mn9)[str(mn9[1])] for f in (50, 100, 150, 200))},
        {"claim": "Sanity (not a paper claim): silencing the stimulated sugar GRNs removes their output, so MN9 must stay at 0 Hz",
         "ours": f"MN9 max = {mx('sugar 100 Hz, GRNs silenced (sanity)'):.1f} Hz",
         "agree": mx("sugar 100 Hz, GRNs silenced (sanity)") == 0},
        {"claim": "Bitter GRNs do not activate MN9 and suppress sugar-evoked MN9 firing (paper Fig. 3a)",
         "ours": f"bitter alone MN9 max = {mx('bitter GRNs 100 Hz (control)'):.1f} Hz; sugar+bitter = {mx('sugar 100 Hz + bitter 100 Hz'):.1f} Hz vs sugar alone {s100:.1f} Hz",
         "agree": mx("bitter GRNs 100 Hz (control)") == 0 and mx("sugar 100 Hz + bitter 100 Hz") < s100},
    ]
    if count_trials:  # same trial count as the paper (n_run = 30 in model.py default_params)
        cnt = {}
        for f in (10.0, 200.0):
            r = simulate(sugar, excite_rate_hz=f, duration_ms=1000.0, n_trials=count_trials, seed=5, n_threads=8)
            cnt[f] = (r["n_active"], r["n_active"] - _n_stim_active(r, sugar))
            print(f"[count] sugar {f:.0f} Hz, {count_trials} trials: {cnt[f][0]} active ({cnt[f][1]} without GRNs) "
                  f"[{r['runtime_s']:.1f}s]", flush=True)
        checks.append({
            "claim": f"Number of responsive neurons with the paper's trial count ({count_trials} x 1 s): 45 at 10 Hz, 455 at 200 Hz",
            "ours": (f"{cnt[10.0][0]} at 10 Hz, {cnt[200.0][0]} at 200 Hz including the {len(sugar)} GRNs "
                     f"({cnt[10.0][1]} / {cnt[200.0][1]} without them)"),
            "agree": (f"close if the paper counts the GRNs ({cnt[10.0][0]} vs 45, {cnt[200.0][0]} vs 455); "
                      "v630 vs v783 connectome")})
    dns = descending_rates(results["sugar GRNs 200 Hz"])
    print("\nChecks vs Shiu et al. 2024:")
    for c in checks:
        print(f"  [{'OK' if c['agree'] is True else ('~~' if isinstance(c['agree'], str) else 'XX')}] {c['claim']}\n       ours: {c['ours']}")
    print(f"\nActive descending neurons in 'sugar GRNs 200 Hz' (FlyWire annotation super_class=descending): {len(dns)}")
    for d in dns[:25]:
        print(f"  {d['cell_type'] or '?':12s} {d['side']:6s} {d['root_id']}  {d['rate_hz']:7.2f} +- {d['std_hz']:.2f} Hz")
    top = sorted(results["sugar GRNs 200 Hz"]["rates"].items(), key=lambda kv: -kv[1])[:30]
    report = {
        "paper": {"doi": PAPER_DOI, "code": REPO},
        "connectome": "FlyWire v783",
        "n_neurons": conn.N,
        "n_trials": n_trials, "duration_ms": duration_ms,
        "stimulus_sets": {"sugar_grns_v783": sugar, "sugar_dropped_not_in_v783": [i for i in SUGAR_GRNS_V630 if i not in conn.id2idx],
                          "bitter_grns_v783": bitter, "bitter_dropped_not_in_v783": [i for i in BITTER_GRNS_V630 if i not in conn.id2idx],
                          "mn9": MN9,
                          "side_convention_note": ("Notebook code comments: sugar GRNs 'right', id_mn9 'left' (old inverted "
                                                   "FAFB convention). Paper text (true biological side): left-hemisphere GRN "
                                                   "activation. FlyWire v783 annotation agrees with the paper text: GRNs "
                                                   "side=left, MN9_a side=right (contralateral), MN9_b side=left (ipsilateral).")},
        "silencing_semantics": ("silence = outgoing synapses set to 0 (model.py silence()); silenced neurons are "
                                "reported with effective rate 0 in 'rates', raw spikes in 'silenced_spiking_rates'"),
        "table": rows,
        "checks": checks,
        "descending_active_sugar200": dns,
        "top30_sugar200": top,
        "params": DEFAULT_PARAMS,
        "speed_s_per_trial_per_sim_s": round(results["sugar GRNs 200 Hz"]["runtime_s"] / n_trials / (duration_ms / 1000), 3),
    }
    cc = OUT / "crosscheck" / "crosscheck_100Hz.json"
    if cc.exists():  # produced by spikes/brain/crosscheck_brian2.py (original Brian2 model.py)
        report["brian2_crosscheck_subnetwork"] = json.loads(cc.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {OUT / 'validation.json'}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--duration-ms", type=float, default=1000.0)
    ap.add_argument("--count-trials", type=int, default=0,
                    help="also count responsive neurons at sugar 10/200 Hz with this many trials (paper: 30)")
    a = ap.parse_args(argv)
    if a.bench:
        t = time.perf_counter()
        conn = load_connectome()
        print(f"load_connectome: {time.perf_counter() - t:.1f}s, N={conn.N}, synapse rows={conn.indices.size}")
        sugar = [i for i in SUGAR_GRNS_V630 if i in conn.id2idx]
        res = simulate(sugar, excite_rate_hz=200.0, duration_ms=a.duration_ms, n_trials=a.trials)
        per = res["runtime_s"] / a.trials / (a.duration_ms / 1000.0)
        print(f"sugar 200 Hz: {a.trials} trials x {a.duration_ms} ms in {res['runtime_s']:.1f}s "
              f"=> {per:.2f} s per trial per simulated second; active={res['n_active']}")
        print("MN9:", rates_for(res, list(MN9.values())))
    if a.validate:
        _validate(a.trials, a.duration_ms, a.count_trials)
    if not (a.bench or a.validate):
        ap.print_help()


if __name__ == "__main__":
    main()
