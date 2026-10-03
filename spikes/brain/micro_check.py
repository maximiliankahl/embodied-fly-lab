"""Deterministic micro-test: flylab.brain._run_batch vs Brian2 with the reference equations.

Neuron 0 receives input kicks (v += w_syn*f_poi) at fixed steps; it projects with k synapses onto
neuron 1, which projects with k2 synapses onto neuron 2. Spike counts must match exactly.
    uv run --with brian2 python spikes/brain/micro_check.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flylab import brain  # noqa: E402

P = brain.DEFAULT_PARAMS
STEPS = 3000
rng0 = np.random.default_rng(5)
KICKS = np.sort(rng0.choice(STEPS - 50, 150, replace=False))  # input event steps for neuron 0


class FakeRng:
    def __init__(self):
        self.step = 0

    def random(self, n):
        r = np.ones(n)
        if self.step in set(KICKS.tolist()):
            r[:] = 0.0
        self.step += 1
        return r


def ours(k, k2):
    ids = np.array([1, 2, 3])
    indptr = np.array([0, 1, 2, 2])
    conn = brain.Connectome(ids=ids, indptr=indptr, indices=np.array([1, 2], np.int32),
                            syn=np.array([k, k2], np.float32), id2idx={1: 0, 2: 1, 3: 2})
    c = brain._run_batch(conn, np.array([0]), np.array([1.0]), np.zeros(3, bool), STEPS, 1, FakeRng(), P)
    return c[0].tolist()


def b2(k, k2):
    from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor, Network, mV, ms,
                        prefs, defaultclock)
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    ns = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms, "tau": 5 * ms}
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = NeuronGroup(3, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                      refractory="rfc", namespace=ns)
    neu.v = -52 * mV
    neu.rfc = 2.2 * ms
    neu.rfc[0] = 0 * ms
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms)
    syn.connect(i=[0, 1], j=[1, 2])
    syn.w = [k * 0.275 * mV, k2 * 0.275 * mV]
    # kicks are applied in the 'synapses' slot like PoissonInput; times are step indices
    gen = SpikeGeneratorGroup(1, np.zeros(len(KICKS), int), KICKS * 0.1 * ms)
    kick = Synapses(gen, neu, on_pre="v_post += 68.75 * mV")
    kick.connect(i=0, j=0)
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, gen, kick, mon)
    net.run(STEPS * 0.1 * ms)
    return [int(x) for x in mon.count[:]]


if __name__ == "__main__":
    for k, k2 in [(5, 30), (10, 30), (20, 20), (30, 60), (60, 15), (100, 100)]:
        a, b = ours(k, k2), b2(k, k2)
        print(f"k={k:3d} k2={k2:3d}  ours={a}  brian2={b}  {'MATCH' if a == b else 'DIFF'}", flush=True)
