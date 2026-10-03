"""Trace comparison (debug): v/g of neuron 1 for one strong input, ours vs Brian2."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spikes" / "brain"))
from flylab import brain
import micro_check as mc
from brian2 import *
prefs.codegen.target = "numpy"
k = 60
mc.KICKS = mc.KICKS[mc.KICKS < 1500]
mc.STEPS = 1500
ids = np.array([1, 2, 3])
conn = brain.Connectome(ids=ids, indptr=np.array([0, 1, 2, 2]), indices=np.array([1, 2], np.int32),
                        syn=np.array([k, 0], np.float32), id2idx={1: 0, 2: 1, 3: 2})
tr = []
brain._run_batch(conn, np.array([0]), np.array([1.0]), np.zeros(3, bool), mc.STEPS, 1, mc.FakeRng(), brain.DEFAULT_PARAMS, trace=tr)
u = np.array([t[0][1] for t in tr]); g = np.array([t[1][1] for t in tr])
ns = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms, "tau": 5 * ms}
eqs = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc : second
"""
neu = NeuronGroup(3, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns)
neu.v = -52 * mV; neu.rfc = 2.2 * ms; neu.rfc[0] = 0 * ms
syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms); syn.connect(i=[0, 1], j=[1, 2]); syn.w = [k * 0.275 * mV, 0 * mV]
gen = SpikeGeneratorGroup(1, np.zeros(len(mc.KICKS), int), mc.KICKS * 0.1 * ms)
kick = Synapses(gen, neu, on_pre="v_post += 68.75 * mV"); kick.connect(i=0, j=0)
sm = StateMonitor(neu, ["v", "g"], record=[0, 1], when="end")
spk = SpikeMonitor(neu)
net = Network(neu, syn, gen, kick, sm, spk); net.run(mc.STEPS * 0.1 * ms)
bv = sm.v[1] / mV + 52; bg = sm.g[1] / mV
print("brian2 spikes n0", [int(round(t / (0.1 * ms))) for t in spk.t[spk.i == 0]], "n1", [int(round(t / (0.1 * ms))) for t in spk.t[spk.i == 1]])
ours_sp = [i for i in range(1, len(u)) if u[i] == 0 and u[i - 1] > 6.9]
print("first diffs:")
d = np.flatnonzero((np.abs(bv - u) > 1e-3) | (np.abs(bg - g) > 1e-3))
for i in d[:12]:
    print(i, "ours u,g", round(float(u[i]), 4), round(float(g[i]), 4), " b2 u,g", round(float(bv[i]), 4), round(float(bg[i]), 4))
print("step  ours(u,g)  b2(u,g)")
for i in range(84, 135):
    print(i, round(float(u[i]), 3), round(float(g[i]), 3), "|", round(float(bv[i]), 3), round(float(bg[i]), 3))
