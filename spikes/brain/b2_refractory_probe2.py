from brian2 import *
import numpy as np
prefs.codegen.target = "numpy"
ns = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms, "tau": 5 * ms}
eqs = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc : second
"""
for k0 in [11, 12, 13, 14]:
    neu = NeuronGroup(2, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns)
    neu.v = -52 * mV; neu.rfc = 2.2 * ms
    gen = SpikeGeneratorGroup(2, [0, 1], [k0 * 0.1 * ms, 10 * 0.1 * ms])
    kick = Synapses(gen, neu, on_pre="v_post += 68.75 * mV"); kick.connect(j="i")
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms); syn.connect(i=0, j=1); syn.w = 10 * mV
    sm = StateMonitor(neu, ["g"], record=[1], when="end")
    spk = SpikeMonitor(neu)
    net = Network(neu, gen, kick, syn, sm, spk); net.run(4 * ms)
    sp = list(zip(spk.i[:].tolist(), np.round(spk.t[:] / (0.1 * ms)).astype(int).tolist()))
    print("n0 kick", k0, "spikes", sp, "deliver step", sp[0][1] + 18 if sp[0][0] == 0 else None,
          "g[28..36]", [round(float(x / mV), 2) for x in sm.g[0][28:37]])
