from brian2 import *
import numpy as np
prefs.codegen.target = "numpy"
ns = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms, "tau": 5 * ms}
for flag_g in ["(unless refractory)", ""]:
    eqs = f"""
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt {flag_g}
    rfc : second
    """
    neu = NeuronGroup(2, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns)
    neu.v = -52 * mV; neu.rfc = 2.2 * ms
    # neuron 1 kicked to spike at step 10; neuron 0 kicked at step 0 -> syn input to 1 at step ~1+18
    gen = SpikeGeneratorGroup(2, [0, 1], [0 * ms, 10 * 0.1 * ms])
    kick = Synapses(gen, neu, on_pre="v_post += 68.75 * mV"); kick.connect(j="i")
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms); syn.connect(i=0, j=1); syn.w = 10 * mV
    sm = StateMonitor(neu, ["v", "g", "not_refractory"], record=[1], when="end")
    spk = SpikeMonitor(neu)
    net = Network(neu, gen, kick, syn, sm, spk); net.run(4 * ms)
    print("flag_g=", repr(flag_g), "spikes", list(zip(spk.i[:], np.round(spk.t[:] / (0.1 * ms)).astype(int))))
    for s in range(8, 40):
        print(s, round(float(sm.v[0][s] / mV), 2), round(float(sm.g[0][s] / mV), 2), bool(sm.not_refractory[0][s]))
