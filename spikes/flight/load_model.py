"""Spike: load the original flybody fruitfly.xml (shipped with flygym 2.1) with the lazily downloaded meshes."""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco as mj
import numpy as np

from flylab import flight

t0 = time.perf_counter()
m = flight.build_model()
print("compile", round(time.perf_counter() - t0, 2), "s", "nbody", m.nbody, "nq", m.nq, "nu", m.nu, "dt", m.opt.timestep)
d = mj.MjData(m)
tid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "thorax")
print("total mass g", m.body_subtreemass[tid], "density", m.opt.density, "visc", m.opt.viscosity)
t0 = time.perf_counter()
for _ in range(2000):
    mj.mj_step(m, d)
print("2000 steps", round(time.perf_counter() - t0, 2), "s; thorax z", d.xpos[tid])
