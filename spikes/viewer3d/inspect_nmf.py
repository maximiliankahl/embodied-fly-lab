"""Inspect NeuroMechFly MuJoCo model: bodies, geoms, meshes (for flylab.export3d)."""
import time
import numpy as np
import mujoco as mj
from flylab import body

t = time.time()
fly, cam, sim, steps, dof_order, ctrl, thorax_idx = body._build(None, 5)
m = sim.mj_model
print("build", round(time.time() - t, 1), "s; nbody", m.nbody, "ngeom", m.ngeom, "nmesh", m.nmesh,
      "nvert", m.nmeshvert, "nface", m.nmeshface, "timestep", m.opt.timestep)
names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
print(names[:12], "...", names[-5:])
gt = {}
for g in range(m.ngeom):
    gt.setdefault(int(m.geom_type[g]), 0)
    gt[int(m.geom_type[g])] += 1
print("geom types", gt, "(7=mesh, 0=plane, 6=box, 5=cyl, 3=capsule, 2=sphere)")
for g in range(min(m.ngeom, 8)):
    print(g, mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g), "body", names[m.geom_bodyid[g]], "type", m.geom_type[g],
          "size", m.geom_size[g], "mesh", m.geom_dataid[g], "group", m.geom_group[g], "rgba", m.geom_rgba[g], "matid", m.geom_matid[g])
mv = [int(m.mesh_vertnum[i]) for i in range(m.nmesh)]
print("mesh vert counts", sorted(mv)[-10:], "sum", sum(mv))
print("thorax pos", sim.mj_data.xpos[names.index([n for n in names if n.endswith("c_thorax")][0])])
print("bbox xpos", sim.mj_data.xpos.min(0), sim.mj_data.xpos.max(0))
print(hasattr(sim, "mj_model"), type(sim).__name__)
