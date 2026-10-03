import os, sys, time
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (HybridTurningController, HybridControllerObservation,
    LocomotionAction, PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly)

t0 = time.time()
fly = make_locomotion_fly(name="nmf", add_adhesion=True, colorize=True)
cam = fly.add_tracking_camera(name="topcam", pos_offset=(0, 0, 20), rotation=Rotation3D("xyaxes", (1,0,0,0,1,0)), fovy=40.0)
world = FlatGroundWorld()
world.add_fly(fly, [0,0,0.8], Rotation3D("quat",[1,0,0,0]), bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=False)
sim = Simulation(world)
print("build", time.time()-t0, "dt", sim.timestep)
render = len(sys.argv) > 2
if render:
    sim.set_renderer([cam], camera_res=(360,480), playback_speed=0.25, output_fps=25)
pp = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
ctrl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=pp, output_dof_order=dof_order)
sim.reset(); ctrl.reset(seed=0)
apply_locomotion_action(sim, fly.name, LocomotionAction(joint_angles=pp.default_pose_by_dof_order(dof_order), adhesion_onoff=np.ones(6,bool)))
sim.warmup()
sig = np.array([float(x) for x in sys.argv[1].split(",")])
thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
tid = sim._internal_bodyids_by_fly[fly.name][thorax_idx]
T = 0.5; n = int(T/sim.timestep)
t1 = time.time()
p0 = sim.get_body_positions(fly.name)[thorax_idx].copy()
h0 = sim.mj_data.xmat[tid].reshape(3,3)[:,0].copy()
for i in range(n):
    obs = HybridControllerObservation.from_sim(sim, fly.name)
    a = ctrl.step(sig, obs)
    apply_locomotion_action(sim, fly.name, a)
    sim.step()
    if render: sim.render_as_needed()
el = time.time()-t1
p1 = sim.get_body_positions(fly.name)[thorax_idx]
h1 = sim.mj_data.xmat[tid].reshape(3,3)[:,0]
print("sig", sig, "disp", np.round(p1-p0,3), "heading0", np.degrees(np.arctan2(h0[1],h0[0])), "heading1", np.degrees(np.arctan2(h1[1],h1[0])))
print("wall per sim s", el/T)
if render:
    sim.renderer.save_video("spikes/body/out/spike.mp4"); print("video ok")
