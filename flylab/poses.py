"""Record per-frame 3D body poses from a MuJoCo simulation for the Three.js replay viewer.

Shared by flylab.body (walking) and flylab.flight (flight). Format (JSON-serialisable):

    {"format": "flylab-poses-v1", "units": "mm" | model units, "fps": 60,
     "bodies": [name, ...],                       # MuJoCo body names (index i)
     "frames": [{"t": s, "p": [[x, y, z], ...], "q": [[w, x, y, z], ...]}, ...]}

Positions/quaternions are world-frame body poses (``data.xpos`` / ``data.xquat``),
rounded to keep files small. Geometry (meshes/primitives per body) is exported separately
by flylab.export3d so that one geometry file serves many runs of the same model.
"""

from __future__ import annotations

import numpy as np


class PoseRecorder:
    """Call ``maybe_record(t, mj_model, mj_data)`` every physics step; keeps frames at ``fps``."""

    def __init__(self, mj_model, fps: float = 60.0, body_filter=None, decimals: int = 4, units: str = "model"):
        import mujoco as mj

        self.fps = float(fps)
        self.decimals = decimals
        self.units = units
        names = [mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(mj_model.nbody)]
        idx = [i for i, n in enumerate(names) if i > 0 and (body_filter is None or body_filter(n))]
        self.body_ids = np.asarray(idx, dtype=int)
        self.bodies = [names[i] for i in idx]
        self.frames: list[dict] = []
        self._next_t = 0.0

    def maybe_record(self, t: float, mj_model, mj_data, force: bool = False) -> None:
        if not force and t + 1e-12 < self._next_t:
            return
        p = np.round(mj_data.xpos[self.body_ids], self.decimals).tolist()
        q = np.round(mj_data.xquat[self.body_ids], self.decimals).tolist()
        self.frames.append({"t": round(float(t), 5), "p": p, "q": q})
        self._next_t = t + 1.0 / self.fps

    def to_dict(self) -> dict:
        return {"format": "flylab-poses-v1", "units": self.units, "fps": self.fps,
                "bodies": self.bodies, "frames": self.frames}
