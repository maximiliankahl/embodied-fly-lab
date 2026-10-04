"""Movement verification: did the body REALLY perform the expected movement?

Used by the ``movement_verifier`` agent (agents/fly_lab.yaml) after every embodied run.
Two independent methods, combined conservatively:

1. **Kinematic check** - recomputed from the RAW trajectory, independent of
   ``flylab.body.classify`` / ``flylab.flight.classify_flight`` (different formulas,
   own thresholds, documented below):
   * walking  (rows ``[t, x_mm, y_mm, heading_deg]``, world frame, x/y relative to start):
     displacement in the fly's INITIAL body frame, time-averaged velocity in the fly's
     CURRENT body frame (sign of locomotion independent of the start heading), heading
     change and yaw rate (unwrapped), net / path speed, upright (``upright_min_cos`` /
     ``fell_over`` if the body reported them).
   * flight   (rows ``[t, x, y, z, roll, pitch, yaw]``): height gain over the start
     height, airborne time, max height, final height, displacement in the initial frame,
     heading change (+ = left / counter-clockwise), attitude (|roll| at the end).
2. **Vision check (blind)** - a keyframe contact sheet (default 16 frames with frame
   number and time stamp) from the rendered mp4 is sent to Claude vision. The model
   gets NEITHER the expected behaviour NOR the kinematic numbers; it only gets the
   camera conventions and the allowed label set and returns a structured JSON verdict
   (observed_behavior, confidence, observations). ``matches_expected`` is computed here.
   Results are cached by video SHA-256 (+ model, mode, frame count, prompt version).

``final_verdict``: ``correct`` only if both methods say the movement matches,
``incorrect`` only if both say it does not, otherwise ``uncertain`` with a reason
(disagreement, weak effect, fell over, unclear video ...). With ``use_vision=False``
(or no video) the final verdict is the kinematic one and ``methods`` says so.

Thresholds are module constants. They are hand-set for 1-s NeuroMechFly runs (reference
values in flylab/body.py: forward ~15 mm/s, backward ~10 mm/s, turns ~150 deg/s) and are
deliberately different formulas than the body classifier; flight thresholds are
PROVISIONAL (flylab/flight.py was written concurrently and is not calibrated yet).

CLI:  uv run python -m flylab.verify --selftest            (kinematics only, no API)
      uv run python -m flylab.verify --video <mp4> --expected backward [--mode walk]
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "verify_vision"  # gitignored
PROMPT_VERSION = "v2"
DEFAULT_VISION_MODEL = "claude-sonnet-5-5"

WALK_LABELS = ("forward", "backward", "turn_left", "turn_right", "stop")
FLIGHT_LABELS = ("no_takeoff", "takeoff_fall", "hover", "climb", "forward_flight",
                 "flight_turn_left", "flight_turn_right")
# Behaviours that are read out at BRAIN level only (no body movement is expected or checkable).
BRAIN_ONLY = ("groom", "feed")

# --- walking thresholds (own, documented; NOT the body classifier's) -------------------
WALK_MIN_SPEED_MM_S = 2.0       # |mean body-frame velocity| and |initial-frame displacement|/s for forward/backward
WALK_WEAK_FRACTION = 0.5        # effects in [0.5, 1) x threshold with the right sign -> "weak" -> uncertain
WALK_STRAIGHT_MAX_DEG = 90.0    # forward/backward: |heading change| must stay below this
TURN_MIN_DEG = 30.0             # turn: |heading change| >= 30 deg ...
TURN_MIN_RATE_DEG_S = 30.0      # ... and |mean yaw rate| >= 30 deg/s
STOP_MAX_NET_SPEED_MM_S = 1.5   # stop: net displacement speed below this ...
STOP_MAX_HEADING_DEG = 20.0     # ... and |heading change| below this
UPRIGHT_MIN_COS = 0.5           # thorax z . world z below this = fell over (same physical meaning as body.py)

# --- flight thresholds (PROVISIONAL) --------------------------------------------------
AIRBORNE_GAIN_MM = 1.0          # height above the start height that counts as airborne
TAKEOFF_MIN_AIRBORNE_S = 0.05   # airborne for at least this long = a takeoff happened
LANDED_GAIN_MM = 0.5            # final height gain below this after a takeoff = fell / landed
CLIMB_MIN_GAIN_MM = 3.0         # final height gain for "climb"
FLIGHT_MIN_FWD_MM_S = 5.0       # forward flight: initial-frame forward displacement / airborne time
FLIGHT_TURN_MIN_DEG = 30.0      # flight turns
VISION_MIN_CONFIDENCE = 0.65   # vision answers below this confidence count as "unclear"
HOVER_MAX_DRIFT_MM_S = 5.0      # hover: horizontal drift / airborne time below this
FLIGHT_MAX_ROLL_DEG = 90.0      # |roll| at the end above this = flipped over


# =========================================================================== helpers


def _f(x: Any, nd: int = 3) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _unwrap_deg(values: list[float]) -> list[float]:
    out: list[float] = []
    prev = None
    acc = 0.0
    for v in values:
        if prev is None:
            acc = v
        else:
            d = (v - prev + 180.0) % 360.0 - 180.0
            acc += d
        out.append(acc)
        prev = v
    return out


def _extract(result: dict) -> tuple[dict, str | None]:
    """Accept a body/flight result, an embodied artifact ({"body": {...}} / {"flight": {...}})
    or a tool result. Returns (the dict that holds "trajectory", detected mode)."""
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")
    for key, mode in (("flight", "flight"), ("body", "walk")):
        sub = result.get(key)
        if isinstance(sub, dict) and sub.get("trajectory"):
            return sub, mode
    if result.get("trajectory"):
        tr = result["trajectory"]
        mode = "flight" if tr and len(tr[0]) >= 7 else "walk"
        return result, mode
    raise ValueError("no trajectory in result (pass the body/flight result or the embodied artifact JSON; "
                     "tool summaries only carry trajectory_end)")


def _check(name: str, value: Any, threshold: str, ok: bool | None, note: str = "") -> dict:
    d = {"name": name, "value": value, "threshold": threshold, "pass": ok}
    if note:
        d["note"] = note
    return d


def normalize_expected(expected: str, mode: str) -> str:
    e = str(expected or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"walk_forward": "forward", "forward_walking": "forward", "backward_walking": "backward",
               "retreat": "backward", "moonwalk": "backward", "left": "turn_left", "right": "turn_right",
               "takeoff": "escape", "jump": "escape", "escape_takeoff": "escape", "flight": "airborne",
               "fly": "airborne", "airborne": "airborne", "none": "stop", "no_movement": "stop"}
    e = aliases.get(e, e)
    if mode == "flight" and e in ("turn_left", "turn_right"):
        e = "flight_" + e
    if mode == "flight" and e == "forward":
        e = "forward_flight"
    if mode == "flight" and e == "stop":
        e = "no_takeoff"
    return e


# =========================================================================== kinematics


def recompute_walk(body: dict) -> dict:
    tr = [[float(v) for v in row[:4]] for row in body["trajectory"]]
    t = [r[0] for r in tr]
    dur = max(t[-1] - t[0], 1e-6)
    head = _unwrap_deg([r[3] for r in tr])
    h0 = math.radians(head[0])
    dx, dy = tr[-1][1] - tr[0][1], tr[-1][2] - tr[0][2]
    fwd0 = dx * math.cos(h0) + dy * math.sin(h0)
    lat0 = -dx * math.sin(h0) + dy * math.cos(h0)
    # time-averaged velocity in the CURRENT body frame (50 ms steps, smooths stride wobble)
    step = max(1, int(round(0.05 / max(t[1] - t[0], 1e-6)))) if len(t) > 1 else 1
    idx = list(range(0, len(tr), step))
    if idx[-1] != len(tr) - 1:
        idx.append(len(tr) - 1)
    v_body_fwd, v_body_lat, path = 0.0, 0.0, 0.0
    for a, b in zip(idx[:-1], idx[1:]):
        ddx, ddy = tr[b][1] - tr[a][1], tr[b][2] - tr[a][2]
        hm = math.radians(0.5 * (head[a] + head[b]))
        v_body_fwd += ddx * math.cos(hm) + ddy * math.sin(hm)
        v_body_lat += -ddx * math.sin(hm) + ddy * math.cos(hm)
        path += math.hypot(ddx, ddy)
    dh = head[-1] - head[0]
    up = body.get("upright_min_cos")
    return {
        "duration_s": _f(dur), "n_samples": len(tr),
        "forward_disp_initial_frame_mm": _f(fwd0), "lateral_disp_initial_frame_mm": _f(lat0),
        "body_frame_forward_speed_mm_s": _f(v_body_fwd / dur), "body_frame_lateral_speed_mm_s": _f(v_body_lat / dur),
        "net_speed_mm_s": _f(math.hypot(dx, dy) / dur), "path_speed_mm_s": _f(path / dur),
        "heading_change_deg": _f(dh, 2), "yaw_rate_deg_s": _f(dh / dur, 2),
        "upright_min_cos": _f(up), "fell_over": (bool(up < UPRIGHT_MIN_COS) if _f(up) is not None
                                                   else bool(body.get("fell_over")) if "fell_over" in body else None),
    }


def recompute_flight(fl: dict) -> dict:
    tr = [[float(v) for v in row[:7]] for row in fl["trajectory"]]
    t = [r[0] for r in tr]
    dur = max(t[-1] - t[0], 1e-6)
    z0 = tr[0][3]
    gain = [r[3] - z0 for r in tr]
    air = 0.0
    first_air = None
    for i in range(1, len(tr)):
        if gain[i] > AIRBORNE_GAIN_MM:
            air += t[i] - t[i - 1]
            if first_air is None:
                first_air = t[i]
    yaw = _unwrap_deg([r[6] for r in tr])
    y0 = math.radians(yaw[0])
    dx, dy, dz = tr[-1][1] - tr[0][1], tr[-1][2] - tr[0][2], tr[-1][3] - tr[0][3]
    fwd0 = dx * math.cos(y0) + dy * math.sin(y0)
    lat0 = -dx * math.sin(y0) + dy * math.cos(y0)
    return {
        "duration_s": _f(dur), "n_samples": len(tr),
        "start_height_mm": _f(z0), "max_height_gain_mm": _f(max(gain)), "final_height_gain_mm": _f(gain[-1]),
        "airborne_time_s": _f(air), "first_airborne_t_s": _f(first_air),
        "forward_disp_initial_frame_mm": _f(fwd0), "lateral_disp_initial_frame_mm": _f(lat0), "dz_mm": _f(dz),
        "horizontal_disp_mm": _f(math.hypot(dx, dy)), "heading_change_deg": _f(yaw[-1] - yaw[0], 2),
        "final_abs_roll_deg": _f(abs(tr[-1][4]), 1), "max_abs_roll_deg": _f(max(abs(r[4]) for r in tr), 1),
    }


def _graded(value: float, thr: float, sign: int) -> str:
    """'pass' if sign*value >= thr, 'weak' if in [WEAK*thr, thr), else 'fail'."""
    v = sign * value
    if v >= thr:
        return "pass"
    if v >= WALK_WEAK_FRACTION * thr:
        return "weak"
    return "fail"


def kinematic_walk(rc: dict, expected: str) -> dict:
    checks: list[dict] = []
    reasons: list[str] = []
    dur = rc["duration_s"] or 1.0
    vb = rc["body_frame_forward_speed_mm_s"] or 0.0
    v0 = (rc["forward_disp_initial_frame_mm"] or 0.0) / dur
    dh = rc["heading_change_deg"] or 0.0
    yr = rc["yaw_rate_deg_s"] or 0.0
    fell = rc.get("fell_over")
    checks.append(_check("upright", rc.get("upright_min_cos"), f"upright_min_cos >= {UPRIGHT_MIN_COS}",
                         None if fell is None else not fell))
    grades: list[str] = []
    if expected in ("forward", "backward"):
        sign = 1 if expected == "forward" else -1
        g1 = _graded(vb, WALK_MIN_SPEED_MM_S, sign)
        g2 = _graded(v0, WALK_MIN_SPEED_MM_S, sign)
        grades += [g1, g2]
        rel = ">=" if sign > 0 else "<="
        checks.append(_check("body_frame_forward_speed_mm_s", rc["body_frame_forward_speed_mm_s"],
                             f"{rel} {sign * WALK_MIN_SPEED_MM_S}", g1 == "pass", g1))
        checks.append(_check("initial_frame_forward_speed_mm_s", _f(v0), f"{rel} {sign * WALK_MIN_SPEED_MM_S}",
                             g2 == "pass", g2))
        ok = abs(dh) < WALK_STRAIGHT_MAX_DEG
        grades.append("pass" if ok else "fail")
        checks.append(_check("abs_heading_change_deg", _f(abs(dh), 2), f"< {WALK_STRAIGHT_MAX_DEG}", ok))
    elif expected in ("turn_left", "turn_right"):
        sign = 1 if expected == "turn_left" else -1  # heading change > 0 = counter-clockwise = LEFT
        g1 = _graded(dh, TURN_MIN_DEG, sign)
        g2 = _graded(yr, TURN_MIN_RATE_DEG_S, sign)
        grades += [g1, g2]
        rel = ">=" if sign > 0 else "<="
        checks.append(_check("heading_change_deg", rc["heading_change_deg"], f"{rel} {sign * TURN_MIN_DEG} (+ = left)",
                             g1 == "pass", g1))
        checks.append(_check("yaw_rate_deg_s", rc["yaw_rate_deg_s"], f"{rel} {sign * TURN_MIN_RATE_DEG_S}",
                             g2 == "pass", g2))
    elif expected == "stop":
        ok1 = (rc["net_speed_mm_s"] or 0.0) < STOP_MAX_NET_SPEED_MM_S
        ok2 = abs(dh) < STOP_MAX_HEADING_DEG
        grades += ["pass" if ok1 else "fail", "pass" if ok2 else "fail"]
        checks.append(_check("net_speed_mm_s", rc["net_speed_mm_s"], f"< {STOP_MAX_NET_SPEED_MM_S}", ok1))
        checks.append(_check("abs_heading_change_deg", _f(abs(dh), 2), f"< {STOP_MAX_HEADING_DEG}", ok2))
    elif expected in BRAIN_ONLY:
        return {"verdict": "uncertain", "checks": checks, "recomputed": rc,
                "reason": f"'{expected}' is a brain-level readout in this lab; the walking body has no {expected} "
                          "motor program, so the body movement cannot verify it."}
    elif expected in ("escape", "airborne"):
        return {"verdict": "incorrect" if not fell else "uncertain", "checks": checks, "recomputed": rc,
                "reason": "expected a takeoff/flight but this is a WALKING body run (no flight model in this run)."}
    else:
        return {"verdict": "uncertain", "checks": checks, "recomputed": rc,
                "reason": f"unknown expected behaviour {expected!r} for mode walk (allowed: {', '.join(WALK_LABELS)})"}
    if fell:
        verdict = "uncertain"
        reasons.append("fly fell over (upright_min_cos < 0.5): movement label unreliable")
    elif all(g == "pass" for g in grades):
        verdict = "correct"
    elif "fail" in grades:
        verdict = "incorrect"
        reasons.append("at least one core check clearly failed")
    else:
        verdict = "uncertain"
        reasons.append(f"weak effect: right direction but below threshold (>= {WALK_WEAK_FRACTION:.0%} of it)")
    return {"verdict": verdict, "checks": checks, "recomputed": rc, "reason": "; ".join(reasons) or "all checks passed"}


def kinematic_flight(rc: dict, expected: str) -> dict:
    checks: list[dict] = []
    air = rc["airborne_time_s"] or 0.0
    took_off = air >= TAKEOFF_MIN_AIRBORNE_S and (rc["max_height_gain_mm"] or 0.0) > AIRBORNE_GAIN_MM
    flipped = (rc["final_abs_roll_deg"] or 0.0) > FLIGHT_MAX_ROLL_DEG
    checks.append(_check("airborne_time_s", rc["airborne_time_s"],
                         f">= {TAKEOFF_MIN_AIRBORNE_S} s with height gain > {AIRBORNE_GAIN_MM} mm", took_off))
    checks.append(_check("final_abs_roll_deg", rc["final_abs_roll_deg"], f"<= {FLIGHT_MAX_ROLL_DEG}", not flipped))
    fin = rc["final_height_gain_mm"] or 0.0
    fwd_rate = (rc["forward_disp_initial_frame_mm"] or 0.0) / max(air, 1e-6)
    drift_rate = (rc["horizontal_disp_mm"] or 0.0) / max(air, 1e-6)
    dh = rc["heading_change_deg"] or 0.0
    need: list[bool] = []
    if expected == "no_takeoff":
        need = [not took_off]
    elif expected in ("escape", "airborne"):
        need = [took_off]
    elif expected == "takeoff_fall":
        landed = fin < LANDED_GAIN_MM
        checks.append(_check("final_height_gain_mm", rc["final_height_gain_mm"], f"< {LANDED_GAIN_MM} (back down)", landed))
        need = [took_off, landed]
    elif expected == "climb":
        ok = fin >= CLIMB_MIN_GAIN_MM
        checks.append(_check("final_height_gain_mm", rc["final_height_gain_mm"], f">= {CLIMB_MIN_GAIN_MM}", ok))
        need = [took_off, ok]
    elif expected == "hover":
        ok1 = fin > AIRBORNE_GAIN_MM
        ok2 = drift_rate < HOVER_MAX_DRIFT_MM_S
        checks.append(_check("final_height_gain_mm", rc["final_height_gain_mm"], f"> {AIRBORNE_GAIN_MM} (still up)", ok1))
        checks.append(_check("horizontal_drift_mm_per_airborne_s", _f(drift_rate), f"< {HOVER_MAX_DRIFT_MM_S}", ok2))
        need = [took_off, ok1, ok2]
    elif expected == "forward_flight":
        ok = fwd_rate >= FLIGHT_MIN_FWD_MM_S
        checks.append(_check("forward_disp_per_airborne_s", _f(fwd_rate), f">= {FLIGHT_MIN_FWD_MM_S}", ok))
        need = [took_off, ok]
    elif expected in ("flight_turn_left", "flight_turn_right"):
        sign = 1 if expected.endswith("left") else -1
        ok = sign * dh >= FLIGHT_TURN_MIN_DEG
        checks.append(_check("heading_change_deg", rc["heading_change_deg"],
                             f"{'>=' if sign > 0 else '<='} {sign * FLIGHT_TURN_MIN_DEG} (+ = left)", ok))
        need = [took_off, ok]
    elif expected in BRAIN_ONLY:
        return {"verdict": "uncertain", "checks": checks, "recomputed": rc,
                "reason": f"'{expected}' is a brain-level readout; not checkable from flight kinematics."}
    else:
        return {"verdict": "uncertain", "checks": checks, "recomputed": rc,
                "reason": f"unknown expected behaviour {expected!r} for mode flight (allowed: {', '.join(FLIGHT_LABELS)}, escape)"}
    if flipped and expected not in ("no_takeoff", "takeoff_fall"):
        return {"verdict": "uncertain", "checks": checks, "recomputed": rc,
                "reason": "fly flipped over (|roll| > 90 deg at the end); flight label unreliable"}
    verdict = "correct" if all(need) else "incorrect"
    return {"verdict": verdict, "checks": checks, "recomputed": rc,
            "reason": "all checks passed" if verdict == "correct" else "at least one core check failed",
            "thresholds_note": "flight thresholds are PROVISIONAL (not calibrated against flylab.flight yet)"}


def kinematic_verdict(result: dict, expected_behavior: str, mode: str | None = None) -> dict:
    holder, detected = _extract(result)
    mode = mode or detected or "walk"
    if mode != detected and detected:
        mode = detected  # trust the trajectory shape over the argument
    exp = normalize_expected(expected_behavior, mode)
    if mode == "flight":
        out = kinematic_flight(recompute_flight(holder), exp)
    else:
        out = kinematic_walk(recompute_walk(holder), exp)
    out["mode"] = mode
    out["expected_normalized"] = exp
    return out


# =========================================================================== vision


def _read_frames(video_path: str, n: int) -> tuple[list, list[float], float]:
    """n evenly spaced RGB frames (numpy) + their video times (s) + video duration."""
    import imageio.v2 as imageio

    r = imageio.get_reader(video_path)
    try:
        fps = float(r.get_meta_data().get("fps") or 25.0)
        frames_all = [f for f in r]
    finally:
        r.close()
    total = len(frames_all)
    if total == 0:
        raise ValueError(f"no frames in {video_path}")
    n = max(2, min(n, total))
    idx = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    return [frames_all[i] for i in idx], [i / fps for i in idx], total / fps


def contact_sheet(video_path: str, out_path: str | None = None, n_frames: int = 16, cols: int = 4,
                  sim_duration_s: float | None = None, tile_w: int = 320) -> dict:
    """Tile n evenly spaced keyframes (time order, left->right, top->bottom) into one JPEG with
    a label per tile: frame number and simulated time (if ``sim_duration_s`` is known, the video
    time is rescaled to simulation time; videos are rendered in slow motion)."""
    from PIL import Image, ImageDraw

    frames, times, vdur = _read_frames(video_path, n_frames)
    rows = math.ceil(len(frames) / cols)
    h0, w0 = frames[0].shape[:2]
    tile_h = int(round(h0 * tile_w / w0))
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 18)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    labels = []
    for k, (fr, tv) in enumerate(zip(frames, times)):
        img = Image.fromarray(fr).resize((tile_w, tile_h))
        x, y = (k % cols) * tile_w, (k // cols) * (tile_h + 18)
        sheet.paste(img, (x, y + 18))
        ts = tv * sim_duration_s / vdur if sim_duration_s else tv
        lab = f"#{k + 1}  t={ts:.2f} s" + ("" if sim_duration_s else " (video)")
        labels.append(lab)
        draw.rectangle([x, y, x + tile_w - 1, y + 17], fill=(30, 30, 30))
        draw.text((x + 4, y + 3), lab, fill=(255, 255, 255))
    if out_path is None:
        out_path = str(Path(video_path).with_name(Path(video_path).stem + "_contact.jpg"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=80)
    return {"path": out_path, "frames_used": len(frames), "labels": labels, "video_duration_s": round(vdur, 3)}


def _env_value(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    env = ROOT / ".env"
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _client():
    import anthropic

    key = _env_value("ANTHROPIC_API_KEY")
    ws = _env_value("ANTHROPIC_WORKSPACE_ID")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (env or 02_App/.env)")
    headers = {"anthropic-workspace-id": ws} if ws else {}
    # Talk to the API directly (not via an ANTHROPIC_BASE_URL proxy that may not be running).
    return anthropic.Anthropic(api_key=key, default_headers=headers, base_url="https://api.anthropic.com",
                               timeout=120.0, max_retries=2)


_CAMERA_NOTES = {
    "walk": ("NeuroMechFly walking model in MuJoCo, seen from ABOVE by a camera that FOLLOWS the fly, so the fly "
             "stays near the image centre and its displacement shows up as the checkerboard floor shifting the "
             "opposite way between tiles. Neighbouring tiles are close in time, so the floor shifts by LESS than half a checker square between them: follow one checker edge from tile to tile to get the direction of travel (a periodic floor can otherwise look like it moves the wrong way). The head is the end with the red eyes; the wings/abdomen are the rear end. "
             "Moving toward the head = forward, toward the abdomen = backward; a change of the body-axis "
             "orientation between tiles = turning."),
    "flight": ("FlyBody fly model in MuJoCo (wings can beat), seen from a chase/side camera. Judge whether the fly "
               "leaves the ground (gap between legs and floor, rising body), stays up, climbs, moves forward or turns."),
}
_LABEL_HELP = {
    "walk": {"forward": "walks toward its head", "backward": "walks toward its abdomen (retreat)",
             "turn_left": "rotates counter-clockwise seen from above (its head swings to its own left)",
             "turn_right": "rotates clockwise seen from above (its head swings to its own right)",
             "stop": "no clear locomotion (standing, small leg movements only)"},
    "flight": {"no_takeoff": "stays on the ground", "takeoff_fall": "leaves the ground then falls back / lands",
               "hover": "stays airborne roughly in place", "climb": "rises steadily",
               "forward_flight": "flies forward", "flight_turn_left": "airborne and turning left",
               "flight_turn_right": "airborne and turning right"},
}


def _vision_prompt(mode: str, labels: list[str]) -> str:
    helptxt = "\n".join(f"- {k}: {v}" for k, v in _LABEL_HELP[mode].items())
    return (
        "You are a blind movement verifier in a computational neuroscience lab. The image is a contact sheet of "
        f"keyframes from ONE physics-simulation video, in time order (left to right, top to bottom); each tile is "
        f"labelled with its frame number and time. {_CAMERA_NOTES[mode]}\n\n"
        "Classify the movement the fly body actually performs over the whole clip, using exactly one label:\n"
        f"{helptxt}\n- unclear: you cannot tell from these frames\n\n"
        "Rules: describe only what is visible in the frames; do not guess from what a fly would typically do. "
        "If the camera tracks the fly, use the floor texture for displacement. Prefer 'unclear' over a guess. "
        "Give 2-5 short observations that reference tile numbers. confidence is 0..1."
    )


def _vision_schema(labels: list[str]) -> dict:
    return {"type": "object",
            "properties": {"observed_behavior": {"type": "string", "enum": labels + ["unclear"]},
                           "confidence": {"type": "number"},
                           "observations": {"type": "array", "items": {"type": "string"}},
                           "fly_visible": {"type": "boolean"}},
            "required": ["observed_behavior", "confidence", "observations", "fly_visible"],
            "additionalProperties": False}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vision_check(video_path: str, mode: str = "walk", n_frames: int = 16, sim_duration_s: float | None = None,
                 sheet_path: str | None = None, model: str | None = None, use_cache: bool = True) -> dict:
    """Blind Claude-vision classification of the movement in a video. Returns
    {observed_behavior, confidence, observations, fly_visible, model, frames_used, contact_sheet, cached,
     usage, video_sha256}. The expected behaviour is NOT passed to the model."""
    model = model or _env_value("FLYLAB_VISION_MODEL") or _env_value("ANTHROPIC_MODEL") or DEFAULT_VISION_MODEL
    labels = list(WALK_LABELS if mode == "walk" else FLIGHT_LABELS)
    sha = _sha256(video_path)
    key = hashlib.sha256(f"{sha}|{mode}|{model}|{n_frames}|{sim_duration_s}|{PROMPT_VERSION}".encode()).hexdigest()[:24]
    cache_file = CACHE_DIR / f"{key}.json"
    sheet = contact_sheet(video_path, sheet_path, n_frames=n_frames, sim_duration_s=sim_duration_s)
    if use_cache and cache_file.exists():
        out = json.loads(cache_file.read_text(encoding="utf-8"))
        out.update({"cached": True, "contact_sheet": sheet["path"]})
        return out
    img_b64 = base64.standard_b64encode(Path(sheet["path"]).read_bytes()).decode("ascii")
    t0 = time.time()
    resp = _client().messages.create(
        model=model, max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _vision_schema(labels)}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": _vision_prompt(mode, labels)}]}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("vision model refused")
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    data = json.loads(text)
    u = resp.usage
    out = {**data, "model": model, "frames_used": sheet["frames_used"], "frame_labels": sheet["labels"],
           "video_sha256": sha, "prompt_version": PROMPT_VERSION, "latency_s": round(time.time() - t0, 2),
           "usage": {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)},
           "blind": "model saw neither the expected behaviour nor the kinematic numbers"}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out, indent=1), encoding="utf-8")
    out.update({"cached": False, "contact_sheet": sheet["path"]})
    return out


def _vision_matches(observed: str, expected: str, mode: str, confidence: Any = None) -> bool | None:
    if observed in ("unclear", "", None):
        return None
    c = _f(confidence)
    if c is not None and c < VISION_MIN_CONFIDENCE:
        return None
    if mode == "flight" and expected in ("escape", "airborne"):
        return observed not in ("no_takeoff",)
    return observed == expected


# =========================================================================== main entry


def verify_movement(result: dict, expected_behavior: str, mode: str = "walk", use_vision: bool = True,
                    video_path: str | None = None, sheet_path: str | None = None, n_frames: int = 16) -> dict:
    """Verify that the body performed ``expected_behavior`` (contract: docs/CONTRACTS.md Phase 3)."""
    kin = kinematic_verdict(result, expected_behavior, mode)
    mode = kin["mode"]
    exp = kin["expected_normalized"]
    holder, _ = _extract(result)
    vid = video_path or holder.get("video") or result.get("video")
    if vid and not Path(vid).is_absolute():
        vid = str(ROOT / vid)
    vision = None
    vision_error = None
    sheet = None
    if use_vision and exp not in BRAIN_ONLY:
        if not vid or not Path(vid).exists() or Path(vid).suffix.lower() != ".mp4":
            vision_error = f"no mp4 video available ({vid})"
        else:
            try:
                dur = holder.get("sim_duration_s") or (holder["trajectory"][-1][0] if holder.get("trajectory") else None)
                v = vision_check(vid, mode=mode, n_frames=n_frames, sim_duration_s=_f(dur), sheet_path=sheet_path)
                sheet = v.pop("contact_sheet", None)
                m = _vision_matches(v.get("observed_behavior"), exp, mode, v.get("confidence"))
                vision = {"verdict": "uncertain" if m is None else ("correct" if m else "incorrect"),
                          "observed_behavior": v.get("observed_behavior"), "matches_expected": m,
                          "confidence": v.get("confidence"), "observations": v.get("observations"),
                          "fly_visible": v.get("fly_visible"), "model": v.get("model"),
                          "frames_used": v.get("frames_used"), "cached": v.get("cached"), "usage": v.get("usage"),
                          "video_sha256": v.get("video_sha256"), "blind": v.get("blind")}
            except Exception as exc:  # noqa: BLE001 - vision is optional evidence, never fatal
                vision_error = f"{type(exc).__name__}: {exc}"[:400]
    kv = kin["verdict"]
    if vision is None:
        final, agreement = kv, None
        reason = f"kinematic only ({vision_error or 'vision disabled'}): {kin.get('reason', '')}"
        methods = ["kinematic"]
    else:
        vv = vision["verdict"]
        methods = ["kinematic", "vision"]
        if kv == vv and kv in ("correct", "incorrect"):
            final, agreement = kv, True
            reason = f"kinematics and blind vision agree: {kv}"
        elif kv == "uncertain" or vv == "uncertain":
            final, agreement = "uncertain", (kv == vv)
            reason = (f"kinematic={kv} ({kin.get('reason', '')}), vision={vv} "
                      f"(observed {vision['observed_behavior']}, conf {vision['confidence']})")
        else:
            final, agreement = "uncertain", False
            reason = (f"DISAGREEMENT: kinematic={kv} ({kin.get('reason', '')}) vs vision={vv} "
                      f"(observed {vision['observed_behavior']}, conf {vision['confidence']}) - needs review / rerun")
    return {"expected_behavior": exp, "mode": mode,
            "kinematic": {k: kin[k] for k in ("verdict", "checks", "recomputed", "reason") if k in kin},
            "vision": vision, "vision_error": vision_error, "final_verdict": final, "agreement": agreement,
            "methods": methods, "reason": reason, "contact_sheet": sheet, "video": vid,
            "independence_note": ("kinematics recomputed from the raw trajectory with own formulas/thresholds "
                                  "(not body.classify / flight.classify_flight); vision is blind to both")}


# =========================================================================== self-test / CLI


def _synthetic_walk(vx_body: float, yaw_rate: float, dur: float = 1.0, h0_deg: float = 37.0) -> dict:
    tr = []
    x = y = 0.0
    h = h0_deg
    dt = 0.01
    for i in range(int(dur / dt) + 1):
        tr.append([round(i * dt, 3), x, y, h])
        hr = math.radians(h)
        x += vx_body * dt * math.cos(hr)
        y += vx_body * dt * math.sin(hr)
        h += yaw_rate * dt
    return {"trajectory": tr, "upright_min_cos": 0.99}


def _selftest() -> int:
    cases = [
        (_synthetic_walk(12, 0), "forward", "correct"), (_synthetic_walk(-10, 0), "backward", "correct"),
        (_synthetic_walk(-10, 0), "forward", "incorrect"), (_synthetic_walk(-1.2, 5), "backward", "uncertain"),
        (_synthetic_walk(0, 150), "turn_left", "correct"), (_synthetic_walk(0, -150), "turn_left", "incorrect"),
        (_synthetic_walk(0, 0), "stop", "correct"), (_synthetic_walk(0, 0), "groom", "uncertain"),
    ]
    fl = {"trajectory": [[i * 0.01, 0.0, 0.0, 1.0 + (min(i, 30) * 0.2), 0.0, 0.0, 10.0] for i in range(101)]}
    cases.append((fl, "escape", "correct"))
    cases.append((fl, "no_takeoff", "incorrect"))
    bad = 0
    for res, exp, want in cases:
        got = verify_movement(res, exp, use_vision=False)["final_verdict"]
        flag = "ok " if got == want else "BAD"
        bad += got != want
        print(f"{flag} expected={exp:<11} want={want:<9} got={got}")
    print("selftest", "PASSED" if not bad else f"FAILED ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--expected", default="")
    ap.add_argument("--mode", default="walk")
    ap.add_argument("--sheet")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.video:
        print(json.dumps(vision_check(a.video, mode=a.mode, sheet_path=a.sheet), indent=1))
