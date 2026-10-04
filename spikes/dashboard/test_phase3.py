"""Phase-3 dashboard test: Flight & 3D page, movement verification + parallel batches in the notebook,
flight checks on Validation & speed, challenge checklist on Method & limits.

Uses SYNTHETIC fixtures (spikes/dashboard/out/fixtures, gitignored, marked mock) because the flight,
verifier and 3D-export workstreams are still producing the real files. Real-data pages are covered by
test_pages.py. Usage (from 02_App): uv run python spikes/dashboard/test_phase3.py
"""
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
FX = ROOT / "spikes" / "dashboard" / "out" / "fixtures"


def make_fixtures() -> None:
    if FX.exists():
        shutil.rmtree(FX)
    for d in ("bench", "web/data/runs", "assets/flight", "runs/fixture_flight_mock", "docs"):
        (FX / d).mkdir(parents=True, exist_ok=True)
    # contact sheet image + a small video copied from the committed walking assets
    from PIL import Image

    Image.new("RGB", (240, 80), (200, 220, 235)).save(FX / "assets/flight/GF_bilateral_contact.png")
    src_vid = ROOT / "assets/embodied/control.mp4"
    if src_vid.exists():
        shutil.copy(src_vid, FX / "assets/flight/GF_bilateral.mp4")
    cs = str((FX / "assets/flight/GF_bilateral_contact.png").resolve())
    ver_ok = {"kinematic": {"verdict": "correct", "checks": [
        {"name": "max_height_mm", "value": 4.2, "threshold": "> 1.0", "pass": True},
        {"name": "airborne_time_s", "value": 0.6, "threshold": "> 0.2", "pass": True}], "recomputed": {}},
        "vision": {"verdict": "correct", "observations": "MOCK: fly leaves the ground", "model": "mock",
                   "frames_used": 6},
        "final_verdict": "correct", "agreement": True, "contact_sheet": cs, "expected_behavior": "climb",
        "mode": "flight"}
    ver_bad = {"kinematic": {"verdict": "incorrect", "checks": [
        {"name": "max_height_mm", "value": 0.1, "threshold": "> 1.0", "pass": False}]},
        "vision": {"verdict": "uncertain", "observations": ["MOCK: legs move", "no takeoff"], "model": "mock",
                   "frames_used": 6},
        "final_verdict": "incorrect", "agreement": False, "contact_sheet": "assets/flight/missing.png"}
    flight = {"mock": True, "what": "SYNTHETIC test fixture", "caveats": ["MOCK fixture for the dashboard test."],
              "adapter": {"takeoff": "GF rate / 148.3 Hz", "frozen": True},
              "rows": [
                  {"condition": "GF_bilateral", "stimulus_groups": ["GF"], "silenced_groups": [],
                   "key_group_rates_hz": {"GF": 160.2, "DNg02": 12.0, "MDN": 0.0},
                   "command": {"takeoff": 1.0, "thrust": 0.4, "yaw": 0.0, "pitch": 0.1, "explain": {"x": 1}},
                   "behavior": "climb", "expected_behavior": "climb", "verification": ver_ok,
                   "checks": [{"gt_id": "gt08_lplc2_activate_escape", "verdict": "consistent"}],
                   "runtimes_s": {"brain": 2.0, "flight": 3.0, "total": 5.0}, "video": "GF_bilateral.mp4",
                   "mock": True},
                  {"condition": "LPLC2_bilateral", "stimulus_groups": ["LPLC2"], "key_group_rates_hz": {"GF": 3.0},
                   "command": {"takeoff": 0.0, "thrust": 0.0, "yaw": 0.0, "pitch": 0.0}, "behavior": "no_takeoff",
                   "expected_behavior": "climb", "verification": ver_bad,
                   "checks": [{"gt_id": "gt08_lplc2_activate_escape", "verdict": "inconsistent"}],
                   "runtimes_s": {"brain": 2.0, "flight": 3.0}, "mock": True},
                  {"condition": "control", "stimulus_groups": [], "behavior": "no_takeoff", "mock": True},
              ]}
    (FX / "bench/flight_validation.json").write_text(json.dumps(flight), encoding="utf-8")
    (FX / "web/index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>viewer</title></head><body>"
        "<script type='module' src='./app.js'></script>FIXTURE VIEWER</body></html>", encoding="utf-8")
    (FX / "web/app.js").write_text("console.log('fixture');", encoding="utf-8")
    (FX / "web/data/runs/GF_bilateral.json").write_text(json.dumps(
        {"mode": "flight", "behavior": "climb", "verification": ver_ok,
         "poses": {"frames": [{"t": 0, "p": [], "q": []}] * 3}}), encoding="utf-8")
    ev = [
        {"seq": 0, "agent": "human", "type": "question", "content": "MOCK: does GF drive takeoff?", "data": None},
        {"seq": 1, "agent": "runner", "type": "experiment_batch", "content": "MOCK batch of 2 in parallel",
         "data": {"batch_id": "b1", "n_workers": 2, "wall_s": 6.0, "mock": True,
                  "experiments": [{"id": "e1", "excite_groups": ["GF"]}, {"id": "e2", "excite_groups": ["LPLC2"]}]}},
        {"seq": 2, "agent": "runner", "type": "experiment_result", "content": "MOCK flight GF",
         "data": {"kind": "flight", "batch_id": "b1", "runtime_s": 5.0, "excite": ["GF"], "mock": True,
                  "command": {"takeoff": 1, "thrust": 0.4, "yaw": 0, "pitch": 0.1},
                  "flight": {"behavior": "climb", "airborne": True, "max_height_mm": 4.2, "flight_time_s": 0.6,
                             "heading_change_deg": 1.0, "video": "assets/flight/none.mp4"},
                  "verification": ver_ok}},
        {"seq": 3, "agent": "runner", "type": "experiment_result", "content": "MOCK flight LPLC2",
         "data": {"kind": "flight", "batch_id": "b1", "runtime_s": 4.0, "excite": ["LPLC2"], "mock": True}},
        {"seq": 4, "agent": "movement_verifier", "type": "movement_verification", "content": "MOCK verified GF",
         "data": {**ver_ok, "condition": "GF_bilateral", "mock": True}},
        {"seq": 5, "agent": "movement_verifier", "type": "movement_verification", "content": "MOCK verified LPLC2",
         "data": {"verification": ver_bad, "condition": "LPLC2_bilateral", "mock": True}},
    ]
    (FX / "runs/fixture_flight_mock/record.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n",
                                                               encoding="utf-8")
    shutil.copy(ROOT / "docs/CHALLENGE_REQUIREMENTS.md", FX / "docs/CHALLENGE_REQUIREMENTS.md")
    (FX / "docs/CHALLENGE_COMPLIANCE.md").write_text(
        "# Compliance (fixture)\n\n| ID | Requirement | Evidence | Status |\n|---|---|---|---|\n"
        "| R1 | Omnigent | [fly_lab.yaml](../agents/fly_lab.yaml), live run | done |\n"
        "| **R6** | parallel | [tools](../flylab/tools.py#L10) | partial |\n\n"
        "- S4: [viewer](https://example.org) pending approval\n\n### G4 connectome decides\nTakeoff trigger only.\n",
        encoding="utf-8")


def run_fx(name: str, fx: str):
    import importlib
    from pathlib import Path

    from flylab import ui

    saved = {k: getattr(ui, k) for k in ("BENCH", "WEB", "FLIGHT_ASSETS", "RUNS", "DOCS")}
    f = Path(fx)
    ui.BENCH, ui.WEB, ui.FLIGHT_ASSETS = f / "bench", f / "web", f / "assets" / "flight"
    ui.RUNS, ui.DOCS = f / "runs", f / "docs"
    try:
        ui.inject_css()
        importlib.import_module(f"flylab.ui_{name}").page()
    finally:
        for k, v in saved.items():
            setattr(ui, k, v)


def check(at, label) -> bool:
    errs = [e.value for e in at.exception]
    print(f"{label}: exceptions={len(errs)} markdown={len(at.markdown)} dataframes={len(at.dataframe)} "
          f"metrics={len(at.metric)} errors={len(at.error)}")
    for e in errs:
        print("   EXC:", str(e)[:2500])
    return not errs


def metric(at, label):
    return next((m.value for m in at.metric if m.label == label), None)


ok = True
t0 = time.time()
make_fixtures()
os.environ["FLYLAB_VIEWER_PORT"] = "0"  # do not take the demo port in tests

at = AppTest.from_function(run_fx, args=("flight", str(FX)), default_timeout=120).run()
ok &= check(at, "fixture flight page")
for lab, want in (("Flight conditions", "3"), ("Verifier: correct movement", "1 / 2"),
                  ("Kinematic vs. vision agree", "1 / 2"), ("Published flight checks consistent", "1 / 2")):
    got = metric(at, lab)
    print(f"   metric {lab!r} = {got!r} (want {want!r})")
    ok &= got == want
md = " ".join(m.value for m in at.markdown)
ok &= "MOCK DATA" in md and "local copy of web/" in md
if len(at.selectbox):
    for i in range(len(at.selectbox[0].options)):
        at.selectbox[0].set_value(i).run()
        ok &= check(at, f"   flight condition {i}")

# local viewer server: index.html and JS with correct MIME types, no directory listing
from flylab import ui_flight  # noqa: E402

srv = ui_flight._local_viewer_server(str(FX / "web"), 0)
base = f"http://127.0.0.1:{srv['port']}/"
with urllib.request.urlopen(base, timeout=5) as r:
    body = r.read().decode()
    print("   viewer index:", r.status, r.headers.get("Content-Type"), "FIXTURE VIEWER" in body)
    ok &= r.status == 200 and "FIXTURE VIEWER" in body
with urllib.request.urlopen(base + "app.js", timeout=5) as r:
    print("   viewer app.js:", r.headers.get("Content-Type"))
    ok &= r.headers.get("Content-Type", "").startswith("text/javascript")
try:
    urllib.request.urlopen(base + "data/", timeout=5)
    print("   directory listing NOT blocked")
    ok = False
except urllib.error.HTTPError as exc:
    print("   directory listing blocked:", exc.code)
try:
    urllib.request.urlopen(base + "../app.py", timeout=5)
    print("   path traversal NOT blocked")
    ok = False
except urllib.error.HTTPError as exc:
    print("   path traversal blocked:", exc.code)

at = AppTest.from_function(run_fx, args=("notebook", str(FX)), default_timeout=120).run()
ok &= check(at, "fixture notebook")
print("   movement checks metric:", metric(at, "Movement checks"))
ok &= metric(at, "Movement checks") == "1 / 2"
md = " ".join(m.value for m in at.markdown)
ok &= "Parallel experiment batches" in md and "Movement verification" in md
batch_df = next((d.value for d in at.dataframe if "observed parallel speed-up" in d.value.columns), None)
print("   batch table:", None if batch_df is None else batch_df.to_dict("records"))
ok &= batch_df is not None and batch_df.iloc[0]["observed parallel speed-up"] == "1.5x"

at = AppTest.from_function(run_fx, args=("validation", str(FX)), default_timeout=180).run()
ok &= check(at, "fixture validation")
md = " ".join(m.value for m in at.markdown)
ok &= "Flight agreement by stimulus type" in md  # walking split: real data, see test_pages.py
print("   flight metric:", metric(at, "Published flight checks consistent"))

at = AppTest.from_function(run_fx, args=("method", str(FX)), default_timeout=120).run()
ok &= check(at, "fixture method")
md = " ".join(m.value for m in at.markdown)
want_link = "github.com/maximiliankahl/embodied-fly-lab/blob/main/agents/fly_lab.yaml"
print("   checklist rows R1/G5:", "| **R1** |" in md, "| **G5** |" in md, "| repo link rewritten:", want_link in md,
      "| status partial:", "partial" in md)
ok &= "| **R1** |" in md and "| **G5** |" in md and want_link in md and "Takeoff trigger only." in md

print("ALL OK" if ok else "FAILURES", f"{time.time() - t0:.1f}s")
sys.exit(0 if ok else 1)
