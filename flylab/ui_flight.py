"""Dashboard page: Flight & 3D - brain-driven flight checks, movement verification, 3D replay viewer.

Reads (each may be missing; other workstreams produce them):
  data/benchmarks/flight_validation.json   flight conditions: stimulus, brain rates, adapter command,
                                           observed behaviour, movement-verifier verdict, ground-truth verdict
  assets/flight/*.mp4|png|jpg              recorded flight videos and verifier contact sheets
  web/index.html (+ web/data/**)           static Three.js replay viewer (also deployed to GitHub Pages)

How the 3D viewer is embedded (most robust option, no symlinks, no copies):
  1. FLYLAB_VIEWER_URL (env) if set: iframe to that URL.
  2. web/index.html exists and the browser runs on this machine (Host header localhost/127.0.0.1):
     the dashboard starts ONE read-only static file server for web/ inside its own process
     (127.0.0.1 only, explicit MIME types so ES modules load on Windows, no directory listing, no caching)
     and embeds http://127.0.0.1:<port>/ (port 8765, or a free port if busy; FLYLAB_VIEWER_PORT overrides).
     Streamlit's own static serving was not used: it only serves ./static next to app.py (would need a
     second copy of web/ in the repo) and takes MIME types from the Windows registry.
  3. The public GitHub Pages URL, if it answers (checked server-side, cached 10 min). Used on a
     cloud deployment where 127.0.0.1 is not the viewer's machine.
  4. Otherwise a placeholder with the command to open the viewer locally.
The viewer replays RECORDED runs (poses exported from MuJoCo); nothing is re-simulated in the browser.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from flylab import ui

DEFAULT_VIEWER_PORT = 8765
_MIME = {
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json", ".map": "application/json",
    ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
    ".bin": "application/octet-stream", ".npy": "application/octet-stream", ".gz": "application/gzip",
    ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".obj": "text/plain; charset=utf-8",
    ".stl": "model/stl", ".wasm": "application/wasm",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".mp4": "video/mp4", ".webm": "video/webm", ".woff": "font/woff", ".woff2": "font/woff2",
}
_VIDEO_EXT = (".mp4", ".webm", ".mov")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")


# --------------------------------------------------------------------------- local viewer server


@st.cache_resource(show_spinner=False)
def _local_viewer_server(web_dir: str, preferred_port: int) -> dict:
    """One read-only HTTP server for web/ per dashboard process (127.0.0.1 only)."""
    import functools
    import http.server
    import threading

    class _Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **_MIME}

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def list_directory(self, path):  # no directory listings
            self.send_error(404, "Not found")
            return None

        def log_message(self, *args: Any) -> None:  # keep the Streamlit console clean
            pass

    class _Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = False  # on Windows SO_REUSEADDR would allow two servers on one port
        daemon_threads = True

    handler = functools.partial(_Handler, directory=web_dir)
    server = None
    for port in (preferred_port, 0):
        try:
            server = _Server(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    if server is None:
        return {"port": None, "error": "could not bind a local port"}
    threading.Thread(target=server.serve_forever, name="flylab-viewer", daemon=True).start()
    return {"port": server.server_address[1], "error": None}


@st.cache_data(ttl=600, show_spinner=False)
def _url_reachable(url: str) -> bool:
    import urllib.request

    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "embodied-fly-lab-dashboard"})
        with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310 (fixed https URL)
            return 200 <= int(r.status) < 400
    except Exception:
        return False


def _browser_is_local() -> bool:
    """True if the browser talks to this server via localhost (unknown, e.g. in tests -> True)."""
    try:
        host = str(st.context.headers.get("Host") or "")
    except Exception:
        return True
    if not host:
        return True
    h = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0]
    h = h.lower()
    return h in ("localhost", "127.0.0.1", "::1") or h.endswith(".localhost")


def viewer_sources() -> tuple[list[tuple[str, str]], list[str]]:
    """[(label, url)] in priority order, plus notes explaining missing options."""
    srcs: list[tuple[str, str]] = []
    notes: list[str] = []
    env_url = os.environ.get("FLYLAB_VIEWER_URL", "").strip()
    if env_url.startswith(("http://", "https://")):
        srcs.append(("configured URL (FLYLAB_VIEWER_URL)", env_url))
    index = ui.WEB / "index.html"
    if index.exists():
        if _browser_is_local():
            try:
                port = int(os.environ.get("FLYLAB_VIEWER_PORT", DEFAULT_VIEWER_PORT))
            except ValueError:
                port = DEFAULT_VIEWER_PORT
            srv = _local_viewer_server(str(ui.WEB), port)
            if srv.get("port"):
                srcs.append(("local copy of web/ (this machine)", f"http://127.0.0.1:{srv['port']}/"))
            else:
                notes.append(f"Local viewer server failed: {srv.get('error')}.")
        else:
            notes.append("web/index.html exists, but your browser is not on the dashboard's machine, so the "
                         "local copy cannot be embedded here.")
    else:
        notes.append("web/index.html is not in this build yet (written by the 3D export workstream).")
    if _url_reachable(ui.PAGES_URL):
        srcs.append(("GitHub Pages (public)", ui.PAGES_URL))
    else:
        notes.append(f"The public viewer ({ui.PAGES_URL}) does not answer yet; GitHub Pages is only enabled "
                     "after the team approves publishing.")
    return srcs, notes


def _iframe(url: str, height: int = 640) -> None:
    if hasattr(st, "iframe"):
        st.iframe(url, height=height)
    else:  # older Streamlit
        import streamlit.components.v1 as components

        components.iframe(url, height=height, scrolling=False)


def _recorded_3d_runs() -> list[dict]:
    """Runs exported for the viewer: web/data/runs/index.json (flylab-run-index-v1) if present, else one row
    per web/data/runs/*.json with a few metadata fields."""
    d = ui.WEB / "data" / "runs"
    if not d.exists():
        return []
    idx = ui.load_json(d / "index.json")
    if isinstance(idx, dict) and isinstance(idx.get("runs"), list):
        rows = []
        for r in idx["runs"]:
            if isinstance(r, dict) and r.get("id"):
                rows.append({k: r.get(k) for k in ("id", "title", "mode", "kind", "expected", "behavior", "verdict",
                                                    "created") if k in r})
        if rows:
            return rows
    rows = []
    for p in sorted(d.glob("*.json")):
        if p.name.lower() in ("index.json", "manifest.json", "runs.json"):
            continue
        row: dict[str, Any] = {"id": p.stem, "size (KB)": round(p.stat().st_size / 1024, 1)}
        if p.stat().st_size < 8_000_000:
            doc = ui.load_json(p)
            if isinstance(doc, dict):
                ver = ui.verification_summary(ui.dig(doc, "verification", "verifier", "verify"))
                row.update({
                    "title": doc.get("title") or "",
                    "mode": ui.dig(doc, "mode", "body.mode", default=""),
                    "behavior": ui.dig(doc, "behavior", "body.behavior", "result.behavior", default=""),
                    "verdict": ver.get("final") or "",
                })
        rows.append(row)
    return rows


def _manifest() -> None:
    """Provenance of the exported viewer data (web/data/manifest.json: generator, git revision, SHA256 per file)."""
    import pandas as pd

    man = ui.load_json(ui.WEB / "data" / "manifest.json")
    if not isinstance(man, dict):
        return
    files = [f for f in (man.get("files") or []) if isinstance(f, dict)]
    total = sum(int(f.get("bytes") or 0) for f in files)
    with st.expander(f"Provenance of the viewer data: {len(files)} files, {total / 1e6:.1f} MB, SHA256 per file"):
        st.caption(f"Generated {man.get('generated', '?')} by `{man.get('generator', '?')}` at git revision "
                   f"`{man.get('git_rev', '?')}`.")
        if files:
            st.dataframe(pd.DataFrame(files), hide_index=True, width="stretch")


def _viewer_section() -> None:
    import pandas as pd

    st.subheader("3D replay viewer")
    st.caption("Three.js viewer of RECORDED simulation runs: body poses exported from the MuJoCo physics, plus the "
               "brain activity summary, adapter output and verifier verdict of each run. Nothing is re-simulated in "
               "the browser.")
    srcs, notes = viewer_sources()
    runs = _recorded_3d_runs()
    if srcs:
        labels = [lab for lab, _ in srcs]
        c1, c2 = st.columns([1, 1])
        with c1:
            pick = st.radio("Viewer source", labels, horizontal=True) if len(srcs) > 1 else labels[0]
        titles = {str(r["id"]): f"{r['id']}: {r.get('title') or ''}".rstrip(": ") for r in runs if r.get("id")}
        with c2:
            run = st.selectbox("Recorded run", [""] + list(titles),
                               format_func=lambda k: titles.get(k, "(viewer default)")) if titles else ""
        url = dict(srcs)[pick] + (f"#{run}" if run else "")
        _iframe(url)
        st.markdown(f"Source: **{pick}** · [open in a new tab]({url})")
    else:
        ui.placeholder("The 3D viewer", "Open it locally with `uv run python -m http.server 8765 -d web` and "
                       "http://127.0.0.1:8765/ once web/index.html exists, or enable GitHub Pages for the public link.")
    for n in notes:
        st.caption(n)
    if runs:
        st.markdown(f"**Recorded runs available to the viewer** ({len(runs)}, `web/data/runs/`)")
        st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    _manifest()
    st.caption("Ground rule G5: a whole-brain point cloud shows neuron positions and recorded firing-rate summaries; it "
               "is a visualisation, not a whole-brain re-simulation. Runs are open loop (no sensory feedback from the "
               "body to the brain).")


# --------------------------------------------------------------------------- flight validation


def _fmt_cmd(cmd: dict) -> str:
    parts = []
    for k in ("takeoff", "thrust", "yaw", "pitch"):
        if cmd.get(k) is not None:
            parts.append(f"{k} {ui.fmt_num(cmd.get(k), 2)}")
    return " · ".join(parts)


def _gf_rate(rates: dict) -> Any:
    for k in ("GF", "DNp01", "GF_bilateral", "giant_fiber"):
        if rates.get(k) is not None:
            return round(float(rates[k]), 1)
    return None


def _other_rates(rates: dict, n: int = 4) -> str:
    items = []
    for k, v in rates.items():
        if k in ("GF", "DNp01", "giant_fiber"):
            continue
        try:
            items.append((k, float(v)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda kv: -kv[1])
    return ", ".join(f"{k} {v:.0f}" for k, v in items[:n] if v > 0)


def _table_rows(rows: list[dict]) -> list[dict]:
    gt = ui.ground_truth_by_id()
    out = []
    for r in rows:
        v = r["verify"]
        verifier = v.get("final") or "not run"
        sub = [x for x in (f"kin. {v['kinematic']}" if v.get("kinematic") else "",
                           f"vision {v['vision']}" if v.get("vision") else "") if x]
        if sub:
            verifier += f" ({', '.join(sub)})"
        gts, doi = [], None
        for c in r["checks"]:
            verdict = str(c.get("verdict") or "not run").replace("_", " ")
            if c.get("informative") is False:
                verdict += " (uninformative)"
            gts.append(f"{c['gt_id']}: {verdict}")
            cit = (gt.get(c["gt_id"]) or {}).get("citation") or {}
            doi = doi or cit.get("doi")
        doi = doi or ui.dig(r["raw"], "citation.doi", "doi")
        stim = "excite " + ", ".join(r["stim"]) if r["stim"] else "no stimulus"
        if r["silenced"]:
            stim += "; silence " + ", ".join(r["silenced"])
        out.append({
            "condition": str(r["condition"]),
            "stimulus": stim,
            "stimulus type": r["stim_type"],
            "GF (Hz)": _gf_rate(r["rates"]),
            "other DN rates (Hz)": _other_rates(r["rates"]),
            "adapter command": _fmt_cmd(r["command"]),
            "behaviour": str(r["behavior"] or ""),
            "expected": str(r["expected"] or ""),
            "verifier": verifier,
            "kin. vs vision agree": "" if v.get("agreement") is None else ("yes" if v["agreement"] else "NO"),
            "ground truth": "; ".join(gts) or "no published comparison",
            "paper": ui.doi_url(doi),
            "wall s": r["wall_s"],
        })
    return out


def _style_cells(col):
    def color(x: Any) -> str:
        s = str(x).lower()
        if s.startswith(("correct", "consistent")) or ": consistent" in s:
            return "background-color: #d8f0de"
        if s.startswith(("incorrect", "inconsistent")) or ": inconsistent" in s or s == "no":
            return "background-color: #fbdcdc"
        if "partially" in s or s.startswith("uncertain"):
            return "background-color: #fff0d6"
        return ""
    return [color(x) for x in col]


def _row_detail(r: dict) -> None:
    import pandas as pd

    v = r["verify"]
    c1, c2 = st.columns([1, 1])
    with c1:
        vid = ui.resolve(r["video"])
        if vid is None and ui.FLIGHT_ASSETS.exists():
            cand = [p for p in ui.FLIGHT_ASSETS.rglob("*") if p.suffix.lower() in _VIDEO_EXT
                    and p.stem.lower() == str(r["condition"]).lower()]
            vid = cand[0] if cand else None
        if vid is not None and vid.suffix.lower() in _VIDEO_EXT:
            st.video(str(vid))
        else:
            st.caption("No video for this condition yet" + (f" (`{r['video']}` not found)." if r["video"] else "."))
        cs = ui.resolve(v.get("contact_sheet"))
        if cs is not None and cs.suffix.lower() in _IMAGE_EXT:
            st.image(str(cs), caption="Contact sheet shown to the vision verifier", width="stretch")
    with c2:
        if v.get("final"):
            st.markdown(f"Movement verifier: {ui.badge(v['final'], ui.VERIFY_COLOR.get(str(v['final']), 'gray'))}"
                        + (f" · expected **{v['expected'] or r['expected']}**" if (v.get('expected') or r['expected']) else ""))
            if v.get("agreement") is not None:
                st.markdown("Kinematic and vision verdicts "
                            + (ui.badge("agree", "green") if v["agreement"] else ui.badge("disagree", "red")))
        if v.get("checks"):
            st.markdown("**Kinematic checks** (recomputed from the raw trajectory, independent of the classifier)")
            df = pd.DataFrame(v["checks"])
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].map(lambda x: "" if x is None else str(x))
            st.dataframe(df, hide_index=True, width="stretch")
        if v.get("observations"):
            st.markdown("**Vision verifier** " + (f"({v.get('model')}, {v.get('frames_used')} frames)" if v.get("model") else ""))
            obs = v["observations"]
            st.caption(obs if isinstance(obs, str) else "; ".join(map(str, obs)) if isinstance(obs, list) else str(obs))
        if r["command"]:
            st.markdown(f"Adapter command: {_fmt_cmd(r['command'])}")
            if isinstance(r["command"].get("explain"), dict):
                with st.expander("How the brain rates became this command"):
                    st.json(r["command"]["explain"], expanded=False)
        for c in r["checks"]:
            if c.get("reason"):
                st.caption(f"{c['gt_id']}: {c['reason']}")
        if r["runtimes"]:
            st.caption("Wall time: " + ", ".join(f"{k} {ui.fmt_num(x, 1)} s" for k, x in r["runtimes"].items()))
    with st.expander("Raw row"):
        st.json(r["raw"], expanded=False)


def _flight_section() -> None:
    import pandas as pd

    st.subheader("Flight checks: brain -> adapter -> winged body -> movement verifier")
    doc = ui.load_json(ui.BENCH / "flight_validation.json")
    rows = ui.flight_rows(doc)
    if not rows:
        ui.placeholder("data/benchmarks/flight_validation.json",
                       "It is written by the flight workstream (flylab/flight.py + bridge.rates_to_flight_command + "
                       "flylab/verify.py). Each row: stimulus, giant-fiber and flight-DN rates, the frozen adapter's "
                       "command, the physics behaviour, the movement verifier's verdict and the published comparison.")
        return
    if any(r["mock"] for r in rows) or (isinstance(doc, dict) and doc.get("mock")):
        ui.mock_banner("Part of the flight validation file")
    fs = ui.flight_summary(rows)
    m = st.columns(4)
    m[0].metric("Flight conditions", fs["n_conditions"])
    m[1].metric("Verifier: correct movement", f"{fs['n_verified_correct']} / {fs['n_verified']}" if fs["n_verified"] else "n/a",
                help="final verdict of flylab.verify (kinematic checks recomputed from the raw trajectory, plus a "
                     "Claude vision check of keyframes when available) against the expected behaviour")
    m[2].metric("Kinematic vs. vision agree", f"{fs['n_agree']} / {fs['n_both_verifiers']}" if fs["n_both_verifiers"] else "n/a")
    m[3].metric("Published flight checks consistent", f"{fs['n_gt_consistent']} / {fs['n_gt']}" if fs["n_gt"] else "n/a",
                help="informative, comparable checks only; kept separate from the walking validation")
    if fs["by_stim_type"]:
        st.caption("By stimulus type: " + "; ".join(f"{k}: {v['consistent']}/{v['n']} consistent"
                                                   for k, v in fs["by_stim_type"].items())
                   + ". Direct descending-neuron stimulation feeds the adapter input directly (partly circular); "
                     "upstream stimulation tests what the connectome model produces.")
    decides = ui.dig(doc, "what_the_connectome_decides", "connectome_decides", "scope", "adapter.scope")
    st.info("**What the connectome decides** (ground rule G4): "
            + (decides if isinstance(decides, str) else
               "per the adapter contract (docs/CONTRACTS.md, Phase 3) the brain model's giant-fiber rate triggers "
               "takeoff, flight-motor descending-neuron rates set thrust and the left/right asymmetry sets yaw. Wing "
               "kinematics and flight stabilisation come from the body controller, not from the connectome. No "
               "'if neuron X fires, play an animation': the behaviour comes from MuJoCo physics."))
    df = pd.DataFrame(_table_rows(rows))
    st.dataframe(df.style.apply(_style_cells, subset=["verifier", "ground truth", "kin. vs vision agree"]),
                 hide_index=True, width="stretch",
                 column_config={"paper": st.column_config.LinkColumn("paper (DOI)", display_text=r"https://doi\.org/(.*)")})
    cav = [c for c in as_strs(ui.dig(doc, "caveats", "limitations", "model_notes")) if c]
    if cav:
        st.info("**How to read these numbers**\n" + "\n".join(f"- {c}" for c in cav))
    adapter = ui.dig(doc, "adapter", "bridge", "protocol")
    if isinstance(adapter, dict):
        with st.expander("Adapter and protocol, frozen before the comparison runs (ground rule G3)"):
            st.json(adapter, expanded=False)
    pick = st.selectbox("Inspect a condition", list(range(len(rows))),
                        format_func=lambda i: f"{rows[i]['condition']}  ({rows[i]['behavior'] or '?'}; verifier "
                                              f"{rows[i]['verify'].get('final') or 'not run'})")
    _row_detail(rows[pick])
    with st.expander("Raw flight_validation.json"):
        st.json(doc, expanded=False)


def as_strs(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(y) for y in x if isinstance(y, (str, int, float))]
    if isinstance(x, dict):
        return [f"{k}: {v}" for k, v in x.items() if isinstance(v, (str, int, float))]
    return []


def _media_section() -> None:
    st.subheader("Recorded flight videos and contact sheets")
    files = sorted(p for p in ui.FLIGHT_ASSETS.rglob("*") if p.is_file()) if ui.FLIGHT_ASSETS.exists() else []
    vids = [p for p in files if p.suffix.lower() in _VIDEO_EXT]
    imgs = [p for p in files if p.suffix.lower() in _IMAGE_EXT]
    if not vids and not imgs:
        ui.placeholder("Flight videos (assets/flight/)", "They are rendered by flylab/flight.py for the validation "
                       "conditions; contact sheets come from flylab/verify.py.")
        return
    if vids:
        by_name = {p.relative_to(ui.FLIGHT_ASSETS).as_posix(): p for p in vids}
        name = st.selectbox("Video", list(by_name))
        pick = by_name[name]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.video(str(pick))
        with c2:
            st.caption(f"`assets/flight/{name}` · {pick.stat().st_size / 1e6:.1f} MB. "
                       "Physics rendering (MuJoCo), not an animation.")
    if imgs:
        st.markdown("**Contact sheets** (keyframes the vision verifier sees)")
        cols = st.columns(3)
        for i, p in enumerate(imgs[:12]):
            cols[i % 3].image(str(p), caption=p.stem, width="stretch")


def _verifier_test_section() -> None:
    """data/benchmarks/movement_verifier.json: how reliable is the movement verifier itself?"""
    import pandas as pd

    st.subheader("How reliable is the movement verifier?")
    doc = ui.load_json(ui.BENCH / "movement_verifier.json")
    if not isinstance(doc, dict):
        ui.placeholder("data/benchmarks/movement_verifier.json", "It is written by the verifier test of flylab/verify.py "
                       "(known labels + a negative control with the opposite label per video).")
        return
    st.caption(f"{doc.get('what', '')} · verifier `{doc.get('verifier', '?')}`, prompt {doc.get('prompt_version', '?')}, "
               f"created {doc.get('created', '?')}")
    s = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}
    n = s.get("n_runs")

    def frac(a, b):
        return f"{a} / {b}" if a is not None and b is not None else "n/a"

    m = st.columns(4)
    m[0].metric("Kinematic: correct on known label", frac(s.get("kinematic_correct_on_known_label"), n))
    m[1].metric("Kinematic: rejects opposite label", frac(s.get("kinematic_incorrect_on_opposite_label"), n),
                help="negative control: the same video checked against the opposite behaviour must fail")
    m[2].metric("Vision: exact label (unique videos)", frac(s.get("vision_exact_on_unique_videos"),
                                                            s.get("vision_unique_videos")))
    m[3].metric("Final: correct / uncertain", f"{s.get('final_correct_on_known_label', '?')} / "
                f"{s.get('final_uncertain_on_known_label', '?')}",
                help=f"final verdict 'correct' needs kinematic AND vision to agree; 'correct' on the opposite label "
                     f"(false accept): {s.get('final_correct_on_opposite_label', '?')}")
    cav = [c for c in (doc.get("caveats") or []) if isinstance(c, str)]
    if cav:
        st.info("**Read with care**\n" + "\n".join(f"- {c}" for c in cav))
    rows = []
    for r in doc.get("rows") or []:
        if not isinstance(r, dict):
            continue
        neg = r.get("negative_control") if isinstance(r.get("negative_control"), dict) else {}
        conf = r.get("vision_confidence")
        rows.append({"condition": r.get("condition"), "known label": r.get("known_label"),
                     "kinematic": r.get("kinematic_verdict"),
                     "vision saw": f"{r.get('vision_observed') or 'n/a'}"
                                   + (f" ({float(conf):.2f})" if isinstance(conf, (int, float)) else ""),
                     "vision correct": "" if r.get("vision_correct") is None else ("yes" if r["vision_correct"] else "no"),
                     "final": r.get("final_verdict"),
                     "negative control (opposite label)": f"{neg.get('expected', '?')}: kinematic "
                                                          f"{neg.get('kinematic', '?')}, final {neg.get('final', '?')}"
                                                          if neg else "",
                     "vision error": r.get("vision_error") or ""})
    if rows:
        st.dataframe(pd.DataFrame(rows).style.apply(_style_cells, subset=["kinematic", "final"]),
                     hide_index=True, width="stretch")


def page() -> None:
    st.title("Flight & 3D")
    st.caption("Can the brain model make the fly take off and steer, and did the body really do it? Every flight run "
               "is checked by a movement-verifier agent: kinematics recomputed from the raw trajectory plus a vision "
               "check of keyframes.")
    tabs = st.tabs(["Flight validation", "3D replay viewer", "Movement verifier test", "Videos"])
    with tabs[0]:
        _flight_section()
    with tabs[1]:
        _viewer_section()
    with tabs[2]:
        _verifier_test_section()
    with tabs[3]:
        _media_section()
