"""Render web/film.html frame by frame (deterministic renderAt(t)) with headless Edge, then encode with ffmpeg.

uv run --with playwright python spikes/film/render_film.py --preview 3,10,17      # a few stills
uv run --with playwright python spikes/film/render_film.py --fps 30               # full film -> spikes/film/out/frames
Needs the web/ folder served on http://localhost:8777 (python -m http.server 8777 --directory web).
"""
import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8777/film.html?capture")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--preview", default="")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    a = ap.parse_args()
    frames_dir = OUT / ("preview" if a.preview else "frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True,
                              args=["--use-angle=d3d11", "--ignore-gpu-blocklist", "--enable-gpu-rasterization", "--enable-webgl"])
        page = b.new_page(viewport={"width": a.w, "height": a.h}, device_scale_factor=1)
        page.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
        page.goto(a.url)
        page.wait_for_function("window.__film && (window.__film.ready || window.__film.error)", timeout=120000)
        err = page.evaluate("window.__film.error || null")
        if err:
            raise SystemExit(f"film error: {err}")
        dur = page.evaluate("window.__film.duration")
        times = [float(x) for x in a.preview.split(",")] if a.preview else None
        if times is None:
            end = a.end if a.end is not None else dur
            n0, n1 = int(round(a.start * a.fps)), int(round(end * a.fps))
            times = [i / a.fps for i in range(n0, n1)]
        t0 = time.time()
        for k, t in enumerate(times):
            page.evaluate("t => window.__film.renderAt(t)", t)
            idx = int(round(t * a.fps)) if not a.preview else k
            name = f"{idx:05d}.jpg" if not a.preview else f"t{t:05.1f}.jpg"
            page.screenshot(path=str(frames_dir / name), type="jpeg", quality=93)
            if k % 60 == 0:
                print(f"{k}/{len(times)} t={t:.2f}s  {time.time() - t0:.0f}s", flush=True)
        b.close()
    print("done", len(times), "frames ->", frames_dir)


if __name__ == "__main__":
    main()
