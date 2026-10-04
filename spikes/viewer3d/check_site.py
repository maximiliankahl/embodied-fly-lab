"""Smoke test for the static viewer: every file the page references must be served (no 404), and the total size is reported.

    uv run python -m http.server 8777 --directory web     (in another shell)
    uv run python spikes/viewer3d/check_site.py [base_url]
"""
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8777").rstrip("/")
WEB = Path(__file__).resolve().parents[2] / "web"
bad = []
seen: dict[str, bytes | None] = {}


def get(path: str) -> bytes | None:
    if path in seen:
        return seen[path]
    seen[path] = None
    url = f"{BASE}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read()
        print(f"200 {path} ({len(body):,} B)")
        seen[path] = body
        return body
    except urllib.error.HTTPError as e:
        print(f"{e.code} {path}  <-- MISSING")
        bad.append(path)
    except Exception as e:  # noqa: BLE001
        print(f"ERR {path}: {e}")
        bad.append(path)
    return None


html = get("index.html").decode("utf-8")
for ref in re.findall(r'(?:src|href)="([^"#:]+?)"', html):
    get(ref)
for js in sorted((WEB / "js").glob("*.js")):
    get(f"js/{js.name}")
manifest = json.loads(get("data/manifest.json"))
for f in manifest["files"]:
    get("data/" + f["path"].removeprefix("data/"))
idx = json.loads(get("data/runs/index.json"))
for r in idx["runs"]:
    get(r["file"])
    run = json.loads((WEB / r["file"]).read_text(encoding="utf-8"))
    get(run["body"]["geometry"])
    # geometry .bin is referenced from the geometry json
    g = json.loads((WEB / run["body"]["geometry"]).read_text(encoding="utf-8"))
    get(run["body"]["geometry"].rsplit("/", 1)[0] + "/" + g["bin"])
    assert run["body"].get("poses"), f"{r['id']}: no poses"
    assert run["brain"]["active_idx"], f"{r['id']}: no active neurons"
meta = json.loads((WEB / "data/brain_points.json").read_text(encoding="utf-8"))
get("data/" + meta["bin"])

total = sum(p.stat().st_size for p in WEB.rglob("*") if p.is_file())
print(f"\nruns: {[r['id'] for r in idx['runs']]}")
print(f"web/ total: {total / 1e6:.2f} MB in {sum(1 for p in WEB.rglob('*') if p.is_file())} files (limit 40 MB)")
print("FAILED: " + ", ".join(bad) if bad else "ALL REFERENCED FILES SERVED (no 404)")
sys.exit(1 if bad else 0)
