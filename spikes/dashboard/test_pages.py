"""Headless test of every dashboard page with streamlit.testing.v1.AppTest.
Usage (from 02_App): uv run python spikes/dashboard/test_pages.py [page ...]"""
import os
import sys
import time

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def run_page(name: str):
    import importlib

    from flylab import ui

    ui.inject_css()
    importlib.import_module(f"flylab.ui_{name}").page()


def check(at, label):
    errs = [e.value for e in at.exception]
    print(f"{label}: exceptions={len(errs)} markdown={len(at.markdown)} errors={len(at.error)} "
          f"warnings={len(at.warning)} info={len(at.info)}")
    for e in errs:
        print("   EXC:", str(e)[:2000])
    for e in at.error:
        print("   ERR:", str(e.value)[:300])
    return not errs


ok = True
pages = sys.argv[1:] or ["notebook", "bench", "validation", "method"]
t0 = time.time()
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
ok &= check(at, "app.py (default page)")
for p in pages:
    at = AppTest.from_function(run_page, args=(p,), default_timeout=120).run()
    ok &= check(at, f"page {p}")
    if p == "notebook" and len(at.selectbox):
        for opt in at.selectbox[0].options:
            at.selectbox[0].select(opt).run()
            ok &= check(at, f"   notebook run={opt}")
# replay mode: bench must render with disabled controls and an explanation
os.environ["FLYLAB_REPLAY"] = "1"
at = AppTest.from_function(run_page, args=("bench",), default_timeout=60).run()
ok &= check(at, "page bench (FLYLAB_REPLAY=1)")
dis = [b.disabled for b in at.button]
print("   replay: run button disabled =", dis, "| info:", [i.value[:60] for i in at.info])
ok &= bool(dis) and all(dis) and len(at.info) >= 1
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
ok &= check(at, "app.py (FLYLAB_REPLAY=1)")
os.environ.pop("FLYLAB_REPLAY")
print("ALL OK" if ok else "FAILURES", f"{time.time() - t0:.1f}s")
sys.exit(0 if ok else 1)
