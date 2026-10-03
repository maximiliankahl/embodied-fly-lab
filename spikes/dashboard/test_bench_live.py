"""Live bench test: MDN activation, short run, through the real page (AppTest)."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
def run_page():
    from flylab import ui_bench
    ui_bench.page()
t0 = time.time()
at = AppTest.from_function(run_page, default_timeout=400).run()
at.select_slider[0].set_value(250)   # brain ms
at.slider[1].set_value(1)            # trials
at.select_slider[1].set_value(1.0)   # body s
at.run()
at.button[0].click().run()
print("exc", [str(e.value)[:1500] for e in at.exception])
print("errors", [e.value[:500] for e in at.error], "warn", [w.value[:200] for w in at.warning])
print("metrics", [(m.label, m.value) for m in at.metric])
print("markdown", [m.value[:200] for m in at.markdown])
print("captions", [c.value[:200] for c in at.caption])
r = at.session_state["bench_result"] if "bench_result" in at.session_state else None
print("result keys", list(r) if r else None, "video", (r or {}).get("video"), "wall", (r or {}).get("wall_s"))
print(f"total {time.time()-t0:.1f}s")
