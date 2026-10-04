from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=d3d11", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 960, "height": 540})
    pg.goto("http://localhost:8777/film.html?capture")
    pg.wait_for_function("window.__film && (window.__film.ready || window.__film.error)", timeout=120000)
    for t in (38.0, 40.5, 42.0, 44.5):
        print(t, pg.evaluate("""t => { window.__film.renderAt(t); const d = window.__dbg; const c = d.camera.position;
          const s = d.S.sugar.position; const th = d.S.fb.bodies.get('thorax');
          return {cam: [c.x, c.y, c.z].map(v => +v.toFixed(2)), sugar: [s.x, s.y, s.z].map(v => +v.toFixed(2)),
                  thoraxLocal: [th.position.x, th.position.y, th.position.z].map(v => +v.toFixed(3)), flyVisible: d.S.fb.root.visible,
                  fade: document.getElementById('fade').style.opacity}; }""", t))
    b.close()
