from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=d3d11", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 960, "height": 540})
    pg.goto("http://localhost:8777/film.html?capture")
    pg.wait_for_function("window.__film && (window.__film.ready || window.__film.error)", timeout=120000)
    for t in (0.5, 38.0, 39.6, 40.5, 41.5):
        print(t, pg.evaluate("""t => { window.__film.renderAt(t); const d = window.__dbg; const bad = [];
          const chk = (n, arr) => { for (const v of arr) if (!Number.isFinite(v)) { bad.push(n); break; } };
          chk('brainPos', d.brain.group.position.toArray()); chk('brainQ', d.brain.group.quaternion.toArray()); chk('brainS', d.brain.group.scale.toArray());
          for (const [n, g] of d.S.fb.bodies) { chk(n, g.position.toArray()); chk(n + 'q', g.quaternion.toArray()); }
          chk('sugarS', d.S.sugar.scale.toArray()); chk('cam', d.camera.position.toArray());
          const th = d.S.fb.bodies.get('thorax').quaternion, hd = d.S.fb.bodies.get('head').quaternion;
          return {bad: bad.slice(0, 8), thQ: th.toArray().map(v => +v.toFixed(3)), headQ: hd.toArray().map(v => +v.toFixed(3))}; }""", t))
    b.close()
