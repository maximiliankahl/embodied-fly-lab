"""Compare controller rates (control_every) on the 5 demo drives. Run from 02_App:
    uv run python spikes/body/control_rate_test.py 1 2 4
"""
import sys, time
sys.path.insert(0, ".")
from flylab import body

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]] or [1, 4]:
        t = time.perf_counter()
        res = body.simulate_many(body.DEMO_CONDITIONS, duration_s=1.0, parallel=True,
                                 control_every=k)
        ok = sum(r["behavior"] == n for n, r in res.items())
        print(f"control_every={k}: {ok}/5 ok, total {time.perf_counter() - t:.1f}s")
        for n, r in res.items():
            print(f"  {n:<11} fwd {r['forward_disp_mm']:7.2f} lat {r['lateral_disp_mm']:6.2f} "
                  f"dH {r['heading_change_deg']:7.1f} -> {r['behavior']:<10} "
                  f"wall/sim {r['wall_per_sim_s']:.1f} upright {r['upright_min_cos']}")
