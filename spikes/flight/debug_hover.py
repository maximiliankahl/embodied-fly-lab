import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from flylab import flight
flight._DEBUG = []
cmd = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"takeoff": 1}
r = flight.simulate_flight(cmd, duration_s=float(sys.argv[2]) if len(sys.argv) > 2 else 0.6)
for row in flight._DEBUG[::2]:
    print(row)
print(r["behavior"], r["metrics_flight_phase"])
