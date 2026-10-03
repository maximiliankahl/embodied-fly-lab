"""Print the active (or given) research record compactly: uv run python spikes/omni/show_record.py [run_id]"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
rid = sys.argv[1] if len(sys.argv) > 1 else (root / "runs" / "CURRENT").read_text(encoding="utf-8").strip()
for line in (root / "runs" / rid / "record.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue
    print(e.get("seq"), str(e.get("ts", ""))[11:19], e.get("agent"), e.get("type"),
          str(e.get("content", ""))[:160].replace("\n", " "))
