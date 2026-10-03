"""Tiny live check of agents/anthropic_ws_proxy.py: one non-streaming + one streaming haiku call
(max_tokens 8 each, cost < $0.001). Reads the key from .env, never prints it."""
import os
import sys
from pathlib import Path

import anthropic

root = Path(__file__).resolve().parents[2]
for line in (root / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
port = sys.argv[1] if len(sys.argv) > 1 else "8788"
c = anthropic.Anthropic(base_url=f"http://127.0.0.1:{port}")  # NO workspace header here: the proxy adds it
m = c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=8, messages=[{"role": "user", "content": "Say OK"}])
print("non-stream:", m.content[0].text, m.usage.output_tokens)
with c.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=8, messages=[{"role": "user", "content": "Say OK"}]) as s:
    txt = "".join(s.text_stream)
print("stream:", txt)
