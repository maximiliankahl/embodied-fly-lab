"""Validate agents/fly_lab.yaml with Omnigent's own loader + validator.

Run from 02_App:  uv run python spikes/omni/validate_yaml.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))  # omnigent resolves callables from cwd

from omnigent.spec import load  # noqa: E402
from omnigent.spec.validator import validate  # noqa: E402
from omnigent.spec.omnigent import agent_spec_to_agent_def  # noqa: E402

path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "agents/fly_lab.yaml")
spec = load(path)
res = validate(spec)
print("valid:", res.valid, [f"{e.path}: {e.message}" for e in res.errors])
print("root:", spec.name, "| harness:", spec.executor.config.get("harness"), "| model:", spec.executor.model)
print("root tools:", [t.name for t in spec.local_tools])
print("sub-agents:", spec.tools.agents)
for sa in spec.sub_agents:
    print(f"  - {sa.name:14s} harness={sa.executor.config.get('harness')} tools={[t.name for t in sa.local_tools]}")
g = spec.guardrails
print("ask_timeout:", g.ask_timeout if g else None)
for p in (g.policies if g else []):
    fn = getattr(p, "function", None)
    print(f"  policy {p.name:18s} on={[s.phase.value for s in (p.on or [])]} -> {fn.path if fn else None} {fn.arguments if fn else ''}")

# Build the policy callables exactly like the engine would (factory + arguments).
import importlib  # noqa: E402
for p in (g.policies if g else []):
    fn = p.function
    mod, _, attr = fn.path.rpartition(".")
    obj = getattr(importlib.import_module(mod), attr)
    cb = obj(**(fn.arguments or {}))
    assert callable(cb), p.name
print("policy factories build: ok")

# Forward translation that the executor performs (needs a model; use a placeholder id).
spec.executor.model = spec.executor.model or "placeholder-model"
adef = agent_spec_to_agent_def(spec)
print("agent_def tools:", sorted(adef.tools))
from omnigent.inner.tools import FunctionTool  # noqa: E402
ft = adef.tools["start_run"]
assert isinstance(ft, FunctionTool) and ft.callable is not None
print("start_run schema:", ft.tool_schema())

# Runtime path: Omnigent wraps local callables as LocalCallableTool (description = docstring lead paragraph).
from omnigent.tools.local_callable import load_local_callable_tools  # noqa: E402
spec2 = load(path)
for sa in [spec2] + list(spec2.sub_agents):
    tools = load_local_callable_tools(sa.local_tools)
    for t in tools:
        sch = t.get_schema()["function"]
        assert sch["parameters"].get("properties"), (sa.name, sch["name"])
        assert sch["description"], (sa.name, sch["name"])
    print(f"runtime schemas ok: {sa.name:14s} {len(tools)} tools")
print("example:", load_local_callable_tools(spec2.sub_agents[4].local_tools)[2].get_schema()["function"]["parameters"])
