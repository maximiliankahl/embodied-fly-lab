"""Exercise the fly_lab.yaml policies through Omnigent's own policy plumbing.

Builds each FunctionPolicy exactly as the engine does (resolve_function_policy on the
parsed spec) and evaluates synthetic TOOL_CALL contexts. No LLM / credentials needed.
Run from 02_App:  uv run python spikes/omni/test_policies.py
"""
import asyncio
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from omnigent.spec import load  # noqa: E402
from omnigent.policies.function import resolve_function_policy  # noqa: E402
from omnigent.policies.types import EvaluationContext  # noqa: E402
from omnigent.spec.types import Phase  # noqa: E402

spec = load(ROOT / "agents/fly_lab.yaml")
pols = {p.name: resolve_function_policy(p) for p in spec.guardrails.policies}
sig = inspect.signature(EvaluationContext)


def ctx(tool, args, state=None):
    kw = dict(phase=Phase.TOOL_CALL, content={"name": tool, "arguments": args}, tool_name=tool)
    if "session_state" in sig.parameters:
        kw["session_state"] = state or {}
    for name, p in sig.parameters.items():  # fill any other required fields with neutral values
        if name not in kw and p.default is inspect.Parameter.empty:
            kw[name] = {} if "state" in name or "labels" in name else None
    return EvaluationContext(**kw)


async def main():
    gate = pols["lab_approval_gate"]
    cases = [
        ("mcp__omnigent__run_embodied_experiment", {"excite_groups": ["MDN"]}, None, "ASK"),
        ("run_embodied_experiment", {"excite_groups": ["MDN"]}, {"_flylab_embodied_runs": 6}, "DENY"),
        ("request_approval", {"action": "2 embodied runs", "reason": "tests H1", "est_cost": 5}, None, "ASK"),
        ("run_brain_experiment", {"excite_groups": ["MDN"], "duration_ms": 5000, "n_trials": 5}, None, "ASK"),
        ("run_brain_experiment", {"excite_groups": ["MDN"], "duration_ms": 1000, "n_trials": 3}, None, None),
        ("search_literature", {"query": "MDN"}, None, None),
    ]
    ok = True
    for tool, args, state, want in cases:
        res = await gate.evaluate(ctx(tool, args, state), {})
        got = getattr(res, "action", res)
        got = getattr(got, "value", got)
        good = (str(got).upper().endswith(want) if want else str(got).upper() in ("NONE", "ALLOW", "UNSPECIFIED", "POLICYACTION.UNSPECIFIED"))
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} lab_approval_gate {tool:42s} -> {got} | {getattr(res, 'reason', '')!s:.90}")
    # Embodied-run cap end to end: approve each ASK, apply its state_updates with Omnigent's own
    # engine helper (what the server does on approve), and expect DENY on the 7th call.
    from omnigent.runtime.policies.engine import _apply_one
    st = {}
    seq = []
    for _ in range(7):
        res = await gate.evaluate(ctx("run_embodied_experiment", {"excite_groups": ["MDN"]}, dict(st)), {})
        seq.append(res.action.value)
        if res.action.value == "ask":
            for u in res.state_updates or []:
                _apply_one(st, u)
    good = seq == ["ask"] * 6 + ["deny"]
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} embodied cap over 7 approved calls -> {seq} state={st}")
    limit = pols["tool_call_limit"]
    res = await limit.evaluate(ctx("search_literature", {}, {"_policy_tool_call_count": 80}), {})
    print("tool_call_limit at 80 ->", res.action.value, "|", res.reason)
    loop = pols["loop_guard"]
    st = {}
    for i in range(3):
        res = await loop.evaluate(ctx("search_literature", {"query": "same"}, st), {})
        for u in getattr(res, "state_updates", None) or []:
            st[u.key] = u.value
    print("loop_guard 3x identical ->", res.action.value)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
