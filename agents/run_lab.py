"""Scripted (headless) run of the Omnigent fly lab that follows the WHOLE multi-agent session.

    powershell -ExecutionPolicy Bypass -File .\\agents\\omni.ps1 -ApproveAtLaunch lab "<question>"
    (omni.ps1 sets the env: Omnigent state dir, workspace-header proxy, claude.exe, pre-approval flag)

Why this exists: `omnigent run <yaml> -p "..."` (Omnigent 0.16 headless mode) decides that an async
orchestrator is finished when the session snapshot reads "idle" right after the supervisor's first
turn. Our PI ends its turn while a specialist works (it is woken by the inbox), so the CLI exited
after ~60 s and then stopped the session (observed 2026-10-04 00:40). This wrapper runs the SAME
Omnigent CLI command in-process and only changes the follow logic: the session counts as running
while any sub-agent in its tree is busy (SessionsNamespace.subtree_busy) or a just-finished
sub-agent's wake-up turn has not started yet (re-checked after a grace period). Everything else
(server, daemon, runner, policies, approvals) is stock Omnigent.

After the run it exports the transcript of the supervisor and every sub-agent session to
runs/<run_id>/omnigent/ (agents/export_transcript.py) and prints the record summary.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRACE_S = 15.0          # wait this long before believing "idle" (wake-up turn may be about to start)
PER_TURN_TIMEOUT_S = 300.0
LOOP_TIMEOUT_S = 3600.0
_state: dict = {"session_id": None, "t0": time.time()}


def _patch() -> None:
    import omnigent.chat as oc
    from omnigent_client import SessionsChat

    oc._PER_TURN_TIMEOUT_S = PER_TURN_TIMEOUT_S
    oc._LOOP_TIMEOUT_S = LOOP_TIMEOUT_S
    orig_refresh = SessionsChat.refresh

    async def _busy(chat) -> bool:
        try:
            return bool(await chat.tree_busy(max_depth=3))
        except Exception:
            return False

    async def refresh(self):  # type: ignore[no-untyped-def]
        _state["session_id"] = self.session_id
        snap = await orig_refresh(self)
        if snap.status in ("running", "launching"):
            return snap
        if await _busy(self):
            self._session = dataclasses.replace(snap, status="running")
            return self._session
        await asyncio.sleep(GRACE_S)  # a finished child's wake-up turn may not have started yet
        snap = await orig_refresh(self)
        if snap.status not in ("running", "launching") and await _busy(self):
            self._session = dataclasses.replace(snap, status="running")
        elapsed = int(time.time() - _state["t0"])
        print(f"[run_lab] {elapsed:5d}s session {self.session_id[:8]} status={self._session.status}", flush=True)
        return self._session

    SessionsChat.refresh = refresh


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    question = " ".join(argv)
    _patch()
    from omnigent.cli import main as omni_main

    sys.argv = ["omnigent", "run", "agents/fly_lab.yaml", "-p", question]
    code = 0
    try:
        omni_main()
    except SystemExit as exc:  # click exits
        code = int(exc.code or 0) if not isinstance(exc.code, str) else 1
    sid = _state["session_id"]
    print(f"[run_lab] omnigent finished (exit {code}) after {int(time.time() - _state['t0'])} s; session {sid}")
    if sid:
        try:
            from agents import export_transcript  # type: ignore
        except ImportError:
            sys.path.insert(0, str(ROOT / "agents"))
            import export_transcript  # type: ignore
        try:
            export_transcript.main([sid])
        except Exception as exc:  # export is best-effort; the record is already on disk
            print(f"[run_lab] transcript export failed: {type(exc).__name__}: {exc}")
    try:
        from flylab import record
        rid = record.current_run_id()
        if rid:
            print("[run_lab] record:", record.summarize(rid))
    except Exception:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
