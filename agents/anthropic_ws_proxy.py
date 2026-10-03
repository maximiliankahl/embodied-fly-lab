"""Tiny local reverse proxy that adds the ``anthropic-workspace-id`` header.

Why: Max's Anthropic API key is not scoped to a workspace, so every API request needs the
header ``anthropic-workspace-id: <ANTHROPIC_WORKSPACE_ID>``. Omnigent's claude-sdk harness runs
``claude.exe`` in a runner process that only receives an env ALLOWLIST (omnigent/host/connect.py
``_RUNNER_ENV_ALLOWLIST`` + ``HARNESS_CREDENTIAL_ENV_VARS``; the CLI->daemon hop in omnigent/cli.py
``_build_host_daemon_env``). ``ANTHROPIC_CUSTOM_HEADERS`` is on neither list (and
``OMNIGENT_RUNNER_ENV_PASSTHROUGH`` only acts on the daemon->runner hop, after the CLI->daemon hop
already dropped it), but ``ANTHROPIC_BASE_URL`` is forwarded on both hops. So agents/omni.ps1 sets
``ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`` and starts this proxy, which forwards every request
unchanged to https://api.anthropic.com and adds the workspace header. The API key itself still
comes from the client (claude.exe apiKeyHelper / x-api-key); the proxy never logs headers or bodies.

Run:  uv run python agents/anthropic_ws_proxy.py [--port 8788]
Health check: GET http://127.0.0.1:8788/__flylab_proxy_health -> 200 "ok"
Binds to 127.0.0.1 only. Stdlib + httpx (already a dependency of anthropic/omnigent).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

UPSTREAM = os.environ.get("FLYLAB_PROXY_UPSTREAM", "https://api.anthropic.com").rstrip("/")
HEALTH_PATH = "/__flylab_proxy_health"
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "proxy-connection",
       "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers", "upgrade"}


def _workspace_id() -> str:
    """ANTHROPIC_WORKSPACE_ID from the environment, else from 02_App/.env (value never printed)."""
    v = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    if v:
        return v
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_WORKSPACE_ID") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


WORKSPACE_ID = _workspace_id()
CLIENT = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=30.0),
                      follow_redirects=False, http2=False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "flylab-ws-proxy/1.0"

    def log_message(self, fmt: str, *args) -> None:  # one short line per request, no headers/bodies
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}\n")

    def _read_body(self) -> bytes:
        n = self.headers.get("Content-Length")
        if n:
            return self.rfile.read(int(n))
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline().strip().split(b";")[0] or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
        return b""

    def _proxy(self) -> None:
        if self.path == HEALTH_PATH:
            body = b"ok" if WORKSPACE_ID else b"missing ANTHROPIC_WORKSPACE_ID"
            self.send_response(200 if WORKSPACE_ID else 500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = self._read_body()
        headers = [(k, v) for k, v in self.headers.items() if k.lower() not in HOP]
        if WORKSPACE_ID and not any(k.lower() == "anthropic-workspace-id" for k, _ in headers):
            headers.append(("anthropic-workspace-id", WORKSPACE_ID))
        req = CLIENT.build_request(self.command, UPSTREAM + self.path, headers=headers, content=body or None)
        try:
            resp = CLIENT.send(req, stream=True)
        except httpx.HTTPError as exc:
            msg = f'{{"type":"error","error":{{"type":"proxy_error","message":"{type(exc).__name__}"}}}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        try:
            self.send_response(resp.status_code)
            no_body = self.command == "HEAD" or resp.status_code in (204, 304) or resp.status_code < 200
            for k, v in resp.headers.multi_items():
                if k.lower() not in HOP or (no_body and k.lower() == "content-length"):
                    self.send_header(k, v)
            if no_body:
                # HEAD / 204 / 304 carry no body: a chunked terminator here would be read by the client as
                # the next status line ("illegal status line: b'0'" in Omnigent's gateway shim).
                self.end_headers()
                self.wfile.flush()
                return
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in resp.iter_raw():  # raw bytes: keeps content-encoding and SSE framing intact
                if chunk:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        finally:
            resp.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _proxy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("FLYLAB_WS_PROXY_PORT", "8788")))
    ap.add_argument("--log", default="", help="append the one-line request log to this file instead of stderr")
    a = ap.parse_args()
    if a.log:  # omni.ps1 starts the proxy without handle redirection (so callers never block on inherited pipes)
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        sys.stderr = open(a.log, "a", encoding="utf-8", buffering=1)
    if not WORKSPACE_ID:
        print("ERROR: ANTHROPIC_WORKSPACE_ID missing (env or 02_App/.env)", file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"flylab workspace-header proxy on http://127.0.0.1:{a.port} -> {UPSTREAM} (workspace id loaded)",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
