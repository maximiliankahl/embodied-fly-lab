# Omnigent launcher for the Embodied Fly Lab (Windows PowerShell 5.1).
#
# Sets project-local Omnigent state + telemetry opt-out, loads non-empty keys
# from 02_App\.env into this process (values are never printed), starts the
# local workspace-header proxy (agents\anthropic_ws_proxy.py) if
# ANTHROPIC_WORKSPACE_ID is set, then runs `uv run omnigent <args>` from the
# 02_App folder (needed so flylab.tools imports).
#
# Examples (from 02_App):
#   .\agents\omni.ps1 server --agent agents/fly_lab.yaml     # web UI on http://localhost:6767 (approve gates in the browser)
#   .\agents\omni.ps1 run agents/fly_lab.yaml                # interactive REPL (approvals as prompts)
#   .\agents\omni.ps1 -ApproveAtLaunch run agents/fly_lab.yaml -p "..."   # scripted/headless: human pre-approves at launch
#   .\agents\omni.ps1 -Mock run agents/fly_lab.yaml -p "..."         # science modules mocked (FLYLAB_MOCK=1)
#   .\agents\omni.ps1 -MockOnly screen run ...                       # mock only some components (FLYLAB_MOCK=screen)
#   .\agents\omni.ps1 session export conv_xxx --output runs\<run_id>\omnigent_transcript.json
#   .\agents\omni.ps1 stop                                   # stop Omnigent server/daemon AND the proxy
# NOTE: deliberately NOT an advanced script (no [Parameter()]), so omnigent's own
# short flags like -p are passed through in $args instead of being parsed by PowerShell.
# Only [switch] params here: a [string] param would be POSITIONAL and swallow the omnigent command
# (e.g. "run"). -MockOnly <components> is therefore parsed from $args by hand.
param(
    [switch]$Mock,
    [switch]$EnvModel,
    [switch]$ApproveAtLaunch
)
$OmniArgs = @()
$MockOnly = ""
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "-MockOnly" -and ($i + 1) -lt $args.Count) { $MockOnly = [string]$args[$i + 1]; $i++ }
    else { $OmniArgs += $args[$i] }
}

$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $PSScriptRoot                 # ...\02_App
$state = Join-Path (Split-Path -Parent $appRoot) ".omnigent" # ...\HackNation-7\.omnigent (outside the git repo)
New-Item -ItemType Directory -Force $state | Out-Null
New-Item -ItemType Directory -Force (Join-Path $state "logs") | Out-Null

# Fresh PATH (uv may be missing in shells started before install).
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')

# Omnigent state (config, credentials store, chat.db sessions, logs) lives OUTSIDE the repo.
$env:OMNIGENT_CONFIG_HOME = $state
$env:OMNIGENT_DATA_DIR = $state
# Telemetry opt-out (any one is enough; see omnigent/telemetry/client.py).
$env:DO_NOT_TRACK = "1"
$env:OMNIGENT_ANALYTICS = "0"
# UTF-8 everywhere: without this the Omnigent host daemon dies on Windows with
# "'charmap' codec can't encode character" and runners report "host is offline".
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Load non-empty KEY=VALUE pairs from .env (never echo values).
$envFile = Join-Path $appRoot ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            $k = $Matches[1]; $v = $Matches[2].Trim().Trim('"').Trim("'")
            if ($v -ne "" -and -not [Environment]::GetEnvironmentVariable($k, 'Process')) {
                Set-Item -Path "Env:$k" -Value $v
                Write-Host "loaded $k from .env"
            }
        }
    }
}

# ---- Anthropic workspace header ------------------------------------------------------------
# Max's key is not workspace-scoped: every request needs "anthropic-workspace-id". Omnigent strips
# ANTHROPIC_CUSTOM_HEADERS on the CLI->daemon->runner hops (env allowlists), but forwards
# ANTHROPIC_BASE_URL. So claude.exe talks to a local proxy that adds the header (see the proxy's docstring).
$proxyPort = if ($env:FLYLAB_WS_PROXY_PORT) { [int]$env:FLYLAB_WS_PROXY_PORT } else { 8788 }
$proxyUrl = "http://127.0.0.1:$proxyPort"
function Test-WsProxy {
    try { return ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "$proxyUrl/__flylab_proxy_health").Content -eq "ok") }
    catch { return $false }
}
function Stop-WsProxy {
    # uv -> venv launcher -> python: stop the whole chain (matched by script name), then anything left on the port.
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "anthropic_ws_proxy\.py" } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "workspace-header proxy process stopped (pid $($_.ProcessId))"
    }
    Get-NetTCPConnection -LocalPort $proxyPort -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -Confirm:$false -ErrorAction SilentlyContinue
    }
}
$isStop = ($OmniArgs.Count -gt 0 -and $OmniArgs[0] -eq "stop")
if ($env:ANTHROPIC_WORKSPACE_ID -and -not $isStop) {
    if (-not (Test-WsProxy)) {
        $plog = Join-Path $state "logs\ws_proxy.log"
        # No -Redirect* here: redirection makes the child inherit this shell's handles, and a caller that
        # captures omni.ps1's output would then wait until the proxy exits. The proxy writes its own log.
        Start-Process -FilePath "uv" -ArgumentList @("run", "python", "agents/anthropic_ws_proxy.py", "--port", "$proxyPort", "--log", "`"$plog`"") `
            -WorkingDirectory $appRoot -WindowStyle Hidden | Out-Null
        $ok = $false
        foreach ($i in 1..30) { Start-Sleep -Milliseconds 500; if (Test-WsProxy) { $ok = $true; break } }
        if ($ok) { Write-Host "workspace-header proxy started on $proxyUrl (log: .omnigent\logs\ws_proxy.log)" }
        else { Write-Host "WARNING: workspace-header proxy did not start - see $plog" }
    } else { Write-Host "workspace-header proxy already running on $proxyUrl" }
    $env:ANTHROPIC_BASE_URL = $proxyUrl
    # If an Omnigent daemon was started earlier WITHOUT this base URL, its runners call the API directly
    # and fail with "400 ... anthropic-workspace-id": run `.\agents\omni.ps1 stop` once, then retry.
}

# Mock mode. Omnigent's host passes only an env allowlist to the runner that executes
# flylab.tools, so the setting is ALSO written to a (gitignored) flag file that tools.py reads.
$mockFlag = Join-Path $appRoot "data\cache\FLYLAB_MOCK"
$mockValue = if ($Mock) { "1" } elseif ($MockOnly) { $MockOnly } else { "" }
if ($mockValue) {
    $env:FLYLAB_MOCK = $mockValue
    New-Item -ItemType Directory -Force (Split-Path -Parent $mockFlag) | Out-Null
    Set-Content -Path $mockFlag -Value $mockValue -Encoding ascii
    Write-Host "FLYLAB_MOCK=$mockValue (mocked components return data labelled MOCK; flag file data\cache\FLYLAB_MOCK)"
} elseif ($OmniArgs.Count -gt 0 -and @("run", "server", "start") -contains $OmniArgs[0]) {
    if (Test-Path $mockFlag) { Remove-Item $mockFlag -Force; Write-Host "mock flag removed (real science modules)" }
}

# Human pre-approval for scripted / headless runs (headless `run -p` declines every ASK).
# Running this command with -ApproveAtLaunch IS Max's approval of the lab's gated actions for this run;
# hard caps (6 embodied runs, <= 40 screen candidates per call, cost_budget, tool-call limit) still apply.
$preFlag = Join-Path $appRoot "data\cache\FLYLAB_PREAPPROVE"
if ($ApproveAtLaunch) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $preFlag) | Out-Null
    $who = "$env:USERNAME (human) at launch via agents/omni.ps1 -ApproveAtLaunch"
    $json = '{"by": "' + $who + '", "ts": "' + (Get-Date -Format s) + '", "caps": "max 6 embodied runs, <= 40 screen candidates per call, cost_budget, 80 tool calls per session"}'
    Set-Content -Path $preFlag -Value $json -Encoding ascii
    Write-Host "PRE-APPROVED: gated lab actions run without prompts for this command (caps still enforced)"
} elseif (Test-Path $preFlag) { Remove-Item $preFlag -Force; Write-Host "stale pre-approval flag removed" }

# The claude-sdk harness needs a native claude.exe (the pip wheel here bundles none and
# npm's claude.cmd shim is refused on Windows). If none is on PATH, reuse the claude.exe
# that ships with the Claude desktop app (newest version) - nothing gets installed.
if (-not $env:OMNIGENT_CLAUDE_PATH -and -not (Get-Command claude.exe -ErrorAction SilentlyContinue)) {
    $cand = Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Packages\Claude_*\LocalCache\Roaming\Claude\claude-code\*\*\claude.exe") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    # PATH (not OMNIGENT_CLAUDE_PATH) because only allowlisted env vars such as PATH reach the runner.
    if ($cand) { $env:Path = $cand.DirectoryName + ";" + $env:Path; Write-Host "claude.exe from Claude desktop app added to PATH" }
    else { Write-Host "WARNING: no claude.exe found; claude-sdk harness will fail (install Claude Code native or add claude.exe to PATH)." }
}

# Omnigent hands an API key to claude.exe via an apiKeyHelper "printf %s <key>" - a POSIX
# command. On Windows it only works if a printf.exe is on PATH: append Git for Windows' usr\bin
# (appended, so Windows tools keep precedence).
if (-not (Get-Command printf -ErrorAction SilentlyContinue)) {
    $gitUsrBin = "C:\Program Files\Git\usr\bin"
    if (Test-Path (Join-Path $gitUsrBin "printf.exe")) { $env:Path = $env:Path + ";" + $gitUsrBin; Write-Host "Git usr\bin appended to PATH (printf for apiKeyHelper)" }
    else { Write-Host "WARNING: printf.exe not found - API-key auth for claude-sdk may fail (Claude subscription login still works)." }
}

Set-Location $appRoot
if (-not $OmniArgs -or $OmniArgs.Count -eq 0) { $OmniArgs = @("--help") }
# -EnvModel: override the SUPERVISOR model with ANTHROPIC_MODEL from .env for `run`
# (sub-agent models are pinned per agent in agents/fly_lab.yaml).
if ($EnvModel -and $OmniArgs[0] -eq "run" -and $env:ANTHROPIC_MODEL) {
    $OmniArgs = $OmniArgs + @("--model", $env:ANTHROPIC_MODEL)
    Write-Host "using --model from ANTHROPIC_MODEL"
}
$code = 1  # stays 1 if uv itself cannot be started
try {
    & uv run omnigent @OmniArgs
    $code = $LASTEXITCODE
} finally {
    # Mock mode and pre-approval only last for this command (flag files must not leak into later runs).
    if ($mockValue -and (Test-Path $mockFlag)) { Remove-Item $mockFlag -Force -ErrorAction SilentlyContinue }
    if ($ApproveAtLaunch -and (Test-Path $preFlag)) { Remove-Item $preFlag -Force -ErrorAction SilentlyContinue; Write-Host "pre-approval flag removed" }
    if ($isStop) { Stop-WsProxy }
}
exit $code
