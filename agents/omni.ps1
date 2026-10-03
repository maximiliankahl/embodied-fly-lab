# Omnigent launcher for the Embodied Fly Lab (Windows PowerShell 5.1).
#
# Sets project-local Omnigent state + telemetry opt-out, loads non-empty keys
# from 02_App\.env into this process (values are never printed), then runs
# `uv run omnigent <args>` from the 02_App folder (needed so flylab.tools imports).
#
# Examples (from 02_App):
#   .\agents\omni.ps1 setup                                   # one-time provider setup (Max, interactively)
#   .\agents\omni.ps1 run agents/fly_lab.yaml -p "Which descending neurons drive backward walking?"
#   .\agents\omni.ps1 -Mock run agents/fly_lab.yaml -p "..." # science modules mocked (FLYLAB_MOCK=1)
#   .\agents\omni.ps1 server --agent agents/fly_lab.yaml     # web UI on http://localhost:6767
#   .\agents\omni.ps1 session export --id conv_xxx --output runs\<run_id>\omnigent_transcript.jsonl
# NOTE: deliberately NOT an advanced script (no [Parameter()]), so omnigent's own
# short flags like -p are passed through in $args instead of being parsed by PowerShell.
param(
    [switch]$Mock,
    [switch]$EnvModel
)
$OmniArgs = @($args)

$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $PSScriptRoot                 # ...\02_App
$state = Join-Path (Split-Path -Parent $appRoot) ".omnigent" # ...\HackNation-7\.omnigent (outside the git repo)
New-Item -ItemType Directory -Force $state | Out-Null

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

# Mock mode. Omnigent's host passes only an env allowlist to the runner that executes
# flylab.tools, so the setting is ALSO written to a (gitignored) flag file that tools.py reads.
$mockFlag = Join-Path $appRoot "data\cache\FLYLAB_MOCK"
if ($Mock) {
    $env:FLYLAB_MOCK = "1"
    New-Item -ItemType Directory -Force (Split-Path -Parent $mockFlag) | Out-Null
    Set-Content -Path $mockFlag -Value "1" -Encoding ascii
    Write-Host "FLYLAB_MOCK=1 (science modules mocked; flag file data\cache\FLYLAB_MOCK)"
} elseif ($OmniArgs.Count -gt 0 -and @("run", "server", "start") -contains $OmniArgs[0]) {
    if (Test-Path $mockFlag) { Remove-Item $mockFlag -Force; Write-Host "mock flag removed (real science modules)" }
}

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
# -EnvModel: use ANTHROPIC_MODEL from .env for `run` (model names live in .env, not in code).
if ($EnvModel -and $OmniArgs[0] -eq "run" -and $env:ANTHROPIC_MODEL) {
    $OmniArgs = $OmniArgs + @("--model", $env:ANTHROPIC_MODEL)
    Write-Host "using --model from ANTHROPIC_MODEL"
}
$code = 1  # stays 1 if uv itself cannot be started
try {
    & uv run omnigent @OmniArgs
    $code = $LASTEXITCODE
} finally {
    # Mock mode only lasts for this command (the flag file must not leak into later real runs).
    if ($Mock -and (Test-Path $mockFlag)) { Remove-Item $mockFlag -Force -ErrorAction SilentlyContinue }
}
exit $code
