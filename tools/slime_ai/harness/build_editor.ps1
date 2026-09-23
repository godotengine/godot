param(
    [int]$Jobs = 8,
    [string]$EvidenceDir = 'docs/slime_ai/evidence'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$evidence = Join-Path $repo $EvidenceDir
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
$run = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss-fff')
$log = Join-Path $evidence "editor-build-$run.log"
$manifest = Join-Path $evidence "editor-build-$run-manifest.txt"
$pythonCode = "import misc; misc.__path__.insert(0, r'$($repo.Replace("'", "\'"))\misc'); import SCons.Script; SCons.Script.main()"

Push-Location $repo
try {
    $revision = (& git rev-parse HEAD).Trim()
    $dirty = & git status --short
    $command = "python -c <process-local misc import correction> platform=windows target=editor arch=x86_64 tests=yes dev_build=yes -j$Jobs"
    @("revision=$revision", "command=$command", "started_utc=$((Get-Date).ToUniversalTime().ToString('o'))", 'dirty_paths:') + $dirty | Set-Content -LiteralPath $manifest
    & python -c $pythonCode platform=windows target=editor arch=x86_64 tests=yes dev_build=yes "-j$Jobs" 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    Add-Content -LiteralPath $manifest -Value "exit_code=$code"
    if ($code -eq 0) {
        $binary = Join-Path $repo 'bin/godot.windows.editor.dev.x86_64.console.exe'
        if (!(Test-Path -LiteralPath $binary)) { throw "Build returned 0 but binary is missing: $binary" }
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash
        Add-Content -LiteralPath $manifest -Value @("binary=$binary", "binary_sha256=$hash", "finished_utc=$((Get-Date).ToUniversalTime().ToString('o'))")
    }
    exit $code
}
finally {
    Pop-Location
}
