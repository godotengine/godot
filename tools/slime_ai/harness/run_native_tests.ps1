param(
    [string]$Filter = '*[SlimeAI]*',
    [string]$EvidenceDir = 'docs/slime_ai/evidence'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$binary = Join-Path $repo 'bin/godot.windows.editor.dev.x86_64.console.exe'
if (!(Test-Path -LiteralPath $binary)) { throw "Editor binary is missing: $binary" }
$evidence = Join-Path $repo $EvidenceDir
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
$log = Join-Path $evidence 'native-tests.log'
$manifest = Join-Path $evidence 'native-tests-manifest.txt'

Push-Location $repo
try {
    $revision = (& git rev-parse HEAD).Trim()
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash
    @("revision=$revision", "binary=$binary", "binary_sha256=$hash", "filter=$Filter", "command=$binary --test --test-case=$Filter --no-colors") | Set-Content -LiteralPath $manifest
    # Capture stderr as data: the malformed-frame regression intentionally emits a diagnostic.
    $stdout = Join-Path $evidence 'native-tests-stdout.tmp'
    $stderr = Join-Path $evidence 'native-tests-stderr.tmp'
    $process = Start-Process -FilePath $binary -ArgumentList @('--test', "--test-case=$Filter", '--no-colors') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $code = $process.ExitCode
    $output = @(Get-Content -LiteralPath $stdout) + @(Get-Content -LiteralPath $stderr)
    $output | Set-Content -LiteralPath $log
    $output | Write-Output
    Remove-Item -LiteralPath $stdout, $stderr
    $summary = Get-Content -LiteralPath $log | Where-Object { $_ -match '^\[doctest\] test cases:' } | Select-Object -Last 1
    $count = 0
    if ($summary -match 'test cases:\s*(\d+)') { $count = [int]$Matches[1] }
    Add-Content -LiteralPath $manifest -Value @("exit_code=$code", "executed_cases=$count", "summary=$summary")
    if ($code -ne 0) { exit $code }
    if ($count -lt 1) { [Console]::Error.WriteLine("No native test cases matched '$Filter'"); exit 3 }
    exit 0
}
finally {
    Pop-Location
}
