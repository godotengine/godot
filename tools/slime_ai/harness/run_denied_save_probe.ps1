param(
    [string]$EvidenceDir = 'docs/slime_ai/evidence',
    [switch]$Injected
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$fixture = (Resolve-Path (Join-Path $repo 'tools/slime_ai/fixtures/2d')).Path
$binary = Join-Path $repo 'bin/godot.windows.editor.dev.x86_64.console.exe'
$evidence = Join-Path $repo $EvidenceDir
$copy = Join-Path ([IO.Path]::GetTempPath()) ('slime-ai-save-denial-' + [Guid]::NewGuid().ToString('N'))
$probe = Join-Path $copy 'probe'
New-Item -ItemType Directory -Path $copy, $probe, $evidence -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $fixture 'project.godot'), (Join-Path $fixture 'main.tscn') -Destination $copy
$scene = Join-Path $copy 'main.tscn'
$originalHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $scene).Hash
$stdout = Join-Path $probe 'editor-stdout.log'
$stderr = Join-Path $probe 'editor-stderr.log'
$names = @('SLIME_AI_TEST_SAVE_PROBE', 'SLIME_AI_SAVE_PROBE_DIR', 'SLIME_AI_SAVE_PROBE_SCENE', 'SLIME_AI_TEST_FORCE_SAVE_FAILURE')
$prior = @{}
foreach ($name in $names) { $prior[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
$process = $null
$lock = $null
function Wait-ProbeFile([string]$path, [int]$seconds) {
    $limit = [DateTime]::UtcNow.AddSeconds($seconds)
    while (!(Test-Path -LiteralPath $path)) {
        if ($process -and $process.HasExited) { throw "Editor exited before $path (exit=$($process.ExitCode))." }
        if ([DateTime]::UtcNow -gt $limit) { throw "Timed out waiting for $path." }
        Start-Sleep -Milliseconds 200
    }
}
try {
    [Environment]::SetEnvironmentVariable('SLIME_AI_TEST_SAVE_PROBE', '1', 'Process')
    [Environment]::SetEnvironmentVariable('SLIME_AI_SAVE_PROBE_DIR', $probe, 'Process')
    [Environment]::SetEnvironmentVariable('SLIME_AI_SAVE_PROBE_SCENE', 'res://main.tscn', 'Process')
    [Environment]::SetEnvironmentVariable('SLIME_AI_TEST_FORCE_SAVE_FAILURE', $(if ($Injected) { '1' } else { $null }), 'Process')
    $process = Start-Process -FilePath $binary -ArgumentList @('--headless', '--editor', '--path', $copy, 'res://main.tscn') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $prior[$name], 'Process') }
    Wait-ProbeFile (Join-Path $probe 'ready.json') 90
    $ready = Get-Content -Raw -LiteralPath (Join-Path $probe 'ready.json') | ConvertFrom-Json
    if ($ready.status -ne 'applied' -or $ready.disk_before -ne $originalHash -or !$ready.editor_unsaved) { throw 'Native preparation did not apply one unsaved edit.' }

    # The default is an actual Win32 sharing denial. The switch exercises the deterministic regression injector separately.
    if (!$Injected) { $lock = [IO.File]::Open($scene, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None) }
    [IO.File]::WriteAllText((Join-Path $probe 'deny_go.flag'), 'go')
    Wait-ProbeFile (Join-Path $probe 'denied.json') 60
    $denied = Get-Content -Raw -LiteralPath (Join-Path $probe 'denied.json') | ConvertFrom-Json
    if ($lock) { $lock.Dispose(); $lock = $null }
    $deniedDiskHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $scene).Hash
    if ($denied.status -ne 'save_failed' -or $deniedDiskHash -ne $originalHash -or !$denied.marker_present -or !$denied.editor_unsaved -or $denied.persistence -ne 'failed') { throw 'Denied save did not preserve disk and unsaved editor state.' }

    [IO.File]::WriteAllText((Join-Path $probe 'retry_go.flag'), 'go')
    Wait-ProbeFile (Join-Path $probe 'retry.json') 60
    $retry = Get-Content -Raw -LiteralPath (Join-Path $probe 'retry.json') | ConvertFrom-Json
    if ($retry.status -ne 'save_confirmed' -or !$retry.duplicate_request -or !$retry.no_second_apply -or $retry.editor_unsaved) { throw 'Retry did not save exactly one existing native effect.' }
    $savedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $scene).Hash
    if ($savedHash -eq $originalHash) { throw 'Retry did not update the scene file.' }
    $text = Get-Content -Raw -LiteralPath $scene
    if (([regex]::Matches($text, '\[node name="SaveDenialMarker"')).Count -ne 1) { throw 'Saved scene does not contain exactly one marker.' }
    $editorPid = $process.Id
    if ($process -and !$process.HasExited) { Stop-Process -Id $process.Id -Force }
    $process = $null
    $reopenOut = Join-Path $probe 'reopen-stdout.log'
    $reopenErr = Join-Path $probe 'reopen-stderr.log'
    $reopen = Start-Process -FilePath $binary -ArgumentList @('--headless', '--editor', '--path', $copy, 'res://main.tscn', '--quit-after', '2') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $reopenOut -RedirectStandardError $reopenErr -Wait -PassThru
    if ($reopen.ExitCode -ne 0) { throw "Editor reopen failed (exit=$($reopen.ExitCode))." }
    $prefix = if ($Injected) { 'p03-injected-save' } else { 'p03-denied-save' }
    Copy-Item -LiteralPath (Join-Path $probe 'ready.json') -Destination (Join-Path $evidence "$prefix-ready.json")
    Copy-Item -LiteralPath (Join-Path $probe 'denied.json') -Destination (Join-Path $evidence "$prefix-denied.json")
    Copy-Item -LiteralPath (Join-Path $probe 'retry.json') -Destination (Join-Path $evidence "$prefix-retry.json")
    Copy-Item -LiteralPath $stderr -Destination (Join-Path $evidence "$prefix-editor-stderr.txt")
    Copy-Item -LiteralPath $reopenErr -Destination (Join-Path $evidence "$prefix-reopen-stderr.txt")
    $manifest = @(
        "source_revision=$((& git -C $repo rev-parse HEAD).Trim())",
        "binary_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash)",
        "fixture_copy=$copy",
        "scene=$scene",
        "scene_before_sha256=$originalHash",
        "scene_after_denied_sha256=$deniedDiskHash",
        "scene_after_sha256=$savedHash",
        "editor_pid=$editorPid",
        "denial=$(if ($Injected) { 'deterministic test-only save error injection' } else { 'exclusive Windows FileStream share mode None on copied main.tscn' })",
        "reopen_exit_code=$($reopen.ExitCode)",
        'result=passed'
    )
    $manifest | Set-Content -LiteralPath (Join-Path $evidence "$prefix-manifest.txt")
    $manifest | Write-Output
}
finally {
    if ($lock) { $lock.Dispose() }
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $prior[$name], 'Process') }
    if ($process -and !$process.HasExited) { Stop-Process -Id $process.Id -Force }
}
