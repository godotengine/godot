[CmdletBinding()]
param(
    [string]$Root = "",
    [string]$GodotBinary = "",
    [string]$OutputRoot = "",
    [ValidateRange(0, 20)]
    [int]$RuntimeLoops = 3,
    [switch]$SkipQA,
    [switch]$SkipPainterly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# PowerShell 7 can convert native stderr lines into PowerShell errors when
# ErrorActionPreference=Stop. Godot logs many "ERROR:" lines to stderr even on
# successful runs, so keep native stderr as plain output.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-RepoRoot {
    param([string]$ExplicitRoot)

    if (-not [string]::IsNullOrWhiteSpace($ExplicitRoot)) {
        return (Resolve-Path -LiteralPath $ExplicitRoot).Path
    }

    $scriptDir = Split-Path -Parent $PSCommandPath
    return (Resolve-Path -LiteralPath (Join-Path $scriptDir "..\..")).Path
}

function Resolve-GodotBinaryPath {
    param(
        [string]$ExplicitBinary,
        [string]$RepoRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitBinary)) {
        return (Resolve-Path -LiteralPath $ExplicitBinary).Path
    }

    $candidates = @(
        (Join-Path $RepoRoot "bin\godot.windows.editor.dev.x86_64.console.exe"),
        (Join-Path $RepoRoot "bin\godot.windows.editor.dev.x86_64.exe"),
        (Join-Path $RepoRoot "bin\godot.windows.editor.x86_64.console.exe"),
        (Join-Path $RepoRoot "bin\godot.windows.editor.x86_64.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Could not resolve Godot binary. Pass -GodotBinary explicitly."
}

function Resolve-PythonCommand {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @("py", "-3")
    }

    throw "Could not find Python launcher (python/py) on PATH."
}

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$ChildPath
    )

    try {
        return [System.IO.Path]::GetRelativePath($BasePath, $ChildPath)
    } catch {
        return $ChildPath
    }
}

function Write-MarkdownFile {
    param(
        [string]$Path,
        [string[]]$Lines
    )

    [string]::Join([Environment]::NewLine, $Lines) | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string[]]$Command,
        [string]$WorkingDirectory,
        [string]$LogPath
    )

    if ($Command.Count -lt 1) {
        throw "Command for '$Name' is empty."
    }

    $exe = $Command[0]
    $args = @()
    if ($Command.Count -gt 1) {
        $args = $Command[1..($Command.Count - 1)]
    }

    Write-Host ""
    Write-Host "=== $Name ==="
    Write-Host "$exe $($args -join ' ')"

    $start = Get-Date
    $exitCode = 1
    $errorText = $null
    $previousErrorActionPreference = $ErrorActionPreference
    $hadNativePreference = $false
    $previousNativePreference = $null

    try {
        Push-Location $WorkingDirectory

        # Native tools (Godot) emit "ERROR:" lines on stderr for recoverable runtime
        # conditions. In strict PowerShell modes this can be promoted into terminating
        # errors, so run the native invocation with local relaxed preferences only.
        if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
            $hadNativePreference = $true
            $previousNativePreference = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
        }
        $ErrorActionPreference = "SilentlyContinue"

        & $exe @args 2>&1 | Tee-Object -FilePath $LogPath | Out-Host
        if ($null -eq $LASTEXITCODE) {
            $exitCode = 0
        } else {
            $exitCode = [int]$LASTEXITCODE
        }
    } catch {
        $errorText = $_.Exception.Message
        ($_ | Out-String) | Tee-Object -FilePath $LogPath -Append | Out-Null
        $exitCode = 1
    } finally {
        if ($hadNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }

    $duration = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)

    return [pscustomobject]@{
        name = $Name
        executable = $exe
        arguments = ($args -join " ")
        working_directory = $WorkingDirectory
        log_path = $LogPath
        exit_code = $exitCode
        duration_seconds = $duration
        error = $errorText
    }
}

function Get-ResultByName {
    param(
        [System.Collections.IEnumerable]$Results,
        [string]$Name
    )

    return $Results | Where-Object { $_.name -eq $Name } | Select-Object -First 1
}

function Test-ResultPassed {
    param(
        [System.Collections.IEnumerable]$Results,
        [string]$Name
    )

    $result = Get-ResultByName -Results $Results -Name $Name
    return ($null -ne $result -and [int]$result.exit_code -eq 0)
}

function Get-RuntimeMetricsFromLog {
    param([string]$LogPath)

    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $null
    }

    $markerMatches = @(Select-String -Path $LogPath -Pattern '^\[RUNTIME_METRICS\]\s*(\{.*\})\s*$' -ErrorAction SilentlyContinue)
    if ($markerMatches.Count -eq 0) {
        return $null
    }

    $lastMatch = $markerMatches[-1]
    $jsonPayload = $lastMatch.Matches[0].Groups[1].Value
    if ([string]::IsNullOrWhiteSpace($jsonPayload)) {
        return $null
    }

    try {
        return $jsonPayload | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Mark-Check {
    param([bool]$Value)

    if ($Value) {
        return "[x]"
    }

    return "[ ]"
}

$Root = Resolve-RepoRoot -ExplicitRoot $Root
$GodotBinary = Resolve-GodotBinaryPath -ExplicitBinary $GodotBinary -RepoRoot $Root
$PythonCommand = @(Resolve-PythonCommand)

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $Root "artifacts\evidence"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutputRoot $timestamp
$loopDir = Join-Path $runDir "runtime_loops"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path $loopDir | Out-Null

$results = @()

Write-Host "Repository root: $Root"
Write-Host "Godot binary:    $GodotBinary"
Write-Host "Output folder:   $runDir"

$commit = (& git -C $Root rev-parse HEAD 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($commit)) {
    $commit = "UNKNOWN"
}
$commit | Set-Content -LiteralPath (Join-Path $runDir "commit.txt") -Encoding ASCII

$versionLog = Join-Path $runDir "godot_version.log"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $GodotBinary --version 2>&1 | Tee-Object -FilePath $versionLog | Out-Null
$ErrorActionPreference = $previousErrorActionPreference

$moduleTestCommand = $PythonCommand + @(
    "tests/ci/run_module_tests.py",
    "--godot-binary",
    $GodotBinary
)
$moduleResult = Invoke-LoggedCommand -Name "module_tests" -Command $moduleTestCommand -WorkingDirectory $Root -LogPath (Join-Path $runDir "module_tests.log")
$results += $moduleResult

$runtimeScripts = @(
    @{ name = "runtime_interactive_state"; script = "tests/runtime/test_interactive_state.gd" },
    @{ name = "runtime_engine_capabilities"; script = "tests/runtime/test_engine_capabilities.gd" },
    @{ name = "runtime_gpu_streaming_stress"; script = "tests/runtime/test_gpu_streaming_stress.gd" },
    @{ name = "runtime_world_streaming_gate"; script = "tests/runtime/test_world_streaming_gate.gd" }
)

foreach ($runtimeScript in $runtimeScripts) {
    $cmd = @(
        $GodotBinary,
        "--path", $Root,
        "--display-driver", "windows",
        "--rendering-driver", "vulkan",
        "--rendering-method", "forward_plus",
        "--render-thread", "safe",
        "--script", $runtimeScript.script
    )
    $result = Invoke-LoggedCommand -Name $runtimeScript.name -Command $cmd -WorkingDirectory $Root -LogPath (Join-Path $runDir ($runtimeScript.name + ".log"))
    $results += $result
}

if ($RuntimeLoops -gt 0) {
    for ($i = 1; $i -le $RuntimeLoops; $i++) {
        foreach ($runtimeScript in $runtimeScripts) {
            $loopName = ("loop{0:D2}_{1}" -f $i, $runtimeScript.name)
            $cmd = @(
                $GodotBinary,
                "--path", $Root,
                "--display-driver", "windows",
                "--rendering-driver", "vulkan",
                "--rendering-method", "forward_plus",
                "--render-thread", "safe",
                "--script", $runtimeScript.script
            )
            $loopResult = Invoke-LoggedCommand -Name $loopName -Command $cmd -WorkingDirectory $Root -LogPath (Join-Path $loopDir ($loopName + ".log"))
            $results += $loopResult
        }
    }
}

if (-not $SkipQA) {
    $qaOutputPath = Join-Path $runDir "qa_results.json"
    $qaCommand = @(
        $GodotBinary,
        "--path", (Join-Path $Root "tests\examples\godot\test_project"),
        "--script", "res://scripts/qa_test_runner.gd",
        "--qa-output", $qaOutputPath
    )
    $qaResult = Invoke-LoggedCommand -Name "qa_runner" -Command $qaCommand -WorkingDirectory $Root -LogPath (Join-Path $runDir "qa_runner.log")
    $results += $qaResult
}

if (-not $SkipPainterly) {
    $painterlyCommand = @(
        $GodotBinary,
        "--headless",
        "--path", $Root,
        "--script", "scripts/tools/run_painterly_regression.gd"
    )
    $painterlyResult = Invoke-LoggedCommand -Name "painterly_regression" -Command $painterlyCommand -WorkingDirectory $Root -LogPath (Join-Path $runDir "painterly_regression.log")
    $results += $painterlyResult

    $painterlyToggleCommand = @(
        $GodotBinary,
        "--headless",
        "--path", $Root,
        "--script", "tests/examples/godot/test_toggle_painterly.gd"
    )
    $painterlyToggleResult = Invoke-LoggedCommand -Name "painterly_toggle_smoke" -Command $painterlyToggleCommand -WorkingDirectory $Root -LogPath (Join-Path $runDir "painterly_toggle_smoke.log")
    $results += $painterlyToggleResult
}

$allLogFiles = @(Get-ChildItem -Path $runDir -Recurse -Filter *.log -File)
$signatureRules = @(
    @{ id = "headless_skip"; regex = "Skipping .*headless mode" },
    @{ id = "gpu_sort_fallback"; regex = "Radix sort failed; falling back|GPU sorter never executed" },
    @{ id = "leak"; regex = "\\bleak(ed)?\\b" },
    @{ id = "invalid_free"; regex = "invalid free|Invalid ID" },
    @{ id = "streaming_invariant"; regex = "\\[Streaming\\] Invalid chunk state" }
)

$signatureHits = @()
foreach ($rule in $signatureRules) {
    $matches = @()
    if ($allLogFiles.Count -gt 0) {
        $matches = @(Select-String -Path $allLogFiles.FullName -Pattern $rule.regex -CaseSensitive:$false -ErrorAction SilentlyContinue)
    }
    foreach ($match in $matches) {
        $signatureHits += [pscustomobject]@{
            rule = $rule.id
            pattern = $rule.regex
            file = Get-RelativePath -BasePath $runDir -ChildPath $match.Path
            line = $match.LineNumber
            text = $match.Line.Trim()
        }
    }
}

if ($signatureHits.Count -gt 0) {
    ($signatureHits | ForEach-Object {
        "{0} | {1}:{2} | {3}" -f $_.rule, $_.file, $_.line, $_.text
    }) | Set-Content -LiteralPath (Join-Path $runDir "signature_hits.txt") -Encoding UTF8
} else {
    "No signature hits." | Set-Content -LiteralPath (Join-Path $runDir "signature_hits.txt") -Encoding UTF8
}

($signatureHits | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $runDir "signature_hits.json") -Encoding UTF8

$qaSummary = $null
$qaResultsPath = Join-Path $runDir "qa_results.json"
if (Test-Path -LiteralPath $qaResultsPath) {
    try {
        $qaSummary = Get-Content -LiteralPath $qaResultsPath -Raw | ConvertFrom-Json
    } catch {
        $qaSummary = $null
    }
}

$qaStreamingVisualScene = "res://scenes/qa/qa_stream_visual_smoke.tscn"
$qaStreamingVisualResult = $null
if ($null -ne $qaSummary -and $null -ne $qaSummary.results) {
    $qaStreamingVisualResult = $qaSummary.results | Where-Object { $_.scene -eq $qaStreamingVisualScene } | Select-Object -First 1
}
$qaStreamingVisualPresent = $null -ne $qaStreamingVisualResult
$qaStreamingVisualPassed = $qaStreamingVisualPresent -and [bool]$qaStreamingVisualResult.passed
$qaStreamingVisualMessage = ""
if ($qaStreamingVisualPresent -and $null -ne $qaStreamingVisualResult.message) {
    $qaStreamingVisualMessage = [string]$qaStreamingVisualResult.message
}
$qaStreamingVisualSkipped = $qaStreamingVisualMessage -match '\[QA_SKIP\]'
$qaStreamingVisualInfraFailurePatterns = @(
    '^Scene not found$',
    '^Failed to load scene$',
    '^Failed to instantiate$',
    '^No GSQATest found$'
)
$qaStreamingVisualInfraFailure = $false
if ($qaStreamingVisualPresent) {
    $qaStreamingVisualInfraFailure = @($qaStreamingVisualInfraFailurePatterns | Where-Object {
        $qaStreamingVisualMessage -match $_
    }).Count -gt 0
}
$qaStreamingVisualGateRequired = $true

$runtimeLogPaths = @()
foreach ($runtimeScript in $runtimeScripts) {
    $runtimeRecord = Get-ResultByName -Results $results -Name $runtimeScript.name
    if ($null -ne $runtimeRecord) {
        $runtimeLogPaths += $runtimeRecord.log_path
    }
}

$runtimeHeadlessHits = @()
if ($runtimeLogPaths.Count -gt 0) {
    $runtimeHeadlessHits = @(Select-String -Path $runtimeLogPaths -Pattern "Skipping .*headless mode" -CaseSensitive:$false -ErrorAction SilentlyContinue)
}

$runtimeMetrics = [ordered]@{}
foreach ($runtimeScript in $runtimeScripts) {
    $runtimeRecord = Get-ResultByName -Results $results -Name $runtimeScript.name
    if ($null -eq $runtimeRecord) {
        continue
    }
    $runtimeMetricPayload = Get-RuntimeMetricsFromLog -LogPath $runtimeRecord.log_path
    if ($null -ne $runtimeMetricPayload) {
        $runtimeMetrics[$runtimeScript.name] = $runtimeMetricPayload
    }
}

$streamingBudgetMetrics = $null
$streamingBudgetBaselineTier = ""
$streamingBudgetBaselinePassed = $false
$streamingBudgetTiers = @()
if ($runtimeMetrics.Contains("runtime_gpu_streaming_stress")) {
    $streamingBudgetMetrics = $runtimeMetrics["runtime_gpu_streaming_stress"]
}
if ($null -ne $streamingBudgetMetrics) {
    if ($streamingBudgetMetrics.PSObject.Properties.Name -contains "baseline_tier" -and
        $null -ne $streamingBudgetMetrics.baseline_tier) {
        $streamingBudgetBaselineTier = [string]$streamingBudgetMetrics.baseline_tier
    }
    if ($streamingBudgetMetrics.PSObject.Properties.Name -contains "baseline_passed") {
        $streamingBudgetBaselinePassed = [bool]$streamingBudgetMetrics.baseline_passed
    }
    if ($streamingBudgetMetrics.PSObject.Properties.Name -contains "tiers" -and
        $null -ne $streamingBudgetMetrics.tiers) {
        $streamingBudgetTiers = @($streamingBudgetMetrics.tiers)
    }
}
$streamingBudgetTierCount = $streamingBudgetTiers.Count

$loopResults = @($results | Where-Object { $_.name -like "loop*_runtime_*" })
$loopFailCount = @($loopResults | Where-Object { [int]$_.exit_code -ne 0 }).Count
$loopLeakHits = @($signatureHits | Where-Object {
    $_.file -like "runtime_loops*" -and ($_.rule -eq "leak" -or $_.rule -eq "invalid_free")
})

$painterlyMarkerFound = $false
$painterlyRecord = Get-ResultByName -Results $results -Name "painterly_regression"
if ($null -ne $painterlyRecord) {
    $painterlyMarkerFound = Select-String -Path $painterlyRecord.log_path -Pattern "PAINTERLY_TEST_PASSED" -SimpleMatch -Quiet
}

$qaRunnerScript = Join-Path $Root "tests\examples\godot\test_project\scripts\qa_test_runner.gd"
$qaStreamingSceneCount = 0
if (Test-Path -LiteralPath $qaRunnerScript) {
    $qaStreamingSceneMatches = @(Select-String -Path $qaRunnerScript -Pattern '^\s*"res://scenes/qa/qa_stream' -ErrorAction SilentlyContinue)
    $qaStreamingSceneCount = $qaStreamingSceneMatches.Count
}

$issue897Ready = (
    (Test-ResultPassed -Results $results -Name "runtime_interactive_state") -and
    (Test-ResultPassed -Results $results -Name "runtime_engine_capabilities") -and
    (Test-ResultPassed -Results $results -Name "runtime_gpu_streaming_stress") -and
    (Test-ResultPassed -Results $results -Name "runtime_world_streaming_gate") -and
    ($runtimeHeadlessHits.Count -eq 0)
)

$issue900Ready = (
    ($loopResults.Count -gt 0) -and
    ($loopFailCount -eq 0) -and
    ($loopLeakHits.Count -eq 0)
)

$gpuStreamingRecord = Get-ResultByName -Results $results -Name "runtime_gpu_streaming_stress"
$gpuStreamingExitText = "MISSING"
if ($null -ne $gpuStreamingRecord) {
    $gpuStreamingExitText = [string]$gpuStreamingRecord.exit_code
}

$issue943Ready = (
    (Test-ResultPassed -Results $results -Name "runtime_gpu_streaming_stress") -and
    ($streamingBudgetTierCount -gt 0) -and
    $streamingBudgetBaselinePassed
)

$qaRecord = Get-ResultByName -Results $results -Name "qa_runner"
$qaExitText = "SKIPPED"
if ($null -ne $qaRecord) {
    $qaExitText = [string]$qaRecord.exit_code
}

$qaPassedCount = 0
$qaFailedCount = 0
if ($null -ne $qaSummary -and $null -ne $qaSummary.summary) {
    $qaPassedCount = [int]$qaSummary.summary.passed
    $qaFailedCount = [int]$qaSummary.summary.failed
}

$qaFailuresExcludingStreamingVisualCount = 0
if ($null -ne $qaSummary -and $null -ne $qaSummary.results) {
    $qaFailuresExcludingStreamingVisualCount = @($qaSummary.results | Where-Object {
        (-not [bool]$_.passed) -and ($_.scene -ne $qaStreamingVisualScene)
    }).Count
}

$qaOnlyStreamingVisualFailure = (
    ($qaFailuresExcludingStreamingVisualCount -eq 0) -and
    $qaStreamingVisualPresent -and
    (-not $qaStreamingVisualPassed) -and
    (-not $qaStreamingVisualSkipped) -and
    (-not $qaStreamingVisualInfraFailure)
)

$qaRunnerPassed = (Test-ResultPassed -Results $results -Name "qa_runner")
$qaRunnerEffectivePass = (
    $qaRunnerPassed -or
    ((-not $qaStreamingVisualGateRequired) -and $qaOnlyStreamingVisualFailure)
)

$issue871Ready = (
    $qaRunnerEffectivePass -and
    ($qaFailuresExcludingStreamingVisualCount -eq 0) -and
    ($qaStreamingSceneCount -gt 0) -and
    $qaStreamingVisualPresent -and
    (-not $qaStreamingVisualSkipped) -and
    ($qaStreamingVisualPassed -or (-not $qaStreamingVisualGateRequired))
)

$painterlyRegression = Get-ResultByName -Results $results -Name "painterly_regression"
$painterlyToggle = Get-ResultByName -Results $results -Name "painterly_toggle_smoke"
$painterlyRegressionExit = "SKIPPED"
$painterlyToggleExit = "SKIPPED"
if ($null -ne $painterlyRegression) { $painterlyRegressionExit = [string]$painterlyRegression.exit_code }
if ($null -ne $painterlyToggle) { $painterlyToggleExit = [string]$painterlyToggle.exit_code }

$issue815Ready = (
    (Test-ResultPassed -Results $results -Name "painterly_regression") -and
    (Test-ResultPassed -Results $results -Name "painterly_toggle_smoke") -and
    $painterlyMarkerFound
)

$issue902Ready = ($issue897Ready -and $issue900Ready -and $issue871Ready -and $issue815Ready)

$summaryObject = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    commit = $commit
    root = $Root
    godot_binary = $GodotBinary
    output_directory = $runDir
    runtime_loops = $RuntimeLoops
    signatures_total = $signatureHits.Count
    runtime_metrics = $runtimeMetrics
    qa_streaming_scene_count = $qaStreamingSceneCount
    qa_streaming_visual_scene = $qaStreamingVisualScene
    qa_streaming_visual_present = $qaStreamingVisualPresent
    qa_streaming_visual_passed = $qaStreamingVisualPassed
    qa_streaming_visual_skipped = $qaStreamingVisualSkipped
    qa_streaming_visual_infra_failure = $qaStreamingVisualInfraFailure
    qa_streaming_visual_gate_required = $qaStreamingVisualGateRequired
    qa_failures_excluding_streaming_visual = $qaFailuresExcludingStreamingVisualCount
    qa_only_streaming_visual_failure = $qaOnlyStreamingVisualFailure
    qa_runner_passed = $qaRunnerPassed
    qa_runner_effective_pass = $qaRunnerEffectivePass
    issues = [ordered]@{
        "897_907_ready" = $issue897Ready
        "900_ready" = $issue900Ready
        "943_ready" = $issue943Ready
        "871_ready" = $issue871Ready
        "815_ready" = $issue815Ready
        "902_ready" = $issue902Ready
    }
    results = $results
}

($summaryObject | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $runDir "summary.json") -Encoding UTF8

$runtimeRows = @()
foreach ($runtimeScript in $runtimeScripts) {
    $record = Get-ResultByName -Results $results -Name $runtimeScript.name
    if ($null -eq $record) { continue }
    $runtimeRows += ("| {0} | {1} | {2} |" -f $runtimeScript.name, $record.exit_code, (Get-RelativePath -BasePath $Root -ChildPath $record.log_path))
}

$issue897Lines = @()
$issue897Lines += "# Evidence Update: #897 / #907"
$issue897Lines += ""
$issue897Lines += "- Commit: $commit"
$issue897Lines += "- Evidence folder: $(Get-RelativePath -BasePath $Root -ChildPath $runDir)"
$issue897Lines += ""
$issue897Lines += "## Runtime Matrix"
$issue897Lines += ""
$issue897Lines += "| Script | Exit Code | Log |"
$issue897Lines += "| --- | ---: | --- |"
$issue897Lines += $runtimeRows
$issue897Lines += @(
    "",
    "## Acceptance Checks",
    "",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "runtime_interactive_state")) + " test_interactive_state.gd passed.",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "runtime_engine_capabilities")) + " test_engine_capabilities.gd passed.",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "runtime_gpu_streaming_stress")) + " test_gpu_streaming_stress.gd passed.",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "runtime_world_streaming_gate")) + " test_world_streaming_gate.gd passed.",
    "- " + (Mark-Check ($runtimeHeadlessHits.Count -eq 0)) + " No headless-skip markers in runtime logs.",
    "",
    "## Signature Hits",
    "",
    "- Runtime headless-skip hits: " + $runtimeHeadlessHits.Count,
    "- Global signature hits file: " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "signature_hits.txt")),
    "",
    "## Suggested Status",
    "",
    "- " + (Mark-Check $issue897Ready) + " Ready to close #897/#907."
)

if ($runtimeMetrics.Contains("runtime_world_streaming_gate")) {
    $worldMetrics = $runtimeMetrics["runtime_world_streaming_gate"]
    $issue897Lines += @(
        "",
        "## World Streaming Metrics",
        "",
        "- monitor_ready_seen: " + [string]$worldMetrics.monitor_ready_seen,
        "- total_chunks_max: " + [string]$worldMetrics.total_chunks_max,
        "- loaded_chunks_max: " + [string]$worldMetrics.loaded_chunks_max,
        "- visible_chunks_max: " + [string]$worldMetrics.visible_chunks_max,
        "- streaming_visible_count_max: " + [string]$worldMetrics.streaming_visible_count_max,
        "- renderer_visible_splats_max: " + [string]$worldMetrics.renderer_visible_splats_max,
        "- renderer_data_source: " + [string]$worldMetrics.renderer_data_source,
        "- streaming_diagnostics_category: " + [string]$worldMetrics.streaming_diagnostics_category,
        "- streaming_diagnostics_reason: " + [string]$worldMetrics.streaming_diagnostics_reason
    )
}

Write-MarkdownFile -Path (Join-Path $runDir "issue_897_907.md") -Lines $issue897Lines

$loopRows = @()
foreach ($loopRecord in $loopResults) {
    $loopRows += ("| {0} | {1} | {2} |" -f $loopRecord.name, $loopRecord.exit_code, (Get-RelativePath -BasePath $Root -ChildPath $loopRecord.log_path))
}

$issue900Lines = @(
    "# Evidence Update: #900",
    "",
    "- Commit: $commit",
    "- Runtime loop count: $RuntimeLoops",
    "- Evidence folder: " + (Get-RelativePath -BasePath $Root -ChildPath $runDir),
    "",
    "## Runtime Loop Results",
    "",
    "| Run | Exit Code | Log |",
    "| --- | ---: | --- |"
)
$issue900Lines += $loopRows
$issue900Lines += @(
    "",
    "## Lifetime / Leak Checks",
    "",
    "- " + (Mark-Check ($loopFailCount -eq 0)) + " No runtime loop command failures.",
    "- " + (Mark-Check ($loopLeakHits.Count -eq 0)) + " No leak/invalid-free signature hits in runtime_loops logs.",
    "- Loop leak/invalid-free hit count: " + $loopLeakHits.Count,
    "",
    "## Suggested Status",
    "",
    "- " + (Mark-Check $issue900Ready) + " Ready to close #900."
)
Write-MarkdownFile -Path (Join-Path $runDir "issue_900.md") -Lines $issue900Lines

$issue943Lines = @(
    "# Evidence Update: #943",
    "",
    "- Commit: $commit",
    "- Evidence folder: " + (Get-RelativePath -BasePath $Root -ChildPath $runDir),
    "",
    "## Streaming Scale Budget Metrics",
    "",
    "- Runtime script exit code: " + $gpuStreamingExitText,
    "- Baseline tier: " + ($(if ([string]::IsNullOrWhiteSpace($streamingBudgetBaselineTier)) { "unknown" } else { $streamingBudgetBaselineTier })),
    "- Baseline passed: " + $streamingBudgetBaselinePassed,
    "- Tier count: " + $streamingBudgetTierCount,
    "",
    "| Tier | Size | Enforced | Within Budget | Source Data | Sort Evidence | Fallback Status | First Visible (ms) | Residency Ratio | Frame P95 (ms) | Fallback Rate |",
    "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"
)

if ($streamingBudgetTierCount -gt 0) {
    foreach ($tier in $streamingBudgetTiers) {
        $tierName = [string]$tier.name
        $tierSize = [int]$tier.dataset_size
        $tierEnforce = [bool]$tier.enforce
        $tierWithin = [bool]$tier.within_budget
        $sourceStatus = "unknown"
        if ($tier.PSObject.Properties.Name -contains "source_data_status" -and
            -not [string]::IsNullOrWhiteSpace([string]$tier.source_data_status)) {
            $sourceStatus = [string]$tier.source_data_status
        } elseif ($tier.PSObject.Properties.Name -contains "source_data_available") {
            $sourceStatus = if ([bool]$tier.source_data_available) { "available" } else { "missing" }
        }
        $sortEvidenceStatus = "unknown"
        if ($tier.PSObject.Properties.Name -contains "sort_evidence_status" -and
            -not [string]::IsNullOrWhiteSpace([string]$tier.sort_evidence_status)) {
            $sortEvidenceStatus = [string]$tier.sort_evidence_status
        } elseif ($tier.PSObject.Properties.Name -contains "sort_evidence_available") {
            $sortEvidenceStatus = if ([bool]$tier.sort_evidence_available) { "available" } else { "missing" }
        }
        $fallbackStatus = "unknown"
        if ($tier.PSObject.Properties.Name -contains "fallback_rate_status" -and
            -not [string]::IsNullOrWhiteSpace([string]$tier.fallback_rate_status)) {
            $fallbackStatus = [string]$tier.fallback_rate_status
        } elseif ($tier.PSObject.Properties.Name -contains "fallback_rate_available") {
            $fallbackStatus = if ([bool]$tier.fallback_rate_available) { "available" } else { "missing" }
        }
        $firstVisibleText = "n/a"
        if ($tier.PSObject.Properties.Name -contains "first_visible_ms" -and [double]$tier.first_visible_ms -ge 0.0) {
            $firstVisibleText = "{0:N2}" -f [double]$tier.first_visible_ms
        }
        $residencyText = "{0:N3}" -f [double]$tier.residency_ratio
        $frameP95Text = "{0:N2}" -f [double]$tier.frame_p95_ms
        $fallbackText = "n/a"
        if ($tier.PSObject.Properties.Name -contains "fallback_rate" -and $null -ne $tier.fallback_rate) {
            $fallbackText = "{0:N3}" -f [double]$tier.fallback_rate
        }
        $issue943Lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} |" -f
            $tierName, $tierSize, $tierEnforce, $tierWithin, $sourceStatus, $sortEvidenceStatus, $fallbackStatus, $firstVisibleText, $residencyText, $frameP95Text, $fallbackText)
    }
} else {
    $issue943Lines += "| (missing) | 0 | false | false | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
}

$issue943Lines += @(
    "",
    "## Acceptance Checks",
    "",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "runtime_gpu_streaming_stress")) + " runtime_gpu_streaming_stress script passed.",
    "- " + (Mark-Check ($streamingBudgetTierCount -gt 0)) + " Streaming tier metrics emitted.",
    "- " + (Mark-Check $streamingBudgetBaselinePassed) + " Baseline tier budget passed.",
    "",
    "## Suggested Status",
    "",
    "- " + (Mark-Check $issue943Ready) + " Ready to close #943."
)
Write-MarkdownFile -Path (Join-Path $runDir "issue_943.md") -Lines $issue943Lines

$issue871Lines = @(
    "# Evidence Update: #871",
    "",
    "- Commit: $commit",
    "- Evidence folder: " + (Get-RelativePath -BasePath $Root -ChildPath $runDir),
    "",
    "## QA Runner",
    "",
    "- Exit code: $qaExitText",
    "- Log: " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "qa_runner.log")),
    "- QA results JSON: " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "qa_results.json")),
    "- QA summary (passed/failed): $qaPassedCount / $qaFailedCount",
    "- QA non-streaming failed scene count: $qaFailuresExcludingStreamingVisualCount",
    "- Enabled streaming QA scenes in runner: $qaStreamingSceneCount",
    "- Streaming visual smoke scene: $qaStreamingVisualScene",
    "- Streaming visual smoke present in results: $qaStreamingVisualPresent",
    "- Streaming visual smoke passed: $qaStreamingVisualPassed",
    "- Streaming visual smoke skipped: $qaStreamingVisualSkipped",
    "- Streaming visual smoke infra/setup failure: $qaStreamingVisualInfraFailure",
    "- Streaming visual smoke message: $qaStreamingVisualMessage",
    "- Streaming visual pass required for readiness: $qaStreamingVisualGateRequired",
    "- QA runner effective pass: $qaRunnerEffectivePass",
    "",
    "## Acceptance Checks",
    "",
    "- " + (Mark-Check $qaRunnerPassed) + " QA runner command passed.",
    "- " + (Mark-Check $qaRunnerEffectivePass) + " QA runner effective pass.",
    "- " + (Mark-Check ($qaFailuresExcludingStreamingVisualCount -eq 0)) + " QA result set contains zero non-streaming failures.",
    "- " + (Mark-Check ($qaStreamingSceneCount -gt 0)) + " Streaming QA scenes are enabled in qa_test_runner.gd.",
    "- " + (Mark-Check $qaStreamingVisualPresent) + " qa_stream_visual_smoke.tscn appears in QA results.",
    "- " + (Mark-Check ($qaStreamingVisualPassed -or (-not $qaStreamingVisualGateRequired))) + " qa_stream_visual_smoke.tscn pass is " + ($(if ($qaStreamingVisualGateRequired) { "required" } else { "waived temporarily" })) + ".",
    "- " + (Mark-Check (-not $qaStreamingVisualSkipped)) + " qa_stream_visual_smoke.tscn was not skipped.",
    "- " + (Mark-Check (-not $qaStreamingVisualInfraFailure)) + " qa_stream_visual_smoke.tscn did not fail due to test infra/setup issues (missing scene/load/instantiate/test node).",
    "",
    "## Suggested Status",
    "",
    "- " + (Mark-Check $issue871Ready) + " Ready to close #871."
)
Write-MarkdownFile -Path (Join-Path $runDir "issue_871.md") -Lines $issue871Lines

$issue815Lines = @(
    "# Evidence Update: #815",
    "",
    "- Commit: $commit",
    "- Evidence folder: " + (Get-RelativePath -BasePath $Root -ChildPath $runDir),
    "",
    "## Painterly Validation",
    "",
    "| Check | Exit Code | Log |",
    "| --- | ---: | --- |",
    "| painterly_regression | $painterlyRegressionExit | " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "painterly_regression.log")) + " |",
    "| painterly_toggle_smoke | $painterlyToggleExit | " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "painterly_toggle_smoke.log")) + " |",
    "",
    "## Acceptance Checks",
    "",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "painterly_regression")) + " Painterly regression command passed.",
    "- " + (Mark-Check $painterlyMarkerFound) + " PAINTERLY_TEST_PASSED marker present in regression log.",
    "- " + (Mark-Check (Test-ResultPassed -Results $results -Name "painterly_toggle_smoke")) + " Painterly toggle smoke test passed.",
    "- [ ] Painterly on/off performance benchmark artifact attached (not produced by this script).",
    "",
    "## Suggested Status",
    "",
    "- " + (Mark-Check $issue815Ready) + " Core painterly stability gate satisfied."
)
Write-MarkdownFile -Path (Join-Path $runDir "issue_815.md") -Lines $issue815Lines

$issue902Lines = @(
    "# Evidence Update: #902",
    "",
    "- Commit: $commit",
    "- Evidence folder: " + (Get-RelativePath -BasePath $Root -ChildPath $runDir),
    "",
    "## Lane Status",
    "",
    "- " + (Mark-Check $issue897Ready) + " Runtime gate (#897/#907) ready.",
    "- " + (Mark-Check $issue900Ready) + " RID ownership lifetime gate (#900) ready.",
    "- " + (Mark-Check $issue943Ready) + " Streaming scale budgets (#943) ready.",
    "- " + (Mark-Check $issue871Ready) + " Tile/visual artifact QA gate (#871) ready.",
    "- " + (Mark-Check $issue815Ready) + " Painterly audit stability gate (#815) ready.",
    "",
    "## Suggested Status",
    "",
    "- " + (Mark-Check $issue902Ready) + " Sprint board can close.",
    "- " + (Mark-Check $issue902Ready) + " Remaining blockers are documented in issue markdown files under " + (Get-RelativePath -BasePath $Root -ChildPath $runDir) + "."
)
Write-MarkdownFile -Path (Join-Path $runDir "issue_902.md") -Lines $issue902Lines

$summaryLines = @(
    "# Production Evidence Summary",
    "",
    "- Commit: $commit",
    "- Run directory: " + (Get-RelativePath -BasePath $Root -ChildPath $runDir),
    "- Summary JSON: " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "summary.json")),
    "- Signature hits: " + (Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir "signature_hits.txt")),
    "",
    "## Issue Markdown Outputs",
    "",
    "- issue_897_907.md",
    "- issue_900.md",
    "- issue_943.md",
    "- issue_871.md",
    "- issue_815.md",
    "- issue_902.md"
)
Write-MarkdownFile -Path (Join-Path $runDir "README.md") -Lines $summaryLines

Write-Host ""
Write-Host "Evidence collection completed."
Write-Host "Summary: $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'README.md'))"
Write-Host "Issue drafts:"
Write-Host "  - $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'issue_897_907.md'))"
Write-Host "  - $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'issue_900.md'))"
Write-Host "  - $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'issue_943.md'))"
Write-Host "  - $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'issue_871.md'))"
Write-Host "  - $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'issue_815.md'))"
Write-Host "  - $(Get-RelativePath -BasePath $Root -ChildPath (Join-Path $runDir 'issue_902.md'))"
