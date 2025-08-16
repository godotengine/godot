# Downloads are done from the official github release page links
$downloadUrl = "https://github.com/OpenCppCoverage/OpenCppCoverage/releases/download/release-0.9.9.0/OpenCppCoverageSetup-x64-0.9.9.0.exe"
$installerPath = [System.IO.Path]::Combine($Env:USERPROFILE, "Downloads", "OpenCppCoverageSetup.exe")

if(-Not (Test-Path $installerPath)) {
    Write-Host -ForegroundColor White ("Downloading OpenCppCoverage from: " + $downloadUrl)
    Start-BitsTransfer $downloadUrl -Destination $installerPath
}

Write-Host -ForegroundColor White "About to install OpenCppCoverage..."

$installProcess = (Start-Process $installerPath -ArgumentList '/VERYSILENT' -PassThru -Wait)
if($installProcess.ExitCode -ne 0) {
    throw [System.String]::Format("Failed to install OpenCppCoverage, ExitCode: {0}.", $installProcess.ExitCode)
}

# Assume standard, boring, installation path of ".../Program Files/OpenCppCoverage"
$installPath = [System.IO.Path]::Combine(${Env:ProgramFiles}, "OpenCppCoverage")
$env:Path="$env:Path;$installPath"
