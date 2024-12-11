#!/usr/bin/env pwsh

#Requires -PSEdition Core
#Requires -Version 7.2

param (
	[Parameter(HelpMessage = "Path to directory with source files", Mandatory)]
	[ValidateNotNullOrEmpty()]
	[string]
	$SourcePath,

	[Parameter(HelpMessage = "Path to directory with compile_commands.json", Mandatory)]
	[ValidateNotNullOrEmpty()]
	[string]
	$BuildPath,

	[Parameter(HelpMessage = "Apply fixes if applicable (warning: slow)")]
	[switch]
	$Fix = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SourceFiles = Get-ChildItem -Recurse -Path $SourcePath -Include "*.cpp"

if ($Fix) {
	clang-tidy -p $BuildPath --quiet --fix-notes @SourceFiles
	exit $LASTEXITCODE
}

$Outputs = [Collections.Concurrent.ConcurrentBag[psobject]]::new()

$SourceFiles | ForEach-Object -Parallel {
	$BuildPath = $using:BuildPath
	$Outputs = $using:Outputs
	$Output = $null
	$($Output = clang-tidy -p $BuildPath --quiet $_ *>&1) || $Outputs.Add($Output)
} -ThrottleLimit ([Environment]::ProcessorCount)

$Outputs | Where-Object { $_ -ne $null } | ForEach-Object {
	Write-Output $_
	Write-Output ""
}

exit $Outputs.IsEmpty ? 0 : 1
