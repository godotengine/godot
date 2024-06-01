#!/usr/bin/env pwsh

#Requires -PSEdition Core
#Requires -Version 7.2

param (
	[Parameter(HelpMessage = "Path to directory with source files", Mandatory)]
	[ValidateNotNullOrEmpty()]
	[string]$SourcePath,

	[Parameter(HelpMessage = "Apply fixes if applicable")]
	[switch]$Fix = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SourceFiles = Get-ChildItem -Recurse -Path $SourcePath -Include ("*.cpp", "*.hpp")

$Outputs = [Collections.Concurrent.ConcurrentBag[psobject]]::new()

$SourceFiles | ForEach-Object -Parallel {
	$Outputs = $using:Outputs
	$Fix = $using:Fix
	$Output = $null
	$($Output = clang-format $($Fix ? "-i" : "-n") --Werror $_ *>&1) || $Outputs.Add($Output)
} -ThrottleLimit ([Environment]::ProcessorCount)

$Outputs | Where-Object { $_ -ne $null } | ForEach-Object {
	Write-Output $_
	Write-Output ""
}

exit $Outputs.IsEmpty ? 0 : 1
