#!/usr/bin/env pwsh

#Requires -PSEdition Core
#Requires -Version 7.2

param (
	[Parameter(HelpMessage = "Version of Visual Studio", Mandatory)]
	[ValidatePattern("^\d+(\.\d+)?$")]
	[string]
	$Version,

	[Parameter(HelpMessage = "Target architecture", Mandatory)]
	[ValidateSet("x64", "x86")]
	[string]
	$Arch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

filter ConvertTo-Hashtable {
	begin { $Result = @{} }
	process { $Result[$_.Key] = $_.Value }
	end { return $Result }
}

$VsArch = ($Arch -eq "x64") ? "amd64" : "x86";

Write-Output "Querying for Visual Studio $Version..."

$VsInstalls = @(vswhere -version $Version -sort -format json | ConvertFrom-Json)

if ($VsInstalls.Count -eq 0) {
	throw "Failed to find Visual Studio $Version"
}

$VsInstall = $VsInstalls[0]

Write-Output "Found Visual Studio $Version, displaying properties..."
Write-Output $($VsInstall | Format-List)
Write-Output "Looking for developer shell..."

$VsRoot = $VsInstall.installationPath
$VsDevShellPaths = @(Get-ChildItem -Recurse -Path $VsRoot -Filter "Launch-VsDevShell.ps1")

if ($VsDevShellPaths.Count -eq 0) {
	throw "Failed to find developer shell"
}

$VsDevShellPath = $VsDevShellPaths[0]

Write-Output "Found developer shell at: $VsDevShellPath"

Write-Output "Looking for clang-cl in PATH..."

$ClangClPath = Get-Command -ErrorAction SilentlyContinue clang-cl

if ($?) {
	$ClangClDir = $ClangClPath | Split-Path -Parent | Resolve-Path

	Write-Output "Found clang-cl at: $ClangClDir"
	Write-Output "Removing clang-cl from PATH..."

	$AllPaths = $env:PATH -split ";"
	$RealPaths = $AllPaths | Where-Object { Test-Path $_ }
	$NormalizedPaths = $RealPaths | ForEach-Object { Resolve-Path $_ }
	$FilteredPaths = $NormalizedPaths | Where-Object { $_.Path -ne $ClangClDir }
	$env:PATH = $FilteredPaths -join ";"
}
else {
	Write-Output "No clang-cl found in PATH, which is good"
}

Write-Output "Caching environment variables..."

$EnvOld = Get-ChildItem env: | ConvertTo-Hashtable

Write-Output "Launching $VsArch developer shell..."

. $VsDevShellPath -HostArch $VsArch -Arch $VsArch -SkipAutomaticLocation

Write-Output "Caching environment variables..."

$EnvNew = Get-ChildItem env: | ConvertTo-Hashtable

Write-Output "Exporting changed environment variables..."

foreach ($Item in $EnvNew.GetEnumerator()) {
	$Key = $Item.Key
	$Value = $Item.Value

	if ($EnvOld[$Key] -ne $EnvNew[$Key]) {
		"$Key=$Value" >> $env:GITHUB_ENV
	}
}
