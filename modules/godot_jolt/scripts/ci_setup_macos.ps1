#!/usr/bin/env pwsh

#Requires -PSEdition Core
#Requires -Version 7.2

param (
	[Parameter(HelpMessage = "Version of Xcode", Mandatory)]
	[ValidatePattern("^\d+\.\d+$")]
	[string]
	$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Output "Setting Xcode $Version as the default..."

xcode-select --switch /Applications/Xcode_$Version.app
