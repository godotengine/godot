#!/usr/bin/env pwsh

#Requires -PSEdition Core
#Requires -Version 7.2

param (
	[Parameter(HelpMessage = "Toolchain to use", Mandatory)]
	[ValidateSet("gcc", "llvm")]
	[string]
	$Toolchain,

	[Parameter(HelpMessage = "Major version of GCC", Mandatory)]
	[ValidatePattern("^\d+$")]
	[string]
	$VersionGcc,

	[Parameter(HelpMessage = "Major version of LLVM")]
	[ValidatePattern("^\d+$")]
	[string]
	$VersionLlvm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Set-DefaultCommand($Name, $Path) {
	Write-Output "Displaying alternatives to '$Name' pre install..."

	update-alternatives --display $Name

	Write-Output "Installing alternatives to '$Name'..."

	update-alternatives --install /usr/bin/$Name $Name $Path 100
	update-alternatives --set $Name $Path

	Write-Output "Displaying alternatives to '$Name' post install..."

	update-alternatives --display $Name
}

Write-Output "Adding the ubuntu-toolchain-r repository..."

add-apt-repository --yes --update ppa:ubuntu-toolchain-r/test

Write-Output "Updating package lists..."

apt update

Write-Output "Installing GCC $VersionGcc..."

apt install --quiet --yes `
	g++-$VersionGcc `
	g++-$VersionGcc-multilib

Write-Output "Setting GCC $VersionGcc as the default..."

Set-DefaultCommand -Name gcc -Path /usr/bin/gcc-$VersionGcc
Set-DefaultCommand -Name g++ -Path /usr/bin/g++-$VersionGcc

if ($Toolchain -eq "llvm") {
	Write-Output "Downloading LLVM installation script..."

	Invoke-WebRequest -Uri https://apt.llvm.org/llvm.sh -OutFile ./llvm.sh

	Write-Output "Making LLVM installation script executable..."

	chmod +x ./llvm.sh

	Write-Output "Installing LLVM $VersionLlvm..."

	./llvm.sh $VersionLlvm all

	Write-Output "Setting LLVM $VersionLlvm as the default..."

	Set-DefaultCommand -Name clang -Path /usr/bin/clang-$VersionLlvm
	Set-DefaultCommand -Name clang++ -Path /usr/bin/clang++-$VersionLlvm
	Set-DefaultCommand -Name clang-format -Path /usr/bin/clang-format-$VersionLlvm
	Set-DefaultCommand -Name clang-tidy -Path /usr/bin/clang-tidy-$VersionLlvm
	Set-DefaultCommand -Name lld -Path /usr/bin/lld-$VersionLlvm
	Set-DefaultCommand -Name ld.lld -Path /usr/bin/ld.lld-$VersionLlvm
}
