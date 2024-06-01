#!/usr/bin/env pwsh

#Requires -PSEdition Core
#Requires -Version 7.2

param (
	[Parameter(
		HelpMessage = "Paths to framework bundles",
		Mandatory,
		ValueFromRemainingArguments,
		ValueFromPipeline
	)]
	[ValidateScript({ Test-Path $_ -PathType Container })]
	[string[]]
	$Frameworks
)

begin {
	Set-StrictMode -Version Latest
	$ErrorActionPreference = "Stop"

	$CodesignPath = Get-Command codesign | Resolve-Path

	$CertificateBase64 = $env:APPLE_CERT_BASE64
	$CertificatePassword = $env:APPLE_CERT_PASSWORD
	$CertificatePath = [IO.Path]::ChangeExtension((New-TemporaryFile), "p12")

	$Keychain = "ephemeral.keychain"
	$KeychainPassword = (New-Guid).ToString().Replace("-", "")

	$DevId = $env:APPLE_DEV_ID
	$DevTeamId = $env:APPLE_DEV_TEAM_ID
	$DevPassword = $env:APPLE_DEV_PASSWORD

	if (!$CertificateBase64) { throw "No certificate provided" }
	if (!$CertificatePassword) { throw "No certificate password provided" }
	if (!$DevId) { throw "No Apple Developer ID provided" }
	if (!$DevTeamId) { throw "No Apple Team ID provided" }
	if (!$DevPassword) { throw "No Apple Developer password provided" }

	Write-Output "Decoding certificate..."

	$Certificate = [Convert]::FromBase64String($CertificateBase64)

	Write-Output "Writing certificate to disk..."

	[IO.File]::WriteAllBytes($CertificatePath, $Certificate)

	Write-Output "Creating keychain..."

	security create-keychain -p $KeychainPassword $Keychain

	Write-Output "Setting keychain as default..."

	security default-keychain -s $Keychain

	Write-Output "Importing certificate into keychain..."

	security import $CertificatePath `
		-k ~/Library/Keychains/$Keychain `
		-P $CertificatePassword `
		-T $CodesignPath

	Write-Output "Granting access to keychain..."

	security set-key-partition-list -S "apple-tool:,apple:" -s -k $KeychainPassword $Keychain
}

process {
	foreach ($Framework in $Frameworks) {
		$Archive = [IO.Path]::ChangeExtension((New-TemporaryFile), "zip")

		Write-Output "Signing '$Framework'..."

		& $CodesignPath --sign "Developer ID" "$Framework"

		Write-Output "Verifying signing..."

		& $CodesignPath --verify "$Framework"

		Write-Output "Archiving framework to '$Archive'..."

		ditto -ck "$Framework" "$Archive"

		Write-Output "Submitting archive for notarization..."

		xcrun notarytool submit "$Archive" `
			--apple-id $DevId `
			--team-id $DevTeamId `
			--password $DevPassword `
			--wait
	}
}
