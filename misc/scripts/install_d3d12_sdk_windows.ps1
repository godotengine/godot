# DirectX Shader Compiler

$dxc_filename="dxc_2023_08_14.zip"
Write-Output "Downloading DirectX Shader Compiler $dxc_filename ..."
Invoke-WebRequest -Uri https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.7.2308/$dxc_filename -OutFile "$env:TEMP\$dxc_filename"

if (Test-Path "$env:LOCALAPPDATA\dxc") {
    Write-Output "Removing existing local DirectX Shader Compiler installation in $env:LOCALAPPDATA\dxc ..."
    # Remove existing local DirectX Shader Compiler if present to allow for updates.
    Remove-Item -Recurse -Force "$env:LOCALAPPDATA\dxc"
}

Write-Output "Extracting DirectX Shader Compiler $dxc_filename ..."
Expand-Archive "$env:TEMP\$dxc_filename" -DestinationPath "$env:LOCALAPPDATA\dxc"

# Remove the original archive as it's not required anymore.
Remove-Item -Recurse -Force "$env:TEMP\$dxc_filename"

Write-Output "DirectX Shader Compiler $dxc_filename installed successfully.`n"

# Mesa NIR

$mesa_filename="godot-nir-23.1.0-1-devel.zip"
Write-Output "Downloading Mesa NIR $mesa_filename ..."
Invoke-WebRequest -Uri https://github.com/godotengine/godot-nir-static/releases/download/23.1.0-devel/$mesa_filename -OutFile "$env:TEMP\$mesa_filename"

if (Test-Path "$env:LOCALAPPDATA\mesa") {
    Write-Output "Removing existing local Mesa NIR installation in $env:LOCALAPPDATA\mesa ..."
    # Remove existing local Mesa NIR if present to allow for updates.
    Remove-Item -Recurse -Force "$env:LOCALAPPDATA\mesa"
}

Write-Output "Extracting Mesa NIR $mesa_filename ..."
Expand-Archive "$env:TEMP\$mesa_filename" -DestinationPath "$env:LOCALAPPDATA\mesa"

# Remove the original archive as it's not required anymore.
Remove-Item -Recurse -Force "$env:TEMP\$mesa_filename"

Write-Output "Mesa NIR $mesa_filename installed successfully.`n"

# WinPixEventRuntime

$pix_version="1.0.231030001"
Write-Output "Downloading WinPixEventRuntime $pix_version ..."
Invoke-WebRequest -Uri https://www.nuget.org/api/v2/package/WinPixEventRuntime/$pix_version -OutFile "$env:TEMP\$pix_version"

if (Test-Path "$env:LOCALAPPDATA\pix") {
    Write-Output "Removing existing local WinPixEventRuntime installation in $env:LOCALAPPDATA\pix ..."
    # Remove existing local WinPixEventRuntime if present to allow for updates.
    Remove-Item -Recurse -Force "$env:LOCALAPPDATA\pix"
}

Write-Output "Extracting WinPixEventRuntime $pix_version ..."
Expand-Archive "$env:TEMP\$pix_version" -DestinationPath "$env:LOCALAPPDATA\pix"

# Remove the original archive as it's not required anymore.
Remove-Item -Recurse -Force "$env:TEMP\$pix_version"

Write-Output "WinPixEventRuntime $pix_version installed successfully.`n"

# DirectX 12 Agility SDK

$agility_sdk_version="1.611.2"
Write-Output "Downloading DirectX 12 Agility SDK $agility_sdk_version ..."
Invoke-WebRequest -Uri https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/$agility_sdk_version -OutFile "$env:TEMP\$agility_sdk_version"

if (Test-Path "$env:LOCALAPPDATA\agility_sdk") {
    Write-Output "Removing existing local DirectX 12 Agility SDK installation in $env:LOCALAPPDATA\agility_sdk ..."
    # Remove existing local DirectX 12 Agility SDK if present to allow for updates.
    Remove-Item -Recurse -Force "$env:LOCALAPPDATA\agility_sdk"
}

Write-Output "Extracting DirectX 12 Agility SDK $agility_sdk_version ..."
Expand-Archive "$env:TEMP\$agility_sdk_version" -DestinationPath "$env:LOCALAPPDATA\agility_sdk"

# Remove the original archive as it's not required anymore.
Remove-Item -Recurse -Force "$env:TEMP\$agility_sdk_version"

Write-Output "DirectX 12 Agility SDK $agility_sdk_version installed successfully.`n"

Write-Output "All Direct3D 12 SDK components were installed successfully!"
Write-Output 'You can now build Godot with Direct3D 12 support enabled by running "scons d3d12=yes" with MSVC installed (MinGW builds are currently not supported).'
