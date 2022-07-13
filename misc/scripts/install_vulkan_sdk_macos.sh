#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Download and install the Vulkan SDK.
curl -LO "https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg"
hdiutil attach vulkan-sdk.dmg -mountpoint /Volumes/vulkan-sdk
/Volumes/vulkan-sdk/InstallVulkan.app/Contents/MacOS/InstallVulkan \
    --accept-licenses --default-answer --confirm-command install
hdiutil detach /Volumes/vulkan-sdk

echo 'Vulkan SDK installed successfully! You can now build Godot by running "scons".'
