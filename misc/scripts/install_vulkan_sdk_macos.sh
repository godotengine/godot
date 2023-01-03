#!/usr/bin/env sh

set -euo pipefail
IFS=$'\n\t'

# Download and install the Vulkan SDK.
curl -L "https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg" -o /tmp/vulkan-sdk.dmg
hdiutil attach /tmp/vulkan-sdk.dmg -mountpoint /Volumes/vulkan-sdk
/Volumes/vulkan-sdk/InstallVulkan.app/Contents/MacOS/InstallVulkan \
    --accept-licenses --default-answer --confirm-command install

cnt=5
until hdiutil detach -force /Volumes/vulkan-sdk
do
   [[ cnt -eq "0" ]] && break
   sleep 1
   ((cnt--))
done

rm -f /tmp/vulkan-sdk.dmg

echo 'Vulkan SDK installed successfully! You can now build Godot by running "scons".'
