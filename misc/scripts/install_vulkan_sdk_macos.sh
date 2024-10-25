#!/usr/bin/env sh

set -euo pipefail
IFS=$'\n\t'

if [ -d "$HOME/VulkanSDK" ]; then
   echo "Vulkan SDK was already installed. Skipping installation."
   exit 0
fi

# Download and install the Vulkan SDK.
curl -L "https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.zip" -o /tmp/vulkan-sdk.zip
unzip /tmp/vulkan-sdk.zip -d /tmp
/tmp/InstallVulkan.app/Contents/MacOS/InstallVulkan \
    --accept-licenses --default-answer --confirm-command install


rm -rf /tmp/InstallVulkan.app
rm -f /tmp/vulkan-sdk.zip

echo 'Vulkan SDK installed successfully! You can now build Godot by running "scons".'
