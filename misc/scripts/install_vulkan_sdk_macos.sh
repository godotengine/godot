#!/usr/bin/env sh

set -euo pipefail
IFS=$'\n\t'

# These can be found in https://vulkan.lunarg.com/sdk/home/
SDK_URL="https://sdk.lunarg.com/sdk/download/1.3.296.0/mac/vulkansdk-macos-1.3.296.0.zip"
SDK_HASH="393fd11f65a4001f12fd34fdd009c38045220ca3f735bc686d97822152b0f33c"

# Download and install the Vulkan SDK.
curl -L $SDK_URL -o /tmp/vulkan-sdk.zip

DL_HASH=$(shasum -a 256 /tmp/vulkan-sdk.zip | awk '{print $1}')
if [ "$SDK_HASH" != "$DL_HASH" ]; then
  echo "SDK_HASH: $SDK_HASH";
  echo "DL_HASH:  $DL_HASH";
  exit 1;
fi

unzip /tmp/vulkan-sdk.zip -d /tmp
/tmp/InstallVulkan.app/Contents/MacOS/InstallVulkan \
    --accept-licenses --default-answer --confirm-command install


rm -rf /tmp/InstallVulkan.app
rm -f /tmp/vulkan-sdk.zip

echo 'Vulkan SDK installed successfully! You can now build Godot by running "scons".'
