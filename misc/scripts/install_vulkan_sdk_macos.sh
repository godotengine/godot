#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'
new_ver_full=''

# Check currently installed and latest available Vulkan SDK versions.
if command -v jq 2>&1 >/dev/null; then
	curl -L "https://sdk.lunarg.com/sdk/download/latest/mac/config.json" -o /tmp/vulkan-sdk.json

	new_ver_full=`jq -r '.version' /tmp/vulkan-sdk.json`
	new_ver=`echo "$new_ver_full" | awk -F. '{ printf("%d%02d%04d%02d\n", $1,$2,$3,$4); }';`

	rm -f /tmp/vulkan-sdk.json

	for f in $HOME/VulkanSDK/*; do
		if [ -d "$f" ]; then
			f=`echo "${f##*/}" | awk -F. '{ printf("%d%02d%04d%02d\n", $1,$2,$3,$4); }';`
			if [ $f -ge $new_ver ]; then
				echo 'Latest or newer Vulkan SDK is already installed. Skipping installation.'
				exit 0
			fi
		fi
	done
fi

# Download and install the Vulkan SDK.
curl -L "https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.zip" -o /tmp/vulkan-sdk.zip
unzip /tmp/vulkan-sdk.zip -d /tmp

if [ -d "/tmp/InstallVulkan-$new_ver_full.app" ]; then
	/tmp/InstallVulkan-$new_ver_full.app/Contents/MacOS/InstallVulkan-$new_ver_full  --accept-licenses --default-answer --confirm-command install
	rm -rf /tmp/InstallVulkan-$new_ver_full.app
elif [ -d "/tmp/InstallVulkan.app" ]; then
	/tmp/InstallVulkan.app/Contents/MacOS/InstallVulkan --accept-licenses --default-answer --confirm-command install
	rm -rf /tmp/InstallVulkan.app
fi

rm -f /tmp/vulkan-sdk.zip

# Find the installer app
INSTALLER_APP=$(find /tmp/vulkan-sdk-extracted -maxdepth 1 -name "*.app" | head -1)
echo "Found installer: $INSTALLER_APP"

# Find the installer executable
INSTALLER_BIN=$(find "$INSTALLER_APP/Contents/MacOS" -type f -name "*ulkan*" | head -1)
echo "Found installer executable: $INSTALLER_BIN"

# Run the installer
echo "Running installer..."
"$INSTALLER_BIN" --accept-licenses --default-answer --confirm-command install

# Clean up temporary files
rm -rf /tmp/vulkan-sdk.dmg /tmp/vulkan-sdk-extracted

# Get installed version
SDK_VERSION=$(ls -1 "$INSTALL_DIR" | sort -V | tail -1)
echo "Vulkan SDK version $SDK_VERSION installed to $INSTALL_DIR/$SDK_VERSION"

# Set environment variable
if ! grep -q "VULKAN_SDK" ~/.zshrc; then
    echo "Setting VULKAN_SDK environment variable..."
    echo "export VULKAN_SDK=\"$INSTALL_DIR/$SDK_VERSION\"" >> ~/.zshrc
    echo "Environment variable added to ~/.zshrc, please run 'source ~/.zshrc' to apply."
else
    echo "VULKAN_SDK environment variable already exists, you may need to manually update it to new version path: $INSTALL_DIR/$SDK_VERSION"
fi

echo "Vulkan SDK installation successful! You can now run 'source ~/.zshrc' to update environment variables, then build Godot with 'scons platform=macos vulkan=yes'."