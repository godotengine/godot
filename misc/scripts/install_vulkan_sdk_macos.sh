#!/usr/bin/env sh

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
else
	echo 'Error: Could not find 'jq' command. Is jq installed? Try running "brew install jq" or "port install jq" and rerunning this script.'
	exit 1
fi

# Download and install the Vulkan SDK.
curl -L "https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.zip" -o /tmp/vulkan-sdk.zip
unzip /tmp/vulkan-sdk.zip -d /tmp

if [ -d "/tmp/vulkansdk-macOS-$new_ver_full.app" ]; then
	/tmp/vulkansdk-macOS-$new_ver_full.app/Contents/MacOS/vulkansdk-macOS-$new_ver_full --accept-licenses --default-answer --confirm-command install
	rm -rf /tmp/vulkansdk-macOS-$new_ver_full.app
else
	echo "Couldn't install the Vulkan SDK, the unzipped contents may no longer match what this script expects."
	exit 1
fi

rm -f /tmp/vulkan-sdk.zip

echo 'Vulkan SDK installed successfully! You can now build Godot by running "scons".'
