#!/usr/bin/env sh

set -euo pipefail
IFS=$'\n\t'
latest_vulkan_version=''
stripped_latest_vulkan_version=''

# Check currently installed and latest available Vulkan SDK versions.
if command -v jq 2>&1 >/dev/null; then
	curl -L "https://sdk.lunarg.com/sdk/download/latest/mac/config.json" -o /tmp/vulkan-sdk.json

	latest_vulkan_version=`jq -r '.version' /tmp/vulkan-sdk.json`
	stripped_latest_vulkan_version=`echo "$latest_vulkan_version" | awk -F. '{ printf("%d%02d%04d%02d\n", $1,$2,$3,$4); }';`

	rm -f /tmp/vulkan-sdk.json

	for f in $HOME/VulkanSDK/*; do
		if [ -d "$f" ]; then
			f=`echo "${f##*/}" | awk -F. '{ printf("%d%02d%04d%02d\n", $1,$2,$3,$4); }';`
			if [ $f -ge $stripped_latest_vulkan_version ]; then
				echo 'Latest or newer Vulkan SDK is already installed. Skipping installation.'
				exit 0
			fi
		fi
	done
else
	echo 'Error: Could not find 'jq' command. Is jq installed? Try running "brew install jq" or "port install jq" and rerunning this script.'
	exit 1
fi

# Download the Vulkan SDK
vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.zip"
tmp_vulkan_archive_file_path="/tmp/vulkan-sdk-$stripped_latest_vulkan_version.zip"

# Fetch Content-Length from the final redirect target
# `tail -n 1` is used because -L (follow redirects) may return multiple
# Use last Content-Length headers as there is one per redirect hop.
remote_size=$(curl -sIL "$vulkan_sdk_url" | grep -i "Content-Length" | awk '{print $2}' | tr -d '\r' | tail -n 1)

# Guard against servers that omit Content-Length (e.g. chunked transfer encoding).
# If the header is absent, remote_size will be empty; force a download in that case.
if [ -z "$remote_size" ]; then
	remote_size=-1
fi

# stat -f%z is the macOS form for reading a file's byte size
if [ -f "$tmp_vulkan_archive_file_path" ] && [ "$(stat -f%z "$tmp_vulkan_archive_file_path")" -eq "$remote_size" ] 2>/dev/null; then
	echo "$tmp_vulkan_archive_file_path is already complete ($remote_size bytes). Skipping download."
else
	if [ -f "$tmp_vulkan_archive_file_path" ]; then
		echo "Continuing to download $vulkan_sdk_url to $tmp_vulkan_archive_file_path"
	else
		echo "Downloading $vulkan_sdk_url to $tmp_vulkan_archive_file_path"
	fi

	# -C -           resume from the last downloaded byte if a partial file exists
	# --fail         treat HTTP error responses (4xx, 5xx) as failures
	# --retry 5      retry up to 5 times on transient failures
	# --retry-delay  wait 5 seconds between retries
	curl -C - -L --fail --retry 5 --retry-delay 5 "$vulkan_sdk_url" -o "$tmp_vulkan_archive_file_path"
fi

unzip "$tmp_vulkan_archive_file_path" -d /tmp

# Install the Vulkan SDK
tmp_vulkan_directory="/tmp/vulkansdk-macOS-$latest_vulkan_version.app"
if [ -d "$tmp_vulkan_directory" ]; then
	"$tmp_vulkan_directory/Contents/MacOS/vulkansdk-macOS-$latest_vulkan_version" --accept-licenses --default-answer --confirm-command install
	rm -rf "$tmp_vulkan_directory"
else
	echo "Couldn't install the Vulkan SDK. Unzipped contents not found at $tmp_vulkan_directory."
	exit 1
fi

rm -f "$tmp_vulkan_archive_file_path"

echo 'Vulkan SDK installed successfully! You can now build Godot by running "scons".'
