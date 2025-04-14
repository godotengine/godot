#!/bin/bash

# A Bash script for cross-compiling Godot Mono C# projects from macOS (Apple Silicon) to Steamdeck/Linux x86_64 binaries using Docker.

# Navigate to script directory
cd "$(dirname "$0")"

# Set variables
APP_NAME="YourGodotMonoAppName"
GODOT_VERSION="4.4"
GODOT_MONO_VERSION="4.4-stable" # Mono version for C# support
DOCKER_IMAGE="archlinux:latest"
PROJECT_DIR="$(pwd)"
DOCKER_PROJECT_DIR="/home/godot/project"
EXPORT_PRESET="Linux/X11" # The export preset name in your project

# Check for Docker installation
echo "🐬 Checking for Docker..."
if ! [ -x "$(command -v docker)" ]; then
    echo "Error: Docker is not installed." >&2
    exit 1
fi

# Pull Arch Linux Docker image
echo "Pulling Arch Linux Docker image..."
docker pull --platform linux/amd64 $DOCKER_IMAGE

# Build project using Docker
echo "Building $APP_NAME for Linux x86_64 using Docker..."

docker run --rm --name Godot-ARM-to-LinuxX64 \
    --platform linux/amd64 \
    -v "$PROJECT_DIR:$DOCKER_PROJECT_DIR" \
    -w "$DOCKER_PROJECT_DIR" \
    $DOCKER_IMAGE bash -c "
    # Update package database
    pacman -Syu --noconfirm
    
    # Install required dependencies
    pacman -S --noconfirm \
        base-devel \
        wget \
        unzip \
        git \
        libxkbcommon-x11 \
        libxcursor \
        libxrandr \
        libxi \
        libxinerama \
        vulkan-headers \
        vulkan-icd-loader \
        vulkan-validation-layers \
        alsa-lib \
        pulseaudio \
        libglvnd \
        gtk3 \
        dotnet-sdk

    # Set DOTNET_ROOT environment variable
    export DOTNET_ROOT=/usr/lib/dotnet
    
    # Create directories
    mkdir -p /home/godot/bin
    mkdir -p /home/godot/exports
    
    # Download Godot Mono Headless for Linux
    wget -q https://github.com/godotengine/godot-builds/releases/download/${GODOT_MONO_VERSION}/Godot_v${GODOT_MONO_VERSION}_mono_linux_x86_64.zip -O /tmp/godot.zip
    unzip -q /tmp/godot.zip -d /tmp
    mv /tmp/Godot_v${GODOT_MONO_VERSION}_mono_linux_x86_64/* /home/godot/bin/
    chmod +x /home/godot/bin/godot
    
    # Download export templates
    wget -q https://github.com/godotengine/godot-builds/releases/download/${GODOT_MONO_VERSION}/Godot_v${GODOT_MONO_VERSION}_mono_export_templates.tpz -O /tmp/templates.tpz
    mkdir -p /home/godot/.local/share/godot/export_templates/${GODOT_VERSION}.stable.mono
    unzip -q /tmp/templates.tpz -d /tmp
    mv /tmp/templates/* /home/godot/.local/share/godot/export_templates/${GODOT_VERSION}.stable.mono/
    
    # Ensure export templates are also in /root/.local
    mkdir -p /root/.local/share/godot/export_templates/${GODOT_VERSION}.stable.mono
    cp -r /home/godot/.local/share/godot/export_templates/${GODOT_VERSION}.stable.mono/* /root/.local/share/godot/export_templates/${GODOT_VERSION}.stable.mono/
    
    # Create export_presets.cfg if it doesn't exist
    if [ ! -f \"$DOCKER_PROJECT_DIR/export_presets.cfg\" ]; then
        echo 'Creating export_presets.cfg...'
        cat > \"$DOCKER_PROJECT_DIR/export_presets.cfg\" << EOF
[preset.0]

name=\"Linux/X11\"
platform=\"Linux/X11\"
runnable=true
export_path=\"exports/${APP_NAME}_linux_x86_64/${APP_NAME}.x86_64\"

[preset.0.options]

binary_format/embed_pck=true
texture_format/bptc=true
binary_format/architecture=\"x86_64\"
EOF
    fi
    
    # Create output directory
    mkdir -p \"$DOCKER_PROJECT_DIR/exports/${APP_NAME}_linux_x86_64\"
    
    # Export the project
    /home/godot/bin/godot --headless --export-release \"$EXPORT_PRESET\" \"$DOCKER_PROJECT_DIR/exports/${APP_NAME}_linux_x86_64/${APP_NAME}.x86_64\"
    
    # Verify export
    if [ -f \"$DOCKER_PROJECT_DIR/exports/${APP_NAME}_linux_x86_64/${APP_NAME}.x86_64\" ]; then
        chmod +x \"$DOCKER_PROJECT_DIR/exports/${APP_NAME}_linux_x86_64/${APP_NAME}.x86_64\"
        echo 'Export succeeded!'
    else
        echo 'Export failed!'
        exit 1
    fi
"

# Post-build tasks
if [ $? -eq 0 ]; then
    echo "Export successful! Binary located at $PROJECT_DIR/exports/${APP_NAME}_linux_x86_64/${APP_NAME}.x86_64"
else
    echo "Export failed."
    exit 1
fi
