#!/usr/bin/env bash
set -euo pipefail

# Determine if sudo is available
if command -v sudo >/dev/null; then
    SUDO="sudo"
else
    SUDO=""
fi

# Update package lists
$SUDO apt-get update

# Install required packages for building Godot and running tests
$SUDO apt-get install -y \
    build-essential pkg-config git curl wget unzip zip \
    python3 python3-pip python3-setuptools python3-venv \
    clang clang-format clang-tidy \
    libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    libasound2-dev libpulse-dev libudev-dev \
    libwayland-dev libwayland-bin wayland-utils \
    mesa-vulkan-drivers libxml2-utils xvfb \
    libembree-dev libenet-dev libfreetype-dev libpng-dev zlib1g-dev \
    libgraphite2-dev libharfbuzz-dev libogg-dev libtheora-dev libvorbis-dev \
    libwebp-dev libmbedtls-dev libminiupnpc-dev libpcre2-dev libzstd-dev \
    libsquish-dev libicu-dev \
    nodejs npm

# Upgrade pip and install Python tooling
python3 -m pip install --upgrade pip
python3 -m pip install \
    scons==4.9.0 \
    pre-commit \
    mypy \
    ruff

# Print versions for debugging
scons --version
clang --version
node --version
python3 --version

echo "Environment setup complete."
