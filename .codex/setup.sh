#!/usr/bin/env bash
set -euo pipefail
[[ $(id -u) -eq 0 ]] && SUDO="" || SUDO="sudo"

echo "🟣  Updating apt sources…"
$SUDO apt-get update -y -qq

echo "🟣  Installing core build deps (no recommends)…"
$SUDO apt-get install -y --no-install-recommends \
  build-essential clang clang-format clang-tidy lld \
  pkg-config git curl wget unzip zip \
  python3-pip python3-setuptools python3-venv \
  libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libxi-dev \
  libgl1-mesa-dev libglu1-mesa-dev libgles-dev libvulkan-dev mesa-vulkan-drivers \
  libasound2-dev libpulse-dev libudev-dev \
  libssl-dev libfreetype-dev libpng-dev \
  libwayland-dev libdecor-0-dev \
  libdrm-dev libenet-dev libembree-dev libtbb-dev \
  libogg-dev libvorbis-dev libopus-dev libtheora-dev libopenal-dev \
  libwebp-dev libmbedtls-dev libminiupnpc-dev libpcre2-dev libzstd-dev libsquish-dev \
  yasm xvfb xauth

# ─── Pip / npm tools ────────────────────────────────────────────────
python3 -m pip install --upgrade --user pip
python3 -m pip install --upgrade --user \
  scons==4.9.0 pre-commit ruff mypy codespell black isort pylint flake8 pytest

PIP_BIN="$(python3 -m site --user-base)/bin"
[[ ":$PATH:" != *":$PIP_BIN:"* ]] && export PATH="$PIP_BIN:$PATH"
grep -qxF "export PATH=\"$PIP_BIN:\$PATH\"" ~/.bashrc 2>/dev/null || \
  echo "export PATH=\"$PIP_BIN:\$PATH\"" >> ~/.bashrc

echo "🟣  Installing global npm tooling…"
npm install -g --silent \
  eslint @eslint/js @html-eslint/eslint-plugin @html-eslint/parser \
  @stylistic/eslint-plugin eslint-plugin-html globals espree jsdoc svgo \
  prettier stylelint markdownlint-cli typescript ts-node yarn

# ─── Pre-commit (if git repo) ───────────────────────────────────────
if [[ -d .git ]]; then
  pre-commit install --install-hooks || true
fi

echo "🟣  Done – runtimes were pre-installed, libs & tools ready."
