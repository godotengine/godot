#!/bin/bash
# Compile all GLSL shader fixtures to SPIR-V 1.3 for Vulkan 1.1, the targets Godot uses for WebGPU shaders.
# Requires: glslangValidator (from glslang or Vulkan SDK)
# Usage: ./compile_fixtures.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURES_DIR="$SCRIPT_DIR/fixtures"

if ! command -v glslangValidator &> /dev/null; then
    echo "ERROR: glslangValidator not found."
    echo "Install via: brew install glslang (macOS) or apt install glslang-tools (Linux)"
    exit 1
fi

echo "Compiling GLSL shaders to SPIR-V..."
echo ""

count=0
errors=0

shopt -s nullglob
for src in "$FIXTURES_DIR"/*.vert "$FIXTURES_DIR"/*.frag "$FIXTURES_DIR"/*.comp; do
    name="$(basename "$src")"
    base="${name%.*}"
    out="$FIXTURES_DIR/${base}.spv"

    printf "  %-35s" "$name"
    if glslangValidator -V --target-env vulkan1.1 "$src" -o "$out" --quiet 2>/dev/null; then
        echo "OK  ($(wc -c < "$out" | tr -d ' ') bytes)"
        count=$((count + 1))
    else
        echo "FAIL"
        glslangValidator -V --target-env vulkan1.1 "$src" -o "$out" 2>&1 | sed 's/^/    /'
        errors=$((errors + 1))
    fi
done

echo ""
echo "Compiled $count shaders, $errors errors."

if [ $errors -gt 0 ]; then
    exit 1
fi
