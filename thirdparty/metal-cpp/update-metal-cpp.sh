#!/bin/bash -e

# metal-cpp update script for Godot
#
# metal-cpp source: https://developer.apple.com/metal/cpp/
# This version includes Metal 4 APIs (macOS 26 / iOS 26).

VERSION="macOS26-iOS26"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# If a zip is provided as argument, extract it
if [ -n "$1" ]; then
    echo "Updating metal-cpp from: $1"

    rm -rf \
        "$SCRIPT_DIR/Foundation" \
        "$SCRIPT_DIR/Metal" \
        "$SCRIPT_DIR/MetalFX" \
        "$SCRIPT_DIR/QuartzCore" \
        "$SCRIPT_DIR/SingleHeader"

    unzip -q "$1" -d "$SCRIPT_DIR"

    echo "Extracted metal-cpp $VERSION"
else
    echo "Applying patches only..."
fi

# =============================================================================
# Apply Godot-specific patches
# =============================================================================

echo "Applying Godot compatibility patches..."

# Apply patch files (idempotent)
PATCH_DIR="$SCRIPT_DIR/patches"
if [ -d "$PATCH_DIR" ]; then
    for PATCH in "$PATCH_DIR"/*.patch; do
        if [ ! -e "$PATCH" ]; then
            echo "  No patches found in $PATCH_DIR"
            break
        fi

        PATCH_NAME="$(basename "$PATCH")"
        if git -C "$REPO_ROOT" apply --check "$PATCH" > /dev/null 2>&1; then
            git -C "$REPO_ROOT" apply "$PATCH"
            echo "  $PATCH_NAME: applied"
        elif git -C "$REPO_ROOT" apply --reverse --check "$PATCH" > /dev/null 2>&1; then
            echo "  $PATCH_NAME: already applied"
        else
            echo "  $PATCH_NAME: failed to apply"
            exit 1
        fi
    done
else
    echo "  Warning: $PATCH_DIR not found"
fi

# Patch 1: Add forward declarations to NSDefines.hpp to avoid conflicts with
#          Godot's global types (String, Object, Error).
#
# Without this patch, metal-cpp's "class String*" forward declarations conflict
# with Godot's ::String, ::Object, and ::Error types.

NSDEFINES="$SCRIPT_DIR/Foundation/NSDefines.hpp"

if [ -f "$NSDEFINES" ]; then
    # Check if patch already applied
    if grep -q "Forward declarations to avoid conflicts with Godot" "$NSDEFINES"; then
        echo "  NSDefines.hpp: already patched"
    else
        # Find the line with #pragma once and insert after the next separator
        sed -i '' '/^#pragma once$/,/^\/\/---/{
            /^\/\/---/ {
                a\
\
// Forward declarations to avoid conflicts with Godot types (String, Object, Error)\
namespace NS {\
class Array;\
class Dictionary;\
class Error;\
class Object;\
class String;\
class URL;\
} // namespace NS
            }
        }' "$NSDEFINES"
        echo "  NSDefines.hpp: patched"
    fi
else
    echo "  Warning: $NSDEFINES not found"
fi

echo "Done."
