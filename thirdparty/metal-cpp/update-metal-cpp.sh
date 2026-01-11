#!/bin/bash -e

# metal-cpp update script for Godot
#
# metal-cpp source: https://developer.apple.com/metal/cpp/
# This version includes Metal 4 APIs (macOS 26 / iOS 26).

VERSION="macOS26-iOS26"

pushd "$(dirname "$0")" > /dev/null
SCRIPT_DIR="$(pwd)"

# If a tarball/zip is provided as argument, extract it
if [ -n "$1" ]; then
    echo "Updating metal-cpp from: $1"

    # Create temp directory for extraction and backup
    TMPDIR=$(mktemp -d)
    trap "rm -rf '$TMPDIR'" EXIT

    # Preserve this script
    cp "$SCRIPT_DIR/update-metal-cpp.sh" "$TMPDIR/update-metal-cpp.sh.bak"

    # Clean existing files (keep this script) - use absolute path for safety
    find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 ! -name 'update-metal-cpp.sh' -exec rm -rf {} +

    # Extract archive
    pushd "$TMPDIR" > /dev/null
    if [[ "$1" == *.zip ]]; then
        unzip -q "$1"
    else
        tar --strip-components=1 -xf "$1"
    fi

    # Copy contents (handle both flat and nested archives)
    if [ -d "metal-cpp" ]; then
        cp -r metal-cpp/* "$SCRIPT_DIR/"
    elif [ -d "Metal" ]; then
        cp -r . "$SCRIPT_DIR/"
    else
        # Try to find the metal-cpp directory
        METAL_DIR=$(find . -type d -name "Metal" -print -quit | xargs dirname)
        if [ -n "$METAL_DIR" ]; then
            cp -r "$METAL_DIR"/* "$SCRIPT_DIR/"
        else
            echo "Error: Could not find metal-cpp files in archive"
            exit 1
        fi
    fi
    popd > /dev/null

    # Restore this script
    mv "$TMPDIR/update-metal-cpp.sh.bak" "$SCRIPT_DIR/update-metal-cpp.sh"

    echo "Extracted metal-cpp $VERSION"
else
    echo "Usage: $0 <path-to-metal-cpp-archive>"
    echo ""
    echo "Download metal-cpp from: https://developer.apple.com/metal/cpp/"
    echo "Then run: $0 /path/to/metal-cpp.zip"
    echo ""
    echo "Applying patches only..."
fi

# =============================================================================
# Apply Godot-specific patches
# =============================================================================

echo "Applying Godot compatibility patches..."

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
popd > /dev/null
