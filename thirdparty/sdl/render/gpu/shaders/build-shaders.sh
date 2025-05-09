#!/usr/bin/env bash

set -e

make-header() {
    xxd -i "$1" | sed \
        -e 's/^unsigned /const unsigned /g' \
        -e 's,^const,static const,' \
        > "$1.h"
}

# Requires shadercross CLI installed from SDL_shadercross
for filename in *.hlsl; do
    if [ -f "$filename" ]; then
        shadercross "$filename" -o "${filename/.hlsl/.spv}"
        make-header "${filename/.hlsl/.spv}"
        shadercross "$filename" -o "${filename/.hlsl/.msl}"
        make-header "${filename/.hlsl/.msl}"
        shadercross "$filename" -o "${filename/.hlsl/.dxil}"
        make-header "${filename/.hlsl/.dxil}"
    fi
done

rm -f *.spv *.msl *.dxil
