#!/bin/bash

set -x
set -e
cd `dirname "$0"`

generate_shaders()
{
    fileplatform=$1
    compileplatform=$2
    sdkplatform=$3
    minversion=$4
    xcrun -sdk $sdkplatform metal -c -std=$compileplatform-metal1.1 -m$sdkplatform-version-min=$minversion -Wall -O3 -o ./sdl.air ./SDL_shaders_metal.metal || exit $?
    xcrun -sdk $sdkplatform metal-ar rc sdl.metalar sdl.air || exit $?
    xcrun -sdk $sdkplatform metallib -o sdl.metallib sdl.metalar || exit $?
    xxd -i sdl.metallib | perl -w -p -e 's/\Aunsigned /const unsigned /;' >./SDL_shaders_metal_$fileplatform.h
    rm -f sdl.air sdl.metalar sdl.metallib
}

generate_shaders macos macos macosx 10.11
generate_shaders ios ios iphoneos 8.0
generate_shaders iphonesimulator ios iphonesimulator 8.0
generate_shaders tvos ios appletvos 9.0
generate_shaders tvsimulator ios appletvsimulator 9.0
