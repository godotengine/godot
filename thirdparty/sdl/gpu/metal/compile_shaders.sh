#!/bin/bash

set -x
set -e
cd `dirname "$0"`

shadernames=(FullscreenVert BlitFrom2D BlitFrom2DArray BlitFrom3D BlitFromCube BlitFromCubeArray)

generate_shaders()
{
        fileplatform=$1
        compileplatform=$2
        sdkplatform=$3
        minversion=$4

        for shadername in "${shadernames[@]}"; do
            xcrun -sdk $sdkplatform metal -c -std=$compileplatform-metal2.0 -m$sdkplatform-version-min=$minversion -Wall -O3 -D COMPILE_$shadername -o ./$shadername.air ./Metal_Blit.metal || exit $?
            xcrun -sdk $sdkplatform metallib -o $shadername.metallib $shadername.air || exit $?
            xxd -i $shadername.metallib | perl -w -p -e 's/\Aunsigned /const unsigned /;' >./${shadername}_$fileplatform.h
            rm -f $shadername.air $shadername.metallib
        done
}

generate_shaders macos macos macosx 10.11
generate_shaders ios ios iphoneos 11.0
generate_shaders iphonesimulator ios iphonesimulator 11.0
generate_shaders tvos ios appletvos 11.0
generate_shaders tvsimulator ios appletvsimulator 11.0

# Bundle together one mega-header
catShaders()
{
    target=$1
    for shadername in "${shadernames[@]}"; do
        cat ${shadername}_$target.h >> Metal_Blit.h
    done
}

rm -f Metal_Blit.h
echo "#if defined(SDL_PLATFORM_IOS)" >> Metal_Blit.h
    echo "#if TARGET_OS_SIMULATOR" >> Metal_Blit.h
        catShaders iphonesimulator
    echo "#else" >> Metal_Blit.h
        catShaders ios
    echo "#endif" >> Metal_Blit.h
echo "#elif defined(SDL_PLATFORM_TVOS)" >> Metal_Blit.h
    echo "#if TARGET_OS_SIMULATOR" >> Metal_Blit.h
        catShaders tvsimulator
    echo "#else" >> Metal_Blit.h
        catShaders tvos
    echo "#endif" >> Metal_Blit.h
echo "#else" >> Metal_Blit.h
    catShaders macos
echo "#endif" >> Metal_Blit.h

# Clean up
cleanupShaders()
{
    target=$1
    for shadername in "${shadernames[@]}"; do
        rm -f ${shadername}_$target.h
    done
}
cleanupShaders iphonesimulator
cleanupShaders ios
cleanupShaders tvsimulator
cleanupShaders tvos
cleanupShaders macos