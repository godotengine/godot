#!/usr/bin/env bash
set -e

mkdir -p build
pushd build
if [ ! -f CMakeCache.txt ]; then
	cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel -DRISCV_BINARY_TRANSLATION=ON
fi
make -j6
popd

VERBOSE=1 ./build/rvdoom
