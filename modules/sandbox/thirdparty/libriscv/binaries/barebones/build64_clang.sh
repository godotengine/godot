#!/bin/bash
set -e
export CC=clang-14
export CXX=clang++-14

mkdir -p build64_clang
pushd build64_clang
cmake .. -DRISCV_ARCH=64 -DLIBC_USE_STDLIB=OFF -DCMAKE_TOOLCHAIN_FILE=micro_toolchain.cmake
make -j4
popd
