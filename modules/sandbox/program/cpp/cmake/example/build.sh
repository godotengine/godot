#!/bin/bash

# Change this to reflect your RISC-V toolchain
# ccache is optional, but recommended
export CXX="ccache riscv64-linux-gnu-g++-12"

# Create build directory
mkdir -p .build
pushd .build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake
ninja
popd
