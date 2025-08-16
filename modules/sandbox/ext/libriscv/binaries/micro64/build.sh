#!/bin/bash
set -e

RISCV_TC=$HOME/riscv
export PATH=$PATH:$RISCV_TC/bin
export CC=$RISCV_TC/bin/riscv64-unknown-elf-gcc
export CXX=$RISCV_TC/bin/riscv64-unknown-elf-g++

mkdir -p build
pushd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
make -j4
popd
