#!/bin/bash
set -e

GCC_TRIPLE="riscv64-unknown-elf"
export CC=riscv64-unknown-linux-gnu-gcc
export CXX=riscv64-unknown-linux-gnu-g++
#GCC_TRIPLE="riscv32-unknown-elf"
#export CC=riscv32-unknown-elf-gcc
#export CXX=riscv32-unknown-elf-g++

mkdir -p build
pushd build
cmake .. -DGCC_TRIPLE=$GCC_TRIPLE -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake $@
make -j4
popd
