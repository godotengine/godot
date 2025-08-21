#!/bin/bash
set -e
export CC=riscv32-unknown-elf-gcc
export CXX=riscv32-unknown-elf-g++

mkdir -p build32
pushd build32
cmake .. -DRISCV_ARCH=32 -DLIBC_USE_STDLIB=ON -DLIBC_WRAP_NATIVE=ON -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
make -j4
popd
