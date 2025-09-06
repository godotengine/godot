#!/bin/bash
set -e

#export CC=riscv64-linux-gnu-gcc-12
#export CXX=riscv64-linux-gnu-g++-12
export CC=riscv64-unknown-linux-gnu-gcc
export CXX=riscv64-unknown-linux-gnu-g++

mkdir -p build
pushd build
cmake ..
make -j4
popd
