#!/bin/bash
set -e

#RISCV_TC=$HOME/riscv
#export PATH=$PATH:$RISCV_TC/bin
#export CC=$RISCV_TC/bin/riscv64-unknown-linux-gnu-gcc
#export CXX=$RISCV_TC/bin/riscv64-unknown-linux-gnu-g++
export CC=riscv64-linux-gnu-gcc-10
export CXX=riscv64-linux-gnu-g++-10

mkdir -p build
pushd build
cmake ..
make -j4
popd
