#!/bin/bash
set -e
GCC_TRIPLE="riscv64-unknown-elf"
export CC=$GCC_TRIPLE-gcc
export CXX=$GCC_TRIPLE-g++
#export CC="riscv64-linux-gnu-gcc-11"
#export CXX="riscv64-linux-gnu-g++-11"

if [[ -z "${DEBUG}" ]]; then
	CDEBUG="-DGCSECTIONS=ON -DLTO=ON -DCMAKE_BUILD_TYPE=Release"
else
	CDEBUG="-DGCSECTIONS=OFF -DLTO=OFF -DCMAKE_BUILD_TYPE=Debug"
fi

mkdir -p $GCC_TRIPLE
pushd $GCC_TRIPLE
cmake .. -DGCC_TRIPLE=$GCC_TRIPLE $CDEBUG -DCMAKE_TOOLCHAIN_FILE=../micro/toolchain.cmake
make -j4
popd
