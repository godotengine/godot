#!/bin/bash

export CXX=em++
export CC=emcc

mkdir -p .build
pushd .build
cmake -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_TOOLCHAIN_FILE=../cmake/wasm.cmake \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	  -DRISCV_32I=OFF \
	  -DRISCV_64I=ON \
	  -DRISCV_EXT_C=OFF \
	  -DRISCV_EXT_V=OFF \
	  -DRISCV_MEMORY_TRAPS=OFF \
	  -DRISCV_BINARY_TRANSLATION=OFF \
	  -DRISCV_EXPERIMENTAL=ON \
	  -DRISCV_ENCOMPASSING_ARENA=ON \
	  -DRISCV_ENCOMPASSING_ARENA_BITS=28 \
	  ..

make -j$(nproc)
popd
