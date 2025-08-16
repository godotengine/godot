#!/usr/bin/env bash
set -e
export CXX=x86_64-unknown-cosmo-c++

mkdir -p .build_cosmo
pushd .build_cosmo
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOSMOPOLITAN=ON -DRISCV_EXT_V=ON -DRISCV_BINARY_TRANSLATION=OFF
ninja
popd
