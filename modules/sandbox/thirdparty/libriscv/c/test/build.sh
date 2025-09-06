#!/usr/bin/env bash
set -e

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRISCV_32I=OFF
make -j6
popd
