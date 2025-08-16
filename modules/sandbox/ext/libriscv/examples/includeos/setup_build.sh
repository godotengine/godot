#!/bin/bash
set -e
mkdir -p build
pushd build
conan install .. -pr clang-6.0-linux-x86_64
source activate.sh
cmake .. -DSMP=ON
make -j
popd
