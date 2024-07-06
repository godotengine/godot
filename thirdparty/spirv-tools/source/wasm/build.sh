#!/bin/bash

# Copyright (c) 2020 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# This is required to run any git command in the docker since owner will
# have changed between the clone environment, and the docker container.
# Marking the root of the repo as safe for ownership changes.
git config --global --add safe.directory /app

NUM_CORES=$(nproc)
echo "Detected $NUM_CORES cores for building"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VERSION=$(sed -n '0,/^v20/ s/^v\(20[0-9.]*\).*/\1/p' $DIR/../../CHANGES).${GITHUB_RUN_NUMBER:-0}
echo "Version: $VERSION"

build() { 
    type=$1
    shift
    args=$@
    mkdir -p build/$type
    pushd build/$type
    echo $args
    emcmake cmake \
        -DCMAKE_BUILD_TYPE=Release \
        $args \
        ../..
    emmake make -j $(( $NUM_CORES )) SPIRV-Tools-static

    echo Building js interface
    emcc \
        --bind \
        -I../../include \
        -std=c++17 \
        ../../source/wasm/spirv-tools.cpp \
        source/libSPIRV-Tools.a \
        -o spirv-tools.js \
        -s MODULARIZE \
        -Oz

    popd
    mkdir -p out/$type

    # copy other js files
    cp source/wasm/spirv-tools.d.ts out/$type/
    sed -e 's/\("version"\s*:\s*\).*/\1"'$VERSION'",/' source/wasm/package.json > out/$type/package.json
    cp source/wasm/README.md out/$type/
    cp LICENSE out/$type/

    cp build/$type/spirv-tools.js out/$type/
    gzip -9 -k -f out/$type/spirv-tools.js
    if [ -e build/$type/spirv-tools.wasm ] ; then
       cp build/$type/spirv-tools.wasm out/$type/
       gzip -9 -k -f out/$type/spirv-tools.wasm
    fi
}

if [ ! -d external/spirv-headers ] ; then
    echo "Fetching deps"
    utils/git-sync-deps
fi

echo Building ${BASH_REMATCH[1]}
build web\
    -DSPIRV_COLOR_TERMINAL=OFF\
    -DSPIRV_SKIP_TESTS=ON\
    -DSPIRV_SKIP_EXECUTABLES=ON

wc -c out/*/*
