#!/bin/bash
set -e

export ENGINE_CC="$CC"
export ENGINE_CXX="$CXX"

# The Nelua program uses a RISC-V compiler
pushd nelua_program
source ./build.sh
popd

# Restore the original compiler
CC=$ENGINE_CC
CXX=$ENGINE_CXX

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
make -j4
popd

./.build/example nelua_program/.build/output.elf
