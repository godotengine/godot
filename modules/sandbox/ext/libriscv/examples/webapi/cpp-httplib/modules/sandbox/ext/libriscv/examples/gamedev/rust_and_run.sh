#!/bin/bash
set -e

# The Rust program uses a RISC-V toolchain
pushd rust_program
source ./build.sh
popd

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
make -j4
popd

./.build/example rust_program/target/riscv64gc-unknown-linux-gnu/release/rust_program
