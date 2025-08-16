#!/usr/bin/env bash
set -e
./rvmicro ../binaries/barebones/build32/hello_world $@
./rvmicro ../binaries/barebones/build32_clang/hello_world $@
./rvmicro ../binaries/barebones/build64/hello_world $@
./rvmicro ../binaries/barebones/build64_clang/hello_world $@
