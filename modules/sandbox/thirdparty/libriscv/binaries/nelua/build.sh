#!/usr/bin/env bash
set -e
#CC=riscv64-unknown-elf-gcc
CC=riscv32-unknown-elf-gcc
source="${1:-example}"

cfile=.build/output.c
binfile=.build/output.elf
mkdir -p .build
nelua -r --print-code $DMODE $source.nelua > $cfile

$CC -static -O2 -g3 -Wall -Wextra env/main.c $cfile -o $binfile

ls -la $binfile
