#!/usr/bin/env bash
set -e
CC=riscv64-unknown-elf-gcc
source="${1:-program}"

cfile=.build/output.c
binfile=.build/output.elf
mkdir -p .build
WRAP="-Wl,--wrap=malloc,--wrap=free,--wrap=calloc,--wrap=realloc,--wrap=memcpy,--wrap=memset,--wrap=memmove,--wrap=memcmp,--wrap=strlen,--wrap=strcmp,--wrap=strncmp"

if [ -z "$GDB" ]; then
	echo "Compiling for Release"
	nelua --ldflags="-static $WRAP" --cflags="api.c -O2 -g3 -Wall -Wextra" --release -o $binfile $source.nelua
else
	echo "Compiling for Debug with Remote GDB"
	nelua --ldflags="-static" --cflags="api.c -O1 -g3 -Wall -Wextra" --debug -o $binfile $source.nelua
fi
