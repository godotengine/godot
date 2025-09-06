#!/bin/bash
FILE="$1"
OUT="$2"
API="api/*.cpp"
INCLUDES="-Iapi"
VERSION=6

echo "Compiling $FILES into $OUT"

WARN="-Wall"
OPTS="-std=gnu++23 -fno-stack-protector -fno-threadsafe-statics $INCLUDES"
MEMOPS=-Wl,--wrap=memcpy,--wrap=memset,--wrap=memcmp,--wrap=memmove
STROPS=-Wl,--wrap=strlen,--wrap=strcmp,--wrap=strncmp
HEAPOPS=-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free
LINKEROPS="$MEMOPS $STROPS $HEAPOPS"

set -v -x
riscv64-unknown-elf-g++ -DVERSION=$VERSION -g -O2 $OPTS $WARN $API "$FILE" -o "$OUT"
