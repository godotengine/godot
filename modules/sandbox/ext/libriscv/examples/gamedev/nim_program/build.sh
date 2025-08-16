#!/usr/bin/env bash
set -e
CC=riscv64-linux-gnu-gcc-12 #riscv64-unknown-elf-gcc
NIMFILE="$PWD/${1:-program}"
NIMCPU="--cpu=riscv64"
NIMAPI="$PWD/api.c"
binfile=output.elf
mkdir -p .build
pushd .build

# Detect nim libraries: find nim and replace /bin/nim with /lib
NIM_LIBS=`whereis nim`
NIM_LIBS="${NIM_LIBS##*: }"
NIM_LIBS="${NIM_LIBS/bin*/lib}"

NIMCACHE=$PWD/nimcache
mkdir -p $NIMCACHE

set -e
if [ -z "$GDB" ]; then
	echo "Compiling for Release"
	nim c --nimcache:$NIMCACHE $NIMCPU --colors:on --os:linux --mm:arc --threads:off -d:release -d:useMalloc=true -c ${NIMFILE}
else
	echo "Compiling for Debug with Remote GDB"
	nim c --nimcache:$NIMCACHE $NIMCPU --colors:on --os:linux --mm:arc --threads:off -d:useMalloc=true --stackTrace:off --debugger=native -c ${NIMFILE}
fi
jq '.compile[] [0]' $NIMCACHE/*.json -r > buildfiles.txt

files=""
for i in $(cat buildfiles.txt); do
    files="$files $i"
done

$CC -static -O2 -ggdb3 -Wall -Wno-unused -Wno-maybe-uninitialized -Wno-discarded-qualifiers -Wl,--wrap=malloc,--wrap=free,--wrap=calloc,--wrap=realloc,--wrap=memcpy,--wrap=memset,--wrap=memmove,--wrap=memcmp,--wrap=strlen,--wrap=strcmp,--wrap=strncmp -I$NIM_LIBS -o $binfile $NIMAPI $files
popd
