#!/usr/bin/env bash
set -e

OPTS=""
EMBED_FILES=""

function usage()
{
   progname=$(basename $0)
   cat << HEREDOC

   Usage: $progname [options] [--embed file] [--embed file] ...

   optional arguments:
     -h, --help           show this help message and exit
     --debug              enable debugging
     --debinfo            enable debugging information
     --defaults           build with default settings
     --perf               build settings for performance profiling
     --no-perf            disable performance profiling
     --static             build static CLI executable
     --no-static          build dynamic CLI executable
     --native             build with -march=native
     --no-native          disable -march=native
     --lto                build with link-time optimization
     --no-lto             disable link-time optimization
     --A                  enable atomic extension
     --no-A               disable atomic extension
     --C                  enable compressed extension
     --no-C               disable compressed extension
     --V                  enable vector extension
     --no-V               disable vector extension
     --32                 enable RV32
     --no-32              disable RV32
     --64                 enable RV64
     --no-64              disable RV64
     --128                enable RV128
     --no-128             disable RV128
     --assembly-dispatch  enable assembly dispatch (experimental feature)
     --no-assembly-dispatch disable assembly dispatch
     --tailcall-dispatch  enable tailcall dispatch
     --no-tailcall-dispatch disable tailcall dispatch
     -b, --bintr          enable binary translation using system compiler
     --no-bintr           disable binary translation
     -t, --jit            jit-compile using tcc
     --no-jit             disable jit-compile using tcc
     -x, --expr           enable experimental features (eg. unbounded 32-bit addressing)
     -N bits              enable N-bits of masked address space (experimental feature)
     --no-expr            disable experimental features
     --embed FILE         embed binary translated sources into the emulator, produced by CLI -o option
     -v, --verbose        increase the verbosity of the bash script

HEREDOC
}

embed_all()
{
	for file in $(ls -1 *.cpp); do
		EMBED_FILES="$EMBED_FILES;$file"
	done
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
		-h|--help ) usage; exit; ;;
		--debug) OPTS="$OPTS -DCMAKE_BUILD_TYPE=Debug" ;;
		--debinfo) OPTS="$OPTS -DCMAKE_BUILD_TYPE=RelWithDebInfo" ;;
		--defaults) OPTS="$OPTS -DCMAKE_BUILD_TYPE=Release -DRISCV_EXT_A=ON -DRISCV_EXT_C=ON -DRISCV_EXT_V=OFF -DRISCV_32I=ON -DRISCV_64I=ON -DRISCV_128I=OFF -DRISCV_BINARY_TRANSLATION=OFF -DRISCV_LIBTCC=OFF -DRISCV_EXPERIMENTAL=OFF -DRISCV_BINARY_TRANSLATION=OFF -DRISCV_LIBTCC=OFF -DLTO=OFF -DPERF=OFF -DSTATIC_BUILD=OFF -DNATIVE=OFF" ;;
		--perf) OPTS="$OPTS -DPERF=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo" ;;
		--no-perf) OPTS="$OPTS -DPERF=OFF" ;;
		--static) OPTS="$OPTS -DSTATIC_BUILD=ON" ;;
		--no-static) OPTS="$OPTS -DSTATIC_BUILD=OFF" ;;
		--native) OPTS="$OPTS -DNATIVE=ON" ;;
		--no-native) OPTS="$OPTS -DNATIVE=OFF" ;;
		--lto) OPTS="$OPTS -DLTO=ON" ;;
		--no-lto) OPTS="$OPTS -DLTO=OFF" ;;
		--A) OPTS="$OPTS -DRISCV_EXT_A=ON" ;;
		--no-A) OPTS="$OPTS -DRISCV_EXT_A=OFF" ;;
		--C) OPTS="$OPTS -DRISCV_EXT_C=ON" ;;
		--no-C) OPTS="$OPTS -DRISCV_EXT_C=OFF" ;;
		--V) OPTS="$OPTS -DRISCV_EXT_V=ON" ;;
		--no-V) OPTS="$OPTS -DRISCV_EXT_V=OFF" ;;
		--32) OPTS="$OPTS -DRISCV_32I=ON" ;;
		--no-32) OPTS="$OPTS -DRISCV_32I=OFF" ;;
		--64) OPTS="$OPTS -DRISCV_64I=ON" ;;
		--no-64) OPTS="$OPTS -DRISCV_64I=OFF" ;;
		--128) OPTS="$OPTS -DRISCV_128I=ON" ;;
		--no-128) OPTS="$OPTS -DRISCV_128I=OFF" ;;
		--assembly-dispatch) OPTS="$OPTS -DRISCV_ASM_DISPATCH=ON" ;;
		--no-assembly-dispatch) OPTS="$OPTS -DRISCV_ASM_DISPATCH=OFF" ;;
		--tailcall-dispatch) OPTS="$OPTS -DRISCV_TAILCALL_DISPATCH=ON" ;;
		--no-tailcall-dispatch) OPTS="$OPTS -DRISCV_TAILCALL_DISPATCH=OFF" ;;
        -b|--bintr) OPTS="$OPTS -DRISCV_BINARY_TRANSLATION=ON -DRISCV_LIBTCC=OFF" ;;
		--no-bintr) OPTS="$OPTS -DRISCV_BINARY_TRANSLATION=OFF" ;;
		--jit|--tcc  ) OPTS="$OPTS -DRISCV_BINARY_TRANSLATION=ON -DRISCV_LIBTCC=ON" ;;
		--no-jit|--no-tcc) OPTS="$OPTS -DRISCV_LIBTCC=OFF" ;;
        -x|--expr ) OPTS="$OPTS -DRISCV_EXPERIMENTAL=ON -DRISCV_ENCOMPASSING_ARENA=ON" ;;
		-N) OPTS="$OPTS -DRISCV_EXPERIMENTAL=ON -DRISCV_ENCOMPASSING_ARENA=ON -DRISCV_ENCOMPASSING_ARENA_BITS=$2"; shift ;;
        --no-expr ) OPTS="$OPTS -DRISCV_EXPERIMENTAL=OFF" ;;
		--embed) EMBED_FILES="$EMBED_FILES;$2"; shift ;;
		--embed-all) embed_all ;;
		-v|--verbose ) set -x ;;
		*) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release $OPTS -DEMBED_FILES="$EMBED_FILES"
make -j6
popd

if test -f ".build/rvmicro"; then
	ln -fs .build/rvmicro .
fi
if test -f ".build/rvnewlib"; then
	ln -fs .build/rvnewlib .
fi
if test -f ".build/libtcc1.a"; then
	ln -fs .build/libtcc1.a .
fi
ln -fs .build/rvlinux .
