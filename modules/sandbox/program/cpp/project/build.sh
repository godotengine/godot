#!/bin/bash
export CC="zig;cc;-target riscv64-linux-musl"
export CXX="zig;c++;-target riscv64-linux-musl"
ARGS=""
BTYPE="Release"

# Find zig
if ! command -v zig &> /dev/null
then
	echo "zig could not be found globally. Please install zig and try again."
	exit
fi

usage()
{
	echo "Usage: $0 [options]"
	echo
	echo "Options:"
	echo "  -h, --help      Display this help and exit"
	echo "  --runtime-api   Download a run-time generated Godot API header"
	echo "  --no-runtime-api Do not download a run-time generated Godot API header"
	echo "  --debug         Build with debug symbols"
	echo "  --debinfo       Build with debug info"
	echo "  --strip         Strip the binary"
	echo "  --no-strip      Do not strip the binary"
	echo "  --single        Build with single precision vectors"
	echo "  --double        Build with double precision vectors"
	echo "  --C             Enable RISC-V C extension (default)"
	echo "  --no-C          Disable RISC-V C extension"
	echo "  --toolchain     Specify a custom toolchain file"
	echo "  --verbose       Enable verbose build"
	echo
}

# Command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
		-h|--help ) usage; exit; ;;
		--runtime-api) ARGS="$ARGS -DDOWNLOAD_RUNTIME_API=ON" ;;
		--no-runtime-api) ARGS="$ARGS -DDOWNLOAD_RUNTIME_API=OFF" ;;
		--debug) BTYPE="Debug" ;;
		--debinfo) BTYPE="RelWithDebInfo" ;;
		--strip) ARGS="$ARGS -DSTRIPPED=ON" ;;
		--no-strip) ARGS="$ARGS -DSTRIPPED=OFF" ;;
		--single) ARGS="$ARGS -DDOUBLE_PRECISION=OFF" ;;
		--double) ARGS="$ARGS -DDOUBLE_PRECISION=ON" ;;
		--C) ARGS="$ARGS -DSANDBOX_RISCV_EXT_C=ON" ;;
		--no-C) ARGS="$ARGS -DSANDBOX_RISCV_EXT_C=OFF" ;;
		--toolchain) ARGS="$ARGS -DCMAKE_TOOLCHAIN_FILE=$2"; shift ;;
		--verbose) ARGS="$ARGS -DCMAKE_VERBOSE_MAKEFILE=ON" ;;
		*) echo "Unknown parameter passed: $1"; exit 1 ;;
	esac
	shift
done

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=$BTYPE $ARGS -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
make -j8
popd
