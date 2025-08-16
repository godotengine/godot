#!/bin/bash
set -e

GCC_TRIPLE="riscv64-linux-gnu"
CXX="$GCC_TRIPLE-g++"

# Try to detect a few common RISC-V compilers
if command -v "riscv64-unknown-elf-g++" &> /dev/null; then
	echo "* Building game scripts with Newlib RISC-V compiler"
	GCC_TRIPLE="riscv64-unknown-elf"
	export CXX="ccache $GCC_TRIPLE-g++"

elif command -v "riscv64-linux-gnu-g++-14" &> /dev/null; then
	echo "* Building game scripts with system GCC/glibc compiler"
	GCC_TRIPLE="riscv64-linux-gnu"
	export CXX="ccache $GCC_TRIPLE-g++-14"

elif command -v "riscv64-linux-gnu-g++-12" &> /dev/null; then
	echo "* Building game scripts with system GCC/glibc compiler"
	GCC_TRIPLE="riscv64-linux-gnu"
	export CXX="ccache $GCC_TRIPLE-g++-12"

elif command -v "riscv64-linux-gnu-g++-10" &> /dev/null; then
	echo "* Building game scripts with system GCC/glibc compiler"
	GCC_TRIPLE="riscv64-linux-gnu"
	export CXX="ccache $GCC_TRIPLE-g++-10"
fi

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
popd
