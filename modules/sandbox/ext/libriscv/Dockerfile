FROM ubuntu:latest

RUN apt update && apt install -y \
	cmake git \
	clang-18 \
	g++-13-riscv64-linux-gnu

ENV CXX=clang++-18

COPY lib /app/lib
COPY emulator/build.sh /app/emulator/build.sh
COPY emulator/CMakeLists.txt /app/emulator/CMakeLists.txt
COPY emulator/src /app/emulator/src
COPY binaries/measure_mips/fib.c /app/emulator/fib.c

# Fast emulation (with TCC JIT compilation)
WORKDIR /app/emulator
RUN ./build.sh -x --tcc && cp .build/rvlinux /app/rvlinux

# Fastest emulator (with binary translation)
WORKDIR /app/emulator
RUN ./build.sh -x --bintr && cp .build/rvlinux /app/rvlinux-fast

# Clean up
RUN rm -rf /app/emulator/.build

# Example program
WORKDIR /app
RUN riscv64-linux-gnu-gcc-13 -march=rv32g -mabi=ilp32d -static -O2 -nostdlib -ffreestanding emulator/fib.c -o fib

# Provdide a path to your cli apps executable
WORKDIR /app
ENTRYPOINT [ "./rvlinux" ]
