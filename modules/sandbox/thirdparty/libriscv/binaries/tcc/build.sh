# TCC can be built to produce 64-bit RISC-V binaries like so:
# ./configure --enable-cross --cpu=riscv64 --triplet=riscv64-linux-gnu --config-backtrace=no
# make
# Change this folder to wherever you have TCC repository:
TCCDIR=$HOME/github/tcc
set -e

mkdir -p build
# TCC RISC-V assembly does not work in any way, so we have to use GAS
riscv64-linux-gnu-as api.s -o build/api.o
$TCCDIR/tcc -static -nostdlib -nostdinc -I$TCCDIR/include -O2 -std=c11 -o build/test test.c build/api.o

ls -la build/test
