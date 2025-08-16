riscv64-linux-gnu-as fib64.S -o fib64.o
riscv64-linux-gnu-ld -Ttext 200000 -o fib64 fib64.o
rm -f fib64.o
