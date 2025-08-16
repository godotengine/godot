#riscv32-unknown-elf-gcc -O2 -static -ffreestanding -nostdlib fib.c -o fib
#riscv32-unknown-elf-gcc -O2 -static -ffreestanding -nostdlib va_fib.c -o va_fib
clang-18 -target riscv64-linux-gnu -march=rv32g -mabi=ilp32d -O3 -static -nostdlib -ffreestanding fib.c -o fib
clang-18 -target riscv64-linux-gnu -march=rv32g -mabi=ilp32d -O3 -static -nostdlib -ffreestanding va_fib.c -o va_fib
