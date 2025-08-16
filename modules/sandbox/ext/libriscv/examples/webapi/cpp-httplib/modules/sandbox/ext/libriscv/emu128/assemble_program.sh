#!/usr/bin/env bash
set -e
riscv64-unknown-elf-as program.asm -o /tmp/program.o
riscv64-unknown-elf-ld -T linker.ld -Ttext 0x100000 -o /tmp/program.elf /tmp/program.o
riscv64-unknown-elf-objdump -drl /tmp/program.elf
riscv64-unknown-elf-objcopy -O binary -j .text /tmp/program.elf /tmp/program.bin
xxd -i /tmp/program.bin > src/program.h
