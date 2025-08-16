# RV128 emulator

This folder contains a simple emulator for a 128-bit computer. Use the bigassm project to build executables for this. Because there exists no 128-bit ELF support, the assembler makes up its own class ELFCLASS128. Every address is 128-bit.

## Build

```sh
./build.sh
```

## Run a 128-bit program

```sh
./build/emulator myprogram.elf128
```

## Debug a 128-bit program

```sh
DEBUG=1 ./build/emulator myprogram.elf128
```

This will enable verbose instruction printing.
