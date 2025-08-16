# Advanced API example

This example shows how to create an API using _libriscv_.

[example.cpp](src/example.cpp) contains the host-side code that:
1. Registers some host functions
2. Sets up a RISC-V emulator with a Linux environment
3. Executes the program

While [micro.c](riscv_program/micro.c) contains the guest code that:
1. Creates a macro helper to place some host function wrappers
2. Defines host functions 0, 1 and 2
3. Prints a blurb and executes each host function

After execution, if host function 2 has been used, the function pointer will get called.

