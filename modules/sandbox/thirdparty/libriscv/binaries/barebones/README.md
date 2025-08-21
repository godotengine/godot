Barebones RISC-V binaries
===========================

To run this you will have to use the _rvmicro emulator_. It activates a non-standard system call implementation specially made for these tiny binaries. It is intended to be usable for sandboxed C/C++ program execution.

## Toolchain for Clang builds

If you want to build the Clang versions, you will need a new version of the compiler. Edit the build32_clang.sh and build64_clang.sh files to match your clang version.

Then, download the latest embedded RISC-V packages from https://github.com/xpack-dev-tools/riscv-none-embed-gcc-xpack/releases/. In my case, it contains GCC 12.1.0, so I will edit my `micro_toolchain.cmake` file to reflect where I extract the toolchain and the compiler version, set in the GCC_VERSION CMake variable.

As I was writing this, I had extracted `xpack-riscv-none-embed-gcc-12.1.0-2` to my home folder, and it contains embedded RISC-V GCC 12.1.0, so I `set(GCC_VERSION 12.1.0)` in `micro_toolchain.cmake` in the same directory as this README. That allowed me to build the Clang variants.

The newlib variants require the RISC-V GNU toolchain from https://github.com/riscv-collab/riscv-gnu-toolchain. The README on their repository explains all the necessary details. Just make sure to build for Newlib. Preferrably rv64g and rv32g respectively, for performance reasons.

## Accelerated runtimes

Has a tiny libc mode. Enough for simple optimized C++. If you enable newlib mode you will get a full standard library, except threads. In both modes functions like `malloc`, `free`, `memcpy`, `memcmp`, `strlen` and many others are implemented as system calls for better performance. The tiny libc is a bit of a mess but it can easily be improved upon as the barebones C standard functions are easy to implement.

The linux64 program takes 11 milliseconds to complete on my machine. The barebones examples both take less than 2 milliseconds to complete, and they do a lot more work testing the multi-threading.

Have a look at `libc/heap.hpp` for the syscall numbers and calling.

## Tiny builds

A minimal 32-bit tiny libc build with MINIMAL, LTO and GCSECTIONS enabled yields a _7kB binary_ that uses 272kB memory.

A minimal 64-bit Newlib build with MINIMAL, LTO and GCSECTIONS enabled yields a _112kB binary_ that uses 196kB memory.

Both executables have been stripped. The reason for the difference in memory use is likely that newlib puts more things in .rodata which is heavily optimized in the emulator. Most rodata pages re-use the binary instead of duplicating the data, and it doesn't count towards the memory usage for various reasons. rodata is shared between all forks of the machine, and it makes sense to not count memory that is only required for the main VM.
