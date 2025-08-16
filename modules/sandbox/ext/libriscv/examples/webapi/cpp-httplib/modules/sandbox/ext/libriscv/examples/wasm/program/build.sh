#!/bin/bash
LUAJIT=$PWD/LuaJIT

riscv64-unknown-elf-g++ -O2 -static \
  main.cpp \
  env.cpp \
  -L$LUAJIT/src \
  -l:libluajit.a \
  -I$LUAJIT/src \
  -o program.elf \
  -Wl,--undefined=run,--undefined=compile \
  -Wl,--wrap=memcpy,--wrap=memset,--wrap=memcmp,--wrap=memmove \
  -Wl,--wrap=malloc,--wrap=free,--wrap=calloc,--wrap=realloc \
  -Wl,--wrap=strlen,--wrap=strcmp,--wrap=strncmp

riscv64-unknown-elf-strip --strip-unneeded program.elf -K run -K compile
