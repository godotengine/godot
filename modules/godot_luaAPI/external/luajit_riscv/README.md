# LJRV - LuaJIT RISC-V 64 Port

LuaJIT is a Just-In-Time (JIT) compiler for the Lua programming language,
RISC-V is a free and open ISA enabling a new era of processor innovation.

## Introduction

LJRV is a ongoing porting project of LuaJIT to the RISC-V 64-bit architecture by PLCT Lab, ISCAS.
The ultimate goal is to provide a RISC-V 64 LuaJIT implementation and have it upstreamed to the official LuaJIT repository.

## Progress

- [x] Interpreter Runtime
- [x] JIT Compiler

LJRV is still of beta quality, particularly the JIT compiler.
For production usage, we suggests disable the JIT compiler during compilation by setting `XCFLAGS+= -DLUAJIT_DISABLE_JIT` in Makefile or environment variable.

## Bug Report

Please report bugs to [Issues](https://github.com/ruyisdk/LuaJIT/issues).

## Copyright

LuaJIT is Copyright (C) 2005-2023 Mike Pall.
LuaJIT is free software, released under the MIT license.
See full Copyright Notice in the COPYRIGHT file or in luajit.h.

LJRV is Copyright (C) 2022-2023 PLCT Lab, ISCAS. Contributed by gns.
LJRV is free software, released under the MIT license.
LJRV is part of RuyiSDK.
