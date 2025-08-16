----------------------------------------------------------------------------
-- LuaJIT RISC-V 64 disassembler wrapper module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------
-- This module just exports the default riscv little-endian functions from the
-- RISC-V disassembler module. All the interesting stuff is there.
------------------------------------------------------------------------------

local dis_riscv = require((string.match(..., ".*%.") or "").."dis_riscv")
return {
  create = dis_riscv.create,
  disass = dis_riscv.disass,
  regname = dis_riscv.regname
}