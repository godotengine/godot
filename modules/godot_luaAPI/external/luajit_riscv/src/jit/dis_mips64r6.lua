----------------------------------------------------------------------------
-- LuaJIT MIPS64R6 disassembler wrapper module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------
-- This module just exports the r6 big-endian functions from the
-- MIPS disassembler module. All the interesting stuff is there.
------------------------------------------------------------------------------

local dis_mips = require((string.match(..., ".*%.") or "").."dis_mips")
return {
  create = dis_mips.create_r6,
  disass = dis_mips.disass_r6,
  regname = dis_mips.regname
}

