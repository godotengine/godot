----------------------------------------------------------------------------
-- LuaJIT ARM64BE disassembler wrapper module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------
-- ARM64 instructions are always little-endian. So just forward to the
-- common ARM64 disassembler module. All the interesting stuff is there.
------------------------------------------------------------------------------

return require((string.match(..., ".*%.") or "").."dis_arm64")

