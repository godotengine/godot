----------------------------------------------------------------------------
-- Lua script to embed the rolling release version in luajit.h.
----------------------------------------------------------------------------
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------

local arg = {...}
local FILE_ROLLING_H = arg[1] or "luajit_rolling.h"
local FILE_RELVER_TXT = arg[2] or "luajit_relver.txt"
local FILE_LUAJIT_H = arg[3] or "luajit.h"

local function file_read(file)
  local fp = assert(io.open(file, "rb"), "run from the wrong directory")
  local data = assert(fp:read("*a"))
  fp:close()
  return data
end

local function file_write_mod(file, data)
  local fp = io.open(file, "rb")
  if fp then
    local odata = assert(fp:read("*a"))
    fp:close()
    if odata == data then return end
  end
  fp = assert(io.open(file, "wb"))
  assert(fp:write(data))
  assert(fp:close())
end

local text = file_read(FILE_ROLLING_H)
local relver = file_read(FILE_RELVER_TXT):match("(%d+)")

if relver then
  text = text:gsub("ROLLING", relver)
else
  io.stderr:write([[
**** WARNING Cannot determine rolling release version from git log.
**** WARNING The 'git' command must be available during the build.
]])
  file_write_mod(FILE_RELVER_TXT, "ROLLING\n") -- Fallback for install target.
end

file_write_mod(FILE_LUAJIT_H, text)
