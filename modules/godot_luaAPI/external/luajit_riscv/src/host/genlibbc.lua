----------------------------------------------------------------------------
-- Lua script to dump the bytecode of the library functions written in Lua.
-- The resulting 'buildvm_libbc.h' is used for the build process of LuaJIT.
----------------------------------------------------------------------------
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------

local ffi = require("ffi")
local bit = require("bit")
local vmdef = require("jit.vmdef")
local bcnames = vmdef.bcnames

local format = string.format

local isbe = (string.byte(string.dump(function() end), 5) % 2 == 1)

local function usage(arg)
  io.stderr:write("Usage: ", arg and arg[0] or "genlibbc",
		  " [-o buildvm_libbc.h] lib_*.c\n")
  os.exit(1)
end

local function parse_arg(arg)
  local outfile = "-"
  if not (arg and arg[1]) then
    usage(arg)
  end
  if arg[1] == "-o" then
    outfile = arg[2]
    if not outfile then usage(arg) end
    table.remove(arg, 1)
    table.remove(arg, 1)
  end
  return outfile
end

local function read_files(names)
  local src = ""
  for _,name in ipairs(names) do
    local fp = assert(io.open(name))
    src = src .. fp:read("*a")
    fp:close()
  end
  return src
end

local function transform_lua(code)
  local fixup = {}
  local n = -30000
  code = string.gsub(code, "CHECK_(%w*)%((.-)%)", function(tp, var)
    n = n + 1
    fixup[n] = { "CHECK", tp }
    return format("%s=%d", var, n)
  end)
  code = string.gsub(code, "PAIRS%((.-)%)", function(var)
    fixup.PAIRS = true
    return format("nil, %s, 0x4dp80", var)
  end)
  return "return "..code, fixup
end

local function read_uleb128(p)
  local v = p[0]; p = p + 1
  if v >= 128 then
    local sh = 7; v = v - 128
    repeat
      local r = p[0]
      v = v + bit.lshift(bit.band(r, 127), sh)
      sh = sh + 7
      p = p + 1
    until r < 128
  end
  return p, v
end

-- ORDER LJ_T
local name2itype = {
  str = 5, func = 9, tab = 12, int = 14, num = 15
}

local BC, BCN = {}, {}
for i=0,#bcnames/6-1 do
  local name = bcnames:sub(i*6+1, i*6+6):gsub(" ", "")
  BC[name] = i
  BCN[i] = name
end
local xop, xra = isbe and 3 or 0, isbe and 2 or 1
local xrc, xrb = isbe and 1 or 2, isbe and 0 or 3

local function fixup_dump(dump, fixup)
  local buf = ffi.new("uint8_t[?]", #dump+1, dump)
  local p = buf+5
  local n, sizebc
  p, n = read_uleb128(p)
  local start = p
  p = p + 4
  p = read_uleb128(p)
  p = read_uleb128(p)
  p, sizebc = read_uleb128(p)
  local startbc = tonumber(p - start)
  local rawtab = {}
  for i=0,sizebc-1 do
    local op = p[xop]
    if op == BC.KSHORT then
      local rd = p[xrc] + 256*p[xrb]
      rd = bit.arshift(bit.lshift(rd, 16), 16)
      local f = fixup[rd]
      if f then
	if f[1] == "CHECK" then
	  local tp = f[2]
	  if tp == "tab" then rawtab[p[xra]] = true end
	  p[xop] = tp == "num" and BC.ISNUM or BC.ISTYPE
	  p[xrb] = 0
	  p[xrc] = name2itype[tp]
	else
	  error("unhandled fixup type: "..f[1])
	end
      end
    elseif op == BC.TGETV then
      if rawtab[p[xrb]] then
	p[xop] = BC.TGETR
      end
    elseif op == BC.TSETV then
      if rawtab[p[xrb]] then
	p[xop] = BC.TSETR
      end
    elseif op == BC.ITERC then
      if fixup.PAIRS then
	p[xop] = BC.ITERN
      end
    end
    p = p + 4
  end
  local ndump = ffi.string(start, n)
  -- Fixup hi-part of 0x4dp80 to LJ_KEYINDEX.
  ndump = ndump:gsub("\x80\x80\xcd\xaa\x04", "\xff\xff\xf9\xff\x0f")
  return { dump = ndump, startbc = startbc, sizebc = sizebc }
end

local function find_defs(src)
  local defs = {}
  for name, code in string.gmatch(src, "LJLIB_LUA%(([^)]*)%)%s*/%*(.-)%*/") do
    local env = {}
    local tcode, fixup = transform_lua(code)
    local func = assert(load(tcode, "", nil, env))()
    defs[name] = fixup_dump(string.dump(func, true), fixup)
    defs[#defs+1] = name
  end
  return defs
end

local function gen_header(defs)
  local t = {}
  local function w(x) t[#t+1] = x end
  w("/* This is a generated file. DO NOT EDIT! */\n\n")
  w("static const int libbc_endian = ") w(isbe and 1 or 0) w(";\n\n")
  local s, sb = "", ""
  for i,name in ipairs(defs) do
    local d = defs[name]
    s = s .. d.dump
    sb = sb .. string.char(i) .. ("\0"):rep(d.startbc - 1)
	    .. (isbe and "\0\0\0\255" or "\255\0\0\0"):rep(d.sizebc)
	    .. ("\0"):rep(#d.dump - d.startbc - d.sizebc*4)
  end
  w("static const uint8_t libbc_code[] = {\n")
  local n = 0
  for i=1,#s do
    local x = string.byte(s, i)
    local xb = string.byte(sb, i)
    if xb == 255 then
      local name = BCN[x]
      local m = #name + 4
      if n + m > 78 then n = 0; w("\n") end
      n = n + m
      w("BC_"); w(name)
    else
      local m = x < 10 and 2 or (x < 100 and 3 or 4)
      if xb == 0 then
	if n + m > 78 then n = 0; w("\n") end
      else
	local name = defs[xb]:gsub("_", ".")
	if n ~= 0 then w("\n") end
	w("/* "); w(name); w(" */ ")
	n = #name + 7
      end
      n = n + m
      w(x)
    end
    w(",")
  end
  w("\n0\n};\n\n")
  w("static const struct { const char *name; int ofs; } libbc_map[] = {\n")
  local m = 0
  for _,name in ipairs(defs) do
    w('{"'); w(name); w('",'); w(m) w('},\n')
    m = m + #defs[name].dump
  end
  w("{NULL,"); w(m); w("}\n};\n\n")
  return table.concat(t)
end

local function write_file(name, data)
  if name == "-" then
    assert(io.write(data))
    assert(io.flush())
  else
    local fp = io.open(name)
    if fp then
      local old = fp:read("*a")
      fp:close()
      if data == old then return end
    end
    fp = assert(io.open(name, "w"))
    assert(fp:write(data))
    assert(fp:close())
  end
end

local outfile = parse_arg(arg)
local src = read_files(arg)
local defs = find_defs(src)
local hdr = gen_header(defs)
write_file(outfile, hdr)

