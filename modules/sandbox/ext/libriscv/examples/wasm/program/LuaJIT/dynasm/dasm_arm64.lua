------------------------------------------------------------------------------
-- DynASM ARM64 module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- See dynasm.lua for full copyright notice.
------------------------------------------------------------------------------

-- Module information:
local _info = {
  arch =	"arm",
  description =	"DynASM ARM64 module",
  version =	"1.5.0",
  vernum =	 10500,
  release =	"2021-05-02",
  author =	"Mike Pall",
  license =	"MIT",
}

-- Exported glue functions for the arch-specific module.
local _M = { _info = _info }

-- Cache library functions.
local type, tonumber, pairs, ipairs = type, tonumber, pairs, ipairs
local assert, setmetatable, rawget = assert, setmetatable, rawget
local _s = string
local format, byte, char = _s.format, _s.byte, _s.char
local match, gmatch, gsub = _s.match, _s.gmatch, _s.gsub
local concat, sort, insert = table.concat, table.sort, table.insert
local bit = bit or require("bit")
local band, shl, shr, sar = bit.band, bit.lshift, bit.rshift, bit.arshift
local ror, tohex, tobit = bit.ror, bit.tohex, bit.tobit

-- Inherited tables and callbacks.
local g_opt, g_arch
local wline, werror, wfatal, wwarn

-- Action name list.
-- CHECK: Keep this in sync with the C code!
local action_names = {
  "STOP", "SECTION", "ESC", "REL_EXT",
  "ALIGN", "REL_LG", "LABEL_LG",
  "REL_PC", "LABEL_PC", "REL_A",
  "IMM", "IMM6", "IMM12", "IMM13W", "IMM13X", "IMML", "IMMV",
  "VREG",
}

-- Maximum number of section buffer positions for dasm_put().
-- CHECK: Keep this in sync with the C code!
local maxsecpos = 25 -- Keep this low, to avoid excessively long C lines.

-- Action name -> action number.
local map_action = {}
for n,name in ipairs(action_names) do
  map_action[name] = n-1
end

-- Action list buffer.
local actlist = {}

-- Argument list for next dasm_put(). Start with offset 0 into action list.
local actargs = { 0 }

-- Current number of section buffer positions for dasm_put().
local secpos = 1

------------------------------------------------------------------------------

-- Dump action names and numbers.
local function dumpactions(out)
  out:write("DynASM encoding engine action codes:\n")
  for n,name in ipairs(action_names) do
    local num = map_action[name]
    out:write(format("  %-10s %02X  %d\n", name, num, num))
  end
  out:write("\n")
end

-- Write action list buffer as a huge static C array.
local function writeactions(out, name)
  local nn = #actlist
  if nn == 0 then nn = 1; actlist[0] = map_action.STOP end
  out:write("static const unsigned int ", name, "[", nn, "] = {\n")
  for i = 1,nn-1 do
    assert(out:write("0x", tohex(actlist[i]), ",\n"))
  end
  assert(out:write("0x", tohex(actlist[nn]), "\n};\n\n"))
end

------------------------------------------------------------------------------

-- Add word to action list.
local function wputxw(n)
  assert(n >= 0 and n <= 0xffffffff and n % 1 == 0, "word out of range")
  actlist[#actlist+1] = n
end

-- Add action to list with optional arg. Advance buffer pos, too.
local function waction(action, val, a, num)
  local w = assert(map_action[action], "bad action name `"..action.."'")
  wputxw(w * 0x10000 + (val or 0))
  if a then actargs[#actargs+1] = a end
  if a or num then secpos = secpos + (num or 1) end
end

-- Flush action list (intervening C code or buffer pos overflow).
local function wflush(term)
  if #actlist == actargs[1] then return end -- Nothing to flush.
  if not term then waction("STOP") end -- Terminate action list.
  wline(format("dasm_put(Dst, %s);", concat(actargs, ", ")), true)
  actargs = { #actlist } -- Actionlist offset is 1st arg to next dasm_put().
  secpos = 1 -- The actionlist offset occupies a buffer position, too.
end

-- Put escaped word.
local function wputw(n)
  if n <= 0x000fffff then waction("ESC") end
  wputxw(n)
end

-- Reserve position for word.
local function wpos()
  local pos = #actlist+1
  actlist[pos] = ""
  return pos
end

-- Store word to reserved position.
local function wputpos(pos, n)
  assert(n >= 0 and n <= 0xffffffff and n % 1 == 0, "word out of range")
  if n <= 0x000fffff then
    insert(actlist, pos+1, n)
    n = map_action.ESC * 0x10000
  end
  actlist[pos] = n
end

------------------------------------------------------------------------------

-- Global label name -> global label number. With auto assignment on 1st use.
local next_global = 20
local map_global = setmetatable({}, { __index = function(t, name)
  if not match(name, "^[%a_][%w_]*$") then werror("bad global label") end
  local n = next_global
  if n > 2047 then werror("too many global labels") end
  next_global = n + 1
  t[name] = n
  return n
end})

-- Dump global labels.
local function dumpglobals(out, lvl)
  local t = {}
  for name, n in pairs(map_global) do t[n] = name end
  out:write("Global labels:\n")
  for i=20,next_global-1 do
    out:write(format("  %s\n", t[i]))
  end
  out:write("\n")
end

-- Write global label enum.
local function writeglobals(out, prefix)
  local t = {}
  for name, n in pairs(map_global) do t[n] = name end
  out:write("enum {\n")
  for i=20,next_global-1 do
    out:write("  ", prefix, t[i], ",\n")
  end
  out:write("  ", prefix, "_MAX\n};\n")
end

-- Write global label names.
local function writeglobalnames(out, name)
  local t = {}
  for name, n in pairs(map_global) do t[n] = name end
  out:write("static const char *const ", name, "[] = {\n")
  for i=20,next_global-1 do
    out:write("  \"", t[i], "\",\n")
  end
  out:write("  (const char *)0\n};\n")
end

------------------------------------------------------------------------------

-- Extern label name -> extern label number. With auto assignment on 1st use.
local next_extern = 0
local map_extern_ = {}
local map_extern = setmetatable({}, { __index = function(t, name)
  -- No restrictions on the name for now.
  local n = next_extern
  if n > 2047 then werror("too many extern labels") end
  next_extern = n + 1
  t[name] = n
  map_extern_[n] = name
  return n
end})

-- Dump extern labels.
local function dumpexterns(out, lvl)
  out:write("Extern labels:\n")
  for i=0,next_extern-1 do
    out:write(format("  %s\n", map_extern_[i]))
  end
  out:write("\n")
end

-- Write extern label names.
local function writeexternnames(out, name)
  out:write("static const char *const ", name, "[] = {\n")
  for i=0,next_extern-1 do
    out:write("  \"", map_extern_[i], "\",\n")
  end
  out:write("  (const char *)0\n};\n")
end

------------------------------------------------------------------------------

-- Arch-specific maps.

-- Ext. register name -> int. name.
local map_archdef = { xzr = "@x31", wzr = "@w31", lr = "x30", }

-- Int. register name -> ext. name.
local map_reg_rev = { ["@x31"] = "xzr", ["@w31"] = "wzr", x30 = "lr", }

local map_type = {}		-- Type name -> { ctype, reg }
local ctypenum = 0		-- Type number (for Dt... macros).

-- Reverse defines for registers.
function _M.revdef(s)
  return map_reg_rev[s] or s
end

local map_shift = { lsl = 0, lsr = 1, asr = 2, }

local map_extend = {
  uxtb = 0, uxth = 1, uxtw = 2, uxtx = 3,
  sxtb = 4, sxth = 5, sxtw = 6, sxtx = 7,
}

local map_cond = {
  eq = 0, ne = 1, cs = 2, cc = 3, mi = 4, pl = 5, vs = 6, vc = 7,
  hi = 8, ls = 9, ge = 10, lt = 11, gt = 12, le = 13, al = 14,
  hs = 2, lo = 3,
}

------------------------------------------------------------------------------

local parse_reg_type

local function parse_reg(expr, shift, no_vreg)
  if not expr then werror("expected register name") end
  local tname, ovreg = match(expr, "^([%w_]+):(@?%l%d+)$")
  if not tname then
    tname, ovreg = match(expr, "^([%w_]+):(R[xwqdshb]%b())$")
  end
  local tp = map_type[tname or expr]
  if tp then
    local reg = ovreg or tp.reg
    if not reg then
      werror("type `"..(tname or expr).."' needs a register override")
    end
    expr = reg
  end
  local ok31, rt, r = match(expr, "^(@?)([xwqdshb])([123]?[0-9])$")
  if r then
    r = tonumber(r)
    if r <= 30 or (r == 31 and ok31 ~= "" or (rt ~= "w" and rt ~= "x")) then
      if not parse_reg_type then
	parse_reg_type = rt
      elseif parse_reg_type ~= rt then
	werror("register size mismatch")
      end
      return shl(r, shift), tp
    end
  end
  local vrt, vreg = match(expr, "^R([xwqdshb])(%b())$")
  if vreg then
    if not parse_reg_type then
      parse_reg_type = vrt
    elseif parse_reg_type ~= vrt then
      werror("register size mismatch")
    end
    if not no_vreg then waction("VREG", shift, vreg) end
    return 0
  end
  werror("bad register name `"..expr.."'")
end

local function parse_reg_base(expr)
  if expr == "sp" then return 0x3e0 end
  local base, tp = parse_reg(expr, 5)
  if parse_reg_type ~= "x" then werror("bad register type") end
  parse_reg_type = false
  return base, tp
end

local parse_ctx = {}

local loadenv = setfenv and function(s)
  local code = loadstring(s, "")
  if code then setfenv(code, parse_ctx) end
  return code
end or function(s)
  return load(s, "", nil, parse_ctx)
end

-- Try to parse simple arithmetic, too, since some basic ops are aliases.
local function parse_number(n)
  local x = tonumber(n)
  if x then return x end
  local code = loadenv("return "..n)
  if code then
    local ok, y = pcall(code)
    if ok and type(y) == "number" then return y end
  end
  return nil
end

local function parse_imm(imm, bits, shift, scale, signed)
  imm = match(imm, "^#(.*)$")
  if not imm then werror("expected immediate operand") end
  local n = parse_number(imm)
  if n then
    local m = sar(n, scale)
    if shl(m, scale) == n then
      if signed then
	local s = sar(m, bits-1)
	if s == 0 then return shl(m, shift)
	elseif s == -1 then return shl(m + shl(1, bits), shift) end
      else
	if sar(m, bits) == 0 then return shl(m, shift) end
      end
    end
    werror("out of range immediate `"..imm.."'")
  else
    waction("IMM", (signed and 32768 or 0)+scale*1024+bits*32+shift, imm)
    return 0
  end
end

local function parse_imm12(imm)
  imm = match(imm, "^#(.*)$")
  if not imm then werror("expected immediate operand") end
  local n = parse_number(imm)
  if n then
    if shr(n, 12) == 0 then
      return shl(n, 10)
    elseif band(n, 0xff000fff) == 0 then
      return shr(n, 2) + 0x00400000
    end
    werror("out of range immediate `"..imm.."'")
  else
    waction("IMM12", 0, imm)
    return 0
  end
end

local function parse_imm13(imm)
  imm = match(imm, "^#(.*)$")
  if not imm then werror("expected immediate operand") end
  local n = parse_number(imm)
  local r64 = parse_reg_type == "x"
  if n and n % 1 == 0 and n >= 0 and n <= 0xffffffff then
    local inv = false
    if band(n, 1) == 1 then n = bit.bnot(n); inv = true end
    local t = {}
    for i=1,32 do t[i] = band(n, 1); n = shr(n, 1) end
    local b = table.concat(t)
    b = b..(r64 and (inv and "1" or "0"):rep(32) or b)
    local p0, p1, p0a, p1a = b:match("^(0+)(1+)(0*)(1*)")
    if p0 then
      local w = p1a == "" and (r64 and 64 or 32) or #p1+#p0a
      if band(w, w-1) == 0 and b == b:sub(1, w):rep(64/w) then
	local s = band(-2*w, 0x3f) - 1
	if w == 64 then s = s + 0x1000 end
	if inv then
	  return shl(w-#p1-#p0, 16) + shl(s+w-#p1, 10)
	else
	  return shl(w-#p0, 16) + shl(s+#p1, 10)
	end
      end
    end
    werror("out of range immediate `"..imm.."'")
  elseif r64 then
    waction("IMM13X", 0, format("(unsigned int)(%s)", imm))
    actargs[#actargs+1] = format("(unsigned int)((unsigned long long)(%s)>>32)", imm)
    return 0
  else
    waction("IMM13W", 0, imm)
    return 0
  end
end

local function parse_imm6(imm)
  imm = match(imm, "^#(.*)$")
  if not imm then werror("expected immediate operand") end
  local n = parse_number(imm)
  if n then
    if n >= 0 and n <= 63 then
      return shl(band(n, 0x1f), 19) + (n >= 32 and 0x80000000 or 0)
    end
    werror("out of range immediate `"..imm.."'")
  else
    waction("IMM6", 0, imm)
    return 0
  end
end

local function parse_imm_load(imm, scale)
  local n = parse_number(imm)
  if n then
    local m = sar(n, scale)
    if shl(m, scale) == n and m >= 0 and m < 0x1000 then
      return shl(m, 10) + 0x01000000 -- Scaled, unsigned 12 bit offset.
    elseif n >= -256 and n < 256 then
      return shl(band(n, 511), 12) -- Unscaled, signed 9 bit offset.
    end
    werror("out of range immediate `"..imm.."'")
  else
    waction("IMML", scale, imm)
    return 0
  end
end

local function parse_fpimm(imm)
  imm = match(imm, "^#(.*)$")
  if not imm then werror("expected immediate operand") end
  local n = parse_number(imm)
  if n then
    local m, e = math.frexp(n)
    local s, e2 = 0, band(e-2, 7)
    if m < 0 then m = -m; s = 0x00100000 end
    m = m*32-16
    if m % 1 == 0 and m >= 0 and m <= 15 and sar(shl(e2, 29), 29)+2 == e then
      return s + shl(e2, 17) + shl(m, 13)
    end
    werror("out of range immediate `"..imm.."'")
  else
    werror("NYI fpimm action")
  end
end

local function parse_shift(expr)
  local s, s2 = match(expr, "^(%S+)%s*(.*)$")
  s = map_shift[s]
  if not s then werror("expected shift operand") end
  return parse_imm(s2, 6, 10, 0, false) + shl(s, 22)
end

local function parse_lslx16(expr)
  local n = match(expr, "^lsl%s*#(%d+)$")
  n = tonumber(n)
  if not n then werror("expected shift operand") end
  if band(n, parse_reg_type == "x" and 0xffffffcf or 0xffffffef) ~= 0 then
    werror("bad shift amount")
  end
  return shl(n, 17)
end

local function parse_extend(expr)
  local s, s2 = match(expr, "^(%S+)%s*(.*)$")
  if s == "lsl" then
    s = parse_reg_type == "x" and 3 or 2
  else
    s = map_extend[s]
  end
  if not s then werror("expected extend operand") end
  return (s2 == "" and 0 or parse_imm(s2, 3, 10, 0, false)) + shl(s, 13)
end

local function parse_cond(expr, inv)
  local c = map_cond[expr]
  if not c then werror("expected condition operand") end
  return shl(bit.bxor(c, inv), 12)
end

local function parse_load(params, nparams, n, op)
  if params[n+2] then werror("too many operands") end
  local scale = shr(op, 30)
  local pn, p2 = params[n], params[n+1]
  local p1, wb = match(pn, "^%[%s*(.-)%s*%](!?)$")
  if not p1 then
    if not p2 then
      local reg, tailr = match(pn, "^([%w_:]+)%s*(.*)$")
      if reg and tailr ~= "" then
	local base, tp = parse_reg_base(reg)
	if tp then
	  waction("IMML", scale, format(tp.ctypefmt, tailr))
	  return op + base
	end
      end
    end
    werror("expected address operand")
  end
  if p2 then
    if wb == "!" then werror("bad use of '!'") end
    op = op + parse_reg_base(p1) + parse_imm(p2, 9, 12, 0, true) + 0x400
  elseif wb == "!" then
    local p1a, p2a = match(p1, "^([^,%s]*)%s*,%s*(.*)$")
    if not p1a then werror("bad use of '!'") end
    op = op + parse_reg_base(p1a) + parse_imm(p2a, 9, 12, 0, true) + 0xc00
  else
    local p1a, p2a = match(p1, "^([^,%s]*)%s*(.*)$")
    op = op + parse_reg_base(p1a)
    if p2a ~= "" then
      local imm = match(p2a, "^,%s*#(.*)$")
      if imm then
	op = op + parse_imm_load(imm, scale)
      else
	local p2b, p3b, p3s = match(p2a, "^,%s*([^,%s]*)%s*,?%s*(%S*)%s*(.*)$")
	op = op + parse_reg(p2b, 16) + 0x00200800
	if parse_reg_type ~= "x" and parse_reg_type ~= "w" then
	  werror("bad index register type")
	end
	if p3b == "" then
	  if parse_reg_type ~= "x" then werror("bad index register type") end
	  op = op + 0x6000
	else
	  if p3s == "" or p3s == "#0" then
	  elseif p3s == "#"..scale then
	    op = op + 0x1000
	  else
	    werror("bad scale")
	  end
	  if parse_reg_type == "x" then
	    if p3b == "lsl" and p3s ~= "" then op = op + 0x6000
	    elseif p3b == "sxtx" then op = op + 0xe000
	    else
	      werror("bad extend/shift specifier")
	    end
	  else
	    if p3b == "uxtw" then op = op + 0x4000
	    elseif p3b == "sxtw" then op = op + 0xc000
	    else
	      werror("bad extend/shift specifier")
	    end
	  end
	end
      end
    else
      if wb == "!" then werror("bad use of '!'") end
      op = op + 0x01000000
    end
  end
  return op
end

local function parse_load_pair(params, nparams, n, op)
  if params[n+2] then werror("too many operands") end
  local pn, p2 = params[n], params[n+1]
  local scale = 2 + shr(op, 31 - band(shr(op, 26), 1))
  local p1, wb = match(pn, "^%[%s*(.-)%s*%](!?)$")
  if not p1 then
    if not p2 then
      local reg, tailr = match(pn, "^([%w_:]+)%s*(.*)$")
      if reg and tailr ~= "" then
	local base, tp = parse_reg_base(reg)
	if tp then
	  waction("IMM", 32768+7*32+15+scale*1024, format(tp.ctypefmt, tailr))
	  return op + base + 0x01000000
	end
      end
    end
    werror("expected address operand")
  end
  if p2 then
    if wb == "!" then werror("bad use of '!'") end
    op = op + 0x00800000
  else
    local p1a, p2a = match(p1, "^([^,%s]*)%s*,%s*(.*)$")
    if p1a then p1, p2 = p1a, p2a else p2 = "#0" end
    op = op + (wb == "!" and 0x01800000 or 0x01000000)
  end
  return op + parse_reg_base(p1) + parse_imm(p2, 7, 15, scale, true)
end

local function parse_label(label, def)
  local prefix = label:sub(1, 2)
  -- =>label (pc label reference)
  if prefix == "=>" then
    return "PC", 0, label:sub(3)
  end
  -- ->name (global label reference)
  if prefix == "->" then
    return "LG", map_global[label:sub(3)]
  end
  if def then
    -- [1-9] (local label definition)
    if match(label, "^[1-9]$") then
      return "LG", 10+tonumber(label)
    end
  else
    -- [<>][1-9] (local label reference)
    local dir, lnum = match(label, "^([<>])([1-9])$")
    if dir then -- Fwd: 1-9, Bkwd: 11-19.
      return "LG", lnum + (dir == ">" and 0 or 10)
    end
    -- extern label (extern label reference)
    local extname = match(label, "^extern%s+(%S+)$")
    if extname then
      return "EXT", map_extern[extname]
    end
    -- &expr (pointer)
    if label:sub(1, 1) == "&" then
      return "A", 0, format("(ptrdiff_t)(%s)", label:sub(2))
    end
  end
end

local function branch_type(op)
  if band(op, 0x7c000000) == 0x14000000 then return 0 -- B, BL
  elseif shr(op, 24) == 0x54 or band(op, 0x7e000000) == 0x34000000 or
	 band(op, 0x3b000000) == 0x18000000 then
    return 0x800 -- B.cond, CBZ, CBNZ, LDR* literal
  elseif band(op, 0x7e000000) == 0x36000000 then return 0x1000 -- TBZ, TBNZ
  elseif band(op, 0x9f000000) == 0x10000000 then return 0x2000 -- ADR
  elseif band(op, 0x9f000000) == band(0x90000000) then return 0x3000 -- ADRP
  else
    assert(false, "unknown branch type")
  end
end

------------------------------------------------------------------------------

local map_op, op_template

local function op_alias(opname, f)
  return function(params, nparams)
    if not params then return "-> "..opname:sub(1, -3) end
    f(params, nparams)
    op_template(params, map_op[opname], nparams)
  end
end

local function alias_bfx(p)
  p[4] = "#("..p[3]:sub(2)..")+("..p[4]:sub(2)..")-1"
end

local function alias_bfiz(p)
  parse_reg(p[1], 0, true)
  if parse_reg_type == "w" then
    p[3] = "#(32-("..p[3]:sub(2).."))%32"
    p[4] = "#("..p[4]:sub(2)..")-1"
  else
    p[3] = "#(64-("..p[3]:sub(2).."))%64"
    p[4] = "#("..p[4]:sub(2)..")-1"
  end
end

local alias_lslimm = op_alias("ubfm_4", function(p)
  parse_reg(p[1], 0, true)
  local sh = p[3]:sub(2)
  if parse_reg_type == "w" then
    p[3] = "#(32-("..sh.."))%32"
    p[4] = "#31-("..sh..")"
  else
    p[3] = "#(64-("..sh.."))%64"
    p[4] = "#63-("..sh..")"
  end
end)

-- Template strings for ARM instructions.
map_op = {
  -- Basic data processing instructions.
  add_3  = "0b000000DNMg|11000000pDpNIg|8b206000pDpNMx",
  add_4  = "0b000000DNMSg|0b200000DNMXg|8b200000pDpNMXx|8b200000pDpNxMwX",
  adds_3 = "2b000000DNMg|31000000DpNIg|ab206000DpNMx",
  adds_4 = "2b000000DNMSg|2b200000DNMXg|ab200000DpNMXx|ab200000DpNxMwX",
  cmn_2  = "2b00001fNMg|3100001fpNIg|ab20601fpNMx",
  cmn_3  = "2b00001fNMSg|2b20001fNMXg|ab20001fpNMXx|ab20001fpNxMwX",

  sub_3  = "4b000000DNMg|51000000pDpNIg|cb206000pDpNMx",
  sub_4  = "4b000000DNMSg|4b200000DNMXg|cb200000pDpNMXx|cb200000pDpNxMwX",
  subs_3 = "6b000000DNMg|71000000DpNIg|eb206000DpNMx",
  subs_4 = "6b000000DNMSg|6b200000DNMXg|eb200000DpNMXx|eb200000DpNxMwX",
  cmp_2  = "6b00001fNMg|7100001fpNIg|eb20601fpNMx",
  cmp_3  = "6b00001fNMSg|6b20001fNMXg|eb20001fpNMXx|eb20001fpNxMwX",

  neg_2  = "4b0003e0DMg",
  neg_3  = "4b0003e0DMSg",
  negs_2 = "6b0003e0DMg",
  negs_3 = "6b0003e0DMSg",

  adc_3  = "1a000000DNMg",
  adcs_3 = "3a000000DNMg",
  sbc_3  = "5a000000DNMg",
  sbcs_3 = "7a000000DNMg",
  ngc_2  = "5a0003e0DMg",
  ngcs_2 = "7a0003e0DMg",

  and_3  = "0a000000DNMg|12000000pDNig",
  and_4  = "0a000000DNMSg",
  orr_3  = "2a000000DNMg|32000000pDNig",
  orr_4  = "2a000000DNMSg",
  eor_3  = "4a000000DNMg|52000000pDNig",
  eor_4  = "4a000000DNMSg",
  ands_3 = "6a000000DNMg|72000000DNig",
  ands_4 = "6a000000DNMSg",
  tst_2  = "6a00001fNMg|7200001fNig",
  tst_3  = "6a00001fNMSg",

  bic_3  = "0a200000DNMg",
  bic_4  = "0a200000DNMSg",
  orn_3  = "2a200000DNMg",
  orn_4  = "2a200000DNMSg",
  eon_3  = "4a200000DNMg",
  eon_4  = "4a200000DNMSg",
  bics_3 = "6a200000DNMg",
  bics_4 = "6a200000DNMSg",

  movn_2 = "12800000DWg",
  movn_3 = "12800000DWRg",
  movz_2 = "52800000DWg",
  movz_3 = "52800000DWRg",
  movk_2 = "72800000DWg",
  movk_3 = "72800000DWRg",

  -- TODO: this doesn't cover all valid immediates for mov reg, #imm.
  mov_2  = "2a0003e0DMg|52800000DW|320003e0pDig|11000000pDpNg",
  mov_3  = "2a0003e0DMSg",
  mvn_2  = "2a2003e0DMg",
  mvn_3  = "2a2003e0DMSg",

  adr_2  = "10000000DBx",
  adrp_2 = "90000000DBx",

  csel_4  = "1a800000DNMCg",
  csinc_4 = "1a800400DNMCg",
  csinv_4 = "5a800000DNMCg",
  csneg_4 = "5a800400DNMCg",
  cset_2  = "1a9f07e0Dcg",
  csetm_2 = "5a9f03e0Dcg",
  cinc_3  = "1a800400DNmcg",
  cinv_3  = "5a800000DNmcg",
  cneg_3  = "5a800400DNmcg",

  ccmn_4 = "3a400000NMVCg|3a400800N5VCg",
  ccmp_4 = "7a400000NMVCg|7a400800N5VCg",

  madd_4 = "1b000000DNMAg",
  msub_4 = "1b008000DNMAg",
  mul_3  = "1b007c00DNMg",
  mneg_3 = "1b00fc00DNMg",

  smaddl_4 = "9b200000DxNMwAx",
  smsubl_4 = "9b208000DxNMwAx",
  smull_3  = "9b207c00DxNMw",
  smnegl_3 = "9b20fc00DxNMw",
  smulh_3  = "9b407c00DNMx",
  umaddl_4 = "9ba00000DxNMwAx",
  umsubl_4 = "9ba08000DxNMwAx",
  umull_3  = "9ba07c00DxNMw",
  umnegl_3 = "9ba0fc00DxNMw",
  umulh_3  = "9bc07c00DNMx",

  udiv_3 = "1ac00800DNMg",
  sdiv_3 = "1ac00c00DNMg",

  -- Bit operations.
  sbfm_4 = "13000000DN12w|93400000DN12x",
  bfm_4  = "33000000DN12w|b3400000DN12x",
  ubfm_4 = "53000000DN12w|d3400000DN12x",
  extr_4 = "13800000DNM2w|93c00000DNM2x",

  sxtb_2 = "13001c00DNw|93401c00DNx",
  sxth_2 = "13003c00DNw|93403c00DNx",
  sxtw_2 = "93407c00DxNw",
  uxtb_2 = "53001c00DNw",
  uxth_2 = "53003c00DNw",

  sbfx_4  = op_alias("sbfm_4", alias_bfx),
  bfxil_4 = op_alias("bfm_4", alias_bfx),
  ubfx_4  = op_alias("ubfm_4", alias_bfx),
  sbfiz_4 = op_alias("sbfm_4", alias_bfiz),
  bfi_4   = op_alias("bfm_4", alias_bfiz),
  ubfiz_4 = op_alias("ubfm_4", alias_bfiz),

  lsl_3  = function(params, nparams)
    if params and params[3]:byte() == 35 then
      return alias_lslimm(params, nparams)
    else
      return op_template(params, "1ac02000DNMg", nparams)
    end
  end,
  lsr_3  = "1ac02400DNMg|53007c00DN1w|d340fc00DN1x",
  asr_3  = "1ac02800DNMg|13007c00DN1w|9340fc00DN1x",
  ror_3  = "1ac02c00DNMg|13800000DNm2w|93c00000DNm2x",

  clz_2   = "5ac01000DNg",
  cls_2   = "5ac01400DNg",
  rbit_2  = "5ac00000DNg",
  rev_2   = "5ac00800DNw|dac00c00DNx",
  rev16_2 = "5ac00400DNg",
  rev32_2 = "dac00800DNx",

  -- Loads and stores.
  ["strb_*"]  = "38000000DwL",
  ["ldrb_*"]  = "38400000DwL",
  ["ldrsb_*"] = "38c00000DwL|38800000DxL",
  ["strh_*"]  = "78000000DwL",
  ["ldrh_*"]  = "78400000DwL",
  ["ldrsh_*"] = "78c00000DwL|78800000DxL",
  ["str_*"]   = "b8000000DwL|f8000000DxL|bc000000DsL|fc000000DdL",
  ["ldr_*"]   = "18000000DwB|58000000DxB|1c000000DsB|5c000000DdB|b8400000DwL|f8400000DxL|bc400000DsL|fc400000DdL",
  ["ldrsw_*"] = "98000000DxB|b8800000DxL",
  -- NOTE: ldur etc. are handled by ldr et al.

  ["stp_*"]   = "28000000DAwP|a8000000DAxP|2c000000DAsP|6c000000DAdP|ac000000DAqP",
  ["ldp_*"]   = "28400000DAwP|a8400000DAxP|2c400000DAsP|6c400000DAdP|ac400000DAqP",
  ["ldpsw_*"] = "68400000DAxP",

  -- Branches.
  b_1    = "14000000B",
  bl_1   = "94000000B",
  blr_1  = "d63f0000Nx",
  br_1   = "d61f0000Nx",
  ret_0  = "d65f03c0",
  ret_1  = "d65f0000Nx",
  -- b.cond is added below.
  cbz_2  = "34000000DBg",
  cbnz_2 = "35000000DBg",
  tbz_3  = "36000000DTBw|36000000DTBx",
  tbnz_3 = "37000000DTBw|37000000DTBx",

  -- ARM64e: Pointer authentication codes (PAC).
  blraaz_1  = "d63f081fNx",
  braa_2    = "d71f0800NDx",
  braaz_1   = "d61f081fNx",
  pacibsp_0 = "d503237f",
  retab_0   = "d65f0fff",

  -- Miscellaneous instructions.
  -- TODO: hlt, hvc, smc, svc, eret, dcps[123], drps, mrs, msr
  -- TODO: sys, sysl, ic, dc, at, tlbi
  -- TODO: hint, yield, wfe, wfi, sev, sevl
  -- TODO: clrex, dsb, dmb, isb
  nop_0  = "d503201f",
  brk_0  = "d4200000",
  brk_1  = "d4200000W",

  -- Floating point instructions.
  fmov_2  = "1e204000DNf|1e260000DwNs|1e270000DsNw|9e660000DxNd|9e670000DdNx|1e201000DFf",
  fabs_2  = "1e20c000DNf",
  fneg_2  = "1e214000DNf",
  fsqrt_2 = "1e21c000DNf",

  fcvt_2  = "1e22c000DdNs|1e624000DsNd",

  -- TODO: half-precision and fixed-point conversions.
  fcvtas_2 = "1e240000DwNs|9e240000DxNs|1e640000DwNd|9e640000DxNd",
  fcvtau_2 = "1e250000DwNs|9e250000DxNs|1e650000DwNd|9e650000DxNd",
  fcvtms_2 = "1e300000DwNs|9e300000DxNs|1e700000DwNd|9e700000DxNd",
  fcvtmu_2 = "1e310000DwNs|9e310000DxNs|1e710000DwNd|9e710000DxNd",
  fcvtns_2 = "1e200000DwNs|9e200000DxNs|1e600000DwNd|9e600000DxNd",
  fcvtnu_2 = "1e210000DwNs|9e210000DxNs|1e610000DwNd|9e610000DxNd",
  fcvtps_2 = "1e280000DwNs|9e280000DxNs|1e680000DwNd|9e680000DxNd",
  fcvtpu_2 = "1e290000DwNs|9e290000DxNs|1e690000DwNd|9e690000DxNd",
  fcvtzs_2 = "1e380000DwNs|9e380000DxNs|1e780000DwNd|9e780000DxNd",
  fcvtzu_2 = "1e390000DwNs|9e390000DxNs|1e790000DwNd|9e790000DxNd",

  scvtf_2  = "1e220000DsNw|9e220000DsNx|1e620000DdNw|9e620000DdNx",
  ucvtf_2  = "1e230000DsNw|9e230000DsNx|1e630000DdNw|9e630000DdNx",

  frintn_2 = "1e244000DNf",
  frintp_2 = "1e24c000DNf",
  frintm_2 = "1e254000DNf",
  frintz_2 = "1e25c000DNf",
  frinta_2 = "1e264000DNf",
  frintx_2 = "1e274000DNf",
  frinti_2 = "1e27c000DNf",

  fadd_3   = "1e202800DNMf",
  fsub_3   = "1e203800DNMf",
  fmul_3   = "1e200800DNMf",
  fnmul_3  = "1e208800DNMf",
  fdiv_3   = "1e201800DNMf",

  fmadd_4  = "1f000000DNMAf",
  fmsub_4  = "1f008000DNMAf",
  fnmadd_4 = "1f200000DNMAf",
  fnmsub_4 = "1f208000DNMAf",

  fmax_3   = "1e204800DNMf",
  fmaxnm_3 = "1e206800DNMf",
  fmin_3   = "1e205800DNMf",
  fminnm_3 = "1e207800DNMf",

  fcmp_2   = "1e202000NMf|1e202008NZf",
  fcmpe_2  = "1e202010NMf|1e202018NZf",

  fccmp_4  = "1e200400NMVCf",
  fccmpe_4 = "1e200410NMVCf",

  fcsel_4  = "1e200c00DNMCf",

  -- TODO: crc32*, aes*, sha*, pmull
  -- TODO: SIMD instructions.
}

for cond,c in pairs(map_cond) do
  map_op["b"..cond.."_1"] = tohex(0x54000000+c).."B"
end

------------------------------------------------------------------------------

-- Handle opcodes defined with template strings.
local function parse_template(params, template, nparams, pos)
  local op = tonumber(template:sub(1, 8), 16)
  local n = 1
  local rtt = {}

  parse_reg_type = false

  -- Process each character.
  for p in gmatch(template:sub(9), ".") do
    local q = params[n]
    if p == "D" then
      op = op + parse_reg(q, 0); n = n + 1
    elseif p == "N" then
      op = op + parse_reg(q, 5); n = n + 1
    elseif p == "M" then
      op = op + parse_reg(q, 16); n = n + 1
    elseif p == "A" then
      op = op + parse_reg(q, 10); n = n + 1
    elseif p == "m" then
      op = op + parse_reg(params[n-1], 16)

    elseif p == "p" then
      if q == "sp" then params[n] = "@x31" end
    elseif p == "g" then
      if parse_reg_type == "x" then
	op = op + 0x80000000
      elseif parse_reg_type ~= "w" then
	werror("bad register type")
      end
      parse_reg_type = false
    elseif p == "f" then
      if parse_reg_type == "d" then
	op = op + 0x00400000
      elseif parse_reg_type ~= "s" then
	werror("bad register type")
      end
      parse_reg_type = false
    elseif p == "x" or p == "w" or p == "d" or p == "s" or p == "q" then
      if parse_reg_type ~= p then
	werror("register size mismatch")
      end
      parse_reg_type = false

    elseif p == "L" then
      op = parse_load(params, nparams, n, op)
    elseif p == "P" then
      op = parse_load_pair(params, nparams, n, op)

    elseif p == "B" then
      local mode, v, s = parse_label(q, false); n = n + 1
      if not mode then werror("bad label `"..q.."'") end
      local m = branch_type(op)
      if mode == "A" then
	waction("REL_"..mode, v+m, format("(unsigned int)(%s)", s))
	actargs[#actargs+1] = format("(unsigned int)((%s)>>32)", s)
      else
	waction("REL_"..mode, v+m, s, 1)
      end

    elseif p == "I" then
      op = op + parse_imm12(q); n = n + 1
    elseif p == "i" then
      op = op + parse_imm13(q); n = n + 1
    elseif p == "W" then
      op = op + parse_imm(q, 16, 5, 0, false); n = n + 1
    elseif p == "T" then
      op = op + parse_imm6(q); n = n + 1
    elseif p == "1" then
      op = op + parse_imm(q, 6, 16, 0, false); n = n + 1
    elseif p == "2" then
      op = op + parse_imm(q, 6, 10, 0, false); n = n + 1
    elseif p == "5" then
      op = op + parse_imm(q, 5, 16, 0, false); n = n + 1
    elseif p == "V" then
      op = op + parse_imm(q, 4, 0, 0, false); n = n + 1
    elseif p == "F" then
      op = op + parse_fpimm(q); n = n + 1
    elseif p == "Z" then
      if q ~= "#0" and q ~= "#0.0" then werror("expected zero immediate") end
      n = n + 1

    elseif p == "S" then
      op = op + parse_shift(q); n = n + 1
    elseif p == "X" then
      op = op + parse_extend(q); n = n + 1
    elseif p == "R" then
      op = op + parse_lslx16(q); n = n + 1
    elseif p == "C" then
      op = op + parse_cond(q, 0); n = n + 1
    elseif p == "c" then
      op = op + parse_cond(q, 1); n = n + 1

    else
      assert(false)
    end
  end
  wputpos(pos, op)
end

function op_template(params, template, nparams)
  if not params then return template:gsub("%x%x%x%x%x%x%x%x", "") end

  -- Limit number of section buffer positions used by a single dasm_put().
  -- A single opcode needs a maximum of 4 positions.
  if secpos+4 > maxsecpos then wflush() end
  local pos = wpos()
  local lpos, apos, spos = #actlist, #actargs, secpos

  local ok, err
  for t in gmatch(template, "[^|]+") do
    ok, err = pcall(parse_template, params, t, nparams, pos)
    if ok then return end
    secpos = spos
    actlist[lpos+1] = nil
    actlist[lpos+2] = nil
    actlist[lpos+3] = nil
    actlist[lpos+4] = nil
    actargs[apos+1] = nil
    actargs[apos+2] = nil
    actargs[apos+3] = nil
    actargs[apos+4] = nil
  end
  error(err, 0)
end

map_op[".template__"] = op_template

------------------------------------------------------------------------------

-- Pseudo-opcode to mark the position where the action list is to be emitted.
map_op[".actionlist_1"] = function(params)
  if not params then return "cvar" end
  local name = params[1] -- No syntax check. You get to keep the pieces.
  wline(function(out) writeactions(out, name) end)
end

-- Pseudo-opcode to mark the position where the global enum is to be emitted.
map_op[".globals_1"] = function(params)
  if not params then return "prefix" end
  local prefix = params[1] -- No syntax check. You get to keep the pieces.
  wline(function(out) writeglobals(out, prefix) end)
end

-- Pseudo-opcode to mark the position where the global names are to be emitted.
map_op[".globalnames_1"] = function(params)
  if not params then return "cvar" end
  local name = params[1] -- No syntax check. You get to keep the pieces.
  wline(function(out) writeglobalnames(out, name) end)
end

-- Pseudo-opcode to mark the position where the extern names are to be emitted.
map_op[".externnames_1"] = function(params)
  if not params then return "cvar" end
  local name = params[1] -- No syntax check. You get to keep the pieces.
  wline(function(out) writeexternnames(out, name) end)
end

------------------------------------------------------------------------------

-- Label pseudo-opcode (converted from trailing colon form).
map_op[".label_1"] = function(params)
  if not params then return "[1-9] | ->global | =>pcexpr" end
  if secpos+1 > maxsecpos then wflush() end
  local mode, n, s = parse_label(params[1], true)
  if not mode or mode == "EXT" then werror("bad label definition") end
  waction("LABEL_"..mode, n, s, 1)
end

------------------------------------------------------------------------------

-- Pseudo-opcodes for data storage.
local function op_data(params)
  if not params then return "imm..." end
  local sz = params.op == ".long" and 4 or 8
  for _,p in ipairs(params) do
    local imm = parse_number(p)
    if imm then
      local n = tobit(imm)
      if n == imm or (n < 0 and n + 2^32 == imm) then
	wputw(n < 0 and n + 2^32 or n)
	if sz == 8 then
	  wputw(imm < 0 and 0xffffffff or 0)
	end
      elseif sz == 4 then
	werror("bad immediate `"..p.."'")
      else
	imm = nil
      end
    end
    if not imm then
      local mode, v, s = parse_label(p, false)
      if sz == 4 then
	if mode then werror("label does not fit into .long") end
	waction("IMMV", 0, p)
      elseif mode and mode ~= "A" then
	waction("REL_"..mode, v+0x8000, s, 1)
      else
	if mode == "A" then p = s end
	waction("IMMV", 0, format("(unsigned int)(%s)", p))
	waction("IMMV", 0, format("(unsigned int)((unsigned long long)(%s)>>32)", p))
      end
    end
    if secpos+2 > maxsecpos then wflush() end
  end
end
map_op[".long_*"] = op_data
map_op[".quad_*"] = op_data
map_op[".addr_*"] = op_data

-- Alignment pseudo-opcode.
map_op[".align_1"] = function(params)
  if not params then return "numpow2" end
  if secpos+1 > maxsecpos then wflush() end
  local align = tonumber(params[1])
  if align then
    local x = align
    -- Must be a power of 2 in the range (2 ... 256).
    for i=1,8 do
      x = x / 2
      if x == 1 then
	waction("ALIGN", align-1, nil, 1) -- Action byte is 2**n-1.
	return
      end
    end
  end
  werror("bad alignment")
end

------------------------------------------------------------------------------

-- Pseudo-opcode for (primitive) type definitions (map to C types).
map_op[".type_3"] = function(params, nparams)
  if not params then
    return nparams == 2 and "name, ctype" or "name, ctype, reg"
  end
  local name, ctype, reg = params[1], params[2], params[3]
  if not match(name, "^[%a_][%w_]*$") then
    werror("bad type name `"..name.."'")
  end
  local tp = map_type[name]
  if tp then
    werror("duplicate type `"..name.."'")
  end
  -- Add #type to defines. A bit unclean to put it in map_archdef.
  map_archdef["#"..name] = "sizeof("..ctype..")"
  -- Add new type and emit shortcut define.
  local num = ctypenum + 1
  map_type[name] = {
    ctype = ctype,
    ctypefmt = format("Dt%X(%%s)", num),
    reg = reg,
  }
  wline(format("#define Dt%X(_V) (int)(ptrdiff_t)&(((%s *)0)_V)", num, ctype))
  ctypenum = num
end
map_op[".type_2"] = map_op[".type_3"]

-- Dump type definitions.
local function dumptypes(out, lvl)
  local t = {}
  for name in pairs(map_type) do t[#t+1] = name end
  sort(t)
  out:write("Type definitions:\n")
  for _,name in ipairs(t) do
    local tp = map_type[name]
    local reg = tp.reg or ""
    out:write(format("  %-20s %-20s %s\n", name, tp.ctype, reg))
  end
  out:write("\n")
end

------------------------------------------------------------------------------

-- Set the current section.
function _M.section(num)
  waction("SECTION", num)
  wflush(true) -- SECTION is a terminal action.
end

------------------------------------------------------------------------------

-- Dump architecture description.
function _M.dumparch(out)
  out:write(format("DynASM %s version %s, released %s\n\n",
    _info.arch, _info.version, _info.release))
  dumpactions(out)
end

-- Dump all user defined elements.
function _M.dumpdef(out, lvl)
  dumptypes(out, lvl)
  dumpglobals(out, lvl)
  dumpexterns(out, lvl)
end

------------------------------------------------------------------------------

-- Pass callbacks from/to the DynASM core.
function _M.passcb(wl, we, wf, ww)
  wline, werror, wfatal, wwarn = wl, we, wf, ww
  return wflush
end

-- Setup the arch-specific module.
function _M.setup(arch, opt)
  g_arch, g_opt = arch, opt
end

-- Merge the core maps and the arch-specific maps.
function _M.mergemaps(map_coreop, map_def)
  setmetatable(map_op, { __index = map_coreop })
  setmetatable(map_def, { __index = map_archdef })
  return map_op, map_def
end

return _M

------------------------------------------------------------------------------

