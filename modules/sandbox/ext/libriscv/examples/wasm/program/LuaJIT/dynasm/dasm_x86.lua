------------------------------------------------------------------------------
-- DynASM x86/x64 module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- See dynasm.lua for full copyright notice.
------------------------------------------------------------------------------

local x64 = x64

-- Module information:
local _info = {
  arch =	x64 and "x64" or "x86",
  description =	"DynASM x86/x64 module",
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
local assert, unpack, setmetatable = assert, unpack or table.unpack, setmetatable
local _s = string
local sub, format, byte, char = _s.sub, _s.format, _s.byte, _s.char
local find, match, gmatch, gsub = _s.find, _s.match, _s.gmatch, _s.gsub
local concat, sort, remove = table.concat, table.sort, table.remove
local bit = bit or require("bit")
local band, bxor, shl, shr = bit.band, bit.bxor, bit.lshift, bit.rshift

-- Inherited tables and callbacks.
local g_opt, g_arch
local wline, werror, wfatal, wwarn

-- Action name list.
-- CHECK: Keep this in sync with the C code!
local action_names = {
  -- int arg, 1 buffer pos:
  "DISP",  "IMM_S", "IMM_B", "IMM_W", "IMM_D",  "IMM_WB", "IMM_DB",
  -- action arg (1 byte), int arg, 1 buffer pos (reg/num):
  "VREG", "SPACE",
  -- ptrdiff_t arg, 1 buffer pos (address): !x64
  "SETLABEL", "REL_A",
  -- action arg (1 byte) or int arg, 2 buffer pos (link, offset):
  "REL_LG", "REL_PC",
  -- action arg (1 byte) or int arg, 1 buffer pos (link):
  "IMM_LG", "IMM_PC",
  -- action arg (1 byte) or int arg, 1 buffer pos (offset):
  "LABEL_LG", "LABEL_PC",
  -- action arg (1 byte), 1 buffer pos (offset):
  "ALIGN",
  -- action args (2 bytes), no buffer pos.
  "EXTERN",
  -- action arg (1 byte), no buffer pos.
  "ESC",
  -- no action arg, no buffer pos.
  "MARK",
  -- action arg (1 byte), no buffer pos, terminal action:
  "SECTION",
  -- no args, no buffer pos, terminal action:
  "STOP"
}

-- Maximum number of section buffer positions for dasm_put().
-- CHECK: Keep this in sync with the C code!
local maxsecpos = 25 -- Keep this low, to avoid excessively long C lines.

-- Action name -> action number (dynamically generated below).
local map_action = {}
-- First action number. Everything below does not need to be escaped.
local actfirst = 256-#action_names

-- Action list buffer and string (only used to remove dupes).
local actlist = {}
local actstr = ""

-- Argument list for next dasm_put(). Start with offset 0 into action list.
local actargs = { 0 }

-- Current number of section buffer positions for dasm_put().
local secpos = 1

-- VREG kind encodings, pre-shifted by 5 bits.
local map_vreg = {
  ["modrm.rm.m"] = 0x00,
  ["modrm.rm.r"] = 0x20,
  ["opcode"] =     0x20,
  ["sib.base"] =   0x20,
  ["sib.index"] =  0x40,
  ["modrm.reg"] =  0x80,
  ["vex.v"] =      0xa0,
  ["imm.hi"] =     0xc0,
}

-- Current number of VREG actions contributing to REX/VEX shrinkage.
local vreg_shrink_count = 0

------------------------------------------------------------------------------

-- Compute action numbers for action names.
for n,name in ipairs(action_names) do
  local num = actfirst + n - 1
  map_action[name] = num
end

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
  local last = actlist[nn] or 255
  actlist[nn] = nil -- Remove last byte.
  if nn == 0 then nn = 1 end
  out:write("static const unsigned char ", name, "[", nn, "] = {\n")
  local s = "  "
  for n,b in ipairs(actlist) do
    s = s..b..","
    if #s >= 75 then
      assert(out:write(s, "\n"))
      s = "  "
    end
  end
  out:write(s, last, "\n};\n\n") -- Add last byte back.
end

------------------------------------------------------------------------------

-- Add byte to action list.
local function wputxb(n)
  assert(n >= 0 and n <= 255 and n % 1 == 0, "byte out of range")
  actlist[#actlist+1] = n
end

-- Add action to list with optional arg. Advance buffer pos, too.
local function waction(action, a, num)
  wputxb(assert(map_action[action], "bad action name `"..action.."'"))
  if a then actargs[#actargs+1] = a end
  if a or num then secpos = secpos + (num or 1) end
end

-- Optionally add a VREG action.
local function wvreg(kind, vreg, psz, sk, defer)
  if not vreg then return end
  waction("VREG", vreg)
  local b = assert(map_vreg[kind], "bad vreg kind `"..vreg.."'")
  if b < (sk or 0) then
    vreg_shrink_count = vreg_shrink_count + 1
  end
  if not defer then
    b = b + vreg_shrink_count * 8
    vreg_shrink_count = 0
  end
  wputxb(b + (psz or 0))
end

-- Add call to embedded DynASM C code.
local function wcall(func, args)
  wline(format("dasm_%s(Dst, %s);", func, concat(args, ", ")), true)
end

-- Delete duplicate action list chunks. A tad slow, but so what.
local function dedupechunk(offset)
  local al, as = actlist, actstr
  local chunk = char(unpack(al, offset+1, #al))
  local orig = find(as, chunk, 1, true)
  if orig then
    actargs[1] = orig-1 -- Replace with original offset.
    for i=offset+1,#al do al[i] = nil end -- Kill dupe.
  else
    actstr = as..chunk
  end
end

-- Flush action list (intervening C code or buffer pos overflow).
local function wflush(term)
  local offset = actargs[1]
  if #actlist == offset then return end -- Nothing to flush.
  if not term then waction("STOP") end -- Terminate action list.
  dedupechunk(offset)
  wcall("put", actargs) -- Add call to dasm_put().
  actargs = { #actlist } -- Actionlist offset is 1st arg to next dasm_put().
  secpos = 1 -- The actionlist offset occupies a buffer position, too.
end

-- Put escaped byte.
local function wputb(n)
  if n >= actfirst then waction("ESC") end -- Need to escape byte.
  wputxb(n)
end

------------------------------------------------------------------------------

-- Global label name -> global label number. With auto assignment on 1st use.
local next_global = 10
local map_global = setmetatable({}, { __index = function(t, name)
  if not match(name, "^[%a_][%w_@]*$") then werror("bad global label") end
  local n = next_global
  if n > 246 then werror("too many global labels") end
  next_global = n + 1
  t[name] = n
  return n
end})

-- Dump global labels.
local function dumpglobals(out, lvl)
  local t = {}
  for name, n in pairs(map_global) do t[n] = name end
  out:write("Global labels:\n")
  for i=10,next_global-1 do
    out:write(format("  %s\n", t[i]))
  end
  out:write("\n")
end

-- Write global label enum.
local function writeglobals(out, prefix)
  local t = {}
  for name, n in pairs(map_global) do t[n] = name end
  out:write("enum {\n")
  for i=10,next_global-1 do
    out:write("  ", prefix, gsub(t[i], "@.*", ""), ",\n")
  end
  out:write("  ", prefix, "_MAX\n};\n")
end

-- Write global label names.
local function writeglobalnames(out, name)
  local t = {}
  for name, n in pairs(map_global) do t[n] = name end
  out:write("static const char *const ", name, "[] = {\n")
  for i=10,next_global-1 do
    out:write("  \"", t[i], "\",\n")
  end
  out:write("  (const char *)0\n};\n")
end

------------------------------------------------------------------------------

-- Extern label name -> extern label number. With auto assignment on 1st use.
local next_extern = -1
local map_extern = setmetatable({}, { __index = function(t, name)
  -- No restrictions on the name for now.
  local n = next_extern
  if n < -256 then werror("too many extern labels") end
  next_extern = n - 1
  t[name] = n
  return n
end})

-- Dump extern labels.
local function dumpexterns(out, lvl)
  local t = {}
  for name, n in pairs(map_extern) do t[-n] = name end
  out:write("Extern labels:\n")
  for i=1,-next_extern-1 do
    out:write(format("  %s\n", t[i]))
  end
  out:write("\n")
end

-- Write extern label names.
local function writeexternnames(out, name)
  local t = {}
  for name, n in pairs(map_extern) do t[-n] = name end
  out:write("static const char *const ", name, "[] = {\n")
  for i=1,-next_extern-1 do
    out:write("  \"", t[i], "\",\n")
  end
  out:write("  (const char *)0\n};\n")
end

------------------------------------------------------------------------------

-- Arch-specific maps.
local map_archdef = {}		-- Ext. register name -> int. name.
local map_reg_rev = {}		-- Int. register name -> ext. name.
local map_reg_num = {}		-- Int. register name -> register number.
local map_reg_opsize = {}	-- Int. register name -> operand size.
local map_reg_valid_base = {}	-- Int. register name -> valid base register?
local map_reg_valid_index = {}	-- Int. register name -> valid index register?
local map_reg_needrex = {}	-- Int. register name -> need rex vs. no rex.
local reg_list = {}		-- Canonical list of int. register names.

local map_type = {}		-- Type name -> { ctype, reg }
local ctypenum = 0		-- Type number (for _PTx macros).

local addrsize = x64 and "q" or "d"	-- Size for address operands.

-- Helper functions to fill register maps.
local function mkrmap(sz, cl, names)
  local cname = format("@%s", sz)
  reg_list[#reg_list+1] = cname
  map_archdef[cl] = cname
  map_reg_rev[cname] = cl
  map_reg_num[cname] = -1
  map_reg_opsize[cname] = sz
  if sz == addrsize or sz == "d" then
    map_reg_valid_base[cname] = true
    map_reg_valid_index[cname] = true
  end
  if names then
    for n,name in ipairs(names) do
      local iname = format("@%s%x", sz, n-1)
      reg_list[#reg_list+1] = iname
      map_archdef[name] = iname
      map_reg_rev[iname] = name
      map_reg_num[iname] = n-1
      map_reg_opsize[iname] = sz
      if sz == "b" and n > 4 then map_reg_needrex[iname] = false end
      if sz == addrsize or sz == "d" then
	map_reg_valid_base[iname] = true
	map_reg_valid_index[iname] = true
      end
    end
  end
  for i=0,(x64 and sz ~= "f") and 15 or 7 do
    local needrex = sz == "b" and i > 3
    local iname = format("@%s%x%s", sz, i, needrex and "R" or "")
    if needrex then map_reg_needrex[iname] = true end
    local name
    if sz == "o" or sz == "y" then name = format("%s%d", cl, i)
    elseif sz == "f" then name = format("st%d", i)
    else name = format("r%d%s", i, sz == addrsize and "" or sz) end
    map_archdef[name] = iname
    if not map_reg_rev[iname] then
      reg_list[#reg_list+1] = iname
      map_reg_rev[iname] = name
      map_reg_num[iname] = i
      map_reg_opsize[iname] = sz
      if sz == addrsize or sz == "d" then
	map_reg_valid_base[iname] = true
	map_reg_valid_index[iname] = true
      end
    end
  end
  reg_list[#reg_list+1] = ""
end

-- Integer registers (qword, dword, word and byte sized).
if x64 then
  mkrmap("q", "Rq", {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"})
end
mkrmap("d", "Rd", {"eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"})
mkrmap("w", "Rw", {"ax", "cx", "dx", "bx", "sp", "bp", "si", "di"})
mkrmap("b", "Rb", {"al", "cl", "dl", "bl", "ah", "ch", "dh", "bh"})
map_reg_valid_index[map_archdef.esp] = false
if x64 then map_reg_valid_index[map_archdef.rsp] = false end
if x64 then map_reg_needrex[map_archdef.Rb] = true end
map_archdef["Ra"] = "@"..addrsize

-- FP registers (internally tword sized, but use "f" as operand size).
mkrmap("f", "Rf")

-- SSE registers (oword sized, but qword and dword accessible).
mkrmap("o", "xmm")

-- AVX registers (yword sized, but oword, qword and dword accessible).
mkrmap("y", "ymm")

-- Operand size prefixes to codes.
local map_opsize = {
  byte = "b", word = "w", dword = "d", qword = "q", oword = "o", yword = "y",
  tword = "t", aword = addrsize,
}

-- Operand size code to number.
local map_opsizenum = {
  b = 1, w = 2, d = 4, q = 8, o = 16, y = 32, t = 10,
}

-- Operand size code to name.
local map_opsizename = {
  b = "byte", w = "word", d = "dword", q = "qword", o = "oword", y = "yword",
  t = "tword", f = "fpword",
}

-- Valid index register scale factors.
local map_xsc = {
  ["1"] = 0, ["2"] = 1, ["4"] = 2, ["8"] = 3,
}

-- Condition codes.
local map_cc = {
  o = 0, no = 1, b = 2, nb = 3, e = 4, ne = 5, be = 6, nbe = 7,
  s = 8, ns = 9, p = 10, np = 11, l = 12, nl = 13, le = 14, nle = 15,
  c = 2, nae = 2, nc = 3, ae = 3, z = 4, nz = 5, na = 6, a = 7,
  pe = 10, po = 11, nge = 12, ge = 13, ng = 14, g = 15,
}


-- Reverse defines for registers.
function _M.revdef(s)
  return gsub(s, "@%w+", map_reg_rev)
end

-- Dump register names and numbers
local function dumpregs(out)
  out:write("Register names, sizes and internal numbers:\n")
  for _,reg in ipairs(reg_list) do
    if reg == "" then
      out:write("\n")
    else
      local name = map_reg_rev[reg]
      local num = map_reg_num[reg]
      local opsize = map_opsizename[map_reg_opsize[reg]]
      out:write(format("  %-5s %-8s %s\n", name, opsize,
		       num < 0 and "(variable)" or num))
    end
  end
end

------------------------------------------------------------------------------

-- Put action for label arg (IMM_LG, IMM_PC, REL_LG, REL_PC).
local function wputlabel(aprefix, imm, num)
  if type(imm) == "number" then
    if imm < 0 then
      waction("EXTERN")
      wputxb(aprefix == "IMM_" and 0 or 1)
      imm = -imm-1
    else
      waction(aprefix.."LG", nil, num);
    end
    wputxb(imm)
  else
    waction(aprefix.."PC", imm, num)
  end
end

-- Put signed byte or arg.
local function wputsbarg(n)
  if type(n) == "number" then
    if n < -128 or n > 127 then
      werror("signed immediate byte out of range")
    end
    if n < 0 then n = n + 256 end
    wputb(n)
  else waction("IMM_S", n) end
end

-- Put unsigned byte or arg.
local function wputbarg(n)
  if type(n) == "number" then
    if n < 0 or n > 255 then
      werror("unsigned immediate byte out of range")
    end
    wputb(n)
  else waction("IMM_B", n) end
end

-- Put unsigned word or arg.
local function wputwarg(n)
  if type(n) == "number" then
    if shr(n, 16) ~= 0 then
      werror("unsigned immediate word out of range")
    end
    wputb(band(n, 255)); wputb(shr(n, 8));
  else waction("IMM_W", n) end
end

-- Put signed or unsigned dword or arg.
local function wputdarg(n)
  local tn = type(n)
  if tn == "number" then
    wputb(band(n, 255))
    wputb(band(shr(n, 8), 255))
    wputb(band(shr(n, 16), 255))
    wputb(shr(n, 24))
  elseif tn == "table" then
    wputlabel("IMM_", n[1], 1)
  else
    waction("IMM_D", n)
  end
end

-- Put signed or unsigned qword or arg.
local function wputqarg(n)
  local tn = type(n)
  if tn == "number" then -- This is only used for numbers from -2^31..2^32-1.
    wputb(band(n, 255))
    wputb(band(shr(n, 8), 255))
    wputb(band(shr(n, 16), 255))
    wputb(shr(n, 24))
    local sign = n < 0 and 255 or 0
    wputb(sign); wputb(sign); wputb(sign); wputb(sign)
  else
    waction("IMM_D", format("(unsigned int)(%s)", n))
    waction("IMM_D", format("(unsigned int)((unsigned long long)(%s)>>32)", n))
  end
end

-- Put operand-size dependent number or arg (defaults to dword).
local function wputszarg(sz, n)
  if not sz or sz == "d" or sz == "q" then wputdarg(n)
  elseif sz == "w" then wputwarg(n)
  elseif sz == "b" then wputbarg(n)
  elseif sz == "s" then wputsbarg(n)
  else werror("bad operand size") end
end

-- Put multi-byte opcode with operand-size dependent modifications.
local function wputop(sz, op, rex, vex, vregr, vregxb)
  local psz, sk = 0, nil
  if vex then
    local tail
    if vex.m == 1 and band(rex, 11) == 0 then
      if x64 and vregxb then
	sk = map_vreg["modrm.reg"]
      else
	wputb(0xc5)
      tail = shl(bxor(band(rex, 4), 4), 5)
      psz = 3
      end
    end
    if not tail then
      wputb(0xc4)
      wputb(shl(bxor(band(rex, 7), 7), 5) + vex.m)
      tail = shl(band(rex, 8), 4)
      psz = 4
    end
    local reg, vreg = 0, nil
    if vex.v then
      reg = vex.v.reg
      if not reg then werror("bad vex operand") end
      if reg < 0 then reg = 0; vreg = vex.v.vreg end
    end
    if sz == "y" or vex.l then tail = tail + 4 end
    wputb(tail + shl(bxor(reg, 15), 3) + vex.p)
    wvreg("vex.v", vreg)
    rex = 0
    if op >= 256 then werror("bad vex opcode") end
  else
    if rex ~= 0 then
      if not x64 then werror("bad operand size") end
    elseif (vregr or vregxb) and x64 then
      rex = 0x10
      sk = map_vreg["vex.v"]
    end
  end
  local r
  if sz == "w" then wputb(102) end
  -- Needs >32 bit numbers, but only for crc32 eax, word [ebx]
  if op >= 4294967296 then r = op%4294967296 wputb((op-r)/4294967296) op = r end
  if op >= 16777216 then wputb(shr(op, 24)); op = band(op, 0xffffff) end
  if op >= 65536 then
    if rex ~= 0 then
      local opc3 = band(op, 0xffff00)
      if opc3 == 0x0f3a00 or opc3 == 0x0f3800 then
	wputb(64 + band(rex, 15)); rex = 0; psz = 2
      end
    end
    wputb(shr(op, 16)); op = band(op, 0xffff); psz = psz + 1
  end
  if op >= 256 then
    local b = shr(op, 8)
    if b == 15 and rex ~= 0 then wputb(64 + band(rex, 15)); rex = 0; psz = 2 end
    wputb(b); op = band(op, 255); psz = psz + 1
  end
  if rex ~= 0 then wputb(64 + band(rex, 15)); psz = 2 end
  if sz == "b" then op = op - 1 end
  wputb(op)
  return psz, sk
end

-- Put ModRM or SIB formatted byte.
local function wputmodrm(m, s, rm, vs, vrm)
  assert(m < 4 and s < 16 and rm < 16, "bad modrm operands")
  wputb(shl(m, 6) + shl(band(s, 7), 3) + band(rm, 7))
end

-- Put ModRM/SIB plus optional displacement.
local function wputmrmsib(t, imark, s, vsreg, psz, sk)
  local vreg, vxreg
  local reg, xreg = t.reg, t.xreg
  if reg and reg < 0 then reg = 0; vreg = t.vreg end
  if xreg and xreg < 0 then xreg = 0; vxreg = t.vxreg end
  if s < 0 then s = 0 end

  -- Register mode.
  if sub(t.mode, 1, 1) == "r" then
    wputmodrm(3, s, reg)
    wvreg("modrm.reg", vsreg, psz+1, sk, vreg)
    wvreg("modrm.rm.r", vreg, psz+1, sk)
    return
  end

  local disp = t.disp
  local tdisp = type(disp)
  -- No base register?
  if not reg then
    local riprel = false
    if xreg then
      -- Indexed mode with index register only.
      -- [xreg*xsc+disp] -> (0, s, esp) (xsc, xreg, ebp)
      wputmodrm(0, s, 4)
      if imark == "I" then waction("MARK") end
      wvreg("modrm.reg", vsreg, psz+1, sk, vxreg)
      wputmodrm(t.xsc, xreg, 5)
      wvreg("sib.index", vxreg, psz+2, sk)
    else
      -- Pure 32 bit displacement.
      if x64 and tdisp ~= "table" then
	wputmodrm(0, s, 4) -- [disp] -> (0, s, esp) (0, esp, ebp)
	wvreg("modrm.reg", vsreg, psz+1, sk)
	if imark == "I" then waction("MARK") end
	wputmodrm(0, 4, 5)
      else
	riprel = x64
	wputmodrm(0, s, 5) -- [disp|rip-label] -> (0, s, ebp)
	wvreg("modrm.reg", vsreg, psz+1, sk)
	if imark == "I" then waction("MARK") end
      end
    end
    if riprel then -- Emit rip-relative displacement.
      if match("UWSiI", imark) then
	werror("NYI: rip-relative displacement followed by immediate")
      end
      -- The previous byte in the action buffer cannot be 0xe9 or 0x80-0x8f.
      if disp[2] == "iPJ" then
	waction("REL_A", disp[1])
      else
	wputlabel("REL_", disp[1], 2)
      end
    else
      wputdarg(disp)
    end
    return
  end

  local m
  if tdisp == "number" then -- Check displacement size at assembly time.
    if disp == 0 and band(reg, 7) ~= 5 then -- [ebp] -> [ebp+0] (in SIB, too)
      if not vreg then m = 0 end -- Force DISP to allow [Rd(5)] -> [ebp+0]
    elseif disp >= -128 and disp <= 127 then m = 1
    else m = 2 end
  elseif tdisp == "table" then
    m = 2
  end

  -- Index register present or esp as base register: need SIB encoding.
  if xreg or band(reg, 7) == 4 then
    wputmodrm(m or 2, s, 4) -- ModRM.
    if m == nil or imark == "I" then waction("MARK") end
    wvreg("modrm.reg", vsreg, psz+1, sk, vxreg or vreg)
    wputmodrm(t.xsc or 0, xreg or 4, reg) -- SIB.
    wvreg("sib.index", vxreg, psz+2, sk, vreg)
    wvreg("sib.base", vreg, psz+2, sk)
  else
    wputmodrm(m or 2, s, reg) -- ModRM.
    if (imark == "I" and (m == 1 or m == 2)) or
       (m == nil and (vsreg or vreg)) then waction("MARK") end
    wvreg("modrm.reg", vsreg, psz+1, sk, vreg)
    wvreg("modrm.rm.m", vreg, psz+1, sk)
  end

  -- Put displacement.
  if m == 1 then wputsbarg(disp)
  elseif m == 2 then wputdarg(disp)
  elseif m == nil then waction("DISP", disp) end
end

------------------------------------------------------------------------------

-- Return human-readable operand mode string.
local function opmodestr(op, args)
  local m = {}
  for i=1,#args do
    local a = args[i]
    m[#m+1] = sub(a.mode, 1, 1)..(a.opsize or "?")
  end
  return op.." "..concat(m, ",")
end

-- Convert number to valid integer or nil.
local function toint(expr, isqword)
  local n = tonumber(expr)
  if n then
    if n % 1 ~= 0 then
      werror("not an integer number `"..expr.."'")
    elseif isqword then
      if n < -2147483648 or n > 2147483647 then
	n = nil -- Handle it as an expression to avoid precision loss.
      end
    elseif n < -2147483648 or n > 4294967295 then
      werror("bad integer number `"..expr.."'")
    end
    return n
  end
end

-- Parse immediate expression.
local function immexpr(expr)
  -- &expr (pointer)
  if sub(expr, 1, 1) == "&" then
    return "iPJ", format("(ptrdiff_t)(%s)", sub(expr,2))
  end

  local prefix = sub(expr, 1, 2)
  -- =>expr (pc label reference)
  if prefix == "=>" then
    return "iJ", sub(expr, 3)
  end
  -- ->name (global label reference)
  if prefix == "->" then
    return "iJ", map_global[sub(expr, 3)]
  end

  -- [<>][1-9] (local label reference)
  local dir, lnum = match(expr, "^([<>])([1-9])$")
  if dir then -- Fwd: 247-255, Bkwd: 1-9.
    return "iJ", lnum + (dir == ">" and 246 or 0)
  end

  local extname = match(expr, "^extern%s+(%S+)$")
  if extname then
    return "iJ", map_extern[extname]
  end

  -- expr (interpreted as immediate)
  return "iI", expr
end

-- Parse displacement expression: +-num, +-expr, +-opsize*num
local function dispexpr(expr)
  local disp = expr == "" and 0 or toint(expr)
  if disp then return disp end
  local c, dispt = match(expr, "^([+-])%s*(.+)$")
  if c == "+" then
    expr = dispt
  elseif not c then
    werror("bad displacement expression `"..expr.."'")
  end
  local opsize, tailops = match(dispt, "^(%w+)%s*%*%s*(.+)$")
  local ops, imm = map_opsize[opsize], toint(tailops)
  if ops and imm then
    if c == "-" then imm = -imm end
    return imm*map_opsizenum[ops]
  end
  local mode, iexpr = immexpr(dispt)
  if mode == "iJ" or mode == "iPJ" then
    if c == "-" then werror("cannot invert label reference") end
    return { iexpr, mode }
  end
  return expr -- Need to return original signed expression.
end

-- Parse register or type expression.
local function rtexpr(expr)
  if not expr then return end
  local tname, ovreg = match(expr, "^([%w_]+):(@[%w_]+)$")
  local tp = map_type[tname or expr]
  if tp then
    local reg = ovreg or tp.reg
    local rnum = map_reg_num[reg]
    if not rnum then
      werror("type `"..(tname or expr).."' needs a register override")
    end
    if not map_reg_valid_base[reg] then
      werror("bad base register override `"..(map_reg_rev[reg] or reg).."'")
    end
    return reg, rnum, tp
  end
  return expr, map_reg_num[expr]
end

-- Parse operand and return { mode, opsize, reg, xreg, xsc, disp, imm }.
local function parseoperand(param, isqword)
  local t = {}

  local expr = param
  local opsize, tailops = match(param, "^(%w+)%s*(.+)$")
  if opsize then
    t.opsize = map_opsize[opsize]
    if t.opsize then expr = tailops end
  end

  local br = match(expr, "^%[%s*(.-)%s*%]$")
  repeat
    if br then
      t.mode = "xm"

      -- [disp]
      t.disp = toint(br)
      if t.disp then
	t.mode = x64 and "xm" or "xmO"
	break
      end

      -- [reg...]
      local tp
      local reg, tailr = match(br, "^([@%w_:]+)%s*(.*)$")
      reg, t.reg, tp = rtexpr(reg)
      if not t.reg then
	-- [expr]
	t.mode = x64 and "xm" or "xmO"
	t.disp = dispexpr("+"..br)
	break
      end

      if t.reg == -1 then
	t.vreg, tailr = match(tailr, "^(%b())(.*)$")
	if not t.vreg then werror("bad variable register expression") end
      end

      -- [xreg*xsc] or [xreg*xsc+-disp] or [xreg*xsc+-expr]
      local xsc, tailsc = match(tailr, "^%*%s*([1248])%s*(.*)$")
      if xsc then
	if not map_reg_valid_index[reg] then
	  werror("bad index register `"..map_reg_rev[reg].."'")
	end
	t.xsc = map_xsc[xsc]
	t.xreg = t.reg
	t.vxreg = t.vreg
	t.reg = nil
	t.vreg = nil
	t.disp = dispexpr(tailsc)
	break
      end
      if not map_reg_valid_base[reg] then
	werror("bad base register `"..map_reg_rev[reg].."'")
      end

      -- [reg] or [reg+-disp]
      t.disp = toint(tailr) or (tailr == "" and 0)
      if t.disp then break end

      -- [reg+xreg...]
      local xreg, tailx = match(tailr, "^%+%s*([@%w_:]+)%s*(.*)$")
      xreg, t.xreg, tp = rtexpr(xreg)
      if not t.xreg then
	-- [reg+-expr]
	t.disp = dispexpr(tailr)
	break
      end
      if not map_reg_valid_index[xreg] then
	werror("bad index register `"..map_reg_rev[xreg].."'")
      end

      if t.xreg == -1 then
	t.vxreg, tailx = match(tailx, "^(%b())(.*)$")
	if not t.vxreg then werror("bad variable register expression") end
      end

      -- [reg+xreg*xsc...]
      local xsc, tailsc = match(tailx, "^%*%s*([1248])%s*(.*)$")
      if xsc then
	t.xsc = map_xsc[xsc]
	tailx = tailsc
      end

      -- [...] or [...+-disp] or [...+-expr]
      t.disp = dispexpr(tailx)
    else
      -- imm or opsize*imm
      local imm = toint(expr, isqword)
      if not imm and sub(expr, 1, 1) == "*" and t.opsize then
	imm = toint(sub(expr, 2))
	if imm then
	  imm = imm * map_opsizenum[t.opsize]
	  t.opsize = nil
	end
      end
      if imm then
	if t.opsize then werror("bad operand size override") end
	local m = "i"
	if imm == 1 then m = m.."1" end
	if imm >= 4294967168 and imm <= 4294967295 then imm = imm-4294967296 end
	if imm >= -128 and imm <= 127 then m = m.."S" end
	t.imm = imm
	t.mode = m
	break
      end

      local tp
      local reg, tailr = match(expr, "^([@%w_:]+)%s*(.*)$")
      reg, t.reg, tp = rtexpr(reg)
      if t.reg then
	if t.reg == -1 then
	  t.vreg, tailr = match(tailr, "^(%b())(.*)$")
	  if not t.vreg then werror("bad variable register expression") end
	end
	-- reg
	if tailr == "" then
	  if t.opsize then werror("bad operand size override") end
	  t.opsize = map_reg_opsize[reg]
	  if t.opsize == "f" then
	    t.mode = t.reg == 0 and "fF" or "f"
	  else
	    if reg == "@w4" or (x64 and reg == "@d4") then
	      wwarn("bad idea, try again with `"..(x64 and "rsp'" or "esp'"))
	    end
	    t.mode = t.reg == 0 and "rmR" or (reg == "@b1" and "rmC" or "rm")
	  end
	  t.needrex = map_reg_needrex[reg]
	  break
	end

	-- type[idx], type[idx].field, type->field -> [reg+offset_expr]
	if not tp then werror("bad operand `"..param.."'") end
	t.mode = "xm"
	t.disp = format(tp.ctypefmt, tailr)
      else
	t.mode, t.imm = immexpr(expr)
	if sub(t.mode, -1) == "J" then
	  if t.opsize and t.opsize ~= addrsize then
	    werror("bad operand size override")
	  end
	  t.opsize = addrsize
	end
      end
    end
  until true
  return t
end

------------------------------------------------------------------------------
-- x86 Template String Description
-- ===============================
--
-- Each template string is a list of [match:]pattern pairs,
-- separated by "|". The first match wins. No match means a
-- bad or unsupported combination of operand modes or sizes.
--
-- The match part and the ":" is omitted if the operation has
-- no operands. Otherwise the first N characters are matched
-- against the mode strings of each of the N operands.
--
-- The mode string for each operand type is (see parseoperand()):
--   Integer register: "rm", +"R" for eax, ax, al, +"C" for cl
--   FP register:      "f",  +"F" for st0
--   Index operand:    "xm", +"O" for [disp] (pure offset)
--   Immediate:        "i",  +"S" for signed 8 bit, +"1" for 1,
--                     +"I" for arg, +"P" for pointer
--   Any:              +"J" for valid jump targets
--
-- So a match character "m" (mixed) matches both an integer register
-- and an index operand (to be encoded with the ModRM/SIB scheme).
-- But "r" matches only a register and "x" only an index operand
-- (e.g. for FP memory access operations).
--
-- The operand size match string starts right after the mode match
-- characters and ends before the ":". "dwb" or "qdwb" is assumed, if empty.
-- The effective data size of the operation is matched against this list.
--
-- If only the regular "b", "w", "d", "q", "t" operand sizes are
-- present, then all operands must be the same size. Unspecified sizes
-- are ignored, but at least one operand must have a size or the pattern
-- won't match (use the "byte", "word", "dword", "qword", "tword"
-- operand size overrides. E.g.: mov dword [eax], 1).
--
-- If the list has a "1" or "2" prefix, the operand size is taken
-- from the respective operand and any other operand sizes are ignored.
-- If the list contains only ".", all operand sizes are ignored.
-- If the list has a "/" prefix, the concatenated (mixed) operand sizes
-- are compared to the match.
--
-- E.g. "rrdw" matches for either two dword registers or two word
-- registers. "Fx2dq" matches an st0 operand plus an index operand
-- pointing to a dword (float) or qword (double).
--
-- Every character after the ":" is part of the pattern string:
--   Hex chars are accumulated to form the opcode (left to right).
--   "n"       disables the standard opcode mods
--             (otherwise: -1 for "b", o16 prefix for "w", rex.w for "q")
--   "X"       Force REX.W.
--   "r"/"R"   adds the reg. number from the 1st/2nd operand to the opcode.
--   "m"/"M"   generates ModRM/SIB from the 1st/2nd operand.
--             The spare 3 bits are either filled with the last hex digit or
--             the result from a previous "r"/"R". The opcode is restored.
--   "u"       Use VEX encoding, vvvv unused.
--   "v"/"V"   Use VEX encoding, vvvv from 1st/2nd operand (the operand is
--             removed from the list used by future characters).
--   "w"       Use VEX encoding, vvvv from 3rd operand.
--   "L"       Force VEX.L
--
-- All of the following characters force a flush of the opcode:
--   "o"/"O"   stores a pure 32 bit disp (offset) from the 1st/2nd operand.
--   "s"       stores a 4 bit immediate from the last register operand,
--             followed by 4 zero bits.
--   "S"       stores a signed 8 bit immediate from the last operand.
--   "U"       stores an unsigned 8 bit immediate from the last operand.
--   "W"       stores an unsigned 16 bit immediate from the last operand.
--   "i"       stores an operand sized immediate from the last operand.
--   "I"       dito, but generates an action code to optionally modify
--             the opcode (+2) for a signed 8 bit immediate.
--   "J"       generates one of the REL action codes from the last operand.
--
------------------------------------------------------------------------------

-- Template strings for x86 instructions. Ordered by first opcode byte.
-- Unimplemented opcodes (deliberate omissions) are marked with *.
local map_op = {
  -- 00-05: add...
  -- 06: *push es
  -- 07: *pop es
  -- 08-0D: or...
  -- 0E: *push cs
  -- 0F: two byte opcode prefix
  -- 10-15: adc...
  -- 16: *push ss
  -- 17: *pop ss
  -- 18-1D: sbb...
  -- 1E: *push ds
  -- 1F: *pop ds
  -- 20-25: and...
  es_0 =	"26",
  -- 27: *daa
  -- 28-2D: sub...
  cs_0 =	"2E",
  -- 2F: *das
  -- 30-35: xor...
  ss_0 =	"36",
  -- 37: *aaa
  -- 38-3D: cmp...
  ds_0 =	"3E",
  -- 3F: *aas
  inc_1 =	x64 and "m:FF0m" or "rdw:40r|m:FF0m",
  dec_1 =	x64 and "m:FF1m" or "rdw:48r|m:FF1m",
  push_1 =	(x64 and "rq:n50r|rw:50r|mq:nFF6m|mw:FF6m" or
			 "rdw:50r|mdw:FF6m").."|S.:6AS|ib:n6Ai|i.:68i",
  pop_1 =	x64 and "rq:n58r|rw:58r|mq:n8F0m|mw:8F0m" or "rdw:58r|mdw:8F0m",
  -- 60: *pusha, *pushad, *pushaw
  -- 61: *popa, *popad, *popaw
  -- 62: *bound rdw,x
  -- 63: x86: *arpl mw,rw
  movsxd_2 =	x64 and "rm/qd:63rM",
  fs_0 =	"64",
  gs_0 =	"65",
  o16_0 =	"66",
  a16_0 =	not x64 and "67" or nil,
  a32_0 =	x64 and "67",
  -- 68: push idw
  -- 69: imul rdw,mdw,idw
  -- 6A: push ib
  -- 6B: imul rdw,mdw,S
  -- 6C: *insb
  -- 6D: *insd, *insw
  -- 6E: *outsb
  -- 6F: *outsd, *outsw
  -- 70-7F: jcc lb
  -- 80: add... mb,i
  -- 81: add... mdw,i
  -- 82: *undefined
  -- 83: add... mdw,S
  test_2 =	"mr:85Rm|rm:85rM|Ri:A9ri|mi:F70mi",
  -- 86: xchg rb,mb
  -- 87: xchg rdw,mdw
  -- 88: mov mb,r
  -- 89: mov mdw,r
  -- 8A: mov r,mb
  -- 8B: mov r,mdw
  -- 8C: *mov mdw,seg
  lea_2 =	"rx1dq:8DrM",
  -- 8E: *mov seg,mdw
  -- 8F: pop mdw
  nop_0 =	"90",
  xchg_2 =	"Rrqdw:90R|rRqdw:90r|rm:87rM|mr:87Rm",
  cbw_0 =	"6698",
  cwde_0 =	"98",
  cdqe_0 =	"4898",
  cwd_0 =	"6699",
  cdq_0 =	"99",
  cqo_0 =	"4899",
  -- 9A: *call iw:idw
  wait_0 =	"9B",
  fwait_0 =	"9B",
  pushf_0 =	"9C",
  pushfd_0 =	not x64 and "9C",
  pushfq_0 =	x64 and "9C",
  popf_0 =	"9D",
  popfd_0 =	not x64 and "9D",
  popfq_0 =	x64 and "9D",
  sahf_0 =	"9E",
  lahf_0 =	"9F",
  mov_2 =	"OR:A3o|RO:A1O|mr:89Rm|rm:8BrM|rib:nB0ri|ridw:B8ri|mi:C70mi",
  movsb_0 =	"A4",
  movsw_0 =	"66A5",
  movsd_0 =	"A5",
  cmpsb_0 =	"A6",
  cmpsw_0 =	"66A7",
  cmpsd_0 =	"A7",
  -- A8: test Rb,i
  -- A9: test Rdw,i
  stosb_0 =	"AA",
  stosw_0 =	"66AB",
  stosd_0 =	"AB",
  lodsb_0 =	"AC",
  lodsw_0 =	"66AD",
  lodsd_0 =	"AD",
  scasb_0 =	"AE",
  scasw_0 =	"66AF",
  scasd_0 =	"AF",
  -- B0-B7: mov rb,i
  -- B8-BF: mov rdw,i
  -- C0: rol... mb,i
  -- C1: rol... mdw,i
  ret_1 =	"i.:nC2W",
  ret_0 =	"C3",
  -- C4: *les rdw,mq
  -- C5: *lds rdw,mq
  -- C6: mov mb,i
  -- C7: mov mdw,i
  -- C8: *enter iw,ib
  leave_0 =	"C9",
  -- CA: *retf iw
  -- CB: *retf
  int3_0 =	"CC",
  int_1 =	"i.:nCDU",
  into_0 =	"CE",
  -- CF: *iret
  -- D0: rol... mb,1
  -- D1: rol... mdw,1
  -- D2: rol... mb,cl
  -- D3: rol... mb,cl
  -- D4: *aam ib
  -- D5: *aad ib
  -- D6: *salc
  -- D7: *xlat
  -- D8-DF: floating point ops
  -- E0: *loopne
  -- E1: *loope
  -- E2: *loop
  -- E3: *jcxz, *jecxz
  -- E4: *in Rb,ib
  -- E5: *in Rdw,ib
  -- E6: *out ib,Rb
  -- E7: *out ib,Rdw
  call_1 =	x64 and "mq:nFF2m|J.:E8nJ" or "md:FF2m|J.:E8J",
  jmp_1 =	x64 and "mq:nFF4m|J.:E9nJ" or "md:FF4m|J.:E9J", -- short: EB
  -- EA: *jmp iw:idw
  -- EB: jmp ib
  -- EC: *in Rb,dx
  -- ED: *in Rdw,dx
  -- EE: *out dx,Rb
  -- EF: *out dx,Rdw
  lock_0 =	"F0",
  int1_0 =	"F1",
  repne_0 =	"F2",
  repnz_0 =	"F2",
  rep_0 =	"F3",
  repe_0 =	"F3",
  repz_0 =	"F3",
  endbr32_0 =	"F30F1EFB",
  endbr64_0 =	"F30F1EFA",
  -- F4: *hlt
  cmc_0 =	"F5",
  -- F6: test... mb,i; div... mb
  -- F7: test... mdw,i; div... mdw
  clc_0 =	"F8",
  stc_0 =	"F9",
  -- FA: *cli
  cld_0 =	"FC",
  std_0 =	"FD",
  -- FE: inc... mb
  -- FF: inc... mdw

  -- misc ops
  not_1 =	"m:F72m",
  neg_1 =	"m:F73m",
  mul_1 =	"m:F74m",
  imul_1 =	"m:F75m",
  div_1 =	"m:F76m",
  idiv_1 =	"m:F77m",

  imul_2 =	"rmqdw:0FAFrM|rIqdw:69rmI|rSqdw:6BrmS|riqdw:69rmi",
  imul_3 =	"rmIqdw:69rMI|rmSqdw:6BrMS|rmiqdw:69rMi",

  movzx_2 =	"rm/db:0FB6rM|rm/qb:|rm/wb:0FB6rM|rm/dw:0FB7rM|rm/qw:",
  movsx_2 =	"rm/db:0FBErM|rm/qb:|rm/wb:0FBErM|rm/dw:0FBFrM|rm/qw:",

  bswap_1 =	"rqd:0FC8r",
  bsf_2 =	"rmqdw:0FBCrM",
  bsr_2 =	"rmqdw:0FBDrM",
  bt_2 =	"mrqdw:0FA3Rm|miqdw:0FBA4mU",
  btc_2 =	"mrqdw:0FBBRm|miqdw:0FBA7mU",
  btr_2 =	"mrqdw:0FB3Rm|miqdw:0FBA6mU",
  bts_2 =	"mrqdw:0FABRm|miqdw:0FBA5mU",

  shld_3 =	"mriqdw:0FA4RmU|mrC/qq:0FA5Rm|mrC/dd:|mrC/ww:",
  shrd_3 =	"mriqdw:0FACRmU|mrC/qq:0FADRm|mrC/dd:|mrC/ww:",

  rdtsc_0 =	"0F31", -- P1+
  rdpmc_0 =	"0F33", -- P6+
  cpuid_0 =	"0FA2", -- P1+

  -- floating point ops
  fst_1 =	"ff:DDD0r|xd:D92m|xq:nDD2m",
  fstp_1 =	"ff:DDD8r|xd:D93m|xq:nDD3m|xt:DB7m",
  fld_1 =	"ff:D9C0r|xd:D90m|xq:nDD0m|xt:DB5m",

  fpop_0 =	"DDD8", -- Alias for fstp st0.

  fist_1 =	"xw:nDF2m|xd:DB2m",
  fistp_1 =	"xw:nDF3m|xd:DB3m|xq:nDF7m",
  fild_1 =	"xw:nDF0m|xd:DB0m|xq:nDF5m",

  fxch_0 =	"D9C9",
  fxch_1 =	"ff:D9C8r",
  fxch_2 =	"fFf:D9C8r|Fff:D9C8R",

  fucom_1 =	"ff:DDE0r",
  fucom_2 =	"Fff:DDE0R",
  fucomp_1 =	"ff:DDE8r",
  fucomp_2 =	"Fff:DDE8R",
  fucomi_1 =	"ff:DBE8r", -- P6+
  fucomi_2 =	"Fff:DBE8R", -- P6+
  fucomip_1 =	"ff:DFE8r", -- P6+
  fucomip_2 =	"Fff:DFE8R", -- P6+
  fcomi_1 =	"ff:DBF0r", -- P6+
  fcomi_2 =	"Fff:DBF0R", -- P6+
  fcomip_1 =	"ff:DFF0r", -- P6+
  fcomip_2 =	"Fff:DFF0R", -- P6+
  fucompp_0 =	"DAE9",
  fcompp_0 =	"DED9",

  fldenv_1 =	"x.:D94m",
  fnstenv_1 =	"x.:D96m",
  fstenv_1 =	"x.:9BD96m",
  fldcw_1 =	"xw:nD95m",
  fstcw_1 =	"xw:n9BD97m",
  fnstcw_1 =	"xw:nD97m",
  fstsw_1 =	"Rw:n9BDFE0|xw:n9BDD7m",
  fnstsw_1 =	"Rw:nDFE0|xw:nDD7m",
  fclex_0 =	"9BDBE2",
  fnclex_0 =	"DBE2",

  fnop_0 =	"D9D0",
  -- D9D1-D9DF: unassigned

  fchs_0 =	"D9E0",
  fabs_0 =	"D9E1",
  -- D9E2: unassigned
  -- D9E3: unassigned
  ftst_0 =	"D9E4",
  fxam_0 =	"D9E5",
  -- D9E6: unassigned
  -- D9E7: unassigned
  fld1_0 =	"D9E8",
  fldl2t_0 =	"D9E9",
  fldl2e_0 =	"D9EA",
  fldpi_0 =	"D9EB",
  fldlg2_0 =	"D9EC",
  fldln2_0 =	"D9ED",
  fldz_0 =	"D9EE",
  -- D9EF: unassigned

  f2xm1_0 =	"D9F0",
  fyl2x_0 =	"D9F1",
  fptan_0 =	"D9F2",
  fpatan_0 =	"D9F3",
  fxtract_0 =	"D9F4",
  fprem1_0 =	"D9F5",
  fdecstp_0 =	"D9F6",
  fincstp_0 =	"D9F7",
  fprem_0 =	"D9F8",
  fyl2xp1_0 =	"D9F9",
  fsqrt_0 =	"D9FA",
  fsincos_0 =	"D9FB",
  frndint_0 =	"D9FC",
  fscale_0 =	"D9FD",
  fsin_0 =	"D9FE",
  fcos_0 =	"D9FF",

  -- SSE, SSE2
  andnpd_2 =	"rmo:660F55rM",
  andnps_2 =	"rmo:0F55rM",
  andpd_2 =	"rmo:660F54rM",
  andps_2 =	"rmo:0F54rM",
  clflush_1 =	"x.:0FAE7m",
  cmppd_3 =	"rmio:660FC2rMU",
  cmpps_3 =	"rmio:0FC2rMU",
  cmpsd_3 =	"rrio:F20FC2rMU|rxi/oq:",
  cmpss_3 =	"rrio:F30FC2rMU|rxi/od:",
  comisd_2 =	"rro:660F2FrM|rx/oq:",
  comiss_2 =	"rro:0F2FrM|rx/od:",
  cvtdq2pd_2 =	"rro:F30FE6rM|rx/oq:",
  cvtdq2ps_2 =	"rmo:0F5BrM",
  cvtpd2dq_2 =	"rmo:F20FE6rM",
  cvtpd2ps_2 =	"rmo:660F5ArM",
  cvtpi2pd_2 =	"rx/oq:660F2ArM",
  cvtpi2ps_2 =	"rx/oq:0F2ArM",
  cvtps2dq_2 =	"rmo:660F5BrM",
  cvtps2pd_2 =	"rro:0F5ArM|rx/oq:",
  cvtsd2si_2 =	"rr/do:F20F2DrM|rr/qo:|rx/dq:|rxq:",
  cvtsd2ss_2 =	"rro:F20F5ArM|rx/oq:",
  cvtsi2sd_2 =	"rm/od:F20F2ArM|rm/oq:F20F2ArXM",
  cvtsi2ss_2 =	"rm/od:F30F2ArM|rm/oq:F30F2ArXM",
  cvtss2sd_2 =	"rro:F30F5ArM|rx/od:",
  cvtss2si_2 =	"rr/do:F30F2DrM|rr/qo:|rxd:|rx/qd:",
  cvttpd2dq_2 =	"rmo:660FE6rM",
  cvttps2dq_2 =	"rmo:F30F5BrM",
  cvttsd2si_2 =	"rr/do:F20F2CrM|rr/qo:|rx/dq:|rxq:",
  cvttss2si_2 =	"rr/do:F30F2CrM|rr/qo:|rxd:|rx/qd:",
  fxsave_1 =	"x.:0FAE0m",
  fxrstor_1 =	"x.:0FAE1m",
  ldmxcsr_1 =	"xd:0FAE2m",
  lfence_0 =	"0FAEE8",
  maskmovdqu_2 = "rro:660FF7rM",
  mfence_0 =	"0FAEF0",
  movapd_2 =	"rmo:660F28rM|mro:660F29Rm",
  movaps_2 =	"rmo:0F28rM|mro:0F29Rm",
  movd_2 =	"rm/od:660F6ErM|rm/oq:660F6ErXM|mr/do:660F7ERm|mr/qo:",
  movdqa_2 =	"rmo:660F6FrM|mro:660F7FRm",
  movdqu_2 =	"rmo:F30F6FrM|mro:F30F7FRm",
  movhlps_2 =	"rro:0F12rM",
  movhpd_2 =	"rx/oq:660F16rM|xr/qo:n660F17Rm",
  movhps_2 =	"rx/oq:0F16rM|xr/qo:n0F17Rm",
  movlhps_2 =	"rro:0F16rM",
  movlpd_2 =	"rx/oq:660F12rM|xr/qo:n660F13Rm",
  movlps_2 =	"rx/oq:0F12rM|xr/qo:n0F13Rm",
  movmskpd_2 =	"rr/do:660F50rM",
  movmskps_2 =	"rr/do:0F50rM",
  movntdq_2 =	"xro:660FE7Rm",
  movnti_2 =	"xrqd:0FC3Rm",
  movntpd_2 =	"xro:660F2BRm",
  movntps_2 =	"xro:0F2BRm",
  movq_2 =	"rro:F30F7ErM|rx/oq:|xr/qo:n660FD6Rm",
  movsd_2 =	"rro:F20F10rM|rx/oq:|xr/qo:nF20F11Rm",
  movss_2 =	"rro:F30F10rM|rx/od:|xr/do:F30F11Rm",
  movupd_2 =	"rmo:660F10rM|mro:660F11Rm",
  movups_2 =	"rmo:0F10rM|mro:0F11Rm",
  orpd_2 =	"rmo:660F56rM",
  orps_2 =	"rmo:0F56rM",
  pause_0 =	"F390",
  pextrw_3 =	"rri/do:660FC5rMU|xri/wo:660F3A15nRmU", -- Mem op: SSE4.1 only.
  pinsrw_3 =	"rri/od:660FC4rMU|rxi/ow:",
  pmovmskb_2 =	"rr/do:660FD7rM",
  prefetchnta_1 = "xb:n0F180m",
  prefetcht0_1 = "xb:n0F181m",
  prefetcht1_1 = "xb:n0F182m",
  prefetcht2_1 = "xb:n0F183m",
  pshufd_3 =	"rmio:660F70rMU",
  pshufhw_3 =	"rmio:F30F70rMU",
  pshuflw_3 =	"rmio:F20F70rMU",
  pslld_2 =	"rmo:660FF2rM|rio:660F726mU",
  pslldq_2 =	"rio:660F737mU",
  psllq_2 =	"rmo:660FF3rM|rio:660F736mU",
  psllw_2 =	"rmo:660FF1rM|rio:660F716mU",
  psrad_2 =	"rmo:660FE2rM|rio:660F724mU",
  psraw_2 =	"rmo:660FE1rM|rio:660F714mU",
  psrld_2 =	"rmo:660FD2rM|rio:660F722mU",
  psrldq_2 =	"rio:660F733mU",
  psrlq_2 =	"rmo:660FD3rM|rio:660F732mU",
  psrlw_2 =	"rmo:660FD1rM|rio:660F712mU",
  rcpps_2 =	"rmo:0F53rM",
  rcpss_2 =	"rro:F30F53rM|rx/od:",
  rsqrtps_2 =	"rmo:0F52rM",
  rsqrtss_2 =	"rmo:F30F52rM",
  sfence_0 =	"0FAEF8",
  shufpd_3 =	"rmio:660FC6rMU",
  shufps_3 =	"rmio:0FC6rMU",
  stmxcsr_1 =   "xd:0FAE3m",
  ucomisd_2 =	"rro:660F2ErM|rx/oq:",
  ucomiss_2 =	"rro:0F2ErM|rx/od:",
  unpckhpd_2 =	"rmo:660F15rM",
  unpckhps_2 =	"rmo:0F15rM",
  unpcklpd_2 =	"rmo:660F14rM",
  unpcklps_2 =	"rmo:0F14rM",
  xorpd_2 =	"rmo:660F57rM",
  xorps_2 =	"rmo:0F57rM",

  -- SSE3 ops
  fisttp_1 =	"xw:nDF1m|xd:DB1m|xq:nDD1m",
  addsubpd_2 =	"rmo:660FD0rM",
  addsubps_2 =	"rmo:F20FD0rM",
  haddpd_2 =	"rmo:660F7CrM",
  haddps_2 =	"rmo:F20F7CrM",
  hsubpd_2 =	"rmo:660F7DrM",
  hsubps_2 =	"rmo:F20F7DrM",
  lddqu_2 =	"rxo:F20FF0rM",
  movddup_2 =	"rmo:F20F12rM",
  movshdup_2 =	"rmo:F30F16rM",
  movsldup_2 =	"rmo:F30F12rM",

  -- SSSE3 ops
  pabsb_2 =	"rmo:660F381CrM",
  pabsd_2 =	"rmo:660F381ErM",
  pabsw_2 =	"rmo:660F381DrM",
  palignr_3 =	"rmio:660F3A0FrMU",
  phaddd_2 =	"rmo:660F3802rM",
  phaddsw_2 =	"rmo:660F3803rM",
  phaddw_2 =	"rmo:660F3801rM",
  phsubd_2 =	"rmo:660F3806rM",
  phsubsw_2 =	"rmo:660F3807rM",
  phsubw_2 =	"rmo:660F3805rM",
  pmaddubsw_2 =	"rmo:660F3804rM",
  pmulhrsw_2 =	"rmo:660F380BrM",
  pshufb_2 =	"rmo:660F3800rM",
  psignb_2 =	"rmo:660F3808rM",
  psignd_2 =	"rmo:660F380ArM",
  psignw_2 =	"rmo:660F3809rM",

  -- SSE4.1 ops
  blendpd_3 =	"rmio:660F3A0DrMU",
  blendps_3 =	"rmio:660F3A0CrMU",
  blendvpd_3 =	"rmRo:660F3815rM",
  blendvps_3 =	"rmRo:660F3814rM",
  dppd_3 =	"rmio:660F3A41rMU",
  dpps_3 =	"rmio:660F3A40rMU",
  extractps_3 =	"mri/do:660F3A17RmU|rri/qo:660F3A17RXmU",
  insertps_3 =	"rrio:660F3A41rMU|rxi/od:",
  movntdqa_2 =	"rxo:660F382ArM",
  mpsadbw_3 =	"rmio:660F3A42rMU",
  packusdw_2 =	"rmo:660F382BrM",
  pblendvb_3 =	"rmRo:660F3810rM",
  pblendw_3 =	"rmio:660F3A0ErMU",
  pcmpeqq_2 =	"rmo:660F3829rM",
  pextrb_3 =	"rri/do:660F3A14nRmU|rri/qo:|xri/bo:",
  pextrd_3 =	"mri/do:660F3A16RmU",
  pextrq_3 =	"mri/qo:660F3A16RmU",
  -- pextrw is SSE2, mem operand is SSE4.1 only
  phminposuw_2 = "rmo:660F3841rM",
  pinsrb_3 =	"rri/od:660F3A20nrMU|rxi/ob:",
  pinsrd_3 =	"rmi/od:660F3A22rMU",
  pinsrq_3 =	"rmi/oq:660F3A22rXMU",
  pmaxsb_2 =	"rmo:660F383CrM",
  pmaxsd_2 =	"rmo:660F383DrM",
  pmaxud_2 =	"rmo:660F383FrM",
  pmaxuw_2 =	"rmo:660F383ErM",
  pminsb_2 =	"rmo:660F3838rM",
  pminsd_2 =	"rmo:660F3839rM",
  pminud_2 =	"rmo:660F383BrM",
  pminuw_2 =	"rmo:660F383ArM",
  pmovsxbd_2 =	"rro:660F3821rM|rx/od:",
  pmovsxbq_2 =	"rro:660F3822rM|rx/ow:",
  pmovsxbw_2 =	"rro:660F3820rM|rx/oq:",
  pmovsxdq_2 =	"rro:660F3825rM|rx/oq:",
  pmovsxwd_2 =	"rro:660F3823rM|rx/oq:",
  pmovsxwq_2 =	"rro:660F3824rM|rx/od:",
  pmovzxbd_2 =	"rro:660F3831rM|rx/od:",
  pmovzxbq_2 =	"rro:660F3832rM|rx/ow:",
  pmovzxbw_2 =	"rro:660F3830rM|rx/oq:",
  pmovzxdq_2 =	"rro:660F3835rM|rx/oq:",
  pmovzxwd_2 =	"rro:660F3833rM|rx/oq:",
  pmovzxwq_2 =	"rro:660F3834rM|rx/od:",
  pmuldq_2 =	"rmo:660F3828rM",
  pmulld_2 =	"rmo:660F3840rM",
  ptest_2 =	"rmo:660F3817rM",
  roundpd_3 =	"rmio:660F3A09rMU",
  roundps_3 =	"rmio:660F3A08rMU",
  roundsd_3 =	"rrio:660F3A0BrMU|rxi/oq:",
  roundss_3 =	"rrio:660F3A0ArMU|rxi/od:",

  -- SSE4.2 ops
  crc32_2 =	"rmqd:F20F38F1rM|rm/dw:66F20F38F1rM|rm/db:F20F38F0rM|rm/qb:",
  pcmpestri_3 =	"rmio:660F3A61rMU",
  pcmpestrm_3 =	"rmio:660F3A60rMU",
  pcmpgtq_2 =	"rmo:660F3837rM",
  pcmpistri_3 =	"rmio:660F3A63rMU",
  pcmpistrm_3 =	"rmio:660F3A62rMU",
  popcnt_2 =	"rmqdw:F30FB8rM",

  -- SSE4a
  extrq_2 =	"rro:660F79rM",
  extrq_3 =	"riio:660F780mUU",
  insertq_2 =	"rro:F20F79rM",
  insertq_4 =	"rriio:F20F78rMUU",
  lzcnt_2 =	"rmqdw:F30FBDrM",
  movntsd_2 =	"xr/qo:nF20F2BRm",
  movntss_2 =	"xr/do:F30F2BRm",
  -- popcnt is also in SSE4.2

  -- AES-NI
  aesdec_2 =	"rmo:660F38DErM",
  aesdeclast_2 = "rmo:660F38DFrM",
  aesenc_2 =	"rmo:660F38DCrM",
  aesenclast_2 = "rmo:660F38DDrM",
  aesimc_2 =	"rmo:660F38DBrM",
  aeskeygenassist_3 = "rmio:660F3ADFrMU",
  pclmulqdq_3 =	"rmio:660F3A44rMU",

   -- AVX FP ops
  vaddsubpd_3 =	"rrmoy:660FVD0rM",
  vaddsubps_3 =	"rrmoy:F20FVD0rM",
  vandpd_3 =	"rrmoy:660FV54rM",
  vandps_3 =	"rrmoy:0FV54rM",
  vandnpd_3 =	"rrmoy:660FV55rM",
  vandnps_3 =	"rrmoy:0FV55rM",
  vblendpd_4 =	"rrmioy:660F3AV0DrMU",
  vblendps_4 =	"rrmioy:660F3AV0CrMU",
  vblendvpd_4 =	"rrmroy:660F3AV4BrMs",
  vblendvps_4 =	"rrmroy:660F3AV4ArMs",
  vbroadcastf128_2 = "rx/yo:660F38u1ArM",
  vcmppd_4 =	"rrmioy:660FVC2rMU",
  vcmpps_4 =	"rrmioy:0FVC2rMU",
  vcmpsd_4 =	"rrrio:F20FVC2rMU|rrxi/ooq:",
  vcmpss_4 =	"rrrio:F30FVC2rMU|rrxi/ood:",
  vcomisd_2 =	"rro:660Fu2FrM|rx/oq:",
  vcomiss_2 =	"rro:0Fu2FrM|rx/od:",
  vcvtdq2pd_2 =	"rro:F30FuE6rM|rx/oq:|rm/yo:",
  vcvtdq2ps_2 =	"rmoy:0Fu5BrM",
  vcvtpd2dq_2 =	"rmoy:F20FuE6rM",
  vcvtpd2ps_2 =	"rmoy:660Fu5ArM",
  vcvtps2dq_2 =	"rmoy:660Fu5BrM",
  vcvtps2pd_2 =	"rro:0Fu5ArM|rx/oq:|rm/yo:",
  vcvtsd2si_2 =	"rr/do:F20Fu2DrM|rx/dq:|rr/qo:|rxq:",
  vcvtsd2ss_3 =	"rrro:F20FV5ArM|rrx/ooq:",
  vcvtsi2sd_3 =	"rrm/ood:F20FV2ArM|rrm/ooq:F20FVX2ArM",
  vcvtsi2ss_3 =	"rrm/ood:F30FV2ArM|rrm/ooq:F30FVX2ArM",
  vcvtss2sd_3 =	"rrro:F30FV5ArM|rrx/ood:",
  vcvtss2si_2 =	"rr/do:F30Fu2DrM|rxd:|rr/qo:|rx/qd:",
  vcvttpd2dq_2 = "rmo:660FuE6rM|rm/oy:660FuLE6rM",
  vcvttps2dq_2 = "rmoy:F30Fu5BrM",
  vcvttsd2si_2 = "rr/do:F20Fu2CrM|rx/dq:|rr/qo:|rxq:",
  vcvttss2si_2 = "rr/do:F30Fu2CrM|rxd:|rr/qo:|rx/qd:",
  vdppd_4 =	"rrmio:660F3AV41rMU",
  vdpps_4 =	"rrmioy:660F3AV40rMU",
  vextractf128_3 = "mri/oy:660F3AuL19RmU",
  vextractps_3 = "mri/do:660F3Au17RmU",
  vhaddpd_3 =	"rrmoy:660FV7CrM",
  vhaddps_3 =	"rrmoy:F20FV7CrM",
  vhsubpd_3 =	"rrmoy:660FV7DrM",
  vhsubps_3 =	"rrmoy:F20FV7DrM",
  vinsertf128_4 = "rrmi/yyo:660F3AV18rMU",
  vinsertps_4 =	"rrrio:660F3AV21rMU|rrxi/ood:",
  vldmxcsr_1 =	"xd:0FuAE2m",
  vmaskmovps_3 = "rrxoy:660F38V2CrM|xrroy:660F38V2ERm",
  vmaskmovpd_3 = "rrxoy:660F38V2DrM|xrroy:660F38V2FRm",
  vmovapd_2 =	"rmoy:660Fu28rM|mroy:660Fu29Rm",
  vmovaps_2 =	"rmoy:0Fu28rM|mroy:0Fu29Rm",
  vmovd_2 =	"rm/od:660Fu6ErM|rm/oq:660FuX6ErM|mr/do:660Fu7ERm|mr/qo:",
  vmovq_2 =	"rro:F30Fu7ErM|rx/oq:|xr/qo:660FuD6Rm",
  vmovddup_2 =	"rmy:F20Fu12rM|rro:|rx/oq:",
  vmovhlps_3 =	"rrro:0FV12rM",
  vmovhpd_2 =	"xr/qo:660Fu17Rm",
  vmovhpd_3 =	"rrx/ooq:660FV16rM",
  vmovhps_2 =	"xr/qo:0Fu17Rm",
  vmovhps_3 =	"rrx/ooq:0FV16rM",
  vmovlhps_3 =	"rrro:0FV16rM",
  vmovlpd_2 =	"xr/qo:660Fu13Rm",
  vmovlpd_3 =	"rrx/ooq:660FV12rM",
  vmovlps_2 =	"xr/qo:0Fu13Rm",
  vmovlps_3 =	"rrx/ooq:0FV12rM",
  vmovmskpd_2 =	"rr/do:660Fu50rM|rr/dy:660FuL50rM",
  vmovmskps_2 =	"rr/do:0Fu50rM|rr/dy:0FuL50rM",
  vmovntpd_2 =	"xroy:660Fu2BRm",
  vmovntps_2 =	"xroy:0Fu2BRm",
  vmovsd_2 =	"rx/oq:F20Fu10rM|xr/qo:F20Fu11Rm",
  vmovsd_3 =	"rrro:F20FV10rM",
  vmovshdup_2 =	"rmoy:F30Fu16rM",
  vmovsldup_2 =	"rmoy:F30Fu12rM",
  vmovss_2 =	"rx/od:F30Fu10rM|xr/do:F30Fu11Rm",
  vmovss_3 =	"rrro:F30FV10rM",
  vmovupd_2 =	"rmoy:660Fu10rM|mroy:660Fu11Rm",
  vmovups_2 =	"rmoy:0Fu10rM|mroy:0Fu11Rm",
  vorpd_3 =	"rrmoy:660FV56rM",
  vorps_3 =	"rrmoy:0FV56rM",
  vpermilpd_3 =	"rrmoy:660F38V0DrM|rmioy:660F3Au05rMU",
  vpermilps_3 =	"rrmoy:660F38V0CrM|rmioy:660F3Au04rMU",
  vperm2f128_4 = "rrmiy:660F3AV06rMU",
  vptestpd_2 =	"rmoy:660F38u0FrM",
  vptestps_2 =	"rmoy:660F38u0ErM",
  vrcpps_2 =	"rmoy:0Fu53rM",
  vrcpss_3 =	"rrro:F30FV53rM|rrx/ood:",
  vrsqrtps_2 =	"rmoy:0Fu52rM",
  vrsqrtss_3 =	"rrro:F30FV52rM|rrx/ood:",
  vroundpd_3 =	"rmioy:660F3Au09rMU",
  vroundps_3 =	"rmioy:660F3Au08rMU",
  vroundsd_4 =	"rrrio:660F3AV0BrMU|rrxi/ooq:",
  vroundss_4 =	"rrrio:660F3AV0ArMU|rrxi/ood:",
  vshufpd_4 =	"rrmioy:660FVC6rMU",
  vshufps_4 =	"rrmioy:0FVC6rMU",
  vsqrtps_2 =	"rmoy:0Fu51rM",
  vsqrtss_2 =	"rro:F30Fu51rM|rx/od:",
  vsqrtpd_2 =	"rmoy:660Fu51rM",
  vsqrtsd_2 =	"rro:F20Fu51rM|rx/oq:",
  vstmxcsr_1 =	"xd:0FuAE3m",
  vucomisd_2 =	"rro:660Fu2ErM|rx/oq:",
  vucomiss_2 =	"rro:0Fu2ErM|rx/od:",
  vunpckhpd_3 =	"rrmoy:660FV15rM",
  vunpckhps_3 =	"rrmoy:0FV15rM",
  vunpcklpd_3 =	"rrmoy:660FV14rM",
  vunpcklps_3 =	"rrmoy:0FV14rM",
  vxorpd_3 =	"rrmoy:660FV57rM",
  vxorps_3 =	"rrmoy:0FV57rM",
  vzeroall_0 =	"0FuL77",
  vzeroupper_0 = "0Fu77",

  -- AVX2 FP ops
  vbroadcastss_2 = "rx/od:660F38u18rM|rx/yd:|rro:|rr/yo:",
  vbroadcastsd_2 = "rx/yq:660F38u19rM|rr/yo:",
  -- *vgather* (!vsib)
  vpermpd_3 =	"rmiy:660F3AuX01rMU",
  vpermps_3 =	"rrmy:660F38V16rM",

  -- AVX, AVX2 integer ops
  -- In general, xmm requires AVX, ymm requires AVX2.
  vaesdec_3 =  "rrmo:660F38VDErM",
  vaesdeclast_3 = "rrmo:660F38VDFrM",
  vaesenc_3 =  "rrmo:660F38VDCrM",
  vaesenclast_3 = "rrmo:660F38VDDrM",
  vaesimc_2 =  "rmo:660F38uDBrM",
  vaeskeygenassist_3 = "rmio:660F3AuDFrMU",
  vlddqu_2 =	"rxoy:F20FuF0rM",
  vmaskmovdqu_2 = "rro:660FuF7rM",
  vmovdqa_2 =	"rmoy:660Fu6FrM|mroy:660Fu7FRm",
  vmovdqu_2 =	"rmoy:F30Fu6FrM|mroy:F30Fu7FRm",
  vmovntdq_2 =	"xroy:660FuE7Rm",
  vmovntdqa_2 =	"rxoy:660F38u2ArM",
  vmpsadbw_4 =	"rrmioy:660F3AV42rMU",
  vpabsb_2 =	"rmoy:660F38u1CrM",
  vpabsd_2 =	"rmoy:660F38u1ErM",
  vpabsw_2 =	"rmoy:660F38u1DrM",
  vpackusdw_3 =	"rrmoy:660F38V2BrM",
  vpalignr_4 =	"rrmioy:660F3AV0FrMU",
  vpblendvb_4 =	"rrmroy:660F3AV4CrMs",
  vpblendw_4 =	"rrmioy:660F3AV0ErMU",
  vpclmulqdq_4 = "rrmio:660F3AV44rMU",
  vpcmpeqq_3 =	"rrmoy:660F38V29rM",
  vpcmpestri_3 = "rmio:660F3Au61rMU",
  vpcmpestrm_3 = "rmio:660F3Au60rMU",
  vpcmpgtq_3 =	"rrmoy:660F38V37rM",
  vpcmpistri_3 = "rmio:660F3Au63rMU",
  vpcmpistrm_3 = "rmio:660F3Au62rMU",
  vpextrb_3 =	"rri/do:660F3Au14nRmU|rri/qo:|xri/bo:",
  vpextrw_3 =	"rri/do:660FuC5rMU|xri/wo:660F3Au15nRmU",
  vpextrd_3 =	"mri/do:660F3Au16RmU",
  vpextrq_3 =	"mri/qo:660F3Au16RmU",
  vphaddw_3 =	"rrmoy:660F38V01rM",
  vphaddd_3 =	"rrmoy:660F38V02rM",
  vphaddsw_3 =	"rrmoy:660F38V03rM",
  vphminposuw_2 = "rmo:660F38u41rM",
  vphsubw_3 =	"rrmoy:660F38V05rM",
  vphsubd_3 =	"rrmoy:660F38V06rM",
  vphsubsw_3 =	"rrmoy:660F38V07rM",
  vpinsrb_4 =	"rrri/ood:660F3AV20rMU|rrxi/oob:",
  vpinsrw_4 =	"rrri/ood:660FVC4rMU|rrxi/oow:",
  vpinsrd_4 =	"rrmi/ood:660F3AV22rMU",
  vpinsrq_4 =	"rrmi/ooq:660F3AVX22rMU",
  vpmaddubsw_3 = "rrmoy:660F38V04rM",
  vpmaxsb_3 =	"rrmoy:660F38V3CrM",
  vpmaxsd_3 =	"rrmoy:660F38V3DrM",
  vpmaxuw_3 =	"rrmoy:660F38V3ErM",
  vpmaxud_3 =	"rrmoy:660F38V3FrM",
  vpminsb_3 =	"rrmoy:660F38V38rM",
  vpminsd_3 =	"rrmoy:660F38V39rM",
  vpminuw_3 =	"rrmoy:660F38V3ArM",
  vpminud_3 =	"rrmoy:660F38V3BrM",
  vpmovmskb_2 =	"rr/do:660FuD7rM|rr/dy:660FuLD7rM",
  vpmovsxbw_2 =	"rroy:660F38u20rM|rx/oq:|rx/yo:",
  vpmovsxbd_2 =	"rroy:660F38u21rM|rx/od:|rx/yq:",
  vpmovsxbq_2 =	"rroy:660F38u22rM|rx/ow:|rx/yd:",
  vpmovsxwd_2 =	"rroy:660F38u23rM|rx/oq:|rx/yo:",
  vpmovsxwq_2 =	"rroy:660F38u24rM|rx/od:|rx/yq:",
  vpmovsxdq_2 =	"rroy:660F38u25rM|rx/oq:|rx/yo:",
  vpmovzxbw_2 =	"rroy:660F38u30rM|rx/oq:|rx/yo:",
  vpmovzxbd_2 =	"rroy:660F38u31rM|rx/od:|rx/yq:",
  vpmovzxbq_2 =	"rroy:660F38u32rM|rx/ow:|rx/yd:",
  vpmovzxwd_2 =	"rroy:660F38u33rM|rx/oq:|rx/yo:",
  vpmovzxwq_2 =	"rroy:660F38u34rM|rx/od:|rx/yq:",
  vpmovzxdq_2 =	"rroy:660F38u35rM|rx/oq:|rx/yo:",
  vpmuldq_3 =	"rrmoy:660F38V28rM",
  vpmulhrsw_3 =	"rrmoy:660F38V0BrM",
  vpmulld_3 =	"rrmoy:660F38V40rM",
  vpshufb_3 =	"rrmoy:660F38V00rM",
  vpshufd_3 =	"rmioy:660Fu70rMU",
  vpshufhw_3 =	"rmioy:F30Fu70rMU",
  vpshuflw_3 =	"rmioy:F20Fu70rMU",
  vpsignb_3 =	"rrmoy:660F38V08rM",
  vpsignw_3 =	"rrmoy:660F38V09rM",
  vpsignd_3 =	"rrmoy:660F38V0ArM",
  vpslldq_3 =	"rrioy:660Fv737mU",
  vpsllw_3 =	"rrmoy:660FVF1rM|rrioy:660Fv716mU",
  vpslld_3 =	"rrmoy:660FVF2rM|rrioy:660Fv726mU",
  vpsllq_3 =	"rrmoy:660FVF3rM|rrioy:660Fv736mU",
  vpsraw_3 =	"rrmoy:660FVE1rM|rrioy:660Fv714mU",
  vpsrad_3 =	"rrmoy:660FVE2rM|rrioy:660Fv724mU",
  vpsrldq_3 =	"rrioy:660Fv733mU",
  vpsrlw_3 =	"rrmoy:660FVD1rM|rrioy:660Fv712mU",
  vpsrld_3 =	"rrmoy:660FVD2rM|rrioy:660Fv722mU",
  vpsrlq_3 =	"rrmoy:660FVD3rM|rrioy:660Fv732mU",
  vptest_2 =	"rmoy:660F38u17rM",

  -- AVX2 integer ops
  vbroadcasti128_2 = "rx/yo:660F38u5ArM",
  vinserti128_4 = "rrmi/yyo:660F3AV38rMU",
  vextracti128_3 = "mri/oy:660F3AuL39RmU",
  vpblendd_4 =	"rrmioy:660F3AV02rMU",
  vpbroadcastb_2 = "rro:660F38u78rM|rx/ob:|rr/yo:|rx/yb:",
  vpbroadcastw_2 = "rro:660F38u79rM|rx/ow:|rr/yo:|rx/yw:",
  vpbroadcastd_2 = "rro:660F38u58rM|rx/od:|rr/yo:|rx/yd:",
  vpbroadcastq_2 = "rro:660F38u59rM|rx/oq:|rr/yo:|rx/yq:",
  vpermd_3 =	"rrmy:660F38V36rM",
  vpermq_3 =	"rmiy:660F3AuX00rMU",
  -- *vpgather* (!vsib)
  vperm2i128_4 = "rrmiy:660F3AV46rMU",
  vpmaskmovd_3 = "rrxoy:660F38V8CrM|xrroy:660F38V8ERm",
  vpmaskmovq_3 = "rrxoy:660F38VX8CrM|xrroy:660F38VX8ERm",
  vpsllvd_3 =	"rrmoy:660F38V47rM",
  vpsllvq_3 =	"rrmoy:660F38VX47rM",
  vpsravd_3 =	"rrmoy:660F38V46rM",
  vpsrlvd_3 =	"rrmoy:660F38V45rM",
  vpsrlvq_3 =	"rrmoy:660F38VX45rM",

  -- Intel ADX
  adcx_2 =	"rmqd:660F38F6rM",
  adox_2 =	"rmqd:F30F38F6rM",

  -- BMI1
  andn_3 =	"rrmqd:0F38VF2rM",
  bextr_3 =	"rmrqd:0F38wF7rM",
  blsi_2 =	"rmqd:0F38vF33m",
  blsmsk_2 =	"rmqd:0F38vF32m",
  blsr_2 =	"rmqd:0F38vF31m",
  tzcnt_2 =	"rmqdw:F30FBCrM",

  -- BMI2
  bzhi_3 =	"rmrqd:0F38wF5rM",
  mulx_3 =	"rrmqd:F20F38VF6rM",
  pdep_3 =	"rrmqd:F20F38VF5rM",
  pext_3 =	"rrmqd:F30F38VF5rM",
  rorx_3 =	"rmSqd:F20F3AuF0rMS",
  sarx_3 =	"rmrqd:F30F38wF7rM",
  shrx_3 =	"rmrqd:F20F38wF7rM",
  shlx_3 =	"rmrqd:660F38wF7rM",

  -- FMA3
  vfmaddsub132pd_3 = "rrmoy:660F38VX96rM",
  vfmaddsub132ps_3 = "rrmoy:660F38V96rM",
  vfmaddsub213pd_3 = "rrmoy:660F38VXA6rM",
  vfmaddsub213ps_3 = "rrmoy:660F38VA6rM",
  vfmaddsub231pd_3 = "rrmoy:660F38VXB6rM",
  vfmaddsub231ps_3 = "rrmoy:660F38VB6rM",

  vfmsubadd132pd_3 = "rrmoy:660F38VX97rM",
  vfmsubadd132ps_3 = "rrmoy:660F38V97rM",
  vfmsubadd213pd_3 = "rrmoy:660F38VXA7rM",
  vfmsubadd213ps_3 = "rrmoy:660F38VA7rM",
  vfmsubadd231pd_3 = "rrmoy:660F38VXB7rM",
  vfmsubadd231ps_3 = "rrmoy:660F38VB7rM",

  vfmadd132pd_3 = "rrmoy:660F38VX98rM",
  vfmadd132ps_3 = "rrmoy:660F38V98rM",
  vfmadd132sd_3 = "rrro:660F38VX99rM|rrx/ooq:",
  vfmadd132ss_3 = "rrro:660F38V99rM|rrx/ood:",
  vfmadd213pd_3 = "rrmoy:660F38VXA8rM",
  vfmadd213ps_3 = "rrmoy:660F38VA8rM",
  vfmadd213sd_3 = "rrro:660F38VXA9rM|rrx/ooq:",
  vfmadd213ss_3 = "rrro:660F38VA9rM|rrx/ood:",
  vfmadd231pd_3 = "rrmoy:660F38VXB8rM",
  vfmadd231ps_3 = "rrmoy:660F38VB8rM",
  vfmadd231sd_3 = "rrro:660F38VXB9rM|rrx/ooq:",
  vfmadd231ss_3 = "rrro:660F38VB9rM|rrx/ood:",

  vfmsub132pd_3 = "rrmoy:660F38VX9ArM",
  vfmsub132ps_3 = "rrmoy:660F38V9ArM",
  vfmsub132sd_3 = "rrro:660F38VX9BrM|rrx/ooq:",
  vfmsub132ss_3 = "rrro:660F38V9BrM|rrx/ood:",
  vfmsub213pd_3 = "rrmoy:660F38VXAArM",
  vfmsub213ps_3 = "rrmoy:660F38VAArM",
  vfmsub213sd_3 = "rrro:660F38VXABrM|rrx/ooq:",
  vfmsub213ss_3 = "rrro:660F38VABrM|rrx/ood:",
  vfmsub231pd_3 = "rrmoy:660F38VXBArM",
  vfmsub231ps_3 = "rrmoy:660F38VBArM",
  vfmsub231sd_3 = "rrro:660F38VXBBrM|rrx/ooq:",
  vfmsub231ss_3 = "rrro:660F38VBBrM|rrx/ood:",

  vfnmadd132pd_3 = "rrmoy:660F38VX9CrM",
  vfnmadd132ps_3 = "rrmoy:660F38V9CrM",
  vfnmadd132sd_3 = "rrro:660F38VX9DrM|rrx/ooq:",
  vfnmadd132ss_3 = "rrro:660F38V9DrM|rrx/ood:",
  vfnmadd213pd_3 = "rrmoy:660F38VXACrM",
  vfnmadd213ps_3 = "rrmoy:660F38VACrM",
  vfnmadd213sd_3 = "rrro:660F38VXADrM|rrx/ooq:",
  vfnmadd213ss_3 = "rrro:660F38VADrM|rrx/ood:",
  vfnmadd231pd_3 = "rrmoy:660F38VXBCrM",
  vfnmadd231ps_3 = "rrmoy:660F38VBCrM",
  vfnmadd231sd_3 = "rrro:660F38VXBDrM|rrx/ooq:",
  vfnmadd231ss_3 = "rrro:660F38VBDrM|rrx/ood:",

  vfnmsub132pd_3 = "rrmoy:660F38VX9ErM",
  vfnmsub132ps_3 = "rrmoy:660F38V9ErM",
  vfnmsub132sd_3 = "rrro:660F38VX9FrM|rrx/ooq:",
  vfnmsub132ss_3 = "rrro:660F38V9FrM|rrx/ood:",
  vfnmsub213pd_3 = "rrmoy:660F38VXAErM",
  vfnmsub213ps_3 = "rrmoy:660F38VAErM",
  vfnmsub213sd_3 = "rrro:660F38VXAFrM|rrx/ooq:",
  vfnmsub213ss_3 = "rrro:660F38VAFrM|rrx/ood:",
  vfnmsub231pd_3 = "rrmoy:660F38VXBErM",
  vfnmsub231ps_3 = "rrmoy:660F38VBErM",
  vfnmsub231sd_3 = "rrro:660F38VXBFrM|rrx/ooq:",
  vfnmsub231ss_3 = "rrro:660F38VBFrM|rrx/ood:",
}

------------------------------------------------------------------------------

-- Arithmetic ops.
for name,n in pairs{ add = 0, ["or"] = 1, adc = 2, sbb = 3,
		     ["and"] = 4, sub = 5, xor = 6, cmp = 7 } do
  local n8 = shl(n, 3)
  map_op[name.."_2"] = format(
    "mr:%02XRm|rm:%02XrM|mI1qdw:81%XmI|mS1qdw:83%XmS|Ri1qdwb:%02Xri|mi1qdwb:81%Xmi",
    1+n8, 3+n8, n, n, 5+n8, n)
end

-- Shift ops.
for name,n in pairs{ rol = 0, ror = 1, rcl = 2, rcr = 3,
		     shl = 4, shr = 5,          sar = 7, sal = 4 } do
  map_op[name.."_2"] = format("m1:D1%Xm|mC1qdwb:D3%Xm|mi:C1%XmU", n, n, n)
end

-- Conditional ops.
for cc,n in pairs(map_cc) do
  map_op["j"..cc.."_1"] = format("J.:n0F8%XJ", n) -- short: 7%X
  map_op["set"..cc.."_1"] = format("mb:n0F9%X2m", n)
  map_op["cmov"..cc.."_2"] = format("rmqdw:0F4%XrM", n) -- P6+
end

-- FP arithmetic ops.
for name,n in pairs{ add = 0, mul = 1, com = 2, comp = 3,
		     sub = 4, subr = 5, div = 6, divr = 7 } do
  local nc = 0xc0 + shl(n, 3)
  local nr = nc + (n < 4 and 0 or (n % 2 == 0 and 8 or -8))
  local fn = "f"..name
  map_op[fn.."_1"] = format("ff:D8%02Xr|xd:D8%Xm|xq:nDC%Xm", nc, n, n)
  if n == 2 or n == 3 then
    map_op[fn.."_2"] = format("Fff:D8%02XR|Fx2d:D8%XM|Fx2q:nDC%XM", nc, n, n)
  else
    map_op[fn.."_2"] = format("Fff:D8%02XR|fFf:DC%02Xr|Fx2d:D8%XM|Fx2q:nDC%XM", nc, nr, n, n)
    map_op[fn.."p_1"] = format("ff:DE%02Xr", nr)
    map_op[fn.."p_2"] = format("fFf:DE%02Xr", nr)
  end
  map_op["fi"..name.."_1"] = format("xd:DA%Xm|xw:nDE%Xm", n, n)
end

-- FP conditional moves.
for cc,n in pairs{ b=0, e=1, be=2, u=3, nb=4, ne=5, nbe=6, nu=7 } do
  local nc = 0xdac0 + shl(band(n, 3), 3) + shl(band(n, 4), 6)
  map_op["fcmov"..cc.."_1"] = format("ff:%04Xr", nc) -- P6+
  map_op["fcmov"..cc.."_2"] = format("Fff:%04XR", nc) -- P6+
end

-- SSE / AVX FP arithmetic ops.
for name,n in pairs{ sqrt = 1, add = 8, mul = 9,
		     sub = 12, min = 13, div = 14, max = 15 } do
  map_op[name.."ps_2"] = format("rmo:0F5%XrM", n)
  map_op[name.."ss_2"] = format("rro:F30F5%XrM|rx/od:", n)
  map_op[name.."pd_2"] = format("rmo:660F5%XrM", n)
  map_op[name.."sd_2"] = format("rro:F20F5%XrM|rx/oq:", n)
  if n ~= 1 then
    map_op["v"..name.."ps_3"] = format("rrmoy:0FV5%XrM", n)
    map_op["v"..name.."ss_3"] = format("rrro:F30FV5%XrM|rrx/ood:", n)
    map_op["v"..name.."pd_3"] = format("rrmoy:660FV5%XrM", n)
    map_op["v"..name.."sd_3"] = format("rrro:F20FV5%XrM|rrx/ooq:", n)
  end
end

-- SSE2 / AVX / AVX2 integer arithmetic ops (66 0F leaf).
for name,n in pairs{
  paddb = 0xFC, paddw = 0xFD, paddd = 0xFE, paddq = 0xD4,
  paddsb = 0xEC, paddsw = 0xED, packssdw = 0x6B,
  packsswb = 0x63, packuswb = 0x67, paddusb = 0xDC,
  paddusw = 0xDD, pand = 0xDB, pandn = 0xDF, pavgb = 0xE0,
  pavgw = 0xE3, pcmpeqb = 0x74, pcmpeqd = 0x76,
  pcmpeqw = 0x75, pcmpgtb = 0x64, pcmpgtd = 0x66,
  pcmpgtw = 0x65, pmaddwd = 0xF5, pmaxsw = 0xEE,
  pmaxub = 0xDE, pminsw = 0xEA, pminub = 0xDA,
  pmulhuw = 0xE4, pmulhw = 0xE5, pmullw = 0xD5,
  pmuludq = 0xF4, por = 0xEB, psadbw = 0xF6, psubb = 0xF8,
  psubw = 0xF9, psubd = 0xFA, psubq = 0xFB, psubsb = 0xE8,
  psubsw = 0xE9, psubusb = 0xD8, psubusw = 0xD9,
  punpckhbw = 0x68, punpckhwd = 0x69, punpckhdq = 0x6A,
  punpckhqdq = 0x6D, punpcklbw = 0x60, punpcklwd = 0x61,
  punpckldq = 0x62, punpcklqdq = 0x6C, pxor = 0xEF
} do
  map_op[name.."_2"] = format("rmo:660F%02XrM", n)
  map_op["v"..name.."_3"] = format("rrmoy:660FV%02XrM", n)
end

------------------------------------------------------------------------------

local map_vexarg = { u = false, v = 1, V = 2, w = 3 }

-- Process pattern string.
local function dopattern(pat, args, sz, op, needrex)
  local digit, addin, vex
  local opcode = 0
  local szov = sz
  local narg = 1
  local rex = 0

  -- Limit number of section buffer positions used by a single dasm_put().
  -- A single opcode needs a maximum of 6 positions.
  if secpos+6 > maxsecpos then wflush() end

  -- Process each character.
  for c in gmatch(pat.."|", ".") do
    if match(c, "%x") then	-- Hex digit.
      digit = byte(c) - 48
      if digit > 48 then digit = digit - 39
      elseif digit > 16 then digit = digit - 7 end
      opcode = opcode*16 + digit
      addin = nil
    elseif c == "n" then	-- Disable operand size mods for opcode.
      szov = nil
    elseif c == "X" then	-- Force REX.W.
      rex = 8
    elseif c == "L" then	-- Force VEX.L.
      vex.l = true
    elseif c == "r" then	-- Merge 1st operand regno. into opcode.
      addin = args[1]; opcode = opcode + (addin.reg % 8)
      if narg < 2 then narg = 2 end
    elseif c == "R" then	-- Merge 2nd operand regno. into opcode.
      addin = args[2]; opcode = opcode + (addin.reg % 8)
      narg = 3
    elseif c == "m" or c == "M" then	-- Encode ModRM/SIB.
      local s
      if addin then
	s = addin.reg
	opcode = opcode - band(s, 7)	-- Undo regno opcode merge.
      else
	s = band(opcode, 15)	-- Undo last digit.
	opcode = shr(opcode, 4)
      end
      local nn = c == "m" and 1 or 2
      local t = args[nn]
      if narg <= nn then narg = nn + 1 end
      if szov == "q" and rex == 0 then rex = rex + 8 end
      if t.reg and t.reg > 7 then rex = rex + 1 end
      if t.xreg and t.xreg > 7 then rex = rex + 2 end
      if s > 7 then rex = rex + 4 end
      if needrex then rex = rex + 16 end
      local psz, sk = wputop(szov, opcode, rex, vex, s < 0, t.vreg or t.vxreg)
      opcode = nil
      local imark = sub(pat, -1) -- Force a mark (ugly).
      -- Put ModRM/SIB with regno/last digit as spare.
      wputmrmsib(t, imark, s, addin and addin.vreg, psz, sk)
      addin = nil
    elseif map_vexarg[c] ~= nil then -- Encode using VEX prefix
      local b = band(opcode, 255); opcode = shr(opcode, 8)
      local m = 1
      if b == 0x38 then m = 2
      elseif b == 0x3a then m = 3 end
      if m ~= 1 then b = band(opcode, 255); opcode = shr(opcode, 8) end
      if b ~= 0x0f then
	werror("expected `0F', `0F38', or `0F3A' to precede `"..c..
	  "' in pattern `"..pat.."' for `"..op.."'")
      end
      local v = map_vexarg[c]
      if v then v = remove(args, v) end
      b = band(opcode, 255)
      local p = 0
      if b == 0x66 then p = 1
      elseif b == 0xf3 then p = 2
      elseif b == 0xf2 then p = 3 end
      if p ~= 0 then opcode = shr(opcode, 8) end
      if opcode ~= 0 then wputop(nil, opcode, 0); opcode = 0 end
      vex = { m = m, p = p, v = v }
    else
      if opcode then -- Flush opcode.
	if szov == "q" and rex == 0 then rex = rex + 8 end
	if needrex then rex = rex + 16 end
	if addin and addin.reg == -1 then
	  local psz, sk = wputop(szov, opcode - 7, rex, vex, true)
	  wvreg("opcode", addin.vreg, psz, sk)
	else
	  if addin and addin.reg > 7 then rex = rex + 1 end
	  wputop(szov, opcode, rex, vex)
	end
	opcode = nil
      end
      if c == "|" then break end
      if c == "o" then -- Offset (pure 32 bit displacement).
	wputdarg(args[1].disp); if narg < 2 then narg = 2 end
      elseif c == "O" then
	wputdarg(args[2].disp); narg = 3
      else
	-- Anything else is an immediate operand.
	local a = args[narg]
	narg = narg + 1
	local mode, imm = a.mode, a.imm
	if mode == "iJ" and not match(x64 and "J" or "iIJ", c) then
	  werror("bad operand size for label")
	end
	if c == "S" then
	  wputsbarg(imm)
	elseif c == "U" then
	  wputbarg(imm)
	elseif c == "W" then
	  wputwarg(imm)
	elseif c == "i" or c == "I" then
	  if mode == "iJ" then
	    wputlabel("IMM_", imm, 1)
	  elseif mode == "iI" and c == "I" then
	    waction(sz == "w" and "IMM_WB" or "IMM_DB", imm)
	  else
	    wputszarg(sz, imm)
	  end
	elseif c == "J" then
	  if mode == "iPJ" then
	    waction("REL_A", imm) -- !x64 (secpos)
	  else
	    wputlabel("REL_", imm, 2)
	  end
	elseif c == "s" then
	  local reg = a.reg
	  if reg < 0 then
	    wputb(0)
	    wvreg("imm.hi", a.vreg)
	  else
	    wputb(shl(reg, 4))
	  end
	else
	  werror("bad char `"..c.."' in pattern `"..pat.."' for `"..op.."'")
	end
      end
    end
  end
end

------------------------------------------------------------------------------

-- Mapping of operand modes to short names. Suppress output with '#'.
local map_modename = {
  r = "reg", R = "eax", C = "cl", x = "mem", m = "mrm", i = "imm",
  f = "stx", F = "st0", J = "lbl", ["1"] = "1",
  I = "#", S = "#", O = "#",
}

-- Return a table/string showing all possible operand modes.
local function templatehelp(template, nparams)
  if nparams == 0 then return "" end
  local t = {}
  for tm in gmatch(template, "[^%|]+") do
    local s = map_modename[sub(tm, 1, 1)]
    s = s..gsub(sub(tm, 2, nparams), ".", function(c)
      return ", "..map_modename[c]
    end)
    if not match(s, "#") then t[#t+1] = s end
  end
  return t
end

-- Match operand modes against mode match part of template.
local function matchtm(tm, args)
  for i=1,#args do
    if not match(args[i].mode, sub(tm, i, i)) then return end
  end
  return true
end

-- Handle opcodes defined with template strings.
map_op[".template__"] = function(params, template, nparams)
  if not params then return templatehelp(template, nparams) end
  local args = {}

  -- Zero-operand opcodes have no match part.
  if #params == 0 then
    dopattern(template, args, "d", params.op, nil)
    return
  end

  -- Determine common operand size (coerce undefined size) or flag as mixed.
  local sz, szmix, needrex
  for i,p in ipairs(params) do
    args[i] = parseoperand(p)
    local nsz = args[i].opsize
    if nsz then
      if sz and sz ~= nsz then szmix = true else sz = nsz end
    end
    local nrex = args[i].needrex
    if nrex ~= nil then
      if needrex == nil then
	needrex = nrex
      elseif needrex ~= nrex then
	werror("bad mix of byte-addressable registers")
      end
    end
  end

  -- Try all match:pattern pairs (separated by '|').
  local gotmatch, lastpat
  for tm in gmatch(template, "[^%|]+") do
    -- Split off size match (starts after mode match) and pattern string.
    local szm, pat = match(tm, "^(.-):(.*)$", #args+1)
    if pat == "" then pat = lastpat else lastpat = pat end
    if matchtm(tm, args) then
      local prefix = sub(szm, 1, 1)
      if prefix == "/" then -- Exactly match leading operand sizes.
	for i = #szm,1,-1 do
	  if i == 1 then
	    dopattern(pat, args, sz, params.op, needrex) -- Process pattern.
	    return
	  elseif args[i-1].opsize ~= sub(szm, i, i) then
	    break
	  end
	end
      else -- Match common operand size.
	local szp = sz
	if szm == "" then szm = x64 and "qdwb" or "dwb" end -- Default sizes.
	if prefix == "1" then szp = args[1].opsize; szmix = nil
	elseif prefix == "2" then szp = args[2].opsize; szmix = nil end
	if not szmix and (prefix == "." or match(szm, szp or "#")) then
	  dopattern(pat, args, szp, params.op, needrex) -- Process pattern.
	  return
	end
      end
      gotmatch = true
    end
  end

  local msg = "bad operand mode"
  if gotmatch then
    if szmix then
      msg = "mixed operand size"
    else
      msg = sz and "bad operand size" or "missing operand size"
    end
  end

  werror(msg.." in `"..opmodestr(params.op, args).."'")
end

------------------------------------------------------------------------------

-- x64-specific opcode for 64 bit immediates and displacements.
if x64 then
  function map_op.mov64_2(params)
    if not params then return { "reg, imm", "reg, [disp]", "[disp], reg" } end
    if secpos+2 > maxsecpos then wflush() end
    local opcode, op64, sz, rex, vreg
    local op64 = match(params[1], "^%[%s*(.-)%s*%]$")
    if op64 then
      local a = parseoperand(params[2])
      if a.mode ~= "rmR" then werror("bad operand mode") end
      sz = a.opsize
      rex = sz == "q" and 8 or 0
      opcode = 0xa3
    else
      op64 = match(params[2], "^%[%s*(.-)%s*%]$")
      local a = parseoperand(params[1])
      if op64 then
	if a.mode ~= "rmR" then werror("bad operand mode") end
	sz = a.opsize
	rex = sz == "q" and 8 or 0
	opcode = 0xa1
      else
	if sub(a.mode, 1, 1) ~= "r" or a.opsize ~= "q" then
	  werror("bad operand mode")
	end
	op64 = params[2]
	if a.reg == -1 then
	  vreg = a.vreg
	  opcode = 0xb8
	else
	  opcode = 0xb8 + band(a.reg, 7)
	end
	rex = a.reg > 7 and 9 or 8
      end
    end
    local psz, sk = wputop(sz, opcode, rex, nil, vreg)
    wvreg("opcode", vreg, psz, sk)
    waction("IMM_D", format("(unsigned int)(%s)", op64))
    waction("IMM_D", format("(unsigned int)((%s)>>32)", op64))
  end
end

------------------------------------------------------------------------------

-- Pseudo-opcodes for data storage.
local function op_data(params)
  if not params then return "imm..." end
  local sz = sub(params.op, 2, 2)
  if sz == "l" then sz = "d" elseif sz == "a" then sz = addrsize end
  for _,p in ipairs(params) do
    local a = parseoperand(p, sz == "q")
    if sub(a.mode, 1, 1) ~= "i" or (a.opsize and a.opsize ~= sz) then
      werror("bad mode or size in `"..p.."'")
    end
    if a.mode == "iJ" then
      wputlabel("IMM_", a.imm, 1)
    elseif sz == "q" then
      wputqarg(a.imm)
    else
      wputszarg(sz, a.imm)
    end
    if secpos+2 > maxsecpos then wflush() end
  end
end

map_op[".byte_*"] = op_data
map_op[".sbyte_*"] = op_data
map_op[".word_*"] = op_data
map_op[".dword_*"] = op_data
map_op[".qword_*"] = op_data
map_op[".aword_*"] = op_data
map_op[".long_*"] = op_data
map_op[".quad_*"] = op_data
map_op[".addr_*"] = op_data

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
map_op[".label_2"] = function(params)
  if not params then return "[1-9] | ->global | =>pcexpr  [, addr]" end
  if secpos+2 > maxsecpos then wflush() end
  local a = parseoperand(params[1])
  local mode, imm = a.mode, a.imm
  if type(imm) == "number" and (mode == "iJ" or (imm >= 1 and imm <= 9)) then
    -- Local label (1: ... 9:) or global label (->global:).
    waction("LABEL_LG", nil, 1)
    wputxb(imm)
  elseif mode == "iJ" then
    -- PC label (=>pcexpr:).
    waction("LABEL_PC", imm)
  else
    werror("bad label definition")
  end
  -- SETLABEL must immediately follow LABEL_LG/LABEL_PC.
  local addr = params[2]
  if addr then
    local a = parseoperand(addr)
    if a.mode == "iPJ" then
      waction("SETLABEL", a.imm)
    else
      werror("bad label assignment")
    end
  end
end
map_op[".label_1"] = map_op[".label_2"]

------------------------------------------------------------------------------

-- Alignment pseudo-opcode.
map_op[".align_1"] = function(params)
  if not params then return "numpow2" end
  if secpos+1 > maxsecpos then wflush() end
  local align = tonumber(params[1]) or map_opsizenum[map_opsize[params[1]]]
  if align then
    local x = align
    -- Must be a power of 2 in the range (2 ... 256).
    for i=1,8 do
      x = x / 2
      if x == 1 then
	waction("ALIGN", nil, 1)
	wputxb(align-1) -- Action byte is 2**n-1.
	return
      end
    end
  end
  werror("bad alignment")
end

-- Spacing pseudo-opcode.
map_op[".space_2"] = function(params)
  if not params then return "num [, filler]" end
  if secpos+1 > maxsecpos then wflush() end
  waction("SPACE", params[1])
  local fill = params[2]
  if fill then
    fill = tonumber(fill)
    if not fill or fill < 0 or fill > 255 then werror("bad filler") end
  end
  wputxb(fill or 0)
end
map_op[".space_1"] = map_op[".space_2"]

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
  if reg and not map_reg_valid_base[reg] then
    werror("bad base register `"..(map_reg_rev[reg] or reg).."'")
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
    local reg = tp.reg and map_reg_rev[tp.reg] or ""
    out:write(format("  %-20s %-20s %s\n", name, tp.ctype, reg))
  end
  out:write("\n")
end

------------------------------------------------------------------------------

-- Set the current section.
function _M.section(num)
  waction("SECTION")
  wputxb(num)
  wflush(true) -- SECTION is a terminal action.
end

------------------------------------------------------------------------------

-- Dump architecture description.
function _M.dumparch(out)
  out:write(format("DynASM %s version %s, released %s\n\n",
    _info.arch, _info.version, _info.release))
  dumpregs(out)
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

