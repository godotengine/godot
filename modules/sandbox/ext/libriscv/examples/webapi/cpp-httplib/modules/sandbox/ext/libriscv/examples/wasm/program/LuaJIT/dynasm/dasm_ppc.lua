------------------------------------------------------------------------------
-- DynASM PPC/PPC64 module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- See dynasm.lua for full copyright notice.
--
-- Support for various extensions contributed by Caio Souza Oliveira.
------------------------------------------------------------------------------

-- Module information:
local _info = {
  arch =	"ppc",
  description =	"DynASM PPC module",
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
local assert, setmetatable = assert, setmetatable
local _s = string
local sub, format, byte, char = _s.sub, _s.format, _s.byte, _s.char
local match, gmatch = _s.match, _s.gmatch
local concat, sort = table.concat, table.sort
local bit = bit or require("bit")
local band, shl, shr, sar = bit.band, bit.lshift, bit.rshift, bit.arshift
local tohex = bit.tohex

-- Inherited tables and callbacks.
local g_opt, g_arch
local wline, werror, wfatal, wwarn

-- Action name list.
-- CHECK: Keep this in sync with the C code!
local action_names = {
  "STOP", "SECTION", "ESC", "REL_EXT",
  "ALIGN", "REL_LG", "LABEL_LG",
  "REL_PC", "LABEL_PC", "IMM", "IMMSH"
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
  if n <= 0xffffff then waction("ESC") end
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
local map_archdef = { sp = "r1" } -- Ext. register name -> int. name.

local map_type = {}		-- Type name -> { ctype, reg }
local ctypenum = 0		-- Type number (for Dt... macros).

-- Reverse defines for registers.
function _M.revdef(s)
  if s == "r1" then return "sp" end
  return s
end

local map_cond = {
  lt = 0, gt = 1, eq = 2, so = 3,
  ge = 4, le = 5, ne = 6, ns = 7,
}

------------------------------------------------------------------------------

local map_op, op_template

local function op_alias(opname, f)
  return function(params, nparams)
    if not params then return "-> "..opname:sub(1, -3) end
    f(params, nparams)
    op_template(params, map_op[opname], nparams)
  end
end

-- Template strings for PPC instructions.
map_op = {
  tdi_3 =	"08000000ARI",
  twi_3 =	"0c000000ARI",
  mulli_3 =	"1c000000RRI",
  subfic_3 =	"20000000RRI",
  cmplwi_3 =	"28000000XRU",
  cmplwi_2 =	"28000000-RU",
  cmpldi_3 =	"28200000XRU",
  cmpldi_2 =	"28200000-RU",
  cmpwi_3 =	"2c000000XRI",
  cmpwi_2 =	"2c000000-RI",
  cmpdi_3 =	"2c200000XRI",
  cmpdi_2 =	"2c200000-RI",
  addic_3 =	"30000000RRI",
  ["addic._3"] = "34000000RRI",
  addi_3 =	"38000000RR0I",
  li_2 =	"38000000RI",
  la_2 =	"38000000RD",
  addis_3 =	"3c000000RR0I",
  lis_2 =	"3c000000RI",
  lus_2 =	"3c000000RU",
  bc_3 =	"40000000AAK",
  bcl_3 =	"40000001AAK",
  bdnz_1 =	"42000000K",
  bdz_1 =	"42400000K",
  sc_0 =	"44000000",
  b_1 =		"48000000J",
  bl_1 =	"48000001J",
  rlwimi_5 =	"50000000RR~AAA.",
  rlwinm_5 =	"54000000RR~AAA.",
  rlwnm_5 =	"5c000000RR~RAA.",
  ori_3 =	"60000000RR~U",
  nop_0 =	"60000000",
  oris_3 =	"64000000RR~U",
  xori_3 =	"68000000RR~U",
  xoris_3 =	"6c000000RR~U",
  ["andi._3"] =	"70000000RR~U",
  ["andis._3"] = "74000000RR~U",
  lwz_2 =	"80000000RD",
  lwzu_2 =	"84000000RD",
  lbz_2 =	"88000000RD",
  lbzu_2 =	"8c000000RD",
  stw_2 =	"90000000RD",
  stwu_2 =	"94000000RD",
  stb_2 =	"98000000RD",
  stbu_2 =	"9c000000RD",
  lhz_2 =	"a0000000RD",
  lhzu_2 =	"a4000000RD",
  lha_2 =	"a8000000RD",
  lhau_2 =	"ac000000RD",
  sth_2 =	"b0000000RD",
  sthu_2 =	"b4000000RD",
  lmw_2 =	"b8000000RD",
  stmw_2 =	"bc000000RD",
  lfs_2 =	"c0000000FD",
  lfsu_2 =	"c4000000FD",
  lfd_2 =	"c8000000FD",
  lfdu_2 =	"cc000000FD",
  stfs_2 =	"d0000000FD",
  stfsu_2 =	"d4000000FD",
  stfd_2 =	"d8000000FD",
  stfdu_2 =	"dc000000FD",
  ld_2 =	"e8000000RD", -- NYI: displacement must be divisible by 4.
  ldu_2 =	"e8000001RD",
  lwa_2 =	"e8000002RD",
  std_2 =	"f8000000RD",
  stdu_2 =	"f8000001RD",

  subi_3 =	op_alias("addi_3", function(p) p[3] = "-("..p[3]..")" end),
  subis_3 =	op_alias("addis_3", function(p) p[3] = "-("..p[3]..")" end),
  subic_3 =	op_alias("addic_3", function(p) p[3] = "-("..p[3]..")" end),
  ["subic._3"] = op_alias("addic._3", function(p) p[3] = "-("..p[3]..")" end),

  rotlwi_3 =	op_alias("rlwinm_5", function(p)
    p[4] = "0"; p[5] = "31"
  end),
  rotrwi_3 =	op_alias("rlwinm_5", function(p)
    p[3] = "32-("..p[3]..")"; p[4] = "0"; p[5] = "31"
  end),
  rotlw_3 =	op_alias("rlwnm_5", function(p)
    p[4] = "0"; p[5] = "31"
  end),
  slwi_3 =	op_alias("rlwinm_5", function(p)
    p[5] = "31-("..p[3]..")"; p[4] = "0"
  end),
  srwi_3 =	op_alias("rlwinm_5", function(p)
    p[4] = p[3]; p[3] = "32-("..p[3]..")"; p[5] = "31"
  end),
  clrlwi_3 =	op_alias("rlwinm_5", function(p)
    p[4] = p[3]; p[3] = "0"; p[5] = "31"
  end),
  clrrwi_3 =	op_alias("rlwinm_5", function(p)
    p[5] = "31-("..p[3]..")"; p[3] = "0"; p[4] = "0"
  end),

  -- Primary opcode 4:
  mulhhwu_3 =		"10000010RRR.",
  machhwu_3 =		"10000018RRR.",
  mulhhw_3 =		"10000050RRR.",
  nmachhw_3 =		"1000005cRRR.",
  machhwsu_3 =		"10000098RRR.",
  machhws_3 =		"100000d8RRR.",
  nmachhws_3 =		"100000dcRRR.",
  mulchwu_3 =		"10000110RRR.",
  macchwu_3 =		"10000118RRR.",
  mulchw_3 =		"10000150RRR.",
  macchw_3 =		"10000158RRR.",
  nmacchw_3 =		"1000015cRRR.",
  macchwsu_3 =		"10000198RRR.",
  macchws_3 =		"100001d8RRR.",
  nmacchws_3 =		"100001dcRRR.",
  mullhw_3 =		"10000350RRR.",
  maclhw_3 =		"10000358RRR.",
  nmaclhw_3 =		"1000035cRRR.",
  maclhwsu_3 =		"10000398RRR.",
  maclhws_3 =		"100003d8RRR.",
  nmaclhws_3 =		"100003dcRRR.",
  machhwuo_3 =		"10000418RRR.",
  nmachhwo_3 =		"1000045cRRR.",
  machhwsuo_3 =		"10000498RRR.",
  machhwso_3 =		"100004d8RRR.",
  nmachhwso_3 =		"100004dcRRR.",
  macchwuo_3 =		"10000518RRR.",
  macchwo_3 =		"10000558RRR.",
  nmacchwo_3 =		"1000055cRRR.",
  macchwsuo_3 =		"10000598RRR.",
  macchwso_3 =		"100005d8RRR.",
  nmacchwso_3 =		"100005dcRRR.",
  maclhwo_3 =		"10000758RRR.",
  nmaclhwo_3 =		"1000075cRRR.",
  maclhwsuo_3 =		"10000798RRR.",
  maclhwso_3 =		"100007d8RRR.",
  nmaclhwso_3 =		"100007dcRRR.",

  vaddubm_3 =		"10000000VVV",
  vmaxub_3 =		"10000002VVV",
  vrlb_3 =		"10000004VVV",
  vcmpequb_3 =		"10000006VVV",
  vmuloub_3 =		"10000008VVV",
  vaddfp_3 =		"1000000aVVV",
  vmrghb_3 =		"1000000cVVV",
  vpkuhum_3 =		"1000000eVVV",
  vmhaddshs_4 =		"10000020VVVV",
  vmhraddshs_4 =	"10000021VVVV",
  vmladduhm_4 =		"10000022VVVV",
  vmsumubm_4 =		"10000024VVVV",
  vmsummbm_4 =		"10000025VVVV",
  vmsumuhm_4 =		"10000026VVVV",
  vmsumuhs_4 =		"10000027VVVV",
  vmsumshm_4 =		"10000028VVVV",
  vmsumshs_4 =		"10000029VVVV",
  vsel_4 =		"1000002aVVVV",
  vperm_4 =		"1000002bVVVV",
  vsldoi_4 =		"1000002cVVVP",
  vpermxor_4 =		"1000002dVVVV",
  vmaddfp_4 =		"1000002eVVVV~",
  vnmsubfp_4 =		"1000002fVVVV~",
  vaddeuqm_4 =		"1000003cVVVV",
  vaddecuq_4 =		"1000003dVVVV",
  vsubeuqm_4 =		"1000003eVVVV",
  vsubecuq_4 =		"1000003fVVVV",
  vadduhm_3 =		"10000040VVV",
  vmaxuh_3 =		"10000042VVV",
  vrlh_3 =		"10000044VVV",
  vcmpequh_3 =		"10000046VVV",
  vmulouh_3 =		"10000048VVV",
  vsubfp_3 =		"1000004aVVV",
  vmrghh_3 =		"1000004cVVV",
  vpkuwum_3 =		"1000004eVVV",
  vadduwm_3 =		"10000080VVV",
  vmaxuw_3 =		"10000082VVV",
  vrlw_3 =		"10000084VVV",
  vcmpequw_3 =		"10000086VVV",
  vmulouw_3 =		"10000088VVV",
  vmuluwm_3 =		"10000089VVV",
  vmrghw_3 =		"1000008cVVV",
  vpkuhus_3 =		"1000008eVVV",
  vaddudm_3 =		"100000c0VVV",
  vmaxud_3 =		"100000c2VVV",
  vrld_3 =		"100000c4VVV",
  vcmpeqfp_3 =		"100000c6VVV",
  vcmpequd_3 =		"100000c7VVV",
  vpkuwus_3 =		"100000ceVVV",
  vadduqm_3 =		"10000100VVV",
  vmaxsb_3 =		"10000102VVV",
  vslb_3 =		"10000104VVV",
  vmulosb_3 =		"10000108VVV",
  vrefp_2 =		"1000010aV-V",
  vmrglb_3 =		"1000010cVVV",
  vpkshus_3 =		"1000010eVVV",
  vaddcuq_3 =		"10000140VVV",
  vmaxsh_3 =		"10000142VVV",
  vslh_3 =		"10000144VVV",
  vmulosh_3 =		"10000148VVV",
  vrsqrtefp_2 =		"1000014aV-V",
  vmrglh_3 =		"1000014cVVV",
  vpkswus_3 =		"1000014eVVV",
  vaddcuw_3 =		"10000180VVV",
  vmaxsw_3 =		"10000182VVV",
  vslw_3 =		"10000184VVV",
  vmulosw_3 =		"10000188VVV",
  vexptefp_2 =		"1000018aV-V",
  vmrglw_3 =		"1000018cVVV",
  vpkshss_3 =		"1000018eVVV",
  vmaxsd_3 =		"100001c2VVV",
  vsl_3 =		"100001c4VVV",
  vcmpgefp_3 =		"100001c6VVV",
  vlogefp_2 =		"100001caV-V",
  vpkswss_3 =		"100001ceVVV",
  vadduhs_3 =		"10000240VVV",
  vminuh_3 =		"10000242VVV",
  vsrh_3 =		"10000244VVV",
  vcmpgtuh_3 =		"10000246VVV",
  vmuleuh_3 =		"10000248VVV",
  vrfiz_2 =		"1000024aV-V",
  vsplth_3 =		"1000024cVV3",
  vupkhsh_2 =		"1000024eV-V",
  vminuw_3 =		"10000282VVV",
  vminud_3 =		"100002c2VVV",
  vcmpgtud_3 =		"100002c7VVV",
  vrfim_2 =		"100002caV-V",
  vcmpgtsb_3 =		"10000306VVV",
  vcfux_3 =		"1000030aVVA~",
  vaddshs_3 =		"10000340VVV",
  vminsh_3 =		"10000342VVV",
  vsrah_3 =		"10000344VVV",
  vcmpgtsh_3 =		"10000346VVV",
  vmulesh_3 =		"10000348VVV",
  vcfsx_3 =		"1000034aVVA~",
  vspltish_2 =		"1000034cVS",
  vupkhpx_2 =		"1000034eV-V",
  vaddsws_3 =		"10000380VVV",
  vminsw_3 =		"10000382VVV",
  vsraw_3 =		"10000384VVV",
  vcmpgtsw_3 =		"10000386VVV",
  vmulesw_3 =		"10000388VVV",
  vctuxs_3 =		"1000038aVVA~",
  vspltisw_2 =		"1000038cVS",
  vminsd_3 =		"100003c2VVV",
  vsrad_3 =		"100003c4VVV",
  vcmpbfp_3 =		"100003c6VVV",
  vcmpgtsd_3 =		"100003c7VVV",
  vctsxs_3 =		"100003caVVA~",
  vupklpx_2 =		"100003ceV-V",
  vsububm_3 =		"10000400VVV",
  ["bcdadd._4"] =	"10000401VVVy.",
  vavgub_3 =		"10000402VVV",
  vand_3 =		"10000404VVV",
  ["vcmpequb._3"] =	"10000406VVV",
  vmaxfp_3 =		"1000040aVVV",
  vsubuhm_3 =		"10000440VVV",
  ["bcdsub._4"] =	"10000441VVVy.",
  vavguh_3 =		"10000442VVV",
  vandc_3 =		"10000444VVV",
  ["vcmpequh._3"] =	"10000446VVV",
  vminfp_3 =		"1000044aVVV",
  vpkudum_3 =		"1000044eVVV",
  vsubuwm_3 =		"10000480VVV",
  vavguw_3 =		"10000482VVV",
  vor_3 =		"10000484VVV",
  ["vcmpequw._3"] =	"10000486VVV",
  vpmsumw_3 =		"10000488VVV",
  ["vcmpeqfp._3"] =	"100004c6VVV",
  ["vcmpequd._3"] =	"100004c7VVV",
  vpkudus_3 =		"100004ceVVV",
  vavgsb_3 =		"10000502VVV",
  vavgsh_3 =		"10000542VVV",
  vorc_3 =		"10000544VVV",
  vbpermq_3 =		"1000054cVVV",
  vpksdus_3 =		"1000054eVVV",
  vavgsw_3 =		"10000582VVV",
  vsld_3 =		"100005c4VVV",
  ["vcmpgefp._3"] =	"100005c6VVV",
  vpksdss_3 =		"100005ceVVV",
  vsububs_3 =		"10000600VVV",
  mfvscr_1 =		"10000604V--",
  vsum4ubs_3 =		"10000608VVV",
  vsubuhs_3 =		"10000640VVV",
  mtvscr_1 =		"10000644--V",
  ["vcmpgtuh._3"] =	"10000646VVV",
  vsum4shs_3 =		"10000648VVV",
  vupkhsw_2 =		"1000064eV-V",
  vsubuws_3 =		"10000680VVV",
  vshasigmaw_4 =	"10000682VVYp",
  veqv_3 =		"10000684VVV",
  vsum2sws_3 =		"10000688VVV",
  vmrgow_3 =		"1000068cVVV",
  vshasigmad_4 =	"100006c2VVYp",
  vsrd_3 =		"100006c4VVV",
  ["vcmpgtud._3"] =	"100006c7VVV",
  vupklsw_2 =		"100006ceV-V",
  vupkslw_2 =		"100006ceV-V",
  vsubsbs_3 =		"10000700VVV",
  vclzb_2 =		"10000702V-V",
  vpopcntb_2 =		"10000703V-V",
  ["vcmpgtsb._3"] =	"10000706VVV",
  vsum4sbs_3 =		"10000708VVV",
  vsubshs_3 =		"10000740VVV",
  vclzh_2 =		"10000742V-V",
  vpopcnth_2 =		"10000743V-V",
  ["vcmpgtsh._3"] =	"10000746VVV",
  vsubsws_3 =		"10000780VVV",
  vclzw_2 =		"10000782V-V",
  vpopcntw_2 =		"10000783V-V",
  ["vcmpgtsw._3"] =	"10000786VVV",
  vsumsws_3 =		"10000788VVV",
  vmrgew_3 =		"1000078cVVV",
  vclzd_2 =		"100007c2V-V",
  vpopcntd_2 =		"100007c3V-V",
  ["vcmpbfp._3"] =	"100007c6VVV",
  ["vcmpgtsd._3"] =	"100007c7VVV",

  -- Primary opcode 19:
  mcrf_2 =	"4c000000XX",
  isync_0 =	"4c00012c",
  crnor_3 =	"4c000042CCC",
  crnot_2 =	"4c000042CC=",
  crandc_3 =	"4c000102CCC",
  crxor_3 =	"4c000182CCC",
  crclr_1 =	"4c000182C==",
  crnand_3 =	"4c0001c2CCC",
  crand_3 =	"4c000202CCC",
  creqv_3 =	"4c000242CCC",
  crset_1 =	"4c000242C==",
  crorc_3 =	"4c000342CCC",
  cror_3 =	"4c000382CCC",
  crmove_2 =	"4c000382CC=",
  bclr_2 =	"4c000020AA",
  bclrl_2 =	"4c000021AA",
  bcctr_2 =	"4c000420AA",
  bcctrl_2 =	"4c000421AA",
  bctar_2 =	"4c000460AA",
  bctarl_2 =	"4c000461AA",
  blr_0 =	"4e800020",
  blrl_0 =	"4e800021",
  bctr_0 =	"4e800420",
  bctrl_0 =	"4e800421",

  -- Primary opcode 31:
  cmpw_3 =	"7c000000XRR",
  cmpw_2 =	"7c000000-RR",
  cmpd_3 =	"7c200000XRR",
  cmpd_2 =	"7c200000-RR",
  tw_3 =	"7c000008ARR",
  lvsl_3 =	"7c00000cVRR",
  subfc_3 =	"7c000010RRR.",
  subc_3 =	"7c000010RRR~.",
  mulhdu_3 =	"7c000012RRR.",
  addc_3 =	"7c000014RRR.",
  mulhwu_3 =	"7c000016RRR.",
  isel_4 =	"7c00001eRRRC",
  isellt_3 =	"7c00001eRRR",
  iselgt_3 =	"7c00005eRRR",
  iseleq_3 =	"7c00009eRRR",
  mfcr_1 =	"7c000026R",
  mfocrf_2 =	"7c100026RG",
  mtcrf_2 =	"7c000120GR",
  mtocrf_2 =	"7c100120GR",
  lwarx_3 =	"7c000028RR0R",
  ldx_3 =	"7c00002aRR0R",
  lwzx_3 =	"7c00002eRR0R",
  slw_3 =	"7c000030RR~R.",
  cntlzw_2 =	"7c000034RR~",
  sld_3 =	"7c000036RR~R.",
  and_3 =	"7c000038RR~R.",
  cmplw_3 =	"7c000040XRR",
  cmplw_2 =	"7c000040-RR",
  cmpld_3 =	"7c200040XRR",
  cmpld_2 =	"7c200040-RR",
  lvsr_3 =	"7c00004cVRR",
  subf_3 =	"7c000050RRR.",
  sub_3 =	"7c000050RRR~.",
  lbarx_3 =	"7c000068RR0R",
  ldux_3 =	"7c00006aRR0R",
  dcbst_2 =	"7c00006c-RR",
  lwzux_3 =	"7c00006eRR0R",
  cntlzd_2 =	"7c000074RR~",
  andc_3 =	"7c000078RR~R.",
  td_3 =	"7c000088ARR",
  lvewx_3 =	"7c00008eVRR",
  mulhd_3 =	"7c000092RRR.",
  addg6s_3 =	"7c000094RRR",
  mulhw_3 =	"7c000096RRR.",
  dlmzb_3 =	"7c00009cRR~R.",
  ldarx_3 =	"7c0000a8RR0R",
  dcbf_2 =	"7c0000ac-RR",
  lbzx_3 =	"7c0000aeRR0R",
  lvx_3 =	"7c0000ceVRR",
  neg_2 =	"7c0000d0RR.",
  lharx_3 =	"7c0000e8RR0R",
  lbzux_3 =	"7c0000eeRR0R",
  popcntb_2 =	"7c0000f4RR~",
  not_2 =	"7c0000f8RR~%.",
  nor_3 =	"7c0000f8RR~R.",
  stvebx_3 =	"7c00010eVRR",
  subfe_3 =	"7c000110RRR.",
  sube_3 =	"7c000110RRR~.",
  adde_3 =	"7c000114RRR.",
  stdx_3 =	"7c00012aRR0R",
  ["stwcx._3"] =	"7c00012dRR0R.",
  stwx_3 =	"7c00012eRR0R",
  prtyw_2 =	"7c000134RR~",
  stvehx_3 =	"7c00014eVRR",
  stdux_3 =	"7c00016aRR0R",
  ["stqcx._3"] =	"7c00016dR:R0R.",
  stwux_3 =	"7c00016eRR0R",
  prtyd_2 =	"7c000174RR~",
  stvewx_3 =	"7c00018eVRR",
  subfze_2 =	"7c000190RR.",
  addze_2 =	"7c000194RR.",
  ["stdcx._3"] =	"7c0001adRR0R.",
  stbx_3 =	"7c0001aeRR0R",
  stvx_3 =	"7c0001ceVRR",
  subfme_2 =	"7c0001d0RR.",
  mulld_3 =	"7c0001d2RRR.",
  addme_2 =	"7c0001d4RR.",
  mullw_3 =	"7c0001d6RRR.",
  dcbtst_2 =	"7c0001ec-RR",
  stbux_3 =	"7c0001eeRR0R",
  bpermd_3 =	"7c0001f8RR~R",
  lvepxl_3 =	"7c00020eVRR",
  add_3 =	"7c000214RRR.",
  lqarx_3 =	"7c000228R:R0R",
  dcbt_2 =	"7c00022c-RR",
  lhzx_3 =	"7c00022eRR0R",
  cdtbcd_2 =	"7c000234RR~",
  eqv_3 =	"7c000238RR~R.",
  lvepx_3 =	"7c00024eVRR",
  eciwx_3 =	"7c00026cRR0R",
  lhzux_3 =	"7c00026eRR0R",
  cbcdtd_2 =	"7c000274RR~",
  xor_3 =	"7c000278RR~R.",
  mfspefscr_1 =	"7c0082a6R",
  mfxer_1 =	"7c0102a6R",
  mflr_1 =	"7c0802a6R",
  mfctr_1 =	"7c0902a6R",
  lwax_3 =	"7c0002aaRR0R",
  lhax_3 =	"7c0002aeRR0R",
  mftb_1 =	"7c0c42e6R",
  mftbu_1 =	"7c0d42e6R",
  lvxl_3 =	"7c0002ceVRR",
  lwaux_3 =	"7c0002eaRR0R",
  lhaux_3 =	"7c0002eeRR0R",
  popcntw_2 =	"7c0002f4RR~",
  divdeu_3 =	"7c000312RRR.",
  divweu_3 =	"7c000316RRR.",
  sthx_3 =	"7c00032eRR0R",
  orc_3 =	"7c000338RR~R.",
  ecowx_3 =	"7c00036cRR0R",
  sthux_3 =	"7c00036eRR0R",
  or_3 =	"7c000378RR~R.",
  mr_2 =	"7c000378RR~%.",
  divdu_3 =	"7c000392RRR.",
  divwu_3 =	"7c000396RRR.",
  mtspefscr_1 =	"7c0083a6R",
  mtxer_1 =	"7c0103a6R",
  mtlr_1 =	"7c0803a6R",
  mtctr_1 =	"7c0903a6R",
  dcbi_2 =	"7c0003ac-RR",
  nand_3 =	"7c0003b8RR~R.",
  dsn_2 =	"7c0003c6-RR",
  stvxl_3 =	"7c0003ceVRR",
  divd_3 =	"7c0003d2RRR.",
  divw_3 =	"7c0003d6RRR.",
  popcntd_2 =	"7c0003f4RR~",
  cmpb_3 =	"7c0003f8RR~R.",
  mcrxr_1 =	"7c000400X",
  lbdx_3 =	"7c000406RRR",
  subfco_3 =	"7c000410RRR.",
  subco_3 =	"7c000410RRR~.",
  addco_3 =	"7c000414RRR.",
  ldbrx_3 =	"7c000428RR0R",
  lswx_3 =	"7c00042aRR0R",
  lwbrx_3 =	"7c00042cRR0R",
  lfsx_3 =	"7c00042eFR0R",
  srw_3 =	"7c000430RR~R.",
  srd_3 =	"7c000436RR~R.",
  lhdx_3 =	"7c000446RRR",
  subfo_3 =	"7c000450RRR.",
  subo_3 =	"7c000450RRR~.",
  lfsux_3 =	"7c00046eFR0R",
  lwdx_3 =	"7c000486RRR",
  lswi_3 =	"7c0004aaRR0A",
  sync_0 =	"7c0004ac",
  lwsync_0 =	"7c2004ac",
  ptesync_0 =	"7c4004ac",
  lfdx_3 =	"7c0004aeFR0R",
  lddx_3 =	"7c0004c6RRR",
  nego_2 =	"7c0004d0RR.",
  lfdux_3 =	"7c0004eeFR0R",
  stbdx_3 =	"7c000506RRR",
  subfeo_3 =	"7c000510RRR.",
  subeo_3 =	"7c000510RRR~.",
  addeo_3 =	"7c000514RRR.",
  stdbrx_3 =	"7c000528RR0R",
  stswx_3 =	"7c00052aRR0R",
  stwbrx_3 =	"7c00052cRR0R",
  stfsx_3 =	"7c00052eFR0R",
  sthdx_3 =	"7c000546RRR",
  ["stbcx._3"] =	"7c00056dRRR",
  stfsux_3 =	"7c00056eFR0R",
  stwdx_3 =	"7c000586RRR",
  subfzeo_2 =	"7c000590RR.",
  addzeo_2 =	"7c000594RR.",
  stswi_3 =	"7c0005aaRR0A",
  ["sthcx._3"] =	"7c0005adRRR",
  stfdx_3 =	"7c0005aeFR0R",
  stddx_3 =	"7c0005c6RRR",
  subfmeo_2 =	"7c0005d0RR.",
  mulldo_3 =	"7c0005d2RRR.",
  addmeo_2 =	"7c0005d4RR.",
  mullwo_3 =	"7c0005d6RRR.",
  dcba_2 =	"7c0005ec-RR",
  stfdux_3 =	"7c0005eeFR0R",
  stvepxl_3 =	"7c00060eVRR",
  addo_3 =	"7c000614RRR.",
  lhbrx_3 =	"7c00062cRR0R",
  lfdpx_3 =	"7c00062eF:RR",
  sraw_3 =	"7c000630RR~R.",
  srad_3 =	"7c000634RR~R.",
  lfddx_3 =	"7c000646FRR",
  stvepx_3 =	"7c00064eVRR",
  srawi_3 =	"7c000670RR~A.",
  sradi_3 =	"7c000674RR~H.",
  eieio_0 =	"7c0006ac",
  lfiwax_3 =	"7c0006aeFR0R",
  divdeuo_3 =	"7c000712RRR.",
  divweuo_3 =	"7c000716RRR.",
  sthbrx_3 =	"7c00072cRR0R",
  stfdpx_3 =	"7c00072eF:RR",
  extsh_2 =	"7c000734RR~.",
  stfddx_3 =	"7c000746FRR",
  divdeo_3 =	"7c000752RRR.",
  divweo_3 =	"7c000756RRR.",
  extsb_2 =	"7c000774RR~.",
  divduo_3 =	"7c000792RRR.",
  divwou_3 =	"7c000796RRR.",
  icbi_2 =	"7c0007ac-RR",
  stfiwx_3 =	"7c0007aeFR0R",
  extsw_2 =	"7c0007b4RR~.",
  divdo_3 =	"7c0007d2RRR.",
  divwo_3 =	"7c0007d6RRR.",
  dcbz_2 =	"7c0007ec-RR",

  ["tbegin._1"] =	"7c00051d1",
  ["tbegin._0"] =	"7c00051d",
  ["tend._1"] =		"7c00055dY",
  ["tend._0"] =		"7c00055d",
  ["tendall._0"] =	"7e00055d",
  tcheck_1 =		"7c00059cX",
  ["tsr._1"] =		"7c0005dd1",
  ["tsuspend._0"] =	"7c0005dd",
  ["tresume._0"] =	"7c2005dd",
  ["tabortwc._3"] =	"7c00061dARR",
  ["tabortdc._3"] =	"7c00065dARR",
  ["tabortwci._3"] =	"7c00069dARS",
  ["tabortdci._3"] =	"7c0006ddARS",
  ["tabort._1"] =	"7c00071d-R-",
  ["treclaim._1"] =	"7c00075d-R",
  ["trechkpt._0"] =	"7c0007dd",

  lxsiwzx_3 =	"7c000018QRR",
  lxsiwax_3 =	"7c000098QRR",
  mfvsrd_2 =	"7c000066-Rq",
  mfvsrwz_2 =	"7c0000e6-Rq",
  stxsiwx_3 =	"7c000118QRR",
  mtvsrd_2 =	"7c000166QR",
  mtvsrwa_2 =	"7c0001a6QR",
  lxvdsx_3 =	"7c000298QRR",
  lxsspx_3 =	"7c000418QRR",
  lxsdx_3 =	"7c000498QRR",
  stxsspx_3 =	"7c000518QRR",
  stxsdx_3 =	"7c000598QRR",
  lxvw4x_3 =	"7c000618QRR",
  lxvd2x_3 =	"7c000698QRR",
  stxvw4x_3 =	"7c000718QRR",
  stxvd2x_3 =	"7c000798QRR",

  -- Primary opcode 30:
  rldicl_4 =	"78000000RR~HM.",
  rldicr_4 =	"78000004RR~HM.",
  rldic_4 =	"78000008RR~HM.",
  rldimi_4 =	"7800000cRR~HM.",
  rldcl_4 =	"78000010RR~RM.",
  rldcr_4 =	"78000012RR~RM.",

  rotldi_3 =	op_alias("rldicl_4", function(p)
    p[4] = "0"
  end),
  rotrdi_3 =	op_alias("rldicl_4", function(p)
    p[3] = "64-("..p[3]..")"; p[4] = "0"
  end),
  rotld_3 =	op_alias("rldcl_4", function(p)
    p[4] = "0"
  end),
  sldi_3 =	op_alias("rldicr_4", function(p)
    p[4] = "63-("..p[3]..")"
  end),
  srdi_3 =	op_alias("rldicl_4", function(p)
    p[4] = p[3]; p[3] = "64-("..p[3]..")"
  end),
  clrldi_3 =	op_alias("rldicl_4", function(p)
    p[4] = p[3]; p[3] = "0"
  end),
  clrrdi_3 =	op_alias("rldicr_4", function(p)
    p[4] = "63-("..p[3]..")"; p[3] = "0"
  end),

  -- Primary opcode 56:
  lq_2 =	"e0000000R:D", -- NYI: displacement must be divisible by 8.

  -- Primary opcode 57:
  lfdp_2 =	"e4000000F:D", -- NYI: displacement must be divisible by 4.

  -- Primary opcode 59:
  fdivs_3 =	"ec000024FFF.",
  fsubs_3 =	"ec000028FFF.",
  fadds_3 =	"ec00002aFFF.",
  fsqrts_2 =	"ec00002cF-F.",
  fres_2 =	"ec000030F-F.",
  fmuls_3 =	"ec000032FF-F.",
  frsqrtes_2 =	"ec000034F-F.",
  fmsubs_4 =	"ec000038FFFF~.",
  fmadds_4 =	"ec00003aFFFF~.",
  fnmsubs_4 =	"ec00003cFFFF~.",
  fnmadds_4 =	"ec00003eFFFF~.",
  fcfids_2 =	"ec00069cF-F.",
  fcfidus_2 =	"ec00079cF-F.",

  dadd_3 =	"ec000004FFF.",
  dqua_4 =	"ec000006FFFZ.",
  dmul_3 =	"ec000044FFF.",
  drrnd_4 =	"ec000046FFFZ.",
  dscli_3 =	"ec000084FF6.",
  dquai_4 =	"ec000086SF~FZ.",
  dscri_3 =	"ec0000c4FF6.",
  drintx_4 =	"ec0000c61F~FZ.",
  dcmpo_3 =	"ec000104XFF",
  dtstex_3 =	"ec000144XFF",
  dtstdc_3 =	"ec000184XF6",
  dtstdg_3 =	"ec0001c4XF6",
  drintn_4 =	"ec0001c61F~FZ.",
  dctdp_2 =	"ec000204F-F.",
  dctfix_2 =	"ec000244F-F.",
  ddedpd_3 =	"ec000284ZF~F.",
  dxex_2 =	"ec0002c4F-F.",
  dsub_3 =	"ec000404FFF.",
  ddiv_3 =	"ec000444FFF.",
  dcmpu_3 =	"ec000504XFF",
  dtstsf_3 =	"ec000544XFF",
  drsp_2 =	"ec000604F-F.",
  dcffix_2 =	"ec000644F-F.",
  denbcd_3 =	"ec000684YF~F.",
  diex_3 =	"ec0006c4FFF.",

  -- Primary opcode 60:
  xsaddsp_3 =		"f0000000QQQ",
  xsmaddasp_3 =		"f0000008QQQ",
  xxsldwi_4 =		"f0000010QQQz",
  xsrsqrtesp_2 =	"f0000028Q-Q",
  xssqrtsp_2 =		"f000002cQ-Q",
  xxsel_4 =		"f0000030QQQQ",
  xssubsp_3 =		"f0000040QQQ",
  xsmaddmsp_3 =		"f0000048QQQ",
  xxpermdi_4 =		"f0000050QQQz",
  xsresp_2 =		"f0000068Q-Q",
  xsmulsp_3 =		"f0000080QQQ",
  xsmsubasp_3 =		"f0000088QQQ",
  xxmrghw_3 =		"f0000090QQQ",
  xsdivsp_3 =		"f00000c0QQQ",
  xsmsubmsp_3 =		"f00000c8QQQ",
  xsadddp_3 =		"f0000100QQQ",
  xsmaddadp_3 =		"f0000108QQQ",
  xscmpudp_3 =		"f0000118XQQ",
  xscvdpuxws_2 =	"f0000120Q-Q",
  xsrdpi_2 =		"f0000124Q-Q",
  xsrsqrtedp_2 =	"f0000128Q-Q",
  xssqrtdp_2 =		"f000012cQ-Q",
  xssubdp_3 =		"f0000140QQQ",
  xsmaddmdp_3 =		"f0000148QQQ",
  xscmpodp_3 =		"f0000158XQQ",
  xscvdpsxws_2 =	"f0000160Q-Q",
  xsrdpiz_2 =		"f0000164Q-Q",
  xsredp_2 =		"f0000168Q-Q",
  xsmuldp_3 =		"f0000180QQQ",
  xsmsubadp_3 =		"f0000188QQQ",
  xxmrglw_3 =		"f0000190QQQ",
  xsrdpip_2 =		"f00001a4Q-Q",
  xstsqrtdp_2 =		"f00001a8X-Q",
  xsrdpic_2 =		"f00001acQ-Q",
  xsdivdp_3 =		"f00001c0QQQ",
  xsmsubmdp_3 =		"f00001c8QQQ",
  xsrdpim_2 =		"f00001e4Q-Q",
  xstdivdp_3 =		"f00001e8XQQ",
  xvaddsp_3 =		"f0000200QQQ",
  xvmaddasp_3 =		"f0000208QQQ",
  xvcmpeqsp_3 =		"f0000218QQQ",
  xvcvspuxws_2 =	"f0000220Q-Q",
  xvrspi_2 =		"f0000224Q-Q",
  xvrsqrtesp_2 =	"f0000228Q-Q",
  xvsqrtsp_2 =		"f000022cQ-Q",
  xvsubsp_3 =		"f0000240QQQ",
  xvmaddmsp_3 =		"f0000248QQQ",
  xvcmpgtsp_3 =		"f0000258QQQ",
  xvcvspsxws_2 =	"f0000260Q-Q",
  xvrspiz_2 =		"f0000264Q-Q",
  xvresp_2 =		"f0000268Q-Q",
  xvmulsp_3 =		"f0000280QQQ",
  xvmsubasp_3 =		"f0000288QQQ",
  xxspltw_3 =		"f0000290QQg~",
  xvcmpgesp_3 =		"f0000298QQQ",
  xvcvuxwsp_2 =		"f00002a0Q-Q",
  xvrspip_2 =		"f00002a4Q-Q",
  xvtsqrtsp_2 =		"f00002a8X-Q",
  xvrspic_2 =		"f00002acQ-Q",
  xvdivsp_3 =		"f00002c0QQQ",
  xvmsubmsp_3 =		"f00002c8QQQ",
  xvcvsxwsp_2 =		"f00002e0Q-Q",
  xvrspim_2 =		"f00002e4Q-Q",
  xvtdivsp_3 =		"f00002e8XQQ",
  xvadddp_3 =		"f0000300QQQ",
  xvmaddadp_3 =		"f0000308QQQ",
  xvcmpeqdp_3 =		"f0000318QQQ",
  xvcvdpuxws_2 =	"f0000320Q-Q",
  xvrdpi_2 =		"f0000324Q-Q",
  xvrsqrtedp_2 =	"f0000328Q-Q",
  xvsqrtdp_2 =		"f000032cQ-Q",
  xvsubdp_3 =		"f0000340QQQ",
  xvmaddmdp_3 =		"f0000348QQQ",
  xvcmpgtdp_3 =		"f0000358QQQ",
  xvcvdpsxws_2 =	"f0000360Q-Q",
  xvrdpiz_2 =		"f0000364Q-Q",
  xvredp_2 =		"f0000368Q-Q",
  xvmuldp_3 =		"f0000380QQQ",
  xvmsubadp_3 =		"f0000388QQQ",
  xvcmpgedp_3 =		"f0000398QQQ",
  xvcvuxwdp_2 =		"f00003a0Q-Q",
  xvrdpip_2 =		"f00003a4Q-Q",
  xvtsqrtdp_2 =		"f00003a8X-Q",
  xvrdpic_2 =		"f00003acQ-Q",
  xvdivdp_3 =		"f00003c0QQQ",
  xvmsubmdp_3 =		"f00003c8QQQ",
  xvcvsxwdp_2 =		"f00003e0Q-Q",
  xvrdpim_2 =		"f00003e4Q-Q",
  xvtdivdp_3 =		"f00003e8XQQ",
  xsnmaddasp_3 =	"f0000408QQQ",
  xxland_3 =		"f0000410QQQ",
  xscvdpsp_2 =		"f0000424Q-Q",
  xscvdpspn_2 =		"f000042cQ-Q",
  xsnmaddmsp_3 =	"f0000448QQQ",
  xxlandc_3 =		"f0000450QQQ",
  xsrsp_2 =		"f0000464Q-Q",
  xsnmsubasp_3 =	"f0000488QQQ",
  xxlor_3 =		"f0000490QQQ",
  xscvuxdsp_2 =		"f00004a0Q-Q",
  xsnmsubmsp_3 =	"f00004c8QQQ",
  xxlxor_3 =		"f00004d0QQQ",
  xscvsxdsp_2 =		"f00004e0Q-Q",
  xsmaxdp_3 =		"f0000500QQQ",
  xsnmaddadp_3 =	"f0000508QQQ",
  xxlnor_3 =		"f0000510QQQ",
  xscvdpuxds_2 =	"f0000520Q-Q",
  xscvspdp_2 =		"f0000524Q-Q",
  xscvspdpn_2 =		"f000052cQ-Q",
  xsmindp_3 =		"f0000540QQQ",
  xsnmaddmdp_3 =	"f0000548QQQ",
  xxlorc_3 =		"f0000550QQQ",
  xscvdpsxds_2 =	"f0000560Q-Q",
  xsabsdp_2 =		"f0000564Q-Q",
  xscpsgndp_3 =		"f0000580QQQ",
  xsnmsubadp_3 =	"f0000588QQQ",
  xxlnand_3 =		"f0000590QQQ",
  xscvuxddp_2 =		"f00005a0Q-Q",
  xsnabsdp_2 =		"f00005a4Q-Q",
  xsnmsubmdp_3 =	"f00005c8QQQ",
  xxleqv_3 =		"f00005d0QQQ",
  xscvsxddp_2 =		"f00005e0Q-Q",
  xsnegdp_2 =		"f00005e4Q-Q",
  xvmaxsp_3 =		"f0000600QQQ",
  xvnmaddasp_3 =	"f0000608QQQ",
  ["xvcmpeqsp._3"] =	"f0000618QQQ",
  xvcvspuxds_2 =	"f0000620Q-Q",
  xvcvdpsp_2 =		"f0000624Q-Q",
  xvminsp_3 =		"f0000640QQQ",
  xvnmaddmsp_3 =	"f0000648QQQ",
  ["xvcmpgtsp._3"] =	"f0000658QQQ",
  xvcvspsxds_2 =	"f0000660Q-Q",
  xvabssp_2 =		"f0000664Q-Q",
  xvcpsgnsp_3 =		"f0000680QQQ",
  xvnmsubasp_3 =	"f0000688QQQ",
  ["xvcmpgesp._3"] =	"f0000698QQQ",
  xvcvuxdsp_2 =		"f00006a0Q-Q",
  xvnabssp_2 =		"f00006a4Q-Q",
  xvnmsubmsp_3 =	"f00006c8QQQ",
  xvcvsxdsp_2 =		"f00006e0Q-Q",
  xvnegsp_2 =		"f00006e4Q-Q",
  xvmaxdp_3 =		"f0000700QQQ",
  xvnmaddadp_3 =	"f0000708QQQ",
  ["xvcmpeqdp._3"] =	"f0000718QQQ",
  xvcvdpuxds_2 =	"f0000720Q-Q",
  xvcvspdp_2 =		"f0000724Q-Q",
  xvmindp_3 =		"f0000740QQQ",
  xvnmaddmdp_3 =	"f0000748QQQ",
  ["xvcmpgtdp._3"] =	"f0000758QQQ",
  xvcvdpsxds_2 =	"f0000760Q-Q",
  xvabsdp_2 =		"f0000764Q-Q",
  xvcpsgndp_3 =		"f0000780QQQ",
  xvnmsubadp_3 =	"f0000788QQQ",
  ["xvcmpgedp._3"] =	"f0000798QQQ",
  xvcvuxddp_2 =		"f00007a0Q-Q",
  xvnabsdp_2 =		"f00007a4Q-Q",
  xvnmsubmdp_3 =	"f00007c8QQQ",
  xvcvsxddp_2 =		"f00007e0Q-Q",
  xvnegdp_2 =		"f00007e4Q-Q",

  -- Primary opcode 61:
  stfdp_2 =	"f4000000F:D", -- NYI: displacement must be divisible by 4.

  -- Primary opcode 62:
  stq_2 =	"f8000002R:D", -- NYI: displacement must be divisible by 8.

  -- Primary opcode 63:
  fdiv_3 =	"fc000024FFF.",
  fsub_3 =	"fc000028FFF.",
  fadd_3 =	"fc00002aFFF.",
  fsqrt_2 =	"fc00002cF-F.",
  fsel_4 =	"fc00002eFFFF~.",
  fre_2 =	"fc000030F-F.",
  fmul_3 =	"fc000032FF-F.",
  frsqrte_2 =	"fc000034F-F.",
  fmsub_4 =	"fc000038FFFF~.",
  fmadd_4 =	"fc00003aFFFF~.",
  fnmsub_4 =	"fc00003cFFFF~.",
  fnmadd_4 =	"fc00003eFFFF~.",
  fcmpu_3 =	"fc000000XFF",
  fcpsgn_3 =	"fc000010FFF.",
  fcmpo_3 =	"fc000040XFF",
  mtfsb1_1 =	"fc00004cA",
  fneg_2 =	"fc000050F-F.",
  mcrfs_2 =	"fc000080XX",
  mtfsb0_1 =	"fc00008cA",
  fmr_2 =	"fc000090F-F.",
  frsp_2 =	"fc000018F-F.",
  fctiw_2 =	"fc00001cF-F.",
  fctiwz_2 =	"fc00001eF-F.",
  ftdiv_2 =	"fc000100X-F.",
  fctiwu_2 =	"fc00011cF-F.",
  fctiwuz_2 =	"fc00011eF-F.",
  mtfsfi_2 =	"fc00010cAA", -- NYI: upshift.
  fnabs_2 =	"fc000110F-F.",
  ftsqrt_2 =	"fc000140X-F.",
  fabs_2 =	"fc000210F-F.",
  frin_2 =	"fc000310F-F.",
  friz_2 =	"fc000350F-F.",
  frip_2 =	"fc000390F-F.",
  frim_2 =	"fc0003d0F-F.",
  mffs_1 =	"fc00048eF.",
  -- NYI: mtfsf, mtfsb0, mtfsb1.
  fctid_2 =	"fc00065cF-F.",
  fctidz_2 =	"fc00065eF-F.",
  fmrgow_3 =	"fc00068cFFF",
  fcfid_2 =	"fc00069cF-F.",
  fctidu_2 =	"fc00075cF-F.",
  fctiduz_2 =	"fc00075eF-F.",
  fmrgew_3 =	"fc00078cFFF",
  fcfidu_2 =	"fc00079cF-F.",

  daddq_3 =	"fc000004F:F:F:.",
  dquaq_4 =	"fc000006F:F:F:Z.",
  dmulq_3 =	"fc000044F:F:F:.",
  drrndq_4 =	"fc000046F:F:F:Z.",
  dscliq_3 =	"fc000084F:F:6.",
  dquaiq_4 =	"fc000086SF:~F:Z.",
  dscriq_3 =	"fc0000c4F:F:6.",
  drintxq_4 =	"fc0000c61F:~F:Z.",
  dcmpoq_3 =	"fc000104XF:F:",
  dtstexq_3 =	"fc000144XF:F:",
  dtstdcq_3 =	"fc000184XF:6",
  dtstdgq_3 =	"fc0001c4XF:6",
  drintnq_4 =	"fc0001c61F:~F:Z.",
  dctqpq_2 =	"fc000204F:-F:.",
  dctfixq_2 =	"fc000244F:-F:.",
  ddedpdq_3 =	"fc000284ZF:~F:.",
  dxexq_2 =	"fc0002c4F:-F:.",
  dsubq_3 =	"fc000404F:F:F:.",
  ddivq_3 =	"fc000444F:F:F:.",
  dcmpuq_3 =	"fc000504XF:F:",
  dtstsfq_3 =	"fc000544XF:F:",
  drdpq_2 =	"fc000604F:-F:.",
  dcffixq_2 =	"fc000644F:-F:.",
  denbcdq_3 =	"fc000684YF:~F:.",
  diexq_3 =	"fc0006c4F:FF:.",

  -- Primary opcode 4, SPE APU extension:
  evaddw_3 =		"10000200RRR",
  evaddiw_3 =		"10000202RAR~",
  evsubw_3 =		"10000204RRR~",
  evsubiw_3 =		"10000206RAR~",
  evabs_2 =		"10000208RR",
  evneg_2 =		"10000209RR",
  evextsb_2 =		"1000020aRR",
  evextsh_2 =		"1000020bRR",
  evrndw_2 =		"1000020cRR",
  evcntlzw_2 =		"1000020dRR",
  evcntlsw_2 =		"1000020eRR",
  brinc_3 =		"1000020fRRR",
  evand_3 =		"10000211RRR",
  evandc_3 =		"10000212RRR",
  evxor_3 =		"10000216RRR",
  evor_3 =		"10000217RRR",
  evmr_2 =		"10000217RR=",
  evnor_3 =		"10000218RRR",
  evnot_2 =		"10000218RR=",
  eveqv_3 =		"10000219RRR",
  evorc_3 =		"1000021bRRR",
  evnand_3 =		"1000021eRRR",
  evsrwu_3 =		"10000220RRR",
  evsrws_3 =		"10000221RRR",
  evsrwiu_3 =		"10000222RRA",
  evsrwis_3 =		"10000223RRA",
  evslw_3 =		"10000224RRR",
  evslwi_3 =		"10000226RRA",
  evrlw_3 =		"10000228RRR",
  evsplati_2 =		"10000229RS",
  evrlwi_3 =		"1000022aRRA",
  evsplatfi_2 =		"1000022bRS",
  evmergehi_3 =		"1000022cRRR",
  evmergelo_3 =		"1000022dRRR",
  evcmpgtu_3 =		"10000230XRR",
  evcmpgtu_2 =		"10000230-RR",
  evcmpgts_3 =		"10000231XRR",
  evcmpgts_2 =		"10000231-RR",
  evcmpltu_3 =		"10000232XRR",
  evcmpltu_2 =		"10000232-RR",
  evcmplts_3 =		"10000233XRR",
  evcmplts_2 =		"10000233-RR",
  evcmpeq_3 =		"10000234XRR",
  evcmpeq_2 =		"10000234-RR",
  evsel_4 =		"10000278RRRW",
  evsel_3 =		"10000278RRR",
  evfsadd_3 =		"10000280RRR",
  evfssub_3 =		"10000281RRR",
  evfsabs_2 =		"10000284RR",
  evfsnabs_2 =		"10000285RR",
  evfsneg_2 =		"10000286RR",
  evfsmul_3 =		"10000288RRR",
  evfsdiv_3 =		"10000289RRR",
  evfscmpgt_3 =		"1000028cXRR",
  evfscmpgt_2 =		"1000028c-RR",
  evfscmplt_3 =		"1000028dXRR",
  evfscmplt_2 =		"1000028d-RR",
  evfscmpeq_3 =		"1000028eXRR",
  evfscmpeq_2 =		"1000028e-RR",
  evfscfui_2 =		"10000290R-R",
  evfscfsi_2 =		"10000291R-R",
  evfscfuf_2 =		"10000292R-R",
  evfscfsf_2 =		"10000293R-R",
  evfsctui_2 =		"10000294R-R",
  evfsctsi_2 =		"10000295R-R",
  evfsctuf_2 =		"10000296R-R",
  evfsctsf_2 =		"10000297R-R",
  evfsctuiz_2 =		"10000298R-R",
  evfsctsiz_2 =		"1000029aR-R",
  evfststgt_3 =		"1000029cXRR",
  evfststgt_2 =		"1000029c-RR",
  evfststlt_3 =		"1000029dXRR",
  evfststlt_2 =		"1000029d-RR",
  evfststeq_3 =		"1000029eXRR",
  evfststeq_2 =		"1000029e-RR",
  efsadd_3 =		"100002c0RRR",
  efssub_3 =		"100002c1RRR",
  efsabs_2 =		"100002c4RR",
  efsnabs_2 =		"100002c5RR",
  efsneg_2 =		"100002c6RR",
  efsmul_3 =		"100002c8RRR",
  efsdiv_3 =		"100002c9RRR",
  efscmpgt_3 =		"100002ccXRR",
  efscmpgt_2 =		"100002cc-RR",
  efscmplt_3 =		"100002cdXRR",
  efscmplt_2 =		"100002cd-RR",
  efscmpeq_3 =		"100002ceXRR",
  efscmpeq_2 =		"100002ce-RR",
  efscfd_2 =		"100002cfR-R",
  efscfui_2 =		"100002d0R-R",
  efscfsi_2 =		"100002d1R-R",
  efscfuf_2 =		"100002d2R-R",
  efscfsf_2 =		"100002d3R-R",
  efsctui_2 =		"100002d4R-R",
  efsctsi_2 =		"100002d5R-R",
  efsctuf_2 =		"100002d6R-R",
  efsctsf_2 =		"100002d7R-R",
  efsctuiz_2 =		"100002d8R-R",
  efsctsiz_2 =		"100002daR-R",
  efststgt_3 =		"100002dcXRR",
  efststgt_2 =		"100002dc-RR",
  efststlt_3 =		"100002ddXRR",
  efststlt_2 =		"100002dd-RR",
  efststeq_3 =		"100002deXRR",
  efststeq_2 =		"100002de-RR",
  efdadd_3 =		"100002e0RRR",
  efdsub_3 =		"100002e1RRR",
  efdcfuid_2 =		"100002e2R-R",
  efdcfsid_2 =		"100002e3R-R",
  efdabs_2 =		"100002e4RR",
  efdnabs_2 =		"100002e5RR",
  efdneg_2 =		"100002e6RR",
  efdmul_3 =		"100002e8RRR",
  efddiv_3 =		"100002e9RRR",
  efdctuidz_2 =		"100002eaR-R",
  efdctsidz_2 =		"100002ebR-R",
  efdcmpgt_3 =		"100002ecXRR",
  efdcmpgt_2 =		"100002ec-RR",
  efdcmplt_3 =		"100002edXRR",
  efdcmplt_2 =		"100002ed-RR",
  efdcmpeq_3 =		"100002eeXRR",
  efdcmpeq_2 =		"100002ee-RR",
  efdcfs_2 =		"100002efR-R",
  efdcfui_2 =		"100002f0R-R",
  efdcfsi_2 =		"100002f1R-R",
  efdcfuf_2 =		"100002f2R-R",
  efdcfsf_2 =		"100002f3R-R",
  efdctui_2 =		"100002f4R-R",
  efdctsi_2 =		"100002f5R-R",
  efdctuf_2 =		"100002f6R-R",
  efdctsf_2 =		"100002f7R-R",
  efdctuiz_2 =		"100002f8R-R",
  efdctsiz_2 =		"100002faR-R",
  efdtstgt_3 =		"100002fcXRR",
  efdtstgt_2 =		"100002fc-RR",
  efdtstlt_3 =		"100002fdXRR",
  efdtstlt_2 =		"100002fd-RR",
  efdtsteq_3 =		"100002feXRR",
  efdtsteq_2 =		"100002fe-RR",
  evlddx_3 =		"10000300RR0R",
  evldd_2 =		"10000301R8",
  evldwx_3 =		"10000302RR0R",
  evldw_2 =		"10000303R8",
  evldhx_3 =		"10000304RR0R",
  evldh_2 =		"10000305R8",
  evlwhex_3 =		"10000310RR0R",
  evlwhe_2 =		"10000311R4",
  evlwhoux_3 =		"10000314RR0R",
  evlwhou_2 =		"10000315R4",
  evlwhosx_3 =		"10000316RR0R",
  evlwhos_2 =		"10000317R4",
  evstddx_3 =		"10000320RR0R",
  evstdd_2 =		"10000321R8",
  evstdwx_3 =		"10000322RR0R",
  evstdw_2 =		"10000323R8",
  evstdhx_3 =		"10000324RR0R",
  evstdh_2 =		"10000325R8",
  evstwhex_3 =		"10000330RR0R",
  evstwhe_2 =		"10000331R4",
  evstwhox_3 =		"10000334RR0R",
  evstwho_2 =		"10000335R4",
  evstwwex_3 =		"10000338RR0R",
  evstwwe_2 =		"10000339R4",
  evstwwox_3 =		"1000033cRR0R",
  evstwwo_2 =		"1000033dR4",
  evmhessf_3 =		"10000403RRR",
  evmhossf_3 =		"10000407RRR",
  evmheumi_3 =		"10000408RRR",
  evmhesmi_3 =		"10000409RRR",
  evmhesmf_3 =		"1000040bRRR",
  evmhoumi_3 =		"1000040cRRR",
  evmhosmi_3 =		"1000040dRRR",
  evmhosmf_3 =		"1000040fRRR",
  evmhessfa_3 =		"10000423RRR",
  evmhossfa_3 =		"10000427RRR",
  evmheumia_3 =		"10000428RRR",
  evmhesmia_3 =		"10000429RRR",
  evmhesmfa_3 =		"1000042bRRR",
  evmhoumia_3 =		"1000042cRRR",
  evmhosmia_3 =		"1000042dRRR",
  evmhosmfa_3 =		"1000042fRRR",
  evmwhssf_3 =		"10000447RRR",
  evmwlumi_3 =		"10000448RRR",
  evmwhumi_3 =		"1000044cRRR",
  evmwhsmi_3 =		"1000044dRRR",
  evmwhsmf_3 =		"1000044fRRR",
  evmwssf_3 =		"10000453RRR",
  evmwumi_3 =		"10000458RRR",
  evmwsmi_3 =		"10000459RRR",
  evmwsmf_3 =		"1000045bRRR",
  evmwhssfa_3 =		"10000467RRR",
  evmwlumia_3 =		"10000468RRR",
  evmwhumia_3 =		"1000046cRRR",
  evmwhsmia_3 =		"1000046dRRR",
  evmwhsmfa_3 =		"1000046fRRR",
  evmwssfa_3 =		"10000473RRR",
  evmwumia_3 =		"10000478RRR",
  evmwsmia_3 =		"10000479RRR",
  evmwsmfa_3 =		"1000047bRRR",
  evmra_2 =		"100004c4RR",
  evdivws_3 =		"100004c6RRR",
  evdivwu_3 =		"100004c7RRR",
  evmwssfaa_3 =		"10000553RRR",
  evmwumiaa_3 =		"10000558RRR",
  evmwsmiaa_3 =		"10000559RRR",
  evmwsmfaa_3 =		"1000055bRRR",
  evmwssfan_3 =		"100005d3RRR",
  evmwumian_3 =		"100005d8RRR",
  evmwsmian_3 =		"100005d9RRR",
  evmwsmfan_3 =		"100005dbRRR",
  evmergehilo_3 =	"1000022eRRR",
  evmergelohi_3 =	"1000022fRRR",
  evlhhesplatx_3 =	"10000308RR0R",
  evlhhesplat_2 =	"10000309R2",
  evlhhousplatx_3 =	"1000030cRR0R",
  evlhhousplat_2 =	"1000030dR2",
  evlhhossplatx_3 =	"1000030eRR0R",
  evlhhossplat_2 =	"1000030fR2",
  evlwwsplatx_3 =	"10000318RR0R",
  evlwwsplat_2 =	"10000319R4",
  evlwhsplatx_3 =	"1000031cRR0R",
  evlwhsplat_2 =	"1000031dR4",
  evaddusiaaw_2 =	"100004c0RR",
  evaddssiaaw_2 =	"100004c1RR",
  evsubfusiaaw_2 =	"100004c2RR",
  evsubfssiaaw_2 =	"100004c3RR",
  evaddumiaaw_2 =	"100004c8RR",
  evaddsmiaaw_2 =	"100004c9RR",
  evsubfumiaaw_2 =	"100004caRR",
  evsubfsmiaaw_2 =	"100004cbRR",
  evmheusiaaw_3 =	"10000500RRR",
  evmhessiaaw_3 =	"10000501RRR",
  evmhessfaaw_3 =	"10000503RRR",
  evmhousiaaw_3 =	"10000504RRR",
  evmhossiaaw_3 =	"10000505RRR",
  evmhossfaaw_3 =	"10000507RRR",
  evmheumiaaw_3 =	"10000508RRR",
  evmhesmiaaw_3 =	"10000509RRR",
  evmhesmfaaw_3 =	"1000050bRRR",
  evmhoumiaaw_3 =	"1000050cRRR",
  evmhosmiaaw_3 =	"1000050dRRR",
  evmhosmfaaw_3 =	"1000050fRRR",
  evmhegumiaa_3 =	"10000528RRR",
  evmhegsmiaa_3 =	"10000529RRR",
  evmhegsmfaa_3 =	"1000052bRRR",
  evmhogumiaa_3 =	"1000052cRRR",
  evmhogsmiaa_3 =	"1000052dRRR",
  evmhogsmfaa_3 =	"1000052fRRR",
  evmwlusiaaw_3 =	"10000540RRR",
  evmwlssiaaw_3 =	"10000541RRR",
  evmwlumiaaw_3 =	"10000548RRR",
  evmwlsmiaaw_3 =	"10000549RRR",
  evmheusianw_3 =	"10000580RRR",
  evmhessianw_3 =	"10000581RRR",
  evmhessfanw_3 =	"10000583RRR",
  evmhousianw_3 =	"10000584RRR",
  evmhossianw_3 =	"10000585RRR",
  evmhossfanw_3 =	"10000587RRR",
  evmheumianw_3 =	"10000588RRR",
  evmhesmianw_3 =	"10000589RRR",
  evmhesmfanw_3 =	"1000058bRRR",
  evmhoumianw_3 =	"1000058cRRR",
  evmhosmianw_3 =	"1000058dRRR",
  evmhosmfanw_3 =	"1000058fRRR",
  evmhegumian_3 =	"100005a8RRR",
  evmhegsmian_3 =	"100005a9RRR",
  evmhegsmfan_3 =	"100005abRRR",
  evmhogumian_3 =	"100005acRRR",
  evmhogsmian_3 =	"100005adRRR",
  evmhogsmfan_3 =	"100005afRRR",
  evmwlusianw_3 =	"100005c0RRR",
  evmwlssianw_3 =	"100005c1RRR",
  evmwlumianw_3 =	"100005c8RRR",
  evmwlsmianw_3 =	"100005c9RRR",

  -- NYI: Book E instructions.
}

-- Add mnemonics for "." variants.
do
  local t = {}
  for k,v in pairs(map_op) do
    if type(v) == "string" and sub(v, -1) == "." then
      local v2 = sub(v, 1, 7)..char(byte(v, 8)+1)..sub(v, 9, -2)
      t[sub(k, 1, -3).."."..sub(k, -2)] = v2
    end
  end
  for k,v in pairs(t) do
    map_op[k] = v
  end
end

-- Add more branch mnemonics.
for cond,c in pairs(map_cond) do
  local b1 = "b"..cond
  local c1 = shl(band(c, 3), 16) + (c < 4 and 0x01000000 or 0)
  -- bX[l]
  map_op[b1.."_1"] = tohex(0x40800000 + c1).."K"
  map_op[b1.."y_1"] = tohex(0x40a00000 + c1).."K"
  map_op[b1.."l_1"] = tohex(0x40800001 + c1).."K"
  map_op[b1.."_2"] = tohex(0x40800000 + c1).."-XK"
  map_op[b1.."y_2"] = tohex(0x40a00000 + c1).."-XK"
  map_op[b1.."l_2"] = tohex(0x40800001 + c1).."-XK"
  -- bXlr[l]
  map_op[b1.."lr_0"] = tohex(0x4c800020 + c1)
  map_op[b1.."lrl_0"] = tohex(0x4c800021 + c1)
  map_op[b1.."ctr_0"] = tohex(0x4c800420 + c1)
  map_op[b1.."ctrl_0"] = tohex(0x4c800421 + c1)
  -- bXctr[l]
  map_op[b1.."lr_1"] = tohex(0x4c800020 + c1).."-X"
  map_op[b1.."lrl_1"] = tohex(0x4c800021 + c1).."-X"
  map_op[b1.."ctr_1"] = tohex(0x4c800420 + c1).."-X"
  map_op[b1.."ctrl_1"] = tohex(0x4c800421 + c1).."-X"
end

------------------------------------------------------------------------------

local function parse_gpr(expr)
  local tname, ovreg = match(expr, "^([%w_]+):(r[1-3]?[0-9])$")
  local tp = map_type[tname or expr]
  if tp then
    local reg = ovreg or tp.reg
    if not reg then
      werror("type `"..(tname or expr).."' needs a register override")
    end
    expr = reg
  end
  local r = match(expr, "^r([1-3]?[0-9])$")
  if r then
    r = tonumber(r)
    if r <= 31 then return r, tp end
  end
  werror("bad register name `"..expr.."'")
end

local function parse_fpr(expr)
  local r = match(expr, "^f([1-3]?[0-9])$")
  if r then
    r = tonumber(r)
    if r <= 31 then return r end
  end
  werror("bad register name `"..expr.."'")
end

local function parse_vr(expr)
  local r = match(expr, "^v([1-3]?[0-9])$")
  if r then
    r = tonumber(r)
    if r <= 31 then return r end
  end
  werror("bad register name `"..expr.."'")
end

local function parse_vs(expr)
  local r = match(expr, "^vs([1-6]?[0-9])$")
  if r then
    r = tonumber(r)
    if r <= 63 then return r end
  end
  werror("bad register name `"..expr.."'")
end

local function parse_cr(expr)
  local r = match(expr, "^cr([0-7])$")
  if r then return tonumber(r) end
  werror("bad condition register name `"..expr.."'")
end

local function parse_cond(expr)
  local r, cond = match(expr, "^4%*cr([0-7])%+(%w%w)$")
  if r then
    r = tonumber(r)
    local c = map_cond[cond]
    if c and c < 4 then return r*4+c end
  end
  werror("bad condition bit name `"..expr.."'")
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
    if ok then return y end
  end
  return nil
end

local function parse_imm(imm, bits, shift, scale, signed)
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
  elseif match(imm, "^[rfv]([1-3]?[0-9])$") or
	 match(imm, "^vs([1-6]?[0-9])$") or
	 match(imm, "^([%w_]+):(r[1-3]?[0-9])$") then
    werror("expected immediate operand, got register")
  else
    waction("IMM", (signed and 32768 or 0)+scale*1024+bits*32+shift, imm)
    return 0
  end
end

local function parse_shiftmask(imm, isshift)
  local n = parse_number(imm)
  if n then
    if shr(n, 6) == 0 then
      local lsb = band(n, 31)
      local msb = n - lsb
      return isshift and (shl(lsb, 11)+shr(msb, 4)) or (shl(lsb, 6)+msb)
    end
    werror("out of range immediate `"..imm.."'")
  elseif match(imm, "^r([1-3]?[0-9])$") or
	 match(imm, "^([%w_]+):(r[1-3]?[0-9])$") then
    werror("expected immediate operand, got register")
  else
    waction("IMMSH", isshift and 1 or 0, imm)
    return 0;
  end
end

local function parse_disp(disp)
  local imm, reg = match(disp, "^(.*)%(([%w_:]+)%)$")
  if imm then
    local r = parse_gpr(reg)
    if r == 0 then werror("cannot use r0 in displacement") end
    return shl(r, 16) + parse_imm(imm, 16, 0, 0, true)
  end
  local reg, tailr = match(disp, "^([%w_:]+)%s*(.*)$")
  if reg and tailr ~= "" then
    local r, tp = parse_gpr(reg)
    if r == 0 then werror("cannot use r0 in displacement") end
    if tp then
      waction("IMM", 32768+16*32, format(tp.ctypefmt, tailr))
      return shl(r, 16)
    end
  end
  werror("bad displacement `"..disp.."'")
end

local function parse_u5disp(disp, scale)
  local imm, reg = match(disp, "^(.*)%(([%w_:]+)%)$")
  if imm then
    local r = parse_gpr(reg)
    if r == 0 then werror("cannot use r0 in displacement") end
    return shl(r, 16) + parse_imm(imm, 5, 11, scale, false)
  end
  local reg, tailr = match(disp, "^([%w_:]+)%s*(.*)$")
  if reg and tailr ~= "" then
    local r, tp = parse_gpr(reg)
    if r == 0 then werror("cannot use r0 in displacement") end
    if tp then
      waction("IMM", scale*1024+5*32+11, format(tp.ctypefmt, tailr))
      return shl(r, 16)
    end
  end
  werror("bad displacement `"..disp.."'")
end

local function parse_label(label, def)
  local prefix = sub(label, 1, 2)
  -- =>label (pc label reference)
  if prefix == "=>" then
    return "PC", 0, sub(label, 3)
  end
  -- ->name (global label reference)
  if prefix == "->" then
    return "LG", map_global[sub(label, 3)]
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
  end
  werror("bad label `"..label.."'")
end

------------------------------------------------------------------------------

-- Handle opcodes defined with template strings.
op_template = function(params, template, nparams)
  if not params then return sub(template, 9) end
  local op = tonumber(sub(template, 1, 8), 16)
  local n, rs = 1, 26

  -- Limit number of section buffer positions used by a single dasm_put().
  -- A single opcode needs a maximum of 3 positions (rlwinm).
  if secpos+3 > maxsecpos then wflush() end
  local pos = wpos()

  -- Process each character.
  for p in gmatch(sub(template, 9), ".") do
    if p == "R" then
      rs = rs - 5; op = op + shl(parse_gpr(params[n]), rs); n = n + 1
    elseif p == "F" then
      rs = rs - 5; op = op + shl(parse_fpr(params[n]), rs); n = n + 1
    elseif p == "V" then
      rs = rs - 5; op = op + shl(parse_vr(params[n]), rs); n = n + 1
    elseif p == "Q" then
      local vs = parse_vs(params[n]); n = n + 1; rs = rs - 5
      local sh = rs == 6 and 2 or 3 + band(shr(rs, 1), 3)
      op = op + shl(band(vs, 31), rs) + shr(band(vs, 32), sh)
    elseif p == "q" then
      local vs = parse_vs(params[n]); n = n + 1
      op = op + shl(band(vs, 31), 21) + shr(band(vs, 32), 5)
    elseif p == "A" then
      rs = rs - 5; op = op + parse_imm(params[n], 5, rs, 0, false); n = n + 1
    elseif p == "S" then
      rs = rs - 5; op = op + parse_imm(params[n], 5, rs, 0, true); n = n + 1
    elseif p == "I" then
      op = op + parse_imm(params[n], 16, 0, 0, true); n = n + 1
    elseif p == "U" then
      op = op + parse_imm(params[n], 16, 0, 0, false); n = n + 1
    elseif p == "D" then
      op = op + parse_disp(params[n]); n = n + 1
    elseif p == "2" then
      op = op + parse_u5disp(params[n], 1); n = n + 1
    elseif p == "4" then
      op = op + parse_u5disp(params[n], 2); n = n + 1
    elseif p == "8" then
      op = op + parse_u5disp(params[n], 3); n = n + 1
    elseif p == "C" then
      rs = rs - 5; op = op + shl(parse_cond(params[n]), rs); n = n + 1
    elseif p == "X" then
      rs = rs - 5; op = op + shl(parse_cr(params[n]), rs+2); n = n + 1
    elseif p == "1" then
      rs = rs - 5; op = op + parse_imm(params[n], 1, rs, 0, false); n = n + 1
    elseif p == "g" then
      rs = rs - 5; op = op + parse_imm(params[n], 2, rs, 0, false); n = n + 1
    elseif p == "3" then
      rs = rs - 5; op = op + parse_imm(params[n], 3, rs, 0, false); n = n + 1
    elseif p == "P" then
      rs = rs - 5; op = op + parse_imm(params[n], 4, rs, 0, false); n = n + 1
    elseif p == "p" then
      op = op + parse_imm(params[n], 4, rs, 0, false); n = n + 1
    elseif p == "6" then
      rs = rs - 6; op = op + parse_imm(params[n], 6, rs, 0, false); n = n + 1
    elseif p == "Y" then
      rs = rs - 5; op = op + parse_imm(params[n], 1, rs+4, 0, false); n = n + 1
    elseif p == "y" then
      rs = rs - 5; op = op + parse_imm(params[n], 1, rs+3, 0, false); n = n + 1
    elseif p == "Z" then
      rs = rs - 5; op = op + parse_imm(params[n], 2, rs+3, 0, false); n = n + 1
    elseif p == "z" then
      rs = rs - 5; op = op + parse_imm(params[n], 2, rs+2, 0, false); n = n + 1
    elseif p == "W" then
      op = op + parse_cr(params[n]); n = n + 1
    elseif p == "G" then
      op = op + parse_imm(params[n], 8, 12, 0, false); n = n + 1
    elseif p == "H" then
      op = op + parse_shiftmask(params[n], true); n = n + 1
    elseif p == "M" then
      op = op + parse_shiftmask(params[n], false); n = n + 1
    elseif p == "J" or p == "K" then
      local mode, m, s = parse_label(params[n], false)
      if p == "K" then m = m + 2048 end
      waction("REL_"..mode, m, s, 1)
      n = n + 1
    elseif p == "0" then
      if band(shr(op, rs), 31) == 0 then werror("cannot use r0") end
    elseif p == "=" or p == "%" then
      local t = band(shr(op, p == "%" and rs+5 or rs), 31)
      rs = rs - 5
      op = op + shl(t, rs)
    elseif p == "~" then
      local mm = shl(31, rs)
      local lo = band(op, mm)
      local hi = band(op, shl(mm, 5))
      op = op - lo - hi + shl(lo, 5) + shr(hi, 5)
    elseif p == ":" then
      if band(shr(op, rs), 1) ~= 0 then werror("register pair expected") end
    elseif p == "-" then
      rs = rs - 5
    elseif p == "." then
      -- Ignored.
    else
      assert(false)
    end
  end
  wputpos(pos, op)
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
  if mode == "EXT" then werror("bad label definition") end
  waction("LABEL_"..mode, n, s, 1)
end

------------------------------------------------------------------------------

-- Pseudo-opcodes for data storage.
map_op[".long_*"] = function(params)
  if not params then return "imm..." end
  for _,p in ipairs(params) do
    local n = tonumber(p)
    if not n then werror("bad immediate `"..p.."'") end
    if n < 0 then n = n + 2^32 end
    wputw(n)
    if secpos+2 > maxsecpos then wflush() end
  end
end

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

