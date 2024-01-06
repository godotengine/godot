------------------------------------------------------------------------------
-- DynASM RISC-V module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- See dynasm.lua for full copyright notice.
------------------------------------------------------------------------------

local riscv32 = riscv32
local riscv64 = riscv64

-- Module information:
local _info = {
  arch =	riscv32 and "riscv32" or riscv64 and "riscv64",
  description =	"DynASM RISC-V module",
  version =	"1.5.0",
  vernum =	 10500,
  release =	"2022-07-12",
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

local function __orderedIndexGen(t)
    local orderedIndex = {}
    for key in pairs(t) do
        table.insert(orderedIndex, key)
    end
    table.sort( orderedIndex )
    return orderedIndex
end

local function __orderedNext(t, state)
    local key = nil
    if state == nil then
        t.__orderedIndex = __orderedIndexGen(t)
        key = t.__orderedIndex[1]
    else
        local j = 0
        for _,_ in pairs(t.__orderedIndex) do j = j + 1 end
        for i = 1, j do
            if t.__orderedIndex[i] == state then
                key = t.__orderedIndex[i+1]
            end
        end
    end

    if key then
        return key, t[key]
    end

    t.__orderedIndex = nil
    return
end

local function opairs(t)
    return __orderedNext, t, nil
end

-- Inherited tables and callbacks.
local g_opt, g_arch
local wline, werror, wfatal, wwarn

-- Action name list.
-- CHECK: Keep this in sync with the C code!
local action_names = {
  "STOP", "SECTION", "ESC", "REL_EXT",
  "ALIGN", "REL_LG", "LABEL_LG",
  "REL_PC", "LABEL_PC", "IMM", "IMMS",
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
  wputxw(w * 0x100000 + (val or 0) * 16)
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
  if band(n, 0xf) == 0 then waction("ESC") end
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
  assert(n >= -0x80000000 and n <= 0xffffffff and n % 1 == 0, "word out of range")
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
local map_archdef = {
  ra = "x1", sp = "x2",
} -- Ext. register name -> int. name.

local map_type = {}		-- Type name -> { ctype, reg }
local ctypenum = 0		-- Type number (for Dt... macros).

-- Reverse defines for registers.
function _M.revdef(s)
  if s == "x1" then return "ra"
  elseif s == "x2" then return "sp" end
  return s
end

------------------------------------------------------------------------------

-- Template strings for RISC-V instructions.
local map_op = {}

local map_op_rv32imafd = {

  -- DASM pseudo-instrs
  empty_0 = "ffffffff",
  call_1 = "7fffffffJ",

  -- RV32I
  lui_2 = "00000037DU",
  auipc_2 = "00000017DA",

  jal_2  = "0000006fDJ",
  jalr_3 = "00000067DRJ",
  -- pseudo-instrs
  j_1 = "0000006fJ",
  jal_1 = "000000efJ",
  jr_1 = "00000067R",
  jalr_1 = "000000e7R",
  jalr_2 = "000000e7RJ",

  beq_3  = "00000063RrB",
  bne_3  = "00001063RrB",
  blt_3  = "00004063RrB",
  bge_3  = "00005063RrB",
  bltu_3 = "00006063RrB",
  bgeu_3 = "00007063RrB",
  -- pseudo-instrs
  bnez_2 = "00001063RB",
  beqz_2 = "00000063RB",
  blez_2 = "00005063rB",
  bgez_2 = "00005063RB",
  bltz_2 = "00004063RB",
  bgtz_2 = "00004063rB",
  bgt_3 = "00004063rRB",
  ble_3 = "00005063rRB",
  bgtu_3 = "00006063rRB",
  bleu_3 = "00007063rRB",

  lb_2  = "00000003DL",
  lh_2  = "00001003DL",
  lw_2  = "00002003DL",
  lbu_2 = "00004003DL",
  lhu_2 = "00005003DL",

  sb_2 = "00000023rS",
  sh_2 = "00001023rS",
  sw_2 = "00002023rS",

  addi_3  = "00000013DRI",
  slti_3  = "00002013DRI",
  sltiu_3 = "00003013DRI",
  xori_3 = "00004013DRI",
  ori_3 = "00006013DRI",
  andi_3 = "00007013DRI",
  slli_3 = "00001013DRi",
  srli_3 = "00005013DRi",
  srai_3 = "40005013DRi",
  -- pseudo-instrs
  seqz_2 = "00103013DR",
  ["zext.b_2"] = "0ff07013DR",

  add_3 = "00000033DRr",
  sub_3 = "40000033DRr",
  sll_3 = "00001033DRr",
  slt_3 = "00002033DRr",
  sltu_3 = "00003033DRr",
  xor_3 = "00004033DRr",
  srl_3 = "00005033DRr",
  sra_3 = "40005033DRr",
  or_3 = "00006033DRr",
  and_3 = "00007033DRr",
  -- pseudo-instrs
  snez_2 = "00003033Dr",
  sltz_2 = "00002033DR",
  sgtz_2 = "00002033Dr",

  ecall_0 = "00000073",
  ebreak_0 = "00100073",

  nop_0 = "00000013",
  li_2 = "00000013DI",
  mv_2 = "00000013DR",
  not_2 = "fff04013DR",
  neg_2 = "40000033Dr",
  ret_0 = "00008067",

  -- RV32M
  mul_3    = "02000033DRr",
  mulh_3   = "02001033DRr",
  mulhsu_3 = "02002033DRr",
  mulhu_3  = "02003033DRr",
  div_3  = "02004033DRr",
  divu_3 = "02005033DRr",
  rem_3  = "02006033DRr",
  remu_3 = "02007033DRr",

  -- RV32A
  ["lr.w_2"] = "c0000053FR",
  ["sc.w_2"] = "c0001053FRr",
  ["amoswap.w_3"] = "c0002053FRr",
  ["amoadd.w_3"] = "c0003053FRr",
  ["amoxor.w_3"] = "c0004053FRr",
  ["amoor.w_3"] = "c0005053FRr",
  ["amoand.w_3"] = "c0006053FRr",
  ["amomin.w_3"] = "c0007053FRr",
  ["amomax.w_3"] = "c0008053FRr",
  ["amominu.w_3"] = "c0009053FRr",
  ["amomaxu.w_3"] = "c000a053FRr",

  -- RV32F
  ["flw_2"] = "00002007FL",
  ["fsw_2"] = "00002027gS",

  ["fmadd.s_4"]  = "00000043FGgH",
  ["fmsub.s_4"]  = "00000047FGgH",
  ["fnmsub.s_4"] = "0000004bFGgH",
  ["fnmadd.s_4"] = "0000004fFGgH",
  ["fmadd.s_5"]  = "00000043FGgHM",
  ["fmsub.s_5"]  = "00000047FGgHM",
  ["fnmsub.s_5"] = "0000004bFGgHM",
  ["fnmadd.s_5"] = "0000004fFGgHM",

  ["fadd.s_3"]  = "00000053FGg",
  ["fsub.s_3"]  = "08000053FGg",
  ["fmul.s_3"]  = "10000053FGg",
  ["fdiv.s_3"]  = "18000053FGg",
  ["fsqrt.s_2"] = "58000053FG",
  ["fadd.s_4"]  = "00000053FGgM",
  ["fsub.s_4"]  = "08000053FGgM",
  ["fmul.s_4"]  = "10000053FGgM",
  ["fdiv.s_4"]  = "18000053FGgM",
  ["fsqrt.s_3"] = "58000053FGM",

  ["fsgnj.s_3"]  = "20000053FGg",
  ["fsgnjn.s_3"] = "20001053FGg",
  ["fsgnjx.s_3"] = "20002053FGg",

  ["fmin.s_3"] = "28000053FGg",
  ["fmax.s_3"] = "28001053FGg",

  ["fcvt.w.s_2"]  = "c0000053DG",
  ["fcvt.wu.s_2"] = "c0100053DG",
  ["fcvt.w.s_3"]  = "c0000053DGM",
  ["fcvt.wu.s_3"] = "c0100053DGM",
  ["fmv.x.w_2"] = "e0000053DG",

  ["feq.s_3"] = "a0002053DGg",
  ["flt.s_3"] = "a0001053DGg",
  ["fle.s_3"] = "a0000053DGg",

  ["fclass.s_2"] = "e0001053DG",

  ["fcvt.s.w_2"]  = "d0000053FR",
  ["fcvt.s.wu_2"] = "d0100053FR",
  ["fcvt.s.w_3"]  = "d0000053FRM",
  ["fcvt.s.wu_3"] = "d0100053FRM",
  ["fmv.w.x_2"] = "f0000053FR",

  -- RV32D
  ["fld_2"] = "00003007FL",
  ["fsd_2"] = "00003027gS",
  
  ["fmadd.d_4"]  = "02000043FGgH",
  ["fmsub.d_4"]  = "02000047FGgH",
  ["fnmsub.d_4"] = "0200004bFGgH",
  ["fnmadd.d_4"] = "0200004fFGgH",
  ["fmadd.d_5"]  = "02000043FGgHM",
  ["fmsub.d_5"]  = "02000047FGgHM",
  ["fnmsub.d_5"] = "0200004bFGgHM",
  ["fnmadd.d_5"] = "0200004fFGgHM",

  ["fadd.d_3"]  = "02000053FGg",
  ["fsub.d_3"]  = "0a000053FGg",
  ["fmul.d_3"]  = "12000053FGg",
  ["fdiv.d_3"]  = "1a000053FGg",
  ["fsqrt.d_2"] = "5a000053FG",
  ["fadd.d_4"]  = "02000053FGgM",
  ["fsub.d_4"]  = "0a000053FGgM",
  ["fmul.d_4"]  = "12000053FGgM",
  ["fdiv.d_4"]  = "1a000053FGgM",
  ["fsqrt.d_3"] = "5a000053FGM",

  ["fsgnj.d_3"]  = "22000053FGg",
  ["fsgnjn.d_3"] = "22001053FGg",
  ["fsgnjx.d_3"] = "22002053FGg",
  ["fmin.d_3"] = "2a000053FGg",
  ["fmax.d_3"] = "2a001053FGg",
  ["fcvt.s.d_2"] = "40100053FG",
  ["fcvt.d.s_2"] = "42000053FG",
  ["feq.d_3"] = "a2002053DGg",
  ["flt.d_3"] = "a2001053DGg",
  ["fle.d_3"] = "a2000053DGg",
  ["fclass.d_2"] = "e2001053DG",
  ["fcvt.w.d_2"]  = "c2000053DG",
  ["fcvt.wu.d_2"] = "c2100053DG",
  ["fcvt.d.w_2"]  = "d2000053FR",
  ["fcvt.d.wu_2"] = "d2100053FR",
  ["fcvt.w.d_3"]  = "c2000053DGM",
  ["fcvt.wu.d_3"] = "c2100053DGM",
  ["fcvt.d.w_3"]  = "d2000053FRM",
  ["fcvt.d.wu_3"] = "d2100053FRM",

  ["fmv.d_2"] = "22000053FY",
  ["fneg.d_2"] = "22001053FY",
  ["fabs.d_2"] = "22002053FY",

}

local map_op_rv64imafd = {

  -- RV64I
  lwu_2 = "00006003DL",
  ld_2  = "00003003DL",

  sd_2 = "00003023rS",

  slli_3 = "00001013DRj",
  srli_3 = "00005013DRj",
  srai_3 = "40005013DRj",

  addiw_3 = "0000001bDRI",
  slliw_3 = "0000101bDRi",
  srliw_3 = "0000501bDRi",
  sraiw_3 = "4000501bDRi",

  addw_3 = "0000003bDRr",
  subw_3 = "4000003bDRr",
  sllw_3 = "0000103bDRr",
  srlw_3 = "0000503bDRr",
  sraw_3 = "4000503bDRr",

  negw_2 = "4000003bDr",
  ["sext.w_2"] = "0000001bDR",

  -- RV64M
  mulw_3  = "0200003bDRr",
  divw_3  = "0200403bDRr",
  divuw_3 = "0200503bDRr",
  remw_3  = "0200603bDRr",
  remuw_3 = "0200703bDRr",

  -- RV64A
  ["lr.d_2"] = "c2000053FR",
  ["sc.d_2"] = "c2001053FRr",
  ["amoswap.d_3"] = "c2002053FRr",
  ["amoadd.d_3"] = "c2003053FRr",
  ["amoxor.d_3"] = "c2004053FRr",
  ["amoor.d_3"] = "c2005053FRr",
  ["amoand.d_3"] = "c2006053FRr",
  ["amomin.d_3"] = "c2007053FRr",
  ["amomax.d_3"] = "c2008053FRr",
  ["amominu.d_3"] = "c2009053FRr",
  ["amomaxu.d_3"] = "c200a053FRr",

  -- RV64F
  ["fcvt.l.s_2"]  = "c0200053DG",
  ["fcvt.lu.s_2"] = "c0300053DG",
  ["fcvt.l.s_3"]  = "c0200053DGM",
  ["fcvt.lu.s_3"] = "c0300053DGM",
  ["fcvt.s.l_2"]  = "d0200053FR",
  ["fcvt.s.lu_2"] = "d0300053FR",
  ["fcvt.s.l_3"]  = "d0200053FRM",
  ["fcvt.s.lu_3"] = "d0300053FRM",

  -- RV64D
  ["fcvt.l.d_2"]  = "c2200053DG",
  ["fcvt.lu.d_2"] = "c2300053DG",
  ["fcvt.l.d_3"]  = "c2200053DGM",
  ["fcvt.lu.d_3"] = "c2300053DGM",
  ["fmv.x.d_2"]   = "e2000053DG",
  ["fcvt.d.l_2"]  = "d2200053FR",
  ["fcvt.d.lu_2"] = "d2300053FR",
  ["fcvt.d.l_3"]  = "d2200053FRM",
  ["fcvt.d.lu_3"] = "d2300053FRM",
  ["fmv.d.x_2"]   = "f2000053FR",

}

local map_op_zicsr = {
  csrrw_3 = "00001073DCR",
  csrrs_3 = "00002073DCR",
  csrrc_3 = "00003073DCR",
  csrrwi_3 = "00005073DCu",
  csrrsi_3 = "00006073DCu",
  csrrci_3 = "00007073DCu",

  -- pseudo-ops
  csrrw_2 = "00001073DC",
  csrrs_2 = "00002073CR",
  csrrc_2 = "00003073CR",
  csrrwi_2 = "00005073Cu",
  csrrsi_2 = "00006073Cu",
  csrrci_2 = "00007073Cu",

  rdinstret_1 = "C0202073D",
  rdcycle_1 = "C0002073D",
  rdtime_1 = "C0102073D",
  rdinstreth_1 = "C8202073D",
  rdcycleh_1 = "C8002073D",
  rdtimeh_1 = "C8102073D",

  frcsr_1 = "00302073D",
  fscsr_2 = "00301073DR",
  fscsr_1 = "00301073R",
  frrm_1 = "00202073D",
  fsrm_2 = "00201073DR",
  fsrm_1 = "00201073R",
  fsrmi_2 = "00205073Du",
  fsrmi_1 = "00205073u",
  frflags_1 = "00102073D",
  fsflags_2 = "00101073DR",
  fsflagsi_2 = "00105073Du",
  fsflagsi_1 = "00105073u",
}

local map_op_zifencei = {
  ["fence.i_3"] = "0000100fDRI",
}

local list_map_op_rv32 = { ['a'] = map_op_rv32imafd, ['b'] = map_op_zifencei, ['c'] = map_op_zicsr }
local list_map_op_rv64 = { ['a'] = map_op_rv32imafd, ['b'] = map_op_rv64imafd, ['c'] = map_op_zifencei, ['d'] = map_op_zicsr }

if riscv32 then for _, map in opairs(list_map_op_rv32) do
  for k, v in pairs(map) do map_op[k] = v end
  end
end
if riscv64 then for _, map in opairs(list_map_op_rv64) do
  for k, v in pairs(map) do map_op[k] = v end
  end
end

------------------------------------------------------------------------------

local function parse_gpr(expr)
  local tname, ovreg = match(expr, "^([%w_]+):(x[1-3]?[0-9])$")
  local tp = map_type[tname or expr]
  if tp then
    local reg = ovreg or tp.reg
    if not reg then
      werror("type `"..(tname or expr).."' needs a register override")
    end
    expr = reg
  end
  local r = match(expr, "^x([1-3]?[0-9])$")
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

local function parse_imm(imm, bits, shift, scale, signed, action)
  local n = tonumber(imm)
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
  elseif match(imm, "^[xf]([1-3]?[0-9])$") or
           match(imm, "^([%w_]+):([xf][1-3]?[0-9])$") then
    werror("expected immediate operand, got register")
  else
    waction(action or "IMM",
        (signed and 32768 or 0)+shl(scale, 10)+shl(bits, 5)+shift, imm)
    return 0
  end
end

local function parse_csr(expr)
  local r = match(expr, "^([1-4]?[0-9]?[0-9]?[0-9])$")
  if r then
    r = tonumber(r)
    if r <= 4095 then return r end
  end
  werror("bad register name `"..expr.."'")
end

local function parse_imms(imm)
  local n = tonumber(imm)
  if n then
    if n >= -2048 and n < 2048 then
      local imm5, imm7 = band(n, 0x1f), shr(band(n, 0xfe0), 5)
      return shl(imm5, 7) + shl(imm7, 25)
    end
    werror("out of range immediate `"..imm.."'")
  elseif match(imm, "^[xf]([1-3]?[0-9])$") or
         match(imm, "^([%w_]+):([xf][1-3]?[0-9])$") then
    werror("expected immediate operand, got register")
  else
    waction("IMMS", 0, imm); return 0
  end
end

local function parse_rm(mode)
  local rnd_mode = {
    rne = 0, rtz = 1, rdn = 2, rup = 3, rmm = 4, dyn = 7
  }
  local n = rnd_mode[mode]
  if n then return n
  else werror("bad rounding mode `"..mode.."'") end
end

local function parse_disp(disp, mode)
  local imm, reg = match(disp, "^(.*)%(([%w_:]+)%)$")
  if imm then
    local r = shl(parse_gpr(reg), 15)
    local extname = match(imm, "^extern%s+(%S+)$")
    if extname then
      waction("REL_EXT", map_extern[extname], nil, 1)
      return r
    else
      if mode == "load" then
        return r + parse_imm(imm, 12, 20, 0, true)
      elseif mode == "store" then
        return r + parse_imms(imm)
      else
        werror("bad displacement mode '"..mode.."'")
      end
    end
  end
  local reg, tailr = match(disp, "^([%w_:]+)%s*(.*)$")
  if reg and tailr ~= "" then
    local r, tp = parse_gpr(reg)
    if tp then
      if mode == "load" then
          waction("IMM", 32768+12*32+20, format(tp.ctypefmt, tailr))
      elseif mode == "store" then
          waction("IMMS", 0, format(tp.ctypefmt, tailr))
      else
        werror("bad displacement mode '"..mode.."'")
      end
      return shl(r, 15)
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
map_op[".template__"] = function(params, template, nparams)
  if not params then return sub(template, 9) end
  local op = tonumber(sub(template, 1, 8), 16)
  local n = 1

  -- Limit number of section buffer positions used by a single dasm_put().
  -- A single opcode needs a maximum of 2 positions (ins/ext).
  if secpos+2 > maxsecpos then wflush() end
  local pos = wpos()

  -- Process each character.
  for p in gmatch(sub(template, 9), ".") do
    if p == "D" then  -- gpr rd
      op = op + shl(parse_gpr(params[n]), 7); n = n + 1
    elseif p == "R" then  -- gpr rs1
      op = op + shl(parse_gpr(params[n]), 15); n = n + 1
    elseif p == "r" then  -- gpr rs2
      op = op + shl(parse_gpr(params[n]), 20); n = n + 1
    elseif p == "F" then  -- fpr rd
      op = op + shl(parse_fpr(params[n]), 7); n = n + 1
    elseif p == "G" then  -- fpr rs1
      op = op + shl(parse_fpr(params[n]), 15); n = n + 1
    elseif p == "g" then  -- fpr rs2
      op = op + shl(parse_fpr(params[n]), 20); n = n + 1
    elseif p == "H" then  -- fpr rs3
      op = op + shl(parse_fpr(params[n]), 27); n = n + 1
    elseif p == "C" then  -- csr
      op = op + shl(parse_csr(params[n]), 20); n = n + 1
    elseif p == "M" then  -- fpr rounding mode
      op = op + shl(parse_rm(params[n]), 12); n = n + 1
    elseif p == "Y" then  -- fpr psuedo-op
      local r = parse_fpr(params[n])
      op = op + shl(r, 15) + shl(r, 20); n = n + 1
    elseif p == "I" then  -- I-type imm12
      op = op + parse_imm(params[n], 12, 20, 0, true); n = n + 1
    elseif p == "i" then  -- I-type shamt5
      op = op + parse_imm(params[n], 5, 20, 0, false); n = n + 1
    elseif p == "j" then  -- I-type shamt6
      op = op + parse_imm(params[n], 6, 20, 0, false); n = n + 1
    elseif p == "u" then  -- I-type uimm
      op = op + parse_imm(params[n], 5, 15, 0, false); n = n + 1
    elseif p == "U" then  -- U-type imm20
      op = op + parse_imm(params[n], 20, 12, 0, false); n = n + 1
    elseif p == "L" then  -- load
      op = op + parse_disp(params[n], "load"); n = n + 1
    elseif p == "S" then  -- store
      op = op + parse_disp(params[n], "store"); n = n + 1
    elseif p == "B" or p == "J" then  -- control flow
      local mode, m, s = parse_label(params[n], false)
      if p == "B" then m = m + 2048 end
      waction("REL_"..mode, m, s, 1); n = n + 1
    elseif p == "A" then  -- AUIPC
      local mode, m, s = parse_label(params[n], false)
      waction("REL_"..mode, m, s, 1); n = n + 1
    else
      assert(false)
    end
  end
  wputpos(pos, op)
end

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

