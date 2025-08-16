------------------------------------------------------------------------------
-- LuaJIT RISC-V disassembler module.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
--
-- Contributed by Milos Poletanovic from Syrmia.com.
-- Contributed by gns from PLCT Lab, ISCAS.
------------------------------------------------------------------------------
-- This is a helper module used by the LuaJIT machine code dumper module.
--
-- It disassembles most standard RISC-V instructions.
-- Mode is little-endian
------------------------------------------------------------------------------

local type = type
local byte, format = string.byte, string.format
local match, gmatch = string.match, string.gmatch
local concat = table.concat
local bit = require("bit")
local band, bor, tohex = bit.band, bit.bor, bit.tohex
local lshift, rshift, arshift = bit.lshift, bit.rshift, bit.arshift
local jit = require("jit")

local jstat = { jit.status() }
local function is_opt_enabled(opt)
  for _, v in ipairs(jstat) do
    if v == opt then
      return true
    end
  end
  return false
end
local xthead = is_opt_enabled("XThead")

------------------------------------------------------------------------------
-- Opcode maps
------------------------------------------------------------------------------

--RVC32 extension

local map_quad0 = {
  shift = 13, mask = 7,
  [0] = "c.addi4spnZW", "c.fldNMh", "c.lwZMn", "c.flwNMn",
  false, "c.fsdNMh", "c.swZMn", "c.fswNMn"
}

local map_sub2quad1 = {
  shift = 5, mask = 3,
  [0] = "c.subMZ", "c.xorMZ", "c.orMZ", "c.andMZ"
}

local map_sub1quad1 = {
  shift = 10, mask = 3,
  [0] = "c.srliM1", "c.sraiM1", "c.andiMx", map_sub2quad1
}

local map_quad1 = {
  shift = 13, mask = 7,
  [0] = {
    shift = 7, mask = 31,
    [0] = "c.nop", _ = "c.addiDx"
  },
  [1] = "c.jalT", [2] = "c.liDx",
  [3] = {
    shift = 7, mask = 31,
    [0] = "c.luiDK", [1] = "c.luiDK", [2] = "c.addi16spX",
    _ = "c.luiDK"
  },
  [4] = map_sub1quad1, [5] = "c.jT", [6] = "c.beqzMq", [7] = "c.bnezMq"
}

local map_sub1quad2 = {
  shift = 12, mask = 1,
  [0] = {
    shift = 2, mask = 31,
    [0] = "c.jrD", _ = "c.mvDE"
  },
  [1] = {
    shift = 2, mask = 31,
    [0] = {
      shift = 7, mask = 31,
      [0] = "c.ebreak", _ = "c.jalrD"
    },
   _ = "c.addDE"
  }
}

local map_quad2 = {
  shift = 13, mask = 7,
  [0] = "c.slliD1", [1] = "c.fldspFQ",[2] = "c.lwspDY", [3] = "c.flwspFY",
  [4] = map_sub1quad2, [5] = "c.fsdspVt", [6] = "c.swspEu", [7] = "c.fswspVu"
}

local map_compr = {
  [0] = map_quad0, map_quad1, map_quad2
}

--RV32M
local map_mext = {
  shift = 12, mask = 7,
  [0] = "mulDRr", "mulhDRr", "mulhsuDRr", "mulhuDRr",
  "divDRr", "divuDRr", "remDRr", "remuDRr"
}

--RV64M
local map_mext64 = {
  shift = 12, mask = 7,
  [0] = "mulwDRr", [4] = "divwDRr", [5] = "divuwDRr", [6] = "remwDRr",
  [7] = "remuwDRr"
}

--RV32F, RV64F, RV32D, RV64D
local map_fload = {
  shift = 12, mask = 7,
  [2] = "flwFL", [3] = "fldFL"
}

local map_fstore = {
  shift = 12, mask = 7,
  [2] = "fswSg", [3] = "fsdSg"
}

local map_fmadd = {
  shift = 25, mask = 3,
  [0] = "fmadd.sFGgH", "fmadd.dFGgH"
}

local map_fmsub = {
  shift = 25, mask = 3,
  [0] = "fmsub.sFGgH", "fmsub.dFGgH"
}

local map_fnmsub = {
  shift = 25, mask = 3,
  [0] = "fnmsub.sFGgH", "fnmsub.dFGgH"
}

local map_fnmadd = {
  shift = 25, mask = 3,
  [0] = "fnmadd.sFGgH", "fnmadd.dFGgH"
}

local map_fsgnjs = {
  shift = 12, mask = 7,
  [0] = "fsgnj.s|fmv.sFGg6", "fsgnjn.s|fneg.sFGg6", "fsgnjx.s|fabs.sFGg6"
}

local map_fsgnjd = {
  shift = 12, mask = 7,
  [0] = "fsgnj.d|fmv.dFGg6", "fsgnjn.d|fneg.dFGg6", "fsgnjx.d|fabs.dFGg6"
}

local map_fms = {
  shift = 12, mask = 7,
  [0] = "fmin.sFGg", "fmax.sFGg"
}

local map_fmd = {
  shift = 12, mask = 7,
  [0] = "fmin.dFGg", "fmax.dFGg"
}

local map_fcomps = {
  shift = 12, mask = 7,
  [0] = "fle.sDGg", "flt.sDGg", "feq.sDGg"
}

local map_fcompd = {
  shift = 12, mask = 7,
  [0] = "fle.dDGg", "flt.dDGg", "feq.dDGg"
}

local map_fcvtwls = {
  shift = 20, mask = 31,
  [0] = "fcvt.w.sDG", "fcvt.wu.sDG", "fcvt.l.sDG", "fcvt.lu.sDG"
}

local map_fcvtwld = {
  shift = 20, mask = 31,
  [0] = "fcvt.w.dDG", "fcvt.wu.dDG", "fcvt.l.dDG", "fcvt.lu.dDG"
}

local map_fcvts = {
  shift = 20, mask = 31,
  [0] = "fcvt.s.wFR", "fcvt.s.wuFR", "fcvt.s.lFR", "fcvt.s.luFR"
}

local map_fcvtd = {
  shift = 20, mask = 31,
  [0] = "fcvt.d.wFR", "fcvt.d.wuFR", "fcvt.d.lFR", "fcvt.d.luFR"
}

local map_fext = {
  shift = 25, mask = 127,
  [0] = "fadd.sFGg", [1] = "fadd.dFGg", [4] = "fsub.sFGg", [5] = "fsub.dFGg",
  [8] = "fmul.sFGg", [9] = "fmul.dFGg", [12] = "fdiv.sFGg", [13] = "fdiv.dFGg",
  [16] = map_fsgnjs, [17] = map_fsgnjd, [20] = map_fms, [21] = map_fmd,
  [32] = "fcvt.s.dFG", [33] = "fcvt.d.sFG",[44] = "fsqrt.sFG", [45] = "fsqrt.dFG",
  [80] = map_fcomps, [81] = map_fcompd, [96] = map_fcvtwls, [97] = map_fcvtwld,
  [104] = map_fcvts, [105] = map_fcvtd,
  [112] = {
    shift = 12, mask = 7,
    [0] = "fmv.x.wDG", "fclass.sDG"
  },
  [113] = {
  shift = 12, mask = 7,
    [0] = "fmv.x.dDG", "fclass.dDG"
  },
  [120] = "fmv.w.xFR", [121] = "fmv.d.xFR"
}

--RV32A, RV64A
local map_aext = {
  shift = 27, mask = 31,
  [0] = {
    shift = 12, mask = 7,
    [2] = "amoadd.wDrO", [3] = "amoadd.dDrO"
  },
  {
    shift = 12, mask = 7,
    [2] = "amoswap.wDrO", [3] = "amoswap.dDrO"
  },
  {
    shift = 12, mask = 7,
    [2] = "lr.wDO", [3] = "lr.dDO"
  },
  {
    shift = 12, mask = 7,
    [2] = "sc.wDrO", [3] = "sc.dDrO"
  },
  {
    shift = 12, mask = 7,
    [2] = "amoxor.wDrO", [3] = "amoxor.dDrO"
  },
  [8] = {
    shift = 12, mask = 7,
    [2] = "amoor.wDrO", [3] = "amoor.dDrO"
  },
  [12] = {
    shift = 12, mask = 7,
    [2] = "amoand.wDrO", [3] = "amoand.dDrO"
  },
  [16] = {
    shift = 12, mask = 7,
    [2] = "amomin.wDrO", [3] = "amomin.dDrO"
  },
  [20] = {
    shift = 12, mask = 7,
    [2] = "amomax.wDrO", [3] = "amomax.dDrO"
  },
  [24] = {
    shift = 12, mask = 7,
    [2] = "amominu.wDrO", [3] = "amominu.dDrO"
  },
  [28] = {
   shift = 12, mask = 7,
   [2] = "amomaxu.wDrO", [3] = "amomaxu.dDrO"
  },
}

-- RV32I, RV64I
local map_load = {
  shift = 12, mask = 7,
  [0] = "lbDL", "lhDL", "lwDL", "ldDL",
  "lbuDL", "lhuDL", "lwuDL"
}

local map_opimm = {
  shift = 12, mask = 7,
  [0] = {
    shift = 7, mask = 0x1ffffff,
    [0] = "nop", _ = "addi|li|mvDR0I2"
  },
  {
    shift = 25, mask = 127,
    [48] = {
      shift = 20, mask = 31,
      [4] = "sext.bDR", [5] = "sext.hDR"
    },
    _ = "slliDRi",
  }, "sltiDRI", "sltiu|seqzDRI5",
  "xori|notDRI4",
  {
    shift = 26, mask = 63,
    [0] = "srliDRi", [16] = "sraiDRi", [24] = "roriDRi",
    [26] = {
      shift = 20, mask = 63,
      [56] = "rev8DR"
    }
  },
  "oriDRI", "andiDRI"
}

local map_branch = {
  shift = 12, mask = 7,
  [0] = "beq|beqzRr0B", "bne|bnezRr0B" , false, false,
  "blt|bgtz|bltzR0r2B", "bge|blez|bgezR0r2B", "bltuRrB", "bgeuRrB"
}

local map_store = {
  shift = 12, mask = 7,
  [0] = "sbSr", "shSr", "swSr", "sdSr"
}

local map_op = {
  shift = 25, mask = 127,
  [0] = {
    shift = 12, mask = 7,
    [0] = "addDRr", "sllDRr", "slt|sgtz|sltzDR0r2", "sltu|snezDR0r",
    "xorDRr", "srlDRr", "orDRr", "andDRr"
  },
  [1] = map_mext,
  [4] = {

  },
  [5] = { -- Zbb
    shift = 12, mask = 7,
    [4] = "minDRr", [5] = "minuDRr", [6] = "maxDRr", [7] = "maxuDRr"
  },
  [7] = { -- Zicond
    shift = 12, mask = 7,
    [5] = "czero.eqzDRr", [7] = "czero.nezDRr"
  },
  [16] = { -- Zba
    shift = 12, mask = 7,
    [2] = "sh1addDRr", [4] = "sh2addDRr", [6] = "sh3addDRr"
  },
  [32] = { -- Zbb
    shift = 12, mask = 7,
    [0] = "sub|negDR0r", [4] = "xnorDRr", [5] = "sraDRr", [6] = "ornDRr", [7] = "andnDRr"
  },
  [48] = { -- Zbb
    shift = 12, mask = 7,
    [1] = "rolDRr", [5] = "rorDRr"
  }
}

--- 64I
local map_opimm32 = {
  shift = 12, mask = 7,
  [0] = "addiw|sext.wDRI0", "slliwDRi",
  [2] = { -- Zba
    shift = 25, mask = 127,
    [1] = "slli.uwDRi"
  },
  [5] = { -- 64I
    shift = 25, mask = 127,
    [0] = "srliwDRi", [32] = "sraiwDRi", [48] = "roriwDRi"
  },
  [48] = { -- Zbb
    shift = 25, mask = 127,
    [5] = "roriwDRi"
  }
}

local map_op32 = {
  shift = 25, mask = 127,
  [0] = { -- 64I
    shift = 12, mask = 7,
    [0] = "addwDRr", [1] = "sllwDRr", [5] = "srlwDRr"
  },
  [1] = map_mext64,
  [4] = { -- Zba & Zbb
    shift = 12, mask = 7,
    [0] = "add.uw|zext.w|DRr0", [4] = "zext.hDRr"
  },
  [16] = { -- Zba
    shift = 12, mask = 7,
    [2] = "sh1add.uw", [4] = "sh2add.uw", [6] = "sh3add.uw"
  },
  [32] = { -- 64I
    shift = 12, mask = 7,
    [0] = "subw|negwDR0r", [5] = "srawDRr"
  },
  [48] = { -- Zbb
    shift = 12, mask = 7,
    [1] = "rolwDRr", [5] = "rorwDRr"
  }
}

local map_ecabre = {
  shift = 12, mask = 7,
  [0] = {
   shift = 20, mask = 4095,
   [0] = "ecall", "ebreak"
  }
}

local map_fence = {
  shift = 12, mask = 1,
  [0] = "fence", --"fence.i" ZIFENCEI EXTENSION
}

local map_jalr = {
  shift = 7, mask = 0x1ffffff,
  _ = "jalr|jrDRI7", [256] = "ret"
}

local map_xthead_custom0 = {
  shift = 12, mask = 7,
  [1] = { -- Arithmetic
    shift = 27, mask = 31,
    [0] = "th.addslDRrv",
    [2] = {
      shift = 26, mask = 63,
      [4] = "th.srriDRi",
      [5] = {
        shift = 25, mask = 127,
        [10] = "th.srriwDRi"
      }
    },
    [4] = { -- XTheadMac
      shift = 25, mask = 3,
      [0] = "th.mulaDRr", "th.mulsDRr", "th.mulawDRr", "th.mulswDRr"
    },
    [5] = { -- XTheadMac
      shift = 25, mask = 3,
      [0] = "th.mulahDRr", "th.mulshDRr"
    },
    [8] = { -- XTheadCondMov
      shift = 25, mask = 3,
      [0] = "th.mveqzDRr", "th.mvnezDRr"
    },
    [16] = { -- XTheadBb
      shift = 20, mask = 31,
      [0] = {
        shift = 25, mask = 3,
        [0] = "th.tstnbzDRi", "th.revDR", "th.ff0DR", "th.ff1DR"
      }
    },
    [17] = { -- XTheadBb
      shift = 26, mask = 1,
      [0] = "th.tstDRi"
    },
    [18] = { -- XTheadBb
      shift = 20, mask = 31,
      [0] = {
        shift = 25, mask = 3,
        [0] = "th.revwDR"
      }
    }
  },
  [2] = "th.extDRji", [3] = "th.extuDRji",
  { -- MemLoad
    shift = 29, mask = 7,
    [7] = { -- XTheadMemPair
      shift = 25, mask = 3,
      [0] = "th.lwdDrP", [2] = "th.lwudDrP", "th.lddDrP"
    }
  },
  { -- MemStore
    shift = 29, mask = 7,
    [7] = { -- XTheadMemPair
      shift = 25, mask = 3,
      [0] = "th.swdDrP", [3] = "th.sddDrP"
    }
  }
}

local map_custom0 = xthead and map_xthead_custom0 or nil

local map_pri = {
  [3] = map_load, [7] = map_fload, [11] = map_custom0, [15] = map_fence, [19] = map_opimm,
  [23] = "auipcDA", [27] = map_opimm32,
  [35] = map_store, [39] = map_fstore, [47] = map_aext, [51] = map_op,
  [55] = "luiDU", [59] = map_op32, [67] = map_fmadd, [71] = map_fmsub,
  [75] = map_fnmsub, [99] = map_branch, [79] = map_fnmadd, [83] = map_fext,
  [103] = map_jalr, [111] = "jal|j|D0J", [115] = map_ecabre
}

------------------------------------------------------------------------------

local map_gpr = {
  [0] = "zero", "ra", "sp", "gp", "tp", "x5", "x6", "x7",
  "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
  "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
  "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31",
}

local map_fgpr = {
  [0] = "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
  "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
  "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
  "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
}

------------------------------------------------------------------------------

-- Output a nicely formatted line with an opcode and operands.
local function putop(ctx, text, operands)
  local pos = ctx.pos
	local extra = ""
  if ctx.rel then
    local sym = ctx.symtab[ctx.rel]
    if sym then extra = "\t->"..sym end
  end
  if ctx.hexdump > 0 then
    ctx.out:write((format("%08x  %s  %-7s %s%s\n",
    ctx.addr+pos, tohex(ctx.op), text, concat(operands, ","), extra)))
  else
    ctx.out(format("%08x  %-7s %s%s\n",
    ctx.addr+pos, text, concat(operands, ", "), extra))
  end
  local pos = ctx.pos
  local first_byte = byte(ctx.code, ctx.pos+1)
  --Examine if the next instruction is 16-bits or 32-bits
  if(band(first_byte, 3) < 3) then
    ctx.pos = pos + 2
  else
    ctx.pos = pos + 4
  end
end

-- Fallback for unknown opcodes.
local function unknown(ctx)
  return putop(ctx, ".long", { "0x"..tohex(ctx.op) })
end

local function get_le(ctx)
  local pos = ctx.pos
  --Examine if the next instruction is 16-bits or 32-bits
  local first_byte = byte(ctx.code, pos+1)
  if(band(first_byte, 3) < 3) then --checking first two bits of opcode
    local b0, b1 = byte(ctx.code, pos+1, pos+2)
    return bor(lshift(b1, 8), b0)
  else
    local b0, b1, b2, b3 = byte(ctx.code, pos+1, pos+4)
    return bor(lshift(b3, 24), lshift(b2, 16), lshift(b1, 8), b0)
  end
end

local function parse_W(opcode)
  local part1 = band(rshift(opcode, 7), 15) --9:6
  local part2 = band(rshift(opcode, 11), 3) --5:4
  local part3 = band(rshift(opcode, 5), 1)--3
  local part4 = band(rshift(opcode, 6), 1)--2
  return bor(lshift(0, 31), lshift(part1, 6) , lshift(part2, 4),
             lshift(part3, 3), lshift(part4, 2))
end

local function parse_x(opcode)
  local part1 = band(rshift(opcode, 12), 1) --5
  local part2 = band(rshift(opcode, 2), 31) --4:0
  if(part1 == 1) then
    return bor(lshift(1, 31), lshift(0x1ffffff, 6), lshift(part1, 5), part2)
  else
    return bor(lshift(0, 31), lshift(part1, 5), part2)
  end
end

local function parse_X(opcode)
  local part1 = band(rshift(opcode, 12), 1) --12
  local part2 = band(rshift(opcode, 3), 3) --8:7
  local part3 = band(rshift(opcode, 5), 1) --6
  local part4 = band(rshift(opcode, 2), 1) --5
  local part5 = band(rshift(opcode, 6), 1) --4
  if(part1 == 1) then
    return bor(lshift(1, 31), lshift(0x3fffff, 9), lshift(part2, 7),
               lshift(part3, 6), lshift(part4, 5), lshift(part5, 4))
  else
    return bor(lshift(0, 31), lshift(part2, 7), lshift(part3, 6),
               lshift(part4, 5), lshift(part5, 4))
  end
end

local function parse_S(opcode)
  local part1 = band(rshift(opcode, 25), 127) --11:5
  local sign = band(rshift(part1, 6), 1)
  local part2 = band(rshift(opcode, 7), 31) --4:0
  if (sign == 1) then
    return bor(lshift(1, 31), lshift(0x7ffff, 12), lshift(part1, 5), part2)
  else
    return bor(lshift(0, 31), lshift(part1, 5), part2)
  end
end

local function parse_B(opcode)
  local part1 = band(rshift(opcode, 7), 1) --11
  local part2 = band(rshift(opcode, 25), 63) --10:5
  local part3 = band(rshift(opcode, 8), 15) -- 4 : 1
  if (part1 == 1) then
    return bor(lshift(1, 31), lshift(0x7ffff, 12), lshift(part1, 11),
               lshift(part2, 5), lshift(part3, 1), 0)
  else
    return bor(lshift(0, 31), lshift(part1, 11), lshift(part2, 5),
               lshift(part3, 1), 0)
  end
end

local function parse_q(opcode)
  local part1 = band(rshift(opcode, 12), 1) --8
  local part2 = band(rshift(opcode, 5), 3) --7:6
  local part3 = band(rshift(opcode, 2), 1) --5
  local part4 = band(rshift(opcode, 10), 3) --4:3
  local part5 = band(rshift(opcode, 3), 3) --2:1
  if(part1 == 1) then
    return bor(lshift(1, 31), lshift(0x7fffff, 8), lshift(part2, 6),
               lshift(part3, 5), lshift(part4, 3), lshift(part5, 1))
  else
    return bor(lshift(0, 31), lshift(part2, 6), lshift(part3, 5),
               lshift(part4, 3), lshift(part5, 1))
  end
end

local function parse_J(opcode)
  local part1 = band(rshift(opcode, 31), 1) --20
  local part2 = band(rshift(opcode, 12), 255) -- 19:12
  local part3 = band(rshift(opcode, 20), 1) --11
  local part4 = band(rshift(opcode, 21), 1023) --10:1
  if(part1 == 1) then
    return bor(lshift(1, 31), lshift(0x7ff, 20), lshift(part2, 12),
               lshift(part3, 11), lshift(part4, 1))
  else
    return bor(lshift(0, 31), lshift(0, 20), lshift(part2, 12),
               lshift(part3, 11), lshift(part4, 1))
  end
end

local function parse_T(opcode)
  local part1 = band(rshift(opcode, 12), 1) --11
  local part2 = band(rshift(opcode, 8), 1) --10
  local part3 = band(rshift(opcode, 9), 3)--9:8
  local part4 = band(rshift(opcode, 6), 1) --7
  local part5 = band(rshift(opcode, 7), 1) -- 6
  local part6 = band(rshift(opcode, 2), 1) --5
  local part7 = band(rshift(opcode, 11), 1) --4
  local part8 = band(rshift(opcode, 3), 7) --3:1
  if(part1 == 1) then
    return bor(lshift(1, 31), lshift(0x7ffff, 12), lshift(part1, 11),
               lshift(part2, 10), lshift(part3, 8), lshift(part4, 7),
               lshift(part5, 6), lshift(part6, 5), lshift(part7, 4),
               lshift(part8, 1))
  else
    return bor(lshift(0, 31), lshift(part1, 11), lshift(part2, 10),
               lshift(part3, 8), lshift(part4, 7), lshift(part5, 6),
               lshift(part6, 5), lshift(part7, 4), lshift(part8, 1))
  end
end

local function parse_K(opcode)
  local part1 = band(rshift(opcode, 12), 1) --5 17
  local part2 = band(rshift(opcode, 2), 31) --4:0  16:12
  if(part1 == 1) then
    return bor(lshift(0, 31), lshift(0x7fff, 5), part2)
  else
    return bor(lshift(0, 31), lshift(part1, 5), part2)
  end
end

-- Disassemble a single instruction.
local function disass_ins(ctx)
  local op = ctx:get()
  local operands = {}
  local last = nil
  ctx.op = op
  ctx.rel =nil

  local opat = 0
  --for compressed instructions
  if(band(op, 3) < 3) then
    opat = ctx.map_compr[band(op, 3)]
    while type(opat) ~= "string" do
      if not opat then return unknown(ctx) end
      local test = band(rshift(op, opat.shift), opat.mask)
      opat = opat[band(rshift(op, opat.shift), opat.mask)] or opat._
    end
  else
    opat = ctx.map_pri[band(op,127)]
    while type(opat) ~= "string" do
      if not opat then return unknown(ctx) end
      opat = opat[band(rshift(op, opat.shift), opat.mask)] or opat._
    end
  end
  local name, pat = match(opat, "^([a-z0-9_.]*)(.*)")
  local altname, pat2 = match(pat, "|([a-z0-9_.|]*)(.*)")
  local a1, a2 = 0
  if altname then
   pat = pat2
  end

  local alias_done = false --variable for the case of 2 pseudoinstructions, if both parameters are x0, 0

  for p in gmatch(pat, ".") do
    local x = nil
    if p == "D" then
      x = map_gpr[band(rshift(op, 7), 31)]
    elseif p == "F" then
      x = map_fgpr[band(rshift(op, 7), 31)]
    elseif p == "R" then
      x = map_gpr[band(rshift(op, 15), 31)]
    elseif p == "G" then
      x = map_fgpr[band(rshift(op, 15), 31)]
    elseif p == "r" then
      x = map_gpr[band(rshift(op, 20), 31)]
      if(name == "sb" or name == "sh" or name == "sw" or name == "sd") then
        local temp = last --because of the diffrent order of the characters
        operands[#operands] = x
        x = temp
      end
    elseif p == "g" then
      x = map_fgpr[band(rshift(op, 20), 31)]
     if(name == "fsw" or name == "fsd") then
        local temp = last
        operands[#operands] = x
        x = temp
     end
    elseif p == "Z" then
      x = map_gpr[8 + band(rshift(op, 2), 7)]
    elseif p == "N" then
      x = map_fgpr[8 + band(rshift(op, 2), 7)]
    elseif p == "M" then
      x = map_gpr[8 + band(rshift(op, 7), 7)]
    elseif p == "E" then
      x = map_gpr[band(rshift(op, 2), 31)]
    elseif p == "W" then
      local uimm = parse_W(op)
      x = format("%s,%d", "sp", uimm)
    elseif p == "x" then
      x = parse_x(op)
    elseif p == "h" then
      local part1 = band(rshift(op, 5), 3) --7:6
      local part2 = band(rshift(op, 10), 7) --5:3
      local uimm = bor(lshift(0, 31), lshift(part1, 6) , lshift(part2, 3))
      operands[#operands] = format("%d(%s)", uimm, last)
    elseif p == "X" then
      local imm = parse_X(op)
      x = format("%s,%d", "sp", imm)
    elseif p == "O" then
      x = format("(%s)", map_gpr[band(rshift(op, 15), 31)])
    elseif p == "H" then
      x = map_fgpr[band(rshift(op, 27), 31)]
    elseif p == "L" then
      local register = map_gpr[band(rshift(op, 15), 31)]
      local disp = arshift(op, 20)
      x = format("%d(%s)", disp, register)
    elseif p == "P" then -- XTheadMemPair
      local register = map_gpr[band(rshift(op, 15), 31)]
      local disp = band(arshift(op, 25), 3)
      local isword = bxor(band(arshift(op, 26), 1), 1)
      x = format("(%s), %d, %d", register, disp, isword and 3 or 4)
    elseif p == "I" then
      x = arshift(op, 20)
      --different for jalr
      if(name == "jalr") then
        local reg = map_gpr[band(rshift(op, 15), 31)]
        if(ctx.reltab[reg] == nil) then
          operands[#operands] = format("%d(%s)", x, last)
        else
          local target = ctx.reltab[reg] + x
          operands[#operands] = format("%d(%s) #0x%08x", x, last, target)
          ctx.rel = target
          ctx.reltab[reg] = nil --assume no reuses of the register
        end
        x = nil --not to add additional operand
      end
    elseif p == "i" then
      --both for RV32I AND RV64I
      local value = band(arshift(op, 20), 63)
      x = string.format("%d", value)
    elseif p == "j" then -- XThead imm1[31..26]
      local value = band(rshift(op, 26), 63)
      x = string.format("%d", value)
    elseif p == "v" then --XThead imm[2][26..25]
      local value = band(rshift(op, 25), 3)
      x = string.format("%d", value)
    elseif p == "S" then
      local register = map_gpr[band(rshift(op, 15), 31)] --register
      local imm = parse_S(op)
      x = format("%d(%s)", imm, register)
    elseif p == "n" then
      local part1 = band(rshift(op, 5), 1) --6
      local part2 = band(rshift(op, 10), 7) --5:3
      local part3 = band(rshift(op, 6), 1) --2
      local uimm = bor(lshift(0, 31), lshift(part1, 6), lshift(part2, 3),
                       lshift(part3, 2))
      operands[#operands] = format("%d(%s)", uimm, last)
    elseif p == "A" then
      local value, dest = band(rshift(op, 12), 0xfffff), map_gpr[band(rshift(op, 7), 31)]
      ctx.reltab[dest] = ctx.addr + ctx.pos + lshift(value, 12)
      x = format("0x%x", value)
    elseif p == "B" then
      x = ctx.addr + ctx.pos + parse_B(op)
      ctx.rel = x
      x = format("0x%08x", x)
    elseif p == "U" then
      local value = band(rshift(op, 12), 0xfffff)
      x = string.format("0x%x", value)
    elseif p == "Q" then
      local part1 = band(rshift(op, 2), 7) --8:6
      local part2 = band(rshift(op, 12), 1) --5
      local part3 = band(rshift(op, 5), 3) --4:3
      local uimm = bor(lshift(0, 31), lshift(part1, 6), lshift(part2, 5),
                       lshift(part3, 3))
      x = format("%d(%s)", uimm, "sp")
   elseif p == "q" then
      x = ctx.addr + ctx.pos + parse_q(op)
      ctx.rel = x
      x = format("0x%08x", x)
    elseif p == "J" then
      x = ctx.addr + ctx.pos + parse_J(op)
      ctx.rel = x
      x = format("0x%08x", x)
    elseif p == "K" then
      local value = parse_K(op)
      x = string.format("0x%x", value)
    elseif p == "Y" then
      local part1 = band(rshift(op, 2), 3) --7:6
      local part2 = band(rshift(op, 12), 1) --5
      local part3 = band(rshift(op, 4), 7) --4:2
      local uimm = bor(lshift(0, 31), lshift(part1, 6), lshift(part2, 5),
                       lshift(part3, 2))
      x = format("%d(%s)", uimm, "sp")
    elseif p == "1" then
      local part1 = band(rshift(op, 12), 1) --5
      local part2 = band(rshift(op, 2), 31) --4:0
      local uimm = bor(lshift(0, 31), lshift(part1, 5), part2)
      x = string.format("0x%x", uimm)
    elseif p == "T" then
      x = ctx.addr + ctx.pos + parse_T(op)
      ctx.rel = x
      x = format("0x%08x", x)
    elseif p == "t" then
      local part1 = band(rshift(op, 7), 7) --8:6
      local part2 = band(rshift(op, 10), 7) --5:3
      local uimm = bor(lshift(0, 31), lshift(part1, 6), lshift(part2, 3))
      x = format("%d(%s)", uimm, "sp")
    elseif p == "u" then
      local part1 = band(rshift(op, 7), 3) --7:6
      local part2 = band(rshift(op, 9), 15) --5:2
      local uimm = bor(lshift(0, 31), lshift(part1, 6), lshift(part2, 2))
      x = format("%d(%s)", uimm, "sp")
    elseif p == "V" then
      x = map_fgpr[band(rshift(op, 2), 31)]
    elseif p == "0" then --PSEUDOINSTRUCTIONS
      if (last == "zero" or last == 0) then
        local n = #operands
        operands[n] = nil
        last = operands[n-1]
        local a1, a2 = match(altname, "([^|]*)|(.*)")
        if a1 then name, altname = a1, a2
        else name = altname end
        alias_done = true
      end
    elseif (p == "4") then
      if(last == -1) then
        name = altname
        operands[#operands] = nil
      end
    elseif (p == "5") then
      if(last == 1) then
        name = altname
        operands[#operands] = nil
      end
    elseif (p == "6") then
      if(last == operands[#operands - 1]) then
        name = altname
        operands[#operands] = nil
      end
    elseif (p == "7") then --jalr rs
      local value = string.sub(operands[#operands], 1, 1)
      local reg = string.sub(operands[#operands], 3, #(operands[#operands]) - 1)
      if(value == "0" and
         (operands[#operands - 1] == "ra" or operands[#operands - 1] == "zero")) then
        if(operands[#operands - 1] == "zero") then
          name = altname
        end
        operands[#operands] = nil
        operands[#operands] = reg
      end
    elseif (p == "2" and alias_done == false) then
      if (last == "zero" or last == 0) then
        local a1, a2 = match(altname, "([^|]*)|(.*)")
        name = a2
        operands[#operands] = nil
      end
    end
    if x then operands[#operands+1] = x; last = x end
  end
  return putop(ctx, name, operands)
end

------------------------------------------------------------------------------

-- Disassemble a block of code.
local function disass_block(ctx, ofs, len)
  if not ofs then
    ofs = 0
  end
  local stop = len and ofs+len or #ctx.code
  --instructions can be both 32 and 16 bits
  stop = stop - stop % 2
  ctx.pos = ofs - ofs % 2
  ctx.rel = nil
  while ctx.pos < stop do disass_ins(ctx) end
end

-- Extended API: create a disassembler context. Then call ctx:disass(ofs, len).
local function create(code, addr, out)
  local ctx = {}
  ctx.code = code
  ctx.addr = addr or 0
  ctx.out = out or io.write
  ctx.symtab = {}
  ctx.disass = disass_block
  ctx.hexdump = 8
  ctx.get = get_le
  ctx.map_pri = map_pri
  ctx.map_compr = map_compr
  ctx.reltab = {}
  return ctx
end

-- Simple API: disassemble code (a string) at address and output via out.
local function disass(code, addr, out)
  create(code, addr, out):disass(addr)
end

-- Return register name for RID.
local function regname(r)
  if r < 32 then return map_gpr[r] end
  return "f"..(r-32)
end

-- Public module functions.
return {
  create = create,
  disass = disass,
  regname = regname
}
