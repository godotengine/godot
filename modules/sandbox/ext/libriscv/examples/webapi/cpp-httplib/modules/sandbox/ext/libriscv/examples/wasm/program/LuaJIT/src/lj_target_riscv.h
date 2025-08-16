/*
** Definitions for RISC-V CPUs.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_TARGET_RISCV_H
#define _LJ_TARGET_RISCV_H

/* -- Registers IDs ------------------------------------------------------- */

#define GPRDEF(_) \
  _(X0) _(RA) _(SP) _(X3) _(X4) _(X5) _(X6) _(X7) \
  _(X8) _(X9) _(X10) _(X11) _(X12) _(X13) _(X14) _(X15) \
  _(X16) _(X17) _(X18) _(X19) _(X20) _(X21) _(X22) _(X23) \
  _(X24) _(X25) _(X26) _(X27) _(X28) _(X29) _(X30) _(X31)
#define FPRDEF(_) \
  _(F0) _(F1) _(F2) _(F3) _(F4) _(F5) _(F6) _(F7) \
  _(F8) _(F9) _(F10) _(F11) _(F12) _(F13) _(F14) _(F15) \
  _(F16) _(F17) _(F18) _(F19) _(F20) _(F21) _(F22) _(F23) \
  _(F24) _(F25) _(F26) _(F27) _(F28) _(F29) _(F30) _(F31)
#define VRIDDEF(_)

#define RIDENUM(name)	RID_##name,

enum {
  GPRDEF(RIDENUM)		/* General-purpose registers (GPRs). */
  FPRDEF(RIDENUM)		/* Floating-point registers (FPRs). */
  RID_MAX,
  RID_ZERO = RID_X0,
  RID_TMP = RID_RA,
  RID_GP = RID_X3,
  RID_TP = RID_X4,

  /* Calling conventions. */
  RID_RET = RID_X10,
  RID_RETLO = RID_X10,
  RID_RETHI = RID_X11,
  RID_FPRET = RID_F10,
  RID_CFUNCADDR = RID_X5,

  /* These definitions must match with the *.dasc file(s): */
  RID_BASE = RID_X18,		/* Interpreter BASE. */
  RID_LPC = RID_X20,		/* Interpreter PC. */
  RID_GL = RID_X21,		/* Interpreter GL. */
  RID_LREG = RID_X23,		/* Interpreter L. */

  /* Register ranges [min, max) and number of registers. */
  RID_MIN_GPR = RID_X0,
  RID_MAX_GPR = RID_X31+1,
  RID_MIN_FPR = RID_MAX_GPR,
  RID_MAX_FPR = RID_F31+1,
  RID_NUM_GPR = RID_MAX_GPR - RID_MIN_GPR,
  RID_NUM_FPR = RID_MAX_FPR - RID_MIN_FPR	/* Only even regs are used. */
};

#define RID_NUM_KREF		RID_NUM_GPR
#define RID_MIN_KREF		RID_X0

/* -- Register sets ------------------------------------------------------- */

/* Make use of all registers, except ZERO, TMP, SP, GP, TP, CFUNCADDR and GL. */
#define RSET_FIXED \
  (RID2RSET(RID_ZERO)|RID2RSET(RID_TMP)|RID2RSET(RID_SP)|\
   RID2RSET(RID_GP)|RID2RSET(RID_TP)|RID2RSET(RID_GL))
#define RSET_GPR	(RSET_RANGE(RID_MIN_GPR, RID_MAX_GPR) - RSET_FIXED)
#define RSET_FPR	RSET_RANGE(RID_MIN_FPR, RID_MAX_FPR)

#define RSET_ALL	(RSET_GPR|RSET_FPR)
#define RSET_INIT	RSET_ALL

#define RSET_SCRATCH_GPR \
  (RSET_RANGE(RID_X5, RID_X7+1)|RSET_RANGE(RID_X28, RID_X31+1)|\
   RSET_RANGE(RID_X10, RID_X17+1))

#define RSET_SCRATCH_FPR \
  (RSET_RANGE(RID_F0, RID_F7+1)|RSET_RANGE(RID_F10, RID_F17+1)|\
   RSET_RANGE(RID_F28, RID_F31+1))
#define RSET_SCRATCH		(RSET_SCRATCH_GPR|RSET_SCRATCH_FPR)

#define REGARG_FIRSTGPR		RID_X10
#define REGARG_LASTGPR		RID_X17
#define REGARG_NUMGPR		8

#define REGARG_FIRSTFPR		RID_F10
#define REGARG_LASTFPR		RID_F17
#define REGARG_NUMFPR		8

/* -- Spill slots --------------------------------------------------------- */

/* Spill slots are 32 bit wide. An even/odd pair is used for FPRs.
**
** SPS_FIXED: Available fixed spill slots in interpreter frame.
** This definition must match with the *.dasc file(s).
**
** SPS_FIRST: First spill slot for general use.
*/
#if LJ_32
#define SPS_FIXED	5
#else
#define SPS_FIXED	4
#endif
#define SPS_FIRST	4

#define SPOFS_TMP	0

#define sps_scale(slot)		(4 * (int32_t)(slot))
#define sps_align(slot)		(((slot) - SPS_FIXED + 3) & ~3)

/* -- Exit state ---------------------------------------------------------- */
/* This definition must match with the *.dasc file(s). */
typedef struct {
  lua_Number fpr[RID_NUM_FPR];	/* Floating-point registers. */
  intptr_t gpr[RID_NUM_GPR];	/* General-purpose registers. */
  int32_t spill[256];		/* Spill slots. */
} ExitState;

/* Highest exit + 1 indicates stack check. */
#define EXITSTATE_CHECKEXIT	1

/* Return the address of a per-trace exit stub. */
static LJ_AINLINE uint32_t *exitstub_trace_addr_(uint32_t *p, uint32_t exitno)
{
  while (*p == 0x00000013) p++;  /* Skip RISCVI_NOP. */
  return p + 4 + exitno;
}
/* Avoid dependence on lj_jit.h if only including lj_target.h. */
#define exitstub_trace_addr(T, exitno) \
  exitstub_trace_addr_((MCode *)((char *)(T)->mcode + (T)->szmcode), (exitno))

/* -- Instructions -------------------------------------------------------- */

/* Instruction fields. */
#define RISCVF_D(d)	(((d)&31) << 7)
#define RISCVF_S1(r)	(((r)&31) << 15)
#define RISCVF_S2(r)	(((r)&31) << 20)
#define RISCVF_S3(r)	(((r)&31) << 27)
#define RISCVF_FUNCT2(f)	(((f)&3) << 25)
#define RISCVF_FUNCT3(f)	(((f)&7) << 12)
#define RISCVF_FUNCT7(f)	(((f)&127) << 25)
#define RISCVF_SHAMT(s)	((s) << 20)
#define RISCVF_RM(m)	(((m)&7) << 12)
#define RISCVF_IMMI(i)	((i) << 20)
#define RISCVF_IMMS(i)	(((i)&0xfe0) << 20 | ((i)&0x1f) << 7)
#define RISCVF_IMMB(i)	(((i)&0x1000) << 19 | ((i)&0x800) >> 4 | ((i)&0x7e0) << 20 | ((i)&0x1e) << 7)
#define RISCVF_IMMU(i)	(((i)&0xfffff) << 12)
#define RISCVF_IMMJ(i)	(((i)&0x100000) << 11 | ((i)&0xff000) | ((i)&0x800) << 9 | ((i)&0x7fe) << 20)

/* Encode helpers. */
#define RISCVF_W_HI(w)  ((w) - ((((w)&0xfff)^0x800) - 0x800))
#define RISCVF_W_LO(w)  ((w)&0xfff)
#define RISCVF_HI(i)	((RISCVF_W_HI(i) >> 12) & 0xfffff)
#define RISCVF_LO(i)	RISCVF_W_LO(i)

/* Check for valid field range. */
#define RISCVF_SIMM_OK(x, b)	((((x) + (1 << (b-1))) >> (b)) == 0)
#define RISCVF_UIMM_OK(x, b)	(((x) >> (b)) == 0)
#define checku11(i)		RISCVF_UIMM_OK(i, 11)
#define checki12(i)		RISCVF_SIMM_OK(i, 12)
#define checki13(i)		RISCVF_SIMM_OK(i, 13)
#define checki20(i)		RISCVF_SIMM_OK(i, 20)
#define checki21(i)		RISCVF_SIMM_OK(i, 21)
#define checki32auipc(i) (checki32(i) && (int32_t)(i) < 0x7ffff800)

typedef enum RISCVIns {

  /* --- RVI --- */
  RISCVI_LUI = 0x00000037,
  RISCVI_AUIPC = 0x00000017,

  RISCVI_JAL = 0x0000006f,
  RISCVI_JALR = 0x00000067,

  RISCVI_ADDI = 0x00000013,
  RISCVI_SLTI = 0x00002013,
  RISCVI_SLTIU = 0x00003013,
  RISCVI_XORI = 0x00004013,
  RISCVI_ORI = 0x00006013,
  RISCVI_ANDI = 0x00007013,

  RISCVI_SLLI = 0x00001013,
  RISCVI_SRLI = 0x00005013,
  RISCVI_SRAI = 0x40005013,

  RISCVI_ADD = 0x00000033,
  RISCVI_SUB = 0x40000033,
  RISCVI_SLL = 0x00001033,
  RISCVI_SLT = 0x00002033,
  RISCVI_SLTU = 0x00003033,
  RISCVI_XOR = 0x00004033,
  RISCVI_SRL = 0x00005033,
  RISCVI_SRA = 0x40005033,
  RISCVI_OR = 0x00006033,
  RISCVI_AND = 0x00007033,

  RISCVI_LB = 0x00000003,
  RISCVI_LH = 0x00001003,
  RISCVI_LW = 0x00002003,
  RISCVI_LBU = 0x00004003,
  RISCVI_LHU = 0x00005003,
  RISCVI_SB = 0x00000023,
  RISCVI_SH = 0x00001023,
  RISCVI_SW = 0x00002023,

  RISCVI_BEQ = 0x00000063,
  RISCVI_BNE = 0x00001063,
  RISCVI_BLT = 0x00004063,
  RISCVI_BGE = 0x00005063,
  RISCVI_BLTU = 0x00006063,
  RISCVI_BGEU = 0x00007063,

  RISCVI_ECALL = 0x00000073,
  RISCVI_EBREAK = 0x00100073,

  RISCVI_NOP = 0x00000013,
  RISCVI_MV = 0x00000013,
  RISCVI_NOT = 0xfff04013,
  RISCVI_NEG = 0x40000033,
  RISCVI_RET = 0x00008067,
  RISCVI_ZEXT_B = 0x0ff07013,

#if LJ_TARGET_RISCV64
  RISCVI_LWU = 0x00007003,
  RISCVI_LD = 0x00003003,
  RISCVI_SD = 0x00003023,

  RISCVI_ADDIW = 0x0000001b,

  RISCVI_SLLIW = 0x0000101b,
  RISCVI_SRLIW = 0x0000501b,
  RISCVI_SRAIW = 0x4000501b,

  RISCVI_ADDW = 0x0000003b,
  RISCVI_SUBW = 0x4000003b,
  RISCVI_SLLW = 0x0000103b,
  RISCVI_SRLW = 0x0000503b,
  RISCVI_SRAW = 0x4000503b,

  RISCVI_NEGW = 0x4000003b,
  RISCVI_SEXT_W = 0x0000001b,
#endif

  /* --- RVM --- */
  RISCVI_MUL = 0x02000033,
  RISCVI_MULH = 0x02001033,
  RISCVI_MULHSU = 0x02002033,
  RISCVI_MULHU = 0x02003033,
  RISCVI_DIV = 0x02004033,
  RISCVI_DIVU = 0x02005033,
  RISCVI_REM = 0x02006033,
  RISCVI_REMU = 0x02007033,
#if LJ_TARGET_RISCV64
  RISCVI_MULW = 0x0200003b,
  RISCVI_DIVW = 0x0200403b,
  RISCVI_DIVUW = 0x0200503b,
  RISCVI_REMW = 0x0200603b,
  RISCVI_REMUW = 0x0200703b,
#endif

  /* --- RVF --- */
  RISCVI_FLW = 0x00002007,
  RISCVI_FSW = 0x00002027,

  RISCVI_FMADD_S = 0x00000043,
  RISCVI_FMSUB_S = 0x00000047,
  RISCVI_FNMSUB_S = 0x0000004b,
  RISCVI_FNMADD_S = 0x0000004f,

  RISCVI_FADD_S = 0x00000053,
  RISCVI_FSUB_S = 0x08000053,
  RISCVI_FMUL_S = 0x10000053,
  RISCVI_FDIV_S = 0x18000053,
  RISCVI_FSQRT_S = 0x58000053,

  RISCVI_FSGNJ_S = 0x20000053,
  RISCVI_FSGNJN_S = 0x20001053,
  RISCVI_FSGNJX_S = 0x20002053,

  RISCVI_FMIN_S = 0x28000053,
  RISCVI_FMAX_S = 0x28001053,

  RISCVI_FCVT_W_S = 0xc0000053,
  RISCVI_FCVT_WU_S = 0xc0100053,

  RISCVI_FMV_X_W = 0xe0000053,

  RISCVI_FEQ_S = 0xa0002053,
  RISCVI_FLT_S = 0xa0001053,
  RISCVI_FLE_S = 0xa0000053,

  RISCVI_FCLASS_S = 0xe0001053,

  RISCVI_FCVT_S_W = 0xd0000053,
  RISCVI_FCVT_S_WU = 0xd0100053,
  RISCVI_FMV_W_X = 0xf0000033,

  RISCVI_FMV_S = 0x20000053,
  RISCVI_FNEG_S = 0x20001053,
  RISCVI_FABS_S = 0x20002053,
#if LJ_TARGET_RISCV64
  RISCVI_FCVT_L_S = 0xc0200053,
  RISCVI_FCVT_LU_S = 0xc0300053,
  RISCVI_FCVT_S_L = 0xd0200053,
  RISCVI_FCVT_S_LU = 0xd0300053,
#endif

  /* --- RVD --- */
  RISCVI_FLD = 0x00003007,
  RISCVI_FSD = 0x00003027,

  RISCVI_FMADD_D = 0x02000043,
  RISCVI_FMSUB_D = 0x02000047,
  RISCVI_FNMSUB_D = 0x0200004b,
  RISCVI_FNMADD_D = 0x0200004f,

  RISCVI_FADD_D = 0x02000053,
  RISCVI_FSUB_D = 0x0a000053,
  RISCVI_FMUL_D = 0x12000053,
  RISCVI_FDIV_D = 0x1a000053,
  RISCVI_FSQRT_D = 0x5a000053,

  RISCVI_FSGNJ_D = 0x22000053,
  RISCVI_FSGNJN_D = 0x22001053,
  RISCVI_FSGNJX_D = 0x22002053,

  RISCVI_FMIN_D = 0x2a000053,
  RISCVI_FMAX_D = 0x2a001053,

  RISCVI_FCVT_S_D = 0x40100053,
  RISCVI_FCVT_D_S = 0x42000053,

  RISCVI_FEQ_D = 0xa2002053,
  RISCVI_FLT_D = 0xa2001053,
  RISCVI_FLE_D = 0xa2000053,

  RISCVI_FCLASS_D = 0xe2001053,

  RISCVI_FCVT_W_D = 0xc2000053,
  RISCVI_FCVT_WU_D = 0xc2100053,
  RISCVI_FCVT_D_W = 0xd2000053,
  RISCVI_FCVT_D_WU = 0xd2100053,

  RISCVI_FMV_D = 0x22000053,
  RISCVI_FNEG_D = 0x22001053,
  RISCVI_FABS_D = 0x22002053,
#if LJ_TARGET_RISCV64
  RISCVI_FCVT_L_D = 0xc2200053,
  RISCVI_FCVT_LU_D = 0xc2300053,
  RISCVI_FMV_X_D = 0xe2000053,
  RISCVI_FCVT_D_L = 0xd2200053,
  RISCVI_FCVT_D_LU = 0xd2300053,
  RISCVI_FMV_D_X = 0xf2000053,
#endif

  /* --- Zifencei --- */
  RISCVI_FENCE = 0x0000000f,
  RISCVI_FENCE_I = 0x0000100f,

  /* --- Zicsr --- */
  RISCVI_CSRRW = 0x00001073,
  RISCVI_CSRRS = 0x00002073,
  RISCVI_CSRRC = 0x00003073,
  RISCVI_CSRRWI = 0x00005073,
  RISCVI_CSRRSI = 0x00006073,
  RISCVI_CSRRCI = 0x00007073,

  /* --- RVB --- */
  /* Zba */
  RISCVI_SH1ADD = 0x20002033,
  RISCVI_SH2ADD = 0x20004033,
  RISCVI_SH3ADD = 0x20006033,
#if LJ_TARGET_RISCV64
  RISCVI_ADD_UW = 0x0800003b,

  RISCVI_SH1ADD_UW = 0x2000203b,
  RISCVI_SH2ADD_UW = 0x2000403b,
  RISCVI_SH3ADD_UW = 0x2000603b,

  RISCVI_SLLI_UW = 0x0800101b,

  RISCVI_ZEXT_W = 0x0800003b,
#endif
  /* Zbb */
  RISCVI_ANDN = 0x40007033,
  RISCVI_ORN = 0x40006033,
  RISCVI_XNOR = 0x40004033,

  RISCVI_CLZ = 0x60001013,
  RISCVI_CTZ = 0x60101013,

  RISCVI_CPOP = 0x60201013,

  RISCVI_MAX = 0x0a006033,
  RISCVI_MAXU = 0x0a007033,
  RISCVI_MIN = 0x0a004033,
  RISCVI_MINU = 0x0a005033,

  RISCVI_SEXT_B = 0x60401013,
  RISCVI_SEXT_H = 0x60501013,
#if LJ_TARGET_RISCV64
  RISCVI_ZEXT_H = 0x0800403b,
#endif

  RISCVI_ROL = 0x60001033,
  RISCVI_ROR = 0x60005033,
  RISCVI_RORI = 0x60005013,

  RISCVI_ORC_B = 0x28705013,

#if LJ_TARGET_RISCV64
  RISCVI_REV8 = 0x6b805013,

  RISCVI_CLZW = 0x6000101b,
  RISCVI_CTZW = 0x6010101b,

  RISCVI_CPOPW = 0x6020101b,

  RISCVI_ROLW = 0x6000103b,
  RISCVI_RORIW = 0x6000501b,
  RISCVI_RORW = 0x6000503b,
#endif
  /* NYI: Zbc, Zbs */

  /* --- Zicond --- */
  RISCVI_CZERO_EQZ = 0x0e005033,
  RISCVI_CZERO_NEZ = 0x0e007033,

  /* TBD: RVV?, RVP?, RVJ? */

  /* --- XThead* --- */
  /* XTHeadBa */
  RISCVI_TH_ADDSL = 0x0000100b,

  /* XTHeadBb */
  RISCVI_TH_SRRI = 0x1000100b,
#if LJ_TARGET_RISCV64
  RISCVI_TH_SRRIW = 0x1400100b,
#endif
  RISCVI_TH_EXT = 0x0000200b,
  RISCVI_TH_EXTU = 0x0000300b,
  RISCVI_TH_FF0 = 0x8400100b,
  RISCVI_TH_FF1 = 0x8600100b,
  RISCVI_TH_REV = 0x8200100b,
#if LJ_TARGET_RISCV64
  RISCVI_TH_REVW = 0x9000100b,
#endif
  RISCVI_TH_TSTNBZ = 0x8000100b,

  /* XTHeadBs */
  RISCVI_TH_TST = 0x8800100b,

  /* XTHeadCondMov */
  RISCVI_TH_MVEQZ = 0x4000100b,
  RISCVI_TH_MVNEZ = 0x4200100b,

  /* XTHeadMac */
  RISCVI_TH_MULA = 0x2000100b,
  RISCVI_TH_MULAH = 0x2800100b,
#if LJ_TARGET_RISCV64
  RISCVI_TH_MULAW = 0x2400100b,
#endif
  RISCVI_TH_MULS = 0x2200100b,
  RISCVI_TH_MULSH = 0x2a00100b,
  RISCVI_TH_MULSW = 0x2600100b,

  /* NYI: XTHeadMemIdx, XTHeadFMemIdx, XTHeadMemPair */
} RISCVIns;

typedef enum RISCVRM {
  RISCVRM_RNE = 0,
  RISCVRM_RTZ = 1,
  RISCVRM_RDN = 2,
  RISCVRM_RUP = 3,
  RISCVRM_RMM = 4,
  RISCVRM_DYN = 7,
} RISCVRM;

#endif
