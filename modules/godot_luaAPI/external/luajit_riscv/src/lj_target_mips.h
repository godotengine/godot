/*
** Definitions for MIPS CPUs.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_TARGET_MIPS_H
#define _LJ_TARGET_MIPS_H

/* -- Registers IDs ------------------------------------------------------- */

#define GPRDEF(_) \
  _(R0) _(R1) _(R2) _(R3) _(R4) _(R5) _(R6) _(R7) \
  _(R8) _(R9) _(R10) _(R11) _(R12) _(R13) _(R14) _(R15) \
  _(R16) _(R17) _(R18) _(R19) _(R20) _(R21) _(R22) _(R23) \
  _(R24) _(R25) _(SYS1) _(SYS2) _(R28) _(SP) _(R30) _(RA)
#if LJ_SOFTFP
#define FPRDEF(_)
#else
#define FPRDEF(_) \
  _(F0) _(F1) _(F2) _(F3) _(F4) _(F5) _(F6) _(F7) \
  _(F8) _(F9) _(F10) _(F11) _(F12) _(F13) _(F14) _(F15) \
  _(F16) _(F17) _(F18) _(F19) _(F20) _(F21) _(F22) _(F23) \
  _(F24) _(F25) _(F26) _(F27) _(F28) _(F29) _(F30) _(F31)
#endif
#define VRIDDEF(_)

#define RIDENUM(name)	RID_##name,

enum {
  GPRDEF(RIDENUM)		/* General-purpose registers (GPRs). */
  FPRDEF(RIDENUM)		/* Floating-point registers (FPRs). */
  RID_MAX,
  RID_ZERO = RID_R0,
  RID_TMP = RID_RA,
  RID_GP = RID_R28,

  /* Calling conventions. */
  RID_RET = RID_R2,
#if LJ_LE
  RID_RETHI = RID_R3,
  RID_RETLO = RID_R2,
#else
  RID_RETHI = RID_R2,
  RID_RETLO = RID_R3,
#endif
#if LJ_SOFTFP
  RID_FPRET = RID_R2,
#else
  RID_FPRET = RID_F0,
#endif
  RID_CFUNCADDR = RID_R25,

  /* These definitions must match with the *.dasc file(s): */
  RID_BASE = RID_R16,		/* Interpreter BASE. */
  RID_LPC = RID_R18,		/* Interpreter PC. */
  RID_DISPATCH = RID_R19,	/* Interpreter DISPATCH table. */
  RID_LREG = RID_R20,		/* Interpreter L. */
  RID_JGL = RID_R30,		/* On-trace: global_State + 32768. */

  /* Register ranges [min, max) and number of registers. */
  RID_MIN_GPR = RID_R0,
  RID_MAX_GPR = RID_RA+1,
  RID_MIN_FPR = RID_MAX_GPR,
#if LJ_SOFTFP
  RID_MAX_FPR = RID_MIN_FPR,
#else
  RID_MAX_FPR = RID_F31+1,
#endif
  RID_NUM_GPR = RID_MAX_GPR - RID_MIN_GPR,
  RID_NUM_FPR = RID_MAX_FPR - RID_MIN_FPR	/* Only even regs are used. */
};

#define RID_NUM_KREF		RID_NUM_GPR
#define RID_MIN_KREF		RID_R0

/* -- Register sets ------------------------------------------------------- */

/* Make use of all registers, except ZERO, TMP, SP, SYS1, SYS2, JGL and GP. */
#define RSET_FIXED \
  (RID2RSET(RID_ZERO)|RID2RSET(RID_TMP)|RID2RSET(RID_SP)|\
   RID2RSET(RID_SYS1)|RID2RSET(RID_SYS2)|RID2RSET(RID_JGL)|RID2RSET(RID_GP))
#define RSET_GPR	(RSET_RANGE(RID_MIN_GPR, RID_MAX_GPR) - RSET_FIXED)
#if LJ_SOFTFP
#define RSET_FPR		0
#else
#if LJ_32
#define RSET_FPR \
  (RID2RSET(RID_F0)|RID2RSET(RID_F2)|RID2RSET(RID_F4)|RID2RSET(RID_F6)|\
   RID2RSET(RID_F8)|RID2RSET(RID_F10)|RID2RSET(RID_F12)|RID2RSET(RID_F14)|\
   RID2RSET(RID_F16)|RID2RSET(RID_F18)|RID2RSET(RID_F20)|RID2RSET(RID_F22)|\
   RID2RSET(RID_F24)|RID2RSET(RID_F26)|RID2RSET(RID_F28)|RID2RSET(RID_F30))
#else
#define RSET_FPR		RSET_RANGE(RID_MIN_FPR, RID_MAX_FPR)
#endif
#endif
#define RSET_ALL		(RSET_GPR|RSET_FPR)
#define RSET_INIT		RSET_ALL

#define RSET_SCRATCH_GPR \
  (RSET_RANGE(RID_R1, RID_R15+1)|\
   RID2RSET(RID_R24)|RID2RSET(RID_R25))
#if LJ_SOFTFP
#define RSET_SCRATCH_FPR	0
#else
#if LJ_32
#define RSET_SCRATCH_FPR \
  (RID2RSET(RID_F0)|RID2RSET(RID_F2)|RID2RSET(RID_F4)|RID2RSET(RID_F6)|\
   RID2RSET(RID_F8)|RID2RSET(RID_F10)|RID2RSET(RID_F12)|RID2RSET(RID_F14)|\
   RID2RSET(RID_F16)|RID2RSET(RID_F18))
#else
#define RSET_SCRATCH_FPR	RSET_RANGE(RID_F0, RID_F24)
#endif
#endif
#define RSET_SCRATCH		(RSET_SCRATCH_GPR|RSET_SCRATCH_FPR)
#define REGARG_FIRSTGPR		RID_R4
#if LJ_32
#define REGARG_LASTGPR		RID_R7
#define REGARG_NUMGPR		4
#else
#define REGARG_LASTGPR		RID_R11
#define REGARG_NUMGPR		8
#endif
#if LJ_ABI_SOFTFP
#define REGARG_FIRSTFPR		0
#define REGARG_LASTFPR		0
#define REGARG_NUMFPR		0
#else
#define REGARG_FIRSTFPR		RID_F12
#if LJ_32
#define REGARG_LASTFPR		RID_F14
#define REGARG_NUMFPR		2
#else
#define REGARG_LASTFPR		RID_F19
#define REGARG_NUMFPR		8
#endif
#endif

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
#define sps_align(slot)		(((slot) - SPS_FIXED + 1) & ~1)

/* -- Exit state ---------------------------------------------------------- */

/* This definition must match with the *.dasc file(s). */
typedef struct {
#if !LJ_SOFTFP
  lua_Number fpr[RID_NUM_FPR];	/* Floating-point registers. */
#endif
  intptr_t gpr[RID_NUM_GPR];	/* General-purpose registers. */
  int32_t spill[256];		/* Spill slots. */
} ExitState;

/* Highest exit + 1 indicates stack check. */
#define EXITSTATE_CHECKEXIT	1

/* Return the address of a per-trace exit stub. */
static LJ_AINLINE uint32_t *exitstub_trace_addr_(uint32_t *p)
{
  while (*p == 0x00000000) p++;  /* Skip MIPSI_NOP. */
  return p;
}
/* Avoid dependence on lj_jit.h if only including lj_target.h. */
#define exitstub_trace_addr(T, exitno) \
  exitstub_trace_addr_((MCode *)((char *)(T)->mcode + (T)->szmcode))

/* -- Instructions -------------------------------------------------------- */

/* Instruction fields. */
#define MIPSF_S(r)	((r) << 21)
#define MIPSF_T(r)	((r) << 16)
#define MIPSF_D(r)	((r) << 11)
#define MIPSF_R(r)	((r) << 21)
#define MIPSF_H(r)	((r) << 16)
#define MIPSF_G(r)	((r) << 11)
#define MIPSF_F(r)	((r) << 6)
#define MIPSF_A(n)	((n) << 6)
#define MIPSF_M(n)	((n) << 11)
#define MIPSF_L(n)	((n) << 6)

typedef enum MIPSIns {
  MIPSI_D = 0x38,
  MIPSI_DV = 0x10,
  MIPSI_D32 = 0x3c,
  /* Integer instructions. */
  MIPSI_MOVE = 0x00000025,
  MIPSI_NOP = 0x00000000,

  MIPSI_LI = 0x24000000,
  MIPSI_LU = 0x34000000,
  MIPSI_LUI = 0x3c000000,

  MIPSI_AND = 0x00000024,
  MIPSI_ANDI = 0x30000000,
  MIPSI_OR = 0x00000025,
  MIPSI_ORI = 0x34000000,
  MIPSI_XOR = 0x00000026,
  MIPSI_XORI = 0x38000000,
  MIPSI_NOR = 0x00000027,

  MIPSI_SLT = 0x0000002a,
  MIPSI_SLTU = 0x0000002b,
  MIPSI_SLTI = 0x28000000,
  MIPSI_SLTIU = 0x2c000000,

  MIPSI_ADDU = 0x00000021,
  MIPSI_ADDIU = 0x24000000,
  MIPSI_SUB = 0x00000022,
  MIPSI_SUBU = 0x00000023,

#if !LJ_TARGET_MIPSR6
  MIPSI_MUL = 0x70000002,
  MIPSI_DIV = 0x0000001a,
  MIPSI_DIVU = 0x0000001b,

  MIPSI_MOVZ = 0x0000000a,
  MIPSI_MOVN = 0x0000000b,
  MIPSI_MFHI = 0x00000010,
  MIPSI_MFLO = 0x00000012,
  MIPSI_MULT = 0x00000018,
#else
  MIPSI_MUL = 0x00000098,
  MIPSI_MUH = 0x000000d8,
  MIPSI_DIV = 0x0000009a,
  MIPSI_DIVU = 0x0000009b,

  MIPSI_SELEQZ = 0x00000035,
  MIPSI_SELNEZ = 0x00000037,
#endif

  MIPSI_SLL = 0x00000000,
  MIPSI_SRL = 0x00000002,
  MIPSI_SRA = 0x00000003,
  MIPSI_ROTR = 0x00200002,	/* MIPSXXR2 */
  MIPSI_DROTR = 0x0020003a,
  MIPSI_DROTR32 = 0x0020003e,
  MIPSI_SLLV = 0x00000004,
  MIPSI_SRLV = 0x00000006,
  MIPSI_SRAV = 0x00000007,
  MIPSI_ROTRV = 0x00000046,	/* MIPSXXR2 */
  MIPSI_DROTRV = 0x00000056,

  MIPSI_INS = 0x7c000004,	/* MIPSXXR2 */

  MIPSI_SEB = 0x7c000420,	/* MIPSXXR2 */
  MIPSI_SEH = 0x7c000620,	/* MIPSXXR2 */
  MIPSI_WSBH = 0x7c0000a0,	/* MIPSXXR2 */
  MIPSI_DSBH = 0x7c0000a4,

  MIPSI_B = 0x10000000,
  MIPSI_J = 0x08000000,
  MIPSI_JAL = 0x0c000000,
#if !LJ_TARGET_MIPSR6
  MIPSI_JALX = 0x74000000,
  MIPSI_JR = 0x00000008,
#else
  MIPSI_JR = 0x00000009,
  MIPSI_BALC = 0xe8000000,
#endif
  MIPSI_JALR = 0x0000f809,

  MIPSI_BEQ = 0x10000000,
  MIPSI_BNE = 0x14000000,
  MIPSI_BLEZ = 0x18000000,
  MIPSI_BGTZ = 0x1c000000,
  MIPSI_BLTZ = 0x04000000,
  MIPSI_BGEZ = 0x04010000,

  /* Load/store instructions. */
  MIPSI_LW = 0x8c000000,
  MIPSI_LD = 0xdc000000,
  MIPSI_SW = 0xac000000,
  MIPSI_SD = 0xfc000000,
  MIPSI_LB = 0x80000000,
  MIPSI_SB = 0xa0000000,
  MIPSI_LH = 0x84000000,
  MIPSI_SH = 0xa4000000,
  MIPSI_LBU = 0x90000000,
  MIPSI_LHU = 0x94000000,
  MIPSI_LWC1 = 0xc4000000,
  MIPSI_SWC1 = 0xe4000000,
  MIPSI_LDC1 = 0xd4000000,
  MIPSI_SDC1 = 0xf4000000,

  /* MIPS64 instructions. */
  MIPSI_DADD = 0x0000002c,
  MIPSI_DADDU = 0x0000002d,
  MIPSI_DADDIU = 0x64000000,
  MIPSI_DSUB = 0x0000002e,
  MIPSI_DSUBU = 0x0000002f,
#if !LJ_TARGET_MIPSR6
  MIPSI_DDIV = 0x0000001e,
  MIPSI_DDIVU = 0x0000001f,
  MIPSI_DMULT = 0x0000001c,
  MIPSI_DMULTU = 0x0000001d,
#else
  MIPSI_DDIV = 0x0000009e,
  MIPSI_DMOD = 0x000000de,
  MIPSI_DDIVU = 0x0000009f,
  MIPSI_DMODU = 0x000000df,
  MIPSI_DMUL = 0x0000009c,
  MIPSI_DMUH = 0x000000dc,
#endif

  MIPSI_DSLL = 0x00000038,
  MIPSI_DSRL = 0x0000003a,
  MIPSI_DSLLV = 0x00000014,
  MIPSI_DSRLV = 0x00000016,
  MIPSI_DSRA = 0x0000003b,
  MIPSI_DSRAV = 0x00000017,
  MIPSI_DSRA32 = 0x0000003f,
  MIPSI_DSLL32 = 0x0000003c,
  MIPSI_DSRL32 = 0x0000003e,
  MIPSI_DSHD = 0x7c000164,

  MIPSI_AADDU = LJ_32 ? MIPSI_ADDU : MIPSI_DADDU,
  MIPSI_AADDIU = LJ_32 ? MIPSI_ADDIU : MIPSI_DADDIU,
  MIPSI_ASUBU = LJ_32 ? MIPSI_SUBU : MIPSI_DSUBU,
  MIPSI_AL = LJ_32 ? MIPSI_LW : MIPSI_LD,
  MIPSI_AS = LJ_32 ? MIPSI_SW : MIPSI_SD,
#if LJ_TARGET_MIPSR6
  MIPSI_LSA = 0x00000005,
  MIPSI_DLSA = 0x00000015,
  MIPSI_ALSA = LJ_32 ? MIPSI_LSA : MIPSI_DLSA,
#endif

  /* Extract/insert instructions. */
  MIPSI_DEXTM = 0x7c000001,
  MIPSI_DEXTU = 0x7c000002,
  MIPSI_DEXT = 0x7c000003,
  MIPSI_DINSM = 0x7c000005,
  MIPSI_DINSU = 0x7c000006,
  MIPSI_DINS = 0x7c000007,

  MIPSI_FLOOR_D = 0x4620000b,

  /* FP instructions. */
  MIPSI_MOV_S = 0x46000006,
  MIPSI_MOV_D = 0x46200006,
#if !LJ_TARGET_MIPSR6
  MIPSI_MOVT_D = 0x46210011,
  MIPSI_MOVF_D = 0x46200011,
#else
  MIPSI_MIN_D = 0x4620001C,
  MIPSI_MAX_D = 0x4620001E,
  MIPSI_SEL_D = 0x46200010,
#endif

  MIPSI_ABS_D = 0x46200005,
  MIPSI_NEG_D = 0x46200007,

  MIPSI_ADD_D = 0x46200000,
  MIPSI_SUB_D = 0x46200001,
  MIPSI_MUL_D = 0x46200002,
  MIPSI_DIV_D = 0x46200003,
  MIPSI_SQRT_D = 0x46200004,

  MIPSI_ADD_S = 0x46000000,
  MIPSI_SUB_S = 0x46000001,

  MIPSI_CVT_D_S = 0x46000021,
  MIPSI_CVT_W_S = 0x46000024,
  MIPSI_CVT_S_D = 0x46200020,
  MIPSI_CVT_W_D = 0x46200024,
  MIPSI_CVT_S_W = 0x46800020,
  MIPSI_CVT_D_W = 0x46800021,
  MIPSI_CVT_S_L = 0x46a00020,
  MIPSI_CVT_D_L = 0x46a00021,

  MIPSI_TRUNC_W_S = 0x4600000d,
  MIPSI_TRUNC_W_D = 0x4620000d,
  MIPSI_TRUNC_L_S = 0x46000009,
  MIPSI_TRUNC_L_D = 0x46200009,
  MIPSI_FLOOR_W_S = 0x4600000f,
  MIPSI_FLOOR_W_D = 0x4620000f,

  MIPSI_MFC1 = 0x44000000,
  MIPSI_MTC1 = 0x44800000,
  MIPSI_DMTC1 = 0x44a00000,
  MIPSI_DMFC1 = 0x44200000,

#if !LJ_TARGET_MIPSR6
  MIPSI_BC1F = 0x45000000,
  MIPSI_BC1T = 0x45010000,
  MIPSI_C_EQ_D = 0x46200032,
  MIPSI_C_OLT_S = 0x46000034,
  MIPSI_C_OLT_D = 0x46200034,
  MIPSI_C_ULT_D = 0x46200035,
  MIPSI_C_OLE_D = 0x46200036,
  MIPSI_C_ULE_D = 0x46200037,
#else
  MIPSI_BC1EQZ = 0x45200000,
  MIPSI_BC1NEZ = 0x45a00000,
  MIPSI_CMP_EQ_D = 0x46a00002,
  MIPSI_CMP_LT_S = 0x46800004,
  MIPSI_CMP_LT_D = 0x46a00004,
#endif

} MIPSIns;

#endif
