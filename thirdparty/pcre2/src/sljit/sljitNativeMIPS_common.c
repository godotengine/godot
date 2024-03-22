/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Latest MIPS architecture. */

#ifdef HAVE_PRCTL
#include <sys/prctl.h>
#endif

#if !defined(__mips_hard_float) || defined(__mips_single_float)
/* Disable automatic detection, covers both -msoft-float and -mno-float */
#define SLJIT_IS_FPU_AVAILABLE 0
#endif

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return "MIPS32-R6" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_MIPS_32 */
	return "MIPS64-R6" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_MIPS_32 */

#elif (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 5)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return "MIPS32-R5" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_MIPS_32 */
	return "MIPS64-R5" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_MIPS_32 */

#elif (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return "MIPS32-R2" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_MIPS_32 */
	return "MIPS64-R2" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_MIPS_32 */

#elif (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return "MIPS32-R1" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_MIPS_32 */
	return "MIPS64-R1" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_MIPS_32 */

#else /* SLJIT_MIPS_REV < 1 */
	return "MIPS III" SLJIT_CPUINFO;
#endif /* SLJIT_MIPS_REV >= 6 */
}

/* Length of an instruction word
   Both for mips-32 and mips-64 */
typedef sljit_u32 sljit_ins;

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3	(SLJIT_NUMBER_OF_REGISTERS + 4)

/* For position independent code, t9 must contain the function address. */
#define PIC_ADDR_REG	TMP_REG2

/* Floating point status register. */
#define FCSR_REG	31
/* Return address register. */
#define RETURN_ADDR_REG	31

/* Flags are kept in volatile registers. */
#define EQUAL_FLAG	3
#define OTHER_FLAG	1

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 7] = {
	0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 23, 22, 21, 20, 19, 18, 17, 16, 29, 4, 25, 31, 3, 1
};

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)
#define TMP_FREG3	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)

static const sljit_u8 freg_map[((SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3) << 1) + 1] = {
	0,
	0, 14, 2, 4, 6, 8, 18, 30, 28, 26, 24, 22, 20,
	12, 10, 16,
	1, 15, 3, 5, 7, 9, 19, 31, 29, 27, 25, 23, 21,
	13, 11, 17
};

#else /* !SLJIT_CONFIG_MIPS_32 */

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 4] = {
	0, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 30, 29, 28, 27, 26, 25, 24, 12, 11, 10
};

#endif /* SLJIT_CONFIG_MIPS_32 */

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

#define S(s)		((sljit_ins)reg_map[s] << 21)
#define T(t)		((sljit_ins)reg_map[t] << 16)
#define D(d)		((sljit_ins)reg_map[d] << 11)
#define FT(t)		((sljit_ins)freg_map[t] << 16)
#define FS(s)		((sljit_ins)freg_map[s] << 11)
#define FD(d)		((sljit_ins)freg_map[d] << 6)
/* Absolute registers. */
#define SA(s)		((sljit_ins)(s) << 21)
#define TA(t)		((sljit_ins)(t) << 16)
#define DA(d)		((sljit_ins)(d) << 11)
#define IMM(imm)	((sljit_ins)(imm) & 0xffff)
#define SH_IMM(imm)	((sljit_ins)(imm) << 6)

#define DR(dr)		(reg_map[dr])
#define FR(dr)		(freg_map[dr])
#define HI(opcode)	((sljit_ins)(opcode) << 26)
#define LO(opcode)	((sljit_ins)(opcode))
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
/* CMP.cond.fmt */
/* S = (20 << 21) D = (21 << 21) */
#define CMP_FMT_S	(20 << 21)
#endif /* SLJIT_MIPS_REV >= 6 */
/* S = (16 << 21) D = (17 << 21) */
#define FMT_S		(16 << 21)
#define FMT_D		(17 << 21)

#define ABS_S		(HI(17) | FMT_S | LO(5))
#define ADD_S		(HI(17) | FMT_S | LO(0))
#define ADDIU		(HI(9))
#define ADDU		(HI(0) | LO(33))
#define AND		(HI(0) | LO(36))
#define ANDI		(HI(12))
#define B		(HI(4))
#define BAL		(HI(1) | (17 << 16))
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define BC1EQZ		(HI(17) | (9 << 21) | FT(TMP_FREG3))
#define BC1NEZ		(HI(17) | (13 << 21) | FT(TMP_FREG3))
#else /* SLJIT_MIPS_REV < 6 */
#define BC1F		(HI(17) | (8 << 21))
#define BC1T		(HI(17) | (8 << 21) | (1 << 16))
#endif /* SLJIT_MIPS_REV >= 6 */
#define BEQ		(HI(4))
#define BGEZ		(HI(1) | (1 << 16))
#define BGTZ		(HI(7))
#define BLEZ		(HI(6))
#define BLTZ		(HI(1) | (0 << 16))
#define BNE		(HI(5))
#define BREAK		(HI(0) | LO(13))
#define CFC1		(HI(17) | (2 << 21))
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define C_EQ_S		(HI(17) | CMP_FMT_S | LO(2))
#define C_OLE_S		(HI(17) | CMP_FMT_S | LO(6))
#define C_OLT_S		(HI(17) | CMP_FMT_S | LO(4))
#define C_UEQ_S		(HI(17) | CMP_FMT_S | LO(3))
#define C_ULE_S		(HI(17) | CMP_FMT_S | LO(7))
#define C_ULT_S		(HI(17) | CMP_FMT_S | LO(5))
#define C_UN_S		(HI(17) | CMP_FMT_S | LO(1))
#define C_FD		(FD(TMP_FREG3))
#else /* SLJIT_MIPS_REV < 6 */
#define C_EQ_S		(HI(17) | FMT_S | LO(50))
#define C_OLE_S		(HI(17) | FMT_S | LO(54))
#define C_OLT_S		(HI(17) | FMT_S | LO(52))
#define C_UEQ_S		(HI(17) | FMT_S | LO(51))
#define C_ULE_S		(HI(17) | FMT_S | LO(55))
#define C_ULT_S		(HI(17) | FMT_S | LO(53))
#define C_UN_S		(HI(17) | FMT_S | LO(49))
#define C_FD		(0)
#endif /* SLJIT_MIPS_REV >= 6 */
#define CVT_S_S		(HI(17) | FMT_S | LO(32))
#define DADDIU		(HI(25))
#define DADDU		(HI(0) | LO(45))
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define DDIV		(HI(0) | (2 << 6) | LO(30))
#define DDIVU		(HI(0) | (2 << 6) | LO(31))
#define DMOD		(HI(0) | (3 << 6) | LO(30))
#define DMODU		(HI(0) | (3 << 6) | LO(31))
#define DIV		(HI(0) | (2 << 6) | LO(26))
#define DIVU		(HI(0) | (2 << 6) | LO(27))
#define DMUH		(HI(0) | (3 << 6) | LO(28))
#define DMUHU		(HI(0) | (3 << 6) | LO(29))
#define DMUL		(HI(0) | (2 << 6) | LO(28))
#define DMULU		(HI(0) | (2 << 6) | LO(29))
#else /* SLJIT_MIPS_REV < 6 */
#define DDIV		(HI(0) | LO(30))
#define DDIVU		(HI(0) | LO(31))
#define DIV		(HI(0) | LO(26))
#define DIVU		(HI(0) | LO(27))
#define DMULT		(HI(0) | LO(28))
#define DMULTU		(HI(0) | LO(29))
#endif /* SLJIT_MIPS_REV >= 6 */
#define DIV_S		(HI(17) | FMT_S | LO(3))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define DINSU		(HI(31) | LO(6))
#endif /* SLJIT_MIPS_REV >= 2 */
#define DMFC1		(HI(17) | (1 << 21))
#define DMTC1		(HI(17) | (5 << 21))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define DROTR		(HI(0) | (1 << 21) | LO(58))
#define DROTR32		(HI(0) | (1 << 21) | LO(62))
#define DROTRV		(HI(0) | (1 << 6) | LO(22))
#define DSBH		(HI(31) | (2 << 6) | LO(36))
#define DSHD		(HI(31) | (5 << 6) | LO(36))
#endif /* SLJIT_MIPS_REV >= 2 */
#define DSLL		(HI(0) | LO(56))
#define DSLL32		(HI(0) | LO(60))
#define DSLLV		(HI(0) | LO(20))
#define DSRA		(HI(0) | LO(59))
#define DSRA32		(HI(0) | LO(63))
#define DSRAV		(HI(0) | LO(23))
#define DSRL		(HI(0) | LO(58))
#define DSRL32		(HI(0) | LO(62))
#define DSRLV		(HI(0) | LO(22))
#define DSUBU		(HI(0) | LO(47))
#define J		(HI(2))
#define JAL		(HI(3))
#define JALR		(HI(0) | LO(9))
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define JR		(HI(0) | LO(9))
#else /* SLJIT_MIPS_REV < 6 */
#define JR		(HI(0) | LO(8))
#endif /* SLJIT_MIPS_REV >= 6 */
#define LD		(HI(55))
#define LDL		(HI(26))
#define LDR		(HI(27))
#define LDC1		(HI(53))
#define LUI		(HI(15))
#define LW		(HI(35))
#define LWL		(HI(34))
#define LWR		(HI(38))
#define LWC1		(HI(49))
#define MFC1		(HI(17))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define MFHC1		(HI(17) | (3 << 21))
#endif /* SLJIT_MIPS_REV >= 2 */
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define MOD		(HI(0) | (3 << 6) | LO(26))
#define MODU		(HI(0) | (3 << 6) | LO(27))
#else /* SLJIT_MIPS_REV < 6 */
#define MFHI		(HI(0) | LO(16))
#define MFLO		(HI(0) | LO(18))
#endif /* SLJIT_MIPS_REV >= 6 */
#define MTC1		(HI(17) | (4 << 21))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define MTHC1		(HI(17) | (7 << 21))
#endif /* SLJIT_MIPS_REV >= 2 */
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define MUH		(HI(0) | (3 << 6) | LO(24))
#define MUHU		(HI(0) | (3 << 6) | LO(25))
#define MUL		(HI(0) | (2 << 6) | LO(24))
#define MULU		(HI(0) | (2 << 6) | LO(25))
#else /* SLJIT_MIPS_REV < 6 */
#define MULT		(HI(0) | LO(24))
#define MULTU		(HI(0) | LO(25))
#endif /* SLJIT_MIPS_REV >= 6 */
#define MUL_S		(HI(17) | FMT_S | LO(2))
#define NEG_S		(HI(17) | FMT_S | LO(7))
#define NOP		(HI(0) | LO(0))
#define NOR		(HI(0) | LO(39))
#define OR		(HI(0) | LO(37))
#define ORI		(HI(13))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define ROTR		(HI(0) | (1 << 21) | LO(2))
#define ROTRV		(HI(0) | (1 << 6) | LO(6))
#endif /* SLJIT_MIPS_REV >= 2 */
#define SD		(HI(63))
#define SDL		(HI(44))
#define SDR		(HI(45))
#define SDC1		(HI(61))
#define SLT		(HI(0) | LO(42))
#define SLTI		(HI(10))
#define SLTIU		(HI(11))
#define SLTU		(HI(0) | LO(43))
#define SLL		(HI(0) | LO(0))
#define SLLV		(HI(0) | LO(4))
#define SRL		(HI(0) | LO(2))
#define SRLV		(HI(0) | LO(6))
#define SRA		(HI(0) | LO(3))
#define SRAV		(HI(0) | LO(7))
#define SUB_S		(HI(17) | FMT_S | LO(1))
#define SUBU		(HI(0) | LO(35))
#define SW		(HI(43))
#define SWL		(HI(42))
#define SWR		(HI(46))
#define SWC1		(HI(57))
#define TRUNC_W_S	(HI(17) | FMT_S | LO(13))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define WSBH		(HI(31) | (2 << 6) | LO(32))
#endif /* SLJIT_MIPS_REV >= 2 */
#define XOR		(HI(0) | LO(38))
#define XORI		(HI(14))

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
#define CLZ		(HI(28) | LO(32))
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#define DCLZ		(LO(18))
#else /* SLJIT_MIPS_REV < 6 */
#define DCLZ		(HI(28) | LO(36))
#define MOVF		(HI(0) | (0 << 16) | LO(1))
#define MOVF_S		(HI(17) | FMT_S | (0 << 16) | LO(17))
#define MOVN		(HI(0) | LO(11))
#define MOVN_S		(HI(17) | FMT_S | LO(19))
#define MOVT		(HI(0) | (1 << 16) | LO(1))
#define MOVT_S		(HI(17) | FMT_S | (1 << 16) | LO(17))
#define MOVZ		(HI(0) | LO(10))
#define MOVZ_S		(HI(17) | FMT_S | LO(18))
#define MUL		(HI(28) | LO(2))
#endif /* SLJIT_MIPS_REV >= 6 */
#define PREF		(HI(51))
#define PREFX		(HI(19) | LO(15))
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#define SEB		(HI(31) | (16 << 6) | LO(32))
#define SEH		(HI(31) | (24 << 6) | LO(32))
#endif /* SLJIT_MIPS_REV >= 2 */
#endif /* SLJIT_MIPS_REV >= 1 */

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define ADDU_W		ADDU
#define ADDIU_W		ADDIU
#define SLL_W		SLL
#define SRA_W		SRA
#define SUBU_W		SUBU
#define STORE_W		SW
#define LOAD_W		LW
#else
#define ADDU_W		DADDU
#define ADDIU_W		DADDIU
#define SLL_W		DSLL
#define SRA_W		DSRA
#define SUBU_W		DSUBU
#define STORE_W		SD
#define LOAD_W		LD
#endif

#define MOV_fmt(f)	(HI(17) | f | LO(6))

#define SIMM_MAX	(0x7fff)
#define SIMM_MIN	(-0x8000)
#define UIMM_MAX	(0xffff)

#define CPU_FEATURE_DETECTED	(1 << 0)
#define CPU_FEATURE_FPU		(1 << 1)
#define CPU_FEATURE_FP64	(1 << 2)
#define CPU_FEATURE_FR		(1 << 3)

static sljit_u32 cpu_feature_list = 0;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
	&& (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)

static sljit_s32 function_check_is_freg(struct sljit_compiler *compiler, sljit_s32 fr, sljit_s32 is_32)
{
	if (compiler->scratches == -1)
		return 0;

	if (is_32 && fr >= SLJIT_F64_SECOND(SLJIT_FR0))
		fr -= SLJIT_F64_SECOND(0);

	return (fr >= SLJIT_FR0 && fr < (SLJIT_FR0 + compiler->fscratches))
		|| (fr > (SLJIT_FS0 - compiler->fsaveds) && fr <= SLJIT_FS0)
		|| (fr >= SLJIT_TMP_FREGISTER_BASE && fr < (SLJIT_TMP_FREGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS));
}

#endif /* SLJIT_CONFIG_MIPS_32 && SLJIT_ARGUMENT_CHECKS */

static void get_cpu_features(void)
{
#if !defined(SLJIT_IS_FPU_AVAILABLE) && defined(__GNUC__)
	sljit_u32 fir = 0;
#endif /* !SLJIT_IS_FPU_AVAILABLE && __GNUC__ */
	sljit_u32 feature_list = CPU_FEATURE_DETECTED;

#if defined(SLJIT_IS_FPU_AVAILABLE)
#if SLJIT_IS_FPU_AVAILABLE
	feature_list |= CPU_FEATURE_FPU;
#if SLJIT_IS_FPU_AVAILABLE == 64
	feature_list |= CPU_FEATURE_FP64;
#endif /* SLJIT_IS_FPU_AVAILABLE == 64 */
#endif /* SLJIT_IS_FPU_AVAILABLE */
#elif defined(__GNUC__)
	__asm__ ("cfc1 %0, $0" : "=r"(fir));
	if ((fir & (0x3 << 16)) == (0x3 << 16))
		feature_list |= CPU_FEATURE_FPU;

#if (defined(SLJIT_CONFIG_MIPS_64) && SLJIT_CONFIG_MIPS_64) \
	&& (!defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV < 2)
	if ((feature_list & CPU_FEATURE_FPU))
		feature_list |= CPU_FEATURE_FP64;
#else /* SLJIT_CONFIG_MIPS32 || SLJIT_MIPS_REV >= 2 */
	if ((fir & (1 << 22)))
		feature_list |= CPU_FEATURE_FP64;
#endif /* SLJIT_CONFIG_MIPS_64 && SLJIT_MIPS_REV < 2 */
#endif /* SLJIT_IS_FPU_AVAILABLE */

	if ((feature_list & CPU_FEATURE_FPU) && (feature_list & CPU_FEATURE_FP64)) {
#if defined(SLJIT_CONFIG_MIPS_32) && SLJIT_CONFIG_MIPS_32
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 6
		feature_list |= CPU_FEATURE_FR;
#elif defined(SLJIT_DETECT_FR) && SLJIT_DETECT_FR == 0
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 5
		feature_list |= CPU_FEATURE_FR;
#endif /* SLJIT_MIPS_REV >= 5 */
#else
		sljit_s32 flag = -1;
#ifndef FR_GET_FP_MODE
		sljit_f64 zero = 0.0;
#else /* PR_GET_FP_MODE */
		flag = prctl(PR_GET_FP_MODE);

		if (flag > 0)
			feature_list |= CPU_FEATURE_FR;
#endif /* FP_GET_PR_MODE */
#if ((defined(SLJIT_DETECT_FR) && SLJIT_DETECT_FR == 2) \
	|| (!defined(PR_GET_FP_MODE) && (!defined(SLJIT_DETECT_FR) || SLJIT_DETECT_FR >= 1))) \
	&& (defined(__GNUC__) && (defined(__mips) && __mips >= 2))
		if (flag < 0) {
			__asm__ (".set oddspreg\n"
				"lwc1 $f17, %0\n"
				"ldc1 $f16, %1\n"
				"swc1 $f17, %0\n"
			: "+m" (flag) : "m" (zero) : "$f16", "$f17");
			if (flag)
				feature_list |= CPU_FEATURE_FR;
		}
#endif /* (!PR_GET_FP_MODE || (PR_GET_FP_MODE && SLJIT_DETECT_FR == 2)) && __GNUC__ */
#endif /* SLJIT_MIPS_REV >= 6 */
#else /* !SLJIT_CONFIG_MIPS_32 */
		/* StatusFR=1 is the only mode supported by the code in MIPS64 */
		feature_list |= CPU_FEATURE_FR;
#endif /* SLJIT_CONFIG_MIPS_32 */
	}

	cpu_feature_list = feature_list;
}

/* dest_reg is the absolute name of the register
   Useful for reordering instructions in the delay slot. */
static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins, sljit_s32 delay_slot)
{
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	SLJIT_ASSERT(delay_slot == MOVABLE_INS || delay_slot >= UNMOVABLE_INS
		|| (sljit_ins)delay_slot == ((ins >> 11) & 0x1f)
		|| (sljit_ins)delay_slot == ((ins >> 16) & 0x1f));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	compiler->delay_slot = delay_slot;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_ins invert_branch(sljit_uw flags)
{
	if (flags & IS_BIT26_COND)
		return (1 << 26);
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
	if (flags & IS_BIT23_COND)
		return (1 << 23);
#endif /* SLJIT_MIPS_REV >= 6 */
	return (1 << 16);
}

static SLJIT_INLINE sljit_ins* detect_jump_type(struct sljit_jump *jump, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_ins *inst;
	sljit_ins saved_inst;

	inst = (sljit_ins *)jump->addr;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (jump->flags & (SLJIT_REWRITABLE_JUMP | IS_CALL))
		goto exit;
#else
	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		goto exit;
#endif

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;
	}

	if (jump->flags & IS_COND)
		inst--;

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (jump->flags & IS_CALL)
		goto preserve_addr;
#endif

	/* B instructions. */
	if (jump->flags & IS_MOVABLE) {
		diff = ((sljit_sw)target_addr - (sljit_sw)inst - executable_offset) >> 2;
		if (diff <= SIMM_MAX && diff >= SIMM_MIN) {
			jump->flags |= PATCH_B;

			if (!(jump->flags & IS_COND)) {
				inst[0] = inst[-1];
				inst[-1] = (jump->flags & IS_JAL) ? BAL : B;
				jump->addr -= sizeof(sljit_ins);
				return inst;
			}
			saved_inst = inst[0];
			inst[0] = inst[-1];
			inst[-1] = saved_inst ^ invert_branch(jump->flags);
			jump->addr -= 2 * sizeof(sljit_ins);
			return inst;
		}
	} else {
		diff = ((sljit_sw)target_addr - (sljit_sw)(inst + 1) - executable_offset) >> 2;
		if (diff <= SIMM_MAX && diff >= SIMM_MIN) {
			jump->flags |= PATCH_B;

			if (!(jump->flags & IS_COND)) {
				inst[0] = (jump->flags & IS_JAL) ? BAL : B;
				/* Keep inst[1] */
				return inst + 1;
			}
			inst[0] ^= invert_branch(jump->flags);
			inst[1] = NOP;
			jump->addr -= sizeof(sljit_ins);
			return inst + 1;
		}
	}

	if (jump->flags & IS_COND) {
		if ((jump->flags & IS_MOVABLE) && (target_addr & ~(sljit_uw)0xfffffff) == ((jump->addr + 2 * sizeof(sljit_ins)) & ~(sljit_uw)0xfffffff)) {
			jump->flags |= PATCH_J;
			saved_inst = inst[0];
			inst[0] = inst[-1];
			inst[-1] = (saved_inst & 0xffff0000) | 3;
			inst[1] = J;
			inst[2] = NOP;
			return inst + 2;
		}
		else if ((target_addr & ~(sljit_uw)0xfffffff) == ((jump->addr + 3 * sizeof(sljit_ins)) & ~(sljit_uw)0xfffffff)) {
			jump->flags |= PATCH_J;
			inst[0] = (inst[0] & 0xffff0000) | 3;
			inst[1] = NOP;
			inst[2] = J;
			inst[3] = NOP;
			jump->addr += sizeof(sljit_ins);
			return inst + 3;
		}
	}
	else {
		/* J instuctions. */
		if ((jump->flags & IS_MOVABLE) && (target_addr & ~(sljit_uw)0xfffffff) == (jump->addr & ~(sljit_uw)0xfffffff)) {
			jump->flags |= PATCH_J;
			inst[0] = inst[-1];
			inst[-1] = (jump->flags & IS_JAL) ? JAL : J;
			jump->addr -= sizeof(sljit_ins);
			return inst;
		}

		if ((target_addr & ~(sljit_uw)0xfffffff) == ((jump->addr + sizeof(sljit_ins)) & ~(sljit_uw)0xfffffff)) {
			jump->flags |= PATCH_J;
			inst[0] = (jump->flags & IS_JAL) ? JAL : J;
			/* Keep inst[1] */
			return inst + 1;
		}
	}

	if (jump->flags & IS_COND)
		inst++;

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
preserve_addr:
	if (target_addr <= 0x7fffffff) {
		jump->flags |= PATCH_ABS32;
		if (jump->flags & IS_COND)
			inst[-1] -= 4;

		inst[2] = inst[0];
		inst[3] = inst[1];
		return inst + 3;
	}
	if (target_addr <= 0x7fffffffffffl) {
		jump->flags |= PATCH_ABS48;
		if (jump->flags & IS_COND)
			inst[-1] -= 2;

		inst[4] = inst[0];
		inst[5] = inst[1];
		return inst + 5;
	}
#endif

exit:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	inst[2] = inst[0];
	inst[3] = inst[1];
	return inst + 3;
#else
	inst[6] = inst[0];
	inst[7] = inst[1];
	return inst + 7;
#endif
}

#ifdef __GNUC__
static __attribute__ ((noinline)) void sljit_cache_flush(void* code, void* code_ptr)
{
	SLJIT_CACHE_FLUSH(code, code_ptr);
}
#endif

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)

static SLJIT_INLINE sljit_sw put_label_get_length(struct sljit_put_label *put_label, sljit_uw max_label)
{
	if (max_label < 0x80000000l) {
		put_label->flags = PATCH_ABS32;
		return 1;
	}

	if (max_label < 0x800000000000l) {
		put_label->flags = PATCH_ABS48;
		return 3;
	}

	put_label->flags = 0;
	return 5;
}

#endif /* SLJIT_CONFIG_MIPS_64 */

static SLJIT_INLINE void load_addr_to_reg(void *dst, sljit_u32 reg)
{
	struct sljit_jump *jump;
	struct sljit_put_label *put_label;
	sljit_uw flags;
	sljit_ins *inst;
	sljit_uw addr;

	if (reg != 0) {
		jump = (struct sljit_jump*)dst;
		flags = jump->flags;
		inst = (sljit_ins*)jump->addr;
		addr = (flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
	} else {
		put_label = (struct sljit_put_label*)dst;
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		flags = put_label->flags;
#endif
		inst = (sljit_ins*)put_label->addr;
		addr = put_label->label->addr;
		reg = *inst;
	}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	inst[0] = LUI | T(reg) | IMM(addr >> 16);
#else /* !SLJIT_CONFIG_MIPS_32 */
	if (flags & PATCH_ABS32) {
		SLJIT_ASSERT(addr < 0x80000000l);
		inst[0] = LUI | T(reg) | IMM(addr >> 16);
	}
	else if (flags & PATCH_ABS48) {
		SLJIT_ASSERT(addr < 0x800000000000l);
		inst[0] = LUI | T(reg) | IMM(addr >> 32);
		inst[1] = ORI | S(reg) | T(reg) | IMM((addr >> 16) & 0xffff);
		inst[2] = DSLL | T(reg) | D(reg) | SH_IMM(16);
		inst += 2;
	}
	else {
		inst[0] = LUI | T(reg) | IMM(addr >> 48);
		inst[1] = ORI | S(reg) | T(reg) | IMM((addr >> 32) & 0xffff);
		inst[2] = DSLL | T(reg) | D(reg) | SH_IMM(16);
		inst[3] = ORI | S(reg) | T(reg) | IMM((addr >> 16) & 0xffff);
		inst[4] = DSLL | T(reg) | D(reg) | SH_IMM(16);
		inst += 4;
	}
#endif /* SLJIT_CONFIG_MIPS_32 */

	inst[1] = ORI | S(reg) | T(reg) | IMM(addr & 0xffff);
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw word_count;
	sljit_uw next_addr;
	sljit_sw executable_offset;
	sljit_uw addr;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;
	struct sljit_put_label *put_label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	code = (sljit_ins*)SLJIT_MALLOC_EXEC(compiler->size * sizeof(sljit_ins), compiler->exec_allocator_data);
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

	code_ptr = code;
	word_count = 0;
	next_addr = 0;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	put_label = compiler->put_labels;

	do {
		buf_ptr = (sljit_ins*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			*code_ptr = *buf_ptr++;
			if (next_addr == word_count) {
				SLJIT_ASSERT(!label || label->size >= word_count);
				SLJIT_ASSERT(!jump || jump->addr >= word_count);
				SLJIT_ASSERT(!const_ || const_->addr >= word_count);
				SLJIT_ASSERT(!put_label || put_label->addr >= word_count);

				/* These structures are ordered by their address. */
				if (label && label->size == word_count) {
					label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					label->size = (sljit_uw)(code_ptr - code);
					label = label->next;
				}
				if (jump && jump->addr == word_count) {
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
					word_count += 2;
#else
					word_count += 6;
#endif
					jump->addr = (sljit_uw)(code_ptr - 1);
					code_ptr = detect_jump_type(jump, code, executable_offset);
					jump = jump->next;
				}
				if (const_ && const_->addr == word_count) {
					const_->addr = (sljit_uw)code_ptr;
					const_ = const_->next;
				}
				if (put_label && put_label->addr == word_count) {
					SLJIT_ASSERT(put_label->label);
					put_label->addr = (sljit_uw)code_ptr;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
					code_ptr += 1;
					word_count += 1;
#else
					code_ptr += put_label_get_length(put_label, (sljit_uw)(SLJIT_ADD_EXEC_OFFSET(code, executable_offset) + put_label->label->size));
					word_count += 5;
#endif
					put_label = put_label->next;
				}
				next_addr = compute_next_addr(label, jump, const_, put_label);
			}
			code_ptr++;
			word_count++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		label->addr = (sljit_uw)code_ptr;
		label->size = (sljit_uw)(code_ptr - code);
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(!put_label);
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);

	jump = compiler->jumps;
	while (jump) {
		do {
			addr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
			buf_ptr = (sljit_ins *)jump->addr;

			if (jump->flags & PATCH_B) {
				addr = (sljit_uw)((sljit_sw)(addr - (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset) - sizeof(sljit_ins)) >> 2);
				SLJIT_ASSERT((sljit_sw)addr <= SIMM_MAX && (sljit_sw)addr >= SIMM_MIN);
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((sljit_ins)addr & 0xffff);
				break;
			}
			if (jump->flags & PATCH_J) {
				SLJIT_ASSERT((addr & ~(sljit_uw)0xfffffff)
					== (((sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset) + sizeof(sljit_ins)) & ~(sljit_uw)0xfffffff));
				buf_ptr[0] |= (sljit_ins)(addr >> 2) & 0x03ffffff;
				break;
			}

			load_addr_to_reg(jump, PIC_ADDR_REG);
		} while (0);
		jump = jump->next;
	}

	put_label = compiler->put_labels;
	while (put_label) {
		load_addr_to_reg(put_label, 0);
		put_label = put_label->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (sljit_uw)(code_ptr - code) * sizeof(sljit_ins);

	code = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
	code_ptr = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

#ifndef __GNUC__
	SLJIT_CACHE_FLUSH(code, code_ptr);
#else
	/* GCC workaround for invalid code generation with -O2. */
	sljit_cache_flush(code, code_ptr);
#endif
	SLJIT_UPDATE_WX_FLAGS(code, code_ptr, 1);
	return code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	switch (feature_type) {
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
		&& (!defined(SLJIT_IS_FPU_AVAILABLE) || SLJIT_IS_FPU_AVAILABLE)
	case SLJIT_HAS_F64_AS_F32_PAIR:
		if (!cpu_feature_list)
			get_cpu_features();

		return (cpu_feature_list & CPU_FEATURE_FR) != 0;
#endif /* SLJIT_CONFIG_MIPS_32 && SLJIT_IS_FPU_AVAILABLE */
	case SLJIT_HAS_FPU:
		if (!cpu_feature_list)
			get_cpu_features();

		return (cpu_feature_list & CPU_FEATURE_FPU) != 0;
	case SLJIT_HAS_ZERO_REGISTER:
	case SLJIT_HAS_COPY_F32:
	case SLJIT_HAS_COPY_F64:
		return 1;
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_CMOV:
	case SLJIT_HAS_PREFETCH:
		return 1;

	case SLJIT_HAS_CTZ:
		return 2;
#endif /* SLJIT_MIPS_REV >= 1 */
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
	case SLJIT_HAS_REV:
	case SLJIT_HAS_ROT:
		return 1;
#endif /* SLJIT_MIPS_REV >= 2 */
	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	SLJIT_UNUSED_ARG(type);
	return 0;
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

/* Creates an index in data_transfer_insts array. */
#define LOAD_DATA	0x01
#define WORD_DATA	0x00
#define BYTE_DATA	0x02
#define HALF_DATA	0x04
#define INT_DATA	0x06
#define SIGNED_DATA	0x08
/* Separates integer and floating point registers */
#define GPR_REG		0x0f
#define DOUBLE_DATA	0x10
#define SINGLE_DATA	0x12

#define MEM_MASK	0x1f

#define ARG_TEST	0x00020
#define ALT_KEEP_CACHE	0x00040
#define CUMULATIVE_OP	0x00080
#define LOGICAL_OP	0x00100
#define IMM_OP		0x00200
#define MOVE_OP		0x00400
#define SRC2_IMM	0x00800

#define UNUSED_DEST	0x01000
#define REG_DEST	0x02000
#define REG1_SOURCE	0x04000
#define REG2_SOURCE	0x08000
#define SLOW_SRC1	0x10000
#define SLOW_SRC2	0x20000
#define SLOW_DEST	0x40000

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw);
static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 frame_size, sljit_ins *ins_ptr);

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define SELECT_OP(a, b)	(b)
#else
#define SELECT_OP(a, b)	(!(op & SLJIT_32) ? a : b)
#endif

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#include "sljitNativeMIPS_32.c"
#else
#include "sljitNativeMIPS_64.c"
#endif

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_ins base;
	sljit_s32 i, tmp, offset;
	sljit_s32 arg_count, word_arg_count, float_arg_count;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 1);
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if ((local_size & SSIZE_OF(sw)) != 0)
			local_size += SSIZE_OF(sw);
		local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	}

	local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
#else
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	local_size = (local_size + SLJIT_LOCALS_OFFSET + 31) & ~0x1f;
#endif
	compiler->local_size = local_size;

	offset = 0;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (!(options & SLJIT_ENTER_REG_ARG)) {
		tmp = arg_types >> SLJIT_ARG_SHIFT;
		arg_count = 0;

		while (tmp) {
			offset = arg_count;
			if ((tmp & SLJIT_ARG_MASK) == SLJIT_ARG_TYPE_F64) {
				if ((arg_count & 0x1) != 0)
					arg_count++;
				arg_count++;
			}

			arg_count++;
			tmp >>= SLJIT_ARG_SHIFT;
		}

		compiler->args_size = (sljit_uw)arg_count << 2;
		offset = (offset >= 4) ? (offset << 2) : 0;
	}
#endif /* SLJIT_CONFIG_MIPS_32 */

	if (local_size + offset <= -SIMM_MIN) {
		/* Frequent case. */
		FAIL_IF(push_inst(compiler, ADDIU_W | S(SLJIT_SP) | T(SLJIT_SP) | IMM(-local_size), DR(SLJIT_SP)));
		base = S(SLJIT_SP);
		offset = local_size - SSIZE_OF(sw);
	} else {
		FAIL_IF(load_immediate(compiler, OTHER_FLAG, local_size));
		FAIL_IF(push_inst(compiler, ADDU_W | S(SLJIT_SP) | TA(0) | D(TMP_REG2), DR(TMP_REG2)));
		FAIL_IF(push_inst(compiler, SUBU_W | S(SLJIT_SP) | TA(OTHER_FLAG) | D(SLJIT_SP), DR(SLJIT_SP)));
		base = S(TMP_REG2);
		offset = -SSIZE_OF(sw);
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		local_size = 0;
#endif
	}

	FAIL_IF(push_inst(compiler, STORE_W | base | TA(RETURN_ADDR_REG) | IMM(offset), UNMOVABLE_INS));

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STORE_W | base | T(i) | IMM(offset), MOVABLE_INS));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STORE_W | base | T(i) | IMM(offset), MOVABLE_INS));
	}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	/* This alignment is valid because offset is not used after storing FPU regs. */
	if ((offset & SSIZE_OF(sw)) != 0)
		offset -= SSIZE_OF(sw);
#endif

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, SDC1 | base | FT(i) | IMM(offset), MOVABLE_INS));
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, SDC1 | base | FT(i) | IMM(offset), MOVABLE_INS));
	}

	if (options & SLJIT_ENTER_REG_ARG)
		return SLJIT_SUCCESS;

	arg_types >>= SLJIT_ARG_SHIFT;
	arg_count = 0;
	word_arg_count = 0;
	float_arg_count = 0;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	/* The first maximum two floating point arguments are passed in floating point
	   registers if no integer argument precedes them. The first 16 byte data is
	   passed in four integer registers, the rest is placed onto the stack.
	   The floating point registers are also part of the first 16 byte data, so
	   their corresponding integer registers are not used when they are present. */

	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			float_arg_count++;
			if ((arg_count & 0x1) != 0)
				arg_count++;

			if (word_arg_count == 0 && float_arg_count <= 2) {
				if (float_arg_count == 1)
					FAIL_IF(push_inst(compiler, MOV_fmt(FMT_D) | FS(TMP_FREG1) | FD(SLJIT_FR0), MOVABLE_INS));
			} else if (arg_count < 4) {
				FAIL_IF(push_inst(compiler, MTC1 | TA(4 + arg_count) | FS(float_arg_count), MOVABLE_INS));
				switch (cpu_feature_list & CPU_FEATURE_FR) {
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
				case CPU_FEATURE_FR:
					FAIL_IF(push_inst(compiler, MTHC1 | TA(5 + arg_count) | FS(float_arg_count), MOVABLE_INS));
					break;
#endif /* SLJIT_MIPS_REV >= 2 */
				default:
					FAIL_IF(push_inst(compiler, MTC1 | TA(5 + arg_count) | FS(float_arg_count) | (1 << 11), MOVABLE_INS));
					break;
				}
			} else
				FAIL_IF(push_inst(compiler, LDC1 | base | FT(float_arg_count) | IMM(local_size + (arg_count << 2)), MOVABLE_INS));
			arg_count++;
			break;
		case SLJIT_ARG_TYPE_F32:
			float_arg_count++;

			if (word_arg_count == 0 && float_arg_count <= 2) {
				if (float_arg_count == 1)
					FAIL_IF(push_inst(compiler, MOV_fmt(FMT_S) | FS(TMP_FREG1) | FD(SLJIT_FR0), MOVABLE_INS));
			} else if (arg_count < 4)
				FAIL_IF(push_inst(compiler, MTC1 | TA(4 + arg_count) | FS(float_arg_count), MOVABLE_INS));
			else
				FAIL_IF(push_inst(compiler, LWC1 | base | FT(float_arg_count) | IMM(local_size + (arg_count << 2)), MOVABLE_INS));
			break;
		default:
			word_arg_count++;

			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				tmp = SLJIT_S0 - saved_arg_count;
				saved_arg_count++;
			} else if (word_arg_count != arg_count + 1 || arg_count == 0)
				tmp = word_arg_count;
			else
				break;

			if (arg_count < 4)
				FAIL_IF(push_inst(compiler, ADDU_W | SA(4 + arg_count) | TA(0) | D(tmp), DR(tmp)));
			else
				FAIL_IF(push_inst(compiler, LW | base | T(tmp) | IMM(local_size + (arg_count << 2)), DR(tmp)));
			break;
		}
		arg_count++;
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	SLJIT_ASSERT(compiler->args_size == (sljit_uw)arg_count << 2);
#else /* !SLJIT_CONFIG_MIPS_32 */
	while (arg_types) {
		arg_count++;
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			float_arg_count++;
			if (arg_count != float_arg_count)
				FAIL_IF(push_inst(compiler, MOV_fmt(FMT_D) | FS(arg_count) | FD(float_arg_count), MOVABLE_INS));
			else if (arg_count == 1)
				FAIL_IF(push_inst(compiler, MOV_fmt(FMT_D) | FS(TMP_FREG1) | FD(SLJIT_FR0), MOVABLE_INS));
			break;
		case SLJIT_ARG_TYPE_F32:
			float_arg_count++;
			if (arg_count != float_arg_count)
				FAIL_IF(push_inst(compiler, MOV_fmt(FMT_S) | FS(arg_count) | FD(float_arg_count), MOVABLE_INS));
			else if (arg_count == 1)
				FAIL_IF(push_inst(compiler, MOV_fmt(FMT_S) | FS(TMP_FREG1) | FD(SLJIT_FR0), MOVABLE_INS));
			break;
		default:
			word_arg_count++;

			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				tmp = SLJIT_S0 - saved_arg_count;
				saved_arg_count++;
			} else if (word_arg_count != arg_count || word_arg_count <= 1)
				tmp = word_arg_count;
			else
				break;

			FAIL_IF(push_inst(compiler, ADDU_W | SA(3 + arg_count) | TA(0) | D(tmp), DR(tmp)));
			break;
		}
		arg_types >>= SLJIT_ARG_SHIFT;
	}
#endif /* SLJIT_CONFIG_MIPS_32 */

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 1);
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if ((local_size & SSIZE_OF(sw)) != 0)
			local_size += SSIZE_OF(sw);
		local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	}

	compiler->local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
#else
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	compiler->local_size = (local_size + SLJIT_LOCALS_OFFSET + 31) & ~0x1f;
#endif
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 frame_size, sljit_ins *ins_ptr)
{
	sljit_s32 local_size, i, tmp, offset;
	sljit_s32 load_return_addr = (frame_size == 0);
	sljit_s32 scratches = compiler->scratches;
	sljit_s32 saveds = compiler->saveds;
	sljit_s32 fsaveds = compiler->fsaveds;
	sljit_s32 fscratches = compiler->fscratches;
	sljit_s32 kept_saveds_count = SLJIT_KEPT_SAVEDS_COUNT(compiler->options);

	SLJIT_ASSERT(frame_size == 1 || (frame_size & 0xf) == 0);
	frame_size &= ~0xf;

	local_size = compiler->local_size;

	tmp = GET_SAVED_REGISTERS_SIZE(scratches, saveds - kept_saveds_count, 1);
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if ((tmp & SSIZE_OF(sw)) != 0)
			tmp += SSIZE_OF(sw);
		tmp += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	}
#else
	tmp += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
#endif

	if (local_size <= SIMM_MAX) {
		if (local_size < frame_size) {
			FAIL_IF(push_inst(compiler, ADDIU_W | S(SLJIT_SP) | T(SLJIT_SP) | IMM(local_size - frame_size), DR(SLJIT_SP)));
			local_size = frame_size;
		}
	} else {
		if (tmp < frame_size)
			tmp = frame_size;

		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), local_size - tmp));
		FAIL_IF(push_inst(compiler, ADDU_W | S(SLJIT_SP) | T(TMP_REG1) | D(SLJIT_SP), DR(SLJIT_SP)));
		local_size = tmp;
	}

	SLJIT_ASSERT(local_size >= frame_size);

	offset = local_size - SSIZE_OF(sw);
	if (load_return_addr)
		FAIL_IF(push_inst(compiler, LOAD_W | S(SLJIT_SP) | TA(RETURN_ADDR_REG) | IMM(offset), RETURN_ADDR_REG));

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - kept_saveds_count; i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, LOAD_W | S(SLJIT_SP) | T(i) | IMM(offset), MOVABLE_INS));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, LOAD_W | S(SLJIT_SP) | T(i) | IMM(offset), MOVABLE_INS));
	}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	/* This alignment is valid because offset is not used after storing FPU regs. */
	if ((offset & SSIZE_OF(sw)) != 0)
		offset -= SSIZE_OF(sw);
#endif

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, LDC1 | S(SLJIT_SP) | FT(i) | IMM(offset), MOVABLE_INS));
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, LDC1 | S(SLJIT_SP) | FT(i) | IMM(offset), MOVABLE_INS));
	}

	if (local_size > frame_size)
		*ins_ptr = ADDIU_W | S(SLJIT_SP) | T(SLJIT_SP) | IMM(local_size - frame_size);
	else
		*ins_ptr = NOP;

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	emit_stack_frame_release(compiler, 0, &ins);

	FAIL_IF(push_inst(compiler, JR | SA(RETURN_ADDR_REG), UNMOVABLE_INS));
	return push_inst(compiler, ins, UNMOVABLE_INS);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, DR(PIC_ADDR_REG), src, srcw));
		src = PIC_ADDR_REG;
		srcw = 0;
	} else if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		FAIL_IF(push_inst(compiler, ADDU_W | S(src) | TA(0) | D(PIC_ADDR_REG), DR(PIC_ADDR_REG)));
		src = PIC_ADDR_REG;
		srcw = 0;
	}

	FAIL_IF(emit_stack_frame_release(compiler, 1, &ins));

	if (src != SLJIT_IMM) {
		FAIL_IF(push_inst(compiler, JR | S(src), UNMOVABLE_INS));
		return push_inst(compiler, ins, UNMOVABLE_INS);
	}

	if (ins != NOP)
		FAIL_IF(push_inst(compiler, ins, MOVABLE_INS));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define ARCH_32_64(a, b)	a
#else
#define ARCH_32_64(a, b)	b
#endif

static const sljit_ins data_transfer_insts[16 + 4] = {
/* u w s */ ARCH_32_64(HI(43) /* sw */, HI(63) /* sd */),
/* u w l */ ARCH_32_64(HI(35) /* lw */, HI(55) /* ld */),
/* u b s */ HI(40) /* sb */,
/* u b l */ HI(36) /* lbu */,
/* u h s */ HI(41) /* sh */,
/* u h l */ HI(37) /* lhu */,
/* u i s */ HI(43) /* sw */,
/* u i l */ ARCH_32_64(HI(35) /* lw */, HI(39) /* lwu */),

/* s w s */ ARCH_32_64(HI(43) /* sw */, HI(63) /* sd */),
/* s w l */ ARCH_32_64(HI(35) /* lw */, HI(55) /* ld */),
/* s b s */ HI(40) /* sb */,
/* s b l */ HI(32) /* lb */,
/* s h s */ HI(41) /* sh */,
/* s h l */ HI(33) /* lh */,
/* s i s */ HI(43) /* sw */,
/* s i l */ HI(35) /* lw */,

/* d   s */ HI(61) /* sdc1 */,
/* d   l */ HI(53) /* ldc1 */,
/* s   s */ HI(57) /* swc1 */,
/* s   l */ HI(49) /* lwc1 */,
};

#undef ARCH_32_64

/* reg_ar is an absoulute register! */

/* Can perform an operation using at most 1 instruction. */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw)
{
	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (!(arg & OFFS_REG_MASK) && argw <= SIMM_MAX && argw >= SIMM_MIN) {
		/* Works for both absoulte and relative addresses. */
		if (SLJIT_UNLIKELY(flags & ARG_TEST))
			return 1;
		FAIL_IF(push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(arg & REG_MASK)
			| TA(reg_ar) | IMM(argw), ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA)) ? reg_ar : MOVABLE_INS));
		return -1;
	}
	return 0;
}

#define TO_ARGW_HI(argw) (((argw) & ~0xffff) + (((argw) & 0x8000) ? 0x10000 : 0))

/* See getput_arg below.
   Note: can_cache is called only for binary operators. */
static sljit_s32 can_cache(sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	SLJIT_ASSERT((arg & SLJIT_MEM) && (next_arg & SLJIT_MEM));

	/* Simple operation except for updates. */
	if (arg & OFFS_REG_MASK) {
		argw &= 0x3;
		next_argw &= 0x3;
		if (argw && argw == next_argw && (arg == next_arg || (arg & OFFS_REG_MASK) == (next_arg & OFFS_REG_MASK)))
			return 1;
		return 0;
	}

	if (arg == next_arg) {
		if (((next_argw - argw) <= SIMM_MAX && (next_argw - argw) >= SIMM_MIN)
				|| TO_ARGW_HI(argw) == TO_ARGW_HI(next_argw))
			return 1;
		return 0;
	}

	return 0;
}

/* Emit the necessary instructions. See can_cache above. */
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 tmp_ar, base, delay_slot;
	sljit_sw offset, argw_hi;

	SLJIT_ASSERT(arg & SLJIT_MEM);
	if (!(next_arg & SLJIT_MEM)) {
		next_arg = 0;
		next_argw = 0;
	}

	/* Since tmp can be the same as base or offset registers,
	 * these might be unavailable after modifying tmp. */
	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA)) {
		tmp_ar = reg_ar;
		delay_slot = reg_ar;
	}
	else {
		tmp_ar = DR(TMP_REG1);
		delay_slot = MOVABLE_INS;
	}
	base = arg & REG_MASK;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		/* Using the cache. */
		if (argw == compiler->cache_argw) {
			if (arg == compiler->cache_arg)
				return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar), delay_slot);

			if ((SLJIT_MEM | (arg & OFFS_REG_MASK)) == compiler->cache_arg) {
				if (arg == next_arg && argw == (next_argw & 0x3)) {
					compiler->cache_arg = arg;
					compiler->cache_argw = argw;
					FAIL_IF(push_inst(compiler, ADDU_W | S(base) | T(TMP_REG3) | D(TMP_REG3), DR(TMP_REG3)));
					return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar), delay_slot);
				}
				FAIL_IF(push_inst(compiler, ADDU_W | S(base) | T(TMP_REG3) | DA(tmp_ar), tmp_ar));
				return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar), delay_slot);
			}
		}

		if (SLJIT_UNLIKELY(argw)) {
			compiler->cache_arg = SLJIT_MEM | (arg & OFFS_REG_MASK);
			compiler->cache_argw = argw;
			FAIL_IF(push_inst(compiler, SLL_W | T(OFFS_REG(arg)) | D(TMP_REG3) | SH_IMM(argw), DR(TMP_REG3)));
		}

		if (arg == next_arg && argw == (next_argw & 0x3)) {
			compiler->cache_arg = arg;
			compiler->cache_argw = argw;
			FAIL_IF(push_inst(compiler, ADDU_W | S(base) | T(!argw ? OFFS_REG(arg) : TMP_REG3) | D(TMP_REG3), DR(TMP_REG3)));
			tmp_ar = DR(TMP_REG3);
		}
		else
			FAIL_IF(push_inst(compiler, ADDU_W | S(base) | T(!argw ? OFFS_REG(arg) : TMP_REG3) | DA(tmp_ar), tmp_ar));
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar), delay_slot);
	}

	if (compiler->cache_arg == arg && argw - compiler->cache_argw <= SIMM_MAX && argw - compiler->cache_argw >= SIMM_MIN)
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar) | IMM(argw - compiler->cache_argw), delay_slot);

	if (compiler->cache_arg == SLJIT_MEM && (argw - compiler->cache_argw) <= SIMM_MAX && (argw - compiler->cache_argw) >= SIMM_MIN) {
		offset = argw - compiler->cache_argw;
	} else {
		compiler->cache_arg = SLJIT_MEM;

		argw_hi = TO_ARGW_HI(argw);

		if (next_arg && next_argw - argw <= SIMM_MAX && next_argw - argw >= SIMM_MIN && argw_hi != TO_ARGW_HI(next_argw)) {
			FAIL_IF(load_immediate(compiler, DR(TMP_REG3), argw));
			compiler->cache_argw = argw;
			offset = 0;
		} else {
			FAIL_IF(load_immediate(compiler, DR(TMP_REG3), argw_hi));
			compiler->cache_argw = argw_hi;
			offset = argw & 0xffff;
			argw = argw_hi;
		}
	}

	if (!base)
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar) | IMM(offset), delay_slot);

	if (arg == next_arg && next_argw - argw <= SIMM_MAX && next_argw - argw >= SIMM_MIN) {
		compiler->cache_arg = arg;
		FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | T(base) | D(TMP_REG3), DR(TMP_REG3)));
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar) | IMM(offset), delay_slot);
	}

	FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | T(base) | DA(tmp_ar), tmp_ar));
	return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar) | IMM(offset), delay_slot);
}

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw)
{
	sljit_s32 tmp_ar, base, delay_slot;

	if (getput_arg_fast(compiler, flags, reg_ar, arg, argw))
		return compiler->error;

	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA)) {
		tmp_ar = reg_ar;
		delay_slot = reg_ar;
	}
	else {
		tmp_ar = DR(TMP_REG1);
		delay_slot = MOVABLE_INS;
	}
	base = arg & REG_MASK;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (SLJIT_UNLIKELY(argw)) {
			FAIL_IF(push_inst(compiler, SLL_W | T(OFFS_REG(arg)) | DA(tmp_ar) | SH_IMM(argw), tmp_ar));
			FAIL_IF(push_inst(compiler, ADDU_W | SA(tmp_ar) | T(base) | DA(tmp_ar), tmp_ar));
		}
		else
			FAIL_IF(push_inst(compiler, ADDU_W | S(base) | T(OFFS_REG(arg)) | DA(tmp_ar), tmp_ar));
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar), delay_slot);
	}

	FAIL_IF(load_immediate(compiler, tmp_ar, TO_ARGW_HI(argw)));

	if (base != 0)
		FAIL_IF(push_inst(compiler, ADDU_W | SA(tmp_ar) | T(base) | DA(tmp_ar), tmp_ar));

	return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar) | IMM(argw), delay_slot);
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

#define EMIT_LOGICAL(op_imm, op_reg) \
	if (flags & SRC2_IMM) { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_imm | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_imm | S(src1) | T(dst) | IMM(src2), DR(dst))); \
	} \
	else { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_reg | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_reg | S(src1) | T(src2) | D(dst), DR(dst))); \
	}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)

#define EMIT_SHIFT(dimm, dimm32, imm, dv, v) \
	op_imm = (imm); \
	op_v = (v);

#else /* !SLJIT_CONFIG_MIPS_32 */


#define EMIT_SHIFT(dimm, dimm32, imm, dv, v) \
	op_dimm = (dimm); \
	op_dimm32 = (dimm32); \
	op_imm = (imm); \
	op_dv = (dv); \
	op_v = (v);

#endif /* SLJIT_CONFIG_MIPS_32 */

#if (!defined SLJIT_MIPS_REV || SLJIT_MIPS_REV < 1)

static sljit_s32 emit_clz_ctz(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw src)
{
	sljit_s32 is_clz = (GET_OPCODE(op) == SLJIT_CLZ);
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	sljit_ins word_size = (op & SLJIT_32) ? 32 : 64;
#else /* !SLJIT_CONFIG_MIPS_64 */
	sljit_ins word_size = 32;
#endif /* SLJIT_CONFIG_MIPS_64 */

	/* The TMP_REG2 is the next value. */
	if (src != TMP_REG2)
		FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src) | TA(0) | D(TMP_REG2), DR(TMP_REG2)));

	FAIL_IF(push_inst(compiler, BEQ | S(TMP_REG2) | TA(0) | IMM(is_clz ? 13 : 14), UNMOVABLE_INS));
	/* The OTHER_FLAG is the counter. Delay slot. */
	FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | SA(0) | TA(OTHER_FLAG) | IMM(word_size), OTHER_FLAG));

	if (!is_clz) {
		FAIL_IF(push_inst(compiler, ANDI | S(TMP_REG2) | T(TMP_REG1) | IMM(1), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, BNE | S(TMP_REG1) | TA(0) | IMM(11), UNMOVABLE_INS));
	} else
		FAIL_IF(push_inst(compiler, BLTZ | S(TMP_REG2) | TA(0) | IMM(11), UNMOVABLE_INS));

	/* Delay slot. */
	FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | SA(0) | TA(OTHER_FLAG) | IMM(0), OTHER_FLAG));

	/* The TMP_REG1 is the next shift. */
	FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | SA(0) | T(TMP_REG1) | IMM(word_size), DR(TMP_REG1)));

	FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(TMP_REG2) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
	FAIL_IF(push_inst(compiler, SELECT_OP(DSRL, SRL) | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(1), DR(TMP_REG1)));

	FAIL_IF(push_inst(compiler, (is_clz ? SELECT_OP(DSRLV, SRLV) : SELECT_OP(DSLLV, SLLV)) | S(TMP_REG1) | TA(EQUAL_FLAG) | D(TMP_REG2), DR(TMP_REG2)));
	FAIL_IF(push_inst(compiler, BNE | S(TMP_REG2) | TA(0) | IMM(-4), UNMOVABLE_INS));
	/* Delay slot. */
	FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));

	FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(TMP_REG1) | T(TMP_REG2) | IMM(-1), DR(TMP_REG2)));
	FAIL_IF(push_inst(compiler, (is_clz ? SELECT_OP(DSRLV, SRLV) : SELECT_OP(DSLLV, SLLV)) | S(TMP_REG2) | TA(EQUAL_FLAG) | D(TMP_REG2), DR(TMP_REG2)));

	FAIL_IF(push_inst(compiler, BEQ | S(TMP_REG2) | TA(0) | IMM(-7), UNMOVABLE_INS));
	/* Delay slot. */
	FAIL_IF(push_inst(compiler, OR | SA(OTHER_FLAG) | T(TMP_REG1) | DA(OTHER_FLAG), OTHER_FLAG));

	return push_inst(compiler, SELECT_OP(DADDU, ADDU) | SA(OTHER_FLAG) | TA(0) | D(dst), DR(dst));
}

#endif /* SLJIT_MIPS_REV < 1 */

static sljit_s32 emit_rev(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw src)
{
#if defined(SLJIT_CONFIG_MIPS_64) && SLJIT_CONFIG_MIPS_64
	int is_32 = (op & SLJIT_32);
#endif /* SLJIT_CONFIG_MIPS_64 */

	op = GET_OPCODE(op);
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#if defined(SLJIT_CONFIG_MIPS_64) && SLJIT_CONFIG_MIPS_64
	if (!is_32 && (op == SLJIT_REV)) {
		FAIL_IF(push_inst(compiler, DSBH | T(src) | D(dst), DR(dst)));
		return push_inst(compiler, DSHD | T(dst) | D(dst), DR(dst));
	}
	if (op != SLJIT_REV && src != TMP_REG2) {
		FAIL_IF(push_inst(compiler, SLL | T(src) | D(TMP_REG1), DR(TMP_REG1)));
		src = TMP_REG1;
	}
#endif /* SLJIT_CONFIG_MIPS_64 */
	FAIL_IF(push_inst(compiler, WSBH | T(src) | D(dst), DR(dst)));
	FAIL_IF(push_inst(compiler, ROTR | T(dst) | D(dst) | SH_IMM(16), DR(dst)));
#if defined(SLJIT_CONFIG_MIPS_64) && SLJIT_CONFIG_MIPS_64
	if (op == SLJIT_REV_U32 && dst != TMP_REG2 && dst != TMP_REG3)
		FAIL_IF(push_inst(compiler, DINSU | T(dst) | SA(0) | (31 << 11), DR(dst)));
#endif /* SLJIT_CONFIG_MIPS_64 */
#else /* SLJIT_MIPS_REV < 2 */
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (!is_32) {
		FAIL_IF(push_inst(compiler, DSRL32 | T(src) | D(TMP_REG1) | SH_IMM(0), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, ORI | SA(0) | TA(OTHER_FLAG) | 0xffff, OTHER_FLAG));
		FAIL_IF(push_inst(compiler, DSLL32 | T(src) | D(dst) | SH_IMM(0), DR(dst)));
		FAIL_IF(push_inst(compiler, DSLL32 | TA(OTHER_FLAG) | DA(OTHER_FLAG) | SH_IMM(0), OTHER_FLAG));
		FAIL_IF(push_inst(compiler, OR | S(dst) | T(TMP_REG1) | D(dst), DR(dst)));

		FAIL_IF(push_inst(compiler, DSRL | T(dst) | D(TMP_REG1) | SH_IMM(16), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, ORI | SA(OTHER_FLAG) | TA(OTHER_FLAG) | 0xffff, OTHER_FLAG));
		FAIL_IF(push_inst(compiler, AND | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
		FAIL_IF(push_inst(compiler, AND | S(TMP_REG1) | TA(OTHER_FLAG) | D(TMP_REG1), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, DSLL | TA(OTHER_FLAG) | DA(EQUAL_FLAG) | SH_IMM(8), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, DSLL | T(dst) | D(dst) | SH_IMM(16), DR(dst)));
		FAIL_IF(push_inst(compiler, XOR | SA(OTHER_FLAG) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		FAIL_IF(push_inst(compiler, OR | S(dst) | T(TMP_REG1) | D(dst), DR(dst)));

		FAIL_IF(push_inst(compiler, DSRL | T(dst) | D(TMP_REG1) | SH_IMM(8), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, AND | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
		FAIL_IF(push_inst(compiler, AND | S(TMP_REG1) | TA(OTHER_FLAG) | D(TMP_REG1), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, DSLL | T(dst) | D(dst) | SH_IMM(8), DR(dst)));
		return push_inst(compiler, OR | S(dst) | T(TMP_REG1) | D(dst), DR(dst));
	}

	if (op != SLJIT_REV && src != TMP_REG2) {
		FAIL_IF(push_inst(compiler, SLL | T(src) | D(TMP_REG2) | SH_IMM(0), DR(TMP_REG2)));
		src = TMP_REG2;
	}
#endif /* SLJIT_CONFIG_MIPS_64 */

	FAIL_IF(push_inst(compiler, SRL | T(src) | D(TMP_REG1) | SH_IMM(16), DR(TMP_REG1)));
	FAIL_IF(push_inst(compiler, LUI | TA(OTHER_FLAG) | 0xff, OTHER_FLAG));
	FAIL_IF(push_inst(compiler, SLL | T(src) | D(dst) | SH_IMM(16), DR(dst)));
	FAIL_IF(push_inst(compiler, ORI | SA(OTHER_FLAG) | TA(OTHER_FLAG) | 0xff, OTHER_FLAG));
	FAIL_IF(push_inst(compiler, OR | S(dst) | T(TMP_REG1) | D(dst), DR(dst)));

	FAIL_IF(push_inst(compiler, SRL | T(dst) | D(TMP_REG1) | SH_IMM(8), DR(TMP_REG1)));
	FAIL_IF(push_inst(compiler, AND | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
	FAIL_IF(push_inst(compiler, AND | S(TMP_REG1) | TA(OTHER_FLAG) | D(TMP_REG1), DR(TMP_REG1)));
	FAIL_IF(push_inst(compiler, SLL | T(dst) | D(dst) | SH_IMM(8), DR(dst)));
	FAIL_IF(push_inst(compiler, OR | S(dst) | T(TMP_REG1) | D(dst), DR(dst)));

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (op == SLJIT_REV_U32 && dst != TMP_REG2 && dst != TMP_REG3) {
		FAIL_IF(push_inst(compiler, DSLL32 | T(dst) | D(dst) | SH_IMM(0), DR(dst)));
		FAIL_IF(push_inst(compiler, DSRL32 | T(dst) | D(dst) | SH_IMM(0), DR(dst)));
	}
#endif /* SLJIT_CONFIG_MIPS_64 */
#endif /* SLJIT_MIPR_REV >= 2 */
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_rev16(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw src)
{
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
#if defined(SLJIT_CONFIG_MIPS_32) && SLJIT_CONFIG_MIPS_32
	FAIL_IF(push_inst(compiler, WSBH | T(src) | D(dst), DR(dst)));
#else /* !SLJIT_CONFIG_MIPS_32 */
	FAIL_IF(push_inst(compiler, DSBH | T(src) | D(dst), DR(dst)));
#endif /* SLJIT_CONFIG_MIPS_32 */
	if (GET_OPCODE(op) == SLJIT_REV_U16)
		return push_inst(compiler, ANDI | S(dst) | T(dst) | 0xffff, DR(dst));
	else
		return push_inst(compiler, SEH | T(dst) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 2 */
	FAIL_IF(push_inst(compiler, SELECT_OP(DSRL, SRL) | T(src) | D(TMP_REG1) | SH_IMM(8), DR(TMP_REG1)));
	FAIL_IF(push_inst(compiler, SELECT_OP(DSLL32, SLL) | T(src) | D(dst) | SH_IMM(24), DR(dst)));
	FAIL_IF(push_inst(compiler, ANDI | S(TMP_REG1) | T(TMP_REG1) | 0xff, DR(TMP_REG1)));
	FAIL_IF(push_inst(compiler, (GET_OPCODE(op) == SLJIT_REV_U16 ? SELECT_OP(DSRL32, SRL) : SELECT_OP(DSRA32, SRA)) | T(dst) | D(dst) | SH_IMM(16), DR(dst)));
	return push_inst(compiler, OR | S(dst) | T(TMP_REG1) | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */
}

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_s32 is_overflow, is_carry, carry_src_ar, is_handled;
	sljit_ins op_imm, op_v;
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	sljit_ins ins, op_dimm, op_dimm32, op_dv;
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src2) | TA(0) | D(dst), DR(dst));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, ANDI | S(src2) | T(dst) | IMM(0xff), DR(dst));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
			return push_inst(compiler, SEB | T(src2) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 2 */
			FAIL_IF(push_inst(compiler, SLL | T(src2) | D(dst) | SH_IMM(24), DR(dst)));
			return push_inst(compiler, SRA | T(dst) | D(dst) | SH_IMM(24), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */
#else /* !SLJIT_CONFIG_MIPS_32 */
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
			if (op & SLJIT_32)
				return push_inst(compiler, SEB | T(src2) | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */
			FAIL_IF(push_inst(compiler, DSLL32 | T(src2) | D(dst) | SH_IMM(24), DR(dst)));
			return push_inst(compiler, DSRA32 | T(dst) | D(dst) | SH_IMM(24), DR(dst));
#endif /* SLJIT_CONFIG_MIPS_32 */
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, ANDI | S(src2) | T(dst) | IMM(0xffff), DR(dst));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
			return push_inst(compiler, SEH | T(src2) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 2 */
			FAIL_IF(push_inst(compiler, SLL | T(src2) | D(dst) | SH_IMM(16), DR(dst)));
			return push_inst(compiler, SRA | T(dst) | D(dst) | SH_IMM(16), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */
#else /* !SLJIT_CONFIG_MIPS_32 */
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
			if (op & SLJIT_32)
				return push_inst(compiler, SEH | T(src2) | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */
			FAIL_IF(push_inst(compiler, DSLL32 | T(src2) | D(dst) | SH_IMM(16), DR(dst)));
			return push_inst(compiler, DSRA32 | T(dst) | D(dst) | SH_IMM(16), DR(dst));
#endif /* SLJIT_CONFIG_MIPS_32 */
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	case SLJIT_MOV_U32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM) && !(op & SLJIT_32));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
			if (dst == src2)
				return push_inst(compiler, DINSU | T(src2) | SA(0) | (31 << 11), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */
			FAIL_IF(push_inst(compiler, DSLL32 | T(src2) | D(dst) | SH_IMM(0), DR(dst)));
			return push_inst(compiler, DSRL32 | T(dst) | D(dst) | SH_IMM(0), DR(dst));
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM) && !(op & SLJIT_32));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			return push_inst(compiler, SLL | T(src2) | D(dst) | SH_IMM(0), DR(dst));
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;
#endif /* SLJIT_CONFIG_MIPS_64 */

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
		return push_inst(compiler, SELECT_OP(DCLZ, CLZ) | S(src2) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 6 */
		return push_inst(compiler, SELECT_OP(DCLZ, CLZ) | S(src2) | T(dst) | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 6 */
	case SLJIT_CTZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | SA(0) | T(src2) | D(TMP_REG1), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, AND | S(src2) | T(TMP_REG1) | D(dst), DR(dst)));
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
		FAIL_IF(push_inst(compiler, SELECT_OP(DCLZ, CLZ) | S(dst) | D(dst), DR(dst)));
#else /* SLJIT_MIPS_REV < 6 */
		FAIL_IF(push_inst(compiler, SELECT_OP(DCLZ, CLZ) | S(dst) | T(dst) | D(dst), DR(dst)));
#endif /* SLJIT_MIPS_REV >= 6 */
		FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(dst) | T(TMP_REG1) | IMM(SELECT_OP(-64, -32)), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, SELECT_OP(DSRL32, SRL) | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(SELECT_OP(26, 27)), DR(TMP_REG1)));
		return push_inst(compiler, XOR | S(dst) | T(TMP_REG1) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 1 */
	case SLJIT_CLZ:
	case SLJIT_CTZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return emit_clz_ctz(compiler, op, dst, src2);
#endif /* SLJIT_MIPS_REV >= 1 */

	case SLJIT_REV:
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM) && src2 != TMP_REG1 && dst != TMP_REG1);
		return emit_rev(compiler, op, dst, src2);

	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return emit_rev16(compiler, op, dst, src2);

	case SLJIT_ADD:
		/* Overflow computation (both add and sub): overflow = src1_sign ^ src2_sign ^ result_sign ^ carry_flag */
		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		carry_src_ar = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, OR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
				else
					FAIL_IF(push_inst(compiler, NOR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(src2), DR(dst)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			if (is_overflow || carry_src_ar != 0) {
				if (src1 != dst)
					carry_src_ar = DR(src1);
				else if (src2 != dst)
					carry_src_ar = DR(src2);
				else {
					FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | TA(0) | DA(OTHER_FLAG), OTHER_FLAG));
					carry_src_ar = OTHER_FLAG;
				}
			}

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		/* Carry is zero if a + b >= a or a + b >= b, otherwise it is 1. */
		if (is_overflow || carry_src_ar != 0) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLTIU | S(dst) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
			else
				FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(carry_src_ar) | DA(OTHER_FLAG), OTHER_FLAG));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, XOR | S(dst) | TA(EQUAL_FLAG) | D(TMP_REG1), DR(TMP_REG1)));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(dst) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, SELECT_OP(DSRL32, SRL) | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(31), DR(TMP_REG1)));
		return push_inst(compiler, XOR | S(TMP_REG1) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

	case SLJIT_ADDC:
		carry_src_ar = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(src2), DR(dst)));
		} else {
			if (carry_src_ar != 0) {
				if (src1 != dst)
					carry_src_ar = DR(src1);
				else if (src2 != dst)
					carry_src_ar = DR(src2);
				else {
					FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
					carry_src_ar = EQUAL_FLAG;
				}
			}

			FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		/* Carry is zero if a + b >= a or a + b >= b, otherwise it is 1. */
		if (carry_src_ar != 0) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLTIU | S(dst) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));
			else
				FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(carry_src_ar) | DA(EQUAL_FLAG), EQUAL_FLAG));
		}

		FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));

		if (carry_src_ar == 0)
			return SLJIT_SUCCESS;

		/* Set ULESS_FLAG (dst == 0) && (OTHER_FLAG == 1). */
		FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		/* Set carry flag. */
		return push_inst(compiler, OR | SA(OTHER_FLAG) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

	case SLJIT_SUB:
		if ((flags & SRC2_IMM) && src2 == SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(TMP_REG2) | IMM(src2), DR(TMP_REG2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_handled = 0;

		if (flags & SRC2_IMM) {
			if (GET_FLAG_TYPE(op) == SLJIT_LESS) {
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
				is_handled = 1;
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_SIG_LESS) {
				FAIL_IF(push_inst(compiler, SLTI | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
				is_handled = 1;
			}
		}

		if (!is_handled && GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL) {
			is_handled = 1;

			if (flags & SRC2_IMM) {
				FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(TMP_REG2) | IMM(src2), DR(TMP_REG2)));
				src2 = TMP_REG2;
				flags &= ~SRC2_IMM;
			}

			switch (GET_FLAG_TYPE(op)) {
			case SLJIT_LESS:
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
				break;
			case SLJIT_GREATER:
				FAIL_IF(push_inst(compiler, SLTU | S(src2) | T(src1) | DA(OTHER_FLAG), OTHER_FLAG));
				break;
			case SLJIT_SIG_LESS:
				FAIL_IF(push_inst(compiler, SLT | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
				break;
			case SLJIT_SIG_GREATER:
				FAIL_IF(push_inst(compiler, SLT | S(src2) | T(src1) | DA(OTHER_FLAG), OTHER_FLAG));
				break;
			}
		}

		if (is_handled) {
			if (flags & SRC2_IMM) {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | TA(EQUAL_FLAG) | IMM(-src2), EQUAL_FLAG));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(-src2), DR(dst));
			}
			else {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | D(dst), DR(dst));
			}
			return SLJIT_SUCCESS;
		}

		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		is_carry = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, OR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
				else
					FAIL_IF(push_inst(compiler, NOR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | TA(EQUAL_FLAG) | IMM(-src2), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(-src2), DR(dst)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, XOR | S(dst) | TA(EQUAL_FLAG) | D(TMP_REG1), DR(TMP_REG1)));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(dst) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, SELECT_OP(DSRL32, SRL) | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(31), DR(TMP_REG1)));
		return push_inst(compiler, XOR | S(TMP_REG1) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

	case SLJIT_SUBC:
		if ((flags & SRC2_IMM) && src2 == SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(TMP_REG2) | IMM(src2), DR(TMP_REG2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_carry = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));

			FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(-src2), DR(dst)));
		}
		else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | D(TMP_REG1), DR(TMP_REG1)));

		FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));

		if (!is_carry)
			return SLJIT_SUCCESS;

		return push_inst(compiler, OR | SA(EQUAL_FLAG) | T(TMP_REG1) | DA(OTHER_FLAG), OTHER_FLAG);

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & SRC2_IMM));

		if (GET_FLAG_TYPE(op) != SLJIT_OVERFLOW) {
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
			return push_inst(compiler, SELECT_OP(DMUL, MUL) | S(src1) | T(src2) | D(dst), DR(dst));
#elif (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
			return push_inst(compiler, MUL | S(src1) | T(src2) | D(dst), DR(dst));
#else /* !SLJIT_CONFIG_MIPS_32 */
			if (op & SLJIT_32)
				return push_inst(compiler, MUL | S(src1) | T(src2) | D(dst), DR(dst));
			FAIL_IF(push_inst(compiler, DMULT | S(src1) | T(src2), MOVABLE_INS));
			return push_inst(compiler, MFLO | D(dst), DR(dst));
#endif /* SLJIT_CONFIG_MIPS_32 */
#else /* SLJIT_MIPS_REV < 1 */
			FAIL_IF(push_inst(compiler, SELECT_OP(DMULT, MULT) | S(src1) | T(src2), MOVABLE_INS));
			return push_inst(compiler, MFLO | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 6 */
		}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
		FAIL_IF(push_inst(compiler, SELECT_OP(DMUL, MUL) | S(src1) | T(src2) | D(dst), DR(dst)));
		FAIL_IF(push_inst(compiler, SELECT_OP(DMUH, MUH) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
#else /* SLJIT_MIPS_REV < 6 */
		FAIL_IF(push_inst(compiler, SELECT_OP(DMULT, MULT) | S(src1) | T(src2), MOVABLE_INS));
		FAIL_IF(push_inst(compiler, MFHI | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, MFLO | D(dst), DR(dst)));
#endif /* SLJIT_MIPS_REV >= 6 */
		FAIL_IF(push_inst(compiler, SELECT_OP(DSRA32, SRA) | T(dst) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG));
		return push_inst(compiler, SELECT_OP(DSUBU, SUBU) | SA(EQUAL_FLAG) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

	case SLJIT_AND:
		EMIT_LOGICAL(ANDI, AND);
		return SLJIT_SUCCESS;

	case SLJIT_OR:
		EMIT_LOGICAL(ORI, OR);
		return SLJIT_SUCCESS;

	case SLJIT_XOR:
		if (!(flags & LOGICAL_OP)) {
			SLJIT_ASSERT((flags & SRC2_IMM) && src2 == -1);
			if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, NOR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
			if (!(flags & UNUSED_DEST))
				FAIL_IF(push_inst(compiler, NOR | S(src1) | T(src1) | D(dst), DR(dst)));
			return SLJIT_SUCCESS;
		}
		EMIT_LOGICAL(XORI, XOR);
		return SLJIT_SUCCESS;

	case SLJIT_SHL:
	case SLJIT_MSHL:
		EMIT_SHIFT(DSLL, DSLL32, SLL, DSLLV, SLLV);
		break;

	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		EMIT_SHIFT(DSRL, DSRL32, SRL, DSRLV, SRLV);
		break;

	case SLJIT_ASHR:
	case SLJIT_MASHR:
		EMIT_SHIFT(DSRA, DSRA32, SRA, DSRAV, SRAV);
		break;

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
	case SLJIT_ROTL:
		if ((flags & SRC2_IMM) || src2 == 0) {
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
			src2 = -src2 & 0x1f;
#else /* !SLJIT_CONFIG_MIPS_32 */
			src2 = -src2 & ((op & SLJIT_32) ? 0x1f : 0x3f);
#endif /* SLJIT_CONFIG_MIPS_32 */
		} else {
			FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | SA(0) | T(src2) | D(TMP_REG2), DR(TMP_REG2)));
			src2 = TMP_REG2;
		}
		/* fallthrough */

	case SLJIT_ROTR:
		EMIT_SHIFT(DROTR, DROTR32, ROTR, DROTRV, ROTRV);
		break;
#else /* SLJIT_MIPS_REV < 1 */
	case SLJIT_ROTL:
	case SLJIT_ROTR:
		if (flags & SRC2_IMM) {
			SLJIT_ASSERT(src2 != 0);
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
			if (!(op & SLJIT_32)) {
				if (GET_OPCODE(op) == SLJIT_ROTL)
					op_imm = ((src2 < 32) ? DSLL : DSLL32);
				else
					op_imm = ((src2 < 32) ? DSRL : DSRL32);

				FAIL_IF(push_inst(compiler, op_imm | T(src1) | DA(OTHER_FLAG) | (((sljit_ins)src2 & 0x1f) << 6), OTHER_FLAG));

				src2 = 64 - src2;
				if (GET_OPCODE(op) == SLJIT_ROTL)
					op_imm = ((src2 < 32) ? DSRL : DSRL32);
				else
					op_imm = ((src2 < 32) ? DSLL : DSLL32);

				FAIL_IF(push_inst(compiler, op_imm | T(src1) | D(dst) | (((sljit_ins)src2 & 0x1f) << 6), DR(dst)));
				return push_inst(compiler, OR | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst));
			}
#endif /* SLJIT_CONFIG_MIPS_64 */

			op_imm = (GET_OPCODE(op) == SLJIT_ROTL) ? SLL : SRL;
			FAIL_IF(push_inst(compiler, op_imm | T(src1) | DA(OTHER_FLAG) | ((sljit_ins)src2 << 6), OTHER_FLAG));

			src2 = 32 - src2;
			op_imm = (GET_OPCODE(op) == SLJIT_ROTL) ? SRL : SLL;
			FAIL_IF(push_inst(compiler, op_imm | T(src1) | D(dst) | (((sljit_ins)src2 & 0x1f) << 6), DR(dst)));
			return push_inst(compiler, OR | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst));
		}

		if (src2 == 0) {
			if (dst != src1)
				return push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | TA(0) | D(dst), DR(dst));
			return SLJIT_SUCCESS;
		}

		FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | SA(0) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (!(op & SLJIT_32)) {
			op_v = (GET_OPCODE(op) == SLJIT_ROTL) ? DSLLV : DSRLV;
			FAIL_IF(push_inst(compiler, op_v | S(src2) | T(src1) | DA(OTHER_FLAG), OTHER_FLAG));
			op_v = (GET_OPCODE(op) == SLJIT_ROTL) ? DSRLV : DSLLV;
			FAIL_IF(push_inst(compiler, op_v | SA(EQUAL_FLAG) | T(src1) | D(dst), DR(dst)));
			return push_inst(compiler, OR | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst));
		}
#endif /* SLJIT_CONFIG_MIPS_64 */

		op_v = (GET_OPCODE(op) == SLJIT_ROTL) ? SLLV : SRLV;
		FAIL_IF(push_inst(compiler, op_v | S(src2) | T(src1) | DA(OTHER_FLAG), OTHER_FLAG));
		op_v = (GET_OPCODE(op) == SLJIT_ROTL) ? SRLV : SLLV;
		FAIL_IF(push_inst(compiler, op_v | SA(EQUAL_FLAG) | T(src1) | D(dst), DR(dst)));
		return push_inst(compiler, OR | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 2 */

	default:
		SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if ((flags & SRC2_IMM) || src2 == 0) {
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, op_imm | T(src1) | DA(EQUAL_FLAG) | SH_IMM(src2), EQUAL_FLAG));

		if (flags & UNUSED_DEST)
			return SLJIT_SUCCESS;
		return push_inst(compiler, op_imm | T(src1) | D(dst) | SH_IMM(src2), DR(dst));
	}

	if (op & SLJIT_SET_Z)
		FAIL_IF(push_inst(compiler, op_v | S(src2) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));

	if (flags & UNUSED_DEST)
		return SLJIT_SUCCESS;
	return push_inst(compiler, op_v | S(src2) | T(src1) | D(dst), DR(dst));
#else /* !SLJIT_CONFIG_MIPS_32 */
	if ((flags & SRC2_IMM) || src2 == 0) {
		if (src2 >= 32) {
			SLJIT_ASSERT(!(op & SLJIT_32));
			ins = op_dimm32;
			src2 -= 32;
		}
		else
			ins = (op & SLJIT_32) ? op_imm : op_dimm;

		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, ins | T(src1) | DA(EQUAL_FLAG) | SH_IMM(src2), EQUAL_FLAG));

		if (flags & UNUSED_DEST)
			return SLJIT_SUCCESS;
		return push_inst(compiler, ins | T(src1) | D(dst) | SH_IMM(src2), DR(dst));
	}

	ins = (op & SLJIT_32) ? op_v : op_dv;
	if (op & SLJIT_SET_Z)
		FAIL_IF(push_inst(compiler, ins | S(src2) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));

	if (flags & UNUSED_DEST)
		return SLJIT_SUCCESS;
	return push_inst(compiler, ins | S(src2) | T(src1) | D(dst), DR(dst));
#endif /* SLJIT_CONFIG_MIPS_32 */
}

#define CHECK_IMM(flags, srcw) \
	((!((flags) & LOGICAL_OP) && ((srcw) <= SIMM_MAX && (srcw) >= SIMM_MIN)) \
		|| (((flags) & LOGICAL_OP) && !((srcw) & ~UIMM_MAX)))

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* arg1 goes to TMP_REG1 or src reg
	   arg2 goes to TMP_REG2, imm or src reg
	   TMP_REG3 can be used for caching
	   result goes to TMP_REG2, so put result can use TMP_REG1 and TMP_REG3. */
	sljit_s32 dst_r = TMP_REG2;
	sljit_s32 src1_r;
	sljit_sw src2_r = 0;
	sljit_s32 sugg_src2_r = TMP_REG2;

	if (!(flags & ALT_KEEP_CACHE)) {
		compiler->cache_arg = 0;
		compiler->cache_argw = 0;
	}

	if (dst == 0) {
		SLJIT_ASSERT(HAS_FLAGS(op));
		flags |= UNUSED_DEST;
		dst = TMP_REG2;
	}
	else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (flags & MOVE_OP)
			sugg_src2_r = dst_r;
	}
	else if ((dst & SLJIT_MEM) && !getput_arg_fast(compiler, flags | ARG_TEST, DR(TMP_REG1), dst, dstw))
		flags |= SLOW_DEST;

	if (flags & IMM_OP) {
		if (src2 == SLJIT_IMM && src2w != 0 && CHECK_IMM(flags, src2w)) {
			flags |= SRC2_IMM;
			src2_r = src2w;
		} else if ((flags & CUMULATIVE_OP) && src1 == SLJIT_IMM && src1w != 0 && CHECK_IMM(flags, src1w)) {
			flags |= SRC2_IMM;
			src2_r = src1w;

			/* And swap arguments. */
			src1 = src2;
			src1w = src2w;
			src2 = SLJIT_IMM;
			/* src2w = src2_r unneeded. */
		}
	}

	/* Source 1. */
	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	}
	else if (src1 == SLJIT_IMM) {
		if (src1w) {
			FAIL_IF(load_immediate(compiler, DR(TMP_REG1), src1w));
			src1_r = TMP_REG1;
		}
		else
			src1_r = 0;
	}
	else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, DR(TMP_REG1), src1, src1w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC1;
		src1_r = TMP_REG1;
	}

	/* Source 2. */
	if (FAST_IS_REG(src2)) {
		src2_r = src2;
		flags |= REG2_SOURCE;
		if ((flags & (REG_DEST | MOVE_OP)) == MOVE_OP)
			dst_r = (sljit_s32)src2_r;
	}
	else if (src2 == SLJIT_IMM) {
		if (!(flags & SRC2_IMM)) {
			if (src2w) {
				FAIL_IF(load_immediate(compiler, DR(sugg_src2_r), src2w));
				src2_r = sugg_src2_r;
			}
			else {
				src2_r = 0;
				if (flags & MOVE_OP) {
					if (dst & SLJIT_MEM)
						dst_r = 0;
					else
						op = SLJIT_MOV;
				}
			}
		}
	}
	else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, DR(sugg_src2_r), src2, src2w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC2;
		src2_r = sugg_src2_r;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		SLJIT_ASSERT(src2_r == TMP_REG2);
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, DR(TMP_REG2), src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, DR(TMP_REG1), src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, DR(TMP_REG1), src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, DR(TMP_REG2), src2, src2w, dst, dstw));
		}
	}
	else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, DR(TMP_REG1), src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, DR(sugg_src2_r), src2, src2w, dst, dstw));

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (dst & SLJIT_MEM) {
		if (!(flags & SLOW_DEST)) {
			getput_arg_fast(compiler, flags, DR(dst_r), dst, dstw);
			return compiler->error;
		}
		return getput_arg(compiler, flags, DR(dst_r), dst, dstw, 0, 0);
	}

	return SLJIT_SUCCESS;
}

#undef CHECK_IMM

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	sljit_s32 int_op = op & SLJIT_32;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	op = GET_OPCODE(op);
	switch (op) {
	case SLJIT_BREAKPOINT:
		return push_inst(compiler, BREAK, UNMOVABLE_INS);
	case SLJIT_NOP:
		return push_inst(compiler, NOP, UNMOVABLE_INS);
	case SLJIT_LMUL_UW:
	case SLJIT_LMUL_SW:
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? DMULU : DMUL) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? DMUHU : DMUH) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
#else /* !SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? MULU : MUL) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? MUHU : MUH) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | TA(0) | D(SLJIT_R0), DR(SLJIT_R0)));
		return push_inst(compiler, ADDU_W | S(TMP_REG1) | TA(0) | D(SLJIT_R1), DR(SLJIT_R1));
#else /* SLJIT_MIPS_REV < 6 */
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? DMULTU : DMULT) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
#else /* !SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? MULTU : MULT) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, MFLO | D(SLJIT_R0), DR(SLJIT_R0)));
		return push_inst(compiler, MFHI | D(SLJIT_R1), DR(SLJIT_R1));
#endif /* SLJIT_MIPS_REV >= 6 */
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		SLJIT_COMPILE_ASSERT((SLJIT_DIVMOD_UW & 0x2) == 0 && SLJIT_DIV_UW - 0x2 == SLJIT_DIVMOD_UW, bad_div_opcode_assignments);
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (int_op) {
			FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DIVU : DIV) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
			FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? MODU : MOD) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
		}
		else {
			FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DDIVU : DDIV) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
			FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DMODU : DMOD) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
		}
#else /* !SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DIVU : DIV) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
		FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? MODU : MOD) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | TA(0) | D(SLJIT_R0), DR(SLJIT_R0)));
		return (op >= SLJIT_DIV_UW) ? SLJIT_SUCCESS : push_inst(compiler, ADDU_W | S(TMP_REG1) | TA(0) | D(SLJIT_R1), DR(SLJIT_R1));
#else /* SLJIT_MIPS_REV < 6 */
#if !(defined SLJIT_MIPS_REV)
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* !SLJIT_MIPS_REV */
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (int_op)
			FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DIVU : DIV) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
		else
			FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DDIVU : DDIV) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
#else /* !SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, ((op | 0x2) == SLJIT_DIV_UW ? DIVU : DIV) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, MFLO | D(SLJIT_R0), DR(SLJIT_R0)));
		return (op >= SLJIT_DIV_UW) ? SLJIT_SUCCESS : push_inst(compiler, MFHI | D(SLJIT_R1), DR(SLJIT_R1));
#endif /* SLJIT_MIPS_REV >= 6 */
	case SLJIT_ENDBR:
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
static sljit_s32 emit_prefetch(struct sljit_compiler *compiler,
        sljit_s32 src, sljit_sw srcw)
{
	if (!(src & OFFS_REG_MASK)) {
		if (srcw <= SIMM_MAX && srcw >= SIMM_MIN)
			return push_inst(compiler, PREF | S(src & REG_MASK) | IMM(srcw), MOVABLE_INS);

		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), srcw));
		return push_inst(compiler, PREFX | S(src & REG_MASK) | T(TMP_REG1), MOVABLE_INS);
	}

	srcw &= 0x3;

	if (SLJIT_UNLIKELY(srcw != 0)) {
		FAIL_IF(push_inst(compiler, SLL_W | T(OFFS_REG(src)) | D(TMP_REG1) | SH_IMM(srcw), DR(TMP_REG1)));
		return push_inst(compiler, PREFX | S(src & REG_MASK) | T(TMP_REG1), MOVABLE_INS);
	}

	return push_inst(compiler, PREFX | S(src & REG_MASK) | T(OFFS_REG(src)), MOVABLE_INS);
}
#endif /* SLJIT_MIPS_REV >= 1 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (op & SLJIT_32)
		flags = INT_DATA | SIGNED_DATA;
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
#endif
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, WORD_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, srcw);

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	case SLJIT_MOV_U32:
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u32)srcw : srcw);

	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s32)srcw : srcw);
#endif

	case SLJIT_MOV_U8:
		return emit_op(compiler, op, BYTE_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, op, BYTE_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, op, HALF_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, op, HALF_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_CLZ:
	case SLJIT_CTZ:
	case SLJIT_REV:
		return emit_op(compiler, op, flags, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		return emit_op(compiler, op, HALF_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		return emit_op(compiler, op | SLJIT_32, INT_DATA, dst, dstw, TMP_REG1, 0, src, srcw);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (op & SLJIT_32) {
		flags |= INT_DATA | SIGNED_DATA;
		if (src1 == SLJIT_IMM)
			src1w = (sljit_s32)src1w;
		if (src2 == SLJIT_IMM)
			src2w = (sljit_s32)src2w;
	}
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
		return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
	case SLJIT_SUBC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		compiler->status_flags_state = 0;
		return emit_op(compiler, op, flags | CUMULATIVE_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_XOR:
		if ((src1 == SLJIT_IMM && src1w == -1) || (src2 == SLJIT_IMM && src2w == -1)) {
			return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);
		}
		/* fallthrough */
	case SLJIT_AND:
	case SLJIT_OR:
		return emit_op(compiler, op, flags | CUMULATIVE_OP | LOGICAL_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
	case SLJIT_ROTL:
	case SLJIT_ROTR:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		if (src2 == SLJIT_IMM)
			src2w &= 0x1f;
#else
		if (src2 == SLJIT_IMM) {
			if (op & SLJIT_32)
				src2w &= 0x1f;
			else
				src2w &= 0x3f;
		}
#endif
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 1, 0, 0, src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, 0, 0, src1, src1w, src2, src2w);
}

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
#define SELECT_OP3(op, src2w, D, D32, W) (((op & SLJIT_32) ? (W) : ((src2w) < 32) ? (D) : (D32)) | (((sljit_ins)src2w & 0x1f) << 6))
#define SELECT_OP2(op, D, W) ((op & SLJIT_32) ? (W) : (D))
#else /* !SLJIT_CONFIG_MIPS_64 */
#define SELECT_OP3(op, src2w, D, D32, W) ((W) | ((sljit_ins)(src2w) << 6))
#define SELECT_OP2(op, D, W) (W)
#endif /* SLJIT_CONFIG_MIPS_64 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_s32 is_left;
	sljit_ins ins1, ins2, ins3;
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	sljit_s32 inp_flags = ((op & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
	sljit_sw bit_length = (op & SLJIT_32) ? 32 : 64;
#else /* !SLJIT_CONFIG_MIPS_64 */
	sljit_s32 inp_flags = WORD_DATA | LOAD_DATA;
	sljit_sw bit_length = 32;
#endif /* SLJIT_CONFIG_MIPS_64 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, dst_reg, src1_reg, src2_reg, src3, src3w));

	is_left = (GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_MSHL);

	if (src1_reg == src2_reg) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, (is_left ? SLJIT_ROTL : SLJIT_ROTR) | (op & SLJIT_32), dst_reg, 0, src1_reg, 0, src3, src3w);
	}

	ADJUST_LOCAL_OFFSET(src3, src3w);

	if (src3 == SLJIT_IMM) {
		src3w &= bit_length - 1;

		if (src3w == 0)
			return SLJIT_SUCCESS;

		if (is_left) {
			ins1 = SELECT_OP3(op, src3w, DSLL, DSLL32, SLL);
			src3w = bit_length - src3w;
			ins2 = SELECT_OP3(op, src3w, DSRL, DSRL32, SRL);
		} else {
			ins1 = SELECT_OP3(op, src3w, DSRL, DSRL32, SRL);
			src3w = bit_length - src3w;
			ins2 = SELECT_OP3(op, src3w, DSLL, DSLL32, SLL);
		}

		FAIL_IF(push_inst(compiler, ins1 | T(src1_reg) | D(dst_reg), DR(dst_reg)));
		FAIL_IF(push_inst(compiler, ins2 | T(src2_reg) | D(TMP_REG1), DR(TMP_REG1)));
		return push_inst(compiler, OR | S(dst_reg) | T(TMP_REG1) | D(dst_reg), DR(dst_reg));
	}

	if (src3 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, DR(TMP_REG2), src3, src3w));
		src3 = TMP_REG2;
	} else if (dst_reg == src3) {
		FAIL_IF(push_inst(compiler, SELECT_OP2(op, DADDU, ADDU) | S(src3) | TA(0) | D(TMP_REG2), DR(TMP_REG2)));
		src3 = TMP_REG2;
	}

	if (is_left) {
		ins1 = SELECT_OP2(op, DSRL, SRL);
		ins2 = SELECT_OP2(op, DSLLV, SLLV);
		ins3 = SELECT_OP2(op, DSRLV, SRLV);
	} else {
		ins1 = SELECT_OP2(op, DSLL, SLL);
		ins2 = SELECT_OP2(op, DSRLV, SRLV);
		ins3 = SELECT_OP2(op, DSLLV, SLLV);
	}

	FAIL_IF(push_inst(compiler, ins2 | S(src3) | T(src1_reg) | D(dst_reg), DR(dst_reg)));

	if (!(op & SLJIT_SHIFT_INTO_NON_ZERO)) {
		FAIL_IF(push_inst(compiler, ins1 | T(src2_reg) | D(TMP_REG1) | (1 << 6), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XORI | S(src3) | T(TMP_REG2) | ((sljit_ins)bit_length - 1), DR(TMP_REG2)));
		src2_reg = TMP_REG1;
	} else
		FAIL_IF(push_inst(compiler, SELECT_OP2(op, DSUBU, SUBU) | SA(0) | T(src3) | D(TMP_REG2), DR(TMP_REG2)));

	FAIL_IF(push_inst(compiler, ins3 | S(TMP_REG2) | T(src2_reg) | D(TMP_REG1), DR(TMP_REG1)));
	return push_inst(compiler, OR | S(dst_reg) | T(TMP_REG1) | D(dst_reg), DR(dst_reg));
}

#undef SELECT_OP3
#undef SELECT_OP2

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, ADDU_W | S(src) | TA(0) | DA(RETURN_ADDR_REG), RETURN_ADDR_REG));
		else
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, RETURN_ADDR_REG, src, srcw));

		FAIL_IF(push_inst(compiler, JR | SA(RETURN_ADDR_REG), UNMOVABLE_INS));
		return push_inst(compiler, NOP, UNMOVABLE_INS);
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
		return emit_prefetch(compiler, src, srcw);
#else /* SLJIT_MIPS_REV < 1 */
		return SLJIT_SUCCESS;
#endif /* SLJIT_MIPS_REV >= 1 */
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 dst_ar = RETURN_ADDR_REG;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_dst(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	switch (op) {
	case SLJIT_FAST_ENTER:
		if (FAST_IS_REG(dst))
			return push_inst(compiler, ADDU_W | SA(RETURN_ADDR_REG) | TA(0) | D(dst), UNMOVABLE_INS);
		break;
	case SLJIT_GET_RETURN_ADDRESS:
		dst_ar = DR(FAST_IS_REG(dst) ? dst : TMP_REG2);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, dst_ar, SLJIT_MEM1(SLJIT_SP), compiler->local_size - SSIZE_OF(sw)));
		break;
	}

	if (dst & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_DATA, dst_ar, dst, dstw));

		if (op == SLJIT_FAST_ENTER)
			compiler->delay_slot = UNMOVABLE_INS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(type, reg));

	if (type == SLJIT_GP_REGISTER)
		return reg_map[reg];

	if (type != SLJIT_FLOAT_REGISTER)
		return -1;

	return FR(reg);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	SLJIT_UNUSED_ARG(size);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_ins*)instruction, UNMOVABLE_INS);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_32) >> 7))
#define FMT(op) (FMT_S | (~(sljit_ins)op & SLJIT_32) << (21 - (5 + 3)))

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_u32 flags = 0;
#else
	sljit_u32 flags = ((sljit_u32)(GET_OPCODE(op) == SLJIT_CONV_SW_FROM_F64)) << 21;
#endif

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src, srcw, dst, dstw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, (TRUNC_W_S ^ (flags >> 19)) | FMT(op) | FS(src) | FD(TMP_FREG1), MOVABLE_INS));

	if (FAST_IS_REG(dst)) {
		FAIL_IF(push_inst(compiler, MFC1 | flags | T(dst) | FS(TMP_FREG1), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV) || (SLJIT_CONFIG_MIPS_32 && SLJIT_MIPS_REV <= 1)
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
		return SLJIT_SUCCESS;
	}

	return emit_op_mem2(compiler, flags ? DOUBLE_DATA : SINGLE_DATA, FR(TMP_FREG1), dst, dstw, 0, 0);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_u32 flags = 0;
#else
	sljit_u32 flags = ((sljit_u32)(GET_OPCODE(op) == SLJIT_CONV_F64_FROM_SW)) << 21;
#endif
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM)
		FAIL_IF(emit_op_mem2(compiler, (flags ? DOUBLE_DATA : SINGLE_DATA) | LOAD_DATA, FR(TMP_FREG1), src, srcw, dst, dstw));
	else {
		if (src == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
			if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
				srcw = (sljit_s32)srcw;
#endif
			FAIL_IF(load_immediate(compiler, DR(TMP_REG1), srcw));
			src = TMP_REG1;
		}

		FAIL_IF(push_inst(compiler, MTC1 | flags | T(src) | FS(TMP_FREG1), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV) || (SLJIT_CONFIG_MIPS_32 && SLJIT_MIPS_REV <= 1)
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
	}

	FAIL_IF(push_inst(compiler, CVT_S_S | flags | (4 << 21) | ((~(sljit_ins)op & SLJIT_32) >> 8) | FS(TMP_FREG1) | FD(dst_r), MOVABLE_INS));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG1), dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_uw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_u32 flags = 0;
#else
	sljit_u32 flags = 1 << 21;
#endif
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_UW ? WORD_DATA : INT_DATA) | LOAD_DATA, DR(TMP_REG1), src, srcw, dst, dstw));
		src = TMP_REG1;
	} else if (src == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_U32)
			srcw = (sljit_u32)srcw;
#endif
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), srcw));
		src = TMP_REG1;
	}

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_U32) {
		if (src != TMP_REG1) {
			FAIL_IF(push_inst(compiler, DSLL32 | T(src) | D(TMP_REG1) | SH_IMM(0), DR(TMP_REG1)));
			FAIL_IF(push_inst(compiler, DSRL32 | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(0), DR(TMP_REG1)));
		}

		FAIL_IF(push_inst(compiler, MTC1 | flags | T(TMP_REG1) | FS(TMP_FREG1), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV)
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */

		FAIL_IF(push_inst(compiler, CVT_S_S | flags | (4 << 21) | ((~(sljit_ins)op & SLJIT_32) >> 8) | FS(TMP_FREG1) | FD(dst_r), MOVABLE_INS));

		if (dst & SLJIT_MEM)
			return emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG1), dst, dstw, 0, 0);
		return SLJIT_SUCCESS;
	}
#else /* !SLJIT_CONFIG_MIPS_64 */
	if (!(op & SLJIT_32)) {
		FAIL_IF(push_inst(compiler, SLL | T(src) | D(TMP_REG2) | SH_IMM(1), DR(TMP_REG2)));
		FAIL_IF(push_inst(compiler, SRL | T(TMP_REG2) | D(TMP_REG2) | SH_IMM(1), DR(TMP_REG2)));

		FAIL_IF(push_inst(compiler, MTC1 | flags | T(TMP_REG2) | FS(TMP_FREG1), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV <= 1
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */

		FAIL_IF(push_inst(compiler, CVT_S_S | flags | (4 << 21) | 1 | FS(TMP_FREG1) | FD(dst_r), MOVABLE_INS));

#if (!defined SLJIT_MIPS_REV || SLJIT_MIPS_REV <= 1)
		FAIL_IF(push_inst(compiler, BGEZ | S(src) | 5, UNMOVABLE_INS));
#else /* SLJIT_MIPS_REV >= 1 */
		FAIL_IF(push_inst(compiler, BGEZ | S(src) | 4, UNMOVABLE_INS));
#endif  /* SLJIT_MIPS_REV < 1 */

		FAIL_IF(push_inst(compiler, LUI | T(TMP_REG2) | IMM(0x41e0), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, MTC1 | TA(0) | FS(TMP_FREG2), UNMOVABLE_INS));
		switch (cpu_feature_list & CPU_FEATURE_FR) {
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
		case CPU_FEATURE_FR:
			FAIL_IF(push_inst(compiler, MTHC1 | T(TMP_REG2) | FS(TMP_FREG2), UNMOVABLE_INS));
			break;
#endif /* SLJIT_MIPS_REV >= 2 */
		default:
			FAIL_IF(push_inst(compiler, MTC1 | T(TMP_REG2) | FS(TMP_FREG2) | (1 << 11), UNMOVABLE_INS));
#if !defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV <= 1
			FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
			break;
		}
		FAIL_IF(push_inst(compiler, ADD_S | FMT(op) | FT(TMP_FREG2) | FS(dst_r) | FD(dst_r), UNMOVABLE_INS));

		if (dst & SLJIT_MEM)
			return emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG1), dst, dstw, 0, 0);
		return SLJIT_SUCCESS;
	}
#endif /* SLJIT_CONFIG_MIPS_64 */

#if (!defined SLJIT_MIPS_REV || SLJIT_MIPS_REV <= 1)
	FAIL_IF(push_inst(compiler, BLTZ | S(src) | 5, UNMOVABLE_INS));
#else /* SLJIT_MIPS_REV >= 1 */
	FAIL_IF(push_inst(compiler, BLTZ | S(src) | 4, UNMOVABLE_INS));
#endif  /* SLJIT_MIPS_REV < 1 */
	FAIL_IF(push_inst(compiler, ANDI | S(src) | T(TMP_REG2) | IMM(1), DR(TMP_REG2)));

	FAIL_IF(push_inst(compiler, MTC1 | flags | T(src) | FS(TMP_FREG1), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV)
	FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* !SLJIT_MIPS_REV */

	FAIL_IF(push_inst(compiler, CVT_S_S | flags | (4 << 21) | ((~(sljit_ins)op & SLJIT_32) >> 8) | FS(TMP_FREG1) | FD(dst_r), MOVABLE_INS));

#if (!defined SLJIT_MIPS_REV || SLJIT_MIPS_REV <= 1)
	FAIL_IF(push_inst(compiler, BEQ | 6, UNMOVABLE_INS));
#else /* SLJIT_MIPS_REV >= 1 */
	FAIL_IF(push_inst(compiler, BEQ | 5, UNMOVABLE_INS));
#endif  /* SLJIT_MIPS_REV < 1 */

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	FAIL_IF(push_inst(compiler, DSRL | T(src) | D(TMP_REG1) | SH_IMM(1), DR(TMP_REG1)));
#else /* !SLJIT_CONFIG_MIPS_64 */
	FAIL_IF(push_inst(compiler, SRL | T(src) | D(TMP_REG1) | SH_IMM(1), DR(TMP_REG1)));
#endif /* SLJIT_CONFIG_MIPS_64 */

	FAIL_IF(push_inst(compiler, OR | S(TMP_REG1) | T(TMP_REG2) | D(TMP_REG1), DR(TMP_REG1)));

	FAIL_IF(push_inst(compiler, MTC1 | flags | T(TMP_REG1) | FS(TMP_FREG1), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV)
	FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* !SLJIT_MIPS_REV */

	FAIL_IF(push_inst(compiler, CVT_S_S | flags | (4 << 21) | ((~(sljit_ins)op & SLJIT_32) >> 8) | FS(TMP_FREG1) | FD(dst_r), MOVABLE_INS));
	FAIL_IF(push_inst(compiler, ADD_S | FMT(op) | FT(dst_r) | FS(dst_r) | FD(dst_r), UNMOVABLE_INS));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG1), dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_ins inst;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src1, src1w, src2, src2w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG2), src2, src2w, 0, 0));
		src2 = TMP_FREG2;
	}

	switch (GET_FLAG_TYPE(op)) {
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
		inst = C_EQ_S;
		break;
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		inst = C_UEQ_S;
		break;
	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
		inst = C_OLT_S;
		break;
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_LESS:
		inst = C_ULT_S;
		break;
	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
		inst = C_ULE_S;
		break;
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER:
		inst = C_OLE_S;
		break;
	default:
		SLJIT_ASSERT(GET_FLAG_TYPE(op) == SLJIT_UNORDERED);
		inst = C_UN_S;
		break;
	}
	return push_inst(compiler, inst | FMT(op) | FT(src2) | FS(src1) | C_FD, UNMOVABLE_INS);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	SLJIT_COMPILE_ASSERT((SLJIT_32 == 0x100) && !(DOUBLE_DATA & 0x2), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_32;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(dst_r), src, srcw, dst, dstw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, MOV_fmt(FMT(op)) | FS(src) | FD(dst_r), MOVABLE_INS));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, NEG_S | FMT(op) | FS(src) | FD(dst_r), MOVABLE_INS));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, ABS_S | FMT(op) | FS(src) | FD(dst_r), MOVABLE_INS));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		/* The SLJIT_32 bit is inverted because sljit_f32 needs to be loaded from the memory. */
		FAIL_IF(push_inst(compiler, CVT_S_S | (sljit_ins)((op & SLJIT_32) ? 1 : (1 << 21)) | FS(src) | FD(dst_r), MOVABLE_INS));
		op ^= SLJIT_32;
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), FR(dst_r), dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r, flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG2;

	if (src1 & SLJIT_MEM) {
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src1, src1w)) {
			FAIL_IF(compiler->error);
			src1 = TMP_FREG1;
		} else
			flags |= SLOW_SRC1;
	}

	if (src2 & SLJIT_MEM) {
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG2), src2, src2w)) {
			FAIL_IF(compiler->error);
			src2 = TMP_FREG2;
		} else
			flags |= SLOW_SRC2;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG2), src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG2), src2, src2w, dst, dstw));
		}
	}
	else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG2), src2, src2w, dst, dstw));

	if (flags & SLOW_SRC1)
		src1 = TMP_FREG1;
	if (flags & SLOW_SRC2)
		src2 = TMP_FREG2;

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, ADD_S | FMT(op) | FT(src2) | FS(src1) | FD(dst_r), MOVABLE_INS));
		break;
	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, SUB_S | FMT(op) | FT(src2) | FS(src1) | FD(dst_r), MOVABLE_INS));
		break;
	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, MUL_S | FMT(op) | FT(src2) | FS(src1) | FD(dst_r), MOVABLE_INS));
		break;
	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, DIV_S | FMT(op) | FT(src2) | FS(src1) | FD(dst_r), MOVABLE_INS));
		break;
	case SLJIT_COPYSIGN_F64:
		return emit_copysign(compiler, op, src1, src2, dst_r);
	}

	if (dst_r == TMP_FREG2)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG2), dst, dstw, 0, 0));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset32(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f32 value)
{
	union {
		sljit_s32 imm;
		sljit_f32 value;
	} u;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fset32(compiler, freg, value));

	u.value = value;

	if (u.imm == 0)
		return push_inst(compiler, MTC1 | TA(0) | FS(freg), MOVABLE_INS);

	FAIL_IF(load_immediate(compiler, DR(TMP_REG1), u.imm));
	return push_inst(compiler, MTC1 | T(TMP_REG1) | FS(freg), MOVABLE_INS);
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler)
{
	struct sljit_label *label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_label(compiler));

	if (compiler->last_label && compiler->last_label->size == compiler->size)
		return compiler->last_label;

	label = (struct sljit_label*)ensure_abuf(compiler, sizeof(struct sljit_label));
	PTR_FAIL_IF(!label);
	set_label(label, compiler);
	compiler->delay_slot = UNMOVABLE_INS;
	return label;
}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define BRANCH_LENGTH	4
#else
#define BRANCH_LENGTH	8
#endif

#define BR_Z(src) \
	inst = BEQ | SA(src) | TA(0) | BRANCH_LENGTH; \
	flags = IS_BIT26_COND; \
	delay_check = src;

#define BR_NZ(src) \
	inst = BNE | SA(src) | TA(0) | BRANCH_LENGTH; \
	flags = IS_BIT26_COND; \
	delay_check = src;

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)

#define BR_T() \
	inst = BC1NEZ; \
	flags = IS_BIT23_COND; \
	delay_check = FCSR_FCC;
#define BR_F() \
	inst = BC1EQZ; \
	flags = IS_BIT23_COND; \
	delay_check = FCSR_FCC;

#else /* SLJIT_MIPS_REV < 6 */

#define BR_T() \
	inst = BC1T | BRANCH_LENGTH; \
	flags = IS_BIT16_COND; \
	delay_check = FCSR_FCC;
#define BR_F() \
	inst = BC1F | BRANCH_LENGTH; \
	flags = IS_BIT16_COND; \
	delay_check = FCSR_FCC;

#endif /* SLJIT_MIPS_REV >= 6 */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins inst;
	sljit_u32 flags = 0;
	sljit_s32 delay_check = UNMOVABLE_INS;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	switch (type) {
	case SLJIT_EQUAL:
		BR_NZ(EQUAL_FLAG);
		break;
	case SLJIT_NOT_EQUAL:
		BR_Z(EQUAL_FLAG);
		break;
	case SLJIT_LESS:
	case SLJIT_GREATER:
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER:
	case SLJIT_OVERFLOW:
	case SLJIT_CARRY:
		BR_Z(OTHER_FLAG);
		break;
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_NOT_CARRY:
		BR_NZ(OTHER_FLAG);
		break;
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_F_GREATER:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_ORDERED:
		BR_T();
		break;
	case SLJIT_F_EQUAL:
	case SLJIT_F_LESS:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_UNORDERED:
		BR_F();
		break;
	default:
		/* Not conditional branch. */
		inst = 0;
		break;
	}

	jump->flags |= flags;
	if (compiler->delay_slot == MOVABLE_INS || (compiler->delay_slot != UNMOVABLE_INS && compiler->delay_slot != delay_check))
		jump->flags |= IS_MOVABLE;

	if (inst)
		PTR_FAIL_IF(push_inst(compiler, inst, UNMOVABLE_INS));

	if (type <= SLJIT_JUMP)
		PTR_FAIL_IF(push_inst(compiler, JR | S(TMP_REG2), UNMOVABLE_INS));
	else {
		jump->flags |= IS_JAL;
		PTR_FAIL_IF(push_inst(compiler, JALR | S(TMP_REG2) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	}

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));

	/* Maximum number of instructions required for generating a constant. */
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	compiler->size += 2;
#else
	compiler->size += 6;
#endif
	return jump;
}

#define RESOLVE_IMM1() \
	if (src1 == SLJIT_IMM) { \
		if (src1w) { \
			PTR_FAIL_IF(load_immediate(compiler, DR(TMP_REG1), src1w)); \
			src1 = TMP_REG1; \
		} \
		else \
			src1 = 0; \
	}

#define RESOLVE_IMM2() \
	if (src2 == SLJIT_IMM) { \
		if (src2w) { \
			PTR_FAIL_IF(load_immediate(compiler, DR(TMP_REG2), src2w)); \
			src2 = TMP_REG2; \
		} \
		else \
			src2 = 0; \
	}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	struct sljit_jump *jump;
	sljit_s32 flags;
	sljit_ins inst;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_cmp(compiler, type, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	flags = WORD_DATA | LOAD_DATA;
#else /* !SLJIT_CONFIG_MIPS_32 */
	flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
#endif /* SLJIT_CONFIG_MIPS_32 */

	if (src1 & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem2(compiler, flags, DR(TMP_REG1), src1, src1w, src2, src2w));
		src1 = TMP_REG1;
	}

	if (src2 & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem2(compiler, flags, DR(TMP_REG2), src2, src2w, 0, 0));
		src2 = TMP_REG2;
	}

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	if (type <= SLJIT_NOT_EQUAL) {
		RESOLVE_IMM1();
		RESOLVE_IMM2();
		jump->flags |= IS_BIT26_COND;
		if (compiler->delay_slot == MOVABLE_INS || (compiler->delay_slot != UNMOVABLE_INS && compiler->delay_slot != DR(src1) && compiler->delay_slot != DR(src2)))
			jump->flags |= IS_MOVABLE;
		PTR_FAIL_IF(push_inst(compiler, (type == SLJIT_EQUAL ? BNE : BEQ) | S(src1) | T(src2) | BRANCH_LENGTH, UNMOVABLE_INS));
	} else if (type >= SLJIT_SIG_LESS && ((src1 == SLJIT_IMM && src1w == 0) || (src2 == SLJIT_IMM && src2w == 0))) {
		inst = NOP;
		if (src1 == SLJIT_IMM && src1w == 0) {
			RESOLVE_IMM2();
			switch (type) {
			case SLJIT_SIG_LESS:
				inst = BLEZ;
				jump->flags |= IS_BIT26_COND;
				break;
			case SLJIT_SIG_GREATER_EQUAL:
				inst = BGTZ;
				jump->flags |= IS_BIT26_COND;
				break;
			case SLJIT_SIG_GREATER:
				inst = BGEZ;
				jump->flags |= IS_BIT16_COND;
				break;
			case SLJIT_SIG_LESS_EQUAL:
				inst = BLTZ;
				jump->flags |= IS_BIT16_COND;
				break;
			}
			src1 = src2;
		}
		else {
			RESOLVE_IMM1();
			switch (type) {
			case SLJIT_SIG_LESS:
				inst = BGEZ;
				jump->flags |= IS_BIT16_COND;
				break;
			case SLJIT_SIG_GREATER_EQUAL:
				inst = BLTZ;
				jump->flags |= IS_BIT16_COND;
				break;
			case SLJIT_SIG_GREATER:
				inst = BLEZ;
				jump->flags |= IS_BIT26_COND;
				break;
			case SLJIT_SIG_LESS_EQUAL:
				inst = BGTZ;
				jump->flags |= IS_BIT26_COND;
				break;
			}
		}
		PTR_FAIL_IF(push_inst(compiler, inst | S(src1) | BRANCH_LENGTH, UNMOVABLE_INS));
	}
	else {
		if (type == SLJIT_LESS || type == SLJIT_GREATER_EQUAL || type == SLJIT_SIG_LESS || type == SLJIT_SIG_GREATER_EQUAL) {
			RESOLVE_IMM1();
			if (src2 == SLJIT_IMM && src2w <= SIMM_MAX && src2w >= SIMM_MIN)
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTIU : SLTI) | S(src1) | T(TMP_REG1) | IMM(src2w), DR(TMP_REG1)));
			else {
				RESOLVE_IMM2();
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTU : SLT) | S(src1) | T(src2) | D(TMP_REG1), DR(TMP_REG1)));
			}
			type = (type == SLJIT_LESS || type == SLJIT_SIG_LESS) ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;
		}
		else {
			RESOLVE_IMM2();
			if (src1 == SLJIT_IMM && src1w <= SIMM_MAX && src1w >= SIMM_MIN)
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTIU : SLTI) | S(src2) | T(TMP_REG1) | IMM(src1w), DR(TMP_REG1)));
			else {
				RESOLVE_IMM1();
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTU : SLT) | S(src2) | T(src1) | D(TMP_REG1), DR(TMP_REG1)));
			}
			type = (type == SLJIT_GREATER || type == SLJIT_SIG_GREATER) ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;
		}

		jump->flags |= IS_BIT26_COND;
		PTR_FAIL_IF(push_inst(compiler, (type == SLJIT_EQUAL ? BNE : BEQ) | S(TMP_REG1) | TA(0) | BRANCH_LENGTH, UNMOVABLE_INS));
	}

	PTR_FAIL_IF(push_inst(compiler, JR | S(TMP_REG2), UNMOVABLE_INS));
	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));

	/* Maximum number of instructions required for generating a constant. */
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	compiler->size += 2;
#else
	compiler->size += 6;
#endif
	return jump;
}

#undef RESOLVE_IMM1
#undef RESOLVE_IMM2

#undef BRANCH_LENGTH
#undef BR_Z
#undef BR_NZ
#undef BR_T
#undef BR_F

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump = NULL;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));

	if (src == SLJIT_IMM) {
		jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
		FAIL_IF(!jump);
		set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_JAL : 0));
		jump->u.target = (sljit_uw)srcw;

		if (compiler->delay_slot != UNMOVABLE_INS)
			jump->flags |= IS_MOVABLE;

		src = TMP_REG2;
	} else if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, DR(TMP_REG2), src, srcw));
		src = TMP_REG2;
	}

	if (type <= SLJIT_JUMP)
		FAIL_IF(push_inst(compiler, JR | S(src), UNMOVABLE_INS));
	else
		FAIL_IF(push_inst(compiler, JALR | S(src) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));

	if (jump != NULL) {
		jump->addr = compiler->size;

		/* Maximum number of instructions required for generating a constant. */
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		compiler->size += 2;
#else
		compiler->size += 6;
#endif
	}

	return push_inst(compiler, NOP, UNMOVABLE_INS);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_s32 src_ar, dst_ar, invert;
	sljit_s32 saved_op = op;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_s32 mem_type = WORD_DATA;
#else
	sljit_s32 mem_type = ((op & SLJIT_32) || op == SLJIT_MOV32) ? (INT_DATA | SIGNED_DATA) : WORD_DATA;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	dst_ar = DR((op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2);

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	if (op >= SLJIT_ADD && (dst & SLJIT_MEM))
		FAIL_IF(emit_op_mem2(compiler, mem_type | LOAD_DATA, DR(TMP_REG1), dst, dstw, dst, dstw));

	if (type < SLJIT_F_EQUAL) {
		src_ar = OTHER_FLAG;
		invert = type & 0x1;

		switch (type) {
		case SLJIT_EQUAL:
		case SLJIT_NOT_EQUAL:
			FAIL_IF(push_inst(compiler, SLTIU | SA(EQUAL_FLAG) | TA(dst_ar) | IMM(1), dst_ar));
			src_ar = dst_ar;
			break;
		case SLJIT_OVERFLOW:
		case SLJIT_NOT_OVERFLOW:
			if (compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)) {
				src_ar = OTHER_FLAG;
				break;
			}
			FAIL_IF(push_inst(compiler, SLTIU | SA(OTHER_FLAG) | TA(dst_ar) | IMM(1), dst_ar));
			src_ar = dst_ar;
			invert ^= 0x1;
			break;
		}
	} else {
		invert = 0;

		switch (type) {
		case SLJIT_F_NOT_EQUAL:
		case SLJIT_F_GREATER_EQUAL:
		case SLJIT_F_GREATER:
		case SLJIT_UNORDERED_OR_NOT_EQUAL:
		case SLJIT_ORDERED_NOT_EQUAL:
		case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		case SLJIT_ORDERED_GREATER_EQUAL:
		case SLJIT_ORDERED_GREATER:
		case SLJIT_UNORDERED_OR_GREATER:
		case SLJIT_ORDERED:
			invert = 1;
			break;
		}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
		FAIL_IF(push_inst(compiler, MFC1 | TA(dst_ar) | FS(TMP_FREG3), dst_ar));
#else /* SLJIT_MIPS_REV < 6 */
		FAIL_IF(push_inst(compiler, CFC1 | TA(dst_ar) | DA(FCSR_REG), dst_ar));
#endif /* SLJIT_MIPS_REV >= 6 */
		FAIL_IF(push_inst(compiler, SRL | TA(dst_ar) | DA(dst_ar) | SH_IMM(23), dst_ar));
		FAIL_IF(push_inst(compiler, ANDI | SA(dst_ar) | TA(dst_ar) | IMM(1), dst_ar));
		src_ar = dst_ar;
	}

	if (invert) {
		FAIL_IF(push_inst(compiler, XORI | SA(src_ar) | TA(dst_ar) | IMM(1), dst_ar));
		src_ar = dst_ar;
	}

	if (op < SLJIT_ADD) {
		if (dst & SLJIT_MEM)
			return emit_op_mem(compiler, mem_type, src_ar, dst, dstw);

		if (src_ar != dst_ar)
			return push_inst(compiler, ADDU_W | SA(src_ar) | TA(0) | DA(dst_ar), dst_ar);
		return SLJIT_SUCCESS;
	}

	/* OTHER_FLAG cannot be specified as src2 argument at the moment. */
	if (DR(TMP_REG2) != src_ar)
		FAIL_IF(push_inst(compiler, ADDU_W | SA(src_ar) | TA(0) | D(TMP_REG2), DR(TMP_REG2)));

	mem_type |= CUMULATIVE_OP | LOGICAL_OP | IMM_OP | ALT_KEEP_CACHE;

	if (dst & SLJIT_MEM)
		return emit_op(compiler, saved_op, mem_type, dst, dstw, TMP_REG1, 0, TMP_REG2, 0);
	return emit_op(compiler, saved_op, mem_type, dst, dstw, dst, dstw, TMP_REG2, 0);
}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6)

static sljit_ins get_select_cc(sljit_s32 type, sljit_s32 is_float)
{
	switch (type & ~SLJIT_32) {
	case SLJIT_EQUAL:
		return (is_float ? MOVZ_S : MOVZ) | TA(EQUAL_FLAG);
	case SLJIT_NOT_EQUAL:
		return (is_float ? MOVN_S : MOVN) | TA(EQUAL_FLAG);
	case SLJIT_LESS:
	case SLJIT_GREATER:
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER:
	case SLJIT_OVERFLOW:
	case SLJIT_CARRY:
		return (is_float ? MOVN_S : MOVN) | TA(OTHER_FLAG);
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_NOT_CARRY:
		return (is_float ? MOVZ_S : MOVZ) | TA(OTHER_FLAG);
	case SLJIT_F_EQUAL:
	case SLJIT_F_LESS:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_UNORDERED:
		return is_float ? MOVT_S : MOVT;
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_F_GREATER:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_ORDERED:
		return is_float ? MOVF_S : MOVF;
	default:
		SLJIT_UNREACHABLE();
		return (is_float ? MOVZ_S : MOVZ) | TA(OTHER_FLAG);
	}
}

#endif /* SLJIT_MIPS_REV >= 1 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	sljit_s32 inp_flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
	sljit_ins mov_ins = (type & SLJIT_32) ? ADDU : DADDU;
#else /* !SLJIT_CONFIG_MIPS_64 */
	sljit_s32 inp_flags = WORD_DATA | LOAD_DATA;
	sljit_ins mov_ins = ADDU;
#endif /* SLJIT_CONFIG_MIPS_64 */

#if !(defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6)
	struct sljit_label *label;
	struct sljit_jump *jump;
#endif /* !(SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6) */

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));
	ADJUST_LOCAL_OFFSET(src1, src1w);

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6)
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, DR(TMP_REG2), src1, src1w));
		src1 = TMP_REG2;
	} else if (src1 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (type & SLJIT_32)
			src1w = (sljit_s32)src1w;
#endif
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), src1w));
		src1 = TMP_REG1;
	}

	if (dst_reg != src2_reg) {
		if (dst_reg == src1) {
			src1 = src2_reg;
			type ^= 0x1;
		} else
			FAIL_IF(push_inst(compiler, mov_ins | S(src2_reg) | TA(0) | D(dst_reg), DR(dst_reg)));
	}

	return push_inst(compiler, get_select_cc(type, 0) | S(src1) | D(dst_reg), DR(dst_reg));

#else /* SLJIT_MIPS_REV < 1 || SLJIT_MIPS_REV >= 6 */
	if (dst_reg != src2_reg) {
		if (dst_reg == src1) {
			src1 = src2_reg;
			src1w = 0;
			type ^= 0x1;
		} else {
			if (ADDRESSING_DEPENDS_ON(src1, dst_reg)) {
				FAIL_IF(push_inst(compiler, ADDU_W | S(dst_reg) | TA(0) | D(TMP_REG2), DR(TMP_REG2)));

				if ((src1 & REG_MASK) == dst_reg)
					src1 = (src1 & ~REG_MASK) | TMP_REG2;

				if (OFFS_REG(src1) == dst_reg)
					src1 = (src1 & ~OFFS_REG_MASK) | TO_OFFS_REG(TMP_REG2);
			}

			FAIL_IF(push_inst(compiler, mov_ins | S(src2_reg) | TA(0) | D(dst_reg), DR(dst_reg)));
		}
	}

	SLJIT_SKIP_CHECKS(compiler);
	jump = sljit_emit_jump(compiler, (type & ~SLJIT_32) ^ 0x1);
	FAIL_IF(!jump);

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, DR(dst_reg), src1, src1w));
	} else if (src1 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (type & SLJIT_32)
			src1w = (sljit_s32)src1w;
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(load_immediate(compiler, DR(dst_reg), src1w));
	} else
		FAIL_IF(push_inst(compiler, mov_ins | S(src1) | TA(0) | D(dst_reg), DR(dst_reg)));

	SLJIT_SKIP_CHECKS(compiler);
	label = sljit_emit_label(compiler);
	FAIL_IF(!label);

	sljit_set_label(jump, label);
	return SLJIT_SUCCESS;
#endif /* SLJIT_MIPS_REV >= 1 */
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
#if !(defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6)
	struct sljit_label *label;
	struct sljit_jump *jump;
#endif /* !(SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6) */

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (dst_freg != src2_freg) {
		if (dst_freg == src1) {
			src1 = src2_freg;
			src1w = 0;
			type ^= 0x1;
		} else
			FAIL_IF(push_inst(compiler, MOV_fmt(FMT(type)) | FS(src2_freg) | FD(dst_freg), MOVABLE_INS));
	}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6)
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(type) | LOAD_DATA, FR(TMP_FREG1), src1, src1w));
		src1 = TMP_FREG1;
	}

	return push_inst(compiler, get_select_cc(type, 1) | FMT(type) | FS(src1) | FD(dst_freg), MOVABLE_INS);

#else /* SLJIT_MIPS_REV < 1 || SLJIT_MIPS_REV >= 6 */
	SLJIT_SKIP_CHECKS(compiler);
	jump = sljit_emit_jump(compiler, (type & ~SLJIT_32) ^ 0x1);
	FAIL_IF(!jump);

	if (src1 & SLJIT_MEM)
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(type) | LOAD_DATA, FR(dst_freg), src1, src1w));
	else
		FAIL_IF(push_inst(compiler, MOV_fmt(FMT(type)) | FS(src1) | FD(dst_freg), MOVABLE_INS));

	SLJIT_SKIP_CHECKS(compiler);
	label = sljit_emit_label(compiler);
	FAIL_IF(!label);

	sljit_set_label(jump, label);
	return SLJIT_SUCCESS;
#endif /* SLJIT_MIPS_REV >= 1 */
}

#undef FLOAT_DATA
#undef FMT

static sljit_s32 update_mem_addr(struct sljit_compiler *compiler, sljit_s32 *mem, sljit_sw *memw, sljit_s16 max_offset)
{
	sljit_s32 arg = *mem;
	sljit_sw argw = *memw;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (SLJIT_UNLIKELY(argw)) {
			FAIL_IF(push_inst(compiler, SLL_W | T(OFFS_REG(arg)) | D(TMP_REG1) | SH_IMM(argw), DR(TMP_REG1)));
			FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG1) | T(arg & REG_MASK) | D(TMP_REG1), DR(TMP_REG1)));
		} else
			FAIL_IF(push_inst(compiler, ADDU_W | S(arg & REG_MASK) | T(OFFS_REG(arg)) | D(TMP_REG1), DR(TMP_REG1)));

		*mem = TMP_REG1;
		*memw = 0;

		return SLJIT_SUCCESS;
	}

	if (argw <= max_offset && argw >= SIMM_MIN) {
		*mem = arg & REG_MASK;
		return SLJIT_SUCCESS;
	}

	*mem = TMP_REG1;

	if ((sljit_s16)argw > max_offset) {
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), argw));
		*memw = 0;
	} else {
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), TO_ARGW_HI(argw)));
		*memw = (sljit_s16)argw;
	}

	if ((arg & REG_MASK) == 0)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ADDU_W | S(TMP_REG1) | T(arg & REG_MASK) | D(TMP_REG1), DR(TMP_REG1));
}

#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define IMM_LEFT(memw)			IMM((memw) + SSIZE_OF(sw) - 1)
#define IMM_RIGHT(memw)			IMM(memw)
#define IMM_32_LEFT(memw)		IMM((memw) + SSIZE_OF(s32) - 1)
#define IMM_32_RIGHT(memw)		IMM(memw)
#define IMM_F64_FIRST_LEFT(memw)	IMM((memw) + SSIZE_OF(s32) - 1)
#define IMM_F64_FIRST_RIGHT(memw)	IMM(memw)
#define IMM_F64_SECOND_LEFT(memw)	IMM((memw) + SSIZE_OF(f64) - 1)
#define IMM_F64_SECOND_RIGHT(memw)	IMM((memw) + SSIZE_OF(s32))
#define IMM_16_FIRST(memw)		IMM((memw) + 1)
#define IMM_16_SECOND(memw)		IMM(memw)
#else /* !SLJIT_LITTLE_ENDIAN */
#define IMM_LEFT(memw)			IMM(memw)
#define IMM_RIGHT(memw)			IMM((memw) + SSIZE_OF(sw) - 1)
#define IMM_32_LEFT(memw)		IMM(memw)
#define IMM_32_RIGHT(memw)		IMM((memw) + SSIZE_OF(s32) - 1)
#define IMM_F64_FIRST_LEFT(memw)	IMM((memw) + SSIZE_OF(s32))
#define IMM_F64_FIRST_RIGHT(memw)	IMM((memw) + SSIZE_OF(f64) - 1)
#define IMM_F64_SECOND_LEFT(memw)	IMM(memw)
#define IMM_F64_SECOND_RIGHT(memw)	IMM((memw) + SSIZE_OF(s32) - 1)
#define IMM_16_FIRST(memw)		IMM(memw)
#define IMM_16_SECOND(memw)		IMM((memw) + 1)
#endif /* SLJIT_LITTLE_ENDIAN */

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define MEM_CHECK_UNALIGNED(type) ((type) & (SLJIT_MEM_UNALIGNED | SLJIT_MEM_ALIGNED_16))
#else /* !SLJIT_CONFIG_MIPS_32 */
#define MEM_CHECK_UNALIGNED(type) ((type) & (SLJIT_MEM_UNALIGNED | SLJIT_MEM_ALIGNED_16 | SLJIT_MEM_ALIGNED_32))
#endif /* SLJIT_CONFIG_MIPS_32 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 op = type & 0xff;
	sljit_s32 flags = 0;
	sljit_ins ins;
#if !(defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
	sljit_ins ins_right;
#endif /* !(SLJIT_MIPS_REV >= 6) */

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (reg & REG_PAIR_MASK) {
		ADJUST_LOCAL_OFFSET(mem, memw);

#if !(defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
		if (MEM_CHECK_UNALIGNED(type)) {
			FAIL_IF(update_mem_addr(compiler, &mem, &memw, SIMM_MAX - (2 * SSIZE_OF(sw) - 1)));

			if (!(type & SLJIT_MEM_STORE) && (mem == REG_PAIR_FIRST(reg) || mem == REG_PAIR_SECOND(reg))) {
				FAIL_IF(push_inst(compiler, ADDU_W | S(mem) | TA(0) | D(TMP_REG1), DR(TMP_REG1)));
				mem = TMP_REG1;
			}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
			ins = ((type & SLJIT_MEM_STORE) ? SWL : LWL) | S(mem);
			ins_right = ((type & SLJIT_MEM_STORE) ? SWR : LWR) | S(mem);
#else /* !SLJIT_CONFIG_MIPS_32 */
			ins = ((type & SLJIT_MEM_STORE) ? SDL : LDL) | S(mem);
			ins_right = ((type & SLJIT_MEM_STORE) ? SDR : LDR) | S(mem);
#endif /* SLJIT_CONFIG_MIPS_32 */

			FAIL_IF(push_inst(compiler, ins | T(REG_PAIR_FIRST(reg)) | IMM_LEFT(memw), DR(REG_PAIR_FIRST(reg))));
			FAIL_IF(push_inst(compiler, ins_right | T(REG_PAIR_FIRST(reg)) | IMM_RIGHT(memw), DR(REG_PAIR_FIRST(reg))));
			FAIL_IF(push_inst(compiler, ins | T(REG_PAIR_SECOND(reg)) | IMM_LEFT(memw + SSIZE_OF(sw)), DR(REG_PAIR_SECOND(reg))));
			return push_inst(compiler, ins_right | T(REG_PAIR_SECOND(reg)) | IMM_RIGHT(memw + SSIZE_OF(sw)), DR(REG_PAIR_SECOND(reg)));
		}
#endif /* !(SLJIT_MIPS_REV >= 6) */

		FAIL_IF(update_mem_addr(compiler, &mem, &memw, SIMM_MAX - SSIZE_OF(sw)));

		ins = ((type & SLJIT_MEM_STORE) ? STORE_W : LOAD_W) | S(mem);

		if (!(type & SLJIT_MEM_STORE) && mem == REG_PAIR_FIRST(reg)) {
			FAIL_IF(push_inst(compiler, ins | T(REG_PAIR_SECOND(reg)) | IMM(memw + SSIZE_OF(sw)), DR(REG_PAIR_SECOND(reg))));
			return push_inst(compiler, ins | T(REG_PAIR_FIRST(reg)) | IMM(memw), DR(REG_PAIR_FIRST(reg)));
		}

		FAIL_IF(push_inst(compiler, ins | T(REG_PAIR_FIRST(reg)) | IMM(memw), DR(REG_PAIR_FIRST(reg))));
		return push_inst(compiler, ins | T(REG_PAIR_SECOND(reg)) | IMM(memw + SSIZE_OF(sw)), DR(REG_PAIR_SECOND(reg)));
	}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
	return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);
#else /* !(SLJIT_MIPS_REV >= 6) */
	ADJUST_LOCAL_OFFSET(mem, memw);

	switch (op) {
	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		flags = BYTE_DATA;
		if (!(type & SLJIT_MEM_STORE))
			flags |= LOAD_DATA;

		if (op == SLJIT_MOV_S8)
			flags |= SIGNED_DATA;

		return emit_op_mem(compiler, flags, DR(reg), mem, memw);

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		FAIL_IF(update_mem_addr(compiler, &mem, &memw, SIMM_MAX - 1));
		SLJIT_ASSERT(FAST_IS_REG(mem) && mem != TMP_REG2);

		if (type & SLJIT_MEM_STORE) {
			FAIL_IF(push_inst(compiler, SRA_W | T(reg) | D(TMP_REG2) | SH_IMM(8), DR(TMP_REG2)));
			FAIL_IF(push_inst(compiler, data_transfer_insts[BYTE_DATA] | S(mem) | T(TMP_REG2) | IMM_16_FIRST(memw), MOVABLE_INS));
			return push_inst(compiler, data_transfer_insts[BYTE_DATA] | S(mem) | T(reg) | IMM_16_SECOND(memw), MOVABLE_INS);
		}

		flags = BYTE_DATA | LOAD_DATA;

		if (op == SLJIT_MOV_S16)
			flags |= SIGNED_DATA;

		FAIL_IF(push_inst(compiler, data_transfer_insts[flags] | S(mem) | T(TMP_REG2) | IMM_16_FIRST(memw), DR(TMP_REG2)));
		FAIL_IF(push_inst(compiler, data_transfer_insts[BYTE_DATA | LOAD_DATA] | S(mem) | T(reg) | IMM_16_SECOND(memw), DR(reg)));
		FAIL_IF(push_inst(compiler, SLL_W | T(TMP_REG2) | D(TMP_REG2) | SH_IMM(8), DR(TMP_REG2)));
		return push_inst(compiler, OR | S(reg) | T(TMP_REG2) | D(reg), DR(reg));

	case SLJIT_MOV:
	case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		if (type & SLJIT_MEM_ALIGNED_32) {
			flags = WORD_DATA;
			if (!(type & SLJIT_MEM_STORE))
				flags |= LOAD_DATA;

			return emit_op_mem(compiler, flags, DR(reg), mem, memw);
		}
#else /* !SLJIT_CONFIG_MIPS_32 */
		FAIL_IF(update_mem_addr(compiler, &mem, &memw, SIMM_MAX - 7));
		SLJIT_ASSERT(FAST_IS_REG(mem) && mem != TMP_REG2);

		if (type & SLJIT_MEM_STORE) {
			FAIL_IF(push_inst(compiler, SDL | S(mem) | T(reg) | IMM_LEFT(memw), MOVABLE_INS));
			return push_inst(compiler, SDR | S(mem) | T(reg) | IMM_RIGHT(memw), MOVABLE_INS);
		}

		if (mem == reg) {
			FAIL_IF(push_inst(compiler, ADDU_W | S(mem) | TA(0) | D(TMP_REG1), DR(TMP_REG1)));
			mem = TMP_REG1;
		}

		FAIL_IF(push_inst(compiler, LDL | S(mem) | T(reg) | IMM_LEFT(memw), DR(reg)));
		return push_inst(compiler, LDR | S(mem) | T(reg) | IMM_RIGHT(memw), DR(reg));
#endif /* SLJIT_CONFIG_MIPS_32 */
	}

	FAIL_IF(update_mem_addr(compiler, &mem, &memw, SIMM_MAX - 3));
	SLJIT_ASSERT(FAST_IS_REG(mem) && mem != TMP_REG2);

	if (type & SLJIT_MEM_STORE) {
		FAIL_IF(push_inst(compiler, SWL | S(mem) | T(reg) | IMM_32_LEFT(memw), MOVABLE_INS));
		return push_inst(compiler, SWR | S(mem) | T(reg) | IMM_32_RIGHT(memw), MOVABLE_INS);
	}

	if (mem == reg) {
		FAIL_IF(push_inst(compiler, ADDU_W | S(mem) | TA(0) | D(TMP_REG1), DR(TMP_REG1)));
		mem = TMP_REG1;
	}

	FAIL_IF(push_inst(compiler, LWL | S(mem) | T(reg) | IMM_32_LEFT(memw), DR(reg)));
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return push_inst(compiler, LWR | S(mem) | T(reg) | IMM_32_RIGHT(memw), DR(reg));
#else /* !SLJIT_CONFIG_MIPS_32 */
	FAIL_IF(push_inst(compiler, LWR | S(mem) | T(reg) | IMM_32_RIGHT(memw), DR(reg)));

	if (op != SLJIT_MOV_U32)
		return SLJIT_SUCCESS;

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 2)
	return push_inst(compiler, DINSU | T(reg) | SA(0) | (31 << 11), DR(reg));
#else  /* SLJIT_MIPS_REV < 2 */
	FAIL_IF(push_inst(compiler, DSLL32 | T(reg) | D(reg) | SH_IMM(0), DR(reg)));
	return push_inst(compiler, DSRL32 | T(reg) | D(reg) | SH_IMM(0), DR(reg));
#endif /* SLJIT_MIPS_REV >= 2 */
#endif /* SLJIT_CONFIG_MIPS_32 */
#endif /* SLJIT_MIPS_REV >= 6 */
}

#if !(defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem(compiler, type, freg, mem, memw));

	FAIL_IF(update_mem_addr(compiler, &mem, &memw, SIMM_MAX - (type & SLJIT_32) ? 3 : 7));
	SLJIT_ASSERT(FAST_IS_REG(mem) && mem != TMP_REG2);

	if (type & SLJIT_MEM_STORE) {
		if (type & SLJIT_32) {
			FAIL_IF(push_inst(compiler, MFC1 | T(TMP_REG2) | FS(freg), DR(TMP_REG2)));
#if !defined(SLJIT_MIPS_REV) || (SLJIT_CONFIG_MIPS_32 && SLJIT_MIPS_REV <= 1)
			FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
			FAIL_IF(push_inst(compiler, SWL | S(mem) | T(TMP_REG2) | IMM_32_LEFT(memw), MOVABLE_INS));
			return push_inst(compiler, SWR | S(mem) | T(TMP_REG2) | IMM_32_RIGHT(memw), MOVABLE_INS);
		}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		FAIL_IF(push_inst(compiler, MFC1 | T(TMP_REG2) | FS(freg), DR(TMP_REG2)));
#if !defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV <= 1
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
		FAIL_IF(push_inst(compiler, SWL | S(mem) | T(TMP_REG2) | IMM_F64_FIRST_LEFT(memw), MOVABLE_INS));
		FAIL_IF(push_inst(compiler, SWR | S(mem) | T(TMP_REG2) | IMM_F64_FIRST_RIGHT(memw), MOVABLE_INS));
		switch (cpu_feature_list & CPU_FEATURE_FR) {
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
		case CPU_FEATURE_FR:
			FAIL_IF(push_inst(compiler, MFHC1 | T(TMP_REG2) | FS(freg), DR(TMP_REG2)));
			break;
#endif /* SLJIT_MIPS_REV >= 2 */
		default:
			FAIL_IF(push_inst(compiler, MFC1 | T(TMP_REG2) | FS(freg) | (1 << 11), DR(TMP_REG2)));
#if !defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV <= 1
			FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif
			break;
		}

		FAIL_IF(push_inst(compiler, SWL | S(mem) | T(TMP_REG2) | IMM_F64_SECOND_LEFT(memw), MOVABLE_INS));
		return push_inst(compiler, SWR | S(mem) | T(TMP_REG2) | IMM_F64_SECOND_RIGHT(memw), MOVABLE_INS);
#else /* !SLJIT_CONFIG_MIPS_32 */
		FAIL_IF(push_inst(compiler, DMFC1 | T(TMP_REG2) | FS(freg), DR(TMP_REG2)));
#if !defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV <= 1
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
		FAIL_IF(push_inst(compiler, SDL | S(mem) | T(TMP_REG2) | IMM_LEFT(memw), MOVABLE_INS));
		return push_inst(compiler, SDR | S(mem) | T(TMP_REG2) | IMM_RIGHT(memw), MOVABLE_INS);
#endif /* SLJIT_CONFIG_MIPS_32 */
	}

	if (type & SLJIT_32) {
		FAIL_IF(push_inst(compiler, LWL | S(mem) | T(TMP_REG2) | IMM_32_LEFT(memw), DR(TMP_REG2)));
		FAIL_IF(push_inst(compiler, LWR | S(mem) | T(TMP_REG2) | IMM_32_RIGHT(memw), DR(TMP_REG2)));

		FAIL_IF(push_inst(compiler, MTC1 | T(TMP_REG2) | FS(freg), MOVABLE_INS));
#if !defined(SLJIT_MIPS_REV) || (SLJIT_CONFIG_MIPS_32 && SLJIT_MIPS_REV <= 1)
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
		return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	FAIL_IF(push_inst(compiler, LWL | S(mem) | T(TMP_REG2) | IMM_F64_FIRST_LEFT(memw), DR(TMP_REG2)));
	FAIL_IF(push_inst(compiler, LWR | S(mem) | T(TMP_REG2) | IMM_F64_FIRST_RIGHT(memw), DR(TMP_REG2)));
	FAIL_IF(push_inst(compiler, MTC1 | T(TMP_REG2) | FS(freg), MOVABLE_INS));

	FAIL_IF(push_inst(compiler, LWL | S(mem) | T(TMP_REG2) | IMM_F64_SECOND_LEFT(memw), DR(TMP_REG2)));
	FAIL_IF(push_inst(compiler, LWR | S(mem) | T(TMP_REG2) | IMM_F64_SECOND_RIGHT(memw), DR(TMP_REG2)));
	switch (cpu_feature_list & CPU_FEATURE_FR) {
#if defined(SLJIT_MIPS_REV) && SLJIT_MIPS_REV >= 2
	case CPU_FEATURE_FR:
		return push_inst(compiler, MTHC1 | T(TMP_REG2) | FS(freg), MOVABLE_INS);
#endif /* SLJIT_MIPS_REV >= 2 */
	default:
		FAIL_IF(push_inst(compiler, MTC1 | T(TMP_REG2) | FS(freg) | (1 << 11), MOVABLE_INS));
		break;
	}
#else /* !SLJIT_CONFIG_MIPS_32 */
	FAIL_IF(push_inst(compiler, LDL | S(mem) | T(TMP_REG2) | IMM_LEFT(memw), DR(TMP_REG2)));
	FAIL_IF(push_inst(compiler, LDR | S(mem) | T(TMP_REG2) | IMM_RIGHT(memw), DR(TMP_REG2)));

	FAIL_IF(push_inst(compiler, DMTC1 | T(TMP_REG2) | FS(freg), MOVABLE_INS));
#endif /* SLJIT_CONFIG_MIPS_32 */
#if !defined(SLJIT_MIPS_REV) || SLJIT_MIPS_REV <= 1
	FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* MIPS III */
	return SLJIT_SUCCESS;
}

#endif /* !SLJIT_MIPS_REV || SLJIT_MIPS_REV < 6 */

#undef IMM_16_SECOND
#undef IMM_16_FIRST
#undef IMM_F64_SECOND_RIGHT
#undef IMM_F64_SECOND_LEFT
#undef IMM_F64_FIRST_RIGHT
#undef IMM_F64_FIRST_LEFT
#undef IMM_32_RIGHT
#undef IMM_32_LEFT
#undef IMM_RIGHT
#undef IMM_LEFT
#undef MEM_CHECK_UNALIGNED

#undef TO_ARGW_HI

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	struct sljit_const *const_;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	PTR_FAIL_IF(emit_const(compiler, dst_r, init_value));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, DR(TMP_REG2), dst, dstw));

	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_put_label *put_label;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_put_label(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	put_label = (struct sljit_put_label*)ensure_abuf(compiler, sizeof(struct sljit_put_label));
	PTR_FAIL_IF(!put_label);
	set_put_label(put_label, compiler, 0);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	PTR_FAIL_IF(push_inst(compiler, (sljit_ins)dst_r, UNMOVABLE_INS));
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	compiler->size += 1;
#else
	compiler->size += 5;
#endif

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, DR(TMP_REG2), dst, dstw));

	return put_label;
}
