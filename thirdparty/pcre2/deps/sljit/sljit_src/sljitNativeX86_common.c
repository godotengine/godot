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

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
	return "x86" SLJIT_CPUINFO;
}

/*
   32b register indexes:
     0 - EAX
     1 - ECX
     2 - EDX
     3 - EBX
     4 - ESP
     5 - EBP
     6 - ESI
     7 - EDI
*/

/*
   64b register indexes:
     0 - RAX
     1 - RCX
     2 - RDX
     3 - RBX
     4 - RSP
     5 - RBP
     6 - RSI
     7 - RDI
     8 - R8   - From now on REX prefix is required
     9 - R9
    10 - R10
    11 - R11
    12 - R12
    13 - R13
    14 - R14
    15 - R15
*/

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_FREG	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 3] = {
	0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 5, 7, 6, 4, 3
};

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2] = {
	0, 1, 2, 3, 4, 5, 6, 7, 0
};

#define CHECK_EXTRA_REGS(p, w, do) \
	if (p >= SLJIT_R3 && p <= SLJIT_S3) { \
		w = (2 * SSIZE_OF(sw)) + ((p) - SLJIT_R3) * SSIZE_OF(sw); \
		p = SLJIT_MEM1(SLJIT_SP); \
		do; \
	}

#else /* SLJIT_CONFIG_X86_32 */

#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)

/* Note: r12 & 0x7 == 0b100, which decoded as SIB byte present
   Note: avoid to use r12 and r13 for memory addressing
   therefore r12 is better to be a higher saved register. */
#ifndef _WIN64
/* Args: rdi(=7), rsi(=6), rdx(=2), rcx(=1), r8, r9. Scratches: rax(=0), r10, r11 */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 0, 6, 7, 1, 8, 11, 10, 12, 5, 13, 14, 15, 3, 4, 2, 9
};
/* low-map. reg_map & 0x7. */
static const sljit_u8 reg_lmap[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 0, 6, 7, 1, 0,  3,  2,  4, 5,  5,  6,  7, 3, 4, 2, 1
};
#else
/* Args: rcx(=1), rdx(=2), r8, r9. Scratches: rax(=0), r10, r11 */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 0, 2, 8, 1, 11, 12, 5, 13, 14, 15, 7, 6, 3, 4, 9, 10
};
/* low-map. reg_map & 0x7. */
static const sljit_u8 reg_lmap[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 0, 2, 0, 1,  3,  4, 5,  5,  6,  7, 7, 6, 3, 4, 1,  2
};
#endif

/* Args: xmm0-xmm3 */
static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2] = {
	0, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4
};
/* low-map. freg_map & 0x7. */
static const sljit_u8 freg_lmap[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2] = {
	0, 0, 1, 2, 3, 5, 6, 7, 0, 1,  2,  3,  4,  5,  6,  7, 4
};

#define REX_W		0x48
#define REX_R		0x44
#define REX_X		0x42
#define REX_B		0x41
#define REX		0x40

#ifndef _WIN64
#define HALFWORD_MAX 0x7fffffffl
#define HALFWORD_MIN -0x80000000l
#else
#define HALFWORD_MAX 0x7fffffffll
#define HALFWORD_MIN -0x80000000ll
#endif

#define IS_HALFWORD(x)		((x) <= HALFWORD_MAX && (x) >= HALFWORD_MIN)
#define NOT_HALFWORD(x)		((x) > HALFWORD_MAX || (x) < HALFWORD_MIN)

#define CHECK_EXTRA_REGS(p, w, do)

#endif /* SLJIT_CONFIG_X86_32 */

#define U8(v)			((sljit_u8)(v))

/* Size flags for emit_x86_instruction: */
#define EX86_BIN_INS		((sljit_uw)0x000010)
#define EX86_SHIFT_INS		((sljit_uw)0x000020)
#define EX86_BYTE_ARG		((sljit_uw)0x000040)
#define EX86_HALF_ARG		((sljit_uw)0x000080)
/* Size flags for both emit_x86_instruction and emit_vex_instruction: */
#define EX86_REX		((sljit_uw)0x000100)
#define EX86_NO_REXW		((sljit_uw)0x000200)
#define EX86_PREF_66		((sljit_uw)0x000400)
#define EX86_PREF_F2		((sljit_uw)0x000800)
#define EX86_PREF_F3		((sljit_uw)0x001000)
#define EX86_SSE2_OP1		((sljit_uw)0x002000)
#define EX86_SSE2_OP2		((sljit_uw)0x004000)
#define EX86_SSE2		(EX86_SSE2_OP1 | EX86_SSE2_OP2)
#define EX86_VEX_EXT		((sljit_uw)0x008000)
/* Op flags for emit_vex_instruction: */
#define VEX_OP_0F38		((sljit_uw)0x010000)
#define VEX_OP_0F3A		((sljit_uw)0x020000)
#define VEX_SSE2_OPV		((sljit_uw)0x040000)
#define VEX_AUTO_W		((sljit_uw)0x080000)
#define VEX_W			((sljit_uw)0x100000)
#define VEX_256			((sljit_uw)0x200000)

#define EX86_SELECT_66(op)	(((op) & SLJIT_32) ? 0 : EX86_PREF_66)
#define EX86_SELECT_F2_F3(op)	(((op) & SLJIT_32) ? EX86_PREF_F3 : EX86_PREF_F2)

/* --------------------------------------------------------------------- */
/*  Instruction forms                                                    */
/* --------------------------------------------------------------------- */

#define ADD			(/* BINARY */ 0 << 3)
#define ADD_EAX_i32		0x05
#define ADD_r_rm		0x03
#define ADD_rm_r		0x01
#define ADDSD_x_xm		0x58
#define ADC			(/* BINARY */ 2 << 3)
#define ADC_EAX_i32		0x15
#define ADC_r_rm		0x13
#define ADC_rm_r		0x11
#define AND			(/* BINARY */ 4 << 3)
#define AND_EAX_i32		0x25
#define AND_r_rm		0x23
#define AND_rm_r		0x21
#define ANDPD_x_xm		0x54
#define BSR_r_rm		(/* GROUP_0F */ 0xbd)
#define BSF_r_rm		(/* GROUP_0F */ 0xbc)
#define BSWAP_r			(/* GROUP_0F */ 0xc8)
#define CALL_i32		0xe8
#define CALL_rm			(/* GROUP_FF */ 2 << 3)
#define CDQ			0x99
#define CMOVE_r_rm		(/* GROUP_0F */ 0x44)
#define CMP			(/* BINARY */ 7 << 3)
#define CMP_EAX_i32		0x3d
#define CMP_r_rm		0x3b
#define CMP_rm_r		0x39
#define CMPS_x_xm		0xc2
#define CMPXCHG_rm_r		0xb1
#define CMPXCHG_rm8_r		0xb0
#define CVTPD2PS_x_xm		0x5a
#define CVTPS2PD_x_xm		0x5a
#define CVTSI2SD_x_rm		0x2a
#define CVTTSD2SI_r_xm		0x2c
#define DIV			(/* GROUP_F7 */ 6 << 3)
#define DIVSD_x_xm		0x5e
#define EXTRACTPS_x_xm		0x17
#define FLDS			0xd9
#define FLDL			0xdd
#define FSTPS			0xd9
#define FSTPD			0xdd
#define INSERTPS_x_xm		0x21
#define INT3			0xcc
#define IDIV			(/* GROUP_F7 */ 7 << 3)
#define IMUL			(/* GROUP_F7 */ 5 << 3)
#define IMUL_r_rm		(/* GROUP_0F */ 0xaf)
#define IMUL_r_rm_i8		0x6b
#define IMUL_r_rm_i32		0x69
#define JL_i8			0x7c
#define JE_i8			0x74
#define JNC_i8			0x73
#define JNE_i8			0x75
#define JMP_i8			0xeb
#define JMP_i32			0xe9
#define JMP_rm			(/* GROUP_FF */ 4 << 3)
#define LEA_r_m			0x8d
#define LOOP_i8			0xe2
#define LZCNT_r_rm		(/* GROUP_F3 */ /* GROUP_0F */ 0xbd)
#define MOV_r_rm		0x8b
#define MOV_r_i32		0xb8
#define MOV_rm_r		0x89
#define MOV_rm_i32		0xc7
#define MOV_rm8_i8		0xc6
#define MOV_rm8_r8		0x88
#define MOVAPS_x_xm		0x28
#define MOVAPS_xm_x		0x29
#define MOVD_x_rm		0x6e
#define MOVD_rm_x		0x7e
#define MOVDDUP_x_xm		0x12
#define MOVDQA_x_xm		0x6f
#define MOVDQA_xm_x		0x7f
#define MOVDQU_x_xm		0x6f
#define MOVHLPS_x_x		0x12
#define MOVHPD_m_x		0x17
#define MOVHPD_x_m		0x16
#define MOVLHPS_x_x		0x16
#define MOVLPD_m_x		0x13
#define MOVLPD_x_m		0x12
#define MOVMSKPS_r_x		(/* GROUP_0F */ 0x50)
#define MOVQ_x_xm		(/* GROUP_0F */ 0x7e)
#define MOVSD_x_xm		0x10
#define MOVSD_xm_x		0x11
#define MOVSHDUP_x_xm		0x16
#define MOVSXD_r_rm		0x63
#define MOVSX_r_rm8		(/* GROUP_0F */ 0xbe)
#define MOVSX_r_rm16		(/* GROUP_0F */ 0xbf)
#define MOVUPS_x_xm		0x10
#define MOVZX_r_rm8		(/* GROUP_0F */ 0xb6)
#define MOVZX_r_rm16		(/* GROUP_0F */ 0xb7)
#define MUL			(/* GROUP_F7 */ 4 << 3)
#define MULSD_x_xm		0x59
#define NEG_rm			(/* GROUP_F7 */ 3 << 3)
#define NOP			0x90
#define NOT_rm			(/* GROUP_F7 */ 2 << 3)
#define OR			(/* BINARY */ 1 << 3)
#define OR_r_rm			0x0b
#define OR_EAX_i32		0x0d
#define OR_rm_r			0x09
#define OR_rm8_r8		0x08
#define ORPD_x_xm		0x56
#define PACKSSWB_x_xm		(/* GROUP_0F */ 0x63)
#define PAND_x_xm		0xdb
#define PCMPEQD_x_xm		0x76
#define PINSRB_x_rm_i8		0x20
#define PINSRW_x_rm_i8		0xc4
#define PINSRD_x_rm_i8		0x22
#define PEXTRB_rm_x_i8		0x14
#define PEXTRW_rm_x_i8		0x15
#define PEXTRD_rm_x_i8		0x16
#define PMOVMSKB_r_x		(/* GROUP_0F */ 0xd7)
#define PMOVSXBD_x_xm		0x21
#define PMOVSXBQ_x_xm		0x22
#define PMOVSXBW_x_xm		0x20
#define PMOVSXDQ_x_xm		0x25
#define PMOVSXWD_x_xm		0x23
#define PMOVSXWQ_x_xm		0x24
#define PMOVZXBD_x_xm		0x31
#define PMOVZXBQ_x_xm		0x32
#define PMOVZXBW_x_xm		0x30
#define PMOVZXDQ_x_xm		0x35
#define PMOVZXWD_x_xm		0x33
#define PMOVZXWQ_x_xm		0x34
#define POP_r			0x58
#define POP_rm			0x8f
#define POPF			0x9d
#define POR_x_xm		0xeb
#define PREFETCH		0x18
#define PSHUFB_x_xm		0x00
#define PSHUFD_x_xm		0x70
#define PSHUFLW_x_xm		0x70
#define PSRLDQ_x		0x73
#define PSLLD_x_i8		0x72
#define PSLLQ_x_i8		0x73
#define PUSH_i32		0x68
#define PUSH_r			0x50
#define PUSH_rm			(/* GROUP_FF */ 6 << 3)
#define PUSHF			0x9c
#define PXOR_x_xm		0xef
#define ROL			(/* SHIFT */ 0 << 3)
#define ROR			(/* SHIFT */ 1 << 3)
#define RET_near		0xc3
#define RET_i16			0xc2
#define SBB			(/* BINARY */ 3 << 3)
#define SBB_EAX_i32		0x1d
#define SBB_r_rm		0x1b
#define SBB_rm_r		0x19
#define SAR			(/* SHIFT */ 7 << 3)
#define SHL			(/* SHIFT */ 4 << 3)
#define SHLD			(/* GROUP_0F */ 0xa5)
#define SHRD			(/* GROUP_0F */ 0xad)
#define SHR			(/* SHIFT */ 5 << 3)
#define SHUFPS_x_xm		0xc6
#define SUB			(/* BINARY */ 5 << 3)
#define SUB_EAX_i32		0x2d
#define SUB_r_rm		0x2b
#define SUB_rm_r		0x29
#define SUBSD_x_xm		0x5c
#define TEST_EAX_i32		0xa9
#define TEST_rm_r		0x85
#define TZCNT_r_rm		(/* GROUP_F3 */ /* GROUP_0F */ 0xbc)
#define UCOMISD_x_xm		0x2e
#define UNPCKLPD_x_xm		0x14
#define UNPCKLPS_x_xm		0x14
#define VBROADCASTSD_x_xm	0x19
#define VBROADCASTSS_x_xm	0x18
#define VEXTRACTF128_x_ym	0x19
#define VEXTRACTI128_x_ym	0x39
#define VINSERTF128_y_y_xm	0x18
#define VINSERTI128_y_y_xm	0x38
#define VPBROADCASTB_x_xm	0x78
#define VPBROADCASTD_x_xm	0x58
#define VPBROADCASTQ_x_xm	0x59
#define VPBROADCASTW_x_xm	0x79
#define VPERMPD_y_ym		0x01
#define VPERMQ_y_ym		0x00
#define XCHG_EAX_r		0x90
#define XCHG_r_rm		0x87
#define XOR			(/* BINARY */ 6 << 3)
#define XOR_EAX_i32		0x35
#define XOR_r_rm		0x33
#define XOR_rm_r		0x31
#define XORPD_x_xm		0x57

#define GROUP_0F		0x0f
#define GROUP_66		0x66
#define GROUP_F3		0xf3
#define GROUP_F7		0xf7
#define GROUP_FF		0xff
#define GROUP_BINARY_81		0x81
#define GROUP_BINARY_83		0x83
#define GROUP_SHIFT_1		0xd1
#define GROUP_SHIFT_N		0xc1
#define GROUP_SHIFT_CL		0xd3
#define GROUP_LOCK		0xf0

#define MOD_REG			0xc0
#define MOD_DISP8		0x40

#define INC_SIZE(s)		(*inst++ = U8(s), compiler->size += (s))

#define PUSH_REG(r)		(*inst++ = U8(PUSH_r + (r)))
#define POP_REG(r)		(*inst++ = U8(POP_r + (r)))
#define RET()			(*inst++ = RET_near)
#define RET_I16(n)		(*inst++ = RET_i16, *inst++ = U8(n), *inst++ = 0)

#define SLJIT_INST_LABEL	255
#define SLJIT_INST_JUMP		254
#define SLJIT_INST_MOV_ADDR	253
#define SLJIT_INST_CONST	252

/* Multithreading does not affect these static variables, since they store
   built-in CPU features. Therefore they can be overwritten by different threads
   if they detect the CPU features in the same time. */
#define CPU_FEATURE_DETECTED		0x001
#if (defined SLJIT_DETECT_SSE2 && SLJIT_DETECT_SSE2)
#define CPU_FEATURE_SSE2		0x002
#endif
#define CPU_FEATURE_SSE41		0x004
#define CPU_FEATURE_LZCNT		0x008
#define CPU_FEATURE_TZCNT		0x010
#define CPU_FEATURE_CMOV		0x020
#define CPU_FEATURE_AVX			0x040
#define CPU_FEATURE_AVX2		0x080
#define CPU_FEATURE_OSXSAVE		0x100

static sljit_u32 cpu_feature_list = 0;

#ifdef _WIN32_WCE
#include <cmnintrin.h>
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#include <intrin.h>
#elif defined(__INTEL_COMPILER)
#include <cpuid.h>
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1400) || defined(__INTEL_COMPILER) \
	|| (defined(__INTEL_LLVM_COMPILER) && defined(__XSAVE__))
#include <immintrin.h>
#endif

/******************************************************/
/*    Unaligned-store functions                       */
/******************************************************/

static SLJIT_INLINE void sljit_unaligned_store_s16(void *addr, sljit_s16 value)
{
	SLJIT_MEMCPY(addr, &value, sizeof(value));
}

static SLJIT_INLINE void sljit_unaligned_store_s32(void *addr, sljit_s32 value)
{
	SLJIT_MEMCPY(addr, &value, sizeof(value));
}

static SLJIT_INLINE void sljit_unaligned_store_sw(void *addr, sljit_sw value)
{
	SLJIT_MEMCPY(addr, &value, sizeof(value));
}

/******************************************************/
/*    Utility functions                               */
/******************************************************/

static void execute_cpu_id(sljit_u32 info[4])
{
#if (defined(_MSC_VER) && _MSC_VER >= 1400) \
	|| (defined(__INTEL_COMPILER) && __INTEL_COMPILER == 2021 && __INTEL_COMPILER_UPDATE >= 7)

	__cpuidex((int*)info, (int)info[0], (int)info[2]);

#elif (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1900)

	__get_cpuid_count(info[0], info[2], info, info + 1, info + 2, info + 3);

#elif (defined(_MSC_VER) || defined(__INTEL_COMPILER)) \
	&& (defined(SLJIT_CONFIG_X86_32) && SLJIT_CONFIG_X86_32)

	/* Intel syntax. */
	__asm {
		mov esi, info
		mov eax, [esi]
		mov ecx, [esi + 8]
		cpuid
		mov [esi], eax
		mov [esi + 4], ebx
		mov [esi + 8], ecx
		mov [esi + 12], edx
	}

#else

	__asm__ __volatile__ (
		"cpuid\n"
		: "=a" (info[0]), "=b" (info[1]), "=c" (info[2]), "=d" (info[3])
		: "0" (info[0]), "2" (info[2])
	);

#endif
}

static sljit_u32 execute_get_xcr0_low(void)
{
	sljit_u32 xcr0;

#if (defined(_MSC_VER) && _MSC_VER >= 1400) || defined(__INTEL_COMPILER) \
	|| (defined(__INTEL_LLVM_COMPILER) && defined(__XSAVE__))

	xcr0 = (sljit_u32)_xgetbv(0);

#elif defined(__TINYC__)

	__asm__ (
		"xorl %%ecx, %%ecx\n"
		".byte 0x0f\n"
		".byte 0x01\n"
		".byte 0xd0\n"
		: "=a" (xcr0)
		:
#if defined(SLJIT_CONFIG_X86_32) && SLJIT_CONFIG_X86_32
		: "ecx", "edx"
#else /* !SLJIT_CONFIG_X86_32 */
		: "rcx", "rdx"
#endif /* SLJIT_CONFIG_X86_32 */
	);

#elif (defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20220100) \
	|| (defined(__clang__) && __clang_major__ < 14) \
	|| (defined(__GNUC__) && __GNUC__ < 3) \
	|| defined(__SUNPRO_C) || defined(__SUNPRO_CC)

	/* AT&T syntax. */
	__asm__ (
		"xorl %%ecx, %%ecx\n"
		"xgetbv\n"
		: "=a" (xcr0)
		:
#if defined(SLJIT_CONFIG_X86_32) && SLJIT_CONFIG_X86_32
		: "ecx", "edx"
#else /* !SLJIT_CONFIG_X86_32 */
		: "rcx", "rdx"
#endif /* SLJIT_CONFIG_X86_32 */
	);

#elif defined(_MSC_VER)

	/* Intel syntax. */
	__asm {
		xor ecx, ecx
		xgetbv
		mov xcr0, eax
	}

#else

	__asm__ (
		"xor{l %%ecx, %%ecx | ecx, ecx}\n"
		"xgetbv\n"
		: "=a" (xcr0)
		:
#if defined(SLJIT_CONFIG_X86_32) && SLJIT_CONFIG_X86_32
		: "ecx", "edx"
#else /* !SLJIT_CONFIG_X86_32 */
		: "rcx", "rdx"
#endif /* SLJIT_CONFIG_X86_32 */
	);

#endif
	return xcr0;
}

static void get_cpu_features(void)
{
	sljit_u32 feature_list = CPU_FEATURE_DETECTED;
	sljit_u32 info[4] = {0};
	sljit_u32 max_id;

	execute_cpu_id(info);
	max_id = info[0];

	if (max_id >= 7) {
		info[0] = 7;
		info[2] = 0;
		execute_cpu_id(info);

		if (info[1] & 0x8)
			feature_list |= CPU_FEATURE_TZCNT;
		if (info[1] & 0x20)
			feature_list |= CPU_FEATURE_AVX2;
	}

	if (max_id >= 1) {
		info[0] = 1;
#if defined(SLJIT_CONFIG_X86_32) && SLJIT_CONFIG_X86_32
		/* Winchip 2 and Cyrix MII bugs */
		info[1] = info[2] = 0;
#endif
		execute_cpu_id(info);

		if (info[2] & 0x80000)
			feature_list |= CPU_FEATURE_SSE41;
		if (info[2] & 0x8000000)
			feature_list |= CPU_FEATURE_OSXSAVE;
		if (info[2] & 0x10000000)
			feature_list |= CPU_FEATURE_AVX;
#if (defined SLJIT_DETECT_SSE2 && SLJIT_DETECT_SSE2)
		if (info[3] & 0x4000000)
			feature_list |= CPU_FEATURE_SSE2;
#endif
		if (info[3] & 0x8000)
			feature_list |= CPU_FEATURE_CMOV;
	}

	info[0] = 0x80000000;
	execute_cpu_id(info);
	max_id = info[0];

	if (max_id >= 0x80000001) {
		info[0] = 0x80000001;
		execute_cpu_id(info);

		if (info[2] & 0x20)
			feature_list |= CPU_FEATURE_LZCNT;
	}

	if ((feature_list & CPU_FEATURE_OSXSAVE) && (execute_get_xcr0_low() & 0x4) == 0)
		feature_list &= ~(sljit_u32)(CPU_FEATURE_AVX | CPU_FEATURE_AVX2);

	cpu_feature_list = feature_list;
}

static sljit_u8 get_jump_code(sljit_uw type)
{
	switch (type) {
	case SLJIT_EQUAL:
	case SLJIT_ATOMIC_STORED:
	case SLJIT_F_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		return 0x84 /* je */;

	case SLJIT_NOT_EQUAL:
	case SLJIT_ATOMIC_NOT_STORED:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
		return 0x85 /* jne */;

	case SLJIT_LESS:
	case SLJIT_CARRY:
	case SLJIT_F_LESS:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_UNORDERED_OR_GREATER:
		return 0x82 /* jc */;

	case SLJIT_GREATER_EQUAL:
	case SLJIT_NOT_CARRY:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
		return 0x83 /* jae */;

	case SLJIT_GREATER:
	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_LESS:
	case SLJIT_ORDERED_GREATER:
		return 0x87 /* jnbe */;

	case SLJIT_LESS_EQUAL:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		return 0x86 /* jbe */;

	case SLJIT_SIG_LESS:
		return 0x8c /* jl */;

	case SLJIT_SIG_GREATER_EQUAL:
		return 0x8d /* jnl */;

	case SLJIT_SIG_GREATER:
		return 0x8f /* jnle */;

	case SLJIT_SIG_LESS_EQUAL:
		return 0x8e /* jle */;

	case SLJIT_OVERFLOW:
		return 0x80 /* jo */;

	case SLJIT_NOT_OVERFLOW:
		return 0x81 /* jno */;

	case SLJIT_UNORDERED:
	case SLJIT_ORDERED_EQUAL: /* NaN. */
		return 0x8a /* jp */;

	case SLJIT_ORDERED:
	case SLJIT_UNORDERED_OR_NOT_EQUAL: /* Not NaN. */
		return 0x8b /* jpo */;
	}
	return 0;
}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
static sljit_u8* detect_far_jump_type(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_sw executable_offset);
#else /* !SLJIT_CONFIG_X86_32 */
static sljit_u8* detect_far_jump_type(struct sljit_jump *jump, sljit_u8 *code_ptr);
static sljit_u8* generate_mov_addr_code(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_u8 *code, sljit_sw executable_offset);
#endif /* SLJIT_CONFIG_X86_32 */

static sljit_u8* detect_near_jump_type(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_u8 *code, sljit_sw executable_offset)
{
	sljit_uw type = jump->flags >> TYPE_SHIFT;
	sljit_s32 short_jump;
	sljit_uw label_addr;
	sljit_uw jump_addr;

	jump_addr = (sljit_uw)code_ptr;
	if (!(jump->flags & JUMP_ADDR)) {
		label_addr = (sljit_uw)(code + jump->u.label->size);

		if (jump->u.label->size > jump->addr)
			jump_addr = (sljit_uw)(code + jump->addr);
	} else
		label_addr = jump->u.target - (sljit_uw)executable_offset;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if ((sljit_sw)(label_addr - (jump_addr + 6)) > HALFWORD_MAX || (sljit_sw)(label_addr - (jump_addr + 5)) < HALFWORD_MIN)
		return detect_far_jump_type(jump, code_ptr);
#endif /* SLJIT_CONFIG_X86_64 */

	short_jump = (sljit_sw)(label_addr - (jump_addr + 2)) >= -0x80 && (sljit_sw)(label_addr - (jump_addr + 2)) <= 0x7f;

	if (type == SLJIT_JUMP) {
		if (short_jump)
			*code_ptr++ = JMP_i8;
		else
			*code_ptr++ = JMP_i32;
	} else if (type > SLJIT_JUMP) {
		short_jump = 0;
		*code_ptr++ = CALL_i32;
	} else if (short_jump) {
		*code_ptr++ = U8(get_jump_code(type) - 0x10);
	} else {
		*code_ptr++ = GROUP_0F;
		*code_ptr++ = get_jump_code(type);
	}

	jump->addr = (sljit_uw)code_ptr;

	if (short_jump) {
		jump->flags |= PATCH_MB;
		code_ptr += sizeof(sljit_s8);
	} else {
		jump->flags |= PATCH_MW;
		code_ptr += sizeof(sljit_s32);
	}

	return code_ptr;
}

static void generate_jump_or_mov_addr(struct sljit_jump *jump, sljit_sw executable_offset)
{
	sljit_uw flags = jump->flags;
	sljit_uw addr = (flags & JUMP_ADDR) ? jump->u.target : jump->u.label->u.addr;
	sljit_uw jump_addr = jump->addr;
	SLJIT_UNUSED_ARG(executable_offset);

	if (SLJIT_UNLIKELY(flags & JUMP_MOV_ADDR)) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		sljit_unaligned_store_sw((void*)(jump_addr - sizeof(sljit_sw)), (sljit_sw)addr);
#else /* SLJIT_CONFIG_X86_32 */
		if (flags & PATCH_MD) {
			SLJIT_ASSERT(addr > HALFWORD_MAX);
			sljit_unaligned_store_sw((void*)(jump_addr - sizeof(sljit_sw)), (sljit_sw)addr);
			return;
		}

		if (flags & PATCH_MW) {
			addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET((sljit_u8*)jump_addr, executable_offset);
			SLJIT_ASSERT((sljit_sw)addr <= HALFWORD_MAX && (sljit_sw)addr >= HALFWORD_MIN);
		} else {
			SLJIT_ASSERT(addr <= HALFWORD_MAX);
		}
		sljit_unaligned_store_s32((void*)(jump_addr - sizeof(sljit_s32)), (sljit_s32)addr);
#endif /* !SLJIT_CONFIG_X86_32 */
		return;
	}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (SLJIT_UNLIKELY(flags & PATCH_MD)) {
		SLJIT_ASSERT(!(flags & JUMP_ADDR));
		sljit_unaligned_store_sw((void*)jump_addr, (sljit_sw)addr);
		return;
	}
#endif /* SLJIT_CONFIG_X86_64 */

	addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET((sljit_u8*)jump_addr, executable_offset);

	if (flags & PATCH_MB) {
		addr -= sizeof(sljit_s8);
		SLJIT_ASSERT((sljit_sw)addr <= 0x7f && (sljit_sw)addr >= -0x80);
		*(sljit_u8*)jump_addr = U8(addr);
		return;
	} else if (flags & PATCH_MW) {
		addr -= sizeof(sljit_s32);
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		sljit_unaligned_store_sw((void*)jump_addr, (sljit_sw)addr);
#else /* !SLJIT_CONFIG_X86_32 */
		SLJIT_ASSERT((sljit_sw)addr <= HALFWORD_MAX && (sljit_sw)addr >= HALFWORD_MIN);
		sljit_unaligned_store_s32((void*)jump_addr, (sljit_s32)addr);
#endif /* SLJIT_CONFIG_X86_32 */
	}
}

static void reduce_code_size(struct sljit_compiler *compiler)
{
	struct sljit_label *label;
	struct sljit_jump *jump;
	sljit_uw next_label_size;
	sljit_uw next_jump_addr;
	sljit_uw next_min_addr;
	sljit_uw size_reduce = 0;
	sljit_sw diff;
	sljit_uw type;
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
	sljit_uw size_reduce_max;
#endif /* SLJIT_DEBUG */

	label = compiler->labels;
	jump = compiler->jumps;

	next_label_size = SLJIT_GET_NEXT_SIZE(label);
	next_jump_addr = SLJIT_GET_NEXT_ADDRESS(jump);

	while (1) {
		next_min_addr = next_label_size;
		if (next_jump_addr < next_min_addr)
			next_min_addr = next_jump_addr;

		if (next_min_addr == SLJIT_MAX_ADDRESS)
			break;

		if (next_min_addr == next_label_size) {
			label->size -= size_reduce;

			label = label->next;
			next_label_size = SLJIT_GET_NEXT_SIZE(label);
		}

		if (next_min_addr != next_jump_addr)
			continue;

		jump->addr -= size_reduce;
		if (!(jump->flags & JUMP_MOV_ADDR)) {
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
			size_reduce_max = size_reduce + (((jump->flags >> TYPE_SHIFT) < SLJIT_JUMP) ? CJUMP_MAX_SIZE : JUMP_MAX_SIZE);
#endif /* SLJIT_DEBUG */

			if (!(jump->flags & SLJIT_REWRITABLE_JUMP)) {
				if (jump->flags & JUMP_ADDR) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
					if (jump->u.target <= 0xffffffffl)
						size_reduce += sizeof(sljit_s32);
#endif /* SLJIT_CONFIG_X86_64 */
				} else {
					/* Unit size: instruction. */
					diff = (sljit_sw)jump->u.label->size - (sljit_sw)jump->addr;
					if (jump->u.label->size > jump->addr) {
						SLJIT_ASSERT(jump->u.label->size - size_reduce >= jump->addr);
						diff -= (sljit_sw)size_reduce;
					}
					type = jump->flags >> TYPE_SHIFT;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
					if (type == SLJIT_JUMP) {
						if (diff <= 0x7f + 2 && diff >= -0x80 + 2)
							size_reduce += JUMP_MAX_SIZE - 2;
						else if (diff <= HALFWORD_MAX + 5 && diff >= HALFWORD_MIN + 5)
							size_reduce += JUMP_MAX_SIZE - 5;
					} else if (type < SLJIT_JUMP) {
						if (diff <= 0x7f + 2 && diff >= -0x80 + 2)
							size_reduce += CJUMP_MAX_SIZE - 2;
						else if (diff <= HALFWORD_MAX + 6 && diff >= HALFWORD_MIN + 6)
							size_reduce += CJUMP_MAX_SIZE - 6;
					} else  {
						if (diff <= HALFWORD_MAX + 5 && diff >= HALFWORD_MIN + 5)
							size_reduce += JUMP_MAX_SIZE - 5;
					}
#else /* !SLJIT_CONFIG_X86_64 */
					if (type == SLJIT_JUMP) {
						if (diff <= 0x7f + 2 && diff >= -0x80 + 2)
							size_reduce += JUMP_MAX_SIZE - 2;
					} else if (type < SLJIT_JUMP) {
						if (diff <= 0x7f + 2 && diff >= -0x80 + 2)
							size_reduce += CJUMP_MAX_SIZE - 2;
					}
#endif /* SLJIT_CONFIG_X86_64 */
				}
			}

#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
			jump->flags |= (size_reduce_max - size_reduce) << JUMP_SIZE_SHIFT;
#endif /* SLJIT_DEBUG */
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		} else {
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
			size_reduce_max = size_reduce + 10;
#endif /* SLJIT_DEBUG */

			if (!(jump->flags & JUMP_ADDR)) {
				diff = (sljit_sw)jump->u.label->size - (sljit_sw)(jump->addr - 3);

				if (diff <= HALFWORD_MAX && diff >= HALFWORD_MIN)
					size_reduce += 3;
			} else if (jump->u.target <= 0xffffffffl)
				size_reduce += (jump->flags & MOV_ADDR_HI) ? 4 : 5;

#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
			jump->flags |= (size_reduce_max - size_reduce) << JUMP_SIZE_SHIFT;
#endif /* SLJIT_DEBUG */
#endif /* SLJIT_CONFIG_X86_64 */
		}

		jump = jump->next;
		next_jump_addr = SLJIT_GET_NEXT_ADDRESS(jump);
	}

	compiler->size -= size_reduce;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler, sljit_s32 options, void *exec_allocator_data)
{
	struct sljit_memory_fragment *buf;
	sljit_u8 *code;
	sljit_u8 *code_ptr;
	sljit_u8 *buf_ptr;
	sljit_u8 *buf_end;
	sljit_u8 len;
	sljit_sw executable_offset;
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
	sljit_uw addr;
#endif /* SLJIT_DEBUG */

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));

	reduce_code_size(compiler);

	/* Second code generation pass. */
	code = (sljit_u8*)allocate_executable_memory(compiler->size, options, exec_allocator_data, &executable_offset);
	PTR_FAIL_WITH_EXEC_IF(code);

	reverse_buf(compiler);
	buf = compiler->buf;

	code_ptr = code;
	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;

	do {
		buf_ptr = buf->memory;
		buf_end = buf_ptr + buf->used_size;
		do {
			len = *buf_ptr++;
			SLJIT_ASSERT(len > 0);
			if (len < SLJIT_INST_CONST) {
				/* The code is already generated. */
				SLJIT_MEMCPY(code_ptr, buf_ptr, len);
				code_ptr += len;
				buf_ptr += len;
			} else {
				switch (len) {
				case SLJIT_INST_LABEL:
					label->u.addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					label->size = (sljit_uw)(code_ptr - code);
					label = label->next;
					break;
				case SLJIT_INST_JUMP:
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
					addr = (sljit_uw)code_ptr;
#endif /* SLJIT_DEBUG */
					if (!(jump->flags & SLJIT_REWRITABLE_JUMP))
						code_ptr = detect_near_jump_type(jump, code_ptr, code, executable_offset);
					else {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
						code_ptr = detect_far_jump_type(jump, code_ptr, executable_offset);
#else /* !SLJIT_CONFIG_X86_32 */
						code_ptr = detect_far_jump_type(jump, code_ptr);
#endif /* SLJIT_CONFIG_X86_32 */
					}

					SLJIT_ASSERT((sljit_uw)code_ptr - addr <= ((jump->flags >> JUMP_SIZE_SHIFT) & 0x1f));
					jump = jump->next;
					break;
				case SLJIT_INST_MOV_ADDR:
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
					code_ptr = generate_mov_addr_code(jump, code_ptr, code, executable_offset);
#endif /* SLJIT_CONFIG_X86_64 */
					jump->addr = (sljit_uw)code_ptr;
					jump = jump->next;
					break;
				default:
					SLJIT_ASSERT(len == SLJIT_INST_CONST);
					const_->addr = ((sljit_uw)code_ptr) - sizeof(sljit_sw);
					const_ = const_->next;
					break;
				}
			}
		} while (buf_ptr < buf_end);

		SLJIT_ASSERT(buf_ptr == buf_end);
		buf = buf->next;
	} while (buf);

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(code_ptr <= code + compiler->size);

	jump = compiler->jumps;
	while (jump) {
		generate_jump_or_mov_addr(jump, executable_offset);
		jump = jump->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (sljit_uw)(code_ptr - code);

	code = (sljit_u8*)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);

	SLJIT_UPDATE_WX_FLAGS(code, (sljit_u8*)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset), 1);
	return (void*)code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	switch (feature_type) {
	case SLJIT_HAS_FPU:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#elif (defined SLJIT_DETECT_SSE2 && SLJIT_DETECT_SSE2)
		if (cpu_feature_list == 0)
			get_cpu_features();
		return (cpu_feature_list & CPU_FEATURE_SSE2) != 0;
#else /* SLJIT_DETECT_SSE2 */
		return 1;
#endif /* SLJIT_DETECT_SSE2 */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	case SLJIT_HAS_VIRTUAL_REGISTERS:
		return 1;
#endif /* SLJIT_CONFIG_X86_32 */

	case SLJIT_HAS_CLZ:
		if (cpu_feature_list == 0)
			get_cpu_features();

		return (cpu_feature_list & CPU_FEATURE_LZCNT) ? 1 : 2;

	case SLJIT_HAS_CTZ:
		if (cpu_feature_list == 0)
			get_cpu_features();

		return (cpu_feature_list & CPU_FEATURE_TZCNT) ? 1 : 2;

	case SLJIT_HAS_CMOV:
		if (cpu_feature_list == 0)
			get_cpu_features();
		return (cpu_feature_list & CPU_FEATURE_CMOV) != 0;

	case SLJIT_HAS_REV:
	case SLJIT_HAS_ROT:
	case SLJIT_HAS_PREFETCH:
	case SLJIT_HAS_COPY_F32:
	case SLJIT_HAS_COPY_F64:
	case SLJIT_HAS_ATOMIC:
	case SLJIT_HAS_MEMORY_BARRIER:
		return 1;

#if !(defined SLJIT_IS_FPU_AVAILABLE) || SLJIT_IS_FPU_AVAILABLE
	case SLJIT_HAS_AVX:
		if (cpu_feature_list == 0)
			get_cpu_features();
		return (cpu_feature_list & CPU_FEATURE_AVX) != 0;
	case SLJIT_HAS_AVX2:
		if (cpu_feature_list == 0)
			get_cpu_features();
		return (cpu_feature_list & CPU_FEATURE_AVX2) != 0;
	case SLJIT_HAS_SIMD:
		if (cpu_feature_list == 0)
			get_cpu_features();
		return (cpu_feature_list & CPU_FEATURE_SSE41) != 0;
#endif /* SLJIT_IS_FPU_AVAILABLE */
	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	switch (type) {
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
		return 2;
	}

	return 0;
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

#define BINARY_OPCODE(opcode) (((opcode ## _EAX_i32) << 24) | ((opcode ## _r_rm) << 16) | ((opcode ## _rm_r) << 8) | (opcode))

#define BINARY_IMM32(op_imm, immw, arg, argw) \
	do { \
		inst = emit_x86_instruction(compiler, 1 | EX86_BIN_INS, SLJIT_IMM, immw, arg, argw); \
		FAIL_IF(!inst); \
		*(inst + 1) |= (op_imm); \
	} while (0)

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)

#define BINARY_IMM(op_imm, op_mr, immw, arg, argw) \
	do { \
		if (IS_HALFWORD(immw) || compiler->mode32) { \
			BINARY_IMM32(op_imm, immw, arg, argw); \
		} \
		else { \
			FAIL_IF(emit_load_imm64(compiler, FAST_IS_REG(arg) ? TMP_REG2 : TMP_REG1, immw)); \
			inst = emit_x86_instruction(compiler, 1, FAST_IS_REG(arg) ? TMP_REG2 : TMP_REG1, 0, arg, argw); \
			FAIL_IF(!inst); \
			*inst = (op_mr); \
		} \
	} while (0)

#define BINARY_EAX_IMM(op_eax_imm, immw) \
	FAIL_IF(emit_do_imm32(compiler, (!compiler->mode32) ? REX_W : 0, (op_eax_imm), immw))

#else /* !SLJIT_CONFIG_X86_64 */

#define BINARY_IMM(op_imm, op_mr, immw, arg, argw) \
	BINARY_IMM32(op_imm, immw, arg, argw)

#define BINARY_EAX_IMM(op_eax_imm, immw) \
	FAIL_IF(emit_do_imm(compiler, (op_eax_imm), immw))

#endif /* SLJIT_CONFIG_X86_64 */

static sljit_s32 emit_byte(struct sljit_compiler *compiler, sljit_u8 byte)
{
	sljit_u8 *inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
	FAIL_IF(!inst);
	INC_SIZE(1);
	*inst = byte;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_mov(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

#define EMIT_MOV(compiler, dst, dstw, src, srcw) \
	FAIL_IF(emit_mov(compiler, dst, dstw, src, srcw));

static sljit_s32 emit_groupf(struct sljit_compiler *compiler,
	sljit_uw op,
	sljit_s32 dst, sljit_s32 src, sljit_sw srcw);

static sljit_s32 emit_groupf_ext(struct sljit_compiler *compiler,
	sljit_uw op,
	sljit_s32 dst, sljit_s32 src, sljit_sw srcw);

static SLJIT_INLINE sljit_s32 emit_sse2_store(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_sw dstw, sljit_s32 src);

static SLJIT_INLINE sljit_s32 emit_sse2_load(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_s32 src, sljit_sw srcw);

static sljit_s32 emit_cmp_binary(struct sljit_compiler *compiler,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

static sljit_s32 emit_cmov_generic(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw);

static SLJIT_INLINE sljit_s32 emit_endbranch(struct sljit_compiler *compiler)
{
#if (defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET)
	/* Emit endbr32/endbr64 when CET is enabled.  */
	sljit_u8 *inst;
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 4);
	FAIL_IF(!inst);
	INC_SIZE(4);
	inst[0] = GROUP_F3;
	inst[1] = GROUP_0F;
	inst[2] = 0x1e;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	inst[3] = 0xfb;
#else /* !SLJIT_CONFIG_X86_32 */
	inst[3] = 0xfa;
#endif /* SLJIT_CONFIG_X86_32 */
#else /* !SLJIT_CONFIG_X86_CET */
	SLJIT_UNUSED_ARG(compiler);
#endif /* SLJIT_CONFIG_X86_CET */
	return SLJIT_SUCCESS;
}

#if (defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET) && defined (__SHSTK__)

static SLJIT_INLINE sljit_s32 emit_rdssp(struct sljit_compiler *compiler, sljit_s32 reg)
{
	sljit_u8 *inst;
	sljit_s32 size;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	size = 5;
#else
	size = 4;
#endif

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);
	*inst++ = GROUP_F3;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	*inst++ = REX_W | (reg_map[reg] <= 7 ? 0 : REX_B);
#endif
	inst[0] = GROUP_0F;
	inst[1] = 0x1e;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	inst[2] = U8(MOD_REG | (0x1 << 3) | reg_lmap[reg]);
#else
	inst[2] = U8(MOD_REG | (0x1 << 3) | reg_map[reg]);
#endif
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_incssp(struct sljit_compiler *compiler, sljit_s32 reg)
{
	sljit_u8 *inst;
	sljit_s32 size;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	size = 5;
#else
	size = 4;
#endif

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);
	*inst++ = GROUP_F3;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	*inst++ = REX_W | (reg_map[reg] <= 7 ? 0 : REX_B);
#endif
	inst[0] = GROUP_0F;
	inst[1] = 0xae;
	inst[2] = (0x3 << 6) | (0x5 << 3) | (reg_map[reg] & 0x7);
	return SLJIT_SUCCESS;
}

#endif /* SLJIT_CONFIG_X86_CET && __SHSTK__ */

static SLJIT_INLINE sljit_s32 cpu_has_shadow_stack(void)
{
#if (defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET) && defined (__SHSTK__)
	return _get_ssp() != 0;
#else /* !SLJIT_CONFIG_X86_CET || !__SHSTK__ */
	return 0;
#endif /* SLJIT_CONFIG_X86_CET && __SHSTK__ */
}

static SLJIT_INLINE sljit_s32 adjust_shadow_stack(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET) && defined (__SHSTK__)
	sljit_u8 *inst, *jz_after_cmp_inst;
	sljit_uw size_jz_after_cmp_inst;

	sljit_uw size_before_rdssp_inst = compiler->size;

	/* Generate "RDSSP TMP_REG1". */
	FAIL_IF(emit_rdssp(compiler, TMP_REG1));

	/* Load return address on shadow stack into TMP_REG1. */
	EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(TMP_REG1), 0);

	/* Compare return address against TMP_REG1. */
	FAIL_IF(emit_cmp_binary (compiler, TMP_REG1, 0, src, srcw));

	/* Generate JZ to skip shadow stack ajdustment when shadow
	   stack matches normal stack. */
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(2);
	*inst++ = get_jump_code(SLJIT_EQUAL) - 0x10;
	size_jz_after_cmp_inst = compiler->size;
	jz_after_cmp_inst = inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	/* REX_W is not necessary. */
	compiler->mode32 = 1;
#endif
	/* Load 1 into TMP_REG1. */
	EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, 1);

	/* Generate "INCSSP TMP_REG1". */
	FAIL_IF(emit_incssp(compiler, TMP_REG1));

	/* Jump back to "RDSSP TMP_REG1" to check shadow stack again. */
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(2);
	inst[0] = JMP_i8;
	inst[1] = size_before_rdssp_inst - compiler->size;

	*jz_after_cmp_inst = compiler->size - size_jz_after_cmp_inst;
#else /* !SLJIT_CONFIG_X86_CET || !__SHSTK__ */
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
#endif /* SLJIT_CONFIG_X86_CET && __SHSTK__ */
	return SLJIT_SUCCESS;
}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
#include "sljitNativeX86_32.c"
#else
#include "sljitNativeX86_64.c"
#endif

static sljit_s32 emit_mov(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;

	if (FAST_IS_REG(src)) {
		inst = emit_x86_instruction(compiler, 1, src, 0, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm_r;
		return SLJIT_SUCCESS;
	}

	if (src == SLJIT_IMM) {
		if (FAST_IS_REG(dst)) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
			return emit_do_imm(compiler, MOV_r_i32 | reg_map[dst], srcw);
#else
			if (!compiler->mode32) {
				if (NOT_HALFWORD(srcw))
					return emit_load_imm64(compiler, dst, srcw);
			}
			else
				return emit_do_imm32(compiler, (reg_map[dst] >= 8) ? REX_B : 0, U8(MOV_r_i32 | reg_lmap[dst]), srcw);
#endif
		}
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (!compiler->mode32 && NOT_HALFWORD(srcw)) {
			/* Immediate to memory move. Only SLJIT_MOV operation copies
			   an immediate directly into memory so TMP_REG1 can be used. */
			FAIL_IF(emit_load_imm64(compiler, TMP_REG1, srcw));
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, dst, dstw);
			FAIL_IF(!inst);
			*inst = MOV_rm_r;
			return SLJIT_SUCCESS;
		}
#endif
		inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, srcw, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm_i32;
		return SLJIT_SUCCESS;
	}
	if (FAST_IS_REG(dst)) {
		inst = emit_x86_instruction(compiler, 1, dst, 0, src, srcw);
		FAIL_IF(!inst);
		*inst = MOV_r_rm;
		return SLJIT_SUCCESS;
	}

	/* Memory to memory move. Only SLJIT_MOV operation copies
	   data from memory to memory so TMP_REG1 can be used. */
	inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src, srcw);
	FAIL_IF(!inst);
	*inst = MOV_r_rm;
	inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, dst, dstw);
	FAIL_IF(!inst);
	*inst = MOV_rm_r;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_cmov_generic(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;
	sljit_uw size;

	SLJIT_ASSERT(type >= SLJIT_EQUAL && type <= SLJIT_ORDERED_LESS_EQUAL);

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(2);
	inst[0] = U8(get_jump_code((sljit_uw)type ^ 0x1) - 0x10);

	size = compiler->size;
	EMIT_MOV(compiler, dst_reg, 0, src, srcw);

	inst[1] = U8(compiler->size - size);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	sljit_u8 *inst;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_uw size;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	switch (GET_OPCODE(op)) {
	case SLJIT_BREAKPOINT:
		return emit_byte(compiler, INT3);
	case SLJIT_NOP:
		return emit_byte(compiler, NOP);
	case SLJIT_LMUL_UW:
	case SLJIT_LMUL_SW:
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
#ifdef _WIN64
		SLJIT_ASSERT(
			reg_map[SLJIT_R0] == 0
			&& reg_map[SLJIT_R1] == 2
			&& reg_map[TMP_REG1] > 7);
#else
		SLJIT_ASSERT(
			reg_map[SLJIT_R0] == 0
			&& reg_map[SLJIT_R1] < 7
			&& reg_map[TMP_REG1] == 2);
#endif
		compiler->mode32 = op & SLJIT_32;
#endif
		SLJIT_COMPILE_ASSERT((SLJIT_DIVMOD_UW & 0x2) == 0 && SLJIT_DIV_UW - 0x2 == SLJIT_DIVMOD_UW, bad_div_opcode_assignments);

		op = GET_OPCODE(op);
		if ((op | 0x2) == SLJIT_DIV_UW) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) || defined(_WIN64)
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_R1, 0);
			inst = emit_x86_instruction(compiler, 1, SLJIT_R1, 0, SLJIT_R1, 0);
#else
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, TMP_REG1, 0);
#endif
			FAIL_IF(!inst);
			*inst = XOR_r_rm;
		}

		if ((op | 0x2) == SLJIT_DIV_SW) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) || defined(_WIN64)
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_R1, 0);
#endif

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
			FAIL_IF(emit_byte(compiler, CDQ));
#else
			if (!compiler->mode32) {
				inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
				FAIL_IF(!inst);
				INC_SIZE(2);
				inst[0] = REX_W;
				inst[1] = CDQ;
			} else
				FAIL_IF(emit_byte(compiler, CDQ));
#endif
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
		FAIL_IF(!inst);
		INC_SIZE(2);
		inst[0] = GROUP_F7;
		inst[1] = MOD_REG | ((op >= SLJIT_DIVMOD_UW) ? reg_map[TMP_REG1] : reg_map[SLJIT_R1]);
#else /* !SLJIT_CONFIG_X86_32 */
#ifdef _WIN64
		size = (!compiler->mode32 || op >= SLJIT_DIVMOD_UW) ? 3 : 2;
#else /* !_WIN64 */
		size = (!compiler->mode32) ? 3 : 2;
#endif /* _WIN64 */
		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
#ifdef _WIN64
		if (!compiler->mode32)
			*inst++ = REX_W | ((op >= SLJIT_DIVMOD_UW) ? REX_B : 0);
		else if (op >= SLJIT_DIVMOD_UW)
			*inst++ = REX_B;
		inst[0] = GROUP_F7;
		inst[1] = MOD_REG | ((op >= SLJIT_DIVMOD_UW) ? reg_lmap[TMP_REG1] : reg_lmap[SLJIT_R1]);
#else /* !_WIN64 */
		if (!compiler->mode32)
			*inst++ = REX_W;
		inst[0] = GROUP_F7;
		inst[1] = MOD_REG | reg_map[SLJIT_R1];
#endif /* _WIN64 */
#endif /* SLJIT_CONFIG_X86_32 */
		switch (op) {
		case SLJIT_LMUL_UW:
			inst[1] |= MUL;
			break;
		case SLJIT_LMUL_SW:
			inst[1] |= IMUL;
			break;
		case SLJIT_DIVMOD_UW:
		case SLJIT_DIV_UW:
			inst[1] |= DIV;
			break;
		case SLJIT_DIVMOD_SW:
		case SLJIT_DIV_SW:
			inst[1] |= IDIV;
			break;
		}
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) && !defined(_WIN64)
		if (op <= SLJIT_DIVMOD_SW)
			EMIT_MOV(compiler, SLJIT_R1, 0, TMP_REG1, 0);
#else
		if (op >= SLJIT_DIV_UW)
			EMIT_MOV(compiler, SLJIT_R1, 0, TMP_REG1, 0);
#endif
		break;
	case SLJIT_MEMORY_BARRIER:
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 3);
		FAIL_IF(!inst);
		INC_SIZE(3);
		inst[0] = GROUP_0F;
		inst[1] = 0xae;
		inst[2] = 0xf0;
		return SLJIT_SUCCESS;
	case SLJIT_ENDBR:
		return emit_endbranch(compiler);
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return skip_frames_before_return(compiler);
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_mov_byte(struct sljit_compiler *compiler, sljit_s32 sign,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;
	sljit_s32 dst_r;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
#endif

	if (src == SLJIT_IMM) {
		if (FAST_IS_REG(dst)) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
			return emit_do_imm(compiler, MOV_r_i32 | reg_map[dst], srcw);
#else
			inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, srcw, dst, 0);
			FAIL_IF(!inst);
			*inst = MOV_rm_i32;
			return SLJIT_SUCCESS;
#endif
		}
		inst = emit_x86_instruction(compiler, 1 | EX86_BYTE_ARG | EX86_NO_REXW, SLJIT_IMM, srcw, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm8_i8;
		return SLJIT_SUCCESS;
	}

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if ((dst & SLJIT_MEM) && FAST_IS_REG(src)) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (reg_map[src] >= 4) {
			SLJIT_ASSERT(dst_r == TMP_REG1);
			EMIT_MOV(compiler, TMP_REG1, 0, src, 0);
		} else
			dst_r = src;
#else
		dst_r = src;
#endif
	} else {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (FAST_IS_REG(src) && reg_map[src] >= 4) {
			/* Both src and dst are registers. */
			SLJIT_ASSERT(FAST_IS_REG(dst));

			if (src == dst && !sign) {
				inst = emit_x86_instruction(compiler, 1 | EX86_BIN_INS, SLJIT_IMM, 0xff, dst, 0);
				FAIL_IF(!inst);
				*(inst + 1) |= AND;
				return SLJIT_SUCCESS;
			}

			EMIT_MOV(compiler, TMP_REG1, 0, src, 0);
			src = TMP_REG1;
			srcw = 0;
		}
#endif /* !SLJIT_CONFIG_X86_32 */

		/* src can be memory addr or reg_map[src] < 4 on x86_32 architectures. */
		FAIL_IF(emit_groupf(compiler, sign ? MOVSX_r_rm8 : MOVZX_r_rm8, dst_r, src, srcw));
	}

	if (dst & SLJIT_MEM) {
		inst = emit_x86_instruction(compiler, 1 | EX86_REX | EX86_NO_REXW, dst_r, 0, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm8_r8;
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_prefetch(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif

	inst = emit_x86_instruction(compiler, 2, 0, 0, src, srcw);
	FAIL_IF(!inst);
	inst[0] = GROUP_0F;
	inst[1] = PREFETCH;

	if (op == SLJIT_PREFETCH_L1)
		inst[2] |= (1 << 3);
	else if (op == SLJIT_PREFETCH_L2)
		inst[2] |= (2 << 3);
	else if (op == SLJIT_PREFETCH_L3)
		inst[2] |= (3 << 3);

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_mov_half(struct sljit_compiler *compiler, sljit_s32 sign,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;
	sljit_s32 dst_r;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
#endif

	if (src == SLJIT_IMM) {
		if (FAST_IS_REG(dst)) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
			return emit_do_imm(compiler, MOV_r_i32 | reg_map[dst], srcw);
#else
			inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, srcw, dst, 0);
			FAIL_IF(!inst);
			*inst = MOV_rm_i32;
			return SLJIT_SUCCESS;
#endif
		}
		inst = emit_x86_instruction(compiler, 1 | EX86_HALF_ARG | EX86_NO_REXW | EX86_PREF_66, SLJIT_IMM, srcw, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm_i32;
		return SLJIT_SUCCESS;
	}

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if ((dst & SLJIT_MEM) && FAST_IS_REG(src))
		dst_r = src;
	else
		FAIL_IF(emit_groupf(compiler, sign ? MOVSX_r_rm16 : MOVZX_r_rm16, dst_r, src, srcw));

	if (dst & SLJIT_MEM) {
		inst = emit_x86_instruction(compiler, 1 | EX86_NO_REXW | EX86_PREF_66, dst_r, 0, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm_r;
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_unary(struct sljit_compiler *compiler, sljit_u8 opcode,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;

	if (dst == src && dstw == srcw) {
		/* Same input and output */
		inst = emit_x86_instruction(compiler, 1, 0, 0, dst, dstw);
		FAIL_IF(!inst);
		inst[0] = GROUP_F7;
		inst[1] |= opcode;
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(dst)) {
		EMIT_MOV(compiler, dst, 0, src, srcw);
		inst = emit_x86_instruction(compiler, 1, 0, 0, dst, 0);
		FAIL_IF(!inst);
		inst[0] = GROUP_F7;
		inst[1] |= opcode;
		return SLJIT_SUCCESS;
	}

	EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
	inst = emit_x86_instruction(compiler, 1, 0, 0, TMP_REG1, 0);
	FAIL_IF(!inst);
	inst[0] = GROUP_F7;
	inst[1] |= opcode;
	EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
	return SLJIT_SUCCESS;
}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
static const sljit_sw emit_clz_arg = 32 + 31;
static const sljit_sw emit_ctz_arg = 32;
#endif

static sljit_s32 emit_clz_ctz(struct sljit_compiler *compiler, sljit_s32 is_clz,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;
	sljit_s32 dst_r;
	sljit_sw max;

	SLJIT_ASSERT(cpu_feature_list != 0);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (is_clz ? (cpu_feature_list & CPU_FEATURE_LZCNT) : (cpu_feature_list & CPU_FEATURE_TZCNT)) {
		FAIL_IF(emit_groupf(compiler, (is_clz ? LZCNT_r_rm : TZCNT_r_rm) | EX86_PREF_F3, dst_r, src, srcw));

		if (dst & SLJIT_MEM)
			EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
		return SLJIT_SUCCESS;
	}

	FAIL_IF(emit_groupf(compiler, is_clz ? BSR_r_rm : BSF_r_rm, dst_r, src, srcw));

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	max = is_clz ? (32 + 31) : 32;

	if (cpu_feature_list & CPU_FEATURE_CMOV) {
		if (dst_r != TMP_REG1) {
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, max);
			inst = emit_x86_instruction(compiler, 2, dst_r, 0, TMP_REG1, 0);
		}
		else
			inst = emit_x86_instruction(compiler, 2, dst_r, 0, SLJIT_MEM0(), is_clz ? (sljit_sw)&emit_clz_arg : (sljit_sw)&emit_ctz_arg);

		FAIL_IF(!inst);
		inst[0] = GROUP_0F;
		inst[1] = CMOVE_r_rm;
	}
	else
		FAIL_IF(emit_cmov_generic(compiler, SLJIT_EQUAL, dst_r, SLJIT_IMM, max));

	if (is_clz) {
		inst = emit_x86_instruction(compiler, 1 | EX86_BIN_INS, SLJIT_IMM, 31, dst_r, 0);
		FAIL_IF(!inst);
		*(inst + 1) |= XOR;
	}
#else
	if (is_clz)
		max = compiler->mode32 ? (32 + 31) : (64 + 63);
	else
		max = compiler->mode32 ? 32 : 64;

	if (cpu_feature_list & CPU_FEATURE_CMOV) {
		EMIT_MOV(compiler, TMP_REG2, 0, SLJIT_IMM, max);
		FAIL_IF(emit_groupf(compiler, CMOVE_r_rm, dst_r, TMP_REG2, 0));
	} else
		FAIL_IF(emit_cmov_generic(compiler, SLJIT_EQUAL, dst_r, SLJIT_IMM, max));

	if (is_clz) {
		inst = emit_x86_instruction(compiler, 1 | EX86_BIN_INS, SLJIT_IMM, max >> 1, dst_r, 0);
		FAIL_IF(!inst);
		*(inst + 1) |= XOR;
	}
#endif

	if (dst & SLJIT_MEM)
		EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_bswap(struct sljit_compiler *compiler,
	sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst;
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	sljit_uw size;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_u8 rex = 0;
#else /* !SLJIT_CONFIG_X86_64 */
	sljit_s32 dst_is_ereg = op & SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (op == SLJIT_REV_U32 || op == SLJIT_REV_S32)
		compiler->mode32 = 1;
#else /* !SLJIT_CONFIG_X86_64 */
	op &= ~SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */

	if (src != dst_r) {
		/* Only the lower 16 bit is read for eregs. */
		if (op == SLJIT_REV_U16 || op == SLJIT_REV_S16)
			FAIL_IF(emit_mov_half(compiler, 0, dst_r, 0, src, srcw));
		else
			EMIT_MOV(compiler, dst_r, 0, src, srcw);
	}

	size = 2;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (!compiler->mode32)
		rex = REX_W;

	if (reg_map[dst_r] >= 8)
		rex |= REX_B;

	if (rex != 0)
		size++;
#endif /* SLJIT_CONFIG_X86_64 */

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (rex != 0)
		*inst++ = rex;

	inst[0] = GROUP_0F;
	inst[1] = BSWAP_r | reg_lmap[dst_r];
#else /* !SLJIT_CONFIG_X86_64 */
	inst[0] = GROUP_0F;
	inst[1] = BSWAP_r | reg_map[dst_r];
#endif /* SLJIT_CONFIG_X86_64 */

	if (op == SLJIT_REV_U16 || op == SLJIT_REV_S16) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		size = compiler->mode32 ? 16 : 48;
#else /* !SLJIT_CONFIG_X86_64 */
		size = 16;
#endif /* SLJIT_CONFIG_X86_64 */

		inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_IMM, (sljit_sw)size, dst_r, 0);
		FAIL_IF(!inst);
		if (op == SLJIT_REV_U16)
			inst[1] |= SHR;
		else
			inst[1] |= SAR;
	}

	if (dst & SLJIT_MEM) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (dst_is_ereg)
			op = SLJIT_REV;
#endif /* SLJIT_CONFIG_X86_32 */
		if (op == SLJIT_REV_U16 || op == SLJIT_REV_S16)
			return emit_mov_half(compiler, 0, dst, dstw, TMP_REG1, 0);

		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
	}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (op == SLJIT_REV_S32) {
		compiler->mode32 = 0;
		inst = emit_x86_instruction(compiler, 1, dst, 0, dst, 0);
		FAIL_IF(!inst);
		*inst = MOVSXD_r_rm;
	}
#endif /* SLJIT_CONFIG_X86_64 */

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 dst_is_ereg = 0;
#else /* !SLJIT_CONFIG_X86_32 */
	sljit_s32 op_flags = GET_ALL_FLAGS(op);
#endif /* SLJIT_CONFIG_X86_32 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	CHECK_EXTRA_REGS(dst, dstw, dst_is_ereg = 1);
	CHECK_EXTRA_REGS(src, srcw, (void)0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op_flags & SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */

	op = GET_OPCODE(op);

	if (op >= SLJIT_MOV && op <= SLJIT_MOV_P) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;
#endif /* SLJIT_CONFIG_X86_64 */

		if (FAST_IS_REG(src) && src == dst) {
			if (!TYPE_CAST_NEEDED(op))
				return SLJIT_SUCCESS;
		}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (op_flags & SLJIT_32) {
			if (src & SLJIT_MEM) {
				if (op == SLJIT_MOV_S32)
					op = SLJIT_MOV_U32;
			}
			else if (src == SLJIT_IMM) {
				if (op == SLJIT_MOV_U32)
					op = SLJIT_MOV_S32;
			}
		}
#endif /* SLJIT_CONFIG_X86_64 */

		if (src == SLJIT_IMM) {
			switch (op) {
			case SLJIT_MOV_U8:
				srcw = (sljit_u8)srcw;
				break;
			case SLJIT_MOV_S8:
				srcw = (sljit_s8)srcw;
				break;
			case SLJIT_MOV_U16:
				srcw = (sljit_u16)srcw;
				break;
			case SLJIT_MOV_S16:
				srcw = (sljit_s16)srcw;
				break;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			case SLJIT_MOV_U32:
				srcw = (sljit_u32)srcw;
				break;
			case SLJIT_MOV_S32:
				srcw = (sljit_s32)srcw;
				break;
#endif /* SLJIT_CONFIG_X86_64 */
			}
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
			if (SLJIT_UNLIKELY(dst_is_ereg))
				return emit_mov(compiler, dst, dstw, src, srcw);
#endif /* SLJIT_CONFIG_X86_32 */
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (SLJIT_UNLIKELY(dst_is_ereg) && (!(op == SLJIT_MOV || op == SLJIT_MOV_U32 || op == SLJIT_MOV_S32 || op == SLJIT_MOV_P) || (src & SLJIT_MEM))) {
			SLJIT_ASSERT(dst == SLJIT_MEM1(SLJIT_SP));
			dst = TMP_REG1;
		}
#endif /* SLJIT_CONFIG_X86_32 */

		switch (op) {
		case SLJIT_MOV:
		case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		case SLJIT_MOV_U32:
		case SLJIT_MOV_S32:
		case SLJIT_MOV32:
#endif /* SLJIT_CONFIG_X86_32 */
			EMIT_MOV(compiler, dst, dstw, src, srcw);
			break;
		case SLJIT_MOV_U8:
			FAIL_IF(emit_mov_byte(compiler, 0, dst, dstw, src, srcw));
			break;
		case SLJIT_MOV_S8:
			FAIL_IF(emit_mov_byte(compiler, 1, dst, dstw, src, srcw));
			break;
		case SLJIT_MOV_U16:
			FAIL_IF(emit_mov_half(compiler, 0, dst, dstw, src, srcw));
			break;
		case SLJIT_MOV_S16:
			FAIL_IF(emit_mov_half(compiler, 1, dst, dstw, src, srcw));
			break;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		case SLJIT_MOV_U32:
			FAIL_IF(emit_mov_int(compiler, 0, dst, dstw, src, srcw));
			break;
		case SLJIT_MOV_S32:
			FAIL_IF(emit_mov_int(compiler, 1, dst, dstw, src, srcw));
			break;
		case SLJIT_MOV32:
			compiler->mode32 = 1;
			EMIT_MOV(compiler, dst, dstw, src, srcw);
			compiler->mode32 = 0;
			break;
#endif /* SLJIT_CONFIG_X86_64 */
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (SLJIT_UNLIKELY(dst_is_ereg) && dst == TMP_REG1)
			return emit_mov(compiler, SLJIT_MEM1(SLJIT_SP), dstw, TMP_REG1, 0);
#endif /* SLJIT_CONFIG_X86_32 */
		return SLJIT_SUCCESS;
	}

	switch (op) {
	case SLJIT_CLZ:
	case SLJIT_CTZ:
		return emit_clz_ctz(compiler, (op == SLJIT_CLZ), dst, dstw, src, srcw);
	case SLJIT_REV:
	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (dst_is_ereg)
			op |= SLJIT_32;
#endif /* SLJIT_CONFIG_X86_32 */
		return emit_bswap(compiler, op, dst, dstw, src, srcw);
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_cum_binary(struct sljit_compiler *compiler,
	sljit_u32 op_types,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;
	sljit_u8 op_eax_imm = U8(op_types >> 24);
	sljit_u8 op_rm = U8((op_types >> 16) & 0xff);
	sljit_u8 op_mr = U8((op_types >> 8) & 0xff);
	sljit_u8 op_imm = U8(op_types & 0xff);

	if (dst == src1 && dstw == src1w) {
		if (src2 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if ((dst == SLJIT_R0) && (src2w > 127 || src2w < -128) && (compiler->mode32 || IS_HALFWORD(src2w))) {
#else
			if ((dst == SLJIT_R0) && (src2w > 127 || src2w < -128)) {
#endif
				BINARY_EAX_IMM(op_eax_imm, src2w);
			}
			else {
				BINARY_IMM(op_imm, op_mr, src2w, dst, dstw);
			}
		}
		else if (FAST_IS_REG(dst)) {
			inst = emit_x86_instruction(compiler, 1, dst, dstw, src2, src2w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
		else if (FAST_IS_REG(src2)) {
			/* Special exception for sljit_emit_op_flags. */
			inst = emit_x86_instruction(compiler, 1, src2, src2w, dst, dstw);
			FAIL_IF(!inst);
			*inst = op_mr;
		}
		else {
			EMIT_MOV(compiler, TMP_REG1, 0, src2, src2w);
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, dst, dstw);
			FAIL_IF(!inst);
			*inst = op_mr;
		}
		return SLJIT_SUCCESS;
	}

	/* Only for cumulative operations. */
	if (dst == src2 && dstw == src2w) {
		if (src1 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if ((dst == SLJIT_R0) && (src1w > 127 || src1w < -128) && (compiler->mode32 || IS_HALFWORD(src1w))) {
#else
			if ((dst == SLJIT_R0) && (src1w > 127 || src1w < -128)) {
#endif
				BINARY_EAX_IMM(op_eax_imm, src1w);
			}
			else {
				BINARY_IMM(op_imm, op_mr, src1w, dst, dstw);
			}
		}
		else if (FAST_IS_REG(dst)) {
			inst = emit_x86_instruction(compiler, 1, dst, dstw, src1, src1w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
		else if (FAST_IS_REG(src1)) {
			inst = emit_x86_instruction(compiler, 1, src1, src1w, dst, dstw);
			FAIL_IF(!inst);
			*inst = op_mr;
		}
		else {
			EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, dst, dstw);
			FAIL_IF(!inst);
			*inst = op_mr;
		}
		return SLJIT_SUCCESS;
	}

	/* General version. */
	if (FAST_IS_REG(dst)) {
		EMIT_MOV(compiler, dst, 0, src1, src1w);
		if (src2 == SLJIT_IMM) {
			BINARY_IMM(op_imm, op_mr, src2w, dst, 0);
		}
		else {
			inst = emit_x86_instruction(compiler, 1, dst, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
	}
	else {
		/* This version requires less memory writing. */
		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		if (src2 == SLJIT_IMM) {
			BINARY_IMM(op_imm, op_mr, src2w, TMP_REG1, 0);
		}
		else {
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
		EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_non_cum_binary(struct sljit_compiler *compiler,
	sljit_u32 op_types,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;
	sljit_u8 op_eax_imm = U8(op_types >> 24);
	sljit_u8 op_rm = U8((op_types >> 16) & 0xff);
	sljit_u8 op_mr = U8((op_types >> 8) & 0xff);
	sljit_u8 op_imm = U8(op_types & 0xff);

	if (dst == src1 && dstw == src1w) {
		if (src2 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if ((dst == SLJIT_R0) && (src2w > 127 || src2w < -128) && (compiler->mode32 || IS_HALFWORD(src2w))) {
#else
			if ((dst == SLJIT_R0) && (src2w > 127 || src2w < -128)) {
#endif
				BINARY_EAX_IMM(op_eax_imm, src2w);
			}
			else {
				BINARY_IMM(op_imm, op_mr, src2w, dst, dstw);
			}
		}
		else if (FAST_IS_REG(dst)) {
			inst = emit_x86_instruction(compiler, 1, dst, dstw, src2, src2w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
		else if (FAST_IS_REG(src2)) {
			inst = emit_x86_instruction(compiler, 1, src2, src2w, dst, dstw);
			FAIL_IF(!inst);
			*inst = op_mr;
		}
		else {
			EMIT_MOV(compiler, TMP_REG1, 0, src2, src2w);
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, dst, dstw);
			FAIL_IF(!inst);
			*inst = op_mr;
		}
		return SLJIT_SUCCESS;
	}

	/* General version. */
	if (FAST_IS_REG(dst) && dst != src2) {
		EMIT_MOV(compiler, dst, 0, src1, src1w);
		if (src2 == SLJIT_IMM) {
			BINARY_IMM(op_imm, op_mr, src2w, dst, 0);
		}
		else {
			inst = emit_x86_instruction(compiler, 1, dst, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
	}
	else {
		/* This version requires less memory writing. */
		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		if (src2 == SLJIT_IMM) {
			BINARY_IMM(op_imm, op_mr, src2w, TMP_REG1, 0);
		}
		else {
			inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = op_rm;
		}
		EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_mul(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	/* Register destination. */
	if (dst_r == src1 && src2 != SLJIT_IMM) {
		FAIL_IF(emit_groupf(compiler, IMUL_r_rm, dst_r, src2, src2w));
	} else if (dst_r == src2 && src1 != SLJIT_IMM) {
		FAIL_IF(emit_groupf(compiler, IMUL_r_rm, dst_r, src1, src1w));
	} else if (src1 == SLJIT_IMM) {
		if (src2 == SLJIT_IMM) {
			EMIT_MOV(compiler, dst_r, 0, SLJIT_IMM, src2w);
			src2 = dst_r;
			src2w = 0;
		}

		if (src1w <= 127 && src1w >= -128) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i8;

			FAIL_IF(emit_byte(compiler, U8(src1w)));
		}
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		else {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i32;
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 4);
			FAIL_IF(!inst);
			INC_SIZE(4);
			sljit_unaligned_store_sw(inst, src1w);
		}
#else
		else if (IS_HALFWORD(src1w)) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i32;
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 4);
			FAIL_IF(!inst);
			INC_SIZE(4);
			sljit_unaligned_store_s32(inst, (sljit_s32)src1w);
		}
		else {
			if (dst_r != src2)
				EMIT_MOV(compiler, dst_r, 0, src2, src2w);
			FAIL_IF(emit_load_imm64(compiler, TMP_REG2, src1w));
			FAIL_IF(emit_groupf(compiler, IMUL_r_rm, dst_r, TMP_REG2, 0));
		}
#endif
	}
	else if (src2 == SLJIT_IMM) {
		/* Note: src1 is NOT immediate. */

		if (src2w <= 127 && src2w >= -128) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src1, src1w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i8;

			FAIL_IF(emit_byte(compiler, U8(src2w)));
		}
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		else {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src1, src1w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i32;

			inst = (sljit_u8*)ensure_buf(compiler, 1 + 4);
			FAIL_IF(!inst);
			INC_SIZE(4);
			sljit_unaligned_store_sw(inst, src2w);
		}
#else
		else if (IS_HALFWORD(src2w)) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src1, src1w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i32;

			inst = (sljit_u8*)ensure_buf(compiler, 1 + 4);
			FAIL_IF(!inst);
			INC_SIZE(4);
			sljit_unaligned_store_s32(inst, (sljit_s32)src2w);
		} else {
			if (dst_r != src1)
				EMIT_MOV(compiler, dst_r, 0, src1, src1w);
			FAIL_IF(emit_load_imm64(compiler, TMP_REG2, src2w));
			FAIL_IF(emit_groupf(compiler, IMUL_r_rm, dst_r, TMP_REG2, 0));
		}
#endif
	} else {
		/* Neither argument is immediate. */
		if (ADDRESSING_DEPENDS_ON(src2, dst_r))
			dst_r = TMP_REG1;
		EMIT_MOV(compiler, dst_r, 0, src1, src1w);
		FAIL_IF(emit_groupf(compiler, IMUL_r_rm, dst_r, src2, src2w));
	}

	if (dst & SLJIT_MEM)
		EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_lea_binary(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;
	sljit_s32 dst_r, done = 0;

	/* These cases better be left to handled by normal way. */
	if (dst == src1 && dstw == src1w)
		return SLJIT_ERR_UNSUPPORTED;
	if (dst == src2 && dstw == src2w)
		return SLJIT_ERR_UNSUPPORTED;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (FAST_IS_REG(src1)) {
		if (FAST_IS_REG(src2)) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM2(src1, src2), 0);
			FAIL_IF(!inst);
			*inst = LEA_r_m;
			done = 1;
		}
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (src2 == SLJIT_IMM && (compiler->mode32 || IS_HALFWORD(src2w))) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src1), (sljit_s32)src2w);
#else
		if (src2 == SLJIT_IMM) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src1), src2w);
#endif
			FAIL_IF(!inst);
			*inst = LEA_r_m;
			done = 1;
		}
	}
	else if (FAST_IS_REG(src2)) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (src1 == SLJIT_IMM && (compiler->mode32 || IS_HALFWORD(src1w))) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src2), (sljit_s32)src1w);
#else
		if (src1 == SLJIT_IMM) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src2), src1w);
#endif
			FAIL_IF(!inst);
			*inst = LEA_r_m;
			done = 1;
		}
	}

	if (done) {
		if (dst_r == TMP_REG1)
			return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
		return SLJIT_SUCCESS;
	}
	return SLJIT_ERR_UNSUPPORTED;
}

static sljit_s32 emit_cmp_binary(struct sljit_compiler *compiler,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (src1 == SLJIT_R0 && src2 == SLJIT_IMM && (src2w > 127 || src2w < -128) && (compiler->mode32 || IS_HALFWORD(src2w))) {
#else
	if (src1 == SLJIT_R0 && src2 == SLJIT_IMM && (src2w > 127 || src2w < -128)) {
#endif
		BINARY_EAX_IMM(CMP_EAX_i32, src2w);
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(src1)) {
		if (src2 == SLJIT_IMM) {
			BINARY_IMM(CMP, CMP_rm_r, src2w, src1, 0);
		}
		else {
			inst = emit_x86_instruction(compiler, 1, src1, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = CMP_r_rm;
		}
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(src2) && src1 != SLJIT_IMM) {
		inst = emit_x86_instruction(compiler, 1, src2, 0, src1, src1w);
		FAIL_IF(!inst);
		*inst = CMP_rm_r;
		return SLJIT_SUCCESS;
	}

	if (src2 == SLJIT_IMM) {
		if (src1 == SLJIT_IMM) {
			EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
			src1 = TMP_REG1;
			src1w = 0;
		}
		BINARY_IMM(CMP, CMP_rm_r, src2w, src1, src1w);
	}
	else {
		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src2, src2w);
		FAIL_IF(!inst);
		*inst = CMP_r_rm;
	}
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_test_binary(struct sljit_compiler *compiler,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (src1 == SLJIT_R0 && src2 == SLJIT_IMM && (src2w > 127 || src2w < -128) && (compiler->mode32 || IS_HALFWORD(src2w))) {
#else
	if (src1 == SLJIT_R0 && src2 == SLJIT_IMM && (src2w > 127 || src2w < -128)) {
#endif
		BINARY_EAX_IMM(TEST_EAX_i32, src2w);
		return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (src2 == SLJIT_R0 && src1 == SLJIT_IMM && (src1w > 127 || src1w < -128) && (compiler->mode32 || IS_HALFWORD(src1w))) {
#else
	if (src2 == SLJIT_R0 && src1 == SLJIT_IMM && (src1w > 127 || src1w < -128)) {
#endif
		BINARY_EAX_IMM(TEST_EAX_i32, src1w);
		return SLJIT_SUCCESS;
	}

	if (src1 != SLJIT_IMM) {
		if (src2 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if (IS_HALFWORD(src2w) || compiler->mode32) {
				inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, src2w, src1, src1w);
				FAIL_IF(!inst);
				*inst = GROUP_F7;
			} else {
				FAIL_IF(emit_load_imm64(compiler, FAST_IS_REG(src1) ? TMP_REG2 : TMP_REG1, src2w));
				inst = emit_x86_instruction(compiler, 1, FAST_IS_REG(src1) ? TMP_REG2 : TMP_REG1, 0, src1, src1w);
				FAIL_IF(!inst);
				*inst = TEST_rm_r;
			}
#else
			inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, src2w, src1, src1w);
			FAIL_IF(!inst);
			*inst = GROUP_F7;
#endif
			return SLJIT_SUCCESS;
		}
		else if (FAST_IS_REG(src1)) {
			inst = emit_x86_instruction(compiler, 1, src1, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = TEST_rm_r;
			return SLJIT_SUCCESS;
		}
	}

	if (src2 != SLJIT_IMM) {
		if (src1 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if (IS_HALFWORD(src1w) || compiler->mode32) {
				inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, src1w, src2, src2w);
				FAIL_IF(!inst);
				*inst = GROUP_F7;
			}
			else {
				FAIL_IF(emit_load_imm64(compiler, TMP_REG1, src1w));
				inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src2, src2w);
				FAIL_IF(!inst);
				*inst = TEST_rm_r;
			}
#else
			inst = emit_x86_instruction(compiler, 1, src1, src1w, src2, src2w);
			FAIL_IF(!inst);
			*inst = GROUP_F7;
#endif
			return SLJIT_SUCCESS;
		}
		else if (FAST_IS_REG(src2)) {
			inst = emit_x86_instruction(compiler, 1, src2, 0, src1, src1w);
			FAIL_IF(!inst);
			*inst = TEST_rm_r;
			return SLJIT_SUCCESS;
		}
	}

	EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
	if (src2 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (IS_HALFWORD(src2w) || compiler->mode32) {
			inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, src2w, TMP_REG1, 0);
			FAIL_IF(!inst);
			*inst = GROUP_F7;
		}
		else {
			FAIL_IF(emit_load_imm64(compiler, TMP_REG2, src2w));
			inst = emit_x86_instruction(compiler, 1, TMP_REG2, 0, TMP_REG1, 0);
			FAIL_IF(!inst);
			*inst = TEST_rm_r;
		}
#else
		inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, src2w, TMP_REG1, 0);
		FAIL_IF(!inst);
		*inst = GROUP_F7;
#endif
	}
	else {
		inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src2, src2w);
		FAIL_IF(!inst);
		*inst = TEST_rm_r;
	}
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_shift(struct sljit_compiler *compiler,
	sljit_u8 mode,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 mode32;
#endif
	sljit_u8* inst;

	if (src2 == SLJIT_IMM || src2 == SLJIT_PREF_SHIFT_REG) {
		if (dst == src1 && dstw == src1w) {
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, src2, src2w, dst, dstw);
			FAIL_IF(!inst);
			inst[1] |= mode;
			return SLJIT_SUCCESS;
		}
		if (dst == SLJIT_PREF_SHIFT_REG && src2 == SLJIT_PREF_SHIFT_REG) {
			EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
			FAIL_IF(!inst);
			inst[1] |= mode;
			EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
			return SLJIT_SUCCESS;
		}
		if (FAST_IS_REG(dst)) {
			EMIT_MOV(compiler, dst, 0, src1, src1w);
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, src2, src2w, dst, 0);
			FAIL_IF(!inst);
			inst[1] |= mode;
			return SLJIT_SUCCESS;
		}

		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, src2, src2w, TMP_REG1, 0);
		FAIL_IF(!inst);
		inst[1] |= mode;
		EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
		return SLJIT_SUCCESS;
	}

	if (dst == SLJIT_PREF_SHIFT_REG) {
		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src2, src2w);
		inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
		FAIL_IF(!inst);
		inst[1] |= mode;
		return emit_mov(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
	}

	if (FAST_IS_REG(dst) && dst != src2 && dst != TMP_REG1 && !ADDRESSING_DEPENDS_ON(src2, dst)) {
		if (src1 != dst)
			EMIT_MOV(compiler, dst, 0, src1, src1w);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		mode32 = compiler->mode32;
		compiler->mode32 = 0;
#endif
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_PREF_SHIFT_REG, 0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = mode32;
#endif
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src2, src2w);
		inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_PREF_SHIFT_REG, 0, dst, 0);
		FAIL_IF(!inst);
		inst[1] |= mode;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;
#endif
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = mode32;
#endif
		return SLJIT_SUCCESS;
	}

	/* This case is complex since ecx itself may be used for
	   addressing, and this case must be supported as well. */
	EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, SLJIT_PREF_SHIFT_REG, 0);
#else /* !SLJIT_CONFIG_X86_32 */
	mode32 = compiler->mode32;
	compiler->mode32 = 0;
	EMIT_MOV(compiler, TMP_REG2, 0, SLJIT_PREF_SHIFT_REG, 0);
	compiler->mode32 = mode32;
#endif /* SLJIT_CONFIG_X86_32 */

	EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src2, src2w);
	inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
	FAIL_IF(!inst);
	inst[1] |= mode;

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, SLJIT_MEM1(SLJIT_SP), 0);
#else
	compiler->mode32 = 0;
	EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG2, 0);
	compiler->mode32 = mode32;
#endif /* SLJIT_CONFIG_X86_32 */

	if (dst != TMP_REG1)
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_shift_with_flags(struct sljit_compiler *compiler,
	sljit_u8 mode, sljit_s32 set_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* The CPU does not set flags if the shift count is 0. */
	if (src2 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		src2w &= compiler->mode32 ? 0x1f : 0x3f;
#else /* !SLJIT_CONFIG_X86_64 */
		src2w &= 0x1f;
#endif /* SLJIT_CONFIG_X86_64 */
		if (src2w != 0)
			return emit_shift(compiler, mode, dst, dstw, src1, src1w, src2, src2w);

		if (!set_flags)
			return emit_mov(compiler, dst, dstw, src1, src1w);
		/* OR dst, src, 0 */
		return emit_cum_binary(compiler, BINARY_OPCODE(OR),
			dst, dstw, src1, src1w, SLJIT_IMM, 0);
	}

	if (!set_flags)
		return emit_shift(compiler, mode, dst, dstw, src1, src1w, src2, src2w);

	if (!FAST_IS_REG(dst))
		FAIL_IF(emit_cmp_binary(compiler, src1, src1w, SLJIT_IMM, 0));

	FAIL_IF(emit_shift(compiler, mode, dst, dstw, src1, src1w, src2, src2w));

	if (FAST_IS_REG(dst))
		return emit_cmp_binary(compiler, dst, dstw, SLJIT_IMM, 0);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);
	CHECK_EXTRA_REGS(src1, src1w, (void)0);
	CHECK_EXTRA_REGS(src2, src2w, (void)0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op & SLJIT_32;
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
		if (!HAS_FLAGS(op)) {
			if (emit_lea_binary(compiler, dst, dstw, src1, src1w, src2, src2w) != SLJIT_ERR_UNSUPPORTED)
				return compiler->error;
		}
		return emit_cum_binary(compiler, BINARY_OPCODE(ADD),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_ADDC:
		return emit_cum_binary(compiler, BINARY_OPCODE(ADC),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_SUB:
		if (src1 == SLJIT_IMM && src1w == 0)
			return emit_unary(compiler, NEG_rm, dst, dstw, src2, src2w);

		if (!HAS_FLAGS(op)) {
			if (src2 == SLJIT_IMM && emit_lea_binary(compiler, dst, dstw, src1, src1w, SLJIT_IMM, -src2w) != SLJIT_ERR_UNSUPPORTED)
				return compiler->error;
			if (FAST_IS_REG(dst) && src2 == dst) {
				FAIL_IF(emit_non_cum_binary(compiler, BINARY_OPCODE(SUB), dst, 0, dst, 0, src1, src1w));
				return emit_unary(compiler, NEG_rm, dst, 0, dst, 0);
			}
		}

		return emit_non_cum_binary(compiler, BINARY_OPCODE(SUB),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_SUBC:
		return emit_non_cum_binary(compiler, BINARY_OPCODE(SBB),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_MUL:
		return emit_mul(compiler, dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_AND:
		return emit_cum_binary(compiler, BINARY_OPCODE(AND),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_OR:
		return emit_cum_binary(compiler, BINARY_OPCODE(OR),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_XOR:
		if (!HAS_FLAGS(op)) {
			if (src2 == SLJIT_IMM && src2w == -1)
				return emit_unary(compiler, NOT_rm, dst, dstw, src1, src1w);
			if (src1 == SLJIT_IMM && src1w == -1)
				return emit_unary(compiler, NOT_rm, dst, dstw, src2, src2w);
		}

		return emit_cum_binary(compiler, BINARY_OPCODE(XOR),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_SHL:
	case SLJIT_MSHL:
		return emit_shift_with_flags(compiler, SHL, HAS_FLAGS(op),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		return emit_shift_with_flags(compiler, SHR, HAS_FLAGS(op),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_ASHR:
	case SLJIT_MASHR:
		return emit_shift_with_flags(compiler, SAR, HAS_FLAGS(op),
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_ROTL:
		return emit_shift_with_flags(compiler, ROL, 0,
			dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_ROTR:
		return emit_shift_with_flags(compiler, ROR, 0,
			dst, dstw, src1, src1w, src2, src2w);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 opcode = GET_OPCODE(op);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 1, 0, 0, src1, src1w, src2, src2w));

	if (opcode != SLJIT_SUB && opcode != SLJIT_AND) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, op, TMP_REG1, 0, src1, src1w, src2, src2w);
	}

	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	CHECK_EXTRA_REGS(src1, src1w, (void)0);
	CHECK_EXTRA_REGS(src2, src2w, (void)0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op & SLJIT_32;
#endif

	if (opcode == SLJIT_SUB)
		return emit_cmp_binary(compiler, src1, src1w, src2, src2w);

	return emit_test_binary(compiler, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_u8* inst;
	sljit_sw dstw = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2r(compiler, op, dst_reg, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	CHECK_EXTRA_REGS(dst_reg, dstw, (void)0);
	CHECK_EXTRA_REGS(src1, src1w, (void)0);
	CHECK_EXTRA_REGS(src2, src2w, (void)0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op & SLJIT_32;
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_MULADD:
		FAIL_IF(emit_mul(compiler, TMP_REG1, 0, src1, src1w, src2, src2w));
		inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, dst_reg, dstw);
		FAIL_IF(!inst);
		*inst = ADD_rm_r;
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_s32 is_rotate, is_left, move_src1;
	sljit_u8* inst;
	sljit_sw src1w = 0;
	sljit_sw dstw = 0;
	/* The whole register must be saved even for 32 bit operations. */
	sljit_u8 restore_ecx = 0;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_sw src2w = 0;
	sljit_s32 restore_sp4 = 0;
#endif /* SLJIT_CONFIG_X86_32 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, dst_reg, src1_reg, src2_reg, src3, src3w));
	ADJUST_LOCAL_OFFSET(src3, src3w);

	CHECK_EXTRA_REGS(dst_reg, dstw, (void)0);
	CHECK_EXTRA_REGS(src3, src3w, (void)0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op & SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */

	if (src3 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		src3w &= 0x1f;
#else /* !SLJIT_CONFIG_X86_32 */
		src3w &= (op & SLJIT_32) ? 0x1f : 0x3f;
#endif /* SLJIT_CONFIG_X86_32 */

		if (src3w == 0)
			return SLJIT_SUCCESS;
	}

	is_left = (GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_MSHL);

	is_rotate = (src1_reg == src2_reg);
	CHECK_EXTRA_REGS(src1_reg, src1w, (void)0);
	CHECK_EXTRA_REGS(src2_reg, src2w, (void)0);

	if (is_rotate)
		return emit_shift(compiler, is_left ? ROL : ROR, dst_reg, dstw, src1_reg, src1w, src3, src3w);

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	if (src2_reg & SLJIT_MEM) {
		EMIT_MOV(compiler, TMP_REG1, 0, src2_reg, src2w);
		src2_reg = TMP_REG1;
	}
#endif /* SLJIT_CONFIG_X86_32 */

	if (dst_reg == SLJIT_PREF_SHIFT_REG && src3 != SLJIT_IMM && (src3 != SLJIT_PREF_SHIFT_REG || src1_reg != SLJIT_PREF_SHIFT_REG)) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		EMIT_MOV(compiler, TMP_REG1, 0, src1_reg, src1w);
		src1_reg = TMP_REG1;
		src1w = 0;
#else /* !SLJIT_CONFIG_X86_64 */
		if (src2_reg != TMP_REG1) {
			EMIT_MOV(compiler, TMP_REG1, 0, src1_reg, src1w);
			src1_reg = TMP_REG1;
			src1w = 0;
		} else if ((src1_reg & SLJIT_MEM) || src1_reg == SLJIT_PREF_SHIFT_REG) {
			restore_sp4 = (src3 == SLJIT_R0) ? SLJIT_R1 : SLJIT_R0;
			EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), sizeof(sljit_s32), restore_sp4, 0);
			EMIT_MOV(compiler, restore_sp4, 0, src1_reg, src1w);
			src1_reg = restore_sp4;
			src1w = 0;
		} else {
			EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), sizeof(sljit_s32), src1_reg, 0);
			restore_sp4 = src1_reg;
		}
#endif /* SLJIT_CONFIG_X86_64 */

		if (src3 != SLJIT_PREF_SHIFT_REG)
			EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src3, src3w);
	} else {
		if (src2_reg == SLJIT_PREF_SHIFT_REG && src3 != SLJIT_IMM && src3 != SLJIT_PREF_SHIFT_REG) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			compiler->mode32 = 0;
#endif /* SLJIT_CONFIG_X86_64 */
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_PREF_SHIFT_REG, 0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			compiler->mode32 = op & SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */
			src2_reg = TMP_REG1;
			restore_ecx = 1;
		}

		move_src1 = 0;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (dst_reg != src1_reg) {
			if (dst_reg != src3) {
				EMIT_MOV(compiler, dst_reg, 0, src1_reg, src1w);
				src1_reg = dst_reg;
				src1w = 0;
			} else
				move_src1 = 1;
		}
#else /* !SLJIT_CONFIG_X86_64 */
		if (dst_reg & SLJIT_MEM) {
			if (src2_reg != TMP_REG1) {
				EMIT_MOV(compiler, TMP_REG1, 0, src1_reg, src1w);
				src1_reg = TMP_REG1;
				src1w = 0;
			} else if ((src1_reg & SLJIT_MEM) || src1_reg == SLJIT_PREF_SHIFT_REG) {
				restore_sp4 = (src3 == SLJIT_R0) ? SLJIT_R1 : SLJIT_R0;
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), sizeof(sljit_s32), restore_sp4, 0);
				EMIT_MOV(compiler, restore_sp4, 0, src1_reg, src1w);
				src1_reg = restore_sp4;
				src1w = 0;
			} else {
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), sizeof(sljit_s32), src1_reg, 0);
				restore_sp4 = src1_reg;
			}
		} else if (dst_reg != src1_reg) {
			if (dst_reg != src3) {
				EMIT_MOV(compiler, dst_reg, 0, src1_reg, src1w);
				src1_reg = dst_reg;
				src1w = 0;
			} else
				move_src1 = 1;
		}
#endif /* SLJIT_CONFIG_X86_64 */

		if (src3 != SLJIT_IMM && src3 != SLJIT_PREF_SHIFT_REG) {
			if (!restore_ecx) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
				compiler->mode32 = 0;
				EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_PREF_SHIFT_REG, 0);
				compiler->mode32 = op & SLJIT_32;
				restore_ecx = 1;
#else /* !SLJIT_CONFIG_X86_64 */
				if (src1_reg != TMP_REG1 && src2_reg != TMP_REG1) {
					EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_PREF_SHIFT_REG, 0);
					restore_ecx = 1;
				} else {
					EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, SLJIT_PREF_SHIFT_REG, 0);
					restore_ecx = 2;
				}
#endif /* SLJIT_CONFIG_X86_64 */
			}
			EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src3, src3w);
		}

		if (move_src1) {
			EMIT_MOV(compiler, dst_reg, 0, src1_reg, src1w);
			src1_reg = dst_reg;
			src1w = 0;
		}
	}

	inst = emit_x86_instruction(compiler, 2, src2_reg, 0, src1_reg, src1w);
	FAIL_IF(!inst);
	inst[0] = GROUP_0F;

	if (src3 == SLJIT_IMM) {
		inst[1] = U8((is_left ? SHLD : SHRD) - 1);

		/* Immediate argument is added separately. */
		FAIL_IF(emit_byte(compiler, U8(src3w)));
	} else
		inst[1] = U8(is_left ? SHLD : SHRD);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (restore_ecx) {
		compiler->mode32 = 0;
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
	}

	if (src1_reg != dst_reg) {
		compiler->mode32 = op & SLJIT_32;
		return emit_mov(compiler, dst_reg, dstw, src1_reg, 0);
	}
#else /* !SLJIT_CONFIG_X86_64 */
	if (restore_ecx)
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, restore_ecx == 1 ? TMP_REG1 : SLJIT_MEM1(SLJIT_SP), 0);

	if (src1_reg != dst_reg)
		EMIT_MOV(compiler, dst_reg, dstw, src1_reg, 0);

	if (restore_sp4)
		return emit_mov(compiler, restore_sp4, 0, SLJIT_MEM1(SLJIT_SP), sizeof(sljit_s32));
#endif /* SLJIT_CONFIG_X86_32 */

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	CHECK_EXTRA_REGS(src, srcw, (void)0);

	switch (op) {
	case SLJIT_FAST_RETURN:
		return emit_fast_return(compiler, src, srcw);
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		/* Don't adjust shadow stack if it isn't enabled.  */
		if (!cpu_has_shadow_stack ())
			return SLJIT_SUCCESS;
		return adjust_shadow_stack(compiler, src, srcw);
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		return emit_prefetch(compiler, op, src, srcw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_dst(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

	switch (op) {
	case SLJIT_FAST_ENTER:
		return emit_fast_enter(compiler, dst, dstw);
	case SLJIT_GET_RETURN_ADDRESS:
		return sljit_emit_get_return_address(compiler, dst, dstw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(type, reg));

	if (type == SLJIT_GP_REGISTER) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (reg >= SLJIT_R3 && reg <= SLJIT_R8)
			return -1;
#endif /* SLJIT_CONFIG_X86_32 */
		return reg_map[reg];
	}

	if (type != SLJIT_FLOAT_REGISTER && type != SLJIT_SIMD_REG_128 && type != SLJIT_SIMD_REG_256 && type != SLJIT_SIMD_REG_512)
		return -1;

	return freg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);
	SLJIT_MEMCPY(inst, instruction, size);
	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

/* Alignment(3) + 4 * 16 bytes. */
static sljit_u32 sse2_data[3 + (4 * 4)];
static sljit_u32 *sse2_buffer;

static void init_compiler(void)
{
	get_cpu_features();

	/* Align to 16 bytes. */
	sse2_buffer = (sljit_u32*)(((sljit_uw)sse2_data + 15) & ~(sljit_uw)0xf);

	/* Single precision constants (each constant is 16 byte long). */
	sse2_buffer[0] = 0x80000000;
	sse2_buffer[4] = 0x7fffffff;
	/* Double precision constants (each constant is 16 byte long). */
	sse2_buffer[8] = 0;
	sse2_buffer[9] = 0x80000000;
	sse2_buffer[12] = 0xffffffff;
	sse2_buffer[13] = 0x7fffffff;
}

static sljit_s32 emit_groupf(struct sljit_compiler *compiler,
	sljit_uw op,
	sljit_s32 dst, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst = emit_x86_instruction(compiler, 2 | (op & ~(sljit_uw)0xff), dst, 0, src, srcw);
	FAIL_IF(!inst);
	inst[0] = GROUP_0F;
	inst[1] = op & 0xff;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_groupf_ext(struct sljit_compiler *compiler,
	sljit_uw op,
	sljit_s32 dst, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst;

	SLJIT_ASSERT((op & EX86_SSE2) && ((op & VEX_OP_0F38) || (op & VEX_OP_0F3A)));

	inst = emit_x86_instruction(compiler, 3 | (op & ~((sljit_uw)0xff | VEX_OP_0F38 | VEX_OP_0F3A)), dst, 0, src, srcw);
	FAIL_IF(!inst);
	inst[0] = GROUP_0F;
	inst[1] = U8((op & VEX_OP_0F38) ? 0x38 : 0x3A);
	inst[2] = op & 0xff;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_sse2_load(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_s32 src, sljit_sw srcw)
{
	return emit_groupf(compiler, MOVSD_x_xm | (single ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2, dst, src, srcw);
}

static SLJIT_INLINE sljit_s32 emit_sse2_store(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_sw dstw, sljit_s32 src)
{
	return emit_groupf(compiler, MOVSD_xm_x | (single ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2, src, dst, dstw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_EXTRA_REGS(dst, dstw, (void)0);
	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (GET_OPCODE(op) == SLJIT_CONV_SW_FROM_F64)
		compiler->mode32 = 0;
#endif

	FAIL_IF(emit_groupf(compiler, CVTTSD2SI_r_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2_OP2, dst_r, src, srcw));

	if (dst & SLJIT_MEM)
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG;

	CHECK_EXTRA_REGS(src, srcw, (void)0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_SW)
		compiler->mode32 = 0;
#endif

	if (src == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;
#endif
		EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
		src = TMP_REG1;
		srcw = 0;
	}

	FAIL_IF(emit_groupf(compiler, CVTSI2SD_x_rm | EX86_SELECT_F2_F3(op) | EX86_SSE2_OP1, dst_r, src, srcw));

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif
	if (dst_r == TMP_FREG)
		return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	switch (GET_FLAG_TYPE(op)) {
	case SLJIT_ORDERED_EQUAL:
		/* Also: SLJIT_UNORDERED_OR_NOT_EQUAL */
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
		FAIL_IF(emit_groupf(compiler, CMPS_x_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2, TMP_FREG, src2, src2w));

		/* EQ */
		FAIL_IF(emit_byte(compiler, 0));

		src1 = TMP_FREG;
		src2 = TMP_FREG;
		src2w = 0;
		break;

	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_GREATER:
		/* Also: SLJIT_UNORDERED_OR_GREATER_EQUAL, SLJIT_ORDERED_LESS_EQUAL  */
		if (!FAST_IS_REG(src2)) {
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src2, src2w));
			src2 = TMP_FREG;
		}

		return emit_groupf(compiler, UCOMISD_x_xm | EX86_SELECT_66(op) | EX86_SSE2, src2, src1, src1w);
	}

	if (!FAST_IS_REG(src1)) {
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
		src1 = TMP_FREG;
	}

	return emit_groupf(compiler, UCOMISD_x_xm | EX86_SELECT_66(op) | EX86_SSE2, src1, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;
	sljit_u8 *inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif

	CHECK_ERROR();
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_MOV_F64) {
		if (FAST_IS_REG(dst))
			return emit_sse2_load(compiler, op & SLJIT_32, dst, src, srcw);
		if (FAST_IS_REG(src))
			return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, src);
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src, srcw));
		return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
	}

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32) {
		dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG;
		if (FAST_IS_REG(src)) {
			/* We overwrite the high bits of source. From SLJIT point of view,
			   this is not an issue.
			   Note: In SSE3, we could also use MOVDDUP and MOVSLDUP. */
			FAIL_IF(emit_groupf(compiler, UNPCKLPD_x_xm | ((op & SLJIT_32) ? EX86_PREF_66 : 0) | EX86_SSE2, src, src, 0));
		} else {
			FAIL_IF(emit_sse2_load(compiler, !(op & SLJIT_32), TMP_FREG, src, srcw));
			src = TMP_FREG;
		}

		FAIL_IF(emit_groupf(compiler, CVTPD2PS_x_xm | ((op & SLJIT_32) ? EX86_PREF_66 : 0) | EX86_SSE2, dst_r, src, 0));
		if (dst_r == TMP_FREG)
			return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(dst)) {
		dst_r = (dst == src) ? TMP_FREG : dst;

		if (src & SLJIT_MEM)
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src, srcw));

		FAIL_IF(emit_groupf(compiler, PCMPEQD_x_xm | EX86_PREF_66 | EX86_SSE2, dst_r, dst_r, 0));

		inst = emit_x86_instruction(compiler, 2 | EX86_PREF_66 | EX86_SSE2_OP2, 0, 0, dst_r, 0);
		inst[0] = GROUP_0F;
		/* Same as PSRLD_x / PSRLQ_x */
		inst[1] = (op & SLJIT_32) ? PSLLD_x_i8 : PSLLQ_x_i8;

		if (GET_OPCODE(op) == SLJIT_ABS_F64) {
			inst[2] |= 2 << 3;
			FAIL_IF(emit_byte(compiler, 1));
		} else {
			inst[2] |= 6 << 3;
			FAIL_IF(emit_byte(compiler, ((op & SLJIT_32) ? 31 : 63)));
		}

		if (dst_r != TMP_FREG)
			dst_r = (src & SLJIT_MEM) ? TMP_FREG : src;
		return emit_groupf(compiler, (GET_OPCODE(op) == SLJIT_NEG_F64 ? XORPD_x_xm : ANDPD_x_xm) | EX86_SSE2, dst, dst_r, 0);
	}

	FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src, srcw));

	switch (GET_OPCODE(op)) {
	case SLJIT_NEG_F64:
		FAIL_IF(emit_groupf(compiler, XORPD_x_xm | EX86_SELECT_66(op) | EX86_SSE2, TMP_FREG, SLJIT_MEM0(), (sljit_sw)((op & SLJIT_32) ? sse2_buffer : sse2_buffer + 8)));
		break;

	case SLJIT_ABS_F64:
		FAIL_IF(emit_groupf(compiler, ANDPD_x_xm | EX86_SELECT_66(op) | EX86_SSE2, TMP_FREG, SLJIT_MEM0(), (sljit_sw)((op & SLJIT_32) ? sse2_buffer + 4 : sse2_buffer + 12)));
		break;
	}

	return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif

	if (FAST_IS_REG(dst)) {
		dst_r = dst;
		if (dst == src1)
			; /* Do nothing here. */
		else if (dst == src2 && (GET_OPCODE(op) == SLJIT_ADD_F64 || GET_OPCODE(op) == SLJIT_MUL_F64)) {
			/* Swap arguments. */
			src2 = src1;
			src2w = src1w;
		} else if (dst != src2)
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, dst_r, src1, src1w));
		else {
			dst_r = TMP_FREG;
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
		}
	} else {
		dst_r = TMP_FREG;
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(emit_groupf(compiler, ADDSD_x_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2, dst_r, src2, src2w));
		break;

	case SLJIT_SUB_F64:
		FAIL_IF(emit_groupf(compiler, SUBSD_x_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2, dst_r, src2, src2w));
		break;

	case SLJIT_MUL_F64:
		FAIL_IF(emit_groupf(compiler, MULSD_x_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2, dst_r, src2, src2w));
		break;

	case SLJIT_DIV_F64:
		FAIL_IF(emit_groupf(compiler, DIVSD_x_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2, dst_r, src2, src2w));
		break;
	}

	if (dst_r != dst)
		return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_uw pref;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2r(compiler, op, dst_freg, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif

	if (dst_freg == src1) {
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src2, src2w));
		pref = EX86_SELECT_66(op) | EX86_SSE2;
		FAIL_IF(emit_groupf(compiler, XORPD_x_xm | pref, TMP_FREG, src1, src1w));
		FAIL_IF(emit_groupf(compiler, ANDPD_x_xm | pref, TMP_FREG, SLJIT_MEM0(), (sljit_sw)((op & SLJIT_32) ? sse2_buffer : sse2_buffer + 8)));
		return emit_groupf(compiler, XORPD_x_xm | pref, dst_freg, TMP_FREG, 0);
	}

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
		src1 = TMP_FREG;
		src1w = 0;
	}

	if (dst_freg != src2)
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, dst_freg, src2, src2w));

	pref = EX86_SELECT_66(op) | EX86_SSE2;
	FAIL_IF(emit_groupf(compiler, XORPD_x_xm | pref, dst_freg, src1, src1w));
	FAIL_IF(emit_groupf(compiler, ANDPD_x_xm | pref, dst_freg, SLJIT_MEM0(), (sljit_sw)((op & SLJIT_32) ? sse2_buffer : sse2_buffer + 8)));
	return emit_groupf(compiler, XORPD_x_xm | pref, dst_freg, src1, src1w);
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler)
{
	sljit_u8 *inst;
	struct sljit_label *label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_label(compiler));

	if (compiler->last_label && compiler->last_label->size == compiler->size)
		return compiler->last_label;

	label = (struct sljit_label*)ensure_abuf(compiler, sizeof(struct sljit_label));
	PTR_FAIL_IF(!label);
	set_label(label, compiler);

	inst = (sljit_u8*)ensure_buf(compiler, 1);
	PTR_FAIL_IF(!inst);
	inst[0] = SLJIT_INST_LABEL;

	return label;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	sljit_u8 *inst;
	struct sljit_jump *jump;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF_NULL(jump);
	set_jump(jump, compiler, (sljit_u32)((type & SLJIT_REWRITABLE_JUMP) | ((type & 0xff) << TYPE_SHIFT)));
	type &= 0xff;

	jump->addr = compiler->size;
	/* Worst case size. */
	compiler->size += (type >= SLJIT_JUMP) ? JUMP_MAX_SIZE : CJUMP_MAX_SIZE;
	inst = (sljit_u8*)ensure_buf(compiler, 1);
	PTR_FAIL_IF_NULL(inst);

	inst[0] = SLJIT_INST_JUMP;
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst;
	struct sljit_jump *jump;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	CHECK_EXTRA_REGS(src, srcw, (void)0);

	if (src == SLJIT_IMM) {
		jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
		FAIL_IF_NULL(jump);
		set_jump(jump, compiler, (sljit_u32)(JUMP_ADDR | (type << TYPE_SHIFT)));
		jump->u.target = (sljit_uw)srcw;

		jump->addr = compiler->size;
		/* Worst case size. */
		compiler->size += JUMP_MAX_SIZE;
		inst = (sljit_u8*)ensure_buf(compiler, 1);
		FAIL_IF_NULL(inst);

		inst[0] = SLJIT_INST_JUMP;
	} else {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		/* REX_W is not necessary (src is not immediate). */
		compiler->mode32 = 1;
#endif
		inst = emit_x86_instruction(compiler, 1, 0, 0, src, srcw);
		FAIL_IF(!inst);
		inst[0] = GROUP_FF;
		inst[1] = U8(inst[1] | ((type >= SLJIT_FAST_CALL) ? CALL_rm : JMP_rm));
	}
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_u8 *inst;
	sljit_u8 cond_set;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 reg;
	sljit_uw size;
#endif /* !SLJIT_CONFIG_X86_64 */
	/* ADJUST_LOCAL_OFFSET and CHECK_EXTRA_REGS might overwrite these values. */
	sljit_s32 dst_save = dst;
	sljit_sw dstw_save = dstw;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));

	ADJUST_LOCAL_OFFSET(dst, dstw);
	CHECK_EXTRA_REGS(dst, dstw, (void)0);

	/* setcc = jcc + 0x10. */
	cond_set = U8(get_jump_code((sljit_uw)type) + 0x10);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (GET_OPCODE(op) == SLJIT_OR && !GET_ALL_FLAGS(op) && FAST_IS_REG(dst)) {
		size = 3 + 2;
		if (reg_map[TMP_REG1] >= 4)
			size += 1 + 1;
		else if (reg_map[dst] >= 4)
			size++;

		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
		/* Set low register to conditional flag. */
		if (reg_map[TMP_REG1] >= 4)
			*inst++ = (reg_map[TMP_REG1] <= 7) ? REX : REX_B;

		inst[0] = GROUP_0F;
		inst[1] = cond_set;
		inst[2] = MOD_REG | reg_lmap[TMP_REG1];
		inst += 3;

		if (reg_map[TMP_REG1] >= 4 || reg_map[dst] >= 4)
			*inst++ = U8(REX | (reg_map[TMP_REG1] <= 7 ? 0 : REX_R) | (reg_map[dst] <= 7 ? 0 : REX_B));

		inst[0] = OR_rm8_r8;
		inst[1] = U8(MOD_REG | (reg_lmap[TMP_REG1] << 3) | reg_lmap[dst]);
		return SLJIT_SUCCESS;
	}

	reg = (GET_OPCODE(op) < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG1;

	size = 3 + (reg_map[reg] >= 4) + 4;
	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);
	/* Set low register to conditional flag. */

	if (reg_map[reg] >= 4)
		*inst++ = (reg_map[reg] <= 7) ? REX : REX_B;

	inst[0] = GROUP_0F;
	inst[1] = cond_set;
	inst[2] = MOD_REG | reg_lmap[reg];

	inst[3] = REX_W | (reg_map[reg] <= 7 ? 0 : (REX_B | REX_R));
	/* The movzx instruction does not affect flags. */
	inst[4] = GROUP_0F;
	inst[5] = MOVZX_r_rm8;
	inst[6] = U8(MOD_REG | (reg_lmap[reg] << 3) | reg_lmap[reg]);

	if (reg != TMP_REG1)
		return SLJIT_SUCCESS;

	if (GET_OPCODE(op) < SLJIT_ADD) {
		compiler->mode32 = GET_OPCODE(op) != SLJIT_MOV;
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, dst_save, dstw_save, dst_save, dstw_save, TMP_REG1, 0);

#else /* !SLJIT_CONFIG_X86_64 */
	SLJIT_ASSERT(reg_map[TMP_REG1] < 4);

	/* The SLJIT_CONFIG_X86_32 code path starts here. */
	if (GET_OPCODE(op) < SLJIT_ADD && FAST_IS_REG(dst) && reg_map[dst] <= 4) {
		/* Low byte is accessible. */
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 3 + 3);
		FAIL_IF(!inst);
		INC_SIZE(3 + 3);
		/* Set low byte to conditional flag. */
		inst[0] = GROUP_0F;
		inst[1] = cond_set;
		inst[2] = U8(MOD_REG | reg_map[dst]);

		inst[3] = GROUP_0F;
		inst[4] = MOVZX_r_rm8;
		inst[5] = U8(MOD_REG | (reg_map[dst] << 3) | reg_map[dst]);
		return SLJIT_SUCCESS;
	}

	if (GET_OPCODE(op) == SLJIT_OR && !GET_ALL_FLAGS(op) && FAST_IS_REG(dst) && reg_map[dst] <= 4) {
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 3 + 2);
		FAIL_IF(!inst);
		INC_SIZE(3 + 2);

		/* Set low byte to conditional flag. */
		inst[0] = GROUP_0F;
		inst[1] = cond_set;
		inst[2] = U8(MOD_REG | reg_map[TMP_REG1]);

		inst[3] = OR_rm8_r8;
		inst[4] = U8(MOD_REG | (reg_map[TMP_REG1] << 3) | reg_map[dst]);
		return SLJIT_SUCCESS;
	}

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 3 + 3);
	FAIL_IF(!inst);
	INC_SIZE(3 + 3);
	/* Set low byte to conditional flag. */
	inst[0] = GROUP_0F;
	inst[1] = cond_set;
	inst[2] = U8(MOD_REG | reg_map[TMP_REG1]);

	inst[3] = GROUP_0F;
	inst[4] = MOVZX_r_rm8;
	inst[5] = U8(MOD_REG | (reg_map[TMP_REG1] << 3) | reg_map[TMP_REG1]);

	if (GET_OPCODE(op) < SLJIT_ADD)
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, dst_save, dstw_save, dst_save, dstw_save, TMP_REG1, 0);
#endif /* SLJIT_CONFIG_X86_64 */
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
	sljit_u8* inst;
	sljit_uw size;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */

	if (dst_freg != src2_freg) {
		if (dst_freg == src1) {
			src1 = src2_freg;
			src1w = 0;
			type ^= 0x1;
		} else
			FAIL_IF(emit_sse2_load(compiler, type & SLJIT_32, dst_freg, src2_freg, 0));
	}

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(2);
	inst[0] = U8(get_jump_code((sljit_uw)(type & ~SLJIT_32) ^ 0x1) - 0x10);

	size = compiler->size;
	FAIL_IF(emit_sse2_load(compiler, type & SLJIT_32, dst_freg, src1, src1w));

	inst[1] = U8(compiler->size - size);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 alignment = SLJIT_SIMD_GET_ELEM2_SIZE(type);
	sljit_uw op;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_mov(compiler, type, vreg, srcdst, srcdstw));

	ADJUST_LOCAL_OFFSET(srcdst, srcdstw);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */

	switch (reg_size) {
	case 4:
		op = EX86_SSE2;
		break;
	case 5:
		if (!(cpu_feature_list & CPU_FEATURE_AVX2))
			return SLJIT_ERR_UNSUPPORTED;
		op = EX86_SSE2 | VEX_256;
		break;
	default:
		return SLJIT_ERR_UNSUPPORTED;
	}

	if (!(srcdst & SLJIT_MEM))
		alignment = reg_size;

	if (type & SLJIT_SIMD_FLOAT) {
		if (elem_size == 2 || elem_size == 3) {
			op |= alignment >= reg_size ? MOVAPS_x_xm : MOVUPS_x_xm;

			if (elem_size == 3)
				op |= EX86_PREF_66;

			if (type & SLJIT_SIMD_STORE)
				op += 1;
		} else
			return SLJIT_ERR_UNSUPPORTED;
	} else {
		op |= ((type & SLJIT_SIMD_STORE) ? MOVDQA_xm_x : MOVDQA_x_xm)
			| (alignment >= reg_size ? EX86_PREF_66 : EX86_PREF_F3);
	}

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if ((op & VEX_256) || ((cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX)))
		return emit_vex_instruction(compiler, op, vreg, 0, srcdst, srcdstw);

	return emit_groupf(compiler, op, vreg, srcdst, srcdstw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 use_vex = (cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX);
	sljit_u8 *inst;
	sljit_u8 opcode = 0;
	sljit_uw op;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_replicate(compiler, type, vreg, src, srcw));

	ADJUST_LOCAL_OFFSET(src, srcw);

	if (!(type & SLJIT_SIMD_FLOAT)) {
		CHECK_EXTRA_REGS(src, srcw, (void)0);
	}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	if ((type & SLJIT_SIMD_FLOAT) ? (elem_size < 2 || elem_size > 3) : (elem_size > 2))
		return SLJIT_ERR_UNSUPPORTED;
#else /* !SLJIT_CONFIG_X86_32 */
	compiler->mode32 = 1;

	if (elem_size > 3 || ((type & SLJIT_SIMD_FLOAT) && elem_size < 2))
		return SLJIT_ERR_UNSUPPORTED;
#endif /* SLJIT_CONFIG_X86_32 */

	if (reg_size != 4 && (reg_size != 5 || !(cpu_feature_list & CPU_FEATURE_AVX2)))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 5)
		use_vex = 1;

	if (use_vex && src != SLJIT_IMM) {
		op = 0;

		switch (elem_size) {
		case 0:
			if (cpu_feature_list & CPU_FEATURE_AVX2)
				op = VPBROADCASTB_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		case 1:
			if (cpu_feature_list & CPU_FEATURE_AVX2)
				op = VPBROADCASTW_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		case 2:
			if (type & SLJIT_SIMD_FLOAT) {
				if ((cpu_feature_list & CPU_FEATURE_AVX2) || ((cpu_feature_list & CPU_FEATURE_AVX) && (src & SLJIT_MEM)))
					op = VBROADCASTSS_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			} else if (cpu_feature_list & CPU_FEATURE_AVX2)
				op = VPBROADCASTD_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		default:
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if (!(type & SLJIT_SIMD_FLOAT)) {
				if (cpu_feature_list & CPU_FEATURE_AVX2)
					op = VPBROADCASTQ_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
				break;
			}
#endif /* SLJIT_CONFIG_X86_64 */

			if (reg_size == 5)
				op = VBROADCASTSD_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		}

		if (op != 0) {
			if (!(src & SLJIT_MEM) && !(type & SLJIT_SIMD_FLOAT)) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
				if (elem_size >= 3)
					compiler->mode32 = 0;
#endif /* SLJIT_CONFIG_X86_64 */
				FAIL_IF(emit_vex_instruction(compiler, MOVD_x_rm | VEX_AUTO_W | EX86_PREF_66 | EX86_SSE2_OP1, vreg, 0, src, srcw));
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
				compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */
				src = vreg;
				srcw = 0;
			}

			if (reg_size == 5)
				op |= VEX_256;

			return emit_vex_instruction(compiler, op, vreg, 0, src, srcw);
		}
	}

	if (type & SLJIT_SIMD_FLOAT) {
		if (src == SLJIT_IMM) {
			if (use_vex)
				return emit_vex_instruction(compiler, XORPD_x_xm | (reg_size == 5 ? VEX_256 : 0) | (elem_size == 3 ? EX86_PREF_66 : 0) | EX86_SSE2 | VEX_SSE2_OPV, vreg, vreg, vreg, 0);

			return emit_groupf(compiler, XORPD_x_xm | (elem_size == 3 ? EX86_PREF_66 : 0) | EX86_SSE2, vreg, vreg, 0);
		}

		SLJIT_ASSERT(reg_size == 4);

		if (use_vex) {
			if (elem_size == 3)
				return emit_vex_instruction(compiler, MOVDDUP_x_xm | EX86_PREF_F2 | EX86_SSE2, vreg, 0, src, srcw);

			SLJIT_ASSERT(!(src & SLJIT_MEM));
			FAIL_IF(emit_vex_instruction(compiler, SHUFPS_x_xm | EX86_SSE2 | VEX_SSE2_OPV, vreg, src, src, 0));
			return emit_byte(compiler, 0);
		}

		if (elem_size == 2 && vreg != src) {
			FAIL_IF(emit_sse2_load(compiler, 1, vreg, src, srcw));
			src = vreg;
			srcw = 0;
		}

		op = (elem_size == 2 ? SHUFPS_x_xm : MOVDDUP_x_xm) | (elem_size == 2 ? 0 : EX86_PREF_F2) | EX86_SSE2;
		FAIL_IF(emit_groupf(compiler, op, vreg, src, srcw));

		if (elem_size == 2)
			return emit_byte(compiler, 0);
		return SLJIT_SUCCESS;
	}

	if (src == SLJIT_IMM) {
		if (elem_size == 0) {
			srcw = (sljit_u8)srcw;
			srcw |= srcw << 8;
			srcw |= srcw << 16;
			elem_size = 2;
		} else if (elem_size == 1) {
			srcw = (sljit_u16)srcw;
			srcw |= srcw << 16;
			elem_size = 2;
		}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (elem_size == 2 && (sljit_s32)srcw == -1)
			srcw = -1;
#endif /* SLJIT_CONFIG_X86_64 */

		if (srcw == 0 || srcw == -1) {
			if (use_vex)
				return emit_vex_instruction(compiler, (srcw == 0 ? PXOR_x_xm : PCMPEQD_x_xm) | (reg_size == 5 ? VEX_256 : 0) | EX86_PREF_66 | EX86_SSE2 | VEX_SSE2_OPV, vreg, vreg, vreg, 0);

			return emit_groupf(compiler, (srcw == 0 ? PXOR_x_xm : PCMPEQD_x_xm) | EX86_PREF_66 | EX86_SSE2, vreg, vreg, 0);
		}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (elem_size == 3)
			FAIL_IF(emit_load_imm64(compiler, TMP_REG1, srcw));
		else
#endif /* SLJIT_CONFIG_X86_64 */
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, srcw);

		src = TMP_REG1;
		srcw = 0;

	}

	op = 2;
	opcode = MOVD_x_rm;

	switch (elem_size) {
	case 0:
		if (!FAST_IS_REG(src)) {
			opcode = 0x3a /* Prefix of PINSRB_x_rm_i8. */;
			op = 3;
		}
		break;
	case 1:
		if (!FAST_IS_REG(src))
			opcode = PINSRW_x_rm_i8;
		break;
	case 2:
		break;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	case 3:
		/* MOVQ */
		compiler->mode32 = 0;
		break;
#endif /* SLJIT_CONFIG_X86_64 */
	}

	if (use_vex) {
		if (opcode != MOVD_x_rm) {
			op = (opcode == 0x3a) ? (PINSRB_x_rm_i8 | VEX_OP_0F3A) : opcode;
			FAIL_IF(emit_vex_instruction(compiler, op | EX86_PREF_66 | EX86_SSE2_OP1 | VEX_SSE2_OPV, vreg, vreg, src, srcw));
		} else
			FAIL_IF(emit_vex_instruction(compiler, MOVD_x_rm | VEX_AUTO_W | EX86_PREF_66 | EX86_SSE2_OP1, vreg, 0, src, srcw));
	} else {
		inst = emit_x86_instruction(compiler, op | EX86_PREF_66 | EX86_SSE2_OP1, vreg, 0, src, srcw);
		FAIL_IF(!inst);
		inst[0] = GROUP_0F;
		inst[1] = opcode;

		if (op == 3) {
			SLJIT_ASSERT(opcode == 0x3a);
			inst[2] = PINSRB_x_rm_i8;
		}
	}

	if ((cpu_feature_list & CPU_FEATURE_AVX2) && use_vex && elem_size >= 2) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		op = VPBROADCASTD_x_xm;
#else /* !SLJIT_CONFIG_X86_32 */
		op = (elem_size == 3) ? VPBROADCASTQ_x_xm : VPBROADCASTD_x_xm;
#endif /* SLJIT_CONFIG_X86_32 */
		return emit_vex_instruction(compiler, op | ((reg_size == 5) ? VEX_256 : 0) | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, vreg, 0);
	}

	SLJIT_ASSERT(reg_size == 4);

	if (opcode != MOVD_x_rm)
		FAIL_IF(emit_byte(compiler, 0));

	switch (elem_size) {
	case 0:
		if (use_vex) {
			FAIL_IF(emit_vex_instruction(compiler, PXOR_x_xm | EX86_PREF_66 | EX86_SSE2 | VEX_SSE2_OPV, TMP_FREG, TMP_FREG, TMP_FREG, 0));
			return emit_vex_instruction(compiler, PSHUFB_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2 | VEX_SSE2_OPV, vreg, vreg, TMP_FREG, 0);
		}
		FAIL_IF(emit_groupf(compiler, PXOR_x_xm | EX86_PREF_66 | EX86_SSE2, TMP_FREG, TMP_FREG, 0));
		return emit_groupf_ext(compiler, PSHUFB_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, TMP_FREG, 0);
	case 1:
		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, PSHUFLW_x_xm | EX86_PREF_F2 | EX86_SSE2, vreg, 0, vreg, 0));
		else
			FAIL_IF(emit_groupf(compiler, PSHUFLW_x_xm | EX86_PREF_F2 | EX86_SSE2, vreg, vreg, 0));
		FAIL_IF(emit_byte(compiler, 0));
		/* fallthrough */
	default:
		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, 0, vreg, 0));
		else
			FAIL_IF(emit_groupf(compiler, PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, vreg, 0));
		return emit_byte(compiler, 0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	case 3:
		compiler->mode32 = 1;
		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, 0, vreg, 0));
		else
			FAIL_IF(emit_groupf(compiler, PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, vreg, 0));
		return emit_byte(compiler, 0x44);
#endif /* SLJIT_CONFIG_X86_64 */
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg, sljit_s32 lane_index,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 use_vex = (cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX);
	sljit_u8 *inst;
	sljit_u8 opcode = 0;
	sljit_uw op;
	sljit_s32 vreg_orig = vreg;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 srcdst_is_ereg = 0;
	sljit_s32 srcdst_orig = 0;
	sljit_sw srcdstw_orig = 0;
#endif /* SLJIT_CONFIG_X86_32 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_mov(compiler, type, vreg, lane_index, srcdst, srcdstw));

	ADJUST_LOCAL_OFFSET(srcdst, srcdstw);

	if (reg_size == 5) {
		if (!(cpu_feature_list & CPU_FEATURE_AVX2))
			return SLJIT_ERR_UNSUPPORTED;
		use_vex = 1;
	} else if (reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	if ((type & SLJIT_SIMD_FLOAT) ? (elem_size < 2 || elem_size > 3) : elem_size > 2)
		return SLJIT_ERR_UNSUPPORTED;
#else /* SLJIT_CONFIG_X86_32 */
	if (elem_size > 3 || ((type & SLJIT_SIMD_FLOAT) && elem_size < 2))
		return SLJIT_ERR_UNSUPPORTED;
#endif /* SLJIT_CONFIG_X86_32 */

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#else /* !SLJIT_CONFIG_X86_64 */
	if (!(type & SLJIT_SIMD_FLOAT)) {
		CHECK_EXTRA_REGS(srcdst, srcdstw, srcdst_is_ereg = 1);

		if ((type & SLJIT_SIMD_STORE) && ((srcdst_is_ereg && elem_size < 2) || (elem_size == 0 && (type & SLJIT_SIMD_LANE_SIGNED) && FAST_IS_REG(srcdst) && reg_map[srcdst] >= 4))) {
			srcdst_orig = srcdst;
			srcdstw_orig = srcdstw;
			srcdst = TMP_REG1;
			srcdstw = 0;
		}
	}
#endif /* SLJIT_CONFIG_X86_64 */

	if (type & SLJIT_SIMD_LANE_ZERO) {
		if (lane_index == 0) {
			if (!(type & SLJIT_SIMD_FLOAT)) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
				if (elem_size == 3) {
					compiler->mode32 = 0;
					elem_size = 2;
				}
#endif /* SLJIT_CONFIG_X86_64 */
				if (srcdst == SLJIT_IMM) {
					if (elem_size == 0)
						srcdstw = (sljit_u8)srcdstw;
					else if (elem_size == 1)
						srcdstw = (sljit_u16)srcdstw;

					EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, srcdstw);
					srcdst = TMP_REG1;
					srcdstw = 0;
					elem_size = 2;
				}

				if (elem_size == 2) {
					if (use_vex)
						return emit_vex_instruction(compiler, MOVD_x_rm | VEX_AUTO_W | EX86_PREF_66 | EX86_SSE2_OP1, vreg, 0, srcdst, srcdstw);
					return emit_groupf(compiler, MOVD_x_rm | EX86_PREF_66 | EX86_SSE2_OP1, vreg, srcdst, srcdstw);
				}
			} else if (srcdst & SLJIT_MEM) {
				SLJIT_ASSERT(elem_size == 2 || elem_size == 3);

				if (use_vex)
					return emit_vex_instruction(compiler, MOVSD_x_xm | (elem_size == 2 ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2, vreg, 0, srcdst, srcdstw);
				return emit_groupf(compiler, MOVSD_x_xm | (elem_size == 2 ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2, vreg, srcdst, srcdstw);
			} else if (elem_size == 3) {
				if (use_vex)
					return emit_vex_instruction(compiler, MOVQ_x_xm | EX86_PREF_F3 | EX86_SSE2, vreg, 0, srcdst, 0);
				return emit_groupf(compiler, MOVQ_x_xm | EX86_PREF_F3 | EX86_SSE2, vreg, srcdst, 0);
			} else if (use_vex) {
				FAIL_IF(emit_vex_instruction(compiler, XORPD_x_xm | EX86_SSE2 | VEX_SSE2_OPV, TMP_FREG, TMP_FREG, TMP_FREG, 0));
				return emit_vex_instruction(compiler, MOVSD_x_xm | EX86_PREF_F3 | EX86_SSE2 | VEX_SSE2_OPV, vreg, TMP_FREG, srcdst, 0);
			}
		}

		if (reg_size == 5 && lane_index >= (1 << (4 - elem_size))) {
			vreg = TMP_FREG;
			lane_index -= (1 << (4 - elem_size));
		} else if ((type & SLJIT_SIMD_FLOAT) && vreg == srcdst) {
			if (use_vex)
				FAIL_IF(emit_vex_instruction(compiler, MOVSD_x_xm | (elem_size == 2 ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2 | VEX_SSE2_OPV, TMP_FREG, TMP_FREG, srcdst, srcdstw));
			else
				FAIL_IF(emit_sse2_load(compiler, elem_size == 2, TMP_FREG, srcdst, srcdstw));
			srcdst = TMP_FREG;
			srcdstw = 0;
		}

		op = ((!(type & SLJIT_SIMD_FLOAT) || elem_size != 2) ? EX86_PREF_66 : 0)
			| ((type & SLJIT_SIMD_FLOAT) ? XORPD_x_xm : PXOR_x_xm) | EX86_SSE2;

		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, op | (reg_size == 5 ? VEX_256 : 0) | VEX_SSE2_OPV, vreg, vreg, vreg, 0));
		else
			FAIL_IF(emit_groupf(compiler, op, vreg, vreg, 0));
	} else if (reg_size == 5 && lane_index >= (1 << (4 - elem_size))) {
		FAIL_IF(emit_vex_instruction(compiler, ((type & SLJIT_SIMD_FLOAT) ? VEXTRACTF128_x_ym : VEXTRACTI128_x_ym) | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2, vreg, 0, TMP_FREG, 0));
		FAIL_IF(emit_byte(compiler, 1));

		vreg = TMP_FREG;
		lane_index -= (1 << (4 - elem_size));
	}

	if (type & SLJIT_SIMD_FLOAT) {
		if (elem_size == 3) {
			if (srcdst & SLJIT_MEM) {
				if (type & SLJIT_SIMD_STORE)
					op = lane_index == 0 ? MOVLPD_m_x : MOVHPD_m_x;
				else
					op = lane_index == 0 ? MOVLPD_x_m : MOVHPD_x_m;

				/* VEX prefix clears upper bits of the target register. */
				if (use_vex && ((type & SLJIT_SIMD_STORE) || reg_size == 4 || vreg == TMP_FREG))
					FAIL_IF(emit_vex_instruction(compiler, op | EX86_PREF_66 | EX86_SSE2
						| ((type & SLJIT_SIMD_STORE) ? 0 : VEX_SSE2_OPV), vreg, (type & SLJIT_SIMD_STORE) ? 0 : vreg, srcdst, srcdstw));
				else
					FAIL_IF(emit_groupf(compiler, op | EX86_PREF_66 | EX86_SSE2, vreg, srcdst, srcdstw));

				/* In case of store, vreg is not TMP_FREG. */
			} else if (type & SLJIT_SIMD_STORE) {
				if (lane_index == 1) {
					if (use_vex)
						return emit_vex_instruction(compiler, MOVHLPS_x_x | EX86_SSE2 | VEX_SSE2_OPV, srcdst, srcdst, vreg, 0);
					return emit_groupf(compiler, MOVHLPS_x_x | EX86_SSE2, srcdst, vreg, 0);
				}
				if (use_vex)
					return emit_vex_instruction(compiler, MOVSD_x_xm | EX86_PREF_F2 | EX86_SSE2 | VEX_SSE2_OPV, srcdst, srcdst, vreg, 0);
				return emit_sse2_load(compiler, 0, srcdst, vreg, 0);
			} else if (use_vex && (reg_size == 4 || vreg == TMP_FREG)) {
				if (lane_index == 1)
					FAIL_IF(emit_vex_instruction(compiler, MOVLHPS_x_x | EX86_SSE2 | VEX_SSE2_OPV, vreg, vreg, srcdst, 0));
				else
					FAIL_IF(emit_vex_instruction(compiler, MOVSD_x_xm | EX86_PREF_F2 | EX86_SSE2 | VEX_SSE2_OPV, vreg, vreg, srcdst, 0));
			} else {
				if (lane_index == 1)
					FAIL_IF(emit_groupf(compiler, MOVLHPS_x_x | EX86_SSE2, vreg, srcdst, 0));
				else
					FAIL_IF(emit_sse2_load(compiler, 0, vreg, srcdst, 0));
			}
		} else if (type & SLJIT_SIMD_STORE) {
			if (lane_index == 0) {
				if (use_vex)
					return emit_vex_instruction(compiler, MOVSD_xm_x | EX86_PREF_F3 | EX86_SSE2 | ((srcdst & SLJIT_MEM) ? 0 : VEX_SSE2_OPV),
						vreg, ((srcdst & SLJIT_MEM) ? 0 : srcdst), srcdst, srcdstw);
				return emit_sse2_store(compiler, 1, srcdst, srcdstw, vreg);
			}

			if (srcdst & SLJIT_MEM) {
				if (use_vex)
					FAIL_IF(emit_vex_instruction(compiler, EXTRACTPS_x_xm | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2, vreg, 0, srcdst, srcdstw));
				else
					FAIL_IF(emit_groupf_ext(compiler, EXTRACTPS_x_xm | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2, vreg, srcdst, srcdstw));
				return emit_byte(compiler, U8(lane_index));
			}

			if (use_vex) {
				FAIL_IF(emit_vex_instruction(compiler, SHUFPS_x_xm | EX86_SSE2 | VEX_SSE2_OPV, srcdst, vreg, vreg, 0));
				return emit_byte(compiler, U8(lane_index));
			}

			if (srcdst == vreg)
				op = SHUFPS_x_xm | EX86_SSE2;
			else {
				switch (lane_index) {
				case 1:
					op = MOVSHDUP_x_xm | EX86_PREF_F3 | EX86_SSE2;
					break;
				case 2:
					op = MOVHLPS_x_x | EX86_SSE2;
					break;
				default:
					SLJIT_ASSERT(lane_index == 3);
					op = PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2;
					break;
				}
			}

			FAIL_IF(emit_groupf(compiler, op, srcdst, vreg, 0));

			op &= 0xff;
			if (op == SHUFPS_x_xm || op == PSHUFD_x_xm)
				return emit_byte(compiler, U8(lane_index));

			return SLJIT_SUCCESS;
		} else {
			if (lane_index != 0 || (srcdst & SLJIT_MEM)) {
				FAIL_IF(emit_groupf_ext(compiler, INSERTPS_x_xm | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2, vreg, srcdst, srcdstw));
				FAIL_IF(emit_byte(compiler, U8(lane_index << 4)));
			} else
				FAIL_IF(emit_sse2_store(compiler, 1, vreg, 0, srcdst));
		}

		if (vreg != TMP_FREG || (type & SLJIT_SIMD_STORE))
			return SLJIT_SUCCESS;

		SLJIT_ASSERT(reg_size == 5);

		if (type & SLJIT_SIMD_LANE_ZERO) {
			FAIL_IF(emit_vex_instruction(compiler, VPERMPD_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg_orig, 0, TMP_FREG, 0));
			return emit_byte(compiler, 0x4e);
		}

		FAIL_IF(emit_vex_instruction(compiler, VINSERTF128_y_y_xm | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2 | VEX_SSE2_OPV, vreg_orig, vreg_orig, TMP_FREG, 0));
		return emit_byte(compiler, 1);
	}

	if (srcdst == SLJIT_IMM) {
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, srcdstw);
		srcdst = TMP_REG1;
		srcdstw = 0;
	}

	op = 3;

	switch (elem_size) {
	case 0:
		opcode = (type & SLJIT_SIMD_STORE) ? PEXTRB_rm_x_i8 : PINSRB_x_rm_i8;
		break;
	case 1:
		if (!(type & SLJIT_SIMD_STORE)) {
			op = 2;
			opcode = PINSRW_x_rm_i8;
		} else
			opcode = PEXTRW_rm_x_i8;
		break;
	case 2:
		opcode = (type & SLJIT_SIMD_STORE) ? PEXTRD_rm_x_i8 : PINSRD_x_rm_i8;
		break;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	case 3:
		/* PINSRQ / PEXTRQ */
		opcode = (type & SLJIT_SIMD_STORE) ? PEXTRD_rm_x_i8 : PINSRD_x_rm_i8;
		compiler->mode32 = 0;
		break;
#endif /* SLJIT_CONFIG_X86_64 */
	}

	if (use_vex && (type & SLJIT_SIMD_STORE)) {
		op = opcode | ((op == 3) ? VEX_OP_0F3A : 0);
		FAIL_IF(emit_vex_instruction(compiler, op | EX86_PREF_66 | VEX_AUTO_W | EX86_SSE2_OP1 | VEX_SSE2_OPV, vreg, 0, srcdst, srcdstw));
	} else {
		inst = emit_x86_instruction(compiler, op | EX86_PREF_66 | EX86_SSE2_OP1, vreg, 0, srcdst, srcdstw);
		FAIL_IF(!inst);
		inst[0] = GROUP_0F;

		if (op == 3) {
			inst[1] = 0x3a;
			inst[2] = opcode;
		} else
			inst[1] = opcode;
	}

	FAIL_IF(emit_byte(compiler, U8(lane_index)));

	if (!(type & SLJIT_SIMD_LANE_SIGNED) || (srcdst & SLJIT_MEM)) {
		if (vreg == TMP_FREG && !(type & SLJIT_SIMD_STORE)) {
			SLJIT_ASSERT(reg_size == 5);

			if (type & SLJIT_SIMD_LANE_ZERO) {
				FAIL_IF(emit_vex_instruction(compiler, VPERMQ_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg_orig, 0, TMP_FREG, 0));
				return emit_byte(compiler, 0x4e);
			}

			FAIL_IF(emit_vex_instruction(compiler, VINSERTI128_y_y_xm | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2 | VEX_SSE2_OPV, vreg_orig, vreg_orig, TMP_FREG, 0));
			return emit_byte(compiler, 1);
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (srcdst_orig & SLJIT_MEM)
			return emit_mov(compiler, srcdst_orig, srcdstw_orig, TMP_REG1, 0);
#endif /* SLJIT_CONFIG_X86_32 */
		return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (elem_size >= 3)
		return SLJIT_SUCCESS;

	compiler->mode32 = (type & SLJIT_32);

	op = 2;

	if (elem_size == 0)
		op |= EX86_REX;

	if (elem_size == 2) {
		if (type & SLJIT_32)
			return SLJIT_SUCCESS;

		SLJIT_ASSERT(!(compiler->mode32));
		op = 1;
	}

	inst = emit_x86_instruction(compiler, op, srcdst, 0, srcdst, 0);
	FAIL_IF(!inst);

	if (op != 1) {
		inst[0] = GROUP_0F;
		inst[1] = U8((elem_size == 0) ? MOVSX_r_rm8 : MOVSX_r_rm16);
	} else
		inst[0] = MOVSXD_r_rm;
#else /* !SLJIT_CONFIG_X86_64 */
	if (elem_size >= 2)
		return SLJIT_SUCCESS;

	FAIL_IF(emit_groupf(compiler, (elem_size == 0) ? MOVSX_r_rm8 : MOVSX_r_rm16,
		(srcdst_orig != 0 && FAST_IS_REG(srcdst_orig)) ? srcdst_orig : srcdst, srcdst, 0));

	if (srcdst_orig & SLJIT_MEM)
		return emit_mov(compiler, srcdst_orig, srcdstw_orig, TMP_REG1, 0);
#endif /* SLJIT_CONFIG_X86_64 */
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_s32 src_lane_index)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 use_vex = (cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX);
	sljit_uw pref;
	sljit_u8 byte;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 opcode3 = TMP_REG1;
#else /* !SLJIT_CONFIG_X86_32 */
	sljit_s32 opcode3 = SLJIT_S0;
#endif /* SLJIT_CONFIG_X86_32 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_replicate(compiler, type, vreg, src, src_lane_index));

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */
	SLJIT_ASSERT(reg_map[opcode3] == 3);

	if (reg_size == 5) {
		if (!(cpu_feature_list & CPU_FEATURE_AVX2))
			return SLJIT_ERR_UNSUPPORTED;
		use_vex = 1;
	} else if (reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_FLOAT) {
		pref = 0;
		byte = U8(src_lane_index);

		if (elem_size == 3) {
			if (type & SLJIT_SIMD_TEST)
				return SLJIT_SUCCESS;

			if (reg_size == 5) {
				if (src_lane_index == 0)
					return emit_vex_instruction(compiler, VBROADCASTSD_x_xm | VEX_256 | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, src, 0);

				FAIL_IF(emit_vex_instruction(compiler, VPERMPD_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg, 0, src, 0));

				byte = U8(byte | (byte << 2));
				return emit_byte(compiler, U8(byte | (byte << 4)));
			}

			if (src_lane_index == 0) {
				if (use_vex)
					return emit_vex_instruction(compiler, MOVDDUP_x_xm | EX86_PREF_F2 | EX86_SSE2, vreg, 0, src, 0);
				return emit_groupf(compiler, MOVDDUP_x_xm | EX86_PREF_F2 | EX86_SSE2, vreg, src, 0);
			}

			/* Changes it to SHUFPD_x_xm. */
			pref = EX86_PREF_66;
		} else if (elem_size != 2)
			return SLJIT_ERR_UNSUPPORTED;
		else if (type & SLJIT_SIMD_TEST)
			return SLJIT_SUCCESS;

		if (reg_size == 5) {
			SLJIT_ASSERT(elem_size == 2);

			if (src_lane_index == 0)
				return emit_vex_instruction(compiler, VBROADCASTSS_x_xm | VEX_256 | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, src, 0);

			FAIL_IF(emit_vex_instruction(compiler, VPERMPD_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg, 0, src, 0));

			byte = 0x44;
			if (src_lane_index >= 4) {
				byte = 0xee;
				src_lane_index -= 4;
			}

			FAIL_IF(emit_byte(compiler, byte));
			FAIL_IF(emit_vex_instruction(compiler, SHUFPS_x_xm | VEX_256 | pref | EX86_SSE2 | VEX_SSE2_OPV, vreg, vreg, vreg, 0));
			byte = U8(src_lane_index);
		} else if (use_vex) {
			FAIL_IF(emit_vex_instruction(compiler, SHUFPS_x_xm | pref | EX86_SSE2 | VEX_SSE2_OPV, vreg, src, src, 0));
		} else {
			if (vreg != src)
				FAIL_IF(emit_groupf(compiler, MOVAPS_x_xm | pref | EX86_SSE2, vreg, src, 0));

			FAIL_IF(emit_groupf(compiler, SHUFPS_x_xm | pref | EX86_SSE2, vreg, vreg, 0));
		}

		if (elem_size == 2) {
			byte = U8(byte | (byte << 2));
			byte = U8(byte | (byte << 4));
		} else
			byte = U8(byte | (byte << 1));

		return emit_byte(compiler, U8(byte));
	}

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (elem_size == 0) {
		if (reg_size == 5 && src_lane_index >= 16) {
			FAIL_IF(emit_vex_instruction(compiler, VPERMQ_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg, 0, src, 0));
			FAIL_IF(emit_byte(compiler, src_lane_index >= 24 ? 0xff : 0xaa));
			src_lane_index &= 0x7;
			src = vreg;
		}

		if (src_lane_index != 0 || (vreg != src && (!(cpu_feature_list & CPU_FEATURE_AVX2) || !use_vex))) {
			pref = 0;

			if ((src_lane_index & 0x3) == 0) {
				pref = EX86_PREF_66;
				byte = U8(src_lane_index >> 2);
			} else if (src_lane_index < 8 && (src_lane_index & 0x1) == 0) {
				pref = EX86_PREF_F2;
				byte = U8(src_lane_index >> 1);
			} else {
				if (!use_vex) {
					if (vreg != src)
						FAIL_IF(emit_groupf(compiler, MOVDQA_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, src, 0));

					FAIL_IF(emit_groupf(compiler, PSRLDQ_x | EX86_PREF_66 | EX86_SSE2_OP2, opcode3, vreg, 0));
				} else
					FAIL_IF(emit_vex_instruction(compiler, PSRLDQ_x | EX86_PREF_66 | EX86_SSE2_OP2 | VEX_SSE2_OPV, opcode3, vreg, src, 0));

				FAIL_IF(emit_byte(compiler, U8(src_lane_index)));
			}

			if (pref != 0) {
				if (use_vex)
					FAIL_IF(emit_vex_instruction(compiler, PSHUFLW_x_xm | pref | EX86_SSE2, vreg, 0, src, 0));
				else
					FAIL_IF(emit_groupf(compiler, PSHUFLW_x_xm | pref | EX86_SSE2, vreg, src, 0));
				FAIL_IF(emit_byte(compiler, byte));
			}

			src = vreg;
		}

		if (use_vex && (cpu_feature_list & CPU_FEATURE_AVX2))
			return emit_vex_instruction(compiler, VPBROADCASTB_x_xm | (reg_size == 5 ? VEX_256 : 0) | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, src, 0);

		SLJIT_ASSERT(reg_size == 4);
		FAIL_IF(emit_groupf(compiler, PXOR_x_xm | EX86_PREF_66 | EX86_SSE2, TMP_FREG, TMP_FREG, 0));
		return emit_groupf_ext(compiler, PSHUFB_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, TMP_FREG, 0);
	}

	if ((cpu_feature_list & CPU_FEATURE_AVX2) && use_vex && src_lane_index == 0 && elem_size <= 3) {
		switch (elem_size) {
		case 1:
			pref = VPBROADCASTW_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		case 2:
			pref = VPBROADCASTD_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		default:
			pref = VPBROADCASTQ_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2;
			break;
		}

		if (reg_size == 5)
			pref |= VEX_256;

		return emit_vex_instruction(compiler, pref, vreg, 0, src, 0);
	}

	if (reg_size == 5) {
		switch (elem_size) {
		case 1:
			byte = U8(src_lane_index & 0x3);
			src_lane_index >>= 2;
			pref = PSHUFLW_x_xm | VEX_256 | ((src_lane_index & 1) == 0 ? EX86_PREF_F2 : EX86_PREF_F3) | EX86_SSE2;
			break;
		case 2:
			byte = U8(src_lane_index & 0x3);
			src_lane_index >>= 1;
			pref = PSHUFD_x_xm | VEX_256 | EX86_PREF_66 | EX86_SSE2;
			break;
		case 3:
			pref = 0;
			break;
		default:
			FAIL_IF(emit_vex_instruction(compiler, VPERMQ_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg, 0, src, 0));
			return emit_byte(compiler, U8(src_lane_index == 0 ? 0x44 : 0xee));
		}

		if (pref != 0) {
			FAIL_IF(emit_vex_instruction(compiler, pref, vreg, 0, src, 0));
			byte = U8(byte | (byte << 2));
			FAIL_IF(emit_byte(compiler, U8(byte | (byte << 4))));

			if (src_lane_index == 0)
				return emit_vex_instruction(compiler, VPBROADCASTQ_x_xm | VEX_256 | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, vreg, 0);

			src = vreg;
		}

		FAIL_IF(emit_vex_instruction(compiler, VPERMQ_y_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | VEX_W | EX86_SSE2, vreg, 0, src, 0));
		byte = U8(src_lane_index);
		byte = U8(byte | (byte << 2));
		return emit_byte(compiler, U8(byte | (byte << 4)));
	}

	switch (elem_size) {
	case 1:
		byte = U8(src_lane_index & 0x3);
		src_lane_index >>= 1;
		pref = (src_lane_index & 2) == 0 ? EX86_PREF_F2 : EX86_PREF_F3;

		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, PSHUFLW_x_xm | pref | EX86_SSE2, vreg, 0, src, 0));
		else
			FAIL_IF(emit_groupf(compiler, PSHUFLW_x_xm | pref | EX86_SSE2, vreg, src, 0));
		byte = U8(byte | (byte << 2));
		FAIL_IF(emit_byte(compiler, U8(byte | (byte << 4))));

		if ((cpu_feature_list & CPU_FEATURE_AVX2) && use_vex && pref == EX86_PREF_F2)
			return emit_vex_instruction(compiler, VPBROADCASTD_x_xm | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, vreg, 0);

		src = vreg;
		/* fallthrough */
	case 2:
		byte = U8(src_lane_index);
		byte = U8(byte | (byte << 2));
		break;
	default:
		byte = U8(src_lane_index << 1);
		byte = U8(byte | (byte << 2) | 0x4);
		break;
	}

	if (use_vex)
		FAIL_IF(emit_vex_instruction(compiler, PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, 0, src, 0));
	else
		FAIL_IF(emit_groupf(compiler, PSHUFD_x_xm | EX86_PREF_66 | EX86_SSE2, vreg, src, 0));
	return emit_byte(compiler, U8(byte | (byte << 4)));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_extend(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 elem2_size = SLJIT_SIMD_GET_ELEM2_SIZE(type);
	sljit_s32 use_vex = (cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX);
	sljit_u8 opcode;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_extend(compiler, type, vreg, src, srcw));

	ADJUST_LOCAL_OFFSET(src, srcw);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */

	if (reg_size == 5) {
		if (!(cpu_feature_list & CPU_FEATURE_AVX2))
			return SLJIT_ERR_UNSUPPORTED;
		use_vex = 1;
	} else if (reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_FLOAT) {
		if (elem_size != 2 || elem2_size != 3)
			return SLJIT_ERR_UNSUPPORTED;

		if (type & SLJIT_SIMD_TEST)
			return SLJIT_SUCCESS;

		if (use_vex)
			return emit_vex_instruction(compiler, CVTPS2PD_x_xm | ((reg_size == 5) ? VEX_256 : 0) | EX86_SSE2, vreg, 0, src, srcw);
		return emit_groupf(compiler, CVTPS2PD_x_xm | EX86_SSE2, vreg, src, srcw);
	}

	switch (elem_size) {
	case 0:
		if (elem2_size == 1)
			opcode = (type & SLJIT_SIMD_EXTEND_SIGNED) ? PMOVSXBW_x_xm : PMOVZXBW_x_xm;
		else if (elem2_size == 2)
			opcode = (type & SLJIT_SIMD_EXTEND_SIGNED) ? PMOVSXBD_x_xm : PMOVZXBD_x_xm;
		else if (elem2_size == 3)
			opcode = (type & SLJIT_SIMD_EXTEND_SIGNED) ? PMOVSXBQ_x_xm : PMOVZXBQ_x_xm;
		else
			return SLJIT_ERR_UNSUPPORTED;
		break;
	case 1:
		if (elem2_size == 2)
			opcode = (type & SLJIT_SIMD_EXTEND_SIGNED) ? PMOVSXWD_x_xm : PMOVZXWD_x_xm;
		else if (elem2_size == 3)
			opcode = (type & SLJIT_SIMD_EXTEND_SIGNED) ? PMOVSXWQ_x_xm : PMOVZXWQ_x_xm;
		else
			return SLJIT_ERR_UNSUPPORTED;
		break;
	case 2:
		if (elem2_size == 3)
			opcode = (type & SLJIT_SIMD_EXTEND_SIGNED) ? PMOVSXDQ_x_xm : PMOVZXDQ_x_xm;
		else
			return SLJIT_ERR_UNSUPPORTED;
		break;
	default:
		return SLJIT_ERR_UNSUPPORTED;
	}

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (use_vex)
		return emit_vex_instruction(compiler, opcode | ((reg_size == 5) ? VEX_256 : 0) | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, 0, src, srcw);
	return emit_groupf_ext(compiler, opcode | EX86_PREF_66 | VEX_OP_0F38 | EX86_SSE2, vreg, src, srcw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_sign(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 use_vex = (cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX);
	sljit_s32 dst_r;
	sljit_uw op;
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_sign(compiler, type, vreg, dst, dstw));

	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */

	if (elem_size > 3 || ((type & SLJIT_SIMD_FLOAT) && elem_size < 2))
		return SLJIT_ERR_UNSUPPORTED;

	if (reg_size == 4) {
		if (type & SLJIT_SIMD_TEST)
			return SLJIT_SUCCESS;

		op = EX86_PREF_66 | EX86_SSE2_OP2;

		switch (elem_size) {
		case 1:
			if (use_vex)
				FAIL_IF(emit_vex_instruction(compiler, PACKSSWB_x_xm | EX86_PREF_66 | EX86_SSE2 | VEX_SSE2_OPV, TMP_FREG, vreg, vreg, 0));
			else
				FAIL_IF(emit_groupf(compiler, PACKSSWB_x_xm | EX86_PREF_66 | EX86_SSE2, TMP_FREG, vreg, 0));
			vreg = TMP_FREG;
			break;
		case 2:
			op = EX86_SSE2_OP2;
			break;
		}

		dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
		op |= (elem_size < 2) ? PMOVMSKB_r_x : MOVMSKPS_r_x;

		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, op, dst_r, 0, vreg, 0));
		else
			FAIL_IF(emit_groupf(compiler, op, dst_r, vreg, 0));

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = type & SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */

		if (elem_size == 1) {
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_IMM, 8, dst_r, 0);
			FAIL_IF(!inst);
			inst[1] |= SHR;
		}

		if (dst_r == TMP_REG1)
			return emit_mov(compiler, dst, dstw, TMP_REG1, 0);

		return SLJIT_SUCCESS;
	}

	if (reg_size != 5 || !(cpu_feature_list & CPU_FEATURE_AVX2))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (elem_size == 1) {
		FAIL_IF(emit_vex_instruction(compiler, VEXTRACTI128_x_ym | VEX_256 | EX86_PREF_66 | VEX_OP_0F3A | EX86_SSE2, vreg, 0, TMP_FREG, 0));
		FAIL_IF(emit_byte(compiler, 1));
		FAIL_IF(emit_vex_instruction(compiler, PACKSSWB_x_xm | VEX_256 | EX86_PREF_66 | EX86_SSE2 | VEX_SSE2_OPV, TMP_FREG, vreg, TMP_FREG, 0));
		FAIL_IF(emit_groupf(compiler, PMOVMSKB_r_x | EX86_PREF_66 | EX86_SSE2_OP2, dst_r, TMP_FREG, 0));
	} else {
		op = MOVMSKPS_r_x | VEX_256 | EX86_SSE2_OP2;

		if (elem_size == 0)
			op = PMOVMSKB_r_x | VEX_256 | EX86_PREF_66 | EX86_SSE2_OP2;
		else if (elem_size == 3)
			op |= EX86_PREF_66;

		FAIL_IF(emit_vex_instruction(compiler, op, dst_r, 0, vreg, 0));
	}

	if (dst_r == TMP_REG1) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = type & SLJIT_32;
#endif /* SLJIT_CONFIG_X86_64 */
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_vreg, sljit_s32 src_vreg)
{
	sljit_uw op = ((type & SLJIT_SIMD_FLOAT) ? MOVAPS_x_xm : MOVDQA_x_xm) | EX86_SSE2;

	SLJIT_ASSERT(SLJIT_SIMD_GET_REG_SIZE(type) == 4);

	if (!(type & SLJIT_SIMD_FLOAT) || SLJIT_SIMD_GET_ELEM_SIZE(type) == 3)
		op |= EX86_PREF_66;

	return emit_groupf(compiler, op, dst_vreg, src_vreg, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_op2(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_vreg, sljit_s32 src1_vreg, sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 use_vex = (cpu_feature_list & CPU_FEATURE_AVX) && (compiler->options & SLJIT_ENTER_USE_VEX);
	sljit_uw op = 0;
	sljit_uw mov_op = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_op2(compiler, type, dst_vreg, src1_vreg, src2, src2w));
	ADJUST_LOCAL_OFFSET(src2, src2w);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 1;
#endif /* SLJIT_CONFIG_X86_64 */

	if (reg_size == 5) {
		if (!(cpu_feature_list & CPU_FEATURE_AVX2))
			return SLJIT_ERR_UNSUPPORTED;
	} else if (reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	switch (SLJIT_SIMD_GET_OPCODE(type)) {
	case SLJIT_SIMD_OP2_AND:
		op = (type & SLJIT_SIMD_FLOAT) ? ANDPD_x_xm : PAND_x_xm;

		if (!(type & SLJIT_SIMD_FLOAT) || elem_size == 3)
			op |= EX86_PREF_66;
		break;
	case SLJIT_SIMD_OP2_OR:
		op = (type & SLJIT_SIMD_FLOAT) ? ORPD_x_xm : POR_x_xm;

		if (!(type & SLJIT_SIMD_FLOAT) || elem_size == 3)
			op |= EX86_PREF_66;
		break;
	case SLJIT_SIMD_OP2_XOR:
		op = (type & SLJIT_SIMD_FLOAT) ? XORPD_x_xm : PXOR_x_xm;

		if (!(type & SLJIT_SIMD_FLOAT) || elem_size == 3)
			op |= EX86_PREF_66;
		break;

	case SLJIT_SIMD_OP2_SHUFFLE:
		if (reg_size != 4)
			return SLJIT_ERR_UNSUPPORTED;

		op = PSHUFB_x_xm | EX86_PREF_66 | VEX_OP_0F38;
		break;
	}

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if ((src2 & SLJIT_MEM) && SLJIT_SIMD_GET_ELEM2_SIZE(type) < reg_size) {
		mov_op = ((type & SLJIT_SIMD_FLOAT) ? (MOVUPS_x_xm | (elem_size == 3 ? EX86_PREF_66 : 0)) : (MOVDQU_x_xm | EX86_PREF_F3)) | EX86_SSE2;
		if (use_vex)
			FAIL_IF(emit_vex_instruction(compiler, mov_op, TMP_FREG, 0, src2, src2w));
		else
			FAIL_IF(emit_groupf(compiler, mov_op, TMP_FREG, src2, src2w));

		src2 = TMP_FREG;
		src2w = 0;
	}

	if (reg_size == 5 || use_vex) {
		if (reg_size == 5)
			op |= VEX_256;

		return emit_vex_instruction(compiler, op | EX86_SSE2 | VEX_SSE2_OPV, dst_vreg, src1_vreg, src2, src2w);
	}

	if (dst_vreg != src1_vreg) {
		if (dst_vreg == src2) {
			if (SLJIT_SIMD_GET_OPCODE(type) == SLJIT_SIMD_OP2_SHUFFLE) {
				FAIL_IF(emit_simd_mov(compiler, type, TMP_FREG, src2));
				FAIL_IF(emit_simd_mov(compiler, type, dst_vreg, src1_vreg));
				src2 = TMP_FREG;
				src2w = 0;
			} else
				src2 = src1_vreg;
		} else
			FAIL_IF(emit_simd_mov(compiler, type, dst_vreg, src1_vreg));
	}

	if (op & (VEX_OP_0F38 | VEX_OP_0F3A))
		return emit_groupf_ext(compiler, op | EX86_SSE2, dst_vreg, src2, src2w);
	return emit_groupf(compiler, op | EX86_SSE2, dst_vreg, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_load(compiler, op, dst_reg, mem_reg));

	if ((op & SLJIT_ATOMIC_USE_LS) || GET_OPCODE(op) == SLJIT_MOV_S8 || GET_OPCODE(op) == SLJIT_MOV_S16 || GET_OPCODE(op) == SLJIT_MOV_S32)
		return SLJIT_ERR_UNSUPPORTED;

	if (op & SLJIT_ATOMIC_TEST)
		return SLJIT_SUCCESS;

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op1(compiler, op & ~SLJIT_ATOMIC_USE_CAS, dst_reg, 0, SLJIT_MEM1(mem_reg), 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	sljit_uw pref;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 saved_reg = TMP_REG1;
	sljit_s32 swap_tmp = 0;
	sljit_sw srcw = 0;
	sljit_sw tempw = 0;
#endif /* SLJIT_CONFIG_X86_32 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_store(compiler, op, src_reg, mem_reg, temp_reg));
	CHECK_EXTRA_REGS(src_reg, srcw, (void)0);
	CHECK_EXTRA_REGS(temp_reg, tempw, (void)0);

	SLJIT_ASSERT(FAST_IS_REG(src_reg) || src_reg == SLJIT_MEM1(SLJIT_SP));
	SLJIT_ASSERT(FAST_IS_REG(temp_reg) || temp_reg == SLJIT_MEM1(SLJIT_SP));

	if ((op & SLJIT_ATOMIC_USE_LS) || GET_OPCODE(op) == SLJIT_MOV_S8 || GET_OPCODE(op) == SLJIT_MOV_S16 || GET_OPCODE(op) == SLJIT_MOV_S32)
		return SLJIT_ERR_UNSUPPORTED;

	if (op & SLJIT_ATOMIC_TEST)
		return SLJIT_SUCCESS;

	op = GET_OPCODE(op);

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	if (temp_reg == SLJIT_TMP_DEST_REG) {
		FAIL_IF(emit_byte(compiler, XCHG_EAX_r | reg_map[TMP_REG1]));

		if (src_reg == SLJIT_R0)
			src_reg = TMP_REG1;
		if (mem_reg == SLJIT_R0)
			mem_reg = TMP_REG1;

		temp_reg = SLJIT_R0;
		swap_tmp = 1;
	}

	/* Src is virtual register or its low byte is not accessible. */
	if ((src_reg & SLJIT_MEM) || (op == SLJIT_MOV_U8 && reg_map[src_reg] >= 4)) {
		SLJIT_ASSERT(src_reg != SLJIT_R1 && temp_reg != SLJIT_TMP_DEST_REG);

		if (swap_tmp) {
			saved_reg = (mem_reg != SLJIT_R1) ? SLJIT_R1 : SLJIT_R2;

			EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, saved_reg, 0);
			EMIT_MOV(compiler, saved_reg, 0, src_reg, srcw);
		} else
			EMIT_MOV(compiler, TMP_REG1, 0, src_reg, srcw);

		src_reg = saved_reg;

		if (mem_reg == src_reg)
			mem_reg = saved_reg;
	}
#endif /* SLJIT_CONFIG_X86_32 */

	if (temp_reg != SLJIT_R0) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;

		EMIT_MOV(compiler, TMP_REG2, 0, SLJIT_R0, 0);
		EMIT_MOV(compiler, SLJIT_R0, 0, temp_reg, 0);

		if (src_reg == SLJIT_R0)
			src_reg = TMP_REG2;
		if (mem_reg == SLJIT_R0)
			mem_reg = TMP_REG2;
#else /* !SLJIT_CONFIG_X86_64 */
		SLJIT_ASSERT(!swap_tmp);

		if (src_reg == TMP_REG1) {
			if (mem_reg == SLJIT_R0) {
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, SLJIT_R1, 0);
				EMIT_MOV(compiler, SLJIT_R1, 0, SLJIT_R0, 0);
				EMIT_MOV(compiler, SLJIT_R0, 0, temp_reg, tempw);

				mem_reg = SLJIT_R1;
				saved_reg = SLJIT_R1;
			} else {
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, SLJIT_R0, 0);
				EMIT_MOV(compiler, SLJIT_R0, 0, temp_reg, tempw);
				saved_reg = SLJIT_R0;
			}
		} else {
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_R0, 0);
			EMIT_MOV(compiler, SLJIT_R0, 0, temp_reg, tempw);

			if (src_reg == SLJIT_R0)
				src_reg = TMP_REG1;
			if (mem_reg == SLJIT_R0)
				mem_reg = TMP_REG1;
		}
#endif /* SLJIT_CONFIG_X86_64 */
	}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op != SLJIT_MOV && op != SLJIT_MOV_P;
#endif /* SLJIT_CONFIG_X86_64 */

	/* Lock prefix. */
	FAIL_IF(emit_byte(compiler, GROUP_LOCK));

	pref = 0;
	if (op == SLJIT_MOV_U16)
		pref = EX86_HALF_ARG | EX86_PREF_66;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (op == SLJIT_MOV_U8)
		pref = EX86_REX;
#endif /* SLJIT_CONFIG_X86_64 */

	FAIL_IF(emit_groupf(compiler, (op == SLJIT_MOV_U8 ? CMPXCHG_rm8_r : CMPXCHG_rm_r) | pref, src_reg, SLJIT_MEM1(mem_reg), 0));

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	if (swap_tmp) {
		SLJIT_ASSERT(temp_reg == SLJIT_R0);
		FAIL_IF(emit_byte(compiler, XCHG_EAX_r | reg_map[TMP_REG1]));

		if (saved_reg != TMP_REG1)
			return emit_mov(compiler, saved_reg, 0, SLJIT_MEM1(SLJIT_SP), 0);
		return SLJIT_SUCCESS;
	}
#endif /* SLJIT_CONFIG_X86_32 */

	if (temp_reg != SLJIT_R0) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;
		return emit_mov(compiler, SLJIT_R0, 0, TMP_REG2, 0);
#else /* !SLJIT_CONFIG_X86_64 */
		EMIT_MOV(compiler, SLJIT_R0, 0, (saved_reg == SLJIT_R0) ? SLJIT_MEM1(SLJIT_SP) : saved_reg, 0);
		if (saved_reg == SLJIT_R1)
			return emit_mov(compiler, SLJIT_R1, 0, SLJIT_MEM1(SLJIT_SP), 0);
#endif /* SLJIT_CONFIG_X86_64 */
	}
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	CHECK_ERROR();
	CHECK(check_sljit_get_local_base(compiler, dst, dstw, offset));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
#endif

	ADJUST_LOCAL_OFFSET(SLJIT_MEM1(SLJIT_SP), offset);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (NOT_HALFWORD(offset)) {
		FAIL_IF(emit_load_imm64(compiler, TMP_REG1, offset));
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
		SLJIT_ASSERT(emit_lea_binary(compiler, dst, dstw, SLJIT_SP, 0, TMP_REG1, 0) != SLJIT_ERR_UNSUPPORTED);
		return compiler->error;
#else
		return emit_lea_binary(compiler, dst, dstw, SLJIT_SP, 0, TMP_REG1, 0);
#endif
	}
#endif

	if (offset != 0)
		return emit_lea_binary(compiler, dst, dstw, SLJIT_SP, 0, SLJIT_IMM, offset);
	return emit_mov(compiler, dst, dstw, SLJIT_SP, 0);
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	sljit_u8 *inst;
	struct sljit_const *const_;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 reg;
#endif

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

	const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
	reg = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (emit_load_imm64(compiler, reg, init_value))
		return NULL;
#else
	if (emit_mov(compiler, dst, dstw, SLJIT_IMM, init_value))
		return NULL;
#endif

	inst = (sljit_u8*)ensure_buf(compiler, 1);
	PTR_FAIL_IF(!inst);

	inst[0] = SLJIT_INST_CONST;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (dst & SLJIT_MEM)
		if (emit_mov(compiler, dst, dstw, TMP_REG1, 0))
			return NULL;
#endif

	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_mov_addr(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_jump *jump;
	sljit_u8 *inst;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 reg;
#endif /* SLJIT_CONFIG_X86_64 */

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_mov_addr(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_mov_addr(jump, compiler, 0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
	reg = FAST_IS_REG(dst) ? dst : TMP_REG1;

	PTR_FAIL_IF(emit_load_imm64(compiler, reg, 0));
	jump->addr = compiler->size;

	if (reg_map[reg] >= 8)
		jump->flags |= MOV_ADDR_HI;
#else /* !SLJIT_CONFIG_X86_64 */
	PTR_FAIL_IF(emit_mov(compiler, dst, dstw, SLJIT_IMM, 0));
#endif /* SLJIT_CONFIG_X86_64 */

	inst = (sljit_u8*)ensure_buf(compiler, 1);
	PTR_FAIL_IF(!inst);

	inst[0] = SLJIT_INST_MOV_ADDR;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_mov(compiler, dst, dstw, TMP_REG1, 0));
#endif /* SLJIT_CONFIG_X86_64 */

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS((void*)addr, (void*)(addr + sizeof(sljit_uw)), 0);
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_unaligned_store_sw((void*)addr, (sljit_sw)(new_target - (addr + 4) - (sljit_uw)executable_offset));
#else
	sljit_unaligned_store_sw((void*)addr, (sljit_sw)new_target);
#endif
	SLJIT_UPDATE_WX_FLAGS((void*)addr, (void*)(addr + sizeof(sljit_uw)), 1);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS((void*)addr, (void*)(addr + sizeof(sljit_sw)), 0);
	sljit_unaligned_store_sw((void*)addr, new_constant);
	SLJIT_UPDATE_WX_FLAGS((void*)addr, (void*)(addr + sizeof(sljit_sw)), 1);
}
