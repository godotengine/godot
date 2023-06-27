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

#define TMP_FREG	(0)

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)

/* Last register + 1. */
#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 3] = {
	0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 7, 6, 3, 4, 5
};

#define CHECK_EXTRA_REGS(p, w, do) \
	if (p >= SLJIT_R3 && p <= SLJIT_S3) { \
		w = (2 * SSIZE_OF(sw)) + ((p) - SLJIT_R3) * SSIZE_OF(sw); \
		p = SLJIT_MEM1(SLJIT_SP); \
		do; \
	}

#else /* SLJIT_CONFIG_X86_32 */

/* Last register + 1. */
#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)

/* Note: r12 & 0x7 == 0b100, which decoded as SIB byte present
   Note: avoid to use r12 and r13 for memory addessing
   therefore r12 is better to be a higher saved register. */
#ifndef _WIN64
/* Args: rdi(=7), rsi(=6), rdx(=2), rcx(=1), r8, r9. Scratches: rax(=0), r10, r11 */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 0, 6, 7, 1, 8, 11, 10, 12, 5, 13, 14, 15, 3, 4, 2, 9
};
/* low-map. reg_map & 0x7. */
static const sljit_u8 reg_lmap[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 0, 6, 7, 1, 0, 3,  2,  4,  5,  5,  6,  7, 3, 4, 2, 1
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
static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1] = {
	4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};
/* low-map. freg_map & 0x7. */
static const sljit_u8 freg_lmap[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1] = {
	4, 0, 1, 2, 3, 5, 6, 7, 0, 1, 2,  3,  4,  5,  6,  7
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
#define EX86_BIN_INS		0x0010
#define EX86_SHIFT_INS		0x0020
#define EX86_REX		0x0040
#define EX86_NO_REXW		0x0080
#define EX86_BYTE_ARG		0x0100
#define EX86_HALF_ARG		0x0200
#define EX86_PREF_66		0x0400
#define EX86_PREF_F2		0x0800
#define EX86_PREF_F3		0x1000
#define EX86_SSE2_OP1		0x2000
#define EX86_SSE2_OP2		0x4000
#define EX86_SSE2		(EX86_SSE2_OP1 | EX86_SSE2_OP2)

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

#define ADD		(/* BINARY */ 0 << 3)
#define ADD_EAX_i32	0x05
#define ADD_r_rm	0x03
#define ADD_rm_r	0x01
#define ADDSD_x_xm	0x58
#define ADC		(/* BINARY */ 2 << 3)
#define ADC_EAX_i32	0x15
#define ADC_r_rm	0x13
#define ADC_rm_r	0x11
#define AND		(/* BINARY */ 4 << 3)
#define AND_EAX_i32	0x25
#define AND_r_rm	0x23
#define AND_rm_r	0x21
#define ANDPD_x_xm	0x54
#define BSR_r_rm	(/* GROUP_0F */ 0xbd)
#define BSF_r_rm	(/* GROUP_0F */ 0xbc)
#define CALL_i32	0xe8
#define CALL_rm		(/* GROUP_FF */ 2 << 3)
#define CDQ		0x99
#define CMOVE_r_rm	(/* GROUP_0F */ 0x44)
#define CMP		(/* BINARY */ 7 << 3)
#define CMP_EAX_i32	0x3d
#define CMP_r_rm	0x3b
#define CMP_rm_r	0x39
#define CVTPD2PS_x_xm	0x5a
#define CVTSI2SD_x_rm	0x2a
#define CVTTSD2SI_r_xm	0x2c
#define DIV		(/* GROUP_F7 */ 6 << 3)
#define DIVSD_x_xm	0x5e
#define FLDS		0xd9
#define FLDL		0xdd
#define FSTPS		0xd9
#define FSTPD		0xdd
#define INT3		0xcc
#define IDIV		(/* GROUP_F7 */ 7 << 3)
#define IMUL		(/* GROUP_F7 */ 5 << 3)
#define IMUL_r_rm	(/* GROUP_0F */ 0xaf)
#define IMUL_r_rm_i8	0x6b
#define IMUL_r_rm_i32	0x69
#define JE_i8		0x74
#define JNE_i8		0x75
#define JMP_i8		0xeb
#define JMP_i32		0xe9
#define JMP_rm		(/* GROUP_FF */ 4 << 3)
#define LEA_r_m		0x8d
#define LOOP_i8		0xe2
#define LZCNT_r_rm	(/* GROUP_F3 */ /* GROUP_0F */ 0xbd)
#define MOV_r_rm	0x8b
#define MOV_r_i32	0xb8
#define MOV_rm_r	0x89
#define MOV_rm_i32	0xc7
#define MOV_rm8_i8	0xc6
#define MOV_rm8_r8	0x88
#define MOVAPS_x_xm	0x28
#define MOVAPS_xm_x	0x29
#define MOVSD_x_xm	0x10
#define MOVSD_xm_x	0x11
#define MOVSXD_r_rm	0x63
#define MOVSX_r_rm8	(/* GROUP_0F */ 0xbe)
#define MOVSX_r_rm16	(/* GROUP_0F */ 0xbf)
#define MOVZX_r_rm8	(/* GROUP_0F */ 0xb6)
#define MOVZX_r_rm16	(/* GROUP_0F */ 0xb7)
#define MUL		(/* GROUP_F7 */ 4 << 3)
#define MULSD_x_xm	0x59
#define NEG_rm		(/* GROUP_F7 */ 3 << 3)
#define NOP		0x90
#define NOT_rm		(/* GROUP_F7 */ 2 << 3)
#define OR		(/* BINARY */ 1 << 3)
#define OR_r_rm		0x0b
#define OR_EAX_i32	0x0d
#define OR_rm_r		0x09
#define OR_rm8_r8	0x08
#define POP_r		0x58
#define POP_rm		0x8f
#define POPF		0x9d
#define PREFETCH	0x18
#define PUSH_i32	0x68
#define PUSH_r		0x50
#define PUSH_rm		(/* GROUP_FF */ 6 << 3)
#define PUSHF		0x9c
#define ROL		(/* SHIFT */ 0 << 3)
#define ROR		(/* SHIFT */ 1 << 3)
#define RET_near	0xc3
#define RET_i16		0xc2
#define SBB		(/* BINARY */ 3 << 3)
#define SBB_EAX_i32	0x1d
#define SBB_r_rm	0x1b
#define SBB_rm_r	0x19
#define SAR		(/* SHIFT */ 7 << 3)
#define SHL		(/* SHIFT */ 4 << 3)
#define SHLD		(/* GROUP_0F */ 0xa5)
#define SHRD		(/* GROUP_0F */ 0xad)
#define SHR		(/* SHIFT */ 5 << 3)
#define SUB		(/* BINARY */ 5 << 3)
#define SUB_EAX_i32	0x2d
#define SUB_r_rm	0x2b
#define SUB_rm_r	0x29
#define SUBSD_x_xm	0x5c
#define TEST_EAX_i32	0xa9
#define TEST_rm_r	0x85
#define TZCNT_r_rm	(/* GROUP_F3 */ /* GROUP_0F */ 0xbc)
#define UCOMISD_x_xm	0x2e
#define UNPCKLPD_x_xm	0x14
#define XCHG_EAX_r	0x90
#define XCHG_r_rm	0x87
#define XOR		(/* BINARY */ 6 << 3)
#define XOR_EAX_i32	0x35
#define XOR_r_rm	0x33
#define XOR_rm_r	0x31
#define XORPD_x_xm	0x57

#define GROUP_0F	0x0f
#define GROUP_F3	0xf3
#define GROUP_F7	0xf7
#define GROUP_FF	0xff
#define GROUP_BINARY_81	0x81
#define GROUP_BINARY_83	0x83
#define GROUP_SHIFT_1	0xd1
#define GROUP_SHIFT_N	0xc1
#define GROUP_SHIFT_CL	0xd3

#define MOD_REG		0xc0
#define MOD_DISP8	0x40

#define INC_SIZE(s)			(*inst++ = U8(s), compiler->size += (s))

#define PUSH_REG(r)			(*inst++ = U8(PUSH_r + (r)))
#define POP_REG(r)			(*inst++ = U8(POP_r + (r)))
#define RET()				(*inst++ = RET_near)
#define RET_I16(n)			(*inst++ = RET_i16, *inst++ = U8(n), *inst++ = 0)

/* Multithreading does not affect these static variables, since they store
   built-in CPU features. Therefore they can be overwritten by different threads
   if they detect the CPU features in the same time. */
#define CPU_FEATURE_DETECTED		0x001
#if (defined SLJIT_DETECT_SSE2 && SLJIT_DETECT_SSE2)
#define CPU_FEATURE_SSE2		0x002
#endif
#define CPU_FEATURE_LZCNT		0x004
#define CPU_FEATURE_TZCNT		0x008
#define CPU_FEATURE_CMOV		0x010

static sljit_u32 cpu_feature_list = 0;

#ifdef _WIN32_WCE
#include <cmnintrin.h>
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#include <intrin.h>
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

static void get_cpu_features(void)
{
	sljit_u32 feature_list = CPU_FEATURE_DETECTED;
	sljit_u32 value;

#if defined(_MSC_VER) && _MSC_VER >= 1400

	int CPUInfo[4];

	__cpuid(CPUInfo, 0);
	if (CPUInfo[0] >= 7) {
		__cpuidex(CPUInfo, 7, 0);
		if (CPUInfo[1] & 0x8)
			feature_list |= CPU_FEATURE_TZCNT;
	}

	__cpuid(CPUInfo, (int)0x80000001);
	if (CPUInfo[2] & 0x20)
		feature_list |= CPU_FEATURE_LZCNT;

	__cpuid(CPUInfo, 1);
	value = (sljit_u32)CPUInfo[3];

#elif defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__SUNPRO_C)

	/* AT&T syntax. */
	__asm__ (
		"movl $0x0, %%eax\n"
		"lzcnt %%eax, %%eax\n"
		"setnz %%al\n"
		"movl %%eax, %0\n"
		: "=g" (value)
		:
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		: "eax"
#else
		: "rax"
#endif
	);

	if (value & 0x1)
		feature_list |= CPU_FEATURE_LZCNT;

	__asm__ (
		"movl $0x0, %%eax\n"
		"tzcnt %%eax, %%eax\n"
		"setnz %%al\n"
		"movl %%eax, %0\n"
		: "=g" (value)
		:
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		: "eax"
#else
		: "rax"
#endif
	);

	if (value & 0x1)
		feature_list |= CPU_FEATURE_TZCNT;

	__asm__ (
		"movl $0x1, %%eax\n"
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		/* On x86-32, there is no red zone, so this
		   should work (no need for a local variable). */
		"push %%ebx\n"
#endif
		"cpuid\n"
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		"pop %%ebx\n"
#endif
		"movl %%edx, %0\n"
		: "=g" (value)
		:
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		: "%eax", "%ecx", "%edx"
#else
		: "%rax", "%rbx", "%rcx", "%rdx"
#endif
	);

#else /* _MSC_VER && _MSC_VER >= 1400 */

	/* Intel syntax. */
	__asm {
		mov eax, 0
		lzcnt eax, eax
		setnz al
		mov value, eax
	}

	if (value & 0x1)
		feature_list |= CPU_FEATURE_LZCNT;

	__asm {
		mov eax, 0
		tzcnt eax, eax
		setnz al
		mov value, eax
	}

	if (value & 0x1)
		feature_list |= CPU_FEATURE_TZCNT;

	__asm {
		mov eax, 1
		cpuid
		mov value, edx
	}

#endif /* _MSC_VER && _MSC_VER >= 1400 */

#if (defined SLJIT_DETECT_SSE2 && SLJIT_DETECT_SSE2)
	if (value & 0x4000000)
		feature_list |= CPU_FEATURE_SSE2;
#endif
	if (value & 0x8000)
		feature_list |= CPU_FEATURE_CMOV;

	cpu_feature_list = feature_list;
}

static sljit_u8 get_jump_code(sljit_uw type)
{
	switch (type) {
	case SLJIT_EQUAL:
	case SLJIT_F_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_EQUAL: /* Not supported. */
		return 0x84 /* je */;

	case SLJIT_NOT_EQUAL:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL: /* Not supported. */
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
		return 0x8a /* jp */;

	case SLJIT_ORDERED:
		return 0x8b /* jpo */;
	}
	return 0;
}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
static sljit_u8* generate_far_jump_code(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_sw executable_offset);
#else
static sljit_u8* generate_far_jump_code(struct sljit_jump *jump, sljit_u8 *code_ptr);
static sljit_u8* generate_put_label_code(struct sljit_put_label *put_label, sljit_u8 *code_ptr, sljit_uw max_label);
#endif

static sljit_u8* generate_near_jump_code(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_u8 *code, sljit_sw executable_offset)
{
	sljit_uw type = jump->flags >> TYPE_SHIFT;
	sljit_s32 short_jump;
	sljit_uw label_addr;

	if (jump->flags & JUMP_LABEL)
		label_addr = (sljit_uw)(code + jump->u.label->size);
	else
		label_addr = jump->u.target - (sljit_uw)executable_offset;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if ((sljit_sw)(label_addr - (jump->addr + 1)) > HALFWORD_MAX || (sljit_sw)(label_addr - (jump->addr + 1)) < HALFWORD_MIN)
		return generate_far_jump_code(jump, code_ptr);
#endif

	short_jump = (sljit_sw)(label_addr - (jump->addr + 2)) >= -128 && (sljit_sw)(label_addr - (jump->addr + 2)) <= 127;

	if (type == SLJIT_JUMP) {
		if (short_jump)
			*code_ptr++ = JMP_i8;
		else
			*code_ptr++ = JMP_i32;
		jump->addr++;
	}
	else if (type >= SLJIT_FAST_CALL) {
		short_jump = 0;
		*code_ptr++ = CALL_i32;
		jump->addr++;
	}
	else if (short_jump) {
		*code_ptr++ = U8(get_jump_code(type) - 0x10);
		jump->addr++;
	}
	else {
		*code_ptr++ = GROUP_0F;
		*code_ptr++ = get_jump_code(type);
		jump->addr += 2;
	}

	if (short_jump) {
		jump->flags |= PATCH_MB;
		code_ptr += sizeof(sljit_s8);
	} else {
		jump->flags |= PATCH_MW;
		code_ptr += sizeof(sljit_s32);
	}

	return code_ptr;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_u8 *code;
	sljit_u8 *code_ptr;
	sljit_u8 *buf_ptr;
	sljit_u8 *buf_end;
	sljit_u8 len;
	sljit_sw executable_offset;
	sljit_uw jump_addr;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;
	struct sljit_put_label *put_label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	/* Second code generation pass. */
	code = (sljit_u8*)SLJIT_MALLOC_EXEC(compiler->size, compiler->exec_allocator_data);
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

	code_ptr = code;
	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	put_label = compiler->put_labels;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	do {
		buf_ptr = buf->memory;
		buf_end = buf_ptr + buf->used_size;
		do {
			len = *buf_ptr++;
			if (len > 0) {
				/* The code is already generated. */
				SLJIT_MEMCPY(code_ptr, buf_ptr, len);
				code_ptr += len;
				buf_ptr += len;
			}
			else {
				switch (*buf_ptr) {
				case 0:
					label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					label->size = (sljit_uw)(code_ptr - code);
					label = label->next;
					break;
				case 1:
					jump->addr = (sljit_uw)code_ptr;
					if (!(jump->flags & SLJIT_REWRITABLE_JUMP))
						code_ptr = generate_near_jump_code(jump, code_ptr, code, executable_offset);
					else {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
						code_ptr = generate_far_jump_code(jump, code_ptr, executable_offset);
#else
						code_ptr = generate_far_jump_code(jump, code_ptr);
#endif
					}
					jump = jump->next;
					break;
				case 2:
					const_->addr = ((sljit_uw)code_ptr) - sizeof(sljit_sw);
					const_ = const_->next;
					break;
				default:
					SLJIT_ASSERT(*buf_ptr == 3);
					SLJIT_ASSERT(put_label->label);
					put_label->addr = (sljit_uw)code_ptr;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
					code_ptr = generate_put_label_code(put_label, code_ptr, (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code, executable_offset) + put_label->label->size);
#endif
					put_label = put_label->next;
					break;
				}
				buf_ptr++;
			}
		} while (buf_ptr < buf_end);
		SLJIT_ASSERT(buf_ptr == buf_end);
		buf = buf->next;
	} while (buf);

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(!put_label);
	SLJIT_ASSERT(code_ptr <= code + compiler->size);

	jump = compiler->jumps;
	while (jump) {
		if (jump->flags & (PATCH_MB | PATCH_MW)) {
			if (jump->flags & JUMP_LABEL)
				jump_addr = jump->u.label->addr;
			else
				jump_addr = jump->u.target;

			jump_addr -= jump->addr + (sljit_uw)executable_offset;

			if (jump->flags & PATCH_MB) {
				jump_addr -= sizeof(sljit_s8);
				SLJIT_ASSERT((sljit_sw)jump_addr >= -128 && (sljit_sw)jump_addr <= 127);
				*(sljit_u8*)jump->addr = U8(jump_addr);
			} else {
				jump_addr -= sizeof(sljit_s32);
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
				sljit_unaligned_store_sw((void*)jump->addr, (sljit_sw)jump_addr);
#else
				SLJIT_ASSERT((sljit_sw)jump_addr >= HALFWORD_MIN && (sljit_sw)jump_addr <= HALFWORD_MAX);
				sljit_unaligned_store_s32((void*)jump->addr, (sljit_s32)jump_addr);
#endif
			}
		}
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		else if (jump->flags & PATCH_MD) {
				SLJIT_ASSERT(jump->flags & JUMP_LABEL);
				sljit_unaligned_store_sw((void*)jump->addr, (sljit_sw)jump->u.label->addr);
		}
#endif

		jump = jump->next;
	}

	put_label = compiler->put_labels;
	while (put_label) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		sljit_unaligned_store_sw((void*)(put_label->addr - sizeof(sljit_sw)), (sljit_sw)put_label->label->addr);
#else
		if (put_label->flags & PATCH_MD) {
			SLJIT_ASSERT(put_label->label->addr > HALFWORD_MAX);
			sljit_unaligned_store_sw((void*)(put_label->addr - sizeof(sljit_sw)), (sljit_sw)put_label->label->addr);
		}
		else {
			SLJIT_ASSERT(put_label->label->addr <= HALFWORD_MAX);
			sljit_unaligned_store_s32((void*)(put_label->addr - sizeof(sljit_s32)), (sljit_s32)put_label->label->addr);
		}
#endif

		put_label = put_label->next;
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
		return SLJIT_IS_FPU_AVAILABLE;
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

	case SLJIT_HAS_ROT:
	case SLJIT_HAS_PREFETCH:
		return 1;

	case SLJIT_HAS_SSE2:
#if (defined SLJIT_DETECT_SSE2 && SLJIT_DETECT_SSE2)
		if (cpu_feature_list == 0)
			get_cpu_features();
		return (cpu_feature_list & CPU_FEATURE_SSE2) != 0;
#else /* !SLJIT_DETECT_SSE2 */
		return 1;
#endif /* SLJIT_DETECT_SSE2 */

	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	if (type < SLJIT_UNORDERED || type > SLJIT_ORDERED_LESS_EQUAL)
		return 0;

	switch (type) {
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
		return 0;
	}

	return 1;
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
			FAIL_IF(emit_load_imm64(compiler, (arg == TMP_REG1) ? TMP_REG2 : TMP_REG1, immw)); \
			inst = emit_x86_instruction(compiler, 1, (arg == TMP_REG1) ? TMP_REG2 : TMP_REG1, 0, arg, argw); \
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

static sljit_s32 emit_mov(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

#define EMIT_MOV(compiler, dst, dstw, src, srcw) \
	FAIL_IF(emit_mov(compiler, dst, dstw, src, srcw));

static SLJIT_INLINE sljit_s32 emit_sse2_store(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_sw dstw, sljit_s32 src);

static SLJIT_INLINE sljit_s32 emit_sse2_load(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_s32 src, sljit_sw srcw);

static sljit_s32 emit_cmp_binary(struct sljit_compiler *compiler,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

static SLJIT_INLINE sljit_s32 emit_endbranch(struct sljit_compiler *compiler)
{
#if (defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET)
	/* Emit endbr32/endbr64 when CET is enabled.  */
	sljit_u8 *inst;
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 4);
	FAIL_IF(!inst);
	INC_SIZE(4);
	*inst++ = 0xf3;
	*inst++ = 0x0f;
	*inst++ = 0x1e;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	*inst = 0xfb;
#else
	*inst = 0xfa;
#endif
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
	*inst++ = 0xf3;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	*inst++ = REX_W | (reg_map[reg] <= 7 ? 0 : REX_B);
#endif
	*inst++ = 0x0f;
	*inst++ = 0x1e;
	*inst = (0x3 << 6) | (0x1 << 3) | (reg_map[reg] & 0x7);
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
	*inst++ = 0xf3;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	*inst++ = REX_W | (reg_map[reg] <= 7 ? 0 : REX_B);
#endif
	*inst++ = 0x0f;
	*inst++ = 0xae;
	*inst = (0x3 << 6) | (0x5 << 3) | (reg_map[reg] & 0x7);
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
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	SLJIT_ASSERT(reg_map[TMP_REG1] == 5);

	/* Hand code unsupported "mov 0x0(%ebp),%ebp". */
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 3);
	FAIL_IF(!inst);
	INC_SIZE(3);
	*inst++ = 0x8b;
	*inst++ = 0x6d;
	*inst = 0;
#else /* !SLJIT_CONFIG_X86_32 */
	EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(TMP_REG1), 0);
#endif /* SLJIT_CONFIG_X86_32 */

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
	*inst++ = JMP_i8;
	*inst = size_before_rdssp_inst - compiler->size;

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
	if (src & SLJIT_IMM) {
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
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
		*inst = INT3;
		break;
	case SLJIT_NOP:
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
		*inst = NOP;
		break;
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
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
			FAIL_IF(!inst);
			INC_SIZE(1);
			*inst = CDQ;
#else
			if (compiler->mode32) {
				inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
				FAIL_IF(!inst);
				INC_SIZE(1);
				*inst = CDQ;
			} else {
				inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
				FAIL_IF(!inst);
				INC_SIZE(2);
				*inst++ = REX_W;
				*inst = CDQ;
			}
#endif
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
		FAIL_IF(!inst);
		INC_SIZE(2);
		*inst++ = GROUP_F7;
		*inst = MOD_REG | ((op >= SLJIT_DIVMOD_UW) ? reg_map[TMP_REG1] : reg_map[SLJIT_R1]);
#else
#ifdef _WIN64
		size = (!compiler->mode32 || op >= SLJIT_DIVMOD_UW) ? 3 : 2;
#else
		size = (!compiler->mode32) ? 3 : 2;
#endif
		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
#ifdef _WIN64
		if (!compiler->mode32)
			*inst++ = REX_W | ((op >= SLJIT_DIVMOD_UW) ? REX_B : 0);
		else if (op >= SLJIT_DIVMOD_UW)
			*inst++ = REX_B;
		*inst++ = GROUP_F7;
		*inst = MOD_REG | ((op >= SLJIT_DIVMOD_UW) ? reg_lmap[TMP_REG1] : reg_lmap[SLJIT_R1]);
#else
		if (!compiler->mode32)
			*inst++ = REX_W;
		*inst++ = GROUP_F7;
		*inst = MOD_REG | reg_map[SLJIT_R1];
#endif
#endif
		switch (op) {
		case SLJIT_LMUL_UW:
			*inst |= MUL;
			break;
		case SLJIT_LMUL_SW:
			*inst |= IMUL;
			break;
		case SLJIT_DIVMOD_UW:
		case SLJIT_DIV_UW:
			*inst |= DIV;
			break;
		case SLJIT_DIVMOD_SW:
		case SLJIT_DIV_SW:
			*inst |= IDIV;
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
	case SLJIT_ENDBR:
		return emit_endbranch(compiler);
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return skip_frames_before_return(compiler);
	}

	return SLJIT_SUCCESS;
}

#define ENCODE_PREFIX(prefix) \
	do { \
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1); \
		FAIL_IF(!inst); \
		INC_SIZE(1); \
		*inst = U8(prefix); \
	} while (0)

static sljit_s32 emit_mov_byte(struct sljit_compiler *compiler, sljit_s32 sign,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;
	sljit_s32 dst_r;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 work_r;
#endif

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
#endif

	if (src & SLJIT_IMM) {
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
	}
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	else if (FAST_IS_REG(src) && reg_map[src] >= 4) {
		/* src, dst are registers. */
		SLJIT_ASSERT(FAST_IS_REG(dst));
		if (reg_map[dst] < 4) {
			if (dst != src)
				EMIT_MOV(compiler, dst, 0, src, 0);
			inst = emit_x86_instruction(compiler, 2, dst, 0, dst, 0);
			FAIL_IF(!inst);
			*inst++ = GROUP_0F;
			*inst = sign ? MOVSX_r_rm8 : MOVZX_r_rm8;
		}
		else {
			if (dst != src)
				EMIT_MOV(compiler, dst, 0, src, 0);
			if (sign) {
				/* shl reg, 24 */
				inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_IMM, 24, dst, 0);
				FAIL_IF(!inst);
				*inst |= SHL;
				/* sar reg, 24 */
				inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_IMM, 24, dst, 0);
				FAIL_IF(!inst);
				*inst |= SAR;
			}
			else {
				inst = emit_x86_instruction(compiler, 1 | EX86_BIN_INS, SLJIT_IMM, 0xff, dst, 0);
				FAIL_IF(!inst);
				*(inst + 1) |= AND;
			}
		}
		return SLJIT_SUCCESS;
	}
#endif
	else {
		/* src can be memory addr or reg_map[src] < 4 on x86_32 architectures. */
		inst = emit_x86_instruction(compiler, 2, dst_r, 0, src, srcw);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = sign ? MOVSX_r_rm8 : MOVZX_r_rm8;
	}

	if (dst & SLJIT_MEM) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (dst_r == TMP_REG1) {
			/* Find a non-used register, whose reg_map[src] < 4. */
			if ((dst & REG_MASK) == SLJIT_R0) {
				if ((dst & OFFS_REG_MASK) == TO_OFFS_REG(SLJIT_R1))
					work_r = SLJIT_R2;
				else
					work_r = SLJIT_R1;
			}
			else {
				if ((dst & OFFS_REG_MASK) != TO_OFFS_REG(SLJIT_R0))
					work_r = SLJIT_R0;
				else if ((dst & REG_MASK) == SLJIT_R1)
					work_r = SLJIT_R2;
				else
					work_r = SLJIT_R1;
			}

			if (work_r == SLJIT_R0) {
				ENCODE_PREFIX(XCHG_EAX_r | reg_map[TMP_REG1]);
			}
			else {
				inst = emit_x86_instruction(compiler, 1, work_r, 0, dst_r, 0);
				FAIL_IF(!inst);
				*inst = XCHG_r_rm;
			}

			inst = emit_x86_instruction(compiler, 1, work_r, 0, dst, dstw);
			FAIL_IF(!inst);
			*inst = MOV_rm8_r8;

			if (work_r == SLJIT_R0) {
				ENCODE_PREFIX(XCHG_EAX_r | reg_map[TMP_REG1]);
			}
			else {
				inst = emit_x86_instruction(compiler, 1, work_r, 0, dst_r, 0);
				FAIL_IF(!inst);
				*inst = XCHG_r_rm;
			}
		}
		else {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, dst, dstw);
			FAIL_IF(!inst);
			*inst = MOV_rm8_r8;
		}
#else
		inst = emit_x86_instruction(compiler, 1 | EX86_REX | EX86_NO_REXW, dst_r, 0, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm8_r8;
#endif
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
	*inst++ = GROUP_0F;
	*inst++ = PREFETCH;

	if (op == SLJIT_PREFETCH_L1)
		*inst |= (1 << 3);
	else if (op == SLJIT_PREFETCH_L2)
		*inst |= (2 << 3);
	else if (op == SLJIT_PREFETCH_L3)
		*inst |= (3 << 3);

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

	if (src & SLJIT_IMM) {
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
	else {
		inst = emit_x86_instruction(compiler, 2, dst_r, 0, src, srcw);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = sign ? MOVSX_r_rm16 : MOVZX_r_rm16;
	}

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
		*inst++ = GROUP_F7;
		*inst |= opcode;
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(dst)) {
		EMIT_MOV(compiler, dst, 0, src, srcw);
		inst = emit_x86_instruction(compiler, 1, 0, 0, dst, 0);
		FAIL_IF(!inst);
		*inst++ = GROUP_F7;
		*inst |= opcode;
		return SLJIT_SUCCESS;
	}

	EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
	inst = emit_x86_instruction(compiler, 1, 0, 0, TMP_REG1, 0);
	FAIL_IF(!inst);
	*inst++ = GROUP_F7;
	*inst |= opcode;
	EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_not_with_flags(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;

	if (FAST_IS_REG(dst)) {
		EMIT_MOV(compiler, dst, 0, src, srcw);
		inst = emit_x86_instruction(compiler, 1, 0, 0, dst, 0);
		FAIL_IF(!inst);
		*inst++ = GROUP_F7;
		*inst |= NOT_rm;
		inst = emit_x86_instruction(compiler, 1, dst, 0, dst, 0);
		FAIL_IF(!inst);
		*inst = OR_r_rm;
		return SLJIT_SUCCESS;
	}

	EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
	inst = emit_x86_instruction(compiler, 1, 0, 0, TMP_REG1, 0);
	FAIL_IF(!inst);
	*inst++ = GROUP_F7;
	*inst |= NOT_rm;
	inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, TMP_REG1, 0);
	FAIL_IF(!inst);
	*inst = OR_r_rm;
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

	if (cpu_feature_list == 0)
		get_cpu_features();

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (is_clz ? (cpu_feature_list & CPU_FEATURE_LZCNT) : (cpu_feature_list & CPU_FEATURE_TZCNT)) {
		/* Group prefix added separately. */
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
		*inst++ = GROUP_F3;

		inst = emit_x86_instruction(compiler, 2, dst_r, 0, src, srcw);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = is_clz ? LZCNT_r_rm : TZCNT_r_rm;

		if (dst & SLJIT_MEM)
			EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
		return SLJIT_SUCCESS;
	}

	inst = emit_x86_instruction(compiler, 2, dst_r, 0, src, srcw);
	FAIL_IF(!inst);
	*inst++ = GROUP_0F;
	*inst = is_clz ? BSR_r_rm : BSF_r_rm;

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
		*inst++ = GROUP_0F;
		*inst = CMOVE_r_rm;
	}
	else
		FAIL_IF(sljit_emit_cmov_generic(compiler, SLJIT_EQUAL, dst_r, SLJIT_IMM, max));

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

		inst = emit_x86_instruction(compiler, 2, dst_r, 0, TMP_REG2, 0);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = CMOVE_r_rm;
	}
	else
		FAIL_IF(sljit_emit_cmov_generic(compiler, SLJIT_EQUAL, dst_r, SLJIT_IMM, max));

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

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 op_flags = GET_ALL_FLAGS(op);
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 dst_is_ereg = 0;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	CHECK_EXTRA_REGS(dst, dstw, dst_is_ereg = 1);
	CHECK_EXTRA_REGS(src, srcw, (void)0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op_flags & SLJIT_32;
#endif

	op = GET_OPCODE(op);

	if (op >= SLJIT_MOV && op <= SLJIT_MOV_P) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;
#endif

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
			else if (src & SLJIT_IMM) {
				if (op == SLJIT_MOV_U32)
					op = SLJIT_MOV_S32;
			}
		}
#endif

		if (src & SLJIT_IMM) {
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
#endif
			}
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
			if (SLJIT_UNLIKELY(dst_is_ereg))
				return emit_mov(compiler, dst, dstw, src, srcw);
#endif
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (SLJIT_UNLIKELY(dst_is_ereg) && (!(op == SLJIT_MOV || op == SLJIT_MOV_U32 || op == SLJIT_MOV_S32 || op == SLJIT_MOV_P) || (src & SLJIT_MEM))) {
			SLJIT_ASSERT(dst == SLJIT_MEM1(SLJIT_SP));
			dst = TMP_REG1;
		}
#endif

		switch (op) {
		case SLJIT_MOV:
		case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		case SLJIT_MOV_U32:
		case SLJIT_MOV_S32:
		case SLJIT_MOV32:
#endif
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
#endif
		}

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		if (SLJIT_UNLIKELY(dst_is_ereg) && dst == TMP_REG1)
			return emit_mov(compiler, SLJIT_MEM1(SLJIT_SP), dstw, TMP_REG1, 0);
#endif
		return SLJIT_SUCCESS;
	}

	switch (op) {
	case SLJIT_NOT:
		if (SLJIT_UNLIKELY(op_flags & SLJIT_SET_Z))
			return emit_not_with_flags(compiler, dst, dstw, src, srcw);
		return emit_unary(compiler, NOT_rm, dst, dstw, src, srcw);

	case SLJIT_CLZ:
	case SLJIT_CTZ:
		return emit_clz_ctz(compiler, (op == SLJIT_CLZ), dst, dstw, src, srcw);
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
		if (src2 & SLJIT_IMM) {
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
		if (src1 & SLJIT_IMM) {
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
		if (src2 & SLJIT_IMM) {
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
		if (src2 & SLJIT_IMM) {
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
		if (src2 & SLJIT_IMM) {
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
		if (src2 & SLJIT_IMM) {
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
		if (src2 & SLJIT_IMM) {
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
	if (dst_r == src1 && !(src2 & SLJIT_IMM)) {
		inst = emit_x86_instruction(compiler, 2, dst_r, 0, src2, src2w);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = IMUL_r_rm;
	}
	else if (dst_r == src2 && !(src1 & SLJIT_IMM)) {
		inst = emit_x86_instruction(compiler, 2, dst_r, 0, src1, src1w);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = IMUL_r_rm;
	}
	else if (src1 & SLJIT_IMM) {
		if (src2 & SLJIT_IMM) {
			EMIT_MOV(compiler, dst_r, 0, SLJIT_IMM, src2w);
			src2 = dst_r;
			src2w = 0;
		}

		if (src1w <= 127 && src1w >= -128) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i8;
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
			FAIL_IF(!inst);
			INC_SIZE(1);
			*inst = U8(src1w);
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
			inst = emit_x86_instruction(compiler, 2, dst_r, 0, TMP_REG2, 0);
			FAIL_IF(!inst);
			*inst++ = GROUP_0F;
			*inst = IMUL_r_rm;
		}
#endif
	}
	else if (src2 & SLJIT_IMM) {
		/* Note: src1 is NOT immediate. */

		if (src2w <= 127 && src2w >= -128) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src1, src1w);
			FAIL_IF(!inst);
			*inst = IMUL_r_rm_i8;
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
			FAIL_IF(!inst);
			INC_SIZE(1);
			*inst = U8(src2w);
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
		}
		else {
			if (dst_r != src1)
				EMIT_MOV(compiler, dst_r, 0, src1, src1w);
			FAIL_IF(emit_load_imm64(compiler, TMP_REG2, src2w));
			inst = emit_x86_instruction(compiler, 2, dst_r, 0, TMP_REG2, 0);
			FAIL_IF(!inst);
			*inst++ = GROUP_0F;
			*inst = IMUL_r_rm;
		}
#endif
	}
	else {
		/* Neither argument is immediate. */
		if (ADDRESSING_DEPENDS_ON(src2, dst_r))
			dst_r = TMP_REG1;
		EMIT_MOV(compiler, dst_r, 0, src1, src1w);
		inst = emit_x86_instruction(compiler, 2, dst_r, 0, src2, src2w);
		FAIL_IF(!inst);
		*inst++ = GROUP_0F;
		*inst = IMUL_r_rm;
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
		if ((src2 & SLJIT_IMM) && (compiler->mode32 || IS_HALFWORD(src2w))) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src1), (sljit_s32)src2w);
#else
		if (src2 & SLJIT_IMM) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src1), src2w);
#endif
			FAIL_IF(!inst);
			*inst = LEA_r_m;
			done = 1;
		}
	}
	else if (FAST_IS_REG(src2)) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if ((src1 & SLJIT_IMM) && (compiler->mode32 || IS_HALFWORD(src1w))) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, SLJIT_MEM1(src2), (sljit_s32)src1w);
#else
		if (src1 & SLJIT_IMM) {
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
	if (src1 == SLJIT_R0 && (src2 & SLJIT_IMM) && (src2w > 127 || src2w < -128) && (compiler->mode32 || IS_HALFWORD(src2w))) {
#else
	if (src1 == SLJIT_R0 && (src2 & SLJIT_IMM) && (src2w > 127 || src2w < -128)) {
#endif
		BINARY_EAX_IMM(CMP_EAX_i32, src2w);
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(src1)) {
		if (src2 & SLJIT_IMM) {
			BINARY_IMM(CMP, CMP_rm_r, src2w, src1, 0);
		}
		else {
			inst = emit_x86_instruction(compiler, 1, src1, 0, src2, src2w);
			FAIL_IF(!inst);
			*inst = CMP_r_rm;
		}
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(src2) && !(src1 & SLJIT_IMM)) {
		inst = emit_x86_instruction(compiler, 1, src2, 0, src1, src1w);
		FAIL_IF(!inst);
		*inst = CMP_rm_r;
		return SLJIT_SUCCESS;
	}

	if (src2 & SLJIT_IMM) {
		if (src1 & SLJIT_IMM) {
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
	if (src1 == SLJIT_R0 && (src2 & SLJIT_IMM) && (src2w > 127 || src2w < -128) && (compiler->mode32 || IS_HALFWORD(src2w))) {
#else
	if (src1 == SLJIT_R0 && (src2 & SLJIT_IMM) && (src2w > 127 || src2w < -128)) {
#endif
		BINARY_EAX_IMM(TEST_EAX_i32, src2w);
		return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (src2 == SLJIT_R0 && (src1 & SLJIT_IMM) && (src1w > 127 || src1w < -128) && (compiler->mode32 || IS_HALFWORD(src1w))) {
#else
	if (src2 == SLJIT_R0 && (src1 & SLJIT_IMM) && (src1w > 127 || src1w < -128)) {
#endif
		BINARY_EAX_IMM(TEST_EAX_i32, src1w);
		return SLJIT_SUCCESS;
	}

	if (!(src1 & SLJIT_IMM)) {
		if (src2 & SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
			if (IS_HALFWORD(src2w) || compiler->mode32) {
				inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, src2w, src1, src1w);
				FAIL_IF(!inst);
				*inst = GROUP_F7;
			}
			else {
				FAIL_IF(emit_load_imm64(compiler, TMP_REG1, src2w));
				inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, src1, src1w);
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

	if (!(src2 & SLJIT_IMM)) {
		if (src1 & SLJIT_IMM) {
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
	if (src2 & SLJIT_IMM) {
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

	if ((src2 & SLJIT_IMM) || (src2 == SLJIT_PREF_SHIFT_REG)) {
		if (dst == src1 && dstw == src1w) {
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, src2, src2w, dst, dstw);
			FAIL_IF(!inst);
			*inst |= mode;
			return SLJIT_SUCCESS;
		}
		if (dst == SLJIT_PREF_SHIFT_REG && src2 == SLJIT_PREF_SHIFT_REG) {
			EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
			FAIL_IF(!inst);
			*inst |= mode;
			EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
			return SLJIT_SUCCESS;
		}
		if (FAST_IS_REG(dst)) {
			EMIT_MOV(compiler, dst, 0, src1, src1w);
			inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, src2, src2w, dst, 0);
			FAIL_IF(!inst);
			*inst |= mode;
			return SLJIT_SUCCESS;
		}

		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, src2, src2w, TMP_REG1, 0);
		FAIL_IF(!inst);
		*inst |= mode;
		EMIT_MOV(compiler, dst, dstw, TMP_REG1, 0);
		return SLJIT_SUCCESS;
	}

	if (dst == SLJIT_PREF_SHIFT_REG) {
		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src2, src2w);
		inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
		FAIL_IF(!inst);
		*inst |= mode;
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
		*inst |= mode;
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
	*inst |= mode;

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
	if (src2 & SLJIT_IMM) {
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

	SLJIT_ASSERT(dst != TMP_REG1 || HAS_FLAGS(op));

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
			if ((src2 & SLJIT_IMM) && emit_lea_binary(compiler, dst, dstw, src1, src1w, SLJIT_IMM, -src2w) != SLJIT_ERR_UNSUPPORTED)
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

	if (opcode == SLJIT_SUB) {
		return emit_cmp_binary(compiler, src1, src1w, src2, src2w);
	}
	return emit_test_binary(compiler, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 restore_ecx = 0;
	sljit_s32 is_rotate, is_left;
	sljit_u8* inst;
	sljit_sw dstw = 0;
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 tmp2 = SLJIT_MEM1(SLJIT_SP);
#else /* !SLJIT_CONFIG_X86_32 */
	sljit_s32 tmp2 = TMP_REG2;
#endif /* SLJIT_CONFIG_X86_32 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, src_dst, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	CHECK_EXTRA_REGS(src1, src1w, (void)0);
	CHECK_EXTRA_REGS(src2, src2w, (void)0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = op & SLJIT_32;
#endif

	if (src2 & SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		src2w &= 0x1f;
#else /* !SLJIT_CONFIG_X86_32 */
		src2w &= (op & SLJIT_32) ? 0x1f : 0x3f;
#endif /* SLJIT_CONFIG_X86_32 */

		if (src2w == 0)
			return SLJIT_SUCCESS;
	}

	is_left = (GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_MSHL);

	is_rotate = (src_dst == src1);
	CHECK_EXTRA_REGS(src_dst, dstw, (void)0);

	if (is_rotate)
		return emit_shift(compiler, is_left ? ROL : ROR, src_dst, dstw, src1, src1w, src2, src2w);

	if ((src2 & SLJIT_IMM) || src2 == SLJIT_PREF_SHIFT_REG) {
		if (!FAST_IS_REG(src1)) {
			EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
			src1 = TMP_REG1;
		}
	} else if (FAST_IS_REG(src1)) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;
#endif
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_PREF_SHIFT_REG, 0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = op & SLJIT_32;
#endif
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src2, src2w);

		if (src1 == SLJIT_PREF_SHIFT_REG)
			src1 = TMP_REG1;

		if (src_dst == SLJIT_PREF_SHIFT_REG)
			src_dst = TMP_REG1;

		restore_ecx = 1;
	} else {
		EMIT_MOV(compiler, TMP_REG1, 0, src1, src1w);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = 0;
#endif
		EMIT_MOV(compiler, tmp2, 0, SLJIT_PREF_SHIFT_REG, 0);
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		compiler->mode32 = op & SLJIT_32;
#endif
		EMIT_MOV(compiler, SLJIT_PREF_SHIFT_REG, 0, src2, src2w);

		src1 = TMP_REG1;

		if (src_dst == SLJIT_PREF_SHIFT_REG) {
			src_dst = tmp2;
			SLJIT_ASSERT(dstw == 0);
		}

		restore_ecx = 2;
	}

	inst = emit_x86_instruction(compiler, 2, src1, 0, src_dst, dstw);
	FAIL_IF(!inst);
	inst[0] = GROUP_0F;

	if (src2 & SLJIT_IMM) {
		inst[1] = U8((is_left ? SHLD : SHRD) - 1);

		/* Immedate argument is added separately. */
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
		*inst = U8(src2w);
	} else
		inst[1] = U8(is_left ? SHLD : SHRD);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
#endif

	if (restore_ecx == 1)
		return emit_mov(compiler, SLJIT_PREF_SHIFT_REG, 0, TMP_REG1, 0);
	if (restore_ecx == 2)
		return emit_mov(compiler, SLJIT_PREF_SHIFT_REG, 0, tmp2, 0);

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

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(reg));
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	if (reg >= SLJIT_R3 && reg <= SLJIT_R8)
		return -1;
#endif
	return reg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_float_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_float_register_index(reg));
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	return reg;
#else
	return freg_map[reg];
#endif
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

static sljit_s32 emit_sse2(struct sljit_compiler *compiler, sljit_u8 opcode,
	sljit_s32 single, sljit_s32 xmm1, sljit_s32 xmm2, sljit_sw xmm2w)
{
	sljit_u8 *inst;

	inst = emit_x86_instruction(compiler, 2 | (single ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2, xmm1, 0, xmm2, xmm2w);
	FAIL_IF(!inst);
	*inst++ = GROUP_0F;
	*inst = opcode;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_sse2_logic(struct sljit_compiler *compiler, sljit_u8 opcode,
	sljit_s32 pref66, sljit_s32 xmm1, sljit_s32 xmm2, sljit_sw xmm2w)
{
	sljit_u8 *inst;

	inst = emit_x86_instruction(compiler, 2 | (pref66 ? EX86_PREF_66 : 0) | EX86_SSE2, xmm1, 0, xmm2, xmm2w);
	FAIL_IF(!inst);
	*inst++ = GROUP_0F;
	*inst = opcode;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_sse2_load(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_s32 src, sljit_sw srcw)
{
	return emit_sse2(compiler, MOVSD_x_xm, single, dst, src, srcw);
}

static SLJIT_INLINE sljit_s32 emit_sse2_store(struct sljit_compiler *compiler,
	sljit_s32 single, sljit_s32 dst, sljit_sw dstw, sljit_s32 src)
{
	return emit_sse2(compiler, MOVSD_xm_x, single, src, dst, dstw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	sljit_u8 *inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (GET_OPCODE(op) == SLJIT_CONV_SW_FROM_F64)
		compiler->mode32 = 0;
#endif

	inst = emit_x86_instruction(compiler, 2 | ((op & SLJIT_32) ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2_OP2, dst_r, 0, src, srcw);
	FAIL_IF(!inst);
	*inst++ = GROUP_0F;
	*inst = CVTTSD2SI_r_xm;

	if (dst & SLJIT_MEM)
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG;
	sljit_u8 *inst;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_SW)
		compiler->mode32 = 0;
#endif

	if (src & SLJIT_IMM) {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;
#endif
		EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
		src = TMP_REG1;
		srcw = 0;
	}

	inst = emit_x86_instruction(compiler, 2 | ((op & SLJIT_32) ? EX86_PREF_F3 : EX86_PREF_F2) | EX86_SSE2_OP1, dst_r, 0, src, srcw);
	FAIL_IF(!inst);
	*inst++ = GROUP_0F;
	*inst = CVTSI2SD_x_rm;

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
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_ORDERED_LESS_EQUAL:
		if (!FAST_IS_REG(src2)) {
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src2, src2w));
			src2 = TMP_FREG;
		}

		return emit_sse2_logic(compiler, UCOMISD_x_xm, !(op & SLJIT_32), src2, src1, src1w);
	}

	if (!FAST_IS_REG(src1)) {
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
		src1 = TMP_FREG;
	}

	return emit_sse2_logic(compiler, UCOMISD_x_xm, !(op & SLJIT_32), src1, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

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
			FAIL_IF(emit_sse2_logic(compiler, UNPCKLPD_x_xm, op & SLJIT_32, src, src, 0));
		}
		else {
			FAIL_IF(emit_sse2_load(compiler, !(op & SLJIT_32), TMP_FREG, src, srcw));
			src = TMP_FREG;
		}

		FAIL_IF(emit_sse2_logic(compiler, CVTPD2PS_x_xm, op & SLJIT_32, dst_r, src, 0));
		if (dst_r == TMP_FREG)
			return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(dst)) {
		dst_r = dst;
		if (dst != src)
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, dst_r, src, srcw));
	}
	else {
		dst_r = TMP_FREG;
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, dst_r, src, srcw));
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_NEG_F64:
		FAIL_IF(emit_sse2_logic(compiler, XORPD_x_xm, 1, dst_r, SLJIT_MEM0(), (sljit_sw)(op & SLJIT_32 ? sse2_buffer : sse2_buffer + 8)));
		break;

	case SLJIT_ABS_F64:
		FAIL_IF(emit_sse2_logic(compiler, ANDPD_x_xm, 1, dst_r, SLJIT_MEM0(), (sljit_sw)(op & SLJIT_32 ? sse2_buffer + 4 : sse2_buffer + 12)));
		break;
	}

	if (dst_r == TMP_FREG)
		return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
	return SLJIT_SUCCESS;
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
		else if (dst == src2 && (op == SLJIT_ADD_F64 || op == SLJIT_MUL_F64)) {
			/* Swap arguments. */
			src2 = src1;
			src2w = src1w;
		}
		else if (dst != src2)
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, dst_r, src1, src1w));
		else {
			dst_r = TMP_FREG;
			FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
		}
	}
	else {
		dst_r = TMP_FREG;
		FAIL_IF(emit_sse2_load(compiler, op & SLJIT_32, TMP_FREG, src1, src1w));
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(emit_sse2(compiler, ADDSD_x_xm, op & SLJIT_32, dst_r, src2, src2w));
		break;

	case SLJIT_SUB_F64:
		FAIL_IF(emit_sse2(compiler, SUBSD_x_xm, op & SLJIT_32, dst_r, src2, src2w));
		break;

	case SLJIT_MUL_F64:
		FAIL_IF(emit_sse2(compiler, MULSD_x_xm, op & SLJIT_32, dst_r, src2, src2w));
		break;

	case SLJIT_DIV_F64:
		FAIL_IF(emit_sse2(compiler, DIVSD_x_xm, op & SLJIT_32, dst_r, src2, src2w));
		break;
	}

	if (dst_r == TMP_FREG)
		return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
	return SLJIT_SUCCESS;
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

	inst = (sljit_u8*)ensure_buf(compiler, 2);
	PTR_FAIL_IF(!inst);

	*inst++ = 0;
	*inst++ = 0;

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

	/* Worst case size. */
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	compiler->size += (type >= SLJIT_JUMP) ? 5 : 6;
#else
	compiler->size += (type >= SLJIT_JUMP) ? (10 + 3) : (2 + 10 + 3);
#endif

	inst = (sljit_u8*)ensure_buf(compiler, 2);
	PTR_FAIL_IF_NULL(inst);

	*inst++ = 0;
	*inst++ = 1;
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

		/* Worst case size. */
#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
		compiler->size += 5;
#else
		compiler->size += 10 + 3;
#endif

		inst = (sljit_u8*)ensure_buf(compiler, 2);
		FAIL_IF_NULL(inst);

		*inst++ = 0;
		*inst++ = 1;
	}
	else {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		/* REX_W is not necessary (src is not immediate). */
		compiler->mode32 = 1;
#endif
		inst = emit_x86_instruction(compiler, 1, 0, 0, src, srcw);
		FAIL_IF(!inst);
		*inst++ = GROUP_FF;
		*inst = U8(*inst | ((type >= SLJIT_FAST_CALL) ? CALL_rm : JMP_rm));
	}
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_u8 *inst;
	sljit_u8 cond_set = 0;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 reg;
#endif
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
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 4 + 3);
		FAIL_IF(!inst);
		INC_SIZE(4 + 3);
		/* Set low register to conditional flag. */
		*inst++ = (reg_map[TMP_REG1] <= 7) ? REX : REX_B;
		*inst++ = GROUP_0F;
		*inst++ = cond_set;
		*inst++ = MOD_REG | reg_lmap[TMP_REG1];
		*inst++ = U8(REX | (reg_map[TMP_REG1] <= 7 ? 0 : REX_R) | (reg_map[dst] <= 7 ? 0 : REX_B));
		*inst++ = OR_rm8_r8;
		*inst++ = U8(MOD_REG | (reg_lmap[TMP_REG1] << 3) | reg_lmap[dst]);
		return SLJIT_SUCCESS;
	}

	reg = (GET_OPCODE(op) < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG1;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 4 + 4);
	FAIL_IF(!inst);
	INC_SIZE(4 + 4);
	/* Set low register to conditional flag. */
	*inst++ = (reg_map[reg] <= 7) ? REX : REX_B;
	*inst++ = GROUP_0F;
	*inst++ = cond_set;
	*inst++ = MOD_REG | reg_lmap[reg];
	*inst++ = REX_W | (reg_map[reg] <= 7 ? 0 : (REX_B | REX_R));
	/* The movzx instruction does not affect flags. */
	*inst++ = GROUP_0F;
	*inst++ = MOVZX_r_rm8;
	*inst = U8(MOD_REG | (reg_lmap[reg] << 3) | reg_lmap[reg]);

	if (reg != TMP_REG1)
		return SLJIT_SUCCESS;

	if (GET_OPCODE(op) < SLJIT_ADD) {
		compiler->mode32 = GET_OPCODE(op) != SLJIT_MOV;
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, dst_save, dstw_save, dst_save, dstw_save, TMP_REG1, 0);

#else
	/* The SLJIT_CONFIG_X86_32 code path starts here. */
	if (GET_OPCODE(op) < SLJIT_ADD && FAST_IS_REG(dst)) {
		if (reg_map[dst] <= 4) {
			/* Low byte is accessible. */
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 3 + 3);
			FAIL_IF(!inst);
			INC_SIZE(3 + 3);
			/* Set low byte to conditional flag. */
			*inst++ = GROUP_0F;
			*inst++ = cond_set;
			*inst++ = U8(MOD_REG | reg_map[dst]);

			*inst++ = GROUP_0F;
			*inst++ = MOVZX_r_rm8;
			*inst = U8(MOD_REG | (reg_map[dst] << 3) | reg_map[dst]);
			return SLJIT_SUCCESS;
		}

		/* Low byte is not accessible. */
		if (cpu_feature_list == 0)
			get_cpu_features();

		if (cpu_feature_list & CPU_FEATURE_CMOV) {
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, 1);
			/* a xor reg, reg operation would overwrite the flags. */
			EMIT_MOV(compiler, dst, 0, SLJIT_IMM, 0);

			inst = (sljit_u8*)ensure_buf(compiler, 1 + 3);
			FAIL_IF(!inst);
			INC_SIZE(3);

			*inst++ = GROUP_0F;
			/* cmovcc = setcc - 0x50. */
			*inst++ = U8(cond_set - 0x50);
			*inst++ = U8(MOD_REG | (reg_map[dst] << 3) | reg_map[TMP_REG1]);
			return SLJIT_SUCCESS;
		}

		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + 3 + 3 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1 + 3 + 3 + 1);
		*inst++ = U8(XCHG_EAX_r | reg_map[TMP_REG1]);
		/* Set al to conditional flag. */
		*inst++ = GROUP_0F;
		*inst++ = cond_set;
		*inst++ = MOD_REG | 0 /* eax */;

		*inst++ = GROUP_0F;
		*inst++ = MOVZX_r_rm8;
		*inst++ = U8(MOD_REG | (reg_map[dst] << 3) | 0 /* eax */);
		*inst++ = U8(XCHG_EAX_r | reg_map[TMP_REG1]);
		return SLJIT_SUCCESS;
	}

	if (GET_OPCODE(op) == SLJIT_OR && !GET_ALL_FLAGS(op) && FAST_IS_REG(dst) && reg_map[dst] <= 4) {
		SLJIT_ASSERT(reg_map[SLJIT_R0] == 0);

		if (dst != SLJIT_R0) {
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + 3 + 2 + 1);
			FAIL_IF(!inst);
			INC_SIZE(1 + 3 + 2 + 1);
			/* Set low register to conditional flag. */
			*inst++ = U8(XCHG_EAX_r | reg_map[TMP_REG1]);
			*inst++ = GROUP_0F;
			*inst++ = cond_set;
			*inst++ = MOD_REG | 0 /* eax */;
			*inst++ = OR_rm8_r8;
			*inst++ = MOD_REG | (0 /* eax */ << 3) | reg_map[dst];
			*inst++ = U8(XCHG_EAX_r | reg_map[TMP_REG1]);
		}
		else {
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 2 + 3 + 2 + 2);
			FAIL_IF(!inst);
			INC_SIZE(2 + 3 + 2 + 2);
			/* Set low register to conditional flag. */
			*inst++ = XCHG_r_rm;
			*inst++ = U8(MOD_REG | (1 /* ecx */ << 3) | reg_map[TMP_REG1]);
			*inst++ = GROUP_0F;
			*inst++ = cond_set;
			*inst++ = MOD_REG | 1 /* ecx */;
			*inst++ = OR_rm8_r8;
			*inst++ = MOD_REG | (1 /* ecx */ << 3) | 0 /* eax */;
			*inst++ = XCHG_r_rm;
			*inst++ = U8(MOD_REG | (1 /* ecx */ << 3) | reg_map[TMP_REG1]);
		}
		return SLJIT_SUCCESS;
	}

	/* Set TMP_REG1 to the bit. */
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + 3 + 3 + 1);
	FAIL_IF(!inst);
	INC_SIZE(1 + 3 + 3 + 1);
	*inst++ = U8(XCHG_EAX_r | reg_map[TMP_REG1]);
	/* Set al to conditional flag. */
	*inst++ = GROUP_0F;
	*inst++ = cond_set;
	*inst++ = MOD_REG | 0 /* eax */;

	*inst++ = GROUP_0F;
	*inst++ = MOVZX_r_rm8;
	*inst++ = MOD_REG | (0 << 3) /* eax */ | 0 /* eax */;

	*inst++ = U8(XCHG_EAX_r | reg_map[TMP_REG1]);

	if (GET_OPCODE(op) < SLJIT_ADD)
		return emit_mov(compiler, dst, dstw, TMP_REG1, 0);

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, dst_save, dstw_save, dst_save, dstw_save, TMP_REG1, 0);
#endif /* SLJIT_CONFIG_X86_64 */
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_cmov(compiler, type, dst_reg, src, srcw));

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	type &= ~SLJIT_32;

	if (!sljit_has_cpu_feature(SLJIT_HAS_CMOV) || (dst_reg >= SLJIT_R3 && dst_reg <= SLJIT_S3))
		return sljit_emit_cmov_generic(compiler, type, dst_reg, src, srcw);
#else
	if (!sljit_has_cpu_feature(SLJIT_HAS_CMOV))
		return sljit_emit_cmov_generic(compiler, type, dst_reg, src, srcw);
#endif

	/* ADJUST_LOCAL_OFFSET is not needed. */
	CHECK_EXTRA_REGS(src, srcw, (void)0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = type & SLJIT_32;
	type &= ~SLJIT_32;
#endif

	if (SLJIT_UNLIKELY(src & SLJIT_IMM)) {
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, srcw);
		src = TMP_REG1;
		srcw = 0;
	}

	inst = emit_x86_instruction(compiler, 2, dst_reg, 0, src, srcw);
	FAIL_IF(!inst);
	*inst++ = GROUP_0F;
	*inst = U8(get_jump_code((sljit_uw)type) - 0x40);
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

	inst = (sljit_u8*)ensure_buf(compiler, 2);
	PTR_FAIL_IF(!inst);

	*inst++ = 0;
	*inst++ = 2;

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (dst & SLJIT_MEM)
		if (emit_mov(compiler, dst, dstw, TMP_REG1, 0))
			return NULL;
#endif

	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_put_label *put_label;
	sljit_u8 *inst;
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 reg;
	sljit_uw start_size;
#endif

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_put_label(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

	put_label = (struct sljit_put_label*)ensure_abuf(compiler, sizeof(struct sljit_put_label));
	PTR_FAIL_IF(!put_label);
	set_put_label(put_label, compiler, 0);

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	compiler->mode32 = 0;
	reg = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (emit_load_imm64(compiler, reg, 0))
		return NULL;
#else
	if (emit_mov(compiler, dst, dstw, SLJIT_IMM, 0))
		return NULL;
#endif

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	if (dst & SLJIT_MEM) {
		start_size = compiler->size;
		if (emit_mov(compiler, dst, dstw, TMP_REG1, 0))
			return NULL;
		put_label->flags = compiler->size - start_size;
	}
#endif

	inst = (sljit_u8*)ensure_buf(compiler, 2);
	PTR_FAIL_IF(!inst);

	*inst++ = 0;
	*inst++ = 3;

	return put_label;
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
