/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright 2009-2012 Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
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
	return "PowerPC" SLJIT_CPUINFO;
}

/* Length of an instruction word.
   Both for ppc-32 and ppc-64. */
typedef sljit_u32 sljit_ins;

#if ((defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) && (defined _AIX)) \
	|| (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define SLJIT_PPC_STACK_FRAME_V2 1
#endif

#ifdef _AIX
#include <sys/cache.h>
#endif

#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define SLJIT_PASS_ENTRY_ADDR_TO_CALL 1
#endif

#if (defined SLJIT_CACHE_FLUSH_OWN_IMPL && SLJIT_CACHE_FLUSH_OWN_IMPL)

static void ppc_cache_flush(sljit_ins *from, sljit_ins *to)
{
#ifdef _AIX
	_sync_cache_range((caddr_t)from, (int)((size_t)to - (size_t)from));
#elif defined(__GNUC__) || (defined(__IBM_GCC_ASM) && __IBM_GCC_ASM)
#	if defined(_ARCH_PWR) || defined(_ARCH_PWR2)
	/* Cache flush for POWER architecture. */
	while (from < to) {
		__asm__ volatile (
			"clf 0, %0\n"
			"dcs\n"
			: : "r"(from)
		);
		from++;
	}
	__asm__ volatile ( "ics" );
#	elif defined(_ARCH_COM) && !defined(_ARCH_PPC)
#	error "Cache flush is not implemented for PowerPC/POWER common mode."
#	else
	/* Cache flush for PowerPC architecture. */
	while (from < to) {
		__asm__ volatile (
			"dcbf 0, %0\n"
			"sync\n"
			"icbi 0, %0\n"
			: : "r"(from)
		);
		from++;
	}
	__asm__ volatile ( "isync" );
#	endif
#	ifdef __xlc__
#	warning "This file may fail to compile if -qfuncsect is used"
#	endif
#elif defined(__xlc__)
#error "Please enable GCC syntax for inline assembly statements with -qasm=gcc"
#else
#error "This platform requires a cache flush implementation."
#endif /* _AIX */
}

#endif /* (defined SLJIT_CACHE_FLUSH_OWN_IMPL && SLJIT_CACHE_FLUSH_OWN_IMPL) */

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3	(SLJIT_NUMBER_OF_REGISTERS + 4)
#define TMP_ZERO	(SLJIT_NUMBER_OF_REGISTERS + 5)

#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
#define TMP_CALL_REG	(SLJIT_NUMBER_OF_REGISTERS + 6)
#else
#define TMP_CALL_REG	TMP_REG2
#endif

#define TMP_FREG1	(0)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 7] = {
	0, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 8, 9, 10, 31, 12
};

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */
#define D(d)		(reg_map[d] << 21)
#define S(s)		(reg_map[s] << 21)
#define A(a)		(reg_map[a] << 16)
#define B(b)		(reg_map[b] << 11)
#define C(c)		(reg_map[c] << 6)
#define FD(fd)		((fd) << 21)
#define FS(fs)		((fs) << 21)
#define FA(fa)		((fa) << 16)
#define FB(fb)		((fb) << 11)
#define FC(fc)		((fc) << 6)
#define IMM(imm)	((imm) & 0xffff)
#define CRD(d)		((d) << 21)

/* Instruction bit sections.
   OE and Rc flag (see ALT_SET_FLAGS). */
#define OERC(flags)	(((flags & ALT_SET_FLAGS) >> 10) | (flags & ALT_SET_FLAGS))
/* Rc flag (see ALT_SET_FLAGS). */
#define RC(flags)	((flags & ALT_SET_FLAGS) >> 10)
#define HI(opcode)	((opcode) << 26)
#define LO(opcode)	((opcode) << 1)

#define ADD		(HI(31) | LO(266))
#define ADDC		(HI(31) | LO(10))
#define ADDE		(HI(31) | LO(138))
#define ADDI		(HI(14))
#define ADDIC		(HI(13))
#define ADDIS		(HI(15))
#define ADDME		(HI(31) | LO(234))
#define AND		(HI(31) | LO(28))
#define ANDI		(HI(28))
#define ANDIS		(HI(29))
#define Bx		(HI(18))
#define BCx		(HI(16))
#define BCCTR		(HI(19) | LO(528) | (3 << 11))
#define BLR		(HI(19) | LO(16) | (0x14 << 21))
#define CNTLZD		(HI(31) | LO(58))
#define CNTLZW		(HI(31) | LO(26))
#define CMP		(HI(31) | LO(0))
#define CMPI		(HI(11))
#define CMPL		(HI(31) | LO(32))
#define CMPLI		(HI(10))
#define CROR		(HI(19) | LO(449))
#define DIVD		(HI(31) | LO(489))
#define DIVDU		(HI(31) | LO(457))
#define DIVW		(HI(31) | LO(491))
#define DIVWU		(HI(31) | LO(459))
#define EXTSB		(HI(31) | LO(954))
#define EXTSH		(HI(31) | LO(922))
#define EXTSW		(HI(31) | LO(986))
#define FABS		(HI(63) | LO(264))
#define FADD		(HI(63) | LO(21))
#define FADDS		(HI(59) | LO(21))
#define FCFID		(HI(63) | LO(846))
#define FCMPU		(HI(63) | LO(0))
#define FCTIDZ		(HI(63) | LO(815))
#define FCTIWZ		(HI(63) | LO(15))
#define FDIV		(HI(63) | LO(18))
#define FDIVS		(HI(59) | LO(18))
#define FMR		(HI(63) | LO(72))
#define FMUL		(HI(63) | LO(25))
#define FMULS		(HI(59) | LO(25))
#define FNEG		(HI(63) | LO(40))
#define FRSP		(HI(63) | LO(12))
#define FSUB		(HI(63) | LO(20))
#define FSUBS		(HI(59) | LO(20))
#define LD		(HI(58) | 0)
#define LWZ		(HI(32))
#define MFCR		(HI(31) | LO(19))
#define MFLR		(HI(31) | LO(339) | 0x80000)
#define MFXER		(HI(31) | LO(339) | 0x10000)
#define MTCTR		(HI(31) | LO(467) | 0x90000)
#define MTLR		(HI(31) | LO(467) | 0x80000)
#define MTXER		(HI(31) | LO(467) | 0x10000)
#define MULHD		(HI(31) | LO(73))
#define MULHDU		(HI(31) | LO(9))
#define MULHW		(HI(31) | LO(75))
#define MULHWU		(HI(31) | LO(11))
#define MULLD		(HI(31) | LO(233))
#define MULLI		(HI(7))
#define MULLW		(HI(31) | LO(235))
#define NEG		(HI(31) | LO(104))
#define NOP		(HI(24))
#define NOR		(HI(31) | LO(124))
#define OR		(HI(31) | LO(444))
#define ORI		(HI(24))
#define ORIS		(HI(25))
#define RLDICL		(HI(30))
#define RLWINM		(HI(21))
#define SLD		(HI(31) | LO(27))
#define SLW		(HI(31) | LO(24))
#define SRAD		(HI(31) | LO(794))
#define SRADI		(HI(31) | LO(413 << 1))
#define SRAW		(HI(31) | LO(792))
#define SRAWI		(HI(31) | LO(824))
#define SRD		(HI(31) | LO(539))
#define SRW		(HI(31) | LO(536))
#define STD		(HI(62) | 0)
#define STDU		(HI(62) | 1)
#define STDUX		(HI(31) | LO(181))
#define STFIWX		(HI(31) | LO(983))
#define STW		(HI(36))
#define STWU		(HI(37))
#define STWUX		(HI(31) | LO(183))
#define SUBF		(HI(31) | LO(40))
#define SUBFC		(HI(31) | LO(8))
#define SUBFE		(HI(31) | LO(136))
#define SUBFIC		(HI(8))
#define XOR		(HI(31) | LO(316))
#define XORI		(HI(26))
#define XORIS		(HI(27))

#define SIMM_MAX	(0x7fff)
#define SIMM_MIN	(-0x8000)
#define UIMM_MAX	(0xffff)

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_function_context(void** func_ptr, struct sljit_function_context* context, sljit_sw addr, void* func)
{
	sljit_sw* ptrs;
	if (func_ptr)
		*func_ptr = (void*)context;
	ptrs = (sljit_sw*)func;
	context->addr = addr ? addr : ptrs[0];
	context->r2 = ptrs[1];
	context->r11 = ptrs[2];
}
#endif

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 detect_jump_type(struct sljit_jump *jump, sljit_ins *code_ptr, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_sw extra_jump_flags;

#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL) && (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	if (jump->flags & (SLJIT_REWRITABLE_JUMP | IS_CALL))
		return 0;
#else
	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		return 0;
#endif

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;
	}

#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL) && (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (jump->flags & IS_CALL)
		goto keep_address;
#endif

	diff = ((sljit_sw)target_addr - (sljit_sw)(code_ptr) - executable_offset) & ~0x3l;

	extra_jump_flags = 0;
	if (jump->flags & IS_COND) {
		if (diff <= 0x7fff && diff >= -0x8000) {
			jump->flags |= PATCH_B;
			return 1;
		}
		if (target_addr <= 0xffff) {
			jump->flags |= PATCH_B | PATCH_ABS_B;
			return 1;
		}
		extra_jump_flags = REMOVE_COND;

		diff -= sizeof(sljit_ins);
	}

	if (diff <= 0x01ffffff && diff >= -0x02000000) {
		jump->flags |= PATCH_B | extra_jump_flags;
		return 1;
	}

	if (target_addr <= 0x03ffffff) {
		jump->flags |= PATCH_B | PATCH_ABS_B | extra_jump_flags;
		return 1;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
keep_address:
#endif
	if (target_addr <= 0x7fffffff) {
		jump->flags |= PATCH_ABS32;
		return 1;
	}

	if (target_addr <= 0x7fffffffffffl) {
		jump->flags |= PATCH_ABS48;
		return 1;
	}
#endif

	return 0;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw word_count;
	sljit_sw executable_offset;
	sljit_uw addr;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	compiler->size += (compiler->size & 0x1) + (sizeof(struct sljit_function_context) / sizeof(sljit_ins));
#else
	compiler->size += (sizeof(struct sljit_function_context) / sizeof(sljit_ins));
#endif
#endif
	code = (sljit_ins*)SLJIT_MALLOC_EXEC(compiler->size * sizeof(sljit_ins));
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

	code_ptr = code;
	word_count = 0;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;

	do {
		buf_ptr = (sljit_ins*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			*code_ptr = *buf_ptr++;
			SLJIT_ASSERT(!label || label->size >= word_count);
			SLJIT_ASSERT(!jump || jump->addr >= word_count);
			SLJIT_ASSERT(!const_ || const_->addr >= word_count);
			/* These structures are ordered by their address. */
			if (label && label->size == word_count) {
				/* Just recording the address. */
				label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
				label->size = code_ptr - code;
				label = label->next;
			}
			if (jump && jump->addr == word_count) {
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
				jump->addr = (sljit_uw)(code_ptr - 3);
#else
				jump->addr = (sljit_uw)(code_ptr - 6);
#endif
				if (detect_jump_type(jump, code_ptr, code, executable_offset)) {
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
					code_ptr[-3] = code_ptr[0];
					code_ptr -= 3;
#else
					if (jump->flags & PATCH_ABS32) {
						code_ptr -= 3;
						code_ptr[-1] = code_ptr[2];
						code_ptr[0] = code_ptr[3];
					}
					else if (jump->flags & PATCH_ABS48) {
						code_ptr--;
						code_ptr[-1] = code_ptr[0];
						code_ptr[0] = code_ptr[1];
						/* rldicr rX,rX,32,31 -> rX,rX,16,47 */
						SLJIT_ASSERT((code_ptr[-3] & 0xfc00ffff) == 0x780007c6);
						code_ptr[-3] ^= 0x8422;
						/* oris -> ori */
						code_ptr[-2] ^= 0x4000000;
					}
					else {
						code_ptr[-6] = code_ptr[0];
						code_ptr -= 6;
					}
#endif
					if (jump->flags & REMOVE_COND) {
						code_ptr[0] = BCx | (2 << 2) | ((code_ptr[0] ^ (8 << 21)) & 0x03ff0001);
						code_ptr++;
						jump->addr += sizeof(sljit_ins);
						code_ptr[0] = Bx;
						jump->flags -= IS_COND;
					}
				}
				jump = jump->next;
			}
			if (const_ && const_->addr == word_count) {
				const_->addr = (sljit_uw)code_ptr;
				const_ = const_->next;
			}
			code_ptr ++;
			word_count ++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
		label->size = code_ptr - code;
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size - (sizeof(struct sljit_function_context) / sizeof(sljit_ins)));
#else
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);
#endif

	jump = compiler->jumps;
	while (jump) {
		do {
			addr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
			buf_ptr = (sljit_ins *)jump->addr;

			if (jump->flags & PATCH_B) {
				if (jump->flags & IS_COND) {
					if (!(jump->flags & PATCH_ABS_B)) {
						addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset);
						SLJIT_ASSERT((sljit_sw)addr <= 0x7fff && (sljit_sw)addr >= -0x8000);
						*buf_ptr = BCx | (addr & 0xfffc) | ((*buf_ptr) & 0x03ff0001);
					}
					else {
						SLJIT_ASSERT(addr <= 0xffff);
						*buf_ptr = BCx | (addr & 0xfffc) | 0x2 | ((*buf_ptr) & 0x03ff0001);
					}
				}
				else {
					if (!(jump->flags & PATCH_ABS_B)) {
						addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset);
						SLJIT_ASSERT((sljit_sw)addr <= 0x01ffffff && (sljit_sw)addr >= -0x02000000);
						*buf_ptr = Bx | (addr & 0x03fffffc) | ((*buf_ptr) & 0x1);
					}
					else {
						SLJIT_ASSERT(addr <= 0x03ffffff);
						*buf_ptr = Bx | (addr & 0x03fffffc) | 0x2 | ((*buf_ptr) & 0x1);
					}
				}
				break;
			}

			/* Set the fields of immediate loads. */
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
			buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 16) & 0xffff);
			buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | (addr & 0xffff);
#else
			if (jump->flags & PATCH_ABS32) {
				SLJIT_ASSERT(addr <= 0x7fffffff);
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 16) & 0xffff);
				buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | (addr & 0xffff);
				break;
			}
			if (jump->flags & PATCH_ABS48) {
				SLJIT_ASSERT(addr <= 0x7fffffffffff);
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 32) & 0xffff);
				buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | ((addr >> 16) & 0xffff);
				buf_ptr[3] = (buf_ptr[3] & 0xffff0000) | (addr & 0xffff);
				break;
			}
			buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 48) & 0xffff);
			buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | ((addr >> 32) & 0xffff);
			buf_ptr[3] = (buf_ptr[3] & 0xffff0000) | ((addr >> 16) & 0xffff);
			buf_ptr[4] = (buf_ptr[4] & 0xffff0000) | (addr & 0xffff);
#endif
		} while (0);
		jump = jump->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (code_ptr - code) * sizeof(sljit_ins);

	code = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (((sljit_sw)code_ptr) & 0x4)
		code_ptr++;
#endif
	sljit_set_function_context(NULL, (struct sljit_function_context*)code_ptr, (sljit_sw)code, (void*)sljit_generate_code);
#endif

	code_ptr = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

	SLJIT_CACHE_FLUSH(code, code_ptr);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
	return code_ptr;
#else
	return code;
#endif
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

/* inp_flags: */

/* Creates an index in data_transfer_insts array. */
#define LOAD_DATA	0x01
#define INDEXED		0x02
#define WRITE_BACK	0x04
#define WORD_DATA	0x00
#define BYTE_DATA	0x08
#define HALF_DATA	0x10
#define INT_DATA	0x18
#define SIGNED_DATA	0x20
/* Separates integer and floating point registers */
#define GPR_REG		0x3f
#define DOUBLE_DATA	0x40

#define MEM_MASK	0x7f

/* Other inp_flags. */

#define ARG_TEST	0x000100
/* Integer opertion and set flags -> requires exts on 64 bit systems. */
#define ALT_SIGN_EXT	0x000200
/* This flag affects the RC() and OERC() macros. */
#define ALT_SET_FLAGS	0x000400
#define ALT_KEEP_CACHE	0x000800
#define ALT_FORM1	0x010000
#define ALT_FORM2	0x020000
#define ALT_FORM3	0x040000
#define ALT_FORM4	0x080000
#define ALT_FORM5	0x100000
#define ALT_FORM6	0x200000

/* Source and destination is register. */
#define REG_DEST	0x000001
#define REG1_SOURCE	0x000002
#define REG2_SOURCE	0x000004
/* getput_arg_fast returned true. */
#define FAST_DEST	0x000008
/* Multiple instructions are required. */
#define SLOW_DEST	0x000010
/*
ALT_SIGN_EXT		0x000200
ALT_SET_FLAGS		0x000400
ALT_FORM1		0x010000
...
ALT_FORM6		0x200000 */

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
#include "sljitNativePPC_32.c"
#else
#include "sljitNativePPC_64.c"
#endif

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
#define STACK_STORE	STW
#define STACK_LOAD	LWZ
#else
#define STACK_STORE	STD
#define STACK_LOAD	LD
#endif

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 i, tmp, offs;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	FAIL_IF(push_inst(compiler, MFLR | D(0)));
	offs = -(sljit_s32)(sizeof(sljit_sw));
	FAIL_IF(push_inst(compiler, STACK_STORE | S(TMP_ZERO) | A(SLJIT_SP) | IMM(offs)));

	tmp = saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = SLJIT_S0; i >= tmp; i--) {
		offs -= (sljit_s32)(sizeof(sljit_sw));
		FAIL_IF(push_inst(compiler, STACK_STORE | S(i) | A(SLJIT_SP) | IMM(offs)));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offs -= (sljit_s32)(sizeof(sljit_sw));
		FAIL_IF(push_inst(compiler, STACK_STORE | S(i) | A(SLJIT_SP) | IMM(offs)));
	}

	SLJIT_ASSERT(offs == -(sljit_s32)GET_SAVED_REGISTERS_SIZE(compiler->scratches, compiler->saveds, 1));

#if (defined SLJIT_PPC_STACK_FRAME_V2 && SLJIT_PPC_STACK_FRAME_V2)
	FAIL_IF(push_inst(compiler, STACK_STORE | S(0) | A(SLJIT_SP) | IMM(2 * sizeof(sljit_sw))));
#else
	FAIL_IF(push_inst(compiler, STACK_STORE | S(0) | A(SLJIT_SP) | IMM(sizeof(sljit_sw))));
#endif

	FAIL_IF(push_inst(compiler, ADDI | D(TMP_ZERO) | A(0) | 0));
	if (args >= 1)
		FAIL_IF(push_inst(compiler, OR | S(SLJIT_R0) | A(SLJIT_S0) | B(SLJIT_R0)));
	if (args >= 2)
		FAIL_IF(push_inst(compiler, OR | S(SLJIT_R1) | A(SLJIT_S1) | B(SLJIT_R1)));
	if (args >= 3)
		FAIL_IF(push_inst(compiler, OR | S(SLJIT_R2) | A(SLJIT_S2) | B(SLJIT_R2)));

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1) + SLJIT_LOCALS_OFFSET;
	local_size = (local_size + 15) & ~0xf;
	compiler->local_size = local_size;

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	if (local_size <= SIMM_MAX)
		FAIL_IF(push_inst(compiler, STWU | S(SLJIT_SP) | A(SLJIT_SP) | IMM(-local_size)));
	else {
		FAIL_IF(load_immediate(compiler, 0, -local_size));
		FAIL_IF(push_inst(compiler, STWUX | S(SLJIT_SP) | A(SLJIT_SP) | B(0)));
	}
#else
	if (local_size <= SIMM_MAX)
		FAIL_IF(push_inst(compiler, STDU | S(SLJIT_SP) | A(SLJIT_SP) | IMM(-local_size)));
	else {
		FAIL_IF(load_immediate(compiler, 0, -local_size));
		FAIL_IF(push_inst(compiler, STDUX | S(SLJIT_SP) | A(SLJIT_SP) | B(0)));
	}
#endif

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1) + SLJIT_LOCALS_OFFSET;
	compiler->local_size = (local_size + 15) & ~0xf;
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 i, tmp, offs;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));

	FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));

	if (compiler->local_size <= SIMM_MAX)
		FAIL_IF(push_inst(compiler, ADDI | D(SLJIT_SP) | A(SLJIT_SP) | IMM(compiler->local_size)));
	else {
		FAIL_IF(load_immediate(compiler, 0, compiler->local_size));
		FAIL_IF(push_inst(compiler, ADD | D(SLJIT_SP) | A(SLJIT_SP) | B(0)));
	}

#if (defined SLJIT_PPC_STACK_FRAME_V2 && SLJIT_PPC_STACK_FRAME_V2)
	FAIL_IF(push_inst(compiler, STACK_LOAD | D(0) | A(SLJIT_SP) | IMM(2 * sizeof(sljit_sw))));
#else
	FAIL_IF(push_inst(compiler, STACK_LOAD | D(0) | A(SLJIT_SP) | IMM(sizeof(sljit_sw))));
#endif

	offs = -(sljit_s32)GET_SAVED_REGISTERS_SIZE(compiler->scratches, compiler->saveds, 1);

	tmp = compiler->scratches;
	for (i = SLJIT_FIRST_SAVED_REG; i <= tmp; i++) {
		FAIL_IF(push_inst(compiler, STACK_LOAD | D(i) | A(SLJIT_SP) | IMM(offs)));
		offs += (sljit_s32)(sizeof(sljit_sw));
	}

	tmp = compiler->saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - compiler->saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = tmp; i <= SLJIT_S0; i++) {
		FAIL_IF(push_inst(compiler, STACK_LOAD | D(i) | A(SLJIT_SP) | IMM(offs)));
		offs += (sljit_s32)(sizeof(sljit_sw));
	}

	FAIL_IF(push_inst(compiler, STACK_LOAD | D(TMP_ZERO) | A(SLJIT_SP) | IMM(offs)));
	SLJIT_ASSERT(offs == -(sljit_sw)(sizeof(sljit_sw)));

	FAIL_IF(push_inst(compiler, MTLR | S(0)));
	FAIL_IF(push_inst(compiler, BLR));

	return SLJIT_SUCCESS;
}

#undef STACK_STORE
#undef STACK_LOAD

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

/* i/x - immediate/indexed form
   n/w - no write-back / write-back (1 bit)
   s/l - store/load (1 bit)
   u/s - signed/unsigned (1 bit)
   w/b/h/i - word/byte/half/int allowed (2 bit)
   It contans 32 items, but not all are different. */

/* 64 bit only: [reg+imm] must be aligned to 4 bytes. */
#define INT_ALIGNED	0x10000
/* 64-bit only: there is no lwau instruction. */
#define UPDATE_REQ	0x20000

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
#define ARCH_32_64(a, b)	a
#define INST_CODE_AND_DST(inst, flags, reg) \
	((inst) | (((flags) & MEM_MASK) <= GPR_REG ? D(reg) : FD(reg)))
#else
#define ARCH_32_64(a, b)	b
#define INST_CODE_AND_DST(inst, flags, reg) \
	(((inst) & ~(INT_ALIGNED | UPDATE_REQ)) | (((flags) & MEM_MASK) <= GPR_REG ? D(reg) : FD(reg)))
#endif

static const sljit_ins data_transfer_insts[64 + 8] = {

/* -------- Unsigned -------- */

/* Word. */

/* u w n i s */ ARCH_32_64(HI(36) /* stw */, HI(62) | INT_ALIGNED | 0x0 /* std */),
/* u w n i l */ ARCH_32_64(HI(32) /* lwz */, HI(58) | INT_ALIGNED | 0x0 /* ld */),
/* u w n x s */ ARCH_32_64(HI(31) | LO(151) /* stwx */, HI(31) | LO(149) /* stdx */),
/* u w n x l */ ARCH_32_64(HI(31) | LO(23) /* lwzx */, HI(31) | LO(21) /* ldx */),

/* u w w i s */ ARCH_32_64(HI(37) /* stwu */, HI(62) | INT_ALIGNED | 0x1 /* stdu */),
/* u w w i l */ ARCH_32_64(HI(33) /* lwzu */, HI(58) | INT_ALIGNED | 0x1 /* ldu */),
/* u w w x s */ ARCH_32_64(HI(31) | LO(183) /* stwux */, HI(31) | LO(181) /* stdux */),
/* u w w x l */ ARCH_32_64(HI(31) | LO(55) /* lwzux */, HI(31) | LO(53) /* ldux */),

/* Byte. */

/* u b n i s */ HI(38) /* stb */, 
/* u b n i l */ HI(34) /* lbz */,
/* u b n x s */ HI(31) | LO(215) /* stbx */,
/* u b n x l */ HI(31) | LO(87) /* lbzx */,

/* u b w i s */ HI(39) /* stbu */,
/* u b w i l */ HI(35) /* lbzu */,
/* u b w x s */ HI(31) | LO(247) /* stbux */,
/* u b w x l */ HI(31) | LO(119) /* lbzux */,

/* Half. */

/* u h n i s */ HI(44) /* sth */,
/* u h n i l */ HI(40) /* lhz */,
/* u h n x s */ HI(31) | LO(407) /* sthx */,
/* u h n x l */ HI(31) | LO(279) /* lhzx */,

/* u h w i s */ HI(45) /* sthu */,
/* u h w i l */ HI(41) /* lhzu */,
/* u h w x s */ HI(31) | LO(439) /* sthux */,
/* u h w x l */ HI(31) | LO(311) /* lhzux */,

/* Int. */

/* u i n i s */ HI(36) /* stw */,
/* u i n i l */ HI(32) /* lwz */,
/* u i n x s */ HI(31) | LO(151) /* stwx */,
/* u i n x l */ HI(31) | LO(23) /* lwzx */,

/* u i w i s */ HI(37) /* stwu */,
/* u i w i l */ HI(33) /* lwzu */,
/* u i w x s */ HI(31) | LO(183) /* stwux */,
/* u i w x l */ HI(31) | LO(55) /* lwzux */,

/* -------- Signed -------- */

/* Word. */

/* s w n i s */ ARCH_32_64(HI(36) /* stw */, HI(62) | INT_ALIGNED | 0x0 /* std */),
/* s w n i l */ ARCH_32_64(HI(32) /* lwz */, HI(58) | INT_ALIGNED | 0x0 /* ld */),
/* s w n x s */ ARCH_32_64(HI(31) | LO(151) /* stwx */, HI(31) | LO(149) /* stdx */),
/* s w n x l */ ARCH_32_64(HI(31) | LO(23) /* lwzx */, HI(31) | LO(21) /* ldx */),

/* s w w i s */ ARCH_32_64(HI(37) /* stwu */, HI(62) | INT_ALIGNED | 0x1 /* stdu */),
/* s w w i l */ ARCH_32_64(HI(33) /* lwzu */, HI(58) | INT_ALIGNED | 0x1 /* ldu */),
/* s w w x s */ ARCH_32_64(HI(31) | LO(183) /* stwux */, HI(31) | LO(181) /* stdux */),
/* s w w x l */ ARCH_32_64(HI(31) | LO(55) /* lwzux */, HI(31) | LO(53) /* ldux */),

/* Byte. */

/* s b n i s */ HI(38) /* stb */,
/* s b n i l */ HI(34) /* lbz */ /* EXTS_REQ */,
/* s b n x s */ HI(31) | LO(215) /* stbx */,
/* s b n x l */ HI(31) | LO(87) /* lbzx */ /* EXTS_REQ */,

/* s b w i s */ HI(39) /* stbu */,
/* s b w i l */ HI(35) /* lbzu */ /* EXTS_REQ */,
/* s b w x s */ HI(31) | LO(247) /* stbux */,
/* s b w x l */ HI(31) | LO(119) /* lbzux */ /* EXTS_REQ */,

/* Half. */

/* s h n i s */ HI(44) /* sth */,
/* s h n i l */ HI(42) /* lha */,
/* s h n x s */ HI(31) | LO(407) /* sthx */,
/* s h n x l */ HI(31) | LO(343) /* lhax */,

/* s h w i s */ HI(45) /* sthu */,
/* s h w i l */ HI(43) /* lhau */,
/* s h w x s */ HI(31) | LO(439) /* sthux */,
/* s h w x l */ HI(31) | LO(375) /* lhaux */,

/* Int. */

/* s i n i s */ HI(36) /* stw */,
/* s i n i l */ ARCH_32_64(HI(32) /* lwz */, HI(58) | INT_ALIGNED | 0x2 /* lwa */),
/* s i n x s */ HI(31) | LO(151) /* stwx */,
/* s i n x l */ ARCH_32_64(HI(31) | LO(23) /* lwzx */, HI(31) | LO(341) /* lwax */),

/* s i w i s */ HI(37) /* stwu */,
/* s i w i l */ ARCH_32_64(HI(33) /* lwzu */, HI(58) | INT_ALIGNED | UPDATE_REQ | 0x2 /* lwa */),
/* s i w x s */ HI(31) | LO(183) /* stwux */,
/* s i w x l */ ARCH_32_64(HI(31) | LO(55) /* lwzux */, HI(31) | LO(373) /* lwaux */),

/* -------- Double -------- */

/* d   n i s */ HI(54) /* stfd */,
/* d   n i l */ HI(50) /* lfd */,
/* d   n x s */ HI(31) | LO(727) /* stfdx */,
/* d   n x l */ HI(31) | LO(599) /* lfdx */,

/* s   n i s */ HI(52) /* stfs */,
/* s   n i l */ HI(48) /* lfs */,
/* s   n x s */ HI(31) | LO(663) /* stfsx */,
/* s   n x l */ HI(31) | LO(535) /* lfsx */,

};

#undef ARCH_32_64

/* Simple cases, (no caching is required). */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 inp_flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_ins inst;

	/* Should work when (arg & REG_MASK) == 0. */
	SLJIT_COMPILE_ASSERT(A(0) == 0, a0_must_be_0);
	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (arg & OFFS_REG_MASK) {
		if (argw & 0x3)
			return 0;
		if (inp_flags & ARG_TEST)
			return 1;

		inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
		SLJIT_ASSERT(!(inst & (INT_ALIGNED | UPDATE_REQ)));
		FAIL_IF(push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | B(OFFS_REG(arg))));
		return -1;
	}

	if (SLJIT_UNLIKELY(!(arg & REG_MASK)))
		inp_flags &= ~WRITE_BACK;

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	inst = data_transfer_insts[inp_flags & MEM_MASK];
	SLJIT_ASSERT((arg & REG_MASK) || !(inst & UPDATE_REQ));

	if (argw > SIMM_MAX || argw < SIMM_MIN || ((inst & INT_ALIGNED) && (argw & 0x3)) || (inst & UPDATE_REQ))
		return 0;
	if (inp_flags & ARG_TEST)
		return 1;
#endif

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	if (argw > SIMM_MAX || argw < SIMM_MIN)
		return 0;
	if (inp_flags & ARG_TEST)
		return 1;

	inst = data_transfer_insts[inp_flags & MEM_MASK];
	SLJIT_ASSERT(!(inst & (INT_ALIGNED | UPDATE_REQ)));
#endif

	FAIL_IF(push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | IMM(argw)));
	return -1;
}

/* See getput_arg below.
   Note: can_cache is called only for binary operators. Those operator always
   uses word arguments without write back. */
static sljit_s32 can_cache(sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_sw high_short, next_high_short;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_sw diff;
#endif

	SLJIT_ASSERT((arg & SLJIT_MEM) && (next_arg & SLJIT_MEM));

	if (arg & OFFS_REG_MASK)
		return ((arg & OFFS_REG_MASK) == (next_arg & OFFS_REG_MASK) && (argw & 0x3) == (next_argw & 0x3));

	if (next_arg & OFFS_REG_MASK)
		return 0;

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	high_short = (argw + ((argw & 0x8000) << 1)) & ~0xffff;
	next_high_short = (next_argw + ((next_argw & 0x8000) << 1)) & ~0xffff;
	return high_short == next_high_short;
#else
	if (argw <= 0x7fffffffl && argw >= -0x80000000l) {
		high_short = (argw + ((argw & 0x8000) << 1)) & ~0xffff;
		next_high_short = (next_argw + ((next_argw & 0x8000) << 1)) & ~0xffff;
		if (high_short == next_high_short)
			return 1;
	}

	diff = argw - next_argw;
	if (!(arg & REG_MASK))
		return diff <= SIMM_MAX && diff >= SIMM_MIN;

	if (arg == next_arg && diff <= SIMM_MAX && diff >= SIMM_MIN)
		return 1;

	return 0;
#endif
}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define ADJUST_CACHED_IMM(imm) \
	if ((inst & INT_ALIGNED) && (imm & 0x3)) { \
		/* Adjust cached value. Fortunately this is really a rare case */ \
		compiler->cache_argw += imm & 0x3; \
		FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG3) | A(TMP_REG3) | (imm & 0x3))); \
		imm &= ~0x3; \
	}
#endif

/* Emit the necessary instructions. See can_cache above. */
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 inp_flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 tmp_r;
	sljit_ins inst;
	sljit_sw high_short, next_high_short;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_sw diff;
#endif

	SLJIT_ASSERT(arg & SLJIT_MEM);

	tmp_r = ((inp_flags & LOAD_DATA) && ((inp_flags) & MEM_MASK) <= GPR_REG) ? reg : TMP_REG1;
	/* Special case for "mov reg, [reg, ... ]". */
	if ((arg & REG_MASK) == tmp_r)
		tmp_r = TMP_REG1;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;
		/* Otherwise getput_arg_fast would capture it. */
		SLJIT_ASSERT(argw);

		if ((SLJIT_MEM | (arg & OFFS_REG_MASK)) == compiler->cache_arg && argw == compiler->cache_argw)
			tmp_r = TMP_REG3;
		else {
			if ((arg & OFFS_REG_MASK) == (next_arg & OFFS_REG_MASK) && argw == (next_argw & 0x3)) {
				compiler->cache_arg = SLJIT_MEM | (arg & OFFS_REG_MASK);
				compiler->cache_argw = argw;
				tmp_r = TMP_REG3;
			}
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
			FAIL_IF(push_inst(compiler, RLWINM | S(OFFS_REG(arg)) | A(tmp_r) | (argw << 11) | ((31 - argw) << 1)));
#else
			FAIL_IF(push_inst(compiler, RLDI(tmp_r, OFFS_REG(arg), argw, 63 - argw, 1)));
#endif
		}
		inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
		SLJIT_ASSERT(!(inst & (INT_ALIGNED | UPDATE_REQ)));
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | B(tmp_r));
	}

	if (SLJIT_UNLIKELY(!(arg & REG_MASK)))
		inp_flags &= ~WRITE_BACK;

	inst = data_transfer_insts[inp_flags & MEM_MASK];
	SLJIT_ASSERT((arg & REG_MASK) || !(inst & UPDATE_REQ));

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (argw <= 0x7fff7fffl && argw >= -0x80000000l
			&& (!(inst & INT_ALIGNED) || !(argw & 0x3)) && !(inst & UPDATE_REQ)) {
#endif

		arg &= REG_MASK;
		high_short = (sljit_s32)(argw + ((argw & 0x8000) << 1)) & ~0xffff;
		/* The getput_arg_fast should handle this otherwise. */
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		SLJIT_ASSERT(high_short && high_short <= 0x7fffffffl && high_short >= -0x80000000l);
#else
		SLJIT_ASSERT(high_short && !(inst & (INT_ALIGNED | UPDATE_REQ)));
#endif

		if (inp_flags & WRITE_BACK) {
			if (arg == reg) {
				FAIL_IF(push_inst(compiler, OR | S(reg) | A(tmp_r) | B(reg)));
				reg = tmp_r;
			}
			tmp_r = arg;
			FAIL_IF(push_inst(compiler, ADDIS | D(arg) | A(arg) | IMM(high_short >> 16)));
		}
		else if (compiler->cache_arg != (SLJIT_MEM | arg) || high_short != compiler->cache_argw) {
			if ((next_arg & SLJIT_MEM) && !(next_arg & OFFS_REG_MASK)) {
				next_high_short = (sljit_s32)(next_argw + ((next_argw & 0x8000) << 1)) & ~0xffff;
				if (high_short == next_high_short) {
					compiler->cache_arg = SLJIT_MEM | arg;
					compiler->cache_argw = high_short;
					tmp_r = TMP_REG3;
				}
			}
			FAIL_IF(push_inst(compiler, ADDIS | D(tmp_r) | A(arg & REG_MASK) | IMM(high_short >> 16)));
		}
		else
			tmp_r = TMP_REG3;

		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(tmp_r) | IMM(argw));

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	}

	/* Everything else is PPC-64 only. */
	if (SLJIT_UNLIKELY(!(arg & REG_MASK))) {
		diff = argw - compiler->cache_argw;
		if ((compiler->cache_arg & SLJIT_IMM) && diff <= SIMM_MAX && diff >= SIMM_MIN) {
			ADJUST_CACHED_IMM(diff);
			return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(TMP_REG3) | IMM(diff));
		}

		diff = argw - next_argw;
		if ((next_arg & SLJIT_MEM) && diff <= SIMM_MAX && diff >= SIMM_MIN) {
			SLJIT_ASSERT(inp_flags & LOAD_DATA);

			compiler->cache_arg = SLJIT_IMM;
			compiler->cache_argw = argw;
			tmp_r = TMP_REG3;
		}

		FAIL_IF(load_immediate(compiler, tmp_r, argw));
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(tmp_r));
	}

	diff = argw - compiler->cache_argw;
	if (compiler->cache_arg == arg && diff <= SIMM_MAX && diff >= SIMM_MIN) {
		SLJIT_ASSERT(!(inp_flags & WRITE_BACK) && !(inst & UPDATE_REQ));
		ADJUST_CACHED_IMM(diff);
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(TMP_REG3) | IMM(diff));
	}

	if ((compiler->cache_arg & SLJIT_IMM) && diff <= SIMM_MAX && diff >= SIMM_MIN) {
		inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
		SLJIT_ASSERT(!(inst & (INT_ALIGNED | UPDATE_REQ)));
		if (compiler->cache_argw != argw) {
			FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG3) | A(TMP_REG3) | IMM(diff)));
			compiler->cache_argw = argw;
		}
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | B(TMP_REG3));
	}

	if (argw == next_argw && (next_arg & SLJIT_MEM)) {
		SLJIT_ASSERT(inp_flags & LOAD_DATA);
		FAIL_IF(load_immediate(compiler, TMP_REG3, argw));

		compiler->cache_arg = SLJIT_IMM;
		compiler->cache_argw = argw;

		inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
		SLJIT_ASSERT(!(inst & (INT_ALIGNED | UPDATE_REQ)));
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | B(TMP_REG3));
	}

	diff = argw - next_argw;
	if (arg == next_arg && !(inp_flags & WRITE_BACK) && diff <= SIMM_MAX && diff >= SIMM_MIN) {
		SLJIT_ASSERT(inp_flags & LOAD_DATA);
		FAIL_IF(load_immediate(compiler, TMP_REG3, argw));
		FAIL_IF(push_inst(compiler, ADD | D(TMP_REG3) | A(TMP_REG3) | B(arg & REG_MASK)));

		compiler->cache_arg = arg;
		compiler->cache_argw = argw;

		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(TMP_REG3));
	}

	if ((next_arg & SLJIT_MEM) && !(next_arg & OFFS_REG_MASK) && diff <= SIMM_MAX && diff >= SIMM_MIN) {
		SLJIT_ASSERT(inp_flags & LOAD_DATA);
		FAIL_IF(load_immediate(compiler, TMP_REG3, argw));

		compiler->cache_arg = SLJIT_IMM;
		compiler->cache_argw = argw;
		tmp_r = TMP_REG3;
	}
	else
		FAIL_IF(load_immediate(compiler, tmp_r, argw));

	/* Get the indexed version instead of the normal one. */
	inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
	SLJIT_ASSERT(!(inst & (INT_ALIGNED | UPDATE_REQ)));
	return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | B(tmp_r));
#endif
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 input_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* arg1 goes to TMP_REG1 or src reg
	   arg2 goes to TMP_REG2, imm or src reg
	   TMP_REG3 can be used for caching
	   result goes to TMP_REG2, so put result can use TMP_REG1 and TMP_REG3. */
	sljit_s32 dst_r;
	sljit_s32 src1_r;
	sljit_s32 src2_r;
	sljit_s32 sugg_src2_r = TMP_REG2;
	sljit_s32 flags = input_flags & (ALT_FORM1 | ALT_FORM2 | ALT_FORM3 | ALT_FORM4 | ALT_FORM5 | ALT_FORM6 | ALT_SIGN_EXT | ALT_SET_FLAGS);

	if (!(input_flags & ALT_KEEP_CACHE)) {
		compiler->cache_arg = 0;
		compiler->cache_argw = 0;
	}

	/* Destination check. */
	if (SLJIT_UNLIKELY(dst == SLJIT_UNUSED)) {
		if (op >= SLJIT_MOV && op <= SLJIT_MOVU_S32 && !(src2 & SLJIT_MEM))
			return SLJIT_SUCCESS;
		dst_r = TMP_REG2;
	}
	else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (op >= SLJIT_MOV && op <= SLJIT_MOVU_S32)
			sugg_src2_r = dst_r;
	}
	else {
		SLJIT_ASSERT(dst & SLJIT_MEM);
		if (getput_arg_fast(compiler, input_flags | ARG_TEST, TMP_REG2, dst, dstw)) {
			flags |= FAST_DEST;
			dst_r = TMP_REG2;
		}
		else {
			flags |= SLOW_DEST;
			dst_r = 0;
		}
	}

	/* Source 1. */
	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	}
	else if (src1 & SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, src1w));
		src1_r = TMP_REG1;
	}
	else if (getput_arg_fast(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w)) {
		FAIL_IF(compiler->error);
		src1_r = TMP_REG1;
	}
	else
		src1_r = 0;

	/* Source 2. */
	if (FAST_IS_REG(src2)) {
		src2_r = src2;
		flags |= REG2_SOURCE;
		if (!(flags & REG_DEST) && op >= SLJIT_MOV && op <= SLJIT_MOVU_S32)
			dst_r = src2_r;
	}
	else if (src2 & SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, sugg_src2_r, src2w));
		src2_r = sugg_src2_r;
	}
	else if (getput_arg_fast(compiler, input_flags | LOAD_DATA, sugg_src2_r, src2, src2w)) {
		FAIL_IF(compiler->error);
		src2_r = sugg_src2_r;
	}
	else
		src2_r = 0;

	/* src1_r, src2_r and dst_r can be zero (=unprocessed).
	   All arguments are complex addressing modes, and it is a binary operator. */
	if (src1_r == 0 && src2_r == 0 && dst_r == 0) {
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG2, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG2, src2, src2w, dst, dstw));
		}
		src1_r = TMP_REG1;
		src2_r = TMP_REG2;
	}
	else if (src1_r == 0 && src2_r == 0) {
		FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w, src2, src2w));
		src1_r = TMP_REG1;
	}
	else if (src1_r == 0 && dst_r == 0) {
		FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
		src1_r = TMP_REG1;
	}
	else if (src2_r == 0 && dst_r == 0) {
		FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, sugg_src2_r, src2, src2w, dst, dstw));
		src2_r = sugg_src2_r;
	}

	if (dst_r == 0)
		dst_r = TMP_REG2;

	if (src1_r == 0) {
		FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w, 0, 0));
		src1_r = TMP_REG1;
	}

	if (src2_r == 0) {
		FAIL_IF(getput_arg(compiler, input_flags | LOAD_DATA, sugg_src2_r, src2, src2w, 0, 0));
		src2_r = sugg_src2_r;
	}

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (flags & (FAST_DEST | SLOW_DEST)) {
		if (flags & FAST_DEST)
			FAIL_IF(getput_arg_fast(compiler, input_flags, dst_r, dst, dstw));
		else
			FAIL_IF(getput_arg(compiler, input_flags, dst_r, dst, dstw, 0, 0));
	}
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 int_op = op & SLJIT_I32_OP;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	op = GET_OPCODE(op);
	switch (op) {
	case SLJIT_BREAKPOINT:
	case SLJIT_NOP:
		return push_inst(compiler, NOP);
	case SLJIT_LMUL_UW:
	case SLJIT_LMUL_SW:
		FAIL_IF(push_inst(compiler, OR | S(SLJIT_R0) | A(TMP_REG1) | B(SLJIT_R0)));
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		FAIL_IF(push_inst(compiler, MULLD | D(SLJIT_R0) | A(TMP_REG1) | B(SLJIT_R1)));
		return push_inst(compiler, (op == SLJIT_LMUL_UW ? MULHDU : MULHD) | D(SLJIT_R1) | A(TMP_REG1) | B(SLJIT_R1));
#else
		FAIL_IF(push_inst(compiler, MULLW | D(SLJIT_R0) | A(TMP_REG1) | B(SLJIT_R1)));
		return push_inst(compiler, (op == SLJIT_LMUL_UW ? MULHWU : MULHW) | D(SLJIT_R1) | A(TMP_REG1) | B(SLJIT_R1));
#endif
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
		FAIL_IF(push_inst(compiler, OR | S(SLJIT_R0) | A(TMP_REG1) | B(SLJIT_R0)));
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		FAIL_IF(push_inst(compiler, (int_op ? (op == SLJIT_DIVMOD_UW ? DIVWU : DIVW) : (op == SLJIT_DIVMOD_UW ? DIVDU : DIVD)) | D(SLJIT_R0) | A(SLJIT_R0) | B(SLJIT_R1)));
		FAIL_IF(push_inst(compiler, (int_op ? MULLW : MULLD) | D(SLJIT_R1) | A(SLJIT_R0) | B(SLJIT_R1)));
#else
		FAIL_IF(push_inst(compiler, (op == SLJIT_DIVMOD_UW ? DIVWU : DIVW) | D(SLJIT_R0) | A(SLJIT_R0) | B(SLJIT_R1)));
		FAIL_IF(push_inst(compiler, MULLW | D(SLJIT_R1) | A(SLJIT_R0) | B(SLJIT_R1)));
#endif
		return push_inst(compiler, SUBF | D(SLJIT_R1) | A(SLJIT_R1) | B(TMP_REG1));
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		return push_inst(compiler, (int_op ? (op == SLJIT_DIV_UW ? DIVWU : DIVW) : (op == SLJIT_DIV_UW ? DIVDU : DIVD)) | D(SLJIT_R0) | A(SLJIT_R0) | B(SLJIT_R1));
#else
		return push_inst(compiler, (op == SLJIT_DIV_UW ? DIVWU : DIVW) | D(SLJIT_R0) | A(SLJIT_R0) | B(SLJIT_R1));
#endif
	}

	return SLJIT_SUCCESS;
}

#define EMIT_MOV(type, type_flags, type_cast) \
	emit_op(compiler, (src & SLJIT_IMM) ? SLJIT_MOV : type, flags | (type_flags), dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? type_cast srcw : srcw)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 flags = GET_FLAGS(op) ? ALT_SET_FLAGS : 0;
	sljit_s32 op_flags = GET_ALL_FLAGS(op);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	op = GET_OPCODE(op);
	if ((src & SLJIT_IMM) && srcw == 0)
		src = TMP_ZERO;

	if (op_flags & SLJIT_SET_O)
		FAIL_IF(push_inst(compiler, MTXER | S(TMP_ZERO)));

	if (op_flags & SLJIT_I32_OP) {
		if (op < SLJIT_NOT) {
			if (FAST_IS_REG(src) && src == dst) {
				if (!TYPE_CAST_NEEDED(op))
					return SLJIT_SUCCESS;
			}
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
			if (op == SLJIT_MOV_S32 && (src & SLJIT_MEM))
				op = SLJIT_MOV_U32;
			if (op == SLJIT_MOVU_S32 && (src & SLJIT_MEM))
				op = SLJIT_MOVU_U32;
			if (op == SLJIT_MOV_U32 && (src & SLJIT_IMM))
				op = SLJIT_MOV_S32;
			if (op == SLJIT_MOVU_U32 && (src & SLJIT_IMM))
				op = SLJIT_MOVU_S32;
#endif
		}
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		else {
			/* Most operations expect sign extended arguments. */
			flags |= INT_DATA | SIGNED_DATA;
			if (src & SLJIT_IMM)
				srcw = (sljit_s32)srcw;
		}
#endif
	}

	switch (op) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
#endif
		return emit_op(compiler, SLJIT_MOV, flags | WORD_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	case SLJIT_MOV_U32:
		return EMIT_MOV(SLJIT_MOV_U32, INT_DATA, (sljit_u32));

	case SLJIT_MOV_S32:
		return EMIT_MOV(SLJIT_MOV_S32, INT_DATA | SIGNED_DATA, (sljit_s32));
#endif

	case SLJIT_MOV_U8:
		return EMIT_MOV(SLJIT_MOV_U8, BYTE_DATA, (sljit_u8));

	case SLJIT_MOV_S8:
		return EMIT_MOV(SLJIT_MOV_S8, BYTE_DATA | SIGNED_DATA, (sljit_s8));

	case SLJIT_MOV_U16:
		return EMIT_MOV(SLJIT_MOV_U16, HALF_DATA, (sljit_u16));

	case SLJIT_MOV_S16:
		return EMIT_MOV(SLJIT_MOV_S16, HALF_DATA | SIGNED_DATA, (sljit_s16));

	case SLJIT_MOVU:
	case SLJIT_MOVU_P:
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	case SLJIT_MOVU_U32:
	case SLJIT_MOVU_S32:
#endif
		return emit_op(compiler, SLJIT_MOV, flags | WORD_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, srcw);

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	case SLJIT_MOVU_U32:
		return EMIT_MOV(SLJIT_MOV_U32, INT_DATA | WRITE_BACK, (sljit_u32));

	case SLJIT_MOVU_S32:
		return EMIT_MOV(SLJIT_MOV_S32, INT_DATA | SIGNED_DATA | WRITE_BACK, (sljit_s32));
#endif

	case SLJIT_MOVU_U8:
		return EMIT_MOV(SLJIT_MOV_U8, BYTE_DATA | WRITE_BACK, (sljit_u8));

	case SLJIT_MOVU_S8:
		return EMIT_MOV(SLJIT_MOV_S8, BYTE_DATA | SIGNED_DATA | WRITE_BACK, (sljit_s8));

	case SLJIT_MOVU_U16:
		return EMIT_MOV(SLJIT_MOV_U16, HALF_DATA | WRITE_BACK, (sljit_u16));

	case SLJIT_MOVU_S16:
		return EMIT_MOV(SLJIT_MOV_S16, HALF_DATA | SIGNED_DATA | WRITE_BACK, (sljit_s16));

	case SLJIT_NOT:
		return emit_op(compiler, SLJIT_NOT, flags, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_NEG:
		return emit_op(compiler, SLJIT_NEG, flags, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_CLZ:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		return emit_op(compiler, SLJIT_CLZ, flags | (!(op_flags & SLJIT_I32_OP) ? 0 : ALT_FORM1), dst, dstw, TMP_REG1, 0, src, srcw);
#else
		return emit_op(compiler, SLJIT_CLZ, flags, dst, dstw, TMP_REG1, 0, src, srcw);
#endif
	}

	return SLJIT_SUCCESS;
}

#undef EMIT_MOV

#define TEST_SL_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && (srcw) <= SIMM_MAX && (srcw) >= SIMM_MIN)

#define TEST_UL_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && !((srcw) & ~0xffff))

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define TEST_SH_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && !((srcw) & 0xffff) && (srcw) <= 0x7fffffffl && (srcw) >= -0x80000000l)
#else
#define TEST_SH_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && !((srcw) & 0xffff))
#endif

#define TEST_UH_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && !((srcw) & ~0xffff0000))

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define TEST_ADD_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && (srcw) <= 0x7fff7fffl && (srcw) >= -0x80000000l)
#else
#define TEST_ADD_IMM(src, srcw) \
	((src) & SLJIT_IMM)
#endif

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define TEST_UI_IMM(src, srcw) \
	(((src) & SLJIT_IMM) && !((srcw) & ~0xffffffff))
#else
#define TEST_UI_IMM(src, srcw) \
	((src) & SLJIT_IMM)
#endif

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 flags = GET_FLAGS(op) ? ALT_SET_FLAGS : 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	if ((src1 & SLJIT_IMM) && src1w == 0)
		src1 = TMP_ZERO;
	if ((src2 & SLJIT_IMM) && src2w == 0)
		src2 = TMP_ZERO;

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (op & SLJIT_I32_OP) {
		/* Most operations expect sign extended arguments. */
		flags |= INT_DATA | SIGNED_DATA;
		if (src1 & SLJIT_IMM)
			src1w = (sljit_s32)(src1w);
		if (src2 & SLJIT_IMM)
			src2w = (sljit_s32)(src2w);
		if (GET_FLAGS(op))
			flags |= ALT_SIGN_EXT;
	}
#endif
	if (op & SLJIT_SET_O)
		FAIL_IF(push_inst(compiler, MTXER | S(TMP_ZERO)));
	if (src2 == TMP_REG2)
		flags |= ALT_KEEP_CACHE;

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
		if (!GET_FLAGS(op) && ((src1 | src2) & SLJIT_IMM)) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = src2w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = src1w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM1, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			if (TEST_SH_IMM(src2, src2w)) {
				compiler->imm = (src2w >> 16) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SH_IMM(src1, src1w)) {
				compiler->imm = (src1w >> 16) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			/* Range between -1 and -32768 is covered above. */
			if (TEST_ADD_IMM(src2, src2w)) {
				compiler->imm = src2w & 0xffffffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM4, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_ADD_IMM(src1, src1w)) {
				compiler->imm = src1w & 0xffffffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM4, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		if (!(GET_FLAGS(op) & (SLJIT_SET_E | SLJIT_SET_O))) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = src2w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = src1w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM3, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		return emit_op(compiler, SLJIT_ADD, flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_ADDC:
		return emit_op(compiler, SLJIT_ADDC, flags | (!(op & SLJIT_KEEP_FLAGS) ? 0 : ALT_FORM1), dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
		if (!GET_FLAGS(op) && ((src1 | src2) & SLJIT_IMM)) {
			if (TEST_SL_IMM(src2, -src2w)) {
				compiler->imm = (-src2w) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = src1w & 0xffff;
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM1, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			if (TEST_SH_IMM(src2, -src2w)) {
				compiler->imm = ((-src2w) >> 16) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			/* Range between -1 and -32768 is covered above. */
			if (TEST_ADD_IMM(src2, -src2w)) {
				compiler->imm = -src2w & 0xffffffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM4, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
		}
		if (dst == SLJIT_UNUSED && (op & (SLJIT_SET_E | SLJIT_SET_U | SLJIT_SET_S)) && !(op & (SLJIT_SET_O | SLJIT_SET_C))) {
			if (!(op & SLJIT_SET_U)) {
				/* We know ALT_SIGN_EXT is set if it is an SLJIT_I32_OP on 64 bit systems. */
				if (TEST_SL_IMM(src2, src2w)) {
					compiler->imm = src2w & 0xffff;
					return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
				}
				if (GET_FLAGS(op) == SLJIT_SET_E && TEST_SL_IMM(src1, src1w)) {
					compiler->imm = src1w & 0xffff;
					return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2, dst, dstw, src2, src2w, TMP_REG2, 0);
				}
			}
			if (!(op & (SLJIT_SET_E | SLJIT_SET_S))) {
				/* We know ALT_SIGN_EXT is set if it is an SLJIT_I32_OP on 64 bit systems. */
				if (TEST_UL_IMM(src2, src2w)) {
					compiler->imm = src2w & 0xffff;
					return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
				}
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM4, dst, dstw, src1, src1w, src2, src2w);
			}
			if ((src2 & SLJIT_IMM) && src2w >= 0 && src2w <= 0x7fff) {
				compiler->imm = src2w;
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2 | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			return emit_op(compiler, SLJIT_SUB, flags | ((op & SLJIT_SET_U) ? ALT_FORM4 : 0) | ((op & (SLJIT_SET_E | SLJIT_SET_S)) ? ALT_FORM5 : 0), dst, dstw, src1, src1w, src2, src2w);
		}
		if (!(op & (SLJIT_SET_E | SLJIT_SET_U | SLJIT_SET_S | SLJIT_SET_O))) {
			if (TEST_SL_IMM(src2, -src2w)) {
				compiler->imm = (-src2w) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
		}
		/* We know ALT_SIGN_EXT is set if it is an SLJIT_I32_OP on 64 bit systems. */
		return emit_op(compiler, SLJIT_SUB, flags | (!(op & SLJIT_SET_U) ? 0 : ALT_FORM6), dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUBC:
		return emit_op(compiler, SLJIT_SUBC, flags | (!(op & SLJIT_KEEP_FLAGS) ? 0 : ALT_FORM1), dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op & SLJIT_I32_OP)
			flags |= ALT_FORM2;
#endif
		if (!GET_FLAGS(op)) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = src2w & 0xffff;
				return emit_op(compiler, SLJIT_MUL, flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = src1w & 0xffff;
				return emit_op(compiler, SLJIT_MUL, flags | ALT_FORM1, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		return emit_op(compiler, SLJIT_MUL, flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		/* Commutative unsigned operations. */
		if (!GET_FLAGS(op) || GET_OPCODE(op) == SLJIT_AND) {
			if (TEST_UL_IMM(src2, src2w)) {
				compiler->imm = src2w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_UL_IMM(src1, src1w)) {
				compiler->imm = src1w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM1, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			if (TEST_UH_IMM(src2, src2w)) {
				compiler->imm = (src2w >> 16) & 0xffff;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_UH_IMM(src1, src1w)) {
				compiler->imm = (src1w >> 16) & 0xffff;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM2, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		if (!GET_FLAGS(op) && GET_OPCODE(op) != SLJIT_AND) {
			if (TEST_UI_IMM(src2, src2w)) {
				compiler->imm = src2w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_UI_IMM(src1, src1w)) {
				compiler->imm = src1w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM3, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		return emit_op(compiler, GET_OPCODE(op), flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_ASHR:
		if (op & SLJIT_KEEP_FLAGS)
			flags |= ALT_FORM3;
		/* Fall through. */
	case SLJIT_SHL:
	case SLJIT_LSHR:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op & SLJIT_I32_OP)
			flags |= ALT_FORM2;
#endif
		if (src2 & SLJIT_IMM) {
			compiler->imm = src2w;
			return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
		}
		return emit_op(compiler, GET_OPCODE(op), flags, dst, dstw, src1, src1w, src2, src2w);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(reg));
	return reg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_float_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_float_register_index(reg));
	return reg;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_s32 size)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_ins*)instruction);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_is_fpu_available(void)
{
#ifdef SLJIT_IS_FPU_AVAILABLE
	return SLJIT_IS_FPU_AVAILABLE;
#else
	/* Available by default. */
	return 1;
#endif
}

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_F32_OP) >> 6))
#define SELECT_FOP(op, single, double) ((op & SLJIT_F32_OP) ? single : double)

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define FLOAT_TMP_MEM_OFFSET (6 * sizeof(sljit_sw))
#else
#define FLOAT_TMP_MEM_OFFSET (2 * sizeof(sljit_sw))

#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define FLOAT_TMP_MEM_OFFSET_LOW (2 * sizeof(sljit_sw))
#define FLOAT_TMP_MEM_OFFSET_HI (3 * sizeof(sljit_sw))
#else
#define FLOAT_TMP_MEM_OFFSET_LOW (3 * sizeof(sljit_sw))
#define FLOAT_TMP_MEM_OFFSET_HI (2 * sizeof(sljit_sw))
#endif

#endif /* SLJIT_CONFIG_PPC_64 */

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (src & SLJIT_MEM) {
		/* We can ignore the temporary data store on the stack from caching point of view. */
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src, srcw, dst, dstw));
		src = TMP_FREG1;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	op = GET_OPCODE(op);
	FAIL_IF(push_inst(compiler, (op == SLJIT_CONV_S32_FROM_F64 ? FCTIWZ : FCTIDZ) | FD(TMP_FREG1) | FB(src)));

	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;

	if (op == SLJIT_CONV_SW_FROM_F64) {
		if (FAST_IS_REG(dst)) {
			FAIL_IF(emit_op_mem2(compiler, DOUBLE_DATA, TMP_FREG1, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, 0, 0));
			return emit_op_mem2(compiler, WORD_DATA | LOAD_DATA, dst, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, 0, 0);
		}
		return emit_op_mem2(compiler, DOUBLE_DATA, TMP_FREG1, dst, dstw, 0, 0);
	}

#else
	FAIL_IF(push_inst(compiler, FCTIWZ | FD(TMP_FREG1) | FB(src)));

	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;
#endif

	if (FAST_IS_REG(dst)) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, FLOAT_TMP_MEM_OFFSET));
		FAIL_IF(push_inst(compiler, STFIWX | FS(TMP_FREG1) | A(SLJIT_SP) | B(TMP_REG1)));
		return emit_op_mem2(compiler, INT_DATA | LOAD_DATA, dst, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, 0, 0);
	}

	SLJIT_ASSERT(dst & SLJIT_MEM);

	if (dst & OFFS_REG_MASK) {
		dstw &= 0x3;
		if (dstw) {
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
			FAIL_IF(push_inst(compiler, RLWINM | S(OFFS_REG(dst)) | A(TMP_REG1) | (dstw << 11) | ((31 - dstw) << 1)));
#else
			FAIL_IF(push_inst(compiler, RLDI(TMP_REG1, OFFS_REG(dst), dstw, 63 - dstw, 1)));
#endif
			dstw = TMP_REG1;
		}
		else
			dstw = OFFS_REG(dst);
	}
	else {
		if ((dst & REG_MASK) && !dstw) {
			dstw = dst & REG_MASK;
			dst = 0;
		}
		else {
			/* This works regardless we have SLJIT_MEM1 or SLJIT_MEM0. */
			FAIL_IF(load_immediate(compiler, TMP_REG1, dstw));
			dstw = TMP_REG1;
		}
	}

	return push_inst(compiler, STFIWX | FS(TMP_FREG1) | A(dst & REG_MASK) | B(dstw));
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)

	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_IMM) {
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
	}
	else if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32) {
		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, EXTSW | S(src) | A(TMP_REG1)));
		else
			FAIL_IF(emit_op_mem2(compiler, INT_DATA | SIGNED_DATA | LOAD_DATA, TMP_REG1, src, srcw, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET));
		src = TMP_REG1;
	}

	if (FAST_IS_REG(src)) {
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA, src, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET));
		FAIL_IF(emit_op_mem2(compiler, DOUBLE_DATA | LOAD_DATA, TMP_FREG1, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, dst, dstw));
	}
	else
		FAIL_IF(emit_op_mem2(compiler, DOUBLE_DATA | LOAD_DATA, TMP_FREG1, src, srcw, dst, dstw));

	FAIL_IF(push_inst(compiler, FCFID | FD(dst_r) | FB(TMP_FREG1)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG1, dst, dstw, 0, 0);
	if (op & SLJIT_F32_OP)
		return push_inst(compiler, FRSP | FD(dst_r) | FB(dst_r));
	return SLJIT_SUCCESS;

#else

	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;
	sljit_s32 invert_sign = 1;

	if (src & SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw ^ 0x80000000));
		src = TMP_REG1;
		invert_sign = 0;
	}
	else if (!FAST_IS_REG(src)) {
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA | SIGNED_DATA | LOAD_DATA, TMP_REG1, src, srcw, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_LOW));
		src = TMP_REG1;
	}

	/* First, a special double floating point value is constructed: (2^53 + (input xor (2^31)))
	   The double precision format has exactly 53 bit precision, so the lower 32 bit represents
	   the lower 32 bit of such value. The result of xor 2^31 is the same as adding 0x80000000
	   to the input, which shifts it into the 0 - 0xffffffff range. To get the converted floating
	   point value, we need to substract 2^53 + 2^31 from the constructed value. */
	FAIL_IF(push_inst(compiler, ADDIS | D(TMP_REG2) | A(0) | 0x4330));
	if (invert_sign)
		FAIL_IF(push_inst(compiler, XORIS | S(src) | A(TMP_REG1) | 0x8000));
	FAIL_IF(emit_op_mem2(compiler, WORD_DATA, TMP_REG2, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_HI, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET));
	FAIL_IF(emit_op_mem2(compiler, WORD_DATA, TMP_REG1, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_LOW, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_HI));
	FAIL_IF(push_inst(compiler, ADDIS | D(TMP_REG1) | A(0) | 0x8000));
	FAIL_IF(emit_op_mem2(compiler, DOUBLE_DATA | LOAD_DATA, TMP_FREG1, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_LOW));
	FAIL_IF(emit_op_mem2(compiler, WORD_DATA, TMP_REG1, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_LOW, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET));
	FAIL_IF(emit_op_mem2(compiler, DOUBLE_DATA | LOAD_DATA, TMP_FREG2, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET, SLJIT_MEM1(SLJIT_SP), FLOAT_TMP_MEM_OFFSET_LOW));

	FAIL_IF(push_inst(compiler, FSUB | FD(dst_r) | FA(TMP_FREG1) | FB(TMP_FREG2)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG1, dst, dstw, 0, 0);
	if (op & SLJIT_F32_OP)
		return push_inst(compiler, FRSP | FD(dst_r) | FB(dst_r));
	return SLJIT_SUCCESS;

#endif
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, src2, src2w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, 0, 0));
		src2 = TMP_FREG2;
	}

	return push_inst(compiler, FCMPU | CRD(4) | FA(src1) | FB(src2));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	SLJIT_COMPILE_ASSERT((SLJIT_F32_OP == 0x100) && !(DOUBLE_DATA & 0x4), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_F32_OP;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, dst_r, src, srcw, dst, dstw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_CONV_F64_FROM_F32:
		op ^= SLJIT_F32_OP;
		if (op & SLJIT_F32_OP) {
			FAIL_IF(push_inst(compiler, FRSP | FD(dst_r) | FB(src)));
			break;
		}
		/* Fall through. */
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, FMR | FD(dst_r) | FB(src)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, FNEG | FD(dst_r) | FB(src)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, FABS | FD(dst_r) | FB(src)));
		break;
	}

	if (dst & SLJIT_MEM)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw, 0, 0));
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
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w)) {
			FAIL_IF(compiler->error);
			src1 = TMP_FREG1;
		} else
			flags |= ALT_FORM1;
	}

	if (src2 & SLJIT_MEM) {
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w)) {
			FAIL_IF(compiler->error);
			src2 = TMP_FREG2;
		} else
			flags |= ALT_FORM2;
	}

	if ((flags & (ALT_FORM1 | ALT_FORM2)) == (ALT_FORM1 | ALT_FORM2)) {
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, dst, dstw));
		}
	}
	else if (flags & ALT_FORM1)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, dst, dstw));
	else if (flags & ALT_FORM2)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, dst, dstw));

	if (flags & ALT_FORM1)
		src1 = TMP_FREG1;
	if (flags & ALT_FORM2)
		src2 = TMP_FREG2;

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, SELECT_FOP(op, FADDS, FADD) | FD(dst_r) | FA(src1) | FB(src2)));
		break;

	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, SELECT_FOP(op, FSUBS, FSUB) | FD(dst_r) | FA(src1) | FB(src2)));
		break;

	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, SELECT_FOP(op, FMULS, FMUL) | FD(dst_r) | FA(src1) | FC(src2) /* FMUL use FC as src2 */));
		break;

	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, SELECT_FOP(op, FDIVS, FDIV) | FD(dst_r) | FA(src1) | FB(src2)));
		break;
	}

	if (dst_r == TMP_FREG2)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG2, dst, dstw, 0, 0));

	return SLJIT_SUCCESS;
}

#undef FLOAT_DATA
#undef SELECT_FOP

/* --------------------------------------------------------------------- */
/*  Other instructions                                                   */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	/* For UNUSED dst. Uncommon, but possible. */
	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;

	if (FAST_IS_REG(dst))
		return push_inst(compiler, MFLR | D(dst));

	/* Memory. */
	FAIL_IF(push_inst(compiler, MFLR | D(TMP_REG2)));
	return emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, TMP_REG2, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_return(compiler, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, MTLR | S(src)));
	else {
		if (src & SLJIT_MEM)
			FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, TMP_REG2, 0, TMP_REG1, 0, src, srcw));
		else if (src & SLJIT_IMM)
			FAIL_IF(load_immediate(compiler, TMP_REG2, srcw));
		FAIL_IF(push_inst(compiler, MTLR | S(TMP_REG2)));
	}
	return push_inst(compiler, BLR);
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
	return label;
}

static sljit_ins get_bo_bi_flags(sljit_s32 type)
{
	switch (type) {
	case SLJIT_EQUAL:
		return (12 << 21) | (2 << 16);

	case SLJIT_NOT_EQUAL:
		return (4 << 21) | (2 << 16);

	case SLJIT_LESS:
	case SLJIT_LESS_F64:
		return (12 << 21) | ((4 + 0) << 16);

	case SLJIT_GREATER_EQUAL:
	case SLJIT_GREATER_EQUAL_F64:
		return (4 << 21) | ((4 + 0) << 16);

	case SLJIT_GREATER:
	case SLJIT_GREATER_F64:
		return (12 << 21) | ((4 + 1) << 16);

	case SLJIT_LESS_EQUAL:
	case SLJIT_LESS_EQUAL_F64:
		return (4 << 21) | ((4 + 1) << 16);

	case SLJIT_SIG_LESS:
		return (12 << 21) | (0 << 16);

	case SLJIT_SIG_GREATER_EQUAL:
		return (4 << 21) | (0 << 16);

	case SLJIT_SIG_GREATER:
		return (12 << 21) | (1 << 16);

	case SLJIT_SIG_LESS_EQUAL:
		return (4 << 21) | (1 << 16);

	case SLJIT_OVERFLOW:
	case SLJIT_MUL_OVERFLOW:
		return (12 << 21) | (3 << 16);

	case SLJIT_NOT_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		return (4 << 21) | (3 << 16);

	case SLJIT_EQUAL_F64:
		return (12 << 21) | ((4 + 2) << 16);

	case SLJIT_NOT_EQUAL_F64:
		return (4 << 21) | ((4 + 2) << 16);

	case SLJIT_UNORDERED_F64:
		return (12 << 21) | ((4 + 3) << 16);

	case SLJIT_ORDERED_F64:
		return (4 << 21) | ((4 + 3) << 16);

	default:
		SLJIT_ASSERT(type >= SLJIT_JUMP && type <= SLJIT_CALL3);
		return (20 << 21);
	}
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins bo_bi_flags;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	bo_bi_flags = get_bo_bi_flags(type & 0xff);
	if (!bo_bi_flags)
		return NULL;

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	/* In PPC, we don't need to touch the arguments. */
	if (type < SLJIT_JUMP)
		jump->flags |= IS_COND;
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
	if (type >= SLJIT_CALL0)
		jump->flags |= IS_CALL;
#endif

	PTR_FAIL_IF(emit_const(compiler, TMP_CALL_REG, 0));
	PTR_FAIL_IF(push_inst(compiler, MTCTR | S(TMP_CALL_REG)));
	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, BCCTR | bo_bi_flags | (type >= SLJIT_FAST_CALL ? 1 : 0)));
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump = NULL;
	sljit_s32 src_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (FAST_IS_REG(src)) {
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
		if (type >= SLJIT_CALL0) {
			FAIL_IF(push_inst(compiler, OR | S(src) | A(TMP_CALL_REG) | B(src)));
			src_r = TMP_CALL_REG;
		}
		else
			src_r = src;
#else
		src_r = src;
#endif
	} else if (src & SLJIT_IMM) {
		jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
		FAIL_IF(!jump);
		set_jump(jump, compiler, JUMP_ADDR);
		jump->u.target = srcw;
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
		if (type >= SLJIT_CALL0)
			jump->flags |= IS_CALL;
#endif
		FAIL_IF(emit_const(compiler, TMP_CALL_REG, 0));
		src_r = TMP_CALL_REG;
	}
	else {
		FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, TMP_CALL_REG, 0, TMP_REG1, 0, src, srcw));
		src_r = TMP_CALL_REG;
	}

	FAIL_IF(push_inst(compiler, MTCTR | S(src_r)));
	if (jump)
		jump->addr = compiler->size;
	return push_inst(compiler, BCCTR | (20 << 21) | (type >= SLJIT_FAST_CALL ? 1 : 0));
}

/* Get a bit from CR, all other bits are zeroed. */
#define GET_CR_BIT(bit, dst) \
	FAIL_IF(push_inst(compiler, MFCR | D(dst))); \
	FAIL_IF(push_inst(compiler, RLWINM | S(dst) | A(dst) | ((1 + (bit)) << 11) | (31 << 6) | (31 << 1)));

#define INVERT_BIT(dst) \
	FAIL_IF(push_inst(compiler, XORI | S(dst) | A(dst) | 0x1));

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw,
	sljit_s32 type)
{
	sljit_s32 reg, input_flags;
	sljit_s32 flags = GET_ALL_FLAGS(op);
	sljit_sw original_dstw = dstw;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, src, srcw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;

	op = GET_OPCODE(op);
	reg = (op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2;

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	if (op >= SLJIT_ADD && (src & SLJIT_MEM)) {
		ADJUST_LOCAL_OFFSET(src, srcw);
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		input_flags = (flags & SLJIT_I32_OP) ? INT_DATA : WORD_DATA;
#else
		input_flags = WORD_DATA;
#endif
		FAIL_IF(emit_op_mem2(compiler, input_flags | LOAD_DATA, TMP_REG1, src, srcw, dst, dstw));
		src = TMP_REG1;
		srcw = 0;
	}

	switch (type & 0xff) {
	case SLJIT_EQUAL:
		GET_CR_BIT(2, reg);
		break;

	case SLJIT_NOT_EQUAL:
		GET_CR_BIT(2, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_LESS:
	case SLJIT_LESS_F64:
		GET_CR_BIT(4 + 0, reg);
		break;

	case SLJIT_GREATER_EQUAL:
	case SLJIT_GREATER_EQUAL_F64:
		GET_CR_BIT(4 + 0, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_GREATER:
	case SLJIT_GREATER_F64:
		GET_CR_BIT(4 + 1, reg);
		break;

	case SLJIT_LESS_EQUAL:
	case SLJIT_LESS_EQUAL_F64:
		GET_CR_BIT(4 + 1, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_SIG_LESS:
		GET_CR_BIT(0, reg);
		break;

	case SLJIT_SIG_GREATER_EQUAL:
		GET_CR_BIT(0, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_SIG_GREATER:
		GET_CR_BIT(1, reg);
		break;

	case SLJIT_SIG_LESS_EQUAL:
		GET_CR_BIT(1, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_OVERFLOW:
	case SLJIT_MUL_OVERFLOW:
		GET_CR_BIT(3, reg);
		break;

	case SLJIT_NOT_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		GET_CR_BIT(3, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_EQUAL_F64:
		GET_CR_BIT(4 + 2, reg);
		break;

	case SLJIT_NOT_EQUAL_F64:
		GET_CR_BIT(4 + 2, reg);
		INVERT_BIT(reg);
		break;

	case SLJIT_UNORDERED_F64:
		GET_CR_BIT(4 + 3, reg);
		break;

	case SLJIT_ORDERED_F64:
		GET_CR_BIT(4 + 3, reg);
		INVERT_BIT(reg);
		break;

	default:
		SLJIT_ASSERT_STOP();
		break;
	}

	if (op < SLJIT_ADD) {
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op == SLJIT_MOV)
			input_flags = WORD_DATA;
		else {
			op = SLJIT_MOV_U32;
			input_flags = INT_DATA;
		}
#else
		op = SLJIT_MOV;
		input_flags = WORD_DATA;
#endif
		if (reg != TMP_REG2)
			return SLJIT_SUCCESS;
		return emit_op(compiler, op, input_flags, dst, dstw, TMP_REG1, 0, TMP_REG2, 0);
	}

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE) \
		|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->skip_checks = 1;
#endif
	return sljit_emit_op2(compiler, op | flags, dst, original_dstw, src, srcw, TMP_REG2, 0);
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	struct sljit_const *const_;
	sljit_s32 reg;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

	reg = SLOW_IS_REG(dst) ? dst : TMP_REG2;

	PTR_FAIL_IF(emit_const(compiler, reg, init_value));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, TMP_REG2, 0));
	return const_;
}
