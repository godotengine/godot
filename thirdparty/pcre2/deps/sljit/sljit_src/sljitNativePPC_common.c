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

#if (defined _CALL_ELF && _CALL_ELF == 2)
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
#define TMP_ZERO	(SLJIT_NUMBER_OF_REGISTERS + 4)

#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
#define TMP_CALL_REG	(SLJIT_NUMBER_OF_REGISTERS + 5)
#else
#define TMP_CALL_REG	TMP_REG1
#endif

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 7] = {
	0, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 9, 10, 31, 12
};

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3] = {
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 0, 13
};

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */
#define D(d)		((sljit_ins)reg_map[d] << 21)
#define S(s)		((sljit_ins)reg_map[s] << 21)
#define A(a)		((sljit_ins)reg_map[a] << 16)
#define B(b)		((sljit_ins)reg_map[b] << 11)
#define C(c)		((sljit_ins)reg_map[c] << 6)
#define FD(fd)		((sljit_ins)freg_map[fd] << 21)
#define FS(fs)		((sljit_ins)freg_map[fs] << 21)
#define FA(fa)		((sljit_ins)freg_map[fa] << 16)
#define FB(fb)		((sljit_ins)freg_map[fb] << 11)
#define FC(fc)		((sljit_ins)freg_map[fc] << 6)
#define IMM(imm)	((sljit_ins)(imm) & 0xffff)
#define CRD(d)		((sljit_ins)(d) << 21)

/* Instruction bit sections.
   OE and Rc flag (see ALT_SET_FLAGS). */
#define OE(flags)	((flags) & ALT_SET_FLAGS)
/* Rc flag (see ALT_SET_FLAGS). */
#define RC(flags)	((sljit_ins)((flags) & ALT_SET_FLAGS) >> 10)
#define HI(opcode)	((sljit_ins)(opcode) << 26)
#define LO(opcode)	((sljit_ins)(opcode) << 1)

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
#if defined(_ARCH_PWR10) && _ARCH_PWR10
#define BRD		(HI(31) | LO(187))
#endif /* POWER10 */
#define CNTLZD		(HI(31) | LO(58))
#define CNTLZW		(HI(31) | LO(26))
#define CMP		(HI(31) | LO(0))
#define CMPI		(HI(11))
#define CMPL		(HI(31) | LO(32))
#define CMPLI		(HI(10))
#define CROR		(HI(19) | LO(449))
#define DCBT		(HI(31) | LO(278))
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
#define LFD		(HI(50))
#define LFS		(HI(48))
#define LDARX		(HI(31) | LO(84))
#if defined(_ARCH_PWR7) && _ARCH_PWR7
#define LDBRX		(HI(31) | LO(532))
#endif /* POWER7 */
#define LHBRX		(HI(31) | LO(790))
#define LWARX		(HI(31) | LO(20))
#define LWBRX		(HI(31) | LO(534))
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
#define RLDCL		(HI(30) | LO(8))
#define RLDICL		(HI(30) | LO(0 << 1))
#define RLDICR		(HI(30) | LO(1 << 1))
#define RLDIMI		(HI(30) | LO(3 << 1))
#define RLWIMI		(HI(20))
#define RLWINM		(HI(21))
#define RLWNM		(HI(23))
#define SLD		(HI(31) | LO(27))
#define SLW		(HI(31) | LO(24))
#define SRAD		(HI(31) | LO(794))
#define SRADI		(HI(31) | LO(413 << 1))
#define SRAW		(HI(31) | LO(792))
#define SRAWI		(HI(31) | LO(824))
#define SRD		(HI(31) | LO(539))
#define SRW		(HI(31) | LO(536))
#define STD		(HI(62) | 0)
#if defined(_ARCH_PWR7) && _ARCH_PWR7
#define STDBRX		(HI(31) | LO(660))
#endif /* POWER7 */
#define STDCX		(HI(31) | LO(214))
#define STDU		(HI(62) | 1)
#define STDUX		(HI(31) | LO(181))
#define STFD		(HI(54))
#define STFIWX		(HI(31) | LO(983))
#define STFS		(HI(52))
#define STHBRX		(HI(31) | LO(918))
#define STW		(HI(36))
#define STWBRX		(HI(31) | LO(662))
#define STWCX		(HI(31) | LO(150))
#define STWU		(HI(37))
#define STWUX		(HI(31) | LO(183))
#define SUBF		(HI(31) | LO(40))
#define SUBFC		(HI(31) | LO(8))
#define SUBFE		(HI(31) | LO(136))
#define SUBFIC		(HI(8))
#define SYNC		(HI(31) | LO(598))
#define XOR		(HI(31) | LO(316))
#define XORI		(HI(26))
#define XORIS		(HI(27))

#define SIMM_MAX	(0x7fff)
#define SIMM_MIN	(-0x8000)
#define UIMM_MAX	(0xffff)

/* Shift helpers. */
#define RLWI_SH(sh) ((sljit_ins)(sh) << 11)
#define RLWI_MBE(mb, me) (((sljit_ins)(mb) << 6) | ((sljit_ins)(me) << 1))
#define RLDI_SH(sh) ((((sljit_ins)(sh) & 0x1f) << 11) | (((sljit_ins)(sh) & 0x20) >> 4))
#define RLDI_MB(mb) ((((sljit_ins)(mb) & 0x1f) << 6) | ((sljit_ins)(mb) & 0x20))
#define RLDI_ME(me) RLDI_MB(me)

#define SLWI(shift) (RLWINM | RLWI_SH(shift) | RLWI_MBE(0, 31 - (shift)))
#define SLDI(shift) (RLDICR | RLDI_SH(shift) | RLDI_ME(63 - (shift)))
/* shift > 0 */
#define SRWI(shift) (RLWINM | RLWI_SH(32 - (shift)) | RLWI_MBE((shift), 31))
#define SRDI(shift) (RLDICL | RLDI_SH(64 - (shift)) | RLDI_MB(shift))

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
#define SLWI_W(shift) SLWI(shift)
#define TMP_MEM_OFFSET (2 * sizeof(sljit_sw))
#else /* !SLJIT_CONFIG_PPC_32 */
#define SLWI_W(shift) SLDI(shift)
#define TMP_MEM_OFFSET (6 * sizeof(sljit_sw))
#endif /* SLJIT_CONFIG_PPC_32 */

#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define TMP_MEM_OFFSET_LO	(TMP_MEM_OFFSET)
#define TMP_MEM_OFFSET_HI	(TMP_MEM_OFFSET + sizeof(sljit_s32))
#define LWBRX_FIRST_REG		S(TMP_REG1)
#define LWBRX_SECOND_REG	S(dst)
#else /* !SLJIT_LITTLE_ENDIAN */
#define TMP_MEM_OFFSET_LO	(TMP_MEM_OFFSET + sizeof(sljit_s32))
#define TMP_MEM_OFFSET_HI	(TMP_MEM_OFFSET)
#define LWBRX_FIRST_REG		S(dst)
#define LWBRX_SECOND_REG	S(TMP_REG1)
#endif /* SLJIT_LITTLE_ENDIAN */

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_function_context(void** func_ptr, struct sljit_function_context* context, sljit_uw addr, void* func)
{
	sljit_uw* ptrs;

	if (func_ptr)
		*func_ptr = (void*)context;

	ptrs = (sljit_uw*)func;
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

static SLJIT_INLINE sljit_ins* detect_jump_type(struct sljit_jump *jump, sljit_ins *code_ptr, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_uw jump_addr = (sljit_uw)code_ptr;
	sljit_uw orig_addr = jump->addr;
	SLJIT_UNUSED_ARG(executable_offset);

	jump->addr = jump_addr;
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL) && (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	if (jump->flags & (SLJIT_REWRITABLE_JUMP | IS_CALL))
		goto exit;
#else
	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		goto exit;
#endif

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->u.label != NULL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;

		if (jump->u.label->size > orig_addr)
			jump_addr = (sljit_uw)(code + orig_addr);
	}

#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL) && (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (jump->flags & IS_CALL)
		goto keep_address;
#endif

	diff = (sljit_sw)target_addr - (sljit_sw)SLJIT_ADD_EXEC_OFFSET(jump_addr, executable_offset);

	if (jump->flags & IS_COND) {
		if (diff <= 0x7fff && diff >= -0x8000) {
			jump->flags |= PATCH_B;
			return code_ptr;
		}
		if (target_addr <= 0xffff) {
			jump->flags |= PATCH_B | PATCH_ABS_B;
			return code_ptr;
		}

		diff -= SSIZE_OF(ins);
	}

	if (diff <= 0x01ffffff && diff >= -0x02000000) {
		jump->flags |= PATCH_B;
	} else if (target_addr <= 0x01ffffff) {
		jump->flags |= PATCH_B | PATCH_ABS_B;
	}

	if (jump->flags & PATCH_B) {
		if (!(jump->flags & IS_COND))
			return code_ptr;

		code_ptr[0] = BCx | (2 << 2) | ((code_ptr[0] ^ (8 << 21)) & 0x03ff0001);
		code_ptr[1] = Bx;
		jump->addr += sizeof(sljit_ins);
		jump->flags -= IS_COND;
		return code_ptr + 1;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
keep_address:
#endif /* SLJIT_PASS_ENTRY_ADDR_TO_CALL */
	if (target_addr < 0x80000000l) {
		jump->flags |= PATCH_ABS32;
		code_ptr[2] = MTCTR | S(TMP_CALL_REG);
		code_ptr[3] = code_ptr[0];
		return code_ptr + 3;
	}

	if (target_addr < 0x800000000000l) {
		jump->flags |= PATCH_ABS48;
		code_ptr[4] = MTCTR | S(TMP_CALL_REG);
		code_ptr[5] = code_ptr[0];
		return code_ptr + 5;
	}
#endif /* SLJIT_CONFIG_PPC_64 */

exit:
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	code_ptr[2] = MTCTR | S(TMP_CALL_REG);
	code_ptr[3] = code_ptr[0];
#else /* !SLJIT_CONFIG_PPC_32 */
	code_ptr[5] = MTCTR | S(TMP_CALL_REG);
	code_ptr[6] = code_ptr[0];
#endif /* SLJIT_CONFIG_PPC_32 */
	return code_ptr + JUMP_MAX_SIZE - 1;
}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)

static SLJIT_INLINE sljit_sw mov_addr_get_length(struct sljit_jump *jump, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_uw addr;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_ASSERT(jump->flags < ((sljit_uw)5 << JUMP_SIZE_SHIFT));
	if (jump->flags & JUMP_ADDR)
		addr = jump->u.target;
	else
		addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code + jump->u.label->size, executable_offset);

	if (addr < 0x80000000l) {
		SLJIT_ASSERT(jump->flags >= ((sljit_uw)1 << JUMP_SIZE_SHIFT));
		jump->flags |= PATCH_ABS32;
		return 1;
	}

	if (addr < 0x800000000000l) {
		SLJIT_ASSERT(jump->flags >= ((sljit_uw)3 << JUMP_SIZE_SHIFT));
		jump->flags |= PATCH_ABS48;
		return 3;
	}

	SLJIT_ASSERT(jump->flags >= ((sljit_uw)4 << JUMP_SIZE_SHIFT));
	return 4;
}

#endif /* SLJIT_CONFIG_PPC_64 */

static void generate_jump_or_mov_addr(struct sljit_jump *jump, sljit_sw executable_offset)
{
	sljit_uw flags = jump->flags;
	sljit_uw addr = (flags & JUMP_ADDR) ? jump->u.target : jump->u.label->u.addr;
	sljit_ins *ins = (sljit_ins*)jump->addr;
	sljit_s32 reg;
	SLJIT_UNUSED_ARG(executable_offset);

	if (flags & PATCH_B) {
		if (flags & IS_COND) {
			if (!(flags & PATCH_ABS_B)) {
				addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET(ins, executable_offset);
				SLJIT_ASSERT((sljit_sw)addr <= 0x7fff && (sljit_sw)addr >= -0x8000);
				ins[0] = BCx | ((sljit_ins)addr & 0xfffc) | (ins[0] & 0x03ff0001);
			} else {
				SLJIT_ASSERT(addr <= 0xffff);
				ins[0] = BCx | ((sljit_ins)addr & 0xfffc) | 0x2 | ((*ins) & 0x03ff0001);
			}
			return;
		}

		if (!(flags & PATCH_ABS_B)) {
			addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET(ins, executable_offset);
			SLJIT_ASSERT((sljit_sw)addr <= 0x01ffffff && (sljit_sw)addr >= -0x02000000);
			ins[0] = Bx | ((sljit_ins)addr & 0x03fffffc) | (ins[0] & 0x1);
		} else {
			SLJIT_ASSERT(addr <= 0x03ffffff);
			ins[0] = Bx | ((sljit_ins)addr & 0x03fffffc) | 0x2 | (ins[0] & 0x1);
		}
		return;
	}

	reg = (flags & JUMP_MOV_ADDR) ? (sljit_s32)ins[0] : TMP_CALL_REG;

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	ins[0] = ADDIS | D(reg) | A(0) | IMM(addr >> 16);
	ins[1] = ORI | S(reg) | A(reg) | IMM(addr);
#else /* !SLJIT_CONFIG_PPC_32 */

	/* The TMP_ZERO cannot be used because it is restored for tail calls. */
	if (flags & PATCH_ABS32) {
		SLJIT_ASSERT(addr < 0x80000000l);
		ins[0] = ADDIS | D(reg) | A(0) | IMM(addr >> 16);
		ins[1] = ORI | S(reg) | A(reg) | IMM(addr);
		return;
	}

	if (flags & PATCH_ABS48) {
		SLJIT_ASSERT(addr < 0x800000000000l);
		ins[0] = ADDIS | D(reg) | A(0) | IMM(addr >> 32);
		ins[1] = ORI | S(reg) | A(reg) | IMM(addr >> 16);
		ins[2] = SLDI(16) | S(reg) | A(reg);
		ins[3] = ORI | S(reg) | A(reg) | IMM(addr);
		return;
	}

	ins[0] = ADDIS | D(reg) | A(0) | IMM(addr >> 48);
	ins[1] = ORI | S(reg) | A(reg) | IMM(addr >> 32);
	ins[2] = SLDI(32) | S(reg) | A(reg);
	ins[3] = ORIS | S(reg) | A(reg) | IMM(addr >> 16);
	ins[4] = ORI | S(reg) | A(reg) | IMM(addr);
#endif /* SLJIT_CONFIG_PPC_32 */
}

static void reduce_code_size(struct sljit_compiler *compiler)
{
	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;
	SLJIT_NEXT_DEFINE_TYPES;
	sljit_uw total_size;
	sljit_uw size_reduce = 0;
	sljit_sw diff;

	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	SLJIT_NEXT_INIT_TYPES();

	while (1) {
		SLJIT_GET_NEXT_MIN();

		if (next_min_addr == SLJIT_MAX_ADDRESS)
			break;

		if (next_min_addr == next_label_size) {
			label->size -= size_reduce;

			label = label->next;
			next_label_size = SLJIT_GET_NEXT_SIZE(label);
		}

		if (next_min_addr == next_const_addr) {
			const_->addr -= size_reduce;
			const_ = const_->next;
			next_const_addr = SLJIT_GET_NEXT_ADDRESS(const_);
			continue;
		}

		if (next_min_addr != next_jump_addr)
			continue;

		jump->addr -= size_reduce;
		if (!(jump->flags & JUMP_MOV_ADDR)) {
			total_size = JUMP_MAX_SIZE - 1;

			if (!(jump->flags & SLJIT_REWRITABLE_JUMP)) {
				if (jump->flags & JUMP_ADDR) {
					if (jump->u.target <= 0x01ffffff)
						total_size = 1 - 1;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
					else if (jump->u.target < 0x80000000l)
						total_size = 4 - 1;
					else if (jump->u.target < 0x800000000000l)
						total_size = 6 - 1;
#endif /* SLJIT_CONFIG_PPC_64 */
				} else {
					/* Unit size: instruction. */
					diff = (sljit_sw)jump->u.label->size - (sljit_sw)jump->addr;
					if (jump->u.label->size > jump->addr) {
						SLJIT_ASSERT(jump->u.label->size - size_reduce >= jump->addr);
						diff -= (sljit_sw)size_reduce;
					}

					if (jump->flags & IS_COND) {
						if (diff <= (0x7fff / SSIZE_OF(ins)) && diff >= (-0x8000 / SSIZE_OF(ins)))
							total_size = 1 - 1;
						else if ((diff - 1) <= (0x01ffffff / SSIZE_OF(ins)) && (diff - 1) >= (-0x02000000 / SSIZE_OF(ins)))
							total_size = 2 - 1;
					} else if (diff <= (0x01ffffff / SSIZE_OF(ins)) && diff >= (-0x02000000 / SSIZE_OF(ins)))
						total_size = 1 - 1;
				}
			}

			size_reduce += (JUMP_MAX_SIZE - 1) - total_size;
			jump->flags |= total_size << JUMP_SIZE_SHIFT;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		} else {
			total_size = (sljit_uw)4 << JUMP_SIZE_SHIFT;

			if (jump->flags & JUMP_ADDR) {
				if (jump->u.target < 0x80000000l) {
					total_size = (sljit_uw)1 << JUMP_SIZE_SHIFT;
					size_reduce += 3;
				} else if (jump->u.target < 0x800000000000l) {
					total_size = (sljit_uw)3 << JUMP_SIZE_SHIFT;
					size_reduce += 1;
				}
			}
			jump->flags |= total_size;
#endif /* SLJIT_CONFIG_PPC_64 */
		}

		jump = jump->next;
		next_jump_addr = SLJIT_GET_NEXT_ADDRESS(jump);
	}

	compiler->size -= size_reduce;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler, sljit_s32 options, void *exec_allocator_data)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw word_count;
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
	sljit_uw jump_addr;
#endif
	SLJIT_NEXT_DEFINE_TYPES;
	sljit_sw executable_offset;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));

	reduce_code_size(compiler);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
	/* add to compiler->size additional instruction space to hold the trampoline and padding */
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	compiler->size += (compiler->size & 0x1) + (sizeof(struct sljit_function_context) / sizeof(sljit_ins));
#else
	compiler->size += (sizeof(struct sljit_function_context) / sizeof(sljit_ins));
#endif
#endif
	code = (sljit_ins*)allocate_executable_memory(compiler->size * sizeof(sljit_ins), options, exec_allocator_data, &executable_offset);
	PTR_FAIL_WITH_EXEC_IF(code);

	reverse_buf(compiler);
	buf = compiler->buf;

	code_ptr = code;
	word_count = 0;
	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	SLJIT_NEXT_INIT_TYPES();
	SLJIT_GET_NEXT_MIN();

	do {
		buf_ptr = (sljit_ins*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			*code_ptr = *buf_ptr++;
			if (next_min_addr == word_count) {
				SLJIT_ASSERT(!label || label->size >= word_count);
				SLJIT_ASSERT(!jump || jump->addr >= word_count);
				SLJIT_ASSERT(!const_ || const_->addr >= word_count);

				/* These structures are ordered by their address. */
				if (next_min_addr == next_label_size) {
					/* Just recording the address. */
					label->u.addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					label->size = (sljit_uw)(code_ptr - code);
					label = label->next;
					next_label_size = SLJIT_GET_NEXT_SIZE(label);
				}

				if (next_min_addr == next_jump_addr) {
					if (!(jump->flags & JUMP_MOV_ADDR)) {
						word_count += jump->flags >> JUMP_SIZE_SHIFT;
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
						jump_addr = (sljit_uw)code_ptr;
#endif
						code_ptr = detect_jump_type(jump, code_ptr, code, executable_offset);
						SLJIT_ASSERT(((sljit_uw)code_ptr - jump_addr <= (jump->flags >> JUMP_SIZE_SHIFT) * sizeof(sljit_ins)));
					} else {
						jump->addr = (sljit_uw)code_ptr;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
						word_count += jump->flags >> JUMP_SIZE_SHIFT;
						code_ptr += mov_addr_get_length(jump, code, executable_offset);
#else /* !SLJIT_CONFIG_PPC_64 */
						word_count++;
						code_ptr++;
#endif /* SLJIT_CONFIG_PPC_64 */
					}
					jump = jump->next;
					next_jump_addr = SLJIT_GET_NEXT_ADDRESS(jump);
				} else if (next_min_addr == next_const_addr) {
					const_->addr = (sljit_uw)code_ptr;
					const_ = const_->next;
					next_const_addr = SLJIT_GET_NEXT_ADDRESS(const_);
				}

				SLJIT_GET_NEXT_MIN();
			}
			code_ptr++;
			word_count++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		label->u.addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
		label->size = (sljit_uw)(code_ptr - code);
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)(compiler->size - (sizeof(struct sljit_function_context) / sizeof(sljit_ins))));
#else
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);
#endif

	jump = compiler->jumps;
	while (jump) {
		generate_jump_or_mov_addr(jump, executable_offset);
		jump = jump->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;

	code = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (((sljit_sw)code_ptr) & 0x4)
		code_ptr++;
#endif
	sljit_set_function_context(NULL, (struct sljit_function_context*)code_ptr, (sljit_uw)code, (void*)sljit_generate_code);
#endif

	code_ptr = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

	SLJIT_CACHE_FLUSH(code, code_ptr);
	SLJIT_UPDATE_WX_FLAGS(code, code_ptr, 1);

#if (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
	compiler->executable_size = (sljit_uw)(code_ptr - code) * sizeof(sljit_ins) + sizeof(struct sljit_function_context);
	return code_ptr;
#else
	compiler->executable_size = (sljit_uw)(code_ptr - code) * sizeof(sljit_ins);
	return code;
#endif
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	switch (feature_type) {
	case SLJIT_HAS_FPU:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#else
		/* Available by default. */
		return 1;
#endif
	case SLJIT_HAS_REV:
#if defined(_ARCH_PWR10) && _ARCH_PWR10
		return 1;
#else /* !POWER10 */
		return 2;
#endif /* POWER10 */
	/* A saved register is set to a zero value. */
	case SLJIT_HAS_ZERO_REGISTER:
	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_ROT:
	case SLJIT_HAS_PREFETCH:
	case SLJIT_HAS_ATOMIC:
	case SLJIT_HAS_MEMORY_BARRIER:
		return 1;

	case SLJIT_HAS_CTZ:
		return 2;

	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	switch (type) {
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_ORDERED_LESS_EQUAL:
		return 1;
	}

	return 0;
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

/* inp_flags: */

/* Creates an index in data_transfer_insts array. */
#define LOAD_DATA	0x01
#define INDEXED		0x02
#define SIGNED_DATA	0x04

#define WORD_DATA	0x00
#define BYTE_DATA	0x08
#define HALF_DATA	0x10
#define INT_DATA	0x18
/* Separates integer and floating point registers */
#define GPR_REG		0x1f
#define DOUBLE_DATA	0x20

#define MEM_MASK	0x7f

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_32) >> 6))

/* Other inp_flags. */

/* Integer opertion and set flags -> requires exts on 64 bit systems. */
#define ALT_SIGN_EXT	0x000100
/* This flag affects the RC() and OERC() macros. */
#define ALT_SET_FLAGS	0x000400
#define ALT_FORM1	0x001000
#define ALT_FORM2	0x002000
#define ALT_FORM3	0x004000
#define ALT_FORM4	0x008000
#define ALT_FORM5	0x010000

/* Source and destination is register. */
#define REG_DEST	0x000001
#define REG1_SOURCE	0x000002
#define REG2_SOURCE	0x000004
/*
ALT_SIGN_EXT		0x000100
ALT_SET_FLAGS		0x000200
ALT_FORM1		0x001000
...
ALT_FORM5		0x010000 */

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 inp_flags, sljit_s32 reg,
	sljit_s32 arg, sljit_sw argw, sljit_s32 tmp_reg);

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

#if (defined SLJIT_PPC_STACK_FRAME_V2 && SLJIT_PPC_STACK_FRAME_V2)
#define LR_SAVE_OFFSET		(2 * SSIZE_OF(sw))
#else
#define LR_SAVE_OFFSET		SSIZE_OF(sw)
#endif

#define STACK_MAX_DISTANCE	(0x8000 - SSIZE_OF(sw) - LR_SAVE_OFFSET)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	sljit_s32 fscratches = ENTER_GET_FLOAT_REGS(scratches);
	sljit_s32 fsaveds = ENTER_GET_FLOAT_REGS(saveds);
	sljit_s32 i, tmp, base, offset;
	sljit_s32 word_arg_count = 0;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 arg_count = 0;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, local_size);

	scratches = ENTER_GET_REGS(scratches);
	saveds = ENTER_GET_REGS(saveds);
	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 0)
		+ GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);

	if (!(options & SLJIT_ENTER_REG_ARG))
		local_size += SSIZE_OF(sw);

	local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
	compiler->local_size = local_size;

	FAIL_IF(push_inst(compiler, MFLR | D(0)));

	base = SLJIT_SP;
	offset = local_size;

	if (local_size <= STACK_MAX_DISTANCE) {
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
		FAIL_IF(push_inst(compiler, STWU | S(SLJIT_SP) | A(SLJIT_SP) | IMM(-local_size)));
#else
		FAIL_IF(push_inst(compiler, STDU | S(SLJIT_SP) | A(SLJIT_SP) | IMM(-local_size)));
#endif
	} else {
		base = TMP_REG1;
		FAIL_IF(push_inst(compiler, OR | S(SLJIT_SP) | A(TMP_REG1) | B(SLJIT_SP)));
		FAIL_IF(load_immediate(compiler, TMP_REG2, -local_size));
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
		FAIL_IF(push_inst(compiler, STWUX | S(SLJIT_SP) | A(SLJIT_SP) | B(TMP_REG2)));
#else
		FAIL_IF(push_inst(compiler, STDUX | S(SLJIT_SP) | A(SLJIT_SP) | B(TMP_REG2)));
#endif
		local_size = 0;
		offset = 0;
	}

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, STFD | FS(i) | A(base) | IMM(offset)));
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, STFD | FS(i) | A(base) | IMM(offset)));
	}

	if (!(options & SLJIT_ENTER_REG_ARG)) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_STORE | S(TMP_ZERO) | A(base) | IMM(offset)));
	}

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_STORE | S(i) | A(base) | IMM(offset)));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_STORE | S(i) | A(base) | IMM(offset)));
	}

	FAIL_IF(push_inst(compiler, STACK_STORE | S(0) | A(base) | IMM(local_size + LR_SAVE_OFFSET)));

	if (options & SLJIT_ENTER_REG_ARG)
		return SLJIT_SUCCESS;

	FAIL_IF(push_inst(compiler, ADDI | D(TMP_ZERO) | A(0) | 0));

	arg_types >>= SLJIT_ARG_SHIFT;
	saved_arg_count = 0;

	while (arg_types > 0) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
			do {
				if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
					tmp = SLJIT_S0 - saved_arg_count;
					saved_arg_count++;
				} else if (arg_count != word_arg_count)
					tmp = SLJIT_R0 + word_arg_count;
				else
					break;

				FAIL_IF(push_inst(compiler, OR | S(SLJIT_R0 + arg_count) | A(tmp) | B(SLJIT_R0 + arg_count)));
			} while (0);
#else
			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				FAIL_IF(push_inst(compiler, OR | S(SLJIT_R0 + word_arg_count) | A(SLJIT_S0 - saved_arg_count) | B(SLJIT_R0 + word_arg_count)));
				saved_arg_count++;
			}
#endif
			word_arg_count++;
		}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		arg_count++;
#endif
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	sljit_s32 fscratches = ENTER_GET_FLOAT_REGS(scratches);
	sljit_s32 fsaveds = ENTER_GET_FLOAT_REGS(saveds);

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, local_size);

	scratches = ENTER_GET_REGS(scratches);
	saveds = ENTER_GET_REGS(saveds);
	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 0)
		+ GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);

	if (!(options & SLJIT_ENTER_REG_ARG))
		local_size += SSIZE_OF(sw);

	compiler->local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_s32 i, tmp, base, offset;
	sljit_s32 local_size = compiler->local_size;

	SLJIT_ASSERT(TMP_CALL_REG != TMP_REG2);

	base = SLJIT_SP;
	if (local_size > STACK_MAX_DISTANCE) {
		base = TMP_REG2;
		if (local_size > 2 * STACK_MAX_DISTANCE + LR_SAVE_OFFSET) {
			FAIL_IF(push_inst(compiler, STACK_LOAD | D(base) | A(SLJIT_SP) | IMM(0)));
			local_size = 0;
		} else {
			FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(SLJIT_SP) | IMM(local_size - STACK_MAX_DISTANCE)));
			local_size = STACK_MAX_DISTANCE;
		}
	}

	offset = local_size;
	if (!is_return_to)
		FAIL_IF(push_inst(compiler, STACK_LOAD | S(0) | A(base) | IMM(offset + LR_SAVE_OFFSET)));

	tmp = SLJIT_FS0 - compiler->fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, LFD | FS(i) | A(base) | IMM(offset)));
	}

	for (i = compiler->fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, LFD | FS(i) | A(base) | IMM(offset)));
	}

	if (!(compiler->options & SLJIT_ENTER_REG_ARG)) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | S(TMP_ZERO) | A(base) | IMM(offset)));
	}

	tmp = SLJIT_S0 - compiler->saveds;
	for (i = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options); i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | S(i) | A(base) | IMM(offset)));
	}

	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | S(i) | A(base) | IMM(offset)));
	}

	if (!is_return_to)
		push_inst(compiler, MTLR | S(0));

	if (local_size > 0)
		return push_inst(compiler, ADDI | D(SLJIT_SP) | A(base) | IMM(local_size));

	SLJIT_ASSERT(base == TMP_REG2);
	return push_inst(compiler, OR | S(base) | A(SLJIT_SP) | B(base));
}

#undef STACK_STORE
#undef STACK_LOAD

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	FAIL_IF(emit_stack_frame_release(compiler, 0));
	return push_inst(compiler, BLR);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_CALL_REG, src, srcw, TMP_CALL_REG));
		src = TMP_CALL_REG;
		srcw = 0;
	} else if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		FAIL_IF(push_inst(compiler, OR | S(src) | A(TMP_CALL_REG) | B(src)));
		src = TMP_CALL_REG;
		srcw = 0;
	}

	FAIL_IF(emit_stack_frame_release(compiler, 1));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

/* s/l - store/load (1 bit)
   i/x - immediate/indexed form
   u/s - signed/unsigned (1 bit)
   w/b/h/i - word/byte/half/int allowed (2 bit)

   Some opcodes are repeated (e.g. store signed / unsigned byte is the same instruction). */

/* 64 bit only: [reg+imm] must be aligned to 4 bytes. */
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define INT_ALIGNED	0x10000
#endif

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
#define ARCH_32_64(a, b)	a
#define INST_CODE_AND_DST(inst, flags, reg) \
	((sljit_ins)(inst) | (sljit_ins)(((flags) & MEM_MASK) <= GPR_REG ? D(reg) : FD(reg)))
#else
#define ARCH_32_64(a, b)	b
#define INST_CODE_AND_DST(inst, flags, reg) \
	(((sljit_ins)(inst) & ~(sljit_ins)INT_ALIGNED) | (sljit_ins)(((flags) & MEM_MASK) <= GPR_REG ? D(reg) : FD(reg)))
#endif

static const sljit_ins data_transfer_insts[64 + 16] = {

/* -------- Integer -------- */

/* Word. */

/* w u i s */ ARCH_32_64(HI(36) /* stw */, HI(62) | INT_ALIGNED | 0x0 /* std */),
/* w u i l */ ARCH_32_64(HI(32) /* lwz */, HI(58) | INT_ALIGNED | 0x0 /* ld */),
/* w u x s */ ARCH_32_64(HI(31) | LO(151) /* stwx */, HI(31) | LO(149) /* stdx */),
/* w u x l */ ARCH_32_64(HI(31) | LO(23) /* lwzx */, HI(31) | LO(21) /* ldx */),

/* w s i s */ ARCH_32_64(HI(36) /* stw */, HI(62) | INT_ALIGNED | 0x0 /* std */),
/* w s i l */ ARCH_32_64(HI(32) /* lwz */, HI(58) | INT_ALIGNED | 0x0 /* ld */),
/* w s x s */ ARCH_32_64(HI(31) | LO(151) /* stwx */, HI(31) | LO(149) /* stdx */),
/* w s x l */ ARCH_32_64(HI(31) | LO(23) /* lwzx */, HI(31) | LO(21) /* ldx */),

/* Byte. */

/* b u i s */ HI(38) /* stb */,
/* b u i l */ HI(34) /* lbz */,
/* b u x s */ HI(31) | LO(215) /* stbx */,
/* b u x l */ HI(31) | LO(87) /* lbzx */,

/* b s i s */ HI(38) /* stb */,
/* b s i l */ HI(34) /* lbz */ /* EXTS_REQ */,
/* b s x s */ HI(31) | LO(215) /* stbx */,
/* b s x l */ HI(31) | LO(87) /* lbzx */ /* EXTS_REQ */,

/* Half. */

/* h u i s */ HI(44) /* sth */,
/* h u i l */ HI(40) /* lhz */,
/* h u x s */ HI(31) | LO(407) /* sthx */,
/* h u x l */ HI(31) | LO(279) /* lhzx */,

/* h s i s */ HI(44) /* sth */,
/* h s i l */ HI(42) /* lha */,
/* h s x s */ HI(31) | LO(407) /* sthx */,
/* h s x l */ HI(31) | LO(343) /* lhax */,

/* Int. */

/* i u i s */ HI(36) /* stw */,
/* i u i l */ HI(32) /* lwz */,
/* i u x s */ HI(31) | LO(151) /* stwx */,
/* i u x l */ HI(31) | LO(23) /* lwzx */,

/* i s i s */ HI(36) /* stw */,
/* i s i l */ ARCH_32_64(HI(32) /* lwz */, HI(58) | INT_ALIGNED | 0x2 /* lwa */),
/* i s x s */ HI(31) | LO(151) /* stwx */,
/* i s x l */ ARCH_32_64(HI(31) | LO(23) /* lwzx */, HI(31) | LO(341) /* lwax */),

/* -------- Floating point -------- */

/* d   i s */ HI(54) /* stfd */,
/* d   i l */ HI(50) /* lfd */,
/* d   x s */ HI(31) | LO(727) /* stfdx */,
/* d   x l */ HI(31) | LO(599) /* lfdx */,

/* s   i s */ HI(52) /* stfs */,
/* s   i l */ HI(48) /* lfs */,
/* s   x s */ HI(31) | LO(663) /* stfsx */,
/* s   x l */ HI(31) | LO(535) /* lfsx */,
};

static const sljit_ins updated_data_transfer_insts[64] = {

/* -------- Integer -------- */

/* Word. */

/* w u i s */ ARCH_32_64(HI(37) /* stwu */, HI(62) | INT_ALIGNED | 0x1 /* stdu */),
/* w u i l */ ARCH_32_64(HI(33) /* lwzu */, HI(58) | INT_ALIGNED | 0x1 /* ldu */),
/* w u x s */ ARCH_32_64(HI(31) | LO(183) /* stwux */, HI(31) | LO(181) /* stdux */),
/* w u x l */ ARCH_32_64(HI(31) | LO(55) /* lwzux */, HI(31) | LO(53) /* ldux */),

/* w s i s */ ARCH_32_64(HI(37) /* stwu */, HI(62) | INT_ALIGNED | 0x1 /* stdu */),
/* w s i l */ ARCH_32_64(HI(33) /* lwzu */, HI(58) | INT_ALIGNED | 0x1 /* ldu */),
/* w s x s */ ARCH_32_64(HI(31) | LO(183) /* stwux */, HI(31) | LO(181) /* stdux */),
/* w s x l */ ARCH_32_64(HI(31) | LO(55) /* lwzux */, HI(31) | LO(53) /* ldux */),

/* Byte. */

/* b u i s */ HI(39) /* stbu */,
/* b u i l */ HI(35) /* lbzu */,
/* b u x s */ HI(31) | LO(247) /* stbux */,
/* b u x l */ HI(31) | LO(119) /* lbzux */,

/* b s i s */ HI(39) /* stbu */,
/* b s i l */ 0 /* no such instruction */,
/* b s x s */ HI(31) | LO(247) /* stbux */,
/* b s x l */ 0 /* no such instruction */,

/* Half. */

/* h u i s */ HI(45) /* sthu */,
/* h u i l */ HI(41) /* lhzu */,
/* h u x s */ HI(31) | LO(439) /* sthux */,
/* h u x l */ HI(31) | LO(311) /* lhzux */,

/* h s i s */ HI(45) /* sthu */,
/* h s i l */ HI(43) /* lhau */,
/* h s x s */ HI(31) | LO(439) /* sthux */,
/* h s x l */ HI(31) | LO(375) /* lhaux */,

/* Int. */

/* i u i s */ HI(37) /* stwu */,
/* i u i l */ HI(33) /* lwzu */,
/* i u x s */ HI(31) | LO(183) /* stwux */,
/* i u x l */ HI(31) | LO(55) /* lwzux */,

/* i s i s */ HI(37) /* stwu */,
/* i s i l */ ARCH_32_64(HI(33) /* lwzu */, 0 /* no such instruction */),
/* i s x s */ HI(31) | LO(183) /* stwux */,
/* i s x l */ ARCH_32_64(HI(31) | LO(55) /* lwzux */, HI(31) | LO(373) /* lwaux */),

/* -------- Floating point -------- */

/* d   i s */ HI(55) /* stfdu */,
/* d   i l */ HI(51) /* lfdu */,
/* d   x s */ HI(31) | LO(759) /* stfdux */,
/* d   x l */ HI(31) | LO(631) /* lfdux */,

/* s   i s */ HI(53) /* stfsu */,
/* s   i l */ HI(49) /* lfsu */,
/* s   x s */ HI(31) | LO(695) /* stfsux */,
/* s   x l */ HI(31) | LO(567) /* lfsux */,
};

#undef ARCH_32_64

/* Simple cases, (no caching is required). */
static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 inp_flags, sljit_s32 reg,
	sljit_s32 arg, sljit_sw argw, sljit_s32 tmp_reg)
{
	sljit_ins inst;
	sljit_s32 offs_reg;

	/* Should work when (arg & REG_MASK) == 0. */
	SLJIT_ASSERT(A(0) == 0);
	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;
		offs_reg = OFFS_REG(arg);

		if (argw != 0) {
			FAIL_IF(push_inst(compiler, SLWI_W(argw) | S(OFFS_REG(arg)) | A(tmp_reg)));
			offs_reg = tmp_reg;
		}

		inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		SLJIT_ASSERT(!(inst & INT_ALIGNED));
#endif /* SLJIT_CONFIG_PPC_64 */

		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg & REG_MASK) | B(offs_reg));
	}

	inst = data_transfer_insts[inp_flags & MEM_MASK];
	arg &= REG_MASK;

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if ((inst & INT_ALIGNED) && (argw & 0x3) != 0) {
		FAIL_IF(load_immediate(compiler, tmp_reg, argw));

		inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg) | B(tmp_reg));
	}
#endif /* SLJIT_CONFIG_PPC_64 */

	if (argw <= SIMM_MAX && argw >= SIMM_MIN)
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg) | IMM(argw));

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (argw <= 0x7fff7fffl && argw >= -0x80000000l) {
#endif /* SLJIT_CONFIG_PPC_64 */
		FAIL_IF(push_inst(compiler, ADDIS | D(tmp_reg) | A(arg) | IMM((argw + 0x8000) >> 16)));
		return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(tmp_reg) | IMM(argw));
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	}

	FAIL_IF(load_immediate(compiler, tmp_reg, argw));

	inst = data_transfer_insts[(inp_flags | INDEXED) & MEM_MASK];
	return push_inst(compiler, INST_CODE_AND_DST(inst, inp_flags, reg) | A(arg) | B(tmp_reg));
#endif /* SLJIT_CONFIG_PPC_64 */
}

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 input_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* arg1 goes to TMP_REG1 or src reg
	   arg2 goes to TMP_REG2, imm or src reg
	   result goes to TMP_REG2, so put result can use TMP_REG1. */
	sljit_s32 dst_r = TMP_REG2;
	sljit_s32 src1_r;
	sljit_s32 src2_r;
	sljit_s32 src2_tmp_reg = (!(input_flags & ALT_SIGN_EXT) && GET_OPCODE(op) >= SLJIT_OP2_BASE && FAST_IS_REG(src1)) ? TMP_REG1 : TMP_REG2;
	sljit_s32 flags = input_flags & (ALT_FORM1 | ALT_FORM2 | ALT_FORM3 | ALT_FORM4 | ALT_FORM5 | ALT_SIGN_EXT | ALT_SET_FLAGS);

	/* Destination check. */
	if (FAST_IS_REG(dst)) {
		dst_r = dst;
		/* The REG_DEST is only used by SLJIT_MOV operations, although
		 * it is set for op2 operations with unset destination. */
		flags |= REG_DEST;

		if (op >= SLJIT_MOV && op <= SLJIT_MOV_P)
			src2_tmp_reg = dst_r;
	}

	/* Source 2. */
	if (FAST_IS_REG(src2)) {
		src2_r = src2;
		flags |= REG2_SOURCE;

		if (!(flags & REG_DEST) && op >= SLJIT_MOV && op <= SLJIT_MOV_P)
			dst_r = src2_r;
	} else if (src2 == SLJIT_IMM) {
		src2_r = TMP_ZERO;
		if (src2w != 0) {
			FAIL_IF(load_immediate(compiler, src2_tmp_reg, src2w));
			src2_r = src2_tmp_reg;
		}
	} else {
		FAIL_IF(emit_op_mem(compiler, input_flags | LOAD_DATA, src2_tmp_reg, src2, src2w, TMP_REG1));
		src2_r = src2_tmp_reg;
	}

	/* Source 1. */
	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	} else if (src1 == SLJIT_IMM) {
		src1_r = TMP_ZERO;
		if (src1w != 0) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, src1w));
			src1_r = TMP_REG1;
		}
	} else {
		FAIL_IF(emit_op_mem(compiler, input_flags | LOAD_DATA, TMP_REG1, src1, src1w, TMP_REG1));
		src1_r = TMP_REG1;
	}

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (!(dst & SLJIT_MEM))
		return SLJIT_SUCCESS;

	return emit_op_mem(compiler, input_flags, dst_r, dst, dstw, TMP_REG1);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 int_op = op & SLJIT_32;
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
	case SLJIT_MEMORY_BARRIER:
		return push_inst(compiler, SYNC);
	case SLJIT_ENDBR:
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_rev(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 mem, offs_reg, inp_flags;
	sljit_sw memw;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 is_32 = op & SLJIT_32;

	op = GET_OPCODE(op);
#endif /* SLJIT_CONFIG_PPC_64 */

	if (!((dst | src) & SLJIT_MEM)) {
		/* Both are registers. */
		if (op == SLJIT_REV_U16 || op == SLJIT_REV_S16) {
			if (src == dst) {
				FAIL_IF(push_inst(compiler, RLWIMI | S(dst) | A(dst) | RLWI_SH(16) | RLWI_MBE(8, 15)));
				FAIL_IF(push_inst(compiler, RLWINM | S(dst) | A(dst) | RLWI_SH(24) | RLWI_MBE(16, 31)));
			} else {
				FAIL_IF(push_inst(compiler, RLWINM | S(src) | A(dst) | RLWI_SH(8) | RLWI_MBE(16, 23)));
				FAIL_IF(push_inst(compiler, RLWIMI | S(src) | A(dst) | RLWI_SH(24) | RLWI_MBE(24, 31)));
			}

			if (op == SLJIT_REV_U16)
				return SLJIT_SUCCESS;
			return push_inst(compiler, EXTSH | S(dst) | A(dst));
		}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (!is_32) {
#if defined(_ARCH_PWR10) && _ARCH_PWR10
			return push_inst(compiler, BRD | S(src) | A(dst));
#else /* !POWER10 */
			FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(0) | IMM(TMP_MEM_OFFSET_HI)));
			FAIL_IF(push_inst(compiler, RLDICL | S(src) | A(TMP_REG1) | RLDI_SH(32) | RLDI_MB(32)));
			FAIL_IF(push_inst(compiler, STWBRX | S(src) | A(SLJIT_SP) | B(TMP_REG2)));
			FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(0) | IMM(TMP_MEM_OFFSET_LO)));
			FAIL_IF(push_inst(compiler, STWBRX | S(TMP_REG1) | A(SLJIT_SP) | B(TMP_REG2)));
			return push_inst(compiler, LD | D(dst) | A(SLJIT_SP) | TMP_MEM_OFFSET);
#endif /* POWER10 */
		}
#endif /* SLJIT_CONFIG_PPC_64 */

		FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(0) | IMM(TMP_MEM_OFFSET)));
		FAIL_IF(push_inst(compiler, STWBRX | S(src) | A(SLJIT_SP) | B(TMP_REG2)));
		FAIL_IF(push_inst(compiler, LWZ | D(dst) | A(SLJIT_SP) | TMP_MEM_OFFSET));

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op == SLJIT_REV_S32)
			return push_inst(compiler, EXTSW | S(dst) | A(dst));
#endif /* SLJIT_CONFIG_PPC_64 */
		return SLJIT_SUCCESS;
	}

	mem = src;
	memw = srcw;

	if (dst & SLJIT_MEM) {
		mem = dst;
		memw = dstw;

		if (src & SLJIT_MEM) {
			inp_flags = HALF_DATA | LOAD_DATA;

			if (op != SLJIT_REV_U16 && op != SLJIT_REV_S16) {
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
				inp_flags = (is_32 ? INT_DATA : WORD_DATA) | LOAD_DATA;
#else /* !SLJIT_CONFIG_PPC_64 */
				inp_flags = WORD_DATA | LOAD_DATA;
#endif /* SLJIT_CONFIG_PPC_64 */
			}

			FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG1, src, srcw, TMP_REG2));
			src = TMP_REG1;
		}
	}

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		offs_reg = OFFS_REG(mem);
		mem &= REG_MASK;
		memw &= 0x3;

		if (memw != 0) {
			FAIL_IF(push_inst(compiler, SLWI_W(memw) | S(offs_reg) | A(TMP_REG2)));
			offs_reg = TMP_REG2;
		}
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	} else if (memw > 0x7fff7fffl || memw < -0x80000000l) {
		FAIL_IF(load_immediate(compiler, TMP_REG2, memw));
		offs_reg = TMP_REG2;
		mem &= REG_MASK;
#endif /* SLJIT_CONFIG_PPC_64 */
	} else {
		FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(mem & REG_MASK) | IMM(memw)));
		if (memw > SIMM_MAX || memw < SIMM_MIN)
			FAIL_IF(push_inst(compiler, ADDIS | D(TMP_REG2) | A(TMP_REG2) | IMM((memw + 0x8000) >> 16)));

		mem = 0;
		offs_reg = TMP_REG2;
	}

	if (op == SLJIT_REV_U16 || op == SLJIT_REV_S16) {
		if (dst & SLJIT_MEM)
			return push_inst(compiler, STHBRX | S(src) | A(mem) | B(offs_reg));

		FAIL_IF(push_inst(compiler, LHBRX | S(dst) | A(mem) | B(offs_reg)));

		if (op == SLJIT_REV_U16)
			return SLJIT_SUCCESS;
		return push_inst(compiler, EXTSH | S(dst) | A(dst));
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (!is_32) {
		if (dst & SLJIT_MEM) {
#if defined(_ARCH_PWR7) && _ARCH_PWR7
			return push_inst(compiler, STDBRX | S(src) | A(mem) | B(offs_reg));
#else /* !POWER7 */
#if defined(SLJIT_LITTLE_ENDIAN) && SLJIT_LITTLE_ENDIAN
			FAIL_IF(push_inst(compiler, RLDICL | S(src) | A(TMP_REG1) | RLDI_SH(32) | RLDI_MB(32)));
			FAIL_IF(push_inst(compiler, STWBRX | S(TMP_REG1) | A(mem) | B(offs_reg)));
			FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(offs_reg) | IMM(SSIZE_OF(s32))));
			return push_inst(compiler, STWBRX | S(src) | A(mem) | B(TMP_REG2));
#else /* !SLJIT_LITTLE_ENDIAN */
			FAIL_IF(push_inst(compiler, STWBRX | S(src) | A(mem) | B(offs_reg)));
			FAIL_IF(push_inst(compiler, RLDICL | S(src) | A(TMP_REG1) | RLDI_SH(32) | RLDI_MB(32)));
			FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(offs_reg) | IMM(SSIZE_OF(s32))));
			return push_inst(compiler, STWBRX | S(TMP_REG1) | A(mem) | B(TMP_REG2));
#endif /* SLJIT_LITTLE_ENDIAN */
#endif /* POWER7 */
		}
#if defined(_ARCH_PWR7) && _ARCH_PWR7
		return push_inst(compiler, LDBRX | S(dst) | A(mem) | B(offs_reg));
#else /* !POWER7 */
		FAIL_IF(push_inst(compiler, LWBRX | LWBRX_FIRST_REG | A(mem) | B(offs_reg)));
		FAIL_IF(push_inst(compiler, ADDI | D(TMP_REG2) | A(offs_reg) | IMM(SSIZE_OF(s32))));
		FAIL_IF(push_inst(compiler, LWBRX | LWBRX_SECOND_REG | A(mem) | B(TMP_REG2)));
		return push_inst(compiler, RLDIMI | S(TMP_REG1) | A(dst) | RLDI_SH(32) | RLDI_MB(0));
#endif /* POWER7 */
	}
#endif /* SLJIT_CONFIG_PPC_64 */

	if (dst & SLJIT_MEM)
		return push_inst(compiler, STWBRX | S(src) | A(mem) | B(offs_reg));

	FAIL_IF(push_inst(compiler, LWBRX | S(dst) | A(mem) | B(offs_reg)));
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (op == SLJIT_REV_S32)
		return push_inst(compiler, EXTSW | S(dst) | A(dst));
#endif /* SLJIT_CONFIG_PPC_64 */
	return SLJIT_SUCCESS;
}

#define EMIT_MOV(type, type_flags, type_cast) \
	emit_op(compiler, (src == SLJIT_IMM) ? SLJIT_MOV : type, flags | (type_flags), dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? type_cast srcw : srcw)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 flags = HAS_FLAGS(op) ? ALT_SET_FLAGS : 0;
	sljit_s32 op_flags = GET_ALL_FLAGS(op);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	op = GET_OPCODE(op);

	if (GET_FLAG_TYPE(op_flags) == SLJIT_OVERFLOW)
		FAIL_IF(push_inst(compiler, MTXER | S(TMP_ZERO)));

	if (op <= SLJIT_MOV_P && FAST_IS_REG(src) && src == dst) {
		if (!TYPE_CAST_NEEDED(op))
			return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (op_flags & SLJIT_32) {
		if (op <= SLJIT_MOV_P) {
			if (src & SLJIT_MEM) {
				if (op == SLJIT_MOV_S32)
					op = SLJIT_MOV_U32;
			}
			else if (src == SLJIT_IMM) {
				if (op == SLJIT_MOV_U32)
					op = SLJIT_MOV_S32;
			}
		}
		else {
			/* Most operations expect sign extended arguments. */
			flags |= INT_DATA | SIGNED_DATA;
			if (HAS_FLAGS(op_flags))
				flags |= ALT_SIGN_EXT;
		}
	}
#endif

	switch (op) {
	case SLJIT_MOV:
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
#endif
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, flags | WORD_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	case SLJIT_MOV_U32:
		return EMIT_MOV(SLJIT_MOV_U32, INT_DATA, (sljit_u32));

	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
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

	case SLJIT_CLZ:
	case SLJIT_CTZ:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op_flags & SLJIT_32)
			flags |= ALT_FORM1;
#endif /* SLJIT_CONFIG_PPC_64 */
		return emit_op(compiler, op, flags, dst, dstw, TMP_REG1, 0, src, srcw);
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		op |= SLJIT_32;
#endif /* SLJIT_CONFIG_PPC_64 */
		/* fallthrough */
	case SLJIT_REV:
	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		op |= (op_flags & SLJIT_32);
#endif /* SLJIT_CONFIG_PPC_64 */
		return emit_rev(compiler, op, dst, dstw, src, srcw);
	}

	return SLJIT_SUCCESS;
}

#undef EMIT_MOV

/* Macros for checking different operand types / values. */
#define TEST_SL_IMM(src, srcw) \
	((src) == SLJIT_IMM && (srcw) <= SIMM_MAX && (srcw) >= SIMM_MIN)
#define TEST_UL_IMM(src, srcw) \
	((src) == SLJIT_IMM && !((srcw) & ~0xffff))
#define TEST_UH_IMM(src, srcw) \
	((src) == SLJIT_IMM && !((srcw) & ~(sljit_sw)0xffff0000))

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define TEST_SH_IMM(src, srcw) \
	((src) == SLJIT_IMM && !((srcw) & 0xffff) && (srcw) <= 0x7fffffffl && (srcw) >= -0x80000000l)
#define TEST_ADD_IMM(src, srcw) \
	((src) == SLJIT_IMM && (srcw) <= 0x7fff7fffl && (srcw) >= -0x80000000l)
#define TEST_UI_IMM(src, srcw) \
	((src) == SLJIT_IMM && !((srcw) & ~0xffffffff))

#define TEST_ADD_FORM1(op) \
	(GET_FLAG_TYPE(op) == SLJIT_OVERFLOW \
		|| (op & (SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK)) == (SLJIT_32 | SLJIT_SET_Z | SLJIT_SET_CARRY))
#define TEST_SUB_FORM2(op) \
	((GET_FLAG_TYPE(op) >= SLJIT_SIG_LESS && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL) \
		|| (op & (SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK)) == (SLJIT_32 | SLJIT_SET_Z))
#define TEST_SUB_FORM3(op) \
	(GET_FLAG_TYPE(op) == SLJIT_OVERFLOW \
		|| (op & (SLJIT_32 | SLJIT_SET_Z)) == (SLJIT_32 | SLJIT_SET_Z))

#else /* !SLJIT_CONFIG_PPC_64 */
#define TEST_SH_IMM(src, srcw) \
	((src) == SLJIT_IMM && !((srcw) & 0xffff))
#define TEST_ADD_IMM(src, srcw) \
	((src) == SLJIT_IMM)
#define TEST_UI_IMM(src, srcw) \
	((src) == SLJIT_IMM)

#define TEST_ADD_FORM1(op) \
	(GET_FLAG_TYPE(op) == SLJIT_OVERFLOW)
#define TEST_SUB_FORM2(op) \
	(GET_FLAG_TYPE(op) >= SLJIT_SIG_LESS && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL)
#define TEST_SUB_FORM3(op) \
	(GET_FLAG_TYPE(op) == SLJIT_OVERFLOW)
#endif /* SLJIT_CONFIG_PPC_64 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 flags = HAS_FLAGS(op) ? ALT_SET_FLAGS : 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (op & SLJIT_32) {
		/* Most operations expect sign extended arguments. */
		flags |= INT_DATA | SIGNED_DATA;
		if (src1 == SLJIT_IMM)
			src1w = (sljit_s32)(src1w);
		if (src2 == SLJIT_IMM)
			src2w = (sljit_s32)(src2w);
		if (HAS_FLAGS(op))
			flags |= ALT_SIGN_EXT;
	}
#endif
	if (GET_FLAG_TYPE(op) == SLJIT_OVERFLOW)
		FAIL_IF(push_inst(compiler, MTXER | S(TMP_ZERO)));

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;

		if (TEST_ADD_FORM1(op))
			return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM1, dst, dstw, src1, src1w, src2, src2w);

		if (!HAS_FLAGS(op) && (src1 == SLJIT_IMM || src2 == SLJIT_IMM)) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			if (TEST_SH_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)(src2w >> 16) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2 | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SH_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)(src1w >> 16) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2 | ALT_FORM3, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			/* Range between -1 and -32768 is covered above. */
			if (TEST_ADD_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w & 0xffffffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2 | ALT_FORM4, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_ADD_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w & 0xffffffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2 | ALT_FORM4, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if ((op & (SLJIT_32 | SLJIT_SET_Z)) == (SLJIT_32 | SLJIT_SET_Z)) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM4 | ALT_FORM5, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM4 | ALT_FORM5, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM4, dst, dstw, src1, src1w, src2, src2w);
		}
#endif
		if (HAS_FLAGS(op)) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM3, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		return emit_op(compiler, SLJIT_ADD, flags | ((GET_FLAG_TYPE(op) == SLJIT_CARRY) ? ALT_FORM5 : 0), dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_ADDC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
		return emit_op(compiler, SLJIT_ADDC, flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;

		if (GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_LESS_EQUAL) {
			if (dst == TMP_REG1) {
				if (TEST_UL_IMM(src2, src2w)) {
					compiler->imm = (sljit_ins)src2w & 0xffff;
					return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM1 | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
				}
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM1, dst, dstw, src1, src1w, src2, src2w);
			}

			if (src2 == SLJIT_IMM && src2w >= 0 && src2w <= (SIMM_MAX + 1)) {
				compiler->imm = (sljit_ins)src2w;
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM1 | ALT_FORM2 | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM1 | ALT_FORM3, dst, dstw, src1, src1w, src2, src2w);
		}

		if (dst == TMP_REG1 && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w & 0xffff;
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2 | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2, dst, dstw, src1, src1w, src2, src2w);
		}

		if (TEST_SUB_FORM2(op)) {
			if (src2 == SLJIT_IMM && src2w >= -SIMM_MAX && src2w <= SIMM_MAX) {
				compiler->imm = (sljit_ins)src2w & 0xffff;
				return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2 | ALT_FORM3 | ALT_FORM4, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM2 | ALT_FORM4, dst, dstw, src1, src1w, src2, src2w);
		}

		if (TEST_SUB_FORM3(op))
			return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM3, dst, dstw, src1, src1w, src2, src2w);

		if (TEST_SL_IMM(src2, -src2w)) {
			compiler->imm = (sljit_ins)(-src2w) & 0xffff;
			return emit_op(compiler, SLJIT_ADD, flags | (!HAS_FLAGS(op) ? ALT_FORM2 : ALT_FORM3), dst, dstw, src1, src1w, TMP_REG2, 0);
		}

		if (TEST_SL_IMM(src1, src1w) && !(op & SLJIT_SET_Z)) {
			compiler->imm = (sljit_ins)src1w & 0xffff;
			return emit_op(compiler, SLJIT_SUB, flags | ALT_FORM4, dst, dstw, src2, src2w, TMP_REG2, 0);
		}

		if (!HAS_FLAGS(op)) {
			if (TEST_SH_IMM(src2, -src2w)) {
				compiler->imm = (sljit_ins)((-src2w) >> 16) & 0xffff;
				return emit_op(compiler, SLJIT_ADD, flags |  ALT_FORM2 | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			/* Range between -1 and -32768 is covered above. */
			if (TEST_ADD_IMM(src2, -src2w)) {
				compiler->imm = (sljit_ins)-src2w;
				return emit_op(compiler, SLJIT_ADD, flags | ALT_FORM2 | ALT_FORM4, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
		}

		/* We know ALT_SIGN_EXT is set if it is an SLJIT_32 on 64 bit systems. */
		return emit_op(compiler, SLJIT_SUB, flags | ((GET_FLAG_TYPE(op) == SLJIT_CARRY) ? ALT_FORM5 : 0), dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUBC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
		return emit_op(compiler, SLJIT_SUBC, flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op & SLJIT_32)
			flags |= ALT_FORM2;
#endif
		if (!HAS_FLAGS(op)) {
			if (TEST_SL_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w & 0xffff;
				return emit_op(compiler, SLJIT_MUL, flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_SL_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w & 0xffff;
				return emit_op(compiler, SLJIT_MUL, flags | ALT_FORM1, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		else
			FAIL_IF(push_inst(compiler, MTXER | S(TMP_ZERO)));
		return emit_op(compiler, SLJIT_MUL, flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_XOR:
		if (src2 == SLJIT_IMM && src2w == -1) {
			return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM4, dst, dstw, TMP_REG1, 0, src1, src1w);
		}
		if (src1 == SLJIT_IMM && src1w == -1) {
			return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM4, dst, dstw, TMP_REG1, 0, src2, src2w);
		}
		/* fallthrough */
	case SLJIT_AND:
	case SLJIT_OR:
		/* Commutative unsigned operations. */
		if (!HAS_FLAGS(op) || GET_OPCODE(op) == SLJIT_AND) {
			if (TEST_UL_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_UL_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM1, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
			if (TEST_UH_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)(src2w >> 16) & 0xffff;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM2, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_UH_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)(src1w >> 16) & 0xffff;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM2, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		if (!HAS_FLAGS(op) && GET_OPCODE(op) != SLJIT_AND) {
			/* Unlike or and xor, the and resets unwanted bits as well. */
			if (TEST_UI_IMM(src2, src2w)) {
				compiler->imm = (sljit_ins)src2w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM3, dst, dstw, src1, src1w, TMP_REG2, 0);
			}
			if (TEST_UI_IMM(src1, src1w)) {
				compiler->imm = (sljit_ins)src1w;
				return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM3, dst, dstw, src2, src2w, TMP_REG2, 0);
			}
		}
		return emit_op(compiler, GET_OPCODE(op), flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
	case SLJIT_ROTL:
	case SLJIT_ROTR:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (op & SLJIT_32)
			flags |= ALT_FORM2;
#endif
		if (src2 == SLJIT_IMM) {
			compiler->imm = (sljit_ins)src2w;
			return emit_op(compiler, GET_OPCODE(op), flags | ALT_FORM1, dst, dstw, src1, src1w, TMP_REG2, 0);
		}
		return emit_op(compiler, GET_OPCODE(op), flags, dst, dstw, src1, src1w, src2, src2w);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 1, 0, 0, src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, TMP_REG1, 0, src1, src1w, src2, src2w);
}

#undef TEST_ADD_FORM1
#undef TEST_SUB_FORM2
#undef TEST_SUB_FORM3

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2r(compiler, op, dst_reg, src1, src1w, src2, src2w));

	switch (GET_OPCODE(op)) {
	case SLJIT_MULADD:
		SLJIT_SKIP_CHECKS(compiler);
		FAIL_IF(sljit_emit_op2(compiler, SLJIT_MUL | (op & SLJIT_32), TMP_REG2, 0, src1, src1w, src2, src2w));
		return push_inst(compiler, ADD | D(dst_reg) | A(dst_reg) | B(TMP_REG2));
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_s32 is_right;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 inp_flags = ((op & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
	sljit_sw bit_length = (op & SLJIT_32) ? 32 : 64;
#else /* !SLJIT_CONFIG_PPC_64 */
	sljit_s32 inp_flags = WORD_DATA | LOAD_DATA;
	sljit_sw bit_length = 32;
#endif /* SLJIT_CONFIG_PPC_64 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, dst_reg, src1_reg, src2_reg, src3, src3w));

	is_right = (GET_OPCODE(op) == SLJIT_LSHR || GET_OPCODE(op) == SLJIT_MLSHR);

	if (src1_reg == src2_reg) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, (is_right ? SLJIT_ROTR : SLJIT_ROTL) | (op & SLJIT_32), dst_reg, 0, src1_reg, 0, src3, src3w);
	}

	ADJUST_LOCAL_OFFSET(src3, src3w);

	if (src3 == SLJIT_IMM) {
		src3w &= bit_length - 1;

		if (src3w == 0)
			return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if (!(op & SLJIT_32)) {
			if (is_right) {
				FAIL_IF(push_inst(compiler, SRDI(src3w) | S(src1_reg) | A(dst_reg)));
				return push_inst(compiler, RLDIMI | S(src2_reg) | A(dst_reg) | RLDI_SH(64 - src3w) | RLDI_MB(0));
			}

			FAIL_IF(push_inst(compiler, SLDI(src3w) | S(src1_reg) | A(dst_reg)));
			/* Computes SRDI(64 - src2w). */
			FAIL_IF(push_inst(compiler, RLDICL | S(src2_reg) | A(TMP_REG1) | RLDI_SH(src3w) | RLDI_MB(64 - src3w)));
			return push_inst(compiler, OR | S(dst_reg) | A(dst_reg) | B(TMP_REG1));
		}
#endif /* SLJIT_CONFIG_PPC_64 */

		if (is_right) {
			FAIL_IF(push_inst(compiler, SRWI(src3w) | S(src1_reg) | A(dst_reg)));
			return push_inst(compiler, RLWIMI | S(src2_reg) | A(dst_reg) | RLWI_SH(32 - src3w) | RLWI_MBE(0, src3w - 1));
		}

		FAIL_IF(push_inst(compiler, SLWI(src3w) | S(src1_reg) | A(dst_reg)));
		return push_inst(compiler, RLWIMI | S(src2_reg) | A(dst_reg) | RLWI_SH(src3w) | RLWI_MBE(32 - src3w, 31));
	}

	if (src3 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG2, src3, src3w, TMP_REG2));
		src3 = TMP_REG2;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if (!(op & SLJIT_32)) {
		if (GET_OPCODE(op) == SLJIT_MSHL || GET_OPCODE(op) == SLJIT_MLSHR || dst_reg == src3) {
			FAIL_IF(push_inst(compiler, ANDI | S(src3) | A(TMP_REG2) | 0x3f));
			src3 = TMP_REG2;
		}

		FAIL_IF(push_inst(compiler, (is_right ? SRD : SLD) | S(src1_reg) | A(dst_reg) | B(src3)));
		FAIL_IF(push_inst(compiler, (is_right ? SLDI(1) : SRDI(1)) | S(src2_reg) | A(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XORI | S(src3) | A(TMP_REG2) | 0x3f));
		FAIL_IF(push_inst(compiler, (is_right ? SLD : SRD) | S(TMP_REG1) | A(TMP_REG1) | B(TMP_REG2)));
		return push_inst(compiler, OR | S(dst_reg) | A(dst_reg) | B(TMP_REG1));
	}
#endif /* SLJIT_CONFIG_PPC_64 */

	if (GET_OPCODE(op) == SLJIT_MSHL || GET_OPCODE(op) == SLJIT_MLSHR || dst_reg == src3) {
		FAIL_IF(push_inst(compiler, ANDI | S(src3) | A(TMP_REG2) | 0x1f));
		src3 = TMP_REG2;
	}

	FAIL_IF(push_inst(compiler, (is_right ? SRW : SLW) | S(src1_reg) | A(dst_reg) | B(src3)));
	FAIL_IF(push_inst(compiler, (is_right ? SLWI(1) : SRWI(1)) | S(src2_reg) | A(TMP_REG1)));
	FAIL_IF(push_inst(compiler, XORI | S(src3) | A(TMP_REG2) | 0x1f));
	FAIL_IF(push_inst(compiler, (is_right ? SLW : SRW) | S(TMP_REG1) | A(TMP_REG1) | B(TMP_REG2)));
	return push_inst(compiler, OR | S(dst_reg) | A(dst_reg) | B(TMP_REG1));
}

static sljit_s32 emit_prefetch(struct sljit_compiler *compiler,
        sljit_s32 src, sljit_sw srcw)
{
	if (!(src & OFFS_REG_MASK)) {
		if (srcw == 0 && (src & REG_MASK))
			return push_inst(compiler, DCBT | A(0) | B(src & REG_MASK));

		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		/* Works with SLJIT_MEM0() case as well. */
		return push_inst(compiler, DCBT | A(src & REG_MASK) | B(TMP_REG1));
	}

	srcw &= 0x3;

	if (srcw == 0)
		return push_inst(compiler, DCBT | A(src & REG_MASK) | B(OFFS_REG(src)));

	FAIL_IF(push_inst(compiler, SLWI_W(srcw) | S(OFFS_REG(src)) | A(TMP_REG1)));
	return push_inst(compiler, DCBT | A(src & REG_MASK) | B(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, MTLR | S(src)));
		else {
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG2, src, srcw, TMP_REG2));
			FAIL_IF(push_inst(compiler, MTLR | S(TMP_REG2)));
		}

		return push_inst(compiler, BLR);
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		return emit_prefetch(compiler, src, srcw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_dst(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	switch (op) {
	case SLJIT_FAST_ENTER:
		if (FAST_IS_REG(dst))
			return push_inst(compiler, MFLR | D(dst));

		FAIL_IF(push_inst(compiler, MFLR | D(TMP_REG1)));
		break;
	case SLJIT_GET_RETURN_ADDRESS:
		dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, dst_r, SLJIT_MEM1(SLJIT_SP), compiler->local_size + LR_SAVE_OFFSET, TMP_REG2));
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, WORD_DATA, TMP_REG1, dst, dstw, TMP_REG2);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(type, reg));

	if (type == SLJIT_GP_REGISTER)
		return reg_map[reg];

	if (type != SLJIT_FLOAT_REGISTER)
		return -1;

	return freg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	SLJIT_UNUSED_ARG(size);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_ins*)instruction);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#define SELECT_FOP(op, single, double) ((sljit_ins)((op & SLJIT_32) ? single : double))

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (src & SLJIT_MEM) {
		/* We can ignore the temporary data store on the stack from caching point of view. */
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src, srcw, TMP_REG1));
		src = TMP_FREG1;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	op = GET_OPCODE(op);
	FAIL_IF(push_inst(compiler, (op == SLJIT_CONV_S32_FROM_F64 ? FCTIWZ : FCTIDZ) | FD(TMP_FREG1) | FB(src)));

	if (op == SLJIT_CONV_SW_FROM_F64) {
		if (FAST_IS_REG(dst)) {
			FAIL_IF(push_inst(compiler, STFD | FS(TMP_FREG1) | A(SLJIT_SP) | TMP_MEM_OFFSET));
			return push_inst(compiler, LD | S(dst) | A(SLJIT_SP) | TMP_MEM_OFFSET);
		}
		return emit_op_mem(compiler, DOUBLE_DATA, TMP_FREG1, dst, dstw, TMP_REG1);
	}
#else /* !SLJIT_CONFIG_PPC_64 */
	FAIL_IF(push_inst(compiler, FCTIWZ | FD(TMP_FREG1) | FB(src)));
#endif /* SLJIT_CONFIG_PPC_64 */

	if (FAST_IS_REG(dst)) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, TMP_MEM_OFFSET));
		FAIL_IF(push_inst(compiler, STFIWX | FS(TMP_FREG1) | A(SLJIT_SP) | B(TMP_REG1)));
		return push_inst(compiler, LWZ | S(dst) | A(SLJIT_SP) | TMP_MEM_OFFSET);
	}

	SLJIT_ASSERT(dst & SLJIT_MEM);

	if (dst & OFFS_REG_MASK) {
		dstw &= 0x3;
		if (dstw) {
			FAIL_IF(push_inst(compiler, SLWI_W(dstw) | S(OFFS_REG(dst)) | A(TMP_REG1)));
			dstw = TMP_REG1;
		} else
			dstw = OFFS_REG(dst);
	}
	else {
		if ((dst & REG_MASK) && !dstw) {
			dstw = dst & REG_MASK;
			dst = 0;
		} else {
			/* This works regardless we have SLJIT_MEM1 or SLJIT_MEM0. */
			FAIL_IF(load_immediate(compiler, TMP_REG1, dstw));
			dstw = TMP_REG1;
		}
	}

	return push_inst(compiler, STFIWX | FS(TMP_FREG1) | A(dst & REG_MASK) | B(dstw));
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, TMP_REG1));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, TMP_REG2));
		src2 = TMP_FREG2;
	}

	FAIL_IF(push_inst(compiler, FCMPU | CRD(4) | FA(src1) | FB(src2)));

	switch (GET_FLAG_TYPE(op)) {
	case SLJIT_UNORDERED_OR_EQUAL:
		return push_inst(compiler, CROR | ((4 + 2) << 21) | ((4 + 2) << 16) | ((4 + 3) << 11));
	case SLJIT_UNORDERED_OR_LESS:
		return push_inst(compiler, CROR | ((4 + 0) << 21) | ((4 + 0) << 16) | ((4 + 3) << 11));
	case SLJIT_UNORDERED_OR_GREATER:
		return push_inst(compiler, CROR | ((4 + 1) << 21) | ((4 + 1) << 16) | ((4 + 3) << 11));
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();

	SLJIT_COMPILE_ASSERT((SLJIT_32 == 0x100) && !(DOUBLE_DATA & 0x4), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_32;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op) | LOAD_DATA, dst_r, src, srcw, TMP_REG1));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_CONV_F64_FROM_F32:
		op ^= SLJIT_32;
		if (op & SLJIT_32) {
			FAIL_IF(push_inst(compiler, FRSP | FD(dst_r) | FB(src)));
			break;
		}
		/* Fall through. */
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (!(dst & SLJIT_MEM))
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
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op), dst_r, dst, dstw, TMP_REG1));
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

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG2;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, TMP_REG1));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, TMP_REG1));
		src2 = TMP_FREG2;
	}

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
	case SLJIT_COPYSIGN_F64:
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? STFS : STFD) | FS(src2) | A(SLJIT_SP) | TMP_MEM_OFFSET));
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
		FAIL_IF(push_inst(compiler, LWZ | S(TMP_REG1) | A(SLJIT_SP) | ((op & SLJIT_32) ? TMP_MEM_OFFSET : TMP_MEM_OFFSET_HI)));
#else /* !SLJIT_CONFIG_PPC_32 */
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? LWZ : LD) | S(TMP_REG1) | A(SLJIT_SP) | TMP_MEM_OFFSET));
#endif /* SLJIT_CONFIG_PPC_32 */
		FAIL_IF(push_inst(compiler, FABS | FD(dst_r) | FB(src1)));
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
		FAIL_IF(push_inst(compiler, CMPI | CRD(0) | A(TMP_REG1) | 0));
#else /* !SLJIT_CONFIG_PPC_32 */
		FAIL_IF(push_inst(compiler, CMPI | CRD(0 | ((op & SLJIT_32) ? 0 : 1)) | A(TMP_REG1) | 0));
#endif /* SLJIT_CONFIG_PPC_32 */
		FAIL_IF(push_inst(compiler, BCx | (4 << 21) | (0 << 16) | 8));
		return push_inst(compiler, FNEG | FD(dst_r) | FB(dst_r));
	}

	if (dst & SLJIT_MEM)
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(op), TMP_FREG2, dst, dstw, TMP_REG1));

	return SLJIT_SUCCESS;
}

#undef SELECT_FOP

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

	if (u.imm != 0)
		FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm));

	FAIL_IF(push_inst(compiler, STW | S(u.imm != 0 ? TMP_REG1 : TMP_ZERO) | A(SLJIT_SP) | TMP_MEM_OFFSET));
	return push_inst(compiler, LFS | FS(freg) | A(SLJIT_SP) | TMP_MEM_OFFSET);
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

static sljit_ins get_bo_bi_flags(struct sljit_compiler *compiler, sljit_s32 type)
{
	switch (type) {
	case SLJIT_NOT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_SUB)
			return (4 << 21) | (2 << 16);
		/* fallthrough */

	case SLJIT_EQUAL:
	case SLJIT_ATOMIC_STORED:
		return (12 << 21) | (2 << 16);

	case SLJIT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_SUB)
			return (12 << 21) | (2 << 16);
		/* fallthrough */

	case SLJIT_NOT_EQUAL:
	case SLJIT_ATOMIC_NOT_STORED:
		return (4 << 21) | (2 << 16);

	case SLJIT_LESS:
	case SLJIT_SIG_LESS:
		return (12 << 21) | (0 << 16);

	case SLJIT_GREATER_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
		return (4 << 21) | (0 << 16);

	case SLJIT_GREATER:
	case SLJIT_SIG_GREATER:
		return (12 << 21) | (1 << 16);

	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
		return (4 << 21) | (1 << 16);

	case SLJIT_OVERFLOW:
		return (12 << 21) | (3 << 16);

	case SLJIT_NOT_OVERFLOW:
		return (4 << 21) | (3 << 16);

	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_LESS:
		return (12 << 21) | ((4 + 0) << 16);

	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		return (4 << 21) | ((4 + 0) << 16);

	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
		return (12 << 21) | ((4 + 1) << 16);

	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		return (4 << 21) | ((4 + 1) << 16);

	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		return (12 << 21) | ((4 + 2) << 16);

	case SLJIT_F_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
		return (4 << 21) | ((4 + 2) << 16);

	case SLJIT_UNORDERED:
		return (12 << 21) | ((4 + 3) << 16);

	case SLJIT_ORDERED:
		return (4 << 21) | ((4 + 3) << 16);

	default:
		SLJIT_ASSERT(type >= SLJIT_JUMP && type <= SLJIT_CALL_REG_ARG);
		return (20 << 21);
	}
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins bo_bi_flags;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	bo_bi_flags = get_bo_bi_flags(compiler, type & 0xff);
	if (!bo_bi_flags)
		return NULL;

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, (sljit_u32)type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	if ((type | 0x1) == SLJIT_NOT_CARRY)
		PTR_FAIL_IF(push_inst(compiler, ADDE | RC(ALT_SET_FLAGS) | D(TMP_REG2) | A(TMP_ZERO) | B(TMP_ZERO)));

	/* In PPC, we don't need to touch the arguments. */
	if (type < SLJIT_JUMP)
		jump->flags |= IS_COND;
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
	if (type >= SLJIT_CALL)
		jump->flags |= IS_CALL;
#endif

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, BCCTR | bo_bi_flags | (type >= SLJIT_FAST_CALL ? 1 : 0)));

	/* Maximum number of instructions required for generating a constant. */
	compiler->size += JUMP_MAX_SIZE - 1;
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	SLJIT_UNUSED_ARG(arg_types);

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		PTR_FAIL_IF(call_with_args(compiler, arg_types, NULL));
#endif

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump = NULL;
	sljit_s32 src_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));

	if (src == SLJIT_IMM) {
		/* These jumps are converted to jump/call instructions when possible. */
		jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
		FAIL_IF(!jump);
		set_jump(jump, compiler, JUMP_ADDR);
		jump->u.target = (sljit_uw)srcw;

#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
		if (type >= SLJIT_CALL)
			jump->flags |= IS_CALL;
#endif /* SLJIT_PASS_ENTRY_ADDR_TO_CALL */

		jump->addr = compiler->size;
		FAIL_IF(push_inst(compiler, BCCTR | (20 << 21) | (type >= SLJIT_FAST_CALL ? 1 : 0)));

		/* Maximum number of instructions required for generating a constant. */
		compiler->size += JUMP_MAX_SIZE - 1;
		return SLJIT_SUCCESS;
	}

	if (FAST_IS_REG(src)) {
#if (defined SLJIT_PASS_ENTRY_ADDR_TO_CALL && SLJIT_PASS_ENTRY_ADDR_TO_CALL)
		if (type >= SLJIT_CALL && src != TMP_CALL_REG) {
			FAIL_IF(push_inst(compiler, OR | S(src) | A(TMP_CALL_REG) | B(src)));
			src_r = TMP_CALL_REG;
		} else
			src_r = src;
#else /* SLJIT_PASS_ENTRY_ADDR_TO_CALL */
		src_r = src;
#endif /* SLJIT_PASS_ENTRY_ADDR_TO_CALL */
	} else {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_CALL_REG, src, srcw, TMP_CALL_REG));
		src_r = TMP_CALL_REG;
	}

	FAIL_IF(push_inst(compiler, MTCTR | S(src_r)));
	return push_inst(compiler, BCCTR | (20 << 21) | (type >= SLJIT_FAST_CALL ? 1 : 0));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(arg_types);

	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_CALL_REG, src, srcw, TMP_CALL_REG));
		src = TMP_CALL_REG;
	}

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
			FAIL_IF(push_inst(compiler, OR | S(src) | A(TMP_CALL_REG) | B(src)));
			src = TMP_CALL_REG;
		}

		FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP;
	}

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		FAIL_IF(call_with_args(compiler, arg_types, &src));
#endif

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, type, src, srcw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_s32 reg, invert;
	sljit_u32 bit, from_xer;
	sljit_s32 saved_op = op;
	sljit_sw saved_dstw = dstw;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 input_flags = ((op & SLJIT_32) || op == SLJIT_MOV32) ? INT_DATA : WORD_DATA;
#else
	sljit_s32 input_flags = WORD_DATA;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	reg = (op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2;

	if (op >= SLJIT_ADD && (dst & SLJIT_MEM))
		FAIL_IF(emit_op_mem(compiler, input_flags | LOAD_DATA, TMP_REG1, dst, dstw, TMP_REG1));

	invert = 0;
	bit = 0;
	from_xer = 0;

	switch (type) {
	case SLJIT_LESS:
	case SLJIT_SIG_LESS:
		break;

	case SLJIT_GREATER_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
		invert = 1;
		break;

	case SLJIT_GREATER:
	case SLJIT_SIG_GREATER:
		bit = 1;
		break;

	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
		bit = 1;
		invert = 1;
		break;

	case SLJIT_EQUAL:
	case SLJIT_ATOMIC_STORED:
		bit = 2;
		break;

	case SLJIT_NOT_EQUAL:
	case SLJIT_ATOMIC_NOT_STORED:
		bit = 2;
		invert = 1;
		break;

	case SLJIT_OVERFLOW:
		from_xer = 1;
		bit = 1;
		break;

	case SLJIT_NOT_OVERFLOW:
		from_xer = 1;
		bit = 1;
		invert = 1;
		break;

	case SLJIT_CARRY:
		from_xer = 1;
		bit = 2;
		invert = (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_SUB) != 0;
		break;

	case SLJIT_NOT_CARRY:
		from_xer = 1;
		bit = 2;
		invert = (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_ADD) != 0;
		break;

	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_LESS:
		bit = 4 + 0;
		break;

	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		bit = 4 + 0;
		invert = 1;
		break;

	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
		bit = 4 + 1;
		break;

	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		bit = 4 + 1;
		invert = 1;
		break;

	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		bit = 4 + 2;
		break;

	case SLJIT_F_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
		bit = 4 + 2;
		invert = 1;
		break;

	case SLJIT_UNORDERED:
		bit = 4 + 3;
		break;

	case SLJIT_ORDERED:
		bit = 4 + 3;
		invert = 1;
		break;

	default:
		SLJIT_UNREACHABLE();
		break;
	}

	FAIL_IF(push_inst(compiler, (from_xer ? MFXER : MFCR) | D(reg)));
	/* Simplified mnemonics: extrwi. */
	FAIL_IF(push_inst(compiler, RLWINM | S(reg) | A(reg) | RLWI_SH(1 + bit) | RLWI_MBE(31, 31)));

	if (invert)
		FAIL_IF(push_inst(compiler, XORI | S(reg) | A(reg) | 0x1));

	if (op < SLJIT_ADD) {
		if (!(dst & SLJIT_MEM))
			return SLJIT_SUCCESS;
		return emit_op_mem(compiler, input_flags, reg, dst, dstw, TMP_REG1);
	}

	SLJIT_SKIP_CHECKS(compiler);

	if (dst & SLJIT_MEM)
		return sljit_emit_op2(compiler, saved_op, dst, saved_dstw, TMP_REG1, 0, TMP_REG2, 0);
	return sljit_emit_op2(compiler, saved_op, dst, 0, dst, 0, TMP_REG2, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
	sljit_ins *ptr;
	sljit_uw size;
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	sljit_s32 inp_flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
#else /* !SLJIT_CONFIG_PPC_64 */
        sljit_s32 inp_flags = WORD_DATA | LOAD_DATA;
#endif /* SLJIT_CONFIG_PPC_64 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (dst_reg != src2_reg) {
		if (dst_reg == src1) {
				src1 = src2_reg;
				src1w = 0;
				type ^= 0x1;
		} else {
			if (ADDRESSING_DEPENDS_ON(src1, dst_reg)) {
				FAIL_IF(push_inst(compiler, OR | S(dst_reg) | A(TMP_REG1) | B(dst_reg)));

				if ((src1 & REG_MASK) == dst_reg)
					src1 = (src1 & ~REG_MASK) | TMP_REG1;

				if (OFFS_REG(src1) == dst_reg)
					src1 = (src1 & ~OFFS_REG_MASK) | TO_OFFS_REG(TMP_REG1);
			}

			FAIL_IF(push_inst(compiler, OR | S(src2_reg) | A(dst_reg) | B(src2_reg)));
		}
	}

	if (((type & ~SLJIT_32) | 0x1) == SLJIT_NOT_CARRY)
		FAIL_IF(push_inst(compiler, ADDE | RC(ALT_SET_FLAGS) | D(TMP_REG1) | A(TMP_ZERO) | B(TMP_ZERO)));

	size = compiler->size;

	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	compiler->size++;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, dst_reg, src1, src1w, TMP_REG1));
	} else if (src1 == SLJIT_IMM) {
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
		if (type & SLJIT_32)
			src1w = (sljit_s32)src1w;
#endif /* SLJIT_CONFIG_RISCV_64 */
		FAIL_IF(load_immediate(compiler, dst_reg, src1w));
	} else
		FAIL_IF(push_inst(compiler, OR | S(src1) | A(dst_reg) | B(src1)));

	*ptr = BCx | get_bo_bi_flags(compiler, (type ^ 0x1) & ~SLJIT_32) | (sljit_ins)((compiler->size - size) << 2);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
	sljit_ins *ptr;
	sljit_uw size;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (dst_freg != src2_freg) {
		if (dst_freg == src1) {
			src1 = src2_freg;
			src1w = 0;
			type ^= 0x1;
		} else
			FAIL_IF(push_inst(compiler, FMR | FD(dst_freg) | FB(src2_freg)));
	}

	if (((type & ~SLJIT_32) | 0x1) == SLJIT_NOT_CARRY)
		FAIL_IF(push_inst(compiler, ADDE | RC(ALT_SET_FLAGS) | D(TMP_REG1) | A(TMP_ZERO) | B(TMP_ZERO)));

	size = compiler->size;

	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	compiler->size++;

	if (src1 & SLJIT_MEM)
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(type) | LOAD_DATA, dst_freg, src1, src1w, TMP_REG1));
	else
		FAIL_IF(push_inst(compiler, FMR | FD(dst_freg) | FB(src1)));

	*ptr = BCx | get_bo_bi_flags(compiler, (type ^ 0x1) & ~SLJIT_32) | (sljit_ins)((compiler->size - size) << 2);
	return SLJIT_SUCCESS;
}

#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)

#define EMIT_MEM_LOAD_IMM(inst, mem, memw) \
	((sljit_s16)(memw) > SIMM_MAX - SSIZE_OF(sw))

#else /* !SLJIT_CONFIG_PPC_32 */

#define EMIT_MEM_LOAD_IMM(inst, mem, memw) \
	((((inst) & INT_ALIGNED) && ((memw) & 0x3) != 0) \
		|| ((sljit_s16)(memw) > SIMM_MAX - SSIZE_OF(sw)) \
		|| ((memw) > 0x7fff7fffl || (memw) < -0x80000000l)) \

#endif /* SLJIT_CONFIG_PPC_32 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK))
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);

	ADJUST_LOCAL_OFFSET(mem, memw);

	inst = data_transfer_insts[WORD_DATA | ((type & SLJIT_MEM_STORE) ? 0 : LOAD_DATA)];

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		memw &= 0x3;

		if (memw != 0) {
			FAIL_IF(push_inst(compiler, SLWI_W(memw) | S(OFFS_REG(mem)) | A(TMP_REG1)));
			FAIL_IF(push_inst(compiler, ADD | D(TMP_REG1) | A(TMP_REG1) | B(mem & REG_MASK)));
		} else
			FAIL_IF(push_inst(compiler, ADD | D(TMP_REG1) | A(mem & REG_MASK) | B(OFFS_REG(mem))));

		mem = TMP_REG1;
		memw = 0;
	} else {
		if (EMIT_MEM_LOAD_IMM(inst, mem, memw)) {
			if ((mem & REG_MASK) != 0) {
				SLJIT_SKIP_CHECKS(compiler);
				FAIL_IF(sljit_emit_op2(compiler, SLJIT_ADD, TMP_REG1, 0, mem & REG_MASK, 0, SLJIT_IMM, memw));
			} else
				FAIL_IF(load_immediate(compiler, TMP_REG1, memw));

			memw = 0;
			mem = TMP_REG1;
		} else if (memw > SIMM_MAX || memw < SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDIS | D(TMP_REG1) | A(mem & REG_MASK) | IMM((memw + 0x8000) >> 16)));

			memw &= 0xffff;
			mem = TMP_REG1;
		} else {
			memw &= 0xffff;
			mem &= REG_MASK;
		}
	}

	SLJIT_ASSERT((memw >= 0 && memw <= SIMM_MAX - SSIZE_OF(sw)) || (memw >= 0x8000 && memw <= 0xffff));

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	inst &= (sljit_ins)~INT_ALIGNED;
#endif /* SLJIT_CONFIG_PPC_64 */

	if (!(type & SLJIT_MEM_STORE) && mem == REG_PAIR_FIRST(reg)) {
		FAIL_IF(push_inst(compiler, inst | D(REG_PAIR_SECOND(reg)) | A(mem) | IMM(memw + SSIZE_OF(sw))));
		return push_inst(compiler, inst | D(REG_PAIR_FIRST(reg)) | A(mem) | IMM(memw));
	}

	FAIL_IF(push_inst(compiler, inst | D(REG_PAIR_FIRST(reg)) | A(mem) | IMM(memw)));
	return push_inst(compiler, inst | D(REG_PAIR_SECOND(reg)) | A(mem) | IMM(memw + SSIZE_OF(sw)));
}

#undef EMIT_MEM_LOAD_IMM

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 mem_flags;
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem_update(compiler, type, reg, mem, memw));

	if (type & SLJIT_MEM_POST)
		return SLJIT_ERR_UNSUPPORTED;

	switch (type & 0xff) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
#endif
		mem_flags = WORD_DATA;
		break;

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
	case SLJIT_MOV_U32:
	case SLJIT_MOV32:
		mem_flags = INT_DATA;
		break;

	case SLJIT_MOV_S32:
		mem_flags = INT_DATA;

		if (!(type & SLJIT_MEM_STORE) && !(type & SLJIT_32)) {
			if (mem & OFFS_REG_MASK)
				mem_flags |= SIGNED_DATA;
			else
				return SLJIT_ERR_UNSUPPORTED;
		}
		break;
#endif

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		mem_flags = BYTE_DATA;
		break;

	case SLJIT_MOV_U16:
		mem_flags = HALF_DATA;
		break;

	case SLJIT_MOV_S16:
		mem_flags = HALF_DATA | SIGNED_DATA;
		break;

	default:
		SLJIT_UNREACHABLE();
		mem_flags = WORD_DATA;
		break;
	}

	if (!(type & SLJIT_MEM_STORE))
		mem_flags |= LOAD_DATA;

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		if (memw != 0)
			return SLJIT_ERR_UNSUPPORTED;

		if (type & SLJIT_MEM_SUPP)
			return SLJIT_SUCCESS;

		inst = updated_data_transfer_insts[mem_flags | INDEXED];
		FAIL_IF(push_inst(compiler, INST_CODE_AND_DST(inst, 0, reg) | A(mem & REG_MASK) | B(OFFS_REG(mem))));
	}
	else {
		if (memw > SIMM_MAX || memw < SIMM_MIN)
			return SLJIT_ERR_UNSUPPORTED;

		inst = updated_data_transfer_insts[mem_flags];

#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		if ((inst & INT_ALIGNED) && (memw & 0x3) != 0)
			return SLJIT_ERR_UNSUPPORTED;
#endif

		if (type & SLJIT_MEM_SUPP)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, INST_CODE_AND_DST(inst, 0, reg) | A(mem & REG_MASK) | IMM(memw)));
	}

	if ((mem_flags & LOAD_DATA) && (type & 0xff) == SLJIT_MOV_S8)
		return push_inst(compiler, EXTSB | S(reg) | A(reg));
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 mem_flags;
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem_update(compiler, type, freg, mem, memw));

	if (type & SLJIT_MEM_POST)
		return SLJIT_ERR_UNSUPPORTED;

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		if (memw != 0)
			return SLJIT_ERR_UNSUPPORTED;
	}
	else {
		if (memw > SIMM_MAX || memw < SIMM_MIN)
			return SLJIT_ERR_UNSUPPORTED;
	}

	if (type & SLJIT_MEM_SUPP)
		return SLJIT_SUCCESS;

	mem_flags = FLOAT_DATA(type);

	if (!(type & SLJIT_MEM_STORE))
		mem_flags |= LOAD_DATA;

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		inst = updated_data_transfer_insts[mem_flags | INDEXED];
		return push_inst(compiler, INST_CODE_AND_DST(inst, DOUBLE_DATA, freg) | A(mem & REG_MASK) | B(OFFS_REG(mem)));
	}

	inst = updated_data_transfer_insts[mem_flags];
	return push_inst(compiler, INST_CODE_AND_DST(inst, DOUBLE_DATA, freg) | A(mem & REG_MASK) | IMM(memw));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_load(compiler, op, dst_reg, mem_reg));

	if (op & SLJIT_ATOMIC_USE_CAS)
		return SLJIT_ERR_UNSUPPORTED;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		ins = LDARX;
		break;
#endif /* SLJIT_CONFIG_RISCV_64 */
	case SLJIT_MOV_U32:
	case SLJIT_MOV32:
		ins = LWARX;
		break;

	default:
		return SLJIT_ERR_UNSUPPORTED;
	}

	if (op & SLJIT_ATOMIC_TEST)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ins | D(dst_reg) | B(mem_reg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	sljit_ins ins;

	/* temp_reg == mem_reg is undefined so use another temp register */
	SLJIT_UNUSED_ARG(temp_reg);

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_store(compiler, op, src_reg, mem_reg, temp_reg));

	if (op & SLJIT_ATOMIC_USE_CAS)
		return SLJIT_ERR_UNSUPPORTED;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
		ins = STDCX | 0x1;
		break;
#endif /* SLJIT_CONFIG_RISCV_64 */
	case SLJIT_MOV_U32:
	case SLJIT_MOV32:
		ins = STWCX | 0x1;
		break;

	default:
		return SLJIT_ERR_UNSUPPORTED;
	}

	if (op & SLJIT_ATOMIC_TEST)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ins | D(src_reg) | B(mem_reg));
}

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
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, dst_r, dst, dstw, TMP_REG1));

	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_mov_addr(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_jump *jump;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_mov_addr(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_mov_addr(jump, compiler, 0);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	PTR_FAIL_IF(push_inst(compiler, (sljit_ins)dst_r));
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
	compiler->size++;
#else
	compiler->size += 4;
#endif

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, TMP_REG2, 0));

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}
