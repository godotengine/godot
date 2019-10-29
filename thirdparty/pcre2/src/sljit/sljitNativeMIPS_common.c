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
/* Automatically detect SLJIT_MIPS_R1 */

#if (defined __mips_isa_rev) && (__mips_isa_rev >= 6)
#define SLJIT_MIPS_R6 1
#endif

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return "MIPS32-R6" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_MIPS_32 */
	return "MIPS64-R6" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_MIPS_32 */

#elif (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	return "MIPS32-R1" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_MIPS_32 */
	return "MIPS64-R1" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_MIPS_32 */

#else /* SLJIT_MIPS_R1 */
	return "MIPS III" SLJIT_CPUINFO;
#endif /* SLJIT_MIPS_R6 */
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

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)
#define TMP_FREG3	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 5] = {
	0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 23, 22, 21, 20, 19, 18, 17, 16, 29, 4, 25, 31
};

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 4] = {
	0, 0, 14, 2, 4, 6, 8, 12, 10, 16
};

#else

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 4] = {
	0, 0, 13, 14, 15, 16, 17, 12, 18, 10
};

#endif

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

#define S(s)		(reg_map[s] << 21)
#define T(t)		(reg_map[t] << 16)
#define D(d)		(reg_map[d] << 11)
#define FT(t)		(freg_map[t] << 16)
#define FS(s)		(freg_map[s] << 11)
#define FD(d)		(freg_map[d] << 6)
/* Absolute registers. */
#define SA(s)		((s) << 21)
#define TA(t)		((t) << 16)
#define DA(d)		((d) << 11)
#define IMM(imm)	((imm) & 0xffff)
#define SH_IMM(imm)	((imm) << 6)

#define DR(dr)		(reg_map[dr])
#define FR(dr)		(freg_map[dr])
#define HI(opcode)	((opcode) << 26)
#define LO(opcode)	(opcode)
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
/* CMP.cond.fmt */
/* S = (20 << 21) D = (21 << 21) */
#define CMP_FMT_S	(20 << 21)
#endif /* SLJIT_MIPS_R6 */
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
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define BC1EQZ		(HI(17) | (9 << 21) | FT(TMP_FREG3))
#define BC1NEZ		(HI(17) | (13 << 21) | FT(TMP_FREG3))
#else /* !SLJIT_MIPS_R6 */
#define BC1F		(HI(17) | (8 << 21))
#define BC1T		(HI(17) | (8 << 21) | (1 << 16))
#endif /* SLJIT_MIPS_R6 */
#define BEQ		(HI(4))
#define BGEZ		(HI(1) | (1 << 16))
#define BGTZ		(HI(7))
#define BLEZ		(HI(6))
#define BLTZ		(HI(1) | (0 << 16))
#define BNE		(HI(5))
#define BREAK		(HI(0) | LO(13))
#define CFC1		(HI(17) | (2 << 21))
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define C_UEQ_S		(HI(17) | CMP_FMT_S | LO(3))
#define C_ULE_S		(HI(17) | CMP_FMT_S | LO(7))
#define C_ULT_S		(HI(17) | CMP_FMT_S | LO(5))
#define C_UN_S		(HI(17) | CMP_FMT_S | LO(1))
#define C_FD		(FD(TMP_FREG3))
#else /* !SLJIT_MIPS_R6 */
#define C_UEQ_S		(HI(17) | FMT_S | LO(51))
#define C_ULE_S		(HI(17) | FMT_S | LO(55))
#define C_ULT_S		(HI(17) | FMT_S | LO(53))
#define C_UN_S		(HI(17) | FMT_S | LO(49))
#define C_FD		(0)
#endif /* SLJIT_MIPS_R6 */
#define CVT_S_S		(HI(17) | FMT_S | LO(32))
#define DADDIU		(HI(25))
#define DADDU		(HI(0) | LO(45))
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
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
#else /* !SLJIT_MIPS_R6 */
#define DDIV		(HI(0) | LO(30))
#define DDIVU		(HI(0) | LO(31))
#define DIV		(HI(0) | LO(26))
#define DIVU		(HI(0) | LO(27))
#define DMULT		(HI(0) | LO(28))
#define DMULTU		(HI(0) | LO(29))
#endif /* SLJIT_MIPS_R6 */
#define DIV_S		(HI(17) | FMT_S | LO(3))
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
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define JR		(HI(0) | LO(9))
#else /* !SLJIT_MIPS_R6 */
#define JR		(HI(0) | LO(8))
#endif /* SLJIT_MIPS_R6 */
#define LD		(HI(55))
#define LUI		(HI(15))
#define LW		(HI(35))
#define MFC1		(HI(17))
#if !(defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define MFHI		(HI(0) | LO(16))
#define MFLO		(HI(0) | LO(18))
#else /* SLJIT_MIPS_R6 */
#define MOD		(HI(0) | (3 << 6) | LO(26))
#define MODU		(HI(0) | (3 << 6) | LO(27))
#endif /* !SLJIT_MIPS_R6 */
#define MOV_S		(HI(17) | FMT_S | LO(6))
#define MTC1		(HI(17) | (4 << 21))
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define MUH		(HI(0) | (3 << 6) | LO(24))
#define MUHU		(HI(0) | (3 << 6) | LO(25))
#define MUL		(HI(0) | (2 << 6) | LO(24))
#define MULU		(HI(0) | (2 << 6) | LO(25))
#else /* !SLJIT_MIPS_R6 */
#define MULT		(HI(0) | LO(24))
#define MULTU		(HI(0) | LO(25))
#endif /* SLJIT_MIPS_R6 */
#define MUL_S		(HI(17) | FMT_S | LO(2))
#define NEG_S		(HI(17) | FMT_S | LO(7))
#define NOP		(HI(0) | LO(0))
#define NOR		(HI(0) | LO(39))
#define OR		(HI(0) | LO(37))
#define ORI		(HI(13))
#define SD		(HI(63))
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
#define SWC1		(HI(57))
#define TRUNC_W_S	(HI(17) | FMT_S | LO(13))
#define XOR		(HI(0) | LO(38))
#define XORI		(HI(14))

#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1) || (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define CLZ		(HI(28) | LO(32))
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#define DCLZ		(LO(18))
#else /* !SLJIT_MIPS_R6 */
#define DCLZ		(HI(28) | LO(36))
#define MOVF		(HI(0) | (0 << 16) | LO(1))
#define MOVN		(HI(0) | LO(11))
#define MOVT		(HI(0) | (1 << 16) | LO(1))
#define MOVZ		(HI(0) | LO(10))
#define MUL		(HI(28) | LO(2))
#endif /* SLJIT_MIPS_R6 */
#define PREF		(HI(51))
#define PREFX		(HI(19) | LO(15))
#define SEB		(HI(31) | (16 << 6) | LO(32))
#define SEH		(HI(31) | (24 << 6) | LO(32))
#endif

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define ADDU_W		ADDU
#define ADDIU_W		ADDIU
#define SLL_W		SLL
#define SUBU_W		SUBU
#else
#define ADDU_W		DADDU
#define ADDIU_W		DADDIU
#define SLL_W		DSLL
#define SUBU_W		DSUBU
#endif

#define SIMM_MAX	(0x7fff)
#define SIMM_MIN	(-0x8000)
#define UIMM_MAX	(0xffff)

/* dest_reg is the absolute name of the register
   Useful for reordering instructions in the delay slot. */
static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins, sljit_s32 delay_slot)
{
	SLJIT_ASSERT(delay_slot == MOVABLE_INS || delay_slot >= UNMOVABLE_INS
		|| delay_slot == ((ins >> 11) & 0x1f) || delay_slot == ((ins >> 16) & 0x1f));
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	compiler->delay_slot = delay_slot;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_ins invert_branch(sljit_s32 flags)
{
	if (flags & IS_BIT26_COND)
		return (1 << 26);
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
	if (flags & IS_BIT23_COND)
		return (1 << 23);
#endif /* SLJIT_MIPS_R6 */
	return (1 << 16);
}

static SLJIT_INLINE sljit_ins* detect_jump_type(struct sljit_jump *jump, sljit_ins *code_ptr, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_ins *inst;
	sljit_ins saved_inst;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (jump->flags & (SLJIT_REWRITABLE_JUMP | IS_CALL))
		return code_ptr;
#else
	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		return code_ptr;
#endif

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;
	}

	inst = (sljit_ins *)jump->addr;
	if (jump->flags & IS_COND)
		inst--;

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (jump->flags & IS_CALL)
		goto keep_address;
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
	}
	else {
		diff = ((sljit_sw)target_addr - (sljit_sw)(inst + 1) - executable_offset) >> 2;
		if (diff <= SIMM_MAX && diff >= SIMM_MIN) {
			jump->flags |= PATCH_B;

			if (!(jump->flags & IS_COND)) {
				inst[0] = (jump->flags & IS_JAL) ? BAL : B;
				inst[1] = NOP;
				return inst + 1;
			}
			inst[0] = inst[0] ^ invert_branch(jump->flags);
			inst[1] = NOP;
			jump->addr -= sizeof(sljit_ins);
			return inst + 1;
		}
	}

	if (jump->flags & IS_COND) {
		if ((jump->flags & IS_MOVABLE) && (target_addr & ~0xfffffff) == ((jump->addr + 2 * sizeof(sljit_ins)) & ~0xfffffff)) {
			jump->flags |= PATCH_J;
			saved_inst = inst[0];
			inst[0] = inst[-1];
			inst[-1] = (saved_inst & 0xffff0000) | 3;
			inst[1] = J;
			inst[2] = NOP;
			return inst + 2;
		}
		else if ((target_addr & ~0xfffffff) == ((jump->addr + 3 * sizeof(sljit_ins)) & ~0xfffffff)) {
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
		if ((jump->flags & IS_MOVABLE) && (target_addr & ~0xfffffff) == (jump->addr & ~0xfffffff)) {
			jump->flags |= PATCH_J;
			inst[0] = inst[-1];
			inst[-1] = (jump->flags & IS_JAL) ? JAL : J;
			jump->addr -= sizeof(sljit_ins);
			return inst;
		}

		if ((target_addr & ~0xfffffff) == ((jump->addr + sizeof(sljit_ins)) & ~0xfffffff)) {
			jump->flags |= PATCH_J;
			inst[0] = (jump->flags & IS_JAL) ? JAL : J;
			inst[1] = NOP;
			return inst + 1;
		}
	}

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
keep_address:
	if (target_addr <= 0x7fffffff) {
		jump->flags |= PATCH_ABS32;
		if (jump->flags & IS_COND) {
			inst[0] -= 4;
			inst++;
		}
		inst[2] = inst[6];
		inst[3] = inst[7];
		return inst + 3;
	}
	if (target_addr <= 0x7fffffffffffl) {
		jump->flags |= PATCH_ABS48;
		if (jump->flags & IS_COND) {
			inst[0] -= 2;
			inst++;
		}
		inst[4] = inst[6];
		inst[5] = inst[7];
		return inst + 5;
	}
#endif

	return code_ptr;
}

#ifdef __GNUC__
static __attribute__ ((noinline)) void sljit_cache_flush(void* code, void* code_ptr)
{
	SLJIT_CACHE_FLUSH(code, code_ptr);
}
#endif

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
				label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
				label->size = code_ptr - code;
				label = label->next;
			}
			if (jump && jump->addr == word_count) {
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
				jump->addr = (sljit_uw)(code_ptr - 3);
#else
				jump->addr = (sljit_uw)(code_ptr - 7);
#endif
				code_ptr = detect_jump_type(jump, code_ptr, code, executable_offset);
				jump = jump->next;
			}
			if (const_ && const_->addr == word_count) {
				/* Just recording the address. */
				const_->addr = (sljit_uw)code_ptr;
				const_ = const_->next;
			}
			code_ptr ++;
			word_count ++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		label->addr = (sljit_uw)code_ptr;
		label->size = code_ptr - code;
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);

	jump = compiler->jumps;
	while (jump) {
		do {
			addr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
			buf_ptr = (sljit_ins *)jump->addr;

			if (jump->flags & PATCH_B) {
				addr = (sljit_sw)(addr - ((sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset) + sizeof(sljit_ins))) >> 2;
				SLJIT_ASSERT((sljit_sw)addr <= SIMM_MAX && (sljit_sw)addr >= SIMM_MIN);
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | (addr & 0xffff);
				break;
			}
			if (jump->flags & PATCH_J) {
				SLJIT_ASSERT((addr & ~0xfffffff) == (((sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset) + sizeof(sljit_ins)) & ~0xfffffff));
				buf_ptr[0] |= (addr >> 2) & 0x03ffffff;
				break;
			}

			/* Set the fields of immediate loads. */
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
			buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 16) & 0xffff);
			buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | (addr & 0xffff);
#else
			if (jump->flags & PATCH_ABS32) {
				SLJIT_ASSERT(addr <= 0x7fffffff);
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 16) & 0xffff);
				buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | (addr & 0xffff);
			}
			else if (jump->flags & PATCH_ABS48) {
				SLJIT_ASSERT(addr <= 0x7fffffffffffl);
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 32) & 0xffff);
				buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | ((addr >> 16) & 0xffff);
				buf_ptr[3] = (buf_ptr[3] & 0xffff0000) | (addr & 0xffff);
			}
			else {
				buf_ptr[0] = (buf_ptr[0] & 0xffff0000) | ((addr >> 48) & 0xffff);
				buf_ptr[1] = (buf_ptr[1] & 0xffff0000) | ((addr >> 32) & 0xffff);
				buf_ptr[3] = (buf_ptr[3] & 0xffff0000) | ((addr >> 16) & 0xffff);
				buf_ptr[5] = (buf_ptr[5] & 0xffff0000) | (addr & 0xffff);
			}
#endif
		} while (0);
		jump = jump->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (code_ptr - code) * sizeof(sljit_ins);

	code = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
	code_ptr = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

#ifndef __GNUC__
	SLJIT_CACHE_FLUSH(code, code_ptr);
#else
	/* GCC workaround for invalid code generation with -O2. */
	sljit_cache_flush(code, code_ptr);
#endif
	return code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	sljit_sw fir = 0;

	switch (feature_type) {
	case SLJIT_HAS_FPU:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return SLJIT_IS_FPU_AVAILABLE;
#elif defined(__GNUC__)
		asm ("cfc1 %0, $0" : "=r"(fir));
		return (fir >> 22) & 0x1;
#else
#error "FIR check is not implemented for this architecture"
#endif

#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_CMOV:
		return 1;
#endif

	default:
		return fir;
	}
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
#define SRC2_IMM	0x00400

#define UNUSED_DEST	0x00800
#define REG_DEST	0x01000
#define REG1_SOURCE	0x02000
#define REG2_SOURCE	0x04000
#define SLOW_SRC1	0x08000
#define SLOW_SRC2	0x10000
#define SLOW_DEST	0x20000

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define STACK_STORE	SW
#define STACK_LOAD	LW
#else
#define STACK_STORE	SD
#define STACK_LOAD	LD
#endif

static SLJIT_INLINE sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw);

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
	sljit_s32 args, i, tmp, offs;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1) + SLJIT_LOCALS_OFFSET;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	local_size = (local_size + 15) & ~0xf;
#else
	local_size = (local_size + 31) & ~0x1f;
#endif
	compiler->local_size = local_size;

	if (local_size <= SIMM_MAX) {
		/* Frequent case. */
		FAIL_IF(push_inst(compiler, ADDIU_W | S(SLJIT_SP) | T(SLJIT_SP) | IMM(-local_size), DR(SLJIT_SP)));
		base = S(SLJIT_SP);
		offs = local_size - (sljit_sw)sizeof(sljit_sw);
	}
	else {
		FAIL_IF(load_immediate(compiler, DR(OTHER_FLAG), local_size));
		FAIL_IF(push_inst(compiler, ADDU_W | S(SLJIT_SP) | TA(0) | D(TMP_REG2), DR(TMP_REG2)));
		FAIL_IF(push_inst(compiler, SUBU_W | S(SLJIT_SP) | T(OTHER_FLAG) | D(SLJIT_SP), DR(SLJIT_SP)));
		base = S(TMP_REG2);
		local_size = 0;
		offs = -(sljit_sw)sizeof(sljit_sw);
	}

	FAIL_IF(push_inst(compiler, STACK_STORE | base | TA(RETURN_ADDR_REG) | IMM(offs), MOVABLE_INS));

	tmp = saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = SLJIT_S0; i >= tmp; i--) {
		offs -= (sljit_s32)(sizeof(sljit_sw));
		FAIL_IF(push_inst(compiler, STACK_STORE | base | T(i) | IMM(offs), MOVABLE_INS));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offs -= (sljit_s32)(sizeof(sljit_sw));
		FAIL_IF(push_inst(compiler, STACK_STORE | base | T(i) | IMM(offs), MOVABLE_INS));
	}

	args = get_arg_count(arg_types);

	if (args >= 1)
		FAIL_IF(push_inst(compiler, ADDU_W | SA(4) | TA(0) | D(SLJIT_S0), DR(SLJIT_S0)));
	if (args >= 2)
		FAIL_IF(push_inst(compiler, ADDU_W | SA(5) | TA(0) | D(SLJIT_S1), DR(SLJIT_S1)));
	if (args >= 3)
		FAIL_IF(push_inst(compiler, ADDU_W | SA(6) | TA(0) | D(SLJIT_S2), DR(SLJIT_S2)));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1) + SLJIT_LOCALS_OFFSET;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	compiler->local_size = (local_size + 15) & ~0xf;
#else
	compiler->local_size = (local_size + 31) & ~0x1f;
#endif
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 local_size, i, tmp, offs;
	sljit_ins base;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));

	FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));

	local_size = compiler->local_size;
	if (local_size <= SIMM_MAX)
		base = S(SLJIT_SP);
	else {
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), local_size));
		FAIL_IF(push_inst(compiler, ADDU_W | S(SLJIT_SP) | T(TMP_REG1) | D(TMP_REG1), DR(TMP_REG1)));
		base = S(TMP_REG1);
		local_size = 0;
	}

	FAIL_IF(push_inst(compiler, STACK_LOAD | base | TA(RETURN_ADDR_REG) | IMM(local_size - (sljit_s32)sizeof(sljit_sw)), RETURN_ADDR_REG));
	offs = local_size - (sljit_s32)GET_SAVED_REGISTERS_SIZE(compiler->scratches, compiler->saveds, 1);

	tmp = compiler->scratches;
	for (i = SLJIT_FIRST_SAVED_REG; i <= tmp; i++) {
		FAIL_IF(push_inst(compiler, STACK_LOAD | base | T(i) | IMM(offs), DR(i)));
		offs += (sljit_s32)(sizeof(sljit_sw));
	}

	tmp = compiler->saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - compiler->saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = tmp; i <= SLJIT_S0; i++) {
		FAIL_IF(push_inst(compiler, STACK_LOAD | base | T(i) | IMM(offs), DR(i)));
		offs += (sljit_s32)(sizeof(sljit_sw));
	}

	SLJIT_ASSERT(offs == local_size - (sljit_sw)(sizeof(sljit_sw)));

	FAIL_IF(push_inst(compiler, JR | SA(RETURN_ADDR_REG), UNMOVABLE_INS));
	if (compiler->local_size <= SIMM_MAX)
		return push_inst(compiler, ADDIU_W | S(SLJIT_SP) | T(SLJIT_SP) | IMM(compiler->local_size), UNMOVABLE_INS);
	else
		return push_inst(compiler, ADDU_W | S(TMP_REG1) | TA(0) | D(SLJIT_SP), UNMOVABLE_INS);
}

#undef STACK_STORE
#undef STACK_LOAD

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

/* See getput_arg below.
   Note: can_cache is called only for binary operators. Those
   operators always uses word arguments without write back. */
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
		if (((next_argw - argw) <= SIMM_MAX && (next_argw - argw) >= SIMM_MIN))
			return 1;
		return 0;
	}

	return 0;
}

/* Emit the necessary instructions. See can_cache above. */
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 tmp_ar, base, delay_slot;

	SLJIT_ASSERT(arg & SLJIT_MEM);
	if (!(next_arg & SLJIT_MEM)) {
		next_arg = 0;
		next_argw = 0;
	}

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

	if (compiler->cache_arg == arg && argw - compiler->cache_argw <= SIMM_MAX && argw - compiler->cache_argw >= SIMM_MIN) {
		if (argw != compiler->cache_argw) {
			FAIL_IF(push_inst(compiler, ADDIU_W | S(TMP_REG3) | T(TMP_REG3) | IMM(argw - compiler->cache_argw), DR(TMP_REG3)));
			compiler->cache_argw = argw;
		}
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar), delay_slot);
	}

	if (compiler->cache_arg == SLJIT_MEM && argw - compiler->cache_argw <= SIMM_MAX && argw - compiler->cache_argw >= SIMM_MIN) {
		if (argw != compiler->cache_argw)
			FAIL_IF(push_inst(compiler, ADDIU_W | S(TMP_REG3) | T(TMP_REG3) | IMM(argw - compiler->cache_argw), DR(TMP_REG3)));
	}
	else {
		compiler->cache_arg = SLJIT_MEM;
		FAIL_IF(load_immediate(compiler, DR(TMP_REG3), argw));
	}
	compiler->cache_argw = argw;

	if (!base)
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar), delay_slot);

	if (arg == next_arg && next_argw - argw <= SIMM_MAX && next_argw - argw >= SIMM_MIN) {
		compiler->cache_arg = arg;
		FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | T(base) | D(TMP_REG3), DR(TMP_REG3)));
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | S(TMP_REG3) | TA(reg_ar), delay_slot);
	}

	FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | T(base) | DA(tmp_ar), tmp_ar));
	return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar), delay_slot);
}

static SLJIT_INLINE sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw)
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
			FAIL_IF(push_inst(compiler, ADDU_W | S(base) | TA(tmp_ar) | DA(tmp_ar), tmp_ar));
		}
		else
			FAIL_IF(push_inst(compiler, ADDU_W | S(base) | T(OFFS_REG(arg)) | DA(tmp_ar), tmp_ar));
		return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar), delay_slot);
	}

	FAIL_IF(load_immediate(compiler, tmp_ar, argw));

	if (base != 0)
		FAIL_IF(push_inst(compiler, ADDU_W | S(base) | TA(tmp_ar) | DA(tmp_ar), tmp_ar));

	return push_inst(compiler, data_transfer_insts[flags & MEM_MASK] | SA(tmp_ar) | TA(reg_ar), delay_slot);
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

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

	if (SLJIT_UNLIKELY(dst == SLJIT_UNUSED)) {
		SLJIT_ASSERT(HAS_FLAGS(op));
		flags |= UNUSED_DEST;
	}
	else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (op >= SLJIT_MOV && op <= SLJIT_MOV_P)
			sugg_src2_r = dst_r;
	}
	else if ((dst & SLJIT_MEM) && !getput_arg_fast(compiler, flags | ARG_TEST, DR(TMP_REG1), dst, dstw))
		flags |= SLOW_DEST;

	if (flags & IMM_OP) {
		if ((src2 & SLJIT_IMM) && src2w) {
			if ((!(flags & LOGICAL_OP) && (src2w <= SIMM_MAX && src2w >= SIMM_MIN))
				|| ((flags & LOGICAL_OP) && !(src2w & ~UIMM_MAX))) {
				flags |= SRC2_IMM;
				src2_r = src2w;
			}
		}
		if (!(flags & SRC2_IMM) && (flags & CUMULATIVE_OP) && (src1 & SLJIT_IMM) && src1w) {
			if ((!(flags & LOGICAL_OP) && (src1w <= SIMM_MAX && src1w >= SIMM_MIN))
				|| ((flags & LOGICAL_OP) && !(src1w & ~UIMM_MAX))) {
				flags |= SRC2_IMM;
				src2_r = src1w;

				/* And swap arguments. */
				src1 = src2;
				src1w = src2w;
				src2 = SLJIT_IMM;
				/* src2w = src2_r unneeded. */
			}
		}
	}

	/* Source 1. */
	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	}
	else if (src1 & SLJIT_IMM) {
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
		if (!(flags & REG_DEST) && op >= SLJIT_MOV && op <= SLJIT_MOV_P)
			dst_r = src2_r;
	}
	else if (src2 & SLJIT_IMM) {
		if (!(flags & SRC2_IMM)) {
			if (src2w) {
				FAIL_IF(load_immediate(compiler, DR(sugg_src2_r), src2w));
				src2_r = sugg_src2_r;
			}
			else {
				src2_r = 0;
				if ((op >= SLJIT_MOV && op <= SLJIT_MOV_P) && (dst & SLJIT_MEM))
					dst_r = 0;
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

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	sljit_s32 int_op = op & SLJIT_I32_OP;
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
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? DMULU : DMUL) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? DMUHU : DMUH) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
#else /* !SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? MULU : MUL) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG3), DR(TMP_REG3)));
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? MUHU : MUH) | S(SLJIT_R0) | T(SLJIT_R1) | D(TMP_REG1), DR(TMP_REG1)));
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, ADDU_W | S(TMP_REG3) | TA(0) | D(SLJIT_R0), DR(SLJIT_R0)));
		return push_inst(compiler, ADDU_W | S(TMP_REG1) | TA(0) | D(SLJIT_R1), DR(SLJIT_R1));
#else /* !SLJIT_MIPS_R6 */
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? DMULTU : DMULT) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
#else /* !SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, (op == SLJIT_LMUL_UW ? MULTU : MULT) | S(SLJIT_R0) | T(SLJIT_R1), MOVABLE_INS));
#endif /* SLJIT_CONFIG_MIPS_64 */
		FAIL_IF(push_inst(compiler, MFLO | D(SLJIT_R0), DR(SLJIT_R0)));
		return push_inst(compiler, MFHI | D(SLJIT_R1), DR(SLJIT_R1));
#endif /* SLJIT_MIPS_R6 */
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		SLJIT_COMPILE_ASSERT((SLJIT_DIVMOD_UW & 0x2) == 0 && SLJIT_DIV_UW - 0x2 == SLJIT_DIVMOD_UW, bad_div_opcode_assignments);
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
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
#else /* !SLJIT_MIPS_R6 */
#if !(defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
#endif /* !SLJIT_MIPS_R1 */
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
#endif /* SLJIT_MIPS_R6 */
	}

	return SLJIT_SUCCESS;
}

#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
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
#endif

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	define flags 0
#else
	sljit_s32 flags = 0;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (dst == SLJIT_UNUSED && !HAS_FLAGS(op)) {
#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
		if (op <= SLJIT_MOV_P && (src & SLJIT_MEM))
			return emit_prefetch(compiler, src, srcw);
#endif
		return SLJIT_SUCCESS;
	}

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if ((op & SLJIT_I32_OP) && GET_OPCODE(op) >= SLJIT_NOT)
		flags |= INT_DATA | SIGNED_DATA;
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_U32:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA, dst, dstw, TMP_REG1, 0, src, srcw);
#else
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u32)srcw : srcw);
#endif

	case SLJIT_MOV_S32:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, srcw);
#else
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s32)srcw : srcw);
#endif

	case SLJIT_MOV_U8:
		return emit_op(compiler, SLJIT_MOV_U8, BYTE_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, SLJIT_MOV_S8, BYTE_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, SLJIT_MOV_U16, HALF_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, SLJIT_MOV_S16, HALF_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_NOT:
		return emit_op(compiler, op, flags, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_NEG:
		return emit_op(compiler, SLJIT_SUB | GET_ALL_FLAGS(op), flags | IMM_OP, dst, dstw, SLJIT_IMM, 0, src, srcw);

	case SLJIT_CLZ:
		return emit_op(compiler, op, flags, dst, dstw, TMP_REG1, 0, src, srcw);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	undef flags
#endif
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	define flags 0
#else
	sljit_s32 flags = 0;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	if (dst == SLJIT_UNUSED && !HAS_FLAGS(op))
		return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (op & SLJIT_I32_OP) {
		flags |= INT_DATA | SIGNED_DATA;
		if (src1 & SLJIT_IMM)
			src1w = (sljit_s32)src1w;
		if (src2 & SLJIT_IMM)
			src2w = (sljit_s32)src2w;
	}
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
		return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
	case SLJIT_SUBC:
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		return emit_op(compiler, op, flags | CUMULATIVE_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		return emit_op(compiler, op, flags | CUMULATIVE_OP | LOGICAL_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_LSHR:
	case SLJIT_ASHR:
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
		if (src2 & SLJIT_IMM)
			src2w &= 0x1f;
#else
		if (src2 & SLJIT_IMM) {
			if (op & SLJIT_I32_OP)
				src2w &= 0x1f;
			else
				src2w &= 0x3f;
		}
#endif
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	undef flags
#endif
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(reg));
	return reg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_float_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_float_register_index(reg));
	return FR(reg);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_s32 size)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_ins*)instruction, UNMOVABLE_INS);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_F32_OP) >> 7))
#define FMT(op) (((op & SLJIT_F32_OP) ^ SLJIT_F32_OP) << (21 - 8))

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	define flags 0
#else
	sljit_s32 flags = (GET_OPCODE(op) == SLJIT_CONV_SW_FROM_F64) << 21;
#endif

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(TMP_FREG1), src, srcw, dst, dstw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, (TRUNC_W_S ^ (flags >> 19)) | FMT(op) | FS(src) | FD(TMP_FREG1), MOVABLE_INS));

	if (FAST_IS_REG(dst))
		return push_inst(compiler, MFC1 | flags | T(dst) | FS(TMP_FREG1), MOVABLE_INS);

	/* Store the integer value from a VFP register. */
	return emit_op_mem2(compiler, flags ? DOUBLE_DATA : SINGLE_DATA, FR(TMP_FREG1), dst, dstw, 0, 0);

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	undef is_long
#endif
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	define flags 0
#else
	sljit_s32 flags = (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_SW) << 21;
#endif

	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, MTC1 | flags | T(src) | FS(TMP_FREG1), MOVABLE_INS));
	else if (src & SLJIT_MEM) {
		/* Load the integer value into a VFP register. */
		FAIL_IF(emit_op_mem2(compiler, ((flags) ? DOUBLE_DATA : SINGLE_DATA) | LOAD_DATA, FR(TMP_FREG1), src, srcw, dst, dstw));
	}
	else {
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;
#endif
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), srcw));
		FAIL_IF(push_inst(compiler, MTC1 | flags | T(TMP_REG1) | FS(TMP_FREG1), MOVABLE_INS));
	}

	FAIL_IF(push_inst(compiler, CVT_S_S | flags | (4 << 21) | (((op & SLJIT_F32_OP) ^ SLJIT_F32_OP) >> 8) | FS(TMP_FREG1) | FD(dst_r), MOVABLE_INS));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG1), dst, dstw, 0, 0);
	return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#	undef flags
#endif
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
	case SLJIT_EQUAL_F64:
	case SLJIT_NOT_EQUAL_F64:
		inst = C_UEQ_S;
		break;
	case SLJIT_LESS_F64:
	case SLJIT_GREATER_EQUAL_F64:
		inst = C_ULT_S;
		break;
	case SLJIT_GREATER_F64:
	case SLJIT_LESS_EQUAL_F64:
		inst = C_ULE_S;
		break;
	default:
		SLJIT_ASSERT(GET_FLAG_TYPE(op) == SLJIT_UNORDERED_F64 || GET_FLAG_TYPE(op) == SLJIT_ORDERED_F64);
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

	SLJIT_COMPILE_ASSERT((SLJIT_F32_OP == 0x100) && !(DOUBLE_DATA & 0x2), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_F32_OP;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, FR(dst_r), src, srcw, dst, dstw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, MOV_S | FMT(op) | FS(src) | FD(dst_r), MOVABLE_INS));
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
		FAIL_IF(push_inst(compiler, CVT_S_S | ((op & SLJIT_F32_OP) ? 1 : (1 << 21)) | FS(src) | FD(dst_r), MOVABLE_INS));
		op ^= SLJIT_F32_OP;
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
	}

	if (dst_r == TMP_FREG2)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), FR(TMP_FREG2), dst, dstw, 0, 0));

	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  Other instructions                                                   */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	if (FAST_IS_REG(dst))
		return push_inst(compiler, ADDU_W | SA(RETURN_ADDR_REG) | TA(0) | D(dst), DR(dst));

	/* Memory. */
	return emit_op_mem(compiler, WORD_DATA, RETURN_ADDR_REG, dst, dstw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_return(compiler, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, ADDU_W | S(src) | TA(0) | DA(RETURN_ADDR_REG), RETURN_ADDR_REG));
	else
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, RETURN_ADDR_REG, src, srcw));

	FAIL_IF(push_inst(compiler, JR | SA(RETURN_ADDR_REG), UNMOVABLE_INS));
	return push_inst(compiler, NOP, UNMOVABLE_INS);
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
#define JUMP_LENGTH	4
#else
#define JUMP_LENGTH	8
#endif

#define BR_Z(src) \
	inst = BEQ | SA(src) | TA(0) | JUMP_LENGTH; \
	flags = IS_BIT26_COND; \
	delay_check = src;

#define BR_NZ(src) \
	inst = BNE | SA(src) | TA(0) | JUMP_LENGTH; \
	flags = IS_BIT26_COND; \
	delay_check = src;

#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)

#define BR_T() \
	inst = BC1NEZ; \
	flags = IS_BIT23_COND; \
	delay_check = FCSR_FCC;
#define BR_F() \
	inst = BC1EQZ; \
	flags = IS_BIT23_COND; \
	delay_check = FCSR_FCC;

#else /* !SLJIT_MIPS_R6 */

#define BR_T() \
	inst = BC1T | JUMP_LENGTH; \
	flags = IS_BIT16_COND; \
	delay_check = FCSR_FCC;
#define BR_F() \
	inst = BC1F | JUMP_LENGTH; \
	flags = IS_BIT16_COND; \
	delay_check = FCSR_FCC;

#endif /* SLJIT_MIPS_R6 */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins inst;
	sljit_s32 flags = 0;
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
	case SLJIT_MUL_OVERFLOW:
		BR_Z(OTHER_FLAG);
		break;
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		BR_NZ(OTHER_FLAG);
		break;
	case SLJIT_NOT_EQUAL_F64:
	case SLJIT_GREATER_EQUAL_F64:
	case SLJIT_GREATER_F64:
	case SLJIT_ORDERED_F64:
		BR_T();
		break;
	case SLJIT_EQUAL_F64:
	case SLJIT_LESS_F64:
	case SLJIT_LESS_EQUAL_F64:
	case SLJIT_UNORDERED_F64:
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

	PTR_FAIL_IF(emit_const(compiler, TMP_REG2, 0));

	if (type <= SLJIT_JUMP)
		PTR_FAIL_IF(push_inst(compiler, JR | S(TMP_REG2), UNMOVABLE_INS));
	else {
		jump->flags |= IS_JAL;
		PTR_FAIL_IF(push_inst(compiler, JALR | S(TMP_REG2) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	}

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
	return jump;
}

#define RESOLVE_IMM1() \
	if (src1 & SLJIT_IMM) { \
		if (src1w) { \
			PTR_FAIL_IF(load_immediate(compiler, DR(TMP_REG1), src1w)); \
			src1 = TMP_REG1; \
		} \
		else \
			src1 = 0; \
	}

#define RESOLVE_IMM2() \
	if (src2 & SLJIT_IMM) { \
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
	flags = ((type & SLJIT_I32_OP) ? INT_DATA : WORD_DATA) | LOAD_DATA;
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
		PTR_FAIL_IF(push_inst(compiler, (type == SLJIT_EQUAL ? BNE : BEQ) | S(src1) | T(src2) | JUMP_LENGTH, UNMOVABLE_INS));
	}
	else if (type >= SLJIT_SIG_LESS && (((src1 & SLJIT_IMM) && (src1w == 0)) || ((src2 & SLJIT_IMM) && (src2w == 0)))) {
		inst = NOP;
		if ((src1 & SLJIT_IMM) && (src1w == 0)) {
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
		PTR_FAIL_IF(push_inst(compiler, inst | S(src1) | JUMP_LENGTH, UNMOVABLE_INS));
	}
	else {
		if (type == SLJIT_LESS || type == SLJIT_GREATER_EQUAL || type == SLJIT_SIG_LESS || type == SLJIT_SIG_GREATER_EQUAL) {
			RESOLVE_IMM1();
			if ((src2 & SLJIT_IMM) && src2w <= SIMM_MAX && src2w >= SIMM_MIN)
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTIU : SLTI) | S(src1) | T(TMP_REG1) | IMM(src2w), DR(TMP_REG1)));
			else {
				RESOLVE_IMM2();
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTU : SLT) | S(src1) | T(src2) | D(TMP_REG1), DR(TMP_REG1)));
			}
			type = (type == SLJIT_LESS || type == SLJIT_SIG_LESS) ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;
		}
		else {
			RESOLVE_IMM2();
			if ((src1 & SLJIT_IMM) && src1w <= SIMM_MAX && src1w >= SIMM_MIN)
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTIU : SLTI) | S(src2) | T(TMP_REG1) | IMM(src1w), DR(TMP_REG1)));
			else {
				RESOLVE_IMM1();
				PTR_FAIL_IF(push_inst(compiler, (type <= SLJIT_LESS_EQUAL ? SLTU : SLT) | S(src2) | T(src1) | D(TMP_REG1), DR(TMP_REG1)));
			}
			type = (type == SLJIT_GREATER || type == SLJIT_SIG_GREATER) ? SLJIT_NOT_EQUAL : SLJIT_EQUAL;
		}

		jump->flags |= IS_BIT26_COND;
		PTR_FAIL_IF(push_inst(compiler, (type == SLJIT_EQUAL ? BNE : BEQ) | S(TMP_REG1) | TA(0) | JUMP_LENGTH, UNMOVABLE_INS));
	}

	PTR_FAIL_IF(emit_const(compiler, TMP_REG2, 0));
	PTR_FAIL_IF(push_inst(compiler, JR | S(TMP_REG2), UNMOVABLE_INS));
	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
	return jump;
}

#undef RESOLVE_IMM1
#undef RESOLVE_IMM2

#undef JUMP_LENGTH
#undef BR_Z
#undef BR_NZ
#undef BR_T
#undef BR_F

#undef FLOAT_DATA
#undef FMT

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump = NULL;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (src & SLJIT_IMM) {
		jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
		FAIL_IF(!jump);
		set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_JAL : 0));
		jump->u.target = srcw;

		if (compiler->delay_slot != UNMOVABLE_INS)
			jump->flags |= IS_MOVABLE;

		FAIL_IF(emit_const(compiler, TMP_REG2, 0));
		src = TMP_REG2;
	}
	else if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, DR(TMP_REG2), src, srcw));
		src = TMP_REG2;
	}

	FAIL_IF(push_inst(compiler, JR | S(src), UNMOVABLE_INS));
	if (jump)
		jump->addr = compiler->size;
	FAIL_IF(push_inst(compiler, NOP, UNMOVABLE_INS));
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_s32 src_ar, dst_ar;
	sljit_s32 saved_op = op;
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_s32 mem_type = WORD_DATA;
#else
	sljit_s32 mem_type = (op & SLJIT_I32_OP) ? (INT_DATA | SIGNED_DATA) : WORD_DATA;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
	if (op == SLJIT_MOV_S32)
		mem_type = INT_DATA | SIGNED_DATA;
#endif
	dst_ar = DR((op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2);

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	if (op >= SLJIT_ADD && (dst & SLJIT_MEM))
		FAIL_IF(emit_op_mem2(compiler, mem_type | LOAD_DATA, DR(TMP_REG1), dst, dstw, dst, dstw));

	switch (type & 0xff) {
	case SLJIT_EQUAL:
	case SLJIT_NOT_EQUAL:
		FAIL_IF(push_inst(compiler, SLTIU | SA(EQUAL_FLAG) | TA(dst_ar) | IMM(1), dst_ar));
		src_ar = dst_ar;
		break;
	case SLJIT_MUL_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		FAIL_IF(push_inst(compiler, SLTIU | SA(OTHER_FLAG) | TA(dst_ar) | IMM(1), dst_ar));
		src_ar = dst_ar;
		type ^= 0x1; /* Flip type bit for the XORI below. */
		break;
	case SLJIT_GREATER_F64:
	case SLJIT_LESS_EQUAL_F64:
		type ^= 0x1; /* Flip type bit for the XORI below. */
	case SLJIT_EQUAL_F64:
	case SLJIT_NOT_EQUAL_F64:
	case SLJIT_LESS_F64:
	case SLJIT_GREATER_EQUAL_F64:
	case SLJIT_UNORDERED_F64:
	case SLJIT_ORDERED_F64:
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
		FAIL_IF(push_inst(compiler, MFC1 | TA(dst_ar) | FS(TMP_FREG3), dst_ar));
#else /* !SLJIT_MIPS_R6 */
		FAIL_IF(push_inst(compiler, CFC1 | TA(dst_ar) | DA(FCSR_REG), dst_ar));
#endif /* SLJIT_MIPS_R6 */
		FAIL_IF(push_inst(compiler, SRL | TA(dst_ar) | DA(dst_ar) | SH_IMM(23), dst_ar));
		FAIL_IF(push_inst(compiler, ANDI | SA(dst_ar) | TA(dst_ar) | IMM(1), dst_ar));
		src_ar = dst_ar;
		break;

	default:
		src_ar = OTHER_FLAG;
		break;
	}

	if (type & 0x1) {
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

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
	sljit_ins ins;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_cmov(compiler, type, dst_reg, src, srcw));

#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)

	if (SLJIT_UNLIKELY(src & SLJIT_IMM)) {
#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
		if (dst_reg & SLJIT_I32_OP)
			srcw = (sljit_s32)srcw;
#endif
		FAIL_IF(load_immediate(compiler, DR(TMP_REG1), srcw));
		src = TMP_REG1;
		srcw = 0;
	}

	dst_reg &= ~SLJIT_I32_OP;

	switch (type & 0xff) {
	case SLJIT_EQUAL:
		ins = MOVZ | TA(EQUAL_FLAG);
		break;
	case SLJIT_NOT_EQUAL:
		ins = MOVN | TA(EQUAL_FLAG);
		break;
	case SLJIT_LESS:
	case SLJIT_GREATER:
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER:
	case SLJIT_OVERFLOW:
	case SLJIT_MUL_OVERFLOW:
		ins = MOVN | TA(OTHER_FLAG);
		break;
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		ins = MOVZ | TA(OTHER_FLAG);
		break;
	case SLJIT_EQUAL_F64:
	case SLJIT_LESS_F64:
	case SLJIT_LESS_EQUAL_F64:
	case SLJIT_UNORDERED_F64:
		ins = MOVT;
		break;
	case SLJIT_NOT_EQUAL_F64:
	case SLJIT_GREATER_EQUAL_F64:
	case SLJIT_GREATER_F64:
	case SLJIT_ORDERED_F64:
		ins = MOVF;
		break;
	default:
		ins = MOVZ | TA(OTHER_FLAG);
		SLJIT_UNREACHABLE();
		break;
	}

	return push_inst(compiler, ins | S(src) | D(dst_reg), DR(dst_reg));

#else
	return sljit_emit_cmov_generic(compiler, type, dst_reg, src, srcw);
#endif
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

	reg = FAST_IS_REG(dst) ? dst : TMP_REG2;

	PTR_FAIL_IF(emit_const(compiler, reg, init_value));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, TMP_REG2, 0));
	return const_;
}
