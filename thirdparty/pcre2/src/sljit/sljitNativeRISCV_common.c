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
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	return "RISC-V-32" SLJIT_CPUINFO;
#else /* !SLJIT_CONFIG_RISCV_32 */
	return "RISC-V-64" SLJIT_CPUINFO;
#endif /* SLJIT_CONFIG_RISCV_32 */
}

/* Length of an instruction word
   Both for riscv-32 and riscv-64 */
typedef sljit_u32 sljit_ins;

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3	(SLJIT_NUMBER_OF_REGISTERS + 4)
#define TMP_ZERO	0

/* Flags are kept in volatile registers. */
#define EQUAL_FLAG	(SLJIT_NUMBER_OF_REGISTERS + 5)
#define RETURN_ADDR_REG	TMP_REG2
#define OTHER_FLAG	(SLJIT_NUMBER_OF_REGISTERS + 6)

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 7] = {
	0, 10, 11, 12, 13, 14, 15, 16, 17, 29, 30, 31, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 9, 8, 2, 6, 1, 7, 5, 28
};

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3] = {
	0, 10, 11, 12, 13, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 9, 8, 0, 1,
};

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

#define RD(rd)		((sljit_ins)reg_map[rd] << 7)
#define RS1(rs1)	((sljit_ins)reg_map[rs1] << 15)
#define RS2(rs2)	((sljit_ins)reg_map[rs2] << 20)
#define FRD(rd)		((sljit_ins)freg_map[rd] << 7)
#define FRS1(rs1)	((sljit_ins)freg_map[rs1] << 15)
#define FRS2(rs2)	((sljit_ins)freg_map[rs2] << 20)
#define IMM_I(imm)	((sljit_ins)(imm) << 20)
#define IMM_S(imm)	((((sljit_ins)(imm) & 0xfe0) << 20) | (((sljit_ins)(imm) & 0x1f) << 7))

/* Represents funct(i) parts of the instructions. */
#define OPC(o)		((sljit_ins)(o))
#define F3(f)		((sljit_ins)(f) << 12)
#define F12(f)		((sljit_ins)(f) << 20)
#define F7(f)		((sljit_ins)(f) << 25)

#define ADD		(F7(0x0) | F3(0x0) | OPC(0x33))
#define ADDI		(F3(0x0) | OPC(0x13))
#define AND		(F7(0x0) | F3(0x7) | OPC(0x33))
#define ANDI		(F3(0x7) | OPC(0x13))
#define AUIPC		(OPC(0x17))
#define BEQ		(F3(0x0) | OPC(0x63))
#define BNE		(F3(0x1) | OPC(0x63))
#define BLT		(F3(0x4) | OPC(0x63))
#define BGE		(F3(0x5) | OPC(0x63))
#define BLTU		(F3(0x6) | OPC(0x63))
#define BGEU		(F3(0x7) | OPC(0x63))
#define DIV		(F7(0x1) | F3(0x4) | OPC(0x33))
#define DIVU		(F7(0x1) | F3(0x5) | OPC(0x33))
#define EBREAK		(F12(0x1) | F3(0x0) | OPC(0x73))
#define FADD_S		(F7(0x0) | F3(0x7) | OPC(0x53))
#define FDIV_S		(F7(0xc) | F3(0x7) | OPC(0x53))
#define FEQ_S		(F7(0x50) | F3(0x2) | OPC(0x53))
#define FLD		(F3(0x3) | OPC(0x7))
#define FLE_S		(F7(0x50) | F3(0x0) | OPC(0x53))
#define FLT_S		(F7(0x50) | F3(0x1) | OPC(0x53))
#define FSD		(F3(0x3) | OPC(0x27))
/* These conversion opcodes are partly defined. */
#define FCVT_S_D	(F7(0x20) | OPC(0x53))
#define FCVT_S_W	(F7(0x68) | OPC(0x53))
#define FCVT_W_S	(F7(0x60) | F3(0x1) | OPC(0x53))
#define FMUL_S		(F7(0x8) | F3(0x7) | OPC(0x53))
#define FSGNJ_S		(F7(0x10) | F3(0x0) | OPC(0x53))
#define FSGNJN_S	(F7(0x10) | F3(0x1) | OPC(0x53))
#define FSGNJX_S	(F7(0x10) | F3(0x2) | OPC(0x53))
#define FSUB_S		(F7(0x4) | F3(0x7) | OPC(0x53))
#define JAL		(OPC(0x6f))
#define JALR		(F3(0x0) | OPC(0x67))
#define LD		(F3(0x3) | OPC(0x3))
#define LUI		(OPC(0x37))
#define LW		(F3(0x2) | OPC(0x3))
#define MUL		(F7(0x1) | F3(0x0) | OPC(0x33))
#define MULH		(F7(0x1) | F3(0x1) | OPC(0x33))
#define MULHU		(F7(0x1) | F3(0x3) | OPC(0x33))
#define OR		(F7(0x0) | F3(0x6) | OPC(0x33))
#define ORI		(F3(0x6) | OPC(0x13))
#define REM		(F7(0x1) | F3(0x6) | OPC(0x33))
#define REMU		(F7(0x1) | F3(0x7) | OPC(0x33))
#define SD		(F3(0x3) | OPC(0x23))
#define SLL		(F7(0x0) | F3(0x1) | OPC(0x33))
#define SLLI		(IMM_I(0x0) | F3(0x1) | OPC(0x13))
#define SLT		(F7(0x0) | F3(0x2) | OPC(0x33))
#define SLTI		(F3(0x2) | OPC(0x13))
#define SLTU		(F7(0x0) | F3(0x3) | OPC(0x33))
#define SLTUI		(F3(0x3) | OPC(0x13))
#define SRL		(F7(0x0) | F3(0x5) | OPC(0x33))
#define SRLI		(IMM_I(0x0) | F3(0x5) | OPC(0x13))
#define SRA		(F7(0x20) | F3(0x5) | OPC(0x33))
#define SRAI		(IMM_I(0x400) | F3(0x5) | OPC(0x13))
#define SUB		(F7(0x20) | F3(0x0) | OPC(0x33))
#define SW		(F3(0x2) | OPC(0x23))
#define XOR		(F7(0x0) | F3(0x4) | OPC(0x33))
#define XORI		(F3(0x4) | OPC(0x13))

#define SIMM_MAX	(0x7ff)
#define SIMM_MIN	(-0x800)
#define BRANCH_MAX	(0xfff)
#define BRANCH_MIN	(-0x1000)
#define JUMP_MAX	(0xfffff)
#define JUMP_MIN	(-0x100000)

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
#define S32_MAX		(0x7ffff7ffl)
#define S32_MIN		(-0x80000000l)
#define S44_MAX		(0x7fffffff7ffl)
#define S52_MAX		(0x7ffffffffffffl)
#endif

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_imm_s_inst(struct sljit_compiler *compiler, sljit_ins ins, sljit_sw imm)
{
	return push_inst(compiler, ins | IMM_S(imm));
}

static SLJIT_INLINE sljit_ins* detect_jump_type(struct sljit_jump *jump, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_ins *inst;

	inst = (sljit_ins *)jump->addr;

	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		goto exit;

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;
	}

	diff = (sljit_sw)target_addr - (sljit_sw)inst - executable_offset;

	if (jump->flags & IS_COND) {
		inst--;
		diff += SSIZE_OF(ins);

		if (diff >= BRANCH_MIN && diff <= BRANCH_MAX) {
			jump->flags |= PATCH_B;
			inst[0] = (inst[0] & 0x1fff07f) ^ 0x1000;
			jump->addr = (sljit_uw)inst;
			return inst;
		}

		inst++;
		diff -= SSIZE_OF(ins);
	}

	if (diff >= JUMP_MIN && diff <= JUMP_MAX) {
		if (jump->flags & IS_COND) {
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
			inst[-1] -= (sljit_ins)(1 * sizeof(sljit_ins)) << 7;
#else
			inst[-1] -= (sljit_ins)(5 * sizeof(sljit_ins)) << 7;
#endif
		}

		jump->flags |= PATCH_J;
		return inst;
	}

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	if (diff >= S32_MIN && diff <= S32_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] -= (sljit_ins)(4 * sizeof(sljit_ins)) << 7;

		jump->flags |= PATCH_REL32;
		inst[1] = inst[0];
		return inst + 1;
	}

	if (target_addr <= (sljit_uw)S32_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] -= (sljit_ins)(4 * sizeof(sljit_ins)) << 7;

		jump->flags |= PATCH_ABS32;
		inst[1] = inst[0];
		return inst + 1;
	}

	if (target_addr <= S44_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] -= (sljit_ins)(2 * sizeof(sljit_ins)) << 7;

		jump->flags |= PATCH_ABS44;
		inst[3] = inst[0];
		return inst + 3;
	}

	if (target_addr <= S52_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] -= (sljit_ins)(1 * sizeof(sljit_ins)) << 7;

		jump->flags |= PATCH_ABS52;
		inst[4] = inst[0];
		return inst + 4;
	}
#endif

exit:
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	inst[1] = inst[0];
	return inst + 1;
#else
	inst[5] = inst[0];
	return inst + 5;
#endif
}

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)

static SLJIT_INLINE sljit_sw put_label_get_length(struct sljit_put_label *put_label, sljit_uw max_label)
{
	if (max_label <= (sljit_uw)S32_MAX) {
		put_label->flags = PATCH_ABS32;
		return 1;
	}

	if (max_label <= S44_MAX) {
		put_label->flags = PATCH_ABS44;
		return 3;
	}

	if (max_label <= S52_MAX) {
		put_label->flags = PATCH_ABS52;
		return 4;
	}

	put_label->flags = 0;
	return 5;
}

#endif /* SLJIT_CONFIG_RISCV_64 */

static SLJIT_INLINE void load_addr_to_reg(void *dst, sljit_u32 reg)
{
	struct sljit_jump *jump = NULL;
	struct sljit_put_label *put_label;
	sljit_uw flags;
	sljit_ins *inst;
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	sljit_sw high;
#endif
	sljit_uw addr;

	if (reg != 0) {
		jump = (struct sljit_jump*)dst;
		flags = jump->flags;
		inst = (sljit_ins*)jump->addr;
		addr = (flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
	} else {
		put_label = (struct sljit_put_label*)dst;
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
		flags = put_label->flags;
#endif
		inst = (sljit_ins*)put_label->addr;
		addr = put_label->label->addr;
		reg = *inst;
	}

	if ((addr & 0x800) != 0)
		addr += 0x1000;

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	inst[0] = LUI | RD(reg) | (sljit_ins)((sljit_sw)addr & ~0xfff);
#else /* !SLJIT_CONFIG_RISCV_32 */

	if (flags & PATCH_ABS32) {
		SLJIT_ASSERT(addr <= S32_MAX);
		inst[0] = LUI | RD(reg) | (sljit_ins)((sljit_sw)addr & ~0xfff);
	} else if (flags & PATCH_ABS44) {
		high = (sljit_sw)addr >> 12;
		SLJIT_ASSERT((sljit_uw)high <= 0x7fffffff);

		if (high > S32_MAX) {
			SLJIT_ASSERT((high & 0x800) != 0);
			inst[0] = LUI | RD(reg) | (sljit_ins)0x80000000u;
			inst[1] = XORI | RD(reg) | RS1(reg) | IMM_I(high);
		} else {
			if ((high & 0x800) != 0)
				high += 0x1000;

			inst[0] = LUI | RD(reg) | (sljit_ins)(high & ~0xfff);
			inst[1] = ADDI | RD(reg) | RS1(reg) | IMM_I(high);
		}

		inst[2] = SLLI | RD(reg) | RS1(reg) | IMM_I(12);
		inst += 2;
	} else {
		high = (sljit_sw)addr >> 32;

		if ((addr & 0x80000000l) != 0)
			high = ~high;

		if ((high & 0x800) != 0)
			high += 0x1000;

		if (flags & PATCH_ABS52) {
			SLJIT_ASSERT(addr <= S52_MAX);
			inst[0] = LUI | RD(TMP_REG3) | (sljit_ins)(high << 12);
		} else {
			inst[0] = LUI | RD(TMP_REG3) | (sljit_ins)(high & ~0xfff);
			inst[1] = ADDI | RD(TMP_REG3) | RS1(TMP_REG3) | IMM_I(high);
			inst++;
		}

		inst[1] = LUI | RD(reg) | (sljit_ins)((sljit_sw)addr & ~0xfff);
		inst[2] = SLLI | RD(TMP_REG3) | RS1(TMP_REG3) | IMM_I((flags & PATCH_ABS52) ? 20 : 32);
		inst[3] = XOR | RD(reg) | RS1(reg) | RS2(TMP_REG3);
		inst += 3;
	}
#endif /* !SLJIT_CONFIG_RISCV_32 */

	if (jump != NULL) {
		SLJIT_ASSERT((inst[1] & 0x707f) == JALR);
		inst[1] = (inst[1] & 0xfffff) | IMM_I(addr);
	} else
		inst[1] = ADDI | RD(reg) | RS1(reg) | IMM_I(addr);
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
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
					word_count += 1;
#else
					word_count += 5;
#endif
					jump->addr = (sljit_uw)code_ptr;
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
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
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
			if (!(jump->flags & (PATCH_B | PATCH_J | PATCH_REL32))) {
				load_addr_to_reg(jump, TMP_REG1);
				break;
			}

			addr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
			buf_ptr = (sljit_ins *)jump->addr;
			addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset);

			if (jump->flags & PATCH_B) {
				SLJIT_ASSERT((sljit_sw)addr >= BRANCH_MIN && (sljit_sw)addr <= BRANCH_MAX);
				addr = ((addr & 0x800) >> 4) | ((addr & 0x1e) << 7) | ((addr & 0x7e0) << 20) | ((addr & 0x1000) << 19);
				buf_ptr[0] |= (sljit_ins)addr;
				break;
			}

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
			if (jump->flags & PATCH_REL32) {
				SLJIT_ASSERT((sljit_sw)addr >= S32_MIN && (sljit_sw)addr <= S32_MAX);

				if ((addr & 0x800) != 0)
					addr += 0x1000;

				buf_ptr[0] = AUIPC | RD(TMP_REG1) | (sljit_ins)((sljit_sw)addr & ~0xfff);
				SLJIT_ASSERT((buf_ptr[1] & 0x707f) == JALR);
				buf_ptr[1] |= IMM_I(addr);
				break;
			}
#endif

			SLJIT_ASSERT((sljit_sw)addr >= JUMP_MIN && (sljit_sw)addr <= JUMP_MAX);
			addr = (addr & 0xff000) | ((addr & 0x800) << 9) | ((addr & 0x7fe) << 20) | ((addr & 0x100000) << 11);
			buf_ptr[0] = JAL | RD((jump->flags & IS_CALL) ? RETURN_ADDR_REG : TMP_ZERO) | (sljit_ins)addr;
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

	SLJIT_CACHE_FLUSH(code, code_ptr);
	SLJIT_UPDATE_WX_FLAGS(code, code_ptr, 1);
	return code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	switch (feature_type) {
	case SLJIT_HAS_FPU:
	case SLJIT_HAS_ZERO_REGISTER:
		return 1;
	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	return (type >= SLJIT_ORDERED_EQUAL && type <= SLJIT_ORDERED_LESS_EQUAL);
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
#define IMM_OP		0x00100
#define MOVE_OP		0x00200
#define SRC2_IMM	0x00400

#define UNUSED_DEST	0x00800
#define REG_DEST	0x01000
#define REG1_SOURCE	0x02000
#define REG2_SOURCE	0x04000
#define SLOW_SRC1	0x08000
#define SLOW_SRC2	0x10000
#define SLOW_DEST	0x20000

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#define STACK_STORE	SW
#define STACK_LOAD	LW
#else
#define STACK_STORE	SD
#define STACK_LOAD	LD
#endif

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#include "sljitNativeRISCV_32.c"
#else
#include "sljitNativeRISCV_64.c"
#endif

#define STACK_MAX_DISTANCE (-SIMM_MIN)

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw);

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 i, tmp, offset;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 1);
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if ((local_size & SSIZE_OF(sw)) != 0)
			local_size += SSIZE_OF(sw);
		local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sizeof(sljit_f64));
	}
#else
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sizeof(sljit_f64));
#endif
	local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
	compiler->local_size = local_size;

	if (local_size <= STACK_MAX_DISTANCE) {
		/* Frequent case. */
		FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_SP) | RS1(SLJIT_SP) | IMM_I(-local_size)));
		offset = local_size - SSIZE_OF(sw);
		local_size = 0;
	} else {
		FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_SP) | RS1(SLJIT_SP) | IMM_I(STACK_MAX_DISTANCE)));
		local_size -= STACK_MAX_DISTANCE;

		if (local_size > STACK_MAX_DISTANCE)
			FAIL_IF(load_immediate(compiler, TMP_REG1, local_size, TMP_REG3));
		offset = STACK_MAX_DISTANCE - SSIZE_OF(sw);
	}

	FAIL_IF(push_imm_s_inst(compiler, STACK_STORE | RS1(SLJIT_SP) | RS2(RETURN_ADDR_REG), offset));

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_imm_s_inst(compiler, STACK_STORE | RS1(SLJIT_SP) | RS2(i), offset));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_imm_s_inst(compiler, STACK_STORE | RS1(SLJIT_SP) | RS2(i), offset));
	}

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	/* This alignment is valid because offset is not used after storing FPU regs. */
	if ((offset & SSIZE_OF(sw)) != 0)
		offset -= SSIZE_OF(sw);
#endif

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_imm_s_inst(compiler, FSD | RS1(SLJIT_SP) | FRS2(i), offset));
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_imm_s_inst(compiler, FSD | RS1(SLJIT_SP) | FRS2(i), offset));
	}

	if (local_size > STACK_MAX_DISTANCE)
		FAIL_IF(push_inst(compiler, SUB | RD(SLJIT_SP) | RS1(SLJIT_SP) | RS2(TMP_REG1)));
	else if (local_size > 0)
		FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_SP) | RS1(SLJIT_SP) | IMM_I(-local_size)));

	if (options & SLJIT_ENTER_REG_ARG)
		return SLJIT_SUCCESS;

	arg_types >>= SLJIT_ARG_SHIFT;
	saved_arg_count = 0;
	tmp = SLJIT_R0;

	while (arg_types > 0) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_S0 - saved_arg_count) | RS1(tmp) | IMM_I(0)));
				saved_arg_count++;
			}
			tmp++;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

#undef STACK_MAX_DISTANCE

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 1);
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if ((local_size & SSIZE_OF(sw)) != 0)
			local_size += SSIZE_OF(sw);
		local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sizeof(sljit_f64));
	}
#else
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sizeof(sljit_f64));
#endif
	compiler->local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;

	return SLJIT_SUCCESS;
}

#define STACK_MAX_DISTANCE (-SIMM_MIN - 16)

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_s32 i, tmp, offset;
	sljit_s32 local_size = compiler->local_size;

	if (local_size > STACK_MAX_DISTANCE) {
		local_size -= STACK_MAX_DISTANCE;

		if (local_size > STACK_MAX_DISTANCE) {
			FAIL_IF(load_immediate(compiler, TMP_REG2, local_size, TMP_REG3));
			FAIL_IF(push_inst(compiler, ADD | RD(SLJIT_SP) | RS1(SLJIT_SP) | RS2(TMP_REG2)));
		} else
			FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_SP) | RS1(SLJIT_SP) | IMM_I(local_size)));

		local_size = STACK_MAX_DISTANCE;
	}

	SLJIT_ASSERT(local_size > 0);

	offset = local_size - SSIZE_OF(sw);
	if (!is_return_to)
		FAIL_IF(push_inst(compiler, STACK_LOAD | RD(RETURN_ADDR_REG) | RS1(SLJIT_SP) | IMM_I(offset)));

	tmp = SLJIT_S0 - compiler->saveds;
	for (i = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options); i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | RD(i) | RS1(SLJIT_SP) | IMM_I(offset)));
	}

	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | RD(i) | RS1(SLJIT_SP) | IMM_I(offset)));
	}

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	/* This alignment is valid because offset is not used after storing FPU regs. */
	if ((offset & SSIZE_OF(sw)) != 0)
		offset -= SSIZE_OF(sw);
#endif

	tmp = SLJIT_FS0 - compiler->fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, FLD | FRD(i) | RS1(SLJIT_SP) | IMM_I(offset)));
	}

	for (i = compiler->fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, FLD | FRD(i) | RS1(SLJIT_SP) | IMM_I(offset)));
	}

	return push_inst(compiler, ADDI | RD(SLJIT_SP) | RS1(SLJIT_SP) | IMM_I(local_size));
}

#undef STACK_MAX_DISTANCE

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	FAIL_IF(emit_stack_frame_release(compiler, 0));
	return push_inst(compiler, JALR | RD(TMP_ZERO) | RS1(RETURN_ADDR_REG) | IMM_I(0));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
		src = TMP_REG1;
		srcw = 0;
	} else if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG1) | RS1(src) | IMM_I(0)));
		src = TMP_REG1;
		srcw = 0;
	}

	FAIL_IF(emit_stack_frame_release(compiler, 1));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#define ARCH_32_64(a, b)	a
#else
#define ARCH_32_64(a, b)	b
#endif

static const sljit_ins data_transfer_insts[16 + 4] = {
/* u w s */ ARCH_32_64(F3(0x2) | OPC(0x23) /* sw */, F3(0x3) | OPC(0x23) /* sd */),
/* u w l */ ARCH_32_64(F3(0x2) | OPC(0x3) /* lw */, F3(0x3) | OPC(0x3) /* ld */),
/* u b s */ F3(0x0) | OPC(0x23) /* sb */,
/* u b l */ F3(0x4) | OPC(0x3) /* lbu */,
/* u h s */ F3(0x1) | OPC(0x23) /* sh */,
/* u h l */ F3(0x5) | OPC(0x3) /* lhu */,
/* u i s */ F3(0x2) | OPC(0x23) /* sw */,
/* u i l */ ARCH_32_64(F3(0x2) | OPC(0x3) /* lw */, F3(0x6) | OPC(0x3) /* lwu */),

/* s w s */ ARCH_32_64(F3(0x2) | OPC(0x23) /* sw */, F3(0x3) | OPC(0x23) /* sd */),
/* s w l */ ARCH_32_64(F3(0x2) | OPC(0x3) /* lw */, F3(0x3) | OPC(0x3) /* ld */),
/* s b s */ F3(0x0) | OPC(0x23) /* sb */,
/* s b l */ F3(0x0) | OPC(0x3) /* lb */,
/* s h s */ F3(0x1) | OPC(0x23) /* sh */,
/* s h l */ F3(0x1) | OPC(0x3) /* lh */,
/* s i s */ F3(0x2) | OPC(0x23) /* sw */,
/* s i l */ F3(0x2) | OPC(0x3) /* lw */,

/* d   s */ F3(0x3) | OPC(0x27) /* fsd */,
/* d   l */ F3(0x3) | OPC(0x7) /* fld */,
/* s   s */ F3(0x2) | OPC(0x27) /* fsw */,
/* s   l */ F3(0x2) | OPC(0x7) /* flw */,
};

#undef ARCH_32_64

static sljit_s32 push_mem_inst(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 base, sljit_sw offset)
{
	sljit_ins ins;

	SLJIT_ASSERT(FAST_IS_REG(base) && offset <= 0xfff && offset >= SIMM_MIN);

	ins = data_transfer_insts[flags & MEM_MASK] | RS1(base);
	if (flags & LOAD_DATA)
		ins |= ((flags & MEM_MASK) <= GPR_REG ? RD(reg) : FRD(reg)) | IMM_I(offset);
	else
		ins |= ((flags & MEM_MASK) <= GPR_REG ? RS2(reg) : FRS2(reg)) | IMM_S(offset);

	return push_inst(compiler, ins);
}

/* Can perform an operation using at most 1 instruction. */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{

	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (!(arg & OFFS_REG_MASK) && argw <= SIMM_MAX && argw >= SIMM_MIN) {
		/* Works for both absoulte and relative addresses. */
		if (SLJIT_UNLIKELY(flags & ARG_TEST))
			return 1;

		FAIL_IF(push_mem_inst(compiler, flags, reg, arg & REG_MASK, argw));
		return -1;
	}
	return 0;
}

#define TO_ARGW_HI(argw) (((argw) & ~0xfff) + (((argw) & 0x800) ? 0x1000 : 0))

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
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 base = arg & REG_MASK;
	sljit_s32 tmp_r = TMP_REG1;
	sljit_sw offset, argw_hi;

	SLJIT_ASSERT(arg & SLJIT_MEM);
	if (!(next_arg & SLJIT_MEM)) {
		next_arg = 0;
		next_argw = 0;
	}

	/* Since tmp can be the same as base or offset registers,
	 * these might be unavailable after modifying tmp. */
	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA))
		tmp_r = reg;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		/* Using the cache. */
		if (argw == compiler->cache_argw) {
			if (arg == compiler->cache_arg)
				return push_mem_inst(compiler, flags, reg, TMP_REG3, 0);

			if ((SLJIT_MEM | (arg & OFFS_REG_MASK)) == compiler->cache_arg) {
				if (arg == next_arg && argw == (next_argw & 0x3)) {
					compiler->cache_arg = arg;
					compiler->cache_argw = argw;
					FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG3) | RS1(TMP_REG3) | RS2(base)));
					return push_mem_inst(compiler, flags, reg, TMP_REG3, 0);
				}
				FAIL_IF(push_inst(compiler, ADD | RD(tmp_r) | RS1(base) | RS2(TMP_REG3)));
				return push_mem_inst(compiler, flags, reg, tmp_r, 0);
			}
		}

		if (SLJIT_UNLIKELY(argw)) {
			compiler->cache_arg = SLJIT_MEM | (arg & OFFS_REG_MASK);
			compiler->cache_argw = argw;
			FAIL_IF(push_inst(compiler, SLLI | RD(TMP_REG3) | RS1(OFFS_REG(arg)) | IMM_I(argw)));
		}

		if (arg == next_arg && argw == (next_argw & 0x3)) {
			compiler->cache_arg = arg;
			compiler->cache_argw = argw;
			FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG3) | RS1(base) | RS2(!argw ? OFFS_REG(arg) : TMP_REG3)));
			tmp_r = TMP_REG3;
		}
		else
			FAIL_IF(push_inst(compiler, ADD | RD(tmp_r) | RS1(base) | RS2(!argw ? OFFS_REG(arg) : TMP_REG3)));
		return push_mem_inst(compiler, flags, reg, tmp_r, 0);
	}

	if (compiler->cache_arg == arg && argw - compiler->cache_argw <= SIMM_MAX && argw - compiler->cache_argw >= SIMM_MIN)
		return push_mem_inst(compiler, flags, reg, TMP_REG3, argw - compiler->cache_argw);

	if (compiler->cache_arg == SLJIT_MEM && (argw - compiler->cache_argw <= SIMM_MAX) && (argw - compiler->cache_argw >= SIMM_MIN)) {
		offset = argw - compiler->cache_argw;
	} else {
		compiler->cache_arg = SLJIT_MEM;

		argw_hi = TO_ARGW_HI(argw);

		if (next_arg && next_argw - argw <= SIMM_MAX && next_argw - argw >= SIMM_MIN && argw_hi != TO_ARGW_HI(next_argw)) {
			FAIL_IF(load_immediate(compiler, TMP_REG3, argw, tmp_r));
			compiler->cache_argw = argw;
			offset = 0;
		} else {
			FAIL_IF(load_immediate(compiler, TMP_REG3, argw_hi, tmp_r));
			compiler->cache_argw = argw_hi;
			offset = argw & 0xfff;
			argw = argw_hi;
		}
	}

	if (!base)
		return push_mem_inst(compiler, flags, reg, TMP_REG3, offset);

	if (arg == next_arg && next_argw - argw <= SIMM_MAX && next_argw - argw >= SIMM_MIN) {
		compiler->cache_arg = arg;
		FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG3) | RS1(TMP_REG3) | RS2(base)));
		return push_mem_inst(compiler, flags, reg, TMP_REG3, offset);
	}

	FAIL_IF(push_inst(compiler, ADD | RD(tmp_r) | RS1(TMP_REG3) | RS2(base)));
	return push_mem_inst(compiler, flags, reg, tmp_r, offset);
}

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_s32 base = arg & REG_MASK;
	sljit_s32 tmp_r = TMP_REG1;

	if (getput_arg_fast(compiler, flags, reg, arg, argw))
		return compiler->error;

	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA))
		tmp_r = reg;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (SLJIT_UNLIKELY(argw)) {
			FAIL_IF(push_inst(compiler, SLLI | RD(tmp_r) | RS1(OFFS_REG(arg)) | IMM_I(argw)));
			FAIL_IF(push_inst(compiler, ADD | RD(tmp_r) | RS1(tmp_r) | RS2(base)));
		}
		else
			FAIL_IF(push_inst(compiler, ADD | RD(tmp_r) | RS1(base) | RS2(OFFS_REG(arg))));

		argw = 0;
	} else {
		FAIL_IF(load_immediate(compiler, tmp_r, TO_ARGW_HI(argw), TMP_REG3));

		if (base != 0)
			FAIL_IF(push_inst(compiler, ADD | RD(tmp_r) | RS1(tmp_r) | RS2(base)));
	}

	return push_mem_inst(compiler, flags, reg, tmp_r, argw & 0xfff);
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#define WORD 0
#define IMM_EXTEND(v) (IMM_I(v))
#else /* !SLJIT_CONFIG_RISCV_32 */
#define WORD word
#define IMM_EXTEND(v) (IMM_I((op & SLJIT_32) ? (v) : (32 + (v))))
#endif /* SLJIT_CONFIG_RISCV_32 */

static sljit_s32 emit_clz_ctz(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw src)
{
	sljit_s32 is_clz = (GET_OPCODE(op) == SLJIT_CLZ);
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	sljit_ins word = (op & SLJIT_32) >> 5;
	sljit_ins max = (op & SLJIT_32) ? 32 : 64;
#else /* !SLJIT_CONFIG_RISCV_64 */
	sljit_ins max = 32;
#endif /* SLJIT_CONFIG_RISCV_64 */

	SLJIT_ASSERT(WORD == 0 || WORD == 0x8);

	/* The OTHER_FLAG is the counter. */
	FAIL_IF(push_inst(compiler, ADDI | WORD | RD(OTHER_FLAG) | RS1(TMP_ZERO) | IMM_I(max)));

	/* The TMP_REG2 is the next value. */
	if (src != TMP_REG2)
		FAIL_IF(push_inst(compiler, ADDI | WORD | RD(TMP_REG2) | RS1(src) | IMM_I(0)));

	FAIL_IF(push_inst(compiler, BEQ | RS1(TMP_REG2) | RS2(TMP_ZERO) | ((sljit_ins)((is_clz ? 4 : 5) * SSIZE_OF(ins)) << 7) | ((sljit_ins)(8 * SSIZE_OF(ins)) << 20)));

	FAIL_IF(push_inst(compiler, ADDI | WORD | RD(OTHER_FLAG) | RS1(TMP_ZERO) | IMM_I(0)));
	if (!is_clz) {
		FAIL_IF(push_inst(compiler, ANDI | RD(TMP_REG1) | RS1(TMP_REG2) | IMM_I(1)));
		FAIL_IF(push_inst(compiler, BNE | RS1(TMP_REG1) | RS2(TMP_ZERO) | ((sljit_ins)(2 * SSIZE_OF(ins)) << 7) | ((sljit_ins)(8 * SSIZE_OF(ins)) << 20)));
	} else
		FAIL_IF(push_inst(compiler, BLT | RS1(TMP_REG2) | RS2(TMP_ZERO) | ((sljit_ins)(2 * SSIZE_OF(ins)) << 7) | ((sljit_ins)(8 * SSIZE_OF(ins)) << 20)));

	/* The TMP_REG1 is the next shift. */
	FAIL_IF(push_inst(compiler, ADDI | WORD | RD(TMP_REG1) | RS1(TMP_ZERO) | IMM_I(max)));

	FAIL_IF(push_inst(compiler, ADDI | WORD | RD(EQUAL_FLAG) | RS1(TMP_REG2) | IMM_I(0)));
	FAIL_IF(push_inst(compiler, SRLI | WORD | RD(TMP_REG1) | RS1(TMP_REG1) | IMM_I(1)));

	FAIL_IF(push_inst(compiler, (is_clz ? SRL : SLL) | WORD | RD(TMP_REG2) | RS1(EQUAL_FLAG) | RS2(TMP_REG1)));
	FAIL_IF(push_inst(compiler, BNE | RS1(TMP_REG2) | RS2(TMP_ZERO) | ((sljit_ins)0xfe000e80 - ((2 * SSIZE_OF(ins)) << 7))));
	FAIL_IF(push_inst(compiler, ADDI | WORD | RD(TMP_REG2) | RS1(TMP_REG1) | IMM_I(-1)));
	FAIL_IF(push_inst(compiler, (is_clz ? SRL : SLL) | WORD | RD(TMP_REG2) | RS1(EQUAL_FLAG) | RS2(TMP_REG2)));
	FAIL_IF(push_inst(compiler, OR | RD(OTHER_FLAG) | RS1(OTHER_FLAG) | RS2(TMP_REG1)));
	FAIL_IF(push_inst(compiler, BEQ | RS1(TMP_REG2) | RS2(TMP_ZERO) | ((sljit_ins)0xfe000e80 - ((5 * SSIZE_OF(ins)) << 7))));

	return push_inst(compiler, ADDI | WORD | RD(dst) | RS1(OTHER_FLAG) | IMM_I(0));
}

#define EMIT_LOGICAL(op_imm, op_reg) \
	if (flags & SRC2_IMM) { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_imm | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(src2))); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_imm | RD(dst) | RS1(src1) | IMM_I(src2))); \
	} \
	else { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_reg | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2))); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_reg | RD(dst) | RS1(src1) | RS2(src2))); \
	}

#define EMIT_SHIFT(imm, reg) \
	op_imm = (imm); \
	op_reg = (reg);

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_s32 is_overflow, is_carry, carry_src_r, is_handled;
	sljit_ins op_imm, op_reg;
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	sljit_ins word = (op & SLJIT_32) >> 5;
#endif /* SLJIT_CONFIG_RISCV_64 */

	SLJIT_ASSERT(WORD == 0 || WORD == 0x8);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return push_inst(compiler, ADDI | RD(dst) | RS1(src2) | IMM_I(0));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, ANDI | RD(dst) | RS1(src2) | IMM_I(0xff));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			FAIL_IF(push_inst(compiler, SLLI | WORD | RD(dst) | RS1(src2) | IMM_EXTEND(24)));
			return push_inst(compiler, SRAI | WORD | RD(dst) | RS1(dst) | IMM_EXTEND(24));
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			FAIL_IF(push_inst(compiler, SLLI | WORD | RD(dst) | RS1(src2) | IMM_EXTEND(16)));
			return push_inst(compiler, SRLI | WORD | RD(dst) | RS1(dst) | IMM_EXTEND(16));
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			FAIL_IF(push_inst(compiler, SLLI | WORD | RD(dst) | RS1(src2) | IMM_EXTEND(16)));
			return push_inst(compiler, SRAI | WORD | RD(dst) | RS1(dst) | IMM_EXTEND(16));
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	case SLJIT_MOV_U32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			FAIL_IF(push_inst(compiler, SLLI | RD(dst) | RS1(src2) | IMM_I(32)));
			return push_inst(compiler, SRLI | RD(dst) | RS1(dst) | IMM_I(32));
		}
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, ADDI | 0x8 | RD(dst) | RS1(src2) | IMM_I(0));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;
#endif /* SLJIT_CONFIG_RISCV_64 */

	case SLJIT_CLZ:
	case SLJIT_CTZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return emit_clz_ctz(compiler, op, dst, src2);

	case SLJIT_ADD:
		/* Overflow computation (both add and sub): overflow = src1_sign ^ src2_sign ^ result_sign ^ carry_flag */
		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		carry_src_r = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, ADDI | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(0)));
				else
					FAIL_IF(push_inst(compiler, XORI | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(-1)));
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, ADDI | WORD | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(src2)));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, ADDI | WORD | RD(dst) | RS1(src1) | IMM_I(src2)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, ADD | WORD | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));

			if (is_overflow || carry_src_r != 0) {
				if (src1 != dst)
					carry_src_r = (sljit_s32)src1;
				else if (src2 != dst)
					carry_src_r = (sljit_s32)src2;
				else {
					FAIL_IF(push_inst(compiler, ADDI | RD(OTHER_FLAG) | RS1(src1) | IMM_I(0)));
					carry_src_r = OTHER_FLAG;
				}
			}

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, ADD | WORD | RD(dst) | RS1(src1) | RS2(src2)));
		}

		/* Carry is zero if a + b >= a or a + b >= b, otherwise it is 1. */
		if (is_overflow || carry_src_r != 0) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLTUI | RD(OTHER_FLAG) | RS1(dst) | IMM_I(src2)));
			else
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RS1(dst) | RS2(carry_src_r)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, XOR | RD(TMP_REG1) | RS1(dst) | RS2(EQUAL_FLAG)));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, ADDI | RD(EQUAL_FLAG) | RS1(dst) | IMM_I(0)));
		FAIL_IF(push_inst(compiler, SRLI | WORD | RD(TMP_REG1) | RS1(TMP_REG1) | IMM_EXTEND(31)));
		return push_inst(compiler, XOR | RD(OTHER_FLAG) | RS1(TMP_REG1) | RS2(OTHER_FLAG));

	case SLJIT_ADDC:
		carry_src_r = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			FAIL_IF(push_inst(compiler, ADDI | WORD | RD(dst) | RS1(src1) | IMM_I(src2)));
		} else {
			if (carry_src_r != 0) {
				if (src1 != dst)
					carry_src_r = (sljit_s32)src1;
				else if (src2 != dst)
					carry_src_r = (sljit_s32)src2;
				else {
					FAIL_IF(push_inst(compiler, ADDI | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(0)));
					carry_src_r = EQUAL_FLAG;
				}
			}

			FAIL_IF(push_inst(compiler, ADD | WORD | RD(dst) | RS1(src1) | RS2(src2)));
		}

		/* Carry is zero if a + b >= a or a + b >= b, otherwise it is 1. */
		if (carry_src_r != 0) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLTUI | RD(EQUAL_FLAG) | RS1(dst) | IMM_I(src2)));
			else
				FAIL_IF(push_inst(compiler, SLTU | RD(EQUAL_FLAG) | RS1(dst) | RS2(carry_src_r)));
		}

		FAIL_IF(push_inst(compiler, ADD | WORD | RD(dst) | RS1(dst) | RS2(OTHER_FLAG)));

		if (carry_src_r == 0)
			return SLJIT_SUCCESS;

		/* Set ULESS_FLAG (dst == 0) && (OTHER_FLAG == 1). */
		FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RS1(dst) | RS2(OTHER_FLAG)));
		/* Set carry flag. */
		return push_inst(compiler, OR | RD(OTHER_FLAG) | RS1(OTHER_FLAG) | RS2(EQUAL_FLAG));

	case SLJIT_SUB:
		if ((flags & SRC2_IMM) && src2 == SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG2) | RS1(TMP_ZERO) | IMM_I(src2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_handled = 0;

		if (flags & SRC2_IMM) {
			if (GET_FLAG_TYPE(op) == SLJIT_LESS || GET_FLAG_TYPE(op) == SLJIT_GREATER_EQUAL) {
				FAIL_IF(push_inst(compiler, SLTUI | RD(OTHER_FLAG) | RS1(src1) | IMM_I(src2)));
				is_handled = 1;
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_SIG_LESS || GET_FLAG_TYPE(op) == SLJIT_SIG_GREATER_EQUAL) {
				FAIL_IF(push_inst(compiler, SLTI | RD(OTHER_FLAG) | RS1(src1) | IMM_I(src2)));
				is_handled = 1;
			}
		}

		if (!is_handled && GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL) {
			is_handled = 1;

			if (flags & SRC2_IMM) {
				FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG2) | RS1(TMP_ZERO) | IMM_I(src2)));
				src2 = TMP_REG2;
				flags &= ~SRC2_IMM;
			}

			switch (GET_FLAG_TYPE(op)) {
			case SLJIT_LESS:
			case SLJIT_GREATER_EQUAL:
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RS1(src1) | RS2(src2)));
				break;
			case SLJIT_GREATER:
			case SLJIT_LESS_EQUAL:
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RS1(src2) | RS2(src1)));
				break;
			case SLJIT_SIG_LESS:
			case SLJIT_SIG_GREATER_EQUAL:
				FAIL_IF(push_inst(compiler, SLT | RD(OTHER_FLAG) | RS1(src1) | RS2(src2)));
				break;
			case SLJIT_SIG_GREATER:
			case SLJIT_SIG_LESS_EQUAL:
				FAIL_IF(push_inst(compiler, SLT | RD(OTHER_FLAG) | RS1(src2) | RS2(src1)));
				break;
			}
		}

		if (is_handled) {
			if (flags & SRC2_IMM) {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, ADDI | WORD | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(-src2)));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, ADDI | WORD | RD(dst) | RS1(src1) | IMM_I(-src2));
			}
			else {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, SUB | WORD | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, SUB | WORD | RD(dst) | RS1(src1) | RS2(src2));
			}
			return SLJIT_SUCCESS;
		}

		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, ADDI | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(0)));
				else
					FAIL_IF(push_inst(compiler, XORI | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(-1)));
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, ADDI | WORD | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(-src2)));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTUI | RD(OTHER_FLAG) | RS1(src1) | IMM_I(src2)));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, ADDI | WORD | RD(dst) | RS1(src1) | IMM_I(-src2)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SUB | WORD | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RS1(src1) | RS2(src2)));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SUB | WORD | RD(dst) | RS1(src1) | RS2(src2)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, XOR | RD(TMP_REG1) | RS1(dst) | RS2(EQUAL_FLAG)));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, ADDI | RD(EQUAL_FLAG) | RS1(dst) | IMM_I(0)));
		FAIL_IF(push_inst(compiler, SRLI | WORD | RD(TMP_REG1) | RS1(TMP_REG1) | IMM_EXTEND(31)));
		return push_inst(compiler, XOR | RD(OTHER_FLAG) | RS1(TMP_REG1) | RS2(OTHER_FLAG));

	case SLJIT_SUBC:
		if ((flags & SRC2_IMM) && src2 == SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG2) | RS1(TMP_ZERO) | IMM_I(src2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTUI | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(src2)));

			FAIL_IF(push_inst(compiler, ADDI | WORD | RD(dst) | RS1(src1) | IMM_I(-src2)));
		}
		else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTU | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));

			FAIL_IF(push_inst(compiler, SUB | WORD | RD(dst) | RS1(src1) | RS2(src2)));
		}

		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | RD(TMP_REG1) | RS1(dst) | RS2(OTHER_FLAG)));

		FAIL_IF(push_inst(compiler, SUB | WORD | RD(dst) | RS1(dst) | RS2(OTHER_FLAG)));

		if (!is_carry)
			return SLJIT_SUCCESS;

		return push_inst(compiler, OR | RD(OTHER_FLAG) | RS1(EQUAL_FLAG) | RS2(TMP_REG1));

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & SRC2_IMM));

		if (GET_FLAG_TYPE(op) != SLJIT_OVERFLOW)
			return push_inst(compiler, MUL | WORD | RD(dst) | RS1(src1) | RS2(src2));

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
		if (word) {
			FAIL_IF(push_inst(compiler, MUL | RD(OTHER_FLAG) | RS1(src1) | RS2(src2)));
			FAIL_IF(push_inst(compiler, MUL | 0x8 | RD(dst) | RS1(src1) | RS2(src2)));
			return push_inst(compiler, SUB | RD(OTHER_FLAG) | RS1(dst) | RS2(OTHER_FLAG));
		}
#endif /* SLJIT_CONFIG_RISCV_64 */

		FAIL_IF(push_inst(compiler, MULH | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));
		FAIL_IF(push_inst(compiler, MUL | RD(dst) | RS1(src1) | RS2(src2)));
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
		FAIL_IF(push_inst(compiler, SRAI | RD(OTHER_FLAG) | RS1(dst) | IMM_I(31)));
#else /* !SLJIT_CONFIG_RISCV_32 */
		FAIL_IF(push_inst(compiler, SRAI | RD(OTHER_FLAG) | RS1(dst) | IMM_I(63)));
#endif /* SLJIT_CONFIG_RISCV_32 */
		return push_inst(compiler, SUB | RD(OTHER_FLAG) | RS1(EQUAL_FLAG) | RS2(OTHER_FLAG));

	case SLJIT_AND:
		EMIT_LOGICAL(ANDI, AND);
		return SLJIT_SUCCESS;

	case SLJIT_OR:
		EMIT_LOGICAL(ORI, OR);
		return SLJIT_SUCCESS;

	case SLJIT_XOR:
		EMIT_LOGICAL(XORI, XOR);
		return SLJIT_SUCCESS;

	case SLJIT_SHL:
	case SLJIT_MSHL:
		EMIT_SHIFT(SLLI, SLL);
		break;

	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		EMIT_SHIFT(SRLI, SRL);
		break;

	case SLJIT_ASHR:
	case SLJIT_MASHR:
		EMIT_SHIFT(SRAI, SRA);
		break;

	case SLJIT_ROTL:
	case SLJIT_ROTR:
		if (flags & SRC2_IMM) {
			SLJIT_ASSERT(src2 != 0);

			op_imm = (GET_OPCODE(op) == SLJIT_ROTL) ? SLLI : SRLI;
			FAIL_IF(push_inst(compiler, op_imm | WORD | RD(OTHER_FLAG) | RS1(src1) | IMM_I(src2)));

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
			src2 = ((op & SLJIT_32) ? 32 : 64) - src2;
#else /* !SLJIT_CONFIG_RISCV_64 */
			src2 = 32 - src2;
#endif /* SLJIT_CONFIG_RISCV_64 */
			op_imm = (GET_OPCODE(op) == SLJIT_ROTL) ? SRLI : SLLI;
			FAIL_IF(push_inst(compiler, op_imm | WORD | RD(dst) | RS1(src1) | IMM_I(src2)));
			return push_inst(compiler, OR | RD(dst) | RS1(dst) | RS2(OTHER_FLAG));
		}

		if (src2 == TMP_ZERO) {
			if (dst != src1)
				return push_inst(compiler, ADDI | WORD | RD(dst) | RS1(src1) | IMM_I(0));
			return SLJIT_SUCCESS;
		}

		FAIL_IF(push_inst(compiler, SUB | WORD | RD(EQUAL_FLAG) | RS1(TMP_ZERO) | RS2(src2)));
		op_reg = (GET_OPCODE(op) == SLJIT_ROTL) ? SLL : SRL;
		FAIL_IF(push_inst(compiler, op_reg | WORD | RD(OTHER_FLAG) | RS1(src1) | RS2(src2)));
		op_reg = (GET_OPCODE(op) == SLJIT_ROTL) ? SRL : SLL;
		FAIL_IF(push_inst(compiler, op_reg | WORD | RD(dst) | RS1(src1) | RS2(EQUAL_FLAG)));
		return push_inst(compiler, OR | RD(dst) | RS1(dst) | RS2(OTHER_FLAG));

	default:
		SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;
	}

	if (flags & SRC2_IMM) {
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, op_imm | WORD | RD(EQUAL_FLAG) | RS1(src1) | IMM_I(src2)));

		if (flags & UNUSED_DEST)
			return SLJIT_SUCCESS;
		return push_inst(compiler, op_imm | WORD | RD(dst) | RS1(src1) | IMM_I(src2));
	}

	if (op & SLJIT_SET_Z)
		FAIL_IF(push_inst(compiler, op_reg | WORD | RD(EQUAL_FLAG) | RS1(src1) | RS2(src2)));

	if (flags & UNUSED_DEST)
		return SLJIT_SUCCESS;
	return push_inst(compiler, op_reg | WORD | RD(dst) | RS1(src1) | RS2(src2));
}

#undef IMM_EXTEND

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

	if (dst == TMP_REG2) {
		SLJIT_ASSERT(HAS_FLAGS(op));
		flags |= UNUSED_DEST;
	}
	else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (flags & MOVE_OP)
			sugg_src2_r = dst_r;
	}
	else if ((dst & SLJIT_MEM) && !getput_arg_fast(compiler, flags | ARG_TEST, TMP_REG1, dst, dstw))
		flags |= SLOW_DEST;

	if (flags & IMM_OP) {
		if ((src2 & SLJIT_IMM) && src2w != 0 && src2w <= SIMM_MAX && src2w >= SIMM_MIN) {
			flags |= SRC2_IMM;
			src2_r = src2w;
		}
		else if ((flags & CUMULATIVE_OP) && (src1 & SLJIT_IMM) && src1w != 0 && src1w <= SIMM_MAX && src1w >= SIMM_MIN) {
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
	else if (src1 & SLJIT_IMM) {
		if (src1w) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, src1w, TMP_REG3));
			src1_r = TMP_REG1;
		}
		else
			src1_r = TMP_ZERO;
	}
	else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w))
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
	else if (src2 & SLJIT_IMM) {
		if (!(flags & SRC2_IMM)) {
			if (src2w) {
				FAIL_IF(load_immediate(compiler, sugg_src2_r, src2w, TMP_REG3));
				src2_r = sugg_src2_r;
			}
			else {
				src2_r = TMP_ZERO;
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
		if (getput_arg_fast(compiler, flags | LOAD_DATA, sugg_src2_r, src2, src2w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC2;
		src2_r = sugg_src2_r;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		SLJIT_ASSERT(src2_r == TMP_REG2);
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2, src2, src2w, dst, dstw));
		}
	}
	else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, sugg_src2_r, src2, src2w, dst, dstw));

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (dst & SLJIT_MEM) {
		if (!(flags & SLOW_DEST)) {
			getput_arg_fast(compiler, flags, dst_r, dst, dstw);
			return compiler->error;
		}
		return getput_arg(compiler, flags, dst_r, dst, dstw, 0, 0);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	sljit_ins word = (op & SLJIT_32) >> 5;

	SLJIT_ASSERT(word == 0 || word == 0x8);
#endif /* SLJIT_CONFIG_RISCV_64 */

	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	switch (GET_OPCODE(op)) {
	case SLJIT_BREAKPOINT:
		return push_inst(compiler, EBREAK);
	case SLJIT_NOP:
		return push_inst(compiler, ADDI | RD(TMP_ZERO) | RS1(TMP_ZERO) | IMM_I(0));
	case SLJIT_LMUL_UW:
		FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG1) | RS1(SLJIT_R1) | IMM_I(0)));
		FAIL_IF(push_inst(compiler, MULHU | RD(SLJIT_R1) | RS1(SLJIT_R0) | RS2(SLJIT_R1)));
		return push_inst(compiler, MUL | RD(SLJIT_R0) | RS1(SLJIT_R0) | RS2(TMP_REG1));
	case SLJIT_LMUL_SW:
		FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG1) | RS1(SLJIT_R1) | IMM_I(0)));
		FAIL_IF(push_inst(compiler, MULH | RD(SLJIT_R1) | RS1(SLJIT_R0) | RS2(SLJIT_R1)));
		return push_inst(compiler, MUL | RD(SLJIT_R0) | RS1(SLJIT_R0) | RS2(TMP_REG1));
	case SLJIT_DIVMOD_UW:
		FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG1) | RS1(SLJIT_R0) | IMM_I(0)));
		FAIL_IF(push_inst(compiler, DIVU | WORD | RD(SLJIT_R0) | RS1(SLJIT_R0) | RS2(SLJIT_R1)));
		return push_inst(compiler, REMU | WORD | RD(SLJIT_R1) | RS1(TMP_REG1) | RS2(SLJIT_R1));
	case SLJIT_DIVMOD_SW:
		FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG1) | RS1(SLJIT_R0) | IMM_I(0)));
		FAIL_IF(push_inst(compiler, DIV | WORD | RD(SLJIT_R0) | RS1(SLJIT_R0) | RS2(SLJIT_R1)));
		return push_inst(compiler, REM | WORD | RD(SLJIT_R1) | RS1(TMP_REG1) | RS2(SLJIT_R1));
	case SLJIT_DIV_UW:
		return push_inst(compiler, DIVU | WORD | RD(SLJIT_R0) | RS1(SLJIT_R0) | RS2(SLJIT_R1));
	case SLJIT_DIV_SW:
		return push_inst(compiler, DIV | WORD | RD(SLJIT_R0) | RS1(SLJIT_R0) | RS2(SLJIT_R1));
	case SLJIT_ENDBR:
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	if (op & SLJIT_32)
		flags = INT_DATA | SIGNED_DATA;
#endif

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
#endif
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, WORD_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, srcw);

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	case SLJIT_MOV_U32:
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u32)srcw : srcw);

	case SLJIT_MOV_S32:
	/* Logical operators have no W variant, so sign extended input is necessary for them. */
	case SLJIT_MOV32:
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s32)srcw : srcw);
#endif

	case SLJIT_MOV_U8:
		return emit_op(compiler, op, BYTE_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, op, BYTE_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, op, HALF_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, op, HALF_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_NOT:
		return emit_op(compiler, SLJIT_XOR | (op & (SLJIT_32 | SLJIT_SET_Z)), flags, dst, dstw, src, srcw, SLJIT_IMM, -1);

	case SLJIT_CLZ:
	case SLJIT_CTZ:
		return emit_op(compiler, op, flags, dst, dstw, TMP_REG1, 0, src, srcw);
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

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	if (op & SLJIT_32) {
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
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
		return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
	case SLJIT_SUBC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		compiler->status_flags_state = 0;
		return emit_op(compiler, op, flags | CUMULATIVE_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
	case SLJIT_ROTL:
	case SLJIT_ROTR:
		if (src2 & SLJIT_IMM) {
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
			src2w &= 0x1f;
#else /* !SLJIT_CONFIG_RISCV_32 */
			if (op & SLJIT_32)
				src2w &= 0x1f;
			else
				src2w &= 0x3f;
#endif /* SLJIT_CONFIG_RISCV_32 */
		}

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
	return sljit_emit_op2(compiler, op, TMP_REG2, 0, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 is_left;
	sljit_ins ins1, ins2, ins3;
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	sljit_ins word = (op & SLJIT_32) >> 5;
	sljit_s32 inp_flags = ((op & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
	sljit_sw bit_length = (op & SLJIT_32) ? 32 : 64;
#else /* !SLJIT_CONFIG_RISCV_64 */
	sljit_s32 inp_flags = WORD_DATA | LOAD_DATA;
	sljit_sw bit_length = 32;
#endif /* SLJIT_CONFIG_RISCV_64 */

	SLJIT_ASSERT(WORD == 0 || WORD == 0x8);

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, src_dst, src1, src1w, src2, src2w));

	is_left = (GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_MSHL);

	if (src_dst == src1) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, (is_left ? SLJIT_ROTL : SLJIT_ROTR) | (op & SLJIT_32), src_dst, 0, src_dst, 0, src2, src2w);
	}

	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	if (src2 & SLJIT_IMM) {
		src2w &= bit_length - 1;

		if (src2w == 0)
			return SLJIT_SUCCESS;
	} else if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG2, src2, src2w));
		src2 = TMP_REG2;
	}

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG1, src1, src1w));
		src1 = TMP_REG1;
	} else if (src1 & SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, src1w, TMP_REG3));
		src1 = TMP_REG1;
	}

	if (src2 & SLJIT_IMM) {
		if (is_left) {
			ins1 = SLLI | WORD | IMM_I(src2w);
			src2w = bit_length - src2w;
			ins2 = SRLI | WORD | IMM_I(src2w);
		} else {
			ins1 = SRLI | WORD | IMM_I(src2w);
			src2w = bit_length - src2w;
			ins2 = SLLI | WORD | IMM_I(src2w);
		}

		FAIL_IF(push_inst(compiler, ins1 | RD(src_dst) | RS1(src_dst)));
		FAIL_IF(push_inst(compiler, ins2 | RD(TMP_REG1) | RS1(src1)));
		return push_inst(compiler, OR | RD(src_dst) | RS1(src_dst) | RS2(TMP_REG1));
	}

	if (is_left) {
		ins1 = SLL;
		ins2 = SRLI;
		ins3 = SRL;
	} else {
		ins1 = SRL;
		ins2 = SLLI;
		ins3 = SLL;
	}

	FAIL_IF(push_inst(compiler, ins1 | WORD | RD(src_dst) | RS1(src_dst) | RS2(src2)));

	if (!(op & SLJIT_SHIFT_INTO_NON_ZERO)) {
		FAIL_IF(push_inst(compiler, ins2 | WORD | RD(TMP_REG1) | RS1(src1) | IMM_I(1)));
		FAIL_IF(push_inst(compiler, XORI | RD(TMP_REG2) | RS1(src2) | IMM_I((sljit_ins)bit_length - 1)));
		src1 = TMP_REG1;
	} else
		FAIL_IF(push_inst(compiler, SUB | WORD | RD(TMP_REG2) | RS1(TMP_ZERO) | RS2(src2)));

	FAIL_IF(push_inst(compiler, ins3 | WORD | RD(TMP_REG1) | RS1(src1) | RS2(TMP_REG2)));
	return push_inst(compiler, OR | RD(src_dst) | RS1(src_dst) | RS2(TMP_REG1));
}

#undef WORD

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, ADDI | RD(RETURN_ADDR_REG) | RS1(src) | IMM_I(0)));
		else
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, RETURN_ADDR_REG, src, srcw));

		return push_inst(compiler, JALR | RD(TMP_ZERO) | RS1(RETURN_ADDR_REG) | IMM_I(0));
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		return SLJIT_SUCCESS;
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
	return freg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_ins*)instruction);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_32) >> 7))
#define FMT(op) ((sljit_ins)((op & SLJIT_32) ^ SLJIT_32) << 17)

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#	define flags (sljit_u32)0
#else
	sljit_u32 flags = ((sljit_u32)(GET_OPCODE(op) == SLJIT_CONV_SW_FROM_F64)) << 21;
#endif
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src, srcw, dst, dstw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, FCVT_W_S | FMT(op) | flags | RD(dst_r) | FRS1(src)));

	/* Store the integer value from a VFP register. */
	if (dst & SLJIT_MEM) {
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
		return emit_op_mem2(compiler, WORD_DATA, TMP_REG2, dst, dstw, 0, 0);
#else
		return emit_op_mem2(compiler, flags ? WORD_DATA : INT_DATA, TMP_REG2, dst, dstw, 0, 0);
#endif
	}
	return SLJIT_SUCCESS;

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#	undef flags
#endif
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins inst;
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
	sljit_u32 flags = ((sljit_u32)(GET_OPCODE(op) == SLJIT_CONV_F64_FROM_SW)) << 21;
#endif

	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw, dst, dstw));
#else
		FAIL_IF(emit_op_mem2(compiler, (flags ? WORD_DATA : INT_DATA) | LOAD_DATA, TMP_REG1, src, srcw, dst, dstw));
#endif
		src = TMP_REG1;
	} else if (src & SLJIT_IMM) {
#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;
#endif

		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw, TMP_REG3));
		src = TMP_REG1;
	}

	inst = FCVT_S_W | FMT(op) | FRD(dst_r) | RS1(src);

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	if (op & SLJIT_32)
		inst |= F3(0x7);
#else
	inst |= flags;

	if (op != SLJIT_CONV_F64_FROM_S32)
		inst |= F3(0x7);
#endif

	FAIL_IF(push_inst(compiler, inst));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG1, dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_ins inst;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, src2, src2w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, 0, 0));
		src2 = TMP_FREG2;
	}

	switch (GET_FLAG_TYPE(op)) {
	case SLJIT_F_EQUAL:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
		inst = FEQ_S | FMT(op) | RD(OTHER_FLAG) | FRS1(src1) | FRS2(src2);
		break;
	case SLJIT_F_LESS:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		inst = FLT_S | FMT(op) | RD(OTHER_FLAG) | FRS1(src1) | FRS2(src2);
		break;
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		inst = FLT_S | FMT(op) | RD(OTHER_FLAG) | FRS1(src2) | FRS2(src1);
		break;
	case SLJIT_F_GREATER:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_ORDERED_LESS_EQUAL:
		inst = FLE_S | FMT(op) | RD(OTHER_FLAG) | FRS1(src1) | FRS2(src2);
		break;
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_ORDERED_GREATER_EQUAL:
		inst = FLE_S | FMT(op) | RD(OTHER_FLAG) | FRS1(src2) | FRS2(src1);
		break;
	case SLJIT_UNORDERED_OR_EQUAL: /* Not supported. */
	case SLJIT_ORDERED_NOT_EQUAL: /* Not supported. */
		FAIL_IF(push_inst(compiler, FLT_S | FMT(op) | RD(OTHER_FLAG) | FRS1(src1) | FRS2(src2)));
		FAIL_IF(push_inst(compiler, FLT_S | FMT(op) | RD(TMP_REG1) | FRS1(src2) | FRS2(src1)));
		inst = OR | RD(OTHER_FLAG) | RS1(OTHER_FLAG) | RS2(TMP_REG1);
		break;
	default: /* SLJIT_UNORDERED, SLJIT_ORDERED */
		FAIL_IF(push_inst(compiler, FADD_S | FMT(op) | FRD(TMP_FREG1) | FRS1(src1) | FRS2(src2)));
		inst = FEQ_S | FMT(op) | RD(OTHER_FLAG) | FRS1(TMP_FREG1) | FRS2(TMP_FREG1);
		break;
	}

	return push_inst(compiler, inst);
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
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, dst_r, src, srcw, dst, dstw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, FSGNJ_S | FMT(op) | FRD(dst_r) | FRS1(src) | FRS2(src)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, FSGNJN_S | FMT(op) | FRD(dst_r) | FRS1(src) | FRS2(src)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, FSGNJX_S | FMT(op) | FRD(dst_r) | FRS1(src) | FRS2(src)));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		/* The SLJIT_32 bit is inverted because sljit_f32 needs to be loaded from the memory. */
		FAIL_IF(push_inst(compiler, FCVT_S_D | ((op & SLJIT_32) ? (1 << 25) : ((1 << 20) | F3(7))) | FRD(dst_r) | FRS1(src)));
		op ^= SLJIT_32;
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw, 0, 0);
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
			flags |= SLOW_SRC1;
	}

	if (src2 & SLJIT_MEM) {
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w)) {
			FAIL_IF(compiler->error);
			src2 = TMP_FREG2;
		} else
			flags |= SLOW_SRC2;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, dst, dstw));
		}
	}
	else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, dst, dstw));

	if (flags & SLOW_SRC1)
		src1 = TMP_FREG1;
	if (flags & SLOW_SRC2)
		src2 = TMP_FREG2;

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, FADD_S | FMT(op) | FRD(dst_r) | FRS1(src1) | FRS2(src2)));
		break;

	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, FSUB_S | FMT(op) | FRD(dst_r) | FRS1(src1) | FRS2(src2)));
		break;

	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, FMUL_S | FMT(op) | FRD(dst_r) | FRS1(src1) | FRS2(src2)));
		break;

	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, FDIV_S | FMT(op) | FRD(dst_r) | FRS1(src1) | FRS2(src2)));
		break;
	}

	if (dst_r == TMP_FREG2)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG2, dst, dstw, 0, 0));

	return SLJIT_SUCCESS;
}

#undef FLOAT_DATA
#undef FMT

/* --------------------------------------------------------------------- */
/*  Other instructions                                                   */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	if (FAST_IS_REG(dst))
		return push_inst(compiler, ADDI | RD(dst) | RS1(RETURN_ADDR_REG) | IMM_I(0));

	/* Memory. */
	return emit_op_mem(compiler, WORD_DATA, RETURN_ADDR_REG, dst, dstw);
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

#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
#define BRANCH_LENGTH	((sljit_ins)(3 * sizeof(sljit_ins)) << 7)
#else
#define BRANCH_LENGTH	((sljit_ins)(7 * sizeof(sljit_ins)) << 7)
#endif

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins inst;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	switch (type) {
	case SLJIT_EQUAL:
		inst = BNE | RS1(EQUAL_FLAG) | RS2(TMP_ZERO) | BRANCH_LENGTH;
		break;
	case SLJIT_NOT_EQUAL:
		inst = BEQ | RS1(EQUAL_FLAG) | RS2(TMP_ZERO) | BRANCH_LENGTH;
		break;
	case SLJIT_LESS:
	case SLJIT_GREATER:
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER:
	case SLJIT_OVERFLOW:
	case SLJIT_CARRY:
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL: /* Not supported. */
	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_ORDERED:
		inst = BEQ | RS1(OTHER_FLAG) | RS2(TMP_ZERO) | BRANCH_LENGTH;
		break;
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_NOT_CARRY:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL: /* Not supported. */
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
	case SLJIT_F_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_UNORDERED:
		inst = BNE | RS1(OTHER_FLAG) | RS2(TMP_ZERO) | BRANCH_LENGTH;
		break;
	default:
		/* Not conditional branch. */
		inst = 0;
		break;
	}

	if (inst != 0) {
		PTR_FAIL_IF(push_inst(compiler, inst));
		jump->flags |= IS_COND;
	}

	jump->addr = compiler->size;
	inst = JALR | RS1(TMP_REG1) | IMM_I(0);

	if (type >= SLJIT_FAST_CALL) {
		jump->flags |= IS_CALL;
		inst |= RD(RETURN_ADDR_REG);
	}

	PTR_FAIL_IF(push_inst(compiler, inst));

	/* Maximum number of instructions required for generating a constant. */
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	compiler->size += 1;
#else
	compiler->size += 5;
#endif
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	SLJIT_UNUSED_ARG(arg_types);
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
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
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	flags = WORD_DATA | LOAD_DATA;
#else /* !SLJIT_CONFIG_RISCV_32 */
	flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
#endif /* SLJIT_CONFIG_RISCV_32 */

	if (src1 & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem2(compiler, flags, TMP_REG1, src1, src1w, src2, src2w));
		src1 = TMP_REG1;
	}

	if (src2 & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem2(compiler, flags, TMP_REG2, src2, src2w, 0, 0));
		src2 = TMP_REG2;
	}

	if (src1 & SLJIT_IMM) {
		if (src1w != 0) {
			PTR_FAIL_IF(load_immediate(compiler, TMP_REG1, src1w, TMP_REG3));
			src1 = TMP_REG1;
		}
		else
			src1 = TMP_ZERO;
	}

	if (src2 & SLJIT_IMM) {
		if (src2w != 0) {
			PTR_FAIL_IF(load_immediate(compiler, TMP_REG2, src2w, TMP_REG3));
			src2 = TMP_REG2;
		}
		else
			src2 = TMP_ZERO;
	}

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, (sljit_u32)((type & SLJIT_REWRITABLE_JUMP) | IS_COND));
	type &= 0xff;

	switch (type) {
	case SLJIT_EQUAL:
		inst = BNE | RS1(src1) | RS2(src2) | BRANCH_LENGTH;
		break;
	case SLJIT_NOT_EQUAL:
		inst = BEQ | RS1(src1) | RS2(src2) | BRANCH_LENGTH;
		break;
	case SLJIT_LESS:
		inst = BGEU | RS1(src1) | RS2(src2) | BRANCH_LENGTH;
		break;
	case SLJIT_GREATER_EQUAL:
		inst = BLTU | RS1(src1) | RS2(src2) | BRANCH_LENGTH;
		break;
	case SLJIT_GREATER:
		inst = BGEU | RS1(src2) | RS2(src1) | BRANCH_LENGTH;
		break;
	case SLJIT_LESS_EQUAL:
		inst = BLTU | RS1(src2) | RS2(src1) | BRANCH_LENGTH;
		break;
	case SLJIT_SIG_LESS:
		inst = BGE | RS1(src1) | RS2(src2) | BRANCH_LENGTH;
		break;
	case SLJIT_SIG_GREATER_EQUAL:
		inst = BLT | RS1(src1) | RS2(src2) | BRANCH_LENGTH;
		break;
	case SLJIT_SIG_GREATER:
		inst = BGE | RS1(src2) | RS2(src1) | BRANCH_LENGTH;
		break;
	case SLJIT_SIG_LESS_EQUAL:
		inst = BLT | RS1(src2) | RS2(src1) | BRANCH_LENGTH;
		break;
	}

	PTR_FAIL_IF(push_inst(compiler, inst));

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, JALR | RD(TMP_ZERO) | RS1(TMP_REG1) | IMM_I(0)));

	/* Maximum number of instructions required for generating a constant. */
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	compiler->size += 1;
#else
	compiler->size += 5;
#endif
	return jump;
}

#undef BRANCH_LENGTH

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));

	if (!(src & SLJIT_IMM)) {
		if (src & SLJIT_MEM) {
			ADJUST_LOCAL_OFFSET(src, srcw);
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
			src = TMP_REG1;
		}
		return push_inst(compiler, JALR | RD((type >= SLJIT_FAST_CALL) ? RETURN_ADDR_REG : TMP_ZERO) | RS1(src) | IMM_I(0));
	}

	/* These jumps are converted to jump/call instructions when possible. */
	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	FAIL_IF(!jump);
	set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_CALL : 0));
	jump->u.target = (sljit_uw)srcw;

	jump->addr = compiler->size;
	FAIL_IF(push_inst(compiler, JALR | RD((type >= SLJIT_FAST_CALL) ? RETURN_ADDR_REG : TMP_ZERO) | RS1(TMP_REG1) | IMM_I(0)));

	/* Maximum number of instructions required for generating a constant. */
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	compiler->size += 1;
#else
	compiler->size += 5;
#endif
	return SLJIT_SUCCESS;
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
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
		src = TMP_REG1;
	}

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
			FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG1) | RS1(src) | IMM_I(0)));
			src = TMP_REG1;
		}

		FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP;
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, type, src, srcw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_s32 src_r, dst_r, invert;
	sljit_s32 saved_op = op;
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	sljit_s32 mem_type = WORD_DATA;
#else
	sljit_s32 mem_type = ((op & SLJIT_32) || op == SLJIT_MOV32) ? (INT_DATA | SIGNED_DATA) : WORD_DATA;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	dst_r = (op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2;

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	if (op >= SLJIT_ADD && (dst & SLJIT_MEM))
		FAIL_IF(emit_op_mem2(compiler, mem_type | LOAD_DATA, TMP_REG1, dst, dstw, dst, dstw));

	if (type < SLJIT_F_EQUAL) {
		src_r = OTHER_FLAG;
		invert = type & 0x1;

		switch (type) {
		case SLJIT_EQUAL:
		case SLJIT_NOT_EQUAL:
			FAIL_IF(push_inst(compiler, SLTUI | RD(dst_r) | RS1(EQUAL_FLAG) | IMM_I(1)));
			src_r = dst_r;
			break;
		case SLJIT_OVERFLOW:
		case SLJIT_NOT_OVERFLOW:
			if (compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)) {
				src_r = OTHER_FLAG;
				break;
			}
			FAIL_IF(push_inst(compiler, SLTUI | RD(dst_r) | RS1(OTHER_FLAG) | IMM_I(1)));
			src_r = dst_r;
			invert ^= 0x1;
			break;
		}
	} else {
		invert = 0;
		src_r = OTHER_FLAG;

		switch (type) {
		case SLJIT_F_NOT_EQUAL:
		case SLJIT_UNORDERED_OR_NOT_EQUAL:
		case SLJIT_UNORDERED_OR_EQUAL: /* Not supported. */
		case SLJIT_F_GREATER_EQUAL:
		case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		case SLJIT_UNORDERED_OR_LESS_EQUAL:
		case SLJIT_F_GREATER:
		case SLJIT_UNORDERED_OR_GREATER:
		case SLJIT_UNORDERED_OR_LESS:
		case SLJIT_UNORDERED:
			invert = 1;
			break;
		}
	}

	if (invert) {
		FAIL_IF(push_inst(compiler, XORI | RD(dst_r) | RS1(src_r) | IMM_I(1)));
		src_r = dst_r;
	}

	if (op < SLJIT_ADD) {
		if (dst & SLJIT_MEM)
			return emit_op_mem(compiler, mem_type, src_r, dst, dstw);

		if (src_r != dst_r)
			return push_inst(compiler, ADDI | RD(dst_r) | RS1(src_r) | IMM_I(0));
		return SLJIT_SUCCESS;
	}

	mem_type |= CUMULATIVE_OP | IMM_OP | ALT_KEEP_CACHE;

	if (dst & SLJIT_MEM)
		return emit_op(compiler, saved_op, mem_type, dst, dstw, TMP_REG1, 0, src_r, 0);
	return emit_op(compiler, saved_op, mem_type, dst, dstw, dst, dstw, src_r, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_cmov(compiler, type, dst_reg, src, srcw));

	return sljit_emit_cmov_generic(compiler, type, dst_reg, src, srcw);;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 flags;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK))
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		memw &= 0x3;

		if (SLJIT_UNLIKELY(memw != 0)) {
			FAIL_IF(push_inst(compiler, SLLI | RD(TMP_REG1) | RS1(OFFS_REG(mem)) | IMM_I(memw)));
			FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG1) | RS1(TMP_REG1) | RS2(mem & REG_MASK)));
		} else
			FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG1) | RS1(mem & REG_MASK) | RS2(OFFS_REG(mem))));

		mem = TMP_REG1;
		memw = 0;
	} else if (memw > SIMM_MAX - SSIZE_OF(sw) || memw < SIMM_MIN) {
		if (((memw + 0x800) & 0xfff) <= 0xfff - SSIZE_OF(sw)) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, TO_ARGW_HI(memw), TMP_REG3));
			memw &= 0xfff;
		} else {
			FAIL_IF(load_immediate(compiler, TMP_REG1, memw, TMP_REG3));
			memw = 0;
		}

		if (mem & REG_MASK)
			FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG1) | RS1(TMP_REG1) | RS2(mem & REG_MASK)));

		mem = TMP_REG1;
	} else {
		mem &= REG_MASK;
		memw &= 0xfff;
	}

	SLJIT_ASSERT((memw >= 0 && memw <= SIMM_MAX - SSIZE_OF(sw)) || (memw > SIMM_MAX && memw <= 0xfff));

	if (!(type & SLJIT_MEM_STORE) && mem == REG_PAIR_FIRST(reg)) {
		FAIL_IF(push_mem_inst(compiler, WORD_DATA | LOAD_DATA, REG_PAIR_SECOND(reg), mem, (memw + SSIZE_OF(sw)) & 0xfff));
		return push_mem_inst(compiler, WORD_DATA | LOAD_DATA, REG_PAIR_FIRST(reg), mem, memw);
	}

	flags = WORD_DATA | (!(type & SLJIT_MEM_STORE) ? LOAD_DATA : 0);

	FAIL_IF(push_mem_inst(compiler, flags, REG_PAIR_FIRST(reg), mem, memw));
	return push_mem_inst(compiler, flags, REG_PAIR_SECOND(reg), mem, (memw + SSIZE_OF(sw)) & 0xfff);
}

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
	PTR_FAIL_IF(emit_const(compiler, dst_r, init_value, ADDI | RD(dst_r)));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw));

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
	PTR_FAIL_IF(push_inst(compiler, (sljit_ins)dst_r));
#if (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32)
	compiler->size += 1;
#else
	compiler->size += 5;
#endif

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw));

	return put_label;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}
