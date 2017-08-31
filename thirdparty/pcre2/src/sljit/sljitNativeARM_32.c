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
#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
	return "ARMv7" SLJIT_CPUINFO;
#elif (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	return "ARMv5" SLJIT_CPUINFO;
#else
#error "Internal error: Unknown ARM architecture"
#endif
}

/* Last register + 1. */
#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3	(SLJIT_NUMBER_OF_REGISTERS + 4)
#define TMP_PC		(SLJIT_NUMBER_OF_REGISTERS + 5)

#define TMP_FREG1	(0)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)

/* In ARM instruction words.
   Cache lines are usually 32 byte aligned. */
#define CONST_POOL_ALIGNMENT	8
#define CONST_POOL_EMPTY	0xffffffff

#define ALIGN_INSTRUCTION(ptr) \
	(sljit_uw*)(((sljit_uw)(ptr) + (CONST_POOL_ALIGNMENT * sizeof(sljit_uw)) - 1) & ~((CONST_POOL_ALIGNMENT * sizeof(sljit_uw)) - 1))
#define MAX_DIFFERENCE(max_diff) \
	(((max_diff) / (sljit_s32)sizeof(sljit_uw)) - (CONST_POOL_ALIGNMENT - 1))

/* See sljit_emit_enter and sljit_emit_op0 if you want to change them. */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 6] = {
	0, 0, 1, 2, 11, 10, 9, 8, 7, 6, 5, 4, 13, 3, 12, 14, 15
};

#define RM(rm) (reg_map[rm])
#define RD(rd) (reg_map[rd] << 12)
#define RN(rn) (reg_map[rn] << 16)

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

/* The instruction includes the AL condition.
   INST_NAME - CONDITIONAL remove this flag. */
#define COND_MASK	0xf0000000
#define CONDITIONAL	0xe0000000
#define PUSH_POOL	0xff000000

/* DP - Data Processing instruction (use with EMIT_DATA_PROCESS_INS). */
#define ADC_DP		0x5
#define ADD_DP		0x4
#define AND_DP		0x0
#define B		0xea000000
#define BIC_DP		0xe
#define BL		0xeb000000
#define BLX		0xe12fff30
#define BX		0xe12fff10
#define CLZ		0xe16f0f10
#define CMP_DP		0xa
#define BKPT		0xe1200070
#define EOR_DP		0x1
#define MOV_DP		0xd
#define MUL		0xe0000090
#define MVN_DP		0xf
#define NOP		0xe1a00000
#define ORR_DP		0xc
#define PUSH		0xe92d0000
#define POP		0xe8bd0000
#define RSB_DP		0x3
#define RSC_DP		0x7
#define SBC_DP		0x6
#define SMULL		0xe0c00090
#define SUB_DP		0x2
#define UMULL		0xe0800090
#define VABS_F32	0xeeb00ac0
#define VADD_F32	0xee300a00
#define VCMP_F32	0xeeb40a40
#define VCVT_F32_S32	0xeeb80ac0
#define VCVT_F64_F32	0xeeb70ac0
#define VCVT_S32_F32	0xeebd0ac0
#define VDIV_F32	0xee800a00
#define VMOV_F32	0xeeb00a40
#define VMOV		0xee000a10
#define VMRS		0xeef1fa10
#define VMUL_F32	0xee200a00
#define VNEG_F32	0xeeb10a40
#define VSTR_F32	0xed000a00
#define VSUB_F32	0xee300a40

#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
/* Arm v7 specific instructions. */
#define MOVW		0xe3000000
#define MOVT		0xe3400000
#define SXTB		0xe6af0070
#define SXTH		0xe6bf0070
#define UXTB		0xe6ef0070
#define UXTH		0xe6ff0070
#endif

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)

static sljit_s32 push_cpool(struct sljit_compiler *compiler)
{
	/* Pushing the constant pool into the instruction stream. */
	sljit_uw* inst;
	sljit_uw* cpool_ptr;
	sljit_uw* cpool_end;
	sljit_s32 i;

	/* The label could point the address after the constant pool. */
	if (compiler->last_label && compiler->last_label->size == compiler->size)
		compiler->last_label->size += compiler->cpool_fill + (CONST_POOL_ALIGNMENT - 1) + 1;

	SLJIT_ASSERT(compiler->cpool_fill > 0 && compiler->cpool_fill <= CPOOL_SIZE);
	inst = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
	FAIL_IF(!inst);
	compiler->size++;
	*inst = 0xff000000 | compiler->cpool_fill;

	for (i = 0; i < CONST_POOL_ALIGNMENT - 1; i++) {
		inst = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
		FAIL_IF(!inst);
		compiler->size++;
		*inst = 0;
	}

	cpool_ptr = compiler->cpool;
	cpool_end = cpool_ptr + compiler->cpool_fill;
	while (cpool_ptr < cpool_end) {
		inst = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
		FAIL_IF(!inst);
		compiler->size++;
		*inst = *cpool_ptr++;
	}
	compiler->cpool_diff = CONST_POOL_EMPTY;
	compiler->cpool_fill = 0;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_uw inst)
{
	sljit_uw* ptr;

	if (SLJIT_UNLIKELY(compiler->cpool_diff != CONST_POOL_EMPTY && compiler->size - compiler->cpool_diff >= MAX_DIFFERENCE(4092)))
		FAIL_IF(push_cpool(compiler));

	ptr = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst_with_literal(struct sljit_compiler *compiler, sljit_uw inst, sljit_uw literal)
{
	sljit_uw* ptr;
	sljit_uw cpool_index = CPOOL_SIZE;
	sljit_uw* cpool_ptr;
	sljit_uw* cpool_end;
	sljit_u8* cpool_unique_ptr;

	if (SLJIT_UNLIKELY(compiler->cpool_diff != CONST_POOL_EMPTY && compiler->size - compiler->cpool_diff >= MAX_DIFFERENCE(4092)))
		FAIL_IF(push_cpool(compiler));
	else if (compiler->cpool_fill > 0) {
		cpool_ptr = compiler->cpool;
		cpool_end = cpool_ptr + compiler->cpool_fill;
		cpool_unique_ptr = compiler->cpool_unique;
		do {
			if ((*cpool_ptr == literal) && !(*cpool_unique_ptr)) {
				cpool_index = cpool_ptr - compiler->cpool;
				break;
			}
			cpool_ptr++;
			cpool_unique_ptr++;
		} while (cpool_ptr < cpool_end);
	}

	if (cpool_index == CPOOL_SIZE) {
		/* Must allocate a new entry in the literal pool. */
		if (compiler->cpool_fill < CPOOL_SIZE) {
			cpool_index = compiler->cpool_fill;
			compiler->cpool_fill++;
		}
		else {
			FAIL_IF(push_cpool(compiler));
			cpool_index = 0;
			compiler->cpool_fill = 1;
		}
	}

	SLJIT_ASSERT((inst & 0xfff) == 0);
	ptr = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst | cpool_index;

	compiler->cpool[cpool_index] = literal;
	compiler->cpool_unique[cpool_index] = 0;
	if (compiler->cpool_diff == CONST_POOL_EMPTY)
		compiler->cpool_diff = compiler->size;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst_with_unique_literal(struct sljit_compiler *compiler, sljit_uw inst, sljit_uw literal)
{
	sljit_uw* ptr;
	if (SLJIT_UNLIKELY((compiler->cpool_diff != CONST_POOL_EMPTY && compiler->size - compiler->cpool_diff >= MAX_DIFFERENCE(4092)) || compiler->cpool_fill >= CPOOL_SIZE))
		FAIL_IF(push_cpool(compiler));

	SLJIT_ASSERT(compiler->cpool_fill < CPOOL_SIZE && (inst & 0xfff) == 0);
	ptr = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst | compiler->cpool_fill;

	compiler->cpool[compiler->cpool_fill] = literal;
	compiler->cpool_unique[compiler->cpool_fill] = 1;
	compiler->cpool_fill++;
	if (compiler->cpool_diff == CONST_POOL_EMPTY)
		compiler->cpool_diff = compiler->size;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 prepare_blx(struct sljit_compiler *compiler)
{
	/* Place for at least two instruction (doesn't matter whether the first has a literal). */
	if (SLJIT_UNLIKELY(compiler->cpool_diff != CONST_POOL_EMPTY && compiler->size - compiler->cpool_diff >= MAX_DIFFERENCE(4088)))
		return push_cpool(compiler);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_blx(struct sljit_compiler *compiler)
{
	/* Must follow tightly the previous instruction (to be able to convert it to bl instruction). */
	SLJIT_ASSERT(compiler->cpool_diff == CONST_POOL_EMPTY || compiler->size - compiler->cpool_diff < MAX_DIFFERENCE(4092));
	return push_inst(compiler, BLX | RM(TMP_REG1));
}

static sljit_uw patch_pc_relative_loads(sljit_uw *last_pc_patch, sljit_uw *code_ptr, sljit_uw* const_pool, sljit_uw cpool_size)
{
	sljit_uw diff;
	sljit_uw ind;
	sljit_uw counter = 0;
	sljit_uw* clear_const_pool = const_pool;
	sljit_uw* clear_const_pool_end = const_pool + cpool_size;

	SLJIT_ASSERT(const_pool - code_ptr <= CONST_POOL_ALIGNMENT);
	/* Set unused flag for all literals in the constant pool.
	   I.e.: unused literals can belong to branches, which can be encoded as B or BL.
	   We can "compress" the constant pool by discarding these literals. */
	while (clear_const_pool < clear_const_pool_end)
		*clear_const_pool++ = (sljit_uw)(-1);

	while (last_pc_patch < code_ptr) {
		/* Data transfer instruction with Rn == r15. */
		if ((*last_pc_patch & 0x0c0f0000) == 0x040f0000) {
			diff = const_pool - last_pc_patch;
			ind = (*last_pc_patch) & 0xfff;

			/* Must be a load instruction with immediate offset. */
			SLJIT_ASSERT(ind < cpool_size && !(*last_pc_patch & (1 << 25)) && (*last_pc_patch & (1 << 20)));
			if ((sljit_s32)const_pool[ind] < 0) {
				const_pool[ind] = counter;
				ind = counter;
				counter++;
			}
			else
				ind = const_pool[ind];

			SLJIT_ASSERT(diff >= 1);
			if (diff >= 2 || ind > 0) {
				diff = (diff + ind - 2) << 2;
				SLJIT_ASSERT(diff <= 0xfff);
				*last_pc_patch = (*last_pc_patch & ~0xfff) | diff;
			}
			else
				*last_pc_patch = (*last_pc_patch & ~(0xfff | (1 << 23))) | 0x004;
		}
		last_pc_patch++;
	}
	return counter;
}

/* In some rare ocasions we may need future patches. The probability is close to 0 in practice. */
struct future_patch {
	struct future_patch* next;
	sljit_s32 index;
	sljit_s32 value;
};

static sljit_s32 resolve_const_pool_index(struct sljit_compiler *compiler, struct future_patch **first_patch, sljit_uw cpool_current_index, sljit_uw *cpool_start_address, sljit_uw *buf_ptr)
{
	sljit_s32 value;
	struct future_patch *curr_patch, *prev_patch;

	SLJIT_UNUSED_ARG(compiler);

	/* Using the values generated by patch_pc_relative_loads. */
	if (!*first_patch)
		value = (sljit_s32)cpool_start_address[cpool_current_index];
	else {
		curr_patch = *first_patch;
		prev_patch = NULL;
		while (1) {
			if (!curr_patch) {
				value = (sljit_s32)cpool_start_address[cpool_current_index];
				break;
			}
			if ((sljit_uw)curr_patch->index == cpool_current_index) {
				value = curr_patch->value;
				if (prev_patch)
					prev_patch->next = curr_patch->next;
				else
					*first_patch = curr_patch->next;
				SLJIT_FREE(curr_patch, compiler->allocator_data);
				break;
			}
			prev_patch = curr_patch;
			curr_patch = curr_patch->next;
		}
	}

	if (value >= 0) {
		if ((sljit_uw)value > cpool_current_index) {
			curr_patch = (struct future_patch*)SLJIT_MALLOC(sizeof(struct future_patch), compiler->allocator_data);
			if (!curr_patch) {
				while (*first_patch) {
					curr_patch = *first_patch;
					*first_patch = (*first_patch)->next;
					SLJIT_FREE(curr_patch, compiler->allocator_data);
				}
				return SLJIT_ERR_ALLOC_FAILED;
			}
			curr_patch->next = *first_patch;
			curr_patch->index = value;
			curr_patch->value = cpool_start_address[value];
			*first_patch = curr_patch;
		}
		cpool_start_address[value] = *buf_ptr;
	}
	return SLJIT_SUCCESS;
}

#else

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_uw inst)
{
	sljit_uw* ptr;

	ptr = (sljit_uw*)ensure_buf(compiler, sizeof(sljit_uw));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_imm(struct sljit_compiler *compiler, sljit_s32 reg, sljit_sw imm)
{
	FAIL_IF(push_inst(compiler, MOVW | RD(reg) | ((imm << 4) & 0xf0000) | (imm & 0xfff)));
	return push_inst(compiler, MOVT | RD(reg) | ((imm >> 12) & 0xf0000) | ((imm >> 16) & 0xfff));
}

#endif

static SLJIT_INLINE sljit_s32 detect_jump_type(struct sljit_jump *jump, sljit_uw *code_ptr, sljit_uw *code, sljit_sw executable_offset)
{
	sljit_sw diff;

	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		return 0;

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	if (jump->flags & IS_BL)
		code_ptr--;

	if (jump->flags & JUMP_ADDR)
		diff = ((sljit_sw)jump->u.target - (sljit_sw)(code_ptr + 2) - executable_offset);
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		diff = ((sljit_sw)(code + jump->u.label->size) - (sljit_sw)(code_ptr + 2));
	}

	/* Branch to Thumb code has not been optimized yet. */
	if (diff & 0x3)
		return 0;

	if (jump->flags & IS_BL) {
		if (diff <= 0x01ffffff && diff >= -0x02000000) {
			*code_ptr = (BL - CONDITIONAL) | (*(code_ptr + 1) & COND_MASK);
			jump->flags |= PATCH_B;
			return 1;
		}
	}
	else {
		if (diff <= 0x01ffffff && diff >= -0x02000000) {
			*code_ptr = (B - CONDITIONAL) | (*code_ptr & COND_MASK);
			jump->flags |= PATCH_B;
		}
	}
#else
	if (jump->flags & JUMP_ADDR)
		diff = ((sljit_sw)jump->u.target - (sljit_sw)code_ptr - executable_offset);
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		diff = ((sljit_sw)(code + jump->u.label->size) - (sljit_sw)code_ptr);
	}

	/* Branch to Thumb code has not been optimized yet. */
	if (diff & 0x3)
		return 0;

	if (diff <= 0x01ffffff && diff >= -0x02000000) {
		code_ptr -= 2;
		*code_ptr = ((jump->flags & IS_BL) ? (BL - CONDITIONAL) : (B - CONDITIONAL)) | (code_ptr[2] & COND_MASK);
		jump->flags |= PATCH_B;
		return 1;
	}
#endif
	return 0;
}

static SLJIT_INLINE void inline_set_jump_addr(sljit_uw jump_ptr, sljit_sw executable_offset, sljit_uw new_addr, sljit_s32 flush_cache)
{
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	sljit_uw *ptr = (sljit_uw *)jump_ptr;
	sljit_uw *inst = (sljit_uw *)ptr[0];
	sljit_uw mov_pc = ptr[1];
	sljit_s32 bl = (mov_pc & 0x0000f000) != RD(TMP_PC);
	sljit_sw diff = (sljit_sw)(((sljit_sw)new_addr - (sljit_sw)(inst + 2) - executable_offset) >> 2);

	if (diff <= 0x7fffff && diff >= -0x800000) {
		/* Turn to branch. */
		if (!bl) {
			inst[0] = (mov_pc & COND_MASK) | (B - CONDITIONAL) | (diff & 0xffffff);
			if (flush_cache) {
				inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
				SLJIT_CACHE_FLUSH(inst, inst + 1);
			}
		} else {
			inst[0] = (mov_pc & COND_MASK) | (BL - CONDITIONAL) | (diff & 0xffffff);
			inst[1] = NOP;
			if (flush_cache) {
				inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
				SLJIT_CACHE_FLUSH(inst, inst + 2);
			}
		}
	} else {
		/* Get the position of the constant. */
		if (mov_pc & (1 << 23))
			ptr = inst + ((mov_pc & 0xfff) >> 2) + 2;
		else
			ptr = inst + 1;

		if (*inst != mov_pc) {
			inst[0] = mov_pc;
			if (!bl) {
				if (flush_cache) {
					inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
					SLJIT_CACHE_FLUSH(inst, inst + 1);
				}
			} else {
				inst[1] = BLX | RM(TMP_REG1);
				if (flush_cache) {
					inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
					SLJIT_CACHE_FLUSH(inst, inst + 2);
				}
			}
		}
		*ptr = new_addr;
	}
#else
	sljit_uw *inst = (sljit_uw*)jump_ptr;
	SLJIT_ASSERT((inst[0] & 0xfff00000) == MOVW && (inst[1] & 0xfff00000) == MOVT);
	inst[0] = MOVW | (inst[0] & 0xf000) | ((new_addr << 4) & 0xf0000) | (new_addr & 0xfff);
	inst[1] = MOVT | (inst[1] & 0xf000) | ((new_addr >> 12) & 0xf0000) | ((new_addr >> 16) & 0xfff);
	if (flush_cache) {
		inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
		SLJIT_CACHE_FLUSH(inst, inst + 2);
	}
#endif
}

static sljit_uw get_imm(sljit_uw imm);

static SLJIT_INLINE void inline_set_const(sljit_uw addr, sljit_sw executable_offset, sljit_sw new_constant, sljit_s32 flush_cache)
{
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	sljit_uw *ptr = (sljit_uw*)addr;
	sljit_uw *inst = (sljit_uw*)ptr[0];
	sljit_uw ldr_literal = ptr[1];
	sljit_uw src2;

	src2 = get_imm(new_constant);
	if (src2) {
		*inst = 0xe3a00000 | (ldr_literal & 0xf000) | src2;
		if (flush_cache) {
			inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
			SLJIT_CACHE_FLUSH(inst, inst + 1);
		}
		return;
	}

	src2 = get_imm(~new_constant);
	if (src2) {
		*inst = 0xe3e00000 | (ldr_literal & 0xf000) | src2;
		if (flush_cache) {
			inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
			SLJIT_CACHE_FLUSH(inst, inst + 1);
		}
		return;
	}

	if (ldr_literal & (1 << 23))
		ptr = inst + ((ldr_literal & 0xfff) >> 2) + 2;
	else
		ptr = inst + 1;

	if (*inst != ldr_literal) {
		*inst = ldr_literal;
		if (flush_cache) {
			inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
			SLJIT_CACHE_FLUSH(inst, inst + 1);
		}
	}
	*ptr = new_constant;
#else
	sljit_uw *inst = (sljit_uw*)addr;
	SLJIT_ASSERT((inst[0] & 0xfff00000) == MOVW && (inst[1] & 0xfff00000) == MOVT);
	inst[0] = MOVW | (inst[0] & 0xf000) | ((new_constant << 4) & 0xf0000) | (new_constant & 0xfff);
	inst[1] = MOVT | (inst[1] & 0xf000) | ((new_constant >> 12) & 0xf0000) | ((new_constant >> 16) & 0xfff);
	if (flush_cache) {
		inst = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
		SLJIT_CACHE_FLUSH(inst, inst + 2);
	}
#endif
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_uw *code;
	sljit_uw *code_ptr;
	sljit_uw *buf_ptr;
	sljit_uw *buf_end;
	sljit_uw size;
	sljit_uw word_count;
	sljit_sw executable_offset;
	sljit_sw jump_addr;
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	sljit_uw cpool_size;
	sljit_uw cpool_skip_alignment;
	sljit_uw cpool_current_index;
	sljit_uw *cpool_start_address;
	sljit_uw *last_pc_patch;
	struct future_patch *first_patch;
#endif

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	/* Second code generation pass. */
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	size = compiler->size + (compiler->patches << 1);
	if (compiler->cpool_fill > 0)
		size += compiler->cpool_fill + CONST_POOL_ALIGNMENT - 1;
#else
	size = compiler->size;
#endif
	code = (sljit_uw*)SLJIT_MALLOC_EXEC(size * sizeof(sljit_uw));
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	cpool_size = 0;
	cpool_skip_alignment = 0;
	cpool_current_index = 0;
	cpool_start_address = NULL;
	first_patch = NULL;
	last_pc_patch = code;
#endif

	code_ptr = code;
	word_count = 0;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;

	if (label && label->size == 0) {
		label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
		label = label->next;
	}

	do {
		buf_ptr = (sljit_uw*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			word_count++;
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
			if (cpool_size > 0) {
				if (cpool_skip_alignment > 0) {
					buf_ptr++;
					cpool_skip_alignment--;
				}
				else {
					if (SLJIT_UNLIKELY(resolve_const_pool_index(compiler, &first_patch, cpool_current_index, cpool_start_address, buf_ptr))) {
						SLJIT_FREE_EXEC(code);
						compiler->error = SLJIT_ERR_ALLOC_FAILED;
						return NULL;
					}
					buf_ptr++;
					if (++cpool_current_index >= cpool_size) {
						SLJIT_ASSERT(!first_patch);
						cpool_size = 0;
						if (label && label->size == word_count) {
							/* Points after the current instruction. */
							label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
							label->size = code_ptr - code;
							label = label->next;
						}
					}
				}
			}
			else if ((*buf_ptr & 0xff000000) != PUSH_POOL) {
#endif
				*code_ptr = *buf_ptr++;
				/* These structures are ordered by their address. */
				SLJIT_ASSERT(!label || label->size >= word_count);
				SLJIT_ASSERT(!jump || jump->addr >= word_count);
				SLJIT_ASSERT(!const_ || const_->addr >= word_count);
				if (jump && jump->addr == word_count) {
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
					if (detect_jump_type(jump, code_ptr, code, executable_offset))
						code_ptr--;
					jump->addr = (sljit_uw)code_ptr;
#else
					jump->addr = (sljit_uw)(code_ptr - 2);
					if (detect_jump_type(jump, code_ptr, code, executable_offset))
						code_ptr -= 2;
#endif
					jump = jump->next;
				}
				if (label && label->size == word_count) {
					/* code_ptr can be affected above. */
					label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr + 1, executable_offset);
					label->size = (code_ptr + 1) - code;
					label = label->next;
				}
				if (const_ && const_->addr == word_count) {
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
					const_->addr = (sljit_uw)code_ptr;
#else
					const_->addr = (sljit_uw)(code_ptr - 1);
#endif
					const_ = const_->next;
				}
				code_ptr++;
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
			}
			else {
				/* Fortunately, no need to shift. */
				cpool_size = *buf_ptr++ & ~PUSH_POOL;
				SLJIT_ASSERT(cpool_size > 0);
				cpool_start_address = ALIGN_INSTRUCTION(code_ptr + 1);
				cpool_current_index = patch_pc_relative_loads(last_pc_patch, code_ptr, cpool_start_address, cpool_size);
				if (cpool_current_index > 0) {
					/* Unconditional branch. */
					*code_ptr = B | (((cpool_start_address - code_ptr) + cpool_current_index - 2) & ~PUSH_POOL);
					code_ptr = cpool_start_address + cpool_current_index;
				}
				cpool_skip_alignment = CONST_POOL_ALIGNMENT - 1;
				cpool_current_index = 0;
				last_pc_patch = code_ptr;
			}
#endif
		} while (buf_ptr < buf_end);
		buf = buf->next;
	} while (buf);

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	SLJIT_ASSERT(cpool_size == 0);
	if (compiler->cpool_fill > 0) {
		cpool_start_address = ALIGN_INSTRUCTION(code_ptr);
		cpool_current_index = patch_pc_relative_loads(last_pc_patch, code_ptr, cpool_start_address, compiler->cpool_fill);
		if (cpool_current_index > 0)
			code_ptr = cpool_start_address + cpool_current_index;

		buf_ptr = compiler->cpool;
		buf_end = buf_ptr + compiler->cpool_fill;
		cpool_current_index = 0;
		while (buf_ptr < buf_end) {
			if (SLJIT_UNLIKELY(resolve_const_pool_index(compiler, &first_patch, cpool_current_index, cpool_start_address, buf_ptr))) {
				SLJIT_FREE_EXEC(code);
				compiler->error = SLJIT_ERR_ALLOC_FAILED;
				return NULL;
			}
			buf_ptr++;
			cpool_current_index++;
		}
		SLJIT_ASSERT(!first_patch);
	}
#endif

	jump = compiler->jumps;
	while (jump) {
		buf_ptr = (sljit_uw *)jump->addr;

		if (jump->flags & PATCH_B) {
			jump_addr = (sljit_sw)SLJIT_ADD_EXEC_OFFSET(buf_ptr + 2, executable_offset);
			if (!(jump->flags & JUMP_ADDR)) {
				SLJIT_ASSERT(jump->flags & JUMP_LABEL);
				SLJIT_ASSERT(((sljit_sw)jump->u.label->addr - jump_addr) <= 0x01ffffff && ((sljit_sw)jump->u.label->addr - jump_addr) >= -0x02000000);
				*buf_ptr |= (((sljit_sw)jump->u.label->addr - jump_addr) >> 2) & 0x00ffffff;
			}
			else {
				SLJIT_ASSERT(((sljit_sw)jump->u.target - jump_addr) <= 0x01ffffff && ((sljit_sw)jump->u.target - jump_addr) >= -0x02000000);
				*buf_ptr |= (((sljit_sw)jump->u.target - jump_addr) >> 2) & 0x00ffffff;
			}
		}
		else if (jump->flags & SLJIT_REWRITABLE_JUMP) {
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
			jump->addr = (sljit_uw)code_ptr;
			code_ptr[0] = (sljit_uw)buf_ptr;
			code_ptr[1] = *buf_ptr;
			inline_set_jump_addr((sljit_uw)code_ptr, executable_offset, (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target, 0);
			code_ptr += 2;
#else
			inline_set_jump_addr((sljit_uw)buf_ptr, executable_offset, (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target, 0);
#endif
		}
		else {
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
			if (jump->flags & IS_BL)
				buf_ptr--;
			if (*buf_ptr & (1 << 23))
				buf_ptr += ((*buf_ptr & 0xfff) >> 2) + 2;
			else
				buf_ptr += 1;
			*buf_ptr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
#else
			inline_set_jump_addr((sljit_uw)buf_ptr, executable_offset, (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target, 0);
#endif
		}
		jump = jump->next;
	}

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	const_ = compiler->consts;
	while (const_) {
		buf_ptr = (sljit_uw*)const_->addr;
		const_->addr = (sljit_uw)code_ptr;

		code_ptr[0] = (sljit_uw)buf_ptr;
		code_ptr[1] = *buf_ptr;
		if (*buf_ptr & (1 << 23))
			buf_ptr += ((*buf_ptr & 0xfff) >> 2) + 2;
		else
			buf_ptr += 1;
		/* Set the value again (can be a simple constant). */
		inline_set_const((sljit_uw)code_ptr, executable_offset, *buf_ptr, 0);
		code_ptr += 2;

		const_ = const_->next;
	}
#endif

	SLJIT_ASSERT(code_ptr - code <= (sljit_s32)size);

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (code_ptr - code) * sizeof(sljit_uw);

	code = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
	code_ptr = (sljit_uw *)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

	SLJIT_CACHE_FLUSH(code, code_ptr);
	return code;
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

/* emit_op inp_flags.
   WRITE_BACK must be the first, since it is a flag. */
#define WRITE_BACK	0x01
#define ALLOW_IMM	0x02
#define ALLOW_INV_IMM	0x04
#define ALLOW_ANY_IMM	(ALLOW_IMM | ALLOW_INV_IMM)
#define ARG_TEST	0x08

/* Creates an index in data_transfer_insts array. */
#define WORD_DATA	0x00
#define BYTE_DATA	0x10
#define HALF_DATA	0x20
#define SIGNED_DATA	0x40
#define LOAD_DATA	0x80

/* Condition: AL. */
#define EMIT_DATA_PROCESS_INS(opcode, set_flags, dst, src1, src2) \
	(0xe0000000 | ((opcode) << 21) | (set_flags) | RD(dst) | RN(src1) | (src2))

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 inp_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 size, i, tmp;
	sljit_uw push;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	/* Push saved registers, temporary registers
	   stmdb sp!, {..., lr} */
	push = PUSH | (1 << 14);

	tmp = saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = SLJIT_S0; i >= tmp; i--)
		push |= 1 << reg_map[i];

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--)
		push |= 1 << reg_map[i];

	FAIL_IF(push_inst(compiler, push));

	/* Stack must be aligned to 8 bytes: */
	size = GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1);
	local_size = ((size + local_size + 7) & ~7) - size;
	compiler->local_size = local_size;
	if (local_size > 0)
		FAIL_IF(emit_op(compiler, SLJIT_SUB, ALLOW_IMM, SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, local_size));

	if (args >= 1)
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, SLJIT_S0, SLJIT_UNUSED, RM(SLJIT_R0))));
	if (args >= 2)
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, SLJIT_S1, SLJIT_UNUSED, RM(SLJIT_R1))));
	if (args >= 3)
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, SLJIT_S2, SLJIT_UNUSED, RM(SLJIT_R2))));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 size;

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	size = GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1);
	compiler->local_size = ((size + local_size + 7) & ~7) - size;
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 i, tmp;
	sljit_uw pop;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));

	FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));

	if (compiler->local_size > 0)
		FAIL_IF(emit_op(compiler, SLJIT_ADD, ALLOW_IMM, SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, compiler->local_size));

	/* Push saved registers, temporary registers
	   ldmia sp!, {..., pc} */
	pop = POP | (1 << 15);

	tmp = compiler->saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - compiler->saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = SLJIT_S0; i >= tmp; i--)
		pop |= 1 << reg_map[i];

	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--)
		pop |= 1 << reg_map[i];

	return push_inst(compiler, pop);
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

/* s/l - store/load (1 bit)
   u/s - signed/unsigned (1 bit)
   w/b/h/N - word/byte/half/NOT allowed (2 bit)
   It contans 16 items, but not all are different. */

static sljit_sw data_transfer_insts[16] = {
/* s u w */ 0xe5000000 /* str */,
/* s u b */ 0xe5400000 /* strb */,
/* s u h */ 0xe10000b0 /* strh */,
/* s u N */ 0x00000000 /* not allowed */,
/* s s w */ 0xe5000000 /* str */,
/* s s b */ 0xe5400000 /* strb */,
/* s s h */ 0xe10000b0 /* strh */,
/* s s N */ 0x00000000 /* not allowed */,

/* l u w */ 0xe5100000 /* ldr */,
/* l u b */ 0xe5500000 /* ldrb */,
/* l u h */ 0xe11000b0 /* ldrh */,
/* l u N */ 0x00000000 /* not allowed */,
/* l s w */ 0xe5100000 /* ldr */,
/* l s b */ 0xe11000d0 /* ldrsb */,
/* l s h */ 0xe11000f0 /* ldrsh */,
/* l s N */ 0x00000000 /* not allowed */,
};

#define EMIT_DATA_TRANSFER(type, add, wb, target, base1, base2) \
	(data_transfer_insts[(type) >> 4] | ((add) << 23) | ((wb) << 21) | (reg_map[target] << 12) | (reg_map[base1] << 16) | (base2))
/* Normal ldr/str instruction.
   Type2: ldrsb, ldrh, ldrsh */
#define IS_TYPE1_TRANSFER(type) \
	(data_transfer_insts[(type) >> 4] & 0x04000000)
#define TYPE2_TRANSFER_IMM(imm) \
	(((imm) & 0xf) | (((imm) & 0xf0) << 4) | (1 << 22))

/* flags: */
  /* Arguments are swapped. */
#define ARGS_SWAPPED	0x01
  /* Inverted immediate. */
#define INV_IMM		0x02
  /* Source and destination is register. */
#define REG_DEST	0x04
#define REG_SOURCE	0x08
  /* One instruction is enough. */
#define FAST_DEST	0x10
  /* Multiple instructions are required. */
#define SLOW_DEST	0x20
/* SET_FLAGS must be (1 << 20) as it is also the value of S bit (can be used for optimization). */
#define SET_FLAGS	(1 << 20)
/* dst: reg
   src1: reg
   src2: reg or imm (if allowed)
   SRC2_IMM must be (1 << 25) as it is also the value of I bit (can be used for optimization). */
#define SRC2_IMM	(1 << 25)

#define EMIT_DATA_PROCESS_INS_AND_RETURN(opcode) \
	return push_inst(compiler, EMIT_DATA_PROCESS_INS(opcode, flags & SET_FLAGS, dst, src1, (src2 & SRC2_IMM) ? src2 : RM(src2)))

#define EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(opcode, dst, src1, src2) \
	return push_inst(compiler, EMIT_DATA_PROCESS_INS(opcode, flags & SET_FLAGS, dst, src1, src2))

#define EMIT_SHIFT_INS_AND_RETURN(opcode) \
	SLJIT_ASSERT(!(flags & INV_IMM) && !(src2 & SRC2_IMM)); \
	if (compiler->shift_imm != 0x20) { \
		SLJIT_ASSERT(src1 == TMP_REG1); \
		SLJIT_ASSERT(!(flags & ARGS_SWAPPED)); \
		if (compiler->shift_imm != 0) \
			return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, flags & SET_FLAGS, dst, SLJIT_UNUSED, (compiler->shift_imm << 7) | (opcode << 5) | reg_map[src2])); \
		return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, flags & SET_FLAGS, dst, SLJIT_UNUSED, reg_map[src2])); \
	} \
	return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, flags & SET_FLAGS, dst, SLJIT_UNUSED, (reg_map[(flags & ARGS_SWAPPED) ? src1 : src2] << 8) | (opcode << 5) | 0x10 | ((flags & ARGS_SWAPPED) ? reg_map[src2] : reg_map[src1])));

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_s32 src2)
{
	sljit_sw mul_inst;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		if (dst != src2) {
			if (src2 & SRC2_IMM) {
				if (flags & INV_IMM)
					EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MVN_DP, dst, SLJIT_UNUSED, src2);
				EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MOV_DP, dst, SLJIT_UNUSED, src2);
			}
			EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MOV_DP, dst, SLJIT_UNUSED, reg_map[src2]);
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		if ((flags & (REG_DEST | REG_SOURCE)) == (REG_DEST | REG_SOURCE)) {
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
			if (op == SLJIT_MOV_U8)
				return push_inst(compiler, EMIT_DATA_PROCESS_INS(AND_DP, 0, dst, src2, SRC2_IMM | 0xff));
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst, SLJIT_UNUSED, (24 << 7) | reg_map[src2])));
			return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst, SLJIT_UNUSED, (24 << 7) | (op == SLJIT_MOV_U8 ? 0x20 : 0x40) | reg_map[dst]));
#else
			return push_inst(compiler, (op == SLJIT_MOV_U8 ? UXTB : SXTB) | RD(dst) | RM(src2));
#endif
		}
		else if (dst != src2) {
			SLJIT_ASSERT(src2 & SRC2_IMM);
			if (flags & INV_IMM)
				EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MVN_DP, dst, SLJIT_UNUSED, src2);
			EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MOV_DP, dst, SLJIT_UNUSED, src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		if ((flags & (REG_DEST | REG_SOURCE)) == (REG_DEST | REG_SOURCE)) {
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst, SLJIT_UNUSED, (16 << 7) | reg_map[src2])));
			return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst, SLJIT_UNUSED, (16 << 7) | (op == SLJIT_MOV_U16 ? 0x20 : 0x40) | reg_map[dst]));
#else
			return push_inst(compiler, (op == SLJIT_MOV_U16 ? UXTH : SXTH) | RD(dst) | RM(src2));
#endif
		}
		else if (dst != src2) {
			SLJIT_ASSERT(src2 & SRC2_IMM);
			if (flags & INV_IMM)
				EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MVN_DP, dst, SLJIT_UNUSED, src2);
			EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MOV_DP, dst, SLJIT_UNUSED, src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_NOT:
		if (src2 & SRC2_IMM) {
			if (flags & INV_IMM)
				EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MOV_DP, dst, SLJIT_UNUSED, src2);
			EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MVN_DP, dst, SLJIT_UNUSED, src2);
		}
		EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(MVN_DP, dst, SLJIT_UNUSED, RM(src2));

	case SLJIT_CLZ:
		SLJIT_ASSERT(!(flags & INV_IMM));
		SLJIT_ASSERT(!(src2 & SRC2_IMM));
		FAIL_IF(push_inst(compiler, CLZ | RD(dst) | RM(src2)));
		if (flags & SET_FLAGS)
			EMIT_FULL_DATA_PROCESS_INS_AND_RETURN(CMP_DP, SLJIT_UNUSED, dst, SRC2_IMM);
		return SLJIT_SUCCESS;

	case SLJIT_ADD:
		SLJIT_ASSERT(!(flags & INV_IMM));
		EMIT_DATA_PROCESS_INS_AND_RETURN(ADD_DP);

	case SLJIT_ADDC:
		SLJIT_ASSERT(!(flags & INV_IMM));
		EMIT_DATA_PROCESS_INS_AND_RETURN(ADC_DP);

	case SLJIT_SUB:
		SLJIT_ASSERT(!(flags & INV_IMM));
		if (!(flags & ARGS_SWAPPED))
			EMIT_DATA_PROCESS_INS_AND_RETURN(SUB_DP);
		EMIT_DATA_PROCESS_INS_AND_RETURN(RSB_DP);

	case SLJIT_SUBC:
		SLJIT_ASSERT(!(flags & INV_IMM));
		if (!(flags & ARGS_SWAPPED))
			EMIT_DATA_PROCESS_INS_AND_RETURN(SBC_DP);
		EMIT_DATA_PROCESS_INS_AND_RETURN(RSC_DP);

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & INV_IMM));
		SLJIT_ASSERT(!(src2 & SRC2_IMM));
		if (SLJIT_UNLIKELY(op & SLJIT_SET_O))
			mul_inst = SMULL | (reg_map[TMP_REG3] << 16) | (reg_map[dst] << 12);
		else
			mul_inst = MUL | (reg_map[dst] << 16);

		if (dst != src2)
			FAIL_IF(push_inst(compiler, mul_inst | (reg_map[src1] << 8) | reg_map[src2]));
		else if (dst != src1)
			FAIL_IF(push_inst(compiler, mul_inst | (reg_map[src2] << 8) | reg_map[src1]));
		else {
			/* Rm and Rd must not be the same register. */
			SLJIT_ASSERT(dst != TMP_REG1);
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, TMP_REG1, SLJIT_UNUSED, reg_map[src2])));
			FAIL_IF(push_inst(compiler, mul_inst | (reg_map[src2] << 8) | reg_map[TMP_REG1]));
		}

		if (!(op & SLJIT_SET_O))
			return SLJIT_SUCCESS;

		/* We need to use TMP_REG3. */
		compiler->cache_arg = 0;
		compiler->cache_argw = 0;
		/* cmp TMP_REG2, dst asr #31. */
		return push_inst(compiler, EMIT_DATA_PROCESS_INS(CMP_DP, SET_FLAGS, SLJIT_UNUSED, TMP_REG3, RM(dst) | 0xfc0));

	case SLJIT_AND:
		if (!(flags & INV_IMM))
			EMIT_DATA_PROCESS_INS_AND_RETURN(AND_DP);
		EMIT_DATA_PROCESS_INS_AND_RETURN(BIC_DP);

	case SLJIT_OR:
		SLJIT_ASSERT(!(flags & INV_IMM));
		EMIT_DATA_PROCESS_INS_AND_RETURN(ORR_DP);

	case SLJIT_XOR:
		SLJIT_ASSERT(!(flags & INV_IMM));
		EMIT_DATA_PROCESS_INS_AND_RETURN(EOR_DP);

	case SLJIT_SHL:
		EMIT_SHIFT_INS_AND_RETURN(0);

	case SLJIT_LSHR:
		EMIT_SHIFT_INS_AND_RETURN(1);

	case SLJIT_ASHR:
		EMIT_SHIFT_INS_AND_RETURN(2);
	}
	SLJIT_ASSERT_STOP();
	return SLJIT_SUCCESS;
}

#undef EMIT_DATA_PROCESS_INS_AND_RETURN
#undef EMIT_FULL_DATA_PROCESS_INS_AND_RETURN
#undef EMIT_SHIFT_INS_AND_RETURN

/* Tests whether the immediate can be stored in the 12 bit imm field.
   Returns with 0 if not possible. */
static sljit_uw get_imm(sljit_uw imm)
{
	sljit_s32 rol;

	if (imm <= 0xff)
		return SRC2_IMM | imm;

	if (!(imm & 0xff000000)) {
		imm <<= 8;
		rol = 8;
	}
	else {
		imm = (imm << 24) | (imm >> 8);
		rol = 0;
	}

	if (!(imm & 0xff000000)) {
		imm <<= 8;
		rol += 4;
	}

	if (!(imm & 0xf0000000)) {
		imm <<= 4;
		rol += 2;
	}

	if (!(imm & 0xc0000000)) {
		imm <<= 2;
		rol += 1;
	}

	if (!(imm & 0x00ffffff))
		return SRC2_IMM | (imm >> 24) | (rol << 8);
	else
		return 0;
}

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
static sljit_s32 generate_int(struct sljit_compiler *compiler, sljit_s32 reg, sljit_uw imm, sljit_s32 positive)
{
	sljit_uw mask;
	sljit_uw imm1;
	sljit_uw imm2;
	sljit_s32 rol;

	/* Step1: Search a zero byte (8 continous zero bit). */
	mask = 0xff000000;
	rol = 8;
	while(1) {
		if (!(imm & mask)) {
			/* Rol imm by rol. */
			imm = (imm << rol) | (imm >> (32 - rol));
			/* Calculate arm rol. */
			rol = 4 + (rol >> 1);
			break;
		}
		rol += 2;
		mask >>= 2;
		if (mask & 0x3) {
			/* rol by 8. */
			imm = (imm << 8) | (imm >> 24);
			mask = 0xff00;
			rol = 24;
			while (1) {
				if (!(imm & mask)) {
					/* Rol imm by rol. */
					imm = (imm << rol) | (imm >> (32 - rol));
					/* Calculate arm rol. */
					rol = (rol >> 1) - 8;
					break;
				}
				rol += 2;
				mask >>= 2;
				if (mask & 0x3)
					return 0;
			}
			break;
		}
	}

	/* The low 8 bit must be zero. */
	SLJIT_ASSERT(!(imm & 0xff));

	if (!(imm & 0xff000000)) {
		imm1 = SRC2_IMM | ((imm >> 16) & 0xff) | (((rol + 4) & 0xf) << 8);
		imm2 = SRC2_IMM | ((imm >> 8) & 0xff) | (((rol + 8) & 0xf) << 8);
	}
	else if (imm & 0xc0000000) {
		imm1 = SRC2_IMM | ((imm >> 24) & 0xff) | ((rol & 0xf) << 8);
		imm <<= 8;
		rol += 4;

		if (!(imm & 0xff000000)) {
			imm <<= 8;
			rol += 4;
		}

		if (!(imm & 0xf0000000)) {
			imm <<= 4;
			rol += 2;
		}

		if (!(imm & 0xc0000000)) {
			imm <<= 2;
			rol += 1;
		}

		if (!(imm & 0x00ffffff))
			imm2 = SRC2_IMM | (imm >> 24) | ((rol & 0xf) << 8);
		else
			return 0;
	}
	else {
		if (!(imm & 0xf0000000)) {
			imm <<= 4;
			rol += 2;
		}

		if (!(imm & 0xc0000000)) {
			imm <<= 2;
			rol += 1;
		}

		imm1 = SRC2_IMM | ((imm >> 24) & 0xff) | ((rol & 0xf) << 8);
		imm <<= 8;
		rol += 4;

		if (!(imm & 0xf0000000)) {
			imm <<= 4;
			rol += 2;
		}

		if (!(imm & 0xc0000000)) {
			imm <<= 2;
			rol += 1;
		}

		if (!(imm & 0x00ffffff))
			imm2 = SRC2_IMM | (imm >> 24) | ((rol & 0xf) << 8);
		else
			return 0;
	}

	FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(positive ? MOV_DP : MVN_DP, 0, reg, SLJIT_UNUSED, imm1)));
	FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(positive ? ORR_DP : BIC_DP, 0, reg, reg, imm2)));
	return 1;
}
#endif

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 reg, sljit_uw imm)
{
	sljit_uw tmp;

#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
	if (!(imm & ~0xffff))
		return push_inst(compiler, MOVW | RD(reg) | ((imm << 4) & 0xf0000) | (imm & 0xfff));
#endif

	/* Create imm by 1 inst. */
	tmp = get_imm(imm);
	if (tmp)
		return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, reg, SLJIT_UNUSED, tmp));

	tmp = get_imm(~imm);
	if (tmp)
		return push_inst(compiler, EMIT_DATA_PROCESS_INS(MVN_DP, 0, reg, SLJIT_UNUSED, tmp));

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	/* Create imm by 2 inst. */
	FAIL_IF(generate_int(compiler, reg, imm, 1));
	FAIL_IF(generate_int(compiler, reg, ~imm, 0));

	/* Load integer. */
	return push_inst_with_literal(compiler, EMIT_DATA_TRANSFER(WORD_DATA | LOAD_DATA, 1, 0, reg, TMP_PC, 0), imm);
#else
	return emit_imm(compiler, reg, imm);
#endif
}

/* Helper function. Dst should be reg + value, using at most 1 instruction, flags does not set. */
static sljit_s32 emit_set_delta(struct sljit_compiler *compiler, sljit_s32 dst, sljit_s32 reg, sljit_sw value)
{
	if (value >= 0) {
		value = get_imm(value);
		if (value)
			return push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, dst, reg, value));
	}
	else {
		value = get_imm(-value);
		if (value)
			return push_inst(compiler, EMIT_DATA_PROCESS_INS(SUB_DP, 0, dst, reg, value));
	}
	return SLJIT_ERR_UNSUPPORTED;
}

/* Can perform an operation using at most 1 instruction. */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 inp_flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_uw imm;

	if (arg & SLJIT_IMM) {
		imm = get_imm(argw);
		if (imm) {
			if (inp_flags & ARG_TEST)
				return 1;
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, reg, SLJIT_UNUSED, imm)));
			return -1;
		}
		imm = get_imm(~argw);
		if (imm) {
			if (inp_flags & ARG_TEST)
				return 1;
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MVN_DP, 0, reg, SLJIT_UNUSED, imm)));
			return -1;
		}
		return 0;
	}

	SLJIT_ASSERT(arg & SLJIT_MEM);

	/* Fast loads/stores. */
	if (!(arg & REG_MASK))
		return 0;

	if (arg & OFFS_REG_MASK) {
		if ((argw & 0x3) != 0 && !IS_TYPE1_TRANSFER(inp_flags))
			return 0;

		if (inp_flags & ARG_TEST)
			return 1;
		FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, inp_flags & WRITE_BACK, reg, arg & REG_MASK,
			RM(OFFS_REG(arg)) | (IS_TYPE1_TRANSFER(inp_flags) ? SRC2_IMM : 0) | ((argw & 0x3) << 7))));
		return -1;
	}

	if (IS_TYPE1_TRANSFER(inp_flags)) {
		if (argw >= 0 && argw <= 0xfff) {
			if (inp_flags & ARG_TEST)
				return 1;
			FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, inp_flags & WRITE_BACK, reg, arg & REG_MASK, argw)));
			return -1;
		}
		if (argw < 0 && argw >= -0xfff) {
			if (inp_flags & ARG_TEST)
				return 1;
			FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 0, inp_flags & WRITE_BACK, reg, arg & REG_MASK, -argw)));
			return -1;
		}
	}
	else {
		if (argw >= 0 && argw <= 0xff) {
			if (inp_flags & ARG_TEST)
				return 1;
			FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, inp_flags & WRITE_BACK, reg, arg & REG_MASK, TYPE2_TRANSFER_IMM(argw))));
			return -1;
		}
		if (argw < 0 && argw >= -0xff) {
			if (inp_flags & ARG_TEST)
				return 1;
			argw = -argw;
			FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 0, inp_flags & WRITE_BACK, reg, arg & REG_MASK, TYPE2_TRANSFER_IMM(argw))));
			return -1;
		}
	}

	return 0;
}

/* See getput_arg below.
   Note: can_cache is called only for binary operators. Those
   operators always uses word arguments without write back. */
static sljit_s32 can_cache(sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	/* Immediate caching is not supported as it would be an operation on constant arguments. */
	if (arg & SLJIT_IMM)
		return 0;

	/* Always a simple operation. */
	if (arg & OFFS_REG_MASK)
		return 0;

	if (!(arg & REG_MASK)) {
		/* Immediate access. */
		if ((next_arg & SLJIT_MEM) && ((sljit_uw)argw - (sljit_uw)next_argw <= 0xfff || (sljit_uw)next_argw - (sljit_uw)argw <= 0xfff))
			return 1;
		return 0;
	}

	if (argw <= 0xfffff && argw >= -0xfffff)
		return 0;

	if (argw == next_argw && (next_arg & SLJIT_MEM))
		return 1;

	if (arg == next_arg && ((sljit_uw)argw - (sljit_uw)next_argw <= 0xfff || (sljit_uw)next_argw - (sljit_uw)argw <= 0xfff))
		return 1;

	return 0;
}

#define GETPUT_ARG_DATA_TRANSFER(add, wb, target, base, imm) \
	if (max_delta & 0xf00) \
		FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, add, wb, target, base, imm))); \
	else \
		FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, add, wb, target, base, TYPE2_TRANSFER_IMM(imm))));

#define TEST_WRITE_BACK() \
	if (inp_flags & WRITE_BACK) { \
		tmp_r = arg & REG_MASK; \
		if (reg == tmp_r) { \
			/* This can only happen for stores */ \
			/* since ldr reg, [reg, ...]! has no meaning */ \
			SLJIT_ASSERT(!(inp_flags & LOAD_DATA)); \
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, TMP_REG3, SLJIT_UNUSED, RM(reg)))); \
			reg = TMP_REG3; \
		} \
	}

/* Emit the necessary instructions. See can_cache above. */
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 inp_flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 tmp_r;
	sljit_sw max_delta;
	sljit_sw sign;
	sljit_uw imm;

	if (arg & SLJIT_IMM) {
		SLJIT_ASSERT(inp_flags & LOAD_DATA);
		return load_immediate(compiler, reg, argw);
	}

	SLJIT_ASSERT(arg & SLJIT_MEM);

	tmp_r = (inp_flags & LOAD_DATA) ? reg : TMP_REG3;
	max_delta = IS_TYPE1_TRANSFER(inp_flags) ? 0xfff : 0xff;

	if ((arg & REG_MASK) == SLJIT_UNUSED) {
		/* Write back is not used. */
		imm = (sljit_uw)(argw - compiler->cache_argw);
		if ((compiler->cache_arg & SLJIT_IMM) && (imm <= (sljit_uw)max_delta || imm >= (sljit_uw)-max_delta)) {
			if (imm <= (sljit_uw)max_delta) {
				sign = 1;
				argw = argw - compiler->cache_argw;
			}
			else {
				sign = 0;
				argw = compiler->cache_argw - argw;
			}

			GETPUT_ARG_DATA_TRANSFER(sign, 0, reg, TMP_REG3, argw);
			return SLJIT_SUCCESS;
		}

		/* With write back, we can create some sophisticated loads, but
		   it is hard to decide whether we should convert downward (0s) or upward (1s). */
		imm = (sljit_uw)(argw - next_argw);
		if ((next_arg & SLJIT_MEM) && (imm <= (sljit_uw)max_delta || imm >= (sljit_uw)-max_delta)) {
			SLJIT_ASSERT(inp_flags & LOAD_DATA);

			compiler->cache_arg = SLJIT_IMM;
			compiler->cache_argw = argw;
			tmp_r = TMP_REG3;
		}

		FAIL_IF(load_immediate(compiler, tmp_r, argw));
		GETPUT_ARG_DATA_TRANSFER(1, 0, reg, tmp_r, 0);
		return SLJIT_SUCCESS;
	}

	if (arg & OFFS_REG_MASK) {
		SLJIT_ASSERT((argw & 0x3) && !(max_delta & 0xf00));
		if (inp_flags & WRITE_BACK)
			tmp_r = arg & REG_MASK;
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, tmp_r, arg & REG_MASK, RM(OFFS_REG(arg)) | ((argw & 0x3) << 7))));
		return push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, 0, reg, tmp_r, TYPE2_TRANSFER_IMM(0)));
	}

	imm = (sljit_uw)(argw - compiler->cache_argw);
	if (compiler->cache_arg == arg && imm <= (sljit_uw)max_delta) {
		SLJIT_ASSERT(!(inp_flags & WRITE_BACK));
		GETPUT_ARG_DATA_TRANSFER(1, 0, reg, TMP_REG3, imm);
		return SLJIT_SUCCESS;
	}
	if (compiler->cache_arg == arg && imm >= (sljit_uw)-max_delta) {
		SLJIT_ASSERT(!(inp_flags & WRITE_BACK));
		imm = (sljit_uw)-(sljit_sw)imm;
		GETPUT_ARG_DATA_TRANSFER(0, 0, reg, TMP_REG3, imm);
		return SLJIT_SUCCESS;
	}

	imm = get_imm(argw & ~max_delta);
	if (imm) {
		TEST_WRITE_BACK();
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, tmp_r, arg & REG_MASK, imm)));
		GETPUT_ARG_DATA_TRANSFER(1, inp_flags & WRITE_BACK, reg, tmp_r, argw & max_delta);
		return SLJIT_SUCCESS;
	}

	imm = get_imm(-argw & ~max_delta);
	if (imm) {
		argw = -argw;
		TEST_WRITE_BACK();
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(SUB_DP, 0, tmp_r, arg & REG_MASK, imm)));
		GETPUT_ARG_DATA_TRANSFER(0, inp_flags & WRITE_BACK, reg, tmp_r, argw & max_delta);
		return SLJIT_SUCCESS;
	}

	if ((compiler->cache_arg & SLJIT_IMM) && compiler->cache_argw == argw) {
		TEST_WRITE_BACK();
		return push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, inp_flags & WRITE_BACK, reg, arg & REG_MASK, RM(TMP_REG3) | (max_delta & 0xf00 ? SRC2_IMM : 0)));
	}

	if (argw == next_argw && (next_arg & SLJIT_MEM)) {
		SLJIT_ASSERT(inp_flags & LOAD_DATA);
		FAIL_IF(load_immediate(compiler, TMP_REG3, argw));

		compiler->cache_arg = SLJIT_IMM;
		compiler->cache_argw = argw;

		TEST_WRITE_BACK();
		return push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, inp_flags & WRITE_BACK, reg, arg & REG_MASK, RM(TMP_REG3) | (max_delta & 0xf00 ? SRC2_IMM : 0)));
	}

	imm = (sljit_uw)(argw - next_argw);
	if (arg == next_arg && !(inp_flags & WRITE_BACK) && (imm <= (sljit_uw)max_delta || imm >= (sljit_uw)-max_delta)) {
		SLJIT_ASSERT(inp_flags & LOAD_DATA);
		FAIL_IF(load_immediate(compiler, TMP_REG3, argw));
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, TMP_REG3, TMP_REG3, reg_map[arg & REG_MASK])));

		compiler->cache_arg = arg;
		compiler->cache_argw = argw;

		GETPUT_ARG_DATA_TRANSFER(1, 0, reg, TMP_REG3, 0);
		return SLJIT_SUCCESS;
	}

	if ((arg & REG_MASK) == tmp_r) {
		compiler->cache_arg = SLJIT_IMM;
		compiler->cache_argw = argw;
		tmp_r = TMP_REG3;
	}

	FAIL_IF(load_immediate(compiler, tmp_r, argw));
	return push_inst(compiler, EMIT_DATA_TRANSFER(inp_flags, 1, inp_flags & WRITE_BACK, reg, arg & REG_MASK, reg_map[tmp_r] | (max_delta & 0xf00 ? SRC2_IMM : 0)));
}

static SLJIT_INLINE sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	if (getput_arg_fast(compiler, flags, reg, arg, argw))
		return compiler->error;
	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	return getput_arg(compiler, flags, reg, arg, argw, 0, 0);
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 inp_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* arg1 goes to TMP_REG1 or src reg
	   arg2 goes to TMP_REG2, imm or src reg
	   TMP_REG3 can be used for caching
	   result goes to TMP_REG2, so put result can use TMP_REG1 and TMP_REG3. */

	/* We prefers register and simple consts. */
	sljit_s32 dst_r;
	sljit_s32 src1_r;
	sljit_s32 src2_r = 0;
	sljit_s32 sugg_src2_r = TMP_REG2;
	sljit_s32 flags = GET_FLAGS(op) ? SET_FLAGS : 0;

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

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
		if (getput_arg_fast(compiler, inp_flags | ARG_TEST, TMP_REG2, dst, dstw)) {
			flags |= FAST_DEST;
			dst_r = TMP_REG2;
		}
		else {
			flags |= SLOW_DEST;
			dst_r = 0;
		}
	}

	/* Source 1. */
	if (FAST_IS_REG(src1))
		src1_r = src1;
	else if (FAST_IS_REG(src2)) {
		flags |= ARGS_SWAPPED;
		src1_r = src2;
		src2 = src1;
		src2w = src1w;
	}
	else do { /* do { } while(0) is used because of breaks. */
		src1_r = 0;
		if ((inp_flags & ALLOW_ANY_IMM) && (src1 & SLJIT_IMM)) {
			/* The second check will generate a hit. */
			src2_r = get_imm(src1w);
			if (src2_r) {
				flags |= ARGS_SWAPPED;
				src1 = src2;
				src1w = src2w;
				break;
			}
			if (inp_flags & ALLOW_INV_IMM) {
				src2_r = get_imm(~src1w);
				if (src2_r) {
					flags |= ARGS_SWAPPED | INV_IMM;
					src1 = src2;
					src1w = src2w;
					break;
				}
			}
			if (GET_OPCODE(op) == SLJIT_ADD) {
				src2_r = get_imm(-src1w);
				if (src2_r) {
					/* Note: ARGS_SWAPPED is intentionally not applied! */
					src1 = src2;
					src1w = src2w;
					op = SLJIT_SUB | GET_ALL_FLAGS(op);
					break;
				}
			}
		}

		if (getput_arg_fast(compiler, inp_flags | LOAD_DATA, TMP_REG1, src1, src1w)) {
			FAIL_IF(compiler->error);
			src1_r = TMP_REG1;
		}
	} while (0);

	/* Source 2. */
	if (src2_r == 0) {
		if (FAST_IS_REG(src2)) {
			src2_r = src2;
			flags |= REG_SOURCE;
			if (!(flags & REG_DEST) && op >= SLJIT_MOV && op <= SLJIT_MOVU_S32)
				dst_r = src2_r;
		}
		else do { /* do { } while(0) is used because of breaks. */
			if ((inp_flags & ALLOW_ANY_IMM) && (src2 & SLJIT_IMM)) {
				src2_r = get_imm(src2w);
				if (src2_r)
					break;
				if (inp_flags & ALLOW_INV_IMM) {
					src2_r = get_imm(~src2w);
					if (src2_r) {
						flags |= INV_IMM;
						break;
					}
				}
				if (GET_OPCODE(op) == SLJIT_ADD) {
					src2_r = get_imm(-src2w);
					if (src2_r) {
						op = SLJIT_SUB | GET_ALL_FLAGS(op);
						flags &= ~ARGS_SWAPPED;
						break;
					}
				}
				if (GET_OPCODE(op) == SLJIT_SUB && !(flags & ARGS_SWAPPED)) {
					src2_r = get_imm(-src2w);
					if (src2_r) {
						op = SLJIT_ADD | GET_ALL_FLAGS(op);
						flags &= ~ARGS_SWAPPED;
						break;
					}
				}
			}

			/* src2_r is 0. */
			if (getput_arg_fast(compiler, inp_flags | LOAD_DATA, sugg_src2_r, src2, src2w)) {
				FAIL_IF(compiler->error);
				src2_r = sugg_src2_r;
			}
		} while (0);
	}

	/* src1_r, src2_r and dst_r can be zero (=unprocessed) or non-zero.
	   If they are zero, they must not be registers. */
	if (src1_r == 0 && src2_r == 0 && dst_r == 0) {
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			SLJIT_ASSERT(!(flags & ARGS_SWAPPED));
			flags |= ARGS_SWAPPED;
			FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG1, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG2, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG2, src2, src2w, dst, dstw));
		}
		src1_r = TMP_REG1;
		src2_r = TMP_REG2;
	}
	else if (src1_r == 0 && src2_r == 0) {
		FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG1, src1, src1w, src2, src2w));
		src1_r = TMP_REG1;
	}
	else if (src1_r == 0 && dst_r == 0) {
		FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
		src1_r = TMP_REG1;
	}
	else if (src2_r == 0 && dst_r == 0) {
		FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, sugg_src2_r, src2, src2w, dst, dstw));
		src2_r = sugg_src2_r;
	}

	if (dst_r == 0)
		dst_r = TMP_REG2;

	if (src1_r == 0) {
		FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, TMP_REG1, src1, src1w, 0, 0));
		src1_r = TMP_REG1;
	}

	if (src2_r == 0) {
		FAIL_IF(getput_arg(compiler, inp_flags | LOAD_DATA, sugg_src2_r, src2, src2w, 0, 0));
		src2_r = sugg_src2_r;
	}

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (flags & (FAST_DEST | SLOW_DEST)) {
		if (flags & FAST_DEST)
			FAIL_IF(getput_arg_fast(compiler, inp_flags, dst_r, dst, dstw));
		else
			FAIL_IF(getput_arg(compiler, inp_flags, dst_r, dst, dstw, 0, 0));
	}
	return SLJIT_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
extern unsigned int __aeabi_uidivmod(unsigned int numerator, unsigned int denominator);
extern int __aeabi_idivmod(int numerator, int denominator);
#else
#error "Software divmod functions are needed"
#endif

#ifdef __cplusplus
}
#endif

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	op = GET_OPCODE(op);
	switch (op) {
	case SLJIT_BREAKPOINT:
		FAIL_IF(push_inst(compiler, BKPT));
		break;
	case SLJIT_NOP:
		FAIL_IF(push_inst(compiler, NOP));
		break;
	case SLJIT_LMUL_UW:
	case SLJIT_LMUL_SW:
#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
		return push_inst(compiler, (op == SLJIT_LMUL_UW ? UMULL : SMULL)
			| (reg_map[SLJIT_R1] << 16)
			| (reg_map[SLJIT_R0] << 12)
			| (reg_map[SLJIT_R0] << 8)
			| reg_map[SLJIT_R1]);
#else
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, TMP_REG1, SLJIT_UNUSED, RM(SLJIT_R1))));
		return push_inst(compiler, (op == SLJIT_LMUL_UW ? UMULL : SMULL)
			| (reg_map[SLJIT_R1] << 16)
			| (reg_map[SLJIT_R0] << 12)
			| (reg_map[SLJIT_R0] << 8)
			| reg_map[TMP_REG1]);
#endif
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		SLJIT_COMPILE_ASSERT((SLJIT_DIVMOD_UW & 0x2) == 0 && SLJIT_DIV_UW - 0x2 == SLJIT_DIVMOD_UW, bad_div_opcode_assignments);
		SLJIT_COMPILE_ASSERT(reg_map[2] == 1 && reg_map[3] == 2, bad_register_mapping);

		if ((op >= SLJIT_DIV_UW) && (compiler->scratches >= 3)) {
			FAIL_IF(push_inst(compiler, 0xe52d2008 /* str r2, [sp, #-8]! */));
			FAIL_IF(push_inst(compiler, 0xe58d1004 /* str r1, [sp, #4] */));
		}
		else if ((op >= SLJIT_DIV_UW) || (compiler->scratches >= 3))
			FAIL_IF(push_inst(compiler, 0xe52d0008 | (op >= SLJIT_DIV_UW ? 0x1000 : 0x2000) /* str r1/r2, [sp, #-8]! */));

#if defined(__GNUC__)
		FAIL_IF(sljit_emit_ijump(compiler, SLJIT_FAST_CALL, SLJIT_IMM,
			((op | 0x2) == SLJIT_DIV_UW ? SLJIT_FUNC_OFFSET(__aeabi_uidivmod) : SLJIT_FUNC_OFFSET(__aeabi_idivmod))));
#else
#error "Software divmod functions are needed"
#endif

		if ((op >= SLJIT_DIV_UW) && (compiler->scratches >= 3)) {
			FAIL_IF(push_inst(compiler, 0xe59d1004 /* ldr r1, [sp, #4] */));
			FAIL_IF(push_inst(compiler, 0xe49d2008 /* ldr r2, [sp], #8 */));
		}
		else if ((op >= SLJIT_DIV_UW) || (compiler->scratches >= 3))
			return push_inst(compiler, 0xe49d0008 | (op >= SLJIT_DIV_UW ? 0x1000 : 0x2000) /* ldr r1/r2, [sp], #8 */);
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, ALLOW_ANY_IMM, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_U8:
		return emit_op(compiler, SLJIT_MOV_U8, ALLOW_ANY_IMM | BYTE_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, SLJIT_MOV_S8, ALLOW_ANY_IMM | SIGNED_DATA | BYTE_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, SLJIT_MOV_U16, ALLOW_ANY_IMM | HALF_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, SLJIT_MOV_S16, ALLOW_ANY_IMM | SIGNED_DATA | HALF_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_MOVU:
	case SLJIT_MOVU_U32:
	case SLJIT_MOVU_S32:
	case SLJIT_MOVU_P:
		return emit_op(compiler, SLJIT_MOV, ALLOW_ANY_IMM | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOVU_U8:
		return emit_op(compiler, SLJIT_MOV_U8, ALLOW_ANY_IMM | BYTE_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOVU_S8:
		return emit_op(compiler, SLJIT_MOV_S8, ALLOW_ANY_IMM | SIGNED_DATA | BYTE_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOVU_U16:
		return emit_op(compiler, SLJIT_MOV_U16, ALLOW_ANY_IMM | HALF_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOVU_S16:
		return emit_op(compiler, SLJIT_MOV_S16, ALLOW_ANY_IMM | SIGNED_DATA | HALF_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_NOT:
		return emit_op(compiler, op, ALLOW_ANY_IMM, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_NEG:
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE) \
			|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
		compiler->skip_checks = 1;
#endif
		return sljit_emit_op2(compiler, SLJIT_SUB | GET_ALL_FLAGS(op), dst, dstw, SLJIT_IMM, 0, src, srcw);

	case SLJIT_CLZ:
		return emit_op(compiler, op, 0, dst, dstw, TMP_REG1, 0, src, srcw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
	case SLJIT_SUB:
	case SLJIT_SUBC:
	case SLJIT_OR:
	case SLJIT_XOR:
		return emit_op(compiler, op, ALLOW_IMM, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		return emit_op(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
		return emit_op(compiler, op, ALLOW_ANY_IMM, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_LSHR:
	case SLJIT_ASHR:
		if (src2 & SLJIT_IMM) {
			compiler->shift_imm = src2w & 0x1f;
			return emit_op(compiler, op, 0, dst, dstw, TMP_REG1, 0, src1, src1w);
		}
		else {
			compiler->shift_imm = 0x20;
			return emit_op(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w);
		}
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
	return reg << 1;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_s32 size)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_uw*)instruction);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)

/* 0 - no fpu
   1 - vfp */
static sljit_s32 arm_fpu_type = -1;

static void init_compiler(void)
{
	if (arm_fpu_type != -1)
		return;

	/* TODO: Only the OS can help to determine the correct fpu type. */
	arm_fpu_type = 1;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_is_fpu_available(void)
{
#ifdef SLJIT_IS_FPU_AVAILABLE
	return SLJIT_IS_FPU_AVAILABLE;
#else
	if (arm_fpu_type == -1)
		init_compiler();
	return arm_fpu_type;
#endif
}

#else

#define arm_fpu_type 1

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_is_fpu_available(void)
{
	/* Always available. */
	return 1;
}

#endif

#define FPU_LOAD (1 << 20)
#define EMIT_FPU_DATA_TRANSFER(inst, add, base, freg, offs) \
	((inst) | ((add) << 23) | (reg_map[base] << 16) | (freg << 12) | (offs))
#define EMIT_FPU_OPERATION(opcode, mode, dst, src1, src2) \
	((opcode) | (mode) | ((dst) << 12) | (src1) | ((src2) << 16))

static sljit_s32 emit_fop_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_sw tmp;
	sljit_uw imm;
	sljit_sw inst = VSTR_F32 | (flags & (SLJIT_F32_OP | FPU_LOAD));
	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, TMP_REG1, arg & REG_MASK, RM(OFFS_REG(arg)) | ((argw & 0x3) << 7))));
		arg = SLJIT_MEM | TMP_REG1;
		argw = 0;
	}

	/* Fast loads and stores. */
	if ((arg & REG_MASK)) {
		if (!(argw & ~0x3fc))
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, arg & REG_MASK, reg, argw >> 2));
		if (!(-argw & ~0x3fc))
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 0, arg & REG_MASK, reg, (-argw) >> 2));
	}

	if (compiler->cache_arg == arg) {
		tmp = argw - compiler->cache_argw;
		if (!(tmp & ~0x3fc))
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG3, reg, tmp >> 2));
		if (!(-tmp & ~0x3fc))
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 0, TMP_REG3, reg, -tmp >> 2));
		if (emit_set_delta(compiler, TMP_REG3, TMP_REG3, tmp) != SLJIT_ERR_UNSUPPORTED) {
			FAIL_IF(compiler->error);
			compiler->cache_argw = argw;
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG3, reg, 0));
		}
	}

	if (arg & REG_MASK) {
		if (emit_set_delta(compiler, TMP_REG1, arg & REG_MASK, argw) != SLJIT_ERR_UNSUPPORTED) {
			FAIL_IF(compiler->error);
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG1, reg, 0));
		}
		imm = get_imm(argw & ~0x3fc);
		if (imm) {
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, TMP_REG1, arg & REG_MASK, imm)));
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG1, reg, (argw & 0x3fc) >> 2));
		}
		imm = get_imm(-argw & ~0x3fc);
		if (imm) {
			argw = -argw;
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(SUB_DP, 0, TMP_REG1, arg & REG_MASK, imm)));
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 0, TMP_REG1, reg, (argw & 0x3fc) >> 2));
		}
	}

	compiler->cache_arg = arg;
	compiler->cache_argw = argw;
	if (arg & REG_MASK) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, argw));
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(ADD_DP, 0, TMP_REG3, arg & REG_MASK, reg_map[TMP_REG1])));
	}
	else
		FAIL_IF(load_immediate(compiler, TMP_REG3, argw));

	return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG3, reg, 0));
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (src & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP) | FPU_LOAD, TMP_FREG1, src, srcw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCVT_S32_F32, op & SLJIT_F32_OP, TMP_FREG1, src, 0)));

	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;

	if (FAST_IS_REG(dst))
		return push_inst(compiler, VMOV | (1 << 20) | RD(dst) | (TMP_FREG1 << 16));

	/* Store the integer value from a VFP register. */
	return emit_fop_mem(compiler, 0, TMP_FREG1, dst, dstw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, VMOV | RD(src) | (TMP_FREG1 << 16)));
	else if (src & SLJIT_MEM) {
		/* Load the integer value into a VFP register. */
		FAIL_IF(emit_fop_mem(compiler, FPU_LOAD, TMP_FREG1, src, srcw));
	}
	else {
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		FAIL_IF(push_inst(compiler, VMOV | RD(TMP_REG1) | (TMP_FREG1 << 16)));
	}

	FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCVT_F32_S32, op & SLJIT_F32_OP, dst_r, TMP_FREG1, 0)));

	if (dst & SLJIT_MEM)
		return emit_fop_mem(compiler, (op & SLJIT_F32_OP), TMP_FREG1, dst, dstw);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP) | FPU_LOAD, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP) | FPU_LOAD, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCMP_F32, op & SLJIT_F32_OP, src1, src2, 0)));
	return push_inst(compiler, VMRS);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	if (GET_OPCODE(op) != SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_F32_OP;

	SLJIT_COMPILE_ASSERT((SLJIT_F32_OP == 0x100), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP) | FPU_LOAD, dst_r, src, srcw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMOV_F32, op & SLJIT_F32_OP, dst_r, src, 0)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VNEG_F32, op & SLJIT_F32_OP, dst_r, src, 0)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VABS_F32, op & SLJIT_F32_OP, dst_r, src, 0)));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCVT_F64_F32, op & SLJIT_F32_OP, dst_r, src, 0)));
		op ^= SLJIT_F32_OP;
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_fop_mem(compiler, (op & SLJIT_F32_OP), dst_r, dst, dstw);
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

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	op ^= SLJIT_F32_OP;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP) | FPU_LOAD, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP) | FPU_LOAD, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VADD_F32, op & SLJIT_F32_OP, dst_r, src2, src1)));
		break;

	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VSUB_F32, op & SLJIT_F32_OP, dst_r, src2, src1)));
		break;

	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMUL_F32, op & SLJIT_F32_OP, dst_r, src2, src1)));
		break;

	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VDIV_F32, op & SLJIT_F32_OP, dst_r, src2, src1)));
		break;
	}

	if (dst_r == TMP_FREG1)
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_F32_OP), TMP_FREG1, dst, dstw));

	return SLJIT_SUCCESS;
}

#undef FPU_LOAD
#undef EMIT_FPU_DATA_TRANSFER
#undef EMIT_FPU_OPERATION

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
		return push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst, SLJIT_UNUSED, RM(TMP_REG3)));

	/* Memory. */
	if (getput_arg_fast(compiler, WORD_DATA, TMP_REG3, dst, dstw))
		return compiler->error;
	/* TMP_REG3 is used for caching. */
	FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, TMP_REG2, SLJIT_UNUSED, RM(TMP_REG3))));
	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	return getput_arg(compiler, WORD_DATA, TMP_REG2, dst, dstw, 0, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_return(compiler, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, TMP_REG3, SLJIT_UNUSED, RM(src))));
	else if (src & SLJIT_MEM) {
		if (getput_arg_fast(compiler, WORD_DATA | LOAD_DATA, TMP_REG3, src, srcw))
			FAIL_IF(compiler->error);
		else {
			compiler->cache_arg = 0;
			compiler->cache_argw = 0;
			FAIL_IF(getput_arg(compiler, WORD_DATA | LOAD_DATA, TMP_REG2, src, srcw, 0, 0));
			FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, TMP_REG3, SLJIT_UNUSED, RM(TMP_REG2))));
		}
	}
	else if (src & SLJIT_IMM)
		FAIL_IF(load_immediate(compiler, TMP_REG3, srcw));
	return push_inst(compiler, BLX | RM(TMP_REG3));
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

static sljit_uw get_cc(sljit_s32 type)
{
	switch (type) {
	case SLJIT_EQUAL:
	case SLJIT_MUL_NOT_OVERFLOW:
	case SLJIT_EQUAL_F64:
		return 0x00000000;

	case SLJIT_NOT_EQUAL:
	case SLJIT_MUL_OVERFLOW:
	case SLJIT_NOT_EQUAL_F64:
		return 0x10000000;

	case SLJIT_LESS:
	case SLJIT_LESS_F64:
		return 0x30000000;

	case SLJIT_GREATER_EQUAL:
	case SLJIT_GREATER_EQUAL_F64:
		return 0x20000000;

	case SLJIT_GREATER:
	case SLJIT_GREATER_F64:
		return 0x80000000;

	case SLJIT_LESS_EQUAL:
	case SLJIT_LESS_EQUAL_F64:
		return 0x90000000;

	case SLJIT_SIG_LESS:
		return 0xb0000000;

	case SLJIT_SIG_GREATER_EQUAL:
		return 0xa0000000;

	case SLJIT_SIG_GREATER:
		return 0xc0000000;

	case SLJIT_SIG_LESS_EQUAL:
		return 0xd0000000;

	case SLJIT_OVERFLOW:
	case SLJIT_UNORDERED_F64:
		return 0x60000000;

	case SLJIT_NOT_OVERFLOW:
	case SLJIT_ORDERED_F64:
		return 0x70000000;

	default:
		SLJIT_ASSERT(type >= SLJIT_JUMP && type <= SLJIT_CALL3);
		return 0xe0000000;
	}
}

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

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	/* In ARM, we don't need to touch the arguments. */
#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	if (type >= SLJIT_FAST_CALL)
		PTR_FAIL_IF(prepare_blx(compiler));
	PTR_FAIL_IF(push_inst_with_unique_literal(compiler, ((EMIT_DATA_TRANSFER(WORD_DATA | LOAD_DATA, 1, 0,
		type <= SLJIT_JUMP ? TMP_PC : TMP_REG1, TMP_PC, 0)) & ~COND_MASK) | get_cc(type), 0));

	if (jump->flags & SLJIT_REWRITABLE_JUMP) {
		jump->addr = compiler->size;
		compiler->patches++;
	}

	if (type >= SLJIT_FAST_CALL) {
		jump->flags |= IS_BL;
		PTR_FAIL_IF(emit_blx(compiler));
	}

	if (!(jump->flags & SLJIT_REWRITABLE_JUMP))
		jump->addr = compiler->size;
#else
	if (type >= SLJIT_FAST_CALL)
		jump->flags |= IS_BL;
	PTR_FAIL_IF(emit_imm(compiler, TMP_REG1, 0));
	PTR_FAIL_IF(push_inst(compiler, (((type <= SLJIT_JUMP ? BX : BLX) | RM(TMP_REG1)) & ~COND_MASK) | get_cc(type)));
	jump->addr = compiler->size;
#endif
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	/* In ARM, we don't need to touch the arguments. */
	if (!(src & SLJIT_IMM)) {
		if (FAST_IS_REG(src))
			return push_inst(compiler, (type <= SLJIT_JUMP ? BX : BLX) | RM(src));

		SLJIT_ASSERT(src & SLJIT_MEM);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG2, src, srcw));
		return push_inst(compiler, (type <= SLJIT_JUMP ? BX : BLX) | RM(TMP_REG2));
	}

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	FAIL_IF(!jump);
	set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_BL : 0));
	jump->u.target = srcw;

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	if (type >= SLJIT_FAST_CALL)
		FAIL_IF(prepare_blx(compiler));
	FAIL_IF(push_inst_with_unique_literal(compiler, EMIT_DATA_TRANSFER(WORD_DATA | LOAD_DATA, 1, 0, type <= SLJIT_JUMP ? TMP_PC : TMP_REG1, TMP_PC, 0), 0));
	if (type >= SLJIT_FAST_CALL)
		FAIL_IF(emit_blx(compiler));
#else
	FAIL_IF(emit_imm(compiler, TMP_REG1, 0));
	FAIL_IF(push_inst(compiler, (type <= SLJIT_JUMP ? BX : BLX) | RM(TMP_REG1)));
#endif
	jump->addr = compiler->size;
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw,
	sljit_s32 type)
{
	sljit_s32 dst_r, flags = GET_ALL_FLAGS(op);
	sljit_uw cc, ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, src, srcw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;

	op = GET_OPCODE(op);
	cc = get_cc(type & 0xff);
	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

	if (op < SLJIT_ADD) {
		FAIL_IF(push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst_r, SLJIT_UNUSED, SRC2_IMM | 0)));
		FAIL_IF(push_inst(compiler, (EMIT_DATA_PROCESS_INS(MOV_DP, 0, dst_r, SLJIT_UNUSED, SRC2_IMM | 1) & ~COND_MASK) | cc));
		return (dst_r == TMP_REG2) ? emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw) : SLJIT_SUCCESS;
	}

	ins = (op == SLJIT_AND ? AND_DP : (op == SLJIT_OR ? ORR_DP : EOR_DP));
	if ((op == SLJIT_OR || op == SLJIT_XOR) && FAST_IS_REG(dst) && dst == src) {
		FAIL_IF(push_inst(compiler, (EMIT_DATA_PROCESS_INS(ins, 0, dst, dst, SRC2_IMM | 1) & ~COND_MASK) | cc));
		/* The condition must always be set, even if the ORR/EOR is not executed above. */
		return (flags & SLJIT_SET_E) ? push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, SET_FLAGS, TMP_REG1, SLJIT_UNUSED, RM(dst))) : SLJIT_SUCCESS;
	}

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw, dst, dstw));
		src = TMP_REG1;
		srcw = 0;
	} else if (src & SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
		srcw = 0;
	}

	FAIL_IF(push_inst(compiler, (EMIT_DATA_PROCESS_INS(ins, 0, dst_r, src, SRC2_IMM | 1) & ~COND_MASK) | cc));
	FAIL_IF(push_inst(compiler, (EMIT_DATA_PROCESS_INS(ins, 0, dst_r, src, SRC2_IMM | 0) & ~COND_MASK) | (cc ^ 0x10000000)));
	if (dst_r == TMP_REG2)
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA, TMP_REG2, dst, dstw, 0, 0));

	return (flags & SLJIT_SET_E) ? push_inst(compiler, EMIT_DATA_PROCESS_INS(MOV_DP, SET_FLAGS, TMP_REG1, SLJIT_UNUSED, RM(dst_r))) : SLJIT_SUCCESS;
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

	reg = SLOW_IS_REG(dst) ? dst : TMP_REG2;

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	PTR_FAIL_IF(push_inst_with_unique_literal(compiler, EMIT_DATA_TRANSFER(WORD_DATA | LOAD_DATA, 1, 0, reg, TMP_PC, 0), init_value));
	compiler->patches++;
#else
	PTR_FAIL_IF(emit_imm(compiler, reg, init_value));
#endif
	set_const(const_, compiler);

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw));
	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	inline_set_jump_addr(addr, executable_offset, new_target, 1);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	inline_set_const(addr, executable_offset, new_constant, 1);
}
