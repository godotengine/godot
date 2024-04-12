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

#ifdef __SOFTFP__
#define ARM_ABI_INFO " ABI:softfp"
#else
#define ARM_ABI_INFO " ABI:hardfp"
#endif

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
	return "ARMv7" SLJIT_CPUINFO ARM_ABI_INFO;
#elif (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	return "ARMv6" SLJIT_CPUINFO ARM_ABI_INFO;
#else
#error "Internal error: Unknown ARM architecture"
#endif
}

/* Length of an instruction word. */
typedef sljit_u32 sljit_ins;

/* Last register + 1. */
#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_PC		(SLJIT_NUMBER_OF_REGISTERS + 4)

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)

/* In ARM instruction words.
   Cache lines are usually 32 byte aligned. */
#define CONST_POOL_ALIGNMENT	8
#define CONST_POOL_EMPTY	0xffffffff

#define ALIGN_INSTRUCTION(ptr) \
	(sljit_ins*)(((sljit_ins)(ptr) + (CONST_POOL_ALIGNMENT * sizeof(sljit_ins)) - 1) & ~((CONST_POOL_ALIGNMENT * sizeof(sljit_ins)) - 1))
#define MAX_DIFFERENCE(max_diff) \
	(((max_diff) / (sljit_s32)sizeof(sljit_ins)) - (CONST_POOL_ALIGNMENT - 1))

/* See sljit_emit_enter and sljit_emit_op0 if you want to change them. */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 5] = {
	0, 0, 1, 2, 3, 11, 10, 9, 8, 7, 6, 5, 4, 13, 12, 14, 15
};

static const sljit_u8 freg_map[((SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2) << 1) + 1] = {
	0,
	0, 1, 2, 3, 4, 5, 15, 14, 13, 12, 11, 10, 9, 8,
	7, 6,
	0, 1, 2, 3, 4, 5, 15, 14, 13, 12, 11, 10, 9, 8,
	7, 6
};

static const sljit_u8 freg_ebit_map[((SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2) << 1) + 1] = {
	0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1
};

#define RM(rm) ((sljit_ins)reg_map[rm])
#define RM8(rm) ((sljit_ins)reg_map[rm] << 8)
#define RD(rd) ((sljit_ins)reg_map[rd] << 12)
#define RN(rn) ((sljit_ins)reg_map[rn] << 16)

#define VM(vm) (((sljit_ins)freg_map[vm]) | ((sljit_ins)freg_ebit_map[vm] << 5))
#define VD(vd) (((sljit_ins)freg_map[vd] << 12) | ((sljit_ins)freg_ebit_map[vd] << 22))
#define VN(vn) (((sljit_ins)freg_map[vn] << 16) | ((sljit_ins)freg_ebit_map[vn] << 7))

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

/* The instruction includes the AL condition.
   INST_NAME - CONDITIONAL remove this flag. */
#define COND_MASK	0xf0000000
#define CONDITIONAL	0xe0000000
#define PUSH_POOL	0xff000000

#define ADC		0xe0a00000
#define ADD		0xe0800000
#define AND		0xe0000000
#define B		0xea000000
#define BIC		0xe1c00000
#define BKPT		0xe1200070
#define BL		0xeb000000
#define BLX		0xe12fff30
#define BX		0xe12fff10
#define CLZ		0xe16f0f10
#define CMN		0xe1600000
#define CMP		0xe1400000
#define EOR		0xe0200000
#define LDR		0xe5100000
#define LDR_POST	0xe4100000
#define LDREX		0xe1900f9f
#define LDREXB		0xe1d00f9f
#define LDREXH		0xe1f00f9f
#define MOV		0xe1a00000
#define MUL		0xe0000090
#define MVN		0xe1e00000
#define NOP		0xe1a00000
#define ORR		0xe1800000
#define PUSH		0xe92d0000
#define POP		0xe8bd0000
#define REV		0xe6bf0f30
#define REV16		0xe6bf0fb0
#define RSB		0xe0600000
#define RSC		0xe0e00000
#define SBC		0xe0c00000
#define SMULL		0xe0c00090
#define STR		0xe5000000
#define STREX		0xe1800f90
#define STREXB		0xe1c00f90
#define STREXH		0xe1e00f90
#define SUB		0xe0400000
#define SXTB		0xe6af0070
#define SXTH		0xe6bf0070
#define TST		0xe1000000
#define UMULL		0xe0800090
#define UXTB		0xe6ef0070
#define UXTH		0xe6ff0070
#define VABS_F32	0xeeb00ac0
#define VADD_F32	0xee300a00
#define VAND		0xf2000110
#define VCMP_F32	0xeeb40a40
#define VCVT_F32_S32	0xeeb80ac0
#define VCVT_F32_U32	0xeeb80a40
#define VCVT_F64_F32	0xeeb70ac0
#define VCVT_S32_F32	0xeebd0ac0
#define VDIV_F32	0xee800a00
#define VDUP		0xee800b10
#define VDUP_s		0xf3b00c00
#define VEOR		0xf3000110
#define VLD1		0xf4200000
#define VLD1_r		0xf4a00c00
#define VLD1_s		0xf4a00000
#define VLDR_F32	0xed100a00
#define VMOV_F32	0xeeb00a40
#define VMOV		0xee000a10
#define VMOV2		0xec400a10
#define VMOV_i		0xf2800010
#define VMOV_s		0xee000b10
#define VMOVN		0xf3b20200
#define VMRS		0xeef1fa10
#define VMUL_F32	0xee200a00
#define VNEG_F32	0xeeb10a40
#define VORR		0xf2200110
#define VPOP		0xecbd0b00
#define VPUSH		0xed2d0b00
#define VSHLL		0xf2800a10
#define VSHR		0xf2800010
#define VSRA		0xf2800110
#define VST1		0xf4000000
#define VST1_s		0xf4800000
#define VSTR_F32	0xed000a00
#define VSUB_F32	0xee300a40

#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
/* Arm v7 specific instructions. */
#define MOVT		0xe3400000
#define MOVW		0xe3000000
#define RBIT		0xe6ff0f30
#endif

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)

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

#endif /* SLJIT_ARGUMENT_CHECKS */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)

static sljit_s32 push_cpool(struct sljit_compiler *compiler)
{
	/* Pushing the constant pool into the instruction stream. */
	sljit_ins* inst;
	sljit_uw* cpool_ptr;
	sljit_uw* cpool_end;
	sljit_s32 i;

	/* The label could point the address after the constant pool. */
	if (compiler->last_label && compiler->last_label->size == compiler->size)
		compiler->last_label->size += compiler->cpool_fill + (CONST_POOL_ALIGNMENT - 1) + 1;

	SLJIT_ASSERT(compiler->cpool_fill > 0 && compiler->cpool_fill <= CPOOL_SIZE);
	inst = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!inst);
	compiler->size++;
	*inst = 0xff000000 | compiler->cpool_fill;

	for (i = 0; i < CONST_POOL_ALIGNMENT - 1; i++) {
		inst = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
		FAIL_IF(!inst);
		compiler->size++;
		*inst = 0;
	}

	cpool_ptr = compiler->cpool;
	cpool_end = cpool_ptr + compiler->cpool_fill;
	while (cpool_ptr < cpool_end) {
		inst = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
		FAIL_IF(!inst);
		compiler->size++;
		*inst = *cpool_ptr++;
	}
	compiler->cpool_diff = CONST_POOL_EMPTY;
	compiler->cpool_fill = 0;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins inst)
{
	sljit_ins* ptr;

	if (SLJIT_UNLIKELY(compiler->cpool_diff != CONST_POOL_EMPTY && compiler->size - compiler->cpool_diff >= MAX_DIFFERENCE(4092)))
		FAIL_IF(push_cpool(compiler));

	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst_with_literal(struct sljit_compiler *compiler, sljit_ins inst, sljit_uw literal)
{
	sljit_ins* ptr;
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
				cpool_index = (sljit_uw)(cpool_ptr - compiler->cpool);
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
	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst | cpool_index;

	compiler->cpool[cpool_index] = literal;
	compiler->cpool_unique[cpool_index] = 0;
	if (compiler->cpool_diff == CONST_POOL_EMPTY)
		compiler->cpool_diff = compiler->size;
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst_with_unique_literal(struct sljit_compiler *compiler, sljit_ins inst, sljit_uw literal)
{
	sljit_ins* ptr;

	if (SLJIT_UNLIKELY((compiler->cpool_diff != CONST_POOL_EMPTY && compiler->size - compiler->cpool_diff >= MAX_DIFFERENCE(4092)) || compiler->cpool_fill >= CPOOL_SIZE))
		FAIL_IF(push_cpool(compiler));

	SLJIT_ASSERT(compiler->cpool_fill < CPOOL_SIZE && (inst & 0xfff) == 0);
	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
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
	SLJIT_ASSERT(reg_map[TMP_REG1] != 14);

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
		if ((*last_pc_patch & 0x0e0f0000) == 0x040f0000) {
			diff = (sljit_uw)(const_pool - last_pc_patch);
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
				diff = (diff + (sljit_uw)ind - 2) << 2;
				SLJIT_ASSERT(diff <= 0xfff);
				*last_pc_patch = (*last_pc_patch & ~(sljit_uw)0xfff) | diff;
			}
			else
				*last_pc_patch = (*last_pc_patch & ~(sljit_uw)(0xfff | (1 << 23))) | 0x004;
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
	sljit_u32 value;
	struct future_patch *curr_patch, *prev_patch;

	SLJIT_UNUSED_ARG(compiler);

	/* Using the values generated by patch_pc_relative_loads. */
	if (!*first_patch)
		value = cpool_start_address[cpool_current_index];
	else {
		curr_patch = *first_patch;
		prev_patch = NULL;
		while (1) {
			if (!curr_patch) {
				value = cpool_start_address[cpool_current_index];
				break;
			}
			if ((sljit_uw)curr_patch->index == cpool_current_index) {
				value = (sljit_uw)curr_patch->value;
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

	if ((sljit_sw)value >= 0) {
		if (value > cpool_current_index) {
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
			curr_patch->index = (sljit_sw)value;
			curr_patch->value = (sljit_sw)cpool_start_address[value];
			*first_patch = curr_patch;
		}
		cpool_start_address[value] = *buf_ptr;
	}
	return SLJIT_SUCCESS;
}

#else

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins inst)
{
	sljit_ins* ptr;

	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	compiler->size++;
	*ptr = inst;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_imm(struct sljit_compiler *compiler, sljit_s32 reg, sljit_sw imm)
{
	FAIL_IF(push_inst(compiler, MOVW | RD(reg) | ((imm << 4) & 0xf0000) | ((sljit_u32)imm & 0xfff)));
	return push_inst(compiler, MOVT | RD(reg) | ((imm >> 12) & 0xf0000) | (((sljit_u32)imm >> 16) & 0xfff));
}

#endif

static SLJIT_INLINE sljit_s32 detect_jump_type(struct sljit_jump *jump, sljit_uw *code_ptr, sljit_uw *code, sljit_sw executable_offset)
{
	sljit_sw diff;

	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		return 0;

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
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
#else /* !SLJIT_CONFIG_ARM_V6 */
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
#endif /* SLJIT_CONFIG_ARM_V6 */
	return 0;
}

static SLJIT_INLINE void inline_set_jump_addr(sljit_uw jump_ptr, sljit_sw executable_offset, sljit_uw new_addr, sljit_s32 flush_cache)
{
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	sljit_ins *ptr = (sljit_ins*)jump_ptr;
	sljit_ins *inst = (sljit_ins*)ptr[0];
	sljit_ins mov_pc = ptr[1];
	sljit_s32 bl = (mov_pc & 0x0000f000) != RD(TMP_PC);
	sljit_sw diff = (sljit_sw)(((sljit_sw)new_addr - (sljit_sw)(inst + 2) - executable_offset) >> 2);

	SLJIT_UNUSED_ARG(executable_offset);

	if (diff <= 0x7fffff && diff >= -0x800000) {
		/* Turn to branch. */
		if (!bl) {
			if (flush_cache) {
				SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 0);
			}
			inst[0] = (mov_pc & COND_MASK) | (B - CONDITIONAL) | (diff & 0xffffff);
			if (flush_cache) {
				SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 1);
				inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
				SLJIT_CACHE_FLUSH(inst, inst + 1);
			}
		} else {
			if (flush_cache) {
				SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 0);
			}
			inst[0] = (mov_pc & COND_MASK) | (BL - CONDITIONAL) | (diff & 0xffffff);
			inst[1] = NOP;
			if (flush_cache) {
				SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 1);
				inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
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
			if (flush_cache) {
				SLJIT_UPDATE_WX_FLAGS(inst, inst + (!bl ? 1 : 2), 0);
			}
			inst[0] = mov_pc;
			if (!bl) {
				if (flush_cache) {
					SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 1);
					inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
					SLJIT_CACHE_FLUSH(inst, inst + 1);
				}
			} else {
				inst[1] = BLX | RM(TMP_REG1);
				if (flush_cache) {
					SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 1);
					inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
					SLJIT_CACHE_FLUSH(inst, inst + 2);
				}
			}
		}

		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(ptr, ptr + 1, 0);
		}

		*ptr = new_addr;

		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(ptr, ptr + 1, 1);
		}
	}
#else /* !SLJIT_CONFIG_ARM_V6 */
	sljit_ins *inst = (sljit_ins*)jump_ptr;

	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_ASSERT((inst[0] & 0xfff00000) == MOVW && (inst[1] & 0xfff00000) == MOVT);

	if (flush_cache) {
		SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 0);
	}

	inst[0] = MOVW | (inst[0] & 0xf000) | ((new_addr << 4) & 0xf0000) | (new_addr & 0xfff);
	inst[1] = MOVT | (inst[1] & 0xf000) | ((new_addr >> 12) & 0xf0000) | ((new_addr >> 16) & 0xfff);

	if (flush_cache) {
		SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 1);
		inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
		SLJIT_CACHE_FLUSH(inst, inst + 2);
	}
#endif /* SLJIT_CONFIG_ARM_V6 */
}

static sljit_uw get_imm(sljit_uw imm);
static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 reg, sljit_uw imm);
static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw, sljit_s32 tmp_reg);

static SLJIT_INLINE void inline_set_const(sljit_uw addr, sljit_sw executable_offset, sljit_uw new_constant, sljit_s32 flush_cache)
{
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	sljit_ins *ptr = (sljit_ins*)addr;
	sljit_ins *inst = (sljit_ins*)ptr[0];
	sljit_uw ldr_literal = ptr[1];
	sljit_uw src2;

	SLJIT_UNUSED_ARG(executable_offset);

	src2 = get_imm(new_constant);
	if (src2) {
		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 0);
		}

		*inst = 0xe3a00000 | (ldr_literal & 0xf000) | src2;

		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 1);
			inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
			SLJIT_CACHE_FLUSH(inst, inst + 1);
		}
		return;
	}

	src2 = get_imm(~new_constant);
	if (src2) {
		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 0);
		}

		*inst = 0xe3e00000 | (ldr_literal & 0xf000) | src2;

		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 1);
			inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
			SLJIT_CACHE_FLUSH(inst, inst + 1);
		}
		return;
	}

	if (ldr_literal & (1 << 23))
		ptr = inst + ((ldr_literal & 0xfff) >> 2) + 2;
	else
		ptr = inst + 1;

	if (*inst != ldr_literal) {
		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 0);
		}

		*inst = ldr_literal;

		if (flush_cache) {
			SLJIT_UPDATE_WX_FLAGS(inst, inst + 1, 1);
			inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
			SLJIT_CACHE_FLUSH(inst, inst + 1);
		}
	}

	if (flush_cache) {
		SLJIT_UPDATE_WX_FLAGS(ptr, ptr + 1, 0);
	}

	*ptr = new_constant;

	if (flush_cache) {
		SLJIT_UPDATE_WX_FLAGS(ptr, ptr + 1, 1);
	}
#else /* !SLJIT_CONFIG_ARM_V6 */
	sljit_ins *inst = (sljit_ins*)addr;

	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_ASSERT((inst[0] & 0xfff00000) == MOVW && (inst[1] & 0xfff00000) == MOVT);

	if (flush_cache) {
		SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 0);
	}

	inst[0] = MOVW | (inst[0] & 0xf000) | ((new_constant << 4) & 0xf0000) | (new_constant & 0xfff);
	inst[1] = MOVT | (inst[1] & 0xf000) | ((new_constant >> 12) & 0xf0000) | ((new_constant >> 16) & 0xfff);

	if (flush_cache) {
		SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 1);
		inst = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
		SLJIT_CACHE_FLUSH(inst, inst + 2);
	}
#endif /* SLJIT_CONFIG_ARM_V6 */
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw size;
	sljit_uw word_count;
	sljit_uw next_addr;
	sljit_sw executable_offset;
	sljit_uw addr;
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	sljit_uw cpool_size;
	sljit_uw cpool_skip_alignment;
	sljit_uw cpool_current_index;
	sljit_ins *cpool_start_address;
	sljit_ins *last_pc_patch;
	struct future_patch *first_patch;
#endif

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;
	struct sljit_put_label *put_label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	/* Second code generation pass. */
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	size = compiler->size + (compiler->patches << 1);
	if (compiler->cpool_fill > 0)
		size += compiler->cpool_fill + CONST_POOL_ALIGNMENT - 1;
#else /* !SLJIT_CONFIG_ARM_V6 */
	size = compiler->size;
#endif /* SLJIT_CONFIG_ARM_V6 */
	code = (sljit_ins*)SLJIT_MALLOC_EXEC(size * sizeof(sljit_ins), compiler->exec_allocator_data);
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	cpool_size = 0;
	cpool_skip_alignment = 0;
	cpool_current_index = 0;
	cpool_start_address = NULL;
	first_patch = NULL;
	last_pc_patch = code;
#endif /* SLJIT_CONFIG_ARM_V6 */

	code_ptr = code;
	word_count = 0;
	next_addr = 1;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	put_label = compiler->put_labels;

	if (label && label->size == 0) {
		label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
		label = label->next;
	}

	do {
		buf_ptr = (sljit_ins*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			word_count++;
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
			if (cpool_size > 0) {
				if (cpool_skip_alignment > 0) {
					buf_ptr++;
					cpool_skip_alignment--;
				}
				else {
					if (SLJIT_UNLIKELY(resolve_const_pool_index(compiler, &first_patch, cpool_current_index, cpool_start_address, buf_ptr))) {
						SLJIT_FREE_EXEC(code, compiler->exec_allocator_data);
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
							label->size = (sljit_uw)(code_ptr - code);
							label = label->next;

							next_addr = compute_next_addr(label, jump, const_, put_label);
						}
					}
				}
			}
			else if ((*buf_ptr & 0xff000000) != PUSH_POOL) {
#endif /* SLJIT_CONFIG_ARM_V6 */
				*code_ptr = *buf_ptr++;
				if (next_addr == word_count) {
					SLJIT_ASSERT(!label || label->size >= word_count);
					SLJIT_ASSERT(!jump || jump->addr >= word_count);
					SLJIT_ASSERT(!const_ || const_->addr >= word_count);
					SLJIT_ASSERT(!put_label || put_label->addr >= word_count);

				/* These structures are ordered by their address. */
					if (jump && jump->addr == word_count) {
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
						if (detect_jump_type(jump, code_ptr, code, executable_offset))
							code_ptr--;
						jump->addr = (sljit_uw)code_ptr;
#else /* !SLJIT_CONFIG_ARM_V6 */
						jump->addr = (sljit_uw)(code_ptr - 2);
						if (detect_jump_type(jump, code_ptr, code, executable_offset))
							code_ptr -= 2;
#endif /* SLJIT_CONFIG_ARM_V6 */
						jump = jump->next;
					}
					if (label && label->size == word_count) {
						/* code_ptr can be affected above. */
						label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr + 1, executable_offset);
						label->size = (sljit_uw)((code_ptr + 1) - code);
						label = label->next;
					}
					if (const_ && const_->addr == word_count) {
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
						const_->addr = (sljit_uw)code_ptr;
#else /* !SLJIT_CONFIG_ARM_V6 */
						const_->addr = (sljit_uw)(code_ptr - 1);
#endif /* SLJIT_CONFIG_ARM_V6 */
						const_ = const_->next;
					}
					if (put_label && put_label->addr == word_count) {
						SLJIT_ASSERT(put_label->label);
						put_label->addr = (sljit_uw)code_ptr;
						put_label = put_label->next;
					}
					next_addr = compute_next_addr(label, jump, const_, put_label);
				}
				code_ptr++;
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
			} else {
				/* Fortunately, no need to shift. */
				cpool_size = *buf_ptr++ & ~PUSH_POOL;
				SLJIT_ASSERT(cpool_size > 0);
				cpool_start_address = ALIGN_INSTRUCTION(code_ptr + 1);
				cpool_current_index = patch_pc_relative_loads(last_pc_patch, code_ptr, cpool_start_address, cpool_size);
				if (cpool_current_index > 0) {
					/* Unconditional branch. */
					*code_ptr = B | (((sljit_ins)(cpool_start_address - code_ptr) + cpool_current_index - 2) & ~PUSH_POOL);
					code_ptr = (sljit_ins*)(cpool_start_address + cpool_current_index);
				}
				cpool_skip_alignment = CONST_POOL_ALIGNMENT - 1;
				cpool_current_index = 0;
				last_pc_patch = code_ptr;
			}
#endif /* SLJIT_CONFIG_ARM_V6 */
		} while (buf_ptr < buf_end);
		buf = buf->next;
	} while (buf);

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(!put_label);

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	SLJIT_ASSERT(cpool_size == 0);
	if (compiler->cpool_fill > 0) {
		cpool_start_address = ALIGN_INSTRUCTION(code_ptr);
		cpool_current_index = patch_pc_relative_loads(last_pc_patch, code_ptr, cpool_start_address, compiler->cpool_fill);
		if (cpool_current_index > 0)
			code_ptr = (sljit_ins*)(cpool_start_address + cpool_current_index);

		buf_ptr = compiler->cpool;
		buf_end = buf_ptr + compiler->cpool_fill;
		cpool_current_index = 0;
		while (buf_ptr < buf_end) {
			if (SLJIT_UNLIKELY(resolve_const_pool_index(compiler, &first_patch, cpool_current_index, cpool_start_address, buf_ptr))) {
				SLJIT_FREE_EXEC(code, compiler->exec_allocator_data);
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
		buf_ptr = (sljit_ins*)jump->addr;

		if (jump->flags & PATCH_B) {
			addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr + 2, executable_offset);
			if (!(jump->flags & JUMP_ADDR)) {
				SLJIT_ASSERT(jump->flags & JUMP_LABEL);
				SLJIT_ASSERT((sljit_sw)(jump->u.label->addr - addr) <= 0x01ffffff && (sljit_sw)(jump->u.label->addr - addr) >= -0x02000000);
				*buf_ptr |= ((jump->u.label->addr - addr) >> 2) & 0x00ffffff;
			}
			else {
				SLJIT_ASSERT((sljit_sw)(jump->u.target - addr) <= 0x01ffffff && (sljit_sw)(jump->u.target - addr) >= -0x02000000);
				*buf_ptr |= ((jump->u.target - addr) >> 2) & 0x00ffffff;
			}
		}
		else if (jump->flags & SLJIT_REWRITABLE_JUMP) {
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
			jump->addr = (sljit_uw)code_ptr;
			code_ptr[0] = (sljit_ins)buf_ptr;
			code_ptr[1] = *buf_ptr;
			inline_set_jump_addr((sljit_uw)code_ptr, executable_offset, (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target, 0);
			code_ptr += 2;
#else /* !SLJIT_CONFIG_ARM_V6 */
			inline_set_jump_addr((sljit_uw)buf_ptr, executable_offset, (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target, 0);
#endif /* SLJIT_CONFIG_ARM_V6 */
		} else {
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
			if (jump->flags & IS_BL)
				buf_ptr--;
			if (*buf_ptr & (1 << 23))
				buf_ptr += ((*buf_ptr & 0xfff) >> 2) + 2;
			else
				buf_ptr += 1;
			*buf_ptr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
#else /* !SLJIT_CONFIG_ARM_V6 */
			inline_set_jump_addr((sljit_uw)buf_ptr, executable_offset, (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target, 0);
#endif /* SLJIT_CONFIG_ARM_V6 */
		}
		jump = jump->next;
	}

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	const_ = compiler->consts;
	while (const_) {
		buf_ptr = (sljit_ins*)const_->addr;
		const_->addr = (sljit_uw)code_ptr;

		code_ptr[0] = (sljit_ins)buf_ptr;
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
#endif /* SLJIT_CONFIG_ARM_V6 */

	put_label = compiler->put_labels;
	while (put_label) {
		addr = put_label->label->addr;
		buf_ptr = (sljit_ins*)put_label->addr;

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
		SLJIT_ASSERT((buf_ptr[0] & 0xffff0000) == 0xe59f0000);
		buf_ptr[((buf_ptr[0] & 0xfff) >> 2) + 2] = addr;
#else /* !SLJIT_CONFIG_ARM_V6 */
		SLJIT_ASSERT((buf_ptr[-1] & 0xfff00000) == MOVW && (buf_ptr[0] & 0xfff00000) == MOVT);
		buf_ptr[-1] |= ((addr << 4) & 0xf0000) | (addr & 0xfff);
		buf_ptr[0] |= ((addr >> 12) & 0xf0000) | ((addr >> 16) & 0xfff);
#endif /* SLJIT_CONFIG_ARM_V6 */
		put_label = put_label->next;
	}

	SLJIT_ASSERT(code_ptr - code <= (sljit_s32)size);

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (sljit_uw)(code_ptr - code) * sizeof(sljit_uw);

	code = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
	code_ptr = (sljit_ins*)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

	SLJIT_CACHE_FLUSH(code, code_ptr);
	SLJIT_UPDATE_WX_FLAGS(code, code_ptr, 1);
	return code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	switch (feature_type) {
	case SLJIT_HAS_FPU:
	case SLJIT_HAS_F64_AS_F32_PAIR:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#else
		/* Available by default. */
		return 1;
#endif /* SLJIT_IS_FPU_AVAILABLE */
	case SLJIT_HAS_SIMD:
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
		return 0;
#else
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#else
		/* Available by default. */
		return 1;
#endif /* SLJIT_IS_FPU_AVAILABLE */
#endif /* SLJIT_CONFIG_ARM_V6 */

	case SLJIT_SIMD_REGS_ARE_PAIRS:
	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_ROT:
	case SLJIT_HAS_CMOV:
	case SLJIT_HAS_REV:
	case SLJIT_HAS_PREFETCH:
	case SLJIT_HAS_COPY_F32:
	case SLJIT_HAS_COPY_F64:
	case SLJIT_HAS_ATOMIC:
		return 1;

	case SLJIT_HAS_CTZ:
#if defined(SLJIT_CONFIG_ARM_V6) && SLJIT_CONFIG_ARM_V6
		return 2;
#else
		return 1;
#endif /* SLJIT_CONFIG_ARM_V6 */

	default:
		return 0;
	}
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

/* Creates an index in data_transfer_insts array. */
#define WORD_SIZE	0x00
#define BYTE_SIZE	0x01
#define HALF_SIZE	0x02
#define PRELOAD		0x03
#define SIGNED		0x04
#define LOAD_DATA	0x08

/* Flag bits for emit_op. */
#define ALLOW_IMM		0x10
#define ALLOW_INV_IMM		0x20
#define ALLOW_ANY_IMM		(ALLOW_IMM | ALLOW_INV_IMM)
#define ALLOW_NEG_IMM		0x40
#define ALLOW_DOUBLE_IMM	0x80

/* s/l - store/load (1 bit)
   u/s - signed/unsigned (1 bit)
   w/b/h/N - word/byte/half/NOT allowed (2 bit)
   Storing signed and unsigned values are the same operations. */

static const sljit_ins data_transfer_insts[16] = {
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
/* l u p */ 0xf5500000 /* preload */,
/* l s w */ 0xe5100000 /* ldr */,
/* l s b */ 0xe11000d0 /* ldrsb */,
/* l s h */ 0xe11000f0 /* ldrsh */,
/* l s N */ 0x00000000 /* not allowed */,
};

#define EMIT_DATA_TRANSFER(type, add, target_reg, base_reg, arg) \
	(data_transfer_insts[(type) & 0xf] | ((add) << 23) | RD(target_reg) | RN(base_reg) | (sljit_ins)(arg))

/* Normal ldr/str instruction.
   Type2: ldrsb, ldrh, ldrsh */
#define IS_TYPE1_TRANSFER(type) \
	(data_transfer_insts[(type) & 0xf] & 0x04000000)
#define TYPE2_TRANSFER_IMM(imm) \
	(((imm) & 0xf) | (((imm) & 0xf0) << 4) | (1 << 22))

#define EMIT_FPU_OPERATION(opcode, mode, dst, src1, src2) \
	((sljit_ins)(opcode) | (sljit_ins)(mode) | VD(dst) | VM(src1) | VN(src2))

/* Flags for emit_op: */
  /* Arguments are swapped. */
#define ARGS_SWAPPED	0x01
  /* Inverted immediate. */
#define INV_IMM		0x02
  /* Source and destination is register. */
#define MOVE_REG_CONV	0x04
  /* Unused return value. */
#define UNUSED_RETURN	0x08
/* SET_FLAGS must be (1 << 20) as it is also the value of S bit (can be used for optimization). */
#define SET_FLAGS	(1 << 20)
/* dst: reg
   src1: reg
   src2: reg or imm (if allowed)
   SRC2_IMM must be (1 << 25) as it is also the value of I bit (can be used for optimization). */
#define SRC2_IMM	(1 << 25)

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 inp_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_uw imm, offset;
	sljit_s32 i, tmp, size, word_arg_count;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);
#ifdef __SOFTFP__
	sljit_u32 float_arg_count;
#else
	sljit_u32 old_offset, f32_offset;
	sljit_u32 remap[3];
	sljit_u32 *remap_ptr = remap;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	imm = 0;

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--)
		imm |= (sljit_uw)1 << reg_map[i];

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--)
		imm |= (sljit_uw)1 << reg_map[i];

	SLJIT_ASSERT(reg_map[TMP_REG2] == 14);

	/* Push saved and temporary registers
	   multiple registers: stmdb sp!, {..., lr}
	   single register: str reg, [sp, #-4]! */
	if (imm != 0)
		FAIL_IF(push_inst(compiler, PUSH | (1 << 14) | imm));
	else
		FAIL_IF(push_inst(compiler, 0xe52d0004 | RD(TMP_REG2)));

	/* Stack must be aligned to 8 bytes: */
	size = GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 1);

	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if ((size & SSIZE_OF(sw)) != 0) {
			FAIL_IF(push_inst(compiler, SUB | RD(SLJIT_SP) | RN(SLJIT_SP) | SRC2_IMM | sizeof(sljit_sw)));
			size += SSIZE_OF(sw);
		}

		if (fsaveds + fscratches >= SLJIT_NUMBER_OF_FLOAT_REGISTERS) {
			FAIL_IF(push_inst(compiler, VPUSH | VD(SLJIT_FS0) | ((sljit_ins)SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS << 1)));
		} else {
			if (fsaveds > 0)
				FAIL_IF(push_inst(compiler, VPUSH | VD(SLJIT_FS0) | ((sljit_ins)fsaveds << 1)));
			if (fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG)
				FAIL_IF(push_inst(compiler, VPUSH | VD(fscratches) | ((sljit_ins)(fscratches - (SLJIT_FIRST_SAVED_FLOAT_REG - 1)) << 1)));
		}
	}

	local_size = ((size + local_size + 0x7) & ~0x7) - size;
	compiler->local_size = local_size;

	if (options & SLJIT_ENTER_REG_ARG)
		arg_types = 0;

	arg_types >>= SLJIT_ARG_SHIFT;
	word_arg_count = 0;
	saved_arg_count = 0;
#ifdef __SOFTFP__
	SLJIT_COMPILE_ASSERT(SLJIT_FR0 == 1, float_register_index_start);

	offset = 0;
	float_arg_count = 0;

	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			if (offset & 0x7)
				offset += sizeof(sljit_sw);

			if (offset < 4 * sizeof(sljit_sw))
				FAIL_IF(push_inst(compiler, VMOV2 | (offset << 10) | ((offset + sizeof(sljit_sw)) << 14) | float_arg_count));
			else
				FAIL_IF(push_inst(compiler, VLDR_F32 | 0x800100 | RN(SLJIT_SP)
						| (float_arg_count << 12) | ((offset + (sljit_ins)size - 4 * sizeof(sljit_sw)) >> 2)));
			float_arg_count++;
			offset += sizeof(sljit_f64) - sizeof(sljit_sw);
			break;
		case SLJIT_ARG_TYPE_F32:
			if (offset < 4 * sizeof(sljit_sw))
				FAIL_IF(push_inst(compiler, VMOV | (float_arg_count << 16) | (offset << 10)));
			else
				FAIL_IF(push_inst(compiler, VLDR_F32 | 0x800000 | RN(SLJIT_SP)
						| (float_arg_count << 12) | ((offset + (sljit_ins)size - 4 * sizeof(sljit_sw)) >> 2)));
			float_arg_count++;
			break;
		default:
			word_arg_count++;

			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				tmp = SLJIT_S0 - saved_arg_count;
				saved_arg_count++;
			} else if (word_arg_count - 1 != (sljit_s32)(offset >> 2))
				tmp = word_arg_count;
			else
				break;

			if (offset < 4 * sizeof(sljit_sw))
				FAIL_IF(push_inst(compiler, MOV | RD(tmp) | (offset >> 2)));
			else
				FAIL_IF(push_inst(compiler, LDR | 0x800000 | RN(SLJIT_SP) | RD(tmp) | (offset + (sljit_ins)size - 4 * sizeof(sljit_sw))));
			break;
		}

		offset += sizeof(sljit_sw);
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	compiler->args_size = offset;
#else
	offset = SLJIT_FR0;
	old_offset = SLJIT_FR0;
	f32_offset = 0;

	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			if (offset != old_offset)
				*remap_ptr++ = EMIT_FPU_OPERATION(VMOV_F32, SLJIT_32, offset, old_offset, 0);
			old_offset++;
			offset++;
			break;
		case SLJIT_ARG_TYPE_F32:
			if (f32_offset != 0) {
				*remap_ptr++ = EMIT_FPU_OPERATION(VMOV_F32, 0x20, offset, f32_offset, 0);
				f32_offset = 0;
			} else {
				if (offset != old_offset)
					*remap_ptr++ = EMIT_FPU_OPERATION(VMOV_F32, 0, offset, old_offset, 0);
				f32_offset = old_offset;
				old_offset++;
			}
			offset++;
			break;
		default:
			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				FAIL_IF(push_inst(compiler, MOV | RD(SLJIT_S0 - saved_arg_count) | RM(SLJIT_R0 + word_arg_count)));
				saved_arg_count++;
			}

			word_arg_count++;
			break;
		}
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	SLJIT_ASSERT((sljit_uw)(remap_ptr - remap) <= sizeof(remap));

	while (remap_ptr > remap)
		FAIL_IF(push_inst(compiler, *(--remap_ptr)));
#endif

	if (local_size > 0)
		FAIL_IF(emit_op(compiler, SLJIT_SUB, ALLOW_IMM | ALLOW_DOUBLE_IMM, SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, local_size));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 size;

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	size = GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 1);

	/* Doubles are saved, so alignment is unaffected. */
	if ((size & SSIZE_OF(sw)) != 0 && (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG))
		size += SSIZE_OF(sw);

	compiler->local_size = ((size + local_size + 0x7) & ~0x7) - size;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_add_sp(struct sljit_compiler *compiler, sljit_uw imm)
{
	sljit_uw imm2 = get_imm(imm);

	if (imm2 == 0)
		return emit_op(compiler, SLJIT_ADD, ALLOW_IMM | ALLOW_DOUBLE_IMM, SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, (sljit_sw)imm);

	return push_inst(compiler, ADD | RD(SLJIT_SP) | RN(SLJIT_SP) | imm2);
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 frame_size)
{
	sljit_s32 local_size, fscratches, fsaveds, i, tmp;
	sljit_s32 restored_reg = 0;
	sljit_s32 lr_dst = TMP_PC;
	sljit_uw reg_list = 0;

	SLJIT_ASSERT(reg_map[TMP_REG2] == 14 && frame_size <= 128);

	local_size = compiler->local_size;
	fscratches = compiler->fscratches;
	fsaveds = compiler->fsaveds;

	if (fsaveds > 0 || fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
		if (local_size > 0)
			FAIL_IF(emit_add_sp(compiler, (sljit_uw)local_size));

		if (fsaveds + fscratches >= SLJIT_NUMBER_OF_FLOAT_REGISTERS) {
			FAIL_IF(push_inst(compiler, VPOP | VD(SLJIT_FS0) | ((sljit_ins)SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS << 1)));
		} else {
			if (fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG)
				FAIL_IF(push_inst(compiler, VPOP | VD(fscratches) | ((sljit_ins)(fscratches - (SLJIT_FIRST_SAVED_FLOAT_REG - 1)) << 1)));
			if (fsaveds > 0)
				FAIL_IF(push_inst(compiler, VPOP | VD(SLJIT_FS0) | ((sljit_ins)fsaveds << 1)));
		}

		local_size = GET_SAVED_REGISTERS_SIZE(compiler->scratches, compiler->saveds, 1) & 0x7;
	}

	if (frame_size < 0) {
		lr_dst = TMP_REG2;
		frame_size = 0;
	} else if (frame_size > 0) {
		SLJIT_ASSERT(frame_size == 1 || (frame_size & 0x7) == 0);
		lr_dst = 0;
		frame_size &= ~0x7;
	}

	if (lr_dst != 0)
		reg_list |= (sljit_uw)1 << reg_map[lr_dst];

	tmp = SLJIT_S0 - compiler->saveds;
	i = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options);
	if (tmp < i) {
		restored_reg = i;
		do {
			reg_list |= (sljit_uw)1 << reg_map[i];
		} while (--i > tmp);
	}

	i = compiler->scratches;
	if (i >= SLJIT_FIRST_SAVED_REG) {
		restored_reg = i;
		do {
			reg_list |= (sljit_uw)1 << reg_map[i];
		} while (--i >= SLJIT_FIRST_SAVED_REG);
	}

	if (lr_dst == TMP_REG2 && reg_list == 0) {
		restored_reg = TMP_REG2;
		lr_dst = 0;
	}

	if (lr_dst == 0 && (reg_list & (reg_list - 1)) == 0) {
		/* The local_size does not include the saved registers. */
		tmp = 0;
		if (reg_list != 0) {
			tmp = 2;
			if (local_size <= 0xfff) {
				if (local_size == 0) {
					SLJIT_ASSERT(restored_reg != TMP_REG2);
					if (frame_size == 0)
						return push_inst(compiler, LDR_POST | RN(SLJIT_SP) | RD(restored_reg) | 0x800008);
					if (frame_size > 2 * SSIZE_OF(sw))
						return push_inst(compiler, LDR_POST | RN(SLJIT_SP) | RD(restored_reg) | (sljit_ins)(frame_size - (2 * SSIZE_OF(sw))));
				}

				FAIL_IF(push_inst(compiler, LDR | 0x800000 | RN(SLJIT_SP) | RD(restored_reg) | (sljit_ins)local_size));
				tmp = 1;
			} else if (frame_size == 0) {
				frame_size = (restored_reg == TMP_REG2) ? SSIZE_OF(sw) : 2 * SSIZE_OF(sw);
				tmp = 3;
			}

			/* Place for the saved register. */
			if (restored_reg != TMP_REG2)
				local_size += SSIZE_OF(sw);
		}

		/* Place for the lr register. */
		local_size += SSIZE_OF(sw);

		if (frame_size > local_size)
			FAIL_IF(push_inst(compiler, SUB | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 25) | (sljit_ins)(frame_size - local_size)));
		else if (frame_size < local_size)
			FAIL_IF(emit_add_sp(compiler, (sljit_uw)(local_size - frame_size)));

		if (tmp <= 1)
			return SLJIT_SUCCESS;

		if (tmp == 2) {
			frame_size -= SSIZE_OF(sw);
			if (restored_reg != TMP_REG2)
				frame_size -= SSIZE_OF(sw);

			return push_inst(compiler, LDR | 0x800000 | RN(SLJIT_SP) | RD(restored_reg) | (sljit_ins)frame_size);
		}

		tmp = (restored_reg == TMP_REG2) ? 0x800004 : 0x800008;
		return push_inst(compiler, LDR_POST | RN(SLJIT_SP) | RD(restored_reg) | (sljit_ins)tmp);
	}

	if (local_size > 0)
		FAIL_IF(emit_add_sp(compiler, (sljit_uw)local_size));

	/* Pop saved and temporary registers
	   multiple registers: ldmia sp!, {...}
	   single register: ldr reg, [sp], #4 */
	if ((reg_list & (reg_list - 1)) == 0) {
		SLJIT_ASSERT(lr_dst != 0);
		SLJIT_ASSERT(reg_list == (sljit_uw)1 << reg_map[lr_dst]);

		return push_inst(compiler, LDR_POST | RN(SLJIT_SP) | RD(lr_dst) | 0x800004);
	}

	FAIL_IF(push_inst(compiler, POP | reg_list));

	if (frame_size > 0)
		return push_inst(compiler, SUB | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 25) | ((sljit_ins)frame_size - sizeof(sljit_sw)));

	if (lr_dst != 0)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ADD | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 25) | sizeof(sljit_sw));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	return emit_stack_frame_release(compiler, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG1, src, srcw, TMP_REG1));
		src = TMP_REG1;
		srcw = 0;
	} else if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | RM(src)));
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

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_uw dst, sljit_uw src1, sljit_uw src2)
{
	sljit_s32 is_masked;
	sljit_uw shift_type;

	switch (op) {
	case SLJIT_MOV:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		if (dst != src2) {
			if (src2 & SRC2_IMM) {
				return push_inst(compiler, ((flags & INV_IMM) ? MVN : MOV) | RD(dst) | src2);
			}
			return push_inst(compiler, MOV | RD(dst) | RM(src2));
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		if (flags & MOVE_REG_CONV)
			return push_inst(compiler, (op == SLJIT_MOV_U8 ? UXTB : SXTB) | RD(dst) | RM(src2));

		if (dst != src2) {
			SLJIT_ASSERT(src2 & SRC2_IMM);
			return push_inst(compiler, ((flags & INV_IMM) ? MVN : MOV) | RD(dst) | src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		if (flags & MOVE_REG_CONV)
			return push_inst(compiler, (op == SLJIT_MOV_U16 ? UXTH : SXTH) | RD(dst) | RM(src2));

		if (dst != src2) {
			SLJIT_ASSERT(src2 & SRC2_IMM);
			return push_inst(compiler, ((flags & INV_IMM) ? MVN : MOV) | RD(dst) | src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_CLZ:
		SLJIT_ASSERT(!(flags & INV_IMM) && !(src2 & SRC2_IMM));
		FAIL_IF(push_inst(compiler, CLZ | RD(dst) | RM(src2)));
		return SLJIT_SUCCESS;

	case SLJIT_CTZ:
		SLJIT_ASSERT(!(flags & INV_IMM) && !(src2 & SRC2_IMM));
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
		FAIL_IF(push_inst(compiler, RSB | SRC2_IMM | RD(TMP_REG1) | RN(src2) | 0));
		FAIL_IF(push_inst(compiler, AND | RD(TMP_REG2) | RN(src2) | RM(TMP_REG1)));
		FAIL_IF(push_inst(compiler, CLZ | RD(dst) | RM(TMP_REG2)));
		FAIL_IF(push_inst(compiler, CMP | SET_FLAGS | SRC2_IMM | RN(dst) | 32));
		return push_inst(compiler, (EOR ^ 0xf0000000) | SRC2_IMM | RD(dst) | RN(dst) | 0x1f);
#else /* !SLJIT_CONFIG_ARM_V6 */
		FAIL_IF(push_inst(compiler, RBIT | RD(dst) | RM(src2)));
		return push_inst(compiler, CLZ | RD(dst) | RM(dst));
#endif /* SLJIT_CONFIG_ARM_V6 */

	case SLJIT_REV:
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED));
		return push_inst(compiler, REV | RD(dst) | RM(src2));

	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & ARGS_SWAPPED) && src2 != TMP_REG1 && dst != TMP_REG1);
		FAIL_IF(push_inst(compiler, REV16 | RD(dst) | RM(src2)));
		if (dst == TMP_REG2 || (src2 == TMP_REG2 && op == SLJIT_REV_U16))
			return SLJIT_SUCCESS;
		return push_inst(compiler, (op == SLJIT_REV_U16 ? UXTH : SXTH) | RD(dst) | RM(dst));
	case SLJIT_ADD:
		SLJIT_ASSERT(!(flags & INV_IMM));

		if ((flags & (UNUSED_RETURN | ARGS_SWAPPED)) == UNUSED_RETURN)
			return push_inst(compiler, CMN | SET_FLAGS | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));
		return push_inst(compiler, ADD | (flags & SET_FLAGS) | RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_ADDC:
		SLJIT_ASSERT(!(flags & INV_IMM));
		return push_inst(compiler, ADC | (flags & SET_FLAGS) | RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_SUB:
		SLJIT_ASSERT(!(flags & INV_IMM));

		if ((flags & (UNUSED_RETURN | ARGS_SWAPPED)) == UNUSED_RETURN)
			return push_inst(compiler, CMP | SET_FLAGS | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

		return push_inst(compiler, (!(flags & ARGS_SWAPPED) ? SUB : RSB) | (flags & SET_FLAGS)
			| RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_SUBC:
		SLJIT_ASSERT(!(flags & INV_IMM));
		return push_inst(compiler, (!(flags & ARGS_SWAPPED) ? SBC : RSC) | (flags & SET_FLAGS)
			| RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & INV_IMM));
		SLJIT_ASSERT(!(src2 & SRC2_IMM));
		compiler->status_flags_state = 0;

		if (!(flags & SET_FLAGS))
			return push_inst(compiler, MUL | RN(dst) | RM8(src2) | RM(src1));

		FAIL_IF(push_inst(compiler, SMULL | RN(TMP_REG1) | RD(dst) | RM8(src2) | RM(src1)));

		/* cmp TMP_REG1, dst asr #31. */
		return push_inst(compiler, CMP | SET_FLAGS | RN(TMP_REG1) | RM(dst) | 0xfc0);

	case SLJIT_AND:
		if ((flags & (UNUSED_RETURN | INV_IMM)) == UNUSED_RETURN)
			return push_inst(compiler, TST | SET_FLAGS | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));
		return push_inst(compiler, (!(flags & INV_IMM) ? AND : BIC) | (flags & SET_FLAGS)
			| RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_OR:
		SLJIT_ASSERT(!(flags & INV_IMM));
		return push_inst(compiler, ORR | (flags & SET_FLAGS) | RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_XOR:
		if (flags & INV_IMM) {
			SLJIT_ASSERT(src2 == SRC2_IMM);
			return push_inst(compiler, MVN | (flags & SET_FLAGS) | RD(dst) | RM(src1));
		}
		return push_inst(compiler, EOR | (flags & SET_FLAGS) | RD(dst) | RN(src1) | ((src2 & SRC2_IMM) ? src2 : RM(src2)));

	case SLJIT_SHL:
	case SLJIT_MSHL:
		shift_type = 0;
		is_masked = op == SLJIT_MSHL;
		break;

	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		shift_type = 1;
		is_masked = op == SLJIT_MLSHR;
		break;

	case SLJIT_ASHR:
	case SLJIT_MASHR:
		shift_type = 2;
		is_masked = op == SLJIT_MASHR;
		break;

	case SLJIT_ROTL:
		if (compiler->shift_imm == 0x20) {
			FAIL_IF(push_inst(compiler, RSB | SRC2_IMM | RD(TMP_REG2) | RN(src2) | 0));
			src2 = TMP_REG2;
		} else
			compiler->shift_imm = (sljit_uw)(-(sljit_sw)compiler->shift_imm) & 0x1f;
		/* fallthrough */

	case SLJIT_ROTR:
		shift_type = 3;
		is_masked = 0;
		break;

	default:
		SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;
	}

	SLJIT_ASSERT(!(flags & ARGS_SWAPPED) && !(flags & INV_IMM) && !(src2 & SRC2_IMM));

	if (compiler->shift_imm != 0x20) {
		SLJIT_ASSERT(src1 == TMP_REG1);

		if (compiler->shift_imm != 0)
			return push_inst(compiler, MOV | (flags & SET_FLAGS) |
				RD(dst) | (compiler->shift_imm << 7) | (shift_type << 5) | RM(src2));
		return push_inst(compiler, MOV | (flags & SET_FLAGS) | RD(dst) | RM(src2));
	}

	SLJIT_ASSERT(src1 != TMP_REG2);

	if (is_masked) {
		FAIL_IF(push_inst(compiler, AND | RD(TMP_REG2) | RN(src2) | SRC2_IMM | 0x1f));
		src2 = TMP_REG2;
	}

	return push_inst(compiler, MOV | (flags & SET_FLAGS) | RD(dst)
		| RM8(src2) | (sljit_ins)(shift_type << 5) | 0x10 | RM(src1));
}

#undef EMIT_SHIFT_INS_AND_RETURN

/* Tests whether the immediate can be stored in the 12 bit imm field.
   Returns with 0 if not possible. */
static sljit_uw get_imm(sljit_uw imm)
{
	sljit_u32 rol;

	if (imm <= 0xff)
		return SRC2_IMM | imm;

	if (!(imm & 0xff000000)) {
		imm <<= 8;
		rol = 8;
	} else {
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
	return 0;
}

static sljit_uw compute_imm(sljit_uw imm, sljit_uw* imm2)
{
	sljit_uw mask;
	sljit_uw imm1;
	sljit_uw rol;

	/* Step1: Search a zero byte (8 continous zero bit). */
	mask = 0xff000000;
	rol = 8;
	while (1) {
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
		*imm2 = SRC2_IMM | ((imm >> 8) & 0xff) | (((rol + 8) & 0xf) << 8);
	} else if (imm & 0xc0000000) {
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
			*imm2 = SRC2_IMM | (imm >> 24) | ((rol & 0xf) << 8);
		else
			return 0;
	} else {
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
			*imm2 = SRC2_IMM | (imm >> 24) | ((rol & 0xf) << 8);
		else
			return 0;
	}

	return imm1;
}

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 reg, sljit_uw imm)
{
	sljit_uw tmp;
#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	sljit_uw imm1, imm2;
#else /* !SLJIT_CONFIG_ARM_V6 */
	if (!(imm & ~(sljit_uw)0xffff))
		return push_inst(compiler, MOVW | RD(reg) | ((imm << 4) & 0xf0000) | (imm & 0xfff));
#endif /* SLJIT_CONFIG_ARM_V6 */

	/* Create imm by 1 inst. */
	tmp = get_imm(imm);
	if (tmp)
		return push_inst(compiler, MOV | RD(reg) | tmp);

	tmp = get_imm(~imm);
	if (tmp)
		return push_inst(compiler, MVN | RD(reg) | tmp);

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	/* Create imm by 2 inst. */
	imm1 = compute_imm(imm, &imm2);
	if (imm1 != 0) {
		FAIL_IF(push_inst(compiler, MOV | RD(reg) | imm1));
		return push_inst(compiler, ORR | RD(reg) | RN(reg) | imm2);
	}

	imm1 = compute_imm(~imm, &imm2);
	if (imm1 != 0) {
		FAIL_IF(push_inst(compiler, MVN | RD(reg) | imm1));
		return push_inst(compiler, BIC | RD(reg) | RN(reg) | imm2);
	}

	/* Load integer. */
	return push_inst_with_literal(compiler, EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1, reg, TMP_PC, 0), imm);
#else /* !SLJIT_CONFIG_ARM_V6 */
	FAIL_IF(push_inst(compiler, MOVW | RD(reg) | ((imm << 4) & 0xf0000) | (imm & 0xfff)));
	if (imm <= 0xffff)
		return SLJIT_SUCCESS;
	return push_inst(compiler, MOVT | RD(reg) | ((imm >> 12) & 0xf0000) | ((imm >> 16) & 0xfff));
#endif /* SLJIT_CONFIG_ARM_V6 */
}

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg,
	sljit_s32 arg, sljit_sw argw, sljit_s32 tmp_reg)
{
	sljit_uw imm, offset_reg, tmp;
	sljit_sw mask = IS_TYPE1_TRANSFER(flags) ? 0xfff : 0xff;
	sljit_sw sign = IS_TYPE1_TRANSFER(flags) ? 0x1000 : 0x100;

	SLJIT_ASSERT(arg & SLJIT_MEM);
	SLJIT_ASSERT((arg & REG_MASK) != tmp_reg || (arg == SLJIT_MEM1(tmp_reg) && argw >= -mask && argw <= mask));

	if (SLJIT_UNLIKELY(!(arg & REG_MASK))) {
		tmp = (sljit_uw)(argw & (sign | mask));
		tmp = (sljit_uw)((argw + (tmp <= (sljit_uw)sign ? 0 : sign)) & ~mask);

		FAIL_IF(load_immediate(compiler, tmp_reg, tmp));

		argw -= (sljit_sw)tmp;
		tmp = 1;

		if (argw < 0) {
			argw = -argw;
			tmp = 0;
		}

		return push_inst(compiler, EMIT_DATA_TRANSFER(flags, tmp, reg, tmp_reg,
			(mask == 0xff) ? TYPE2_TRANSFER_IMM(argw) : argw));
	}

	if (arg & OFFS_REG_MASK) {
		offset_reg = OFFS_REG(arg);
		arg &= REG_MASK;
		argw &= 0x3;

		if (argw != 0 && (mask == 0xff)) {
			FAIL_IF(push_inst(compiler, ADD | RD(tmp_reg) | RN(arg) | RM(offset_reg) | ((sljit_ins)argw << 7)));
			return push_inst(compiler, EMIT_DATA_TRANSFER(flags, 1, reg, tmp_reg, TYPE2_TRANSFER_IMM(0)));
		}

		/* Bit 25: RM is offset. */
		return push_inst(compiler, EMIT_DATA_TRANSFER(flags, 1, reg, arg,
			RM(offset_reg) | (mask == 0xff ? 0 : (1 << 25)) | ((sljit_ins)argw << 7)));
	}

	arg &= REG_MASK;

	if (argw > mask) {
		tmp = (sljit_uw)(argw & (sign | mask));
		tmp = (sljit_uw)((argw + (tmp <= (sljit_uw)sign ? 0 : sign)) & ~mask);
		imm = get_imm(tmp);

		if (imm) {
			FAIL_IF(push_inst(compiler, ADD | RD(tmp_reg) | RN(arg) | imm));
			argw -= (sljit_sw)tmp;
			arg = tmp_reg;

			SLJIT_ASSERT(argw >= -mask && argw <= mask);
		}
	} else if (argw < -mask) {
		tmp = (sljit_uw)(-argw & (sign | mask));
		tmp = (sljit_uw)((-argw + (tmp <= (sljit_uw)sign ? 0 : sign)) & ~mask);
		imm = get_imm(tmp);

		if (imm) {
			FAIL_IF(push_inst(compiler, SUB | RD(tmp_reg) | RN(arg) | imm));
			argw += (sljit_sw)tmp;
			arg = tmp_reg;

			SLJIT_ASSERT(argw >= -mask && argw <= mask);
		}
	}

	if (argw <= mask && argw >= -mask) {
		if (argw >= 0) {
			if (mask == 0xff)
				argw = TYPE2_TRANSFER_IMM(argw);
			return push_inst(compiler, EMIT_DATA_TRANSFER(flags, 1, reg, arg, argw));
		}

		argw = -argw;

		if (mask == 0xff)
			argw = TYPE2_TRANSFER_IMM(argw);

		return push_inst(compiler, EMIT_DATA_TRANSFER(flags, 0, reg, arg, argw));
	}

	FAIL_IF(load_immediate(compiler, tmp_reg, (sljit_uw)argw));
	return push_inst(compiler, EMIT_DATA_TRANSFER(flags, 1, reg, arg,
		RM(tmp_reg) | (mask == 0xff ? 0 : (1 << 25))));
}

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 inp_flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* src1 is reg or TMP_REG1
	   src2 is reg, TMP_REG2, or imm
	   result goes to TMP_REG2, so put result can use TMP_REG1. */

	/* We prefers register and simple consts. */
	sljit_s32 dst_reg;
	sljit_s32 src1_reg = 0;
	sljit_s32 src2_reg = 0;
	sljit_s32 flags = HAS_FLAGS(op) ? SET_FLAGS : 0;
	sljit_s32 neg_op = 0;
	sljit_u32 imm2;

	op = GET_OPCODE(op);

	if (flags & SET_FLAGS)
		inp_flags &= ~ALLOW_DOUBLE_IMM;

	if (dst == TMP_REG2)
		flags |= UNUSED_RETURN;

	SLJIT_ASSERT(!(inp_flags & ALLOW_INV_IMM) || (inp_flags & ALLOW_IMM));

	if (inp_flags & ALLOW_NEG_IMM) {
		switch (op) {
		case SLJIT_ADD:
			compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
			neg_op = SLJIT_SUB;
			break;
		case SLJIT_ADDC:
			compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
			neg_op = SLJIT_SUBC;
			break;
		case SLJIT_SUB:
			compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
			neg_op = SLJIT_ADD;
			break;
		case SLJIT_SUBC:
			compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
			neg_op = SLJIT_ADDC;
			break;
		}
	}

	do {
		if (!(inp_flags & ALLOW_IMM))
			break;

		if (src2 == SLJIT_IMM) {
			src2_reg = (sljit_s32)get_imm((sljit_uw)src2w);
			if (src2_reg)
				break;

			if (inp_flags & ALLOW_INV_IMM) {
				src2_reg = (sljit_s32)get_imm(~(sljit_uw)src2w);
				if (src2_reg) {
					flags |= INV_IMM;
					break;
				}
			}

			if (neg_op != 0) {
				src2_reg = (sljit_s32)get_imm((neg_op == SLJIT_ADD || neg_op == SLJIT_SUB) ? (sljit_uw)-src2w : ~(sljit_uw)src2w);
				if (src2_reg) {
					op = neg_op | GET_ALL_FLAGS(op);
					break;
				}
			}
		}

		if (src1 == SLJIT_IMM) {
			src2_reg = (sljit_s32)get_imm((sljit_uw)src1w);
			if (src2_reg) {
				flags |= ARGS_SWAPPED;
				src1 = src2;
				src1w = src2w;
				break;
			}

			if (inp_flags & ALLOW_INV_IMM) {
				src2_reg = (sljit_s32)get_imm(~(sljit_uw)src1w);
				if (src2_reg) {
					flags |= ARGS_SWAPPED | INV_IMM;
					src1 = src2;
					src1w = src2w;
					break;
				}
			}

			if (neg_op >= SLJIT_SUB) {
				/* Note: additive operation (commutative). */
				SLJIT_ASSERT(op == SLJIT_ADD || op == SLJIT_ADDC);

				src2_reg = (sljit_s32)get_imm((sljit_uw)-src1w);
				if (src2_reg) {
					src1 = src2;
					src1w = src2w;
					op = neg_op | GET_ALL_FLAGS(op);
					break;
				}
			}
		}
	} while(0);

	/* Source 1. */
	if (FAST_IS_REG(src1))
		src1_reg = src1;
	else if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags | LOAD_DATA, TMP_REG1, src1, src1w, TMP_REG1));
		src1_reg = TMP_REG1;
	} else if (!(inp_flags & ALLOW_DOUBLE_IMM) || src2_reg != 0 || op == SLJIT_SUB || op == SLJIT_SUBC) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)src1w));
		src1_reg = TMP_REG1;
	}

	/* Destination. */
	dst_reg = FAST_IS_REG(dst) ? dst : TMP_REG2;

	if (op <= SLJIT_MOV_P) {
		if (dst & SLJIT_MEM) {
			if (inp_flags & BYTE_SIZE)
				inp_flags &= ~SIGNED;

			if (FAST_IS_REG(src2))
				return emit_op_mem(compiler, inp_flags, src2, dst, dstw, TMP_REG2);
		}

		if (FAST_IS_REG(src2) && dst_reg != TMP_REG2)
			flags |= MOVE_REG_CONV;
	}

	/* Source 2. */
	if (src2_reg == 0) {
		src2_reg = (op <= SLJIT_MOV_P) ? dst_reg : TMP_REG2;

		if (FAST_IS_REG(src2))
			src2_reg = src2;
		else if (src2 & SLJIT_MEM)
			FAIL_IF(emit_op_mem(compiler, inp_flags | LOAD_DATA, src2_reg, src2, src2w, TMP_REG2));
		else if (!(inp_flags & ALLOW_DOUBLE_IMM))
			FAIL_IF(load_immediate(compiler, src2_reg, (sljit_uw)src2w));
		else {
			SLJIT_ASSERT(!(flags & SET_FLAGS));

			if (src1_reg == 0) {
				FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)src1w));
				src1_reg = TMP_REG1;
			}

			src2_reg = (sljit_s32)compute_imm((sljit_uw)src2w, &imm2);

			if (src2_reg == 0 && neg_op != 0) {
				src2_reg = (sljit_s32)compute_imm((sljit_uw)-src2w, &imm2);
				if (src2_reg != 0)
					op = neg_op;
			}

			if (src2_reg == 0) {
				FAIL_IF(load_immediate(compiler, TMP_REG2, (sljit_uw)src2w));
				src2_reg = TMP_REG2;
			} else {
				FAIL_IF(emit_single_op(compiler, op, flags, (sljit_uw)dst_reg, (sljit_uw)src1_reg, (sljit_uw)src2_reg));
				src1_reg = dst_reg;
				src2_reg = (sljit_s32)imm2;

				if (op == SLJIT_ADDC)
					op = SLJIT_ADD;
				else if (op == SLJIT_SUBC)
					op = SLJIT_SUB;
			}
		}
	}

	if (src1_reg == 0) {
		SLJIT_ASSERT((inp_flags & ALLOW_DOUBLE_IMM) && !(flags & SET_FLAGS));

		src1_reg = (sljit_s32)compute_imm((sljit_uw)src1w, &imm2);

		if (src1_reg == 0 && neg_op != 0) {
			src1_reg = (sljit_s32)compute_imm((sljit_uw)-src1w, &imm2);
			if (src1_reg != 0)
				op = neg_op;
		}

		if (src1_reg == 0) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)src1w));
			src1_reg = TMP_REG1;
		} else {
			FAIL_IF(emit_single_op(compiler, op, flags, (sljit_uw)dst_reg, (sljit_uw)src2_reg, (sljit_uw)src1_reg));
			src1_reg = dst_reg;
			src2_reg = (sljit_s32)imm2;

			if (op == SLJIT_ADDC)
				op = SLJIT_ADD;
		}
	}

	FAIL_IF(emit_single_op(compiler, op, flags, (sljit_uw)dst_reg, (sljit_uw)src1_reg, (sljit_uw)src2_reg));

	if (!(dst & SLJIT_MEM))
		return SLJIT_SUCCESS;

	return emit_op_mem(compiler, inp_flags, dst_reg, dst, dstw, TMP_REG1);
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
	sljit_uw saved_reg_list[3];
	sljit_sw saved_reg_count;

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
		return push_inst(compiler, (op == SLJIT_LMUL_UW ? UMULL : SMULL)
			| RN(SLJIT_R1) | RD(SLJIT_R0) | RM8(SLJIT_R0) | RM(SLJIT_R1));
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		SLJIT_COMPILE_ASSERT((SLJIT_DIVMOD_UW & 0x2) == 0 && SLJIT_DIV_UW - 0x2 == SLJIT_DIVMOD_UW, bad_div_opcode_assignments);
		SLJIT_ASSERT(reg_map[2] == 1 && reg_map[3] == 2 && reg_map[4] == 3);

		saved_reg_count = 0;
		if (compiler->scratches >= 4)
			saved_reg_list[saved_reg_count++] = 3;
		if (compiler->scratches >= 3)
			saved_reg_list[saved_reg_count++] = 2;
		if (op >= SLJIT_DIV_UW)
			saved_reg_list[saved_reg_count++] = 1;

		if (saved_reg_count > 0) {
			FAIL_IF(push_inst(compiler, STR | 0x2d0000 | (saved_reg_count >= 3 ? 16 : 8)
						| (saved_reg_list[0] << 12) /* str rX, [sp, #-8/-16]! */));
			if (saved_reg_count >= 2) {
				SLJIT_ASSERT(saved_reg_list[1] < 8);
				FAIL_IF(push_inst(compiler, STR | 0x8d0004 | (saved_reg_list[1] << 12) /* str rX, [sp, #4] */));
			}
			if (saved_reg_count >= 3) {
				SLJIT_ASSERT(saved_reg_list[2] < 8);
				FAIL_IF(push_inst(compiler, STR | 0x8d0008 | (saved_reg_list[2] << 12) /* str rX, [sp, #8] */));
			}
		}

#if defined(__GNUC__)
		FAIL_IF(sljit_emit_ijump(compiler, SLJIT_FAST_CALL, SLJIT_IMM,
			((op | 0x2) == SLJIT_DIV_UW ? SLJIT_FUNC_ADDR(__aeabi_uidivmod) : SLJIT_FUNC_ADDR(__aeabi_idivmod))));
#else
#error "Software divmod functions are needed"
#endif

		if (saved_reg_count > 0) {
			if (saved_reg_count >= 3) {
				SLJIT_ASSERT(saved_reg_list[2] < 8);
				FAIL_IF(push_inst(compiler, LDR | 0x8d0008 | (saved_reg_list[2] << 12) /* ldr rX, [sp, #8] */));
			}
			if (saved_reg_count >= 2) {
				SLJIT_ASSERT(saved_reg_list[1] < 8);
				FAIL_IF(push_inst(compiler, LDR | 0x8d0004 | (saved_reg_list[1] << 12) /* ldr rX, [sp, #4] */));
			}
			return push_inst(compiler, (LDR ^ (1 << 24)) | 0x8d0000 | (sljit_ins)(saved_reg_count >= 3 ? 16 : 8)
						| (saved_reg_list[0] << 12) /* ldr rX, [sp], #8/16 */);
		}
		return SLJIT_SUCCESS;
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
	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, ALLOW_ANY_IMM, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_U8:
		return emit_op(compiler, SLJIT_MOV_U8, ALLOW_ANY_IMM | BYTE_SIZE, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, SLJIT_MOV_S8, ALLOW_ANY_IMM | SIGNED | BYTE_SIZE, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, SLJIT_MOV_U16, ALLOW_ANY_IMM | HALF_SIZE, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, SLJIT_MOV_S16, ALLOW_ANY_IMM | SIGNED | HALF_SIZE, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_CLZ:
	case SLJIT_CTZ:
	case SLJIT_REV:
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		return emit_op(compiler, op, 0, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		return emit_op(compiler, op, HALF_SIZE, dst, dstw, TMP_REG1, 0, src, srcw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 inp_flags;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
	case SLJIT_SUB:
	case SLJIT_SUBC:
		return emit_op(compiler, op, ALLOW_IMM | ALLOW_NEG_IMM | ALLOW_DOUBLE_IMM, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_OR:
		return emit_op(compiler, op, ALLOW_IMM | ALLOW_DOUBLE_IMM, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_XOR:
		inp_flags = ALLOW_IMM | ALLOW_DOUBLE_IMM;
		if ((src1 == SLJIT_IMM && src1w == -1) || (src2 == SLJIT_IMM && src2w == -1)) {
			inp_flags |= ALLOW_INV_IMM;
		}
		return emit_op(compiler, op, inp_flags, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		return emit_op(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
		return emit_op(compiler, op, ALLOW_ANY_IMM, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
	case SLJIT_ROTL:
	case SLJIT_ROTR:
		if (src2 == SLJIT_IMM) {
			compiler->shift_imm = src2w & 0x1f;
			return emit_op(compiler, op, 0, dst, dstw, TMP_REG1, 0, src1, src1w);
		} else {
			compiler->shift_imm = 0x20;
			return emit_op(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w);
		}
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
	return sljit_emit_op2(compiler, op, TMP_REG2, 0, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_s32 is_left;

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, dst_reg, src1_reg, src2_reg, src3, src3w));

	op = GET_OPCODE(op);
	is_left = (op == SLJIT_SHL || op == SLJIT_MSHL);

	if (src1_reg == src2_reg) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, is_left ? SLJIT_ROTL : SLJIT_ROTR, dst_reg, 0, src1_reg, 0, src3, src3w);
	}

	ADJUST_LOCAL_OFFSET(src3, src3w);

	/* Shift type of ROR is 3. */
	if (src3 == SLJIT_IMM) {
		src3w &= 0x1f;

		if (src3w == 0)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, MOV | RD(dst_reg) | RM(src1_reg) | ((sljit_ins)(is_left ? 0 : 1) << 5) | ((sljit_ins)src3w << 7)));
		src3w = (src3w ^ 0x1f) + 1;
		return push_inst(compiler, ORR | RD(dst_reg) | RN(dst_reg) | RM(src2_reg) | ((sljit_ins)(is_left ? 1 : 0) << 5) | ((sljit_ins)src3w << 7));
	}

	if (src3 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG2, src3, src3w, TMP_REG2));
		src3 = TMP_REG2;
	}

	if (op == SLJIT_MSHL || op == SLJIT_MLSHR || dst_reg == src3) {
		FAIL_IF(push_inst(compiler, AND | SRC2_IMM | RD(TMP_REG2) | RN(src3) | 0x1f));
		src3 = TMP_REG2;
	}

	FAIL_IF(push_inst(compiler, MOV | RD(dst_reg) | RM8(src3) | ((sljit_ins)(is_left ? 0 : 1) << 5) | 0x10 | RM(src1_reg)));
	FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | RM(src2_reg) | ((sljit_ins)(is_left ? 1 : 0) << 5) | (1 << 7)));
	FAIL_IF(push_inst(compiler, EOR | SRC2_IMM | RD(TMP_REG2) | RN(src3) | 0x1f));
	return push_inst(compiler, ORR | RD(dst_reg) | RN(dst_reg) | RM8(TMP_REG2) | ((sljit_ins)(is_left ? 1 : 0) << 5) | 0x10 | RM(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		SLJIT_ASSERT(reg_map[TMP_REG2] == 14);

		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG2) | RM(src)));
		else
			FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG2, src, srcw, TMP_REG1));

		return push_inst(compiler, BX | RM(TMP_REG2));
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		SLJIT_ASSERT(src & SLJIT_MEM);
		return emit_op_mem(compiler, PRELOAD | LOAD_DATA, TMP_PC, src, srcw, TMP_REG1);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 size, dst_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_dst(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	switch (op) {
	case SLJIT_FAST_ENTER:
		SLJIT_ASSERT(reg_map[TMP_REG2] == 14);

		if (FAST_IS_REG(dst))
			return push_inst(compiler, MOV | RD(dst) | RM(TMP_REG2));
		break;
	case SLJIT_GET_RETURN_ADDRESS:
		size = GET_SAVED_REGISTERS_SIZE(compiler->scratches, compiler->saveds - SLJIT_KEPT_SAVEDS_COUNT(compiler->options), 0);

		if (compiler->fsaveds > 0 || compiler->fscratches >= SLJIT_FIRST_SAVED_FLOAT_REG) {
			/* The size of pc is not added above. */
			if ((size & SSIZE_OF(sw)) == 0)
				size += SSIZE_OF(sw);

			size += GET_SAVED_FLOAT_REGISTERS_SIZE(compiler->fscratches, compiler->fsaveds, f64);
		}

		SLJIT_ASSERT(((compiler->local_size + size + SSIZE_OF(sw)) & 0x7) == 0);

		dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, dst_r, SLJIT_MEM1(SLJIT_SP), compiler->local_size + size, TMP_REG1));
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, WORD_SIZE, TMP_REG2, dst, dstw, TMP_REG1);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(type, reg));

	if (type == SLJIT_GP_REGISTER)
		return reg_map[reg];

	if (type == SLJIT_FLOAT_REGISTER || type == SLJIT_SIMD_REG_64)
		return freg_map[reg];

	if (type != SLJIT_SIMD_REG_128)
		return freg_map[reg] & ~0x1;

	return -1;
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

#define FPU_LOAD (1 << 20)
#define EMIT_FPU_DATA_TRANSFER(inst, add, base, freg, offs) \
	((inst) | (sljit_ins)((add) << 23) | RN(base) | VD(freg) | (sljit_ins)(offs))

static sljit_s32 emit_fop_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_uw imm;
	sljit_ins inst = VSTR_F32 | (flags & (SLJIT_32 | FPU_LOAD));

	SLJIT_ASSERT(arg & SLJIT_MEM);
	arg &= ~SLJIT_MEM;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG2) | RN(arg & REG_MASK) | RM(OFFS_REG(arg)) | (((sljit_ins)argw & 0x3) << 7)));
		arg = TMP_REG2;
		argw = 0;
	}

	/* Fast loads and stores. */
	if (arg) {
		if (!(argw & ~0x3fc))
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, arg & REG_MASK, reg, argw >> 2));
		if (!(-argw & ~0x3fc))
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 0, arg & REG_MASK, reg, (-argw) >> 2));

		imm = get_imm((sljit_uw)argw & ~(sljit_uw)0x3fc);
		if (imm) {
			FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG2) | RN(arg & REG_MASK) | imm));
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG2, reg, (argw & 0x3fc) >> 2));
		}
		imm = get_imm((sljit_uw)-argw & ~(sljit_uw)0x3fc);
		if (imm) {
			argw = -argw;
			FAIL_IF(push_inst(compiler, SUB | RD(TMP_REG2) | RN(arg & REG_MASK) | imm));
			return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 0, TMP_REG2, reg, (argw & 0x3fc) >> 2));
		}
	}

	if (arg) {
		FAIL_IF(load_immediate(compiler, TMP_REG2, (sljit_uw)argw));
		FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG2) | RN(arg & REG_MASK) | RM(TMP_REG2)));
	}
	else
		FAIL_IF(load_immediate(compiler, TMP_REG2, (sljit_uw)argw));

	return push_inst(compiler, EMIT_FPU_DATA_TRANSFER(inst, 1, TMP_REG2, reg, 0));
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	op ^= SLJIT_32;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) | FPU_LOAD, TMP_FREG1, src, srcw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCVT_S32_F32, op & SLJIT_32, TMP_FREG1, src, 0)));

	if (FAST_IS_REG(dst))
		return push_inst(compiler, VMOV | (1 << 20) | RD(dst) | VN(TMP_FREG1));

	/* Store the integer value from a VFP register. */
	return emit_fop_mem(compiler, 0, TMP_FREG1, dst, dstw);
}

static sljit_s32 sljit_emit_fop1_conv_f64_from_w(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, VMOV | RD(src) | VN(TMP_FREG1)));
	else if (src & SLJIT_MEM) {
		/* Load the integer value into a VFP register. */
		FAIL_IF(emit_fop_mem(compiler, FPU_LOAD, TMP_FREG1, src, srcw));
	}
	else {
		FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)srcw));
		FAIL_IF(push_inst(compiler, VMOV | RD(TMP_REG1) | VN(TMP_FREG1)));
	}

	FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(ins, ins & SLJIT_32, dst_r, TMP_FREG1, 0)));

	if (dst & SLJIT_MEM)
		return emit_fop_mem(compiler, (ins & SLJIT_32), TMP_FREG1, dst, dstw);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	return sljit_emit_fop1_conv_f64_from_w(compiler, VCVT_F32_S32 | (~op & SLJIT_32), dst, dstw, src, srcw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_uw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	return sljit_emit_fop1_conv_f64_from_w(compiler, VCVT_F32_U32 | (~op & SLJIT_32), dst, dstw, src, srcw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	op ^= SLJIT_32;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) | FPU_LOAD, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) | FPU_LOAD, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCMP_F32, op & SLJIT_32, src1, src2, 0)));
	FAIL_IF(push_inst(compiler, VMRS));

	if (GET_FLAG_TYPE(op) != SLJIT_UNORDERED_OR_EQUAL)
		return SLJIT_SUCCESS;

	return push_inst(compiler, (CMP - CONDITIONAL) | (0x60000000 /* VS */) | SET_FLAGS | RN(TMP_REG1) | RM(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();

	SLJIT_COMPILE_ASSERT((SLJIT_32 == 0x100), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (GET_OPCODE(op) != SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_32;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) | FPU_LOAD, dst_r, src, srcw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMOV_F32, op & SLJIT_32, dst_r, src, 0)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VNEG_F32, op & SLJIT_32, dst_r, src, 0)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VABS_F32, op & SLJIT_32, dst_r, src, 0)));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VCVT_F64_F32, op & SLJIT_32, dst_r, src, 0)));
		op ^= SLJIT_32;
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_fop_mem(compiler, (op & SLJIT_32), dst_r, dst, dstw);
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

	op ^= SLJIT_32;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) | FPU_LOAD, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) | FPU_LOAD, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VADD_F32, op & SLJIT_32, dst_r, src2, src1)));
		break;
	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VSUB_F32, op & SLJIT_32, dst_r, src2, src1)));
		break;
	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMUL_F32, op & SLJIT_32, dst_r, src2, src1)));
		break;
	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VDIV_F32, op & SLJIT_32, dst_r, src2, src1)));
		break;
	case SLJIT_COPYSIGN_F64:
		FAIL_IF(push_inst(compiler, VMOV | (1 << 20) | VN(src2) | RD(TMP_REG1) | ((op & SLJIT_32) ? (1 << 7) : 0)));
		FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VABS_F32, op & SLJIT_32, dst_r, src1, 0)));
		FAIL_IF(push_inst(compiler, CMP | SET_FLAGS | RN(TMP_REG1) | SRC2_IMM | 0));
		return push_inst(compiler, EMIT_FPU_OPERATION((VNEG_F32 & ~COND_MASK) | 0xb0000000, op & SLJIT_32, dst_r, dst_r, 0));
	}

	if (dst_r == TMP_FREG1)
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32), TMP_FREG1, dst, dstw));

	return SLJIT_SUCCESS;
}

#undef EMIT_FPU_DATA_TRANSFER

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset32(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f32 value)
{
#if defined(__ARM_NEON) && __ARM_NEON
	sljit_u32 exp;
	sljit_ins ins;
#endif /* NEON */
	union {
		sljit_u32 imm;
		sljit_f32 value;
	} u;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fset32(compiler, freg, value));

	u.value = value;

#if defined(__ARM_NEON) && __ARM_NEON
	if ((u.imm << (32 - 19)) == 0) {
		exp = (u.imm >> (23 + 2)) & 0x3f;

		if (exp == 0x20 || exp == 0x1f) {
			ins = ((u.imm >> 24) & 0x80) | ((u.imm >> 19) & 0x7f);
			return push_inst(compiler, (VMOV_F32 ^ (1 << 6)) | ((ins & 0xf0) << 12) | VD(freg) | (ins & 0xf));
		}
	}
#endif /* NEON */

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm));
	return push_inst(compiler, VMOV | VN(freg) | RD(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset64(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f64 value)
{
#if defined(__ARM_NEON) && __ARM_NEON
	sljit_u32 exp;
	sljit_ins ins;
#endif /* NEON */
	union {
		sljit_u32 imm[2];
		sljit_f64 value;
	} u;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fset64(compiler, freg, value));

	u.value = value;

#if defined(__ARM_NEON) && __ARM_NEON
	if (u.imm[0] == 0 && (u.imm[1] << (64 - 48)) == 0) {
		exp = (u.imm[1] >> ((52 - 32) + 2)) & 0x1ff;

		if (exp == 0x100 || exp == 0xff) {
			ins = ((u.imm[1] >> (56 - 32)) & 0x80) | ((u.imm[1] >> (48 - 32)) & 0x7f);
			return push_inst(compiler, (VMOV_F32 ^ (1 << 6)) | (1 << 8) | ((ins & 0xf0) << 12) | VD(freg) | (ins & 0xf));
		}
	}
#endif /* NEON */

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm[0]));
	if (u.imm[0] == u.imm[1])
		return push_inst(compiler, VMOV2 | RN(TMP_REG1) | RD(TMP_REG1) | VM(freg));

	FAIL_IF(load_immediate(compiler, TMP_REG2, u.imm[1]));
	return push_inst(compiler, VMOV2 | RN(TMP_REG2) | RD(TMP_REG1) | VM(freg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
	sljit_s32 reg2;
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fcopy(compiler, op, freg, reg));

	if (reg & REG_PAIR_MASK) {
		reg2 = REG_PAIR_SECOND(reg);
		reg = REG_PAIR_FIRST(reg);

		inst = VMOV2 | RN(reg) | RD(reg2) | VM(freg);
	} else {
		inst = VMOV | VN(freg) | RD(reg);

		if (!(op & SLJIT_32))
			inst |= 1 << 7;
	}

	if (GET_OPCODE(op) == SLJIT_COPY_FROM_F64)
		inst |= 1 << 20;

	return push_inst(compiler, inst);
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

static sljit_ins get_cc(struct sljit_compiler *compiler, sljit_s32 type)
{
	switch (type) {
	case SLJIT_EQUAL:
	case SLJIT_ATOMIC_STORED:
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		return 0x00000000;

	case SLJIT_NOT_EQUAL:
	case SLJIT_ATOMIC_NOT_STORED:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
		return 0x10000000;

	case SLJIT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_ADD)
			return 0x20000000;
		/* fallthrough */

	case SLJIT_LESS:
		return 0x30000000;

	case SLJIT_NOT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_ADD)
			return 0x30000000;
		/* fallthrough */

	case SLJIT_GREATER_EQUAL:
		return 0x20000000;

	case SLJIT_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
		return 0x80000000;

	case SLJIT_LESS_EQUAL:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
		return 0x90000000;

	case SLJIT_SIG_LESS:
	case SLJIT_UNORDERED_OR_LESS:
		return 0xb0000000;

	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
		return 0xa0000000;

	case SLJIT_SIG_GREATER:
	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
		return 0xc0000000;

	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		return 0xd0000000;

	case SLJIT_OVERFLOW:
		if (!(compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)))
			return 0x10000000;
		/* fallthrough */

	case SLJIT_UNORDERED:
		return 0x60000000;

	case SLJIT_NOT_OVERFLOW:
		if (!(compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)))
			return 0x00000000;
		/* fallthrough */

	case SLJIT_ORDERED:
		return 0x70000000;

	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
		return 0x40000000;

	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		return 0x50000000;

	default:
		SLJIT_ASSERT(type >= SLJIT_JUMP && type <= SLJIT_CALL_REG_ARG);
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

	SLJIT_ASSERT(reg_map[TMP_REG1] != 14);

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	if (type >= SLJIT_FAST_CALL)
		PTR_FAIL_IF(prepare_blx(compiler));
	PTR_FAIL_IF(push_inst_with_unique_literal(compiler, ((EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1,
		type <= SLJIT_JUMP ? TMP_PC : TMP_REG1, TMP_PC, 0)) & ~COND_MASK) | get_cc(compiler, type), 0));

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
#else /* !SLJIT_CONFIG_ARM_V6 */
	if (type >= SLJIT_FAST_CALL)
		jump->flags |= IS_BL;
	PTR_FAIL_IF(emit_imm(compiler, TMP_REG1, 0));
	PTR_FAIL_IF(push_inst(compiler, (((type <= SLJIT_JUMP ? BX : BLX) | RM(TMP_REG1)) & ~COND_MASK) | get_cc(compiler, type)));
	jump->addr = compiler->size;
#endif /* SLJIT_CONFIG_ARM_V6 */
	return jump;
}

#ifdef __SOFTFP__

static sljit_s32 softfloat_call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_s32 *src, sljit_u32 *extra_space)
{
	sljit_u32 is_tail_call = *extra_space & SLJIT_CALL_RETURN;
	sljit_u32 offset = 0;
	sljit_u32 word_arg_offset = 0;
	sljit_u32 src_offset = 4 * sizeof(sljit_sw);
	sljit_u32 float_arg_count = 0;
	sljit_s32 types = 0;
	sljit_u8 offsets[4];
	sljit_u8 *offset_ptr = offsets;

	if (src && FAST_IS_REG(*src))
		src_offset = (sljit_u32)reg_map[*src] * sizeof(sljit_sw);

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		types = (types << SLJIT_ARG_SHIFT) | (arg_types & SLJIT_ARG_MASK);

		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			if (offset & 0x7)
				offset += sizeof(sljit_sw);
			*offset_ptr++ = (sljit_u8)offset;
			offset += sizeof(sljit_f64);
			float_arg_count++;
			break;
		case SLJIT_ARG_TYPE_F32:
			*offset_ptr++ = (sljit_u8)offset;
			offset += sizeof(sljit_f32);
			float_arg_count++;
			break;
		default:
			*offset_ptr++ = (sljit_u8)offset;
			offset += sizeof(sljit_sw);
			word_arg_offset += sizeof(sljit_sw);
			break;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (offset > 4 * sizeof(sljit_sw) && (!is_tail_call || offset > compiler->args_size)) {
		/* Keep lr register on the stack. */
		if (is_tail_call)
			offset += sizeof(sljit_sw);

		offset = ((offset - 4 * sizeof(sljit_sw)) + 0x7) & ~(sljit_u32)0x7;

		*extra_space = offset;

		if (is_tail_call)
			FAIL_IF(emit_stack_frame_release(compiler, (sljit_s32)offset));
		else
			FAIL_IF(push_inst(compiler, SUB | RD(SLJIT_SP) | RN(SLJIT_SP) | SRC2_IMM | offset));
	} else {
		if (is_tail_call)
			FAIL_IF(emit_stack_frame_release(compiler, -1));
		*extra_space = 0;
	}

	/* Process arguments in reversed direction. */
	while (types) {
		switch (types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			float_arg_count--;
			offset = *(--offset_ptr);

			SLJIT_ASSERT((offset & 0x7) == 0);

			if (offset < 4 * sizeof(sljit_sw)) {
				if (src_offset == offset || src_offset == offset + sizeof(sljit_sw)) {
					FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | (src_offset >> 2)));
					*src = TMP_REG1;
				}
				FAIL_IF(push_inst(compiler, VMOV2 | 0x100000 | (offset << 10) | ((offset + sizeof(sljit_sw)) << 14) | float_arg_count));
			} else
				FAIL_IF(push_inst(compiler, VSTR_F32 | 0x800100 | RN(SLJIT_SP)
						| (float_arg_count << 12) | ((offset - 4 * sizeof(sljit_sw)) >> 2)));
			break;
		case SLJIT_ARG_TYPE_F32:
			float_arg_count--;
			offset = *(--offset_ptr);

			if (offset < 4 * sizeof(sljit_sw)) {
				if (src_offset == offset) {
					FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | (src_offset >> 2)));
					*src = TMP_REG1;
				}
				FAIL_IF(push_inst(compiler, VMOV | 0x100000 | (float_arg_count << 16) | (offset << 10)));
			} else
				FAIL_IF(push_inst(compiler, VSTR_F32 | 0x800000 | RN(SLJIT_SP)
						| (float_arg_count << 12) | ((offset - 4 * sizeof(sljit_sw)) >> 2)));
			break;
		default:
			word_arg_offset -= sizeof(sljit_sw);
			offset = *(--offset_ptr);

			SLJIT_ASSERT(offset >= word_arg_offset);

			if (offset != word_arg_offset) {
				if (offset < 4 * sizeof(sljit_sw)) {
					if (src_offset == offset) {
						FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | (src_offset >> 2)));
						*src = TMP_REG1;
					}
					else if (src_offset == word_arg_offset) {
						*src = (sljit_s32)(SLJIT_R0 + (offset >> 2));
						src_offset = offset;
					}
					FAIL_IF(push_inst(compiler, MOV | (offset << 10) | (word_arg_offset >> 2)));
				} else
					FAIL_IF(push_inst(compiler, STR | 0x800000 | RN(SLJIT_SP) | (word_arg_offset << 10) | (offset - 4 * sizeof(sljit_sw))));
			}
			break;
		}

		types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 softfloat_post_call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types)
{
	if ((arg_types & SLJIT_ARG_MASK) == SLJIT_ARG_TYPE_F64)
		FAIL_IF(push_inst(compiler, VMOV2 | (1 << 16) | (0 << 12) | 0));
	if ((arg_types & SLJIT_ARG_MASK) == SLJIT_ARG_TYPE_F32)
		FAIL_IF(push_inst(compiler, VMOV | (0 << 16) | (0 << 12)));

	return SLJIT_SUCCESS;
}

#else /* !__SOFTFP__ */

static sljit_s32 hardfloat_call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types)
{
	sljit_u32 offset = SLJIT_FR0;
	sljit_u32 new_offset = SLJIT_FR0;
	sljit_u32 f32_offset = 0;

	/* Remove return value. */
	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			if (offset != new_offset)
				FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMOV_F32,
					SLJIT_32, new_offset, offset, 0)));

			new_offset++;
			offset++;
			break;
		case SLJIT_ARG_TYPE_F32:
			if (f32_offset != 0) {
				FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMOV_F32,
					0x400000, f32_offset, offset, 0)));
				f32_offset = 0;
			} else {
				if (offset != new_offset)
					FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMOV_F32,
						0, new_offset, offset, 0)));
				f32_offset = new_offset;
				new_offset++;
			}
			offset++;
			break;
		}
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

#endif /* __SOFTFP__ */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
#ifdef __SOFTFP__
	struct sljit_jump *jump;
	sljit_u32 extra_space = (sljit_u32)type;
#endif

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

#ifdef __SOFTFP__
	if ((type & 0xff) != SLJIT_CALL_REG_ARG) {
		PTR_FAIL_IF(softfloat_call_with_args(compiler, arg_types, NULL, &extra_space));
		SLJIT_ASSERT((extra_space & 0x7) == 0);

		if ((type & SLJIT_CALL_RETURN) && extra_space == 0)
			type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);

		SLJIT_SKIP_CHECKS(compiler);
		jump = sljit_emit_jump(compiler, type);
		PTR_FAIL_IF(jump == NULL);

		if (extra_space > 0) {
			if (type & SLJIT_CALL_RETURN)
				PTR_FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1,
					TMP_REG2, SLJIT_SP, extra_space - sizeof(sljit_sw))));

			PTR_FAIL_IF(push_inst(compiler, ADD | RD(SLJIT_SP) | RN(SLJIT_SP) | SRC2_IMM | extra_space));

			if (type & SLJIT_CALL_RETURN) {
				PTR_FAIL_IF(push_inst(compiler, BX | RM(TMP_REG2)));
				return jump;
			}
		}

		SLJIT_ASSERT(!(type & SLJIT_CALL_RETURN));
		PTR_FAIL_IF(softfloat_post_call_with_args(compiler, arg_types));
		return jump;
	}
#endif /* __SOFTFP__ */

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler, -1));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

#ifndef __SOFTFP__
	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		PTR_FAIL_IF(hardfloat_call_with_args(compiler, arg_types));
#endif /* !__SOFTFP__ */

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	SLJIT_ASSERT(reg_map[TMP_REG1] != 14);

	if (src != SLJIT_IMM) {
		if (FAST_IS_REG(src)) {
			SLJIT_ASSERT(reg_map[src] != 14);
			return push_inst(compiler, (type <= SLJIT_JUMP ? BX : BLX) | RM(src));
		}

		SLJIT_ASSERT(src & SLJIT_MEM);
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG1, src, srcw, TMP_REG1));
		return push_inst(compiler, (type <= SLJIT_JUMP ? BX : BLX) | RM(TMP_REG1));
	}

	/* These jumps are converted to jump/call instructions when possible. */
	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	FAIL_IF(!jump);
	set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_BL : 0));
	jump->u.target = (sljit_uw)srcw;

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	if (type >= SLJIT_FAST_CALL)
		FAIL_IF(prepare_blx(compiler));
	FAIL_IF(push_inst_with_unique_literal(compiler, EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1, type <= SLJIT_JUMP ? TMP_PC : TMP_REG1, TMP_PC, 0), 0));
	if (type >= SLJIT_FAST_CALL)
		FAIL_IF(emit_blx(compiler));
#else /* !SLJIT_CONFIG_ARM_V6 */
	FAIL_IF(emit_imm(compiler, TMP_REG1, 0));
	FAIL_IF(push_inst(compiler, (type <= SLJIT_JUMP ? BX : BLX) | RM(TMP_REG1)));
#endif /* SLJIT_CONFIG_ARM_V6 */
	jump->addr = compiler->size;
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
#ifdef __SOFTFP__
	sljit_u32 extra_space = (sljit_u32)type;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG1, src, srcw, TMP_REG1));
		src = TMP_REG1;
	}

	if ((type & SLJIT_CALL_RETURN) && (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options)))) {
		FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | RM(src)));
		src = TMP_REG1;
	}

#ifdef __SOFTFP__
	if ((type & 0xff) != SLJIT_CALL_REG_ARG) {
		FAIL_IF(softfloat_call_with_args(compiler, arg_types, &src, &extra_space));
		SLJIT_ASSERT((extra_space & 0x7) == 0);

		if ((type & SLJIT_CALL_RETURN) && extra_space == 0)
			type = SLJIT_JUMP;

		SLJIT_SKIP_CHECKS(compiler);
		FAIL_IF(sljit_emit_ijump(compiler, type, src, srcw));

		if (extra_space > 0) {
			if (type & SLJIT_CALL_RETURN)
				FAIL_IF(push_inst(compiler, EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1,
					TMP_REG2, SLJIT_SP, extra_space - sizeof(sljit_sw))));

			FAIL_IF(push_inst(compiler, ADD | RD(SLJIT_SP) | RN(SLJIT_SP) | SRC2_IMM | extra_space));

			if (type & SLJIT_CALL_RETURN)
				return push_inst(compiler, BX | RM(TMP_REG2));
		}

		SLJIT_ASSERT(!(type & SLJIT_CALL_RETURN));
		return softfloat_post_call_with_args(compiler, arg_types);
	}
#endif /* __SOFTFP__ */

	if (type & SLJIT_CALL_RETURN) {
		FAIL_IF(emit_stack_frame_release(compiler, -1));
		type = SLJIT_JUMP;
	}

#ifndef __SOFTFP__
	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		FAIL_IF(hardfloat_call_with_args(compiler, arg_types));
#endif /* !__SOFTFP__ */

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, type, src, srcw);
}

#ifdef __SOFTFP__

static SLJIT_INLINE sljit_s32 emit_fmov_before_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	if (compiler->options & SLJIT_ENTER_REG_ARG) {
		if (src == SLJIT_FR0)
			return SLJIT_SUCCESS;

		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_fop1(compiler, op, SLJIT_RETURN_FREG, 0, src, srcw);
	}

	if (FAST_IS_REG(src)) {
		if (op & SLJIT_32)
			return push_inst(compiler, VMOV | (1 << 20) | RD(SLJIT_R0) | VN(src));
		return push_inst(compiler, VMOV2 | (1 << 20) | RD(SLJIT_R0) | RN(SLJIT_R1) | VM(src));
	}

	SLJIT_SKIP_CHECKS(compiler);

	if (op & SLJIT_32)
		return sljit_emit_op1(compiler, SLJIT_MOV, SLJIT_R0, 0, src, srcw);
	return sljit_emit_mem(compiler, SLJIT_MOV, SLJIT_REG_PAIR(SLJIT_R0, SLJIT_R1), src, srcw);
}

#endif /* __SOFTFP__ */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_s32 dst_reg, flags = GET_ALL_FLAGS(op);
	sljit_ins cc, ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	cc = get_cc(compiler, type);
	dst_reg = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (op < SLJIT_ADD) {
		FAIL_IF(push_inst(compiler, MOV | RD(dst_reg) | SRC2_IMM | 0));
		FAIL_IF(push_inst(compiler, ((MOV | RD(dst_reg) | SRC2_IMM | 1) & ~COND_MASK) | cc));
		if (dst & SLJIT_MEM)
			return emit_op_mem(compiler, WORD_SIZE, TMP_REG1, dst, dstw, TMP_REG2);
		return SLJIT_SUCCESS;
	}

	ins = (op == SLJIT_AND ? AND : (op == SLJIT_OR ? ORR : EOR));

	if (dst & SLJIT_MEM)
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG1, dst, dstw, TMP_REG2));

	FAIL_IF(push_inst(compiler, ((ins | RD(dst_reg) | RN(dst_reg) | SRC2_IMM | 1) & ~COND_MASK) | cc));

	if (op == SLJIT_AND)
		FAIL_IF(push_inst(compiler, ((ins | RD(dst_reg) | RN(dst_reg) | SRC2_IMM | 0) & ~COND_MASK) | (cc ^ 0x10000000)));

	if (dst & SLJIT_MEM)
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG1, dst, dstw, TMP_REG2));

	if (flags & SLJIT_SET_Z)
		return push_inst(compiler, MOV | SET_FLAGS | RD(TMP_REG2) | RM(dst_reg));
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
	sljit_ins cc, tmp;

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (src2_reg != dst_reg && src1 == dst_reg) {
		src1 = src2_reg;
		src1w = 0;
		src2_reg = dst_reg;
		type ^= 0x1;
	}

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, (src2_reg != dst_reg) ? dst_reg : TMP_REG1, src1, src1w, TMP_REG2));

		if (src2_reg != dst_reg) {
			src1 = src2_reg;
			src1w = 0;
			type ^= 0x1;
		} else {
			src1 = TMP_REG1;
			src1w = 0;
		}
	} else if (dst_reg != src2_reg)
		FAIL_IF(push_inst(compiler, MOV | RD(dst_reg) | RM(src2_reg)));

	cc = get_cc(compiler, type & ~SLJIT_32);

	if (SLJIT_UNLIKELY(src1 == SLJIT_IMM)) {
		tmp = get_imm((sljit_uw)src1w);
		if (tmp)
			return push_inst(compiler, ((MOV | RD(dst_reg) | tmp) & ~COND_MASK) | cc);

		tmp = get_imm(~(sljit_uw)src1w);
		if (tmp)
			return push_inst(compiler, ((MVN | RD(dst_reg) | tmp) & ~COND_MASK) | cc);

#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
		tmp = (sljit_ins)src1w;
		FAIL_IF(push_inst(compiler, (MOVW & ~COND_MASK) | cc | RD(dst_reg) | ((tmp << 4) & 0xf0000) | (tmp & 0xfff)));
		if (tmp <= 0xffff)
			return SLJIT_SUCCESS;
		return push_inst(compiler, (MOVT & ~COND_MASK) | cc | RD(dst_reg) | ((tmp >> 12) & 0xf0000) | ((tmp >> 16) & 0xfff));
#else /* !SLJIT_CONFIG_ARM_V7 */
		FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)src1w));
		src1 = TMP_REG1;
#endif /* SLJIT_CONFIG_ARM_V7 */
	}

	return push_inst(compiler, ((MOV | RD(dst_reg) | RM(src1)) & ~COND_MASK) | cc);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
	sljit_ins cc;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	type ^= SLJIT_32;

	if (dst_freg != src2_freg) {
		if (dst_freg == src1) {
			src1 = src2_freg;
			src1w = 0;
			type ^= 0x1;
		} else
			FAIL_IF(push_inst(compiler, EMIT_FPU_OPERATION(VMOV_F32, (type & SLJIT_32), dst_freg, src2_freg, 0)));
	}

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (type & SLJIT_32) | FPU_LOAD, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	cc = get_cc(compiler, type & ~SLJIT_32);
	return push_inst(compiler, EMIT_FPU_OPERATION((VMOV_F32 & ~COND_MASK) | cc, (type & SLJIT_32), dst_freg, src1, 0));
}

#undef EMIT_FPU_OPERATION

static sljit_s32 update_mem_addr(struct sljit_compiler *compiler, sljit_s32 *mem, sljit_sw *memw, sljit_s32 max_offset)
{
	sljit_s32 arg = *mem;
	sljit_sw argw = *memw;
	sljit_uw imm, tmp;
	sljit_sw mask = 0xfff;
	sljit_sw sign = 0x1000;

	SLJIT_ASSERT(max_offset >= 0xf00);

	*mem = TMP_REG1;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		*memw = 0;
		return push_inst(compiler, ADD | RD(TMP_REG1) | RN(arg & REG_MASK) | RM(OFFS_REG(arg)) | ((sljit_ins)(argw & 0x3) << 7));
	}

	arg &= REG_MASK;

	if (arg) {
		if (argw <= max_offset && argw >= -mask) {
			*mem = arg;
			return SLJIT_SUCCESS;
		}

		if (argw >= 0) {
			tmp = (sljit_uw)(argw & (sign | mask));
			tmp = (sljit_uw)((argw + ((tmp <= (sljit_uw)max_offset || tmp == (sljit_uw)sign) ? 0 : sign)) & ~mask);
			imm = get_imm(tmp);

			if (imm) {
				*memw = argw - (sljit_sw)tmp;
				SLJIT_ASSERT(*memw >= -mask && *memw <= max_offset);

				return push_inst(compiler, ADD | RD(TMP_REG1) | RN(arg) | imm);
			}
		} else {
			tmp = (sljit_uw)(-argw & (sign | mask));
			tmp = (sljit_uw)((-argw + ((tmp <= (sljit_uw)((sign << 1) - max_offset - 1)) ? 0 : sign)) & ~mask);
			imm = get_imm(tmp);

			if (imm) {
				*memw = argw + (sljit_sw)tmp;
				SLJIT_ASSERT(*memw >= -mask && *memw <= max_offset);

				return push_inst(compiler, SUB | RD(TMP_REG1) | RN(arg) | imm);
			}
		}
	}

	tmp = (sljit_uw)(argw & (sign | mask));
	tmp = (sljit_uw)((argw + ((tmp <= (sljit_uw)max_offset || tmp == (sljit_uw)sign) ? 0 : sign)) & ~mask);
	*memw = argw - (sljit_sw)tmp;

	FAIL_IF(load_immediate(compiler, TMP_REG1, tmp));

	if (arg == 0)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ADD | RD(TMP_REG1) | RN(TMP_REG1) | RM(arg));
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

	ADJUST_LOCAL_OFFSET(mem, memw);

	FAIL_IF(update_mem_addr(compiler, &mem, &memw, 0xfff - 4));

	flags = WORD_SIZE;

	if (!(type & SLJIT_MEM_STORE)) {
		if (REG_PAIR_FIRST(reg) == (mem & REG_MASK)) {
			FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, REG_PAIR_SECOND(reg), SLJIT_MEM1(mem), memw + SSIZE_OF(sw), TMP_REG1));
			return emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, REG_PAIR_FIRST(reg), SLJIT_MEM1(mem), memw, TMP_REG1);
		}

		flags = WORD_SIZE | LOAD_DATA;
	}

	FAIL_IF(emit_op_mem(compiler, flags, REG_PAIR_FIRST(reg), SLJIT_MEM1(mem), memw, TMP_REG1));
	return emit_op_mem(compiler, flags, REG_PAIR_SECOND(reg), SLJIT_MEM1(mem), memw + SSIZE_OF(sw), TMP_REG1);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 flags;
	sljit_ins is_type1_transfer, inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem_update(compiler, type, reg, mem, memw));

	is_type1_transfer = 1;

	switch (type & 0xff) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
	case SLJIT_MOV_P:
		flags = WORD_SIZE;
		break;
	case SLJIT_MOV_U8:
		flags = BYTE_SIZE;
		break;
	case SLJIT_MOV_S8:
		if (!(type & SLJIT_MEM_STORE))
			is_type1_transfer = 0;
		flags = BYTE_SIZE | SIGNED;
		break;
	case SLJIT_MOV_U16:
		is_type1_transfer = 0;
		flags = HALF_SIZE;
		break;
	case SLJIT_MOV_S16:
		is_type1_transfer = 0;
		flags = HALF_SIZE | SIGNED;
		break;
	default:
		SLJIT_UNREACHABLE();
		flags = WORD_SIZE;
		break;
	}

	if (!(type & SLJIT_MEM_STORE))
		flags |= LOAD_DATA;

	SLJIT_ASSERT(is_type1_transfer == !!IS_TYPE1_TRANSFER(flags));

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		if (!is_type1_transfer && memw != 0)
			return SLJIT_ERR_UNSUPPORTED;
	} else {
		if (is_type1_transfer) {
			if (memw > 4095 || memw < -4095)
				return SLJIT_ERR_UNSUPPORTED;
		} else if (memw > 255 || memw < -255)
			return SLJIT_ERR_UNSUPPORTED;
	}

	if (type & SLJIT_MEM_SUPP)
		return SLJIT_SUCCESS;

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		memw &= 0x3;

		inst = EMIT_DATA_TRANSFER(flags, 1, reg, mem & REG_MASK, RM(OFFS_REG(mem)) | ((sljit_ins)memw << 7));

		if (is_type1_transfer)
			inst |= (1 << 25);

		if (type & SLJIT_MEM_POST)
			inst ^= (1 << 24);
		else
			inst |= (1 << 21);

		return push_inst(compiler, inst);
	}

	inst = EMIT_DATA_TRANSFER(flags, 0, reg, mem & REG_MASK, 0);

	if (type & SLJIT_MEM_POST)
		inst ^= (1 << 24);
	else
		inst |= (1 << 21);

	if (is_type1_transfer) {
		if (memw >= 0)
			inst |= (1 << 23);
		else
			memw = -memw;

		return push_inst(compiler, inst | (sljit_ins)memw);
	}

	if (memw >= 0)
		inst |= (1 << 23);
	else
		memw = -memw;

	return push_inst(compiler, inst | TYPE2_TRANSFER_IMM((sljit_ins)memw));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem(compiler, type, freg, mem, memw));

	if (type & SLJIT_MEM_ALIGNED_32)
		return emit_fop_mem(compiler, ((type ^ SLJIT_32) & SLJIT_32) | ((type & SLJIT_MEM_STORE) ? 0 : FPU_LOAD), freg, mem, memw);

	if (type & SLJIT_MEM_STORE) {
		FAIL_IF(push_inst(compiler, VMOV | (1 << 20) | VN(freg) | RD(TMP_REG2)));

		if (type & SLJIT_32)
			return emit_op_mem(compiler, WORD_SIZE, TMP_REG2, mem, memw, TMP_REG1);

		FAIL_IF(update_mem_addr(compiler, &mem, &memw, 0xfff - 4));
		mem |= SLJIT_MEM;

		FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG2, mem, memw, TMP_REG1));
		FAIL_IF(push_inst(compiler, VMOV | (1 << 20) | VN(freg) | 0x80 | RD(TMP_REG2)));
		return emit_op_mem(compiler, WORD_SIZE, TMP_REG2, mem, memw + 4, TMP_REG1);
	}

	if (type & SLJIT_32) {
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG2, mem, memw, TMP_REG1));
		return push_inst(compiler, VMOV | VN(freg) | RD(TMP_REG2));
	}

	FAIL_IF(update_mem_addr(compiler, &mem, &memw, 0xfff - 4));
	mem |= SLJIT_MEM;

	FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG2, mem, memw, TMP_REG1));
	FAIL_IF(emit_op_mem(compiler, WORD_SIZE | LOAD_DATA, TMP_REG1, mem, memw + 4, TMP_REG1));
	return push_inst(compiler, VMOV2 | VM(freg) | RD(TMP_REG2) | RN(TMP_REG1));
}

static sljit_s32 sljit_emit_simd_mem_offset(struct sljit_compiler *compiler, sljit_s32 *mem_ptr, sljit_sw memw)
{
	sljit_s32 mem = *mem_ptr;
	sljit_uw imm;

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		*mem_ptr = TMP_REG1;
		return push_inst(compiler, ADD | RD(TMP_REG1) | RN(mem & REG_MASK) | RM(OFFS_REG(mem)) | ((sljit_ins)(memw & 0x3) << 7));
	}

	if (SLJIT_UNLIKELY(!(mem & REG_MASK))) {
		*mem_ptr = TMP_REG1;
		return load_immediate(compiler, TMP_REG1, (sljit_uw)memw);
	}

	mem &= REG_MASK;

	if (memw == 0) {
		*mem_ptr = mem;
		return SLJIT_SUCCESS;
	}

	*mem_ptr = TMP_REG1;
	imm = get_imm((sljit_uw)(memw < 0 ? -memw : memw));

	if (imm != 0)
		return push_inst(compiler, ((memw < 0) ? SUB : ADD) | RD(TMP_REG1) | RN(mem) | imm);

	FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)memw));
	return push_inst(compiler, ADD | RD(TMP_REG1) | RN(TMP_REG1) | RM(mem));
}

static SLJIT_INLINE sljit_s32 simd_get_quad_reg_index(sljit_s32 freg)
{
	freg += freg & 0x1;

	SLJIT_ASSERT((freg_map[freg] & 0x1) == (freg <= SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS));

	if (freg <= SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS)
		freg--;

	return freg;
}

#define SLJIT_QUAD_OTHER_HALF(freg) ((((freg) & 0x1) << 1) - 1)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 alignment = SLJIT_SIMD_GET_ELEM2_SIZE(type);
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_mov(compiler, type, freg, srcdst, srcdstw));

	ADJUST_LOCAL_OFFSET(srcdst, srcdstw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4)
		freg = simd_get_quad_reg_index(freg);

	if (!(srcdst & SLJIT_MEM)) {
		if (reg_size == 4)
			srcdst = simd_get_quad_reg_index(srcdst);

		if (type & SLJIT_SIMD_STORE)
			ins = VD(srcdst) | VN(freg) | VM(freg);
		else
			ins = VD(freg) | VN(srcdst) | VM(srcdst);

		if (reg_size == 4)
			ins |= (sljit_ins)1 << 6;

		return push_inst(compiler, VORR | ins);
	}

	FAIL_IF(sljit_emit_simd_mem_offset(compiler, &srcdst, srcdstw));

	if (elem_size > 3)
		elem_size = 3;

	ins = ((type & SLJIT_SIMD_STORE) ? VST1 : VLD1) | VD(freg)
		| (sljit_ins)((reg_size == 3) ? (0x7 << 8) : (0xa << 8));

	SLJIT_ASSERT(reg_size >= alignment);

	if (alignment == 3)
		ins |= 0x10;
	else if (alignment >= 3)
		ins |= 0x20;

	return push_inst(compiler, ins | RN(srcdst) | ((sljit_ins)elem_size) << 6 | 0xf);
}

static sljit_ins simd_get_imm(sljit_s32 elem_size, sljit_uw value)
{
	sljit_ins result;

	if (elem_size > 1 && (sljit_u16)value == (value >> 16)) {
		elem_size = 1;
		value = (sljit_u16)value;
	}

	if (elem_size == 1 && (sljit_u8)value == (value >> 8)) {
		elem_size = 0;
		value = (sljit_u8)value;
	}

	switch (elem_size) {
	case 0:
		SLJIT_ASSERT(value <= 0xff);
		result = 0xe00;
		break;
	case 1:
		SLJIT_ASSERT(value <= 0xffff);
		result = 0;

		while (1) {
			if (value <= 0xff) {
				result |= 0x800;
				break;
			}

			if ((value & 0xff) == 0) {
				value >>= 8;
				result |= 0xa00;
				break;
			}

			if (result != 0)
				return ~(sljit_ins)0;

			value ^= (sljit_uw)0xffff;
			result = (1 << 5);
		}
		break;
	default:
		SLJIT_ASSERT(value <= 0xffffffff);
		result = 0;

		while (1) {
			if (value <= 0xff) {
				result |= 0x000;
				break;
			}

			if ((value & ~(sljit_uw)0xff00) == 0) {
				value >>= 8;
				result |= 0x200;
				break;
			}

			if ((value & ~(sljit_uw)0xff0000) == 0) {
				value >>= 16;
				result |= 0x400;
				break;
			}

			if ((value & ~(sljit_uw)0xff000000) == 0) {
				value >>= 24;
				result |= 0x600;
				break;
			}

			if ((value & (sljit_uw)0xff) == 0xff && (value >> 16) == 0) {
				value >>= 8;
				result |= 0xc00;
				break;
			}

			if ((value & (sljit_uw)0xffff) == 0xffff && (value >> 24) == 0) {
				value >>= 16;
				result |= 0xd00;
				break;
			}

			if (result != 0)
				return ~(sljit_ins)0;

			value = ~value;
			result = (1 << 5);
		}
		break;
	}

	return ((sljit_ins)value & 0xf) | (((sljit_ins)value & 0x70) << 12) | (((sljit_ins)value & 0x80) << 17) | result;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins, imm;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_replicate(compiler, type, freg, src, srcw));

	ADJUST_LOCAL_OFFSET(src, srcw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) ? (elem_size < 2 || elem_size > 3) : (elem_size > 2))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4)
		freg = simd_get_quad_reg_index(freg);

	if (src == SLJIT_IMM && srcw == 0)
		return push_inst(compiler, VMOV_i | ((reg_size == 4) ? (1 << 6) : 0) | VD(freg));

	if (SLJIT_UNLIKELY(elem_size == 3)) {
		SLJIT_ASSERT(type & SLJIT_SIMD_FLOAT);

		if (src & SLJIT_MEM) {
			FAIL_IF(emit_fop_mem(compiler, FPU_LOAD | SLJIT_32, freg, src, srcw));
			src = freg;
		} else if (freg != src)
			FAIL_IF(push_inst(compiler, VORR | VD(freg) | VN(src) | VM(src)));

		freg += SLJIT_QUAD_OTHER_HALF(freg);

		if (freg != src)
			return push_inst(compiler, VORR | VD(freg) | VN(src) | VM(src));
		return SLJIT_SUCCESS;
	}

	if (src & SLJIT_MEM) {
		FAIL_IF(sljit_emit_simd_mem_offset(compiler, &src, srcw));

		ins = (sljit_ins)(elem_size << 6);

		if (reg_size == 4)
			ins |= (sljit_ins)1 << 5;

		return push_inst(compiler, VLD1_r | ins | VD(freg) | RN(src) | 0xf);
	}

	if (type & SLJIT_SIMD_FLOAT) {
		SLJIT_ASSERT(elem_size == 2);
		ins = ((sljit_ins)freg_ebit_map[src] << (16 + 2 + 1)) | ((sljit_ins)1 << (16 + 2));

		if (reg_size == 4)
			ins |= (sljit_ins)1 << 6;

		return push_inst(compiler, VDUP_s | ins | VD(freg) | (sljit_ins)freg_map[src]);
	}

	if (src == SLJIT_IMM) {
		if (elem_size < 2)
			srcw &= ((sljit_sw)1 << (((sljit_sw)1 << elem_size) << 3)) - 1;

		imm = simd_get_imm(elem_size, (sljit_uw)srcw);

		if (imm != ~(sljit_ins)0) {
			if (reg_size == 4)
				imm |= (sljit_ins)1 << 6;

			return push_inst(compiler, VMOV_i | imm | VD(freg));
		}

		FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)srcw));
		src = TMP_REG1;
	}

	switch (elem_size) {
	case 0:
		ins = 1 << 22;
		break;
	case 1:
		ins = 1 << 5;
		break;
	default:
		ins = 0;
		break;
	}

	if (reg_size == 4)
		ins |= (sljit_ins)1 << 21;

	return push_inst(compiler, VDUP | ins | VN(freg) | RD(src));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg, sljit_s32 lane_index,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_mov(compiler, type, freg, lane_index, srcdst, srcdstw));

	ADJUST_LOCAL_OFFSET(srcdst, srcdstw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) ? (elem_size < 2 || elem_size > 3) : (elem_size > 2))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4)
		freg = simd_get_quad_reg_index(freg);

	if (type & SLJIT_SIMD_LANE_ZERO) {
		ins = (reg_size == 3) ? 0 : ((sljit_ins)1 << 6);

		if (type & SLJIT_SIMD_FLOAT) {
			if (elem_size == 3 && !(srcdst & SLJIT_MEM)) {
				if (lane_index == 1)
					freg += SLJIT_QUAD_OTHER_HALF(freg);

				if (srcdst != freg)
					FAIL_IF(push_inst(compiler, VORR | VD(freg) | VN(srcdst) | VM(srcdst)));

				freg += SLJIT_QUAD_OTHER_HALF(freg);
				return push_inst(compiler, VMOV_i | VD(freg));
			}

			if (srcdst == freg || (elem_size == 3 && srcdst == (freg + SLJIT_QUAD_OTHER_HALF(freg)))) {
				FAIL_IF(push_inst(compiler, VORR | ins | VD(TMP_FREG2) | VN(freg) | VM(freg)));
				srcdst = TMP_FREG2;
				srcdstw = 0;
			}
		}

		FAIL_IF(push_inst(compiler, VMOV_i | ins | VD(freg)));
	}

	if (reg_size == 4 && lane_index >= (0x8 >> elem_size)) {
		lane_index -= (0x8 >> elem_size);
		freg += SLJIT_QUAD_OTHER_HALF(freg);
	}

	if (srcdst & SLJIT_MEM) {
		if (elem_size == 3)
			return emit_fop_mem(compiler, ((type & SLJIT_SIMD_STORE) ? 0 : FPU_LOAD) | SLJIT_32, freg, srcdst, srcdstw);

		FAIL_IF(sljit_emit_simd_mem_offset(compiler, &srcdst, srcdstw));

		lane_index = lane_index << elem_size;
		ins = (sljit_ins)((elem_size << 10) | (lane_index << 5));
		return push_inst(compiler, ((type & SLJIT_SIMD_STORE) ? VST1_s : VLD1_s) | ins | VD(freg) | RN(srcdst) | 0xf);
	}

	if (type & SLJIT_SIMD_FLOAT) {
		if (elem_size == 3) {
			if (type & SLJIT_SIMD_STORE)
				return push_inst(compiler, VORR | VD(srcdst) | VN(freg) | VM(freg));
			return push_inst(compiler, VMOV_F32 | SLJIT_32 | VD(freg) | VM(srcdst));
		}

		if (type & SLJIT_SIMD_STORE) {
			if (freg_ebit_map[freg] == 0) {
				if (lane_index == 1)
					freg = SLJIT_F64_SECOND(freg);

				return push_inst(compiler, VMOV_F32 | VD(srcdst) | VM(freg));
			}

			FAIL_IF(push_inst(compiler, VMOV_s | (1 << 20) | ((sljit_ins)lane_index << 21) | VN(freg) | RD(TMP_REG1)));
			return push_inst(compiler, VMOV | VN(srcdst) | RD(TMP_REG1));
		}

		FAIL_IF(push_inst(compiler, VMOV | (1 << 20) | VN(srcdst) | RD(TMP_REG1)));
		return push_inst(compiler, VMOV_s | ((sljit_ins)lane_index << 21) | VN(freg) | RD(TMP_REG1));
	}

	if (srcdst == SLJIT_IMM) {
		if (elem_size < 2)
			srcdstw &= ((sljit_sw)1 << (((sljit_sw)1 << elem_size) << 3)) - 1;

		FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_uw)srcdstw));
		srcdst = TMP_REG1;
	}

	if (elem_size == 0)
		ins = 0x400000;
	else if (elem_size == 1)
		ins = 0x20;
	else
		ins = 0;

	lane_index = lane_index << elem_size;
	ins |= (sljit_ins)(((lane_index & 0x4) << 19) | ((lane_index & 0x3) << 5));

	if (type & SLJIT_SIMD_STORE) {
		ins |= (1 << 20);

		if (elem_size < 2 && !(type & SLJIT_SIMD_LANE_SIGNED))
			ins |= (1 << 23);
	}

	return push_inst(compiler, VMOV_s | ins | VN(freg) | RD(srcdst));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 src, sljit_s32 src_lane_index)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_replicate(compiler, type, freg, src, src_lane_index));

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4) {
		freg = simd_get_quad_reg_index(freg);
		src = simd_get_quad_reg_index(src);

		if (src_lane_index >= (0x8 >> elem_size)) {
			src_lane_index -= (0x8 >> elem_size);
			src += SLJIT_QUAD_OTHER_HALF(src);
		}
	}

	if (elem_size == 3) {
		if (freg != src)
			FAIL_IF(push_inst(compiler, VORR | VD(freg) | VN(src) | VM(src)));

		freg += SLJIT_QUAD_OTHER_HALF(freg);

		if (freg != src)
			return push_inst(compiler, VORR | VD(freg) | VN(src) | VM(src));
		return SLJIT_SUCCESS;
	}

	ins = ((((sljit_ins)src_lane_index << 1) | 1) << (16 + elem_size));

	if (reg_size == 4)
		ins |= (sljit_ins)1 << 6;

	return push_inst(compiler, VDUP_s | ins | VD(freg) | VM(src));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_extend(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 elem2_size = SLJIT_SIMD_GET_ELEM2_SIZE(type);
	sljit_s32 dst_reg;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_extend(compiler, type, freg, src, srcw));

	ADJUST_LOCAL_OFFSET(src, srcw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size != 2 || elem2_size != 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4)
		freg = simd_get_quad_reg_index(freg);

	if (src & SLJIT_MEM) {
		FAIL_IF(sljit_emit_simd_mem_offset(compiler, &src, srcw));
		if (reg_size == 4 && elem2_size - elem_size == 1)
			FAIL_IF(push_inst(compiler, VLD1 | (0x7 << 8) | VD(freg) | RN(src) | 0xf));
		else
			FAIL_IF(push_inst(compiler, VLD1_s | (sljit_ins)((reg_size - elem2_size + elem_size) << 10) | VD(freg) | RN(src) | 0xf));
		src = freg;
	} else if (reg_size == 4)
		src = simd_get_quad_reg_index(src);

	if (!(type & SLJIT_SIMD_FLOAT)) {
		dst_reg = (reg_size == 4) ? freg : TMP_FREG2;

		do {
			FAIL_IF(push_inst(compiler, VSHLL | ((type & SLJIT_SIMD_EXTEND_SIGNED) ? 0 : (1 << 24))
				| ((sljit_ins)1 << (19 + elem_size)) | VD(dst_reg) | VM(src)));
			src = dst_reg;
		} while (++elem_size < elem2_size);

		if (dst_reg == TMP_FREG2)
			return push_inst(compiler, VORR | VD(freg) | VN(TMP_FREG2) | VM(TMP_FREG2));
		return SLJIT_SUCCESS;
	}

	/* No SIMD variant, must use VFP instead. */
	SLJIT_ASSERT(reg_size == 4);

	if (freg == src) {
		freg += SLJIT_QUAD_OTHER_HALF(freg);
		FAIL_IF(push_inst(compiler, VCVT_F64_F32 | VD(freg) | VM(src) | 0x20));
		freg += SLJIT_QUAD_OTHER_HALF(freg);
		return push_inst(compiler, VCVT_F64_F32 | VD(freg) | VM(src));
	}

	FAIL_IF(push_inst(compiler, VCVT_F64_F32 | VD(freg) | VM(src)));
	freg += SLJIT_QUAD_OTHER_HALF(freg);
	return push_inst(compiler, VCVT_F64_F32 | VD(freg) | VM(src) | 0x20);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_sign(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins, imms;
	sljit_s32 dst_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_sign(compiler, type, freg, dst, dstw));

	ADJUST_LOCAL_OFFSET(dst, dstw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	switch (elem_size) {
	case 0:
		imms = 0x243219;
		ins = VSHR | (1 << 24) | (0x9 << 16);
		break;
	case 1:
		imms = (reg_size == 4) ? 0x243219 : 0x2231;
		ins = VSHR | (1 << 24) | (0x11 << 16);
		break;
	case 2:
		imms = (reg_size == 4) ? 0x2231 : 0x21;
		ins = VSHR | (1 << 24) | (0x21 << 16);
		break;
	default:
		imms = 0x21;
		ins = VSHR | (1 << 24) | (0x1 << 16) | (1 << 7);
		break;
	}

	if (reg_size == 4) {
		freg = simd_get_quad_reg_index(freg);
		ins |= (sljit_ins)1 << 6;
	}

	SLJIT_ASSERT((freg_map[TMP_FREG2] & 0x1) == 0);
	FAIL_IF(push_inst(compiler, ins | VD(TMP_FREG2) | VM(freg)));

	if (reg_size == 4 && elem_size > 0)
		FAIL_IF(push_inst(compiler, VMOVN | ((sljit_ins)(elem_size - 1) << 18) | VD(TMP_FREG2) | VM(TMP_FREG2)));

	ins = (reg_size == 4 && elem_size == 0) ? (1 << 6) : 0;

	while (imms >= 0x100) {
		FAIL_IF(push_inst(compiler, VSRA | (1 << 24) | ins | ((imms & 0xff) << 16) | VD(TMP_FREG2) | VM(TMP_FREG2)));
		imms >>= 8;
	}

	FAIL_IF(push_inst(compiler, VSRA | (1 << 24) | ins | (1 << 7) | (imms << 16) | VD(TMP_FREG2) | VM(TMP_FREG2)));

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	FAIL_IF(push_inst(compiler, VMOV_s | (1 << 20) | (1 << 23) | (0x2 << 21) | RD(dst_r) | VN(TMP_FREG2)));

	if (reg_size == 4 && elem_size == 0) {
		SLJIT_ASSERT(freg_map[TMP_FREG2] + 1 == freg_map[TMP_FREG1]);
		FAIL_IF(push_inst(compiler, VMOV_s | (1 << 20) | (1 << 23) | (0x2 << 21) | RD(TMP_REG2) | VN(TMP_FREG1)));
		FAIL_IF(push_inst(compiler, ORR | RD(dst_r) | RN(dst_r) | RM(TMP_REG2) | (0x8 << 7)));
	}

	if (dst_r == TMP_REG1)
		return emit_op_mem(compiler, WORD_SIZE, TMP_REG1, dst, dstw, TMP_REG2);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_op2(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg, sljit_s32 src1_freg, sljit_s32 src2_freg)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_op2(compiler, type, dst_freg, src1_freg, src2_freg));

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	switch (SLJIT_SIMD_GET_OPCODE(type)) {
	case SLJIT_SIMD_OP2_AND:
		ins = VAND;
		break;
	case SLJIT_SIMD_OP2_OR:
		ins = VORR;
		break;
	case SLJIT_SIMD_OP2_XOR:
		ins = VEOR;
		break;
	}

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4) {
		dst_freg = simd_get_quad_reg_index(dst_freg);
		src1_freg = simd_get_quad_reg_index(src1_freg);
		src2_freg = simd_get_quad_reg_index(src2_freg);
		ins |= (sljit_ins)1 << 6;
	}

	return push_inst(compiler, ins | VD(dst_freg) | VN(src1_freg) | VM(src2_freg));
}

#undef FPU_LOAD

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	sljit_u32 ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_load(compiler, op, dst_reg, mem_reg));

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_U8:
		ins = LDREXB;
		break;
	case SLJIT_MOV_U16:
		ins = LDREXH;
		break;
	default:
		ins = LDREX;
		break;
	}

	return push_inst(compiler, ins | RN(mem_reg) | RD(dst_reg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	sljit_u32 ins;

	/* temp_reg == mem_reg is undefined so use another temp register */
	SLJIT_UNUSED_ARG(temp_reg);

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_store(compiler, op, src_reg, mem_reg, temp_reg));

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_U8:
		ins = STREXB;
		break;
	case SLJIT_MOV_U16:
		ins = STREXH;
		break;
	default:
		ins = STREX;
		break;
	}

	FAIL_IF(push_inst(compiler, ins | RN(mem_reg) | RD(TMP_REG1) | RM(src_reg)));
	if (op & SLJIT_SET_ATOMIC_STORED)
		return push_inst(compiler, CMP | SET_FLAGS | SRC2_IMM | RN(TMP_REG1));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	struct sljit_const *const_;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	PTR_FAIL_IF(push_inst_with_unique_literal(compiler,
		EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1, dst_r, TMP_PC, 0), (sljit_ins)init_value));
	compiler->patches++;
#else /* !SLJIT_CONFIG_ARM_V6 */
	PTR_FAIL_IF(emit_imm(compiler, dst_r, init_value));
#endif /* SLJIT_CONFIG_ARM_V6 */

	const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG2, dst, dstw, TMP_REG1));
	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_put_label *put_label;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_put_label(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	PTR_FAIL_IF(push_inst_with_unique_literal(compiler, EMIT_DATA_TRANSFER(WORD_SIZE | LOAD_DATA, 1, dst_r, TMP_PC, 0), 0));
	compiler->patches++;
#else /* !SLJIT_CONFIG_ARM_V6 */
	PTR_FAIL_IF(emit_imm(compiler, dst_r, 0));
#endif /* SLJIT_CONFIG_ARM_V6 */

	put_label = (struct sljit_put_label*)ensure_abuf(compiler, sizeof(struct sljit_put_label));
	PTR_FAIL_IF(!put_label);
	set_put_label(put_label, compiler, 0);

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG2, dst, dstw, TMP_REG1));
	return put_label;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	inline_set_jump_addr(addr, executable_offset, new_target, 1);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	inline_set_const(addr, executable_offset, (sljit_uw)new_constant, 1);
}
