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

/* x86 32-bit arch dependent functions. */

static sljit_s32 emit_do_imm(struct sljit_compiler *compiler, sljit_u8 opcode, sljit_sw imm)
{
	sljit_u8 *inst;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + sizeof(sljit_sw));
	FAIL_IF(!inst);
	INC_SIZE(1 + sizeof(sljit_sw));
	*inst++ = opcode;
	sljit_unaligned_store_sw(inst, imm);
	return SLJIT_SUCCESS;
}

static sljit_u8* generate_far_jump_code(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_s32 type, sljit_sw executable_offset)
{
	if (type == SLJIT_JUMP) {
		*code_ptr++ = JMP_i32;
		jump->addr++;
	}
	else if (type >= SLJIT_FAST_CALL) {
		*code_ptr++ = CALL_i32;
		jump->addr++;
	}
	else {
		*code_ptr++ = GROUP_0F;
		*code_ptr++ = get_jump_code(type);
		jump->addr += 2;
	}

	if (jump->flags & JUMP_LABEL)
		jump->flags |= PATCH_MW;
	else
		sljit_unaligned_store_sw(code_ptr, jump->u.target - (jump->addr + 4) - (sljit_uw)executable_offset);
	code_ptr += 4;

	return code_ptr;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 size;
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	compiler->args = args;
	compiler->flags_saved = 0;

	size = 1 + (scratches > 7 ? (scratches - 7) : 0) + (saveds <= 3 ? saveds : 3);
#if (defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
	size += (args > 0 ? (args * 2) : 0) + (args > 2 ? 2 : 0);
#else
	size += (args > 0 ? (2 + args * 3) : 0);
#endif
	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);

	INC_SIZE(size);
	PUSH_REG(reg_map[TMP_REG1]);
#if !(defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
	if (args > 0) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_REG | (reg_map[TMP_REG1] << 3) | 0x4 /* esp */;
	}
#endif
	if (saveds > 2 || scratches > 7)
		PUSH_REG(reg_map[SLJIT_S2]);
	if (saveds > 1 || scratches > 8)
		PUSH_REG(reg_map[SLJIT_S1]);
	if (saveds > 0 || scratches > 9)
		PUSH_REG(reg_map[SLJIT_S0]);

#if (defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
	if (args > 0) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_REG | (reg_map[SLJIT_S0] << 3) | reg_map[SLJIT_R2];
	}
	if (args > 1) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_REG | (reg_map[SLJIT_S1] << 3) | reg_map[SLJIT_R1];
	}
	if (args > 2) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_DISP8 | (reg_map[SLJIT_S2] << 3) | 0x4 /* esp */;
		*inst++ = 0x24;
		*inst++ = sizeof(sljit_sw) * (3 + 2); /* saveds >= 3 as well. */
	}
#else
	if (args > 0) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_DISP8 | (reg_map[SLJIT_S0] << 3) | reg_map[TMP_REG1];
		*inst++ = sizeof(sljit_sw) * 2;
	}
	if (args > 1) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_DISP8 | (reg_map[SLJIT_S1] << 3) | reg_map[TMP_REG1];
		*inst++ = sizeof(sljit_sw) * 3;
	}
	if (args > 2) {
		*inst++ = MOV_r_rm;
		*inst++ = MOD_DISP8 | (reg_map[SLJIT_S2] << 3) | reg_map[TMP_REG1];
		*inst++ = sizeof(sljit_sw) * 4;
	}
#endif

	SLJIT_COMPILE_ASSERT(SLJIT_LOCALS_OFFSET >= (2 + 4) * sizeof(sljit_uw), require_at_least_two_words);
#if defined(__APPLE__)
	/* Ignore pushed registers and SLJIT_LOCALS_OFFSET when computing the aligned local size. */
	saveds = (2 + (scratches > 7 ? (scratches - 7) : 0) + (saveds <= 3 ? saveds : 3)) * sizeof(sljit_uw);
	local_size = ((SLJIT_LOCALS_OFFSET + saveds + local_size + 15) & ~15) - saveds;
#else
	if (options & SLJIT_DOUBLE_ALIGNMENT) {
		local_size = SLJIT_LOCALS_OFFSET + ((local_size + 7) & ~7);

		inst = (sljit_u8*)ensure_buf(compiler, 1 + 17);
		FAIL_IF(!inst);

		INC_SIZE(17);
		inst[0] = MOV_r_rm;
		inst[1] = MOD_REG | (reg_map[TMP_REG1] << 3) | reg_map[SLJIT_SP];
		inst[2] = GROUP_F7;
		inst[3] = MOD_REG | (0 << 3) | reg_map[SLJIT_SP];
		sljit_unaligned_store_sw(inst + 4, 0x4);
		inst[8] = JNE_i8;
		inst[9] = 6;
		inst[10] = GROUP_BINARY_81;
		inst[11] = MOD_REG | (5 << 3) | reg_map[SLJIT_SP];
		sljit_unaligned_store_sw(inst + 12, 0x4);
		inst[16] = PUSH_r + reg_map[TMP_REG1];
	}
	else
		local_size = SLJIT_LOCALS_OFFSET + ((local_size + 3) & ~3);
#endif

	compiler->local_size = local_size;
#ifdef _WIN32
	if (local_size > 1024) {
#if (defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
		FAIL_IF(emit_do_imm(compiler, MOV_r_i32 + reg_map[SLJIT_R0], local_size));
#else
		local_size -= SLJIT_LOCALS_OFFSET;
		FAIL_IF(emit_do_imm(compiler, MOV_r_i32 + reg_map[SLJIT_R0], local_size));
		FAIL_IF(emit_non_cum_binary(compiler, SUB_r_rm, SUB_rm_r, SUB, SUB_EAX_i32,
			SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, SLJIT_LOCALS_OFFSET));
#endif
		FAIL_IF(sljit_emit_ijump(compiler, SLJIT_CALL1, SLJIT_IMM, SLJIT_FUNC_OFFSET(sljit_grow_stack)));
	}
#endif

	SLJIT_ASSERT(local_size > 0);
	return emit_non_cum_binary(compiler, SUB_r_rm, SUB_rm_r, SUB, SUB_EAX_i32,
		SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, local_size);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	compiler->args = args;

#if defined(__APPLE__)
	saveds = (2 + (scratches > 7 ? (scratches - 7) : 0) + (saveds <= 3 ? saveds : 3)) * sizeof(sljit_uw);
	compiler->local_size = ((SLJIT_LOCALS_OFFSET + saveds + local_size + 15) & ~15) - saveds;
#else
	if (options & SLJIT_DOUBLE_ALIGNMENT)
		compiler->local_size = SLJIT_LOCALS_OFFSET + ((local_size + 7) & ~7);
	else
		compiler->local_size = SLJIT_LOCALS_OFFSET + ((local_size + 3) & ~3);
#endif
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 size;
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));
	SLJIT_ASSERT(compiler->args >= 0);

	compiler->flags_saved = 0;
	FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));

	SLJIT_ASSERT(compiler->local_size > 0);
	FAIL_IF(emit_cum_binary(compiler, ADD_r_rm, ADD_rm_r, ADD, ADD_EAX_i32,
		SLJIT_SP, 0, SLJIT_SP, 0, SLJIT_IMM, compiler->local_size));

#if !defined(__APPLE__)
	if (compiler->options & SLJIT_DOUBLE_ALIGNMENT) {
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 3);
		FAIL_IF(!inst);

		INC_SIZE(3);
		inst[0] = MOV_r_rm;
		inst[1] = (reg_map[SLJIT_SP] << 3) | 0x4 /* SIB */;
		inst[2] = (4 << 3) | reg_map[SLJIT_SP];
	}
#endif

	size = 2 + (compiler->scratches > 7 ? (compiler->scratches - 7) : 0) +
		(compiler->saveds <= 3 ? compiler->saveds : 3);
#if (defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
	if (compiler->args > 2)
		size += 2;
#else
	if (compiler->args > 0)
		size += 2;
#endif
	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);

	INC_SIZE(size);

	if (compiler->saveds > 0 || compiler->scratches > 9)
		POP_REG(reg_map[SLJIT_S0]);
	if (compiler->saveds > 1 || compiler->scratches > 8)
		POP_REG(reg_map[SLJIT_S1]);
	if (compiler->saveds > 2 || compiler->scratches > 7)
		POP_REG(reg_map[SLJIT_S2]);
	POP_REG(reg_map[TMP_REG1]);
#if (defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
	if (compiler->args > 2)
		RET_I16(sizeof(sljit_sw));
	else
		RET();
#else
	RET();
#endif

	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

/* Size contains the flags as well. */
static sljit_u8* emit_x86_instruction(struct sljit_compiler *compiler, sljit_s32 size,
	/* The register or immediate operand. */
	sljit_s32 a, sljit_sw imma,
	/* The general operand (not immediate). */
	sljit_s32 b, sljit_sw immb)
{
	sljit_u8 *inst;
	sljit_u8 *buf_ptr;
	sljit_s32 flags = size & ~0xf;
	sljit_s32 inst_size;

	/* Both cannot be switched on. */
	SLJIT_ASSERT((flags & (EX86_BIN_INS | EX86_SHIFT_INS)) != (EX86_BIN_INS | EX86_SHIFT_INS));
	/* Size flags not allowed for typed instructions. */
	SLJIT_ASSERT(!(flags & (EX86_BIN_INS | EX86_SHIFT_INS)) || (flags & (EX86_BYTE_ARG | EX86_HALF_ARG)) == 0);
	/* Both size flags cannot be switched on. */
	SLJIT_ASSERT((flags & (EX86_BYTE_ARG | EX86_HALF_ARG)) != (EX86_BYTE_ARG | EX86_HALF_ARG));
	/* SSE2 and immediate is not possible. */
	SLJIT_ASSERT(!(a & SLJIT_IMM) || !(flags & EX86_SSE2));
	SLJIT_ASSERT((flags & (EX86_PREF_F2 | EX86_PREF_F3)) != (EX86_PREF_F2 | EX86_PREF_F3)
		&& (flags & (EX86_PREF_F2 | EX86_PREF_66)) != (EX86_PREF_F2 | EX86_PREF_66)
		&& (flags & (EX86_PREF_F3 | EX86_PREF_66)) != (EX86_PREF_F3 | EX86_PREF_66));

	size &= 0xf;
	inst_size = size;

	if (flags & (EX86_PREF_F2 | EX86_PREF_F3))
		inst_size++;
	if (flags & EX86_PREF_66)
		inst_size++;

	/* Calculate size of b. */
	inst_size += 1; /* mod r/m byte. */
	if (b & SLJIT_MEM) {
		if ((b & REG_MASK) == SLJIT_UNUSED)
			inst_size += sizeof(sljit_sw);
		else if (immb != 0 && !(b & OFFS_REG_MASK)) {
			/* Immediate operand. */
			if (immb <= 127 && immb >= -128)
				inst_size += sizeof(sljit_s8);
			else
				inst_size += sizeof(sljit_sw);
		}

		if ((b & REG_MASK) == SLJIT_SP && !(b & OFFS_REG_MASK))
			b |= TO_OFFS_REG(SLJIT_SP);

		if ((b & OFFS_REG_MASK) != SLJIT_UNUSED)
			inst_size += 1; /* SIB byte. */
	}

	/* Calculate size of a. */
	if (a & SLJIT_IMM) {
		if (flags & EX86_BIN_INS) {
			if (imma <= 127 && imma >= -128) {
				inst_size += 1;
				flags |= EX86_BYTE_ARG;
			} else
				inst_size += 4;
		}
		else if (flags & EX86_SHIFT_INS) {
			imma &= 0x1f;
			if (imma != 1) {
				inst_size ++;
				flags |= EX86_BYTE_ARG;
			}
		} else if (flags & EX86_BYTE_ARG)
			inst_size++;
		else if (flags & EX86_HALF_ARG)
			inst_size += sizeof(short);
		else
			inst_size += sizeof(sljit_sw);
	}
	else
		SLJIT_ASSERT(!(flags & EX86_SHIFT_INS) || a == SLJIT_PREF_SHIFT_REG);

	inst = (sljit_u8*)ensure_buf(compiler, 1 + inst_size);
	PTR_FAIL_IF(!inst);

	/* Encoding the byte. */
	INC_SIZE(inst_size);
	if (flags & EX86_PREF_F2)
		*inst++ = 0xf2;
	if (flags & EX86_PREF_F3)
		*inst++ = 0xf3;
	if (flags & EX86_PREF_66)
		*inst++ = 0x66;

	buf_ptr = inst + size;

	/* Encode mod/rm byte. */
	if (!(flags & EX86_SHIFT_INS)) {
		if ((flags & EX86_BIN_INS) && (a & SLJIT_IMM))
			*inst = (flags & EX86_BYTE_ARG) ? GROUP_BINARY_83 : GROUP_BINARY_81;

		if ((a & SLJIT_IMM) || (a == 0))
			*buf_ptr = 0;
		else if (!(flags & EX86_SSE2_OP1))
			*buf_ptr = reg_map[a] << 3;
		else
			*buf_ptr = a << 3;
	}
	else {
		if (a & SLJIT_IMM) {
			if (imma == 1)
				*inst = GROUP_SHIFT_1;
			else
				*inst = GROUP_SHIFT_N;
		} else
			*inst = GROUP_SHIFT_CL;
		*buf_ptr = 0;
	}

	if (!(b & SLJIT_MEM))
		*buf_ptr++ |= MOD_REG + ((!(flags & EX86_SSE2_OP2)) ? reg_map[b] : b);
	else if ((b & REG_MASK) != SLJIT_UNUSED) {
		if ((b & OFFS_REG_MASK) == SLJIT_UNUSED || (b & OFFS_REG_MASK) == TO_OFFS_REG(SLJIT_SP)) {
			if (immb != 0) {
				if (immb <= 127 && immb >= -128)
					*buf_ptr |= 0x40;
				else
					*buf_ptr |= 0x80;
			}

			if ((b & OFFS_REG_MASK) == SLJIT_UNUSED)
				*buf_ptr++ |= reg_map[b & REG_MASK];
			else {
				*buf_ptr++ |= 0x04;
				*buf_ptr++ = reg_map[b & REG_MASK] | (reg_map[OFFS_REG(b)] << 3);
			}

			if (immb != 0) {
				if (immb <= 127 && immb >= -128)
					*buf_ptr++ = immb; /* 8 bit displacement. */
				else {
					sljit_unaligned_store_sw(buf_ptr, immb); /* 32 bit displacement. */
					buf_ptr += sizeof(sljit_sw);
				}
			}
		}
		else {
			*buf_ptr++ |= 0x04;
			*buf_ptr++ = reg_map[b & REG_MASK] | (reg_map[OFFS_REG(b)] << 3) | (immb << 6);
		}
	}
	else {
		*buf_ptr++ |= 0x05;
		sljit_unaligned_store_sw(buf_ptr, immb); /* 32 bit displacement. */
		buf_ptr += sizeof(sljit_sw);
	}

	if (a & SLJIT_IMM) {
		if (flags & EX86_BYTE_ARG)
			*buf_ptr = imma;
		else if (flags & EX86_HALF_ARG)
			sljit_unaligned_store_s16(buf_ptr, imma);
		else if (!(flags & EX86_SHIFT_INS))
			sljit_unaligned_store_sw(buf_ptr, imma);
	}

	return !(flags & EX86_SHIFT_INS) ? inst : (inst + 1);
}

/* --------------------------------------------------------------------- */
/*  Call / return instructions                                           */
/* --------------------------------------------------------------------- */

static SLJIT_INLINE sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 type)
{
	sljit_u8 *inst;

#if (defined SLJIT_X86_32_FASTCALL && SLJIT_X86_32_FASTCALL)
	inst = (sljit_u8*)ensure_buf(compiler, type >= SLJIT_CALL3 ? 1 + 2 + 1 : 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(type >= SLJIT_CALL3 ? 2 + 1 : 2);

	if (type >= SLJIT_CALL3)
		PUSH_REG(reg_map[SLJIT_R2]);
	*inst++ = MOV_r_rm;
	*inst++ = MOD_REG | (reg_map[SLJIT_R2] << 3) | reg_map[SLJIT_R0];
#else
	inst = (sljit_u8*)ensure_buf(compiler, 1 + 4 * (type - SLJIT_CALL0));
	FAIL_IF(!inst);
	INC_SIZE(4 * (type - SLJIT_CALL0));

	*inst++ = MOV_rm_r;
	*inst++ = MOD_DISP8 | (reg_map[SLJIT_R0] << 3) | 0x4 /* SIB */;
	*inst++ = (0x4 /* none*/ << 3) | reg_map[SLJIT_SP];
	*inst++ = 0;
	if (type >= SLJIT_CALL2) {
		*inst++ = MOV_rm_r;
		*inst++ = MOD_DISP8 | (reg_map[SLJIT_R1] << 3) | 0x4 /* SIB */;
		*inst++ = (0x4 /* none*/ << 3) | reg_map[SLJIT_SP];
		*inst++ = sizeof(sljit_sw);
	}
	if (type >= SLJIT_CALL3) {
		*inst++ = MOV_rm_r;
		*inst++ = MOD_DISP8 | (reg_map[SLJIT_R2] << 3) | 0x4 /* SIB */;
		*inst++ = (0x4 /* none*/ << 3) | reg_map[SLJIT_SP];
		*inst++ = 2 * sizeof(sljit_sw);
	}
#endif
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

	/* For UNUSED dst. Uncommon, but possible. */
	if (dst == SLJIT_UNUSED)
		dst = TMP_REG1;

	if (FAST_IS_REG(dst)) {
		/* Unused dest is possible here. */
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);

		INC_SIZE(1);
		POP_REG(reg_map[dst]);
		return SLJIT_SUCCESS;
	}

	/* Memory. */
	inst = emit_x86_instruction(compiler, 1, 0, 0, dst, dstw);
	FAIL_IF(!inst);
	*inst++ = POP_rm;
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_return(compiler, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	CHECK_EXTRA_REGS(src, srcw, (void)0);

	if (FAST_IS_REG(src)) {
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + 1);
		FAIL_IF(!inst);

		INC_SIZE(1 + 1);
		PUSH_REG(reg_map[src]);
	}
	else if (src & SLJIT_MEM) {
		inst = emit_x86_instruction(compiler, 1, 0, 0, src, srcw);
		FAIL_IF(!inst);
		*inst++ = GROUP_FF;
		*inst |= PUSH_rm;

		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
	}
	else {
		/* SLJIT_IMM. */
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 5 + 1);
		FAIL_IF(!inst);

		INC_SIZE(5 + 1);
		*inst++ = PUSH_i32;
		sljit_unaligned_store_sw(inst, srcw);
		inst += sizeof(sljit_sw);
	}

	RET();
	return SLJIT_SUCCESS;
}
