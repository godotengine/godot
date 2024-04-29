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

/* x86 32-bit arch dependent functions. */

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

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

/* Size contains the flags as well. */
static sljit_u8* emit_x86_instruction(struct sljit_compiler *compiler, sljit_uw size,
	/* The register or immediate operand. */
	sljit_s32 a, sljit_sw imma,
	/* The general operand (not immediate). */
	sljit_s32 b, sljit_sw immb)
{
	sljit_u8 *inst;
	sljit_u8 *buf_ptr;
	sljit_u8 reg_map_b;
	sljit_uw flags = size;
	sljit_uw inst_size;

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
		if (!(b & REG_MASK))
			inst_size += sizeof(sljit_sw);
		else {
			if (immb != 0 && !(b & OFFS_REG_MASK)) {
				/* Immediate operand. */
				if (immb <= 127 && immb >= -128)
					inst_size += sizeof(sljit_s8);
				else
					inst_size += sizeof(sljit_sw);
			}
			else if (reg_map[b & REG_MASK] == 5) {
				/* Swap registers if possible. */
				if ((b & OFFS_REG_MASK) && (immb & 0x3) == 0 && reg_map[OFFS_REG(b)] != 5)
					b = SLJIT_MEM | OFFS_REG(b) | TO_OFFS_REG(b & REG_MASK);
				else
					inst_size += sizeof(sljit_s8);
			}

			if (reg_map[b & REG_MASK] == 4 && !(b & OFFS_REG_MASK))
				b |= TO_OFFS_REG(SLJIT_SP);

			if (b & OFFS_REG_MASK)
				inst_size += 1; /* SIB byte. */
		}
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
			SLJIT_ASSERT(imma <= 0x1f);
			if (imma != 1) {
				inst_size++;
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

		if (a & SLJIT_IMM)
			*buf_ptr = 0;
		else if (!(flags & EX86_SSE2_OP1))
			*buf_ptr = U8(reg_map[a] << 3);
		else
			*buf_ptr = U8(a << 3);
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

	if (!(b & SLJIT_MEM)) {
		*buf_ptr = U8(*buf_ptr | MOD_REG | (!(flags & EX86_SSE2_OP2) ? reg_map[b] : b));
		buf_ptr++;
	} else if (b & REG_MASK) {
		reg_map_b = reg_map[b & REG_MASK];

		if (!(b & OFFS_REG_MASK) || (b & OFFS_REG_MASK) == TO_OFFS_REG(SLJIT_SP)) {
			if (immb != 0 || reg_map_b == 5) {
				if (immb <= 127 && immb >= -128)
					*buf_ptr |= 0x40;
				else
					*buf_ptr |= 0x80;
			}

			if (!(b & OFFS_REG_MASK))
				*buf_ptr++ |= reg_map_b;
			else {
				*buf_ptr++ |= 0x04;
				*buf_ptr++ = U8(reg_map_b | (reg_map[OFFS_REG(b)] << 3));
			}

			if (immb != 0 || reg_map_b == 5) {
				if (immb <= 127 && immb >= -128)
					*buf_ptr++ = U8(immb); /* 8 bit displacement. */
				else {
					sljit_unaligned_store_sw(buf_ptr, immb); /* 32 bit displacement. */
					buf_ptr += sizeof(sljit_sw);
				}
			}
		}
		else {
			if (reg_map_b == 5)
				*buf_ptr |= 0x40;

			*buf_ptr++ |= 0x04;
			*buf_ptr++ = U8(reg_map_b | (reg_map[OFFS_REG(b)] << 3) | (immb << 6));

			if (reg_map_b == 5)
				*buf_ptr++ = 0;
		}
	}
	else {
		*buf_ptr++ |= 0x05;
		sljit_unaligned_store_sw(buf_ptr, immb); /* 32 bit displacement. */
		buf_ptr += sizeof(sljit_sw);
	}

	if (a & SLJIT_IMM) {
		if (flags & EX86_BYTE_ARG)
			*buf_ptr = U8(imma);
		else if (flags & EX86_HALF_ARG)
			sljit_unaligned_store_s16(buf_ptr, (sljit_s16)imma);
		else if (!(flags & EX86_SHIFT_INS))
			sljit_unaligned_store_sw(buf_ptr, imma);
	}

	return !(flags & EX86_SHIFT_INS) ? inst : (inst + 1);
}

/* --------------------------------------------------------------------- */
/*  Enter / return                                                       */
/* --------------------------------------------------------------------- */

static sljit_u8* generate_far_jump_code(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_sw executable_offset)
{
	sljit_uw type = jump->flags >> TYPE_SHIFT;

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
		sljit_unaligned_store_sw(code_ptr, (sljit_sw)(jump->u.target - (jump->addr + 4) - (sljit_uw)executable_offset));
	code_ptr += 4;

	return code_ptr;
}

#define ENTER_TMP_TO_R4		0x00001
#define ENTER_TMP_TO_S		0x00002

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 word_arg_count, saved_arg_count, float_arg_count;
	sljit_s32 size, args_size, types, status;
	sljit_s32 kept_saveds_count = SLJIT_KEPT_SAVEDS_COUNT(options);
	sljit_u8 *inst;
#ifdef _WIN32
	sljit_s32 r2_offset = -1;
#endif

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	/* Emit ENDBR32 at function entry if needed.  */
	FAIL_IF(emit_endbranch(compiler));

	SLJIT_COMPILE_ASSERT(SLJIT_FR0 == 1, float_register_index_start);

	arg_types >>= SLJIT_ARG_SHIFT;
	word_arg_count = 0;
	status = 0;

	if (options & SLJIT_ENTER_REG_ARG) {
		args_size = 3 * SSIZE_OF(sw);

		while (arg_types) {
			if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
				word_arg_count++;
				if (word_arg_count >= 4)
					status |= ENTER_TMP_TO_R4;
			}

			arg_types >>= SLJIT_ARG_SHIFT;
		}

		compiler->args_size = 0;
	} else {
		types = arg_types;
		saved_arg_count = 0;
		float_arg_count = 0;
		args_size = SSIZE_OF(sw);
		while (types) {
			switch (types & SLJIT_ARG_MASK) {
			case SLJIT_ARG_TYPE_F64:
				float_arg_count++;
				FAIL_IF(emit_sse2_load(compiler, 0, float_arg_count, SLJIT_MEM1(SLJIT_SP), args_size));
				args_size += SSIZE_OF(f64);
				break;
			case SLJIT_ARG_TYPE_F32:
				float_arg_count++;
				FAIL_IF(emit_sse2_load(compiler, 1, float_arg_count, SLJIT_MEM1(SLJIT_SP), args_size));
				args_size += SSIZE_OF(f32);
				break;
			default:
				word_arg_count++;

				if (!(types & SLJIT_ARG_TYPE_SCRATCH_REG))
					saved_arg_count++;

				if (word_arg_count == 4) {
					if (types & SLJIT_ARG_TYPE_SCRATCH_REG) {
						status |= ENTER_TMP_TO_R4;
						arg_types &= ~(SLJIT_ARG_FULL_MASK << 3 * SLJIT_ARG_SHIFT);
					} else if (saved_arg_count == 4) {
						status |= ENTER_TMP_TO_S;
						arg_types &= ~(SLJIT_ARG_FULL_MASK << 3 * SLJIT_ARG_SHIFT);
					}
				}

				args_size += SSIZE_OF(sw);
				break;
			}
			types >>= SLJIT_ARG_SHIFT;
		}

		args_size -= SSIZE_OF(sw);
		compiler->args_size = args_size;
	}

	size = (scratches > 9 ? (scratches - 9) : 0) + (saveds <= 3 ? saveds : 3) - kept_saveds_count;
	if (!(options & SLJIT_ENTER_REG_ARG))
		size++;

	if (size != 0) {
		inst = (sljit_u8*)ensure_buf(compiler, (sljit_uw)(size + 1));
		FAIL_IF(!inst);

		INC_SIZE((sljit_uw)size);

		if (!(options & SLJIT_ENTER_REG_ARG))
			PUSH_REG(reg_map[TMP_REG1]);

		if ((saveds > 2 && kept_saveds_count <= 2) || scratches > 9)
			PUSH_REG(reg_map[SLJIT_S2]);
		if ((saveds > 1 && kept_saveds_count <= 1) || scratches > 10)
			PUSH_REG(reg_map[SLJIT_S1]);
		if ((saveds > 0 && kept_saveds_count == 0) || scratches > 11)
			PUSH_REG(reg_map[SLJIT_S0]);

		size *= SSIZE_OF(sw);
	}

	if (status & (ENTER_TMP_TO_R4 | ENTER_TMP_TO_S))
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), args_size + size);

	size += SSIZE_OF(sw);

	local_size = ((SLJIT_LOCALS_OFFSET_BASE + local_size + size + 0xf) & ~0xf) - size;
	compiler->local_size = local_size;

	word_arg_count = 0;
	saved_arg_count = 0;
	args_size = size;
	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			args_size += SSIZE_OF(f64);
			break;
		case SLJIT_ARG_TYPE_F32:
			args_size += SSIZE_OF(f32);
			break;
		default:
			word_arg_count++;
			SLJIT_ASSERT(word_arg_count <= 3 || (word_arg_count == 4 && !(status & (ENTER_TMP_TO_R4 | ENTER_TMP_TO_S))));

			if (arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) {
#ifdef _WIN32
				if (word_arg_count == 3 && local_size > 4 * 4096)
					r2_offset = local_size + args_size;
				else
#endif
					EMIT_MOV(compiler, word_arg_count, 0, SLJIT_MEM1(SLJIT_SP), args_size);

			} else {
				EMIT_MOV(compiler, SLJIT_S0 - saved_arg_count, 0, SLJIT_MEM1(SLJIT_SP), args_size);
				saved_arg_count++;
			}

			args_size += SSIZE_OF(sw);
			break;
		}
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	SLJIT_ASSERT(SLJIT_LOCALS_OFFSET > 0);

#ifdef _WIN32
	SLJIT_ASSERT(r2_offset == -1 || local_size > 4 * 4096);

	if (local_size > 4096) {
		if (local_size <= 4 * 4096) {
			BINARY_IMM32(OR, 0, SLJIT_MEM1(SLJIT_SP), -4096);

			if (local_size > 2 * 4096)
				BINARY_IMM32(OR, 0, SLJIT_MEM1(SLJIT_SP), -4096 * 2);
			if (local_size > 3 * 4096)
				BINARY_IMM32(OR, 0, SLJIT_MEM1(SLJIT_SP), -4096 * 3);
		}
		else {
			if (options & SLJIT_ENTER_REG_ARG) {
				SLJIT_ASSERT(r2_offset == -1);

				inst = (sljit_u8*)ensure_buf(compiler, (sljit_uw)(1 + 1));
				FAIL_IF(!inst);
				INC_SIZE(1);
				PUSH_REG(reg_map[SLJIT_R2]);

				local_size -= SSIZE_OF(sw);
				r2_offset = local_size;
			}

			EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_IMM, local_size >> 12);

			BINARY_IMM32(OR, 0, SLJIT_MEM1(SLJIT_SP), -4096);
			BINARY_IMM32(SUB, 4096, SLJIT_SP, 0);

			inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
			FAIL_IF(!inst);

			INC_SIZE(2);
			inst[0] = LOOP_i8;
			inst[1] = (sljit_u8)-16;
			local_size &= 0xfff;
		}
	}

	if (local_size > 0) {
		BINARY_IMM32(OR, 0, SLJIT_MEM1(SLJIT_SP), -local_size);
		BINARY_IMM32(SUB, local_size, SLJIT_SP, 0);
	}

	if (r2_offset != -1)
		EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), r2_offset);

#else /* !_WIN32 */

	SLJIT_ASSERT(local_size > 0);

	BINARY_IMM32(SUB, local_size, SLJIT_SP, 0);

#endif /* _WIN32 */

	size = SLJIT_LOCALS_OFFSET_BASE - SSIZE_OF(sw);
	kept_saveds_count = SLJIT_R3 - kept_saveds_count;

	while (saved_arg_count > 3) {
		EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), size, kept_saveds_count, 0);
		kept_saveds_count++;
		size -= SSIZE_OF(sw);
		saved_arg_count--;
	}

	if (status & (ENTER_TMP_TO_R4 | ENTER_TMP_TO_S)) {
		if (status & ENTER_TMP_TO_R4)
			size = 2 * SSIZE_OF(sw);

		EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), size, TMP_REG1, 0);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 args_size;

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	arg_types >>= SLJIT_ARG_SHIFT;
	args_size = 0;

	if (!(options & SLJIT_ENTER_REG_ARG)) {
		while (arg_types) {
			switch (arg_types & SLJIT_ARG_MASK) {
			case SLJIT_ARG_TYPE_F64:
				args_size += SSIZE_OF(f64);
				break;
			case SLJIT_ARG_TYPE_F32:
				args_size += SSIZE_OF(f32);
				break;
			default:
				args_size += SSIZE_OF(sw);
				break;
			}
			arg_types >>= SLJIT_ARG_SHIFT;
		}
	}

	compiler->args_size = args_size;

	/* [esp+0] for saving temporaries and for function calls. */

	saveds = (1 + (scratches > 9 ? (scratches - 9) : 0) + (saveds <= 3 ? saveds : 3) - SLJIT_KEPT_SAVEDS_COUNT(options)) * SSIZE_OF(sw);

	/* Saving ebp. */
	if (!(options & SLJIT_ENTER_REG_ARG))
		saveds += SSIZE_OF(sw);

	compiler->local_size = ((SLJIT_LOCALS_OFFSET_BASE + local_size + saveds + 0xf) & ~0xf) - saveds;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_s32 kept_saveds_count = SLJIT_KEPT_SAVEDS_COUNT(compiler->options);
	sljit_s32 local_size, saveds;
	sljit_uw size;
	sljit_u8 *inst;

	size = (sljit_uw)((compiler->scratches > 9 ? (compiler->scratches - 9) : 0) +
		(compiler->saveds <= 3 ? compiler->saveds : 3) - kept_saveds_count);

	local_size = compiler->local_size;

	if (!(compiler->options & SLJIT_ENTER_REG_ARG))
		size++;
	else if (is_return_to && size == 0) {
		local_size += SSIZE_OF(sw);
		is_return_to = 0;
	}

	if (local_size > 0)
		BINARY_IMM32(ADD, local_size, SLJIT_SP, 0);

	if (size == 0)
		return SLJIT_SUCCESS;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);

	INC_SIZE(size);

	saveds = compiler->saveds;

	if ((saveds > 0 && kept_saveds_count == 0) || compiler->scratches > 11)
		POP_REG(reg_map[SLJIT_S0]);
	if ((saveds > 1 && kept_saveds_count <= 1) || compiler->scratches > 10)
		POP_REG(reg_map[SLJIT_S1]);
	if ((saveds > 2 && kept_saveds_count <= 2) || compiler->scratches > 9)
		POP_REG(reg_map[SLJIT_S2]);

	if (!(compiler->options & SLJIT_ENTER_REG_ARG))
		POP_REG(reg_map[TMP_REG1]);

	if (is_return_to)
		BINARY_IMM32(ADD, sizeof(sljit_sw), SLJIT_SP, 0);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	SLJIT_ASSERT(compiler->args_size >= 0);
	SLJIT_ASSERT(compiler->local_size > 0);

	FAIL_IF(emit_stack_frame_release(compiler, 0));

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
	FAIL_IF(!inst);
	INC_SIZE(1);
	RET();
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 src_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if ((src & SLJIT_MEM) || (src > SLJIT_R2 && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options)))) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		CHECK_EXTRA_REGS(src, srcw, (void)0);

		src_r = (compiler->options & SLJIT_ENTER_REG_ARG) ? TMP_REG1 : SLJIT_R1;

		EMIT_MOV(compiler, src_r, 0, src, srcw);
		src = src_r;
		srcw = 0;
	}

	FAIL_IF(emit_stack_frame_release(compiler, 1));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
}

/* --------------------------------------------------------------------- */
/*  Call / return instructions                                           */
/* --------------------------------------------------------------------- */

static sljit_s32 call_get_stack_size(sljit_s32 arg_types, sljit_s32 *word_arg_count_ptr)
{
	sljit_sw stack_size = 0;
	sljit_s32 word_arg_count = 0;

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			stack_size += SSIZE_OF(f64);
			break;
		case SLJIT_ARG_TYPE_F32:
			stack_size += SSIZE_OF(f32);
			break;
		default:
			word_arg_count++;
			stack_size += SSIZE_OF(sw);
			break;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (word_arg_count_ptr)
		*word_arg_count_ptr = word_arg_count;

	if (stack_size <= 4 * SSIZE_OF(sw))
		return 0;

	return ((stack_size - (4 * SSIZE_OF(sw)) + 0xf) & ~0xf);
}

static sljit_s32 call_with_args(struct sljit_compiler *compiler,
	sljit_s32 arg_types, sljit_sw stack_size, sljit_s32 word_arg_count, sljit_s32 keep_tmp1)
{
	sljit_s32 float_arg_count = 0, arg4_reg = 0, arg_offset;
	sljit_u8 *inst;

	if (word_arg_count >= 4) {
		arg4_reg = SLJIT_R0;

		if (!keep_tmp1) {
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), 2 * SSIZE_OF(sw));
			arg4_reg = TMP_REG1;
		}
	}

	if (stack_size > 0)
		BINARY_IMM32(SUB, stack_size, SLJIT_SP, 0);

	arg_offset = 0;
	word_arg_count = 0;
	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			float_arg_count++;
			FAIL_IF(emit_sse2_store(compiler, 0, SLJIT_MEM1(SLJIT_SP), arg_offset, float_arg_count));
			arg_offset += SSIZE_OF(f64);
			break;
		case SLJIT_ARG_TYPE_F32:
			float_arg_count++;
			FAIL_IF(emit_sse2_store(compiler, 1, SLJIT_MEM1(SLJIT_SP), arg_offset, float_arg_count));
			arg_offset += SSIZE_OF(f32);
			break;
		default:
			word_arg_count++;
			EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), arg_offset, (word_arg_count >= 4) ? arg4_reg : word_arg_count, 0);

			if (word_arg_count == 1 && arg4_reg == SLJIT_R0)
				EMIT_MOV(compiler, SLJIT_R0, 0, SLJIT_MEM1(SLJIT_SP), 2 * SSIZE_OF(sw) + stack_size);

			arg_offset += SSIZE_OF(sw);
			break;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 post_call_with_args(struct sljit_compiler *compiler,
	sljit_s32 arg_types, sljit_s32 stack_size)
{
	sljit_u8 *inst;
	sljit_s32 single;

	if (stack_size > 0)
		BINARY_IMM32(ADD, stack_size, SLJIT_SP, 0);

	if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64)
		return SLJIT_SUCCESS;

	single = ((arg_types & SLJIT_ARG_MASK) == SLJIT_ARG_TYPE_F32);

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 3);
	FAIL_IF(!inst);
	INC_SIZE(3);
	inst[0] = single ? FSTPS : FSTPD;
	inst[1] = (0x03 << 3) | 0x04;
	inst[2] = (0x04 << 3) | reg_map[SLJIT_SP];

	return emit_sse2_load(compiler, single, SLJIT_FR0, SLJIT_MEM1(SLJIT_SP), 0);
}

static sljit_s32 tail_call_with_args(struct sljit_compiler *compiler,
	sljit_s32 *extra_space, sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_sw args_size, saved_regs_size;
	sljit_sw types, word_arg_count, float_arg_count;
	sljit_sw stack_size, prev_stack_size, min_size, offset;
	sljit_sw word_arg4_offset;
	sljit_u8 r2_offset = 0;
	sljit_s32 kept_saveds_count = SLJIT_KEPT_SAVEDS_COUNT(compiler->options);
	sljit_u8* inst;

	ADJUST_LOCAL_OFFSET(src, srcw);
	CHECK_EXTRA_REGS(src, srcw, (void)0);

	saved_regs_size = (1 + (compiler->scratches > 9 ? (compiler->scratches - 9) : 0)
		+ (compiler->saveds <= 3 ? compiler->saveds : 3) - kept_saveds_count) * SSIZE_OF(sw);

	word_arg_count = 0;
	float_arg_count = 0;
	arg_types >>= SLJIT_ARG_SHIFT;
	types = 0;
	args_size = 0;

	while (arg_types != 0) {
		types = (types << SLJIT_ARG_SHIFT) | (arg_types & SLJIT_ARG_MASK);

		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			args_size += SSIZE_OF(f64);
			float_arg_count++;
			break;
		case SLJIT_ARG_TYPE_F32:
			args_size += SSIZE_OF(f32);
			float_arg_count++;
			break;
		default:
			word_arg_count++;
			args_size += SSIZE_OF(sw);
			break;
		}
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (args_size <= compiler->args_size) {
		*extra_space = 0;
		stack_size = args_size + SSIZE_OF(sw) + saved_regs_size;

		offset = stack_size + compiler->local_size;

		if (!(src & SLJIT_IMM) && src != SLJIT_R0) {
			if (word_arg_count >= 1) {
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, SLJIT_R0, 0);
				r2_offset = sizeof(sljit_sw);
			}
			EMIT_MOV(compiler, SLJIT_R0, 0, src, srcw);
		}

		while (types != 0) {
			switch (types & SLJIT_ARG_MASK) {
			case SLJIT_ARG_TYPE_F64:
				offset -= SSIZE_OF(f64);
				FAIL_IF(emit_sse2_store(compiler, 0, SLJIT_MEM1(SLJIT_SP), offset, float_arg_count));
				float_arg_count--;
				break;
			case SLJIT_ARG_TYPE_F32:
				offset -= SSIZE_OF(f32);
				FAIL_IF(emit_sse2_store(compiler, 0, SLJIT_MEM1(SLJIT_SP), offset, float_arg_count));
				float_arg_count--;
				break;
			default:
				switch (word_arg_count) {
				case 1:
					offset -= SSIZE_OF(sw);
					if (r2_offset != 0) {
						EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), 0);
						EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R2, 0);
					} else
						EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R0, 0);
					break;
				case 2:
					offset -= SSIZE_OF(sw);
					EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R1, 0);
					break;
				case 3:
					offset -= SSIZE_OF(sw);
					break;
				case 4:
					offset -= SSIZE_OF(sw);
					EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), 2 * SSIZE_OF(sw));
					EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R2, 0);
					break;
				}
				word_arg_count--;
				break;
			}
			types >>= SLJIT_ARG_SHIFT;
		}

		return emit_stack_frame_release(compiler, 0);
	}

	stack_size = args_size + SSIZE_OF(sw);

	if (word_arg_count >= 1 && !(src & SLJIT_IMM) && src != SLJIT_R0) {
		r2_offset = SSIZE_OF(sw);
		stack_size += SSIZE_OF(sw);
	}

	if (word_arg_count >= 3)
		stack_size += SSIZE_OF(sw);

	prev_stack_size = SSIZE_OF(sw) + saved_regs_size;
	min_size = prev_stack_size + compiler->local_size;

	word_arg4_offset = 2 * SSIZE_OF(sw);

	if (stack_size > min_size) {
		BINARY_IMM32(SUB, stack_size - min_size, SLJIT_SP, 0);
		if (src == SLJIT_MEM1(SLJIT_SP))
			srcw += stack_size - min_size;
		word_arg4_offset += stack_size - min_size;
	}
	else
		stack_size = min_size;

	if (word_arg_count >= 3) {
		EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), r2_offset, SLJIT_R2, 0);

		if (word_arg_count >= 4)
			EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), word_arg4_offset);
	}

	if (!(src & SLJIT_IMM) && src != SLJIT_R0) {
		if (word_arg_count >= 1) {
			SLJIT_ASSERT(r2_offset == sizeof(sljit_sw));
			EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), 0, SLJIT_R0, 0);
		}
		EMIT_MOV(compiler, SLJIT_R0, 0, src, srcw);
	}

	/* Restore saved registers. */
	offset = stack_size - 2 * SSIZE_OF(sw);
	EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), offset);

	if (compiler->saveds > 2 || compiler->scratches > 9) {
		offset -= SSIZE_OF(sw);
		EMIT_MOV(compiler, SLJIT_S2, 0, SLJIT_MEM1(SLJIT_SP), offset);
	}
	if ((compiler->saveds > 1 && kept_saveds_count <= 1) || compiler->scratches > 10) {
		offset -= SSIZE_OF(sw);
		EMIT_MOV(compiler, SLJIT_S1, 0, SLJIT_MEM1(SLJIT_SP), offset);
	}
	if ((compiler->saveds > 0 && kept_saveds_count == 0) || compiler->scratches > 11) {
		offset -= SSIZE_OF(sw);
		EMIT_MOV(compiler, SLJIT_S0, 0, SLJIT_MEM1(SLJIT_SP), offset);
	}

	/* Copy fourth argument and return address. */
	offset = stack_size - SSIZE_OF(sw);
	*extra_space = args_size;

	if (word_arg_count >= 4) {
		offset -= SSIZE_OF(sw);
		EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R2, 0);
	}

	while (types != 0) {
		switch (types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			offset -= SSIZE_OF(f64);
			FAIL_IF(emit_sse2_store(compiler, 0, SLJIT_MEM1(SLJIT_SP), offset, float_arg_count));
			float_arg_count--;
			break;
		case SLJIT_ARG_TYPE_F32:
			offset -= SSIZE_OF(f32);
			FAIL_IF(emit_sse2_store(compiler, 0, SLJIT_MEM1(SLJIT_SP), offset, float_arg_count));
			float_arg_count--;
			break;
		default:
			switch (word_arg_count) {
			case 1:
				offset -= SSIZE_OF(sw);
				if (r2_offset != 0) {
					EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), 0);
					EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R2, 0);
				} else
					EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R0, 0);
				break;
			case 2:
				offset -= SSIZE_OF(sw);
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R1, 0);
				break;
			case 3:
				offset -= SSIZE_OF(sw);
				EMIT_MOV(compiler, SLJIT_R2, 0, SLJIT_MEM1(SLJIT_SP), r2_offset);
				EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, SLJIT_R2, 0);
				break;
			}
			word_arg_count--;
			break;
		}
		types >>= SLJIT_ARG_SHIFT;
	}

	SLJIT_ASSERT(offset >= 0);

	if (offset == 0)
		return SLJIT_SUCCESS;

	BINARY_IMM32(ADD, offset, SLJIT_SP, 0);
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_tail_call_end(struct sljit_compiler *compiler, sljit_s32 extra_space)
{
	/* Called when stack consumption cannot be reduced to 0. */
	sljit_u8 *inst;

	BINARY_IMM32(ADD, extra_space, SLJIT_SP, 0);

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
	FAIL_IF(!inst);
	INC_SIZE(1);
	RET();

	return SLJIT_SUCCESS;
}

static sljit_s32 tail_call_reg_arg_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types)
{
	sljit_s32 word_arg_count = 0;
	sljit_s32 kept_saveds_count, offset;

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64)
			word_arg_count++;

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (word_arg_count < 4)
		return SLJIT_SUCCESS;

	EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), 2 * SSIZE_OF(sw));

	kept_saveds_count = SLJIT_KEPT_SAVEDS_COUNT(compiler->options);
	offset = compiler->local_size + 3 * SSIZE_OF(sw);

	if ((compiler->saveds > 0 && kept_saveds_count == 0) || compiler->scratches > 11)
		offset += SSIZE_OF(sw);
	if ((compiler->saveds > 1 && kept_saveds_count <= 1) || compiler->scratches > 10)
		offset += SSIZE_OF(sw);
	if ((compiler->saveds > 2 && kept_saveds_count <= 2) || compiler->scratches > 9)
		offset += SSIZE_OF(sw);

	return emit_mov(compiler, SLJIT_MEM1(SLJIT_SP), offset, TMP_REG1, 0);
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	struct sljit_jump *jump;
	sljit_sw stack_size = 0;
	sljit_s32 word_arg_count;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	if (type & SLJIT_CALL_RETURN) {
		if ((type & 0xff) == SLJIT_CALL_REG_ARG) {
			PTR_FAIL_IF(tail_call_reg_arg_with_args(compiler, arg_types));
			PTR_FAIL_IF(emit_stack_frame_release(compiler, 0));

			SLJIT_SKIP_CHECKS(compiler);
			return sljit_emit_jump(compiler, SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP));
		}

		stack_size = type;
		PTR_FAIL_IF(tail_call_with_args(compiler, &stack_size, arg_types, SLJIT_IMM, 0));

		SLJIT_SKIP_CHECKS(compiler);

		if (stack_size == 0)
			return sljit_emit_jump(compiler, SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP));

		jump = sljit_emit_jump(compiler, type);
		PTR_FAIL_IF(jump == NULL);

		PTR_FAIL_IF(emit_tail_call_end(compiler, stack_size));
		return jump;
	}

	if ((type & 0xff) == SLJIT_CALL_REG_ARG) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_jump(compiler, type);
	}

	stack_size = call_get_stack_size(arg_types, &word_arg_count);
	PTR_FAIL_IF(call_with_args(compiler, arg_types, stack_size, word_arg_count, 0));

	SLJIT_SKIP_CHECKS(compiler);
	jump = sljit_emit_jump(compiler, type);
	PTR_FAIL_IF(jump == NULL);

	PTR_FAIL_IF(post_call_with_args(compiler, arg_types, stack_size));
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_sw stack_size = 0;
	sljit_s32 word_arg_count;

	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	if (type & SLJIT_CALL_RETURN) {
		if ((type & 0xff) == SLJIT_CALL_REG_ARG) {
			FAIL_IF(tail_call_reg_arg_with_args(compiler, arg_types));

			if ((src & SLJIT_MEM) || (src > SLJIT_R2 && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options)))) {
				ADJUST_LOCAL_OFFSET(src, srcw);
				CHECK_EXTRA_REGS(src, srcw, (void)0);

				EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
				src = TMP_REG1;
				srcw = 0;
			}

			FAIL_IF(emit_stack_frame_release(compiler, 0));

			SLJIT_SKIP_CHECKS(compiler);
			return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
		}

		stack_size = type;
		FAIL_IF(tail_call_with_args(compiler, &stack_size, arg_types, src, srcw));

		if (!(src & SLJIT_IMM)) {
			src = SLJIT_R0;
			srcw = 0;
		}

		SLJIT_SKIP_CHECKS(compiler);

		if (stack_size == 0)
			return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);

		FAIL_IF(sljit_emit_ijump(compiler, type, src, srcw));
		return emit_tail_call_end(compiler, stack_size);
	}

	if ((type & 0xff) == SLJIT_CALL_REG_ARG) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_ijump(compiler, type, src, srcw);
	}

	ADJUST_LOCAL_OFFSET(src, srcw);
	CHECK_EXTRA_REGS(src, srcw, (void)0);

	if (src & SLJIT_MEM) {
		EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
		src = TMP_REG1;
		srcw = 0;
	}

	stack_size = call_get_stack_size(arg_types, &word_arg_count);
	FAIL_IF(call_with_args(compiler, arg_types, stack_size, word_arg_count, src == TMP_REG1));

	if (stack_size > 0 && src == SLJIT_MEM1(SLJIT_SP))
		srcw += stack_size;

	SLJIT_SKIP_CHECKS(compiler);
	FAIL_IF(sljit_emit_ijump(compiler, type, src, srcw));

	return post_call_with_args(compiler, arg_types, stack_size);
}

static SLJIT_INLINE sljit_s32 emit_fmov_before_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;

	if (compiler->options & SLJIT_ENTER_REG_ARG) {
		if (src == SLJIT_FR0)
			return SLJIT_SUCCESS;

		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_fop1(compiler, op, SLJIT_RETURN_FREG, 0, src, srcw);
	}

	if (FAST_IS_REG(src)) {
		FAIL_IF(emit_sse2_store(compiler, op & SLJIT_32, SLJIT_MEM1(SLJIT_SP), 0, src));

		src = SLJIT_MEM1(SLJIT_SP);
		srcw = 0;
	} else {
		ADJUST_LOCAL_OFFSET(src, srcw);
	}

	inst = emit_x86_instruction(compiler, 1 | EX86_SSE2_OP1, 0, 0, src, srcw);
	*inst = (op & SLJIT_32) ? FLDS : FLDL;

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	CHECK_EXTRA_REGS(dst, dstw, (void)0);

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

static sljit_s32 emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst;

	CHECK_EXTRA_REGS(src, srcw, (void)0);

	if (FAST_IS_REG(src)) {
		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + 1);
		FAIL_IF(!inst);

		INC_SIZE(1 + 1);
		PUSH_REG(reg_map[src]);
	}
	else {
		inst = emit_x86_instruction(compiler, 1, 0, 0, src, srcw);
		FAIL_IF(!inst);
		*inst++ = GROUP_FF;
		*inst |= PUSH_rm;

		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
	}

	RET();
	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  Other operations                                                     */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_u8* inst;
	sljit_s32 i, next, reg_idx, offset;
	sljit_u8 regs[2];

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK))
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);

	ADJUST_LOCAL_OFFSET(mem, memw);

	regs[0] = U8(REG_PAIR_FIRST(reg));
	regs[1] = U8(REG_PAIR_SECOND(reg));

	next = SSIZE_OF(sw);

	if (!(type & SLJIT_MEM_STORE) && (regs[0] == (mem & REG_MASK) || regs[0] == OFFS_REG(mem))) {
		if (regs[1] == (mem & REG_MASK) || regs[1] == OFFS_REG(mem)) {
			/* None of them are virtual register so TMP_REG1 will not be used. */
			EMIT_MOV(compiler, TMP_REG1, 0, OFFS_REG(mem), 0);

			if (regs[1] == OFFS_REG(mem))
				next = -SSIZE_OF(sw);

			mem = (mem & ~OFFS_REG_MASK) | TO_OFFS_REG(TMP_REG1);
		} else {
			next = -SSIZE_OF(sw);

			if (!(mem & OFFS_REG_MASK))
				memw += SSIZE_OF(sw);
		}
	}

	for (i = 0; i < 2; i++) {
		reg_idx = next > 0 ? i : (i ^ 0x1);
		reg = regs[reg_idx];

		offset = -1;

		if (reg >= SLJIT_R3 && reg <= SLJIT_S3) {
			offset = (2 * SSIZE_OF(sw)) + ((reg) - SLJIT_R3) * SSIZE_OF(sw);
			reg = TMP_REG1;

			if (type & SLJIT_MEM_STORE)
				EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), offset);
		}

		if ((mem & OFFS_REG_MASK) && (reg_idx == 1)) {
			inst = (sljit_u8*)ensure_buf(compiler, (sljit_uw)(1 + 4));
			FAIL_IF(!inst);

			INC_SIZE(4);

			inst[0] = (type & SLJIT_MEM_STORE) ? MOV_rm_r : MOV_r_rm;
			inst[1] = 0x44 | U8(reg_map[reg] << 3);
			inst[2] = U8(memw << 6) | U8(reg_map[OFFS_REG(mem)] << 3) | reg_map[mem & REG_MASK];
			inst[3] = sizeof(sljit_sw);
		} else if (type & SLJIT_MEM_STORE) {
			EMIT_MOV(compiler, mem, memw, reg, 0);
		} else {
			EMIT_MOV(compiler, reg, 0, mem, memw);
		}

		if (!(mem & OFFS_REG_MASK))
			memw += next;

		if (!(type & SLJIT_MEM_STORE) && offset != -1)
			EMIT_MOV(compiler, SLJIT_MEM1(SLJIT_SP), offset, TMP_REG1, 0);
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 skip_frames_before_return(struct sljit_compiler *compiler)
{
	sljit_sw size;

	/* Don't adjust shadow stack if it isn't enabled.  */
	if (!cpu_has_shadow_stack())
		return SLJIT_SUCCESS;

	SLJIT_ASSERT(compiler->args_size >= 0);
	SLJIT_ASSERT(compiler->local_size > 0);

	size = compiler->local_size;
	size += (1 + (compiler->scratches > 9 ? (compiler->scratches - 9) : 0)
		+ (compiler->saveds <= 3 ? compiler->saveds : 3)) * SSIZE_OF(sw);

	return adjust_shadow_stack(compiler, SLJIT_MEM1(SLJIT_SP), size);
}
