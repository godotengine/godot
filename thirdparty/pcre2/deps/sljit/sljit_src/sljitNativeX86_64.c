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

/* x86 64-bit arch dependent functions. */

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

static sljit_s32 emit_load_imm64(struct sljit_compiler *compiler, sljit_s32 reg, sljit_sw imm)
{
	sljit_u8 *inst;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2 + sizeof(sljit_sw));
	FAIL_IF(!inst);
	INC_SIZE(2 + sizeof(sljit_sw));
	inst[0] = REX_W | ((reg_map[reg] <= 7) ? 0 : REX_B);
	inst[1] = U8(MOV_r_i32 | reg_lmap[reg]);
	sljit_unaligned_store_sw(inst + 2, imm);
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_do_imm32(struct sljit_compiler *compiler, sljit_u8 rex, sljit_u8 opcode, sljit_sw imm)
{
	sljit_u8 *inst;
	sljit_uw length = (rex ? 2 : 1) + sizeof(sljit_s32);

	inst = (sljit_u8*)ensure_buf(compiler, 1 + length);
	FAIL_IF(!inst);
	INC_SIZE(length);
	if (rex)
		*inst++ = rex;
	*inst++ = opcode;
	sljit_unaligned_store_s32(inst, (sljit_s32)imm);
	return SLJIT_SUCCESS;
}

static sljit_u8* emit_x86_instruction(struct sljit_compiler *compiler, sljit_uw size,
	/* The register or immediate operand. */
	sljit_s32 a, sljit_sw imma,
	/* The general operand (not immediate). */
	sljit_s32 b, sljit_sw immb)
{
	sljit_u8 *inst;
	sljit_u8 *buf_ptr;
	sljit_u8 rex = 0;
	sljit_u8 reg_lmap_b;
	sljit_uw flags = size;
	sljit_uw inst_size;

	/* The immediate operand must be 32 bit. */
	SLJIT_ASSERT(a != SLJIT_IMM || compiler->mode32 || IS_HALFWORD(imma));
	/* Both cannot be switched on. */
	SLJIT_ASSERT((flags & (EX86_BIN_INS | EX86_SHIFT_INS)) != (EX86_BIN_INS | EX86_SHIFT_INS));
	/* Size flags not allowed for typed instructions. */
	SLJIT_ASSERT(!(flags & (EX86_BIN_INS | EX86_SHIFT_INS)) || (flags & (EX86_BYTE_ARG | EX86_HALF_ARG)) == 0);
	/* Both size flags cannot be switched on. */
	SLJIT_ASSERT((flags & (EX86_BYTE_ARG | EX86_HALF_ARG)) != (EX86_BYTE_ARG | EX86_HALF_ARG));
	/* SSE2 and immediate is not possible. */
	SLJIT_ASSERT(a != SLJIT_IMM || !(flags & EX86_SSE2));
	SLJIT_ASSERT(((flags & (EX86_PREF_F2 | EX86_PREF_F3 | EX86_PREF_66))
			& ((flags & (EX86_PREF_F2 | EX86_PREF_F3 | EX86_PREF_66)) - 1)) == 0);
	SLJIT_ASSERT((flags & (EX86_VEX_EXT | EX86_REX)) != EX86_VEX_EXT);

	size &= 0xf;
	/* The mod r/m byte is always present. */
	inst_size = size + 1;

	if (!compiler->mode32 && !(flags & EX86_NO_REXW))
		rex |= REX_W;
	else if (flags & EX86_REX)
		rex |= REX;

	if (flags & (EX86_PREF_F2 | EX86_PREF_F3 | EX86_PREF_66))
		inst_size++;

	/* Calculate size of b. */
	if (b & SLJIT_MEM) {
		if (!(b & OFFS_REG_MASK) && NOT_HALFWORD(immb)) {
			PTR_FAIL_IF(emit_load_imm64(compiler, TMP_REG2, immb));
			immb = 0;
			if (b & REG_MASK)
				b |= TO_OFFS_REG(TMP_REG2);
			else
				b |= TMP_REG2;
		}

		if (!(b & REG_MASK))
			inst_size += 1 + sizeof(sljit_s32); /* SIB byte required to avoid RIP based addressing. */
		else {
			if (immb != 0 && !(b & OFFS_REG_MASK)) {
				/* Immediate operand. */
				if (immb <= 127 && immb >= -128)
					inst_size += sizeof(sljit_s8);
				else
					inst_size += sizeof(sljit_s32);
			} else if (reg_lmap[b & REG_MASK] == 5) {
				/* Swap registers if possible. */
				if ((b & OFFS_REG_MASK) && (immb & 0x3) == 0 && reg_lmap[OFFS_REG(b)] != 5)
					b = SLJIT_MEM | OFFS_REG(b) | TO_OFFS_REG(b & REG_MASK);
				else
					inst_size += sizeof(sljit_s8);
			}

			if (reg_map[b & REG_MASK] >= 8)
				rex |= REX_B;

			if (reg_lmap[b & REG_MASK] == 4 && !(b & OFFS_REG_MASK))
				b |= TO_OFFS_REG(SLJIT_SP);

			if (b & OFFS_REG_MASK) {
				inst_size += 1; /* SIB byte. */
				if (reg_map[OFFS_REG(b)] >= 8)
					rex |= REX_X;
			}
		}
	} else if (!(flags & EX86_SSE2_OP2)) {
		if (reg_map[b] >= 8)
			rex |= REX_B;
	} else if (freg_map[b] >= 8)
		rex |= REX_B;

	if ((flags & EX86_VEX_EXT) && (rex & 0x3)) {
		SLJIT_ASSERT(size == 2);
		size++;
		inst_size++;
	}

	if (a == SLJIT_IMM) {
		if (flags & EX86_BIN_INS) {
			if (imma <= 127 && imma >= -128) {
				inst_size += sizeof(sljit_s8);
				flags |= EX86_BYTE_ARG;
			} else
				inst_size += sizeof(sljit_s32);
		} else if (flags & EX86_SHIFT_INS) {
			SLJIT_ASSERT(imma <= (compiler->mode32 ? 0x1f : 0x3f));
			if (imma != 1) {
				inst_size += sizeof(sljit_s8);
				flags |= EX86_BYTE_ARG;
			}
		} else if (flags & EX86_BYTE_ARG)
			inst_size += sizeof(sljit_s8);
		else if (flags & EX86_HALF_ARG)
			inst_size += sizeof(sljit_s16);
		else
			inst_size += sizeof(sljit_s32);
	} else {
		SLJIT_ASSERT(!(flags & EX86_SHIFT_INS) || a == SLJIT_PREF_SHIFT_REG);
		/* reg_map[SLJIT_PREF_SHIFT_REG] is less than 8. */
		if (!(flags & EX86_SSE2_OP1)) {
			if (reg_map[a] >= 8)
				rex |= REX_R;
		}
		else if (freg_map[a] >= 8)
			rex |= REX_R;
	}

	if (rex)
		inst_size++;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + inst_size);
	PTR_FAIL_IF(!inst);

	/* Encoding prefixes. */
	INC_SIZE(inst_size);
	if (flags & EX86_PREF_F2)
		*inst++ = 0xf2;
	else if (flags & EX86_PREF_F3)
		*inst++ = 0xf3;
	else if (flags & EX86_PREF_66)
		*inst++ = 0x66;

	/* Rex is always the last prefix. */
	if (rex)
		*inst++ = rex;

	buf_ptr = inst + size;

	/* Encode mod/rm byte. */
	if (!(flags & EX86_SHIFT_INS)) {
		if ((flags & EX86_BIN_INS) && a == SLJIT_IMM)
			*inst = (flags & EX86_BYTE_ARG) ? GROUP_BINARY_83 : GROUP_BINARY_81;

		if (a == SLJIT_IMM)
			*buf_ptr = 0;
		else if (!(flags & EX86_SSE2_OP1))
			*buf_ptr = U8(reg_lmap[a] << 3);
		else
			*buf_ptr = U8(freg_lmap[a] << 3);
	} else {
		if (a == SLJIT_IMM) {
			if (imma == 1)
				*inst = GROUP_SHIFT_1;
			else
				*inst = GROUP_SHIFT_N;
		} else
			*inst = GROUP_SHIFT_CL;
		*buf_ptr = 0;
	}

	if (!(b & SLJIT_MEM)) {
		*buf_ptr = U8(*buf_ptr | MOD_REG | (!(flags & EX86_SSE2_OP2) ? reg_lmap[b] : freg_lmap[b]));
		buf_ptr++;
	} else if (b & REG_MASK) {
		reg_lmap_b = reg_lmap[b & REG_MASK];

		if (!(b & OFFS_REG_MASK) || (b & OFFS_REG_MASK) == TO_OFFS_REG(SLJIT_SP)) {
			if (immb != 0 || reg_lmap_b == 5) {
				if (immb <= 127 && immb >= -128)
					*buf_ptr |= 0x40;
				else
					*buf_ptr |= 0x80;
			}

			if (!(b & OFFS_REG_MASK))
				*buf_ptr++ |= reg_lmap_b;
			else {
				buf_ptr[0] |= 0x04;
				buf_ptr[1] = U8(reg_lmap_b | (reg_lmap[OFFS_REG(b)] << 3));
				buf_ptr += 2;
			}

			if (immb != 0 || reg_lmap_b == 5) {
				if (immb <= 127 && immb >= -128)
					*buf_ptr++ = U8(immb); /* 8 bit displacement. */
				else {
					sljit_unaligned_store_s32(buf_ptr, (sljit_s32)immb); /* 32 bit displacement. */
					buf_ptr += sizeof(sljit_s32);
				}
			}
		} else {
			if (reg_lmap_b == 5)
				*buf_ptr |= 0x40;

			buf_ptr[0] |= 0x04;
			buf_ptr[1] = U8(reg_lmap_b | (reg_lmap[OFFS_REG(b)] << 3) | (immb << 6));
			buf_ptr += 2;

			if (reg_lmap_b == 5)
				*buf_ptr++ = 0;
		}
	} else {
		buf_ptr[0] |= 0x04;
		buf_ptr[1] = 0x25;
		buf_ptr += 2;
		sljit_unaligned_store_s32(buf_ptr, (sljit_s32)immb); /* 32 bit displacement. */
		buf_ptr += sizeof(sljit_s32);
	}

	if (a == SLJIT_IMM) {
		if (flags & EX86_BYTE_ARG)
			*buf_ptr = U8(imma);
		else if (flags & EX86_HALF_ARG)
			sljit_unaligned_store_s16(buf_ptr, (sljit_s16)imma);
		else if (!(flags & EX86_SHIFT_INS))
			sljit_unaligned_store_s32(buf_ptr, (sljit_s32)imma);
	}

	return inst;
}

static sljit_s32 emit_vex_instruction(struct sljit_compiler *compiler, sljit_uw op,
	/* The first and second register operand. */
	sljit_s32 a, sljit_s32 v,
	/* The general operand (not immediate). */
	sljit_s32 b, sljit_sw immb)
{
	sljit_u8 *inst;
	sljit_u8 vex = 0;
	sljit_u8 vex_m = 0;
	sljit_uw size;

	SLJIT_ASSERT(((op & (EX86_PREF_F2 | EX86_PREF_F3 | EX86_PREF_66))
			& ((op & (EX86_PREF_F2 | EX86_PREF_F3 | EX86_PREF_66)) - 1)) == 0);

	op |= EX86_REX;

	if (op & VEX_OP_0F38)
		vex_m = 0x2;
	else if (op & VEX_OP_0F3A)
		vex_m = 0x3;

	if ((op & VEX_W) || ((op & VEX_AUTO_W) && !compiler->mode32)) {
		if (vex_m == 0)
			vex_m = 0x1;

		vex |= 0x80;
	}

	if (op & EX86_PREF_66)
		vex |= 0x1;
	else if (op & EX86_PREF_F2)
		vex |= 0x3;
	else if (op & EX86_PREF_F3)
		vex |= 0x2;

	op &= ~(EX86_PREF_66 | EX86_PREF_F2 | EX86_PREF_F3);

	if (op & VEX_256)
		vex |= 0x4;

	vex = U8(vex | ((((op & VEX_SSE2_OPV) ? freg_map[v] : reg_map[v]) ^ 0xf) << 3));

	size = op & ~(sljit_uw)0xff;
	size |= (vex_m == 0) ? (EX86_VEX_EXT | 2) : 3;

	inst = emit_x86_instruction(compiler, size, a, 0, b, immb);
	FAIL_IF(!inst);

	SLJIT_ASSERT((inst[-1] & 0xf0) == REX);

	/* If X or B is present in REX prefix. */
	if (vex_m == 0 && inst[-1] & 0x3)
		vex_m = 0x1;

	if (vex_m == 0) {
		vex |= U8(((inst[-1] >> 2) ^ 0x1) << 7);

		inst[-1] = 0xc5;
		inst[0] = vex;
		inst[1] = U8(op);
		return SLJIT_SUCCESS;
	}

	vex_m |= U8((inst[-1] ^ 0x7) << 5);
	inst[-1] = 0xc4;
	inst[0] = vex_m;
	inst[1] = vex;
	inst[2] = U8(op);
	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  Enter / return                                                       */
/* --------------------------------------------------------------------- */

static sljit_u8* detect_far_jump_type(struct sljit_jump *jump, sljit_u8 *code_ptr)
{
	sljit_uw type = jump->flags >> TYPE_SHIFT;

	int short_addr = ((jump->flags & (SLJIT_REWRITABLE_JUMP | JUMP_ADDR)) == JUMP_ADDR) && (jump->u.target <= 0xffffffff);

	/* The relative jump below specialized for this case. */
	SLJIT_ASSERT(reg_map[TMP_REG2] >= 8 && TMP_REG2 != SLJIT_TMP_DEST_REG);

	if (type < SLJIT_JUMP) {
		/* Invert type. */
		code_ptr[0] = U8(get_jump_code(type ^ 0x1) - 0x10);
		code_ptr[1] = short_addr ? (6 + 3) : (10 + 3);
		code_ptr += 2;
	}

	code_ptr[0] = short_addr ? REX_B : (REX_W | REX_B);
	code_ptr[1] = MOV_r_i32 | reg_lmap[TMP_REG2];
	code_ptr += 2;
	jump->addr = (sljit_uw)code_ptr;

	if (!(jump->flags & JUMP_ADDR))
		jump->flags |= PATCH_MD;
	else if (short_addr)
		sljit_unaligned_store_s32(code_ptr, (sljit_s32)jump->u.target);
	else
		sljit_unaligned_store_sw(code_ptr, (sljit_sw)jump->u.target);

	code_ptr += short_addr ? sizeof(sljit_s32) : sizeof(sljit_sw);

	code_ptr[0] = REX_B;
	code_ptr[1] = GROUP_FF;
	code_ptr[2] = U8(MOD_REG | (type >= SLJIT_FAST_CALL ? CALL_rm : JMP_rm) | reg_lmap[TMP_REG2]);

	return code_ptr + 3;
}

static sljit_u8* generate_mov_addr_code(struct sljit_jump *jump, sljit_u8 *code_ptr, sljit_u8 *code, sljit_sw executable_offset)
{
	sljit_uw addr;
	sljit_sw diff;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_ASSERT(((jump->flags >> JUMP_SIZE_SHIFT) & 0x1f) <= 10);
	if (jump->flags & JUMP_ADDR)
		addr = jump->u.target;
	else
		addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code, executable_offset) + jump->u.label->size;

	if (addr > 0xffffffffl) {
		diff = (sljit_sw)addr - (sljit_sw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

		if (diff <= HALFWORD_MAX && diff >= HALFWORD_MIN) {
			SLJIT_ASSERT(((jump->flags >> JUMP_SIZE_SHIFT) & 0x1f) >= 7);
			code_ptr -= SSIZE_OF(s32) - 1;

			SLJIT_ASSERT((code_ptr[-3 - SSIZE_OF(s32)] & 0xf8) == REX_W);
			SLJIT_ASSERT((code_ptr[-2 - SSIZE_OF(s32)] & 0xf8) == MOV_r_i32);

			code_ptr[-3 - SSIZE_OF(s32)] = U8(REX_W | ((code_ptr[-3 - SSIZE_OF(s32)] & 0x1) << 2));
			code_ptr[-1 - SSIZE_OF(s32)] = U8(((code_ptr[-2 - SSIZE_OF(s32)] & 0x7) << 3) | 0x5);
			code_ptr[-2 - SSIZE_OF(s32)] = LEA_r_m;

			jump->flags |= PATCH_MW;
			return code_ptr;
		}

		jump->flags |= PATCH_MD;
		return code_ptr;
	}

	code_ptr -= 2 + sizeof(sljit_uw);

	SLJIT_ASSERT((code_ptr[0] & 0xf8) == REX_W);
	SLJIT_ASSERT((code_ptr[1] & 0xf8) == MOV_r_i32);

	if ((code_ptr[0] & 0x07) != 0) {
		SLJIT_ASSERT(((jump->flags >> JUMP_SIZE_SHIFT) & 0x1f) >= 6);
		code_ptr[0] = U8(code_ptr[0] & ~0x08);
		code_ptr += 2 + sizeof(sljit_s32);
	} else {
		SLJIT_ASSERT(((jump->flags >> JUMP_SIZE_SHIFT) & 0x1f) >= 5);
		code_ptr[0] = code_ptr[1];
		code_ptr += 1 + sizeof(sljit_s32);
	}

	return code_ptr;
}

#ifdef _WIN64
typedef struct {
	sljit_sw regs[2];
} sljit_sse2_reg;
#endif /* _WIN64 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	sljit_uw size;
	sljit_s32 word_arg_count = 0;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);
	sljit_s32 saved_regs_size, tmp, i;
#ifdef _WIN64
	sljit_s32 fscratches;
	sljit_s32 fsaveds;
	sljit_s32 saved_float_regs_size;
	sljit_s32 saved_float_regs_offset = 0;
	sljit_s32 float_arg_count = 0;
#endif /* _WIN64 */
	sljit_u8 *inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, local_size);

	scratches = ENTER_GET_REGS(scratches);
#ifdef _WIN64
	saveds = ENTER_GET_REGS(saveds);
	fscratches = compiler->fscratches;
	fsaveds = compiler->fsaveds;
#endif /* _WIN64 */

	if (options & SLJIT_ENTER_REG_ARG)
		arg_types = 0;

	/* Emit ENDBR64 at function entry if needed.  */
	FAIL_IF(emit_endbranch(compiler));

	compiler->mode32 = 0;

	/* Including the return address saved by the call instruction. */
	saved_regs_size = GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 1);

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		size = reg_map[i] >= 8 ? 2 : 1;
		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
		if (reg_map[i] >= 8)
			*inst++ = REX_B;
		PUSH_REG(reg_lmap[i]);
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		size = reg_map[i] >= 8 ? 2 : 1;
		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
		if (reg_map[i] >= 8)
			*inst++ = REX_B;
		PUSH_REG(reg_lmap[i]);
	}

#ifdef _WIN64
	local_size += SLJIT_LOCALS_OFFSET;
	saved_float_regs_size = GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sse2_reg);

	if (saved_float_regs_size > 0) {
		saved_float_regs_offset = ((local_size + 0xf) & ~0xf);
		local_size = saved_float_regs_offset + saved_float_regs_size;
	}
#else /* !_WIN64 */
	SLJIT_ASSERT(SLJIT_LOCALS_OFFSET == 0);
#endif /* _WIN64 */

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types > 0) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
			tmp = 0;
#ifndef _WIN64
			switch (word_arg_count) {
			case 0:
				tmp = SLJIT_R2;
				break;
			case 1:
				tmp = SLJIT_R1;
				break;
			case 2:
				tmp = TMP_REG1;
				break;
			default:
				tmp = SLJIT_R3;
				break;
			}
#else /* !_WIN64 */
			switch (word_arg_count + float_arg_count) {
			case 0:
				tmp = SLJIT_R3;
				break;
			case 1:
				tmp = SLJIT_R1;
				break;
			case 2:
				tmp = SLJIT_R2;
				break;
			default:
				tmp = TMP_REG1;
				break;
			}
#endif /* _WIN64 */
			if (arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) {
				if (tmp != SLJIT_R0 + word_arg_count)
					EMIT_MOV(compiler, SLJIT_R0 + word_arg_count, 0, tmp, 0);
			} else {
				EMIT_MOV(compiler, SLJIT_S0 - saved_arg_count, 0, tmp, 0);
				saved_arg_count++;
			}
			word_arg_count++;
		} else {
#ifdef _WIN64
			SLJIT_COMPILE_ASSERT(SLJIT_FR0 == 1, float_register_index_start);
			float_arg_count++;
			if (float_arg_count != float_arg_count + word_arg_count)
				FAIL_IF(emit_sse2_load(compiler, (arg_types & SLJIT_ARG_MASK) == SLJIT_ARG_TYPE_F32,
					float_arg_count, float_arg_count + word_arg_count, 0));
#endif /* _WIN64 */
		}
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	local_size = ((local_size + saved_regs_size + 0xf) & ~0xf) - saved_regs_size;
	compiler->local_size = local_size;

#ifdef _WIN64
	if (local_size > 0) {
		if (local_size <= 4 * 4096) {
			if (local_size > 4096)
				EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), -4096);
			if (local_size > 2 * 4096)
				EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), -4096 * 2);
			if (local_size > 3 * 4096)
				EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), -4096 * 3);
		}
		else {
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, local_size >> 12);

			EMIT_MOV(compiler, TMP_REG2, 0, SLJIT_MEM1(SLJIT_SP), -4096);
			BINARY_IMM32(SUB, 4096, SLJIT_SP, 0);
			BINARY_IMM32(SUB, 1, TMP_REG1, 0);

			inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
			FAIL_IF(!inst);

			INC_SIZE(2);
			inst[0] = JNE_i8;
			inst[1] = (sljit_u8)-21;
			local_size &= 0xfff;
		}

		if (local_size > 0)
			EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_MEM1(SLJIT_SP), -local_size);
	}
#endif /* _WIN64 */

	if (local_size > 0)
		BINARY_IMM32(SUB, local_size, SLJIT_SP, 0);

#ifdef _WIN64
	if (saved_float_regs_size > 0) {
		compiler->mode32 = 1;

		tmp = SLJIT_FS0 - fsaveds;
		for (i = SLJIT_FS0; i > tmp; i--) {
			FAIL_IF(emit_groupf(compiler, MOVAPS_xm_x | EX86_SSE2, i, SLJIT_MEM1(SLJIT_SP), saved_float_regs_offset));
			saved_float_regs_offset += 16;
		}

		for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
			FAIL_IF(emit_groupf(compiler, MOVAPS_xm_x | EX86_SSE2, i, SLJIT_MEM1(SLJIT_SP), saved_float_regs_offset));
			saved_float_regs_offset += 16;
		}
	}
#endif /* _WIN64 */

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	sljit_s32 saved_regs_size;
#ifdef _WIN64
	sljit_s32 fscratches;
	sljit_s32 fsaveds;
	sljit_s32 saved_float_regs_size;
#endif /* _WIN64 */

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, local_size);

	scratches = ENTER_GET_REGS(scratches);

#ifdef _WIN64
	saveds = ENTER_GET_REGS(saveds);
	fscratches = compiler->fscratches;
	fsaveds = compiler->fsaveds;

	local_size += SLJIT_LOCALS_OFFSET;
	saved_float_regs_size = GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sse2_reg);

	if (saved_float_regs_size > 0)
		local_size = ((local_size + 0xf) & ~0xf) + saved_float_regs_size;
#else /* !_WIN64 */
	SLJIT_ASSERT(SLJIT_LOCALS_OFFSET == 0);
#endif /* _WIN64 */

	/* Including the return address saved by the call instruction. */
	saved_regs_size = GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 1);
	compiler->local_size = ((local_size + saved_regs_size + 0xf) & ~0xf) - saved_regs_size;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_uw size;
	sljit_s32 local_size, i, tmp;
	sljit_u8 *inst;
#ifdef _WIN64
	sljit_s32 saved_float_regs_offset;
	sljit_s32 fscratches = compiler->fscratches;
	sljit_s32 fsaveds = compiler->fsaveds;
#endif /* _WIN64 */

#ifdef _WIN64
	saved_float_regs_offset = GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, sse2_reg);

	if (saved_float_regs_offset > 0) {
		compiler->mode32 = 1;
		saved_float_regs_offset = (compiler->local_size - saved_float_regs_offset) & ~0xf;

		tmp = SLJIT_FS0 - fsaveds;
		for (i = SLJIT_FS0; i > tmp; i--) {
			FAIL_IF(emit_groupf(compiler, MOVAPS_x_xm | EX86_SSE2, i, SLJIT_MEM1(SLJIT_SP), saved_float_regs_offset));
			saved_float_regs_offset += 16;
		}

		for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
			FAIL_IF(emit_groupf(compiler, MOVAPS_x_xm | EX86_SSE2, i, SLJIT_MEM1(SLJIT_SP), saved_float_regs_offset));
			saved_float_regs_offset += 16;
		}

		compiler->mode32 = 0;
	}
#endif /* _WIN64 */

	local_size = compiler->local_size;

	if (is_return_to && compiler->scratches < SLJIT_FIRST_SAVED_REG && (compiler->saveds == SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		local_size += SSIZE_OF(sw);
		is_return_to = 0;
	}

	if (local_size > 0)
		BINARY_IMM32(ADD, local_size, SLJIT_SP, 0);

	tmp = compiler->scratches;
	for (i = SLJIT_FIRST_SAVED_REG; i <= tmp; i++) {
		size = reg_map[i] >= 8 ? 2 : 1;
		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
		if (reg_map[i] >= 8)
			*inst++ = REX_B;
		POP_REG(reg_lmap[i]);
	}

	tmp = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options);
	for (i = SLJIT_S0 + 1 - compiler->saveds; i <= tmp; i++) {
		size = reg_map[i] >= 8 ? 2 : 1;
		inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
		FAIL_IF(!inst);
		INC_SIZE(size);
		if (reg_map[i] >= 8)
			*inst++ = REX_B;
		POP_REG(reg_lmap[i]);
	}

	if (is_return_to)
		BINARY_IMM32(ADD, sizeof(sljit_sw), SLJIT_SP, 0);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	compiler->mode32 = 0;

	FAIL_IF(emit_stack_frame_release(compiler, 0));
	return emit_byte(compiler, RET_near);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	compiler->mode32 = 0;

	if ((src & SLJIT_MEM) || (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options)))) {
		ADJUST_LOCAL_OFFSET(src, srcw);

		EMIT_MOV(compiler, TMP_REG2, 0, src, srcw);
		src = TMP_REG2;
		srcw = 0;
	}

	FAIL_IF(emit_stack_frame_release(compiler, 1));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
}

/* --------------------------------------------------------------------- */
/*  Call / return instructions                                           */
/* --------------------------------------------------------------------- */

#ifndef _WIN64

static sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_s32 *src_ptr)
{
	sljit_s32 src = src_ptr ? (*src_ptr) : 0;
	sljit_s32 word_arg_count = 0;

	SLJIT_ASSERT(reg_map[SLJIT_R1] == 6 && reg_map[SLJIT_R3] == 1 && reg_map[TMP_REG1] == 2);
	SLJIT_ASSERT(!(src & SLJIT_MEM));

	/* Remove return value. */
	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64)
			word_arg_count++;
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (word_arg_count == 0)
		return SLJIT_SUCCESS;

	if (word_arg_count >= 3 || src == SLJIT_R2) {
		if (src == SLJIT_R2)
			*src_ptr = TMP_REG1;
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_R2, 0);
	}

	return emit_mov(compiler, SLJIT_R2, 0, SLJIT_R0, 0);
}

#else

static sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_s32 *src_ptr)
{
	sljit_s32 src = src_ptr ? (*src_ptr) : 0;
	sljit_s32 arg_count = 0;
	sljit_s32 word_arg_count = 0;
	sljit_s32 float_arg_count = 0;
	sljit_s32 types = 0;
	sljit_s32 data_trandfer = 0;
	static sljit_u8 word_arg_regs[5] = { 0, SLJIT_R3, SLJIT_R1, SLJIT_R2, TMP_REG1 };

	SLJIT_ASSERT(reg_map[SLJIT_R3] == 1 && reg_map[SLJIT_R1] == 2 && reg_map[SLJIT_R2] == 8 && reg_map[TMP_REG1] == 9);
	SLJIT_ASSERT(!(src & SLJIT_MEM));

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		types = (types << SLJIT_ARG_SHIFT) | (arg_types & SLJIT_ARG_MASK);

		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
		case SLJIT_ARG_TYPE_F32:
			arg_count++;
			float_arg_count++;

			if (arg_count != float_arg_count)
				data_trandfer = 1;
			break;
		default:
			arg_count++;
			word_arg_count++;

			if (arg_count != word_arg_count || arg_count != word_arg_regs[arg_count]) {
				data_trandfer = 1;

				if (src == word_arg_regs[arg_count]) {
					EMIT_MOV(compiler, TMP_REG2, 0, src, 0);
					*src_ptr = TMP_REG2;
				}
			}
			break;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (!data_trandfer)
		return SLJIT_SUCCESS;

	while (types) {
		switch (types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			if (arg_count != float_arg_count)
				FAIL_IF(emit_sse2_load(compiler, 0, arg_count, float_arg_count, 0));
			arg_count--;
			float_arg_count--;
			break;
		case SLJIT_ARG_TYPE_F32:
			if (arg_count != float_arg_count)
				FAIL_IF(emit_sse2_load(compiler, 1, arg_count, float_arg_count, 0));
			arg_count--;
			float_arg_count--;
			break;
		default:
			if (arg_count != word_arg_count || arg_count != word_arg_regs[arg_count])
				EMIT_MOV(compiler, word_arg_regs[arg_count], 0, word_arg_count, 0);
			arg_count--;
			word_arg_count--;
			break;
		}

		types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

#endif

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	compiler->mode32 = 0;

	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		PTR_FAIL_IF(call_with_args(compiler, arg_types, NULL));

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	compiler->mode32 = 0;

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		EMIT_MOV(compiler, TMP_REG2, 0, src, srcw);
		src = TMP_REG2;
	}

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
			EMIT_MOV(compiler, TMP_REG2, 0, src, srcw);
			src = TMP_REG2;
		}

		FAIL_IF(emit_stack_frame_release(compiler, 0));
	}

	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		FAIL_IF(call_with_args(compiler, arg_types, &src));

	if (type & SLJIT_CALL_RETURN)
		type = SLJIT_JUMP;

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, type, src, srcw);
}

static sljit_s32 emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	sljit_u8 *inst;

	if (FAST_IS_REG(dst)) {
		if (reg_map[dst] < 8)
			return emit_byte(compiler, U8(POP_r + reg_lmap[dst]));

		inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
		FAIL_IF(!inst);
		INC_SIZE(2);
		*inst++ = REX_B;
		POP_REG(reg_lmap[dst]);
		return SLJIT_SUCCESS;
	}

	/* REX_W is not necessary (src is not immediate). */
	compiler->mode32 = 1;
	inst = emit_x86_instruction(compiler, 1, 0, 0, dst, dstw);
	FAIL_IF(!inst);
	*inst = POP_rm;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 *inst;

	if (FAST_IS_REG(src)) {
		if (reg_map[src] < 8) {
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 1 + 1);
			FAIL_IF(!inst);

			INC_SIZE(1 + 1);
			PUSH_REG(reg_lmap[src]);
		}
		else {
			inst = (sljit_u8*)ensure_buf(compiler, 1 + 2 + 1);
			FAIL_IF(!inst);

			INC_SIZE(2 + 1);
			*inst++ = REX_B;
			PUSH_REG(reg_lmap[src]);
		}
	}
	else {
		/* REX_W is not necessary (src is not immediate). */
		compiler->mode32 = 1;
		inst = emit_x86_instruction(compiler, 1, 0, 0, src, srcw);
		FAIL_IF(!inst);
		inst[0] = GROUP_FF;
		inst[1] |= PUSH_rm;

		inst = (sljit_u8*)ensure_buf(compiler, 1 + 1);
		FAIL_IF(!inst);
		INC_SIZE(1);
	}

	RET();
	return SLJIT_SUCCESS;
}

static sljit_s32 sljit_emit_get_return_address(struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 saved_regs_size;

	compiler->mode32 = 0;
	saved_regs_size = GET_SAVED_REGISTERS_SIZE(compiler->scratches, compiler->saveds - SLJIT_KEPT_SAVEDS_COUNT(compiler->options), 0);
	return emit_mov(compiler, dst, dstw, SLJIT_MEM1(SLJIT_SP), compiler->local_size + saved_regs_size);
}

/* --------------------------------------------------------------------- */
/*  Other operations                                                     */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
	sljit_u8* inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	compiler->mode32 = type & SLJIT_32;

	if (type & SLJIT_COMPARE_SELECT) {
		if (!FAST_IS_REG(src1)) {
			EMIT_MOV(compiler, TMP_REG2, 0, src1, src1w);
			src1 = TMP_REG2;
			src1w = 0;
		}

		inst = emit_x86_instruction(compiler, 1, src1, 0, src2_reg, 0);
		FAIL_IF(!inst);
		*inst = CMP_r_rm;
	}

	type &= ~(SLJIT_32 | SLJIT_COMPARE_SELECT);

	if (dst_reg != src2_reg) {
		if (dst_reg == src1) {
			src1 = src2_reg;
			src1w = 0;
			type ^= 0x1;
		} else if (ADDRESSING_DEPENDS_ON(src1, dst_reg)) {
			EMIT_MOV(compiler, dst_reg, 0, src1, src1w);
			src1 = src2_reg;
			src1w = 0;
			type ^= 0x1;
		} else
			EMIT_MOV(compiler, dst_reg, 0, src2_reg, 0);
	}

	if (sljit_has_cpu_feature(SLJIT_HAS_CMOV)) {
		if (SLJIT_UNLIKELY(src1 == SLJIT_IMM)) {
			EMIT_MOV(compiler, TMP_REG2, 0, src1, src1w);
			src1 = TMP_REG2;
			src1w = 0;
		}

		return emit_groupf(compiler, U8(get_jump_code((sljit_uw)type) - 0x40), dst_reg, src1, src1w);
	}

	return emit_cmov_generic(compiler, type, dst_reg, src1, src1w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_u8* inst;
	sljit_s32 i, next, reg_idx;
	sljit_u8 regs[2];

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK))
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);

	ADJUST_LOCAL_OFFSET(mem, memw);

	compiler->mode32 = 0;

	if ((mem & REG_MASK) == 0) {
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, memw);

		mem = SLJIT_MEM1(TMP_REG1);
		memw = 0;
	} else if (!(mem & OFFS_REG_MASK) && ((memw < HALFWORD_MIN) || (memw > HALFWORD_MAX - SSIZE_OF(sw)))) {
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, memw);

		mem = SLJIT_MEM2(mem & REG_MASK, TMP_REG1);
		memw = 0;
	}

	regs[0] = U8(REG_PAIR_FIRST(reg));
	regs[1] = U8(REG_PAIR_SECOND(reg));

	next = SSIZE_OF(sw);

	if (!(type & SLJIT_MEM_STORE) && (regs[0] == (mem & REG_MASK) || regs[0] == OFFS_REG(mem))) {
		if (regs[1] == (mem & REG_MASK) || regs[1] == OFFS_REG(mem)) {
			/* Base and offset cannot be TMP_REG1. */
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

		if ((mem & OFFS_REG_MASK) && (reg_idx == 1)) {
			inst = (sljit_u8*)ensure_buf(compiler, (sljit_uw)(1 + 5));
			FAIL_IF(!inst);

			INC_SIZE(5);

			inst[0] = U8(REX_W | ((reg_map[reg] >= 8) ? REX_R : 0) | ((reg_map[mem & REG_MASK] >= 8) ? REX_B : 0) | ((reg_map[OFFS_REG(mem)] >= 8) ? REX_X : 0));
			inst[1] = (type & SLJIT_MEM_STORE) ? MOV_rm_r : MOV_r_rm;
			inst[2] = 0x44 | U8(reg_lmap[reg] << 3);
			inst[3] = U8(memw << 6) | U8(reg_lmap[OFFS_REG(mem)] << 3) | reg_lmap[mem & REG_MASK];
			inst[4] = sizeof(sljit_sw);
		} else if (type & SLJIT_MEM_STORE) {
			EMIT_MOV(compiler, mem, memw, reg, 0);
		} else {
			EMIT_MOV(compiler, reg, 0, mem, memw);
		}

		if (!(mem & OFFS_REG_MASK))
			memw += next;
	}

	return SLJIT_SUCCESS;
}

static sljit_s32 emit_mov_int(struct sljit_compiler *compiler, sljit_s32 sign,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8* inst;
	sljit_s32 dst_r;

	compiler->mode32 = 0;

	if (src == SLJIT_IMM) {
		if (FAST_IS_REG(dst)) {
			if (!sign || ((sljit_u32)srcw <= 0x7fffffff))
				return emit_do_imm32(compiler, reg_map[dst] <= 7 ? 0 : REX_B, U8(MOV_r_i32 | reg_lmap[dst]), srcw);

			inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, (sljit_sw)(sljit_s32)srcw, dst, dstw);
			FAIL_IF(!inst);
			*inst = MOV_rm_i32;
			return SLJIT_SUCCESS;
		}
		compiler->mode32 = 1;
		inst = emit_x86_instruction(compiler, 1, SLJIT_IMM, (sljit_sw)(sljit_s32)srcw, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm_i32;
		compiler->mode32 = 0;
		return SLJIT_SUCCESS;
	}

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if ((dst & SLJIT_MEM) && FAST_IS_REG(src))
		dst_r = src;
	else {
		if (sign) {
			inst = emit_x86_instruction(compiler, 1, dst_r, 0, src, srcw);
			FAIL_IF(!inst);
			*inst = MOVSXD_r_rm;
		} else {
			compiler->mode32 = 1;
			EMIT_MOV(compiler, dst_r, 0, src, srcw);
			compiler->mode32 = 0;
		}
	}

	if (dst & SLJIT_MEM) {
		compiler->mode32 = 1;
		inst = emit_x86_instruction(compiler, 1, dst_r, 0, dst, dstw);
		FAIL_IF(!inst);
		*inst = MOV_rm_r;
		compiler->mode32 = 0;
	}

	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_uw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG;
	sljit_u8 *inst, *jump_inst1, *jump_inst2;
	sljit_uw size1, size2;

	compiler->mode32 = 0;

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_U32) {
		if (src != SLJIT_IMM) {
			compiler->mode32 = 1;
			EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
			compiler->mode32 = 0;
		} else
			FAIL_IF(emit_do_imm32(compiler, reg_map[TMP_REG1] <= 7 ? 0 : REX_B, U8(MOV_r_i32 | reg_lmap[TMP_REG1]), srcw));

		FAIL_IF(emit_groupf(compiler, CVTSI2SD_x_rm | EX86_SELECT_F2_F3(op) | EX86_SSE2_OP1, dst_r, TMP_REG1, 0));

		compiler->mode32 = 1;

		if (dst_r == TMP_FREG)
			return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
		return SLJIT_SUCCESS;
	}

	if (!FAST_IS_REG(src)) {
		EMIT_MOV(compiler, TMP_REG1, 0, src, srcw);
		src = TMP_REG1;
	}

	BINARY_IMM32(CMP, 0, src, 0);

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(2);
	inst[0] = JL_i8;
	jump_inst1 = inst;

	size1 = compiler->size;

	compiler->mode32 = 0;
	FAIL_IF(emit_groupf(compiler, CVTSI2SD_x_rm | EX86_SELECT_F2_F3(op) | EX86_SSE2_OP1, dst_r, src, 0));

	inst = (sljit_u8*)ensure_buf(compiler, 1 + 2);
	FAIL_IF(!inst);
	INC_SIZE(2);
	inst[0] = JMP_i8;
	jump_inst2 = inst;

	size2 = compiler->size;

	jump_inst1[1] = U8(size2 - size1);

	if (src != TMP_REG1)
		EMIT_MOV(compiler, TMP_REG1, 0, src, 0);

	EMIT_MOV(compiler, TMP_REG2, 0, src, 0);

	inst = emit_x86_instruction(compiler, 1 | EX86_SHIFT_INS, SLJIT_IMM, 1, TMP_REG1, 0);
	FAIL_IF(!inst);
	inst[1] |= SHR;

	compiler->mode32 = 1;
	BINARY_IMM32(AND, 1, TMP_REG2, 0);

	compiler->mode32 = 0;
	inst = emit_x86_instruction(compiler, 1, TMP_REG1, 0, TMP_REG2, 0);
	FAIL_IF(!inst);
	inst[0] = OR_r_rm;

	FAIL_IF(emit_groupf(compiler, CVTSI2SD_x_rm | EX86_SELECT_F2_F3(op) | EX86_SSE2_OP1, dst_r, TMP_REG1, 0));
	compiler->mode32 = 1;
	FAIL_IF(emit_groupf(compiler, ADDSD_x_xm | EX86_SELECT_F2_F3(op) | EX86_SSE2, dst_r, dst_r, 0));

	jump_inst2[1] = U8(compiler->size - size2);

	if (dst_r == TMP_FREG)
		return emit_sse2_store(compiler, op & SLJIT_32, dst, dstw, TMP_FREG);
	return SLJIT_SUCCESS;
}

static sljit_s32 sljit_emit_fset(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_u8 rex, sljit_s32 is_zero)
{
	sljit_u8 *inst;
	sljit_u32 size;

	if (is_zero) {
		rex = freg_map[freg] >= 8 ? (REX_R | REX_B) : 0;
	} else {
		if (freg_map[freg] >= 8)
			rex |= REX_R;
		if (reg_map[TMP_REG1] >= 8)
			rex |= REX_B;
	}

	size = (rex != 0) ? 5 : 4;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);

	*inst++ = GROUP_66;
	if (rex != 0)
		*inst++ = rex;
	inst[0] = GROUP_0F;

	if (is_zero) {
		inst[1] = PXOR_x_xm;
		inst[2] = U8(freg_lmap[freg] | (freg_lmap[freg] << 3) | MOD_REG);
	} else {
		inst[1] = MOVD_x_rm;
		inst[2] = U8(reg_lmap[TMP_REG1] | (freg_lmap[freg] << 3) | MOD_REG);
	}

	return SLJIT_SUCCESS;
}

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

	if (u.imm != 0) {
		compiler->mode32 = 1;
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, u.imm);
	}

	return sljit_emit_fset(compiler, freg, 0, u.imm == 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset64(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f64 value)
{
	union {
		sljit_sw imm;
		sljit_f64 value;
	} u;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fset64(compiler, freg, value));

	u.value = value;

	if (u.imm != 0) {
		compiler->mode32 = 0;
		EMIT_MOV(compiler, TMP_REG1, 0, SLJIT_IMM, u.imm);
	}

	return sljit_emit_fset(compiler, freg, REX_W, u.imm == 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
	sljit_u8 *inst;
	sljit_u32 size;
	sljit_u8 rex = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fcopy(compiler, op, freg, reg));

	if (!(op & SLJIT_32))
		rex = REX_W;

	if (freg_map[freg] >= 8)
		rex |= REX_R;

	if (reg_map[reg] >= 8)
		rex |= REX_B;

	size = (rex != 0) ? 5 : 4;

	inst = (sljit_u8*)ensure_buf(compiler, 1 + size);
	FAIL_IF(!inst);
	INC_SIZE(size);

	*inst++ = GROUP_66;
	if (rex != 0)
		*inst++ = rex;
	inst[0] = GROUP_0F;
	inst[1] = GET_OPCODE(op) == SLJIT_COPY_TO_F64 ? MOVD_x_rm : MOVD_rm_x;
	inst[2] = U8(reg_lmap[reg] | (freg_lmap[freg] << 3) | MOD_REG);

	return SLJIT_SUCCESS;
}

static sljit_s32 skip_frames_before_return(struct sljit_compiler *compiler)
{
	sljit_s32 tmp, size;

	/* Don't adjust shadow stack if it isn't enabled.  */
	if (!cpu_has_shadow_stack())
		return SLJIT_SUCCESS;

	size = compiler->local_size;
	tmp = compiler->scratches;
	if (tmp >= SLJIT_FIRST_SAVED_REG)
		size += (tmp - SLJIT_FIRST_SAVED_REG + 1) * SSIZE_OF(sw);
	tmp = compiler->saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - compiler->saveds) : SLJIT_FIRST_SAVED_REG;
	if (SLJIT_S0 >= tmp)
		size += (SLJIT_S0 - tmp + 1) * SSIZE_OF(sw);

	return adjust_shadow_stack(compiler, SLJIT_MEM1(SLJIT_SP), size);
}
