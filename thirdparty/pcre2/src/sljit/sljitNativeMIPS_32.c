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

/* mips 32-bit arch dependent functions. */

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_ar, sljit_sw imm)
{
	if (!(imm & ~0xffff))
		return push_inst(compiler, ORI | SA(0) | TA(dst_ar) | IMM(imm), dst_ar);

	if (imm < 0 && imm >= SIMM_MIN)
		return push_inst(compiler, ADDIU | SA(0) | TA(dst_ar) | IMM(imm), dst_ar);

	FAIL_IF(push_inst(compiler, LUI | TA(dst_ar) | IMM(imm >> 16), dst_ar));
	return (imm & 0xffff) ? push_inst(compiler, ORI | SA(dst_ar) | TA(dst_ar) | IMM(imm), dst_ar) : SLJIT_SUCCESS;
}

#define EMIT_LOGICAL(op_imm, op_norm) \
	if (flags & SRC2_IMM) { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_imm | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_imm | S(src1) | T(dst) | IMM(src2), DR(dst))); \
	} \
	else { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_norm | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_norm | S(src1) | T(src2) | D(dst), DR(dst))); \
	}

#define EMIT_SHIFT(op_imm, op_v) \
	if (flags & SRC2_IMM) { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_imm | T(src1) | DA(EQUAL_FLAG) | SH_IMM(src2), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_imm | T(src1) | D(dst) | SH_IMM(src2), DR(dst))); \
	} \
	else { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_v | S(src2) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_v | S(src2) | T(src1) | D(dst), DR(dst))); \
	}

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_s32 is_overflow, is_carry, is_handled;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return push_inst(compiler, ADDU | S(src2) | TA(0) | D(dst), DR(dst));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S8) {
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
				return push_inst(compiler, SEB | T(src2) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 1 */
				FAIL_IF(push_inst(compiler, SLL | T(src2) | D(dst) | SH_IMM(24), DR(dst)));
				return push_inst(compiler, SRA | T(dst) | D(dst) | SH_IMM(24), DR(dst));
#endif /* SLJIT_MIPS_REV >= 1 */
			}
			return push_inst(compiler, ANDI | S(src2) | T(dst) | IMM(0xff), DR(dst));
		}
		else {
			SLJIT_ASSERT(dst == src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S16) {
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
				return push_inst(compiler, SEH | T(src2) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 1 */
				FAIL_IF(push_inst(compiler, SLL | T(src2) | D(dst) | SH_IMM(16), DR(dst)));
				return push_inst(compiler, SRA | T(dst) | D(dst) | SH_IMM(16), DR(dst));
#endif /* SLJIT_MIPS_REV >= 1 */
			}
			return push_inst(compiler, ANDI | S(src2) | T(dst) | IMM(0xffff), DR(dst));
		}
		else {
			SLJIT_ASSERT(dst == src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_NOT:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, NOR | S(src2) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
		if (!(flags & UNUSED_DEST))
			FAIL_IF(push_inst(compiler, NOR | S(src2) | T(src2) | D(dst), DR(dst)));
		return SLJIT_SUCCESS;

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, CLZ | S(src2) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
		if (!(flags & UNUSED_DEST))
			FAIL_IF(push_inst(compiler, CLZ | S(src2) | T(dst) | D(dst), DR(dst)));
#else /* SLJIT_MIPS_REV < 1 */
		if (SLJIT_UNLIKELY(flags & UNUSED_DEST)) {
			FAIL_IF(push_inst(compiler, SRL | T(src2) | DA(EQUAL_FLAG) | SH_IMM(31), EQUAL_FLAG));
			return push_inst(compiler, XORI | SA(EQUAL_FLAG) | TA(EQUAL_FLAG) | IMM(1), EQUAL_FLAG);
		}
		/* Nearly all instructions are unmovable in the following sequence. */
		FAIL_IF(push_inst(compiler, ADDU | S(src2) | TA(0) | D(TMP_REG1), DR(TMP_REG1)));
		/* Check zero. */
		FAIL_IF(push_inst(compiler, BEQ | S(TMP_REG1) | TA(0) | IMM(5), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, ORI | SA(0) | T(dst) | IMM(32), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(dst) | IMM(-1), DR(dst)));
		/* Loop for searching the highest bit. */
		FAIL_IF(push_inst(compiler, ADDIU | S(dst) | T(dst) | IMM(1), DR(dst)));
		FAIL_IF(push_inst(compiler, BGEZ | S(TMP_REG1) | IMM(-2), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, SLL | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(1), UNMOVABLE_INS));
#endif /* SLJIT_MIPS_REV >= 1 */
		return SLJIT_SUCCESS;

	case SLJIT_ADD:
		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, OR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
				else
					FAIL_IF(push_inst(compiler, NOR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, ADDIU | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));

			if (is_overflow || is_carry) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, ORI | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
				else {
					FAIL_IF(push_inst(compiler, ADDIU | SA(0) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
					FAIL_IF(push_inst(compiler, OR | S(src1) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
				}
			}
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, ADDIU | S(src1) | T(dst) | IMM(src2), DR(dst)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, ADDU | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, OR | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, ADDU | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		/* a + b >= a | b (otherwise, the carry should be set to 1). */
		if (is_overflow || is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		if (!is_overflow)
			return SLJIT_SUCCESS;
		FAIL_IF(push_inst(compiler, SLL | TA(OTHER_FLAG) | D(TMP_REG1) | SH_IMM(31), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XOR | S(TMP_REG1) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, XOR | S(dst) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, ADDU | S(dst) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
		return push_inst(compiler, SRL | TA(OTHER_FLAG) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG);

	case SLJIT_ADDC:
		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_carry) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, ORI | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));
				else {
					FAIL_IF(push_inst(compiler, ADDIU | SA(0) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));
					FAIL_IF(push_inst(compiler, OR | S(src1) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
				}
			}
			FAIL_IF(push_inst(compiler, ADDIU | S(src1) | T(dst) | IMM(src2), DR(dst)));
		} else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, OR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			/* dst may be the same as src1 or src2. */
			FAIL_IF(push_inst(compiler, ADDU | S(src1) | T(src2) | D(dst), DR(dst)));
		}
		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));

		FAIL_IF(push_inst(compiler, ADDU | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
		if (!is_carry)
			return SLJIT_SUCCESS;

		/* Set ULESS_FLAG (dst == 0) && (OTHER_FLAG == 1). */
		FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		/* Set carry flag. */
		return push_inst(compiler, OR | SA(OTHER_FLAG) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

	case SLJIT_SUB:
		if ((flags & SRC2_IMM) && src2 == SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(TMP_REG2) | IMM(src2), DR(TMP_REG2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_handled = 0;

		if (flags & SRC2_IMM) {
			if (GET_FLAG_TYPE(op) == SLJIT_LESS || GET_FLAG_TYPE(op) == SLJIT_GREATER_EQUAL) {
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
				is_handled = 1;
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_SIG_LESS || GET_FLAG_TYPE(op) == SLJIT_SIG_GREATER_EQUAL) {
				FAIL_IF(push_inst(compiler, SLTI | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
				is_handled = 1;
			}
		}

		if (!is_handled && GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL) {
			is_handled = 1;

			if (flags & SRC2_IMM) {
				FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(TMP_REG2) | IMM(src2), DR(TMP_REG2)));
				src2 = TMP_REG2;
				flags &= ~SRC2_IMM;
			}

			if (GET_FLAG_TYPE(op) == SLJIT_LESS || GET_FLAG_TYPE(op) == SLJIT_GREATER_EQUAL) {
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_GREATER || GET_FLAG_TYPE(op) == SLJIT_LESS_EQUAL)
			{
				FAIL_IF(push_inst(compiler, SLTU | S(src2) | T(src1) | DA(OTHER_FLAG), OTHER_FLAG));
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_SIG_LESS || GET_FLAG_TYPE(op) == SLJIT_SIG_GREATER_EQUAL) {
				FAIL_IF(push_inst(compiler, SLT | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_SIG_GREATER || GET_FLAG_TYPE(op) == SLJIT_SIG_LESS_EQUAL)
			{
				FAIL_IF(push_inst(compiler, SLT | S(src2) | T(src1) | DA(OTHER_FLAG), OTHER_FLAG));
			}
		}

		if (is_handled) {
			if (flags & SRC2_IMM) {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, ADDIU | S(src1) | TA(EQUAL_FLAG) | IMM(-src2), EQUAL_FLAG));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, ADDIU | S(src1) | T(dst) | IMM(-src2), DR(dst));
			}
			else {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, SUBU | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, SUBU | S(src1) | T(src2) | D(dst), DR(dst));
			}
			return SLJIT_SUCCESS;
		}

		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, OR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
				else
					FAIL_IF(push_inst(compiler, NOR | S(src1) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG));
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, ADDIU | S(src1) | TA(EQUAL_FLAG) | IMM(-src2), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, ADDIU | S(src1) | T(dst) | IMM(-src2), DR(dst)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SUBU | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SUBU | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;
		FAIL_IF(push_inst(compiler, SLL | TA(OTHER_FLAG) | D(TMP_REG1) | SH_IMM(31), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XOR | S(TMP_REG1) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, XOR | S(dst) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, ADDU | S(dst) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
		return push_inst(compiler, SRL | TA(OTHER_FLAG) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG);

	case SLJIT_SUBC:
		if ((flags & SRC2_IMM) && src2 == SIMM_MIN) {
			FAIL_IF(push_inst(compiler, ADDIU | SA(0) | T(TMP_REG2) | IMM(src2), DR(TMP_REG2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));
			/* dst may be the same as src1 or src2. */
			FAIL_IF(push_inst(compiler, ADDIU | S(src1) | T(dst) | IMM(-src2), DR(dst)));
		}
		else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			/* dst may be the same as src1 or src2. */
			FAIL_IF(push_inst(compiler, SUBU | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | D(TMP_REG1), DR(TMP_REG1)));

		FAIL_IF(push_inst(compiler, SUBU | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
		return (is_carry) ? push_inst(compiler, OR | SA(EQUAL_FLAG) | T(TMP_REG1) | DA(OTHER_FLAG), OTHER_FLAG) : SLJIT_SUCCESS;

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & SRC2_IMM));

		if (GET_FLAG_TYPE(op) != SLJIT_MUL_OVERFLOW) {
#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1)
			return push_inst(compiler, MUL | S(src1) | T(src2) | D(dst), DR(dst));
#else /* SLJIT_MIPS_REV < 1 */
			FAIL_IF(push_inst(compiler, MULT | S(src1) | T(src2), MOVABLE_INS));
			return push_inst(compiler, MFLO | D(dst), DR(dst));
#endif /* SLJIT_MIPS_REV >= 1 */
		}

#if (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)
		FAIL_IF(push_inst(compiler, MUL | S(src1) | T(src2) | D(dst), DR(dst)));
		FAIL_IF(push_inst(compiler, MUH | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
#else /* SLJIT_MIPS_REV < 6 */
		FAIL_IF(push_inst(compiler, MULT | S(src1) | T(src2), MOVABLE_INS));
		FAIL_IF(push_inst(compiler, MFHI | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, MFLO | D(dst), DR(dst)));
#endif /* SLJIT_MIPS_REV >= 6 */
		FAIL_IF(push_inst(compiler, SRA | T(dst) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG));
		return push_inst(compiler, SUBU | SA(EQUAL_FLAG) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

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
		EMIT_SHIFT(SLL, SLLV);
		return SLJIT_SUCCESS;

	case SLJIT_LSHR:
		EMIT_SHIFT(SRL, SRLV);
		return SLJIT_SUCCESS;

	case SLJIT_ASHR:
		EMIT_SHIFT(SRA, SRAV);
		return SLJIT_SUCCESS;
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value)
{
	FAIL_IF(push_inst(compiler, LUI | T(dst) | IMM(init_value >> 16), DR(dst)));
	return push_inst(compiler, ORI | S(dst) | T(dst) | IMM(init_value), DR(dst));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 0);
	SLJIT_ASSERT((inst[0] & 0xffe00000) == LUI && (inst[1] & 0xfc000000) == ORI);
	inst[0] = (inst[0] & 0xffff0000) | ((new_target >> 16) & 0xffff);
	inst[1] = (inst[1] & 0xffff0000) | (new_target & 0xffff);
	SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 1);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 2);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, new_constant, executable_offset);
}

static sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_ins *ins_ptr)
{
	sljit_s32 stack_offset = 0;
	sljit_s32 arg_count = 0;
	sljit_s32 float_arg_count = 0;
	sljit_s32 word_arg_count = 0;
	sljit_s32 types = 0;
	sljit_s32 arg_count_save, types_save;
	sljit_ins prev_ins = NOP;
	sljit_ins ins = NOP;
	sljit_u8 offsets[4];

	SLJIT_ASSERT(reg_map[TMP_REG1] == 4 && freg_map[TMP_FREG1] == 12);

	arg_types >>= SLJIT_DEF_SHIFT;

	while (arg_types) {
		types = (types << SLJIT_DEF_SHIFT) | (arg_types & SLJIT_DEF_MASK);

		switch (arg_types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			offsets[arg_count] = (sljit_u8)stack_offset;

			if (word_arg_count == 0 && arg_count <= 1)
				offsets[arg_count] = 254 + arg_count;

			stack_offset += sizeof(sljit_f32);
			arg_count++;
			float_arg_count++;
			break;
		case SLJIT_ARG_TYPE_F64:
			if (stack_offset & 0x7)
				stack_offset += sizeof(sljit_sw);
			offsets[arg_count] = (sljit_u8)stack_offset;

			if (word_arg_count == 0 && arg_count <= 1)
				offsets[arg_count] = 254 + arg_count;

			stack_offset += sizeof(sljit_f64);
			arg_count++;
			float_arg_count++;
			break;
		default:
			offsets[arg_count] = (sljit_u8)stack_offset;
			stack_offset += sizeof(sljit_sw);
			arg_count++;
			word_arg_count++;
			break;
		}

		arg_types >>= SLJIT_DEF_SHIFT;
	}

	/* Stack is aligned to 16 bytes, max two doubles can be placed on the stack. */
	if (stack_offset > 16)
		FAIL_IF(push_inst(compiler, ADDIU | S(SLJIT_SP) | T(SLJIT_SP) | IMM(-16), DR(SLJIT_SP)));

	types_save = types;
	arg_count_save = arg_count;

	while (types) {
		switch (types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			arg_count--;
			if (offsets[arg_count] < 254)
				ins = SWC1 | S(SLJIT_SP) | FT(float_arg_count) | IMM(offsets[arg_count]);
			float_arg_count--;
			break;
		case SLJIT_ARG_TYPE_F64:
			arg_count--;
			if (offsets[arg_count] < 254)
				ins = SDC1 | S(SLJIT_SP) | FT(float_arg_count) | IMM(offsets[arg_count]);
			float_arg_count--;
			break;
		default:
			if (offsets[arg_count - 1] >= 16)
				ins = SW | S(SLJIT_SP) | T(word_arg_count) | IMM(offsets[arg_count - 1]);
			else if (arg_count != word_arg_count)
				ins = ADDU | S(word_arg_count) | TA(0) | DA(4 + (offsets[arg_count - 1] >> 2));
			else if (arg_count == 1)
				ins = ADDU | S(SLJIT_R0) | TA(0) | DA(4);

			arg_count--;
			word_arg_count--;
			break;
		}

		if (ins != NOP) {
			if (prev_ins != NOP)
				FAIL_IF(push_inst(compiler, prev_ins, MOVABLE_INS));
			prev_ins = ins;
			ins = NOP;
		}

		types >>= SLJIT_DEF_SHIFT;
	}

	types = types_save;
	arg_count = arg_count_save;

	while (types) {
		switch (types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			arg_count--;
			if (offsets[arg_count] == 254)
				ins = MOV_S | FMT_S | FS(SLJIT_FR0) | FD(TMP_FREG1);
			else if (offsets[arg_count] < 16)
				ins = LW | S(SLJIT_SP) | TA(4 + (offsets[arg_count] >> 2)) | IMM(offsets[arg_count]);
			break;
		case SLJIT_ARG_TYPE_F64:
			arg_count--;
			if (offsets[arg_count] == 254)
				ins = MOV_S | FMT_D | FS(SLJIT_FR0) | FD(TMP_FREG1);
			else if (offsets[arg_count] < 16) {
				if (prev_ins != NOP)
					FAIL_IF(push_inst(compiler, prev_ins, MOVABLE_INS));
				prev_ins = LW | S(SLJIT_SP) | TA(4 + (offsets[arg_count] >> 2)) | IMM(offsets[arg_count]);
				ins = LW | S(SLJIT_SP) | TA(5 + (offsets[arg_count] >> 2)) | IMM(offsets[arg_count] + sizeof(sljit_sw));
			}
			break;
		default:
			arg_count--;
			break;
		}

		if (ins != NOP) {
			if (prev_ins != NOP)
				FAIL_IF(push_inst(compiler, prev_ins, MOVABLE_INS));
			prev_ins = ins;
			ins = NOP;
		}

		types >>= SLJIT_DEF_SHIFT;
	}

	*ins_ptr = prev_ins;

	return SLJIT_SUCCESS;
}

static sljit_s32 post_call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types)
{
	sljit_s32 stack_offset = 0;

	arg_types >>= SLJIT_DEF_SHIFT;

	while (arg_types) {
		switch (arg_types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			stack_offset += sizeof(sljit_f32);
			break;
		case SLJIT_ARG_TYPE_F64:
			if (stack_offset & 0x7)
				stack_offset += sizeof(sljit_sw);
			stack_offset += sizeof(sljit_f64);
			break;
		default:
			stack_offset += sizeof(sljit_sw);
			break;
		}

		arg_types >>= SLJIT_DEF_SHIFT;
	}

	/* Stack is aligned to 16 bytes, max two doubles can be placed on the stack. */
	if (stack_offset > 16)
		return push_inst(compiler, ADDIU | S(SLJIT_SP) | T(SLJIT_SP) | IMM(16), DR(SLJIT_SP));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	struct sljit_jump *jump;
	sljit_ins ins;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	PTR_FAIL_IF(call_with_args(compiler, arg_types, &ins));

	SLJIT_ASSERT(DR(PIC_ADDR_REG) == 25 && PIC_ADDR_REG == TMP_REG2);

	PTR_FAIL_IF(emit_const(compiler, PIC_ADDR_REG, 0));

	jump->flags |= IS_JAL | IS_CALL;
	PTR_FAIL_IF(push_inst(compiler, JALR | S(PIC_ADDR_REG) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, ins, UNMOVABLE_INS));

	PTR_FAIL_IF(post_call_with_args(compiler, arg_types));

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	SLJIT_ASSERT(DR(PIC_ADDR_REG) == 25 && PIC_ADDR_REG == TMP_REG2);

	if (src & SLJIT_IMM)
		FAIL_IF(load_immediate(compiler, DR(PIC_ADDR_REG), srcw));
	else if (FAST_IS_REG(src))
		FAIL_IF(push_inst(compiler, ADDU | S(src) | TA(0) | D(PIC_ADDR_REG), DR(PIC_ADDR_REG)));
	else if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, DR(PIC_ADDR_REG), src, srcw));
	}

	FAIL_IF(call_with_args(compiler, arg_types, &ins));

	/* Register input. */
	FAIL_IF(push_inst(compiler, JALR | S(PIC_ADDR_REG) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	FAIL_IF(push_inst(compiler, ins, UNMOVABLE_INS));
	return post_call_with_args(compiler, arg_types);
}
