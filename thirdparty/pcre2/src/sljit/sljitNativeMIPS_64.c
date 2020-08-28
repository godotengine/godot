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

/* mips 64-bit arch dependent functions. */

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_ar, sljit_sw imm)
{
	sljit_s32 shift = 32;
	sljit_s32 shift2;
	sljit_s32 inv = 0;
	sljit_ins ins;
	sljit_uw uimm;

	if (!(imm & ~0xffff))
		return push_inst(compiler, ORI | SA(0) | TA(dst_ar) | IMM(imm), dst_ar);

	if (imm < 0 && imm >= SIMM_MIN)
		return push_inst(compiler, ADDIU | SA(0) | TA(dst_ar) | IMM(imm), dst_ar);

	if (imm <= 0x7fffffffl && imm >= -0x80000000l) {
		FAIL_IF(push_inst(compiler, LUI | TA(dst_ar) | IMM(imm >> 16), dst_ar));
		return (imm & 0xffff) ? push_inst(compiler, ORI | SA(dst_ar) | TA(dst_ar) | IMM(imm), dst_ar) : SLJIT_SUCCESS;
	}

	/* Zero extended number. */
	uimm = imm;
	if (imm < 0) {
		uimm = ~imm;
		inv = 1;
	}

	while (!(uimm & 0xff00000000000000l)) {
		shift -= 8;
		uimm <<= 8;
	}

	if (!(uimm & 0xf000000000000000l)) {
		shift -= 4;
		uimm <<= 4;
	}

	if (!(uimm & 0xc000000000000000l)) {
		shift -= 2;
		uimm <<= 2;
	}

	if ((sljit_sw)uimm < 0) {
		uimm >>= 1;
		shift += 1;
	}
	SLJIT_ASSERT(((uimm & 0xc000000000000000l) == 0x4000000000000000l) && (shift > 0) && (shift <= 32));

	if (inv)
		uimm = ~uimm;

	FAIL_IF(push_inst(compiler, LUI | TA(dst_ar) | IMM(uimm >> 48), dst_ar));
	if (uimm & 0x0000ffff00000000l)
		FAIL_IF(push_inst(compiler, ORI | SA(dst_ar) | TA(dst_ar) | IMM(uimm >> 32), dst_ar));

	imm &= (1l << shift) - 1;
	if (!(imm & ~0xffff)) {
		ins = (shift == 32) ? DSLL32 : DSLL;
		if (shift < 32)
			ins |= SH_IMM(shift);
		FAIL_IF(push_inst(compiler, ins | TA(dst_ar) | DA(dst_ar), dst_ar));
		return !(imm & 0xffff) ? SLJIT_SUCCESS : push_inst(compiler, ORI | SA(dst_ar) | TA(dst_ar) | IMM(imm), dst_ar);
	}

	/* Double shifts needs to be performed. */
	uimm <<= 32;
	shift2 = shift - 16;

	while (!(uimm & 0xf000000000000000l)) {
		shift2 -= 4;
		uimm <<= 4;
	}

	if (!(uimm & 0xc000000000000000l)) {
		shift2 -= 2;
		uimm <<= 2;
	}

	if (!(uimm & 0x8000000000000000l)) {
		shift2--;
		uimm <<= 1;
	}

	SLJIT_ASSERT((uimm & 0x8000000000000000l) && (shift2 > 0) && (shift2 <= 16));

	FAIL_IF(push_inst(compiler, DSLL | TA(dst_ar) | DA(dst_ar) | SH_IMM(shift - shift2), dst_ar));
	FAIL_IF(push_inst(compiler, ORI | SA(dst_ar) | TA(dst_ar) | IMM(uimm >> 48), dst_ar));
	FAIL_IF(push_inst(compiler, DSLL | TA(dst_ar) | DA(dst_ar) | SH_IMM(shift2), dst_ar));

	imm &= (1l << shift2) - 1;
	return !(imm & 0xffff) ? SLJIT_SUCCESS : push_inst(compiler, ORI | SA(dst_ar) | TA(dst_ar) | IMM(imm), dst_ar);
}

#define SELECT_OP(a, b) \
	(!(op & SLJIT_I32_OP) ? a : b)

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

#define EMIT_SHIFT(op_dimm, op_dimm32, op_imm, op_dv, op_v) \
	if (flags & SRC2_IMM) { \
		if (src2 >= 32) { \
			SLJIT_ASSERT(!(op & SLJIT_I32_OP)); \
			ins = op_dimm32; \
			src2 -= 32; \
		} \
		else \
			ins = (op & SLJIT_I32_OP) ? op_imm : op_dimm; \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, ins | T(src1) | DA(EQUAL_FLAG) | SH_IMM(src2), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, ins | T(src1) | D(dst) | SH_IMM(src2), DR(dst))); \
	} \
	else { \
		ins = (op & SLJIT_I32_OP) ? op_v : op_dv; \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, ins | S(src2) | T(src1) | DA(EQUAL_FLAG), EQUAL_FLAG)); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, ins | S(src2) | T(src1) | D(dst), DR(dst))); \
	}

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_ins ins;
	sljit_s32 is_overflow, is_carry, is_handled;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src2) | TA(0) | D(dst), DR(dst));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S8) {
				FAIL_IF(push_inst(compiler, DSLL32 | T(src2) | D(dst) | SH_IMM(24), DR(dst)));
				return push_inst(compiler, DSRA32 | T(dst) | D(dst) | SH_IMM(24), DR(dst));
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
				FAIL_IF(push_inst(compiler, DSLL32 | T(src2) | D(dst) | SH_IMM(16), DR(dst)));
				return push_inst(compiler, DSRA32 | T(dst) | D(dst) | SH_IMM(16), DR(dst));
			}
			return push_inst(compiler, ANDI | S(src2) | T(dst) | IMM(0xffff), DR(dst));
		}
		else {
			SLJIT_ASSERT(dst == src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U32:
		SLJIT_ASSERT(!(op & SLJIT_I32_OP));
		FAIL_IF(push_inst(compiler, DSLL32 | T(src2) | D(dst) | SH_IMM(0), DR(dst)));
		return push_inst(compiler, DSRL32 | T(dst) | D(dst) | SH_IMM(0), DR(dst));

	case SLJIT_MOV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return push_inst(compiler, SLL | T(src2) | D(dst) | SH_IMM(0), DR(dst));

	case SLJIT_NOT:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, NOR | S(src2) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
		if (!(flags & UNUSED_DEST))
			FAIL_IF(push_inst(compiler, NOR | S(src2) | T(src2) | D(dst), DR(dst)));
		return SLJIT_SUCCESS;

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
#if (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, SELECT_OP(DCLZ, CLZ) | S(src2) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
		if (!(flags & UNUSED_DEST))
			FAIL_IF(push_inst(compiler, SELECT_OP(DCLZ, CLZ) | S(src2) | T(dst) | D(dst), DR(dst)));
#else
		if (SLJIT_UNLIKELY(flags & UNUSED_DEST)) {
			FAIL_IF(push_inst(compiler, SELECT_OP(DSRL32, SRL) | T(src2) | DA(EQUAL_FLAG) | SH_IMM(31), EQUAL_FLAG));
			return push_inst(compiler, XORI | SA(EQUAL_FLAG) | TA(EQUAL_FLAG) | IMM(1), EQUAL_FLAG);
		}
		/* Nearly all instructions are unmovable in the following sequence. */
		FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src2) | TA(0) | D(TMP_REG1), DR(TMP_REG1)));
		/* Check zero. */
		FAIL_IF(push_inst(compiler, BEQ | S(TMP_REG1) | TA(0) | IMM(5), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, ORI | SA(0) | T(dst) | IMM((op & SLJIT_I32_OP) ? 32 : 64), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | SA(0) | T(dst) | IMM(-1), DR(dst)));
		/* Loop for searching the highest bit. */
		FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(dst) | T(dst) | IMM(1), DR(dst)));
		FAIL_IF(push_inst(compiler, BGEZ | S(TMP_REG1) | IMM(-2), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, SELECT_OP(DSLL, SLL) | T(TMP_REG1) | D(TMP_REG1) | SH_IMM(1), UNMOVABLE_INS));
#endif
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
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));

			if (is_overflow || is_carry) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, ORI | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
				else {
					FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | SA(0) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
					FAIL_IF(push_inst(compiler, OR | S(src1) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
				}
			}
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(src2), DR(dst)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, OR | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		/* a + b >= a | b (otherwise, the carry should be set to 1). */
		if (is_overflow || is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		if (!is_overflow)
			return SLJIT_SUCCESS;
		FAIL_IF(push_inst(compiler, SELECT_OP(DSLL32, SLL) | TA(OTHER_FLAG) | D(TMP_REG1) | SH_IMM(31), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XOR | S(TMP_REG1) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, XOR | S(dst) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(dst) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
		return push_inst(compiler, SELECT_OP(DSRL32, SRL) | TA(OTHER_FLAG) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG);

	case SLJIT_ADDC:
		is_carry = GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY);

		if (flags & SRC2_IMM) {
			if (is_carry) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, ORI | S(src1) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));
				else {
					FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | SA(0) | TA(EQUAL_FLAG) | IMM(src2), EQUAL_FLAG));
					FAIL_IF(push_inst(compiler, OR | S(src1) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
				}
			}
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(src2), DR(dst)));
		} else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, OR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			/* dst may be the same as src1 or src2. */
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}
		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));

		FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
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
					FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | TA(EQUAL_FLAG) | IMM(-src2), EQUAL_FLAG));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(-src2), DR(dst));
			}
			else {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | D(dst), DR(dst));
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
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | TA(EQUAL_FLAG) | IMM(-src2), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTIU | S(src1) | TA(OTHER_FLAG) | IMM(src2), OTHER_FLAG));
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(-src2), DR(dst)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(OTHER_FLAG), OTHER_FLAG));
			/* dst may be the same as src1 or src2. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;
		FAIL_IF(push_inst(compiler, SELECT_OP(DSLL32, SLL) | TA(OTHER_FLAG) | D(TMP_REG1) | SH_IMM(31), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XOR | S(TMP_REG1) | TA(EQUAL_FLAG) | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, XOR | S(dst) | TA(EQUAL_FLAG) | DA(OTHER_FLAG), OTHER_FLAG));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDU, ADDU) | S(dst) | TA(0) | DA(EQUAL_FLAG), EQUAL_FLAG));
		return push_inst(compiler, SELECT_OP(DSRL32, SRL) | TA(OTHER_FLAG) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG);

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
			FAIL_IF(push_inst(compiler, SELECT_OP(DADDIU, ADDIU) | S(src1) | T(dst) | IMM(-src2), DR(dst)));
		}
		else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTU | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
			/* dst may be the same as src1 or src2. */
			FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(src1) | T(src2) | D(dst), DR(dst)));
		}

		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | S(dst) | TA(OTHER_FLAG) | D(TMP_REG1), DR(TMP_REG1)));

		FAIL_IF(push_inst(compiler, SELECT_OP(DSUBU, SUBU) | S(dst) | TA(OTHER_FLAG) | D(dst), DR(dst)));
		return (is_carry) ? push_inst(compiler, OR | SA(EQUAL_FLAG) | T(TMP_REG1) | DA(OTHER_FLAG), OTHER_FLAG) : SLJIT_SUCCESS;

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & SRC2_IMM));

		if (GET_FLAG_TYPE(op) != SLJIT_MUL_OVERFLOW) {
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
			return push_inst(compiler, SELECT_OP(DMUL, MUL) | S(src1) | T(src2) | D(dst), DR(dst));
#elif (defined SLJIT_MIPS_R1 && SLJIT_MIPS_R1)
			if (op & SLJIT_I32_OP)
				return push_inst(compiler, MUL | S(src1) | T(src2) | D(dst), DR(dst));
			FAIL_IF(push_inst(compiler, DMULT | S(src1) | T(src2), MOVABLE_INS));
			return push_inst(compiler, MFLO | D(dst), DR(dst));
#else /* !SLJIT_MIPS_R6 && !SLJIT_MIPS_R1 */
			FAIL_IF(push_inst(compiler, SELECT_OP(DMULT, MULT) | S(src1) | T(src2), MOVABLE_INS));
			return push_inst(compiler, MFLO | D(dst), DR(dst));
#endif /* SLJIT_MIPS_R6 */
		}
#if (defined SLJIT_MIPS_R6 && SLJIT_MIPS_R6)
		FAIL_IF(push_inst(compiler, SELECT_OP(DMUL, MUL) | S(src1) | T(src2) | D(dst), DR(dst)));
		FAIL_IF(push_inst(compiler, SELECT_OP(DMUH, MUH) | S(src1) | T(src2) | DA(EQUAL_FLAG), EQUAL_FLAG));
#else /* !SLJIT_MIPS_R6 */
		FAIL_IF(push_inst(compiler, SELECT_OP(DMULT, MULT) | S(src1) | T(src2), MOVABLE_INS));
		FAIL_IF(push_inst(compiler, MFHI | DA(EQUAL_FLAG), EQUAL_FLAG));
		FAIL_IF(push_inst(compiler, MFLO | D(dst), DR(dst)));
#endif /* SLJIT_MIPS_R6 */
		FAIL_IF(push_inst(compiler, SELECT_OP(DSRA32, SRA) | T(dst) | DA(OTHER_FLAG) | SH_IMM(31), OTHER_FLAG));
		return push_inst(compiler, SELECT_OP(DSUBU, SUBU) | SA(EQUAL_FLAG) | TA(OTHER_FLAG) | DA(OTHER_FLAG), OTHER_FLAG);

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
		EMIT_SHIFT(DSLL, DSLL32, SLL, DSLLV, SLLV);
		return SLJIT_SUCCESS;

	case SLJIT_LSHR:
		EMIT_SHIFT(DSRL, DSRL32, SRL, DSRLV, SRLV);
		return SLJIT_SUCCESS;

	case SLJIT_ASHR:
		EMIT_SHIFT(DSRA, DSRA32, SRA, DSRAV, SRAV);
		return SLJIT_SUCCESS;
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value)
{
	FAIL_IF(push_inst(compiler, LUI | T(dst) | IMM(init_value >> 48), DR(dst)));
	FAIL_IF(push_inst(compiler, ORI | S(dst) | T(dst) | IMM(init_value >> 32), DR(dst)));
	FAIL_IF(push_inst(compiler, DSLL | T(dst) | D(dst) | SH_IMM(16), DR(dst)));
	FAIL_IF(push_inst(compiler, ORI | S(dst) | T(dst) | IMM(init_value >> 16), DR(dst)));
	FAIL_IF(push_inst(compiler, DSLL | T(dst) | D(dst) | SH_IMM(16), DR(dst)));
	return push_inst(compiler, ORI | S(dst) | T(dst) | IMM(init_value), DR(dst));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;

	inst[0] = (inst[0] & 0xffff0000) | ((new_target >> 48) & 0xffff);
	inst[1] = (inst[1] & 0xffff0000) | ((new_target >> 32) & 0xffff);
	inst[3] = (inst[3] & 0xffff0000) | ((new_target >> 16) & 0xffff);
	inst[5] = (inst[5] & 0xffff0000) | (new_target & 0xffff);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 6);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;

	inst[0] = (inst[0] & 0xffff0000) | ((new_constant >> 48) & 0xffff);
	inst[1] = (inst[1] & 0xffff0000) | ((new_constant >> 32) & 0xffff);
	inst[3] = (inst[3] & 0xffff0000) | ((new_constant >> 16) & 0xffff);
	inst[5] = (inst[5] & 0xffff0000) | (new_constant & 0xffff);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 6);
}

static sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_ins *ins_ptr)
{
	sljit_s32 arg_count = 0;
	sljit_s32 word_arg_count = 0;
	sljit_s32 float_arg_count = 0;
	sljit_s32 types = 0;
	sljit_ins prev_ins = NOP;
	sljit_ins ins = NOP;

	SLJIT_ASSERT(reg_map[TMP_REG1] == 4 && freg_map[TMP_FREG1] == 12);

	arg_types >>= SLJIT_DEF_SHIFT;

	while (arg_types) {
		types = (types << SLJIT_DEF_SHIFT) | (arg_types & SLJIT_DEF_MASK);

		switch (arg_types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
		case SLJIT_ARG_TYPE_F64:
			arg_count++;
			float_arg_count++;
			break;
		default:
			arg_count++;
			word_arg_count++;
			break;
		}

		arg_types >>= SLJIT_DEF_SHIFT;
	}

	while (types) {
		switch (types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			if (arg_count != float_arg_count)
				ins = MOV_S | FMT_S | FS(float_arg_count) | FD(arg_count);
			else if (arg_count == 1)
				ins = MOV_S | FMT_S | FS(SLJIT_FR0) | FD(TMP_FREG1);
			arg_count--;
			float_arg_count--;
			break;
		case SLJIT_ARG_TYPE_F64:
			if (arg_count != float_arg_count)
				ins = MOV_S | FMT_D | FS(float_arg_count) | FD(arg_count);
			else if (arg_count == 1)
				ins = MOV_S | FMT_D | FS(SLJIT_FR0) | FD(TMP_FREG1);
			arg_count--;
			float_arg_count--;
			break;
		default:
			if (arg_count != word_arg_count)
				ins = DADDU | S(word_arg_count) | TA(0) | D(arg_count);
			else if (arg_count == 1)
				ins = DADDU | S(SLJIT_R0) | TA(0) | DA(4);
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

	*ins_ptr = prev_ins;

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
		FAIL_IF(push_inst(compiler, DADDU | S(src) | TA(0) | D(PIC_ADDR_REG), DR(PIC_ADDR_REG)));
	else if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, DR(PIC_ADDR_REG), src, srcw));
	}

	FAIL_IF(call_with_args(compiler, arg_types, &ins));

	/* Register input. */
	FAIL_IF(push_inst(compiler, JALR | S(PIC_ADDR_REG) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	return push_inst(compiler, ins, UNMOVABLE_INS);
}
