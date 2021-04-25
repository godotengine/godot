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

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw imm)
{
	if (imm <= SIMM_MAX && imm >= SIMM_MIN)
		return push_inst(compiler, OR | D(dst) | S1(0) | IMM(imm), DR(dst));

	FAIL_IF(push_inst(compiler, SETHI | D(dst) | ((imm >> 10) & 0x3fffff), DR(dst)));
	return (imm & 0x3ff) ? push_inst(compiler, OR | D(dst) | S1(dst) | IMM_ARG | (imm & 0x3ff), DR(dst)) : SLJIT_SUCCESS;
}

#define ARG2(flags, src2) ((flags & SRC2_IMM) ? IMM(src2) : S2(src2))

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	SLJIT_COMPILE_ASSERT(ICC_IS_SET == SET_FLAGS, icc_is_set_and_set_flags_must_be_the_same);

	switch (op) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return push_inst(compiler, OR | D(dst) | S1(0) | S2(src2), DR(dst));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_U8)
				return push_inst(compiler, AND | D(dst) | S1(src2) | IMM(0xff), DR(dst));
			FAIL_IF(push_inst(compiler, SLL | D(dst) | S1(src2) | IMM(24), DR(dst)));
			return push_inst(compiler, SRA | D(dst) | S1(dst) | IMM(24), DR(dst));
		}
		else if (dst != src2)
			SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			FAIL_IF(push_inst(compiler, SLL | D(dst) | S1(src2) | IMM(16), DR(dst)));
			return push_inst(compiler, (op == SLJIT_MOV_S16 ? SRA : SRL) | D(dst) | S1(dst) | IMM(16), DR(dst));
		}
		else if (dst != src2)
			SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;

	case SLJIT_NOT:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return push_inst(compiler, XNOR | (flags & SET_FLAGS) | D(dst) | S1(0) | S2(src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		FAIL_IF(push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(src2) | S2(0), SET_FLAGS));
		FAIL_IF(push_inst(compiler, OR | D(TMP_REG1) | S1(0) | S2(src2), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, BICC | DA(0x1) | (7 & DISP_MASK), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, OR | D(dst) | S1(0) | IMM(32), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, OR | D(dst) | S1(0) | IMM(-1), DR(dst)));

		/* Loop. */
		FAIL_IF(push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(TMP_REG1) | S2(0), SET_FLAGS));
		FAIL_IF(push_inst(compiler, SLL | D(TMP_REG1) | S1(TMP_REG1) | IMM(1), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, BICC | DA(0xe) | (-2 & DISP_MASK), UNMOVABLE_INS));
		return push_inst(compiler, ADD | D(dst) | S1(dst) | IMM(1), UNMOVABLE_INS);

	case SLJIT_ADD:
		return push_inst(compiler, ADD | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_ADDC:
		return push_inst(compiler, ADDC | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_SUB:
		return push_inst(compiler, SUB | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_SUBC:
		return push_inst(compiler, SUBC | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_MUL:
		FAIL_IF(push_inst(compiler, SMUL | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst)));
		if (!(flags & SET_FLAGS))
			return SLJIT_SUCCESS;
		FAIL_IF(push_inst(compiler, SRA | D(TMP_REG1) | S1(dst) | IMM(31), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, RDY | D(TMP_LINK), DR(TMP_LINK)));
		return push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(TMP_REG1) | S2(TMP_LINK), MOVABLE_INS | SET_FLAGS);

	case SLJIT_AND:
		return push_inst(compiler, AND | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_OR:
		return push_inst(compiler, OR | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_XOR:
		return push_inst(compiler, XOR | (flags & SET_FLAGS) | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_SHL:
		FAIL_IF(push_inst(compiler, SLL | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst)));
		return !(flags & SET_FLAGS) ? SLJIT_SUCCESS : push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(dst) | S2(0), SET_FLAGS);

	case SLJIT_LSHR:
		FAIL_IF(push_inst(compiler, SRL | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst)));
		return !(flags & SET_FLAGS) ? SLJIT_SUCCESS : push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(dst) | S2(0), SET_FLAGS);

	case SLJIT_ASHR:
		FAIL_IF(push_inst(compiler, SRA | D(dst) | S1(src1) | ARG2(flags, src2), DR(dst)));
		return !(flags & SET_FLAGS) ? SLJIT_SUCCESS : push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(dst) | S2(0), SET_FLAGS);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

static sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_s32 *src)
{
	sljit_s32 reg_index = 8;
	sljit_s32 word_reg_index = 8;
	sljit_s32 float_arg_index = 1;
	sljit_s32 double_arg_count = 0;
	sljit_s32 float_offset = (16 + 6) * sizeof(sljit_sw);
	sljit_s32 types = 0;
	sljit_s32 reg = 0;
	sljit_s32 move_to_tmp2 = 0;

	if (src)
		reg = reg_map[*src & REG_MASK];

	arg_types >>= SLJIT_DEF_SHIFT;

	while (arg_types) {
		types = (types << SLJIT_DEF_SHIFT) | (arg_types & SLJIT_DEF_MASK);

		switch (arg_types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			float_arg_index++;
			if (reg_index == reg)
				move_to_tmp2 = 1;
			reg_index++;
			break;
		case SLJIT_ARG_TYPE_F64:
			float_arg_index++;
			double_arg_count++;
			if (reg_index == reg || reg_index + 1 == reg)
				move_to_tmp2 = 1;
			reg_index += 2;
			break;
		default:
			if (reg_index != word_reg_index && reg_index < 14 && reg_index == reg)
				move_to_tmp2 = 1;
			reg_index++;
			word_reg_index++;
			break;
		}

		if (move_to_tmp2) {
			move_to_tmp2 = 0;
			if (reg < 14)
				FAIL_IF(push_inst(compiler, OR | D(TMP_REG1) | S1(0) | S2A(reg), DR(TMP_REG1)));
			*src = TMP_REG1;
		}

		arg_types >>= SLJIT_DEF_SHIFT;
	}

	arg_types = types;

	while (arg_types) {
		switch (arg_types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			float_arg_index--;
			FAIL_IF(push_inst(compiler, STF | FD(float_arg_index) | S1(SLJIT_SP) | IMM(float_offset), MOVABLE_INS));
			float_offset -= sizeof(sljit_f64);
			break;
		case SLJIT_ARG_TYPE_F64:
			float_arg_index--;
			if (float_arg_index == 4 && double_arg_count == 4) {
				FAIL_IF(push_inst(compiler, STF | FD(float_arg_index) | S1(SLJIT_SP) | IMM((16 + 7) * sizeof(sljit_sw)), MOVABLE_INS));
				FAIL_IF(push_inst(compiler, STF | FD(float_arg_index) | (1 << 25) | S1(SLJIT_SP) | IMM((16 + 8) * sizeof(sljit_sw)), MOVABLE_INS));
			}
			else
				FAIL_IF(push_inst(compiler, STDF | FD(float_arg_index) | S1(SLJIT_SP) | IMM(float_offset), MOVABLE_INS));
			float_offset -= sizeof(sljit_f64);
			break;
		default:
			break;
		}

		arg_types >>= SLJIT_DEF_SHIFT;
	}

	float_offset = (16 + 6) * sizeof(sljit_sw);

	while (types) {
		switch (types & SLJIT_DEF_MASK) {
		case SLJIT_ARG_TYPE_F32:
			reg_index--;
			if (reg_index < 14)
				FAIL_IF(push_inst(compiler, LDUW | DA(reg_index) | S1(SLJIT_SP) | IMM(float_offset), reg_index));
			float_offset -= sizeof(sljit_f64);
			break;
		case SLJIT_ARG_TYPE_F64:
			reg_index -= 2;
			if (reg_index < 14) {
				if ((reg_index & 0x1) != 0) {
					FAIL_IF(push_inst(compiler, LDUW | DA(reg_index) | S1(SLJIT_SP) | IMM(float_offset), reg_index));
					if (reg_index < 13)
						FAIL_IF(push_inst(compiler, LDUW | DA(reg_index + 1) | S1(SLJIT_SP) | IMM(float_offset + sizeof(sljit_sw)), reg_index + 1));
				}
				else 
					FAIL_IF(push_inst(compiler, LDD | DA(reg_index) | S1(SLJIT_SP) | IMM(float_offset), reg_index));
			}
			float_offset -= sizeof(sljit_f64);
			break;
		default:
			reg_index--;
			word_reg_index--;

			if (reg_index != word_reg_index) {
				if (reg_index < 14)
					FAIL_IF(push_inst(compiler, OR | DA(reg_index) | S1(0) | S2A(word_reg_index), reg_index));
				else
					FAIL_IF(push_inst(compiler, STW | DA(word_reg_index) | S1(SLJIT_SP) | IMM(92), word_reg_index));
			}
			break;
		}

		types >>= SLJIT_DEF_SHIFT;
	}

	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value)
{
	FAIL_IF(push_inst(compiler, SETHI | D(dst) | ((init_value >> 10) & 0x3fffff), DR(dst)));
	return push_inst(compiler, OR | D(dst) | S1(dst) | IMM_ARG | (init_value & 0x3ff), DR(dst));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 0);
	SLJIT_ASSERT(((inst[0] & 0xc1c00000) == 0x01000000) && ((inst[1] & 0xc1f82000) == 0x80102000));
	inst[0] = (inst[0] & 0xffc00000) | ((new_target >> 10) & 0x3fffff);
	inst[1] = (inst[1] & 0xfffffc00) | (new_target & 0x3ff);
	SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 1);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 2);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, new_constant, executable_offset);
}
