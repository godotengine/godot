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
			SLJIT_ASSERT_STOP();
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			FAIL_IF(push_inst(compiler, SLL | D(dst) | S1(src2) | IMM(16), DR(dst)));
			return push_inst(compiler, (op == SLJIT_MOV_S16 ? SRA : SRL) | D(dst) | S1(dst) | IMM(16), DR(dst));
		}
		else if (dst != src2)
			SLJIT_ASSERT_STOP();
		return SLJIT_SUCCESS;

	case SLJIT_NOT:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return push_inst(compiler, XNOR | (flags & SET_FLAGS) | D(dst) | S1(0) | S2(src2), DR(dst) | (flags & SET_FLAGS));

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		/* sparc 32 does not support SLJIT_KEEP_FLAGS. Not sure I can fix this. */
		FAIL_IF(push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(src2) | S2(0), SET_FLAGS));
		FAIL_IF(push_inst(compiler, OR | D(TMP_REG1) | S1(0) | S2(src2), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, BICC | DA(0x1) | (7 & DISP_MASK), UNMOVABLE_INS));
		FAIL_IF(push_inst(compiler, OR | (flags & SET_FLAGS) | D(dst) | S1(0) | IMM(32), UNMOVABLE_INS | (flags & SET_FLAGS)));
		FAIL_IF(push_inst(compiler, OR | D(dst) | S1(0) | IMM(-1), DR(dst)));

		/* Loop. */
		FAIL_IF(push_inst(compiler, SUB | SET_FLAGS | D(0) | S1(TMP_REG1) | S2(0), SET_FLAGS));
		FAIL_IF(push_inst(compiler, SLL | D(TMP_REG1) | S1(TMP_REG1) | IMM(1), DR(TMP_REG1)));
		FAIL_IF(push_inst(compiler, BICC | DA(0xe) | (-2 & DISP_MASK), UNMOVABLE_INS));
		return push_inst(compiler, ADD | (flags & SET_FLAGS) | D(dst) | S1(dst) | IMM(1), UNMOVABLE_INS | (flags & SET_FLAGS));

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

	SLJIT_ASSERT_STOP();
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

	inst[0] = (inst[0] & 0xffc00000) | ((new_target >> 10) & 0x3fffff);
	inst[1] = (inst[1] & 0xfffffc00) | (new_target & 0x3ff);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 2);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;

	inst[0] = (inst[0] & 0xffc00000) | ((new_constant >> 10) & 0x3fffff);
	inst[1] = (inst[1] & 0xfffffc00) | (new_constant & 0x3ff);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 2);
}
