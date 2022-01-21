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

/* ppc 32-bit arch dependent functions. */

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 reg, sljit_sw imm)
{
	if (imm <= SIMM_MAX && imm >= SIMM_MIN)
		return push_inst(compiler, ADDI | D(reg) | A(0) | IMM(imm));

	if (!(imm & ~0xffff))
		return push_inst(compiler, ORI | S(TMP_ZERO) | A(reg) | IMM(imm));

	FAIL_IF(push_inst(compiler, ADDIS | D(reg) | A(0) | IMM(imm >> 16)));
	return (imm & 0xffff) ? push_inst(compiler, ORI | S(reg) | A(reg) | IMM(imm)) : SLJIT_SUCCESS;
}

#define INS_CLEAR_LEFT(dst, src, from) \
	(RLWINM | S(src) | A(dst) | ((from) << 6) | (31 << 1))

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_s32 src2)
{
	switch (op) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(src1 == TMP_REG1);
		if (dst != src2)
			return push_inst(compiler, OR | S(src2) | A(dst) | B(src2));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1);
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S8)
				return push_inst(compiler, EXTSB | S(src2) | A(dst));
			return push_inst(compiler, INS_CLEAR_LEFT(dst, src2, 24));
		}
		else if ((flags & REG_DEST) && op == SLJIT_MOV_S8)
			return push_inst(compiler, EXTSB | S(src2) | A(dst));
		else {
			SLJIT_ASSERT(dst == src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1);
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S16)
				return push_inst(compiler, EXTSH | S(src2) | A(dst));
			return push_inst(compiler, INS_CLEAR_LEFT(dst, src2, 16));
		}
		else {
			SLJIT_ASSERT(dst == src2);
		}
		return SLJIT_SUCCESS;

	case SLJIT_NOT:
		SLJIT_ASSERT(src1 == TMP_REG1);
		return push_inst(compiler, NOR | RC(flags) | S(src2) | A(dst) | B(src2));

	case SLJIT_NEG:
		SLJIT_ASSERT(src1 == TMP_REG1);
		/* Setting XER SO is not enough, CR SO is also needed. */
		return push_inst(compiler, NEG | OE((flags & ALT_FORM1) ? ALT_SET_FLAGS : 0) | RC(flags) | D(dst) | A(src2));

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1);
		return push_inst(compiler, CNTLZW | S(src2) | A(dst));

	case SLJIT_ADD:
		if (flags & ALT_FORM1) {
			/* Setting XER SO is not enough, CR SO is also needed. */
			return push_inst(compiler, ADD | OE(ALT_SET_FLAGS) | RC(ALT_SET_FLAGS) | D(dst) | A(src1) | B(src2));
		}

		if (flags & ALT_FORM2) {
			/* Flags does not set: BIN_IMM_EXTS unnecessary. */
			SLJIT_ASSERT(src2 == TMP_REG2);

			if (flags & ALT_FORM3)
				return push_inst(compiler, ADDIS | D(dst) | A(src1) | compiler->imm);

			if (flags & ALT_FORM4) {
				FAIL_IF(push_inst(compiler, ADDIS | D(dst) | A(src1) | (((compiler->imm >> 16) & 0xffff) + ((compiler->imm >> 15) & 0x1))));
				src1 = dst;
			}

			return push_inst(compiler, ADDI | D(dst) | A(src1) | (compiler->imm & 0xffff));
		}
		if (flags & ALT_FORM3) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, ADDIC | D(dst) | A(src1) | compiler->imm);
		}
		SLJIT_ASSERT(!(flags & ALT_FORM4));
		if (!(flags & ALT_SET_FLAGS))
			return push_inst(compiler, ADD | D(dst) | A(src1) | B(src2));
		if (flags & ALT_FORM5)
			return push_inst(compiler, ADDC | RC(ALT_SET_FLAGS) | D(dst) | A(src1) | B(src2));
		return push_inst(compiler, ADD | RC(flags) | D(dst) | A(src1) | B(src2));

	case SLJIT_ADDC:
		return push_inst(compiler, ADDE | D(dst) | A(src1) | B(src2));

	case SLJIT_SUB:
		if (flags & ALT_FORM1) {
			if (flags & ALT_FORM2) {
				FAIL_IF(push_inst(compiler, CMPLI | CRD(0) | A(src1) | compiler->imm));
				if (!(flags & ALT_FORM3))
					return SLJIT_SUCCESS;
				return push_inst(compiler, ADDI | D(dst) | A(src1) | (-compiler->imm & 0xffff));
			}
			FAIL_IF(push_inst(compiler, CMPL | CRD(0) | A(src1) | B(src2)));
			if (!(flags & ALT_FORM3))
				return SLJIT_SUCCESS;
			return push_inst(compiler, SUBF | D(dst) | A(src2) | B(src1));
		}

		if (flags & ALT_FORM2) {
			if (flags & ALT_FORM3) {
				FAIL_IF(push_inst(compiler, CMPI | CRD(0) | A(src1) | compiler->imm));
				if (!(flags & ALT_FORM4))
					return SLJIT_SUCCESS;
				return push_inst(compiler, ADDI | D(dst) | A(src1) | (-compiler->imm & 0xffff));
			}
			FAIL_IF(push_inst(compiler, CMP | CRD(0) | A(src1) | B(src2)));
			if (!(flags & ALT_FORM4))
				return SLJIT_SUCCESS;
			return push_inst(compiler, SUBF | D(dst) | A(src2) | B(src1));
		}

		if (flags & ALT_FORM3) {
			/* Setting XER SO is not enough, CR SO is also needed. */
			return push_inst(compiler, SUBF | OE(ALT_SET_FLAGS) | RC(ALT_SET_FLAGS) | D(dst) | A(src2) | B(src1));
		}

		if (flags & ALT_FORM4) {
			/* Flags does not set: BIN_IMM_EXTS unnecessary. */
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, SUBFIC | D(dst) | A(src1) | compiler->imm);
		}

		if (!(flags & ALT_SET_FLAGS))
			return push_inst(compiler, SUBF | D(dst) | A(src2) | B(src1));
		if (flags & ALT_FORM5)
			return push_inst(compiler, SUBFC | RC(ALT_SET_FLAGS) | D(dst) | A(src2) | B(src1));
		return push_inst(compiler, SUBF | RC(flags) | D(dst) | A(src2) | B(src1));

	case SLJIT_SUBC:
		return push_inst(compiler, SUBFE | D(dst) | A(src2) | B(src1));

	case SLJIT_MUL:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, MULLI | D(dst) | A(src1) | compiler->imm);
		}
		return push_inst(compiler, MULLW | OE(flags) | RC(flags) | D(dst) | A(src2) | B(src1));

	case SLJIT_AND:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, ANDI | S(src1) | A(dst) | compiler->imm);
		}
		if (flags & ALT_FORM2) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, ANDIS | S(src1) | A(dst) | compiler->imm);
		}
		return push_inst(compiler, AND | RC(flags) | S(src1) | A(dst) | B(src2));

	case SLJIT_OR:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, ORI | S(src1) | A(dst) | compiler->imm);
		}
		if (flags & ALT_FORM2) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, ORIS | S(src1) | A(dst) | compiler->imm);
		}
		if (flags & ALT_FORM3) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			FAIL_IF(push_inst(compiler, ORI | S(src1) | A(dst) | IMM(compiler->imm)));
			return push_inst(compiler, ORIS | S(dst) | A(dst) | IMM(compiler->imm >> 16));
		}
		return push_inst(compiler, OR | RC(flags) | S(src1) | A(dst) | B(src2));

	case SLJIT_XOR:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, XORI | S(src1) | A(dst) | compiler->imm);
		}
		if (flags & ALT_FORM2) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			return push_inst(compiler, XORIS | S(src1) | A(dst) | compiler->imm);
		}
		if (flags & ALT_FORM3) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			FAIL_IF(push_inst(compiler, XORI | S(src1) | A(dst) | IMM(compiler->imm)));
			return push_inst(compiler, XORIS | S(dst) | A(dst) | IMM(compiler->imm >> 16));
		}
		return push_inst(compiler, XOR | RC(flags) | S(src1) | A(dst) | B(src2));

	case SLJIT_SHL:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			compiler->imm &= 0x1f;
			return push_inst(compiler, RLWINM | RC(flags) | S(src1) | A(dst) | (compiler->imm << 11) | ((31 - compiler->imm) << 1));
		}
		return push_inst(compiler, SLW | RC(flags) | S(src1) | A(dst) | B(src2));

	case SLJIT_LSHR:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			compiler->imm &= 0x1f;
			return push_inst(compiler, RLWINM | RC(flags) | S(src1) | A(dst) | (((32 - compiler->imm) & 0x1f) << 11) | (compiler->imm << 6) | (31 << 1));
		}
		return push_inst(compiler, SRW | RC(flags) | S(src1) | A(dst) | B(src2));

	case SLJIT_ASHR:
		if (flags & ALT_FORM1) {
			SLJIT_ASSERT(src2 == TMP_REG2);
			compiler->imm &= 0x1f;
			return push_inst(compiler, SRAWI | RC(flags) | S(src1) | A(dst) | (compiler->imm << 11));
		}
		return push_inst(compiler, SRAW | RC(flags) | S(src1) | A(dst) | B(src2));
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 reg, sljit_sw init_value)
{
	FAIL_IF(push_inst(compiler, ADDIS | D(reg) | A(0) | IMM(init_value >> 16)));
	return push_inst(compiler, ORI | S(reg) | A(reg) | IMM(init_value));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 2, 0);
	SLJIT_ASSERT((inst[0] & 0xfc1f0000) == ADDIS && (inst[1] & 0xfc000000) == ORI);
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
