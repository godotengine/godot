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

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_r, sljit_sw imm, sljit_s32 tmp_r)
{
	sljit_sw high;

	SLJIT_ASSERT(dst_r != tmp_r);

	if (imm <= SIMM_MAX && imm >= SIMM_MIN)
		return push_inst(compiler, ADDI | RD(dst_r) | RS1(TMP_ZERO) | IMM_I(imm));

	if (imm <= 0x7fffffffl && imm >= S32_MIN) {
		if (imm > S32_MAX) {
			SLJIT_ASSERT((imm & 0x800) != 0);
			FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)0x80000000u));
			return push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
		}

		if ((imm & 0x800) != 0)
			imm += 0x1000;

		FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)(imm & ~0xfff)));

		if ((imm & 0xfff) == 0)
			return SLJIT_SUCCESS;

		return push_inst(compiler, ADDI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
	}

	/* Trailing zeroes could be used to produce shifted immediates. */

	if (imm <= 0x7ffffffffffl && imm >= -0x80000000000l) {
		high = imm >> 12;

		if (imm & 0x800)
			high = ~high;

		if (high > S32_MAX) {
			SLJIT_ASSERT((high & 0x800) != 0);
			FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)0x80000000u));
			FAIL_IF(push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(high)));
		} else {
			if ((high & 0x800) != 0)
				high += 0x1000;

			FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)(high & ~0xfff)));

			if ((high & 0xfff) != 0)
				FAIL_IF(push_inst(compiler, ADDI | RD(dst_r) | RS1(dst_r) | IMM_I(high)));
		}

		FAIL_IF(push_inst(compiler, SLLI | RD(dst_r) | RS1(dst_r) | IMM_I(12)));

		if ((imm & 0xfff) != 0)
			return push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));

		return SLJIT_SUCCESS;
	}

	high = imm >> 32;
	imm = (sljit_s32)imm;

	if ((imm & 0x80000000l) != 0)
		high = ~high;

	if (high <= 0x7ffff && high >= -0x80000) {
		FAIL_IF(push_inst(compiler, LUI | RD(tmp_r) | (sljit_ins)(high << 12)));
		high = 0x1000;
	} else {
		if ((high & 0x800) != 0)
			high += 0x1000;

		FAIL_IF(push_inst(compiler, LUI | RD(tmp_r) | (sljit_ins)(high & ~0xfff)));
		high &= 0xfff;
	}

	if (imm <= SIMM_MAX && imm >= SIMM_MIN) {
		FAIL_IF(push_inst(compiler, ADDI | RD(dst_r) | RS1(TMP_ZERO) | IMM_I(imm)));
		imm = 0;
	} else if (imm > S32_MAX) {
		SLJIT_ASSERT((imm & 0x800) != 0);

		FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)0x80000000u));
		imm = 0x1000 | (imm & 0xfff);
	} else {
		if ((imm & 0x800) != 0)
			imm += 0x1000;

		FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)(imm & ~0xfff)));
		imm &= 0xfff;
	}

	if ((high & 0xfff) != 0)
		FAIL_IF(push_inst(compiler, ADDI | RD(tmp_r) | RS1(tmp_r) | IMM_I(high)));

	if (imm & 0x1000)
		FAIL_IF(push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm)));
	else if (imm != 0)
		FAIL_IF(push_inst(compiler, ADDI | RD(dst_r) | RS1(dst_r) | IMM_I(imm)));

	FAIL_IF(push_inst(compiler, SLLI | RD(tmp_r) | RS1(tmp_r) | IMM_I((high & 0x1000) ? 20 : 32)));
	return push_inst(compiler, XOR | RD(dst_r) | RS1(dst_r) | RS2(tmp_r));
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value, sljit_ins last_ins)
{
	sljit_sw high;

	if ((init_value & 0x800) != 0)
		init_value += 0x1000;

	high = init_value >> 32;

	if ((init_value & 0x80000000l) != 0)
		high = ~high;

	if ((high & 0x800) != 0)
		high += 0x1000;

	FAIL_IF(push_inst(compiler, LUI | RD(TMP_REG3) | (sljit_ins)(high & ~0xfff)));
	FAIL_IF(push_inst(compiler, ADDI | RD(TMP_REG3) | RS1(TMP_REG3) | IMM_I(high)));
	FAIL_IF(push_inst(compiler, LUI | RD(dst) | (sljit_ins)(init_value & ~0xfff)));
	FAIL_IF(push_inst(compiler, SLLI | RD(TMP_REG3) | RS1(TMP_REG3) | IMM_I(32)));
	FAIL_IF(push_inst(compiler, XOR | RD(dst) | RS1(dst) | RS2(TMP_REG3)));
	return push_inst(compiler, last_ins | RS1(dst) | IMM_I(init_value));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins*)addr;
	sljit_sw high;
	SLJIT_UNUSED_ARG(executable_offset);

	if ((new_target & 0x800) != 0)
		new_target += 0x1000;

	high = (sljit_sw)new_target >> 32;

	if ((new_target & 0x80000000l) != 0)
		high = ~high;

	if ((high & 0x800) != 0)
		high += 0x1000;

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 5, 0);

	SLJIT_ASSERT((inst[0] & 0x7f) == LUI);
	inst[0] = (inst[0] & 0xfff) | (sljit_ins)(high & ~0xfff);
	SLJIT_ASSERT((inst[1] & 0x707f) == ADDI);
	inst[1] = (inst[1] & 0xfffff) | IMM_I(high);
	SLJIT_ASSERT((inst[2] & 0x7f) == LUI);
	inst[2] = (inst[2] & 0xfff) | (sljit_ins)((sljit_sw)new_target & ~0xfff);
	SLJIT_ASSERT((inst[5] & 0x707f) == ADDI || (inst[5] & 0x707f) == JALR);
	inst[5] = (inst[5] & 0xfffff) | IMM_I(new_target);
	SLJIT_UPDATE_WX_FLAGS(inst, inst + 5, 1);

	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 5);
}
