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

static sljit_s32 load_immediate32(struct sljit_compiler *compiler, sljit_s32 dst_r, sljit_sw imm)
{
	SLJIT_ASSERT((imm <= 0x7fffffffl && imm > SIMM_MAX) || (imm >= S32_MIN && imm < SIMM_MIN));

	if (imm > S32_MAX) {
		SLJIT_ASSERT((imm & 0x800) != 0);
		FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)0x80000000u));
		return push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
	}

	if (RISCV_HAS_COMPRESSED(200) && imm <= 0x1ffff && imm >= -0x20000) {
		if (imm > 0x1f7ff) {
			SLJIT_ASSERT((imm & 0x800) != 0);
			FAIL_IF(push_inst16(compiler, C_LUI | C_RD(dst_r) | (sljit_u16)0x1000));
			return push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
		}

		if ((imm & 0x800) != 0)
			imm += 0x1000;

		FAIL_IF(push_inst16(compiler, C_LUI | C_RD(dst_r) | ((sljit_u16)(((imm) & 0x1f000) >> 10) | ((imm) & 0x20000) >> 5)));
	} else {
		if ((imm & 0x800) != 0)
			imm += 0x1000;

		FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)(imm & ~(sljit_sw)0xfff)));
	}

	imm &= 0xfff;

	if (imm == 0)
		return SLJIT_SUCCESS;

	if (RISCV_HAS_COMPRESSED(200) && (imm <= 0x1f || imm >= 0xfe0))
		return push_inst16(compiler, C_ADDI | C_RD(dst_r) | C_IMM_I(imm));

	return push_inst(compiler, ADDI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
}

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_r, sljit_sw imm, sljit_s32 tmp_r)
{
	sljit_sw high, shift;

	if (RISCV_HAS_COMPRESSED(200) && imm <= SIMM16_MAX && imm >= SIMM16_MIN)
		return push_inst16(compiler, C_LI | C_RD(dst_r) | C_IMM_I(imm));

	if (imm <= SIMM_MAX && imm >= SIMM_MIN)
		return push_inst(compiler, ADDI | RD(dst_r) | RS1(TMP_ZERO) | IMM_I(imm));

	if (imm <= 0x7fffffffl && imm >= S32_MIN)
		return load_immediate32(compiler, dst_r, imm);

	/* Shifted small immediates. */

	high = imm;
	shift = 0;
	while ((high & 0xff) == 0) {
		high >>= 8;
		shift += 8;
	}

	if ((high & 0xf) == 0) {
		high >>= 4;
		shift += 4;
	}

	if ((high & 0x3) == 0) {
		high >>= 2;
		shift += 2;
	}

	if ((high & 0x1) == 0) {
		high >>= 1;
		shift += 1;
	}

	if (high <= 0x7fffffffl && high >= S32_MIN) {
		load_immediate(compiler, dst_r, high, tmp_r);

		if (RISCV_HAS_COMPRESSED(200))
			return push_inst16(compiler, C_SLLI | C_RD(dst_r) | C_IMM_I(shift));
		return push_inst(compiler, SLLI | RD(dst_r) | RS1(dst_r) | IMM_I(shift));
	}

	/* Trailing zeroes could be used to produce shifted immediates. */

	if (imm <= 0x7ffffffffffl && imm >= -0x80000000000l) {
		high = imm >> 12;

		if (imm & 0x800)
			high = ~high;

		FAIL_IF(load_immediate32(compiler, dst_r, high));

		if (RISCV_HAS_COMPRESSED(200))
			FAIL_IF(push_inst16(compiler, C_SLLI | C_RD(dst_r) | (sljit_u16)(12 << 2)));
		else
			FAIL_IF(push_inst(compiler, SLLI | RD(dst_r) | RS1(dst_r) | IMM_I(12)));

		SLJIT_ASSERT((imm & 0xfff) != 0);
		return push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
	}

	SLJIT_ASSERT(dst_r != tmp_r);

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
		if (RISCV_HAS_COMPRESSED(200) && imm <= 0x1f && imm >= -0x20)
			FAIL_IF(push_inst16(compiler, C_LI | C_RD(dst_r) | C_IMM_I(imm)));
		else
			FAIL_IF(push_inst(compiler, ADDI | RD(dst_r) | RS1(TMP_ZERO) | IMM_I(imm)));
		imm = 0;
	} else if (imm > S32_MAX) {
		SLJIT_ASSERT((imm & 0x800) != 0);

		FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)0x80000000u));
		imm = 0x1000 | (imm & 0xfff);
	} else {
		if ((imm & 0x800) != 0)
			imm += 0x1000;

		if (RISCV_HAS_COMPRESSED(200) && imm <= 0x1ffff && imm >= -0x20000)
			FAIL_IF(push_inst16(compiler, C_LUI | C_RD(dst_r) | ((sljit_u16)(((imm) & 0x1f000) >> 10) | ((imm) & 0x20000) >> 5)));
		else
			FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)(imm & ~0xfff)));
		imm &= 0xfff;
	}

	if ((high & 0xfff) != 0) {
		SLJIT_ASSERT(high <= 0xfff);
		if (RISCV_HAS_COMPRESSED(200) && (high <= 0x1f || high >= 0xfe0))
			FAIL_IF(push_inst16(compiler, C_ADDI | C_RD(tmp_r) | C_IMM_I(high)));
		else
			FAIL_IF(push_inst(compiler, ADDI | RD(tmp_r) | RS1(tmp_r) | IMM_I(high)));
	}

	if (imm & 0x1000)
		FAIL_IF(push_inst(compiler, XORI | RD(dst_r) | RS1(dst_r) | IMM_I(imm)));
	else if (imm != 0) {
		SLJIT_ASSERT(imm <= 0xfff);
		if (RISCV_HAS_COMPRESSED(200) && (imm <= 0x1f || imm >= 0xfe0))
			FAIL_IF(push_inst16(compiler, C_ADDI | C_RD(dst_r) | C_IMM_I(imm)));
		else
			FAIL_IF(push_inst(compiler, ADDI | RD(dst_r) | RS1(dst_r) | IMM_I(imm)));
	}

	if (RISCV_HAS_COMPRESSED(200))
		FAIL_IF(push_inst16(compiler, C_SLLI | C_RD(tmp_r) | (sljit_u16)((high & 0x1000) ? (20 << 2) : (1 << 12))));
	else
		FAIL_IF(push_inst(compiler, SLLI | RD(tmp_r) | RS1(tmp_r) | IMM_I((high & 0x1000) ? 20 : 32)));
	return push_inst(compiler, XOR | RD(dst_r) | RS1(dst_r) | RS2(tmp_r));
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

	if (u.imm == 0)
		return push_inst(compiler, FMV_W_X | (1 << 25) | RS1(TMP_ZERO) | FRD(freg));

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm, TMP_REG3));
	return push_inst(compiler, FMV_W_X | (1 << 25) | RS1(TMP_REG1) | FRD(freg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fcopy(compiler, op, freg, reg));

	if (GET_OPCODE(op) == SLJIT_COPY_TO_F64)
		inst = FMV_W_X | RS1(reg) | FRD(freg);
	else
		inst = FMV_X_W | FRS1(freg) | RD(reg);

	if (!(op & SLJIT_32))
		inst |= (sljit_ins)1 << 25;

	return push_inst(compiler, inst);
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
	sljit_u16 *inst = (sljit_u16*)addr;
	sljit_sw high;
	SLJIT_UNUSED_ARG(executable_offset);

	if ((new_target & 0x800) != 0)
		new_target += 0x1000;

	high = (sljit_sw)new_target >> 32;

	if ((new_target & 0x80000000l) != 0)
		high = ~high;

	if ((high & 0x800) != 0)
		high += 0x1000;

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 12, 0);

	SLJIT_ASSERT((inst[0] & 0x7f) == LUI);
	inst[0] = (sljit_u16)((inst[0] & 0xfff) | (high & 0xf000));
	inst[1] = (sljit_u16)(high >> 16);
	SLJIT_ASSERT((inst[2] & 0x707f) == ADDI);
	inst[3] = (sljit_u16)((inst[3] & 0xf) | (high << 4));
	SLJIT_ASSERT((inst[4] & 0x7f) == LUI);
	inst[4] = (sljit_u16)((inst[4] & 0xfff) | (new_target & 0xf000));
	inst[5] = (sljit_u16)(new_target >> 16);
	SLJIT_ASSERT((inst[10] & 0x707f) == ADDI || (inst[10] & 0x707f) == JALR);
	inst[11] = (sljit_u16)((inst[11] & 0xf) | (new_target << 4));
	SLJIT_UPDATE_WX_FLAGS(inst, inst + 12, 1);

	inst = (sljit_u16 *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 12);
}
