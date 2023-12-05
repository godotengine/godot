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
	SLJIT_UNUSED_ARG(tmp_r);
	SLJIT_ASSERT(dst_r != tmp_r);

	if (imm <= SIMM_MAX && imm >= SIMM_MIN)
		return push_inst(compiler, ADDI | RD(dst_r) | RS1(TMP_ZERO) | IMM_I(imm));

	if (imm & 0x800)
		imm += 0x1000;

	FAIL_IF(push_inst(compiler, LUI | RD(dst_r) | (sljit_ins)(imm & ~0xfff)));

	if ((imm & 0xfff) == 0)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ADDI | RD(dst_r) | RS1(dst_r) | IMM_I(imm));
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value, sljit_ins last_ins)
{
	if ((init_value & 0x800) != 0)
		init_value += 0x1000;

	FAIL_IF(push_inst(compiler, LUI | RD(dst) | (sljit_ins)(init_value & ~0xfff)));
	return push_inst(compiler, last_ins | RS1(dst) | IMM_I(init_value));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins*)addr;
	SLJIT_UNUSED_ARG(executable_offset);

	if ((new_target & 0x800) != 0)
		new_target += 0x1000;

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 5, 0);

	SLJIT_ASSERT((inst[0] & 0x7f) == LUI);
	inst[0] = (inst[0] & 0xfff) | (sljit_ins)((sljit_sw)new_target & ~0xfff);
	SLJIT_ASSERT((inst[1] & 0x707f) == ADDI || (inst[1] & 0x707f) == JALR);
	inst[1] = (inst[1] & 0xfffff) | IMM_I(new_target);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 5, 1);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 5);
}
