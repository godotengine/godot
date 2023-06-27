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
	uimm = (sljit_uw)imm;
	if (imm < 0) {
		uimm = ~(sljit_uw)imm;
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
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 6, 0);
	inst[0] = (inst[0] & 0xffff0000) | ((sljit_ins)(new_target >> 48) & 0xffff);
	inst[1] = (inst[1] & 0xffff0000) | ((sljit_ins)(new_target >> 32) & 0xffff);
	inst[3] = (inst[3] & 0xffff0000) | ((sljit_ins)(new_target >> 16) & 0xffff);
	inst[5] = (inst[5] & 0xffff0000) | ((sljit_ins)new_target & 0xffff);
	SLJIT_UPDATE_WX_FLAGS(inst, inst + 6, 1);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 6);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}

static sljit_s32 call_with_args(struct sljit_compiler *compiler, sljit_s32 arg_types, sljit_ins *ins_ptr)
{
	sljit_s32 arg_count = 0;
	sljit_s32 word_arg_count = 0;
	sljit_s32 float_arg_count = 0;
	sljit_s32 types = 0;
	sljit_ins prev_ins = *ins_ptr;
	sljit_ins ins = NOP;

	SLJIT_ASSERT(reg_map[TMP_REG1] == 4 && freg_map[TMP_FREG1] == 12);

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		types = (types << SLJIT_ARG_SHIFT) | (arg_types & SLJIT_ARG_MASK);

		switch (arg_types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
		case SLJIT_ARG_TYPE_F32:
			arg_count++;
			float_arg_count++;
			break;
		default:
			arg_count++;
			word_arg_count++;
			break;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	while (types) {
		switch (types & SLJIT_ARG_MASK) {
		case SLJIT_ARG_TYPE_F64:
			if (arg_count != float_arg_count)
				ins = MOV_S | FMT_D | FS(float_arg_count) | FD(arg_count);
			else if (arg_count == 1)
				ins = MOV_S | FMT_D | FS(SLJIT_FR0) | FD(TMP_FREG1);
			arg_count--;
			float_arg_count--;
			break;
		case SLJIT_ARG_TYPE_F32:
			if (arg_count != float_arg_count)
				ins = MOV_S | FMT_S | FS(float_arg_count) | FD(arg_count);
			else if (arg_count == 1)
				ins = MOV_S | FMT_S | FS(SLJIT_FR0) | FD(TMP_FREG1);
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

		types >>= SLJIT_ARG_SHIFT;
	}

	*ins_ptr = prev_ins;

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	struct sljit_jump *jump;
	sljit_ins ins = NOP;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);

	if (type & SLJIT_CALL_RETURN)
		PTR_FAIL_IF(emit_stack_frame_release(compiler, 0, &ins));

	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		PTR_FAIL_IF(call_with_args(compiler, arg_types, &ins));

	SLJIT_ASSERT(DR(PIC_ADDR_REG) == 25 && PIC_ADDR_REG == TMP_REG2);

	if (ins == NOP && compiler->delay_slot != UNMOVABLE_INS)
		jump->flags |= IS_MOVABLE;

	if (!(type & SLJIT_CALL_RETURN)) {
		jump->flags |= IS_JAL;

		if ((type & 0xff) != SLJIT_CALL_REG_ARG)
			jump->flags |= IS_CALL;

		PTR_FAIL_IF(push_inst(compiler, JALR | S(PIC_ADDR_REG) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	} else
		PTR_FAIL_IF(push_inst(compiler, JR | S(PIC_ADDR_REG), UNMOVABLE_INS));

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, ins, UNMOVABLE_INS));

	/* Maximum number of instructions required for generating a constant. */
	compiler->size += 6;
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins ins = NOP;

	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, DR(PIC_ADDR_REG), src, srcw));
		src = PIC_ADDR_REG;
		srcw = 0;
	}

	if ((type & 0xff) == SLJIT_CALL_REG_ARG) {
		if (type & SLJIT_CALL_RETURN) {
			if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
				FAIL_IF(push_inst(compiler, DADDU | S(src) | TA(0) | D(PIC_ADDR_REG), DR(PIC_ADDR_REG)));
				src = PIC_ADDR_REG;
				srcw = 0;
			}

			FAIL_IF(emit_stack_frame_release(compiler, 0, &ins));

			if (ins != NOP)
				FAIL_IF(push_inst(compiler, ins, MOVABLE_INS));
		}

		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_ijump(compiler, type, src, srcw);
	}

	SLJIT_ASSERT(DR(PIC_ADDR_REG) == 25 && PIC_ADDR_REG == TMP_REG2);

	if (src & SLJIT_IMM)
		FAIL_IF(load_immediate(compiler, DR(PIC_ADDR_REG), srcw));
	else if (src != PIC_ADDR_REG)
		FAIL_IF(push_inst(compiler, DADDU | S(src) | TA(0) | D(PIC_ADDR_REG), DR(PIC_ADDR_REG)));

	if (type & SLJIT_CALL_RETURN)
		FAIL_IF(emit_stack_frame_release(compiler, 0, &ins));

	FAIL_IF(call_with_args(compiler, arg_types, &ins));

	/* Register input. */
	if (!(type & SLJIT_CALL_RETURN))
		FAIL_IF(push_inst(compiler, JALR | S(PIC_ADDR_REG) | DA(RETURN_ADDR_REG), UNMOVABLE_INS));
	else
		FAIL_IF(push_inst(compiler, JR | S(PIC_ADDR_REG), UNMOVABLE_INS));
	return push_inst(compiler, ins, UNMOVABLE_INS);
}
