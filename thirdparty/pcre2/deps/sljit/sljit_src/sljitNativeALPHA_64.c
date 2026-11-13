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

/* Alpha ISA extension bits: a bit set means the extension is present.
   Same layout as Linux AT_HWCAP and as ~amask(-1). */
#define ALPHA_HWCAP_BWX 0x1
#define ALPHA_HWCAP_FIX 0x2
#define ALPHA_HWCAP_CIX 0x4

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
	return "Alpha" SLJIT_CPUINFO;
}

/* Alpha instructions are fixed 32 bit and processed as instruction units. */
typedef sljit_u32 sljit_ins;

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3	(SLJIT_NUMBER_OF_REGISTERS + 4)
/* r29 (gp) is unused in JIT-compiled code; used as 4th scratch when reg
   aliases TMP_REG2 or TMP_REG3 in the unaligned store synthesizers. */
#define TMP_REG4	(SLJIT_NUMBER_OF_REGISTERS + 7)
#define TMP_ZERO	0

/* Flags are kept in volatile registers. */
#define EQUAL_FLAG	(SLJIT_NUMBER_OF_REGISTERS + 5)
#define RETURN_ADDR_REG	TMP_REG2
#define OTHER_FLAG	(SLJIT_NUMBER_OF_REGISTERS + 6)

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)

/* Maps sljit register indices onto Alpha physical registers. */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 8] = {
	31,
	0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23,
	9, 10, 11, 12, 13, 14, 15,
	30, 28, 26, 27, 25, 24, 29
};

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3] = {
	31,
	0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
	9, 8, 7, 6, 5, 4, 3, 2,
	30, 28
};

/* --------------------------------------------------------------------- */
/*  Instruction forms                                                     */
/* --------------------------------------------------------------------- */

#define OP(op)		((sljit_ins)(op) << 26)
#define FUNC(f)		((sljit_ins)(f) << 5)

#define RA(r)		((sljit_ins)reg_map[r] << 21)
#define RB(r)		((sljit_ins)reg_map[r] << 16)
#define RC(r)		((sljit_ins)reg_map[r])
#define FA(r)		((sljit_ins)freg_map[r] << 21)
#define FB(r)		((sljit_ins)freg_map[r] << 16)
#define FC(r)		((sljit_ins)freg_map[r])
#define ABS_RA(n)	((sljit_ins)(n) << 21)
/* Literal (8 bit, zero extended) operand for operate instructions. */
#define ALPHA_LIT(imm)	((((sljit_ins)(imm) & 0xff) << 13) | ((sljit_ins)1 << 12))
#define DISP16(imm)	((sljit_ins)(imm) & 0xffff)

/* Memory format opcodes. */
#define LDA		OP(0x08)
#define LDAH		OP(0x09)
#define LDBU		OP(0x0a)
#define LDQ_U		OP(0x0b)
#define LDWU		OP(0x0c)
#define STW		OP(0x0d)
#define STB		OP(0x0e)
#define STQ_U		OP(0x0f)
#define LDF		OP(0x20)
#define LDG		OP(0x21)
#define LDS		OP(0x22)
#define LDT		OP(0x23)
#define STF		OP(0x24)
#define STG		OP(0x25)
#define STS		OP(0x26)
#define STT		OP(0x27)
#define LDL		OP(0x28)
#define LDQ		OP(0x29)
#define LDL_L		OP(0x2a)
#define LDQ_L		OP(0x2b)
#define STL		OP(0x2c)
#define STQ		OP(0x2d)
#define STL_C		OP(0x2e)
#define STQ_C		OP(0x2f)

/* Integer arithmetic (op 0x10). */
#define ADDL		(OP(0x10) | FUNC(0x00))
#define SUBL		(OP(0x10) | FUNC(0x09))
#define ADDQ		(OP(0x10) | FUNC(0x20))
#define SUBQ		(OP(0x10) | FUNC(0x29))
#define S4ADDQ		(OP(0x10) | FUNC(0x22))
#define S8ADDQ		(OP(0x10) | FUNC(0x32))
#define CMPEQ		(OP(0x10) | FUNC(0x2d))
#define CMPLT		(OP(0x10) | FUNC(0x4d))
#define CMPLE		(OP(0x10) | FUNC(0x6d))
#define CMPULT		(OP(0x10) | FUNC(0x1d))
#define CMPULE		(OP(0x10) | FUNC(0x3d))
#define CMPBGE		(OP(0x10) | FUNC(0x0f))

/* Logical / conditional move (op 0x11). */
#define AND		(OP(0x11) | FUNC(0x00))
#define BIC		(OP(0x11) | FUNC(0x08))
#define CMOVLBS		(OP(0x11) | FUNC(0x14))
#define CMOVLBC		(OP(0x11) | FUNC(0x16))
#define BIS		(OP(0x11) | FUNC(0x20))
#define CMOVEQ		(OP(0x11) | FUNC(0x24))
#define CMOVNE		(OP(0x11) | FUNC(0x26))
#define ORNOT		(OP(0x11) | FUNC(0x28))
#define XOR		(OP(0x11) | FUNC(0x40))
#define CMOVLT		(OP(0x11) | FUNC(0x44))
#define CMOVGE		(OP(0x11) | FUNC(0x46))
#define EQV		(OP(0x11) | FUNC(0x48))
#define CMOVLE		(OP(0x11) | FUNC(0x64))
#define CMOVGT		(OP(0x11) | FUNC(0x66))

/* Shift / byte manipulation (op 0x12). */
#define EXTBL		(OP(0x12) | FUNC(0x06))
#define EXTWL		(OP(0x12) | FUNC(0x16))
#define EXTLL		(OP(0x12) | FUNC(0x26))
#define EXTQL		(OP(0x12) | FUNC(0x36))
#define INSBL		(OP(0x12) | FUNC(0x0b))
#define MSKBL		(OP(0x12) | FUNC(0x02))
#define INSWL		(OP(0x12) | FUNC(0x1b))
#define INSLL		(OP(0x12) | FUNC(0x2b))
#define INSQL		(OP(0x12) | FUNC(0x3b))
#define MSKWL		(OP(0x12) | FUNC(0x12))
#define MSKLL		(OP(0x12) | FUNC(0x22))
#define MSKQL		(OP(0x12) | FUNC(0x32))
#define ZAPNOT		(OP(0x12) | FUNC(0x31))
#define EXTWH		(OP(0x12) | FUNC(0x5a))
#define EXTLH		(OP(0x12) | FUNC(0x6a))
#define EXTQH		(OP(0x12) | FUNC(0x7a))
#define MSKWH		(OP(0x12) | FUNC(0x52))
#define MSKLH		(OP(0x12) | FUNC(0x62))
#define MSKQH		(OP(0x12) | FUNC(0x72))
#define INSWH		(OP(0x12) | FUNC(0x57))
#define INSLH		(OP(0x12) | FUNC(0x67))
#define INSQH		(OP(0x12) | FUNC(0x77))
#define SRL		(OP(0x12) | FUNC(0x34))
#define SLL		(OP(0x12) | FUNC(0x39))
#define SRA		(OP(0x12) | FUNC(0x3c))

/* Multiply (op 0x13). */
#define MULL		(OP(0x13) | FUNC(0x00))
#define MULQ		(OP(0x13) | FUNC(0x20))
#define UMULH		(OP(0x13) | FUNC(0x30))

/* Sign extension / count (op 0x1c, BWX/CIX). */
#define SEXTB		(OP(0x1c) | FUNC(0x00))
#define SEXTW		(OP(0x1c) | FUNC(0x01))
#define CTPOP		(OP(0x1c) | FUNC(0x30))
#define CTLZ		(OP(0x1c) | FUNC(0x32))
#define CTTZ		(OP(0x1c) | FUNC(0x33))
/* FTOIx / ITOFx leave the unused Rb field as r31. */
#define FTOIT		(OP(0x1c) | FUNC(0x070) | ((sljit_ins)31 << 16))
#define FTOIS		(OP(0x1c) | FUNC(0x078) | ((sljit_ins)31 << 16))

/* Integer to float register move (op 0x14, FIX). */
#define ITOFS		(OP(0x14) | FUNC(0x004) | ((sljit_ins)31 << 16))
#define ITOFT		(OP(0x14) | FUNC(0x024) | ((sljit_ins)31 << 16))

/* Branches. */
#define BR		OP(0x30)
#define FBEQ		OP(0x31)
#define FBLT		OP(0x32)
#define FBLE		OP(0x33)
#define BSR		OP(0x34)
#define FBNE		OP(0x35)
#define FBGE		OP(0x36)
#define FBGT		OP(0x37)
#define BLBC		OP(0x38)
#define BEQ		OP(0x39)
#define BLT		OP(0x3a)
#define BLE		OP(0x3b)
#define BLBS		OP(0x3c)
#define BNE		OP(0x3d)
#define BGE		OP(0x3e)
#define BGT		OP(0x3f)

/* Memory branch (jumps, op 0x1a). */
#define JMP		(OP(0x1a) | ((sljit_ins)0 << 14))
#define JSR		(OP(0x1a) | ((sljit_ins)1 << 14))
#define RETI		(OP(0x1a) | ((sljit_ins)2 << 14))

/* Floating point IEEE T/S (op 0x16). The /SUI (arithmetic, +0x700) and /SU
   (compare, +0x500) software-completion qualifiers keep IEEE exceptions from
   raising SIGFPE: the kernel completes them with the (trap-disabled) FPCR. */
#define ADDS		(OP(0x16) | FUNC(0x780))
#define SUBS		(OP(0x16) | FUNC(0x781))
#define MULS		(OP(0x16) | FUNC(0x782))
#define DIVS		(OP(0x16) | FUNC(0x783))
#define ADDT		(OP(0x16) | FUNC(0x7a0))
#define SUBT		(OP(0x16) | FUNC(0x7a1))
#define MULT		(OP(0x16) | FUNC(0x7a2))
#define DIVT		(OP(0x16) | FUNC(0x7a3))
#define CMPTUN		(OP(0x16) | FUNC(0x5a4))
#define CMPTEQ		(OP(0x16) | FUNC(0x5a5))
#define CMPTLT		(OP(0x16) | FUNC(0x5a6))
#define CMPTLE		(OP(0x16) | FUNC(0x5a7))
/* Unary converts leave the unused Fa field as f31. CVTTQ uses /SVC (software
   completion + integer overflow + chopped/truncating rounding). */
#define CVTTS		(OP(0x16) | FUNC(0x7ac) | ((sljit_ins)31 << 21))
#define CVTTQ		(OP(0x16) | FUNC(0x52f) | ((sljit_ins)31 << 21))
#define CVTQS		(OP(0x16) | FUNC(0x7bc) | ((sljit_ins)31 << 21))
#define CVTQT		(OP(0x16) | FUNC(0x7be) | ((sljit_ins)31 << 21))

/* Floating point copy/convert (op 0x17). */
#define CVTLQ		(OP(0x17) | FUNC(0x010) | ((sljit_ins)31 << 21))
#define CPYS		(OP(0x17) | FUNC(0x020))
#define CPYSN		(OP(0x17) | FUNC(0x021))
#define CPYSE		(OP(0x17) | FUNC(0x022))
#define CVTQL		(OP(0x17) | FUNC(0x030) | ((sljit_ins)31 << 21))

/* Miscellaneous (op 0x18). */
#define MB		(OP(0x18) | 0x4000)

/* NOP: BIS r31, r31, r31. */
#define NOP		(BIS | RA(TMP_ZERO) | RB(TMP_ZERO) | RC(TMP_ZERO))
/* Canonical return: ret r31, (r26). */
#define ALPHA_RET	(RETI | RA(TMP_ZERO) | RB(RETURN_ADDR_REG) | 1)

#define SIMM_MAX	(0x7fff)
#define SIMM_MIN	(-0x8000)
#define S32_MAX		(0x7fffffffl)
#define S32_MIN		(-0x80000000l)

/* Branch displacement is a signed 21 bit count of instructions. */
#define BRANCH_MAX	(0xfffff)
#define BRANCH_MIN	(-0x100000)

/* Local jump flags (bits must not clash with JUMP_ADDR/JUMP_MOV_ADDR/
   SLJIT_REWRITABLE_JUMP defined in sljitLir.c). */
#define IS_COND		0x004
#define IS_CALL		0x008
#define PATCH_B		0x020
#define PATCH_ABS	0x040

/* Reserved instruction slots for a materialized far jump:
   LDAH, LDA, SLL, LDAH, LDA (5) + JMP/JSR (1). */
#define JUMP_MAX_SIZE	((sljit_uw)6)

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  CPU feature detection                                                 */
/* --------------------------------------------------------------------- */

static sljit_u32 cpu_feature_list = 0;

static void get_cpu_features(void)
{
	/* Bit 31 is a sentinel: ensures cpu_feature_list != 0 after
	   initialization even on pre-EV56 CPUs with no extensions. */
	sljit_u32 features = 0x80000000u;
	/* __builtin_alpha_amask clears a bit for each supported extension.
	   On pre-EV56, AMASK copies Rb to Rc unchanged (hardware behavior),
	   so ~amask(~0UL)=0 correctly indicates no extensions. */
	unsigned long hwcap = ~__builtin_alpha_amask(~0UL);

	if (hwcap & ALPHA_HWCAP_BWX)
		features |= ALPHA_HWCAP_BWX;
	if (hwcap & ALPHA_HWCAP_FIX)
		features |= ALPHA_HWCAP_FIX;
	if (hwcap & ALPHA_HWCAP_CIX)
		features |= ALPHA_HWCAP_CIX;

	cpu_feature_list = features;
}

/* Move a raw bit pattern between the integer and floating point register
   files.  ITOFS/ITOFT/FTOIS/FTOIT require the FIX extension (EV6+); older
   CPUs have no such instruction, so the value is bounced through the word
   reserved at the bottom of the frame by SLJIT_LOCALS_OFFSET_BASE.  The
   store/load pair is chosen to perform the same format conversion as the
   instruction it replaces: LDS/STS convert S format, LDT/STT do not. */
static sljit_s32 emit_int_to_float(struct sljit_compiler *compiler, sljit_s32 is_32, sljit_s32 freg, sljit_s32 reg)
{
	if (!cpu_feature_list)
		get_cpu_features();

	if (cpu_feature_list & ALPHA_HWCAP_FIX)
		return push_inst(compiler, (is_32 ? ITOFS : ITOFT) | RA(reg) | FC(freg));

	FAIL_IF(push_inst(compiler, (is_32 ? STL : STQ) | RA(reg) | RB(SLJIT_SP) | DISP16(0)));
	return push_inst(compiler, (is_32 ? LDS : LDT) | FA(freg) | RB(SLJIT_SP) | DISP16(0));
}

/* Number of instructions emitted by the two helpers above, needed by callers
   which branch over a block containing a transfer. */
static sljit_s32 int_float_move_size(void)
{
	if (!cpu_feature_list)
		get_cpu_features();

	return (cpu_feature_list & ALPHA_HWCAP_FIX) ? 1 : 2;
}

static sljit_s32 emit_float_to_int(struct sljit_compiler *compiler, sljit_s32 is_32, sljit_s32 reg, sljit_s32 freg)
{
	if (!cpu_feature_list)
		get_cpu_features();

	if (cpu_feature_list & ALPHA_HWCAP_FIX)
		return push_inst(compiler, (is_32 ? FTOIS : FTOIT) | FA(freg) | RC(reg));

	FAIL_IF(push_inst(compiler, (is_32 ? STS : STT) | FA(freg) | RB(SLJIT_SP) | DISP16(0)));
	return push_inst(compiler, (is_32 ? LDL : LDQ) | RA(reg) | RB(SLJIT_SP) | DISP16(0));
}

/* --------------------------------------------------------------------- */
/*  Jumps                                                                 */
/* --------------------------------------------------------------------- */

static sljit_ins get_jump_instruction(sljit_s32 type);

/* Fill the reserved region [inst .. inst + JUMP_MAX_SIZE) of a jump.
   Returns the last written instruction slot. */
static SLJIT_INLINE sljit_ins* detect_jump_type(struct sljit_jump *jump, sljit_ins *code_ptr, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_ins *inst = code_ptr;
	sljit_s32 i;

	SLJIT_UNUSED_ARG(executable_offset);

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->u.label != NULL);
		target_addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code + jump->u.label->size, executable_offset);
	}

	if (jump->flags & IS_COND) {
		if (!(jump->flags & SLJIT_REWRITABLE_JUMP)) {
			/* The conditional branch is at inst[-1]. */
			diff = (sljit_sw)target_addr - (sljit_sw)SLJIT_ADD_EXEC_OFFSET(inst - 1, executable_offset);
			diff = (diff - 4) >> 2;

			if (diff >= BRANCH_MIN && diff <= BRANCH_MAX) {
				/* Branch straight to the target, pad the rest with NOP. */
				jump->flags |= PATCH_B;
				jump->addr = (sljit_uw)(inst - 1);
				for (i = 0; i < (sljit_s32)JUMP_MAX_SIZE; i++)
					inst[i] = NOP;
				return inst + (JUMP_MAX_SIZE - 1);
			}
		}

		/* Far / rewritable conditional: invert the branch so it skips the
		   materialized jump. */
		inst[-1] = (inst[-1] ^ ((sljit_ins)0x04 << 26)) | (JUMP_MAX_SIZE & 0x1fffff);
	} else if (!(jump->flags & SLJIT_REWRITABLE_JUMP) && !(jump->flags & IS_CALL)) {
		/* Calls always take the materialized path so pv (r27) holds the
		   callee entry point for its ldgp. */
		diff = (sljit_sw)target_addr - (sljit_sw)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
		diff = (diff - 4) >> 2;

		if (diff >= BRANCH_MIN && diff <= BRANCH_MAX) {
			jump->flags |= PATCH_B;
			jump->addr = (sljit_uw)inst;
			inst[0] = BR | RA(TMP_ZERO);
			for (i = 1; i < (sljit_s32)JUMP_MAX_SIZE; i++)
				inst[i] = NOP;
			return inst + (JUMP_MAX_SIZE - 1);
		}
	}

	/* Far / rewritable: materialize the target into TMP_REG3 and jump. */
	jump->flags |= PATCH_ABS;
	jump->addr = (sljit_uw)inst;
	inst[5] = (jump->flags & IS_CALL)
		? (JSR | RA(RETURN_ADDR_REG) | RB(TMP_REG3))
		: (JMP | RA(TMP_ZERO) | RB(TMP_REG3));
	return inst + (JUMP_MAX_SIZE - 1);
}

/* Write a fixed five instruction LDAH/LDA/SLL/LDAH/LDA sequence that loads
   a 64 bit address into reg at inst[0..4]. */
static SLJIT_INLINE void load_addr_to_reg(sljit_ins *inst, sljit_s32 reg, sljit_uw addr)
{
	sljit_sw high;
	sljit_s32 low;

	low = (sljit_s32)addr;
	high = (sljit_sw)addr >> 32;
	if (low < 0)
		high++;

	inst[0] = LDAH | RA(reg) | RB(TMP_ZERO) | ((((sljit_ins)((sljit_uw)high >> 16)) + (((high) & 0x8000) ? 1 : 0)) & 0xffff);
	inst[1] = LDA | RA(reg) | RB(reg) | ((sljit_ins)high & 0xffff);
	inst[2] = SLL | RA(reg) | ALPHA_LIT(32) | RC(reg);
	inst[3] = LDAH | RA(reg) | RB(reg) | ((((sljit_ins)((sljit_u32)low >> 16)) + (((low) & 0x8000) ? 1 : 0)) & 0xffff);
	inst[4] = LDA | RA(reg) | RB(reg) | ((sljit_ins)low & 0xffff);
}

static SLJIT_INLINE sljit_ins *process_extended_label(sljit_ins *code_ptr, struct sljit_extended_label *ext_label)
{
	SLJIT_ASSERT(ext_label->label.u.index == SLJIT_LABEL_ALIGNED);
	return (sljit_ins*)((sljit_uw)code_ptr & ~(ext_label->data));
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler, sljit_s32 options, void *exec_allocator_data)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw word_count;
	SLJIT_NEXT_DEFINE_TYPES;
	sljit_sw executable_offset;
	sljit_uw addr;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler, options));

	code = (sljit_ins*)allocate_executable_memory(compiler->size * sizeof(sljit_ins), options, exec_allocator_data, &executable_offset);
	PTR_FAIL_WITH_EXEC_IF(code);

	reverse_buf(compiler);
	buf = compiler->buf;

	code_ptr = code;
	word_count = 0;
	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	SLJIT_NEXT_INIT_TYPES();
	SLJIT_GET_NEXT_MIN();

	do {
		buf_ptr = (sljit_ins*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			*code_ptr = *buf_ptr++;
			if (next_min_addr == word_count) {
				SLJIT_ASSERT(!label || label->size >= word_count);
				SLJIT_ASSERT(!jump || jump->addr >= word_count);
				SLJIT_ASSERT(!const_ || const_->addr >= word_count);

				if (next_min_addr == next_label_size) {
					if (label->u.index >= SLJIT_LABEL_ALIGNED) {
						code_ptr = process_extended_label(code_ptr, (struct sljit_extended_label*)label);
						*code_ptr = buf_ptr[-1];
					}

					label->u.addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					label->size = (sljit_uw)(code_ptr - code);
					label = label->next;
					next_label_size = SLJIT_GET_NEXT_SIZE(label);
				}

				if (next_min_addr == next_jump_addr) {
					if (!(jump->flags & JUMP_MOV_ADDR)) {
						word_count += JUMP_MAX_SIZE - 1;
						code_ptr = detect_jump_type(jump, code_ptr, code, executable_offset);
					} else {
						word_count += JUMP_MAX_SIZE - 1;
						jump->addr = (sljit_uw)code_ptr;
						/* load_addr_to_reg fills slots [0..4]; NOP the rest. */
						code_ptr[JUMP_MAX_SIZE - 1] = NOP;
						code_ptr += JUMP_MAX_SIZE - 1;
					}
					jump = jump->next;
					next_jump_addr = SLJIT_GET_NEXT_ADDRESS(jump);
				} else if (next_min_addr == next_const_addr) {
					const_->addr = (sljit_uw)code_ptr;
					const_ = const_->next;
					next_const_addr = SLJIT_GET_NEXT_ADDRESS(const_);
				}

				SLJIT_GET_NEXT_MIN();
			}
			code_ptr++;
			word_count++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		if (label->u.index >= SLJIT_LABEL_ALIGNED)
			code_ptr = process_extended_label(code_ptr, (struct sljit_extended_label*)label);

		label->u.addr = (sljit_uw)code_ptr;
		label->size = (sljit_uw)(code_ptr - code);
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);

	jump = compiler->jumps;
	while (jump) {
		do {
			addr = (jump->flags & JUMP_ADDR) ? jump->u.target : jump->u.label->u.addr;
			buf_ptr = (sljit_ins *)jump->addr;

			if (jump->flags & JUMP_MOV_ADDR) {
				load_addr_to_reg(buf_ptr, (sljit_s32)buf_ptr[0], addr);
				break;
			}

			if (jump->flags & PATCH_B) {
				addr = (sljit_uw)((sljit_sw)(addr - (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset) - sizeof(sljit_ins)) >> 2);
				SLJIT_ASSERT((sljit_sw)addr <= BRANCH_MAX && (sljit_sw)addr >= BRANCH_MIN);
				buf_ptr[0] = (buf_ptr[0] & ~(sljit_ins)0x1fffff) | ((sljit_ins)addr & 0x1fffff);
				break;
			}

			SLJIT_ASSERT(jump->flags & PATCH_ABS);
			load_addr_to_reg(buf_ptr, TMP_REG3, addr);
		} while (0);

		jump = jump->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = (sljit_uw)(code_ptr - code) * sizeof(sljit_ins);

	code = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
	code_ptr = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

	SLJIT_CACHE_FLUSH(code, code_ptr);
	SLJIT_UPDATE_WX_FLAGS(code, code_ptr, 1);
	return code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	switch (feature_type) {
	case SLJIT_HAS_FPU:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#else
		return 1;
#endif
	case SLJIT_HAS_ZERO_REGISTER:
	case SLJIT_HAS_CMOV:
	case SLJIT_HAS_REV:
	case SLJIT_HAS_MEMORY_BARRIER:
		return 1;
	case SLJIT_HAS_COPY_F32:
	case SLJIT_HAS_COPY_F64:
		/* Uses ITOFS/ITOFT/FTOIS/FTOIT when the FIX extension (EV6+) is
		   available, and a stack round trip otherwise. */
		return 1;
	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_CTZ:
		if (!cpu_feature_list)
			get_cpu_features();
		return (cpu_feature_list & ALPHA_HWCAP_CIX) ? 1 : 0;
	case SLJIT_HAS_PREFETCH:
		return 1;
	case SLJIT_HAS_ROT:
	case SLJIT_HAS_ATOMIC:
	case SLJIT_HAS_SIMD:
	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	SLJIT_UNUSED_ARG(type);
	return 0;
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

/* Creates an index in data_transfer_insts array. */
#define LOAD_DATA	0x01
#define WORD_DATA	0x00
#define BYTE_DATA	0x02
#define HALF_DATA	0x04
#define INT_DATA	0x06
#define SIGNED_DATA	0x08
/* Separates integer and floating point registers */
#define GPR_REG		0x0f
#define DOUBLE_DATA	0x10
#define SINGLE_DATA	0x12

#define MEM_MASK	0x1f

#define ARG_TEST	0x00020
#define ALT_KEEP_CACHE	0x00040
#define CUMULATIVE_OP	0x00080
#define IMM_OP		0x00100
#define MOVE_OP		0x00200
#define SRC2_IMM	0x00400

#define UNUSED_DEST	0x00800
#define REG_DEST	0x01000
#define REG1_SOURCE	0x02000
#define REG2_SOURCE	0x04000
#define SLOW_SRC1	0x08000
#define SLOW_SRC2	0x10000
#define SLOW_DEST	0x20000
#define MEM_USE_TMP2	0x40000

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw);

/* Split a signed 32 bit value into an LDAH/LDA displacement pair, correcting
   the high part for the sign extension performed by the LDA. */
#define ALPHA_HI16(v) ((sljit_ins)((((sljit_u32)(v) >> 16) + (((v) & 0x8000) ? 1 : 0)) & 0xffff))
#define ALPHA_LO16(v) ((sljit_ins)((v) & 0xffff))

/* Emit "reg = base + (sljit_s32)v32". Normally an LDAH/LDA pair, but values in
   (0x7fff7fff, 0x7fffffff] need the high displacement 0x8000 which sign-extends
   negative, so split off 0x40000000 and use a three instruction sequence. */
static sljit_s32 emit_hi_lo(struct sljit_compiler *compiler, sljit_s32 reg, sljit_s32 base, sljit_s32 v32)
{
	sljit_s32 lo = (sljit_s32)(sljit_s16)(v32 & 0xffff);
	sljit_sw hi = ((sljit_sw)v32 - lo) >> 16;

	if (hi >= -0x8000 && hi <= 0x7fff) {
		FAIL_IF(push_inst(compiler, LDAH | RA(reg) | RB(base) | ((sljit_ins)hi & 0xffff)));
		return push_inst(compiler, LDA | RA(reg) | RB(reg) | ((sljit_ins)lo & 0xffff));
	}

	FAIL_IF(push_inst(compiler, LDAH | RA(reg) | RB(base) | 0x4000));
	v32 -= 0x40000000;
	lo = (sljit_s32)(sljit_s16)(v32 & 0xffff);
	hi = ((sljit_sw)v32 - lo) >> 16;
	FAIL_IF(push_inst(compiler, LDAH | RA(reg) | RB(reg) | ((sljit_ins)hi & 0xffff)));
	return push_inst(compiler, LDA | RA(reg) | RB(reg) | ((sljit_ins)lo & 0xffff));
}

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_r, sljit_sw imm, sljit_s32 tmp_r)
{
	sljit_sw high;
	sljit_s32 low;

	SLJIT_UNUSED_ARG(tmp_r);

	if (imm >= -0x8000 && imm <= 0x7fff)
		return push_inst(compiler, LDA | RA(dst_r) | RB(TMP_ZERO) | ALPHA_LO16(imm));

	if (imm >= S32_MIN && imm <= S32_MAX)
		return emit_hi_lo(compiler, dst_r, TMP_ZERO, (sljit_s32)imm);

	/* Full 64 bit constant: build the high half, shift and add the low half.
	   The low half is added back sign extended, so the high half is biased by
	   one when it is negative. That bias can push high up to 0x80000000, which
	   no longer fits in a signed 32 bit value, but only the low 32 bits of the
	   high half survive the shift, so computing it modulo 2^32 is equivalent. */
	low = (sljit_s32)imm;
	high = (sljit_sw)((sljit_uw)imm >> 32);
	if (low < 0)
		high++;

	FAIL_IF(emit_hi_lo(compiler, dst_r, TMP_ZERO, (sljit_s32)(sljit_u32)(sljit_uw)high));
	FAIL_IF(push_inst(compiler, SLL | RA(dst_r) | ALPHA_LIT(32) | RC(dst_r)));

	if (low == 0)
		return SLJIT_SUCCESS;

	return emit_hi_lo(compiler, dst_r, dst_r, low);
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value)
{
	sljit_sw high;
	sljit_s32 low;

	low = (sljit_s32)init_value;
	high = init_value >> 32;
	if (low < 0)
		high++;

	/* Fixed five instruction sequence so sljit_set_jump_addr can rewrite it. */
	FAIL_IF(push_inst(compiler, LDAH | RA(dst) | RB(TMP_ZERO) | ALPHA_HI16((sljit_s32)high)));
	FAIL_IF(push_inst(compiler, LDA | RA(dst) | RB(dst) | ALPHA_LO16((sljit_s32)high)));
	FAIL_IF(push_inst(compiler, SLL | RA(dst) | ALPHA_LIT(32) | RC(dst)));
	FAIL_IF(push_inst(compiler, LDAH | RA(dst) | RB(dst) | ALPHA_HI16(low)));
	return push_inst(compiler, LDA | RA(dst) | RB(dst) | ALPHA_LO16(low));
}

/* Emit a register move (BIS r31, src, dst). */
static SLJIT_INLINE sljit_s32 emit_mov(struct sljit_compiler *compiler, sljit_s32 dst, sljit_s32 src)
{
	if (dst == src)
		return SLJIT_SUCCESS;
	return push_inst(compiler, BIS | RA(TMP_ZERO) | RB(src) | RC(dst));
}

#define STACK_MAX_DISTANCE (-SIMM_MIN)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	sljit_s32 fscratches = ENTER_GET_FLOAT_REGS(scratches);
	sljit_s32 fsaveds = ENTER_GET_FLOAT_REGS(saveds);
	sljit_s32 i, tmp, offset;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);
	sljit_s32 arg_count;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, local_size);

	scratches = ENTER_GET_REGS(scratches);
	saveds = ENTER_GET_REGS(saveds);
	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 1);
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
	compiler->local_size = local_size;

	if (local_size <= STACK_MAX_DISTANCE) {
		offset = local_size - SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, LDA | RA(SLJIT_SP) | RB(SLJIT_SP) | DISP16(-local_size)));
		local_size = 0;
	} else {
		FAIL_IF(push_inst(compiler, LDA | RA(SLJIT_SP) | RB(SLJIT_SP) | DISP16(-STACK_MAX_DISTANCE)));
		local_size -= STACK_MAX_DISTANCE;

		if (local_size > STACK_MAX_DISTANCE)
			FAIL_IF(load_immediate(compiler, TMP_REG1, local_size, TMP_REG3));
		offset = STACK_MAX_DISTANCE - SSIZE_OF(sw);
	}

	FAIL_IF(push_inst(compiler, STQ | RA(RETURN_ADDR_REG) | RB(SLJIT_SP) | DISP16(offset)));

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STQ | RA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STQ | RA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, STT | FA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, STT | FA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	if (local_size > STACK_MAX_DISTANCE)
		FAIL_IF(push_inst(compiler, SUBQ | RA(SLJIT_SP) | RB(TMP_REG1) | RC(SLJIT_SP)));
	else if (local_size > 0)
		FAIL_IF(push_inst(compiler, LDA | RA(SLJIT_SP) | RB(SLJIT_SP) | DISP16(-local_size)));

	if (options & SLJIT_ENTER_REG_ARG)
		return SLJIT_SUCCESS;

	/* The Nth argument arrives in r(16 + N) (integer) or f(16 + N) (float).
	   Saved arguments are copied to SLJIT_S0.. / SLJIT_FS0..; scratch (_R)
	   arguments are copied to SLJIT_R0.. / SLJIT_FR0.. by type index. */
	arg_types >>= SLJIT_ARG_SHIFT;
	saved_arg_count = 0;
	arg_count = 0;
	offset = 0;            /* word arg index */
	i = 0;                 /* float arg index */

	while (arg_types > 0) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				FAIL_IF(push_inst(compiler, BIS | ABS_RA(16 + arg_count) | RB(TMP_ZERO) | RC(SLJIT_S0 - saved_arg_count)));
				saved_arg_count++;
			} else if (reg_map[SLJIT_R0 + offset] != 16 + arg_count)
				FAIL_IF(push_inst(compiler, BIS | ABS_RA(16 + arg_count) | RB(TMP_ZERO) | RC(SLJIT_R0 + offset)));
			offset++;
		} else {
			/* Floating point arguments land in SLJIT_FR0.. by float index. */
			if (freg_map[SLJIT_FR0 + i] != 16 + arg_count)
				FAIL_IF(push_inst(compiler, CPYS | ((sljit_ins)(16 + arg_count) << 21) | ((sljit_ins)(16 + arg_count) << 16) | FC(SLJIT_FR0 + i)));
			i++;
		}

		arg_count++;
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

#undef STACK_MAX_DISTANCE

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	sljit_s32 fscratches = ENTER_GET_FLOAT_REGS(scratches);
	sljit_s32 fsaveds = ENTER_GET_FLOAT_REGS(saveds);

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, local_size);

	scratches = ENTER_GET_REGS(scratches);
	saveds = ENTER_GET_REGS(saveds);
	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 1);
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);
	compiler->local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;

	return SLJIT_SUCCESS;
}

#define STACK_MAX_DISTANCE (-SIMM_MIN - 16)

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_s32 i, tmp, offset;
	sljit_s32 local_size = compiler->local_size;

	if (local_size > STACK_MAX_DISTANCE) {
		local_size -= STACK_MAX_DISTANCE;

		if (local_size > STACK_MAX_DISTANCE) {
			FAIL_IF(load_immediate(compiler, TMP_REG2, local_size, TMP_REG3));
			FAIL_IF(push_inst(compiler, ADDQ | RA(SLJIT_SP) | RB(TMP_REG2) | RC(SLJIT_SP)));
		} else
			FAIL_IF(push_inst(compiler, LDA | RA(SLJIT_SP) | RB(SLJIT_SP) | DISP16(local_size)));

		local_size = STACK_MAX_DISTANCE;
	}

	offset = local_size - SSIZE_OF(sw);
	if (!is_return_to)
		FAIL_IF(push_inst(compiler, LDQ | RA(RETURN_ADDR_REG) | RB(SLJIT_SP) | DISP16(offset)));

	tmp = SLJIT_S0 - compiler->saveds;
	for (i = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options); i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, LDQ | RA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, LDQ | RA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	tmp = SLJIT_FS0 - compiler->fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, LDT | FA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	for (i = compiler->fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, LDT | FA(i) | RB(SLJIT_SP) | DISP16(offset)));
	}

	return push_inst(compiler, LDA | RA(SLJIT_SP) | RB(SLJIT_SP) | DISP16(local_size));
}

#undef STACK_MAX_DISTANCE

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	FAIL_IF(emit_stack_frame_release(compiler, 0));
	return push_inst(compiler, ALPHA_RET);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
		src = TMP_REG1;
		srcw = 0;
	} else if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		FAIL_IF(emit_mov(compiler, TMP_REG1, src));
		src = TMP_REG1;
		srcw = 0;
	}

	FAIL_IF(emit_stack_frame_release(compiler, 1));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, SLJIT_JUMP, src, srcw);
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

static const sljit_ins data_transfer_insts[16 + 4] = {
/* u w s */ STQ,
/* u w l */ LDQ,
/* u b s */ STB,
/* u b l */ LDBU,
/* u h s */ STW,
/* u h l */ LDWU,
/* u i s */ STL,
/* u i l */ LDL,

/* s w s */ STQ,
/* s w l */ LDQ,
/* s b s */ STB,
/* s b l */ LDBU,
/* s h s */ STW,
/* s h l */ LDWU,
/* s i s */ STL,
/* s i l */ LDL,

/* d   s */ STT,
/* d   l */ LDT,
/* s   s */ STS,
/* s   l */ LDS,
};

/* Synthesize byte or halfword load/store on pre-EV56 Alpha (no BWX extension).
 * Uses LDQ_U + EXTBL/EXTWL[H] / INSBL/INSWL[H] / MSKBL/MSKWL[H] + STQ_U.
 * base must be a register; offset is a signed 16-bit displacement. */
static sljit_s32 emit_no_bwx_mem(struct sljit_compiler *compiler, sljit_s32 flags,
	sljit_s32 reg, sljit_s32 base, sljit_sw offset)
{
	sljit_s32 is_byte = ((flags & 0x6) == BYTE_DATA);
	sljit_s32 is_signed = (flags & SIGNED_DATA) != 0;

	/* TMP_REG1 = actual byte address; LDQ_U will mask the low 3 bits to align. */
	FAIL_IF(push_inst(compiler, LDA | RA(TMP_REG1) | RB(base) | DISP16(offset)));

	if (flags & LOAD_DATA) {
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));

		if (is_byte) {
			FAIL_IF(push_inst(compiler, EXTBL | RA(TMP_REG2) | RB(TMP_REG1) | RC(reg)));
			if (is_signed) {
				FAIL_IF(push_inst(compiler, SLL | RA(reg) | ALPHA_LIT(56) | RC(reg)));
				return push_inst(compiler, SRA | RA(reg) | ALPHA_LIT(56) | RC(reg));
			}
			return SLJIT_SUCCESS;
		}

		/* Halfword may span two aligned quadwords; load both. */
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG3) | RB(TMP_REG1) | DISP16(1)));
		FAIL_IF(push_inst(compiler, EXTWL | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, EXTWH | RA(TMP_REG3) | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(reg)));
		if (is_signed) {
			FAIL_IF(push_inst(compiler, SLL | RA(reg) | ALPHA_LIT(48) | RC(reg)));
			return push_inst(compiler, SRA | RA(reg) | ALPHA_LIT(48) | RC(reg));
		}
		return SLJIT_SUCCESS;
	}

	/* Store: INSBL/INSWL read reg before LDQ_U clobbers TMP_REG2, but for
	   halfword stores the data must survive two LDQ_U/INSWx pairs.  Save to
	   TMP_REG4 (r29/gp, never used in JIT code) when reg aliases a scratch. */

	if (is_byte) {
		/* Emit INSBL first so reg may safely alias TMP_REG2 or TMP_REG3. */
		FAIL_IF(push_inst(compiler, INSBL | RA(reg)      | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
		FAIL_IF(push_inst(compiler, MSKBL | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
		return push_inst(compiler, STQ_U  | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0));
	}

	/* Halfword store: data must survive two LDQ_U loads; save if aliased. */
	if (reg == TMP_REG2 || reg == TMP_REG3) {
		FAIL_IF(push_inst(compiler, BIS | RA(reg) | RB(TMP_ZERO) | RC(TMP_REG4)));
		reg = TMP_REG4;
	}
	/* Lower aligned quadword. */
	FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
	FAIL_IF(push_inst(compiler, MSKWL | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
	FAIL_IF(push_inst(compiler, INSWL | RA(reg)      | RB(TMP_REG1) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
	FAIL_IF(push_inst(compiler, STQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
	/* Upper aligned quadword. */
	FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(1)));
	FAIL_IF(push_inst(compiler, MSKWH | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
	FAIL_IF(push_inst(compiler, INSWH | RA(reg)      | RB(TMP_REG1) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
	return push_inst(compiler, STQ_U  | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(1));
}

static sljit_s32 push_mem_inst(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 base, sljit_sw offset)
{
	sljit_ins ins;

	SLJIT_ASSERT(FAST_IS_REG(base) && offset <= SIMM_MAX && offset >= SIMM_MIN);

	/* LDBU/LDWU/STB/STW require the BWX extension (EV56+).  Synthesize
	 * using LDQ_U + EXT/INS/MSK + STQ_U on pre-EV56 hardware.  Only integer
	 * accesses qualify: SINGLE_DATA also has the BYTE_DATA bits set, but it
	 * transfers a floating point register with LDS/STS, which always exist. */
	if ((flags & MEM_MASK) <= GPR_REG
			&& ((flags & 0x6) == BYTE_DATA || (flags & 0x6) == HALF_DATA)) {
		if (!cpu_feature_list)
			get_cpu_features();
		if (!(cpu_feature_list & ALPHA_HWCAP_BWX))
			return emit_no_bwx_mem(compiler, flags, reg, base, offset);
	}

	ins = data_transfer_insts[flags & MEM_MASK] | RB(base) | DISP16(offset);
	if ((flags & MEM_MASK) <= GPR_REG)
		ins |= RA(reg);
	else
		ins |= FA(reg);

	return push_inst(compiler, ins);
}

/* Can perform an operation using at most 1 instruction. */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (!(arg & OFFS_REG_MASK) && argw <= SIMM_MAX && argw >= SIMM_MIN) {
		if (SLJIT_UNLIKELY(flags & ARG_TEST))
			return 1;

		FAIL_IF(push_mem_inst(compiler, flags, reg, arg & REG_MASK, argw));
		return -1;
	}
	return 0;
}

#define TO_ARGW_HI(argw) (((argw) & ~0xffff) + (((argw) & 0x8000) ? 0x10000 : 0))

static sljit_s32 can_cache(sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	SLJIT_ASSERT((arg & SLJIT_MEM) && (next_arg & SLJIT_MEM));

	if (arg & OFFS_REG_MASK) {
		argw &= 0x3;
		next_argw &= 0x3;
		if (argw && argw == next_argw && (arg == next_arg || (arg & OFFS_REG_MASK) == (next_arg & OFFS_REG_MASK)))
			return 1;
		return 0;
	}

	if (arg == next_arg) {
		if (((next_argw - argw) <= SIMM_MAX && (next_argw - argw) >= SIMM_MIN)
				|| TO_ARGW_HI(argw) == TO_ARGW_HI(next_argw))
			return 1;
		return 0;
	}

	return 0;
}

static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_s32 base = arg & REG_MASK;
	sljit_s32 tmp_r = (flags & MEM_USE_TMP2) ? TMP_REG2 : TMP_REG1;
	sljit_sw offset;

	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (SLJIT_UNLIKELY(argw)) {
			FAIL_IF(push_inst(compiler, SLL | RA(OFFS_REG(arg)) | ALPHA_LIT(argw) | RC(tmp_r)));
			FAIL_IF(push_inst(compiler, ADDQ | RA(tmp_r) | RB(base) | RC(tmp_r)));
		} else
			FAIL_IF(push_inst(compiler, ADDQ | RA(base) | RB(OFFS_REG(arg)) | RC(tmp_r)));

		return push_mem_inst(compiler, flags, reg, tmp_r, 0);
	}

	/* No address cache: materialize the high part and keep the low 16 bits
	   as the memory displacement. */
	offset = argw - TO_ARGW_HI(argw);
	FAIL_IF(load_immediate(compiler, tmp_r, TO_ARGW_HI(argw), TMP_REG3));

	if (base)
		FAIL_IF(push_inst(compiler, ADDQ | RA(tmp_r) | RB(base) | RC(tmp_r)));

	return push_mem_inst(compiler, flags, reg, tmp_r, offset);
}

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_s32 base = arg & REG_MASK;
	sljit_s32 tmp_r = TMP_REG1;

	if (getput_arg_fast(compiler, flags, reg, arg, argw))
		return compiler->error;

	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA))
		tmp_r = reg;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (SLJIT_UNLIKELY(argw)) {
			FAIL_IF(push_inst(compiler, SLL | RA(OFFS_REG(arg)) | ALPHA_LIT(argw) | RC(tmp_r)));
			FAIL_IF(push_inst(compiler, ADDQ | RA(tmp_r) | RB(base) | RC(tmp_r)));
		} else
			FAIL_IF(push_inst(compiler, ADDQ | RA(base) | RB(OFFS_REG(arg)) | RC(tmp_r)));

		argw = 0;
	} else {
		FAIL_IF(load_immediate(compiler, tmp_r, TO_ARGW_HI(argw), TMP_REG3));

		if (base != 0)
			FAIL_IF(push_inst(compiler, ADDQ | RA(tmp_r) | RB(base) | RC(tmp_r)));

		argw -= TO_ARGW_HI(argw);
	}

	return push_mem_inst(compiler, flags, reg, tmp_r, argw);
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w);
}

/* Select ADDQ/ADDL etc. based on the SLJIT_32 flag. */
#define SELECT_OP(op, q, l) (((op) & SLJIT_32) ? (l) : (q))

/* Sign extend the low 32 bits when the result must stay canonical, and deposit
   the zero flag when requested (only shift opcodes route SLJIT_SET_Z here). */
static SLJIT_INLINE sljit_s32 emit_sext32(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst)
{
	if (op & SLJIT_32)
		FAIL_IF(push_inst(compiler, ADDL | RA(dst) | RB(TMP_ZERO) | RC(dst)));
	if (op & SLJIT_SET_Z)
		return push_inst(compiler, BIS | RA(TMP_ZERO) | RB(dst) | RC(EQUAL_FLAG));
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_rev(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_s32 src)
{
	sljit_s32 n, i;
	sljit_s32 res = EQUAL_FLAG;

	switch (GET_OPCODE(op)) {
	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		n = 2;
		break;
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		n = 4;
		break;
	default:
		n = (op & SLJIT_32) ? 4 : 8;
		break;
	}

	FAIL_IF(push_inst(compiler, BIS | RA(TMP_ZERO) | RB(TMP_ZERO) | RC(res)));
	for (i = 0; i < n; i++) {
		FAIL_IF(push_inst(compiler, EXTBL | RA(src) | ALPHA_LIT(i) | RC(TMP_REG1)));
		FAIL_IF(push_inst(compiler, INSBL | RA(TMP_REG1) | ALPHA_LIT(n - 1 - i) | RC(TMP_REG1)));
		FAIL_IF(push_inst(compiler, BIS | RA(res) | RB(TMP_REG1) | RC(res)));
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_REV_U16:
		return push_inst(compiler, ZAPNOT | RA(res) | ALPHA_LIT(0x03) | RC(dst));
	case SLJIT_REV_S16:
		return push_inst(compiler, SEXTW | RA(TMP_ZERO) | RB(res) | RC(dst));
	case SLJIT_REV_U32:
		return push_inst(compiler, ZAPNOT | RA(res) | ALPHA_LIT(0x0f) | RC(dst));
	case SLJIT_REV_S32:
		return push_inst(compiler, ADDL | RA(res) | RB(TMP_ZERO) | RC(dst));
	default:
		return emit_mov(compiler, dst, res);
	}
}

static sljit_s32 emit_cmp_flag(struct sljit_compiler *compiler, sljit_s32 flag_type, sljit_s32 src1, sljit_s32 src2)
{
	/* Deposit the strict comparison result into OTHER_FLAG. */
	switch (flag_type) {
	case SLJIT_LESS:
	case SLJIT_GREATER_EQUAL:
	case SLJIT_CARRY:
	case SLJIT_NOT_CARRY:
		return push_inst(compiler, CMPULT | RA(src1) | RB(src2) | RC(OTHER_FLAG));
	case SLJIT_GREATER:
	case SLJIT_LESS_EQUAL:
		return push_inst(compiler, CMPULT | RA(src2) | RB(src1) | RC(OTHER_FLAG));
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER_EQUAL:
		return push_inst(compiler, CMPLT | RA(src1) | RB(src2) | RC(OTHER_FLAG));
	case SLJIT_SIG_GREATER:
	case SLJIT_SIG_LESS_EQUAL:
		return push_inst(compiler, CMPLT | RA(src2) | RB(src1) | RC(OTHER_FLAG));
	}
	return SLJIT_SUCCESS;
}

/* Count leading zeroes of a 64 bit value without the CIX extension (pre-EV67).
   Branchless binary search: at each step the top half of the remaining window
   is tested, and is either skipped (adding its width to the count) or becomes
   the new window.  A zero input yields 64.

   src_r is read by the first instruction only, so it may be any of the scratch
   registers this clobbers (OTHER_FLAG, EQUAL_FLAG, TMP_REG3). */
static sljit_s32 emit_clz64_no_cix(struct sljit_compiler *compiler, sljit_s32 dst_r, sljit_s32 src_r)
{
	sljit_s32 shift;

	/* The window under consideration, and the count of zeroes above it. */
	FAIL_IF(push_inst(compiler, BIS | RA(src_r) | RB(TMP_ZERO) | RC(OTHER_FLAG)));
	FAIL_IF(push_inst(compiler, LDA | RA(dst_r) | RB(TMP_ZERO) | DISP16(0)));

	for (shift = 32; shift > 0; shift >>= 1) {
		FAIL_IF(push_inst(compiler, SRL | RA(OTHER_FLAG) | ALPHA_LIT(shift) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, ADDQ | RA(dst_r) | ALPHA_LIT(shift) | RC(TMP_REG3)));
		/* Top half empty: skip it and count its width. */
		FAIL_IF(push_inst(compiler, CMOVEQ | RA(EQUAL_FLAG) | RB(TMP_REG3) | RC(dst_r)));
		/* Otherwise the top half becomes the new window. */
		FAIL_IF(push_inst(compiler, CMOVNE | RA(EQUAL_FLAG) | RB(EQUAL_FLAG) | RC(OTHER_FLAG)));
	}

	/* The window is now a single bit; if it is clear the input was zero. */
	FAIL_IF(push_inst(compiler, ADDQ | RA(dst_r) | ALPHA_LIT(1) | RC(TMP_REG3)));
	return push_inst(compiler, CMOVEQ | RA(OTHER_FLAG) | RB(TMP_REG3) | RC(dst_r));
}

/* SLJIT_CLZ / SLJIT_CTZ on CPUs without CIX. */
static sljit_s32 emit_clz_ctz_no_cix(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 opcode, sljit_s32 dst_r, sljit_s32 src_r)
{
	sljit_s32 src = src_r;

	if (op & SLJIT_32) {
		/* Operate on the zero extended low word. */
		FAIL_IF(push_inst(compiler, ZAPNOT | RA(src_r) | ALPHA_LIT(0x0f) | RC(TMP_REG3)));
		src = TMP_REG3;
	}

	if (opcode == SLJIT_CLZ) {
		FAIL_IF(emit_clz64_no_cix(compiler, dst_r, src));

		/* A zero extended word has 32 extra leading zeroes, and a zero input
		   correctly ends up as 64 - 32. */
		if (op & SLJIT_32)
			return push_inst(compiler, SUBQ | RA(dst_r) | ALPHA_LIT(32) | RC(dst_r));
		return SLJIT_SUCCESS;
	}

	/* ctz(x) == 64 - clz(~x & (x - 1)), which needs no special case for zero:
	   the mask is then all ones, so clz is 0 and the result is 64. */
	FAIL_IF(push_inst(compiler, LDA | RA(EQUAL_FLAG) | RB(src) | DISP16(-1)));
	FAIL_IF(push_inst(compiler, BIC | RA(EQUAL_FLAG) | RB(src) | RC(EQUAL_FLAG)));
	FAIL_IF(emit_clz64_no_cix(compiler, dst_r, EQUAL_FLAG));
	FAIL_IF(push_inst(compiler, LDA | RA(OTHER_FLAG) | RB(TMP_ZERO) | DISP16(64)));
	FAIL_IF(push_inst(compiler, SUBQ | RA(OTHER_FLAG) | RB(dst_r) | RC(dst_r)));

	if (!(op & SLJIT_32))
		return SLJIT_SUCCESS;

	/* A zero low word gives 64 here, but the 32 bit result must be 32.  No
	   other input can reach 64, so the value alone identifies the case. */
	FAIL_IF(push_inst(compiler, CMPEQ | RA(dst_r) | ALPHA_LIT(64) | RC(EQUAL_FLAG)));
	FAIL_IF(push_inst(compiler, LDA | RA(OTHER_FLAG) | RB(TMP_ZERO) | DISP16(32)));
	return push_inst(compiler, CMOVNE | RA(EQUAL_FLAG) | RB(OTHER_FLAG) | RC(dst_r));
}

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_s32 opcode = GET_OPCODE(op);
	sljit_s32 flag_type = GET_FLAG_TYPE(op);
	sljit_ins op_imm, op_reg;
	sljit_s32 src2r = (sljit_s32)src2;

	switch (opcode) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return emit_mov(compiler, dst, src2r);

	case SLJIT_MOV_U8:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return push_inst(compiler, ZAPNOT | RA(src2r) | ALPHA_LIT(0x01) | RC(dst));

	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return push_inst(compiler, SEXTB | RA(TMP_ZERO) | RB(src2r) | RC(dst));

	case SLJIT_MOV_U16:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return push_inst(compiler, ZAPNOT | RA(src2r) | ALPHA_LIT(0x03) | RC(dst));

	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return push_inst(compiler, SEXTW | RA(TMP_ZERO) | RB(src2r) | RC(dst));

	case SLJIT_MOV_U32:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return push_inst(compiler, ZAPNOT | RA(src2r) | ALPHA_LIT(0x0f) | RC(dst));

	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return push_inst(compiler, ADDL | RA(src2r) | RB(TMP_ZERO) | RC(dst));

	case SLJIT_CLZ:
	case SLJIT_CTZ:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		/* CTLZ/CTTZ require the CIX extension (EV67+). */
		if (!cpu_feature_list)
			get_cpu_features();
		if (!(cpu_feature_list & ALPHA_HWCAP_CIX))
			return emit_clz_ctz_no_cix(compiler, op, opcode, dst, src2r);

		if (op & SLJIT_32) {
			if (opcode == SLJIT_CLZ) {
				FAIL_IF(push_inst(compiler, ZAPNOT | RA(src2r) | ALPHA_LIT(0x0f) | RC(OTHER_FLAG)));
				FAIL_IF(push_inst(compiler, CTLZ | RA(TMP_ZERO) | RB(OTHER_FLAG) | RC(dst)));
				return push_inst(compiler, SUBQ | RA(dst) | ALPHA_LIT(32) | RC(dst));
			}
			/* Trailing zeroes of a 32 bit value; the all-zero case is 32. */
			FAIL_IF(push_inst(compiler, ZAPNOT | RA(src2r) | ALPHA_LIT(0x0f) | RC(OTHER_FLAG)));
			FAIL_IF(push_inst(compiler, CTTZ | RA(TMP_ZERO) | RB(OTHER_FLAG) | RC(dst)));
			FAIL_IF(push_inst(compiler, LDA | RA(EQUAL_FLAG) | RB(TMP_ZERO) | DISP16(32)));
			return push_inst(compiler, CMOVEQ | RA(OTHER_FLAG) | RB(EQUAL_FLAG) | RC(dst));
		}
		return push_inst(compiler, ((opcode == SLJIT_CLZ) ? CTLZ : CTTZ) | RA(TMP_ZERO) | RB(src2r) | RC(dst));

	case SLJIT_REV:
	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		SLJIT_ASSERT(src1 == TMP_ZERO && !(flags & SRC2_IMM));
		return emit_rev(compiler, op, dst, src2r);

	case SLJIT_ADD: {
		sljit_s32 is_carry = (flag_type == SLJIT_CARRY);
		sljit_s32 is_ovf = (flag_type == SLJIT_OVERFLOW);
		sljit_s32 carry_r = 0;

		if (is_ovf) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, XOR | RA(src1) | ALPHA_LIT(src2) | RC(EQUAL_FLAG)));
			else
				FAIL_IF(push_inst(compiler, XOR | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
		}

		if (is_carry || is_ovf) {
			if (dst != src1)
				carry_r = src1;
			else if (!(flags & SRC2_IMM) && dst != src2r)
				carry_r = src2r;
			else {
				FAIL_IF(emit_mov(compiler, OTHER_FLAG, src1));
				carry_r = OTHER_FLAG;
			}
		}

		{
			/* Flags-only + zero flag: add straight into EQUAL_FLAG so a caller
			   value held in the (unused) destination TMP_REG2 is preserved. */
			sljit_s32 add_dst = ((op & SLJIT_SET_Z) && (flags & UNUSED_DEST) && !is_carry && !is_ovf) ? EQUAL_FLAG : dst;

			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SELECT_OP(op, ADDQ, ADDL) | RA(src1) | ALPHA_LIT(src2) | RC(add_dst)));
			else
				FAIL_IF(push_inst(compiler, SELECT_OP(op, ADDQ, ADDL) | RA(src1) | RB(src2r) | RC(add_dst)));

			if (is_carry || is_ovf)
				FAIL_IF(push_inst(compiler, CMPULT | RA(dst) | RB(carry_r) | RC(OTHER_FLAG)));

			if (is_ovf) {
				FAIL_IF(push_inst(compiler, XOR | RA(dst) | RB(EQUAL_FLAG) | RC(TMP_REG1)));
				if (op & SLJIT_SET_Z)
					FAIL_IF(emit_mov(compiler, EQUAL_FLAG, dst));
				/* Test the sign at the operation width: bit 31 for 32 bit ops
				   (sign extend the low word first), bit 63 otherwise. */
				if (op & SLJIT_32)
					FAIL_IF(push_inst(compiler, ADDL | RA(TMP_REG1) | RB(TMP_ZERO) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, CMPLT | RA(TMP_REG1) | RB(TMP_ZERO) | RC(TMP_REG1)));
				return push_inst(compiler, XOR | RA(TMP_REG1) | RB(OTHER_FLAG) | RC(OTHER_FLAG));
			}

			if ((op & SLJIT_SET_Z) && add_dst != EQUAL_FLAG)
				FAIL_IF(emit_mov(compiler, EQUAL_FLAG, add_dst));
		}
		return SLJIT_SUCCESS;
	}

	case SLJIT_ADDC: {
		sljit_s32 is_carry = (flag_type == SLJIT_CARRY);
		sljit_s32 carry_r = 0;

		if (is_carry) {
			if (dst != src1)
				carry_r = src1;
			else if (!(flags & SRC2_IMM) && dst != src2r)
				carry_r = src2r;
			else {
				FAIL_IF(emit_mov(compiler, EQUAL_FLAG, src1));
				carry_r = EQUAL_FLAG;
			}
		}

		if (flags & SRC2_IMM)
			FAIL_IF(push_inst(compiler, SELECT_OP(op, ADDQ, ADDL) | RA(src1) | ALPHA_LIT(src2) | RC(dst)));
		else
			FAIL_IF(push_inst(compiler, SELECT_OP(op, ADDQ, ADDL) | RA(src1) | RB(src2r) | RC(dst)));

		if (is_carry)
			FAIL_IF(push_inst(compiler, CMPULT | RA(dst) | RB(carry_r) | RC(EQUAL_FLAG)));

		FAIL_IF(push_inst(compiler, SELECT_OP(op, ADDQ, ADDL) | RA(dst) | RB(OTHER_FLAG) | RC(dst)));

		if (!is_carry)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, CMPULT | RA(dst) | RB(OTHER_FLAG) | RC(OTHER_FLAG)));
		return push_inst(compiler, BIS | RA(OTHER_FLAG) | RB(EQUAL_FLAG) | RC(OTHER_FLAG));
	}

	case SLJIT_SUB: {
		sljit_s32 is_carry = (flag_type == SLJIT_CARRY);
		sljit_s32 is_ovf = (flag_type == SLJIT_OVERFLOW);

		if (flag_type >= SLJIT_LESS && flag_type <= SLJIT_SIG_LESS_EQUAL) {
			if (flags & SRC2_IMM) {
				/* Use a scratch that is not SLJIT_TMP_DEST_REG (TMP_REG2) so a
				   caller value held there is not clobbered by the comparison. */
				sljit_s32 imm_reg = (src1 == TMP_REG1) ? TMP_REG2 : TMP_REG1;
				FAIL_IF(load_immediate(compiler, imm_reg, src2, TMP_REG3));
				src2r = imm_reg;
				flags &= ~SRC2_IMM;
			}
			FAIL_IF(emit_cmp_flag(compiler, flag_type, src1, src2r));

			/* Only the comparison flag (and optionally Z) is needed; skip the
			   subtraction when the destination is unused. */
			if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
			if (!(flags & UNUSED_DEST))
				FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(src1) | RB(src2r) | RC(dst)));
			return SLJIT_SUCCESS;
		}

		if (is_ovf) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, XOR | RA(src1) | ALPHA_LIT(src2) | RC(EQUAL_FLAG)));
			else
				FAIL_IF(push_inst(compiler, XOR | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
		}

		if (is_carry || is_ovf) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, CMPULT | RA(src1) | ALPHA_LIT(src2) | RC(OTHER_FLAG)));
			else
				FAIL_IF(push_inst(compiler, CMPULT | RA(src1) | RB(src2r) | RC(OTHER_FLAG)));
		}

		/* When only the zero flag is needed and the destination is unused,
		   subtract straight into EQUAL_FLAG so a caller value held in dst
		   (SLJIT_TMP_DEST_REG maps to TMP_REG2) is preserved. */
		{
			sljit_s32 sub_dst = ((op & SLJIT_SET_Z) && (flags & UNUSED_DEST) && !is_ovf) ? EQUAL_FLAG : dst;

			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(src1) | ALPHA_LIT(src2) | RC(sub_dst)));
			else
				FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(src1) | RB(src2r) | RC(sub_dst)));

			if (is_ovf) {
				FAIL_IF(push_inst(compiler, XOR | RA(dst) | RB(EQUAL_FLAG) | RC(TMP_REG1)));
				if (op & SLJIT_SET_Z)
					FAIL_IF(emit_mov(compiler, EQUAL_FLAG, dst));
				/* Test the sign at the operation width: bit 31 for 32 bit ops
				   (sign extend the low word first), bit 63 otherwise. */
				if (op & SLJIT_32)
					FAIL_IF(push_inst(compiler, ADDL | RA(TMP_REG1) | RB(TMP_ZERO) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, CMPLT | RA(TMP_REG1) | RB(TMP_ZERO) | RC(TMP_REG1)));
				return push_inst(compiler, XOR | RA(TMP_REG1) | RB(OTHER_FLAG) | RC(OTHER_FLAG));
			}

			if ((op & SLJIT_SET_Z) && sub_dst != EQUAL_FLAG)
				FAIL_IF(emit_mov(compiler, EQUAL_FLAG, sub_dst));
		}
		return SLJIT_SUCCESS;
	}

	case SLJIT_SUBC: {
		sljit_s32 is_carry = (flag_type == SLJIT_CARRY);

		if (is_carry) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, CMPULT | RA(src1) | ALPHA_LIT(src2) | RC(EQUAL_FLAG)));
			else
				FAIL_IF(push_inst(compiler, CMPULT | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
		}

		if (flags & SRC2_IMM)
			FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(src1) | ALPHA_LIT(src2) | RC(dst)));
		else
			FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(src1) | RB(src2r) | RC(dst)));

		if (is_carry)
			FAIL_IF(push_inst(compiler, CMPULT | RA(dst) | RB(OTHER_FLAG) | RC(TMP_REG1)));

		FAIL_IF(push_inst(compiler, SELECT_OP(op, SUBQ, SUBL) | RA(dst) | RB(OTHER_FLAG) | RC(dst)));

		if (!is_carry)
			return SLJIT_SUCCESS;

		return push_inst(compiler, BIS | RA(EQUAL_FLAG) | RB(TMP_REG1) | RC(OTHER_FLAG));
	}

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & SRC2_IMM));

		if (flag_type != SLJIT_OVERFLOW)
			return push_inst(compiler, SELECT_OP(op, MULQ, MULL) | RA(src1) | RB(src2r) | RC(dst));

		if (op & SLJIT_32) {
			/* Overflow if the full product differs from its low 32 bits. */
			FAIL_IF(push_inst(compiler, MULQ | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
			FAIL_IF(push_inst(compiler, MULL | RA(src1) | RB(src2r) | RC(dst)));
			FAIL_IF(push_inst(compiler, XOR | RA(EQUAL_FLAG) | RB(dst) | RC(EQUAL_FLAG)));
			return push_inst(compiler, CMPULT | RA(TMP_ZERO) | RB(EQUAL_FLAG) | RC(OTHER_FLAG));
		}

		/* Signed 64x64: overflow if the signed high half differs from the sign
		   extension of the low half. Signed high = UMULH - (a<0?b) - (b<0?a). */
		FAIL_IF(push_inst(compiler, UMULH | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, SRA | RA(src1) | ALPHA_LIT(63) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, AND | RA(TMP_REG3) | RB(src2r) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(EQUAL_FLAG) | RB(TMP_REG3) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, SRA | RA(src2r) | ALPHA_LIT(63) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, AND | RA(TMP_REG3) | RB(src1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(EQUAL_FLAG) | RB(TMP_REG3) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, MULQ | RA(src1) | RB(src2r) | RC(dst)));
		FAIL_IF(push_inst(compiler, SRA | RA(dst) | ALPHA_LIT(63) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, XOR | RA(EQUAL_FLAG) | RB(TMP_REG3) | RC(EQUAL_FLAG)));
		return push_inst(compiler, CMPULT | RA(TMP_ZERO) | RB(EQUAL_FLAG) | RC(OTHER_FLAG));

	case SLJIT_AND:
		op_imm = AND;
		op_reg = AND;
		break;
	case SLJIT_OR:
		op_imm = BIS;
		op_reg = BIS;
		break;
	case SLJIT_XOR:
		op_imm = XOR;
		op_reg = XOR;
		break;

	case SLJIT_SHL:
	case SLJIT_MSHL:
		if (op & SLJIT_32) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLL | RA(src1) | ALPHA_LIT(src2 & 0x1f) | RC(dst)));
			else {
				FAIL_IF(push_inst(compiler, AND | RA(src2r) | ALPHA_LIT(0x1f) | RC(TMP_REG2)));
				FAIL_IF(push_inst(compiler, SLL | RA(src1) | RB(TMP_REG2) | RC(dst)));
			}
			return emit_sext32(compiler, op, dst);
		}
		op_imm = SLL;
		op_reg = SLL;
		break;

	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		if (op & SLJIT_32) {
			if (flags & SRC2_IMM) {
				FAIL_IF(push_inst(compiler, ZAPNOT | RA(src1) | ALPHA_LIT(0x0f) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, SRL | RA(TMP_REG1) | ALPHA_LIT(src2 & 0x1f) | RC(dst)));
			} else {
				/* Mask the count before src1 gets zero-extended (they may share a temp). */
				FAIL_IF(push_inst(compiler, AND | RA(src2r) | ALPHA_LIT(0x1f) | RC(TMP_REG2)));
				FAIL_IF(push_inst(compiler, ZAPNOT | RA(src1) | ALPHA_LIT(0x0f) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, SRL | RA(TMP_REG1) | RB(TMP_REG2) | RC(dst)));
			}
			return emit_sext32(compiler, op, dst);
		}
		op_imm = SRL;
		op_reg = SRL;
		break;

	case SLJIT_ASHR:
	case SLJIT_MASHR:
		if (op & SLJIT_32) {
			if (flags & SRC2_IMM) {
				FAIL_IF(push_inst(compiler, ADDL | RA(src1) | RB(TMP_ZERO) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, SRA | RA(TMP_REG1) | ALPHA_LIT(src2 & 0x1f) | RC(dst)));
			} else {
				FAIL_IF(push_inst(compiler, AND | RA(src2r) | ALPHA_LIT(0x1f) | RC(TMP_REG2)));
				FAIL_IF(push_inst(compiler, ADDL | RA(src1) | RB(TMP_ZERO) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, SRA | RA(TMP_REG1) | RB(TMP_REG2) | RC(dst)));
			}
			return emit_sext32(compiler, op, dst);
		}
		op_imm = SRA;
		op_reg = SRA;
		break;

	case SLJIT_ROTL:
	case SLJIT_ROTR: {
		sljit_s32 bits = (op & SLJIT_32) ? 32 : 64;
		sljit_s32 is_left = (opcode == SLJIT_ROTL);
		sljit_s32 in = src1;
		sljit_ins first = is_left ? SLL : SRL;
		sljit_ins second = is_left ? SRL : SLL;

		if (flags & SRC2_IMM) {
			sljit_sw sh = src2 & (bits - 1);
			if (op & SLJIT_32) {
				FAIL_IF(push_inst(compiler, ZAPNOT | RA(src1) | ALPHA_LIT(0x0f) | RC(TMP_REG1)));
				in = TMP_REG1;
			}
			if (sh == 0) {
				FAIL_IF(emit_mov(compiler, dst, src1));
				return emit_sext32(compiler, op, dst);
			}
			FAIL_IF(push_inst(compiler, first | RA(in) | ALPHA_LIT(sh) | RC(OTHER_FLAG)));
			FAIL_IF(push_inst(compiler, second | RA(in) | ALPHA_LIT(bits - sh) | RC(dst)));
			FAIL_IF(push_inst(compiler, BIS | RA(dst) | RB(OTHER_FLAG) | RC(dst)));
			return emit_sext32(compiler, op, dst);
		}

		if (op & SLJIT_32) {
			/* Mask the count before src1 gets zero-extended (shared temp). */
			FAIL_IF(push_inst(compiler, AND | RA(src2r) | ALPHA_LIT(0x1f) | RC(TMP_REG2)));
			FAIL_IF(push_inst(compiler, ZAPNOT | RA(src1) | ALPHA_LIT(0x0f) | RC(TMP_REG1)));
			FAIL_IF(push_inst(compiler, LDA | RA(EQUAL_FLAG) | RB(TMP_ZERO) | DISP16(32)));
			FAIL_IF(push_inst(compiler, SUBQ | RA(EQUAL_FLAG) | RB(TMP_REG2) | RC(EQUAL_FLAG)));
			FAIL_IF(push_inst(compiler, first | RA(TMP_REG1) | RB(TMP_REG2) | RC(OTHER_FLAG)));
			FAIL_IF(push_inst(compiler, second | RA(TMP_REG1) | RB(EQUAL_FLAG) | RC(dst)));
			FAIL_IF(push_inst(compiler, BIS | RA(dst) | RB(OTHER_FLAG) | RC(dst)));
			return emit_sext32(compiler, op, dst);
		}

		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_ZERO) | RB(src2r) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, first | RA(in) | RB(src2r) | RC(OTHER_FLAG)));
		FAIL_IF(push_inst(compiler, second | RA(in) | RB(EQUAL_FLAG) | RC(dst)));
		return push_inst(compiler, BIS | RA(dst) | RB(OTHER_FLAG) | RC(dst));
	}

	default:
		SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;
	}

	/* Logical operations. Route SLJIT_SET_Z through EQUAL_FLAG and only write
	   the destination when it is actually used, so a live value the caller
	   keeps in dst (e.g. SLJIT_TMP_DEST_REG, which maps to TMP_REG2) survives
	   a flags-only (UNUSED_DEST) operation. */
	if (flags & SRC2_IMM) {
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, op_imm | RA(src1) | ALPHA_LIT(src2) | RC(EQUAL_FLAG)));
		if (!(flags & UNUSED_DEST))
			FAIL_IF(push_inst(compiler, op_imm | RA(src1) | ALPHA_LIT(src2) | RC(dst)));
	} else {
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, op_reg | RA(src1) | RB(src2r) | RC(EQUAL_FLAG)));
		if (!(flags & UNUSED_DEST))
			FAIL_IF(push_inst(compiler, op_reg | RA(src1) | RB(src2r) | RC(dst)));
	}
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r = TMP_REG2;
	sljit_s32 src1_r;
	sljit_sw src2_r = 0;
	sljit_s32 src2_tmp_reg = (GET_OPCODE(op) >= SLJIT_OP2_BASE && FAST_IS_REG(src1)) ? TMP_REG1 : TMP_REG2;

	if (dst == 0) {
		SLJIT_ASSERT(HAS_FLAGS(op));
		flags |= UNUSED_DEST;
		dst = TMP_REG2;
	} else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (flags & MOVE_OP)
			src2_tmp_reg = dst_r;
	} else if ((dst & SLJIT_MEM) && !getput_arg_fast(compiler, flags | ARG_TEST, TMP_REG1, dst, dstw))
		flags |= SLOW_DEST;

	if (flags & IMM_OP) {
		if (src2 == SLJIT_IMM && src2w >= 0 && src2w <= 0xff) {
			flags |= SRC2_IMM;
			src2_r = src2w;
		} else if ((flags & CUMULATIVE_OP) && src1 == SLJIT_IMM && src1w >= 0 && src1w <= 0xff) {
			flags |= SRC2_IMM;
			src2_r = src1w;

			src1 = src2;
			src1w = src2w;
			src2 = SLJIT_IMM;
		}
	}

	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	} else if (src1 == SLJIT_IMM) {
		if (src1w) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, src1w, TMP_REG3));
			src1_r = TMP_REG1;
		} else
			src1_r = TMP_ZERO;
	} else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC1;
		src1_r = TMP_REG1;
	}

	if (FAST_IS_REG(src2)) {
		src2_r = src2;
		flags |= REG2_SOURCE;
		if ((flags & (REG_DEST | MOVE_OP)) == MOVE_OP)
			dst_r = (sljit_s32)src2_r;
	} else if (src2 == SLJIT_IMM) {
		if (!(flags & SRC2_IMM)) {
			if (src2w) {
				FAIL_IF(load_immediate(compiler, src2_tmp_reg, src2w, TMP_REG3));
				src2_r = src2_tmp_reg;
			} else {
				src2_r = TMP_ZERO;
				if (flags & MOVE_OP) {
					if (dst & SLJIT_MEM)
						dst_r = 0;
					else
						op = SLJIT_MOV;
				}
			}
		}
	} else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, src2_tmp_reg, src2, src2w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC2;
		src2_r = src2_tmp_reg;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		SLJIT_ASSERT(src2_r == TMP_REG2);
		if ((flags & SLOW_DEST) && !can_cache(src2, src2w, src1, src1w) && can_cache(src2, src2w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA | MEM_USE_TMP2, TMP_REG2, src2, src2w));
		} else {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2, src2, src2w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w));
		}
	} else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA | ((src1_r == TMP_REG1) ? MEM_USE_TMP2 : 0), src2_tmp_reg, src2, src2w));

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (dst & SLJIT_MEM) {
		if (!(flags & SLOW_DEST)) {
			getput_arg_fast(compiler, flags, dst_r, dst, dstw);
			return compiler->error;
		}
		return getput_arg(compiler, flags, dst_r, dst, dstw);
	}

	return SLJIT_SUCCESS;
}

/* Alpha has no integer divide instruction. Synthesize DIV/DIVMOD with a
   restoring binary long division. Only the backend temporaries are touched so
   the caller's scratch/saved registers (other than R0/R1) are preserved.
   Quotient goes to R0; DIVMOD also writes the remainder to R1, DIV leaves R1
   holding the divisor. */
static sljit_s32 emit_div_mod(struct sljit_compiler *compiler, sljit_s32 op)
{
	sljit_s32 opcode = GET_OPCODE(op);
	sljit_s32 is_32 = op & SLJIT_32;
	sljit_s32 is_signed = (opcode == SLJIT_DIVMOD_SW || opcode == SLJIT_DIV_SW);
	sljit_s32 is_mod = (opcode == SLJIT_DIVMOD_UW || opcode == SLJIT_DIVMOD_SW);
	sljit_s32 a = SLJIT_R0, b = SLJIT_R1;

	/* Canonicalize 32 bit inputs to a full-width value. */
	if (is_32) {
		if (is_signed) {
			FAIL_IF(push_inst(compiler, ADDL | RA(a) | RB(TMP_ZERO) | RC(a)));
			FAIL_IF(push_inst(compiler, ADDL | RA(b) | RB(TMP_ZERO) | RC(b)));
		} else {
			FAIL_IF(push_inst(compiler, ZAPNOT | RA(a) | ALPHA_LIT(0x0f) | RC(a)));
			FAIL_IF(push_inst(compiler, ZAPNOT | RA(b) | ALPHA_LIT(0x0f) | RC(b)));
		}
	}

	if (is_signed) {
		/* TMP_REG1 = |a|, TMP_REG2 = |b|; keep a_sign in R0, q_sign in R1. */
		FAIL_IF(push_inst(compiler, SRA | RA(a) | ALPHA_LIT(63) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, XOR | RA(a) | RB(EQUAL_FLAG) | RC(TMP_REG1)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_REG1) | RB(EQUAL_FLAG) | RC(TMP_REG1)));
		FAIL_IF(push_inst(compiler, SRA | RA(b) | ALPHA_LIT(63) | RC(OTHER_FLAG)));
		FAIL_IF(push_inst(compiler, XOR | RA(b) | RB(OTHER_FLAG) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_REG2) | RB(OTHER_FLAG) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, XOR | RA(EQUAL_FLAG) | RB(OTHER_FLAG) | RC(b)));
		FAIL_IF(emit_mov(compiler, a, EQUAL_FLAG));
	} else {
		FAIL_IF(emit_mov(compiler, TMP_REG1, a));
		FAIL_IF(emit_mov(compiler, TMP_REG2, b));
	}

	/* Unsigned long division of TMP_REG1 by TMP_REG2. TMP_REG1 accumulates the
	   quotient as its bits shift out, TMP_REG3 holds the running remainder. */
	FAIL_IF(push_inst(compiler, BIS | RA(TMP_ZERO) | RB(TMP_ZERO) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, LDA | RA(OTHER_FLAG) | RB(TMP_ZERO) | DISP16(64)));
	/* 11 instruction loop body; the trailing BNE branches back by -11. */
	FAIL_IF(push_inst(compiler, SRL | RA(TMP_REG1) | ALPHA_LIT(63) | RC(EQUAL_FLAG)));
	FAIL_IF(push_inst(compiler, SLL | RA(TMP_REG3) | ALPHA_LIT(1) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, BIS | RA(TMP_REG3) | RB(EQUAL_FLAG) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, SLL | RA(TMP_REG1) | ALPHA_LIT(1) | RC(TMP_REG1)));
	FAIL_IF(push_inst(compiler, CMPULE | RA(TMP_REG2) | RB(TMP_REG3) | RC(EQUAL_FLAG)));
	FAIL_IF(push_inst(compiler, BIS | RA(TMP_REG1) | RB(EQUAL_FLAG) | RC(TMP_REG1)));
	FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_ZERO) | RB(EQUAL_FLAG) | RC(EQUAL_FLAG)));
	FAIL_IF(push_inst(compiler, AND | RA(TMP_REG2) | RB(EQUAL_FLAG) | RC(EQUAL_FLAG)));
	FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_REG3) | RB(EQUAL_FLAG) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, SUBQ | RA(OTHER_FLAG) | ALPHA_LIT(1) | RC(OTHER_FLAG)));
	FAIL_IF(push_inst(compiler, BNE | RA(OTHER_FLAG) | ((sljit_ins)(-11) & 0x1fffff)));

	if (is_signed) {
		/* Apply the result signs: quotient uses a_sign^b_sign, remainder a_sign. */
		FAIL_IF(push_inst(compiler, XOR | RA(TMP_REG1) | RB(b) | RC(TMP_REG1)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_REG1) | RB(b) | RC(TMP_REG1)));
		FAIL_IF(push_inst(compiler, XOR | RA(TMP_REG3) | RB(a) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_REG3) | RB(a) | RC(TMP_REG3)));

		if (is_mod) {
			FAIL_IF(emit_mov(compiler, a, TMP_REG1));
			return emit_mov(compiler, b, TMP_REG3);
		}
		/* DIV: rebuild the signed divisor (b_sign = q_sign ^ a_sign). */
		FAIL_IF(push_inst(compiler, XOR | RA(b) | RB(a) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, XOR | RA(TMP_REG2) | RB(EQUAL_FLAG) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_REG2) | RB(EQUAL_FLAG) | RC(TMP_REG2)));
		FAIL_IF(emit_mov(compiler, a, TMP_REG1));
		return emit_mov(compiler, b, TMP_REG2);
	}

	if (is_mod) {
		FAIL_IF(emit_mov(compiler, a, TMP_REG1));
		return emit_mov(compiler, b, TMP_REG3);
	}
	/* Unsigned DIV: R1 still holds the (possibly zero-extended) divisor. */
	return emit_mov(compiler, a, TMP_REG1);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	switch (GET_OPCODE(op)) {
	case SLJIT_BREAKPOINT:
		return push_inst(compiler, OP(0x00) | 0x80);
	case SLJIT_NOP:
		return push_inst(compiler, NOP);
	case SLJIT_LMUL_UW:
		FAIL_IF(emit_mov(compiler, TMP_REG1, SLJIT_R1));
		FAIL_IF(push_inst(compiler, UMULH | RA(SLJIT_R0) | RB(TMP_REG1) | RC(SLJIT_R1)));
		return push_inst(compiler, MULQ | RA(SLJIT_R0) | RB(TMP_REG1) | RC(SLJIT_R0));
	case SLJIT_LMUL_SW:
		/* Signed high = UMULH(a,b) - (a<0?b:0) - (b<0?a:0). */
		FAIL_IF(emit_mov(compiler, TMP_REG1, SLJIT_R1));
		FAIL_IF(push_inst(compiler, UMULH | RA(SLJIT_R0) | RB(TMP_REG1) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, SRA | RA(SLJIT_R0) | ALPHA_LIT(63) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, AND | RA(TMP_REG3) | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(EQUAL_FLAG) | RB(TMP_REG3) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, SRA | RA(TMP_REG1) | ALPHA_LIT(63) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, AND | RA(TMP_REG3) | RB(SLJIT_R0) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, SUBQ | RA(EQUAL_FLAG) | RB(TMP_REG3) | RC(EQUAL_FLAG)));
		FAIL_IF(push_inst(compiler, MULQ | RA(SLJIT_R0) | RB(TMP_REG1) | RC(SLJIT_R0)));
		return emit_mov(compiler, SLJIT_R1, EQUAL_FLAG);
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		return emit_div_mod(compiler, op);
	case SLJIT_MEMORY_BARRIER:
		return push_inst(compiler, MB);
	case SLJIT_ENDBR:
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (op & SLJIT_32)
		flags = INT_DATA | SIGNED_DATA;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, WORD_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, srcw);

	case SLJIT_MOV_U32:
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, (src == SLJIT_IMM) ? (sljit_u32)srcw : srcw);

	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, (src == SLJIT_IMM) ? (sljit_s32)srcw : srcw);

	case SLJIT_MOV_U8:
		return emit_op(compiler, op, BYTE_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, (src == SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, op, BYTE_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, (src == SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, op, HALF_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, (src == SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, op, HALF_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_ZERO, 0, src, (src == SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_CLZ:
	case SLJIT_CTZ:
	case SLJIT_REV:
		return emit_op(compiler, op, flags, dst, dstw, TMP_ZERO, 0, src, srcw);

	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		return emit_op(compiler, op, HALF_DATA, dst, dstw, TMP_ZERO, 0, src, srcw);

	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		return emit_op(compiler, op | SLJIT_32, INT_DATA, dst, dstw, TMP_ZERO, 0, src, srcw);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	if (op & SLJIT_32) {
		flags |= INT_DATA | SIGNED_DATA;
		if (src1 == SLJIT_IMM)
			src1w = (sljit_s32)src1w;
		if (src2 == SLJIT_IMM)
			src2w = (sljit_s32)src2w;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
		return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
	case SLJIT_SUBC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		compiler->status_flags_state = 0;
		return emit_op(compiler, op, flags | CUMULATIVE_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		return emit_op(compiler, op, flags | CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
	case SLJIT_ROTL:
	case SLJIT_ROTR:
		if (src2 == SLJIT_IMM) {
			if (op & SLJIT_32)
				src2w &= 0x1f;
			else
				src2w &= 0x3f;
		}
		return emit_op(compiler, op, flags | IMM_OP, dst, dstw, src1, src1w, src2, src2w);
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 1, 0, 0, src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, 0, 0, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2r(compiler, op, dst_reg, src1, src1w, src2, src2w));

	switch (GET_OPCODE(op)) {
	case SLJIT_MULADD:
		SLJIT_SKIP_CHECKS(compiler);
		FAIL_IF(sljit_emit_op2(compiler, SLJIT_MUL | (op & SLJIT_32), TMP_REG2, 0, src1, src1w, src2, src2w));
		FAIL_IF(push_inst(compiler, SELECT_OP(op, ADDQ, ADDL) | RA(dst_reg) | RB(TMP_REG2) | RC(dst_reg)));
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_s32 is_left;
	sljit_s32 inp_flags = ((op & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;
	sljit_sw bit_length = (op & SLJIT_32) ? 32 : 64;

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, dst_reg, src1_reg, src2_reg, src3, src3w));

	is_left = (GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_MSHL);

	if (src1_reg == src2_reg) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, (is_left ? SLJIT_ROTL : SLJIT_ROTR) | (op & SLJIT_32), dst_reg, 0, src1_reg, 0, src3, src3w);
	}

	ADJUST_LOCAL_OFFSET(src3, src3w);

	if (op & SLJIT_32) {
		/* Zero-extend src2 so the shifted-in bits come from the low 32. */
		FAIL_IF(push_inst(compiler, ZAPNOT | RA(src2_reg) | ALPHA_LIT(0x0f) | RC(EQUAL_FLAG)));
		src2_reg = EQUAL_FLAG;
		if (!is_left) {
			/* A right shift needs a zero-extended low word so SRL brings in
			   zeros instead of the sign-extended upper bits of src1. */
			FAIL_IF(push_inst(compiler, ZAPNOT | RA(src1_reg) | ALPHA_LIT(0x0f) | RC(OTHER_FLAG)));
			src1_reg = OTHER_FLAG;
		}
	}

	if (src3 == SLJIT_IMM) {
		src3w &= bit_length - 1;

		if (src3w == 0) {
			FAIL_IF(emit_mov(compiler, dst_reg, src1_reg));
			return emit_sext32(compiler, op, dst_reg);
		}

		if (is_left) {
			FAIL_IF(push_inst(compiler, SLL | RA(src1_reg) | ALPHA_LIT(src3w) | RC(dst_reg)));
			FAIL_IF(push_inst(compiler, SRL | RA(src2_reg) | ALPHA_LIT(bit_length - src3w) | RC(TMP_REG1)));
		} else {
			FAIL_IF(push_inst(compiler, SRL | RA(src1_reg) | ALPHA_LIT(src3w) | RC(dst_reg)));
			FAIL_IF(push_inst(compiler, SLL | RA(src2_reg) | ALPHA_LIT(bit_length - src3w) | RC(TMP_REG1)));
		}
		FAIL_IF(push_inst(compiler, BIS | RA(dst_reg) | RB(TMP_REG1) | RC(dst_reg)));
		return emit_sext32(compiler, op, dst_reg);
	}

	if (src3 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG2, src3, src3w));
		src3 = TMP_REG2;
	} else if (dst_reg == src3) {
		FAIL_IF(emit_mov(compiler, TMP_REG2, src3));
		src3 = TMP_REG2;
	}

	if (op & SLJIT_32) {
		FAIL_IF(push_inst(compiler, AND | RA(src3) | ALPHA_LIT(0x1f) | RC(TMP_REG2)));
		src3 = TMP_REG2;
	}

	if (is_left) {
		FAIL_IF(push_inst(compiler, SLL | RA(src1_reg) | RB(src3) | RC(dst_reg)));
		FAIL_IF(push_inst(compiler, SRL | RA(src2_reg) | ALPHA_LIT(1) | RC(TMP_REG1)));
	} else {
		FAIL_IF(push_inst(compiler, SRL | RA(src1_reg) | RB(src3) | RC(dst_reg)));
		FAIL_IF(push_inst(compiler, SLL | RA(src2_reg) | ALPHA_LIT(1) | RC(TMP_REG1)));
	}

	if (!(op & SLJIT_SHIFT_INTO_NON_ZERO) || (op & SLJIT_32)) {
		/* The 32 bit path always uses the (src2 >> 1) form so the complement
		   count is derived with the 5 bit mask. */
		FAIL_IF(push_inst(compiler, XOR | RA(src3) | ALPHA_LIT(bit_length - 1) | RC(TMP_REG2)));
		if (is_left)
			FAIL_IF(push_inst(compiler, SRL | RA(TMP_REG1) | RB(TMP_REG2) | RC(TMP_REG1)));
		else
			FAIL_IF(push_inst(compiler, SLL | RA(TMP_REG1) | RB(TMP_REG2) | RC(TMP_REG1)));
	} else {
		FAIL_IF(push_inst(compiler, SUBQ | RA(TMP_ZERO) | RB(src3) | RC(TMP_REG2)));
		if (is_left)
			FAIL_IF(push_inst(compiler, SRL | RA(src2_reg) | RB(TMP_REG2) | RC(TMP_REG1)));
		else
			FAIL_IF(push_inst(compiler, SLL | RA(src2_reg) | RB(TMP_REG2) | RC(TMP_REG1)));
	}

	FAIL_IF(push_inst(compiler, BIS | RA(dst_reg) | RB(TMP_REG1) | RC(dst_reg)));
	return emit_sext32(compiler, op, dst_reg);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2_shift(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w,
	sljit_sw shift_arg)
{
	sljit_s32 dst_r, tmp_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2_shift(compiler, op, dst, dstw, src1, src1w, src2, src2w, shift_arg));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	shift_arg &= (sljit_sw)((sizeof(sljit_sw) * 8) - 1);

	if (src2 == SLJIT_IMM) {
		src2w = src2w << shift_arg;
		shift_arg = 0;
	}

	if (shift_arg == 0) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, GET_OPCODE(op), dst, dstw, src1, src1w, src2, src2w);
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA | LOAD_DATA, TMP_REG2, src2, src2w));
		src2 = TMP_REG2;
	}

	if (src1 == SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, src1w, TMP_REG3));
		src1 = TMP_REG1;
	} else if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src1, src1w));
		src1 = TMP_REG1;
	}

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	tmp_r = (src1 == TMP_REG1) ? TMP_REG2 : TMP_REG1;

	FAIL_IF(push_inst(compiler, SLL | RA(src2) | ALPHA_LIT(shift_arg) | RC(tmp_r)));
	FAIL_IF(push_inst(compiler, ADDQ | RA(src1) | RB(tmp_r) | RC(dst_r)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, WORD_DATA, dst_r, dst, dstw);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 base;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		if (FAST_IS_REG(src))
			FAIL_IF(emit_mov(compiler, RETURN_ADDR_REG, src));
		else
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, RETURN_ADDR_REG, src, srcw));

		return push_inst(compiler, JMP | RA(TMP_ZERO) | RB(RETURN_ADDR_REG));
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		/* PREFETCH = LDL R31, disp(Rb). This is a hint which is discarded when
		   the address is invalid, so the address is only computed, never
		   dereferenced by an actual load. */
		base = src & REG_MASK;

		if (src & OFFS_REG_MASK) {
			srcw &= 0x3;

			if (srcw) {
				FAIL_IF(push_inst(compiler, SLL | RA(OFFS_REG(src)) | ALPHA_LIT(srcw) | RC(TMP_REG1)));
				FAIL_IF(push_inst(compiler, ADDQ | RA(TMP_REG1) | RB(base) | RC(TMP_REG1)));
			} else
				FAIL_IF(push_inst(compiler, ADDQ | RA(base) | RB(OFFS_REG(src)) | RC(TMP_REG1)));

			return push_inst(compiler, LDL | RA(TMP_ZERO) | RB(TMP_REG1) | DISP16(0));
		}

		if (srcw >= SIMM_MIN && srcw <= SIMM_MAX)
			return push_inst(compiler, LDL | RA(TMP_ZERO) | RB(base) | DISP16(srcw));

		FAIL_IF(load_immediate(compiler, TMP_REG1, TO_ARGW_HI(srcw), TMP_REG3));

		if (base != 0)
			FAIL_IF(push_inst(compiler, ADDQ | RA(TMP_REG1) | RB(base) | RC(TMP_REG1)));

		return push_inst(compiler, LDL | RA(TMP_ZERO) | RB(TMP_REG1) | DISP16(srcw - TO_ARGW_HI(srcw)));
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_dst(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	switch (op) {
	case SLJIT_FAST_ENTER:
		if (FAST_IS_REG(dst))
			return emit_mov(compiler, dst, RETURN_ADDR_REG);

		SLJIT_ASSERT(RETURN_ADDR_REG == TMP_REG2);
		break;
	case SLJIT_GET_RETURN_ADDRESS:
		dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, dst_r, SLJIT_MEM1(SLJIT_SP), compiler->local_size - SSIZE_OF(sw)));
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(type, reg));

	if (type == SLJIT_GP_REGISTER)
		return reg_map[reg];

	return freg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	SLJIT_UNUSED_ARG(size);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	return push_inst(compiler, *(sljit_ins*)instruction);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_32) >> 7))

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	sljit_s32 is_i32 = (GET_OPCODE(op) == SLJIT_CONV_S32_FROM_F64);
	/* Double bit patterns for the saturation thresholds 2^(w-1). */
	sljit_sw hi_bits = is_i32 ? (sljit_sw)0x41e0000000000000 : (sljit_sw)0x43e0000000000000;
	sljit_sw lo_bits = is_i32 ? (sljit_sw)0xc1e0000000000000 : (sljit_sw)0xc3e0000000000000;
	sljit_sw sat_max = is_i32 ? (sljit_sw)0x7fffffff : (sljit_sw)0x7fffffffffffffff;
	sljit_sw sat_min = is_i32 ? (sljit_sw)(sljit_s32)0x80000000 : (sljit_sw)((sljit_uw)1 << 63);

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src, srcw));
		src = TMP_FREG1;
	}

	/* In-range truncation toward zero (valid while the value fits). */
	FAIL_IF(push_inst(compiler, CVTTQ | FB(src) | FC(TMP_FREG2)));
	FAIL_IF(emit_float_to_int(compiler, 0, dst_r, TMP_FREG2));

	/* Saturate high: src >= 2^(w-1) -> max. CMPTLE leaves 2.0/0.0. */
	FAIL_IF(load_immediate(compiler, TMP_REG1, hi_bits, TMP_REG3));
	FAIL_IF(emit_int_to_float(compiler, 0, TMP_FREG2, TMP_REG1));
	FAIL_IF(push_inst(compiler, CMPTLE | FA(TMP_FREG2) | FB(src) | FC(TMP_FREG2)));
	FAIL_IF(emit_float_to_int(compiler, 0, OTHER_FLAG, TMP_FREG2));
	FAIL_IF(load_immediate(compiler, TMP_REG1, sat_max, TMP_REG3));
	FAIL_IF(push_inst(compiler, CMOVNE | RA(OTHER_FLAG) | RB(TMP_REG1) | RC(dst_r)));

	/* Saturate low: src < -2^(w-1) -> min. */
	FAIL_IF(load_immediate(compiler, TMP_REG1, lo_bits, TMP_REG3));
	FAIL_IF(emit_int_to_float(compiler, 0, TMP_FREG2, TMP_REG1));
	FAIL_IF(push_inst(compiler, CMPTLT | FA(src) | FB(TMP_FREG2) | FC(TMP_FREG2)));
	FAIL_IF(emit_float_to_int(compiler, 0, OTHER_FLAG, TMP_FREG2));
	FAIL_IF(load_immediate(compiler, TMP_REG1, sat_min, TMP_REG3));
	FAIL_IF(push_inst(compiler, CMOVNE | RA(OTHER_FLAG) | RB(TMP_REG1) | RC(dst_r)));

	/* NaN -> 0 (an ordered self-compare is false only for NaN). */
	FAIL_IF(push_inst(compiler, CMPTEQ | FA(src) | FB(src) | FC(TMP_FREG2)));
	FAIL_IF(emit_float_to_int(compiler, 0, OTHER_FLAG, TMP_FREG2));
	FAIL_IF(push_inst(compiler, CMOVEQ | RA(OTHER_FLAG) | RB(TMP_ZERO) | RC(dst_r)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, (is_i32 ? INT_DATA : WORD_DATA), dst_r, dst, dstw);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, ((GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32) ? INT_DATA : WORD_DATA) | LOAD_DATA, TMP_REG1, src, srcw));
		src = TMP_REG1;
	} else if (src == SLJIT_IMM) {
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw, TMP_REG3));
		src = TMP_REG1;
	}

	FAIL_IF(emit_int_to_float(compiler, 0, TMP_FREG1, src));
	FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? CVTQS : CVTQT) | FB(TMP_FREG1) | FC(dst_r)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_uw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;
	sljit_ins cvt = (op & SLJIT_32) ? CVTQS : CVTQT;
	sljit_ins add = (op & SLJIT_32) ? ADDS : ADDT;

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_U32) {
		if (src & SLJIT_MEM) {
			FAIL_IF(emit_op_mem2(compiler, INT_DATA | LOAD_DATA, TMP_REG1, src, srcw));
			src = TMP_REG1;
		} else if (src == SLJIT_IMM) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_sw)(sljit_uw)(sljit_u32)srcw, TMP_REG3));
			src = TMP_REG1;
		}

		/* An unsigned 32 bit value is a small positive 64 bit integer, so the
		   signed conversion is exact once the low word is zero-extended. */
		FAIL_IF(push_inst(compiler, ZAPNOT | RA(src) | ALPHA_LIT(0x0f) | RC(TMP_REG1)));
		FAIL_IF(emit_int_to_float(compiler, 0, TMP_FREG1, TMP_REG1));
		FAIL_IF(push_inst(compiler, cvt | FB(TMP_FREG1) | FC(dst_r)));

		if (dst & SLJIT_MEM)
			return emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw);
		return SLJIT_SUCCESS;
	}

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
		src = TMP_REG1;
	} else if (src == SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw, TMP_REG3));
		src = TMP_REG1;
	}

	/* While bit 63 is clear the signed conversion is exact. */
	FAIL_IF(emit_int_to_float(compiler, 0, TMP_FREG1, src));
	FAIL_IF(push_inst(compiler, cvt | FB(TMP_FREG1) | FC(dst_r)));

	/* Otherwise convert (src >> 1) | (src & 1) (round-to-odd keeps the dropped
	   bit sticky) and double the result. The branch skips the fixup, which is
	   five instructions plus the integer to float transfer. */
	FAIL_IF(push_inst(compiler, BGE | RA(src) | ((sljit_ins)(5 + int_float_move_size()) & 0x1fffff)));
	FAIL_IF(push_inst(compiler, SRL | RA(src) | ALPHA_LIT(1) | RC(TMP_REG2)));
	FAIL_IF(push_inst(compiler, AND | RA(src) | ALPHA_LIT(1) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, BIS | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
	FAIL_IF(emit_int_to_float(compiler, 0, TMP_FREG1, TMP_REG2));
	FAIL_IF(push_inst(compiler, cvt | FB(TMP_FREG1) | FC(dst_r)));
	FAIL_IF(push_inst(compiler, add | FA(dst_r) | FB(dst_r) | FC(dst_r)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_ins inst;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	/* Compute a canonical comparison per negation pair into TMP_FREG1
	   (2.0 = true, 0.0 = false). get_jump_instruction picks the branch
	   polarity (BNE for the positive member, BEQ for its negation). */
	switch (GET_FLAG_TYPE(op)) {
	case SLJIT_F_EQUAL:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
		inst = CMPTEQ | FA(src1) | FB(src2) | FC(TMP_FREG1);
		break;
	case SLJIT_F_LESS:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_LESS:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		inst = CMPTLT | FA(src1) | FB(src2) | FC(TMP_FREG1);
		break;
	case SLJIT_F_GREATER:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		inst = CMPTLT | FA(src2) | FB(src1) | FC(TMP_FREG1);
		break;
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER:
		inst = CMPTLE | FA(src1) | FB(src2) | FC(TMP_FREG1);
		break;
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_LESS:
		inst = CMPTLE | FA(src2) | FB(src1) | FC(TMP_FREG1);
		break;
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		/* ordered && a != b  <=>  (a < b) || (a > b). */
		FAIL_IF(push_inst(compiler, CMPTLT | FA(src1) | FB(src2) | FC(TMP_FREG1)));
		FAIL_IF(push_inst(compiler, CMPTLT | FA(src2) | FB(src1) | FC(TMP_FREG2)));
		FAIL_IF(emit_float_to_int(compiler, 0, OTHER_FLAG, TMP_FREG1));
		FAIL_IF(emit_float_to_int(compiler, 0, TMP_REG1, TMP_FREG2));
		FAIL_IF(push_inst(compiler, BIS | RA(OTHER_FLAG) | RB(TMP_REG1) | RC(OTHER_FLAG)));
		/* Normalize the 2.0 bit pattern to a clean 0/1. */
		return push_inst(compiler, CMPULT | RA(TMP_ZERO) | RB(OTHER_FLAG) | RC(OTHER_FLAG));
	default: /* SLJIT_ORDERED, SLJIT_UNORDERED */
		inst = CMPTUN | FA(src1) | FB(src2) | FC(TMP_FREG1);
		break;
	}

	FAIL_IF(push_inst(compiler, inst));
	/* Move the boolean (2.0 or 0.0) result into OTHER_FLAG and normalize to 0/1. */
	FAIL_IF(emit_float_to_int(compiler, 0, OTHER_FLAG, TMP_FREG1));
	return push_inst(compiler, CMPULT | RA(TMP_ZERO) | RB(OTHER_FLAG) | RC(OTHER_FLAG));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_32;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, dst_r, src, srcw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (!(dst & SLJIT_MEM))
				FAIL_IF(push_inst(compiler, CPYS | FA(src) | FB(src) | FC(dst_r)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, CPYSN | FA(src) | FB(src) | FC(dst_r)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, CPYS | FA(TMP_ZERO) | FB(src) | FC(dst_r)));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		/* Value already held in double format after the LDS load. */
		if (src != dst_r)
			FAIL_IF(push_inst(compiler, CPYS | FA(src) | FB(src) | FC(dst_r)));
		op ^= SLJIT_32;
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r, flags = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG2;

	if (src1 & SLJIT_MEM) {
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w)) {
			FAIL_IF(compiler->error);
			src1 = TMP_FREG1;
		} else
			flags |= SLOW_SRC1;
	}

	if (src2 & SLJIT_MEM) {
		if (getput_arg_fast(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w)) {
			FAIL_IF(compiler->error);
			src2 = TMP_FREG2;
		} else
			flags |= SLOW_SRC2;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		if ((dst & SLJIT_MEM) && !can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w));
		} else {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w));
		}
	} else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w));

	if (flags & SLOW_SRC1)
		src1 = TMP_FREG1;
	if (flags & SLOW_SRC2)
		src2 = TMP_FREG2;

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? ADDS : ADDT) | FA(src1) | FB(src2) | FC(dst_r)));
		break;
	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? SUBS : SUBT) | FA(src1) | FB(src2) | FC(dst_r)));
		break;
	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? MULS : MULT) | FA(src1) | FB(src2) | FC(dst_r)));
		break;
	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? DIVS : DIVT) | FA(src1) | FB(src2) | FC(dst_r)));
		break;
	case SLJIT_COPYSIGN_F64:
		return push_inst(compiler, CPYS | FA(src2) | FB(src1) | FC(dst_r));
	}

	if (dst_r != dst)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG2, dst, dstw));

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

	if (u.imm == 0)
		return emit_int_to_float(compiler, 1, freg, TMP_ZERO);

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm, TMP_REG3));
	return emit_int_to_float(compiler, 1, freg, TMP_REG1);
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
		return emit_int_to_float(compiler, 0, freg, TMP_ZERO);

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm, TMP_REG3));
	return emit_int_to_float(compiler, 0, freg, TMP_REG1);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fcopy(compiler, op, freg, reg));

	if (GET_OPCODE(op) == SLJIT_COPY_TO_F64)
		return emit_int_to_float(compiler, op & SLJIT_32, freg, reg);

	return emit_float_to_int(compiler, op & SLJIT_32, reg, freg);
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler)
{
	struct sljit_label *label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_label(compiler));

	if (compiler->last_label && compiler->last_label->size == compiler->size)
		return compiler->last_label;

	label = (struct sljit_label*)ensure_abuf(compiler, sizeof(struct sljit_label));
	PTR_FAIL_IF(!label);
	set_label(label, compiler);
	return label;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_aligned_label(struct sljit_compiler *compiler,
	sljit_s32 alignment, struct sljit_read_only_buffer *buffers)
{
	sljit_uw mask = 0, i;
	struct sljit_label *label = NULL;
	struct sljit_label *next_label;
	struct sljit_extended_label *ext_label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_aligned_label(compiler, alignment, buffers));

	sljit_reset_read_only_buffers(buffers);

	if (alignment <= SLJIT_LABEL_ALIGN_4) {
		SLJIT_SKIP_CHECKS(compiler);
		label = sljit_emit_label(compiler);
		PTR_FAIL_IF(!label);
	} else {
		mask = ((sljit_uw)1 << alignment) - sizeof(sljit_ins);

		for (i = (mask >> 2); i != 0; i--)
			PTR_FAIL_IF(push_inst(compiler, NOP));
	}

	if (label == NULL) {
		ext_label = (struct sljit_extended_label*)ensure_abuf(compiler, sizeof(struct sljit_extended_label));
		PTR_FAIL_IF(!ext_label);
		set_extended_label(ext_label, compiler, SLJIT_LABEL_ALIGNED, mask);
		label = &ext_label->label;
	}

	if (buffers == NULL)
		return label;

	next_label = label;

	while (1) {
		buffers->u.label = next_label;

		for (i = (buffers->size + 3) >> 2; i > 0; i--)
			PTR_FAIL_IF(push_inst(compiler, NOP));

		buffers = buffers->next;

		if (buffers == NULL)
			break;

		SLJIT_SKIP_CHECKS(compiler);
		next_label = sljit_emit_label(compiler);
		PTR_FAIL_IF(!next_label);
	}

	return label;
}

/* Returns the branch instruction that jumps to the target when the
   condition is TRUE, testing the appropriate flag register. */
static sljit_ins get_jump_instruction(sljit_s32 type)
{
	switch (type) {
	case SLJIT_EQUAL:
		return BEQ | RA(EQUAL_FLAG);
	case SLJIT_NOT_EQUAL:
		return BNE | RA(EQUAL_FLAG);
	case SLJIT_LESS:
	case SLJIT_GREATER:
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER:
	case SLJIT_OVERFLOW:
	case SLJIT_CARRY:
	case SLJIT_ATOMIC_STORED:
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_UNORDERED:
		return BNE | RA(OTHER_FLAG);
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_NOT_CARRY:
	case SLJIT_ATOMIC_NOT_STORED:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED:
		return BEQ | RA(OTHER_FLAG);
	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins inst;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	inst = get_jump_instruction(type);

	if (inst != 0) {
		PTR_FAIL_IF(push_inst(compiler, inst));
		jump->flags |= IS_COND;
	}

	if (type >= SLJIT_FAST_CALL)
		jump->flags |= IS_CALL;

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, 0));
	compiler->size += JUMP_MAX_SIZE - 1;
	return jump;
}

/* Move call arguments from the sljit scratch registers into the Alpha
   argument registers r16-r21 / f16-f21 (positional). */
static sljit_s32 emit_call_args(struct sljit_compiler *compiler, sljit_s32 arg_types)
{
	sljit_s32 word_arg = 0, float_arg = 0, pos = 0;

	arg_types >>= SLJIT_ARG_SHIFT;

	while (arg_types) {
		if (pos < 6) {
			switch (arg_types & SLJIT_ARG_MASK) {
			case SLJIT_ARG_TYPE_F64:
			case SLJIT_ARG_TYPE_F32:
				FAIL_IF(push_inst(compiler, CPYS | FA(SLJIT_FR0 + float_arg) | FB(SLJIT_FR0 + float_arg) | (sljit_ins)(16 + pos)));
				float_arg++;
				break;
			default:
				FAIL_IF(push_inst(compiler, BIS | RA(SLJIT_R0 + word_arg) | RB(TMP_ZERO) | (sljit_ins)(16 + pos)));
				word_arg++;
				break;
			}
		} else
			return SLJIT_ERR_UNSUPPORTED;

		pos++;
		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		PTR_FAIL_IF(emit_call_args(compiler, arg_types));

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));

	if (src != SLJIT_IMM) {
		if (src & SLJIT_MEM) {
			ADJUST_LOCAL_OFFSET(src, srcw);
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG3, src, srcw));
			src = TMP_REG3;
		} else if (src != TMP_REG3)
			FAIL_IF(emit_mov(compiler, TMP_REG3, src));

		if (type >= SLJIT_FAST_CALL)
			return push_inst(compiler, JSR | RA(RETURN_ADDR_REG) | RB(TMP_REG3));
		return push_inst(compiler, JMP | RA(TMP_ZERO) | RB(TMP_REG3));
	}

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	FAIL_IF(!jump);
	set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_CALL : 0));
	jump->u.target = (sljit_uw)srcw;

	jump->addr = compiler->size;
	FAIL_IF(push_inst(compiler, 0));
	compiler->size += JUMP_MAX_SIZE - 1;
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG3, src, srcw));
		src = TMP_REG3;
	}

	if ((type & 0xff) != SLJIT_CALL_REG_ARG)
		FAIL_IF(emit_call_args(compiler, arg_types));

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
			FAIL_IF(emit_mov(compiler, TMP_REG3, src));
			src = TMP_REG3;
		}

		FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP;
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_ijump(compiler, type, src, srcw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_s32 src_r, dst_r, invert;
	sljit_s32 saved_op = op;
	sljit_s32 mem_type = ((op & SLJIT_32) || op == SLJIT_MOV32) ? (INT_DATA | SIGNED_DATA) : WORD_DATA;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	dst_r = (op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2;

	if (op >= SLJIT_ADD && (dst & SLJIT_MEM))
		FAIL_IF(emit_op_mem2(compiler, mem_type | LOAD_DATA, TMP_REG1, dst, dstw));

	/* The boolean condition ends up (0 or non-zero) in src_r. */
	invert = 0;
	if (type == SLJIT_EQUAL) {
		FAIL_IF(push_inst(compiler, CMPEQ | RA(EQUAL_FLAG) | RB(TMP_ZERO) | RC(dst_r)));
		src_r = dst_r;
	} else if (type == SLJIT_NOT_EQUAL) {
		FAIL_IF(push_inst(compiler, CMPULT | RA(TMP_ZERO) | RB(EQUAL_FLAG) | RC(dst_r)));
		src_r = dst_r;
	} else {
		src_r = OTHER_FLAG;
		/* get_jump_instruction returns BEQ OTHER_FLAG for the inverted set. */
		if ((get_jump_instruction(type) & OP(0x3f)) == BEQ)
			invert = 1;
	}

	if (invert) {
		FAIL_IF(push_inst(compiler, CMPEQ | RA(src_r) | RB(TMP_ZERO) | RC(dst_r)));
		src_r = dst_r;
	}

	if (op < SLJIT_ADD) {
		if (dst & SLJIT_MEM)
			return emit_op_mem(compiler, mem_type, src_r, dst, dstw);

		return emit_mov(compiler, dst_r, src_r);
	}

	mem_type |= CUMULATIVE_OP | IMM_OP | ALT_KEEP_CACHE;

	if (dst & SLJIT_MEM)
		return emit_op(compiler, saved_op, mem_type, dst, dstw, TMP_REG1, 0, src_r, 0);
	return emit_op(compiler, saved_op, mem_type, dst, dstw, dst, dstw, src_r, 0);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
	sljit_s32 flag_reg;
	sljit_ins cmov_true, cmov_false;
	sljit_s32 inp_flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG1, src1, src1w));
		src1 = TMP_REG1;
	} else if (src1 == SLJIT_IMM) {
		if (type & SLJIT_32)
			src1w = (sljit_s32)src1w;
		FAIL_IF(load_immediate(compiler, TMP_REG1, src1w, TMP_REG3));
		src1 = TMP_REG1;
	}

	/* pick_cmov sets dst = src1 when the condition is true. When dst == src1
	   invert it and conditionally load src2_reg instead (src1 must survive). */
	if (type & SLJIT_COMPARE_SELECT) {
		/* dst = (src1 <cond> src2_reg) ? src1 : src2_reg. */
		sljit_ins cmp;
		flag_reg = OTHER_FLAG;
		cmov_true = CMOVNE;

		switch (type & ~(SLJIT_32 | SLJIT_COMPARE_SELECT)) {
		case SLJIT_EQUAL:
			cmp = CMPEQ | RA(src1) | RB(src2_reg);
			break;
		case SLJIT_NOT_EQUAL:
			cmp = CMPEQ | RA(src1) | RB(src2_reg);
			cmov_true = CMOVEQ;
			break;
		case SLJIT_LESS:
			cmp = CMPULT | RA(src1) | RB(src2_reg);
			break;
		case SLJIT_GREATER_EQUAL:
			cmp = CMPULT | RA(src1) | RB(src2_reg);
			cmov_true = CMOVEQ;
			break;
		case SLJIT_GREATER:
			cmp = CMPULT | RA(src2_reg) | RB(src1);
			break;
		case SLJIT_LESS_EQUAL:
			cmp = CMPULT | RA(src2_reg) | RB(src1);
			cmov_true = CMOVEQ;
			break;
		case SLJIT_SIG_LESS:
			cmp = CMPLT | RA(src1) | RB(src2_reg);
			break;
		case SLJIT_SIG_GREATER_EQUAL:
			cmp = CMPLT | RA(src1) | RB(src2_reg);
			cmov_true = CMOVEQ;
			break;
		case SLJIT_SIG_GREATER:
			cmp = CMPLT | RA(src2_reg) | RB(src1);
			break;
		default: /* SLJIT_SIG_LESS_EQUAL */
			cmp = CMPLT | RA(src2_reg) | RB(src1);
			cmov_true = CMOVEQ;
			break;
		}

		FAIL_IF(push_inst(compiler, cmp | RC(OTHER_FLAG)));

		if (dst_reg == src1) {
			cmov_false = (cmov_true == CMOVNE) ? CMOVEQ : CMOVNE;
			return push_inst(compiler, cmov_false | RA(flag_reg) | RB(src2_reg) | RC(dst_reg));
		}

		if (dst_reg != src2_reg)
			FAIL_IF(emit_mov(compiler, dst_reg, src2_reg));
		return push_inst(compiler, cmov_true | RA(flag_reg) | RB(src1) | RC(dst_reg));
	}

	/* dst = cond ? src1 : src2_reg (condition from the current flags). */
	type &= ~SLJIT_32;

	if (type == SLJIT_EQUAL) {
		flag_reg = EQUAL_FLAG;
		cmov_true = CMOVEQ;
	} else if (type == SLJIT_NOT_EQUAL) {
		flag_reg = EQUAL_FLAG;
		cmov_true = CMOVNE;
	} else {
		flag_reg = OTHER_FLAG;
		cmov_true = ((get_jump_instruction(type) & OP(0x3f)) == BEQ) ? CMOVEQ : CMOVNE;
	}

	if (dst_reg == src1) {
		cmov_false = (cmov_true == CMOVNE) ? CMOVEQ : CMOVNE;
		return push_inst(compiler, cmov_false | RA(flag_reg) | RB(src2_reg) | RC(dst_reg));
	}

	if (dst_reg != src2_reg)
		FAIL_IF(emit_mov(compiler, dst_reg, src2_reg));
	return push_inst(compiler, cmov_true | RA(flag_reg) | RB(src1) | RC(dst_reg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
	sljit_ins branch;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (!(src1 & SLJIT_MEM) && dst_freg == src1) {
		/* dst holds src1: keep it when the condition is true, else copy src2.
		   Branch (non-inverted) over the src2 copy when the condition holds. */
		branch = get_jump_instruction(type & ~SLJIT_32);
		FAIL_IF(push_inst(compiler, branch | (1 & 0x1fffff)));
		return push_inst(compiler, CPYS | FA(src2_freg) | FB(src2_freg) | FC(dst_freg));
	}

	if (dst_freg != src2_freg)
		FAIL_IF(push_inst(compiler, CPYS | FA(src2_freg) | FB(src2_freg) | FC(dst_freg)));

	if (src1 & SLJIT_MEM) {
		/* Load into TMP_FREG2, avoiding TMP_FREG1: the latter is exposed to the
		   caller as SLJIT_TMP_DEST_FREG and may hold a value that must survive
		   when the condition is false (both dst_freg and src2_freg live). */
		sljit_s32 tmp_freg = (dst_freg == TMP_FREG2) ? TMP_FREG1 : TMP_FREG2;
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(type) | LOAD_DATA, tmp_freg, src1, src1w));
		src1 = tmp_freg;
	}

	/* Branch over the copy when the condition is false (invert). */
	branch = get_jump_instruction(type & ~SLJIT_32) ^ ((sljit_ins)0x04 << 26);
	FAIL_IF(push_inst(compiler, branch | (1 & 0x1fffff)));
	return push_inst(compiler, CPYS | FA(src1) | FB(src1) | FC(dst_freg));
}

#undef FLOAT_DATA

/* Materialize the effective address of (mem, memw) into *mem (a register), *memw=0.
 * Uses TMP_REG1 for any computation; TMP_REG3 as scratch for load_immediate. */
static sljit_s32 update_mem_addr(struct sljit_compiler *compiler,
	sljit_s32 *mem, sljit_sw *memw)
{
	sljit_s32 arg = *mem;
	sljit_sw argw = *memw;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;
		if (SLJIT_UNLIKELY(argw != 0)) {
			FAIL_IF(push_inst(compiler, SLL | RA(OFFS_REG(arg)) | ALPHA_LIT(argw) | RC(TMP_REG1)));
			FAIL_IF(push_inst(compiler, ADDQ | RA(TMP_REG1) | RB(arg & REG_MASK) | RC(TMP_REG1)));
		} else
			FAIL_IF(push_inst(compiler, ADDQ | RA(arg & REG_MASK) | RB(OFFS_REG(arg)) | RC(TMP_REG1)));
		*mem = TMP_REG1;
		*memw = 0;
		return SLJIT_SUCCESS;
	}

	*mem = arg & REG_MASK;
	if (argw == 0)
		return SLJIT_SUCCESS;

	if (argw >= SIMM_MIN && argw <= SIMM_MAX) {
		FAIL_IF(push_inst(compiler, LDA | RA(TMP_REG1) | RB(*mem) | DISP16(argw)));
		*mem = TMP_REG1;
		*memw = 0;
		return SLJIT_SUCCESS;
	}

	FAIL_IF(load_immediate(compiler, TMP_REG1, argw, TMP_REG3));
	if (arg & REG_MASK)
		FAIL_IF(push_inst(compiler, ADDQ | RA(TMP_REG1) | RB(arg & REG_MASK) | RC(TMP_REG1)));
	*mem = TMP_REG1;
	*memw = 0;
	return SLJIT_SUCCESS;
}

/* Synthesize an unaligned multi-byte memory access using LDQ_U + EXT/INS/MSK + STQ_U.
 * Called only when SLJIT_MEM_UNALIGNED is set and the operand is wider than 8 bits.
 * Clobbers TMP_REG1, TMP_REG2, TMP_REG3. */
static sljit_s32 emit_unaligned_mem(struct sljit_compiler *compiler,
	sljit_s32 type, sljit_s32 reg, sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 op = GET_OPCODE(type);
	sljit_ins extl, exth, mskl, mskh, insl, insh;
	sljit_sw hi_off;

	ADJUST_LOCAL_OFFSET(mem, memw);
	FAIL_IF(update_mem_addr(compiler, &mem, &memw));

	/* Ensure the effective address is in TMP_REG1: used as the LDQ_U/STQ_U
	 * base and as the byte-position argument (low 3 bits) to EXT/INS/MSK. */
	if (mem != TMP_REG1)
		FAIL_IF(push_inst(compiler, BIS | RA(mem) | RB(TMP_ZERO) | RC(TMP_REG1)));

	switch (op) {
	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		extl = EXTWL; exth = EXTWH;
		mskl = MSKWL; mskh = MSKWH;
		insl = INSWL; insh = INSWH;
		hi_off = 1;
		break;
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		extl = EXTLL; exth = EXTLH;
		mskl = MSKLL; mskh = MSKLH;
		insl = INSLL; insh = INSLH;
		hi_off = 3;
		break;
	default: /* SLJIT_MOV, SLJIT_MOV_P */
		extl = EXTQL; exth = EXTQH;
		mskl = MSKQL; mskh = MSKQH;
		insl = INSQL; insh = INSQH;
		hi_off = 7;
		break;
	}

	if (type & SLJIT_MEM_STORE) {
		/* Read-modify-write both aligned quadwords that overlap the value.
		 * Save reg to TMP_REG4 (r29/gp) if it aliases TMP_REG2 or TMP_REG3
		 * so the data survives the two LDQ_U/INS sequences. */
		if (reg == TMP_REG2 || reg == TMP_REG3) {
			FAIL_IF(push_inst(compiler, BIS | RA(reg) | RB(TMP_ZERO) | RC(TMP_REG4)));
			reg = TMP_REG4;
		}

		/* Upper aligned quadword. */
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(hi_off)));
		FAIL_IF(push_inst(compiler, mskh  | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, insh  | RA(reg)      | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, STQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(hi_off)));

		/* Lower aligned quadword. */
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
		FAIL_IF(push_inst(compiler, mskl  | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, insl  | RA(reg)      | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
		return push_inst(compiler, STQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0));
	}

	/* Load: extract pieces from two aligned quadwords and combine.
	 * Extract hi portion first, then lo, so the final BIS can write to any reg. */
	FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
	FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG3) | RB(TMP_REG1) | DISP16(hi_off)));
	FAIL_IF(push_inst(compiler, exth  | RA(TMP_REG3) | RB(TMP_REG1) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, extl  | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
	FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(reg)));

	switch (op) {
	case SLJIT_MOV_S16:
		FAIL_IF(push_inst(compiler, SLL | RA(reg) | ALPHA_LIT(48) | RC(reg)));
		return push_inst(compiler, SRA | RA(reg) | ALPHA_LIT(48) | RC(reg));
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		/* ADDL sign-extends bits 31:0 to 64 bits. */
		return push_inst(compiler, ADDL | RA(reg) | RB(TMP_ZERO) | RC(reg));
	default:
		return SLJIT_SUCCESS;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 flags;
	sljit_s32 op;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK)) {
		op = GET_OPCODE(type);
		/* When any alignment hint is set, check whether the guaranteed alignment
		 * meets Alpha's natural-alignment requirement for this operand size:
		 *   16-bit: 2B  (ALIGNED_16 or ALIGNED_32 suffices)
		 *   32-bit: 4B  (only ALIGNED_32 suffices)
		 *   64-bit: 8B  (no hint flag provides this guarantee)
		 * If the guarantee falls short, synthesize with LDQ_U + EXT/INS/MSK. */
		if (type & (SLJIT_MEM_UNALIGNED | SLJIT_MEM_ALIGNED_16 | SLJIT_MEM_ALIGNED_32)) {
			if (op == SLJIT_MOV_U8 || op == SLJIT_MOV_S8)
				return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);
			if ((op == SLJIT_MOV_U16 || op == SLJIT_MOV_S16)
					&& !(type & SLJIT_MEM_UNALIGNED))
				return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);
			if ((op == SLJIT_MOV_U32 || op == SLJIT_MOV_S32 || op == SLJIT_MOV32)
					&& (type & SLJIT_MEM_ALIGNED_32))
				return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);
			return emit_unaligned_mem(compiler, type, reg, mem, memw);
		}
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);
	}

	/* Pair register path.  LDQ/STQ require 8B alignment; no hint flag
	 * provides that guarantee, so synthesize both QW accesses when any
	 * alignment/unaligned flag is present.  After emit_unaligned_mem
	 * returns, TMP_REG1 always holds the effective address. */
	if (type & (SLJIT_MEM_UNALIGNED | SLJIT_MEM_ALIGNED_16 | SLJIT_MEM_ALIGNED_32)) {
		FAIL_IF(emit_unaligned_mem(compiler, type, REG_PAIR_FIRST(reg), mem, memw));
		FAIL_IF(push_inst(compiler, LDA | RA(TMP_REG1) | RB(TMP_REG1) | DISP16(SSIZE_OF(sw))));
		return emit_unaligned_mem(compiler, type, REG_PAIR_SECOND(reg), SLJIT_MEM1(TMP_REG1), 0);
	}

	ADJUST_LOCAL_OFFSET(mem, memw);

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		memw &= 0x3;

		if (SLJIT_UNLIKELY(memw != 0)) {
			FAIL_IF(push_inst(compiler, SLL | RA(OFFS_REG(mem)) | ALPHA_LIT(memw) | RC(TMP_REG1)));
			FAIL_IF(push_inst(compiler, ADDQ | RA(TMP_REG1) | RB(mem & REG_MASK) | RC(TMP_REG1)));
		} else
			FAIL_IF(push_inst(compiler, ADDQ | RA(mem & REG_MASK) | RB(OFFS_REG(mem)) | RC(TMP_REG1)));

		mem = TMP_REG1;
		memw = 0;
	} else if (memw > SIMM_MAX - SSIZE_OF(sw) || memw < SIMM_MIN) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, memw, TMP_REG3));

		if (mem & REG_MASK)
			FAIL_IF(push_inst(compiler, ADDQ | RA(TMP_REG1) | RB(mem & REG_MASK) | RC(TMP_REG1)));

		mem = TMP_REG1;
		memw = 0;
	} else
		mem &= REG_MASK;

	if (!(type & SLJIT_MEM_STORE) && mem == REG_PAIR_FIRST(reg)) {
		FAIL_IF(push_mem_inst(compiler, WORD_DATA | LOAD_DATA, REG_PAIR_SECOND(reg), mem, memw + SSIZE_OF(sw)));
		return push_mem_inst(compiler, WORD_DATA | LOAD_DATA, REG_PAIR_FIRST(reg), mem, memw);
	}

	flags = WORD_DATA | (!(type & SLJIT_MEM_STORE) ? LOAD_DATA : 0);

	FAIL_IF(push_mem_inst(compiler, flags, REG_PAIR_FIRST(reg), mem, memw));
	return push_mem_inst(compiler, flags, REG_PAIR_SECOND(reg), mem, memw + SSIZE_OF(sw));
}

#undef TO_ARGW_HI

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_ins extl, exth, mskl, mskh, insl, insh, ftox, itofx;
	sljit_sw hi_off;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem(compiler, type, freg, mem, memw));

	/* Alpha FP load/store (LDS/LDT/STS/STT) require natural alignment.
	 * Synthesize unaligned FP access using integer LDQ_U+EXT/INS/MSK+STQ_U
	 * combined with ITOFS/ITOFT (FIX extension) for the integer<->FP transfer.
	 * Without FIX, fall back to the generic path and accept the kernel fixup. */
	if (!cpu_feature_list)
		get_cpu_features();
	if (!(cpu_feature_list & ALPHA_HWCAP_FIX))
		return sljit_emit_fmem_unaligned(compiler, type, freg, mem, memw);

	/* F32 = SLJIT_MOV_F64 | SLJIT_32; distinguish by the SLJIT_32 flag, not
	 * the low opcode byte (which is always SLJIT_MOV_F64 for both widths). */
	if (type & SLJIT_32) {
		/* LDS requires 4-byte alignment; ALIGNED_32 guarantees 4B. */
		if ((type & SLJIT_MEM_ALIGNED_32) && !(type & SLJIT_MEM_UNALIGNED))
			return sljit_emit_fmem_unaligned(compiler, type, freg, mem, memw);
		ftox = FTOIS; itofx = ITOFS;
		extl = EXTLL; exth = EXTLH;
		mskl = MSKLL; mskh = MSKLH;
		insl = INSLL; insh = INSLH;
		hi_off = 3;
	} else {
		/* LDT requires 8-byte alignment; no hint flag guarantees that. */
		ftox = FTOIT; itofx = ITOFT;
		extl = EXTQL; exth = EXTQH;
		mskl = MSKQL; mskh = MSKQH;
		insl = INSQL; insh = INSQH;
		hi_off = 7;
	}

	ADJUST_LOCAL_OFFSET(mem, memw);
	FAIL_IF(update_mem_addr(compiler, &mem, &memw));
	if (mem != TMP_REG1)
		FAIL_IF(push_inst(compiler, BIS | RA(mem) | RB(TMP_ZERO) | RC(TMP_REG1)));

	if (type & SLJIT_MEM_STORE) {
		/* Copy FP source to TMP_FREG1 so we can read its integer bits twice
		 * (once per aligned quadword) using FTOIS/FTOIT without needing a
		 * spare GPR to hold the value across the two RMW sequences. */
		FAIL_IF(push_inst(compiler, CPYS | FA(freg) | FB(freg) | FC(TMP_FREG1)));

		/* Upper aligned quadword. */
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(hi_off)));
		FAIL_IF(push_inst(compiler, mskh  | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, ftox  | FA(TMP_FREG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, insh  | RA(TMP_REG3) | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, STQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(hi_off)));

		/* Lower aligned quadword. */
		FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
		FAIL_IF(push_inst(compiler, mskl  | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
		FAIL_IF(push_inst(compiler, ftox  | FA(TMP_FREG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, insl  | RA(TMP_REG3) | RB(TMP_REG1) | RC(TMP_REG3)));
		FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG2)));
		return push_inst(compiler, STQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0));
	}

	/* Load: extract pieces from two aligned quadwords, combine, then
	 * move to the FP register via ITOFS/ITOFT. Write the combined integer
	 * result to TMP_REG1 (addr is no longer needed after the EXT steps). */
	FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG2) | RB(TMP_REG1) | DISP16(0)));
	FAIL_IF(push_inst(compiler, LDQ_U | RA(TMP_REG3) | RB(TMP_REG1) | DISP16(hi_off)));
	FAIL_IF(push_inst(compiler, exth  | RA(TMP_REG3) | RB(TMP_REG1) | RC(TMP_REG3)));
	FAIL_IF(push_inst(compiler, extl  | RA(TMP_REG2) | RB(TMP_REG1) | RC(TMP_REG2)));
	FAIL_IF(push_inst(compiler, BIS   | RA(TMP_REG2) | RB(TMP_REG3) | RC(TMP_REG1)));
	return push_inst(compiler, itofx | RA(TMP_REG1) | FC(freg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_load(compiler, op, dst_reg, mem_reg));

	if (op & SLJIT_ATOMIC_USE_CAS)
		return SLJIT_ERR_UNSUPPORTED;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		ins = LDQ_L;
		break;
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		ins = LDL_L;
		break;
	default:
		return SLJIT_ERR_UNSUPPORTED;
	}

	if (op & SLJIT_ATOMIC_TEST)
		return SLJIT_SUCCESS;

	return push_inst(compiler, ins | RA(dst_reg) | RB(mem_reg) | DISP16(0));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_store(compiler, op, src_reg, mem_reg, temp_reg));

	if (op & SLJIT_ATOMIC_USE_CAS)
		return SLJIT_ERR_UNSUPPORTED;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		ins = STQ_C;
		break;
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		ins = STL_C;
		break;
	default:
		return SLJIT_ERR_UNSUPPORTED;
	}

	if (op & SLJIT_ATOMIC_TEST)
		return SLJIT_SUCCESS;

	/* STx_C overwrites src_reg with the success flag (1 = stored). */
	FAIL_IF(emit_mov(compiler, TMP_REG1, src_reg));
	FAIL_IF(push_inst(compiler, ins | RA(TMP_REG1) | RB(mem_reg) | DISP16(0)));
	/* OTHER_FLAG is set (stored) when TMP_REG1 == 1. */
	return push_inst(compiler, BIS | RA(TMP_ZERO) | RB(TMP_REG1) | RC(OTHER_FLAG));
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_sw init_value)
{
	struct sljit_const *const_;
	sljit_s32 dst_r;
	sljit_s32 mem_flags = WORD_DATA;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, op, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_U8:
		/* Match the shared patchable-const encoding: bit 8 sign-extends the
		   byte so a subsequent wider store observes the sign bits. */
		if (init_value & 0x100)
			init_value = (init_value & 0xff) | ~(sljit_sw)0xff;
		else
			init_value &= 0xff;
		mem_flags = BYTE_DATA;
		break;
	case SLJIT_MOV32:
		/* Plain 32 bit: store only the low word (keep the upper memory bits). */
		mem_flags = INT_DATA;
		/* fallthrough */
	case SLJIT_MOV_S32:
		/* Signed 32 bit: sign extend and store the full word. */
		init_value = (sljit_s32)init_value;
		break;
	}

	PTR_FAIL_IF(emit_const(compiler, dst_r, init_value));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, mem_flags, TMP_REG2, dst, dstw));

	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_op_addr(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_jump *jump;
	sljit_s32 dst_r, target_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_op_addr(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

	if (op != SLJIT_ADD_ABS_ADDR)
		target_r = dst_r;
	else {
		target_r = TMP_REG1;

		if (dst & SLJIT_MEM)
			PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG2, dst, dstw));
	}

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_mov_addr(jump, compiler, 0);

	PTR_FAIL_IF(push_inst(compiler, (sljit_ins)target_r));
	compiler->size += JUMP_MAX_SIZE - 1;

	if (op == SLJIT_ADD_ABS_ADDR)
		PTR_FAIL_IF(push_inst(compiler, ADDQ | RA(dst_r) | RB(TMP_REG1) | RC(dst_r)));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw));

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins *)addr;
	sljit_sw high;
	sljit_s32 low;

	SLJIT_UNUSED_ARG(executable_offset);

	low = (sljit_s32)new_target;
	high = (sljit_sw)new_target >> 32;
	if (low < 0)
		high++;

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 5, 0);
	inst[0] = (inst[0] & ~(sljit_ins)0xffff) | ALPHA_HI16((sljit_s32)high);
	inst[1] = (inst[1] & ~(sljit_ins)0xffff) | ALPHA_LO16((sljit_s32)high);
	/* inst[2] is the SLL, unchanged. */
	inst[3] = (inst[3] & ~(sljit_ins)0xffff) | ALPHA_HI16(low);
	inst[4] = (inst[4] & ~(sljit_ins)0xffff) | ALPHA_LO16(low);
	SLJIT_UPDATE_WX_FLAGS(inst, inst + 5, 1);

	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 5);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_s32 op, sljit_sw new_constant, sljit_sw executable_offset)
{
	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_U8:
		if (new_constant & 0x100)
			new_constant = (new_constant & 0xff) | ~(sljit_sw)0xff;
		else
			new_constant &= 0xff;
		break;
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		new_constant = (sljit_s32)new_constant;
		break;
	}

	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}
