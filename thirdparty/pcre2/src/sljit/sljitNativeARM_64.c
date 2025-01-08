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

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
	return "ARM-64" SLJIT_CPUINFO;
}

/* Length of an instruction word */
typedef sljit_u32 sljit_ins;

#define TMP_ZERO	(0)

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_LR		(SLJIT_NUMBER_OF_REGISTERS + 4)
#define TMP_FP		(SLJIT_NUMBER_OF_REGISTERS + 5)

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)

/* r18 - platform register, currently not used */
static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 8] = {
	31, 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 8, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 31, 9, 10, 30, 29
};

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3] = {
	0, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 14, 13, 12, 11, 10, 9, 8, 30, 31
};

#define W_OP ((sljit_ins)1 << 31)
#define RD(rd) ((sljit_ins)reg_map[rd])
#define RT(rt) ((sljit_ins)reg_map[rt])
#define RN(rn) ((sljit_ins)reg_map[rn] << 5)
#define RT2(rt2) ((sljit_ins)reg_map[rt2] << 10)
#define RM(rm) ((sljit_ins)reg_map[rm] << 16)
#define VD(vd) ((sljit_ins)freg_map[vd])
#define VT(vt) ((sljit_ins)freg_map[vt])
#define VT2(vt) ((sljit_ins)freg_map[vt] << 10)
#define VN(vn) ((sljit_ins)freg_map[vn] << 5)
#define VM(vm) ((sljit_ins)freg_map[vm] << 16)

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

#define ADC		0x9a000000
#define ADD		0x8b000000
#define ADDE		0x8b200000
#define ADDI		0x91000000
#define AND		0x8a000000
#define ANDI		0x92000000
#define AND_v		0x0e201c00
#define ASRV		0x9ac02800
#define B		0x14000000
#define B_CC		0x54000000
#define BL		0x94000000
#define BLR		0xd63f0000
#define BR		0xd61f0000
#define BRK		0xd4200000
#define CAS		0xc8a07c00
#define CASB		0x08a07c00
#define CASH		0x48a07c00
#define CBZ		0xb4000000
#define CCMPI		0xfa400800
#define CLZ		0xdac01000
#define CSEL		0x9a800000
#define CSINC		0x9a800400
#define DUP_e		0x0e000400
#define DUP_g		0x0e000c00
#define EOR		0xca000000
#define EOR_v		0x2e201c00
#define EORI		0xd2000000
#define EXTR		0x93c00000
#define FABS		0x1e60c000
#define FADD		0x1e602800
#define FCMP		0x1e602000
#define FCSEL		0x1e600c00
#define FCVT		0x1e224000
#define FCVTL		0x0e217800
#define FCVTZS		0x9e780000
#define FDIV		0x1e601800
#define FMOV		0x1e604000
#define FMOV_R		0x9e660000
#define FMOV_I		0x1e601000
#define FMUL		0x1e600800
#define FNEG		0x1e614000
#define FSUB		0x1e603800
#define INS		0x4e001c00
#define INS_e		0x6e000400
#define LD1		0x0c407000
#define LD1_s		0x0d400000
#define LD1R		0x0d40c000
#define LDRI		0xf9400000
#define LDRI_F64	0xfd400000
#define LDRI_POST	0xf8400400
#define LDP		0xa9400000
#define LDP_F64		0x6d400000
#define LDP_POST	0xa8c00000
#define LDR_PRE		0xf8400c00
#define LDXR		0xc85f7c00
#define LDXRB		0x085f7c00
#define LDXRH		0x485f7c00
#define LSLV		0x9ac02000
#define LSRV		0x9ac02400
#define MADD		0x9b000000
#define MOVI		0x0f000400
#define MOVK		0xf2800000
#define MOVN		0x92800000
#define MOVZ		0xd2800000
#define NOP		0xd503201f
#define ORN		0xaa200000
#define ORR		0xaa000000
#define ORR_v		0x0ea01c00
#define ORRI		0xb2000000
#define RBIT		0xdac00000
#define RET		0xd65f0000
#define REV		0xdac00c00
#define REV16		0xdac00400
#define RORV		0x9ac02c00
#define SBC		0xda000000
#define SBFM		0x93400000
#define SCVTF		0x9e620000
#define SDIV		0x9ac00c00
#define SMADDL		0x9b200000
#define SMOV		0x0e002c00
#define SMULH		0x9b403c00
#define SSHLL		0x0f00a400
#define ST1		0x0c007000
#define ST1_s		0x0d000000
#define STP		0xa9000000
#define STP_F64		0x6d000000
#define STP_PRE		0xa9800000
#define STRB		0x38206800
#define STRBI		0x39000000
#define STRI		0xf9000000
#define STRI_F64	0xfd000000
#define STR_FI		0x3d000000
#define STR_FR		0x3c206800
#define STUR_FI		0x3c000000
#define STURBI		0x38000000
#define STXR		0xc8007c00
#define STXRB		0x8007c00
#define STXRH		0x48007c00
#define SUB		0xcb000000
#define SUBI		0xd1000000
#define SUBS		0xeb000000
#define TBZ		0x36000000
#define UBFM		0xd3400000
#define UCVTF		0x9e630000
#define UDIV		0x9ac00800
#define UMOV		0x0e003c00
#define UMULH		0x9bc03c00
#define USHLL		0x2f00a400
#define USHR		0x2f000400
#define USRA		0x2f001400
#define XTN		0x0e212800

#define CSET		(CSINC | RM(TMP_ZERO) | RN(TMP_ZERO))
#define LDR		(STRI | (1 << 22))
#define LDRB		(STRBI | (1 << 22))
#define LDRH		(LDRB | (1 << 30))
#define MOV		(ORR | RN(TMP_ZERO))

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 emit_imm64_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_uw imm)
{
	FAIL_IF(push_inst(compiler, MOVZ | RD(dst) | ((sljit_ins)(imm & 0xffff) << 5)));
	FAIL_IF(push_inst(compiler, MOVK | RD(dst) | (((sljit_ins)(imm >> 16) & 0xffff) << 5) | (1 << 21)));
	FAIL_IF(push_inst(compiler, MOVK | RD(dst) | (((sljit_ins)(imm >> 32) & 0xffff) << 5) | (2 << 21)));
	return push_inst(compiler, MOVK | RD(dst) | ((sljit_ins)(imm >> 48) << 5) | (3 << 21));
}

static SLJIT_INLINE sljit_sw detect_jump_type(struct sljit_jump *jump, sljit_ins *code_ptr, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;

	if (jump->flags & SLJIT_REWRITABLE_JUMP) {
		jump->flags |= PATCH_ABS64;
		return 0;
	}

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;
	}

	diff = (sljit_sw)target_addr - (sljit_sw)(code_ptr - 4) - executable_offset;

	if (jump->flags & IS_COND) {
		diff += SSIZE_OF(ins);
		if (diff <= 0xfffff && diff >= -0x100000) {
			code_ptr[-5] ^= (jump->flags & IS_CBZ) ? (0x1 << 24) : 0x1;
			jump->addr -= sizeof(sljit_ins);
			jump->flags |= PATCH_COND;
			return 5;
		}
		diff -= SSIZE_OF(ins);
	}

	if (diff <= 0x7ffffff && diff >= -0x8000000) {
		jump->flags |= PATCH_B;
		return 4;
	}

	if (target_addr < 0x100000000l) {
		if (jump->flags & IS_COND)
			code_ptr[-5] -= (2 << 5);
		code_ptr[-2] = code_ptr[0];
		return 2;
	}

	if (target_addr < 0x1000000000000l) {
		if (jump->flags & IS_COND)
			code_ptr[-5] -= (1 << 5);
		jump->flags |= PATCH_ABS48;
		code_ptr[-1] = code_ptr[0];
		return 1;
	}

	jump->flags |= PATCH_ABS64;
	return 0;
}

static SLJIT_INLINE sljit_sw put_label_get_length(struct sljit_put_label *put_label, sljit_uw max_label)
{
	if (max_label < 0x100000000l) {
		put_label->flags = 0;
		return 2;
	}

	if (max_label < 0x1000000000000l) {
		put_label->flags = 1;
		return 1;
	}

	put_label->flags = 2;
	return 0;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw word_count;
	sljit_uw next_addr;
	sljit_sw executable_offset;
	sljit_sw addr;
	sljit_u32 dst;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;
	struct sljit_put_label *put_label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	code = (sljit_ins*)SLJIT_MALLOC_EXEC(compiler->size * sizeof(sljit_ins), compiler->exec_allocator_data);
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

	code_ptr = code;
	word_count = 0;
	next_addr = 0;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	put_label = compiler->put_labels;

	do {
		buf_ptr = (sljit_ins*)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 2);
		do {
			*code_ptr = *buf_ptr++;
			if (next_addr == word_count) {
				SLJIT_ASSERT(!label || label->size >= word_count);
				SLJIT_ASSERT(!jump || jump->addr >= word_count);
				SLJIT_ASSERT(!const_ || const_->addr >= word_count);
				SLJIT_ASSERT(!put_label || put_label->addr >= word_count);

				/* These structures are ordered by their address. */
				if (label && label->size == word_count) {
					label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					label->size = (sljit_uw)(code_ptr - code);
					label = label->next;
				}
				if (jump && jump->addr == word_count) {
						jump->addr = (sljit_uw)(code_ptr - 4);
						code_ptr -= detect_jump_type(jump, code_ptr, code, executable_offset);
						jump = jump->next;
				}
				if (const_ && const_->addr == word_count) {
					const_->addr = (sljit_uw)code_ptr;
					const_ = const_->next;
				}
				if (put_label && put_label->addr == word_count) {
					SLJIT_ASSERT(put_label->label);
					put_label->addr = (sljit_uw)(code_ptr - 3);
					code_ptr -= put_label_get_length(put_label, (sljit_uw)(SLJIT_ADD_EXEC_OFFSET(code, executable_offset) + put_label->label->size));
					put_label = put_label->next;
				}
				next_addr = compute_next_addr(label, jump, const_, put_label);
			}
			code_ptr++;
			word_count++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
		label->size = (sljit_uw)(code_ptr - code);
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(!put_label);
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);

	jump = compiler->jumps;
	while (jump) {
		do {
			addr = (sljit_sw)((jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target);
			buf_ptr = (sljit_ins *)jump->addr;

			if (jump->flags & PATCH_B) {
				addr = (addr - (sljit_sw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset)) >> 2;
				SLJIT_ASSERT(addr <= 0x1ffffff && addr >= -0x2000000);
				buf_ptr[0] = ((jump->flags & IS_BL) ? BL : B) | (sljit_ins)(addr & 0x3ffffff);
				if (jump->flags & IS_COND)
					buf_ptr[-1] -= (4 << 5);
				break;
			}
			if (jump->flags & PATCH_COND) {
				addr = (addr - (sljit_sw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset)) >> 2;
				SLJIT_ASSERT(addr <= 0x3ffff && addr >= -0x40000);
				buf_ptr[0] = (buf_ptr[0] & ~(sljit_ins)0xffffe0) | (sljit_ins)((addr & 0x7ffff) << 5);
				break;
			}

			SLJIT_ASSERT((jump->flags & (PATCH_ABS48 | PATCH_ABS64)) || (sljit_uw)addr <= (sljit_uw)0xffffffff);
			SLJIT_ASSERT((jump->flags & PATCH_ABS64) || (sljit_uw)addr <= (sljit_uw)0xffffffffffff);

			dst = buf_ptr[0] & 0x1f;
			buf_ptr[0] = MOVZ | dst | (((sljit_ins)addr & 0xffff) << 5);
			buf_ptr[1] = MOVK | dst | (((sljit_ins)(addr >> 16) & 0xffff) << 5) | (1 << 21);
			if (jump->flags & (PATCH_ABS48 | PATCH_ABS64))
				buf_ptr[2] = MOVK | dst | (((sljit_ins)(addr >> 32) & 0xffff) << 5) | (2 << 21);
			if (jump->flags & PATCH_ABS64)
				buf_ptr[3] = MOVK | dst | ((sljit_ins)(addr >> 48) << 5) | (3 << 21);
		} while (0);
		jump = jump->next;
	}

	put_label = compiler->put_labels;
	while (put_label) {
		addr = (sljit_sw)put_label->label->addr;
		buf_ptr = (sljit_ins*)put_label->addr;

		buf_ptr[0] |= ((sljit_ins)addr & 0xffff) << 5;
		buf_ptr[1] |= ((sljit_ins)(addr >> 16) & 0xffff) << 5;

		if (put_label->flags >= 1)
			buf_ptr[2] |= ((sljit_ins)(addr >> 32) & 0xffff) << 5;

		if (put_label->flags >= 2)
			buf_ptr[3] |= (sljit_ins)(addr >> 48) << 5;

		put_label = put_label->next;
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
	case SLJIT_HAS_SIMD:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#else
		/* Available by default. */
		return 1;
#endif

	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_CTZ:
	case SLJIT_HAS_REV:
	case SLJIT_HAS_ROT:
	case SLJIT_HAS_CMOV:
	case SLJIT_HAS_PREFETCH:
	case SLJIT_HAS_COPY_F32:
	case SLJIT_HAS_COPY_F64:
	case SLJIT_HAS_ATOMIC:
		return 1;

	default:
		return 0;
	}
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	switch (type) {
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
		return 2;
	}

	return 0;
}

/* --------------------------------------------------------------------- */
/*  Core code generator functions.                                       */
/* --------------------------------------------------------------------- */

#define COUNT_TRAILING_ZERO(value, result) \
	result = 0; \
	if (!(value & 0xffffffff)) { \
		result += 32; \
		value >>= 32; \
	} \
	if (!(value & 0xffff)) { \
		result += 16; \
		value >>= 16; \
	} \
	if (!(value & 0xff)) { \
		result += 8; \
		value >>= 8; \
	} \
	if (!(value & 0xf)) { \
		result += 4; \
		value >>= 4; \
	} \
	if (!(value & 0x3)) { \
		result += 2; \
		value >>= 2; \
	} \
	if (!(value & 0x1)) { \
		result += 1; \
		value >>= 1; \
	}

#define LOGICAL_IMM_CHECK (sljit_ins)0x100

static sljit_ins logical_imm(sljit_sw imm, sljit_u32 len)
{
	sljit_s32 negated;
	sljit_u32 ones, right;
	sljit_uw mask, uimm;
	sljit_ins ins;

	if (len & LOGICAL_IMM_CHECK) {
		len &= ~LOGICAL_IMM_CHECK;
		if (len == 32 && (imm == 0 || imm == -1))
			return 0;
		if (len == 16 && ((sljit_s32)imm == 0 || (sljit_s32)imm == -1))
			return 0;
	}

	SLJIT_ASSERT((len == 32 && imm != 0 && imm != -1)
		|| (len == 16 && (sljit_s32)imm != 0 && (sljit_s32)imm != -1));

	uimm = (sljit_uw)imm;
	while (1) {
		if (len <= 0) {
			SLJIT_UNREACHABLE();
			return 0;
		}

		mask = ((sljit_uw)1 << len) - 1;
		if ((uimm & mask) != ((uimm >> len) & mask))
			break;
		len >>= 1;
	}

	len <<= 1;

	negated = 0;
	if (uimm & 0x1) {
		negated = 1;
		uimm = ~uimm;
	}

	if (len < 64)
		uimm &= ((sljit_uw)1 << len) - 1;

	/* Unsigned right shift. */
	COUNT_TRAILING_ZERO(uimm, right);

	/* Signed shift. We also know that the highest bit is set. */
	imm = (sljit_sw)~uimm;
	SLJIT_ASSERT(imm < 0);

	COUNT_TRAILING_ZERO(imm, ones);

	if (~imm)
		return 0;

	if (len == 64)
		ins = 1 << 22;
	else
		ins = (0x3f - ((len << 1) - 1)) << 10;

	if (negated)
		return ins | ((len - ones - 1) << 10) | ((len - ones - right) << 16);

	return ins | ((ones - 1) << 10) | ((len - right) << 16);
}

#undef COUNT_TRAILING_ZERO

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw simm)
{
	sljit_uw imm = (sljit_uw)simm;
	sljit_u32 i, zeros, ones, first;
	sljit_ins bitmask;

	/* Handling simple immediates first. */
	if (imm <= 0xffff)
		return push_inst(compiler, MOVZ | RD(dst) | ((sljit_ins)imm << 5));

	if (simm < 0 && simm >= -0x10000)
		return push_inst(compiler, MOVN | RD(dst) | (((sljit_ins)~imm & 0xffff) << 5));

	if (imm <= 0xffffffffl) {
		if ((imm & 0xffff) == 0)
			return push_inst(compiler, MOVZ | RD(dst) | ((sljit_ins)(imm >> 16) << 5) | (1 << 21));
		if ((imm & 0xffff0000l) == 0xffff0000)
			return push_inst(compiler, (MOVN ^ W_OP) | RD(dst) | (((sljit_ins)~imm & 0xffff) << 5));
		if ((imm & 0xffff) == 0xffff)
			return push_inst(compiler, (MOVN ^ W_OP) | RD(dst) | (((sljit_ins)~imm & 0xffff0000u) >> (16 - 5)) | (1 << 21));

		bitmask = logical_imm(simm, 16);
		if (bitmask != 0)
			return push_inst(compiler, (ORRI ^ W_OP) | RD(dst) | RN(TMP_ZERO) | bitmask);

		FAIL_IF(push_inst(compiler, MOVZ | RD(dst) | (((sljit_ins)imm & 0xffff) << 5)));
		return push_inst(compiler, MOVK | RD(dst) | (((sljit_ins)imm & 0xffff0000u) >> (16 - 5)) | (1 << 21));
	}

	bitmask = logical_imm(simm, 32);
	if (bitmask != 0)
		return push_inst(compiler, ORRI | RD(dst) | RN(TMP_ZERO) | bitmask);

	if (simm < 0 && simm >= -0x100000000l) {
		if ((imm & 0xffff) == 0xffff)
			return push_inst(compiler, MOVN | RD(dst) | (((sljit_ins)~imm & 0xffff0000u) >> (16 - 5)) | (1 << 21));

		FAIL_IF(push_inst(compiler, MOVN | RD(dst) | (((sljit_ins)~imm & 0xffff) << 5)));
		return push_inst(compiler, MOVK | RD(dst) | (((sljit_ins)imm & 0xffff0000u) >> (16 - 5)) | (1 << 21));
	}

	/* A large amount of number can be constructed from ORR and MOVx, but computing them is costly. */

	zeros = 0;
	ones = 0;
	for (i = 4; i > 0; i--) {
		if ((simm & 0xffff) == 0)
			zeros++;
		if ((simm & 0xffff) == 0xffff)
			ones++;
		simm >>= 16;
	}

	simm = (sljit_sw)imm;
	first = 1;
	if (ones > zeros) {
		simm = ~simm;
		for (i = 0; i < 4; i++) {
			if (!(simm & 0xffff)) {
				simm >>= 16;
				continue;
			}
			if (first) {
				first = 0;
				FAIL_IF(push_inst(compiler, MOVN | RD(dst) | (((sljit_ins)simm & 0xffff) << 5) | (i << 21)));
			}
			else
				FAIL_IF(push_inst(compiler, MOVK | RD(dst) | (((sljit_ins)~simm & 0xffff) << 5) | (i << 21)));
			simm >>= 16;
		}
		return SLJIT_SUCCESS;
	}

	for (i = 0; i < 4; i++) {
		if (!(simm & 0xffff)) {
			simm >>= 16;
			continue;
		}
		if (first) {
			first = 0;
			FAIL_IF(push_inst(compiler, MOVZ | RD(dst) | (((sljit_ins)simm & 0xffff) << 5) | (i << 21)));
		}
		else
			FAIL_IF(push_inst(compiler, MOVK | RD(dst) | (((sljit_ins)simm & 0xffff) << 5) | (i << 21)));
		simm >>= 16;
	}
	return SLJIT_SUCCESS;
}

#define ARG1_IMM	0x0010000
#define ARG2_IMM	0x0020000
#define INT_OP		0x0040000
#define SET_FLAGS	0x0080000
#define UNUSED_RETURN	0x0100000

#define CHECK_FLAGS(flag_bits) \
	if (flags & SET_FLAGS) { \
		inv_bits |= flag_bits; \
		if (flags & UNUSED_RETURN) \
			dst = TMP_ZERO; \
	}

static sljit_s32 emit_op_imm(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 dst, sljit_sw arg1, sljit_sw arg2)
{
	/* dst must be register, TMP_REG1
	   arg1 must be register, TMP_REG1, imm
	   arg2 must be register, TMP_REG2, imm */
	sljit_ins inv_bits = (flags & INT_OP) ? W_OP : 0;
	sljit_ins inst_bits;
	sljit_s32 op = (flags & 0xffff);
	sljit_s32 reg;
	sljit_sw imm, nimm;

	if (SLJIT_UNLIKELY((flags & (ARG1_IMM | ARG2_IMM)) == (ARG1_IMM | ARG2_IMM))) {
		/* Both are immediates. */
		flags &= ~ARG1_IMM;
		if (arg1 == 0 && op != SLJIT_ADD && op != SLJIT_SUB)
			arg1 = TMP_ZERO;
		else {
			FAIL_IF(load_immediate(compiler, TMP_REG1, arg1));
			arg1 = TMP_REG1;
		}
	}

	if (flags & (ARG1_IMM | ARG2_IMM)) {
		reg = (sljit_s32)((flags & ARG2_IMM) ? arg1 : arg2);
		imm = (flags & ARG2_IMM) ? arg2 : arg1;

		switch (op) {
		case SLJIT_MUL:
		case SLJIT_CLZ:
		case SLJIT_CTZ:
		case SLJIT_REV:
		case SLJIT_REV_U16:
		case SLJIT_REV_S16:
		case SLJIT_REV_U32:
		case SLJIT_REV_S32:
		case SLJIT_ADDC:
		case SLJIT_SUBC:
			/* No form with immediate operand (except imm 0, which
			is represented by a ZERO register). */
			break;
		case SLJIT_MOV:
			SLJIT_ASSERT(!(flags & SET_FLAGS) && (flags & ARG2_IMM) && arg1 == TMP_REG1);
			return load_immediate(compiler, dst, imm);
		case SLJIT_SUB:
			compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
			if (flags & ARG1_IMM)
				break;
			imm = -imm;
			/* Fall through. */
		case SLJIT_ADD:
			if (op != SLJIT_SUB)
				compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;

			if (imm == 0) {
				CHECK_FLAGS(1 << 29);
				return push_inst(compiler, ((op == SLJIT_ADD ? ADDI : SUBI) ^ inv_bits) | RD(dst) | RN(reg));
			}
			if (imm > 0 && imm <= 0xfff) {
				CHECK_FLAGS(1 << 29);
				return push_inst(compiler, (ADDI ^ inv_bits) | RD(dst) | RN(reg) | ((sljit_ins)imm << 10));
			}
			nimm = -imm;
			if (nimm > 0 && nimm <= 0xfff) {
				CHECK_FLAGS(1 << 29);
				return push_inst(compiler, (SUBI ^ inv_bits) | RD(dst) | RN(reg) | ((sljit_ins)nimm << 10));
			}
			if (imm > 0 && imm <= 0xffffff && !(imm & 0xfff)) {
				CHECK_FLAGS(1 << 29);
				return push_inst(compiler, (ADDI ^ inv_bits) | RD(dst) | RN(reg) | (((sljit_ins)imm >> 12) << 10) | (1 << 22));
			}
			if (nimm > 0 && nimm <= 0xffffff && !(nimm & 0xfff)) {
				CHECK_FLAGS(1 << 29);
				return push_inst(compiler, (SUBI ^ inv_bits) | RD(dst) | RN(reg) | (((sljit_ins)nimm >> 12) << 10) | (1 << 22));
			}
			if (imm > 0 && imm <= 0xffffff && !(flags & SET_FLAGS)) {
				FAIL_IF(push_inst(compiler, (ADDI ^ inv_bits) | RD(dst) | RN(reg) | (((sljit_ins)imm >> 12) << 10) | (1 << 22)));
				return push_inst(compiler, (ADDI ^ inv_bits) | RD(dst) | RN(dst) | (((sljit_ins)imm & 0xfff) << 10));
			}
			if (nimm > 0 && nimm <= 0xffffff && !(flags & SET_FLAGS)) {
				FAIL_IF(push_inst(compiler, (SUBI ^ inv_bits) | RD(dst) | RN(reg) | (((sljit_ins)nimm >> 12) << 10) | (1 << 22)));
				return push_inst(compiler, (SUBI ^ inv_bits) | RD(dst) | RN(dst) | (((sljit_ins)nimm & 0xfff) << 10));
			}
			break;
		case SLJIT_AND:
			inst_bits = logical_imm(imm, LOGICAL_IMM_CHECK | ((flags & INT_OP) ? 16 : 32));
			if (!inst_bits)
				break;
			CHECK_FLAGS(3 << 29);
			return push_inst(compiler, (ANDI ^ inv_bits) | RD(dst) | RN(reg) | inst_bits);
		case SLJIT_XOR:
			if (imm == -1) {
				FAIL_IF(push_inst(compiler, (ORN ^ inv_bits) | RD(dst) | RN(TMP_ZERO) | RM(reg)));
				goto set_flags;
			}
			/* fallthrough */
		case SLJIT_OR:
			inst_bits = logical_imm(imm, LOGICAL_IMM_CHECK | ((flags & INT_OP) ? 16 : 32));
			if (!inst_bits)
				break;
			if (op == SLJIT_OR)
				inst_bits |= ORRI;
			else
				inst_bits |= EORI;
			FAIL_IF(push_inst(compiler, (inst_bits ^ inv_bits) | RD(dst) | RN(reg)));
			goto set_flags;
		case SLJIT_SHL:
		case SLJIT_MSHL:
			if (flags & ARG1_IMM)
				break;

			if (flags & INT_OP) {
				imm &= 0x1f;
				inst_bits = (((sljit_ins)-imm & 0x1f) << 16) | ((31 - (sljit_ins)imm) << 10);
			} else {
				imm &= 0x3f;
				inst_bits = ((sljit_ins)1 << 22) | (((sljit_ins)-imm & 0x3f) << 16) | ((63 - (sljit_ins)imm) << 10);
			}

			inv_bits |= inv_bits >> 9;
			FAIL_IF(push_inst(compiler, (UBFM ^ inv_bits) | RD(dst) | RN(arg1) | inst_bits));
			goto set_flags;
		case SLJIT_LSHR:
		case SLJIT_MLSHR:
		case SLJIT_ASHR:
		case SLJIT_MASHR:
			if (flags & ARG1_IMM)
				break;

			inv_bits |= inv_bits >> 9;
			if (op >= SLJIT_ASHR)
				inv_bits |= 1 << 30;

			if (flags & INT_OP) {
				imm &= 0x1f;
				inst_bits = ((sljit_ins)imm << 16) | (31 << 10);
			} else {
				imm &= 0x3f;
				inst_bits = ((sljit_ins)1 << 22) | ((sljit_ins)imm << 16) | (63 << 10);
			}

			FAIL_IF(push_inst(compiler, (UBFM ^ inv_bits) | RD(dst) | RN(arg1) | inst_bits));
			goto set_flags;
		case SLJIT_ROTL:
		case SLJIT_ROTR:
			if (flags & ARG1_IMM)
				break;

			if (op == SLJIT_ROTL)
				imm = -imm;

			imm &= (flags & INT_OP) ? 0x1f : 0x3f;
			return push_inst(compiler, (EXTR ^ (inv_bits | (inv_bits >> 9))) | RD(dst) | RN(arg1) | RM(arg1) | ((sljit_ins)imm << 10));
		default:
			SLJIT_UNREACHABLE();
			break;
		}

		if (flags & ARG2_IMM) {
			if (arg2 == 0)
				arg2 = TMP_ZERO;
			else {
				FAIL_IF(load_immediate(compiler, TMP_REG2, arg2));
				arg2 = TMP_REG2;
			}
		}
		else {
			if (arg1 == 0)
				arg1 = TMP_ZERO;
			else {
				FAIL_IF(load_immediate(compiler, TMP_REG1, arg1));
				arg1 = TMP_REG1;
			}
		}
	}

	/* Both arguments are registers. */
	switch (op) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		if (dst == arg2)
			return SLJIT_SUCCESS;
		return push_inst(compiler, MOV | RD(dst) | RM(arg2));
	case SLJIT_MOV_U8:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		inv_bits |= inv_bits >> 9;
		return push_inst(compiler, (UBFM ^ inv_bits) | RD(dst) | RN(arg2) | (7 << 10));
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		inv_bits |= inv_bits >> 9;
		return push_inst(compiler, (SBFM ^ inv_bits) | RD(dst) | RN(arg2) | (7 << 10));
	case SLJIT_MOV_U16:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		inv_bits |= inv_bits >> 9;
		return push_inst(compiler, (UBFM ^ inv_bits) | RD(dst) | RN(arg2) | (15 << 10));
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		inv_bits |= inv_bits >> 9;
		return push_inst(compiler, (SBFM ^ inv_bits) | RD(dst) | RN(arg2) | (15 << 10));
	case SLJIT_MOV32:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		if (dst == arg2)
			return SLJIT_SUCCESS;
		/* fallthrough */
	case SLJIT_MOV_U32:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		return push_inst(compiler, (MOV ^ W_OP) | RD(dst) | RM(arg2));
	case SLJIT_MOV_S32:
		SLJIT_ASSERT(!(flags & SET_FLAGS) && arg1 == TMP_REG1);
		return push_inst(compiler, SBFM | (1 << 22) | RD(dst) | RN(arg2) | (31 << 10));
	case SLJIT_CLZ:
		SLJIT_ASSERT(arg1 == TMP_REG1);
		return push_inst(compiler, (CLZ ^ inv_bits) | RD(dst) | RN(arg2));
	case SLJIT_CTZ:
		SLJIT_ASSERT(arg1 == TMP_REG1);
		FAIL_IF(push_inst(compiler, (RBIT ^ inv_bits) | RD(dst) | RN(arg2)));
		return push_inst(compiler, (CLZ ^ inv_bits) | RD(dst) | RN(dst));
	case SLJIT_REV:
		SLJIT_ASSERT(arg1 == TMP_REG1);
		inv_bits |= inv_bits >> 21;
		return push_inst(compiler, (REV ^ inv_bits) | RD(dst) | RN(arg2));
	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		SLJIT_ASSERT(arg1 == TMP_REG1 && dst != TMP_REG2);
		FAIL_IF(push_inst(compiler, (REV16 ^ (sljit_ins)0x80000000) | RD(dst) | RN(arg2)));
		if (dst == TMP_REG1 || (arg2 == TMP_REG2 && op == SLJIT_REV_U16))
			return SLJIT_SUCCESS;
		inv_bits |= inv_bits >> 9;
		return push_inst(compiler, ((op == SLJIT_REV_U16 ? UBFM : SBFM) ^ inv_bits) | RD(dst) | RN(dst) | (15 << 10));
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		SLJIT_ASSERT(arg1 == TMP_REG1 && dst != TMP_REG2);
		FAIL_IF(push_inst(compiler, (REV ^ (sljit_ins)0x80000400) | RD(dst) | RN(arg2)));
		if (op == SLJIT_REV_U32 || dst == TMP_REG1)
			return SLJIT_SUCCESS;
		return push_inst(compiler, SBFM | (1 << 22) | RD(dst) | RN(dst) | (31 << 10));
	case SLJIT_ADD:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
		CHECK_FLAGS(1 << 29);
		return push_inst(compiler, (ADD ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2));
	case SLJIT_ADDC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_ADD;
		CHECK_FLAGS(1 << 29);
		return push_inst(compiler, (ADC ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2));
	case SLJIT_SUB:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
		CHECK_FLAGS(1 << 29);
		return push_inst(compiler, (SUB ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2));
	case SLJIT_SUBC:
		compiler->status_flags_state = SLJIT_CURRENT_FLAGS_SUB;
		CHECK_FLAGS(1 << 29);
		return push_inst(compiler, (SBC ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2));
	case SLJIT_MUL:
		compiler->status_flags_state = 0;
		if (!(flags & SET_FLAGS))
			return push_inst(compiler, (MADD ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2) | RT2(TMP_ZERO));
		if (flags & INT_OP) {
			FAIL_IF(push_inst(compiler, SMADDL | RD(dst) | RN(arg1) | RM(arg2) | (31 << 10)));
			FAIL_IF(push_inst(compiler, ADD | RD(TMP_LR) | RN(TMP_ZERO) | RM(dst) | (2 << 22) | (31 << 10)));
			return push_inst(compiler, SUBS | RD(TMP_ZERO) | RN(TMP_LR) | RM(dst) | (2 << 22) | (63 << 10));
		}
		FAIL_IF(push_inst(compiler, SMULH | RD(TMP_LR) | RN(arg1) | RM(arg2)));
		FAIL_IF(push_inst(compiler, MADD | RD(dst) | RN(arg1) | RM(arg2) | RT2(TMP_ZERO)));
		return push_inst(compiler, SUBS | RD(TMP_ZERO) | RN(TMP_LR) | RM(dst) | (2 << 22) | (63 << 10));
	case SLJIT_AND:
		CHECK_FLAGS(3 << 29);
		return push_inst(compiler, (AND ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2));
	case SLJIT_OR:
		FAIL_IF(push_inst(compiler, (ORR ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2)));
		break; /* Set flags. */
	case SLJIT_XOR:
		FAIL_IF(push_inst(compiler, (EOR ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2)));
		break; /* Set flags. */
	case SLJIT_SHL:
	case SLJIT_MSHL:
		FAIL_IF(push_inst(compiler, (LSLV ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2)));
		break; /* Set flags. */
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		FAIL_IF(push_inst(compiler, (LSRV ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2)));
		break; /* Set flags. */
	case SLJIT_ASHR:
	case SLJIT_MASHR:
		FAIL_IF(push_inst(compiler, (ASRV ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2)));
		break; /* Set flags. */
	case SLJIT_ROTL:
		FAIL_IF(push_inst(compiler, (SUB ^ inv_bits) | RD(TMP_REG2) | RN(TMP_ZERO) | RM(arg2)));
		arg2 = TMP_REG2;
		/* fallthrough */
	case SLJIT_ROTR:
		return push_inst(compiler, (RORV ^ inv_bits) | RD(dst) | RN(arg1) | RM(arg2));
	default:
		SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;
	}

set_flags:
	if (flags & SET_FLAGS)
		return push_inst(compiler, (SUBS ^ inv_bits) | RD(TMP_ZERO) | RN(dst) | RM(TMP_ZERO));
	return SLJIT_SUCCESS;
}

#define STORE		0x10
#define SIGNED		0x20

#define BYTE_SIZE	0x0
#define HALF_SIZE	0x1
#define INT_SIZE	0x2
#define WORD_SIZE	0x3

#define MEM_SIZE_SHIFT(flags) ((sljit_ins)(flags) & 0x3)

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg,
	sljit_s32 arg, sljit_sw argw, sljit_s32 tmp_reg)
{
	sljit_u32 shift = MEM_SIZE_SHIFT(flags);
	sljit_u32 type = (shift << 30);

	if (!(flags & STORE))
		type |= (flags & SIGNED) ? 0x00800000 : 0x00400000;

	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (argw == 0 || argw == shift)
			return push_inst(compiler, STRB | type | RT(reg)
				| RN(arg & REG_MASK) | RM(OFFS_REG(arg)) | (argw ? (1 << 12) : 0));

		FAIL_IF(push_inst(compiler, ADD | RD(tmp_reg) | RN(arg & REG_MASK) | RM(OFFS_REG(arg)) | ((sljit_ins)argw << 10)));
		return push_inst(compiler, STRBI | type | RT(reg) | RN(tmp_reg));
	}

	arg &= REG_MASK;

	if (!arg) {
		FAIL_IF(load_immediate(compiler, tmp_reg, argw & ~(0xfff << shift)));

		argw = (argw >> shift) & 0xfff;

		return push_inst(compiler, STRBI | type | RT(reg) | RN(tmp_reg) | ((sljit_ins)argw << 10));
	}

	if ((argw & ((1 << shift) - 1)) == 0) {
		if (argw >= 0) {
			if ((argw >> shift) <= 0xfff)
				return push_inst(compiler, STRBI | type | RT(reg) | RN(arg) | ((sljit_ins)argw << (10 - shift)));

			if (argw <= 0xffffff) {
				FAIL_IF(push_inst(compiler, ADDI | (1 << 22) | RD(tmp_reg) | RN(arg) | (((sljit_ins)argw >> 12) << 10)));

				argw = ((argw & 0xfff) >> shift);
				return push_inst(compiler, STRBI | type | RT(reg) | RN(tmp_reg) | ((sljit_ins)argw << 10));
			}
		} else if (argw < -256 && argw >= -0xfff000) {
			FAIL_IF(push_inst(compiler, SUBI | (1 << 22) | RD(tmp_reg) | RN(arg) | (((sljit_ins)(-argw + 0xfff) >> 12) << 10)));
			argw = ((0x1000 + argw) & 0xfff) >> shift;
			return push_inst(compiler, STRBI | type | RT(reg) | RN(tmp_reg) | ((sljit_ins)argw << 10));
		}
	}

	if (argw <= 0xff && argw >= -0x100)
		return push_inst(compiler, STURBI | type | RT(reg) | RN(arg) | (((sljit_ins)argw & 0x1ff) << 12));

	if (argw >= 0) {
		if (argw <= 0xfff0ff && ((argw + 0x100) & 0xfff) <= 0x1ff) {
			FAIL_IF(push_inst(compiler, ADDI | (1 << 22) | RD(tmp_reg) | RN(arg) | (((sljit_ins)argw >> 12) << 10)));
			return push_inst(compiler, STURBI | type | RT(reg) | RN(tmp_reg) | (((sljit_ins)argw & 0x1ff) << 12));
		}
	} else if (argw >= -0xfff100 && ((-argw + 0xff) & 0xfff) <= 0x1ff) {
		FAIL_IF(push_inst(compiler, SUBI | (1 << 22) | RD(tmp_reg) | RN(arg) | (((sljit_ins)-argw >> 12) << 10)));
		return push_inst(compiler, STURBI | type | RT(reg) | RN(tmp_reg) | (((sljit_ins)argw & 0x1ff) << 12));
	}

	FAIL_IF(load_immediate(compiler, tmp_reg, argw));

	return push_inst(compiler, STRB | type | RT(reg) | RN(arg) | RM(tmp_reg));
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 prev, fprev, saved_regs_size, i, tmp;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);
	sljit_ins offs;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	saved_regs_size = GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 2);
	saved_regs_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);

	local_size = (local_size + saved_regs_size + 0xf) & ~0xf;
	compiler->local_size = local_size;

	if (local_size <= 512) {
		FAIL_IF(push_inst(compiler, STP_PRE | RT(TMP_FP) | RT2(TMP_LR)
			| RN(SLJIT_SP) | (sljit_ins)((-(local_size >> 3) & 0x7f) << 15)));
		offs = (sljit_ins)(local_size - 2 * SSIZE_OF(sw)) << (15 - 3);
		local_size = 0;
	} else {
		saved_regs_size = ((saved_regs_size - 2 * SSIZE_OF(sw)) + 0xf) & ~0xf;

		FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | ((sljit_ins)saved_regs_size << 10)));
		offs = (sljit_ins)(saved_regs_size - 2 * SSIZE_OF(sw)) << (15 - 3);
		local_size -= saved_regs_size;
		SLJIT_ASSERT(local_size > 0);
	}

	prev = -1;

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		if (prev == -1) {
			prev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, STP | RT(prev) | RT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		prev = -1;
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		if (prev == -1) {
			prev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, STP | RT(prev) | RT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		prev = -1;
	}

	fprev = -1;

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		if (fprev == -1) {
			fprev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, STP_F64 | VT(fprev) | VT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		fprev = -1;
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		if (fprev == -1) {
			fprev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, STP_F64 | VT(fprev) | VT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		fprev = -1;
	}

	if (fprev != -1)
		FAIL_IF(push_inst(compiler, STRI_F64 | VT(fprev) | RN(SLJIT_SP) | (offs >> 5) | (1 << 10)));

	if (prev != -1)
		FAIL_IF(push_inst(compiler, STRI | RT(prev) | RN(SLJIT_SP) | (offs >> 5) | ((fprev == -1) ? (1 << 10) : 0)));


#ifdef _WIN32
	if (local_size > 4096)
		FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 10) | (1 << 22)));
#endif /* _WIN32 */

	if (!(options & SLJIT_ENTER_REG_ARG)) {
		arg_types >>= SLJIT_ARG_SHIFT;
		saved_arg_count = 0;
		tmp = SLJIT_R0;

		while (arg_types) {
			if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
				if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
					FAIL_IF(push_inst(compiler, MOV | RD(SLJIT_S0 - saved_arg_count) | RM(tmp)));
					saved_arg_count++;
				}
				tmp++;
			}
			arg_types >>= SLJIT_ARG_SHIFT;
		}
	}

#ifdef _WIN32
	if (local_size > 4096) {
		if (local_size < 4 * 4096) {
			/* No need for a loop. */

			if (local_size >= 2 * 4096) {
				if (local_size >= 3 * 4096) {
					FAIL_IF(push_inst(compiler, LDRI | RT(TMP_ZERO) | RN(SLJIT_SP)));
					FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 10) | (1 << 22)));
				}

				FAIL_IF(push_inst(compiler, LDRI | RT(TMP_ZERO) | RN(SLJIT_SP)));
				FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 10) | (1 << 22)));
			}
		}
		else {
			FAIL_IF(push_inst(compiler, MOVZ | RD(TMP_REG1) | ((((sljit_ins)local_size >> 12) - 1) << 5)));
			FAIL_IF(push_inst(compiler, LDRI | RT(TMP_ZERO) | RN(SLJIT_SP)));
			FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | (1 << 10) | (1 << 22)));
			FAIL_IF(push_inst(compiler, SUBI | (1 << 29) | RD(TMP_REG1) | RN(TMP_REG1) | (1 << 10)));
			FAIL_IF(push_inst(compiler, B_CC | ((((sljit_ins) -3) & 0x7ffff) << 5) | 0x1 /* not-equal */));
		}

		local_size &= 0xfff;

		if (local_size > 0)
			FAIL_IF(push_inst(compiler, LDRI | RT(TMP_ZERO) | RN(SLJIT_SP)));
		else
			FAIL_IF(push_inst(compiler, STP | RT(TMP_FP) | RT2(TMP_LR) | RN(SLJIT_SP)));
	}

	if (local_size > 0) {
		if (local_size <= 512)
			FAIL_IF(push_inst(compiler, STP_PRE | RT(TMP_FP) | RT2(TMP_LR)
				| RN(SLJIT_SP) | (sljit_ins)((-(local_size >> 3) & 0x7f) << 15)));
		else {
			if (local_size >= 4096)
				local_size = (1 << (22 - 10));

			FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | ((sljit_ins)local_size << 10)));
			FAIL_IF(push_inst(compiler, STP | RT(TMP_FP) | RT2(TMP_LR) | RN(SLJIT_SP)));
		}
	}

#else /* !_WIN32 */

	/* The local_size does not include saved registers size. */
	if (local_size != 0) {
		if (local_size > 0xfff) {
			FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | (((sljit_ins)local_size >> 12) << 10) | (1 << 22)));
			local_size &= 0xfff;
		}

		if (local_size > 512 || local_size == 0) {
			if (local_size != 0)
				FAIL_IF(push_inst(compiler, SUBI | RD(SLJIT_SP) | RN(SLJIT_SP) | ((sljit_ins)local_size << 10)));

			FAIL_IF(push_inst(compiler, STP | RT(TMP_FP) | RT2(TMP_LR) | RN(SLJIT_SP)));
		} else
			FAIL_IF(push_inst(compiler, STP_PRE | RT(TMP_FP) | RT2(TMP_LR)
				| RN(SLJIT_SP) | (sljit_ins)((-(local_size >> 3) & 0x7f) << 15)));
	}

#endif /* _WIN32 */

	return push_inst(compiler, ADDI | RD(TMP_FP) | RN(SLJIT_SP) | (0 << 10));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 saved_regs_size;

	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	saved_regs_size = GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 2);
	saved_regs_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);

	compiler->local_size = (local_size + saved_regs_size + 0xf) & ~0xf;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_s32 local_size, prev, fprev, i, tmp;
	sljit_ins offs;

	local_size = compiler->local_size;

	if (!is_return_to) {
		if (local_size > 512 && local_size <= 512 + 496) {
			FAIL_IF(push_inst(compiler, LDP_POST | RT(TMP_FP) | RT2(TMP_LR)
				| RN(SLJIT_SP) | ((sljit_ins)(local_size - 512) << (15 - 3))));
			local_size = 512;
		} else
			FAIL_IF(push_inst(compiler, LDP | RT(TMP_FP) | RT2(TMP_LR) | RN(SLJIT_SP)));
	} else {
		if (local_size > 512 && local_size <= 512 + 248) {
			FAIL_IF(push_inst(compiler, LDRI_POST | RT(TMP_FP) | RN(SLJIT_SP) | ((sljit_ins)(local_size - 512) << 12)));
			local_size = 512;
		} else
			FAIL_IF(push_inst(compiler, LDRI | RT(TMP_FP) | RN(SLJIT_SP) | 0));
	}

	if (local_size > 512) {
		local_size -= 512;
		if (local_size > 0xfff) {
			FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_SP) | RN(SLJIT_SP)
				| (((sljit_ins)local_size >> 12) << 10) | (1 << 22)));
			local_size &= 0xfff;
		}

		FAIL_IF(push_inst(compiler, ADDI | RD(SLJIT_SP) | RN(SLJIT_SP) | ((sljit_ins)local_size << 10)));
		local_size = 512;
	}

	offs = (sljit_ins)(local_size - 2 * SSIZE_OF(sw)) << (15 - 3);
	prev = -1;

	tmp = SLJIT_S0 - compiler->saveds;
	for (i = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options); i > tmp; i--) {
		if (prev == -1) {
			prev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, LDP | RT(prev) | RT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		prev = -1;
	}

	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		if (prev == -1) {
			prev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, LDP | RT(prev) | RT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		prev = -1;
	}

	fprev = -1;

	tmp = SLJIT_FS0 - compiler->fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		if (fprev == -1) {
			fprev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, LDP_F64 | VT(fprev) | VT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		fprev = -1;
	}

	for (i = compiler->fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		if (fprev == -1) {
			fprev = i;
			continue;
		}
		FAIL_IF(push_inst(compiler, LDP_F64 | VT(fprev) | VT2(i) | RN(SLJIT_SP) | offs));
		offs -= (sljit_ins)2 << 15;
		fprev = -1;
	}

	if (fprev != -1)
		FAIL_IF(push_inst(compiler, LDRI_F64 | VT(fprev) | RN(SLJIT_SP) | (offs >> 5) | (1 << 10)));

	if (prev != -1)
		FAIL_IF(push_inst(compiler, LDRI | RT(prev) | RN(SLJIT_SP) | (offs >> 5) | ((fprev == -1) ? (1 << 10) : 0)));

	/* This and the next call/jump instruction can be executed parallelly. */
	return push_inst(compiler, ADDI | RD(SLJIT_SP) | RN(SLJIT_SP) | (sljit_ins)(local_size << 10));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	FAIL_IF(emit_stack_frame_release(compiler, 0));

	return push_inst(compiler, RET | RN(TMP_LR));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_to(compiler, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG1, src, srcw, TMP_REG1));
		src = TMP_REG1;
		srcw = 0;
	} else if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
		FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | RM(src)));
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

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	sljit_ins inv_bits = (op & SLJIT_32) ? W_OP : 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	op = GET_OPCODE(op);
	switch (op) {
	case SLJIT_BREAKPOINT:
		return push_inst(compiler, BRK);
	case SLJIT_NOP:
		return push_inst(compiler, NOP);
	case SLJIT_LMUL_UW:
	case SLJIT_LMUL_SW:
		FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | RM(SLJIT_R0)));
		FAIL_IF(push_inst(compiler, MADD | RD(SLJIT_R0) | RN(SLJIT_R0) | RM(SLJIT_R1) | RT2(TMP_ZERO)));
		return push_inst(compiler, (op == SLJIT_LMUL_UW ? UMULH : SMULH) | RD(SLJIT_R1) | RN(TMP_REG1) | RM(SLJIT_R1));
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
		FAIL_IF(push_inst(compiler, (MOV ^ inv_bits) | RD(TMP_REG1) | RM(SLJIT_R0)));
		FAIL_IF(push_inst(compiler, ((op == SLJIT_DIVMOD_UW ? UDIV : SDIV) ^ inv_bits) | RD(SLJIT_R0) | RN(SLJIT_R0) | RM(SLJIT_R1)));
		FAIL_IF(push_inst(compiler, (MADD ^ inv_bits) | RD(SLJIT_R1) | RN(SLJIT_R0) | RM(SLJIT_R1) | RT2(TMP_ZERO)));
		return push_inst(compiler, (SUB ^ inv_bits) | RD(SLJIT_R1) | RN(TMP_REG1) | RM(SLJIT_R1));
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		return push_inst(compiler, ((op == SLJIT_DIV_UW ? UDIV : SDIV) ^ inv_bits) | RD(SLJIT_R0) | RN(SLJIT_R0) | RM(SLJIT_R1));
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
	sljit_s32 dst_r, flags, mem_flags;
	sljit_s32 op_flags = GET_ALL_FLAGS(op);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	op = GET_OPCODE(op);
	if (op >= SLJIT_MOV && op <= SLJIT_MOV_P) {
		/* Both operands are registers. */
		if (dst_r != TMP_REG1 && FAST_IS_REG(src))
			return emit_op_imm(compiler, op | ((op_flags & SLJIT_32) ? INT_OP : 0), dst_r, TMP_REG1, src);

		switch (op) {
		case SLJIT_MOV:
		case SLJIT_MOV_P:
			mem_flags = WORD_SIZE;
			break;
		case SLJIT_MOV_U8:
			mem_flags = BYTE_SIZE;
			if (src == SLJIT_IMM)
				srcw = (sljit_u8)srcw;
			break;
		case SLJIT_MOV_S8:
			mem_flags = BYTE_SIZE | SIGNED;
			if (src == SLJIT_IMM)
				srcw = (sljit_s8)srcw;
			break;
		case SLJIT_MOV_U16:
			mem_flags = HALF_SIZE;
			if (src == SLJIT_IMM)
				srcw = (sljit_u16)srcw;
			break;
		case SLJIT_MOV_S16:
			mem_flags = HALF_SIZE | SIGNED;
			if (src == SLJIT_IMM)
				srcw = (sljit_s16)srcw;
			break;
		case SLJIT_MOV_U32:
			mem_flags = INT_SIZE;
			if (src == SLJIT_IMM)
				srcw = (sljit_u32)srcw;
			break;
		case SLJIT_MOV_S32:
		case SLJIT_MOV32:
			mem_flags = INT_SIZE | SIGNED;
			if (src == SLJIT_IMM)
				srcw = (sljit_s32)srcw;
			break;
		default:
			SLJIT_UNREACHABLE();
			mem_flags = 0;
			break;
		}

		if (src == SLJIT_IMM)
			FAIL_IF(emit_op_imm(compiler, SLJIT_MOV | ARG2_IMM, dst_r, TMP_REG1, srcw));
		else if (!(src & SLJIT_MEM))
			dst_r = src;
		else
			FAIL_IF(emit_op_mem(compiler, mem_flags, dst_r, src, srcw, TMP_REG1));

		if (dst & SLJIT_MEM)
			return emit_op_mem(compiler, mem_flags | STORE, dst_r, dst, dstw, TMP_REG2);
		return SLJIT_SUCCESS;
	}

	flags = HAS_FLAGS(op_flags) ? SET_FLAGS : 0;

	switch (op) {
	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		mem_flags = HALF_SIZE;
		break;
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		mem_flags = INT_SIZE;
		break;
	default:
		mem_flags = WORD_SIZE;

		if (op_flags & SLJIT_32) {
			flags |= INT_OP;
			mem_flags = INT_SIZE;
		}
		break;
	}

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, mem_flags, TMP_REG2, src, srcw, TMP_REG2));
		src = TMP_REG2;
	}

	emit_op_imm(compiler, flags | op, dst_r, TMP_REG1, src);

	if (SLJIT_UNLIKELY(dst & SLJIT_MEM))
		return emit_op_mem(compiler, mem_flags | STORE, dst_r, dst, dstw, TMP_REG2);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r, flags, mem_flags;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	flags = HAS_FLAGS(op) ? SET_FLAGS : 0;
	mem_flags = WORD_SIZE;

	if (op & SLJIT_32) {
		flags |= INT_OP;
		mem_flags = INT_SIZE;
	}

	if (dst == TMP_REG1)
		flags |= UNUSED_RETURN;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, mem_flags, TMP_REG1, src1, src1w, TMP_REG1));
		src1 = TMP_REG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, mem_flags, TMP_REG2, src2, src2w, TMP_REG2));
		src2 = TMP_REG2;
	}

	if (src1 == SLJIT_IMM)
		flags |= ARG1_IMM;
	else
		src1w = src1;

	if (src2 == SLJIT_IMM)
		flags |= ARG2_IMM;
	else
		src2w = src2;

	emit_op_imm(compiler, flags | GET_OPCODE(op), dst_r, src1w, src2w);

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, mem_flags | STORE, dst_r, dst, dstw, TMP_REG2);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 1, 0, 0, src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op2(compiler, op, TMP_REG1, 0, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_ins inv_bits, imm;
	sljit_s32 is_left;
	sljit_sw mask;

	CHECK_ERROR();
	CHECK(check_sljit_emit_shift_into(compiler, op, dst_reg, src1_reg, src2_reg, src3, src3w));

	is_left = (GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_MSHL);

	if (src1_reg == src2_reg) {
		SLJIT_SKIP_CHECKS(compiler);
		return sljit_emit_op2(compiler, (is_left ? SLJIT_ROTL : SLJIT_ROTR) | (op & SLJIT_32), dst_reg, 0, src1_reg, 0, src3, src3w);
	}

	ADJUST_LOCAL_OFFSET(src3, src3w);

	inv_bits = (op & SLJIT_32) ? W_OP : 0;

	if (src3 == SLJIT_IMM) {
		mask = inv_bits ? 0x1f : 0x3f;
		src3w &= mask;

		if (src3w == 0)
			return SLJIT_SUCCESS;

		if (is_left)
			src3w = (src3w ^ mask) + 1;

		return push_inst(compiler, (EXTR ^ (inv_bits | (inv_bits >> 9))) | RD(dst_reg)
			| RN(is_left ? src1_reg : src2_reg) | RM(is_left ? src2_reg : src1_reg) | ((sljit_ins)src3w << 10));
	}

	if (src3 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inv_bits ? INT_SIZE : WORD_SIZE, TMP_REG2, src3, src3w, TMP_REG2));
		src3 = TMP_REG2;
	} else if (dst_reg == src3) {
		FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG2) | RM(src3)));
		src3 = TMP_REG2;
	}

	FAIL_IF(push_inst(compiler, ((is_left ? LSLV : LSRV) ^ inv_bits) | RD(dst_reg) | RN(src1_reg) | RM(src3)));

	if (!(op & SLJIT_SHIFT_INTO_NON_ZERO)) {
		/* Shift left/right by 1. */
		if (is_left)
			imm = (sljit_ins)(inv_bits ? ((1 << 16) | (31 << 10)) : ((1 << 16) | (63 << 10) | (1 << 22)));
		else
			imm = (sljit_ins)(inv_bits ? ((31 << 16) | (30 << 10)) : ((63 << 16) | (62 << 10) | (1 << 22)));

		FAIL_IF(push_inst(compiler, (UBFM ^ (inv_bits | (inv_bits >> 9))) | RD(TMP_REG1) | RN(src2_reg) | imm));

		/* Set imm to mask. */
		imm = (sljit_ins)(inv_bits ? (4 << 10) : ((5 << 10) | (1 << 22)));
		FAIL_IF(push_inst(compiler, (EORI ^ inv_bits) | RD(TMP_REG2) | RN(src3) | imm));

		src2_reg = TMP_REG1;
	} else
		FAIL_IF(push_inst(compiler, (SUB ^ inv_bits) | RD(TMP_REG2) | RN(TMP_ZERO) | RM(src3)));

	FAIL_IF(push_inst(compiler, ((is_left ? LSRV : LSLV) ^ inv_bits) | RD(TMP_REG1) | RN(src2_reg) | RM(TMP_REG2)));
	return push_inst(compiler, (ORR ^ inv_bits) | RD(dst_reg) | RN(dst_reg) | RM(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, MOV | RD(TMP_LR) | RM(src)));
		else
			FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_LR, src, srcw, TMP_REG1));

		return push_inst(compiler, RET | RN(TMP_LR));
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		SLJIT_ASSERT(reg_map[1] == 0 && reg_map[3] == 2 && reg_map[5] == 4);

		/* The reg_map[op] should provide the appropriate constant. */
		if (op == SLJIT_PREFETCH_L1)
			op = 1;
		else if (op == SLJIT_PREFETCH_L2)
			op = 3;
		else if (op == SLJIT_PREFETCH_L3)
			op = 5;
		else
			op = 2;

		/* Signed word sized load is the prefetch instruction. */
		return emit_op_mem(compiler, WORD_SIZE | SIGNED, op, src, srcw, TMP_REG1);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 dst_r = TMP_LR;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_dst(compiler, op, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	switch (op) {
	case SLJIT_FAST_ENTER:
		if (FAST_IS_REG(dst))
			return push_inst(compiler, MOV | RD(dst) | RM(TMP_LR));
		break;
	case SLJIT_GET_RETURN_ADDRESS:
		dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE, dst_r, SLJIT_MEM1(SLJIT_SP), 0x8, TMP_REG2));
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, WORD_SIZE | STORE, dst_r, dst, dstw, TMP_REG2);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(type, reg));

	if (type == SLJIT_GP_REGISTER)
		return reg_map[reg];

	if (type != SLJIT_FLOAT_REGISTER && type != SLJIT_SIMD_REG_64 && type != SLJIT_SIMD_REG_128)
		return -1;

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

static sljit_s32 emit_fop_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_u32 shift = MEM_SIZE_SHIFT(flags);
	sljit_ins type = (shift << 30);

	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (!(flags & STORE))
		type |= 0x00400000;

	if (arg & OFFS_REG_MASK) {
		argw &= 3;
		if (argw == 0 || argw == shift)
			return push_inst(compiler, STR_FR | type | VT(reg)
				| RN(arg & REG_MASK) | RM(OFFS_REG(arg)) | (argw ? (1 << 12) : 0));

		FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG1) | RN(arg & REG_MASK) | RM(OFFS_REG(arg)) | ((sljit_ins)argw << 10)));
		return push_inst(compiler, STR_FI | type | VT(reg) | RN(TMP_REG1));
	}

	arg &= REG_MASK;

	if (!arg) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, argw & ~(0xfff << shift)));

		argw = (argw >> shift) & 0xfff;

		return push_inst(compiler, STR_FI | type | VT(reg) | RN(TMP_REG1) | ((sljit_ins)argw << 10));
	}

	if (argw >= 0 && (argw & ((1 << shift) - 1)) == 0) {
		if ((argw >> shift) <= 0xfff)
			return push_inst(compiler, STR_FI | type | VT(reg) | RN(arg) | ((sljit_ins)argw << (10 - shift)));

		if (argw <= 0xffffff) {
			FAIL_IF(push_inst(compiler, ADDI | (1 << 22) | RD(TMP_REG1) | RN(arg) | (((sljit_ins)argw >> 12) << 10)));

			argw = ((argw & 0xfff) >> shift);
			return push_inst(compiler, STR_FI | type | VT(reg) | RN(TMP_REG1) | ((sljit_ins)argw << 10));
		}
	}

	if (argw <= 255 && argw >= -256)
		return push_inst(compiler, STUR_FI | type | VT(reg) | RN(arg) | (((sljit_ins)argw & 0x1ff) << 12));

	FAIL_IF(load_immediate(compiler, TMP_REG1, argw));
	return push_inst(compiler, STR_FR | type | VT(reg) | RN(arg) | RM(TMP_REG1));
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	sljit_ins inv_bits = (op & SLJIT_32) ? (1 << 22) : 0;

	if (GET_OPCODE(op) == SLJIT_CONV_S32_FROM_F64)
		inv_bits |= W_OP;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (op & SLJIT_32) ? INT_SIZE : WORD_SIZE, TMP_FREG1, src, srcw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, (FCVTZS ^ inv_bits) | RD(dst_r) | VN(src)));

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, ((GET_OPCODE(op) == SLJIT_CONV_S32_FROM_F64) ? INT_SIZE : WORD_SIZE) | STORE, TMP_REG1, dst, dstw, TMP_REG2);
	return SLJIT_SUCCESS;
}

static sljit_s32 sljit_emit_fop1_conv_f64_from_w(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		emit_op_mem(compiler, (ins & W_OP) ? WORD_SIZE : INT_SIZE, TMP_REG1, src, srcw, TMP_REG1);
		src = TMP_REG1;
	} else if (src == SLJIT_IMM) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
	}

	FAIL_IF(push_inst(compiler, ins | VD(dst_r) | RN(src)));

	if (dst & SLJIT_MEM)
		return emit_fop_mem(compiler, ((ins & (1 << 22)) ? WORD_SIZE : INT_SIZE) | STORE, TMP_FREG1, dst, dstw);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins inv_bits = (op & SLJIT_32) ? (1 << 22) : 0;

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32) {
		inv_bits |= W_OP;

		if (src == SLJIT_IMM)
			srcw = (sljit_s32)srcw;
	}

	return sljit_emit_fop1_conv_f64_from_w(compiler, SCVTF ^ inv_bits, dst, dstw, src, srcw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_uw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins inv_bits = (op & SLJIT_32) ? (1 << 22) : 0;

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_U32) {
		inv_bits |= W_OP;

		if (src == SLJIT_IMM)
			srcw = (sljit_u32)srcw;
	}

	return sljit_emit_fop1_conv_f64_from_w(compiler, UCVTF ^ inv_bits, dst, dstw, src, srcw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 mem_flags = (op & SLJIT_32) ? INT_SIZE : WORD_SIZE;
	sljit_ins inv_bits = (op & SLJIT_32) ? (1 << 22) : 0;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, mem_flags, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, mem_flags, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	FAIL_IF(push_inst(compiler, (FCMP ^ inv_bits) | VN(src1) | VM(src2)));

	if (GET_FLAG_TYPE(op) != SLJIT_UNORDERED_OR_EQUAL)
		return SLJIT_SUCCESS;

	FAIL_IF(push_inst(compiler, CSINC | (0x0 << 12) | RD(TMP_REG1) | RN(TMP_ZERO) | RM(TMP_ZERO)));
	return push_inst(compiler, CCMPI | (0x0 << 16) | (0x7 << 12) | RN(TMP_REG1) | 0x4);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r, mem_flags = (op & SLJIT_32) ? INT_SIZE : WORD_SIZE;
	sljit_ins inv_bits;

	CHECK_ERROR();

	SLJIT_COMPILE_ASSERT((INT_SIZE ^ 0x1) == WORD_SIZE, must_be_one_bit_difference);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	inv_bits = (op & SLJIT_32) ? (1 << 22) : 0;
	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32) ? (mem_flags ^ 0x1) : mem_flags, dst_r, src, srcw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, (FMOV ^ inv_bits) | VD(dst_r) | VN(src)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, (FNEG ^ inv_bits) | VD(dst_r) | VN(src)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, (FABS ^ inv_bits) | VD(dst_r) | VN(src)));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		FAIL_IF(push_inst(compiler, FCVT | (sljit_ins)((op & SLJIT_32) ? (1 << 22) : (1 << 15)) | VD(dst_r) | VN(src)));
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_fop_mem(compiler, mem_flags | STORE, dst_r, dst, dstw);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r, mem_flags = (op & SLJIT_32) ? INT_SIZE : WORD_SIZE;
	sljit_ins inv_bits = (op & SLJIT_32) ? (1 << 22) : 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, mem_flags, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}
	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, mem_flags, TMP_FREG2, src2, src2w));
		src2 = TMP_FREG2;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, (FADD ^ inv_bits) | VD(dst_r) | VN(src1) | VM(src2)));
		break;
	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, (FSUB ^ inv_bits) | VD(dst_r) | VN(src1) | VM(src2)));
		break;
	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, (FMUL ^ inv_bits) | VD(dst_r) | VN(src1) | VM(src2)));
		break;
	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, (FDIV ^ inv_bits) | VD(dst_r) | VN(src1) | VM(src2)));
		break;
	case SLJIT_COPYSIGN_F64:
		FAIL_IF(push_inst(compiler, (FMOV_R ^ ((op & SLJIT_32) ? (W_OP | (1 << 22)) : 0)) | VN(src2) | RD(TMP_REG1)));
		FAIL_IF(push_inst(compiler, (FABS ^ inv_bits) | VD(dst_r) | VN(src1)));
		FAIL_IF(push_inst(compiler, TBZ | ((op & SLJIT_32) ? 0 : ((sljit_ins)1 << 31)) | (0x1f << 19) | (2 << 5) | RT(TMP_REG1)));
		return push_inst(compiler, (FNEG ^ inv_bits) | VD(dst_r) | VN(dst_r));
	}

	if (!(dst & SLJIT_MEM))
		return SLJIT_SUCCESS;
	return emit_fop_mem(compiler, mem_flags | STORE, TMP_FREG1, dst, dstw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset32(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f32 value)
{
	sljit_u32 exp;
	union {
		sljit_u32 imm;
		sljit_f32 value;
	} u;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fset32(compiler, freg, value));

	u.value = value;

	if (u.imm == 0)
		return push_inst(compiler, (FMOV_R ^ (W_OP | (1 << 22))) | RN(TMP_ZERO) | VD(freg) | (1 << 16));

	if ((u.imm << (32 - 19)) == 0) {
		exp = (u.imm >> (23 + 2)) & 0x3f;

		if (exp == 0x20 || exp == 0x1f)
			return push_inst(compiler, (FMOV_I ^ (1 << 22)) | (sljit_ins)((((u.imm >> 24) & 0x80) | ((u.imm >> 19) & 0x7f)) << 13) | VD(freg));
	}

	FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_s32)u.imm));
	return push_inst(compiler, (FMOV_R ^ (W_OP | (1 << 22))) | RN(TMP_REG1) | VD(freg) | (1 << 16));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset64(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f64 value)
{
	sljit_uw exp;
	union {
		sljit_uw imm;
		sljit_f64 value;
	} u;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fset64(compiler, freg, value));

	u.value = value;

	if (u.imm == 0)
		return push_inst(compiler, FMOV_R | RN(TMP_ZERO) | VD(freg) | (sljit_ins)1 << 16);

	if ((u.imm << (64 - 48)) == 0) {
		exp = (u.imm >> (52 + 2)) & 0x1ff;

		if (exp == 0x100 || exp == 0xff)
			return push_inst(compiler, FMOV_I | (sljit_ins)((((u.imm >> 56) & 0x80) | ((u.imm >> 48) & 0x7f)) << 13) | VD(freg));
	}

	FAIL_IF(load_immediate(compiler, TMP_REG1, (sljit_sw)u.imm));
	return push_inst(compiler, FMOV_R | RN(TMP_REG1) | VD(freg) | (1 << 16));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fcopy(compiler, op, freg, reg));

	if (GET_OPCODE(op) == SLJIT_COPY_TO_F64)
		inst = FMOV_R | RN(reg) | VD(freg) | (1 << 16);
	else
		inst = FMOV_R | VN(freg) | RD(reg);

	if (op & SLJIT_32)
		inst ^= W_OP | (1 << 22);

	return push_inst(compiler, inst);
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

static sljit_ins get_cc(struct sljit_compiler *compiler, sljit_s32 type)
{
	switch (type) {
	case SLJIT_EQUAL:
	case SLJIT_ATOMIC_STORED:
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_UNORDERED_OR_EQUAL:
		return 0x1;

	case SLJIT_NOT_EQUAL:
	case SLJIT_ATOMIC_NOT_STORED:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
		return 0x0;

	case SLJIT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_ADD)
			return 0x3;
		/* fallthrough */

	case SLJIT_LESS:
		return 0x2;

	case SLJIT_NOT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_ADD)
			return 0x2;
		/* fallthrough */

	case SLJIT_GREATER_EQUAL:
		return 0x3;

	case SLJIT_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
		return 0x9;

	case SLJIT_LESS_EQUAL:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
		return 0x8;

	case SLJIT_SIG_LESS:
	case SLJIT_UNORDERED_OR_LESS:
		return 0xa;

	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
		return 0xb;

	case SLJIT_SIG_GREATER:
	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
		return 0xd;

	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
		return 0xc;

	case SLJIT_OVERFLOW:
		if (!(compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)))
			return 0x0;
		/* fallthrough */

	case SLJIT_UNORDERED:
		return 0x7;

	case SLJIT_NOT_OVERFLOW:
		if (!(compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)))
			return 0x1;
		/* fallthrough */

	case SLJIT_ORDERED:
		return 0x6;

	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
		return 0x5;

	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		return 0x4;

	default:
		SLJIT_UNREACHABLE();
		return 0xe;
	}
}

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

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	if (type < SLJIT_JUMP) {
		jump->flags |= IS_COND;
		PTR_FAIL_IF(push_inst(compiler, B_CC | (6 << 5) | get_cc(compiler, type)));
	}
	else if (type >= SLJIT_FAST_CALL)
		jump->flags |= IS_BL;

	PTR_FAIL_IF(emit_imm64_const(compiler, TMP_REG1, 0));
	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, ((type >= SLJIT_FAST_CALL) ? BLR : BR) | RN(TMP_REG1)));

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	SLJIT_UNUSED_ARG(arg_types);
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler, 0));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

static SLJIT_INLINE struct sljit_jump* emit_cmp_to0(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump;
	sljit_ins inv_bits = (type & SLJIT_32) ? W_OP : 0;

	SLJIT_ASSERT((type & 0xff) == SLJIT_EQUAL || (type & 0xff) == SLJIT_NOT_EQUAL);
	ADJUST_LOCAL_OFFSET(src, srcw);

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	jump->flags |= IS_CBZ | IS_COND;

	if (src & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem(compiler, inv_bits ? INT_SIZE : WORD_SIZE, TMP_REG1, src, srcw, TMP_REG1));
		src = TMP_REG1;
	}
	else if (src == SLJIT_IMM) {
		PTR_FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
	}

	SLJIT_ASSERT(FAST_IS_REG(src));

	if ((type & 0xff) == SLJIT_EQUAL)
		inv_bits |= 1 << 24;

	PTR_FAIL_IF(push_inst(compiler, (CBZ ^ inv_bits) | (6 << 5) | RT(src)));
	PTR_FAIL_IF(emit_imm64_const(compiler, TMP_REG1, 0));
	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, BR | RN(TMP_REG1)));
	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	struct sljit_jump *jump;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));

	if (src != SLJIT_IMM) {
		if (src & SLJIT_MEM) {
			ADJUST_LOCAL_OFFSET(src, srcw);
			FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG1, src, srcw, TMP_REG1));
			src = TMP_REG1;
		}
		return push_inst(compiler, ((type >= SLJIT_FAST_CALL) ? BLR : BR) | RN(src));
	}

	/* These jumps are converted to jump/call instructions when possible. */
	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	FAIL_IF(!jump);
	set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_BL : 0));
	jump->u.target = (sljit_uw)srcw;

	FAIL_IF(emit_imm64_const(compiler, TMP_REG1, 0));
	jump->addr = compiler->size;
	return push_inst(compiler, ((type >= SLJIT_FAST_CALL) ? BLR : BR) | RN(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(arg_types);
	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG1, src, srcw, TMP_REG1));
		src = TMP_REG1;
	}

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
			FAIL_IF(push_inst(compiler, MOV | RD(TMP_REG1) | RM(src)));
			src = TMP_REG1;
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
	sljit_s32 dst_r, src_r, flags, mem_flags;
	sljit_ins cc;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	cc = get_cc(compiler, type);
	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;

	if (GET_OPCODE(op) < SLJIT_ADD) {
		FAIL_IF(push_inst(compiler, CSINC | (cc << 12) | RD(dst_r) | RN(TMP_ZERO) | RM(TMP_ZERO)));

		if (dst_r == TMP_REG1) {
			mem_flags = (GET_OPCODE(op) == SLJIT_MOV ? WORD_SIZE : INT_SIZE) | STORE;
			return emit_op_mem(compiler, mem_flags, TMP_REG1, dst, dstw, TMP_REG2);
		}

		return SLJIT_SUCCESS;
	}

	flags = HAS_FLAGS(op) ? SET_FLAGS : 0;
	mem_flags = WORD_SIZE;

	if (op & SLJIT_32) {
		flags |= INT_OP;
		mem_flags = INT_SIZE;
	}

	src_r = dst;

	if (dst & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, mem_flags, TMP_REG1, dst, dstw, TMP_REG1));
		src_r = TMP_REG1;
	}

	FAIL_IF(push_inst(compiler, CSINC | (cc << 12) | RD(TMP_REG2) | RN(TMP_ZERO) | RM(TMP_ZERO)));
	emit_op_imm(compiler, flags | GET_OPCODE(op), dst_r, src_r, TMP_REG2);

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, mem_flags | STORE, TMP_REG1, dst, dstw, TMP_REG2);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
	sljit_ins inv_bits = (type & SLJIT_32) ? W_OP : 0;
	sljit_ins cc;

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (src1 == SLJIT_IMM) {
		if (type & SLJIT_32)
			src1w = (sljit_s32)src1w;
		FAIL_IF(load_immediate(compiler, TMP_REG1, src1w));
		src1 = TMP_REG1;
	} else if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, WORD_SIZE, TMP_REG1, src1, src1w, TMP_REG2));
		src1 = TMP_REG1;
	}

	cc = get_cc(compiler, type & ~SLJIT_32);
	return push_inst(compiler, (CSEL ^ inv_bits) | (cc << 12) | RD(dst_reg) | RN(src2_reg) | RM(src1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
	sljit_ins inv_bits = (type & SLJIT_32) ? (1 << 22) : 0;
	sljit_ins cc;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_fop_mem(compiler, (type & SLJIT_32) ? INT_SIZE : WORD_SIZE, TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	cc = get_cc(compiler, type & ~SLJIT_32);
	return push_inst(compiler, (FCSEL ^ inv_bits) | (cc << 12) | VD(dst_freg) | VN(src2_freg) | VM(src1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_u32 inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK))
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);

	ADJUST_LOCAL_OFFSET(mem, memw);

	if (!(mem & REG_MASK)) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, memw & ~0x1f8));

		mem = SLJIT_MEM1(TMP_REG1);
		memw &= 0x1f8;
	} else if (mem & OFFS_REG_MASK) {
		FAIL_IF(push_inst(compiler, ADD | RD(TMP_REG1) | RN(mem & REG_MASK) | RM(OFFS_REG(mem)) | ((sljit_ins)(memw & 0x3) << 10)));

		mem = SLJIT_MEM1(TMP_REG1);
		memw = 0;
	} else if ((memw & 0x7) != 0 || memw > 0x1f8 || memw < -0x200) {
		inst = ADDI;

		if (memw < 0) {
			/* Remains negative for integer min. */
			memw = -memw;
			inst = SUBI;
		} else if ((memw & 0x7) == 0 && memw <= 0x7ff0) {
			if (!(type & SLJIT_MEM_STORE) && (mem & REG_MASK) == REG_PAIR_FIRST(reg)) {
				FAIL_IF(push_inst(compiler, LDRI | RD(REG_PAIR_SECOND(reg)) | RN(mem & REG_MASK) | ((sljit_ins)memw << 7)));
				return push_inst(compiler, LDRI | RD(REG_PAIR_FIRST(reg)) | RN(mem & REG_MASK) | ((sljit_ins)(memw + 0x8) << 7));
			}

			inst = (type & SLJIT_MEM_STORE) ? STRI : LDRI;

			FAIL_IF(push_inst(compiler, inst | RD(REG_PAIR_FIRST(reg)) | RN(mem & REG_MASK) | ((sljit_ins)memw << 7)));
			return push_inst(compiler, inst | RD(REG_PAIR_SECOND(reg)) | RN(mem & REG_MASK) | ((sljit_ins)(memw + 0x8) << 7));
		}

		if ((sljit_uw)memw <= 0xfff) {
			FAIL_IF(push_inst(compiler, inst | RD(TMP_REG1) | RN(mem & REG_MASK) | ((sljit_ins)memw << 10)));
			memw = 0;
		} else if ((sljit_uw)memw <= 0xffffff) {
			FAIL_IF(push_inst(compiler, inst | (1 << 22) | RD(TMP_REG1) | RN(mem & REG_MASK) | (((sljit_ins)memw >> 12) << 10)));

			if ((memw & 0xe07) != 0) {
				FAIL_IF(push_inst(compiler, inst | RD(TMP_REG1) | RN(TMP_REG1) | (((sljit_ins)memw & 0xfff) << 10)));
				memw = 0;
			} else {
				memw &= 0xfff;
			}
		} else {
			FAIL_IF(load_immediate(compiler, TMP_REG1, memw));
			FAIL_IF(push_inst(compiler, (inst == ADDI ? ADD : SUB) | RD(TMP_REG1) | RN(mem & REG_MASK) | RM(TMP_REG1)));
			memw = 0;
		}

		mem = SLJIT_MEM1(TMP_REG1);

		if (inst == SUBI)
			memw = -memw;
	}

	SLJIT_ASSERT((memw & 0x7) == 0 && memw <= 0x1f8 && memw >= -0x200);
	return push_inst(compiler, ((type & SLJIT_MEM_STORE) ? STP : LDP) | RT(REG_PAIR_FIRST(reg)) | RT2(REG_PAIR_SECOND(reg)) | RN(mem & REG_MASK) | (sljit_ins)((memw & 0x3f8) << 12));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_u32 sign = 0, inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem_update(compiler, type, reg, mem, memw));

	if ((mem & OFFS_REG_MASK) || (memw > 255 || memw < -256))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_MEM_SUPP)
		return SLJIT_SUCCESS;

	switch (type & 0xff) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		inst = STURBI | (MEM_SIZE_SHIFT(WORD_SIZE) << 30) | 0x400;
		break;
	case SLJIT_MOV_S8:
		sign = 1;
		/* fallthrough */
	case SLJIT_MOV_U8:
		inst = STURBI | (MEM_SIZE_SHIFT(BYTE_SIZE) << 30) | 0x400;
		break;
	case SLJIT_MOV_S16:
		sign = 1;
		/* fallthrough */
	case SLJIT_MOV_U16:
		inst = STURBI | (MEM_SIZE_SHIFT(HALF_SIZE) << 30) | 0x400;
		break;
	case SLJIT_MOV_S32:
		sign = 1;
		/* fallthrough */
	case SLJIT_MOV_U32:
	case SLJIT_MOV32:
		inst = STURBI | (MEM_SIZE_SHIFT(INT_SIZE) << 30) | 0x400;
		break;
	default:
		SLJIT_UNREACHABLE();
		inst = STURBI | (MEM_SIZE_SHIFT(WORD_SIZE) << 30) | 0x400;
		break;
	}

	if (!(type & SLJIT_MEM_STORE))
		inst |= sign ? 0x00800000 : 0x00400000;

	if (!(type & SLJIT_MEM_POST))
		inst |= 0x800;

	return push_inst(compiler, inst | RT(reg) | RN(mem & REG_MASK) | (sljit_ins)((memw & 0x1ff) << 12));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_u32 inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem_update(compiler, type, freg, mem, memw));

	if ((mem & OFFS_REG_MASK) || (memw > 255 || memw < -256))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_MEM_SUPP)
		return SLJIT_SUCCESS;

	inst = STUR_FI | 0x80000400;

	if (!(type & SLJIT_32))
		inst |= 0x40000000;

	if (!(type & SLJIT_MEM_STORE))
		inst |= 0x00400000;

	if (!(type & SLJIT_MEM_POST))
		inst |= 0x800;

	return push_inst(compiler, inst | VT(freg) | RN(mem & REG_MASK) | (sljit_ins)((memw & 0x1ff) << 12));
}

static sljit_s32 sljit_emit_simd_mem_offset(struct sljit_compiler *compiler, sljit_s32 *mem_ptr, sljit_sw memw)
{
	sljit_ins ins;
	sljit_s32 mem = *mem_ptr;

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		*mem_ptr = TMP_REG1;
		return push_inst(compiler, ADD | RD(TMP_REG1) | RN(mem & REG_MASK) | RM(OFFS_REG(mem)) | ((sljit_ins)(memw & 0x3) << 10));
	}

	if (!(mem & REG_MASK)) {
		*mem_ptr = TMP_REG1;
		return load_immediate(compiler, TMP_REG1, memw);
	}

	mem &= REG_MASK;

	if (memw == 0) {
		*mem_ptr = mem;
		return SLJIT_SUCCESS;
	}

	*mem_ptr = TMP_REG1;

	if (memw < -0xffffff || memw > 0xffffff) {
		FAIL_IF(load_immediate(compiler, TMP_REG1, memw));
		return push_inst(compiler, ADD | RD(TMP_REG1) | RN(TMP_REG1) | RM(mem));
	}

	ins = ADDI;

	if (memw < 0) {
		memw = -memw;
		ins = SUBI;
	}

	if (memw > 0xfff) {
		FAIL_IF(push_inst(compiler, ins | (1 << 22) | RD(TMP_REG1) | RN(mem) | ((sljit_ins)(memw >> 12) << 10)));

		memw &= 0xfff;
		if (memw == 0)
			return SLJIT_SUCCESS;

		mem = TMP_REG1;
	}

	return push_inst(compiler, ins | RD(TMP_REG1) | RN(mem) | ((sljit_ins)memw << 10));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_mov(compiler, type, freg, srcdst, srcdstw));

	ADJUST_LOCAL_OFFSET(srcdst, srcdstw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (!(srcdst & SLJIT_MEM)) {
		if (type & SLJIT_SIMD_STORE)
			ins = VD(srcdst) | VN(freg) | VM(freg);
		else
			ins = VD(freg) | VN(srcdst) | VM(srcdst);

		if (reg_size == 4)
			ins |= (1 << 30);

		return push_inst(compiler, ORR_v | ins);
	}

	FAIL_IF(sljit_emit_simd_mem_offset(compiler, &srcdst, srcdstw));

	if (elem_size > 3)
		elem_size = 3;

	ins = (type & SLJIT_SIMD_STORE) ? ST1 : LD1;

	if (reg_size == 4)
		ins |= (1 << 30);

	return push_inst(compiler, ins | ((sljit_ins)elem_size << 10) | RN(srcdst) | VT(freg));
}

static sljit_ins simd_get_imm(sljit_s32 elem_size, sljit_uw value)
{
	sljit_ins result;

	if (elem_size > 2 && (sljit_u32)value == (value >> 32)) {
		elem_size = 2;
		value = (sljit_u32)value;
	}

	if (elem_size == 2 && (sljit_u16)value == (value >> 16)) {
		elem_size = 1;
		value = (sljit_u16)value;
	}

	if (elem_size == 1 && (sljit_u8)value == (value >> 8)) {
		elem_size = 0;
		value = (sljit_u8)value;
	}

	switch (elem_size) {
	case 0:
		SLJIT_ASSERT(value <= 0xff);
		result = 0xe000;
		break;
	case 1:
		SLJIT_ASSERT(value <= 0xffff);
		result = 0;

		while (1) {
			if (value <= 0xff) {
				result |= 0x8000;
				break;
			}

			if ((value & 0xff) == 0) {
				value >>= 8;
				result |= 0xa000;
				break;
			}

			if (result != 0)
				return ~(sljit_ins)0;

			value ^= (sljit_uw)0xffff;
			result = (1 << 29);
		}
		break;
	case 2:
		SLJIT_ASSERT(value <= 0xffffffff);
		result = 0;

		while (1) {
			if (value <= 0xff) {
				result |= 0x0000;
				break;
			}

			if ((value & ~(sljit_uw)0xff00) == 0) {
				value >>= 8;
				result |= 0x2000;
				break;
			}

			if ((value & ~(sljit_uw)0xff0000) == 0) {
				value >>= 16;
				result |= 0x4000;
				break;
			}

			if ((value & ~(sljit_uw)0xff000000) == 0) {
				value >>= 24;
				result |= 0x6000;
				break;
			}

			if ((value & (sljit_uw)0xff) == 0xff && (value >> 16) == 0) {
				value >>= 8;
				result |= 0xc000;
				break;
			}

			if ((value & (sljit_uw)0xffff) == 0xffff && (value >> 24) == 0) {
				value >>= 16;
				result |= 0xd000;
				break;
			}

			if (result != 0)
				return ~(sljit_ins)0;

			value ^= (sljit_uw)0xffffffff;
			result = (1 << 29);
		}
		break;
	default:
		return ~(sljit_ins)0;
	}

	return (((sljit_ins)value & 0x1f) << 5) | (((sljit_ins)value & 0xe0) << 11) | result;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins, imm;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_replicate(compiler, type, freg, src, srcw));

	ADJUST_LOCAL_OFFSET(src, srcw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (src & SLJIT_MEM) {
		FAIL_IF(sljit_emit_simd_mem_offset(compiler, &src, srcw));

		ins = (sljit_ins)elem_size << 10;

		if (reg_size == 4)
			ins |= (sljit_ins)1 << 30;

		return push_inst(compiler, LD1R | ins | RN(src) | VT(freg));
	}

	ins = (sljit_ins)1 << (16 + elem_size);

	if (reg_size == 4)
		ins |= (sljit_ins)1 << 30;

	if (type & SLJIT_SIMD_FLOAT) {
		if (src == SLJIT_IMM)
			return push_inst(compiler, MOVI | (ins & ((sljit_ins)1 << 30)) | VD(freg));

		return push_inst(compiler, DUP_e | ins | VD(freg) | VN(src));
	}

	if (src == SLJIT_IMM) {
		if (elem_size < 3)
			srcw &= ((sljit_sw)1 << (((sljit_sw)1 << elem_size) << 3)) - 1;

		imm = simd_get_imm(elem_size, (sljit_uw)srcw);

		if (imm != ~(sljit_ins)0) {
			imm |= ins & ((sljit_ins)1 << 30);

			return push_inst(compiler, MOVI | imm | VD(freg));
		}

		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
	}

	return push_inst(compiler, DUP_g | ins | VD(freg) | RN(src));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg, sljit_s32 lane_index,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_mov(compiler, type, freg, lane_index, srcdst, srcdstw));

	ADJUST_LOCAL_OFFSET(srcdst, srcdstw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (type & SLJIT_SIMD_LANE_ZERO) {
		ins = (reg_size == 3) ? 0 : ((sljit_ins)1 << 30);

		if ((type & SLJIT_SIMD_FLOAT) && freg == srcdst) {
			FAIL_IF(push_inst(compiler, ORR_v | ins | VD(TMP_FREG1) | VN(freg) | VM(freg)));
			srcdst = TMP_FREG1;
			srcdstw = 0;
		}

		FAIL_IF(push_inst(compiler, MOVI | ins | VD(freg)));
	}

	if (srcdst & SLJIT_MEM) {
		FAIL_IF(sljit_emit_simd_mem_offset(compiler, &srcdst, srcdstw));

		if (elem_size == 3)
			ins = 0x8400;
		else if (elem_size == 0)
			ins = 0;
		else
			ins = (sljit_ins)0x2000 << elem_size;

		lane_index = lane_index << elem_size;
		ins |= (sljit_ins)(((lane_index & 0x8) << 27) | ((lane_index & 0x7) << 10));

		return push_inst(compiler, ((type & SLJIT_SIMD_STORE) ? ST1_s : LD1_s) | ins | RN(srcdst) | VT(freg));
	}

	if (type & SLJIT_SIMD_FLOAT) {
		if (type & SLJIT_SIMD_STORE)
			ins = INS_e | ((sljit_ins)1 << (16 + elem_size)) | ((sljit_ins)lane_index << (11 + elem_size)) | VD(srcdst) | VN(freg);
		else
			ins = INS_e | ((((sljit_ins)lane_index << 1) | 1) << (16 + elem_size)) | VD(freg) | VN(srcdst);

		return push_inst(compiler, ins);
	}

	if (srcdst == SLJIT_IMM) {
		if (elem_size < 3)
			srcdstw &= ((sljit_sw)1 << (((sljit_sw)1 << elem_size) << 3)) - 1;

		FAIL_IF(load_immediate(compiler, TMP_REG1, srcdstw));
		srcdst = TMP_REG1;
	}

	if (type & SLJIT_SIMD_STORE) {
		ins = RD(srcdst) | VN(freg);

		if ((type & SLJIT_SIMD_LANE_SIGNED) && (elem_size < 2 || (elem_size == 2 && !(type & SLJIT_32)))) {
			ins |= SMOV;

			if (!(type & SLJIT_32))
				ins |= (sljit_ins)1 << 30;
		} else
			ins |= UMOV;
	} else
		ins = INS | VD(freg) | RN(srcdst);

	if (elem_size == 3)
		ins |= (sljit_ins)1 << 30;

	return push_inst(compiler, ins | ((((sljit_ins)lane_index << 1) | 1) << (16 + elem_size)));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 src, sljit_s32 src_lane_index)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_replicate(compiler, type, freg, src, src_lane_index));

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	ins = (((sljit_ins)src_lane_index << 1) | 1) << (16 + elem_size);

	if (reg_size == 4)
		ins |= (sljit_ins)1 << 30;

	return push_inst(compiler, DUP_e | ins | VD(freg) | VN(src));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_extend(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_s32 elem2_size = SLJIT_SIMD_GET_ELEM2_SIZE(type);

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_extend(compiler, type, freg, src, srcw));

	ADJUST_LOCAL_OFFSET(src, srcw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size != 2 || elem2_size != 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (src & SLJIT_MEM) {
		FAIL_IF(sljit_emit_simd_mem_offset(compiler, &src, srcw));

		if (reg_size == 4 && elem2_size - elem_size == 1)
			FAIL_IF(push_inst(compiler, LD1 | ((sljit_ins)elem_size << 10) | RN(src) | VT(freg)));
		else
			FAIL_IF(push_inst(compiler, LD1_s | ((sljit_ins)0x2000 << (reg_size - elem2_size + elem_size)) | RN(src) | VT(freg)));
		src = freg;
	}

	if (type & SLJIT_SIMD_FLOAT) {
		SLJIT_ASSERT(reg_size == 4);
		return push_inst(compiler, FCVTL | (1 << 22) | VD(freg) | VN(src));
	}

	do {
		FAIL_IF(push_inst(compiler, ((type & SLJIT_SIMD_EXTEND_SIGNED) ? SSHLL : USHLL)
			| ((sljit_ins)1 << (19 + elem_size)) | VD(freg) | VN(src)));
		src = freg;
	} while (++elem_size < elem2_size);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_sign(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 dst, sljit_sw dstw)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins, imms;
	sljit_s32 dst_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_sign(compiler, type, freg, dst, dstw));

	ADJUST_LOCAL_OFFSET(dst, dstw);

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	switch (elem_size) {
	case 0:
		imms = 0x643219;
		ins = USHR | (0x9 << 16);
		break;
	case 1:
		imms = (reg_size == 4) ? 0x643219 : 0x6231;
		ins = USHR | (0x11 << 16);
		break;
	case 2:
		imms = (reg_size == 4) ? 0x6231 : 0x61;
		ins = USHR | (0x21 << 16);
		break;
	default:
		imms = 0x61;
		ins = USHR | (0x41 << 16);
		break;
	}

	if (reg_size == 4)
		ins |= (1 << 30);

	FAIL_IF(push_inst(compiler, ins | VD(TMP_FREG1) | VN(freg)));

	if (reg_size == 4 && elem_size > 0)
		FAIL_IF(push_inst(compiler, XTN | ((sljit_ins)(elem_size - 1) << 22) | VD(TMP_FREG1) | VN(TMP_FREG1)));

	if (imms >= 0x100) {
		ins = (reg_size == 4 && elem_size == 0) ? (1 << 30) : 0;

		do {
			FAIL_IF(push_inst(compiler, USRA | ins | ((imms & 0xff) << 16) | VD(TMP_FREG1) | VN(TMP_FREG1)));
			imms >>= 8;
		} while (imms >= 0x100);
	}

	FAIL_IF(push_inst(compiler, USRA | (1 << 30) | (imms << 16) | VD(TMP_FREG1) | VN(TMP_FREG1)));

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	ins = (0x1 << 16);

	if (reg_size == 4 && elem_size == 0) {
		FAIL_IF(push_inst(compiler, INS_e | (0x3 << 16) | (0x8 << 11) | VD(TMP_FREG1) | VN(TMP_FREG1)));
		ins = (0x2 << 16);
	}

	FAIL_IF(push_inst(compiler, UMOV | ins | RD(dst_r) | VN(TMP_FREG1)));

	if (dst_r == TMP_REG1)
		return emit_op_mem(compiler, STORE | ((type & SLJIT_32) ? INT_SIZE : WORD_SIZE), TMP_REG1, dst, dstw, TMP_REG2);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_op2(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg, sljit_s32 src1_freg, sljit_s32 src2_freg)
{
	sljit_s32 reg_size = SLJIT_SIMD_GET_REG_SIZE(type);
	sljit_s32 elem_size = SLJIT_SIMD_GET_ELEM_SIZE(type);
	sljit_ins ins = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_op2(compiler, type, dst_freg, src1_freg, src2_freg));

	if (reg_size != 3 && reg_size != 4)
		return SLJIT_ERR_UNSUPPORTED;

	if ((type & SLJIT_SIMD_FLOAT) && (elem_size < 2 || elem_size > 3))
		return SLJIT_ERR_UNSUPPORTED;

	switch (SLJIT_SIMD_GET_OPCODE(type)) {
	case SLJIT_SIMD_OP2_AND:
		ins = AND_v;
		break;
	case SLJIT_SIMD_OP2_OR:
		ins = ORR_v;
		break;
	case SLJIT_SIMD_OP2_XOR:
		ins = EOR_v;
		break;
	}

	if (type & SLJIT_SIMD_TEST)
		return SLJIT_SUCCESS;

	if (reg_size == 4)
		ins |= (sljit_ins)1 << 30;

	return push_inst(compiler, ins | VD(dst_freg) | VN(src1_freg) | VM(src2_freg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_load(compiler, op, dst_reg, mem_reg));

#ifdef __ARM_FEATURE_ATOMICS
	switch (GET_OPCODE(op)) {
	case SLJIT_MOV32:
	case SLJIT_MOV_U32:
		ins = LDR ^ (1 << 30);
		break;
	case SLJIT_MOV_U16:
		ins = LDRH;
		break;
	case SLJIT_MOV_U8:
		ins = LDRB;
		break;
	default:
		ins = LDR;
		break;
	}
#else /* !__ARM_FEATURE_ATOMICS */
	switch (GET_OPCODE(op)) {
	case SLJIT_MOV32:
	case SLJIT_MOV_U32:
		ins = LDXR ^ (1 << 30);
		break;
	case SLJIT_MOV_U8:
		ins = LDXRB;
		break;
	case SLJIT_MOV_U16:
		ins = LDXRH;
		break;
	default:
		ins = LDXR;
		break;
	}
#endif /* ARM_FEATURE_ATOMICS */
	return push_inst(compiler, ins | RN(mem_reg) | RT(dst_reg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	sljit_ins ins;
	sljit_s32 tmp = temp_reg;
	sljit_ins cmp = 0;
	sljit_ins inv_bits = W_OP;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_store(compiler, op, src_reg, mem_reg, temp_reg));

#ifdef __ARM_FEATURE_ATOMICS
	if (op & SLJIT_SET_ATOMIC_STORED)
		cmp = (SUBS ^ W_OP) | RD(TMP_ZERO);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV32:
	case SLJIT_MOV_U32:
		ins = CAS ^ (1 << 30);
		break;
	case SLJIT_MOV_U16:
		ins = CASH;
		break;
	case SLJIT_MOV_U8:
		ins = CASB;
		break;
	default:
		ins = CAS;
		inv_bits = 0;
		if (cmp)
			cmp ^= W_OP;
		break;
	}

	if (cmp) {
		FAIL_IF(push_inst(compiler, (MOV ^ inv_bits) | RM(temp_reg) | RD(TMP_REG1)));
		tmp = TMP_REG1;
	}
	FAIL_IF(push_inst(compiler, ins | RM(tmp) | RN(mem_reg) | RD(src_reg)));
	if (!cmp)
		return SLJIT_SUCCESS;

	FAIL_IF(push_inst(compiler, cmp | RM(tmp) | RN(temp_reg)));
	FAIL_IF(push_inst(compiler, (CSET ^ inv_bits) | RD(tmp)));
	return push_inst(compiler, cmp | RM(tmp) | RN(TMP_ZERO));
#else /* !__ARM_FEATURE_ATOMICS */
	SLJIT_UNUSED_ARG(tmp);
	SLJIT_UNUSED_ARG(inv_bits);

	if (op & SLJIT_SET_ATOMIC_STORED)
		cmp = (SUBI ^ W_OP) | (1 << 29);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV32:
	case SLJIT_MOV_U32:
		ins = STXR ^ (1 << 30);
		break;
	case SLJIT_MOV_U8:
		ins = STXRB;
		break;
	case SLJIT_MOV_U16:
		ins = STXRH;
		break;
	default:
		ins = STXR;
		break;
	}

	FAIL_IF(push_inst(compiler, ins | RM(TMP_REG1) | RN(mem_reg) | RT(src_reg)));
	return cmp ? push_inst(compiler, cmp | RD(TMP_ZERO) | RN(TMP_REG1)) : SLJIT_SUCCESS;
#endif /* __ARM_FEATURE_ATOMICS */
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	sljit_s32 dst_reg;
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_get_local_base(compiler, dst, dstw, offset));
	ADJUST_LOCAL_OFFSET(SLJIT_MEM1(SLJIT_SP), offset);

	dst_reg = FAST_IS_REG(dst) ? dst : TMP_REG1;

	/* Not all instruction forms support accessing SP register. */
	if (offset <= 0xffffff && offset >= -0xffffff) {
		ins = ADDI;
		if (offset < 0) {
			offset = -offset;
			ins = SUBI;
		}

		if (offset <= 0xfff)
			FAIL_IF(push_inst(compiler, ins | RD(dst_reg) | RN(SLJIT_SP) | (sljit_ins)(offset << 10)));
		else {
			FAIL_IF(push_inst(compiler, ins | RD(dst_reg) | RN(SLJIT_SP) | (sljit_ins)((offset & 0xfff000) >> (12 - 10)) | (1 << 22)));

			offset &= 0xfff;
			if (offset != 0)
				FAIL_IF(push_inst(compiler, ins | RD(dst_reg) | RN(dst_reg) | (sljit_ins)(offset << 10)));
		}
	}
	else {
		FAIL_IF(load_immediate (compiler, dst_reg, offset));
		/* Add extended register form. */
		FAIL_IF(push_inst(compiler, ADDE | (0x3 << 13) | RD(dst_reg) | RN(SLJIT_SP) | RM(dst_reg)));
	}

	if (SLJIT_UNLIKELY(dst & SLJIT_MEM))
		return emit_op_mem(compiler, WORD_SIZE | STORE, dst_reg, dst, dstw, TMP_REG1);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	struct sljit_const *const_;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	PTR_FAIL_IF(emit_imm64_const(compiler, dst_r, (sljit_uw)init_value));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_SIZE | STORE, dst_r, dst, dstw, TMP_REG2));
	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_put_label *put_label;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_put_label(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG1;
	PTR_FAIL_IF(emit_imm64_const(compiler, dst_r, 0));

	put_label = (struct sljit_put_label*)ensure_abuf(compiler, sizeof(struct sljit_put_label));
	PTR_FAIL_IF(!put_label);
	set_put_label(put_label, compiler, 1);

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_SIZE | STORE, dst_r, dst, dstw, TMP_REG2));

	return put_label;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins* inst = (sljit_ins*)addr;
	sljit_u32 dst;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 4, 0);

	dst = inst[0] & 0x1f;
	SLJIT_ASSERT((inst[0] & 0xffe00000) == MOVZ && (inst[1] & 0xffe00000) == (MOVK | (1 << 21)));
	inst[0] = MOVZ | dst | (((sljit_u32)new_target & 0xffff) << 5);
	inst[1] = MOVK | dst | (((sljit_u32)(new_target >> 16) & 0xffff) << 5) | (1 << 21);
	inst[2] = MOVK | dst | (((sljit_u32)(new_target >> 32) & 0xffff) << 5) | (2 << 21);
	inst[3] = MOVK | dst | ((sljit_u32)(new_target >> 48) << 5) | (3 << 21);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 4, 1);
	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 4);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}
