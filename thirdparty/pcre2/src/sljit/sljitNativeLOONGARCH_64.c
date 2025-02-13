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
	return "LOONGARCH" SLJIT_CPUINFO;
}

typedef sljit_u32 sljit_ins;

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3	(SLJIT_NUMBER_OF_REGISTERS + 4)
#define TMP_ZERO	0

/* Flags are kept in volatile registers. */
#define EQUAL_FLAG	(SLJIT_NUMBER_OF_REGISTERS + 5)
#define RETURN_ADDR_REG	TMP_REG2
#define OTHER_FLAG	(SLJIT_NUMBER_OF_REGISTERS + 6)

#define TMP_FREG1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define TMP_FREG2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS + 2)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 7] = {
	0, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 22, 31, 30, 29, 28, 27, 26, 25, 24, 23, 3, 13, 1, 14, 12, 15
};

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 3] = {
	0, 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 31, 30, 29, 28, 27, 26, 25, 24, 8, 9
};

/* --------------------------------------------------------------------- */
/*  Instrucion forms                                                     */
/* --------------------------------------------------------------------- */

/*
LoongArch instructions are 32 bits wide, belonging to 9 basic instruction formats (and variants of them):

| Format name  | Composition                 |
| 2R           | Opcode + Rj + Rd            |
| 3R           | Opcode + Rk + Rj + Rd       |
| 4R           | Opcode + Ra + Rk + Rj + Rd  |
| 2RI8         | Opcode + I8 + Rj + Rd       |
| 2RI12        | Opcode + I12 + Rj + Rd      |
| 2RI14        | Opcode + I14 + Rj + Rd      |
| 2RI16        | Opcode + I16 + Rj + Rd      |
| 1RI21        | Opcode + I21L + Rj + I21H   |
| I26          | Opcode + I26L + I26H        |

Rd is the destination register operand, while Rj, Rk and Ra (“a” stands for “additional”) are the source register operands.
I8/I12/I14/I16/I21/I26 are immediate operands of respective width. The longer I21 and I26 are stored in separate higher and
lower parts in the instruction word, denoted by the “L” and “H” suffixes. */

#define RD(rd) ((sljit_ins)reg_map[rd])
#define RJ(rj) ((sljit_ins)reg_map[rj] << 5)
#define RK(rk) ((sljit_ins)reg_map[rk] << 10)
#define RA(ra) ((sljit_ins)reg_map[ra] << 15)

#define FD(fd) ((sljit_ins)reg_map[fd])
#define FRD(fd) ((sljit_ins)freg_map[fd])
#define FRJ(fj) ((sljit_ins)freg_map[fj] << 5)
#define FRK(fk) ((sljit_ins)freg_map[fk] << 10)
#define FRA(fa) ((sljit_ins)freg_map[fa] << 15)

#define IMM_I8(imm) (((sljit_ins)(imm)&0xff) << 10)
#define IMM_I12(imm) (((sljit_ins)(imm)&0xfff) << 10)
#define IMM_I14(imm) (((sljit_ins)(imm)&0xfff3) << 10)
#define IMM_I16(imm) (((sljit_ins)(imm)&0xffff) << 10)
#define IMM_I21(imm) ((((sljit_ins)(imm)&0xffff) << 10) | (((sljit_ins)(imm) >> 16) & 0x1f))
#define IMM_I26(imm) ((((sljit_ins)(imm)&0xffff) << 10) | (((sljit_ins)(imm) >> 16) & 0x3ff))

#define OPC_I26(opc) ((sljit_ins)(opc) << 26)
#define OPC_1RI21(opc) ((sljit_ins)(opc) << 26)
#define OPC_2RI16(opc) ((sljit_ins)(opc) << 26)
#define OPC_2RI14(opc) ((sljit_ins)(opc) << 24)
#define OPC_2RI12(opc) ((sljit_ins)(opc) << 22)
#define OPC_2RI8(opc) ((sljit_ins)(opc) << 18)
#define OPC_4R(opc) ((sljit_ins)(opc) << 20)
#define OPC_3R(opc) ((sljit_ins)(opc) << 15)
#define OPC_2R(opc) ((sljit_ins)(opc) << 10)
#define OPC_1RI20(opc) ((sljit_ins)(opc) << 25)

/* Arithmetic operation instructions */
#define ADD_W OPC_3R(0x20)
#define ADD_D OPC_3R(0x21)
#define SUB_W OPC_3R(0x22)
#define SUB_D OPC_3R(0x23)
#define ADDI_W OPC_2RI12(0xa)
#define ADDI_D OPC_2RI12(0xb)
#define ANDI OPC_2RI12(0xd)
#define ORI OPC_2RI12(0xe)
#define XORI OPC_2RI12(0xf)
#define ADDU16I_D OPC_2RI16(0x4)
#define LU12I_W OPC_1RI20(0xa)
#define LU32I_D OPC_1RI20(0xb)
#define LU52I_D OPC_2RI12(0xc)
#define SLT OPC_3R(0x24)
#define SLTU OPC_3R(0x25)
#define SLTI OPC_2RI12(0x8)
#define SLTUI OPC_2RI12(0x9)
#define PCADDI OPC_1RI20(0xc)
#define PCALAU12I OPC_1RI20(0xd)
#define PCADDU12I OPC_1RI20(0xe)
#define PCADDU18I OPC_1RI20(0xf)
#define NOR OPC_3R(0x28)
#define AND OPC_3R(0x29)
#define OR OPC_3R(0x2a)
#define XOR OPC_3R(0x2b)
#define ORN OPC_3R(0x2c)
#define ANDN OPC_3R(0x2d)
#define MUL_W OPC_3R(0x38)
#define MULH_W OPC_3R(0x39)
#define MULH_WU OPC_3R(0x3a)
#define MUL_D OPC_3R(0x3b)
#define MULH_D OPC_3R(0x3c)
#define MULH_DU OPC_3R(0x3d)
#define MULW_D_W OPC_3R(0x3e)
#define MULW_D_WU OPC_3R(0x3f)
#define DIV_W OPC_3R(0x40)
#define MOD_W OPC_3R(0x41)
#define DIV_WU OPC_3R(0x42)
#define MOD_WU OPC_3R(0x43)
#define DIV_D OPC_3R(0x44)
#define MOD_D OPC_3R(0x45)
#define DIV_DU OPC_3R(0x46)
#define MOD_DU OPC_3R(0x47)

/* Bit-shift instructions */
#define SLL_W OPC_3R(0x2e)
#define SRL_W OPC_3R(0x2f)
#define SRA_W OPC_3R(0x30)
#define SLL_D OPC_3R(0x31)
#define SRL_D OPC_3R(0x32)
#define SRA_D OPC_3R(0x33)
#define ROTR_W OPC_3R(0x36)
#define ROTR_D OPC_3R(0x37)
#define SLLI_W OPC_3R(0x81)
#define SLLI_D ((sljit_ins)(0x41) << 16)
#define SRLI_W OPC_3R(0x89)
#define SRLI_D ((sljit_ins)(0x45) << 16)
#define SRAI_W OPC_3R(0x91)
#define SRAI_D ((sljit_ins)(0x49) << 16)
#define ROTRI_W OPC_3R(0x99)
#define ROTRI_D ((sljit_ins)(0x4d) << 16)

/* Bit-manipulation instructions */
#define CLO_W OPC_2R(0x4)
#define CLZ_W OPC_2R(0x5)
#define CTO_W OPC_2R(0x6)
#define CTZ_W OPC_2R(0x7)
#define CLO_D OPC_2R(0x8)
#define CLZ_D OPC_2R(0x9)
#define CTO_D OPC_2R(0xa)
#define CTZ_D OPC_2R(0xb)
#define REVB_2H OPC_2R(0xc)
#define REVB_4H OPC_2R(0xd)
#define REVB_2W OPC_2R(0xe)
#define REVB_D OPC_2R(0xf)
#define REVH_2W OPC_2R(0x10)
#define REVH_D OPC_2R(0x11)
#define BITREV_4B OPC_2R(0x12)
#define BITREV_8B OPC_2R(0x13)
#define BITREV_W OPC_2R(0x14)
#define BITREV_D OPC_2R(0x15)
#define EXT_W_H OPC_2R(0x16)
#define EXT_W_B OPC_2R(0x17)
#define BSTRINS_W (0x1 << 22 | 1 << 21)
#define BSTRPICK_W (0x1 << 22 | 1 << 21 | 1 << 15)
#define BSTRINS_D (0x2 << 22)
#define BSTRPICK_D (0x3 << 22)

/* Branch instructions */
#define BEQZ  OPC_1RI21(0x10)
#define BNEZ  OPC_1RI21(0x11)
#define JIRL  OPC_2RI16(0x13)
#define B     OPC_I26(0x14)
#define BL    OPC_I26(0x15)
#define BEQ   OPC_2RI16(0x16)
#define BNE   OPC_2RI16(0x17)
#define BLT   OPC_2RI16(0x18)
#define BGE   OPC_2RI16(0x19)
#define BLTU  OPC_2RI16(0x1a)
#define BGEU  OPC_2RI16(0x1b)

/* Memory access instructions */
#define LD_B OPC_2RI12(0xa0)
#define LD_H OPC_2RI12(0xa1)
#define LD_W OPC_2RI12(0xa2)
#define LD_D OPC_2RI12(0xa3)

#define ST_B OPC_2RI12(0xa4)
#define ST_H OPC_2RI12(0xa5)
#define ST_W OPC_2RI12(0xa6)
#define ST_D OPC_2RI12(0xa7)

#define LD_BU OPC_2RI12(0xa8)
#define LD_HU OPC_2RI12(0xa9)
#define LD_WU OPC_2RI12(0xaa)

#define LDX_B OPC_3R(0x7000)
#define LDX_H OPC_3R(0x7008)
#define LDX_W OPC_3R(0x7010)
#define LDX_D OPC_3R(0x7018)

#define STX_B OPC_3R(0x7020)
#define STX_H OPC_3R(0x7028)
#define STX_W OPC_3R(0x7030)
#define STX_D OPC_3R(0x7038)

#define LDX_BU OPC_3R(0x7040)
#define LDX_HU OPC_3R(0x7048)
#define LDX_WU OPC_3R(0x7050)

#define PRELD OPC_2RI12(0xab)

/* Atomic memory access instructions */
#define LL_W OPC_2RI14(0x20)
#define SC_W OPC_2RI14(0x21)
#define LL_D OPC_2RI14(0x22)
#define SC_D OPC_2RI14(0x23)

/* LoongArch V1.10 Instructions */
#define AMCAS_B OPC_3R(0x70B0)
#define AMCAS_H OPC_3R(0x70B1)
#define AMCAS_W OPC_3R(0x70B2)
#define AMCAS_D OPC_3R(0x70B3)

/* Other instructions */
#define BREAK OPC_3R(0x54)
#define DBGCALL OPC_3R(0x55)
#define SYSCALL OPC_3R(0x56)

/* Basic Floating-Point Instructions */
/* Floating-Point Arithmetic Operation Instructions */
#define FADD_S  OPC_3R(0x201)
#define FADD_D  OPC_3R(0x202)
#define FSUB_S  OPC_3R(0x205)
#define FSUB_D  OPC_3R(0x206)
#define FMUL_S  OPC_3R(0x209)
#define FMUL_D  OPC_3R(0x20a)
#define FDIV_S  OPC_3R(0x20d)
#define FDIV_D  OPC_3R(0x20e)
#define FCMP_COND_S  OPC_4R(0xc1)
#define FCMP_COND_D  OPC_4R(0xc2)
#define FCOPYSIGN_S  OPC_3R(0x225)
#define FCOPYSIGN_D  OPC_3R(0x226)
#define FSEL  OPC_4R(0xd0)
#define FABS_S  OPC_2R(0x4501)
#define FABS_D  OPC_2R(0x4502)
#define FNEG_S  OPC_2R(0x4505)
#define FNEG_D  OPC_2R(0x4506)
#define FMOV_S  OPC_2R(0x4525)
#define FMOV_D  OPC_2R(0x4526)

/* Floating-Point Conversion Instructions */
#define FCVT_S_D  OPC_2R(0x4646)
#define FCVT_D_S  OPC_2R(0x4649)
#define FTINTRZ_W_S  OPC_2R(0x46a1)
#define FTINTRZ_W_D  OPC_2R(0x46a2)
#define FTINTRZ_L_S  OPC_2R(0x46a9)
#define FTINTRZ_L_D  OPC_2R(0x46aa)
#define FFINT_S_W  OPC_2R(0x4744)
#define FFINT_S_L  OPC_2R(0x4746)
#define FFINT_D_W  OPC_2R(0x4748)
#define FFINT_D_L  OPC_2R(0x474a)

/* Floating-Point Move Instructions */
#define FMOV_S  OPC_2R(0x4525)
#define FMOV_D  OPC_2R(0x4526)
#define MOVGR2FR_W  OPC_2R(0x4529)
#define MOVGR2FR_D  OPC_2R(0x452a)
#define MOVGR2FRH_W  OPC_2R(0x452b)
#define MOVFR2GR_S  OPC_2R(0x452d)
#define MOVFR2GR_D  OPC_2R(0x452e)
#define MOVFRH2GR_S  OPC_2R(0x452f)
#define MOVGR2FCSR  OPC_2R(0x4530)
#define MOVFCSR2GR  OPC_2R(0x4532)
#define MOVFR2CF  OPC_2R(0x4534)
#define MOVCF2FR  OPC_2R(0x4535)
#define MOVGR2CF  OPC_2R(0x4536)
#define MOVCF2GR  OPC_2R(0x4537)

/* Floating-Point Branch Instructions */
#define BCEQZ OPC_I26(0x12)
#define BCNEZ OPC_I26(0x12)

/* Floating-Point Common Memory Access Instructions */
#define FLD_S OPC_2RI12(0xac)
#define FLD_D OPC_2RI12(0xae)
#define FST_S OPC_2RI12(0xad)
#define FST_D OPC_2RI12(0xaf)

#define FLDX_S OPC_3R(0x7060)
#define FLDX_D OPC_3R(0x7068)
#define FSTX_S OPC_3R(0x7070)
#define FSTX_D OPC_3R(0x7078)

#define I12_MAX (0x7ff)
#define I12_MIN (-0x800)
#define BRANCH16_MAX (0x7fff << 2)
#define BRANCH16_MIN (-(0x8000 << 2))
#define BRANCH21_MAX (0xfffff << 2)
#define BRANCH21_MIN (-(0x100000 << 2))
#define JUMP_MAX (0x1ffffff << 2)
#define JUMP_MIN (-(0x2000000 << 2))
#define JIRL_MAX (0x7fff << 2)
#define JIRL_MIN (-(0x8000 << 2))

#define S32_MAX		(0x7fffffffl)
#define S32_MIN		(-0x80000000l)
#define S52_MAX		(0x7ffffffffffffl)

#define INST(inst, type) ((sljit_ins)((type & SLJIT_32) ? inst##_W : inst##_D))

/* LoongArch CPUCFG register for feature detection */
#define LOONGARCH_CFG2			0x02
#define LOONGARCH_FEATURE_LAMCAS	(1 << 28)

static sljit_u32 cpu_feature_list = 0;

static SLJIT_INLINE sljit_u32 get_cpu_features(void)
{
	if (cpu_feature_list == 0)
		__asm__ ("cpucfg %0, %1" : "+&r"(cpu_feature_list) : "r"(LOONGARCH_CFG2));
	return cpu_feature_list;
}

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_ins* detect_jump_type(struct sljit_jump *jump, sljit_ins *code, sljit_sw executable_offset)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_ins *inst;

	inst = (sljit_ins *)jump->addr;

	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		goto exit;

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size) + (sljit_uw)executable_offset;
	}

	diff = (sljit_sw)target_addr - (sljit_sw)inst - executable_offset;

	if (jump->flags & IS_COND) {
		inst--;
		diff += SSIZE_OF(ins);

		if (diff >= BRANCH16_MIN && diff <= BRANCH16_MAX) {
			jump->flags |= PATCH_B;
			inst[0] = (inst[0] & 0xfc0003ff) ^ 0x4000000;
			jump->addr = (sljit_uw)inst;
			return inst;
		}

		inst++;
		diff -= SSIZE_OF(ins);
	}

	if (diff >= JUMP_MIN && diff <= JUMP_MAX) {
		if (jump->flags & IS_COND) {
			inst[-1] |= (sljit_ins)IMM_I16(2);
		}

		jump->flags |= PATCH_J;
		return inst;
	}

	if (diff >= S32_MIN && diff <= S32_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] |= (sljit_ins)IMM_I16(3);

		jump->flags |= PATCH_REL32;
		inst[1] = inst[0];
		return inst + 1;
	}

	if (target_addr <= (sljit_uw)S32_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] |= (sljit_ins)IMM_I16(3);

		jump->flags |= PATCH_ABS32;
		inst[1] = inst[0];
		return inst + 1;
	}

	if (target_addr <= S52_MAX) {
		if (jump->flags & IS_COND)
			inst[-1] |= (sljit_ins)IMM_I16(4);

		jump->flags |= PATCH_ABS52;
		inst[2] = inst[0];
		return inst + 2;
	}

exit:
	if (jump->flags & IS_COND)
		inst[-1] |= (sljit_ins)IMM_I16(5);
	inst[3] = inst[0];
	return inst + 3;
}

static SLJIT_INLINE sljit_sw put_label_get_length(struct sljit_put_label *put_label, sljit_uw max_label)
{
	if (max_label <= (sljit_uw)S32_MAX) {
		put_label->flags = PATCH_ABS32;
		return 1;
	}

	if (max_label <= S52_MAX) {
		put_label->flags = PATCH_ABS52;
		return 2;
	}

	put_label->flags = 0;
	return 3;
}

static SLJIT_INLINE void load_addr_to_reg(void *dst, sljit_u32 reg)
{
	struct sljit_jump *jump = NULL;
	struct sljit_put_label *put_label;
	sljit_uw flags;
	sljit_ins *inst;
	sljit_uw addr;

	if (reg != 0) {
		jump = (struct sljit_jump*)dst;
		flags = jump->flags;
		inst = (sljit_ins*)jump->addr;
		addr = (flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
	} else {
		put_label = (struct sljit_put_label*)dst;
		flags = put_label->flags;
		inst = (sljit_ins*)put_label->addr;
		addr = put_label->label->addr;
		reg = *inst;
	}

	if (flags & PATCH_ABS32) {
		SLJIT_ASSERT(addr <= S32_MAX);
		inst[0] = LU12I_W | RD(reg) | (sljit_ins)(((addr & 0xffffffff) >> 12) << 5);
	} else if (flags & PATCH_ABS52) {
		inst[0] = LU12I_W | RD(reg) | (sljit_ins)(((addr & 0xffffffff) >> 12) << 5);
		inst[1] = LU32I_D | RD(reg) | (sljit_ins)(((addr >> 32) & 0xfffff) << 5);
		inst += 1;
	} else {
		inst[0] = LU12I_W | RD(reg) | (sljit_ins)(((addr & 0xffffffff) >> 12) << 5);
		inst[1] = LU32I_D | RD(reg) | (sljit_ins)(((addr >> 32) & 0xfffff) << 5);
		inst[2] = LU52I_D | RD(reg) | RJ(reg) | IMM_I12(addr >> 52);
		inst += 2;
	}

	if (jump != NULL) {
		SLJIT_ASSERT((inst[1] & OPC_2RI16(0x3f)) == JIRL);
		inst[1] = (inst[1] & (OPC_2RI16(0x3f) | 0x3ff)) | IMM_I16((addr & 0xfff) >> 2);
	} else
		inst[1] = ORI | RD(reg) | RJ(reg) | IMM_I12(addr);
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
	sljit_uw addr;

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
					word_count += 3;
					jump->addr = (sljit_uw)code_ptr;
					code_ptr = detect_jump_type(jump, code, executable_offset);
					jump = jump->next;
				}
				if (const_ && const_->addr == word_count) {
					const_->addr = (sljit_uw)code_ptr;
					const_ = const_->next;
				}
				if (put_label && put_label->addr == word_count) {
					SLJIT_ASSERT(put_label->label);
					put_label->addr = (sljit_uw)code_ptr;

					code_ptr += put_label_get_length(put_label, (sljit_uw)(SLJIT_ADD_EXEC_OFFSET(code, executable_offset) + put_label->label->size));
					word_count += 3;

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
		label->addr = (sljit_uw)code_ptr;
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
			if (!(jump->flags & (PATCH_B | PATCH_J | PATCH_REL32))) {
				load_addr_to_reg(jump, TMP_REG1);
				break;
			}

			addr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
			buf_ptr = (sljit_ins *)jump->addr;
			addr -= (sljit_uw)SLJIT_ADD_EXEC_OFFSET(buf_ptr, executable_offset);

			if (jump->flags & PATCH_B) {
				SLJIT_ASSERT((sljit_sw)addr >= BRANCH16_MIN && (sljit_sw)addr <= BRANCH16_MAX);
				buf_ptr[0] |= (sljit_ins)IMM_I16(addr >> 2);
				break;
			}

			if (jump->flags & PATCH_REL32) {
				SLJIT_ASSERT((sljit_sw)addr >= S32_MIN && (sljit_sw)addr <= S32_MAX);

				buf_ptr[0] = PCADDU12I | RD(TMP_REG1) | (sljit_ins)((sljit_sw)addr & ~0xfff);
				SLJIT_ASSERT((buf_ptr[1] & OPC_2RI16(0x3f)) == JIRL);
				buf_ptr[1] |= IMM_I16((addr & 0xfff) >> 2);
				break;
			}

			SLJIT_ASSERT((sljit_sw)addr >= JUMP_MIN && (sljit_sw)addr <= JUMP_MAX);
			if (jump->flags & IS_CALL)
				buf_ptr[0] = BL | (sljit_ins)IMM_I26(addr >> 2);
			else
				buf_ptr[0] = B | (sljit_ins)IMM_I26(addr >> 2);
		} while (0);
		jump = jump->next;
	}

	put_label = compiler->put_labels;
	while (put_label) {
		load_addr_to_reg(put_label, 0);
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
	switch (feature_type)
	{
	case SLJIT_HAS_FPU:
#ifdef SLJIT_IS_FPU_AVAILABLE
		return (SLJIT_IS_FPU_AVAILABLE) != 0;
#else
		/* Available by default. */
		return 1;
#endif

	case SLJIT_HAS_ATOMIC:
		return (LOONGARCH_FEATURE_LAMCAS & get_cpu_features());

	case SLJIT_HAS_CLZ:
	case SLJIT_HAS_CTZ:
	case SLJIT_HAS_REV:
	case SLJIT_HAS_ROT:
	case SLJIT_HAS_PREFETCH:
	case SLJIT_HAS_COPY_F32:
	case SLJIT_HAS_COPY_F64:
		return 1;

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

#define STACK_STORE	ST_D
#define STACK_LOAD	LD_D

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_r, sljit_sw imm)
{
	if (imm <= I12_MAX && imm >= I12_MIN)
		return push_inst(compiler, ADDI_D | RD(dst_r) | RJ(TMP_ZERO) | IMM_I12(imm));

	if (imm <= 0x7fffffffl && imm >= -0x80000000l) {
		FAIL_IF(push_inst(compiler, LU12I_W | RD(dst_r) | (sljit_ins)(((imm & 0xffffffff) >> 12) << 5)));
		return push_inst(compiler, ORI | RD(dst_r) | RJ(dst_r) | IMM_I12(imm));
	} else if (imm <= 0x7ffffffffffffl && imm >= -0x8000000000000l) {
		FAIL_IF(push_inst(compiler, LU12I_W | RD(dst_r) | (sljit_ins)(((imm & 0xffffffff) >> 12) << 5)));
		FAIL_IF(push_inst(compiler, ORI | RD(dst_r) | RJ(dst_r) | IMM_I12(imm)));
		return push_inst(compiler, LU32I_D | RD(dst_r) | (sljit_ins)(((imm >> 32) & 0xfffff) << 5));
	}
	FAIL_IF(push_inst(compiler, LU12I_W | RD(dst_r) | (sljit_ins)(((imm & 0xffffffff) >> 12) << 5)));
	FAIL_IF(push_inst(compiler, ORI | RD(dst_r) | RJ(dst_r) | IMM_I12(imm)));
	FAIL_IF(push_inst(compiler, LU32I_D | RD(dst_r) | (sljit_ins)(((imm >> 32) & 0xfffff) << 5)));
	return push_inst(compiler, LU52I_D | RD(dst_r) | RJ(dst_r) | IMM_I12(imm >> 52));
}

#define STACK_MAX_DISTANCE (-I12_MIN)

static sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw);

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 i, tmp, offset;
	sljit_s32 saved_arg_count = SLJIT_KEPT_SAVEDS_COUNT(options);

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - saved_arg_count, 1);
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);

	local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;
	compiler->local_size = local_size;

	if (local_size <= STACK_MAX_DISTANCE) {
		/* Frequent case. */
		FAIL_IF(push_inst(compiler, ADDI_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | IMM_I12(-local_size)));
		offset = local_size - SSIZE_OF(sw);
		local_size = 0;
	} else {
		FAIL_IF(push_inst(compiler, ADDI_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | IMM_I12(STACK_MAX_DISTANCE)));
		local_size -= STACK_MAX_DISTANCE;

		if (local_size > STACK_MAX_DISTANCE)
			FAIL_IF(load_immediate(compiler, TMP_REG1, local_size));
		offset = STACK_MAX_DISTANCE - SSIZE_OF(sw);
	}

	FAIL_IF(push_inst(compiler, STACK_STORE | RD(RETURN_ADDR_REG) | RJ(SLJIT_SP) | IMM_I12(offset)));

	tmp = SLJIT_S0 - saveds;
	for (i = SLJIT_S0 - saved_arg_count; i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_STORE | RD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_STORE | RD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, FST_D | FRD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, FST_D | FRD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	if (local_size > STACK_MAX_DISTANCE)
		FAIL_IF(push_inst(compiler, SUB_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | RK(TMP_REG1)));
	else if (local_size > 0)
		FAIL_IF(push_inst(compiler, ADDI_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | IMM_I12(-local_size)));

	if (options & SLJIT_ENTER_REG_ARG)
		return SLJIT_SUCCESS;

	arg_types >>= SLJIT_ARG_SHIFT;
	saved_arg_count = 0;
	tmp = SLJIT_R0;

	while (arg_types > 0) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				FAIL_IF(push_inst(compiler, ADDI_D | RD(SLJIT_S0 - saved_arg_count) | RJ(tmp) | IMM_I12(0)));
				saved_arg_count++;
			}
			tmp++;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

#undef STACK_MAX_DISTANCE

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds - SLJIT_KEPT_SAVEDS_COUNT(options), 1);
	local_size += GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, f64);

	compiler->local_size = (local_size + SLJIT_LOCALS_OFFSET + 15) & ~0xf;

	return SLJIT_SUCCESS;
}

#define STACK_MAX_DISTANCE (-I12_MIN - 16)

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler, sljit_s32 is_return_to)
{
	sljit_s32 i, tmp, offset;
	sljit_s32 local_size = compiler->local_size;

	if (local_size > STACK_MAX_DISTANCE) {
		local_size -= STACK_MAX_DISTANCE;

		if (local_size > STACK_MAX_DISTANCE) {
			FAIL_IF(load_immediate(compiler, TMP_REG2, local_size));
			FAIL_IF(push_inst(compiler, ADD_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | RK(TMP_REG2)));
		} else
			FAIL_IF(push_inst(compiler, ADDI_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | IMM_I12(local_size)));

		local_size = STACK_MAX_DISTANCE;
	}

	SLJIT_ASSERT(local_size > 0);

	offset = local_size - SSIZE_OF(sw);
	if (!is_return_to)
		FAIL_IF(push_inst(compiler, STACK_LOAD | RD(RETURN_ADDR_REG) | RJ(SLJIT_SP) | IMM_I12(offset)));

	tmp = SLJIT_S0 - compiler->saveds;
	for (i = SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options); i > tmp; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | RD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		offset -= SSIZE_OF(sw);
		FAIL_IF(push_inst(compiler, STACK_LOAD | RD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	tmp = SLJIT_FS0 - compiler->fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, FLD_D | FRD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	for (i = compiler->fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		offset -= SSIZE_OF(f64);
		FAIL_IF(push_inst(compiler, FLD_D | FRD(i) | RJ(SLJIT_SP) | IMM_I12(offset)));
	}

	return push_inst(compiler, ADDI_D | RD(SLJIT_SP) | RJ(SLJIT_SP) | IMM_I12(local_size));
}

#undef STACK_MAX_DISTANCE

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	FAIL_IF(emit_stack_frame_release(compiler, 0));
	return push_inst(compiler, JIRL | RD(TMP_ZERO) | RJ(RETURN_ADDR_REG) | IMM_I12(0));
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
		FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG1) | RJ(src) | IMM_I12(0)));
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
/* u w s */ ST_D /* st.d */,
/* u w l */ LD_D /* ld.d */,
/* u b s */ ST_B /* st.b */,
/* u b l */ LD_BU /* ld.bu */,
/* u h s */ ST_H /* st.h */,
/* u h l */ LD_HU /* ld.hu */,
/* u i s */ ST_W /* st.w */,
/* u i l */ LD_WU /* ld.wu */,

/* s w s */ ST_D /* st.d */,
/* s w l */ LD_D /* ld.d */,
/* s b s */ ST_B /* st.b */,
/* s b l */ LD_B /* ld.b */,
/* s h s */ ST_H /* st.h */,
/* s h l */ LD_H /* ld.h */,
/* s i s */ ST_W /* st.w */,
/* s i l */ LD_W /* ld.w */,

/* d   s */ FST_D /* fst.d */,
/* d   l */ FLD_D /* fld.d */,
/* s   s */ FST_S /* fst.s */,
/* s   l */ FLD_S /* fld.s */,
};

static const sljit_ins data_transfer_insts_x[16 + 4] = {
/* u w s */ STX_D /* stx.d */,
/* u w l */ LDX_D /* ldx.d */,
/* u b s */ STX_B /* stx.b */,
/* u b l */ LDX_BU /* ldx.bu */,
/* u h s */ STX_H /* stx.h */,
/* u h l */ LDX_HU /* ldx.hu */,
/* u i s */ STX_W /* stx.w */,
/* u i l */ LDX_WU /* ldx.wu */,

/* s w s */ STX_D /* stx.d */,
/* s w l */ LDX_D /* ldx.d */,
/* s b s */ STX_B /* stx.b */,
/* s b l */ LDX_B /* ldx.b */,
/* s h s */ STX_H /* stx.h */,
/* s h l */ LDX_H /* ldx.h */,
/* s i s */ STX_W /* stx.w */,
/* s i l */ LDX_W /* ldx.w */,

/* d   s */ FSTX_D /* fstx.d */,
/* d   l */ FLDX_D /* fldx.d */,
/* s   s */ FSTX_S /* fstx.s */,
/* s   l */ FLDX_S /* fldx.s */,
};

static sljit_s32 push_mem_inst(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	sljit_ins ins;
	sljit_s32 base = arg & REG_MASK;

	SLJIT_ASSERT(arg & SLJIT_MEM);

	if (arg & OFFS_REG_MASK) {
		sljit_s32 offs = OFFS_REG(arg);

		SLJIT_ASSERT(!argw);
		ins = data_transfer_insts_x[flags & MEM_MASK] |
			  ((flags & MEM_MASK) <= GPR_REG ? RD(reg) : FRD(reg)) |
			  RJ(base) | RK(offs);
	} else {
		SLJIT_ASSERT(argw <= 0xfff && argw >= I12_MIN);

		ins = data_transfer_insts[flags & MEM_MASK] |
			  ((flags & MEM_MASK) <= GPR_REG ? RD(reg) : FRD(reg)) |
			  RJ(base) | IMM_I12(argw);
	}
	return push_inst(compiler, ins);
}

/* Can perform an operation using at most 1 instruction. */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw)
{
	SLJIT_ASSERT(arg & SLJIT_MEM);

	/* argw == 0 (ldx/stx rd, rj, rk) can be used.
	 * argw in [-2048, 2047] (ld/st rd, rj, imm) can be used. */
	if (!argw || (!(arg & OFFS_REG_MASK) && (argw <= I12_MAX && argw >= I12_MIN))) {
		/* Works for both absolute and relative addresses. */
		if (SLJIT_UNLIKELY(flags & ARG_TEST))
			return 1;

		FAIL_IF(push_mem_inst(compiler, flags, reg, arg, argw));
		return -1;
	}
	return 0;
}

#define TO_ARGW_HI(argw) (((argw) & ~0xfff) + (((argw) & 0x800) ? 0x1000 : 0))

/* See getput_arg below.
   Note: can_cache is called only for binary operators. */
static sljit_s32 can_cache(sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	SLJIT_ASSERT((arg & SLJIT_MEM) && (next_arg & SLJIT_MEM));

	if (arg & OFFS_REG_MASK)
		return 0;

	if (arg == next_arg) {
		if (((next_argw - argw) <= I12_MAX && (next_argw - argw) >= I12_MIN)
				|| TO_ARGW_HI(argw) == TO_ARGW_HI(next_argw))
			return 1;
		return 0;
	}

	return 0;
}

/* Emit the necessary instructions. See can_cache above. */
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 base = arg & REG_MASK;
	sljit_s32 tmp_r = TMP_REG1;
	sljit_sw offset;

	SLJIT_ASSERT(arg & SLJIT_MEM);
	if (!(next_arg & SLJIT_MEM)) {
		next_arg = 0;
		next_argw = 0;
	}

	/* Since tmp can be the same as base or offset registers,
	 * these might be unavailable after modifying tmp. */
	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA))
		tmp_r = reg;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if (SLJIT_UNLIKELY(argw))
			FAIL_IF(push_inst(compiler, SLLI_D | RD(TMP_REG3) | RJ(OFFS_REG(arg)) | IMM_I12(argw)));
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM2(base, TMP_REG3), 0);
	}

	if (compiler->cache_arg == arg && argw - compiler->cache_argw <= I12_MAX && argw - compiler->cache_argw >= I12_MIN)
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM1(TMP_REG3), argw - compiler->cache_argw);

	if (compiler->cache_arg == SLJIT_MEM && (argw - compiler->cache_argw <= I12_MAX) && (argw - compiler->cache_argw >= I12_MIN)) {
		offset = argw - compiler->cache_argw;
	} else {
		sljit_sw argw_hi=TO_ARGW_HI(argw);
		compiler->cache_arg = SLJIT_MEM;

		if (next_arg && next_argw - argw <= I12_MAX && next_argw - argw >= I12_MIN && argw_hi != TO_ARGW_HI(next_argw)) {
			FAIL_IF(load_immediate(compiler, TMP_REG3, argw));
			compiler->cache_argw = argw;
			offset = 0;
		} else {
			FAIL_IF(load_immediate(compiler, TMP_REG3, argw_hi));
			compiler->cache_argw = argw_hi;
			offset = argw & 0xfff;
			argw = argw_hi;
		}
	}

	if (!base)
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM1(TMP_REG3), offset);

	if (arg == next_arg && next_argw - argw <= I12_MAX && next_argw - argw >= I12_MIN) {
		compiler->cache_arg = arg;
		FAIL_IF(push_inst(compiler, ADD_D | RD(TMP_REG3) | RJ(TMP_REG3) | RK(base)));
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM1(TMP_REG3), offset);
	}

	if (!offset)
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM2(base, TMP_REG3), 0);

	FAIL_IF(push_inst(compiler, ADD_D | RD(tmp_r) | RJ(TMP_REG3) | RK(base)));
	return push_mem_inst(compiler, flags, reg, SLJIT_MEM1(tmp_r), offset);
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

		if (SLJIT_UNLIKELY(argw))
			FAIL_IF(push_inst(compiler, SLLI_D | RD(tmp_r) | RJ(OFFS_REG(arg)) | IMM_I12(argw)));
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM2(base, tmp_r), 0);
	} else {
		FAIL_IF(load_immediate(compiler, tmp_r, argw));

		if (base != 0)
			return push_mem_inst(compiler, flags, reg, SLJIT_MEM2(base, tmp_r), 0);
		return push_mem_inst(compiler, flags, reg, SLJIT_MEM1(tmp_r), 0);
	}
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

#define IMM_EXTEND(v) (IMM_I12((op & SLJIT_32) ? (v) : (32 + (v))))

/* andi/ori/xori are zero-extended */
#define EMIT_LOGICAL(op_imm, op_reg) \
	if (flags & SRC2_IMM) { \
		if (op & SLJIT_SET_Z) {\
			FAIL_IF(push_inst(compiler, ADDI_D | RD(EQUAL_FLAG) | RJ(TMP_ZERO) | IMM_I12(src2))); \
			FAIL_IF(push_inst(compiler, op_reg | RD(EQUAL_FLAG) | RJ(src1) | RK(EQUAL_FLAG))); \
		} \
		if (!(flags & UNUSED_DEST)) { \
			if (dst == src1) { \
				FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG1) | RJ(TMP_ZERO) | IMM_I12(src2))); \
				FAIL_IF(push_inst(compiler, op_reg | RD(dst) | RJ(src1) | RK(TMP_REG1))); \
			} else { \
				FAIL_IF(push_inst(compiler, ADDI_D | RD(dst) | RJ(TMP_ZERO) | IMM_I12(src2))); \
				FAIL_IF(push_inst(compiler, op_reg | RD(dst) | RJ(src1) | RK(dst))); \
			} \
		} \
	} \
	else { \
		if (op & SLJIT_SET_Z) \
			FAIL_IF(push_inst(compiler, op_reg | RD(EQUAL_FLAG) | RJ(src1) | RK(src2))); \
		if (!(flags & UNUSED_DEST)) \
			FAIL_IF(push_inst(compiler, op_reg | RD(dst) | RJ(src1) | RK(src2))); \
	} \
	while (0)

#define EMIT_SHIFT(imm, reg) \
	op_imm = (imm); \
	op_reg = (reg)

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_s32 is_overflow, is_carry, carry_src_r, is_handled;
	sljit_ins op_imm, op_reg;
	sljit_ins word_size = ((op & SLJIT_32) ? 32 : 64);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return push_inst(compiler, INST(ADD, op) | RD(dst) | RJ(src2) | IMM_I12(0));
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, ANDI | RD(dst) | RJ(src2) | IMM_I12(0xff));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, EXT_W_B | RD(dst) | RJ(src2));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, INST(BSTRPICK, op) | RD(dst) | RJ(src2) | (15 << 16));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, EXT_W_H | RD(dst) | RJ(src2));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, BSTRPICK_D | RD(dst) | RJ(src2) | (31 << 16));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE))
			return push_inst(compiler, SLLI_W | RD(dst) | RJ(src2) | IMM_I12(0));
		SLJIT_ASSERT(dst == src2);
		return SLJIT_SUCCESS;

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return push_inst(compiler, INST(CLZ, op) | RD(dst) | RJ(src2));

	case SLJIT_CTZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return push_inst(compiler, INST(CTZ, op) | RD(dst) | RJ(src2));

	case SLJIT_REV:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		return push_inst(compiler, ((op & SLJIT_32) ? REVB_2W : REVB_D) | RD(dst) | RJ(src2));

	case SLJIT_REV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		FAIL_IF(push_inst(compiler, REVB_2H | RD(dst) | RJ(src2)));
		return push_inst(compiler, EXT_W_H | RD(dst) | RJ(dst));

	case SLJIT_REV_U16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		FAIL_IF(push_inst(compiler, REVB_2H | RD(dst) | RJ(src2)));
		return push_inst(compiler, INST(BSTRPICK, op) | RD(dst) | RJ(dst) | (15 << 16));

	case SLJIT_REV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM) && dst != TMP_REG1);
		FAIL_IF(push_inst(compiler, REVB_2W | RD(dst) | RJ(src2)));
		return push_inst(compiler, SLLI_W | RD(dst) | RJ(dst) | IMM_I12(0));

	case SLJIT_REV_U32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM) && dst != TMP_REG1);
		FAIL_IF(push_inst(compiler, REVB_2W | RD(dst) | RJ(src2)));
		return push_inst(compiler, BSTRPICK_D | RD(dst) | RJ(dst) | (31 << 16));

	case SLJIT_ADD:
		/* Overflow computation (both add and sub): overflow = src1_sign ^ src2_sign ^ result_sign ^ carry_flag */
		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		carry_src_r = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(0)));
				else {
					FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(TMP_ZERO) | IMM_I12(-1)));
					FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RJ(src1) | RK(EQUAL_FLAG)));
				}
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(src2)));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(dst) | RJ(src1) | IMM_I12(src2)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, INST(ADD, op) | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));

			if (is_overflow || carry_src_r != 0) {
				if (src1 != dst)
					carry_src_r = (sljit_s32)src1;
				else if (src2 != dst)
					carry_src_r = (sljit_s32)src2;
				else {
					FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(OTHER_FLAG) | RJ(src1) | IMM_I12(0)));
					carry_src_r = OTHER_FLAG;
				}
			}

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, INST(ADD, op) | RD(dst) | RJ(src1) | RK(src2)));
		}

		/* Carry is zero if a + b >= a or a + b >= b, otherwise it is 1. */
		if (is_overflow || carry_src_r != 0) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLTUI | RD(OTHER_FLAG) | RJ(dst) | IMM_I12(src2)));
			else
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RJ(dst) | RK(carry_src_r)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, XOR | RD(TMP_REG1) | RJ(dst) | RK(EQUAL_FLAG)));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, INST(ADD, op) | RD(EQUAL_FLAG) | RJ(dst) | IMM_I12(0)));
		FAIL_IF(push_inst(compiler, INST(SRLI, op) | RD(TMP_REG1) | RJ(TMP_REG1) | IMM_EXTEND(31)));
		return push_inst(compiler, XOR | RD(OTHER_FLAG) | RJ(TMP_REG1) | RK(OTHER_FLAG));

	case SLJIT_ADDC:
		carry_src_r = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			FAIL_IF(push_inst(compiler, ADDI_D | RD(dst) | RJ(src1) | IMM_I12(src2)));
		} else {
			if (carry_src_r != 0) {
				if (src1 != dst)
					carry_src_r = (sljit_s32)src1;
				else if (src2 != dst)
					carry_src_r = (sljit_s32)src2;
				else {
					FAIL_IF(push_inst(compiler, ADDI_D | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(0)));
					carry_src_r = EQUAL_FLAG;
				}
			}

			FAIL_IF(push_inst(compiler, ADD_D | RD(dst) | RJ(src1) | RK(src2)));
		}

		/* Carry is zero if a + b >= a or a + b >= b, otherwise it is 1. */
		if (carry_src_r != 0) {
			if (flags & SRC2_IMM)
				FAIL_IF(push_inst(compiler, SLTUI | RD(EQUAL_FLAG) | RJ(dst) | IMM_I12(src2)));
			else
				FAIL_IF(push_inst(compiler, SLTU | RD(EQUAL_FLAG) | RJ(dst) | RK(carry_src_r)));
		}

		FAIL_IF(push_inst(compiler, ADD_D | RD(dst) | RJ(dst) | RK(OTHER_FLAG)));

		if (carry_src_r == 0)
			return SLJIT_SUCCESS;

		/* Set ULESS_FLAG (dst == 0) && (OTHER_FLAG == 1). */
		FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RJ(dst) | RK(OTHER_FLAG)));
		/* Set carry flag. */
		return push_inst(compiler, OR | RD(OTHER_FLAG) | RJ(OTHER_FLAG) | RK(EQUAL_FLAG));

	case SLJIT_SUB:
		if ((flags & SRC2_IMM) && src2 == I12_MIN) {
			FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG2) | RJ(TMP_ZERO) | IMM_I12(src2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_handled = 0;

		if (flags & SRC2_IMM) {
			if (GET_FLAG_TYPE(op) == SLJIT_LESS) {
				FAIL_IF(push_inst(compiler, SLTUI | RD(OTHER_FLAG) | RJ(src1) | IMM_I12(src2)));
				is_handled = 1;
			}
			else if (GET_FLAG_TYPE(op) == SLJIT_SIG_LESS) {
				FAIL_IF(push_inst(compiler, SLTI | RD(OTHER_FLAG) | RJ(src1) | IMM_I12(src2)));
				is_handled = 1;
			}
		}

		if (!is_handled && GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_SIG_LESS_EQUAL) {
			is_handled = 1;

			if (flags & SRC2_IMM) {
				FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG2) | RJ(TMP_ZERO) | IMM_I12(src2)));
				src2 = TMP_REG2;
				flags &= ~SRC2_IMM;
			}

			switch (GET_FLAG_TYPE(op)) {
			case SLJIT_LESS:
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RJ(src1) | RK(src2)));
				break;
			case SLJIT_GREATER:
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RJ(src2) | RK(src1)));
				break;
			case SLJIT_SIG_LESS:
				FAIL_IF(push_inst(compiler, SLT | RD(OTHER_FLAG) | RJ(src1) | RK(src2)));
				break;
			case SLJIT_SIG_GREATER:
				FAIL_IF(push_inst(compiler, SLT | RD(OTHER_FLAG) | RJ(src2) | RK(src1)));
				break;
			}
		}

		if (is_handled) {
			if (flags & SRC2_IMM) {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(-src2)));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, INST(ADDI, op) | RD(dst) | RJ(src1) | IMM_I12(-src2));
			}
			else {
				if (op & SLJIT_SET_Z)
					FAIL_IF(push_inst(compiler, INST(SUB, op) | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));
				if (!(flags & UNUSED_DEST))
					return push_inst(compiler, INST(SUB, op) | RD(dst) | RJ(src1) | RK(src2));
			}
			return SLJIT_SUCCESS;
		}

		is_overflow = GET_FLAG_TYPE(op) == SLJIT_OVERFLOW;
		is_carry = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			if (is_overflow) {
				if (src2 >= 0)
					FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(0)));
				else {
					FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(-1)));
					FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RJ(src1) | RK(EQUAL_FLAG)));
				}
			}
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(-src2)));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTUI | RD(OTHER_FLAG) | RJ(src1) | IMM_I12(src2)));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(dst) | RJ(src1) | IMM_I12(-src2)));
		}
		else {
			if (is_overflow)
				FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));
			else if (op & SLJIT_SET_Z)
				FAIL_IF(push_inst(compiler, INST(SUB, op) | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));

			if (is_overflow || is_carry)
				FAIL_IF(push_inst(compiler, SLTU | RD(OTHER_FLAG) | RJ(src1) | RK(src2)));

			/* Only the zero flag is needed. */
			if (!(flags & UNUSED_DEST) || (op & VARIABLE_FLAG_MASK))
				FAIL_IF(push_inst(compiler, INST(SUB, op) | RD(dst) | RJ(src1) | RK(src2)));
		}

		if (!is_overflow)
			return SLJIT_SUCCESS;

		FAIL_IF(push_inst(compiler, XOR | RD(TMP_REG1) | RJ(dst) | RK(EQUAL_FLAG)));
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(EQUAL_FLAG) | RJ(dst) | IMM_I12(0)));
		FAIL_IF(push_inst(compiler, INST(SRLI, op) | RD(TMP_REG1) | RJ(TMP_REG1) | IMM_EXTEND(31)));
		return push_inst(compiler, XOR | RD(OTHER_FLAG) | RJ(TMP_REG1) | RK(OTHER_FLAG));

	case SLJIT_SUBC:
		if ((flags & SRC2_IMM) && src2 == I12_MIN) {
			FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(TMP_REG2) | RJ(TMP_ZERO) | IMM_I12(src2)));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		is_carry = GET_FLAG_TYPE(op) == SLJIT_CARRY;

		if (flags & SRC2_IMM) {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTUI | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(src2)));

			FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(dst) | RJ(src1) | IMM_I12(-src2)));
		}
		else {
			if (is_carry)
				FAIL_IF(push_inst(compiler, SLTU | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));

			FAIL_IF(push_inst(compiler, INST(SUB, op) | RD(dst) | RJ(src1) | RK(src2)));
		}

		if (is_carry)
			FAIL_IF(push_inst(compiler, SLTU | RD(TMP_REG1) | RJ(dst) | RK(OTHER_FLAG)));

		FAIL_IF(push_inst(compiler, INST(SUB, op) | RD(dst) | RJ(dst) | RK(OTHER_FLAG)));

		if (!is_carry)
			return SLJIT_SUCCESS;

		return push_inst(compiler, OR | RD(OTHER_FLAG) | RJ(EQUAL_FLAG) | RK(TMP_REG1));

	case SLJIT_MUL:
		SLJIT_ASSERT(!(flags & SRC2_IMM));

		if (GET_FLAG_TYPE(op) != SLJIT_OVERFLOW)
			return push_inst(compiler, INST(MUL, op) | RD(dst) | RJ(src1) | RK(src2));

		if (op & SLJIT_32) {
			FAIL_IF(push_inst(compiler, MUL_D | RD(OTHER_FLAG) | RJ(src1) | RK(src2)));
			FAIL_IF(push_inst(compiler, MUL_W | RD(dst) | RJ(src1) | RK(src2)));
			return push_inst(compiler, SUB_D | RD(OTHER_FLAG) | RJ(dst) | RK(OTHER_FLAG));
		}

		FAIL_IF(push_inst(compiler, MULH_D | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));
		FAIL_IF(push_inst(compiler, MUL_D | RD(dst) | RJ(src1) | RK(src2)));
		FAIL_IF(push_inst(compiler, SRAI_D | RD(OTHER_FLAG) | RJ(dst) | IMM_I12((63))));
		return push_inst(compiler, SUB_D | RD(OTHER_FLAG) | RJ(EQUAL_FLAG) | RK(OTHER_FLAG));

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
	case SLJIT_MSHL:
		if (op & SLJIT_32) {
			EMIT_SHIFT(SLLI_W, SLL_W);
		} else {
			EMIT_SHIFT(SLLI_D, SLL_D);
		}
		break;

	case SLJIT_LSHR:
	case SLJIT_MLSHR:
		if (op & SLJIT_32) {
			EMIT_SHIFT(SRLI_W, SRL_W);
		} else {
			EMIT_SHIFT(SRLI_D, SRL_D);
		}
		break;

	case SLJIT_ASHR:
	case SLJIT_MASHR:
		if (op & SLJIT_32) {
			EMIT_SHIFT(SRAI_W, SRA_W);
		} else {
			EMIT_SHIFT(SRAI_D, SRA_D);
		}
		break;

	case SLJIT_ROTL:
	case SLJIT_ROTR:
		if (flags & SRC2_IMM) {
			SLJIT_ASSERT(src2 != 0);

			if (GET_OPCODE(op) == SLJIT_ROTL)
				src2 = word_size - src2;
			return push_inst(compiler, INST(ROTRI, op) | RD(dst) | RJ(src1) | IMM_I12(src2));

		}

		if (src2 == TMP_ZERO) {
			if (dst != src1)
				return push_inst(compiler, INST(ADDI, op) | RD(dst) | RJ(src1) | IMM_I12(0));
			return SLJIT_SUCCESS;
		}

		if (GET_OPCODE(op) == SLJIT_ROTL) {
			FAIL_IF(push_inst(compiler, INST(SUB, op)| RD(OTHER_FLAG) | RJ(TMP_ZERO) | RK(src2)));
			src2 = OTHER_FLAG;
		}
		return push_inst(compiler, INST(ROTR, op) | RD(dst) | RJ(src1) | RK(src2));

	default:
		SLJIT_UNREACHABLE();
		return SLJIT_SUCCESS;
	}

	if (flags & SRC2_IMM) {
		if (op & SLJIT_SET_Z)
			FAIL_IF(push_inst(compiler, op_imm | RD(EQUAL_FLAG) | RJ(src1) | IMM_I12(src2)));

		if (flags & UNUSED_DEST)
			return SLJIT_SUCCESS;
		return push_inst(compiler, op_imm | RD(dst) | RJ(src1) | IMM_I12(src2));
	}

	if (op & SLJIT_SET_Z)
		FAIL_IF(push_inst(compiler, op_reg | RD(EQUAL_FLAG) | RJ(src1) | RK(src2)));

	if (flags & UNUSED_DEST)
		return SLJIT_SUCCESS;
	return push_inst(compiler, op_reg | RD(dst) | RJ(src1) | RK(src2));
}

#undef IMM_EXTEND

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* arg1 goes to TMP_REG1 or src reg
	   arg2 goes to TMP_REG2, imm or src reg
	   TMP_REG3 can be used for caching
	   result goes to TMP_REG2, so put result can use TMP_REG1 and TMP_REG3. */
	sljit_s32 dst_r = TMP_REG2;
	sljit_s32 src1_r;
	sljit_sw src2_r = 0;
	sljit_s32 sugg_src2_r = TMP_REG2;

	if (!(flags & ALT_KEEP_CACHE)) {
		compiler->cache_arg = 0;
		compiler->cache_argw = 0;
	}

	if (dst == 0) {
		SLJIT_ASSERT(HAS_FLAGS(op));
		flags |= UNUSED_DEST;
		dst = TMP_REG2;
	}
	else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (flags & MOVE_OP)
			sugg_src2_r = dst_r;
	}
	else if ((dst & SLJIT_MEM) && !getput_arg_fast(compiler, flags | ARG_TEST, TMP_REG1, dst, dstw))
		flags |= SLOW_DEST;

	if (flags & IMM_OP) {
		if (src2 == SLJIT_IMM && src2w != 0 && src2w <= I12_MAX && src2w >= I12_MIN) {
			flags |= SRC2_IMM;
			src2_r = src2w;
		}
		else if ((flags & CUMULATIVE_OP) && src1 == SLJIT_IMM && src1w != 0 && src1w <= I12_MAX && src1w >= I12_MIN) {
			flags |= SRC2_IMM;
			src2_r = src1w;

			/* And swap arguments. */
			src1 = src2;
			src1w = src2w;
			src2 = SLJIT_IMM;
			/* src2w = src2_r unneeded. */
		}
	}

	/* Source 1. */
	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	}
	else if (src1 == SLJIT_IMM) {
		if (src1w) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, src1w));
			src1_r = TMP_REG1;
		}
		else
			src1_r = TMP_ZERO;
	}
	else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC1;
		src1_r = TMP_REG1;
	}

	/* Source 2. */
	if (FAST_IS_REG(src2)) {
		src2_r = src2;
		flags |= REG2_SOURCE;
		if ((flags & (REG_DEST | MOVE_OP)) == MOVE_OP)
			dst_r = (sljit_s32)src2_r;
	}
	else if (src2 == SLJIT_IMM) {
		if (!(flags & SRC2_IMM)) {
			if (src2w) {
				FAIL_IF(load_immediate(compiler, sugg_src2_r, src2w));
				src2_r = sugg_src2_r;
			}
			else {
				src2_r = TMP_ZERO;
				if (flags & MOVE_OP) {
					if (dst & SLJIT_MEM)
						dst_r = 0;
					else
						op = SLJIT_MOV;
				}
			}
		}
	}
	else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, sugg_src2_r, src2, src2w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC2;

		src2_r = sugg_src2_r;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		SLJIT_ASSERT(src2_r == TMP_REG2);
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2, src2, src2w, dst, dstw));
		}
	}
	else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1, src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, sugg_src2_r, src2, src2w, dst, dstw));

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (dst & SLJIT_MEM) {
		if (!(flags & SLOW_DEST)) {
			getput_arg_fast(compiler, flags, dst_r, dst, dstw);
			return compiler->error;
		}
		return getput_arg(compiler, flags, dst_r, dst, dstw, 0, 0);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	switch (GET_OPCODE(op)) {
	case SLJIT_BREAKPOINT:
		return push_inst(compiler, BREAK);
	case SLJIT_NOP:
		return push_inst(compiler, ANDI | RD(TMP_ZERO) | RJ(TMP_ZERO) | IMM_I12(0));
	case SLJIT_LMUL_UW:
		FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG1) | RJ(SLJIT_R1) | IMM_I12(0)));
		FAIL_IF(push_inst(compiler, MULH_DU | RD(SLJIT_R1) | RJ(SLJIT_R0) | RK(SLJIT_R1)));
		return push_inst(compiler, MUL_D | RD(SLJIT_R0) | RJ(SLJIT_R0) | RK(TMP_REG1));
	case SLJIT_LMUL_SW:
		FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG1) | RJ(SLJIT_R1) | IMM_I12(0)));
		FAIL_IF(push_inst(compiler, MULH_D | RD(SLJIT_R1) | RJ(SLJIT_R0) | RK(SLJIT_R1)));
		return push_inst(compiler, MUL_D | RD(SLJIT_R0) | RJ(SLJIT_R0) | RK(TMP_REG1));
	case SLJIT_DIVMOD_UW:
		FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(TMP_REG1) | RJ(SLJIT_R0) | IMM_I12(0)));
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32)? DIV_WU: DIV_DU) | RD(SLJIT_R0) | RJ(SLJIT_R0) | RK(SLJIT_R1)));
		return push_inst(compiler, ((op & SLJIT_32)? MOD_WU: MOD_DU) | RD(SLJIT_R1) | RJ(TMP_REG1) | RK(SLJIT_R1));
	case SLJIT_DIVMOD_SW:
		FAIL_IF(push_inst(compiler, INST(ADDI, op) | RD(TMP_REG1) | RJ(SLJIT_R0) | IMM_I12(0)));
		FAIL_IF(push_inst(compiler, INST(DIV, op) | RD(SLJIT_R0) | RJ(SLJIT_R0) | RK(SLJIT_R1)));
		return push_inst(compiler, INST(MOD, op) | RD(SLJIT_R1) | RJ(TMP_REG1) | RK(SLJIT_R1));
	case SLJIT_DIV_UW:
		return push_inst(compiler, ((op & SLJIT_32)? DIV_WU: DIV_DU) | RD(SLJIT_R0) | RJ(SLJIT_R0) | RK(SLJIT_R1));
	case SLJIT_DIV_SW:
		return push_inst(compiler, INST(DIV, op) | RD(SLJIT_R0) | RJ(SLJIT_R0) | RK(SLJIT_R1));
	case SLJIT_ENDBR:
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return SLJIT_SUCCESS;
	}

	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
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
		return emit_op(compiler, SLJIT_MOV, WORD_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_U32:
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u32)srcw : srcw);

	case SLJIT_MOV_S32:
	/* Logical operators have no W variant, so sign extended input is necessary for them. */
	case SLJIT_MOV32:
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s32)srcw : srcw);

	case SLJIT_MOV_U8:
		return emit_op(compiler, op, BYTE_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u8)srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, op, BYTE_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s8)srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, op, HALF_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_u16)srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, op, HALF_DATA | SIGNED_DATA | MOVE_OP, dst, dstw, TMP_REG1, 0, src, (src == SLJIT_IMM) ? (sljit_s16)srcw : srcw);

	case SLJIT_CLZ:
	case SLJIT_CTZ:
	case SLJIT_REV:
		return emit_op(compiler, op, flags, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_REV_U16:
	case SLJIT_REV_S16:
		return emit_op(compiler, op, HALF_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		return emit_op(compiler, op | SLJIT_32, INT_DATA, dst, dstw, TMP_REG1, 0, src, srcw);
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

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
	sljit_s32 is_left;
	sljit_ins ins1, ins2, ins3;
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

	if (src3 == SLJIT_IMM) {
		src3w &= bit_length - 1;

		if (src3w == 0)
			return SLJIT_SUCCESS;

		if (is_left) {
			ins1 = INST(SLLI, op) | IMM_I12(src3w);
			src3w = bit_length - src3w;
			ins2 = INST(SRLI, op) | IMM_I12(src3w);
		} else {
			ins1 = INST(SRLI, op) | IMM_I12(src3w);
			src3w = bit_length - src3w;
			ins2 = INST(SLLI, op) | IMM_I12(src3w);
		}

		FAIL_IF(push_inst(compiler, ins1 | RD(dst_reg) | RJ(src1_reg)));
		FAIL_IF(push_inst(compiler, ins2 | RD(TMP_REG1) | RJ(src2_reg)));
		return push_inst(compiler, OR | RD(dst_reg) | RJ(dst_reg) | RK(TMP_REG1));
	}

	if (src3 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, TMP_REG2, src3, src3w));
		src3 = TMP_REG2;
	} else if (dst_reg == src3) {
		push_inst(compiler, INST(ADDI, op) | RD(TMP_REG2) | RJ(src3) | IMM_I12(0));
		src3 = TMP_REG2;
	}

	if (is_left) {
		ins1 = INST(SLL, op);
		ins2 = INST(SRLI, op);
		ins3 = INST(SRL, op);
	} else {
		ins1 = INST(SRL, op);
		ins2 = INST(SLLI, op);
		ins3 = INST(SLL, op);
	}

	FAIL_IF(push_inst(compiler, ins1 | RD(dst_reg) | RJ(src1_reg) | RK(src3)));

	if (!(op & SLJIT_SHIFT_INTO_NON_ZERO)) {
		FAIL_IF(push_inst(compiler, ins2 | RD(TMP_REG1) | RJ(src2_reg) | IMM_I12(1)));
		FAIL_IF(push_inst(compiler, XORI | RD(TMP_REG2) | RJ(src3) | IMM_I12((sljit_ins)bit_length - 1)));
		src2_reg = TMP_REG1;
	} else
		FAIL_IF(push_inst(compiler, INST(SUB, op) | RD(TMP_REG2) | RJ(TMP_ZERO) | RK(src3)));

	FAIL_IF(push_inst(compiler, ins3 | RD(TMP_REG1) | RJ(src2_reg) | RK(TMP_REG2)));
	return push_inst(compiler, OR | RD(dst_reg) | RJ(dst_reg) | RK(TMP_REG1));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 base = src & REG_MASK;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		if (FAST_IS_REG(src))
			FAIL_IF(push_inst(compiler, ADDI_D | RD(RETURN_ADDR_REG) | RJ(src) | IMM_I12(0)));
		else
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, RETURN_ADDR_REG, src, srcw));

		return push_inst(compiler, JIRL | RD(TMP_ZERO) | RJ(RETURN_ADDR_REG) | IMM_I12(0));
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		if (SLJIT_UNLIKELY(src & OFFS_REG_MASK)) {
			srcw &= 0x3;
			if (SLJIT_UNLIKELY(srcw))
				FAIL_IF(push_inst(compiler, SLLI_D | RD(TMP_REG1) | RJ(OFFS_REG(src)) | IMM_I12(srcw)));
			FAIL_IF(push_inst(compiler, ADD_D | RD(TMP_REG1) | RJ(base) | RK(TMP_REG1)));
		} else {
			if (base && srcw <= I12_MAX && srcw >= I12_MIN)
				return push_inst(compiler,PRELD | RJ(base) | IMM_I12(srcw));

			FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
			if (base != 0)
				FAIL_IF(push_inst(compiler, ADD_D | RD(TMP_REG1) | RJ(base) | RK(TMP_REG1)));
		}
		return push_inst(compiler, PRELD | RD(0) | RJ(TMP_REG1));
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
			return push_inst(compiler, ADDI_D | RD(dst) | RJ(RETURN_ADDR_REG) | IMM_I12(0));

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

	if (type != SLJIT_FLOAT_REGISTER)
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
#define SET_COND(cond) (sljit_ins)(cond << 15)

#define COND_CUN SET_COND(0x8)	 /* UN */
#define COND_CEQ SET_COND(0x4)	 /* EQ */
#define COND_CUEQ SET_COND(0xc)	 /* UN EQ */
#define COND_CLT SET_COND(0x2)	 /* LT */
#define COND_CULT SET_COND(0xa)	 /* UN LT */
#define COND_CLE SET_COND(0x6)	 /* LT EQ */
#define COND_CULE SET_COND(0xe)	 /* UN LT EQ */
#define COND_CNE SET_COND(0x10)	 /* GT LT */
#define COND_CUNE SET_COND(0x18) /* UN GT LT */
#define COND_COR SET_COND(0x14)	 /* GT LT EQ */

#define FINST(inst, type) (sljit_ins)((type & SLJIT_32) ? inst##_S : inst##_D)
#define FCD(cd) (sljit_ins)(cd & 0x7)
#define FCJ(cj) (sljit_ins)((cj & 0x7) << 5)
#define FCA(ca) (sljit_ins)((ca & 0x7) << 15)
#define F_OTHER_FLAG 1

#define FLOAT_DATA(op) (DOUBLE_DATA | ((op & SLJIT_32) >> 7))

/* convert to inter exact toward zero */
static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins inst;
	sljit_u32 word_data = 0;
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;

	switch (GET_OPCODE(op))
	{
	case SLJIT_CONV_SW_FROM_F64:
		word_data = 1;
		inst = FINST(FTINTRZ_L, op);
		break;
	case SLJIT_CONV_S32_FROM_F64:
		inst = FINST(FTINTRZ_W, op);
		break;
	default:
		inst = BREAK;
		SLJIT_UNREACHABLE();
	}

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src, srcw, dst, dstw));
		src = TMP_FREG1;
	}

	FAIL_IF(push_inst(compiler, inst | FRD(TMP_FREG1) | FRJ(src)));
	FAIL_IF(push_inst(compiler, FINST(MOVFR2GR, word_data) | RD(dst_r) | FRJ(TMP_FREG1)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, word_data ? WORD_DATA : INT_DATA, TMP_REG2, dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_w(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins inst;
	sljit_u32 word_data = 0;
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	switch (GET_OPCODE(op))
	{
	case SLJIT_CONV_F64_FROM_SW:
		word_data = 1;
		inst = (sljit_ins)((op & SLJIT_32) ? FFINT_S_L : FFINT_D_L);
		break;
	case SLJIT_CONV_F64_FROM_S32:
		inst = (sljit_ins)((op & SLJIT_32) ? FFINT_S_W : FFINT_D_W);
		break;
	default:
		inst = BREAK;
		SLJIT_UNREACHABLE();
	}

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, (word_data ? WORD_DATA : INT_DATA) | LOAD_DATA, TMP_REG1, src, srcw, dst, dstw));
		src = TMP_REG1;
	} else if (src == SLJIT_IMM) {
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32)
			srcw = (sljit_s32)srcw;

		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
	}
	FAIL_IF(push_inst(compiler, (word_data ? MOVGR2FR_D : MOVGR2FR_W) | FRD(dst_r) | RJ(src)));
	FAIL_IF(push_inst(compiler, inst | FRD(dst_r) | FRJ(dst_r)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG1, dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	return sljit_emit_fop1_conv_f64_from_w(compiler, op, dst, dstw, src, srcw);
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_uw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins inst;
	sljit_u32 word_data = 0;
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	switch (GET_OPCODE(op))
	{
	case SLJIT_CONV_F64_FROM_UW:
		word_data = 1;
		inst = (sljit_ins)((op & SLJIT_32) ? FFINT_S_L : FFINT_D_L);
		break;
	case SLJIT_CONV_F64_FROM_U32:
		inst = (sljit_ins)((op & SLJIT_32) ? FFINT_S_W : FFINT_D_W);
		break;
	default:
		inst = BREAK;
		SLJIT_UNREACHABLE();
	}

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, (word_data ? WORD_DATA : INT_DATA) | LOAD_DATA, TMP_REG1, src, srcw, dst, dstw));
		src = TMP_REG1;
	} else if (src == SLJIT_IMM) {
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_U32)
			srcw = (sljit_u32)srcw;

		FAIL_IF(load_immediate(compiler, TMP_REG1, srcw));
		src = TMP_REG1;
	}

	if (!word_data)
		FAIL_IF(push_inst(compiler, SRLI_W | RD(src) | RJ(src) | IMM_I12(0)));

	FAIL_IF(push_inst(compiler, BLT | RJ(src) | RD(TMP_ZERO) | IMM_I16(4)));

	FAIL_IF(push_inst(compiler, (word_data ? MOVGR2FR_D : MOVGR2FR_W) | FRD(dst_r) | RJ(src)));
	FAIL_IF(push_inst(compiler, inst | FRD(dst_r) | FRJ(dst_r)));
	FAIL_IF(push_inst(compiler, B | IMM_I26(7)));

	FAIL_IF(push_inst(compiler, ANDI | RD(TMP_REG2) | RJ(src) | IMM_I12(1)));
	FAIL_IF(push_inst(compiler, (word_data ? SRLI_D : SRLI_W) | RD(TMP_REG1) | RJ(src) | IMM_I12(1)));
	FAIL_IF(push_inst(compiler, OR | RD(TMP_REG1) | RJ(TMP_REG1) | RK(TMP_REG2)));
	FAIL_IF(push_inst(compiler, INST(MOVGR2FR, (!word_data)) | FRD(dst_r) | RJ(TMP_REG1)));
	FAIL_IF(push_inst(compiler, inst | FRD(dst_r) | FRJ(dst_r)));
	FAIL_IF(push_inst(compiler, FINST(FADD, op) | FRD(dst_r) | FRJ(dst_r) | FRK(dst_r)));

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG1, dst, dstw, 0, 0);
	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, src2, src2w));
		src1 = TMP_FREG1;
	}

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, 0, 0));
		src2 = TMP_FREG2;
	}

	FAIL_IF(push_inst(compiler, XOR | RD(OTHER_FLAG) | RJ(OTHER_FLAG) | RK(OTHER_FLAG)));

	switch (GET_FLAG_TYPE(op)) {
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CEQ | FCD(F_OTHER_FLAG) | FRJ(src1) | FRK(src2)));
		break;
	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CLT | FCD(F_OTHER_FLAG) | FRJ(src1) | FRK(src2)));
		break;
	case SLJIT_F_GREATER:
	case SLJIT_ORDERED_GREATER:
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CLT | FCD(F_OTHER_FLAG) | FRJ(src2) | FRK(src1)));
		break;
	case SLJIT_UNORDERED_OR_GREATER:
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CULT | FCD(F_OTHER_FLAG) | FRJ(src2) | FRK(src1)));
		break;
	case SLJIT_UNORDERED_OR_LESS:
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CULT | FCD(F_OTHER_FLAG) | FRJ(src1) | FRK(src2)));
		break;
	case SLJIT_UNORDERED_OR_EQUAL:
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CUEQ | FCD(F_OTHER_FLAG) | FRJ(src1) | FRK(src2)));
		break;
	default: /* SLJIT_UNORDERED */
		FAIL_IF(push_inst(compiler, FINST(FCMP_COND, op) | COND_CUN | FCD(F_OTHER_FLAG) | FRJ(src1) | FRK(src2)));
	}
	return push_inst(compiler, MOVCF2GR | RD(OTHER_FLAG) | FCJ(F_OTHER_FLAG));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;

	CHECK_ERROR();
	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	SLJIT_COMPILE_ASSERT((SLJIT_32 == 0x100) && !(DOUBLE_DATA & 0x2), float_transfer_bit_error);
	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
		op ^= SLJIT_32;

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (src & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, dst_r, src, srcw, dst, dstw));
		src = dst_r;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_F64:
		if (src != dst_r) {
			if (dst_r != TMP_FREG1)
				FAIL_IF(push_inst(compiler, FINST(FMOV, op) | FRD(dst_r) | FRJ(src)));
			else
				dst_r = src;
		}
		break;
	case SLJIT_NEG_F64:
		FAIL_IF(push_inst(compiler, FINST(FNEG, op) | FRD(dst_r) | FRJ(src)));
		break;
	case SLJIT_ABS_F64:
		FAIL_IF(push_inst(compiler, FINST(FABS, op) | FRD(dst_r) | FRJ(src)));
		break;
	case SLJIT_CONV_F64_FROM_F32:
		/* The SLJIT_32 bit is inverted because sljit_f32 needs to be loaded from the memory. */
		FAIL_IF(push_inst(compiler, ((op & SLJIT_32) ? FCVT_D_S : FCVT_S_D) | FRD(dst_r) | FRJ(src)));
		op ^= SLJIT_32;
		break;
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem2(compiler, FLOAT_DATA(op), dst_r, dst, dstw, 0, 0);
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

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

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
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, dst, dstw));
		}
		else {
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, dst, dstw));
		}
	}
	else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG2, src2, src2w, dst, dstw));

	if (flags & SLOW_SRC1)
		src1 = TMP_FREG1;
	if (flags & SLOW_SRC2)
		src2 = TMP_FREG2;

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		FAIL_IF(push_inst(compiler, FINST(FADD, op) | FRD(dst_r) | FRJ(src1) | FRK(src2)));
		break;
	case SLJIT_SUB_F64:
		FAIL_IF(push_inst(compiler, FINST(FSUB, op) | FRD(dst_r) | FRJ(src1) | FRK(src2)));
		break;
	case SLJIT_MUL_F64:
		FAIL_IF(push_inst(compiler, FINST(FMUL, op) | FRD(dst_r) | FRJ(src1) | FRK(src2)));
		break;
	case SLJIT_DIV_F64:
		FAIL_IF(push_inst(compiler, FINST(FDIV, op) | FRD(dst_r) | FRJ(src1) | FRK(src2)));
		break;
	}

	if (dst_r == TMP_FREG2)
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op), TMP_FREG2, dst, dstw, 0, 0));
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 reg;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2r(compiler, op, dst_freg, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	if (src2 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, TMP_FREG1, src2, src2w, 0, 0));
		src2 = TMP_FREG1;
	}

	if (src1 & SLJIT_MEM) {
		reg = (dst_freg == src2) ? TMP_FREG1 : dst_freg;
		FAIL_IF(emit_op_mem2(compiler, FLOAT_DATA(op) | LOAD_DATA, reg, src1, src1w, 0, 0));
		src1 = reg;
	}

	return push_inst(compiler, FINST(FCOPYSIGN, op) | FRD(dst_freg) | FRJ(src1) | FRK(src2));
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
		return push_inst(compiler, MOVGR2FR_W | RJ(TMP_ZERO) | FRD(freg));

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm));
	return push_inst(compiler, MOVGR2FR_W | RJ(TMP_REG1) | FRD(freg));
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
		return push_inst(compiler, MOVGR2FR_D | RJ(TMP_ZERO) | FRD(freg));

	FAIL_IF(load_immediate(compiler, TMP_REG1, u.imm));
	return push_inst(compiler, MOVGR2FR_D | RJ(TMP_REG1) | FRD(freg));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
	sljit_ins inst;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fcopy(compiler, op, freg, reg));

	if (GET_OPCODE(op) == SLJIT_COPY_TO_F64)
		inst = ((op & SLJIT_32) ? MOVGR2FR_W : MOVGR2FR_D) | FRD(freg) | RJ(reg);
	else
		inst = ((op & SLJIT_32) ? MOVFR2GR_S : MOVFR2GR_D) | RD(reg) | FRJ(freg);
	return push_inst(compiler, inst);
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

static sljit_ins get_jump_instruction(sljit_s32 type)
{
	switch (type) {
	case SLJIT_EQUAL:
	case SLJIT_ATOMIC_NOT_STORED:
		return BNE | RJ(EQUAL_FLAG) | RD(TMP_ZERO);
	case SLJIT_NOT_EQUAL:
	case SLJIT_ATOMIC_STORED:
		return BEQ | RJ(EQUAL_FLAG) | RD(TMP_ZERO);
	case SLJIT_LESS:
	case SLJIT_GREATER:
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER:
	case SLJIT_OVERFLOW:
	case SLJIT_CARRY:
		return BEQ | RJ(OTHER_FLAG) | RD(TMP_ZERO);
	case SLJIT_GREATER_EQUAL:
	case SLJIT_LESS_EQUAL:
	case SLJIT_SIG_GREATER_EQUAL:
	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_NOT_CARRY:
		return BNE | RJ(OTHER_FLAG) | RD(TMP_ZERO);
	case SLJIT_F_EQUAL:
	case SLJIT_ORDERED_EQUAL:
	case SLJIT_F_LESS:
	case SLJIT_ORDERED_LESS:
	case SLJIT_ORDERED_GREATER:
	case SLJIT_UNORDERED_OR_GREATER:
	case SLJIT_F_GREATER:
	case SLJIT_UNORDERED_OR_LESS:
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_UNORDERED:
		return BEQ | RJ(OTHER_FLAG) | RD(TMP_ZERO);
	case SLJIT_ORDERED_NOT_EQUAL:
	case SLJIT_ORDERED_LESS_EQUAL:
	case SLJIT_ORDERED_GREATER_EQUAL:
	case SLJIT_F_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_NOT_EQUAL:
	case SLJIT_UNORDERED_OR_GREATER_EQUAL:
	case SLJIT_UNORDERED_OR_LESS_EQUAL:
	case SLJIT_F_LESS_EQUAL:
	case SLJIT_F_GREATER_EQUAL:
	case SLJIT_ORDERED:
		return BNE | RJ(OTHER_FLAG) | RD(TMP_ZERO);
	default:
		/* Not conditional branch. */
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

	jump->addr = compiler->size;
	inst = JIRL | RJ(TMP_REG1) | IMM_I16(0);

	if (type >= SLJIT_FAST_CALL) {
		jump->flags |= IS_CALL;
		inst |= RD(RETURN_ADDR_REG);
	}

	PTR_FAIL_IF(push_inst(compiler, inst));

	/* Maximum number of instructions required for generating a constant. */
	compiler->size += 3;

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

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	struct sljit_jump *jump;
	sljit_s32 flags;
	sljit_ins inst;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_cmp(compiler, type, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;

	if (src1 & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem2(compiler, flags, TMP_REG1, src1, src1w, src2, src2w));
		src1 = TMP_REG1;
	}

	if (src2 & SLJIT_MEM) {
		PTR_FAIL_IF(emit_op_mem2(compiler, flags, TMP_REG2, src2, src2w, 0, 0));
		src2 = TMP_REG2;
	}

	if (src1 == SLJIT_IMM) {
		if (src1w != 0) {
			PTR_FAIL_IF(load_immediate(compiler, TMP_REG1, src1w));
			src1 = TMP_REG1;
		}
		else
			src1 = TMP_ZERO;
	}

	if (src2 == SLJIT_IMM) {
		if (src2w != 0) {
			PTR_FAIL_IF(load_immediate(compiler, TMP_REG2, src2w));
			src2 = TMP_REG2;
		}
		else
			src2 = TMP_ZERO;
	}

	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, (sljit_u32)((type & SLJIT_REWRITABLE_JUMP) | IS_COND));
	type &= 0xff;

	switch (type) {
	case SLJIT_EQUAL:
		inst = BNE | RJ(src1) | RD(src2);
		break;
	case SLJIT_NOT_EQUAL:
		inst = BEQ | RJ(src1) | RD(src2);
		break;
	case SLJIT_LESS:
		inst = BGEU | RJ(src1) | RD(src2);
		break;
	case SLJIT_GREATER_EQUAL:
		inst = BLTU | RJ(src1) | RD(src2);
		break;
	case SLJIT_GREATER:
		inst = BGEU | RJ(src2) | RD(src1);
		break;
	case SLJIT_LESS_EQUAL:
		inst = BLTU | RJ(src2) | RD(src1);
		break;
	case SLJIT_SIG_LESS:
		inst = BGE | RJ(src1) | RD(src2);
		break;
	case SLJIT_SIG_GREATER_EQUAL:
		inst = BLT | RJ(src1) | RD(src2);
		break;
	case SLJIT_SIG_GREATER:
		inst = BGE | RJ(src2) | RD(src1);
		break;
	case SLJIT_SIG_LESS_EQUAL:
		inst = BLT | RJ(src2) | RD(src1);
		break;
	default:
		inst = BREAK;
		SLJIT_UNREACHABLE();
	}

	PTR_FAIL_IF(push_inst(compiler, inst));

	jump->addr = compiler->size;
	PTR_FAIL_IF(push_inst(compiler, JIRL | RD(TMP_ZERO) | RJ(TMP_REG1) | IMM_I12(0)));

	/* Maximum number of instructions required for generating a constant. */
	compiler->size += 3;

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
			FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
			src = TMP_REG1;
		}
		return push_inst(compiler, JIRL | RD((type >= SLJIT_FAST_CALL) ? RETURN_ADDR_REG : TMP_ZERO) | RJ(src) | IMM_I12(0));
	}

	/* These jumps are converted to jump/call instructions when possible. */
	jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
	FAIL_IF(!jump);
	set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_CALL : 0));
	jump->u.target = (sljit_uw)srcw;

	jump->addr = compiler->size;
	FAIL_IF(push_inst(compiler, JIRL | RD((type >= SLJIT_FAST_CALL) ? RETURN_ADDR_REG : TMP_ZERO) | RJ(TMP_REG1) | IMM_I12(0)));

	/* Maximum number of instructions required for generating a constant. */
	compiler->size += 3;

	return SLJIT_SUCCESS;
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
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, TMP_REG1, src, srcw));
		src = TMP_REG1;
	}

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= (SLJIT_S0 - SLJIT_KEPT_SAVEDS_COUNT(compiler->options))) {
			FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG1) | RJ(src) | IMM_I12(0)));
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
	sljit_s32 src_r, dst_r, invert;
	sljit_s32 saved_op = op;
	sljit_s32 mem_type = ((op & SLJIT_32) || op == SLJIT_MOV32) ? (INT_DATA | SIGNED_DATA) : WORD_DATA;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	dst_r = (op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2;

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;

	if (op >= SLJIT_ADD && (dst & SLJIT_MEM))
		FAIL_IF(emit_op_mem2(compiler, mem_type | LOAD_DATA, TMP_REG1, dst, dstw, dst, dstw));

	if (type < SLJIT_F_EQUAL) {
		src_r = OTHER_FLAG;
		invert = type & 0x1;

		switch (type) {
		case SLJIT_EQUAL:
		case SLJIT_NOT_EQUAL:
			FAIL_IF(push_inst(compiler, SLTUI | RD(dst_r) | RJ(EQUAL_FLAG) | IMM_I12(1)));
			src_r = dst_r;
			break;
		case SLJIT_ATOMIC_STORED:
		case SLJIT_ATOMIC_NOT_STORED:
			FAIL_IF(push_inst(compiler, SLTUI | RD(dst_r) | RJ(EQUAL_FLAG) | IMM_I12(1)));
			src_r = dst_r;
			invert ^= 0x1;
			break;
		case SLJIT_OVERFLOW:
		case SLJIT_NOT_OVERFLOW:
			if (compiler->status_flags_state & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)) {
				src_r = OTHER_FLAG;
				break;
			}
			FAIL_IF(push_inst(compiler, SLTUI | RD(dst_r) | RJ(OTHER_FLAG) | IMM_I12(1)));
			src_r = dst_r;
			invert ^= 0x1;
			break;
		}
	} else {
		invert = 0;
		src_r = OTHER_FLAG;

		switch (type) {
		case SLJIT_ORDERED_NOT_EQUAL:
		case SLJIT_ORDERED_LESS_EQUAL:
		case SLJIT_ORDERED_GREATER_EQUAL:
		case SLJIT_F_NOT_EQUAL:
		case SLJIT_UNORDERED_OR_NOT_EQUAL:
		case SLJIT_UNORDERED_OR_GREATER_EQUAL:
		case SLJIT_UNORDERED_OR_LESS_EQUAL:
		case SLJIT_F_LESS_EQUAL:
		case SLJIT_F_GREATER_EQUAL:
		case SLJIT_ORDERED:
			invert = 1;
			break;
		}
	}

	if (invert) {
		FAIL_IF(push_inst(compiler, XORI | RD(dst_r) | RJ(src_r) | IMM_I12(1)));
		src_r = dst_r;
	}

	if (op < SLJIT_ADD) {
		if (dst & SLJIT_MEM)
			return emit_op_mem(compiler, mem_type, src_r, dst, dstw);

		if (src_r != dst_r)
			return push_inst(compiler, ADDI_D | RD(dst_r) | RJ(src_r) | IMM_I12(0));
		return SLJIT_SUCCESS;
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
	sljit_ins *ptr;
	sljit_uw size;
	sljit_s32 inp_flags = ((type & SLJIT_32) ? INT_DATA : WORD_DATA) | LOAD_DATA;

	CHECK_ERROR();
	CHECK(check_sljit_emit_select(compiler, type, dst_reg, src1, src1w, src2_reg));
	ADJUST_LOCAL_OFFSET(src1, src1w);

	if (dst_reg != src2_reg) {
		if (dst_reg == src1) {
			src1 = src2_reg;
			src1w = 0;
			type ^= 0x1;
		} else {
			if (ADDRESSING_DEPENDS_ON(src1, dst_reg)) {
				FAIL_IF(push_inst(compiler, ADDI_D | RD(TMP_REG2) | RJ(dst_reg) | IMM_I12(0)));

				if ((src1 & REG_MASK) == dst_reg)
					src1 = (src1 & ~REG_MASK) | TMP_REG2;

				if (OFFS_REG(src1) == dst_reg)
					src1 = (src1 & ~OFFS_REG_MASK) | TO_OFFS_REG(TMP_REG2);
			}

			FAIL_IF(push_inst(compiler, ADDI_D | RD(dst_reg) | RJ(src2_reg) | IMM_I12(0)));
		}
	}

	size = compiler->size;

	ptr = (sljit_ins*)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	compiler->size++;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, inp_flags, dst_reg, src1, src1w));
	} else if (src1 == SLJIT_IMM) {
		if (type & SLJIT_32)
			src1w = (sljit_s32)src1w;
		FAIL_IF(load_immediate(compiler, dst_reg, src1w));
	} else
		FAIL_IF(push_inst(compiler, ADDI_D | RD(dst_reg) | RJ(src1) | IMM_I12(0)));

	*ptr = get_jump_instruction(type & ~SLJIT_32) | IMM_I16(compiler->size - size);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
	sljit_s32 invert = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fselect(compiler, type, dst_freg, src1, src1w, src2_freg));

	ADJUST_LOCAL_OFFSET(src1, src1w);

	if ((type & ~SLJIT_32) == SLJIT_EQUAL || (type & ~SLJIT_32) == SLJIT_NOT_EQUAL) {
		if ((type & ~SLJIT_32) == SLJIT_EQUAL)
			invert = 1;
		FAIL_IF(push_inst(compiler, MOVGR2CF | FCD(F_OTHER_FLAG) | RJ(EQUAL_FLAG)));
	}
	else
		FAIL_IF(push_inst(compiler, MOVGR2CF | FCD(F_OTHER_FLAG) | RJ(OTHER_FLAG)));

	if (src1 & SLJIT_MEM) {
		FAIL_IF(emit_op_mem(compiler, FLOAT_DATA(type) | LOAD_DATA, dst_freg, src1, src1w));
		if (invert)
			return push_inst(compiler, FSEL | FRD(dst_freg) | FRJ(dst_freg) | FRK(src2_freg) | FCA(F_OTHER_FLAG));
		return push_inst(compiler, FSEL | FRD(dst_freg) | FRJ(src2_freg) | FRK(dst_freg) | FCA(F_OTHER_FLAG));
	} else {
		if (invert)
			return push_inst(compiler, FSEL | FRD(dst_freg) | FRJ(src1) | FRK(src2_freg) | FCA(F_OTHER_FLAG));
		return push_inst(compiler, FSEL | FRD(dst_freg) | FRJ(src2_freg) | FRK(src1) | FCA(F_OTHER_FLAG));
	}
}

#undef FLOAT_DATA

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	sljit_s32 flags;

	CHECK_ERROR();
	CHECK(check_sljit_emit_mem(compiler, type, reg, mem, memw));

	if (!(reg & REG_PAIR_MASK))
		return sljit_emit_mem_unaligned(compiler, type, reg, mem, memw);

	if (SLJIT_UNLIKELY(mem & OFFS_REG_MASK)) {
		memw &= 0x3;

		if (SLJIT_UNLIKELY(memw != 0)) {
			FAIL_IF(push_inst(compiler, SLLI_D | RD(TMP_REG1) | RJ(OFFS_REG(mem)) | IMM_I12(memw)));
			FAIL_IF(push_inst(compiler, ADD_D| RD(TMP_REG1) | RJ(TMP_REG1) | RK(mem & REG_MASK)));
		} else
			FAIL_IF(push_inst(compiler, ADD_D| RD(TMP_REG1) | RJ(mem & REG_MASK) | RK(OFFS_REG(mem))));

		mem = TMP_REG1;
		memw = 0;
	} else if (memw > I12_MAX - SSIZE_OF(sw) || memw < I12_MIN) {
		if (((memw + 0x800) & 0xfff) <= 0xfff - SSIZE_OF(sw)) {
			FAIL_IF(load_immediate(compiler, TMP_REG1, TO_ARGW_HI(memw)));
			memw &= 0xfff;
		} else {
			FAIL_IF(load_immediate(compiler, TMP_REG1, memw));
			memw = 0;
		}

		if (mem & REG_MASK)
			FAIL_IF(push_inst(compiler, ADD_D| RD(TMP_REG1) | RJ(TMP_REG1) | RK(mem & REG_MASK)));

		mem = TMP_REG1;
	} else {
		mem &= REG_MASK;
		memw &= 0xfff;
	}

	SLJIT_ASSERT((memw >= 0 && memw <= I12_MAX - SSIZE_OF(sw)) || (memw > I12_MAX && memw <= 0xfff));

	if (!(type & SLJIT_MEM_STORE) && mem == REG_PAIR_FIRST(reg)) {
		FAIL_IF(push_mem_inst(compiler, WORD_DATA | LOAD_DATA, REG_PAIR_SECOND(reg), SLJIT_MEM1(mem), (memw + SSIZE_OF(sw)) & 0xfff));
		return push_mem_inst(compiler, WORD_DATA | LOAD_DATA, REG_PAIR_FIRST(reg), SLJIT_MEM1(mem), memw);
	}

	flags = WORD_DATA | (!(type & SLJIT_MEM_STORE) ? LOAD_DATA : 0);

	FAIL_IF(push_mem_inst(compiler, flags, REG_PAIR_FIRST(reg), SLJIT_MEM1(mem), memw));
	return push_mem_inst(compiler, flags, REG_PAIR_SECOND(reg), SLJIT_MEM1(mem), (memw + SSIZE_OF(sw)) & 0xfff);
}

#undef TO_ARGW_HI

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler,
	sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	sljit_ins ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_load(compiler, op, dst_reg, mem_reg));

	if (!(LOONGARCH_FEATURE_LAMCAS & get_cpu_features()))
		return SLJIT_ERR_UNSUPPORTED;

	switch(GET_OPCODE(op)) {
	case SLJIT_MOV_U8:
		ins = LD_BU;
		break;
	case SLJIT_MOV_U16:
		ins = LD_HU;
		break;
	case SLJIT_MOV32:
		ins = LD_W;
		break;
	case SLJIT_MOV_U32:
		ins = LD_WU;
		break;
	default:
		ins = LD_D;
		break;
	}

	return push_inst(compiler, ins | RD(dst_reg) | RJ(mem_reg) | IMM_I12(0));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler,
	sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	sljit_ins ins = 0;
	sljit_ins unsign = 0;
	sljit_s32 tmp = temp_reg;

	CHECK_ERROR();
	CHECK(check_sljit_emit_atomic_store(compiler, op, src_reg, mem_reg, temp_reg));

	if (!(LOONGARCH_FEATURE_LAMCAS & get_cpu_features()))
		return SLJIT_ERR_UNSUPPORTED;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV_U8:
		ins = AMCAS_B;
		unsign = BSTRPICK_D | (7 << 16);
		break;
	case SLJIT_MOV_U16:
		ins = AMCAS_H;
		unsign = BSTRPICK_D | (15 << 16);
		break;
	case SLJIT_MOV32:
		ins = AMCAS_W;
		break;
	case SLJIT_MOV_U32:
		ins = AMCAS_W;
		unsign = BSTRPICK_D | (31 << 16);
		break;
	default:
		ins = AMCAS_D;
		break;
	}

	if (op & SLJIT_SET_ATOMIC_STORED) {
		FAIL_IF(push_inst(compiler, XOR | RD(TMP_REG1) | RJ(temp_reg) | RK(TMP_ZERO)));
		tmp = TMP_REG1;
	}
	FAIL_IF(push_inst(compiler, ins | RD(tmp) | RJ(mem_reg) | RK(src_reg)));
	if (!(op & SLJIT_SET_ATOMIC_STORED))
		return SLJIT_SUCCESS;

	if (unsign)
		FAIL_IF(push_inst(compiler, unsign | RD(tmp) | RJ(tmp)));

	FAIL_IF(push_inst(compiler, XOR | RD(EQUAL_FLAG) | RJ(tmp) | RK(temp_reg)));
	return push_inst(compiler, SLTUI | RD(EQUAL_FLAG) | RJ(EQUAL_FLAG) | IMM_I12(1));
}

static SLJIT_INLINE sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw init_value, sljit_ins last_ins)
{
	SLJIT_UNUSED_ARG(last_ins);

	FAIL_IF(push_inst(compiler, LU12I_W | RD(dst) | (sljit_ins)(((init_value & 0xffffffff) >> 12) << 5)));
	FAIL_IF(push_inst(compiler, LU32I_D | RD(dst) | (sljit_ins)(((init_value >> 32) & 0xfffff) << 5)));
	FAIL_IF(push_inst(compiler, LU52I_D | RD(dst) | RJ(dst) | (sljit_ins)(IMM_I12(init_value >> 52))));
	return push_inst(compiler, ORI | RD(dst) | RJ(dst) | IMM_I12(init_value));
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	sljit_ins *inst = (sljit_ins*)addr;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 4, 0);

	SLJIT_ASSERT((inst[0] & OPC_1RI20(0x7f)) == LU12I_W);
	inst[0] = (inst[0] & (OPC_1RI20(0x7f) | 0x1f)) | (sljit_ins)(((new_target & 0xffffffff) >> 12) << 5);

	SLJIT_ASSERT((inst[1] & OPC_1RI20(0x7f)) == LU32I_D);
	inst[1] = (inst[1] & (OPC_1RI20(0x7f) | 0x1f)) | (sljit_ins)(sljit_ins)(((new_target >> 32) & 0xfffff) << 5);

	SLJIT_ASSERT((inst[2] & OPC_2RI12(0x3ff)) == LU52I_D);
	inst[2] = (inst[2] & (OPC_2RI12(0x3ff) | 0x3ff)) | IMM_I12(new_target >> 52);

	SLJIT_ASSERT((inst[3] & OPC_2RI12(0x3ff)) == ORI || (inst[3] & OPC_2RI16(0x3f)) == JIRL);
	if ((inst[3] & OPC_2RI12(0x3ff)) == ORI)
		inst[3] = (inst[3] & (OPC_2RI12(0x3ff) | 0x3ff)) | IMM_I12(new_target);
	else
		inst[3] = (inst[3] & (OPC_2RI16(0x3f) | 0x3ff)) | IMM_I12((new_target & 0xfff) >> 2);

	SLJIT_UPDATE_WX_FLAGS(inst, inst + 4, 1);

	inst = (sljit_ins *)SLJIT_ADD_EXEC_OFFSET(inst, executable_offset);
	SLJIT_CACHE_FLUSH(inst, inst + 4);
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

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	PTR_FAIL_IF(emit_const(compiler, dst_r, init_value, 0));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw));

	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_put_label *put_label;
	sljit_s32 dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_put_label(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	put_label = (struct sljit_put_label*)ensure_abuf(compiler, sizeof(struct sljit_put_label));
	PTR_FAIL_IF(!put_label);
	set_put_label(put_label, compiler, 0);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_REG2;
	PTR_FAIL_IF(push_inst(compiler, (sljit_ins)dst_r));

	compiler->size += 3;

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op_mem(compiler, WORD_DATA, TMP_REG2, dst, dstw));

	return put_label;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}
