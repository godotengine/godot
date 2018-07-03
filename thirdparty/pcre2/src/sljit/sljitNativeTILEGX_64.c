/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright 2013-2013 Tilera Corporation(jiwang@tilera.com). All rights reserved.
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

/* TileGX architecture. */
/* Contributed by Tilera Corporation. */
#include "sljitNativeTILEGX-encoder.c"

#define SIMM_8BIT_MAX (0x7f)
#define SIMM_8BIT_MIN (-0x80)
#define SIMM_16BIT_MAX (0x7fff)
#define SIMM_16BIT_MIN (-0x8000)
#define SIMM_17BIT_MAX (0xffff)
#define SIMM_17BIT_MIN (-0x10000)
#define SIMM_32BIT_MAX (0x7fffffff)
#define SIMM_32BIT_MIN (-0x7fffffff - 1)
#define SIMM_48BIT_MAX (0x7fffffff0000L)
#define SIMM_48BIT_MIN (-0x800000000000L)
#define IMM16(imm) ((imm) & 0xffff)

#define UIMM_16BIT_MAX (0xffff)

#define TMP_REG1 (SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2 (SLJIT_NUMBER_OF_REGISTERS + 3)
#define TMP_REG3 (SLJIT_NUMBER_OF_REGISTERS + 4)
#define ADDR_TMP (SLJIT_NUMBER_OF_REGISTERS + 5)
#define PIC_ADDR_REG TMP_REG2

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 6] = {
	63, 0, 1, 2, 3, 4, 30, 31, 32, 33, 34, 54, 5, 16, 6, 7
};

#define SLJIT_LOCALS_REG_mapped 54
#define TMP_REG1_mapped 5
#define TMP_REG2_mapped 16
#define TMP_REG3_mapped 6
#define ADDR_TMP_mapped 7

/* Flags are keept in volatile registers. */
#define EQUAL_FLAG 8
/* And carry flag as well. */
#define ULESS_FLAG 9
#define UGREATER_FLAG 10
#define LESS_FLAG 11
#define GREATER_FLAG 12
#define OVERFLOW_FLAG 13

#define ZERO 63
#define RA 55
#define TMP_EREG1 14
#define TMP_EREG2 15

#define LOAD_DATA 0x01
#define WORD_DATA 0x00
#define BYTE_DATA 0x02
#define HALF_DATA 0x04
#define INT_DATA 0x06
#define SIGNED_DATA 0x08
#define DOUBLE_DATA 0x10

/* Separates integer and floating point registers */
#define GPR_REG 0xf

#define MEM_MASK 0x1f

#define WRITE_BACK 0x00020
#define ARG_TEST 0x00040
#define ALT_KEEP_CACHE 0x00080
#define CUMULATIVE_OP 0x00100
#define LOGICAL_OP 0x00200
#define IMM_OP 0x00400
#define SRC2_IMM 0x00800

#define UNUSED_DEST 0x01000
#define REG_DEST 0x02000
#define REG1_SOURCE 0x04000
#define REG2_SOURCE 0x08000
#define SLOW_SRC1 0x10000
#define SLOW_SRC2 0x20000
#define SLOW_DEST 0x40000

/* Only these flags are set. UNUSED_DEST is not set when no flags should be set.
 */
#define CHECK_FLAGS(list) (!(flags & UNUSED_DEST) || (op & GET_FLAGS(~(list))))

SLJIT_API_FUNC_ATTRIBUTE const char *sljit_get_platform_name(void)
{
	return "TileGX" SLJIT_CPUINFO;
}

/* Length of an instruction word */
typedef sljit_uw sljit_ins;

struct jit_instr {
	const struct tilegx_opcode* opcode; 
	tilegx_pipeline pipe;
	unsigned long input_registers;
	unsigned long output_registers;
	int operand_value[4];
	int line;
};

/* Opcode Helper Macros */
#define TILEGX_X_MODE 0

#define X_MODE create_Mode(TILEGX_X_MODE)

#define FNOP_X0 \
	create_Opcode_X0(RRR_0_OPCODE_X0) | \
	create_RRROpcodeExtension_X0(UNARY_RRR_0_OPCODE_X0) | \
	create_UnaryOpcodeExtension_X0(FNOP_UNARY_OPCODE_X0)

#define FNOP_X1 \
	create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(UNARY_RRR_0_OPCODE_X1) | \
	create_UnaryOpcodeExtension_X1(FNOP_UNARY_OPCODE_X1)

#define NOP \
	create_Mode(TILEGX_X_MODE) | FNOP_X0 | FNOP_X1

#define ANOP_X0 \
	create_Opcode_X0(RRR_0_OPCODE_X0) | \
	create_RRROpcodeExtension_X0(UNARY_RRR_0_OPCODE_X0) | \
	create_UnaryOpcodeExtension_X0(NOP_UNARY_OPCODE_X0)

#define BPT create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(UNARY_RRR_0_OPCODE_X1) | \
	create_UnaryOpcodeExtension_X1(ILL_UNARY_OPCODE_X1) | \
	create_Dest_X1(0x1C) | create_SrcA_X1(0x25) | ANOP_X0

#define ADD_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(ADD_RRR_0_OPCODE_X1) | FNOP_X0

#define ADDI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(IMM8_OPCODE_X1) | \
	create_Imm8OpcodeExtension_X1(ADDI_IMM8_OPCODE_X1) | FNOP_X0

#define SUB_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(SUB_RRR_0_OPCODE_X1) | FNOP_X0

#define NOR_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(NOR_RRR_0_OPCODE_X1) | FNOP_X0

#define OR_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(OR_RRR_0_OPCODE_X1) | FNOP_X0

#define AND_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(AND_RRR_0_OPCODE_X1) | FNOP_X0

#define XOR_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(XOR_RRR_0_OPCODE_X1) | FNOP_X0

#define CMOVNEZ_X0 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X0(RRR_0_OPCODE_X0) | \
	create_RRROpcodeExtension_X0(CMOVNEZ_RRR_0_OPCODE_X0) | FNOP_X1

#define CMOVEQZ_X0 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X0(RRR_0_OPCODE_X0) | \
	create_RRROpcodeExtension_X0(CMOVEQZ_RRR_0_OPCODE_X0) | FNOP_X1

#define ADDLI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(ADDLI_OPCODE_X1) | FNOP_X0

#define V4INT_L_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(V4INT_L_RRR_0_OPCODE_X1) | FNOP_X0

#define BFEXTU_X0 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X0(BF_OPCODE_X0) | \
	create_BFOpcodeExtension_X0(BFEXTU_BF_OPCODE_X0) | FNOP_X1

#define BFEXTS_X0 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X0(BF_OPCODE_X0) | \
	create_BFOpcodeExtension_X0(BFEXTS_BF_OPCODE_X0) | FNOP_X1

#define SHL16INSLI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(SHL16INSLI_OPCODE_X1) | FNOP_X0

#define ST_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(ST_RRR_0_OPCODE_X1) | create_Dest_X1(0x0) | FNOP_X0

#define LD_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(UNARY_RRR_0_OPCODE_X1) | \
	create_UnaryOpcodeExtension_X1(LD_UNARY_OPCODE_X1) | FNOP_X0

#define JR_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(UNARY_RRR_0_OPCODE_X1) | \
	create_UnaryOpcodeExtension_X1(JR_UNARY_OPCODE_X1) | FNOP_X0

#define JALR_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(UNARY_RRR_0_OPCODE_X1) | \
	create_UnaryOpcodeExtension_X1(JALR_UNARY_OPCODE_X1) | FNOP_X0

#define CLZ_X0 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X0(RRR_0_OPCODE_X0) | \
	create_RRROpcodeExtension_X0(UNARY_RRR_0_OPCODE_X0) | \
	create_UnaryOpcodeExtension_X0(CNTLZ_UNARY_OPCODE_X0) | FNOP_X1

#define CMPLTUI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(IMM8_OPCODE_X1) | \
	create_Imm8OpcodeExtension_X1(CMPLTUI_IMM8_OPCODE_X1) | FNOP_X0

#define CMPLTU_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(CMPLTU_RRR_0_OPCODE_X1) | FNOP_X0

#define CMPLTS_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(CMPLTS_RRR_0_OPCODE_X1) | FNOP_X0

#define XORI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(IMM8_OPCODE_X1) | \
	create_Imm8OpcodeExtension_X1(XORI_IMM8_OPCODE_X1) | FNOP_X0

#define ORI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(IMM8_OPCODE_X1) | \
	create_Imm8OpcodeExtension_X1(ORI_IMM8_OPCODE_X1) | FNOP_X0

#define ANDI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(IMM8_OPCODE_X1) | \
	create_Imm8OpcodeExtension_X1(ANDI_IMM8_OPCODE_X1) | FNOP_X0

#define SHLI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(SHIFT_OPCODE_X1) | \
	create_ShiftOpcodeExtension_X1(SHLI_SHIFT_OPCODE_X1) | FNOP_X0

#define SHL_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(SHL_RRR_0_OPCODE_X1) | FNOP_X0

#define SHRSI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(SHIFT_OPCODE_X1) | \
	create_ShiftOpcodeExtension_X1(SHRSI_SHIFT_OPCODE_X1) | FNOP_X0

#define SHRS_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(SHRS_RRR_0_OPCODE_X1) | FNOP_X0

#define SHRUI_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(SHIFT_OPCODE_X1) | \
	create_ShiftOpcodeExtension_X1(SHRUI_SHIFT_OPCODE_X1) | FNOP_X0

#define SHRU_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(RRR_0_OPCODE_X1) | \
	create_RRROpcodeExtension_X1(SHRU_RRR_0_OPCODE_X1) | FNOP_X0

#define BEQZ_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(BRANCH_OPCODE_X1) | \
	create_BrType_X1(BEQZ_BRANCH_OPCODE_X1) | FNOP_X0

#define BNEZ_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(BRANCH_OPCODE_X1) | \
	create_BrType_X1(BNEZ_BRANCH_OPCODE_X1) | FNOP_X0

#define J_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(JUMP_OPCODE_X1) | \
	create_JumpOpcodeExtension_X1(J_JUMP_OPCODE_X1) | FNOP_X0

#define JAL_X1 \
	create_Mode(TILEGX_X_MODE) | create_Opcode_X1(JUMP_OPCODE_X1) | \
	create_JumpOpcodeExtension_X1(JAL_JUMP_OPCODE_X1) | FNOP_X0

#define DEST_X0(x) create_Dest_X0(x)
#define SRCA_X0(x) create_SrcA_X0(x)
#define SRCB_X0(x) create_SrcB_X0(x)
#define DEST_X1(x) create_Dest_X1(x)
#define SRCA_X1(x) create_SrcA_X1(x)
#define SRCB_X1(x) create_SrcB_X1(x)
#define IMM16_X1(x) create_Imm16_X1(x)
#define IMM8_X1(x) create_Imm8_X1(x)
#define BFSTART_X0(x) create_BFStart_X0(x)
#define BFEND_X0(x) create_BFEnd_X0(x)
#define SHIFTIMM_X1(x) create_ShAmt_X1(x)
#define JOFF_X1(x) create_JumpOff_X1(x)
#define BOFF_X1(x) create_BrOff_X1(x)

static const tilegx_mnemonic data_transfer_insts[16] = {
	/* u w s */ TILEGX_OPC_ST   /* st */,
	/* u w l */ TILEGX_OPC_LD   /* ld */,
	/* u b s */ TILEGX_OPC_ST1  /* st1 */,
	/* u b l */ TILEGX_OPC_LD1U /* ld1u */,
	/* u h s */ TILEGX_OPC_ST2  /* st2 */,
	/* u h l */ TILEGX_OPC_LD2U /* ld2u */,
	/* u i s */ TILEGX_OPC_ST4  /* st4 */,
	/* u i l */ TILEGX_OPC_LD4U /* ld4u */,
	/* s w s */ TILEGX_OPC_ST   /* st */,
	/* s w l */ TILEGX_OPC_LD   /* ld */,
	/* s b s */ TILEGX_OPC_ST1  /* st1 */,
	/* s b l */ TILEGX_OPC_LD1S /* ld1s */,
	/* s h s */ TILEGX_OPC_ST2  /* st2 */,
	/* s h l */ TILEGX_OPC_LD2S /* ld2s */,
	/* s i s */ TILEGX_OPC_ST4  /* st4 */,
	/* s i l */ TILEGX_OPC_LD4S /* ld4s */,
};

#ifdef TILEGX_JIT_DEBUG
static sljit_s32 push_inst_debug(struct sljit_compiler *compiler, sljit_ins ins, int line)
{
	sljit_ins *ptr = (sljit_ins *)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	printf("|%04d|S0|:\t\t", line);
	print_insn_tilegx(ptr);
	return SLJIT_SUCCESS;
}

static sljit_s32 push_inst_nodebug(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins *)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

#define push_inst(a, b) push_inst_debug(a, b, __LINE__)
#else
static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ptr = (sljit_ins *)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ptr);
	*ptr = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}
#endif

#define BUNDLE_FORMAT_MASK(p0, p1, p2) \
	((p0) | ((p1) << 8) | ((p2) << 16))

#define BUNDLE_FORMAT(p0, p1, p2) \
	{ \
		{ \
			(tilegx_pipeline)(p0), \
			(tilegx_pipeline)(p1), \
			(tilegx_pipeline)(p2) \
		}, \
		BUNDLE_FORMAT_MASK(1 << (p0), 1 << (p1), (1 << (p2))) \
	}

#define NO_PIPELINE TILEGX_NUM_PIPELINE_ENCODINGS

#define tilegx_is_x_pipeline(p) ((int)(p) <= (int)TILEGX_PIPELINE_X1)

#define PI(encoding) \
	push_inst(compiler, encoding)

#define PB3(opcode, dst, srca, srcb) \
	push_3_buffer(compiler, opcode, dst, srca, srcb, __LINE__)

#define PB2(opcode, dst, src) \
	push_2_buffer(compiler, opcode, dst, src, __LINE__)

#define JR(reg) \
	push_jr_buffer(compiler, TILEGX_OPC_JR, reg, __LINE__)

#define ADD(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_ADD, dst, srca, srcb, __LINE__)

#define SUB(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_SUB, dst, srca, srcb, __LINE__)

#define MUL(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_MULX, dst, srca, srcb, __LINE__)

#define NOR(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_NOR, dst, srca, srcb, __LINE__)

#define OR(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_OR, dst, srca, srcb, __LINE__)

#define XOR(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_XOR, dst, srca, srcb, __LINE__)

#define AND(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_AND, dst, srca, srcb, __LINE__)

#define CLZ(dst, src) \
	push_2_buffer(compiler, TILEGX_OPC_CLZ, dst, src, __LINE__)

#define SHLI(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_SHLI, dst, srca, srcb, __LINE__)

#define SHRUI(dst, srca, imm) \
	push_3_buffer(compiler, TILEGX_OPC_SHRUI, dst, srca, imm, __LINE__)

#define XORI(dst, srca, imm) \
	push_3_buffer(compiler, TILEGX_OPC_XORI, dst, srca, imm, __LINE__)

#define ORI(dst, srca, imm) \
	push_3_buffer(compiler, TILEGX_OPC_ORI, dst, srca, imm, __LINE__)

#define CMPLTU(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_CMPLTU, dst, srca, srcb, __LINE__)

#define CMPLTS(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_CMPLTS, dst, srca, srcb, __LINE__)

#define CMPLTUI(dst, srca, imm) \
	push_3_buffer(compiler, TILEGX_OPC_CMPLTUI, dst, srca, imm, __LINE__)

#define CMOVNEZ(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_CMOVNEZ, dst, srca, srcb, __LINE__)

#define CMOVEQZ(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_CMOVEQZ, dst, srca, srcb, __LINE__)

#define ADDLI(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_ADDLI, dst, srca, srcb, __LINE__)

#define SHL16INSLI(dst, srca, srcb) \
	push_3_buffer(compiler, TILEGX_OPC_SHL16INSLI, dst, srca, srcb, __LINE__)

#define LD_ADD(dst, addr, adjust) \
	push_3_buffer(compiler, TILEGX_OPC_LD_ADD, dst, addr, adjust, __LINE__)

#define ST_ADD(src, addr, adjust) \
	push_3_buffer(compiler, TILEGX_OPC_ST_ADD, src, addr, adjust, __LINE__)

#define LD(dst, addr) \
	push_2_buffer(compiler, TILEGX_OPC_LD, dst, addr, __LINE__)

#define BFEXTU(dst, src, start, end) \
	push_4_buffer(compiler, TILEGX_OPC_BFEXTU, dst, src, start, end, __LINE__)

#define BFEXTS(dst, src, start, end) \
	push_4_buffer(compiler, TILEGX_OPC_BFEXTS, dst, src, start, end, __LINE__)

#define ADD_SOLO(dest, srca, srcb) \
	push_inst(compiler, ADD_X1 | DEST_X1(dest) | SRCA_X1(srca) | SRCB_X1(srcb))

#define ADDI_SOLO(dest, srca, imm) \
	push_inst(compiler, ADDI_X1 | DEST_X1(dest) | SRCA_X1(srca) | IMM8_X1(imm))

#define ADDLI_SOLO(dest, srca, imm) \
	push_inst(compiler, ADDLI_X1 | DEST_X1(dest) | SRCA_X1(srca) | IMM16_X1(imm))

#define SHL16INSLI_SOLO(dest, srca, imm) \
	push_inst(compiler, SHL16INSLI_X1 | DEST_X1(dest) | SRCA_X1(srca) | IMM16_X1(imm))

#define JALR_SOLO(reg) \
	push_inst(compiler, JALR_X1 | SRCA_X1(reg))

#define JR_SOLO(reg) \
	push_inst(compiler, JR_X1 | SRCA_X1(reg))

struct Format {
	/* Mapping of bundle issue slot to assigned pipe. */
	tilegx_pipeline pipe[TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE];

	/* Mask of pipes used by this bundle. */
	unsigned int pipe_mask;
};

const struct Format formats[] =
{
	/* In Y format we must always have something in Y2, since it has
	* no fnop, so this conveys that Y2 must always be used. */
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y0, TILEGX_PIPELINE_Y2, NO_PIPELINE),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y1, TILEGX_PIPELINE_Y2, NO_PIPELINE),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y2, TILEGX_PIPELINE_Y0, NO_PIPELINE),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y2, TILEGX_PIPELINE_Y1, NO_PIPELINE),

	/* Y format has three instructions. */
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y0, TILEGX_PIPELINE_Y1, TILEGX_PIPELINE_Y2),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y0, TILEGX_PIPELINE_Y2, TILEGX_PIPELINE_Y1),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y1, TILEGX_PIPELINE_Y0, TILEGX_PIPELINE_Y2),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y1, TILEGX_PIPELINE_Y2, TILEGX_PIPELINE_Y0),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y2, TILEGX_PIPELINE_Y0, TILEGX_PIPELINE_Y1),
	BUNDLE_FORMAT(TILEGX_PIPELINE_Y2, TILEGX_PIPELINE_Y1, TILEGX_PIPELINE_Y0),

	/* X format has only two instructions. */
	BUNDLE_FORMAT(TILEGX_PIPELINE_X0, TILEGX_PIPELINE_X1, NO_PIPELINE),
	BUNDLE_FORMAT(TILEGX_PIPELINE_X1, TILEGX_PIPELINE_X0, NO_PIPELINE)
};


struct jit_instr inst_buf[TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE];
unsigned long inst_buf_index;

tilegx_pipeline get_any_valid_pipe(const struct tilegx_opcode* opcode)
{
	/* FIXME: tile: we could pregenerate this. */
	int pipe;
	for (pipe = 0; ((opcode->pipes & (1 << pipe)) == 0 && pipe < TILEGX_NUM_PIPELINE_ENCODINGS); pipe++)
		;
	return (tilegx_pipeline)(pipe);
}

void insert_nop(tilegx_mnemonic opc, int line)
{
	const struct tilegx_opcode* opcode = NULL;

	memmove(&inst_buf[1], &inst_buf[0], inst_buf_index * sizeof inst_buf[0]);

	opcode = &tilegx_opcodes[opc];
	inst_buf[0].opcode = opcode;
	inst_buf[0].pipe = get_any_valid_pipe(opcode);
	inst_buf[0].input_registers = 0;
	inst_buf[0].output_registers = 0;
	inst_buf[0].line = line;
	++inst_buf_index;
}

const struct Format* compute_format()
{
	unsigned int compatible_pipes = BUNDLE_FORMAT_MASK(
		inst_buf[0].opcode->pipes,
		inst_buf[1].opcode->pipes,
		(inst_buf_index == 3 ? inst_buf[2].opcode->pipes : (1 << NO_PIPELINE)));

	const struct Format* match = NULL;
	const struct Format *b = NULL;
	unsigned int i;
	for (i = 0; i < sizeof formats / sizeof formats[0]; i++) {
		b = &formats[i];
		if ((b->pipe_mask & compatible_pipes) == b->pipe_mask) {
			match = b;
			break;
		}
	}

	return match;
}

sljit_s32 assign_pipes()
{
	unsigned long output_registers = 0;
	unsigned int i = 0;

	if (inst_buf_index == 1) {
		tilegx_mnemonic opc = inst_buf[0].opcode->can_bundle
					? TILEGX_OPC_FNOP : TILEGX_OPC_NOP;
		insert_nop(opc, __LINE__);
	}

	const struct Format* match = compute_format();

	if (match == NULL)
		return -1;

	for (i = 0; i < inst_buf_index; i++) {

		if ((i > 0) && ((inst_buf[i].input_registers & output_registers) != 0))
			return -1;

		if ((i > 0) && ((inst_buf[i].output_registers & output_registers) != 0))
			return -1;

		/* Don't include Rzero in the match set, to avoid triggering
		   needlessly on 'prefetch' instrs. */

		output_registers |= inst_buf[i].output_registers & 0xFFFFFFFFFFFFFFL;

		inst_buf[i].pipe = match->pipe[i];
	}

	/* If only 2 instrs, and in Y-mode, insert a nop. */
	if (inst_buf_index == 2 && !tilegx_is_x_pipeline(match->pipe[0])) {
		insert_nop(TILEGX_OPC_FNOP, __LINE__);

		/* Select the yet unassigned pipe. */
		tilegx_pipeline pipe = (tilegx_pipeline)(((TILEGX_PIPELINE_Y0
					+ TILEGX_PIPELINE_Y1 + TILEGX_PIPELINE_Y2)
					- (inst_buf[1].pipe + inst_buf[2].pipe)));

		inst_buf[0].pipe = pipe;
	}

	return 0;
}

tilegx_bundle_bits get_bundle_bit(struct jit_instr *inst)
{
	int i, val;
	const struct tilegx_opcode* opcode = inst->opcode;
	tilegx_bundle_bits bits = opcode->fixed_bit_values[inst->pipe];

	const struct tilegx_operand* operand = NULL;
	for (i = 0; i < opcode->num_operands; i++) {
		operand = &tilegx_operands[opcode->operands[inst->pipe][i]];
		val = inst->operand_value[i];

		bits |= operand->insert(val);
	}

	return bits;
}

static sljit_s32 update_buffer(struct sljit_compiler *compiler)
{
	int i;
	int orig_index = inst_buf_index;
	struct jit_instr inst0 = inst_buf[0];
	struct jit_instr inst1 = inst_buf[1];
	struct jit_instr inst2 = inst_buf[2];
	tilegx_bundle_bits bits = 0;

	/* If the bundle is valid as is, perform the encoding and return 1. */
	if (assign_pipes() == 0) {
		for (i = 0; i < inst_buf_index; i++) {
			bits |= get_bundle_bit(inst_buf + i);
#ifdef TILEGX_JIT_DEBUG
			printf("|%04d", inst_buf[i].line);
#endif
		}
#ifdef TILEGX_JIT_DEBUG
		if (inst_buf_index == 3)
			printf("|M0|:\t");
		else
			printf("|M0|:\t\t");
		print_insn_tilegx(&bits);
#endif

		inst_buf_index = 0;

#ifdef TILEGX_JIT_DEBUG
		return push_inst_nodebug(compiler, bits);
#else
		return push_inst(compiler, bits);
#endif
	}

	/* If the bundle is invalid, split it in two. First encode the first two
	   (or possibly 1) instructions, and then the last, separately. Note that
	   assign_pipes may have re-ordered the instrs (by inserting no-ops in
	   lower slots) so we need to reset them. */

	inst_buf_index = orig_index - 1;
	inst_buf[0] = inst0;
	inst_buf[1] = inst1;
	inst_buf[2] = inst2;
	if (assign_pipes() == 0) {
		for (i = 0; i < inst_buf_index; i++) {
			bits |= get_bundle_bit(inst_buf + i);
#ifdef TILEGX_JIT_DEBUG
			printf("|%04d", inst_buf[i].line);
#endif
		}

#ifdef TILEGX_JIT_DEBUG
		if (inst_buf_index == 3)
			printf("|M1|:\t");
		else
			printf("|M1|:\t\t");
		print_insn_tilegx(&bits);
#endif

		if ((orig_index - 1) == 2) {
			inst_buf[0] = inst2;
			inst_buf_index = 1;
		} else if ((orig_index - 1) == 1) {
			inst_buf[0] = inst1;
			inst_buf_index = 1;
		} else
			SLJIT_UNREACHABLE();

#ifdef TILEGX_JIT_DEBUG
		return push_inst_nodebug(compiler, bits);
#else
		return push_inst(compiler, bits);
#endif
	} else {
		/* We had 3 instrs of which the first 2 can't live in the same bundle.
		   Split those two. Note that we don't try to then combine the second
		   and third instr into a single bundle.  First instruction: */
		inst_buf_index = 1;
		inst_buf[0] = inst0;
		inst_buf[1] = inst1;
		inst_buf[2] = inst2;
		if (assign_pipes() == 0) {
			for (i = 0; i < inst_buf_index; i++) {
				bits |= get_bundle_bit(inst_buf + i);
#ifdef TILEGX_JIT_DEBUG
				printf("|%04d", inst_buf[i].line);
#endif
			}

#ifdef TILEGX_JIT_DEBUG
			if (inst_buf_index == 3)
				printf("|M2|:\t");
			else
				printf("|M2|:\t\t");
			print_insn_tilegx(&bits);
#endif

			inst_buf[0] = inst1;
			inst_buf[1] = inst2;
			inst_buf_index = orig_index - 1;
#ifdef TILEGX_JIT_DEBUG
			return push_inst_nodebug(compiler, bits);
#else
			return push_inst(compiler, bits);
#endif
		} else
			SLJIT_UNREACHABLE();
	}

	SLJIT_UNREACHABLE();
}

static sljit_s32 flush_buffer(struct sljit_compiler *compiler)
{
	while (inst_buf_index != 0) {
		FAIL_IF(update_buffer(compiler));
	}
	return SLJIT_SUCCESS;
}

static sljit_s32 push_4_buffer(struct sljit_compiler *compiler, tilegx_mnemonic opc, int op0, int op1, int op2, int op3, int line)
{
	if (inst_buf_index == TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE)
		FAIL_IF(update_buffer(compiler));

	const struct tilegx_opcode* opcode = &tilegx_opcodes[opc];
	inst_buf[inst_buf_index].opcode = opcode;
	inst_buf[inst_buf_index].pipe = get_any_valid_pipe(opcode);
	inst_buf[inst_buf_index].operand_value[0] = op0;
	inst_buf[inst_buf_index].operand_value[1] = op1;
	inst_buf[inst_buf_index].operand_value[2] = op2;
	inst_buf[inst_buf_index].operand_value[3] = op3;
	inst_buf[inst_buf_index].input_registers = 1L << op1;
	inst_buf[inst_buf_index].output_registers = 1L << op0;
	inst_buf[inst_buf_index].line = line;
	inst_buf_index++;

	return SLJIT_SUCCESS;
}

static sljit_s32 push_3_buffer(struct sljit_compiler *compiler, tilegx_mnemonic opc, int op0, int op1, int op2, int line)
{
	if (inst_buf_index == TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE)
		FAIL_IF(update_buffer(compiler));

	const struct tilegx_opcode* opcode = &tilegx_opcodes[opc];
	inst_buf[inst_buf_index].opcode = opcode;
	inst_buf[inst_buf_index].pipe = get_any_valid_pipe(opcode);
	inst_buf[inst_buf_index].operand_value[0] = op0;
	inst_buf[inst_buf_index].operand_value[1] = op1;
	inst_buf[inst_buf_index].operand_value[2] = op2;
	inst_buf[inst_buf_index].line = line;

	switch (opc) {
	case TILEGX_OPC_ST_ADD:
		inst_buf[inst_buf_index].input_registers = (1L << op0) | (1L << op1);
		inst_buf[inst_buf_index].output_registers = 1L << op0;
		break;
	case TILEGX_OPC_LD_ADD:
		inst_buf[inst_buf_index].input_registers = 1L << op1;
		inst_buf[inst_buf_index].output_registers = (1L << op0) | (1L << op1);
		break;
	case TILEGX_OPC_ADD:
	case TILEGX_OPC_AND:
	case TILEGX_OPC_SUB:
	case TILEGX_OPC_MULX:
	case TILEGX_OPC_OR:
	case TILEGX_OPC_XOR:
	case TILEGX_OPC_NOR:
	case TILEGX_OPC_SHL:
	case TILEGX_OPC_SHRU:
	case TILEGX_OPC_SHRS:
	case TILEGX_OPC_CMPLTU:
	case TILEGX_OPC_CMPLTS:
	case TILEGX_OPC_CMOVEQZ:
	case TILEGX_OPC_CMOVNEZ:
		inst_buf[inst_buf_index].input_registers = (1L << op1) | (1L << op2);
		inst_buf[inst_buf_index].output_registers = 1L << op0;
		break;
	case TILEGX_OPC_ADDLI:
	case TILEGX_OPC_XORI:
	case TILEGX_OPC_ORI:
	case TILEGX_OPC_SHLI:
	case TILEGX_OPC_SHRUI:
	case TILEGX_OPC_SHRSI:
	case TILEGX_OPC_SHL16INSLI:
	case TILEGX_OPC_CMPLTUI:
	case TILEGX_OPC_CMPLTSI:
		inst_buf[inst_buf_index].input_registers = 1L << op1;
		inst_buf[inst_buf_index].output_registers = 1L << op0;
		break;
	default:
		printf("unrecoginzed opc: %s\n", opcode->name);
		SLJIT_UNREACHABLE();
	}

	inst_buf_index++;

	return SLJIT_SUCCESS;
}

static sljit_s32 push_2_buffer(struct sljit_compiler *compiler, tilegx_mnemonic opc, int op0, int op1, int line)
{
	if (inst_buf_index == TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE)
		FAIL_IF(update_buffer(compiler));

	const struct tilegx_opcode* opcode = &tilegx_opcodes[opc];
	inst_buf[inst_buf_index].opcode = opcode;
	inst_buf[inst_buf_index].pipe = get_any_valid_pipe(opcode);
	inst_buf[inst_buf_index].operand_value[0] = op0;
	inst_buf[inst_buf_index].operand_value[1] = op1;
	inst_buf[inst_buf_index].line = line;

	switch (opc) {
	case TILEGX_OPC_BEQZ:
	case TILEGX_OPC_BNEZ:
		inst_buf[inst_buf_index].input_registers = 1L << op0;
		break;
	case TILEGX_OPC_ST:
	case TILEGX_OPC_ST1:
	case TILEGX_OPC_ST2:
	case TILEGX_OPC_ST4:
		inst_buf[inst_buf_index].input_registers = (1L << op0) | (1L << op1);
		inst_buf[inst_buf_index].output_registers = 0;
		break;
	case TILEGX_OPC_CLZ:
	case TILEGX_OPC_LD:
	case TILEGX_OPC_LD1U:
	case TILEGX_OPC_LD1S:
	case TILEGX_OPC_LD2U:
	case TILEGX_OPC_LD2S:
	case TILEGX_OPC_LD4U:
	case TILEGX_OPC_LD4S:
		inst_buf[inst_buf_index].input_registers = 1L << op1;
		inst_buf[inst_buf_index].output_registers = 1L << op0;
		break;
	default:
		printf("unrecoginzed opc: %s\n", opcode->name);
		SLJIT_UNREACHABLE();
	}

	inst_buf_index++;

	return SLJIT_SUCCESS;
}

static sljit_s32 push_0_buffer(struct sljit_compiler *compiler, tilegx_mnemonic opc, int line)
{
	if (inst_buf_index == TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE)
		FAIL_IF(update_buffer(compiler));

	const struct tilegx_opcode* opcode = &tilegx_opcodes[opc];
	inst_buf[inst_buf_index].opcode = opcode;
	inst_buf[inst_buf_index].pipe = get_any_valid_pipe(opcode);
	inst_buf[inst_buf_index].input_registers = 0;
	inst_buf[inst_buf_index].output_registers = 0;
	inst_buf[inst_buf_index].line = line;
	inst_buf_index++;

	return SLJIT_SUCCESS;
}

static sljit_s32 push_jr_buffer(struct sljit_compiler *compiler, tilegx_mnemonic opc, int op0, int line)
{
	if (inst_buf_index == TILEGX_MAX_INSTRUCTIONS_PER_BUNDLE)
		FAIL_IF(update_buffer(compiler));

	const struct tilegx_opcode* opcode = &tilegx_opcodes[opc];
	inst_buf[inst_buf_index].opcode = opcode;
	inst_buf[inst_buf_index].pipe = get_any_valid_pipe(opcode);
	inst_buf[inst_buf_index].operand_value[0] = op0;
	inst_buf[inst_buf_index].input_registers = 1L << op0;
	inst_buf[inst_buf_index].output_registers = 0;
	inst_buf[inst_buf_index].line = line;
	inst_buf_index++;
 
	return flush_buffer(compiler);
}

static SLJIT_INLINE sljit_ins * detect_jump_type(struct sljit_jump *jump, sljit_ins *code_ptr, sljit_ins *code)
{
	sljit_sw diff;
	sljit_uw target_addr;
	sljit_ins *inst;

	if (jump->flags & SLJIT_REWRITABLE_JUMP)
		return code_ptr;

	if (jump->flags & JUMP_ADDR)
		target_addr = jump->u.target;
	else {
		SLJIT_ASSERT(jump->flags & JUMP_LABEL);
		target_addr = (sljit_uw)(code + jump->u.label->size);
	}

	inst = (sljit_ins *)jump->addr;
	if (jump->flags & IS_COND)
		inst--;

	diff = ((sljit_sw) target_addr - (sljit_sw) inst) >> 3;
	if (diff <= SIMM_17BIT_MAX && diff >= SIMM_17BIT_MIN) {
		jump->flags |= PATCH_B;

		if (!(jump->flags & IS_COND)) {
			if (jump->flags & IS_JAL) {
				jump->flags &= ~(PATCH_B);
				jump->flags |= PATCH_J;
				inst[0] = JAL_X1;

#ifdef TILEGX_JIT_DEBUG
				printf("[runtime relocate]%04d:\t", __LINE__);
				print_insn_tilegx(inst);
#endif
			} else {
				inst[0] = BEQZ_X1 | SRCA_X1(ZERO);

#ifdef TILEGX_JIT_DEBUG
				printf("[runtime relocate]%04d:\t", __LINE__);
				print_insn_tilegx(inst);
#endif
			}

			return inst;
		}

		inst[0] = inst[0] ^ (0x7L << 55);

#ifdef TILEGX_JIT_DEBUG
		printf("[runtime relocate]%04d:\t", __LINE__);
		print_insn_tilegx(inst);
#endif
		jump->addr -= sizeof(sljit_ins);
		return inst;
	}

	if (jump->flags & IS_COND) {
		if ((target_addr & ~0x3FFFFFFFL) == ((jump->addr + sizeof(sljit_ins)) & ~0x3FFFFFFFL)) {
			jump->flags |= PATCH_J;
			inst[0] = (inst[0] & ~(BOFF_X1(-1))) | BOFF_X1(2);
			inst[1] = J_X1;
			return inst + 1;
		}

		return code_ptr;
	}

	if ((target_addr & ~0x3FFFFFFFL) == ((jump->addr + sizeof(sljit_ins)) & ~0x3FFFFFFFL)) {
		jump->flags |= PATCH_J;

		if (jump->flags & IS_JAL) {
			inst[0] = JAL_X1;

#ifdef TILEGX_JIT_DEBUG
			printf("[runtime relocate]%04d:\t", __LINE__);
			print_insn_tilegx(inst);
#endif

		} else {
			inst[0] = J_X1;

#ifdef TILEGX_JIT_DEBUG
			printf("[runtime relocate]%04d:\t", __LINE__);
			print_insn_tilegx(inst);
#endif
		}

		return inst;
	}

	return code_ptr;
}

SLJIT_API_FUNC_ATTRIBUTE void * sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	sljit_ins *code;
	sljit_ins *code_ptr;
	sljit_ins *buf_ptr;
	sljit_ins *buf_end;
	sljit_uw word_count;
	sljit_uw addr;

	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	code = (sljit_ins *)SLJIT_MALLOC_EXEC(compiler->size * sizeof(sljit_ins));
	PTR_FAIL_WITH_EXEC_IF(code);
	buf = compiler->buf;

	code_ptr = code;
	word_count = 0;
	label = compiler->labels;
	jump = compiler->jumps;
	const_ = compiler->consts;
	do {
		buf_ptr = (sljit_ins *)buf->memory;
		buf_end = buf_ptr + (buf->used_size >> 3);
		do {
			*code_ptr = *buf_ptr++;
			SLJIT_ASSERT(!label || label->size >= word_count);
			SLJIT_ASSERT(!jump || jump->addr >= word_count);
			SLJIT_ASSERT(!const_ || const_->addr >= word_count);
			/* These structures are ordered by their address. */
			if (label && label->size == word_count) {
				/* Just recording the address. */
				label->addr = (sljit_uw) code_ptr;
				label->size = code_ptr - code;
				label = label->next;
			}

			if (jump && jump->addr == word_count) {
				if (jump->flags & IS_JAL)
					jump->addr = (sljit_uw)(code_ptr - 4);
				else
					jump->addr = (sljit_uw)(code_ptr - 3);

				code_ptr = detect_jump_type(jump, code_ptr, code);
				jump = jump->next;
			}

			if (const_ && const_->addr == word_count) {
				/* Just recording the address. */
				const_->addr = (sljit_uw) code_ptr;
				const_ = const_->next;
			}

			code_ptr++;
			word_count++;
		} while (buf_ptr < buf_end);

		buf = buf->next;
	} while (buf);

	if (label && label->size == word_count) {
		label->addr = (sljit_uw) code_ptr;
		label->size = code_ptr - code;
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!const_);
	SLJIT_ASSERT(code_ptr - code <= (sljit_sw)compiler->size);

	jump = compiler->jumps;
	while (jump) {
		do {
			addr = (jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target;
			buf_ptr = (sljit_ins *)jump->addr;

			if (jump->flags & PATCH_B) {
				addr = (sljit_sw)(addr - (jump->addr)) >> 3;
				SLJIT_ASSERT((sljit_sw) addr <= SIMM_17BIT_MAX && (sljit_sw) addr >= SIMM_17BIT_MIN);
				buf_ptr[0] = (buf_ptr[0] & ~(BOFF_X1(-1))) | BOFF_X1(addr);

#ifdef TILEGX_JIT_DEBUG
				printf("[runtime relocate]%04d:\t", __LINE__);
				print_insn_tilegx(buf_ptr);
#endif
				break;
			}

			if (jump->flags & PATCH_J) {
				SLJIT_ASSERT((addr & ~0x3FFFFFFFL) == ((jump->addr + sizeof(sljit_ins)) & ~0x3FFFFFFFL));
				addr = (sljit_sw)(addr - (jump->addr)) >> 3;
				buf_ptr[0] = (buf_ptr[0] & ~(JOFF_X1(-1))) | JOFF_X1(addr);

#ifdef TILEGX_JIT_DEBUG
				printf("[runtime relocate]%04d:\t", __LINE__);
				print_insn_tilegx(buf_ptr);
#endif
				break;
			}

			SLJIT_ASSERT(!(jump->flags & IS_JAL));

			/* Set the fields of immediate loads. */
			buf_ptr[0] = (buf_ptr[0] & ~(0xFFFFL << 43)) | (((addr >> 32) & 0xFFFFL) << 43);
			buf_ptr[1] = (buf_ptr[1] & ~(0xFFFFL << 43)) | (((addr >> 16) & 0xFFFFL) << 43);
			buf_ptr[2] = (buf_ptr[2] & ~(0xFFFFL << 43)) | ((addr & 0xFFFFL) << 43);
		} while (0);

		jump = jump->next;
	}

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_size = (code_ptr - code) * sizeof(sljit_ins);
	SLJIT_CACHE_FLUSH(code, code_ptr);
	return code;
}

static sljit_s32 load_immediate(struct sljit_compiler *compiler, sljit_s32 dst_ar, sljit_sw imm)
{

	if (imm <= SIMM_16BIT_MAX && imm >= SIMM_16BIT_MIN)
		return ADDLI(dst_ar, ZERO, imm);

	if (imm <= SIMM_32BIT_MAX && imm >= SIMM_32BIT_MIN) {
		FAIL_IF(ADDLI(dst_ar, ZERO, imm >> 16));
		return SHL16INSLI(dst_ar, dst_ar, imm);
	}

	if (imm <= SIMM_48BIT_MAX && imm >= SIMM_48BIT_MIN) {
		FAIL_IF(ADDLI(dst_ar, ZERO, imm >> 32));
		FAIL_IF(SHL16INSLI(dst_ar, dst_ar, imm >> 16));
		return SHL16INSLI(dst_ar, dst_ar, imm);
	}

	FAIL_IF(ADDLI(dst_ar, ZERO, imm >> 48));
	FAIL_IF(SHL16INSLI(dst_ar, dst_ar, imm >> 32));
	FAIL_IF(SHL16INSLI(dst_ar, dst_ar, imm >> 16));
	return SHL16INSLI(dst_ar, dst_ar, imm);
}

static sljit_s32 emit_const(struct sljit_compiler *compiler, sljit_s32 dst_ar, sljit_sw imm, int flush)
{
	/* Should *not* be optimized as load_immediate, as pcre relocation
	   mechanism will match this fixed 4-instruction pattern. */
	if (flush) {
		FAIL_IF(ADDLI_SOLO(dst_ar, ZERO, imm >> 32));
		FAIL_IF(SHL16INSLI_SOLO(dst_ar, dst_ar, imm >> 16));
		return SHL16INSLI_SOLO(dst_ar, dst_ar, imm);
	}

	FAIL_IF(ADDLI(dst_ar, ZERO, imm >> 32));
	FAIL_IF(SHL16INSLI(dst_ar, dst_ar, imm >> 16));
	return SHL16INSLI(dst_ar, dst_ar, imm);
}

static sljit_s32 emit_const_64(struct sljit_compiler *compiler, sljit_s32 dst_ar, sljit_sw imm, int flush)
{
	/* Should *not* be optimized as load_immediate, as pcre relocation
	   mechanism will match this fixed 4-instruction pattern. */
	if (flush) {
		FAIL_IF(ADDLI_SOLO(reg_map[dst_ar], ZERO, imm >> 48));
		FAIL_IF(SHL16INSLI_SOLO(reg_map[dst_ar], reg_map[dst_ar], imm >> 32));
		FAIL_IF(SHL16INSLI_SOLO(reg_map[dst_ar], reg_map[dst_ar], imm >> 16));
		return SHL16INSLI_SOLO(reg_map[dst_ar], reg_map[dst_ar], imm);
	}

	FAIL_IF(ADDLI(reg_map[dst_ar], ZERO, imm >> 48));
	FAIL_IF(SHL16INSLI(reg_map[dst_ar], reg_map[dst_ar], imm >> 32));
	FAIL_IF(SHL16INSLI(reg_map[dst_ar], reg_map[dst_ar], imm >> 16));
	return SHL16INSLI(reg_map[dst_ar], reg_map[dst_ar], imm);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_ins base;
	sljit_s32 i, tmp;
 
	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1);
	local_size = (local_size + 7) & ~7;
	compiler->local_size = local_size;

	if (local_size <= SIMM_16BIT_MAX) {
		/* Frequent case. */
		FAIL_IF(ADDLI(SLJIT_LOCALS_REG_mapped, SLJIT_LOCALS_REG_mapped, -local_size));
		base = SLJIT_LOCALS_REG_mapped;
	} else {
		FAIL_IF(load_immediate(compiler, TMP_REG1_mapped, local_size));
		FAIL_IF(ADD(TMP_REG2_mapped, SLJIT_LOCALS_REG_mapped, ZERO));
		FAIL_IF(SUB(SLJIT_LOCALS_REG_mapped, SLJIT_LOCALS_REG_mapped, TMP_REG1_mapped));
		base = TMP_REG2_mapped;
		local_size = 0;
	}

	/* Save the return address. */
	FAIL_IF(ADDLI(ADDR_TMP_mapped, base, local_size - 8));
	FAIL_IF(ST_ADD(ADDR_TMP_mapped, RA, -8));

	/* Save the S registers. */
	tmp = saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = SLJIT_S0; i >= tmp; i--) {
		FAIL_IF(ST_ADD(ADDR_TMP_mapped, reg_map[i], -8));
	}

	/* Save the R registers that need to be reserved. */
	for (i = scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		FAIL_IF(ST_ADD(ADDR_TMP_mapped, reg_map[i], -8));
	}

	/* Move the arguments to S registers. */
	for (i = 0; i < args; i++) {
		FAIL_IF(ADD(reg_map[SLJIT_S0 - i], i, ZERO));
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, args, scratches, saveds, fscratches, fsaveds, local_size);

	local_size += GET_SAVED_REGISTERS_SIZE(scratches, saveds, 1);
	compiler->local_size = (local_size + 7) & ~7;

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 local_size;
	sljit_ins base;
	sljit_s32 i, tmp;
	sljit_s32 saveds;

	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));

	FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));

	local_size = compiler->local_size;
	if (local_size <= SIMM_16BIT_MAX)
		base = SLJIT_LOCALS_REG_mapped;
	else {
		FAIL_IF(load_immediate(compiler, TMP_REG1_mapped, local_size));
		FAIL_IF(ADD(TMP_REG1_mapped, SLJIT_LOCALS_REG_mapped, TMP_REG1_mapped));
		base = TMP_REG1_mapped;
		local_size = 0;
	}

	/* Restore the return address. */
	FAIL_IF(ADDLI(ADDR_TMP_mapped, base, local_size - 8));
	FAIL_IF(LD_ADD(RA, ADDR_TMP_mapped, -8));

	/* Restore the S registers. */
	saveds = compiler->saveds;
	tmp = saveds < SLJIT_NUMBER_OF_SAVED_REGISTERS ? (SLJIT_S0 + 1 - saveds) : SLJIT_FIRST_SAVED_REG;
	for (i = SLJIT_S0; i >= tmp; i--) {
		FAIL_IF(LD_ADD(reg_map[i], ADDR_TMP_mapped, -8));
	}

	/* Restore the R registers that need to be reserved. */
	for (i = compiler->scratches; i >= SLJIT_FIRST_SAVED_REG; i--) {
		FAIL_IF(LD_ADD(reg_map[i], ADDR_TMP_mapped, -8));
	}

	if (compiler->local_size <= SIMM_16BIT_MAX)
		FAIL_IF(ADDLI(SLJIT_LOCALS_REG_mapped, SLJIT_LOCALS_REG_mapped, compiler->local_size));
	else
		FAIL_IF(ADD(SLJIT_LOCALS_REG_mapped, TMP_REG1_mapped, ZERO));

	return JR(RA);
}

/* reg_ar is an absoulute register! */

/* Can perform an operation using at most 1 instruction. */
static sljit_s32 getput_arg_fast(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw)
{
	SLJIT_ASSERT(arg & SLJIT_MEM);

	if ((!(flags & WRITE_BACK) || !(arg & REG_MASK))
			&& !(arg & OFFS_REG_MASK) && argw <= SIMM_16BIT_MAX && argw >= SIMM_16BIT_MIN) {
		/* Works for both absoulte and relative addresses. */
		if (SLJIT_UNLIKELY(flags & ARG_TEST))
			return 1;

		FAIL_IF(ADDLI(ADDR_TMP_mapped, reg_map[arg & REG_MASK], argw));

		if (flags & LOAD_DATA)
			FAIL_IF(PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, ADDR_TMP_mapped));
		else
			FAIL_IF(PB2(data_transfer_insts[flags & MEM_MASK], ADDR_TMP_mapped, reg_ar));

		return -1;
	}

	return 0;
}

/* See getput_arg below.
   Note: can_cache is called only for binary operators. Those
   operators always uses word arguments without write back. */
static sljit_s32 can_cache(sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	SLJIT_ASSERT((arg & SLJIT_MEM) && (next_arg & SLJIT_MEM));

	/* Simple operation except for updates. */
	if (arg & OFFS_REG_MASK) {
		argw &= 0x3;
		next_argw &= 0x3;
		if (argw && argw == next_argw
				&& (arg == next_arg || (arg & OFFS_REG_MASK) == (next_arg & OFFS_REG_MASK)))
			return 1;
		return 0;
	}

	if (arg == next_arg) {
		if (((next_argw - argw) <= SIMM_16BIT_MAX
				&& (next_argw - argw) >= SIMM_16BIT_MIN))
			return 1;

		return 0;
	}

	return 0;
}

/* Emit the necessary instructions. See can_cache above. */
static sljit_s32 getput_arg(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw, sljit_s32 next_arg, sljit_sw next_argw)
{
	sljit_s32 tmp_ar, base;

	SLJIT_ASSERT(arg & SLJIT_MEM);
	if (!(next_arg & SLJIT_MEM)) {
		next_arg = 0;
		next_argw = 0;
	}

	if ((flags & MEM_MASK) <= GPR_REG && (flags & LOAD_DATA))
		tmp_ar = reg_ar;
	else
		tmp_ar = TMP_REG1_mapped;

	base = arg & REG_MASK;

	if (SLJIT_UNLIKELY(arg & OFFS_REG_MASK)) {
		argw &= 0x3;

		if ((flags & WRITE_BACK) && reg_ar == reg_map[base]) {
			SLJIT_ASSERT(!(flags & LOAD_DATA) && reg_map[TMP_REG1] != reg_ar);
			FAIL_IF(ADD(TMP_REG1_mapped, reg_ar, ZERO));
			reg_ar = TMP_REG1_mapped;
		}

		/* Using the cache. */
		if (argw == compiler->cache_argw) {
			if (!(flags & WRITE_BACK)) {
				if (arg == compiler->cache_arg) {
					if (flags & LOAD_DATA)
						return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, TMP_REG3_mapped);
					else
						return PB2(data_transfer_insts[flags & MEM_MASK], TMP_REG3_mapped, reg_ar);
				}

				if ((SLJIT_MEM | (arg & OFFS_REG_MASK)) == compiler->cache_arg) {
					if (arg == next_arg && argw == (next_argw & 0x3)) {
						compiler->cache_arg = arg;
						compiler->cache_argw = argw;
						FAIL_IF(ADD(TMP_REG3_mapped, reg_map[base], TMP_REG3_mapped));
						if (flags & LOAD_DATA)
							return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, TMP_REG3_mapped);
						else
							return PB2(data_transfer_insts[flags & MEM_MASK], TMP_REG3_mapped, reg_ar);
					}

					FAIL_IF(ADD(tmp_ar, reg_map[base], TMP_REG3_mapped));
					if (flags & LOAD_DATA)
						return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, tmp_ar);
					else
						return PB2(data_transfer_insts[flags & MEM_MASK], tmp_ar, reg_ar);
				}
			} else {
				if ((SLJIT_MEM | (arg & OFFS_REG_MASK)) == compiler->cache_arg) {
					FAIL_IF(ADD(reg_map[base], reg_map[base], TMP_REG3_mapped));
					if (flags & LOAD_DATA)
						return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, reg_map[base]);
					else
						return PB2(data_transfer_insts[flags & MEM_MASK], reg_map[base], reg_ar);
				}
			}
		}

		if (SLJIT_UNLIKELY(argw)) {
			compiler->cache_arg = SLJIT_MEM | (arg & OFFS_REG_MASK);
			compiler->cache_argw = argw;
			FAIL_IF(SHLI(TMP_REG3_mapped, reg_map[OFFS_REG(arg)], argw));
		}

		if (!(flags & WRITE_BACK)) {
			if (arg == next_arg && argw == (next_argw & 0x3)) {
				compiler->cache_arg = arg;
				compiler->cache_argw = argw;
				FAIL_IF(ADD(TMP_REG3_mapped, reg_map[base], reg_map[!argw ? OFFS_REG(arg) : TMP_REG3]));
				tmp_ar = TMP_REG3_mapped;
			} else
				FAIL_IF(ADD(tmp_ar, reg_map[base], reg_map[!argw ? OFFS_REG(arg) : TMP_REG3]));

			if (flags & LOAD_DATA)
				return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, tmp_ar);
			else
				return PB2(data_transfer_insts[flags & MEM_MASK], tmp_ar, reg_ar);
		}

		FAIL_IF(ADD(reg_map[base], reg_map[base], reg_map[!argw ? OFFS_REG(arg) : TMP_REG3]));

		if (flags & LOAD_DATA)
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, reg_map[base]);
		else
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_map[base], reg_ar);
	}

	if (SLJIT_UNLIKELY(flags & WRITE_BACK) && base) {
		/* Update only applies if a base register exists. */
		if (reg_ar == reg_map[base]) {
			SLJIT_ASSERT(!(flags & LOAD_DATA) && TMP_REG1_mapped != reg_ar);
			if (argw <= SIMM_16BIT_MAX && argw >= SIMM_16BIT_MIN) {
				FAIL_IF(ADDLI(ADDR_TMP_mapped, reg_map[base], argw));
				if (flags & LOAD_DATA)
					FAIL_IF(PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, ADDR_TMP_mapped));
				else
					FAIL_IF(PB2(data_transfer_insts[flags & MEM_MASK], ADDR_TMP_mapped, reg_ar));

				if (argw)
					return ADDLI(reg_map[base], reg_map[base], argw);

				return SLJIT_SUCCESS;
			}

			FAIL_IF(ADD(TMP_REG1_mapped, reg_ar, ZERO));
			reg_ar = TMP_REG1_mapped;
		}

		if (argw <= SIMM_16BIT_MAX && argw >= SIMM_16BIT_MIN) {
			if (argw)
				FAIL_IF(ADDLI(reg_map[base], reg_map[base], argw));
		} else {
			if (compiler->cache_arg == SLJIT_MEM
					&& argw - compiler->cache_argw <= SIMM_16BIT_MAX
					&& argw - compiler->cache_argw >= SIMM_16BIT_MIN) {
				if (argw != compiler->cache_argw) {
					FAIL_IF(ADD(TMP_REG3_mapped, TMP_REG3_mapped, argw - compiler->cache_argw));
					compiler->cache_argw = argw;
				}

				FAIL_IF(ADD(reg_map[base], reg_map[base], TMP_REG3_mapped));
			} else {
				compiler->cache_arg = SLJIT_MEM;
				compiler->cache_argw = argw;
				FAIL_IF(load_immediate(compiler, TMP_REG3_mapped, argw));
				FAIL_IF(ADD(reg_map[base], reg_map[base], TMP_REG3_mapped));
			}
		}

		if (flags & LOAD_DATA)
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, reg_map[base]);
		else
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_map[base], reg_ar);
	}

	if (compiler->cache_arg == arg
			&& argw - compiler->cache_argw <= SIMM_16BIT_MAX
			&& argw - compiler->cache_argw >= SIMM_16BIT_MIN) {
		if (argw != compiler->cache_argw) {
			FAIL_IF(ADDLI(TMP_REG3_mapped, TMP_REG3_mapped, argw - compiler->cache_argw));
			compiler->cache_argw = argw;
		}

		if (flags & LOAD_DATA)
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, TMP_REG3_mapped);
		else
			return PB2(data_transfer_insts[flags & MEM_MASK], TMP_REG3_mapped, reg_ar);
	}

	if (compiler->cache_arg == SLJIT_MEM
			&& argw - compiler->cache_argw <= SIMM_16BIT_MAX
			&& argw - compiler->cache_argw >= SIMM_16BIT_MIN) {
		if (argw != compiler->cache_argw)
			FAIL_IF(ADDLI(TMP_REG3_mapped, TMP_REG3_mapped, argw - compiler->cache_argw));
	} else {
		compiler->cache_arg = SLJIT_MEM;
		FAIL_IF(load_immediate(compiler, TMP_REG3_mapped, argw));
	}

	compiler->cache_argw = argw;

	if (!base) {
		if (flags & LOAD_DATA)
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, TMP_REG3_mapped);
		else
			return PB2(data_transfer_insts[flags & MEM_MASK], TMP_REG3_mapped, reg_ar);
	}

	if (arg == next_arg
			&& next_argw - argw <= SIMM_16BIT_MAX
			&& next_argw - argw >= SIMM_16BIT_MIN) {
		compiler->cache_arg = arg;
		FAIL_IF(ADD(TMP_REG3_mapped, TMP_REG3_mapped, reg_map[base]));
		if (flags & LOAD_DATA)
			return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, TMP_REG3_mapped);
		else
			return PB2(data_transfer_insts[flags & MEM_MASK], TMP_REG3_mapped, reg_ar);
	}

	FAIL_IF(ADD(tmp_ar, TMP_REG3_mapped, reg_map[base]));

	if (flags & LOAD_DATA)
		return PB2(data_transfer_insts[flags & MEM_MASK], reg_ar, tmp_ar);
	else
		return PB2(data_transfer_insts[flags & MEM_MASK], tmp_ar, reg_ar);
}

static SLJIT_INLINE sljit_s32 emit_op_mem(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg_ar, sljit_s32 arg, sljit_sw argw)
{
	if (getput_arg_fast(compiler, flags, reg_ar, arg, argw))
		return compiler->error;

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	return getput_arg(compiler, flags, reg_ar, arg, argw, 0, 0);
}

static SLJIT_INLINE sljit_s32 emit_op_mem2(struct sljit_compiler *compiler, sljit_s32 flags, sljit_s32 reg, sljit_s32 arg1, sljit_sw arg1w, sljit_s32 arg2, sljit_sw arg2w)
{
	if (getput_arg_fast(compiler, flags, reg, arg1, arg1w))
		return compiler->error;
	return getput_arg(compiler, flags, reg, arg1, arg1w, arg2, arg2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	/* For UNUSED dst. Uncommon, but possible. */
	if (dst == SLJIT_UNUSED)
		return SLJIT_SUCCESS;

	if (FAST_IS_REG(dst))
		return ADD(reg_map[dst], RA, ZERO);

	/* Memory. */
	return emit_op_mem(compiler, WORD_DATA, RA, dst, dstw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_return(compiler, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (FAST_IS_REG(src))
		FAIL_IF(ADD(RA, reg_map[src], ZERO));

	else if (src & SLJIT_MEM)
		FAIL_IF(emit_op_mem(compiler, WORD_DATA | LOAD_DATA, RA, src, srcw));

	else if (src & SLJIT_IMM)
		FAIL_IF(load_immediate(compiler, RA, srcw));

	return JR(RA);
}

static SLJIT_INLINE sljit_s32 emit_single_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags, sljit_s32 dst, sljit_s32 src1, sljit_sw src2)
{
	sljit_s32 overflow_ra = 0;

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (dst != src2)
			return ADD(reg_map[dst], reg_map[src2], ZERO);
		return SLJIT_SUCCESS;

	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S32)
				return BFEXTS(reg_map[dst], reg_map[src2], 0, 31);

			return BFEXTU(reg_map[dst], reg_map[src2], 0, 31);
		} else if (dst != src2) {
			SLJIT_ASSERT(src2 == 0);
			return ADD(reg_map[dst], reg_map[src2], ZERO);
		}

		return SLJIT_SUCCESS;

	case SLJIT_MOV_U8:
	case SLJIT_MOV_S8:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S8)
				return BFEXTS(reg_map[dst], reg_map[src2], 0, 7);

			return BFEXTU(reg_map[dst], reg_map[src2], 0, 7);
		} else if (dst != src2) {
			SLJIT_ASSERT(src2 == 0);
			return ADD(reg_map[dst], reg_map[src2], ZERO);
		}

		return SLJIT_SUCCESS;

	case SLJIT_MOV_U16:
	case SLJIT_MOV_S16:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if ((flags & (REG_DEST | REG2_SOURCE)) == (REG_DEST | REG2_SOURCE)) {
			if (op == SLJIT_MOV_S16)
				return BFEXTS(reg_map[dst], reg_map[src2], 0, 15);

			return BFEXTU(reg_map[dst], reg_map[src2], 0, 15);
		} else if (dst != src2) {
			SLJIT_ASSERT(src2 == 0);
			return ADD(reg_map[dst], reg_map[src2], ZERO);
		}

		return SLJIT_SUCCESS;

	case SLJIT_NOT:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (op & SLJIT_SET_E)
			FAIL_IF(NOR(EQUAL_FLAG, reg_map[src2], reg_map[src2]));
		if (CHECK_FLAGS(SLJIT_SET_E))
			FAIL_IF(NOR(reg_map[dst], reg_map[src2], reg_map[src2]));

		return SLJIT_SUCCESS;

	case SLJIT_CLZ:
		SLJIT_ASSERT(src1 == TMP_REG1 && !(flags & SRC2_IMM));
		if (op & SLJIT_SET_E)
			FAIL_IF(CLZ(EQUAL_FLAG, reg_map[src2]));
		if (CHECK_FLAGS(SLJIT_SET_E))
			FAIL_IF(CLZ(reg_map[dst], reg_map[src2]));

		return SLJIT_SUCCESS;

	case SLJIT_ADD:
		if (flags & SRC2_IMM) {
			if (op & SLJIT_SET_O) {
				FAIL_IF(SHRUI(TMP_EREG1, reg_map[src1], 63));
				if (src2 < 0)
					FAIL_IF(XORI(TMP_EREG1, TMP_EREG1, 1));
			}

			if (op & SLJIT_SET_E)
				FAIL_IF(ADDLI(EQUAL_FLAG, reg_map[src1], src2));

			if (op & SLJIT_SET_C) {
				if (src2 >= 0)
					FAIL_IF(ORI(ULESS_FLAG ,reg_map[src1], src2));
				else {
					FAIL_IF(ADDLI(ULESS_FLAG ,ZERO, src2));
					FAIL_IF(OR(ULESS_FLAG,reg_map[src1],ULESS_FLAG));
				}
			}

			/* dst may be the same as src1 or src2. */
			if (CHECK_FLAGS(SLJIT_SET_E))
				FAIL_IF(ADDLI(reg_map[dst], reg_map[src1], src2));

			if (op & SLJIT_SET_O) {
				FAIL_IF(SHRUI(OVERFLOW_FLAG, reg_map[dst], 63));

				if (src2 < 0)
					FAIL_IF(XORI(OVERFLOW_FLAG, OVERFLOW_FLAG, 1));
			}
		} else {
			if (op & SLJIT_SET_O) {
				FAIL_IF(XOR(TMP_EREG1, reg_map[src1], reg_map[src2]));
				FAIL_IF(SHRUI(TMP_EREG1, TMP_EREG1, 63));

				if (src1 != dst)
					overflow_ra = reg_map[src1];
				else if (src2 != dst)
					overflow_ra = reg_map[src2];
				else {
					/* Rare ocasion. */
					FAIL_IF(ADD(TMP_EREG2, reg_map[src1], ZERO));
					overflow_ra = TMP_EREG2;
				}
			}

			if (op & SLJIT_SET_E)
				FAIL_IF(ADD(EQUAL_FLAG ,reg_map[src1], reg_map[src2]));

			if (op & SLJIT_SET_C)
				FAIL_IF(OR(ULESS_FLAG,reg_map[src1], reg_map[src2]));

			/* dst may be the same as src1 or src2. */
			if (CHECK_FLAGS(SLJIT_SET_E))
				FAIL_IF(ADD(reg_map[dst],reg_map[src1], reg_map[src2]));

			if (op & SLJIT_SET_O) {
				FAIL_IF(XOR(OVERFLOW_FLAG,reg_map[dst], overflow_ra));
				FAIL_IF(SHRUI(OVERFLOW_FLAG, OVERFLOW_FLAG, 63));
			}
		}

		/* a + b >= a | b (otherwise, the carry should be set to 1). */
		if (op & SLJIT_SET_C)
			FAIL_IF(CMPLTU(ULESS_FLAG ,reg_map[dst] ,ULESS_FLAG));

		if (op & SLJIT_SET_O)
			return CMOVNEZ(OVERFLOW_FLAG, TMP_EREG1, ZERO);

		return SLJIT_SUCCESS;

	case SLJIT_ADDC:
		if (flags & SRC2_IMM) {
			if (op & SLJIT_SET_C) {
				if (src2 >= 0)
					FAIL_IF(ORI(TMP_EREG1, reg_map[src1], src2));
				else {
					FAIL_IF(ADDLI(TMP_EREG1, ZERO, src2));
					FAIL_IF(OR(TMP_EREG1, reg_map[src1], TMP_EREG1));
				}
			}

			FAIL_IF(ADDLI(reg_map[dst], reg_map[src1], src2));

		} else {
			if (op & SLJIT_SET_C)
				FAIL_IF(OR(TMP_EREG1, reg_map[src1], reg_map[src2]));

			/* dst may be the same as src1 or src2. */
			FAIL_IF(ADD(reg_map[dst], reg_map[src1], reg_map[src2]));
		}

		if (op & SLJIT_SET_C)
			FAIL_IF(CMPLTU(TMP_EREG1, reg_map[dst], TMP_EREG1));

		FAIL_IF(ADD(reg_map[dst], reg_map[dst], ULESS_FLAG));

		if (!(op & SLJIT_SET_C))
			return SLJIT_SUCCESS;

		/* Set TMP_EREG2 (dst == 0) && (ULESS_FLAG == 1). */
		FAIL_IF(CMPLTUI(TMP_EREG2, reg_map[dst], 1));
		FAIL_IF(AND(TMP_EREG2, TMP_EREG2, ULESS_FLAG));
		/* Set carry flag. */
		return OR(ULESS_FLAG, TMP_EREG2, TMP_EREG1);

	case SLJIT_SUB:
		if ((flags & SRC2_IMM) && ((op & (SLJIT_SET_U | SLJIT_SET_S)) || src2 == SIMM_16BIT_MIN)) {
			FAIL_IF(ADDLI(TMP_REG2_mapped, ZERO, src2));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		if (flags & SRC2_IMM) {
			if (op & SLJIT_SET_O) {
				FAIL_IF(SHRUI(TMP_EREG1,reg_map[src1], 63));

				if (src2 < 0)
					FAIL_IF(XORI(TMP_EREG1, TMP_EREG1, 1));

				if (src1 != dst)
					overflow_ra = reg_map[src1];
				else {
					/* Rare ocasion. */
					FAIL_IF(ADD(TMP_EREG2, reg_map[src1], ZERO));
					overflow_ra = TMP_EREG2;
				}
			}

			if (op & SLJIT_SET_E)
				FAIL_IF(ADDLI(EQUAL_FLAG, reg_map[src1], -src2));

			if (op & SLJIT_SET_C) {
				FAIL_IF(load_immediate(compiler, ADDR_TMP_mapped, src2));
				FAIL_IF(CMPLTU(ULESS_FLAG, reg_map[src1], ADDR_TMP_mapped));
			}

			/* dst may be the same as src1 or src2. */
			if (CHECK_FLAGS(SLJIT_SET_E))
				FAIL_IF(ADDLI(reg_map[dst], reg_map[src1], -src2));

		} else {

			if (op & SLJIT_SET_O) {
				FAIL_IF(XOR(TMP_EREG1, reg_map[src1], reg_map[src2]));
				FAIL_IF(SHRUI(TMP_EREG1, TMP_EREG1, 63));

				if (src1 != dst)
					overflow_ra = reg_map[src1];
				else {
					/* Rare ocasion. */
					FAIL_IF(ADD(TMP_EREG2, reg_map[src1], ZERO));
					overflow_ra = TMP_EREG2;
				}
			}

			if (op & SLJIT_SET_E)
				FAIL_IF(SUB(EQUAL_FLAG, reg_map[src1], reg_map[src2]));

			if (op & (SLJIT_SET_U | SLJIT_SET_C))
				FAIL_IF(CMPLTU(ULESS_FLAG, reg_map[src1], reg_map[src2]));

			if (op & SLJIT_SET_U)
				FAIL_IF(CMPLTU(UGREATER_FLAG, reg_map[src2], reg_map[src1]));

			if (op & SLJIT_SET_S) {
				FAIL_IF(CMPLTS(LESS_FLAG ,reg_map[src1] ,reg_map[src2]));
				FAIL_IF(CMPLTS(GREATER_FLAG ,reg_map[src2] ,reg_map[src1]));
			}

			/* dst may be the same as src1 or src2. */
			if (CHECK_FLAGS(SLJIT_SET_E | SLJIT_SET_U | SLJIT_SET_S | SLJIT_SET_C))
				FAIL_IF(SUB(reg_map[dst], reg_map[src1], reg_map[src2]));
		}

		if (op & SLJIT_SET_O) {
			FAIL_IF(XOR(OVERFLOW_FLAG, reg_map[dst], overflow_ra));
			FAIL_IF(SHRUI(OVERFLOW_FLAG, OVERFLOW_FLAG, 63));
			return CMOVEQZ(OVERFLOW_FLAG, TMP_EREG1, ZERO);
		}

		return SLJIT_SUCCESS;

	case SLJIT_SUBC:
		if ((flags & SRC2_IMM) && src2 == SIMM_16BIT_MIN) {
			FAIL_IF(ADDLI(TMP_REG2_mapped, ZERO, src2));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		if (flags & SRC2_IMM) {
			if (op & SLJIT_SET_C) {
				FAIL_IF(load_immediate(compiler, ADDR_TMP_mapped, -src2));
				FAIL_IF(CMPLTU(TMP_EREG1, reg_map[src1], ADDR_TMP_mapped));
			}

			/* dst may be the same as src1 or src2. */
			FAIL_IF(ADDLI(reg_map[dst], reg_map[src1], -src2));

		} else {
			if (op & SLJIT_SET_C)
				FAIL_IF(CMPLTU(TMP_EREG1, reg_map[src1], reg_map[src2]));
				/* dst may be the same as src1 or src2. */
			FAIL_IF(SUB(reg_map[dst], reg_map[src1], reg_map[src2]));
		}

		if (op & SLJIT_SET_C)
			FAIL_IF(CMOVEQZ(TMP_EREG1, reg_map[dst], ULESS_FLAG));

		FAIL_IF(SUB(reg_map[dst], reg_map[dst], ULESS_FLAG));

		if (op & SLJIT_SET_C)
			FAIL_IF(ADD(ULESS_FLAG, TMP_EREG1, ZERO));

		return SLJIT_SUCCESS;

	case SLJIT_MUL:
		if (flags & SRC2_IMM) {
			FAIL_IF(load_immediate(compiler, TMP_REG2_mapped, src2));
			src2 = TMP_REG2;
			flags &= ~SRC2_IMM;
		}

		FAIL_IF(MUL(reg_map[dst], reg_map[src1], reg_map[src2]));

		return SLJIT_SUCCESS;

#define EMIT_LOGICAL(op_imm, op_norm) \
	if (flags & SRC2_IMM) { \
		FAIL_IF(load_immediate(compiler, ADDR_TMP_mapped, src2)); \
		if (op & SLJIT_SET_E) \
			FAIL_IF(push_3_buffer( \
				compiler, op_norm, EQUAL_FLAG, reg_map[src1], \
				ADDR_TMP_mapped, __LINE__)); \
		if (CHECK_FLAGS(SLJIT_SET_E)) \
			FAIL_IF(push_3_buffer( \
				compiler, op_norm, reg_map[dst], reg_map[src1], \
				ADDR_TMP_mapped, __LINE__)); \
	} else { \
		if (op & SLJIT_SET_E) \
			FAIL_IF(push_3_buffer( \
				compiler, op_norm, EQUAL_FLAG, reg_map[src1], \
				reg_map[src2], __LINE__)); \
		if (CHECK_FLAGS(SLJIT_SET_E)) \
			FAIL_IF(push_3_buffer( \
				compiler, op_norm, reg_map[dst], reg_map[src1], \
				reg_map[src2], __LINE__)); \
	}

	case SLJIT_AND:
		EMIT_LOGICAL(TILEGX_OPC_ANDI, TILEGX_OPC_AND);
		return SLJIT_SUCCESS;

	case SLJIT_OR:
		EMIT_LOGICAL(TILEGX_OPC_ORI, TILEGX_OPC_OR);
		return SLJIT_SUCCESS;

	case SLJIT_XOR:
		EMIT_LOGICAL(TILEGX_OPC_XORI, TILEGX_OPC_XOR);
		return SLJIT_SUCCESS;

#define EMIT_SHIFT(op_imm, op_norm) \
	if (flags & SRC2_IMM) { \
		if (op & SLJIT_SET_E) \
			FAIL_IF(push_3_buffer( \
				compiler, op_imm, EQUAL_FLAG, reg_map[src1], \
				src2 & 0x3F, __LINE__)); \
		if (CHECK_FLAGS(SLJIT_SET_E)) \
			FAIL_IF(push_3_buffer( \
				compiler, op_imm, reg_map[dst], reg_map[src1], \
				src2 & 0x3F, __LINE__)); \
	} else { \
		if (op & SLJIT_SET_E) \
			FAIL_IF(push_3_buffer( \
				compiler, op_norm, EQUAL_FLAG, reg_map[src1], \
				reg_map[src2], __LINE__)); \
		if (CHECK_FLAGS(SLJIT_SET_E)) \
			FAIL_IF(push_3_buffer( \
				compiler, op_norm, reg_map[dst], reg_map[src1], \
				reg_map[src2], __LINE__)); \
	}

	case SLJIT_SHL:
		EMIT_SHIFT(TILEGX_OPC_SHLI, TILEGX_OPC_SHL);
		return SLJIT_SUCCESS;

	case SLJIT_LSHR:
		EMIT_SHIFT(TILEGX_OPC_SHRUI, TILEGX_OPC_SHRU);
		return SLJIT_SUCCESS;

	case SLJIT_ASHR:
		EMIT_SHIFT(TILEGX_OPC_SHRSI, TILEGX_OPC_SHRS);
		return SLJIT_SUCCESS;
	}

	SLJIT_UNREACHABLE();
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_op(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 flags, sljit_s32 dst, sljit_sw dstw, sljit_s32 src1, sljit_sw src1w, sljit_s32 src2, sljit_sw src2w)
{
	/* arg1 goes to TMP_REG1 or src reg.
	   arg2 goes to TMP_REG2, imm or src reg.
	   TMP_REG3 can be used for caching.
	   result goes to TMP_REG2, so put result can use TMP_REG1 and TMP_REG3. */
	sljit_s32 dst_r = TMP_REG2;
	sljit_s32 src1_r;
	sljit_sw src2_r = 0;
	sljit_s32 sugg_src2_r = TMP_REG2;

	if (!(flags & ALT_KEEP_CACHE)) {
		compiler->cache_arg = 0;
		compiler->cache_argw = 0;
	}

	if (SLJIT_UNLIKELY(dst == SLJIT_UNUSED)) {
		if (op >= SLJIT_MOV && op <= SLJIT_MOVU_S32 && !(src2 & SLJIT_MEM))
			return SLJIT_SUCCESS;
		if (GET_FLAGS(op))
			flags |= UNUSED_DEST;
	} else if (FAST_IS_REG(dst)) {
		dst_r = dst;
		flags |= REG_DEST;
		if (op >= SLJIT_MOV && op <= SLJIT_MOVU_S32)
			sugg_src2_r = dst_r;
	} else if ((dst & SLJIT_MEM) && !getput_arg_fast(compiler, flags | ARG_TEST, TMP_REG1_mapped, dst, dstw))
		flags |= SLOW_DEST;

	if (flags & IMM_OP) {
		if ((src2 & SLJIT_IMM) && src2w) {
			if ((!(flags & LOGICAL_OP)
					&& (src2w <= SIMM_16BIT_MAX && src2w >= SIMM_16BIT_MIN))
					|| ((flags & LOGICAL_OP) && !(src2w & ~UIMM_16BIT_MAX))) {
				flags |= SRC2_IMM;
				src2_r = src2w;
			}
		}

		if (!(flags & SRC2_IMM) && (flags & CUMULATIVE_OP) && (src1 & SLJIT_IMM) && src1w) {
			if ((!(flags & LOGICAL_OP)
					&& (src1w <= SIMM_16BIT_MAX && src1w >= SIMM_16BIT_MIN))
					|| ((flags & LOGICAL_OP) && !(src1w & ~UIMM_16BIT_MAX))) {
				flags |= SRC2_IMM;
				src2_r = src1w;

				/* And swap arguments. */
				src1 = src2;
				src1w = src2w;
				src2 = SLJIT_IMM;
				/* src2w = src2_r unneeded. */
			}
		}
	}

	/* Source 1. */
	if (FAST_IS_REG(src1)) {
		src1_r = src1;
		flags |= REG1_SOURCE;
	} else if (src1 & SLJIT_IMM) {
		if (src1w) {
			FAIL_IF(load_immediate(compiler, TMP_REG1_mapped, src1w));
			src1_r = TMP_REG1;
		} else
			src1_r = 0;
	} else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, TMP_REG1_mapped, src1, src1w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC1;
		src1_r = TMP_REG1;
	}

	/* Source 2. */
	if (FAST_IS_REG(src2)) {
		src2_r = src2;
		flags |= REG2_SOURCE;
		if (!(flags & REG_DEST) && op >= SLJIT_MOV && op <= SLJIT_MOVU_S32)
			dst_r = src2_r;
	} else if (src2 & SLJIT_IMM) {
		if (!(flags & SRC2_IMM)) {
			if (src2w) {
				FAIL_IF(load_immediate(compiler, reg_map[sugg_src2_r], src2w));
				src2_r = sugg_src2_r;
			} else {
				src2_r = 0;
				if ((op >= SLJIT_MOV && op <= SLJIT_MOVU_S32) && (dst & SLJIT_MEM))
					dst_r = 0;
			}
		}
	} else {
		if (getput_arg_fast(compiler, flags | LOAD_DATA, reg_map[sugg_src2_r], src2, src2w))
			FAIL_IF(compiler->error);
		else
			flags |= SLOW_SRC2;
		src2_r = sugg_src2_r;
	}

	if ((flags & (SLOW_SRC1 | SLOW_SRC2)) == (SLOW_SRC1 | SLOW_SRC2)) {
		SLJIT_ASSERT(src2_r == TMP_REG2);
		if (!can_cache(src1, src1w, src2, src2w) && can_cache(src1, src1w, dst, dstw)) {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2_mapped, src2, src2w, src1, src1w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1_mapped, src1, src1w, dst, dstw));
		} else {
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1_mapped, src1, src1w, src2, src2w));
			FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG2_mapped, src2, src2w, dst, dstw));
		}
	} else if (flags & SLOW_SRC1)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, TMP_REG1_mapped, src1, src1w, dst, dstw));
	else if (flags & SLOW_SRC2)
		FAIL_IF(getput_arg(compiler, flags | LOAD_DATA, reg_map[sugg_src2_r], src2, src2w, dst, dstw));

	FAIL_IF(emit_single_op(compiler, op, flags, dst_r, src1_r, src2_r));

	if (dst & SLJIT_MEM) {
		if (!(flags & SLOW_DEST)) {
			getput_arg_fast(compiler, flags, reg_map[dst_r], dst, dstw);
			return compiler->error;
		}

		return getput_arg(compiler, flags, reg_map[dst_r], dst, dstw, 0, 0);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw dstw, sljit_s32 src, sljit_sw srcw, sljit_s32 type)
{
	sljit_s32 sugg_dst_ar, dst_ar;
	sljit_s32 flags = GET_ALL_FLAGS(op);
	sljit_s32 mem_type = (op & SLJIT_I32_OP) ? (INT_DATA | SIGNED_DATA) : WORD_DATA;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, src, srcw, type));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	op = GET_OPCODE(op);
	if (op == SLJIT_MOV_S32 || op == SLJIT_MOV_U32)
		mem_type = INT_DATA | SIGNED_DATA;
	sugg_dst_ar = reg_map[(op < SLJIT_ADD && FAST_IS_REG(dst)) ? dst : TMP_REG2];

	compiler->cache_arg = 0;
	compiler->cache_argw = 0;
	if (op >= SLJIT_ADD && (src & SLJIT_MEM)) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(emit_op_mem2(compiler, mem_type | LOAD_DATA, TMP_REG1_mapped, src, srcw, dst, dstw));
		src = TMP_REG1;
		srcw = 0;
	}

	switch (type & 0xff) {
	case SLJIT_EQUAL:
	case SLJIT_NOT_EQUAL:
		FAIL_IF(CMPLTUI(sugg_dst_ar, EQUAL_FLAG, 1));
		dst_ar = sugg_dst_ar;
		break;
	case SLJIT_LESS:
	case SLJIT_GREATER_EQUAL:
		dst_ar = ULESS_FLAG;
		break;
	case SLJIT_GREATER:
	case SLJIT_LESS_EQUAL:
		dst_ar = UGREATER_FLAG;
		break;
	case SLJIT_SIG_LESS:
	case SLJIT_SIG_GREATER_EQUAL:
		dst_ar = LESS_FLAG;
		break;
	case SLJIT_SIG_GREATER:
	case SLJIT_SIG_LESS_EQUAL:
		dst_ar = GREATER_FLAG;
		break;
	case SLJIT_OVERFLOW:
	case SLJIT_NOT_OVERFLOW:
		dst_ar = OVERFLOW_FLAG;
		break;
	case SLJIT_MUL_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		FAIL_IF(CMPLTUI(sugg_dst_ar, OVERFLOW_FLAG, 1));
		dst_ar = sugg_dst_ar;
		type ^= 0x1; /* Flip type bit for the XORI below. */
		break;

	default:
		SLJIT_UNREACHABLE();
		dst_ar = sugg_dst_ar;
		break;
	}

	if (type & 0x1) {
		FAIL_IF(XORI(sugg_dst_ar, dst_ar, 1));
		dst_ar = sugg_dst_ar;
	}

	if (op >= SLJIT_ADD) {
		if (TMP_REG2_mapped != dst_ar)
			FAIL_IF(ADD(TMP_REG2_mapped, dst_ar, ZERO));
		return emit_op(compiler, op | flags, mem_type | CUMULATIVE_OP | LOGICAL_OP | IMM_OP | ALT_KEEP_CACHE, dst, dstw, src, srcw, TMP_REG2, 0);
	}

	if (dst & SLJIT_MEM)
		return emit_op_mem(compiler, mem_type, dst_ar, dst, dstw);

	if (sugg_dst_ar != dst_ar)
		return ADD(sugg_dst_ar, dst_ar, ZERO);

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op) {
	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	op = GET_OPCODE(op);
	switch (op) {
	case SLJIT_NOP:
		return push_0_buffer(compiler, TILEGX_OPC_FNOP, __LINE__);

	case SLJIT_BREAKPOINT:
		return PI(BPT);

	case SLJIT_LMUL_UW:
	case SLJIT_LMUL_SW:
	case SLJIT_DIVMOD_UW:
	case SLJIT_DIVMOD_SW:
	case SLJIT_DIV_UW:
	case SLJIT_DIV_SW:
		SLJIT_UNREACHABLE();
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw dstw, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		return emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_U32:
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_S32:
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOV_U8:
		return emit_op(compiler, SLJIT_MOV_U8, BYTE_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u8) srcw : srcw);

	case SLJIT_MOV_S8:
		return emit_op(compiler, SLJIT_MOV_S8, BYTE_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s8) srcw : srcw);

	case SLJIT_MOV_U16:
		return emit_op(compiler, SLJIT_MOV_U16, HALF_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u16) srcw : srcw);

	case SLJIT_MOV_S16:
		return emit_op(compiler, SLJIT_MOV_S16, HALF_DATA | SIGNED_DATA, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s16) srcw : srcw);

	case SLJIT_MOVU:
	case SLJIT_MOVU_P:
		return emit_op(compiler, SLJIT_MOV, WORD_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOVU_U32:
		return emit_op(compiler, SLJIT_MOV_U32, INT_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOVU_S32:
		return emit_op(compiler, SLJIT_MOV_S32, INT_DATA | SIGNED_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_MOVU_U8:
		return emit_op(compiler, SLJIT_MOV_U8, BYTE_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u8) srcw : srcw);

	case SLJIT_MOVU_S8:
		return emit_op(compiler, SLJIT_MOV_S8, BYTE_DATA | SIGNED_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s8) srcw : srcw);

	case SLJIT_MOVU_U16:
		return emit_op(compiler, SLJIT_MOV_U16, HALF_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_u16) srcw : srcw);

	case SLJIT_MOVU_S16:
		return emit_op(compiler, SLJIT_MOV_S16, HALF_DATA | SIGNED_DATA | WRITE_BACK, dst, dstw, TMP_REG1, 0, src, (src & SLJIT_IMM) ? (sljit_s16) srcw : srcw);

	case SLJIT_NOT:
		return emit_op(compiler, op, 0, dst, dstw, TMP_REG1, 0, src, srcw);

	case SLJIT_NEG:
		return emit_op(compiler, SLJIT_SUB | GET_ALL_FLAGS(op), IMM_OP, dst, dstw, SLJIT_IMM, 0, src, srcw);

	case SLJIT_CLZ:
		return emit_op(compiler, op, (op & SLJIT_I32_OP) ? INT_DATA : WORD_DATA, dst, dstw, TMP_REG1, 0, src, srcw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw dstw, sljit_s32 src1, sljit_sw src1w, sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
		return emit_op(compiler, op, CUMULATIVE_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SUB:
	case SLJIT_SUBC:
		return emit_op(compiler, op, IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_MUL:
		return emit_op(compiler, op, CUMULATIVE_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		return emit_op(compiler, op, CUMULATIVE_OP | LOGICAL_OP | IMM_OP, dst, dstw, src1, src1w, src2, src2w);

	case SLJIT_SHL:
	case SLJIT_LSHR:
	case SLJIT_ASHR:
		if (src2 & SLJIT_IMM)
			src2w &= 0x3f;
		if (op & SLJIT_I32_OP)
			src2w &= 0x1f;

		return emit_op(compiler, op, IMM_OP, dst, dstw, src1, src1w, src2, src2w);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label * sljit_emit_label(struct sljit_compiler *compiler)
{
	struct sljit_label *label;

	flush_buffer(compiler);

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_label(compiler));

	if (compiler->last_label && compiler->last_label->size == compiler->size)
		return compiler->last_label;

	label = (struct sljit_label *)ensure_abuf(compiler, sizeof(struct sljit_label));
	PTR_FAIL_IF(!label);
	set_label(label, compiler);
	return label;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 src_r = TMP_REG2;
	struct sljit_jump *jump = NULL;

	flush_buffer(compiler);

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (FAST_IS_REG(src)) {
		if (reg_map[src] != 0)
			src_r = src;
		else
			FAIL_IF(ADD_SOLO(TMP_REG2_mapped, reg_map[src], ZERO));
	}

	if (type >= SLJIT_CALL0) {
		SLJIT_ASSERT(reg_map[PIC_ADDR_REG] == 16 && PIC_ADDR_REG == TMP_REG2);
		if (src & (SLJIT_IMM | SLJIT_MEM)) {
			if (src & SLJIT_IMM)
				FAIL_IF(emit_const(compiler, reg_map[PIC_ADDR_REG], srcw, 1));
			else {
				SLJIT_ASSERT(src_r == TMP_REG2 && (src & SLJIT_MEM));
				FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, TMP_REG2, 0, TMP_REG1, 0, src, srcw));
			}

			FAIL_IF(ADD_SOLO(0, reg_map[SLJIT_R0], ZERO));

			FAIL_IF(ADDI_SOLO(54, 54, -16));

			FAIL_IF(JALR_SOLO(reg_map[PIC_ADDR_REG]));

			return ADDI_SOLO(54, 54, 16);
		}

		/* Register input. */
		if (type >= SLJIT_CALL1)
			FAIL_IF(ADD_SOLO(0, reg_map[SLJIT_R0], ZERO));

		FAIL_IF(ADD_SOLO(reg_map[PIC_ADDR_REG], reg_map[src_r], ZERO));

		FAIL_IF(ADDI_SOLO(54, 54, -16));

		FAIL_IF(JALR_SOLO(reg_map[src_r]));

		return ADDI_SOLO(54, 54, 16);
	}

	if (src & SLJIT_IMM) {
		jump = (struct sljit_jump *)ensure_abuf(compiler, sizeof(struct sljit_jump));
		FAIL_IF(!jump);
		set_jump(jump, compiler, JUMP_ADDR | ((type >= SLJIT_FAST_CALL) ? IS_JAL : 0));
		jump->u.target = srcw;
		FAIL_IF(emit_const(compiler, TMP_REG2_mapped, 0, 1));

		if (type >= SLJIT_FAST_CALL) {
			FAIL_IF(ADD_SOLO(ZERO, ZERO, ZERO));
			jump->addr = compiler->size;
			FAIL_IF(JR_SOLO(reg_map[src_r]));
		} else {
			jump->addr = compiler->size;
			FAIL_IF(JR_SOLO(reg_map[src_r]));
		}

		return SLJIT_SUCCESS;

	} else if (src & SLJIT_MEM) {
		FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, TMP_REG2, 0, TMP_REG1, 0, src, srcw));
		flush_buffer(compiler);
	}

	FAIL_IF(JR_SOLO(reg_map[src_r]));

	if (jump)
		jump->addr = compiler->size;

	return SLJIT_SUCCESS;
}

#define BR_Z(src) \
	inst = BEQZ_X1 | SRCA_X1(src); \
	flags = IS_COND;

#define BR_NZ(src) \
	inst = BNEZ_X1 | SRCA_X1(src); \
	flags = IS_COND;

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump * sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	struct sljit_jump *jump;
	sljit_ins inst;
	sljit_s32 flags = 0;

	flush_buffer(compiler);

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	jump = (struct sljit_jump *)ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	type &= 0xff;

	switch (type) {
	case SLJIT_EQUAL:
		BR_NZ(EQUAL_FLAG);
		break;
	case SLJIT_NOT_EQUAL:
		BR_Z(EQUAL_FLAG);
		break;
	case SLJIT_LESS:
		BR_Z(ULESS_FLAG);
		break;
	case SLJIT_GREATER_EQUAL:
		BR_NZ(ULESS_FLAG);
		break;
	case SLJIT_GREATER:
		BR_Z(UGREATER_FLAG);
		break;
	case SLJIT_LESS_EQUAL:
		BR_NZ(UGREATER_FLAG);
		break;
	case SLJIT_SIG_LESS:
		BR_Z(LESS_FLAG);
		break;
	case SLJIT_SIG_GREATER_EQUAL:
		BR_NZ(LESS_FLAG);
		break;
	case SLJIT_SIG_GREATER:
		BR_Z(GREATER_FLAG);
		break;
	case SLJIT_SIG_LESS_EQUAL:
		BR_NZ(GREATER_FLAG);
		break;
	case SLJIT_OVERFLOW:
	case SLJIT_MUL_OVERFLOW:
		BR_Z(OVERFLOW_FLAG);
		break;
	case SLJIT_NOT_OVERFLOW:
	case SLJIT_MUL_NOT_OVERFLOW:
		BR_NZ(OVERFLOW_FLAG);
		break;
	default:
		/* Not conditional branch. */
		inst = 0;
		break;
	}

	jump->flags |= flags;

	if (inst) {
		inst = inst | ((type <= SLJIT_JUMP) ? BOFF_X1(5) : BOFF_X1(6));
		PTR_FAIL_IF(PI(inst));
	}

	PTR_FAIL_IF(emit_const(compiler, TMP_REG2_mapped, 0, 1));
	if (type <= SLJIT_JUMP) {
		jump->addr = compiler->size;
		PTR_FAIL_IF(JR_SOLO(TMP_REG2_mapped));
	} else {
		SLJIT_ASSERT(reg_map[PIC_ADDR_REG] == 16 && PIC_ADDR_REG == TMP_REG2);
		/* Cannot be optimized out if type is >= CALL0. */
		jump->flags |= IS_JAL | (type >= SLJIT_CALL0 ? SLJIT_REWRITABLE_JUMP : 0);
		PTR_FAIL_IF(ADD_SOLO(0, reg_map[SLJIT_R0], ZERO));
		jump->addr = compiler->size;
		PTR_FAIL_IF(JALR_SOLO(TMP_REG2_mapped));
	}

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw dstw, sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 dst, sljit_sw dstw, sljit_s32 src1, sljit_sw src1w, sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const * sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	struct sljit_const *const_;
	sljit_s32 reg;

	flush_buffer(compiler);

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	const_ = (struct sljit_const *)ensure_abuf(compiler, sizeof(struct sljit_const));
	PTR_FAIL_IF(!const_);
	set_const(const_, compiler);

	reg = FAST_IS_REG(dst) ? dst : TMP_REG2;

	PTR_FAIL_IF(emit_const_64(compiler, reg, init_value, 1));

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(emit_op(compiler, SLJIT_MOV, WORD_DATA, dst, dstw, TMP_REG1, 0, TMP_REG2, 0));
	return const_;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target)
{
	sljit_ins *inst = (sljit_ins *)addr;

	inst[0] = (inst[0] & ~(0xFFFFL << 43)) | (((new_target >> 32) & 0xffff) << 43);
	inst[1] = (inst[1] & ~(0xFFFFL << 43)) | (((new_target >> 16) & 0xffff) << 43);
	inst[2] = (inst[2] & ~(0xFFFFL << 43)) | ((new_target & 0xffff) << 43);
	SLJIT_CACHE_FLUSH(inst, inst + 3);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant)
{
	sljit_ins *inst = (sljit_ins *)addr;

	inst[0] = (inst[0] & ~(0xFFFFL << 43)) | (((new_constant >> 48) & 0xFFFFL) << 43);
	inst[1] = (inst[1] & ~(0xFFFFL << 43)) | (((new_constant >> 32) & 0xFFFFL) << 43);
	inst[2] = (inst[2] & ~(0xFFFFL << 43)) | (((new_constant >> 16) & 0xFFFFL) << 43);
	inst[3] = (inst[3] & ~(0xFFFFL << 43)) | ((new_constant & 0xFFFFL) << 43);
	SLJIT_CACHE_FLUSH(inst, inst + 4);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(reg));
	return reg_map[reg];
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_s32 size)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));
	return SLJIT_ERR_UNSUPPORTED;
}

