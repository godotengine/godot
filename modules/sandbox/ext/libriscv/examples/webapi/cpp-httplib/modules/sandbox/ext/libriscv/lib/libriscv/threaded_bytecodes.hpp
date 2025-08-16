#pragma once
#include "libriscv_settings.h"

namespace riscv
{
	// Bytecodes for threaded simulation
	enum
	{
		RV32I_BC_INVALID = 0,
		RV32I_BC_ADDI,
		RV32I_BC_LI,
		RV32I_BC_MV,

		RV32I_BC_SLLI,
		RV32I_BC_SLTI,
		RV32I_BC_SLTIU,
		RV32I_BC_XORI,
		RV32I_BC_SRLI,
		RV32I_BC_SRAI,
		RV32I_BC_ORI,
		RV32I_BC_ANDI,

		RV32I_BC_LUI,
		RV32I_BC_AUIPC,

		RV32I_BC_LDB,
		RV32I_BC_LDBU,
		RV32I_BC_LDH,
		RV32I_BC_LDHU,
		RV32I_BC_LDW,

		RV32I_BC_STB,
		RV32I_BC_STH,
		RV32I_BC_STW,

#ifdef RISCV_64I
		RV32I_BC_LDWU,
		RV32I_BC_LDD,
		RV32I_BC_STD,
#endif

		RV32I_BC_BEQ,
		RV32I_BC_BNE,
		RV32I_BC_BLT,
		RV32I_BC_BGE,
		RV32I_BC_BLTU,
		RV32I_BC_BGEU,
		RV32I_BC_BEQ_FW,
		RV32I_BC_BNE_FW,

		RV32I_BC_JAL,
		RV32I_BC_JALR,
		RV32I_BC_FAST_JAL,
		RV32I_BC_FAST_CALL,

		RV32I_BC_OP_ADD,
		RV32I_BC_OP_SUB,
		RV32I_BC_OP_SLL,
		RV32I_BC_OP_SLT,
		RV32I_BC_OP_SLTU,
		RV32I_BC_OP_XOR,
		RV32I_BC_OP_SRL,
		RV32I_BC_OP_OR,
		RV32I_BC_OP_AND,
		RV32I_BC_OP_MUL,
		RV32I_BC_OP_DIV,
		RV32I_BC_OP_DIVU,
		RV32I_BC_OP_REM,
		RV32I_BC_OP_REMU,
		RV32I_BC_OP_SRA,
		RV32I_BC_OP_ZEXT_H,
		RV32I_BC_OP_SH1ADD,
		RV32I_BC_OP_SH2ADD,
		RV32I_BC_OP_SH3ADD,

		RV32I_BC_SEXT_B,
		RV32I_BC_SEXT_H,
		RV32I_BC_BSETI,
		RV32I_BC_BEXTI,

#ifdef RISCV_64I
		RV64I_BC_ADDIW,
		RV64I_BC_SLLIW,
		RV64I_BC_SRLIW,
		RV64I_BC_SRAIW,
		RV64I_BC_OP_ADDW,
		RV64I_BC_OP_SUBW,
		RV64I_BC_OP_MULW,
		RV64I_BC_OP_ADD_UW,
		RV64I_BC_OP_SH1ADD_UW,
		RV64I_BC_OP_SH2ADD_UW,
#endif

#ifdef RISCV_EXT_COMPRESSED
		RV32C_BC_ADDI,
		RV32C_BC_LI,
		RV32C_BC_MV,
		RV32C_BC_SLLI,
		RV32C_BC_BEQZ,
		RV32C_BC_BNEZ,
		RV32C_BC_JMP,
		RV32C_BC_JR,
		RV32C_BC_JAL_ADDIW,
		RV32C_BC_JALR,
		RV32C_BC_LDD,
		RV32C_BC_STD,
		RV32C_BC_LDW,
		RV32C_BC_STW,
		RV32C_BC_SRLI,
		RV32C_BC_ANDI,
		RV32C_BC_ADD,
		RV32C_BC_XOR,
		RV32C_BC_OR,
		RV32C_BC_FUNCTION,
#endif

		RV32I_BC_SYSCALL,
		RV32I_BC_STOP,

		RV32F_BC_FLW,
		RV32F_BC_FLD,
		RV32F_BC_FSW,
		RV32F_BC_FSD,
		RV32F_BC_FADD,
		RV32F_BC_FSUB,
		RV32F_BC_FMUL,
		RV32F_BC_FDIV,
		RV32F_BC_FMADD,
#ifdef RISCV_EXT_VECTOR
		RV32V_BC_VLE32,
		RV32V_BC_VSE32,
		RV32V_BC_VFADD_VV,
		RV32V_BC_VFMUL_VF,
#endif
		RV32I_BC_FUNCTION,
		RV32I_BC_FUNCBLOCK,
#ifdef RISCV_BINARY_TRANSLATION
		RV32I_BC_TRANSLATOR,
#endif
		RV32I_BC_LIVEPATCH,
		RV32I_BC_SYSTEM,
		BYTECODES_MAX
	};
	static_assert(BYTECODES_MAX <= 256, "A bytecode must fit in a byte");

	union FasterItype
	{
		uint32_t whole;

		struct
		{
			uint16_t imm;
			uint8_t  rs2;
			uint8_t  rs1;
		};

		RISCV_ALWAYS_INLINE
		auto get_rs1() const noexcept {
			return rs1;
		}
		RISCV_ALWAYS_INLINE
		auto get_rs2() const noexcept {
			return rs2;
		}
		RISCV_ALWAYS_INLINE
		int32_t signed_imm() const noexcept {
			return (int16_t)imm;
		}
		RISCV_ALWAYS_INLINE
		auto unsigned_imm() const noexcept {
			return (uint16_t)imm;
		}
	};

	union FasterOpType
	{
		uint32_t whole;

		struct
		{
			uint16_t rd;
			uint8_t rs2;
			uint8_t rs1;
		};

		RISCV_ALWAYS_INLINE
		auto get_rd() const noexcept {
			return rd;
		}
		RISCV_ALWAYS_INLINE
		auto get_rs1() const noexcept {
			return rs1;
		}
		RISCV_ALWAYS_INLINE
		auto get_rs2() const noexcept {
			return rs2;
		}
	};

	union FasterImmediate
	{
		uint32_t whole;

		struct
		{
			uint8_t  rd;
			int16_t  imm;
		};

		RISCV_ALWAYS_INLINE
		auto get_rd() const noexcept {
			return rd;
		}

		RISCV_ALWAYS_INLINE
		int32_t signed_imm() const noexcept {
			return imm;
		}
	};

	union FasterMove
	{
		uint32_t whole;

		struct
		{
			uint8_t rs1;
			uint8_t rd;
		};

		RISCV_ALWAYS_INLINE
		auto get_rd() const noexcept {
			return rd;
		}
		RISCV_ALWAYS_INLINE
		auto get_rs1() const noexcept {
			return rs1;
		}
	};

	union FasterJtype
	{
		uint32_t whole;

		struct
		{
			uint32_t offset : 24;
			uint32_t rd     : 8;
		};

		int32_t signed_imm() const noexcept {
			return int32_t(offset << (32 - 24)) >> (32 - 24);
		}
		int32_t upper_imm() const noexcept {
			return offset << 8;
		}
	};

	union FasterFloatType
	{
		uint32_t whole;

		struct
		{
			uint8_t func;
			uint8_t rd;
			uint8_t rs2;
			uint8_t rs1;
		};

		RISCV_ALWAYS_INLINE
		auto get_rd() const noexcept {
			return rd;
		}
		RISCV_ALWAYS_INLINE
		auto get_rs1() const noexcept {
			return rs1;
		}
		RISCV_ALWAYS_INLINE
		auto get_rs2() const noexcept {
			return rs2;
		}
	};

} // riscv
