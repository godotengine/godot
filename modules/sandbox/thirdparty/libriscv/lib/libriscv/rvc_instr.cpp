#include "rvc.hpp"
#include "instr_helpers.hpp"
#include <inttypes.h>

namespace riscv
{
	COMPRESSED_INSTR(C0_ADDI4SPN,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		cpu.cireg(ci.CIW.srd) = cpu.reg(REG_SP) + ci.CIW.offset();
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		if (UNLIKELY(ci.whole == 0)) {
			return snprintf(buffer, len, "INVALID: All zeroes");
		}
		return snprintf(buffer, len, "C.ADDI4SPN %s, SP+%u (0x%" PRIx64 ")",
						RISCV::ciname(ci.CIW.srd), ci.CIW.offset(),
						uint64_t(cpu.reg(REG_SP) + ci.CIW.offset()));
	});

	// LW, LD, LQ, FLW, FLD
	COMPRESSED_INSTR(C0_REG_FLD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.cireg(ci.CL.srs1) + ci.CSD.offset8();
		cpu.ciflp(ci.CL.srd).load_u64(
				cpu.machine().memory.template read<uint64_t> (address));
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		static const std::array<const char*, 4> f3 = {
			"???", "FLD", "LW", RVIS64BIT(cpu) ? "LD" : "FLW"
		};
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.%s %s, [%s+%u = 0x%lX]",
						f3[ci.CL.funct3], RISCV::ciname(ci.CL.srd),
						RISCV::ciname(ci.CL.srs1), ci.CL.offset(),
						(long) cpu.cireg(ci.CL.srs1) + ci.CL.offset());
	});

	COMPRESSED_INSTR(C0_REG_LW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.cireg(ci.CL.srs1) + ci.CL.offset();
		cpu.cireg(ci.CL.srd) = (int32_t) cpu.machine().memory.template read<uint32_t> (address);
	}, DECODED_COMPR(C0_REG_FLD).printer);

	COMPRESSED_INSTR(C0_REG_LD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.cireg(ci.CSD.srs1) + ci.CSD.offset8();
		cpu.cireg(ci.CSD.srs2) = (int64_t)
				cpu.machine().memory.template read<uint64_t> (address);
	}, DECODED_COMPR(C0_REG_FLD).printer);

	COMPRESSED_INSTR(C0_REG_FLW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.cireg(ci.CL.srs1) + ci.CL.offset();
		cpu.ciflp(ci.CL.srd).load_u32(
			cpu.machine().memory.template read<uint32_t> (address));
	}, DECODED_COMPR(C0_REG_FLD).printer);

	// SW, SD, SQ, FSW, FSD
	COMPRESSED_INSTR(C0_REG_FSD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		const auto address = cpu.cireg(ci.CSD.srs1) + ci.CSD.offset8();
		const auto value   = cpu.ciflp(ci.CSD.srs2).i64;
		cpu.machine().memory.template write<uint64_t> (address, value);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR
	{
		static const std::array<const char*, 4> f3 = {
			"Reserved instruction", "FSD", "SW", "FSW"
		};
		const rv32c_instruction ci { instr };
		if (ci.CS.funct3 == 0x6) {
		return snprintf(buffer, len, "C.%s %s, [%s%+d]", f3[ci.CS.funct3 - 4],
						RISCV::ciname(ci.CS.srs2),
						RISCV::ciname(ci.CS.srs1), ci.CS.offset4());
		}
		const int offset = (ci.CS.funct3 == 0x7) ? ci.CS.offset4() : ci.CSD.offset8();
		return snprintf(buffer, len, "C.%s %s, [%s%+d]", f3[ci.CS.funct3 - 4],
						RISCV::ciflp(ci.CS.srs2),
						RISCV::ciname(ci.CS.srs1), offset);
	});

	COMPRESSED_INSTR(C0_REG_SW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		const auto address = cpu.cireg(ci.CS.srs1) + ci.CS.offset4();
		const auto value   = cpu.cireg(ci.CS.srs2);
		cpu.machine().memory.template write<uint32_t> (address, value);
	}, DECODED_COMPR(C0_REG_FSD).printer);

	COMPRESSED_INSTR(C0_REG_SD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		const auto address = cpu.cireg(ci.CSD.srs1) + ci.CSD.offset8();
		const auto value   = cpu.cireg(ci.CSD.srs2);
		cpu.machine().memory.template write<uint64_t> (address, value);
	}, DECODED_COMPR(C0_REG_FSD).printer);

	COMPRESSED_INSTR(C0_REG_FSW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		const auto address = cpu.cireg(ci.CS.srs1) + ci.CS.offset4();
		const auto value   = cpu.ciflp(ci.CS.srs2).i32[0];
		cpu.machine().memory.template write<uint32_t> (address, value);
	}, DECODED_COMPR(C0_REG_FSD).printer);

	COMPRESSED_INSTR(C1_ADDI,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		// C.ADDI (non-hint, not NOP)
		cpu.reg(ci.CI.rd) += ci.CI.signed_imm();
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		if (ci.CI.rd != 0) {
			return snprintf(buffer, len, "C.ADDI %s, %" PRId32,
							RISCV::regname(ci.CI.rd), ci.CI.signed_imm());
		}
		if (ci.CI.imm1 != 0 || ci.CI.imm2 != 0)
			return snprintf(buffer, len, "C.HINT");
		return snprintf(buffer, len, "C.NOP");
	});

	COMPRESSED_INSTR(C1_JAL,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		cpu.reg(REG_RA) = cpu.pc() + 2; // return instruction
		const auto address = cpu.pc() + ci.CJ.signed_imm();
		cpu.jump(address - 2);
		if constexpr (verbose_branches_enabled) {
			printf(">>> CALL 0x%lX <-- %s = 0x%lX\n", (long) address,
					RISCV::regname(REG_RA), (long) cpu.reg(REG_RA));
		}
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.JAL %s, PC%+" PRId32 " (0x%" PRIX64 ")",
						RISCV::regname(REG_RA),
						ci.CJ.signed_imm(),
						uint64_t(cpu.pc() + ci.CJ.signed_imm()));
	});

	COMPRESSED_INSTR(C1_ADDIW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		// C.ADDIW rd, imm[5:0]
		const uint32_t src = cpu.reg(ci.CI.rd);
		cpu.reg(ci.CI.rd) = (int32_t) (src + ci.CI.signed_imm());
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.ADDIW %s, %+" PRId32,
						RISCV::regname(ci.CI.rd), ci.CI.signed_imm());
	});

	COMPRESSED_INSTR(C1_LI,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		// C.LI rd, imm[5:0]
		cpu.reg(ci.CI.rd) = ci.CI.signed_imm();
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.LI %s, %+" PRId32,
						RISCV::regname(ci.CI.rd), ci.CI.signed_imm());
	});

	COMPRESSED_INSTR(C1_ADDI16SP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		// C.ADDI16SP rd, imm[17:12]
		cpu.reg(REG_SP) += ci.CI16.signed_imm();
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		if (ci.CI.rd != 0 && ci.CI.rd != 2) {
			return snprintf(buffer, len, "C.LUI %s, 0x%" PRIX32,
							RISCV::regname(ci.CI.rd),
							ci.CI.upper_imm());
		} else if (ci.CI.rd == 2) {
			return snprintf(buffer, len, "C.ADDI16SP %s, %+" PRId32,
							RISCV::regname(ci.CI.rd),
							ci.CI16.signed_imm());
		}
		return snprintf(buffer, len, "C.LUI (Invalid values)");
	});

	COMPRESSED_INSTR(C1_LUI,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		// LUI rd, imm[17:12] (sign-extended)
		cpu.reg(ci.CI.rd) = (int32_t) ci.CI.upper_imm();
	}, DECODED_COMPR(C1_ADDI16SP).printer);

	COMPRESSED_INSTR(C1_ALU_OPS,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32c_instruction ci { instr };
		auto& dst = cpu.cireg(ci.CA.srd);
		switch (ci.CA.funct6 & 0x3)
		{
			case 0: // C.SRLI
				if constexpr (RVIS64BIT(cpu)) {
					dst = dst >> ci.CAB.shift64_imm();
				} else {
					dst = dst >> ci.CAB.shift_imm();
				}
				return;
			case 1: // C.SRAI (preserve sign)
				dst = (RVSIGNTYPE(cpu))dst >> (ci.CAB.shift64_imm() & (RVXLEN(cpu)-1));
				return;
			case 2: // C.ANDI
				dst = dst & ci.CAB.signed_imm();
				return;
			case 3: // more ops
				const auto& src = cpu.cireg(ci.CA.srs2);
				switch (ci.CA.funct2 | (ci.CA.funct6 & 0x4))
				{
					case 0: // C.SUB
						dst = dst - src;
						return;
					case 1: // C.XOR
						dst = dst ^ src;
						return;
					case 2: // C.OR
						dst = dst | src;
						return;
					case 3: // C.AND
						dst = dst & src;
						return;
					case 4: // C.SUBW
					if constexpr (RVIS64BIT(cpu)) {
						dst = (int32_t) ((uint32_t)dst - (uint32_t)src);
						return;
					}
						break;
					case 5: // C.ADDW
					if constexpr (RVIS64BIT(cpu)) {
						dst = (int32_t) ((uint32_t)dst + (uint32_t)src);
						return;
					}
						break;
					default:
						break;
				}
		}
		cpu.trigger_exception(ILLEGAL_OPCODE);
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		if ((ci.CA.funct6 & 0x3) < 2) {
			static const std::array<const char*, 2> f3 = {"SRLI", "SRAI"};
			return snprintf(buffer, len, "C.%s %s, %+d",
				f3[ci.CA.funct6 & 0x3], RISCV::ciname(ci.CAB.srd),
					RVIS64BIT(cpu) ? ci.CAB.shift64_imm() : ci.CAB.shift_imm());
		}
		else if ((ci.CA.funct6 & 0x3) == 2) {
			return snprintf(buffer, len, "C.ANDI %s, %+" PRId32,
							RISCV::ciname(ci.CAB.srd), ci.CAB.signed_imm());
		}
		const int op = ci.CA.funct2 | (ci.CA.funct6 & 0x4);
		static const std::array<const char*, 8> f3 = {
			"SUB", "XOR", "OR", "AND", "SUBW", "ADDW", "RESV", "RESV"
		};

		return snprintf(buffer, len, "C.%s %s, %s", f3[op],
						RISCV::ciname(ci.CA.srd), RISCV::ciname(ci.CA.srs2));
	});

	COMPRESSED_INSTR(C1_JUMP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32c_instruction ci { instr };
		cpu.jump(cpu.pc() + ci.CJ.signed_imm() - 2);
		if constexpr (verbose_branches_enabled) {
			printf(">>> C.JMP 0x%lX\n", (long) cpu.pc() + 2);
		}
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.JMP 0x%" PRIX64,
			uint64_t(cpu.pc() + ci.CJ.signed_imm()));
	});

	COMPRESSED_INSTR(C1_BEQZ,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		// condition: register equals zero
		if (cpu.cireg(ci.CB.srs1) == 0) {
			// branch taken
			cpu.jump(cpu.pc() + ci.CB.signed_imm() - 2);
			if constexpr (verbose_branches_enabled) {
				printf(">>> BRANCH jump to 0x%" PRIX64 "\n", uint64_t(cpu.pc() + 2));
			}
		}
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.BEQZ %s, PC%+" PRId32 " (0x%" PRIX64 ")",
						RISCV::ciname(ci.CB.srs1), ci.CB.signed_imm(),
						uint64_t(cpu.pc() + ci.CB.signed_imm()));
	});

	COMPRESSED_INSTR(C1_BNEZ,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32c_instruction ci { instr };
		// condition: register not-equal zero
		if (cpu.cireg(ci.CB.srs1) != 0) {
			// branch taken
			cpu.jump(cpu.pc() + ci.CB.signed_imm() - 2);
			if constexpr (verbose_branches_enabled) {
				printf(">>> BRANCH jump to 0x%" PRIX64 "\n", (uint64_t)(cpu.pc() + 2));
			}
		}
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		return snprintf(buffer, len, "C.BNEZ %s, PC%+" PRId32 " (0x%" PRIX64 ")",
						RISCV::ciname(ci.CB.srs1), ci.CB.signed_imm(),
						uint64_t(cpu.pc() + ci.CB.signed_imm()));
	});

	// C.SLLI, LWSP, LDSP, LQSP, FLWSP, FLDSP
	COMPRESSED_INSTR(C2_SLLI,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32c_instruction ci { instr };
		if constexpr (RVIS64BIT(cpu)) {
			cpu.reg(ci.CI.rd) <<= ci.CI.shift64_imm();
		} else {
			cpu.reg(ci.CI.rd) <<= ci.CI.shift_imm();
		}
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		if (ci.CI2.funct3 == 0x0 && ci.CI2.rd != 0) {
			return snprintf(buffer, len, "C.SLLI %s, %u",
							RISCV::regname(ci.CI.rd),
							(RVIS64BIT(cpu)) ? ci.CI.shift64_imm() : ci.CI.shift_imm());
		}
		else if (ci.CI2.rd != 0) {
			static const std::array<const char*, 4> f3 = {
				"???", "FLDSP", "LWSP", "FLWSP"
			};
			const char* regname = (ci.CI2.funct3 & 1)
			 	? RISCV::flpname(ci.CI2.rd) : RISCV::regname(ci.CI2.rd);
			auto address = (ci.CI2.funct3 != 0x1) ?
						  cpu.reg(REG_SP) + ci.CI2.offset()
						: cpu.reg(REG_SP) + ci.CIFLD.offset();
			return snprintf(buffer, len, "C.%s %s, [SP+%u] (0x%" PRIX64 ")", f3[ci.CI2.funct3],
							regname, ci.CI2.offset(), uint64_t(address));
		}
		return snprintf(buffer, len, "C.HINT %s", RISCV::regname(ci.CI2.rd));
	});

	COMPRESSED_INSTR(C2_FLDSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.reg(REG_SP) + ci.CIFLD.offset();
		auto& dst = cpu.registers().getfl(ci.CIFLD.rd);
		dst.load_u64(cpu.machine().memory.template read <uint64_t> (address));
	}, DECODED_COMPR(C2_SLLI).printer);

	COMPRESSED_INSTR(C2_LWSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.reg(REG_SP) + ci.CI2.offset();
		cpu.reg(ci.CI2.rd) = (int32_t) cpu.machine().memory.template read <uint32_t> (address);
	}, DECODED_COMPR(C2_SLLI).printer);

	COMPRESSED_INSTR(C2_LDSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.reg(REG_SP) + ci.CIFLD.offset();
		cpu.reg(ci.CIFLD.rd) = (int64_t)
			cpu.machine().memory.template read <uint64_t> (address);
	}, DECODED_COMPR(C2_SLLI).printer);

	COMPRESSED_INSTR(C2_FLWSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto address = cpu.reg(REG_SP) + ci.CI2.offset();
		auto& dst = cpu.registers().getfl(ci.CI2.rd);
		dst.load_u32(cpu.machine().memory.template read <uint32_t> (address));
	}, DECODED_COMPR(C2_SLLI).printer);

	// SWSP, SDSP, SQSP, FSWSP, FSDSP
	COMPRESSED_INSTR(C2_FSDSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto addr = cpu.reg(REG_SP) + ci.CSFSD.offset();
		uint64_t value = cpu.registers().getfl(ci.CSFSD.rs2).i64;
		cpu.machine().memory.template write<uint64_t> (addr, value);
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		static const std::array<const char*, 4> f3 = {
			"XXX", "FSDSP", "SWSP", RVIS64BIT(cpu) ? "SDSP" : "FSWSP"
		};
		const rv32c_instruction ci { instr };
		auto address = cpu.reg(REG_SP) + ci.CSS.offset(4);
		return snprintf(buffer, len, "C.%s [SP%+d], %s (0x%lX)",
						f3[ci.CSS.funct3 - 4], ci.CSS.offset(4),
						RISCV::regname(ci.CSS.rs2), (long) address);
	});

	COMPRESSED_INSTR(C2_SWSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto addr = cpu.reg(REG_SP) + ci.CSS.offset(4);
		uint32_t value = cpu.reg(ci.CSS.rs2);
		cpu.machine().memory.template write<uint32_t> (addr, value);
	}, DECODED_COMPR(C2_FSDSP).printer);

	COMPRESSED_INSTR(C2_SDSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto addr = cpu.reg(REG_SP) + ci.CSFSD.offset();
		auto value = cpu.reg(ci.CSFSD.rs2);
		cpu.machine().memory.template write<uint64_t> (addr, value);
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		auto address = cpu.reg(REG_SP) + ci.CSFSD.offset();
		return snprintf(buffer, len, "C.SDSP [SP%+d], %s (0x%lX)",
						ci.CSFSD.offset(), RISCV::regname(ci.CSS.rs2), (long) address);
	});

	COMPRESSED_INSTR(C2_FSWSP,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		auto addr = cpu.reg(REG_SP) + ci.CSS.offset(4);
		uint32_t value = cpu.registers().getfl(ci.CSS.rs2).i32[0];
		cpu.machine().memory.template write<uint32_t> (addr, value);
	}, DECODED_COMPR(C2_FSDSP).printer);

	// C.JR, C.MV, C.JALR, C.ADD
	COMPRESSED_INSTR(C2_JR,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		cpu.jump(cpu.reg(ci.CR.rd) - 2);
		if constexpr (verbose_branches_enabled) {
			printf(">>> RET 0x%lX <-- %s = 0x%lX\n", (long) cpu.pc(),
				RISCV::regname(ci.CR.rd), (long) cpu.reg(ci.CR.rd));
		}
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR
	{
		const rv32c_instruction ci { instr };
		const bool topbit = ci.whole & (1 << 12);
		if (!topbit && ci.CR.rs2 == 0 && ci.CR.rd != 0) {
			if (ci.CR.rd == REG_RA)
				return snprintf(buffer, len, "C.RET");
			return snprintf(buffer, len, "C.JR %s", RISCV::regname(ci.CR.rd));
		} else if (!topbit && ci.CR.rs2 != 0 && ci.CR.rd != 0)
			return snprintf(buffer, len, "C.MV %s, %s",
							RISCV::regname(ci.CR.rd), RISCV::regname(ci.CR.rs2));
		else if (topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0)
			return snprintf(buffer, len, "C.JALR RA, %s (0x%lX)",
							RISCV::regname(ci.CR.rd), (long)cpu.reg(ci.CR.rd));
		else if (ci.CR.rd != 0)
			return snprintf(buffer, len, "C.ADD %s, %s + %s", RISCV::regname(ci.CR.rd),
							RISCV::regname(ci.CR.rd), RISCV::regname(ci.CR.rs2));
		else if (topbit && ci.CR.rd == 0 && ci.CR.rs2 == 0)
			return snprintf(buffer, len, "C.EBREAK");
		return snprintf(buffer, len, "C.HINT");
	});

	COMPRESSED_INSTR(C2_JALR,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		cpu.reg(REG_RA) = cpu.pc() + 0x2;
		cpu.jump(cpu.reg(ci.CR.rd) - 2);
		if constexpr (verbose_branches_enabled) {
			printf(">>> C.JAL RA, 0x%lX <-- %s = 0x%lX\n",
				(long) cpu.reg(REG_RA) - 2,
				RISCV::regname(ci.CR.rd), (long) cpu.reg(ci.CR.rd));
		}
	}, DECODED_COMPR(C2_JR).printer);

	COMPRESSED_INSTR(C2_MV,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		cpu.reg(ci.CR.rd) = cpu.reg(ci.CR.rs2);
	}, DECODED_COMPR(C2_JR).printer);

	COMPRESSED_INSTR(C2_ADD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		const rv32c_instruction ci { instr };
		cpu.reg(ci.CR.rd) += cpu.reg(ci.CR.rs2);
	}, DECODED_COMPR(C2_JR).printer);

	COMPRESSED_INSTR(C2_EBREAK,
	[] (auto& cpu, rv32i_instruction) RVINSTR_COLDATTR {
		cpu.machine().ebreak();
	}, DECODED_COMPR(C2_JR).printer);
}
