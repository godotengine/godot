#include "instr_helpers.hpp"
#include <inttypes.h>

namespace riscv
{
	INSTRUCTION(LOAD_U128,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		auto& reg = cpu.reg(instr.Itype.rd);
		auto addr = cpu.reg(instr.Itype.rs1) + RVIMM(cpu, instr.Itype);
		addr &= ~(__uint128_t)0xF;
		reg = (RVSIGNTYPE(cpu)) cpu.machine().memory.template read<__uint128_t>(addr);
	}, DECODED_INSTR(LOAD_I8).printer);

    INSTRUCTION(OP_IMM64,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		auto& dst = cpu.reg(instr.Itype.rd);
		const uint64_t src = cpu.reg(instr.Itype.rs1);
		switch (instr.Itype.funct3) {
		case 0x0:
			// ADDI.D: Add sign-extended 12-bit immediate
			dst = (int64_t) (src + instr.Itype.shift64_imm());
			return;
		case 0x1: // SLLI.D:
			dst = (int64_t) (src << instr.Itype.shift64_imm());
			return;
		case 0x5: // SRLI.D / SRAI.D:
			if (LIKELY(!instr.Itype.is_srai())) {
				dst = (int64_t) (src >> instr.Itype.shift64_imm());
			} else { // SRAIW: preserve the sign bit
				dst = (int64_t)src >> instr.Itype.shift64_imm();
			}
			return;
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	}, DECODED_INSTR(OP_IMM32_ADDIW).printer);

	INSTRUCTION(OP64,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		auto& dst = cpu.reg(instr.Rtype.rd);
		const uint64_t src1 = cpu.reg(instr.Rtype.rs1);
		const uint64_t src2 = cpu.reg(instr.Rtype.rs2);

		switch (instr.Rtype.jumptable_friendly_op()) {
		case 0x0: // ADD.D
			dst = (int64_t) (src1 + src2);
			return;
		case 0x1: // SLL.D
			dst = (int64_t) (src1 << (src2 & 0x3F));
			return;
		case 0x5: // SRL.D
			dst = (int64_t) (src1 >> (src2 & 0x3F));
			return;
		// M-extension
		case 0x10: // MUL.D
			dst = (int64_t) ((int64_t)src1 * (int64_t)src2);
			return;
		case 0x14: // DIV.D
			// division by zero is not an exception
			if (LIKELY(src2 != 0)) {
				if (LIKELY(!((int64_t)src1 == INT64_MIN && (int64_t)src2 == -1ll)))
					dst = (int64_t) ((int64_t)src1 / (int64_t)src2);
			} else {
				dst = (RVREGTYPE(cpu)) -1;
			}
			return;
		case 0x15: // DIVU.D
			if (LIKELY(src2 != 0)) {
				dst = (int64_t) (src1 / src2);
			} else {
				dst = (RVREGTYPE(cpu)) -1;
			}
			return;
		case 0x16: // REM.D
			if (LIKELY(src2 != 0)) {
				if (LIKELY(!((int64_t)src1 == INT64_MIN && (int64_t)src2 == -1ll)))
					dst = (int64_t) ((int64_t)src1 % (int64_t)src2);
			} else {
				dst = (RVREGTYPE(cpu)) -1;
			}
			return;
		case 0x17: // REMU.D
			if (LIKELY(src2 != 0)) {
				dst = (int64_t) (src1 % src2);
			} else {
				dst = (RVREGTYPE(cpu)) -1;
			}
			return;
		case 0x200: // SUB.D
			dst = (int64_t) (src1 - src2);
			return;
		case 0x205: // SRA.D
			dst = (int64_t)src1 >> (src2 & 63);
			return;
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	}, DECODED_INSTR(OP32).printer);

} // riscv
