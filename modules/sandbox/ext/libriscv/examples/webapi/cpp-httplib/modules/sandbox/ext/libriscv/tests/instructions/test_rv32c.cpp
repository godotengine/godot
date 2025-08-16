#include "testable_instruction.hpp"
#include <libriscv/rvc.hpp>
#include <cassert>
using namespace riscv;


void test_rv32c()
{
	riscv::Machine<RISCV32> machine { std::string_view{}, {
		.memory_max = 65536
	} };

	// C.SRLI imm = [0, 31] CI_CODE(0b100, 0b01):
	for (int i = 0; i < 32; i++)
	{
		rv32c_instruction ci;
		ci.CA.opcode  = 0b01;     // Quadrant 1
		ci.CA.funct6  = 0b100000; // ALU OP: SRLI
		ci.CAB.srd    = 0x2; // A0
		ci.CAB.imm04  = i;
		ci.CAB.imm5   = 0;

		const testable_insn<RISCV32> insn {
			.name  = "C.SRLI",
			.bits  = ci.whole,
			.reg   = REG_ARG0,
			.index = i,
			.initial_value = 0xFFFFFFFF
		};
		bool b = validate<RISCV32>(machine, insn,
		[] (auto& cpu, const auto& insn) -> bool {
			return cpu.reg(insn.reg) == (insn.initial_value >> insn.index);
		});
		assert(b);
	}

	// C.SRAI imm = [0, 31] CI_CODE(0b100, 0b01):
	for (int i = 0; i < 32; i++)
	{
		rv32c_instruction ci;
		ci.CA.opcode  = 0b01;     // Quadrant 1
		ci.CA.funct6  = 0b100001; // ALU OP: SRAI
		ci.CAB.srd    = 0x2; // A0
		ci.CAB.imm04  = i;
		ci.CAB.imm5   = 0;

		const testable_insn<RISCV32> insn {
			.name  = "C.SRAI",
			.bits  = ci.whole,
			.reg   = REG_ARG0,
			.index = i,
			.initial_value = 0xFFFFFFFF
		};
		bool b = validate<RISCV32>(machine, insn,
		[] (auto& cpu, const auto& insn) -> bool {
			return cpu.reg(insn.reg) == insn.initial_value;
		});
		assert(b);
	}

	// C.ANDI imm = [-32, 31] CI_CODE(0b100, 0b01):
	for (int i = 0; i < 64; i++)
	{
		rv32c_instruction ci;
		ci.CA.opcode  = 0b01;     // Quadrant 1
		ci.CA.funct6  = 0b100010; // ALU OP: ANDI
		ci.CAB.srd    = 0x2; // A0
		ci.CAB.imm04  = i & 31;
		ci.CAB.imm5   = i >> 5;

		const testable_insn<RISCV32> insn {
			.name  = "C.ANDI",
			.bits  = ci.whole,
			.reg   = REG_ARG0,
			.index = i,
			.initial_value = 0xFFFFFFFF
		};
		bool b = validate<RISCV32>(machine, insn,
		[] (auto& cpu, const auto& insn) -> bool {
			if (insn.index < 32) {
				return cpu.reg(insn.reg) == (insn.initial_value & insn.index);
			}
			return cpu.reg(insn.reg) == (insn.initial_value & (insn.index-64));
		});
		assert(b);
	}

	// C.SLLI imm = [0, 31] CI_CODE(0b011, 0b10):
	for (int i = 0; i < 32; i++)
	{
		rv32c_instruction ci;
		ci.CI.opcode  = 0b10; // Quadrant 1
		ci.CI.funct3  = 0x0;  // OP: SLLI
		ci.CI.rd      = 0xA;  // A0
		ci.CI.imm1    = i;
		ci.CI.imm2    = 0;

		const testable_insn<RISCV32> insn {
			.name  = "C.SLLI",
			.bits  = ci.whole,
			.reg   = REG_ARG0,
			.index = i,
			.initial_value = 0xA
		};
		bool b = validate<RISCV32>(machine, insn,
		[] (auto& cpu, const auto& insn) -> bool {
			return cpu.reg(insn.reg) == (insn.initial_value << insn.index);
		});
		assert(b);
	}

	printf("%lu instructions passed.\n", machine.instruction_counter());
}
