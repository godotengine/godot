#pragma once
#include <libriscv/machine.hpp>
#include <libriscv/rv32i_instr.hpp>
#include <cstdio>

namespace riscv
{
	template<int W>
	struct testable_insn {
		const char* name;     // test name
		address_type<W> bits; // the instruction bits
		const int reg;        // which register this insn affects
		const int index;      // test loop index
		address_type<W> initial_value; // start value of register
	};

	template <int W>
	static bool
	validate(Machine<W>& machine, const testable_insn<W>& insn,
			std::function<bool(CPU<W>&, const testable_insn<W>&)> callback)
	{
		static const address_type<W> MEMBASE = 0x1000;

		const std::array<uint32_t, 1> instr_page = {
			insn.bits
		};

		DecodedExecuteSegment<W> &des = machine.cpu.init_execute_area(&instr_page[0], MEMBASE, sizeof(instr_page));
		// jump to page containing instruction
		machine.cpu.jump(MEMBASE);
		// execute instruction
		machine.cpu.reg(insn.reg) = insn.initial_value;
		machine.cpu.step_one();
		// There is a max number of execute segments. Evict the latest to avoid the max limit check
		machine.cpu.memory().evict_execute_segment(des);
		// call instruction validation callback
		if ( callback(machine.cpu, insn) ) return true;
		fprintf(stderr, "Failed test: %s on iteration %d\n", insn.name, insn.index);
		fprintf(stderr, "Register value: 0x%X\n", machine.cpu.reg(insn.reg));
		return false;
	}
}
