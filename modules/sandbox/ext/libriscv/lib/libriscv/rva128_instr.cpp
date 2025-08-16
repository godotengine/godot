#include "instr_helpers.hpp"
#include <cstdint>
#include <inttypes.h>
#define AMOSIZE_W 0x2
#define AMOSIZE_D 0x3
#define AMOSIZE_Q 0x4

namespace riscv
{
	ATOMIC_INSTR(AMOADD_Q,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<__int128_t>(instr,
		[] (auto& cpu, auto& value, auto rs2) {
			auto old_value = value;
			value += cpu.reg(rs2);
			return old_value;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOXOR_Q,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<__int128_t>(instr,
		[] (auto& cpu, auto& value, auto rs2) {
			auto old_value = value;
			value ^= cpu.reg(rs2);
			return old_value;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOOR_Q,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<__int128_t>(instr,
		[] (auto& cpu, auto& value, auto rs2) {
			auto old_value = value;
			value |= cpu.reg(rs2);
			return old_value;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOAND_Q,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<__int128_t>(instr,
		[] (auto& cpu, auto& value, auto rs2) {
			auto old_value = value;
			value &= cpu.reg(rs2);
			return old_value;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOSWAP_Q,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<__int128_t>(instr,
		[] (auto& cpu, auto& value, auto rs2) {
			auto old_value = value;
			value = cpu.reg(rs2);
			return old_value;
		});
	}, DECODED_ATOMIC(AMOSWAP_W).printer);

} // riscv
