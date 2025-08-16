#include <libriscv/machine.hpp>
#include <cassert>
using namespace riscv;

#define assert_mem(m, type, addr, value) \
		assert(m.memory.template read<type> (addr) == value)

static const std::array<uint32_t, 4> instructions =
{
	0x00065637, // lui     a2,0x65
	0x000655b7, // lui     a1,0x65
	0x11612023, // sw      s6,256(sp)
	0x0b410b13, // addi    s6,sp,180
};

void test_rv32i()
{
	riscv::Machine<riscv::RISCV32> m { std::string_view{}, {
		.memory_max = 65536
	} };
	// install instructions
	m.cpu.init_execute_area(instructions.data(), 0x1000, 4 * instructions.size());
	m.cpu.jump(0x1000);

	// stack frame
	m.cpu.reg(REG_SP) = 0x120000 - 288;
	const uint32_t current_sp = m.cpu.reg(REG_SP);

	// execute LUI a2, 0x65000
	m.cpu.step_one();
	assert(m.cpu.reg(REG_ARG2) == 0x65000);
	// execute LUI a1, 0x65000
	m.cpu.step_one();
	assert(m.cpu.reg(REG_ARG1) == 0x65000);
	// execute SW  s6, [SP + 256]
	m.cpu.reg(22) = 0x12345678;
	m.cpu.step_one();
	assert_mem(m, uint32_t, current_sp + 256, m.cpu.reg(22));
	// execute ADDI s6, [SP + 180]
	m.cpu.reg(22) = 0x0;
	m.cpu.step_one();
	assert(m.cpu.reg(22) == current_sp + 180);
}
