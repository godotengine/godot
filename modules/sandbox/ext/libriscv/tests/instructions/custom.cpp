#include <libriscv/machine.hpp>
#include <cassert>

void test_custom_machine()
{
	// this is a custom machine with very little virtual memory
	riscv::Machine<riscv::RISCV32> m2 { std::string_view{}, {
		.memory_max = 65536
	} };

	// free the zero-page to reclaim 4k
	m2.memory.free_pages(0x0, riscv::Page::size());

	// fake a start at 0x1068
	const uint32_t entry_point = 0x1068;
	try {
		m2.cpu.jump(entry_point);
	} catch (...) {
	}
	try {
		m2.simulate();
	} catch (...) {
	}

	assert(m2.instruction_counter() == 0);
}
