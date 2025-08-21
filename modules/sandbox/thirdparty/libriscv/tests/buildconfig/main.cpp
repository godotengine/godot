#include <libriscv/machine.hpp>

static constexpr uint64_t MAX_MEMORY = 1024 * 1024 * 200;

int main()
{
	const std::vector<uint8_t> binary;

	riscv::Machine<riscv::RISCV32> machine32 { binary, {
		.memory_max = MAX_MEMORY
	}};

	riscv::Machine<riscv::RISCV64> machine64 { binary, {
		.memory_max = MAX_MEMORY
	}};

	riscv::Machine<riscv::RISCV128> machine128 { binary, {
		.memory_max = MAX_MEMORY
	}};

}
