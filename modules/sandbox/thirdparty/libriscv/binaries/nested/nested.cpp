#include <chrono>
#include <libriscv/machine.hpp>
#include <inttypes.h>
static constexpr uint64_t MAX_MEMORY = 512ULL << 20;
static constexpr int  RISCV_ARCH = 8; // 64-bit
static constexpr bool verbose_enabled = true;
#include "stream.h"

int main()
{
	const std::string_view binary { (const char*)stream_rv64gvb, stream_rv64gvb_len };

	riscv::Machine<RISCV_ARCH> machine { binary, {
		.memory_max = MAX_MEMORY,
		.verbose_loader = verbose_enabled,
	} };

	static const std::vector<std::string> env = {
		"LC_CTYPE=C", "LC_ALL=C"
	};
	machine.setup_linux({"program"}, env);

	// Linux system to open files and access internet
	machine.setup_linux_syscalls();

	auto t0 = std::chrono::high_resolution_clock::now();
	try {
		// Normal RISC-V simulation
		machine.cpu.simulate_inaccurate(machine.cpu.pc());
	} catch (const riscv::MachineException& me) {
		printf("%s\n", machine.cpu.current_instruction_to_string().c_str());
		printf(">>> Machine exception %d: %s (data: 0x%" PRIX64 ")\n",
				me.type(), me.what(), me.data());
		printf("%s\n", machine.cpu.registers().to_string().c_str());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		if (me.type() == riscv::UNIMPLEMENTED_INSTRUCTION || me.type() == riscv::MISALIGNED_INSTRUCTION) {
			printf(">>> Is an instruction extension disabled?\n");
			printf(">>> A-extension: %d  C-extension: %d  V-extension: %d\n",
				riscv::atomics_enabled, riscv::compressed_enabled, riscv::vector_extension);
		}
	} catch (const std::exception& e) {
		printf(">>> Exception: %s\n", e.what());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> runtime = t1 - t0;

	const auto retval = machine.return_value();
	printf(">>> Program exited, exit code = %" PRId64 " (0x%" PRIX64 ")\n",
		int64_t(retval), uint64_t(retval));
	printf("Instructions executed: %" PRIu64 "  Runtime: %.3fms  Insn/s: %.0fmi/s\n",
		machine.instruction_counter(), runtime.count()*1000.0,
		machine.instruction_counter() / (runtime.count() * 1e6));
	printf("Pages in use: %zu (%" PRIu64 " kB virtual memory, total %" PRIu64 " kB)\n",
		machine.memory.pages_active(),
		machine.memory.pages_active() * riscv::Page::size() / uint64_t(1024),
		machine.memory.memory_usage_total() / uint64_t(1024));
}
