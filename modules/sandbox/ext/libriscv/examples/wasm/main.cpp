#include <libriscv/machine.hpp>
#include <chrono>
using Machine = riscv::Machine<riscv::RISCV64>;
static const std::vector<uint8_t> empty;
static constexpr bool USE_EXECUTION_TIMEOUT = true;
static constexpr uint64_t MAX_CYCLES = USE_EXECUTION_TIMEOUT ? 15'000'000'000ull : UINT64_MAX;
static const int HEAP_SYSCALLS_BASE	  = 490; // 490-494 (5)
static const int MEMORY_SYSCALLS_BASE = 495; // 495-509 (15)
#define TIME_POINT(t) \
	asm("" ::: "memory"); \
	auto t = std::chrono::high_resolution_clock::now(); \
	asm("" ::: "memory");
static const uint8_t program_elf[] = {
#embed "program/program.elf"
};

int main()
{
	std::string_view binview{(const char *)program_elf, sizeof(program_elf)};
	Machine machine { binview };
	machine.setup_linux_syscalls();
	// Add native system call interfaces
	const auto heap_area = machine.memory.mmap_allocate(32ULL << 20); // 64 MiB
	machine.setup_native_heap(HEAP_SYSCALLS_BASE, heap_area, 32ULL << 20);
	machine.setup_native_memory(MEMORY_SYSCALLS_BASE);
	machine.setup_linux(
		{"libriscv", "Hello", "World"},
		{"LC_ALL=C", "USER=groot"}
	);

	try {
		// Initialize LuaJIT
		machine.simulate(MAX_CYCLES);
	} catch (const std::exception& e) {
		fprintf(stderr, ">>> Exception: %s\n", e.what());
	}

	// Load the LuaJIT script
	machine.vmcall<MAX_CYCLES>("compile", R"V0G0N(
		-- It's faster to one-shot without JIT
		jit.off(true, true)
		print("Hello, WebAssembly!")
		function fib(n, acc, prev)
			if (n < 1) then
				return acc
			else
				return fib(n - 1, prev + acc, acc)
			end
		end
		print("The 500th fibonacci number is " .. fib(500, 0, 1))
		return 42
	)V0G0N");

	// Run the loaded LuaJIT script
	TIME_POINT(t0);
	machine.vmcall<MAX_CYCLES>("run", 0);
	std::string result = machine.return_value<std::string>();
	TIME_POINT(t1);

	const std::chrono::duration<double, std::micro> exec_time = t1 - t0;
	std::string exec_insns;
	if constexpr (USE_EXECUTION_TIMEOUT) {
		exec_insns.resize(128);
		snprintf(exec_insns.data(), exec_insns.size(), " Insn/s: %.1fmi/s",
			machine.instruction_counter() / exec_time.count());
	}
	printf("\nRuntime: %.fus%s  Result: %s\n",
		exec_time.count(), exec_insns.c_str(),
		result.c_str());
}
