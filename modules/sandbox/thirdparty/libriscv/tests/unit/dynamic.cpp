#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
using namespace riscv;
static bool is_zig() {
	const char* rcc = getenv("RCC");
	if (rcc == nullptr)
		return false;
	return std::string(rcc).find("zig") != std::string::npos;
}

static const std::string pie_compiler_args =
	"-O2 -pie -Wl,--no-dynamic-linker,-z,text";

TEST_CASE("Instantiate machine", "[Dynamic]")
{
	if (is_zig())
		return;

	const auto binary = build_and_load(R"M(
	int main() {
		return 666;
	})M", pie_compiler_args);
	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };

	REQUIRE(machine.memory.is_dynamic_executable());
	// The stack is mmap allocated
	REQUIRE(machine.memory.stack_initial() > machine.memory.mmap_start());
	// The starting address is somewhere in the program area
	REQUIRE(machine.memory.start_address() > Memory<RISCV64>::DYLINK_BASE);
	REQUIRE(machine.memory.start_address() < machine.memory.heap_address());
	// The start address is within the current executable area
	REQUIRE(!machine.cpu.current_execute_segment().empty());
	REQUIRE(machine.cpu.current_execute_segment().is_within(machine.memory.start_address()));
}

TEST_CASE("Instantiate machine using shared ELF", "[Dynamic]")
{
	// This fails because we are not loading the dynamic linker.
	REQUIRE_THROWS([] {
		const auto binary = build_and_load(R"M(
		int main() {
			return 666;
		})M", "-shared");
		riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
		machine.simulate(MAX_INSTRUCTIONS);
	}());
}

TEST_CASE("Calculate fib(50) using dynamic ELF", "[Dynamic]")
{
	if (is_zig())
		return;

	const auto binary = build_and_load(R"M(
	static long fib(long n, long acc, long prev)
	{
		if (n == 0)
			return acc;
		else
			return fib(n - 1, prev + acc, acc);
	}

	inline long syscall(long n, long arg0) {
		register long a0 __asm__("a0") = arg0;
		register long syscall_id __asm__("a7") = n;

		__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));

		return a0;
	}

	void _start()
	{
		const volatile long n = 50;
		syscall(93, fib(n, 0, 1));
	})M", pie_compiler_args + " -nostdlib");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls(false, false);
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"dynamic"},
		{"LC_TYPE=C", "LC_ALL=C"});
	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value<long>() == 12586269025L);
}
