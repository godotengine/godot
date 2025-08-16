#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
#include <libriscv/debug.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const std::string cwd {SRCDIR};
using namespace riscv;
static bool is_zig() {
	const char* rcc = getenv("RCC");
	if (rcc == nullptr)
		return false;
	return std::string(rcc).find("zig") != std::string::npos;
}

/**
 * These tests are designed to be really brutal to support,
 * and most emulators will surely fail here.
*/

TEST_CASE("Calculate fib(50) slowly, basic", "[Compute]")
{
	const auto binary = build_and_load(R"M(
	long fib(long n, long acc, long prev)
	{
		if (n < 1)
			return acc;
		else
			return fib(n - 1, prev + acc, acc);
	}
	__attribute__((used, retain))
	extern long my_start(long n) {
		return fib(n, 0, 1);
	}
	int main() {}
	)M", "-O2 -static");

	riscv::Machine<RISCV64> machine { binary, { .use_memory_arena = false } };

	const auto addr = machine.address_of("my_start");
	REQUIRE(addr != 0x0);

	// Create a manual VM call in order to avoid exercising the C-runtime
	// The goal is to see if the basic start/stop/resume functionality works
	machine.cpu.jump(addr);
	machine.cpu.reg(riscv::REG_ARG0) = 50;
	machine.cpu.reg(riscv::REG_RA) = machine.memory.exit_address();

	riscv::Machine<RISCV64> fork { machine };
	do {
		// No matter how many (or few) instructions we execute before exiting
		// simulation, we should be able to resume and complete the program normally.
		for (int step = 5; step < 105; step++) {
			fork.cpu.registers() = machine.cpu.registers();
			do {
				fork.simulate<false>(step);
			} while (fork.instruction_limit_reached());
			REQUIRE(fork.return_value<long>() == 12586269025L);
		}
		machine.simulate<false>(100);
	} while (machine.instruction_limit_reached());
}

TEST_CASE("Execute libc_start_main, slowly", "[Compute]")
{
	const auto binary = build_and_load(R"M(
	int main() {
		return 1234;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .use_memory_arena = false } };
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"brutal", "50"},
		{"LC_TYPE=C", "LC_ALL=C"});

	do {
		// No matter how many (or few) instructions we execute before exiting
		// simulation, we should be able to resume and complete the program normally.
		for (int step = 5; step < 105; step++) {
			riscv::Machine<RISCV64> fork { machine, { .use_memory_arena = false } };
			do {
				fork.simulate<false>(step);
			} while (fork.instruction_limit_reached());
			REQUIRE(fork.return_value<long>() == 1234);
		}
		machine.simulate<false>(1000);
	} while (machine.instruction_limit_reached());

	REQUIRE(machine.return_value<long>() == 1234);
}

TEST_CASE("Threads test-suite slowly", "[Compute]")
{
	if (is_zig())
		return;

	const auto binary = build_and_load(R"M(
	#include "threads/test_threads.cpp"
	)M", "-O1 -static -pthread -I" + cwd, true);

	riscv::Machine<RISCV64> machine { binary, { .use_memory_arena = false } };
	machine.setup_linux_syscalls();
	machine.setup_posix_threads();
	machine.setup_linux(
		{"brutal", "123"},
		{"LC_TYPE=C", "LC_ALL=C"});

	do {
		// No matter how many (or few) instructions we execute before exiting
		// simulation, we should be able to resume and complete the program normally.
		const int step = 5;
		riscv::Machine<RISCV64> fork { machine };
		fork.set_printer([] (const auto&, const char*, size_t) {});

		do {
			fork.simulate<false>(step);
		} while (fork.instruction_limit_reached());
		REQUIRE(fork.return_value<long>() == 123666123L);

		machine.simulate<false>(100000UL);
	} while (machine.instruction_limit_reached());
	REQUIRE(machine.return_value<long>() == 123666123L);
}
