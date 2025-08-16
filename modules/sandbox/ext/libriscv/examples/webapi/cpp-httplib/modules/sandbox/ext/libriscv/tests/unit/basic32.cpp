#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
using namespace riscv;
static bool is_zig() {
	const char* rcc = getenv("RCC");
	if (rcc == nullptr)
		return false;
	return std::string(rcc).find("zig") != std::string::npos;
}

TEST_CASE("Instantiate 32-bit machine using 64-bit ELF", "[Instantiate]")
{
	REQUIRE_THROWS([] {
		const auto binary = build_and_load(R"M(
		int main() {
			return 666;
		})M");
		riscv::Machine<RISCV32> machine { binary };
	}());
}

TEST_CASE("Validate custom 32-bit ELF", "[Instantiate]")
{
	if (is_zig())
		return;

	const auto binary = build_and_load(R"M(
	__asm__(".global _start\n"
	".section .text\n"
	"_start:\n"
	"	li a0, 0xDEADBEEF\n"
	"	li a7, 1\n"
	"	ecall\n"
	"	nop\n"
	"   wfi\n");
	)M", "-static -march=rv32g -mabi=ilp32d -nostdlib");

	riscv::Machine<RISCV32> machine { binary };

	machine.install_syscall_handler(1,
		[] (auto& machine) { machine.stop(); });
	machine.simulate(10);
	REQUIRE(machine.return_value() == 0xDEADBEEF);
}

TEST_CASE("Validate 32-bit fib()", "[Compute]")
{
	if (is_zig())
		return;

    // 64-bit arguments require two registers
    // and consumes 2 return registers (A0, A1).
    const auto binary = build_and_load(R"M(
    #define u64 long long

	u64 fib(u64 n, u64 acc, u64 prev)
	{
		if (n < 1)
			return acc;
		else
			return fib(n - 1, prev + acc, acc);
	}

	__asm__(".global _start\n"
	".section .text\n"
	"_start:\n"
	"	li a0, 50\n"
	"	li a2, 0\n"
	"	li a4, 1\n"
	"	call fib\n"
	"	li a7, 1\n"
	"	ecall\n"
	"   wfi\n");
	)M", "-static -march=rv32g -mabi=ilp32d -nostdlib");

    riscv::Machine<RISCV32> machine { binary };

	machine.install_syscall_handler(1,
		[] (auto& machine) { machine.stop(); });
	machine.simulate(MAX_INSTRUCTIONS);
	REQUIRE(machine.return_value<uint64_t>() == 12586269025L);
}
