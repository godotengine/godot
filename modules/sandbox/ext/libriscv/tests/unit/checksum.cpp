#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
static const std::string cwd {SRCDIR};
using namespace riscv;

TEST_CASE("Verify CRC32", "[Emulation]")
{
	const auto binary = build_and_load(R"M(
	#include "crc32.hpp"
	#include <cassert>
	int main(int argc, char** argv) {
		constexpr uint8_t arr[] = {'H', 'e', 'l', 'l', 'o', '!', 0};
		// Constexpr version is always good
		constexpr unsigned c[] = {
			crc32(arr, 1),
			crc32(arr, 2),
			crc32(arr, 3),
			crc32(arr, 4),
			crc32(arr, 5),
			crc32(arr, 6),
		};
		for (unsigned i = 0; i < 6; i++) {
			assert(c[i] == crc32(arr, i+1));
		}
		return 0;
	})M", "-static -O2 -I" + cwd, true);

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"checksum"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 0);
}
