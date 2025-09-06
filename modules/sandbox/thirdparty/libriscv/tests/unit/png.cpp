#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 64'000'000ul;
static const std::string cwd {SRCDIR};
using namespace riscv;
#include "crc32.hpp"
#include "png.hpp"
#include "mandelbrot.cpp"
static constexpr size_t COUNTER = 4;

auto locally_produced_png()
{
	constexpr size_t W = 128;
	constexpr size_t H = 128;

	const double factor = powf(2.0, COUNTER * -0.1);
	const double x1 = -1.5;
	const double x2 =  2.0 * factor;
	const double y1 = -1.0 * factor;
	const double y2 =  2.0 * factor;

	auto bitmap = fractal<W, H, 65> (x1, y1, x2, y2);
	auto* data = (const uint8_t *)bitmap.data();

	return encode(W, H, data);
}

TEST_CASE("Verify PNG encoding", "[Emulation]")
{
	const auto binary = build_and_load(R"M(
	#include <cassert>
	#include "png.hpp"
	#include "mandelbrot.cpp"

	__attribute__((noreturn))
	inline void respond(const uint8_t* arg0, size_t arg1)
	{
		register const uint8_t* a0 asm("a0") = (const uint8_t*)arg0;
		register size_t      a1 asm("a1") = arg1;

		asm volatile ("wfi" : :
			"m"(*(const char(*)[arg1]) a0), "r"(a0), "r"(a1) : "memory");
		__builtin_unreachable();
	}

	int main(int argc, char** argv) {
		constexpr size_t W = 128;
		constexpr size_t H = 128;
		const size_t counter = atoi(argv[1]);

		const double factor = powf(2.0, counter * -0.1);
		const double x1 = -1.5;
		const double x2 =  2.0 * factor;
		const double y1 = -1.0 * factor;
		const double y2 =  2.0 * factor;

		auto bitmap = fractal<W, H, 65> (x1, y1, x2, y2);
		auto* data = (const uint8_t *)bitmap.data();

		auto png = encode(W, H, data);
		respond(png.data(), png.size());
	})M", "-static -O2 -I" + cwd, true);

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"verifypng", std::to_string(COUNTER)},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});
	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	const auto good_png = locally_produced_png();
	auto g_png = machine.cpu.reg(riscv::REG_ARG0);
	auto size  = machine.cpu.reg(riscv::REG_ARG1);
	std::vector<uint8_t> png;
	png.resize(size);
	machine.copy_from_guest(png.data(), g_png, size);

	auto guest_crc = crc32(png.data(), png.size());
	auto good_crc  = crc32(good_png.data(), good_png.size());

	REQUIRE(guest_crc == good_crc);

	FILE* fp = fopen("good.png", "w+");
	fwrite(good_png.data(), 1, good_png.size(), fp);
	fclose(fp);

	fp = fopen("guest.png", "w+");
	fwrite(png.data(), 1, png.size(), fp);
	fclose(fp);
}
