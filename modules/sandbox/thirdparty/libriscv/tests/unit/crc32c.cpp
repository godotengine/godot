#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/util/crc32.hpp>
using namespace riscv;

TEST_CASE("Verify CRC32-C", "[CRC32]")
{
	static constexpr char string[] = {
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
	};

	REQUIRE(0x4291922e == crc32c(string, sizeof(string)-1));
}

TEST_CASE("Verify CRC32", "[CRC32]")
{
    constexpr auto crc_code = riscv::crc32("some-id");

	REQUIRE(0x1A61BC96 == crc_code);
}
