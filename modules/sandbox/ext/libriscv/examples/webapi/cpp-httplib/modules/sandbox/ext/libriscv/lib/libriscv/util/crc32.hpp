#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace riscv {

template <uint32_t POLYNOMIAL>
inline constexpr auto gen_crc32_table()
{
	constexpr auto num_iterations = 8;
	auto crc32_table = std::array<uint32_t, 256> {};

	for (auto byte = 0u; byte < crc32_table.size(); ++byte) {
		auto crc = byte;

		for (auto i = 0; i < num_iterations; ++i) {
			auto mask = -(crc & 1);
			crc = (crc >> 1) ^ (POLYNOMIAL & mask);
		}

		crc32_table[byte] = crc;
	}
	return crc32_table;
}

template <uint32_t POLYNOMIAL = 0xEDB88320>
inline constexpr auto crc32(const char* data)
{
	constexpr auto crc32_table = gen_crc32_table<POLYNOMIAL>();

	auto crc = 0xFFFFFFFFu;
	for (auto i = 0u; auto c = data[i]; ++i) {
		crc = crc32_table[(crc ^ c) & 0xFF] ^ (crc >> 8);
	}
	return ~crc;
}

template <uint32_t POLYNOMIAL = 0xEDB88320>
inline constexpr auto crc32(uint32_t crc, const void* vdata, const size_t len)
{
	constexpr auto crc32_table = gen_crc32_table<POLYNOMIAL>();

	auto* data = (const uint8_t*) vdata;
	for (auto i = 0u; i < len; ++i) {
		crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
	}
	return crc;
}

template <uint32_t POLYNOMIAL = 0xEDB88320>
inline constexpr auto crc32(const void* vdata, const size_t len)
{
	return ~crc32<POLYNOMIAL>(0xFFFFFFFF, vdata, len);
}

// CRC32-C with hardware acceleration on amd64
extern uint32_t crc32c(const void* data, size_t);
extern uint32_t crc32c(uint32_t partial, const void* data, size_t);

} // riscv
