#pragma once
#include "types.hpp"
#include <array>

namespace riscv
{
	union alignas(RISCV_EXT_VECTOR) VectorLane {
		static constexpr unsigned VSIZE = RISCV_EXT_VECTOR;
		static constexpr unsigned size() noexcept { return VSIZE; }

		std::array<uint8_t,  VSIZE / 1> u8 = {};
		std::array<uint16_t, VSIZE / 2> u16;
		std::array<uint32_t, VSIZE / 4> u32;
		std::array<uint64_t, VSIZE / 8> u64;

		std::array<float,  VSIZE / 4> f32;
		std::array<double, VSIZE / 8> f64;
	};
	static_assert(sizeof(VectorLane) == RISCV_EXT_VECTOR, "Vectors are 32 bytes");
	static_assert(alignof(VectorLane) == RISCV_EXT_VECTOR, "Vectors are 32-byte aligned");

	template <int W>
	struct alignas(RISCV_EXT_VECTOR) VectorRegisters
	{
		using address_t  = address_type<W>;   // one unsigned memory address
		using register_t = register_type<W>;  // integer register

		auto& get(unsigned idx) noexcept { return m_vec[idx]; }
		auto& f32(unsigned idx) { return m_vec[idx].f32; }
		auto& u32(unsigned idx) { return m_vec[idx].u32; }

		register_t vtype() const noexcept {
			return 0u;
		}


	private:
		std::array<VectorLane, 32> m_vec {};
	};
}