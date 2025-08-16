#pragma once
#include <cstdint>
#include "types.hpp"

namespace riscv
{
	template <int W>
	struct AtomicMemory
	{
		using address_t = address_type<W>;

		bool load_reserve(int size, address_t addr) RISCV_INTERNAL
		{
			if (!check_alignment(size, addr))
				return false;

			m_reservation = addr;
			return true;
		}

		// Volume I: RISC-V Unprivileged ISA V20190608 p.49:
		// An SC can only pair with the most recent LR in program order.
		bool store_conditional(int size, address_t addr) RISCV_INTERNAL
		{
			if (!check_alignment(size, addr))
				return false;

			bool result = m_reservation == addr;
			// Regardless of success or failure, executing an SC.W
			// instruction invalidates any reservation held by this hart.
			m_reservation = 0x0;
			return result;
		}

	private:
		inline bool check_alignment(int size, address_t addr) RISCV_INTERNAL
		{
			return (addr & (size-1)) == 0;
		}

		address_t m_reservation = 0x0;
	};
}
